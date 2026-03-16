import os
import time
import requests
from dotenv import load_dotenv
import argparse

load_dotenv()

API_KEY = os.environ.get("RUNPOD_API_KEY")
URL = "https://api.runpod.io/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

if not API_KEY:
    raise RuntimeError("RUNPOD_API_KEY is not set")

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="SINDy", type=str, help="Name for pod")
args = parser.parse_args()
pod_name = args.name


def run_query(query: str, variables: dict | None = None) -> dict:
    resp = requests.post(
        URL,
        json={"query": query, "variables": variables or {}},
        headers=HEADERS,
        timeout=30,
    )

    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code} error from Runpod:\n{resp.text}")

    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")

    return data["data"]


def find_port_mapping(runtime: dict | None, private_port: int) -> dict | None:
    if not runtime:
        return None
    
    ports = runtime.get("ports") or []
    for p in ports:
        if (
            p.get("privatePort") == private_port
            and p.get("publicPort") is not None
            and p.get("ip")
        ):
            return p
    return None

launch_query = """
mutation DeployPod($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) {
    id
    desiredStatus
  }
}
"""

bootstrap_cmd = r"""bash -lc 'set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
REPO_DIR=/workspace

echo "[bootstrap] installing system packages"
apt-get update
apt-get install -y --no-install-recommends \
  nano \
  git \
  ca-certificates \
  curl \
  ffmpeg \
  rsync \
  openssh-server
rm -rf /var/lib/apt/lists/*

echo "[bootstrap] configuring sshd"
mkdir -p /var/run/sshd
mkdir -p /root/.ssh
chmod 700 /root/.ssh

AUTHORIZED_KEY="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
if [ -z "$AUTHORIZED_KEY" ]; then
  echo "[bootstrap] ERROR: neither SSH_PUBLIC_KEY nor PUBLIC_KEY env var is set"
  exit 1
fi

printf "%s\n" "$AUTHORIZED_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

ssh-keygen -A

cat >/etc/ssh/sshd_config.d/runpod.conf <<EOF
PermitRootLogin yes
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
UsePAM no
AuthorizedKeysFile .ssh/authorized_keys
PidFile /var/run/sshd.pid
EOF

echo "[bootstrap] starting sshd"
/usr/sbin/sshd

echo "[bootstrap] verifying sshd"
ss -tlnp | grep ":22" || (echo "[bootstrap] sshd failed to listen on 22" && exit 1)

echo "[bootstrap] python version"
python --version
pip --version

echo "[bootstrap] cloning/updating repo"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$GIT_REPO" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch --all --tags
fi

cd "$REPO_DIR"

if [ -n "${GIT_REF:-}" ]; then
  git checkout "$GIT_REF"
fi

echo "[bootstrap] upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[bootstrap] installing project dependencies"
python -m pip install \
  "numpy<2" \
  scipy \
  scikit-learn \
  matplotlib \
  tqdm \
  tensorboard \
  torch-tb-profiler==0.4.3

echo "[bootstrap] environment values"
echo "RUNPOD_PUBLIC_IP=${RUNPOD_PUBLIC_IP:-}"
echo "RUNPOD_TCP_PORT_22=${RUNPOD_TCP_PORT_22:-}"
echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-}"

echo "[bootstrap] done"
tail -f /dev/null'"""

launch_vars = {
    "input": {
        "cloudType": "COMMUNITY",
        "gpuTypeId": "NVIDIA GeForce RTX 3090",
        "gpuCount": 1,
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "name": pod_name,
        "containerDiskInGb": 20,
        "volumeInGb": 20,
        "volumeMountPath": "/workspace",
        "ports": "22/tcp,6006/http",
        "env": [
            {"key": "GIT_REPO", "value": "https://github.com/16thomja/SINDy_Autoencoder_PyTorch_EP.git"},
            {"key": "GIT_REF", "value": "main"},
            {"key": "SSH_PUBLIC_KEY", "value": open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()},
        ],
        "dockerArgs": bootstrap_cmd,
    }
}

pod_data = run_query(launch_query, launch_vars)
pod_id = pod_data["podFindAndDeployOnDemand"]["id"]

print(f"Pod created: {pod_id}")

status_query = """
query GetPod($input: PodFilter) {
  pod(input: $input) {
    id
    desiredStatus
    runtime {
      uptimeInSeconds
      ports {
        ip
        privatePort
        publicPort
        isIpPublic
        type
      }
    }
  }
}
"""

while True:
    pod = run_query(status_query, {"input": {"podId": pod_id}})["pod"]
    runtime = pod.get("runtime")

    ssh_mapping = find_port_mapping(runtime, 22)
    tb_mapping = find_port_mapping(runtime, 6006)

    if ssh_mapping:
        public_ip = ssh_mapping["ip"]
        public_ssh_port = ssh_mapping["publicPort"]

        print("Pod is ready.")
        print(f"Public IP: {public_ip}")
        print(f"Public SSH port: {public_ssh_port}")
        print(f'SSH: ssh root@{public_ip} -p {public_ssh_port} -i ~/.ssh/id_ed25519')

        if ssh_mapping:
            public_ip = ssh_mapping["ip"]
            public_ssh_port = ssh_mapping["publicPort"]
            print(f"Public IP: {public_ip}")
            print(f"Public SSH port: {public_ssh_port}")
            print(f'SSH: ssh root@{public_ip} -p {public_ssh_port} -i ~/.ssh/id_ed25519')

            if tb_mapping:
                print(f'TensorBoard proxy/direct mapping: http://{tb_mapping["ip"]}:{tb_mapping["publicPort"]}')
            else:
                print("TensorBoard port mapping not available yet.")

            print("")
            print("Artifact sync command:")
            print(
                f'scripts/rsync_runpod_artifacts.sh {public_ip} {public_ssh_port} ~/.ssh/id_ed25519'
            )

            break

    print(
        f"Waiting... desiredStatus={pod.get('desiredStatus')}, "
        f"ssh_mapped={'yes' if ssh_mapping else 'no'}"
    )
    time.sleep(5)
