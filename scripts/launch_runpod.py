import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("RUNPOD_API_KEY")
URL = "https://api.runpod.io/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

if not API_KEY:
    raise RuntimeError("RUNPOD_API_KEY is not set")


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
  git \
  ca-certificates \
  curl \
  ffmpeg
rm -rf /var/lib/apt/lists/*

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

echo "[bootstrap] done"
tail -f /dev/null'"""

launch_vars = {
    "input": {
        "cloudType": "COMMUNITY",
        "gpuTypeId": "NVIDIA GeForce RTX 3090",
        "gpuCount": 1,
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "name": "SINDy",
        "containerDiskInGb": 20,
        "volumeInGb": 20,
        "volumeMountPath": "/workspace",
        "env": [
            {"key": "GIT_REPO", "value": "https://github.com/16thomja/SINDy_Autoencoder_PyTorch_EP.git"},
            {"key": "GIT_REF", "value": "main"},
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

    if runtime is not None:
        print("Pod is ready.")
        print("TensorBoard tunnel:")
        print("  ssh -L 6006:127.0.0.1:6006 <pod_ip> <ssh_key>")
        print("  open http://localhost:6006")
        break

    print(f"Waiting... desiredStatus={pod.get('desiredStatus')}")
    time.sleep(5)
