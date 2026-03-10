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

bootstrap_cmd = r"""bash -lc 'set -euo pipefail; export DEBIAN_FRONTEND=noninteractive; apt-get update; apt-get install -y --no-install-recommends bash git curl ca-certificates bzip2; rm -rf /var/lib/apt/lists/*; if [ ! -d /workspace/repo/.git ]; then git clone "$GIT_REPO" /workspace/repo; else git -C /workspace/repo fetch --all --tags; fi; cd /workspace/repo; if [ -n "${GIT_REF:-}" ]; then git checkout "$GIT_REF"; fi; chmod +x scripts/bootstrap_runpod.sh; exec bash scripts/bootstrap_runpod.sh'"""

launch_vars = {
    "input": {
        "cloudType": "COMMUNITY",
        "gpuTypeId": "NVIDIA GeForce RTX 3090",
        "gpuCount": 1,
        "imageName": "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        "name": "SINDy",
        "containerDiskInGb": 20,
        "volumeInGb": 20,
        "volumeMountPath": "/workspace",
        "ports": "22/tcp,8888/http",
        "env": [
            {"key": "GIT_REPO", "value": "https://github.com/16thomja/SINDy_Autoencoder_PyTorch_EP.git"},
            {"key": "GIT_REF", "value": "main"},
            {"key": "CONDA_ENV_FILE", "value": "/workspace/repo/environment.cuda.yml"},
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

    print(f"Waiting... desiredStatus={pod.get('desiredStatus')}")
    time.sleep(5)