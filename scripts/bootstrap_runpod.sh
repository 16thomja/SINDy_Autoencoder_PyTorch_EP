#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/workspace/repo
ENV_NAME=sindy-cuda
ENV_FILE="${CONDA_ENV_FILE:-/workspace/repo/environment.cuda.yml}"
MAMBA_ROOT_PREFIX=/workspace/.micromamba

echo "[bootstrap] starting"
export MAMBA_ROOT_PREFIX

echo "[bootstrap] installing micromamba"
mkdir -p /tmp/micromamba /usr/local/bin "$MAMBA_ROOT_PREFIX"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C /tmp/micromamba bin/micromamba
install -m 755 /tmp/micromamba/bin/micromamba /usr/local/bin/micromamba

echo "[bootstrap] micromamba version:"
micromamba --version

cd "$REPO_DIR"

echo "[bootstrap] using env file: $ENV_FILE"
if [ ! -f "$ENV_FILE" ]; then
  echo "[bootstrap] environment file not found: $ENV_FILE"
  exit 1
fi

if micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[bootstrap] updating existing environment: $ENV_NAME"
  micromamba env update -y -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "[bootstrap] creating environment: $ENV_NAME"
  micromamba create -y -n "$ENV_NAME" -f "$ENV_FILE"
fi

echo "[bootstrap] registering jupyter kernel"
micromamba run -n "$ENV_NAME" python -m ipykernel install \
  --user \
  --name "$ENV_NAME" \
  --display-name "$ENV_NAME" || true

echo "[bootstrap] done"
tail -f /dev/null