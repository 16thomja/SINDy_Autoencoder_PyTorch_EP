#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/rsync_runpod_artifacts.sh <public_ip> <public_ssh_port> <ssh_key_path>

Example:
  scripts/rsync_runpod_artifacts.sh 80.15.7.37 40490 ~/.ssh/id_ed25519

Env vars:
  RSYNC_DELETE   If "1", pass --delete to rsync (default: 0)
  RSYNC_DRY_RUN  If "1", pass --dry-run to rsync (default: 0)
EOF
}

PUBLIC_IP="${1:-}"
PUBLIC_SSH_PORT="${2:-}"
SSH_KEY_PATH="${3:-}"

if [[ -z "$PUBLIC_IP" ]]; then
  usage
  exit 2
fi

if [[ -z "$PUBLIC_SSH_PORT" ]]; then
  usage
  exit 2
fi

if [[ -z "$SSH_KEY_PATH" ]]; then
  usage
  exit 2
fi

if [[ ! -f "$SSH_KEY_PATH" ]]; then
  echo "ERROR: SSH key not found: $SSH_KEY_PATH" >&2
  exit 2
fi

LOCAL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_ARGS=(-avh --partial --progress)
if [[ "${RSYNC_DRY_RUN:-0}" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi
if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

SSH_OPTS=(
  -p "$PUBLIC_SSH_PORT"
  -i "$SSH_KEY_PATH"
  -o StrictHostKeyChecking=accept-new
)

REMOTE="root@${PUBLIC_IP}"

echo "[sync] local repo root: $LOCAL_REPO_ROOT"
echo "[sync] remote: ${REMOTE}:/workspace"
echo "[sync] ssh port: ${PUBLIC_SSH_PORT}"

echo "[sync] checking SSH connectivity"
ssh "${SSH_OPTS[@]}" "$REMOTE" "echo ok" >/dev/null

echo "[sync] checking remote rsync availability"
if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "command -v rsync >/dev/null"; then
  cat <<EOF
ERROR: rsync is not installed on the pod.

Install it on the pod (as root):
  apt-get update && apt-get install -y rsync
EOF
  exit 1
fi

RSYNC_SSH="ssh -p $PUBLIC_SSH_PORT -i $SSH_KEY_PATH -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=QUIET"

for d in experiments tb_runs trained_models; do
  echo "[sync] syncing $d/"
  mkdir -p "$LOCAL_REPO_ROOT/$d"
  rsync "${RSYNC_ARGS[@]}" -e "$RSYNC_SSH" \
    "${REMOTE}:/workspace/$d/" \
    "$LOCAL_REPO_ROOT/$d/"
done

echo "[sync] done"