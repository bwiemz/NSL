#!/usr/bin/env bash
#
# setup-gpu-runner.sh — one-time registration of this machine as the
# self-hosted GPU runner that .github/workflows/gpu-cert.yml targets.
#
# Run it YOURSELF, once, on the GPU box:
#
#   scripts/setup-gpu-runner.sh
#
# It is deliberately not run by any automation: registering a runner grants
# github.com workflows code execution on this machine, which is a decision
# the machine's owner makes interactively.
#
# ── What you are agreeing to ─────────────────────────────────────────────
# This repository is PUBLIC. The gpu-cert workflow is trigger-locked to
# `schedule` (default-branch code only) and `workflow_dispatch`
# (write-access users only), so fork code cannot reach this runner through
# it. Two standing rules keep that true:
#   1. never add pull_request/push triggers to gpu-cert.yml;
#   2. in repo Settings -> Actions -> General, keep "Require approval for
#      all outside collaborators" selected.
#
# The runner runs as your user, unprivileged, from ~/actions-runner-nsl.
# Remove it any time with:
#   cd ~/actions-runner-nsl && ./config.sh remove --token \
#     "$(gh api -X POST repos/bwiemz/NSL/actions/runners/remove-token -q .token)"

set -euo pipefail

REPO="bwiemz/NSL"
DIR="${HOME}/actions-runner-nsl"
NAME="rtx4500-blackwell"
LABELS="gpu,cuda,sm120"

command -v gh >/dev/null || { echo "gh CLI required" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver visible" >&2; exit 1; }

# Download the runner if this box does not have it yet.
if [[ ! -x "${DIR}/config.sh" ]]; then
  VER="$(gh api repos/actions/runner/releases/latest -q .tag_name | sed 's/^v//')"
  echo "downloading actions-runner v${VER} into ${DIR}"
  mkdir -p "${DIR}"
  curl -sL -o "${DIR}/runner.tar.gz" \
    "https://github.com/actions/runner/releases/download/v${VER}/actions-runner-linux-x64-${VER}.tar.gz"
  tar xzf "${DIR}/runner.tar.gz" -C "${DIR}"
  rm "${DIR}/runner.tar.gz"
fi

cd "${DIR}"
if [[ -f .runner ]]; then
  echo "runner already configured ($(grep -o '"agentName": *"[^"]*"' .runner || true))"
else
  TOKEN="$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" -q .token)"
  ./config.sh --url "https://github.com/${REPO}" --token "${TOKEN}" \
    --name "${NAME}" --labels "${LABELS}" --work _work --unattended
fi

# Persistent user service (survives logout via linger).
UNIT_DIR="${HOME}/.config/systemd/user"
mkdir -p "${UNIT_DIR}"
cat > "${UNIT_DIR}/nsl-gpu-runner.service" << UNIT
[Unit]
Description=GitHub Actions self-hosted runner (NSL gpu-cert lane)
After=network-online.target

[Service]
ExecStart=${DIR}/run.sh
WorkingDirectory=${DIR}
Restart=always
RestartSec=30
# The lane builds with CUDA and writes warm caches under ~/.cache.
Environment=CUDA_PATH=/opt/cuda

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now nsl-gpu-runner.service
loginctl enable-linger "${USER}" || echo \
  "note: enable-linger failed — the runner stops at logout until a root run of: loginctl enable-linger ${USER}"

echo "runner '${NAME}' registered with labels [${LABELS}] and running."
echo "verify: gh api repos/${REPO}/actions/runners -q '.runners[].name'"
