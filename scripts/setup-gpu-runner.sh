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
  # --fail: without it curl writes the HTTP error page to runner.tar.gz and
  # the failure surfaces as a bogus `tar` error two lines later.
  curl -sL --fail --show-error --retry 3 -o "${DIR}/runner.tar.gz" \
    "https://github.com/actions/runner/releases/download/v${VER}/actions-runner-linux-x64-${VER}.tar.gz"
  tar xzf "${DIR}/runner.tar.gz" -C "${DIR}"
  rm "${DIR}/runner.tar.gz"
fi

cd "${DIR}"
if [[ -f .runner ]]; then
  echo "runner already configured ($(grep -o '"agentName": *"[^"]*"' .runner || true))"
else
  TOKEN="$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" -q .token)"
  # --replace: NAME is fixed, so a server-side registration left over from a
  # previous attempt (or a re-imaged box) otherwise makes config.sh refuse
  # with "already exists" and there is no way forward from this script.
  ./config.sh --url "https://github.com/${REPO}" --token "${TOKEN}" \
    --name "${NAME}" --labels "${LABELS}" --work _work --unattended --replace
fi

# Persistent user service (survives logout via linger).
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_FILE="${UNIT_DIR}/nsl-gpu-runner.service"
mkdir -p "${UNIT_DIR}"
# The lane's TMPDIR must exist before the service starts; /tmp is a 31G tmpfs
# here and a full one surfaces as linker "No space left" errors that read like
# compiler regressions.
mkdir -p "${HOME}/.cache/nsl-gpucert-tmp"

# Written only if absent, and otherwise diffed. This used to be an
# unconditional truncating `cat >`, which destroyed any hand-added
# `Environment=` line every time the script was re-run — so no local fix to
# the unit could survive, including the PATH below.
NEW_UNIT="$(cat << UNIT
[Unit]
Description=GitHub Actions self-hosted runner (NSL gpu-cert lane)
Wants=network-online.target
After=network-online.target

[Service]
ExecStart=${DIR}/run.sh
WorkingDirectory=${DIR}
Restart=always
RestartSec=30
# Actions steps run as NON-LOGIN shells: ~/.cargo/env and ~/.profile are never
# sourced, and there is no ~/.config/environment.d/ on this box. Without an
# explicit PATH the lane cannot find cargo/rustc at all, and a missing ptxas
# makes scripts/gpu-cert.sh REFUSE outright rather than merely run slower.
Environment=PATH=%h/.cargo/bin:/opt/cuda/bin:/usr/local/bin:/usr/bin:/bin
# The lane builds with CUDA and writes warm caches under ~/.cache.
Environment=CUDA_PATH=/opt/cuda
# Keep cargo/scratch off the tmpfs.
Environment=TMPDIR=%h/.cache/nsl-gpucert-tmp

[Install]
WantedBy=default.target
UNIT
)"
if [[ ! -f "${UNIT_FILE}" ]]; then
  printf '%s\n' "${NEW_UNIT}" > "${UNIT_FILE}"
  echo "wrote ${UNIT_FILE}"
elif ! diff -q <(printf '%s\n' "${NEW_UNIT}") "${UNIT_FILE}" > /dev/null; then
  echo "note: ${UNIT_FILE} exists and differs from the template — LEAVING IT ALONE."
  echo "      Review and merge by hand if you want the template's settings:"
  diff -u "${UNIT_FILE}" <(printf '%s\n' "${NEW_UNIT}") || true
fi

systemctl --user daemon-reload
systemctl --user enable --now nsl-gpu-runner.service

# Linger is not optional: without it the service stops at logout and the
# 09:00 UTC schedule silently never fires. A warning here is what let the
# previous registration attempt look successful while doing nothing.
if ! loginctl enable-linger "${USER}"; then
  echo "ERROR: loginctl enable-linger ${USER} failed. The runner will stop at" >&2
  echo "       logout and the nightly will not run. Fix with a root run of:" >&2
  echo "         sudo loginctl enable-linger ${USER}" >&2
  exit 1
fi

# VERIFY, rather than announce. The previous version printed "registered and
# running" unconditionally — including on the run where config.sh had aborted
# under `set -e` and left no .runner at all, which is why this box sat
# unregistered while the script reported success.
echo "waiting for '${NAME}' to appear in the repo's runner list..."
for _ in $(seq 1 30); do
  if gh api "repos/${REPO}/actions/runners" -q '.runners[].name' 2>/dev/null \
     | grep -qx "${NAME}"; then
    echo "runner '${NAME}' registered with labels [${LABELS}] and online."
    gh api "repos/${REPO}/actions/runners" \
      -q '.runners[] | "  \(.name)  status=\(.status)  labels=\([.labels[].name]|join(","))"'
    exit 0
  fi
  sleep 2
done

echo "ERROR: '${NAME}' never appeared in repos/${REPO}/actions/runners." >&2
echo "       Diagnose with:" >&2
echo "         systemctl --user status nsl-gpu-runner" >&2
echo "         journalctl --user -u nsl-gpu-runner -n 50" >&2
exit 1
