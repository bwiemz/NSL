#!/usr/bin/env bash
#
# gpu-guard.sh — mutual exclusion + busy-device refusal for GPU workloads
# (roadmap item 14: "scripts refuse concurrent GPU runs").
#
# WHY THIS EXISTS: on 2026-08-19 an LR sweep printed
# "WARNING: GPU still at 24459 MiB" and started its measurement anyway. The
# run aborted at 0 steps inside nsl_tensor_matmul, three times over, and two
# whole sweep arms were silently lost — the failure read as a compute bug when
# it was a busy device. A precondition that warns and proceeds converts a
# correctly-detected bad state into corrupted data plus a log line nobody
# reads. This guard REFUSES: an absent measurement is recoverable, a wrong one
# is not.
#
# The device can be busy in two distinct ways, and the guard covers both:
#
#   - a COOPERATING run (another tier/campaign started through this guard):
#     excluded by a non-blocking flock on a fixed path, held for the whole
#     workload lifetime;
#   - a NON-cooperating process: caught by the compute-app check. The usual
#     offender is an orphaned `nsl run` child — `nsl run` execs the compiled
#     program at /tmp/nsl_run_<pid>/, so killing the `nsl` parent leaves the
#     child holding its ~22 GB. The device then looks free in `ps` but not in
#     `nvidia-smi --query-compute-apps`.
#
# The workload runs under `setsid`, in its OWN process group, and TERM/INT
# are forwarded to the whole group — so killing the guard can never orphan a
# child on the device (the mechanism that created the incident above).
#
# Subcommands:
#   check              refuse (exit 1) if any compute process holds more than
#                      NSL_GPU_GUARD_THRESHOLD_MIB of VRAM. No lock taken.
#   run -- CMD ARG...  flock (exit 1 if another guarded run holds it), then
#                      check, then run CMD in its own process group. Exits
#                      with CMD's status.
#   path               print the lock path (for non-bash callers; the python
#                      twin models/benchmarks/gpu_guard.py uses the same path
#                      and threshold variables).
#
# Environment:
#   NSL_GPU_LOCK                 lock path. Default: $XDG_RUNTIME_DIR/nsl-gpu.lock,
#                                falling back to /tmp/nsl-gpu.lock. One path per
#                                machine on purpose — the GPU is one shared
#                                resource, so the lock must be too.
#   NSL_GPU_GUARD_THRESHOLD_MIB  per-process refusal threshold (default 256).
#                                Not zero: a Wayland desktop's compositor holds
#                                ~125 MiB as a compute app while completely
#                                idle (measured: kwin_wayland, 125 MiB), so a
#                                zero threshold would refuse on every desktop
#                                boot. 256 clears the compositor and still
#                                catches any real training remnant by orders
#                                of magnitude.
#   NSL_GPU_GUARD=0              skip the guard entirely. For debugging the
#                                guard itself — NOT for getting past a refusal:
#                                the refusal means a measurement made now would
#                                be untrustworthy.
#   NSL_GPU_LOCK_HELD            internal. Set to the guard's pid for the
#                                wrapped workload, so a guarded script invoking
#                                another guarded script does not deadlock on
#                                the lock it already holds (and does not
#                                re-run the busy check against its own
#                                allocations).
#
# A non-numeric memory reading (some platforms report "[N/A]" per process) is
# treated as OVER threshold, not under: the guard cannot prove the device is
# idle, and "cannot prove idle" refuses for the same reason "no nvidia-smi"
# does. NSL_GPU_GUARD=0 is the operator override for platforms where that
# reading never becomes numeric.

set -euo pipefail

LOCK="${NSL_GPU_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/nsl-gpu.lock}"
THRESHOLD_MIB="${NSL_GPU_GUARD_THRESHOLD_MIB:-256}"

# True when an enclosing guard already holds the lock and is still alive. Both
# the lock AND the busy check are skipped then: the outer guard performed both
# before any GPU work started, and by the time a nested invocation runs, the
# workload's own allocations would read as "foreign" and self-refuse.
enclosing_guard_alive() {
    [[ -n "${NSL_GPU_LOCK_HELD:-}" ]] && kill -0 "${NSL_GPU_LOCK_HELD}" 2>/dev/null
}

busy_check() {
    if [[ "${NSL_GPU_GUARD:-1}" == "0" ]]; then
        echo "gpu-guard: NSL_GPU_GUARD=0 — busy check skipped" >&2
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "gpu-guard: REFUSING — nvidia-smi not found; cannot prove the device is idle." >&2
        exit 1
    fi
    local rows offenders
    rows="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
            --format=csv,noheader,nounits 2>/dev/null || true)"
    # Third field over threshold OR not a number (see header). `$3 + 0 != $3`
    # is the numeric test: "[N/A]" fails it, "22000" passes it.
    offenders="$(awk -F', ' -v t="${THRESHOLD_MIB}" '
        NF >= 3 && (($3 + 0) > t || ($3 + 0) != $3)' <<< "${rows}")"
    if [[ -n "${offenders}" ]]; then
        {
            echo "gpu-guard: REFUSING — the device is busy. Compute process(es) over ${THRESHOLD_MIB} MiB:"
            sed 's/^/    pid, process, MiB: /' <<< "${offenders}"
            echo "A measurement started now would be corrupted by the resident allocation."
            echo "If this is an orphaned run, kill its process GROUP — kill -TERM -- -<pgid> —"
            echo "not just the visible pid: 'nsl run' execs a child at /tmp/nsl_run_<pid>/ that"
            echo "outlives its parent and keeps the VRAM."
        } >&2
        exit 1
    fi
}

cmd_check() {
    if enclosing_guard_alive; then
        echo "gpu-guard: enclosing guard (pid ${NSL_GPU_LOCK_HELD}) already checked — ok" >&2
        return 0
    fi
    busy_check
    echo "gpu-guard: device idle (no compute process over ${THRESHOLD_MIB} MiB)"
}

cmd_run() {
    if [[ "${1:-}" != "--" ]]; then
        echo "gpu-guard: usage: gpu-guard.sh run -- CMD [ARG...]" >&2
        exit 2
    fi
    shift
    if [[ $# -eq 0 ]]; then
        echo "gpu-guard: run needs a command after --" >&2
        exit 2
    fi

    if ! enclosing_guard_alive && [[ "${NSL_GPU_GUARD:-1}" != "0" ]]; then
        # Append mode: opening with '>' would truncate the holder metadata of
        # a lock we are about to FAIL to take.
        exec 9>>"${LOCK}"
        if ! flock -n 9; then
            {
                echo "gpu-guard: REFUSING — another guarded GPU run holds ${LOCK}:"
                sed 's/^/    /' "${LOCK}" 2>/dev/null || true
                echo "Wait for it to finish; two concurrent runs corrupt both measurements."
            } >&2
            exit 1
        fi
        # Lock is held on fd 9 (the inode), so rewriting the file's content
        # does not release it.
        printf 'pid\t%s\nstarted\t%s\ncommand\t%s\n' \
            "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "${LOCK}"
        busy_check
    fi

    export NSL_GPU_LOCK_HELD="$$"

    # Own process group via setsid, signals forwarded to the GROUP. `wait`
    # returns early when a trapped signal arrives, so loop until the child is
    # actually gone and report its real status.
    setsid "$@" &
    local child=$!
    forward() { kill -TERM -- "-${child}" 2>/dev/null || true; }
    trap forward TERM INT
    local rc=0
    wait "${child}" && rc=0 || rc=$?
    while kill -0 "${child}" 2>/dev/null; do
        wait "${child}" && rc=0 || rc=$?
    done
    exit "${rc}"
}

case "${1:-}" in
    check) cmd_check ;;
    run)   shift; cmd_run "$@" ;;
    path)  printf '%s\n' "${LOCK}" ;;
    -h|--help) sed -n '2,76p' "$0" | sed 's/^# \{0,1\}//' ;;
    # Anything else exits non-zero: a typo'd subcommand in a CI line must not
    # read as a green check that did nothing (same posture as gpu-cert.sh).
    *) echo "gpu-guard: unknown subcommand: ${1:-<none>} (want check|run|path)" >&2; exit 2 ;;
esac
