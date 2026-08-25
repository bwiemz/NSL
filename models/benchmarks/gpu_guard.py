"""Refuse-to-start guard for GPU campaign scripts.

Python twin of ``scripts/gpu-guard.sh`` — same lock path, same threshold
variable, same posture. A campaign calls :func:`acquire_or_refuse` once at
startup; the flock is held on a module-global fd for the life of the process.

WHY: on 2026-08-19 a sweep printed "WARNING: GPU still at 24459 MiB" and ran
anyway — the measurement died at 0 steps in nsl_tensor_matmul and two sweep
arms were silently lost. A precondition that warns and proceeds converts a
detected bad state into corrupted data. This guard refuses: an absent
measurement is recoverable, a wrong one is not.

Environment (shared with the shell guard):
    NSL_GPU_LOCK                 lock path (default /tmp/nsl-gpu.lock — ONE
                                 fixed path per machine; an $XDG_RUNTIME_DIR
                                 default was rejected because it differs
                                 between an interactive shell and a runner
                                 service, splitting the "machine-wide" lock
                                 into two files that never exclude each other)
    NSL_GPU_GUARD_THRESHOLD_MIB  per-process refusal threshold, default 256
                                 (an idle Wayland compositor holds ~125 MiB as
                                 a compute app, so zero would always refuse)
    NSL_GPU_GUARD=0              skip the guard (debugging the guard itself)
    NSL_GPU_LOCK_HELD            set by an enclosing guard to "pid:starttime";
                                 both the lock and the busy check are skipped
                                 when that exact process is still alive — by
                                 then our own allocations would read as
                                 foreign and self-refuse. The starttime half
                                 is what makes a leaked value harmless: a
                                 recycled pid has a different starttime, so a
                                 stale variable cannot silently disarm the
                                 guard.
"""

from __future__ import annotations

import datetime
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Keeps the fd — and therefore the flock — alive until the process exits.
_lock_file = None


def lock_path() -> Path:
    return Path(os.environ.get("NSL_GPU_LOCK", "/tmp/nsl-gpu.lock"))


def _starttime(pid: int) -> str | None:
    """Kernel start time of `pid` (field 22 of /proc/<pid>/stat), or None if
    the process is gone. The comm field can contain spaces and parens, so
    fields are counted after the LAST ')' — naive whitespace splitting is
    wrong for a process named e.g. 'a b)c'."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    rest = stat.rsplit(")", 1)[-1].split()
    return rest[19] if len(rest) > 19 else None


def _held_token() -> str:
    st = _starttime(os.getpid())
    return f"{os.getpid()}:{st or '0'}"


def _enclosing_guard_alive() -> bool:
    held = os.environ.get("NSL_GPU_LOCK_HELD", "")
    if ":" not in held:
        return False
    pid_s, _, start = held.partition(":")
    try:
        pid = int(pid_s)
    except ValueError:
        return False
    return start != "" and _starttime(pid) == start


def _busy_offenders(threshold_mib: int) -> list[str]:
    """Compute-app rows over threshold. A non-numeric memory reading counts as
    OVER: the guard cannot prove the device is idle, and cannot-prove refuses
    for the same reason a missing nvidia-smi does."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        # Fail CLOSED. An erroring nvidia-smi (a wedged driver, a post-Xid
        # device) correlates with exactly the abnormal states this guard
        # exists to catch; treating its silence as "idle" would pass the one
        # situation most likely to be busy-and-broken.
        sys.exit(
            f"gpu-guard: REFUSING — nvidia-smi exited {out.returncode}; "
            "cannot prove the device is idle."
        )
    offenders: list[str] = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not line.strip():
            continue
        try:
            over = int(parts[-1]) > threshold_mib
        except ValueError:
            over = True
        if over:
            offenders.append(line.strip())
    return offenders


def acquire_or_refuse(label: str) -> None:
    """Take the machine-wide GPU lock and verify the device is idle, or exit(1).

    Never warns-and-proceeds. Call once, before any GPU work is launched.
    """
    global _lock_file
    if os.environ.get("NSL_GPU_GUARD", "1") == "0":
        print(f"[{label}] gpu-guard: NSL_GPU_GUARD=0 — guard skipped", file=sys.stderr)
        return
    if _enclosing_guard_alive():
        # Never skip silently: a skip that prints nothing is indistinguishable
        # from a guard that was never wired.
        print(
            f"[{label}] gpu-guard: enclosing guard "
            f"({os.environ['NSL_GPU_LOCK_HELD']}) holds the lock — ok",
            file=sys.stderr,
        )
        return
    if shutil.which("nvidia-smi") is None:
        sys.exit(f"[{label}] gpu-guard: REFUSING — nvidia-smi not found; cannot prove the device is idle.")

    path = lock_path()
    # Append mode: "w" would truncate the holder metadata of a lock this
    # process is about to FAIL to take.
    f = path.open("a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip()
        f.close()
        sys.exit(
            f"[{label}] gpu-guard: REFUSING — another guarded GPU run holds {path}:\n"
            + "\n".join(f"    {ln}" for ln in holder.splitlines())
            + "\nWait for it to finish; two concurrent runs corrupt both measurements."
        )
    _lock_file = f
    f.truncate(0)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    f.write(f"pid\t{os.getpid()}\nstarted\t{started}\ncommand\t{label}\n")
    f.flush()

    threshold = int(os.environ.get("NSL_GPU_GUARD_THRESHOLD_MIB", "256"))
    offenders = _busy_offenders(threshold)
    if offenders:
        lines = "\n".join(f"    pid, process, MiB: {o}" for o in offenders)
        sys.exit(
            f"[{label}] gpu-guard: REFUSING — the device is busy. "
            f"Compute process(es) over {threshold} MiB:\n{lines}\n"
            "A measurement started now would be corrupted by the resident allocation.\n"
            "If this is an orphaned run, kill its process GROUP (kill -TERM -- -<pgid>):\n"
            "'nsl run' execs a child at /tmp/nsl_run_<pid>/ that outlives its parent."
        )
    # Children we spawn must not re-lock against us or self-refuse on our own
    # allocations. pid:starttime, not a bare pid — see the header.
    os.environ["NSL_GPU_LOCK_HELD"] = _held_token()
