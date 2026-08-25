#!/usr/bin/env python3
"""Instrumented residency/capacity probe for one matrix cell.

`matrix_bench.py` answers "how fast" — it terminates each run after
`--max-steps`, so the runtime's `atexit` reporters (the `[weight-stream]`
counter line, the residency posture) never fire, and it parses only the
`step=` header of `[gpu-mem]`, discarding the per-surface attribution that
says WHERE the peak went.

This script answers "where do the bytes live". It builds the same programs
`matrix_bench` builds — same generator, same arms — against a SHORT token
stream so the run reaches its own end and exits cleanly, then captures the
whole stderr and reports:

  * `[weight-stream]` transfer counters + residency posture (pinned vs
    streamed), i.e. whether any parameter byte crossed PCIe;
  * `[gpu-mem]` peak allocator/driver bytes;
  * the per-surface table — weights / optim_m / optim_v / m_partial / grads /
    activations / attn_workspace — which is the only output that identifies
    the NEXT capacity bottleneck rather than just reporting that the previous
    one is gone. Two forms, both captured: the per-step `surfaces:` line
    (current/peak) from `NSL_DEBUG_MEM_ALL`, and the `at-global-peak` column
    of the end-of-run summary, which needs `NSL_MEMSTATS=1`.

An arm that OOMs is a RESULT here, not a failed run: the OOM panic prints the
same surface table, so running an arm one microbatch above its ceiling is the
sharpest available answer to "what binds now". The stderr is kept either way.

Usage:
  python3 models/benchmarks/residency_probe.py \
      --nsl <path> --scale 1b --seq 2048 \
      --arm layerwise_resident_srbf16:1 --arm layerwise_policy:1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matrix_bench import (  # noqa: E402
    ARMS,
    REPO,
    SCALES,
    build_binary,
    build_program,
)

import gpu_guard  # noqa: E402

WS_RE = re.compile(r"^\[weight-stream\] .*$", re.M)
SR_RE = re.compile(r"^\[sr-bf16\] .*$", re.M)
GPU_MEM_RE = re.compile(
    r"\[gpu-mem\]\s+step=(\d+)\s+driver=(\d+)MB\s+alloc=(\d+)MB\s+reserved=(\d+)MB"
)
SURFACE_RE = re.compile(r"^\[gpu-mem\]\s+surfaces:(.*)$", re.M)
OOM_RE = re.compile(r"out of memory", re.I)


#: How often the nvidia-smi sampler polls while a run is in flight.
SMI_INTERVAL_S = 0.1


def _smi_used_mb(pid: int) -> int:
    """VRAM nvidia-smi attributes to `pid`, in MB, or 0 if it is not listed.

    Per-PID, not whole-device: the box runs a compositor and whatever other
    session is building, so a device-wide reading would fold their bytes into
    the arm's number.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            return int(parts[1]) if parts[1].isdigit() else 0
    return 0


def run_sampled(binary: Path, env: dict[str, str],
                timeout: int) -> tuple[int, str, str, int]:
    """Run `binary`, sampling its VRAM, and return (rc, stdout, stderr, peak_mb).

    The reason this exists: `[gpu-mem]` prints at STEP BOUNDARIES, so both the
    `alloc=` and `driver=` columns miss whatever the step transiently peaked
    at in between. At 1B@2048 that gap is roughly a factor of two, which is
    the difference between "13 GB of a 32 GB card, tons of room" and the
    measured microbatch ceiling of 1.

    Sampling can still MISS a peak narrower than the interval, so the number
    it yields is a LOWER BOUND on the true peak — never report it as the peak
    itself.
    """
    import threading

    proc = subprocess.Popen(
        [str(binary)], cwd=binary.parent, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, errors="replace", env=env,
    )
    peak = 0
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(SMI_INTERVAL_S):
            peak = max(peak, _smi_used_mb(proc.pid))

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    finally:
        stop.set()
        t.join(timeout=5)
    return proc.returncode, stdout, stderr, peak


def parse_arm_spec(spec: str) -> tuple[str, int]:
    """`<arm>:<microbatch>` — the batch is explicit here on purpose.

    `matrix_bench`'s probe returns the largest batch that fits EVERY arm in a
    cell, which is not the batch any single arm's capacity question is about.
    """
    if ":" not in spec:
        raise SystemExit(f"--arm wants <key>:<microbatch>, got {spec!r}")
    key, _, batch = spec.partition(":")
    if key not in ARMS:
        raise SystemExit(f"unknown arm {key!r}; known: {sorted(ARMS)}")
    if not batch.isdigit() or int(batch) < 1:
        raise SystemExit(f"microbatch must be a positive integer, got {batch!r}")
    return key, int(batch)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsl", type=Path, default=REPO / "target/release/nsl")
    ap.add_argument("--scale", default="1b", choices=sorted(SCALES))
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="KEY:MICROBATCH",
                    help="repeatable; each runs in its own process")
    ap.add_argument("--out", type=Path, default=REPO / "target/residency_probe")
    ap.add_argument("--tokens", type=Path, default=None,
                    help="token stream; a SHORT one by default so the run "
                         "ends on its own and atexit reporters fire")
    ap.add_argument("--token-count", type=int, default=120_000)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    # Refuse-to-start GPU guard — a busy device or a concurrent guarded run
    # aborts here instead of corrupting the measurement (see gpu_guard.py).
    gpu_guard.acquire_or_refuse("residency_probe")

    args.out = args.out.resolve()
    args.nsl = args.nsl.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.nsl.exists():
        sys.exit(f"compiler not found at {args.nsl}")

    # Validate every spec BEFORE generating tokens or compiling: a typo in the
    # third `--arm` should not cost a token stream and two 1B builds first.
    specs = [parse_arm_spec(s) for s in args.arm]

    tokens = (args.tokens or (args.out / "tokens_short.bin")).resolve()
    if not tokens.exists():
        print(f"generating {args.token_count} tokens at {tokens}", flush=True)
        subprocess.run(
            [sys.executable, str(REPO / "models/benchmarks/gen_tokens.py"),
             str(tokens), "--tokens", str(args.token_count)],
            check=True,
        )

    scale = SCALES[args.scale]
    for key, batch in specs:
        arm = ARMS[key]
        print(f"\n=== {key} @ {args.scale} seq {args.seq} microbatch {batch} ===",
              flush=True)
        work = args.out / f"{args.scale}_s{args.seq}_b{batch}_{key}"
        prog = build_program(work, scale, tokens, batch, args.seq, arm)
        binary = build_binary(args.nsl, prog, arm, args.timeout)
        if binary is None:
            print("  BUILD REFUSED — see build_stderr.txt", flush=True)
            continue

        env = dict(os.environ)
        env.update(dict(arm.env))
        # The two reporters this script exists for. NSL_DEBUG_MEM_ALL because
        # the memory dump early-returns after step 5 without it.
        env["NSL_WS_COUNTER"] = "1"
        env["NSL_DEBUG_MEM_ALL"] = "1"
        # The end-of-run summary carries the `at-global-peak` column — the
        # per-step `surfaces:` line only has current/peak, and a surface's own
        # peak can occur when the GLOBAL peak was somewhere else entirely.
        env["NSL_MEMSTATS"] = "1"
        rc, stdout, stderr, smi_peak = run_sampled(binary, env, args.timeout)
        log = work / "run_stderr.txt"
        log.write_text(stderr, encoding="utf-8")
        (work / "run_stdout.txt").write_text(stdout, encoding="utf-8")

        if OOM_RE.search(stderr):
            print("  OOM", flush=True)
        print(f"  exit={rc}  stderr -> {log}", flush=True)

        peak_alloc = peak_driver = 0
        peak_step = -1
        for m in GPU_MEM_RE.finditer(stderr):
            if int(m.group(3)) > peak_alloc:
                peak_alloc, peak_step = int(m.group(3)), int(m.group(1))
            peak_driver = max(peak_driver, int(m.group(2)))
        print(f"  peak alloc {peak_alloc} MB (step {peak_step})  "
              f"peak driver {peak_driver} MB  [step-boundary samples]",
              flush=True)
        print(f"  nvidia-smi peak for THIS pid: {smi_peak} MB "
              f"[{SMI_INTERVAL_S * 1000:.0f} ms sampling]", flush=True)

        for line in SR_RE.findall(stderr)[:4]:
            print(f"  {line.strip()}", flush=True)
        for line in WS_RE.findall(stderr):
            print(f"  {line.strip()}", flush=True)
        surfaces = SURFACE_RE.findall(stderr)
        if surfaces:
            print(f"  surfaces (last step):{surfaces[-1]}", flush=True)

        # Persist the parsed numbers, not just the console. The nvidia-smi peak
        # exists NOWHERE in the saved stderr — it is sampled by this process —
        # so without this the headline capacity figure lives only in whatever
        # terminal happened to be scrolled back, and a document citing it is
        # unreproducible.
        (work / "probe_result.json").write_text(
            json.dumps(
                {
                    "arm": key,
                    "scale": args.scale,
                    "seq": args.seq,
                    "microbatch": batch,
                    "exit_code": rc,
                    "oom": bool(OOM_RE.search(stderr)),
                    "step_boundary_peak_alloc_mb": peak_alloc,
                    "step_boundary_peak_driver_mb": peak_driver,
                    "nvidia_smi_peak_mb_lower_bound": smi_peak,
                    "smi_sample_interval_s": SMI_INTERVAL_S,
                    "surfaces_last_step": surfaces[-1].strip() if surfaces else "",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # The NSL_MEMSTATS block carries the true high-water marks and the
        # at-global-peak column — the whole reason this script sets that env.
        # On an OOM the same table comes out of the panic handler instead,
        # which is the most informative output the failing arm produces.
        for marker in ("[nsl] GPU Memory Summary", "Allocator surfaces"):
            at = stderr.find(marker)
            if at < 0:
                continue
            block = stderr[at:].splitlines()[:34]
            print(f"  --- {marker} ---", flush=True)
            for line in block:
                print(f"  {line}", flush=True)
            break


if __name__ == "__main__":
    main()
