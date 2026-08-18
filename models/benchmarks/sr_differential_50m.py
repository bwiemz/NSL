#!/usr/bin/env python3
"""Roadmap item 6: the real-corpus SR-BF16 differential — F32 oracle vs
`--param-dtype bf16-sr` at Coder-50M.

The 2026-07-30 campaign (srbf16_campaign.py / SRBF16_TRAJECTORY_2026_07_30.md)
banked 500-step trajectories on a synthetic memorizable stream whose loss
collapses to ~0.01 — a regime where a rounding-induced quality delta cannot
show. This campaign trains one epoch over an 8.39M-token slice of the REAL
corpus and evaluates a held-out tail slice the training never touches.

Per seed and arm: N replicate runs on the IDENTICAL CSLA + weight-stream
residency schedule (the composition bf16-sr requires), so the comparison
isolates authoritative bf16 storage + the fused stochastically-rounded
AdamW step — the dtype, not the stack.

Replicates, NOT paired determinism: the first version of this harness
assumed `--deterministic` reproduces an arm bit-for-bit and used one
paired run per seed. The determinism control REFUTED that — same-seed
same-arm replicates diverge from the FIRST sampled step (with or without
dropout; ~0.1 VAL scatter over a 4096-update epoch), because the flag
covers RNG seeding and compile-time nondeterminism detection, not the
GPU backward's atomic accumulation order under this composition. So the
verdict is statistical: pooled per-arm VAL means, judged against the
measured replicate + seed scatter.

Usage (from the repo root, needs the local corpus + a CUDA nsl build):
    python models/benchmarks/sr_differential_50m.py \
        --nsl <path-to-cuda-nsl-binary> [--seeds 3]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROGRAM = REPO / "models/coder50m/sr_differential.nsl"
CORPUS = REPO / "data/tokens/train_new.bin"
TRAIN_SLICE = REPO / "data/tokens/sr_train_slice.bin"
VAL_SLICE = REPO / "data/tokens/sr_val_slice.bin"
TRAIN_TOKENS = 8192 * 1024  # one epoch = 8192 micro-steps of seq 1024
VAL_TOKENS = 513 * 1024  # 512 val batches (+1 sample slack for drop_last)

BASE_FLAGS = [
    "--source-ad",
    "--deterministic",
    "--checkpoint-blocks",
    "--layerwise-accum",
    "--weight-stream",
]

LOSS_RE = re.compile(r"^(?:tensor\(\[)?([0-9]+\.[0-9]+(?:e-?[0-9]+)?)\]?\)?$")
SR_TEARDOWN_RE = re.compile(r"\[sr-bf16\] teardown: (\d+) bf16-authoritative param\(s\)")


@dataclass
class Run:
    label: str
    losses: list[float]
    val_loss: float
    wall_s: float


def materialize_slices() -> None:
    if not CORPUS.exists():
        sys.exit(f"missing local corpus: {CORPUS} (not committed; see pretrain_prod.nsl)")
    data = CORPUS.read_bytes()
    n_tok = len(data) // 2
    val_start = n_tok - VAL_TOKENS
    assert val_start > TRAIN_TOKENS, "held-out slice would overlap the training slice"
    want_train = TRAIN_TOKENS * 2
    if not (TRAIN_SLICE.exists() and TRAIN_SLICE.stat().st_size == want_train):
        TRAIN_SLICE.write_bytes(data[:want_train])
        print(f"[slice] wrote {TRAIN_SLICE} ({want_train} bytes)")
    want_val = VAL_TOKENS * 2
    if not (VAL_SLICE.exists() and VAL_SLICE.stat().st_size == want_val):
        VAL_SLICE.write_bytes(data[val_start * 2 :])
        print(f"[slice] wrote {VAL_SLICE} ({want_val} bytes)")


def run_arm(nsl: str, label: str, seed: int, sr: bool, _retried: bool = False) -> Run:
    cmd = [nsl, "run", "--seed", str(seed), *BASE_FLAGS]
    if sr:
        cmd += ["--param-dtype", "bf16-sr"]
    cmd.append(PROGRAM.name)
    env = dict(os.environ)
    env.setdefault("NSL_STDLIB_PATH", str(REPO / "stdlib"))
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, cwd=PROGRAM.parent, env=env, capture_output=True, text=True, timeout=7200
    )
    wall = time.monotonic() - t0
    out, err = proc.stdout, proc.stderr
    if proc.returncode != 0 or "SR_DIFFERENTIAL_COMPLETE" not in out:
        # The display watchdog on a desktop GPU can kill a kernel under
        # transient contention (CUDA_ERROR_LAUNCH_TIMEOUT); the identical
        # arm reruns clean. One retry for exactly that signature — anything
        # else, or a second trip, is a real failure.
        if "CUDA_ERROR_LAUNCH_TIMEOUT" in err and not _retried:
            print(f"[{label}] transient CUDA launch timeout — retrying once")
            return run_arm(nsl, label, seed, sr, _retried=True)
        sys.exit(f"[{label}] run failed (exit {proc.returncode}):\n{err[-3000:]}")
    if "source AD extraction failed" in err:
        sys.exit(f"[{label}] fell back to tape AD — the campaign compares source-AD arms")

    # Honesty: an SR arm that silently ran f32 (or vice versa) poisons the
    # whole comparison. The teardown counters are the runtime's own receipt.
    m = SR_TEARDOWN_RE.search(err)
    if sr:
        if not m or int(m.group(1)) == 0:
            sys.exit(f"[{label}] no [sr-bf16] teardown counters — the SR arm ran f32?\n{err[-1500:]}")
    elif "[sr-bf16]" in err:
        sys.exit(f"[{label}] f32 oracle arm shows [sr-bf16] markers")

    losses: list[float] = []
    in_stream = False
    val_loss = None
    lines = iter(out.splitlines())
    for line in lines:
        s = line.strip()
        if s == "LOSS_STREAM_BEGIN":
            in_stream = True
        elif s == "LOSS_STREAM_END":
            in_stream = False
        elif s == "VAL_LOSS":
            nxt = next(lines, "").strip()
            mm = LOSS_RE.match(nxt)
            if mm:
                val_loss = float(mm.group(1))
        elif in_stream:
            mm = LOSS_RE.match(s)
            if mm:
                losses.append(float(mm.group(1)))
    # on_step prints step and loss every 8 micro-steps; the step numbers are
    # integers and fail LOSS_RE's decimal-point requirement, so `losses`
    # holds only losses.
    if len(losses) < 900 or val_loss is None:
        sys.exit(f"[{label}] incomplete streams: {len(losses)} losses, val={val_loss}")
    bad = [l for l in losses if not (0.0 < l < 20.0)]
    if bad:
        sys.exit(f"[{label}] non-finite/absurd losses: {bad[:5]}")
    tok_s = TRAIN_TOKENS / wall
    print(
        f"[{label}] {len(losses)} sampled losses  first={losses[0]:.6f}  "
        f"last={losses[-1]:.6f}  VAL={val_loss:.6f}  wall={wall:.0f}s  ~{tok_s:,.0f} tok/s"
    )
    return Run(label, losses, val_loss, wall)


def tail_mean(r: Run, n: int = 64) -> float:
    return sum(r.losses[-n:]) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsl", required=True, help="CUDA nsl binary")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--replicates", type=int, default=2)
    args = ap.parse_args()

    materialize_slices()
    runs: dict[str, list[Run]] = {"f32": [], "sr": []}
    for i in range(args.seeds):
        seed = 1000 + i
        print(f"\n=== seed {seed} ===")
        for rep in range(args.replicates):
            runs["f32"].append(run_arm(args.nsl, f"seed{seed}/f32-r{rep}", seed, sr=False))
            runs["sr"].append(run_arm(args.nsl, f"seed{seed}/bf16sr-r{rep}", seed, sr=True))

    def stats(vals: list[float]) -> tuple[float, float]:
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
        return mean, (var**0.5)

    print("\n| arm | n | VAL mean | VAL sd | VAL min..max | tail64 mean |")
    print("|-----|---|----------|--------|--------------|-------------|")
    summary = {}
    for arm in ("f32", "sr"):
        vals = [r.val_loss for r in runs[arm]]
        tails = [tail_mean(r) for r in runs[arm]]
        m, sd = stats(vals)
        tm, _ = stats(tails)
        summary[arm] = (m, sd, len(vals))
        print(
            f"| {arm} | {len(vals)} | {m:.6f} | {sd:.6f} "
            f"| {min(vals):.6f}..{max(vals):.6f} | {tm:.6f} |"
        )

    (mf, sdf, nf), (ms, sds, ns) = summary["f32"], summary["sr"]
    delta = ms - mf
    # Welch standard error of the mean difference; 2×SE ≈ a 95% bound.
    se = (sdf**2 / nf + sds**2 / ns) ** 0.5
    print(f"\ndVAL (SR - F32) = {delta:+.6f}  (Welch SE {se:.6f}; 2SE bound ±{2*se:.6f})")
    for arm in ("f32", "sr"):
        vals = sorted(f"{r.val_loss:.4f}" for r in runs[arm])
        print(f"  {arm} VALs: {vals}")
    if abs(delta) <= 2 * se:
        print(
            "VERDICT: the SR-BF16 quality delta is statistically indistinguishable "
            f"from run noise at 50M/1-epoch (|{delta:+.4f}| <= 2SE {2*se:.4f})"
        )
    else:
        print(
            f"VERDICT: SR-BF16 shifts held-out loss by {delta:+.4f} "
            f"(beyond the 2SE noise bound {2*se:.4f}) — quantified above"
        )


if __name__ == "__main__":
    main()
