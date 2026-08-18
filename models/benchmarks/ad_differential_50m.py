#!/usr/bin/env python3
"""Roadmap item 5: the 50M AD differential — tape-F32 vs source-F32 on GPU.

Measurement design (the one-seed pilot dictated it):
  - The GPU FORWARD is run-to-run deterministic: a same-seed source pair
    reproduces the step-0 loss to the last printed digit. Divergence
    appears only after optimizer updates (backward/optimizer atomics) and
    then amplifies chaotically — by step ~100 same-seed same-mode runs
    disagree by O(1), so a whole-trajectory max-gap envelope is vacuous.
  - Therefore three windows, each judged against BOTH same-mode control
    pairs (source-vs-source AND tape-vs-tape — per-mode runs cost the
    same chaos, so the mode comparison must not be judged against a
    source-only envelope):
      1. step-0: pure forward parity. Same-mode pairs must match exactly;
         tape-vs-source is allowed only the absolute floor (f32
         lowering/order differences between the two forwards).
      2. early window (first EARLY steps): gradient parity before chaos
         amplifies — this is the window that actually certifies the
         backward implementations agree.
      3. endpoint (mean of the last TAIL losses): both modes must learn
         to the same place.

Per seed: source×2, tape×2 (controls) — 4 runs. All runs share `--seed`
(distinct reproducible init per seed), shuffle=false, NSL_MATMUL_TF32=0,
dropout off, no fused LCE (ad_differential.nsl's header explains each
isolation).

Usage (from the repo root, needs the local corpus + a CUDA nsl build):
    python models/benchmarks/ad_differential_50m.py \
        --nsl <path-to-cuda-nsl-binary> [--seeds 3]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROGRAM = REPO / "models/coder50m/ad_differential.nsl"
CORPUS = REPO / "data/tokens/train_new.bin"
SLICE = REPO / "data/tokens/ad_differential_slice.bin"
SLICE_TOKENS = 128 * 1024  # 128 micro-steps of seq_len=1024
EARLY = 10  # steps before backward nondeterminism visibly amplifies
TAIL = 16  # endpoint window
# Loss prints: bare floats on CPU, tensor([x]) on GPU.
LOSS_RE = re.compile(r"^(?:tensor\(\[)?([0-9]+\.[0-9]+(?:e-?[0-9]+)?)\]?\)?$")


@dataclass
class Run:
    label: str
    losses: list[float]


def materialize_slice() -> None:
    """First SLICE_TOKENS u16 tokens of the production corpus."""
    if not CORPUS.exists():
        sys.exit(f"missing local corpus: {CORPUS} (not committed; see pretrain_prod.nsl)")
    want = SLICE_TOKENS * 2
    if SLICE.exists() and SLICE.stat().st_size == want:
        return
    SLICE.write_bytes(CORPUS.read_bytes()[:want])
    print(f"[slice] wrote {SLICE} ({want} bytes)")


def run_once(nsl: str, label: str, seed: int, source_ad: bool) -> Run:
    cmd = [nsl, "run", "--seed", str(seed)]
    if source_ad:
        cmd.append("--source-ad")
    cmd.append(PROGRAM.name)
    env = dict(os.environ)
    env["NSL_MATMUL_TF32"] = "0"
    env.setdefault("NSL_STDLIB_PATH", str(REPO / "stdlib"))
    proc = subprocess.run(
        cmd, cwd=PROGRAM.parent, env=env, capture_output=True, text=True, timeout=3600
    )
    out = proc.stdout
    if proc.returncode != 0 or "AD_DIFFERENTIAL_COMPLETE" not in out:
        sys.exit(f"[{label}] run failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}")
    losses: list[float] = []
    in_stream = False
    for line in out.splitlines():
        line = line.strip()
        if line == "LOSS_STREAM_BEGIN":
            in_stream = True
        elif line == "LOSS_STREAM_END":
            in_stream = False
        elif in_stream:
            m = LOSS_RE.match(line)
            if m:
                losses.append(float(m.group(1)))
    if len(losses) < 100:
        sys.exit(f"[{label}] expected ~128 losses, got {len(losses)}")
    bad = [l for l in losses if not (0.0 < l < 20.0)]
    if bad:
        sys.exit(f"[{label}] non-finite/absurd losses: {bad[:5]}")
    print(f"[{label}] {len(losses)} steps  first={losses[0]:.6f}  last={losses[-1]:.6f}")
    return Run(label, losses)


def window_gap(a: Run, b: Run, n: int) -> float:
    return max(abs(x - y) for x, y in zip(a.losses[:n], b.losses[:n]))


def tail_mean(r: Run) -> float:
    return sum(r.losses[-TAIL:]) / TAIL


@dataclass
class Verdict:
    seed: int
    step0_gap: float
    early_gap: float
    early_allowed: float
    end_gap: float
    end_allowed: float

    @property
    def ok(self) -> bool:
        return (
            self.step0_gap <= STEP0_FLOOR
            and self.early_gap <= self.early_allowed
            and self.end_gap <= self.end_allowed
        )


# Absolute floors. step-0: the two modes lower the SAME forward math
# differently (eager runtime ops vs Cranelift-lowered Wengert primal), so
# f32 reduction-order deltas through 6 blocks × seq 1024 × 49152-way CE
# land ~5e-2 at loss≈10.9; the floor rejects anything structural (a wrong
# op is orders of magnitude larger). early/endpoint floors serve the case
# of a perfectly quiet control pair.
STEP0_FLOOR = 0.15
EARLY_FLOOR = 0.15
END_FLOOR = 0.10
MULT = 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsl", required=True, help="CUDA nsl binary")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    materialize_slice()
    verdicts: list[Verdict] = []
    for i in range(args.seeds):
        seed = 1000 + i
        print(f"\n=== seed {seed} ===")
        src_a = run_once(args.nsl, f"seed{seed}/source-A", seed, True)
        src_b = run_once(args.nsl, f"seed{seed}/source-B", seed, True)
        tape_a = run_once(args.nsl, f"seed{seed}/tape-A", seed, False)
        tape_b = run_once(args.nsl, f"seed{seed}/tape-B", seed, False)

        # All four runs must have seen the SAME micro-step count — the
        # window comparisons below zip() and would silently truncate
        # (review finding on the item-5 commit).
        counts = {len(r.losses) for r in (src_a, src_b, tape_a, tape_b)}
        if len(counts) != 1:
            sys.exit(f"[seed {seed}] runs saw different step counts: {counts}")

        # Same-mode step-0 reproducibility is the foundation everything
        # else stands on — refuse to interpret anything if it fails.
        if src_a.losses[0] != src_b.losses[0] or tape_a.losses[0] != tape_b.losses[0]:
            sys.exit(
                f"[seed {seed}] same-mode step-0 losses differ — the forward is "
                "no longer run-to-run deterministic and no differential can be judged"
            )

        step0_gap = abs(src_a.losses[0] - tape_a.losses[0])
        early_ctrl = max(window_gap(src_a, src_b, EARLY), window_gap(tape_a, tape_b, EARLY))
        early_gap = window_gap(src_a, tape_a, EARLY)
        end_ctrl = max(
            abs(tail_mean(src_a) - tail_mean(src_b)),
            abs(tail_mean(tape_a) - tail_mean(tape_b)),
        )
        end_gap = abs(tail_mean(src_a) - tail_mean(tape_a))
        v = Verdict(
            seed,
            step0_gap,
            early_gap,
            max(MULT * early_ctrl, EARLY_FLOOR),
            end_gap,
            max(MULT * end_ctrl, END_FLOOR),
        )
        verdicts.append(v)
        print(
            f"[seed {seed}] step0={v.step0_gap:.3e} (floor {STEP0_FLOOR})  "
            f"early={v.early_gap:.3e}/{v.early_allowed:.3e}  "
            f"end={v.end_gap:.3e}/{v.end_allowed:.3e}  ->  "
            f"{'AGREE' if v.ok else 'DIVERGE'}"
        )

    print(
        "\n| seed | step0 gap | early gap | early allowed | endpoint gap "
        "| endpoint allowed | verdict |"
    )
    print("|------|-----------|-----------|---------------|--------------|------------------|---------|")
    for v in verdicts:
        print(
            f"| {v.seed} | {v.step0_gap:.3e} | {v.early_gap:.3e} | {v.early_allowed:.3e} "
            f"| {v.end_gap:.3e} | {v.end_allowed:.3e} | {'AGREE' if v.ok else 'DIVERGE'} |"
        )
    if not all(v.ok for v in verdicts):
        sys.exit("AD DIFFERENTIAL FAILED: tape and source AD diverge beyond the control envelope")
    print("\nAD DIFFERENTIAL PASSED: tape-F32 and source-F32 agree within the measured envelope")


if __name__ == "__main__":
    main()
