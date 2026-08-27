#!/usr/bin/env python3
"""Pre-chain posture benchmark — the 1B production model, five compile/run arms.

The final benchmark before the intermediate-run checkpoint chain begins
(2026-08-27). The chain's execution fingerprint treats the wgrad choice as
arithmetic identity, so the posture must be settled BEFORE the first
checkpoint is written, not six hours after. Arms:

    P0  --source-ad --checkpoint-blocks    --fuse-rmsnorm-backward
    P1  P0 + --fuse-wgrad-accum
    P2  P1 + --cuda-graphs
    P3  --source-ad --checkpoint-selective --fuse-rmsnorm-backward --fuse-wgrad-accum
    P4  P3's binary, run under NSL_MATMUL_BF16=1 (cuBLAS math mode is a
        RUNTIME env — same artifact, different run)

P0-P2 answer the wgrad and graphs questions on the shipped posture; P3/P4
measure the proposed run posture (checkpoint-selective + bf16) AT 1B — its
published +33% was measured at 500M, and whether selective's larger
activation surface even fits beside 12.9 GB of resident optimizer state at
1B is exactly what this must establish before the chain starts. The #537
elementwise/RoPE/scalar fusions and #538 multi-warp FA backward are
default-on and present in every arm.

For P2 the tok/s number alone is NOT acceptance: the run stderr's
    [cuda-graph] regions=R captured=C replays=N taints=T ...
summary must show meaningful capture AND replay; a graphs arm that runs
eager is P1 with extra bookkeeping, and known graph-taint history plus
#538's own region-splitting follow-up make silent non-engagement likely.
If graphs do not meaningfully engage, they are not used for the chain.

Reuses resident_bench.py's proven discipline wholesale: generated
step-limited program in the recipe's exact posture, fused-CE build-stderr
refusal, OPT_STEP arrival stamps, first-10-updates warmup discard,
interleaved round-robin rounds, best-round reporting, gpu_guard
refuse-don't-warn.

Usage: python3 models/benchmarks/posture_bench_1b.py \
           --nsl /path/to/nsl --out run-out/posture-bench [--rounds 3]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resident_bench as rb  # noqa: E402
import gpu_guard  # noqa: E402

BASE = ["--source-ad", "--checkpoint-blocks", "--fuse-rmsnorm-backward"]
SELECTIVE = ["--source-ad", "--checkpoint-blocks", "--checkpoint-selective",
             "--fuse-rmsnorm-backward"]


@dataclasses.dataclass
class FlagArm:
    """Duck-types the four attributes resident_bench.build_arm reads."""

    key: str
    flag_list: list[str]
    run_env: dict[str, str]
    micro_batch: int = 2
    accum: int = 4

    @property
    def flags(self) -> list[str]:
        return list(self.flag_list)


ARMS = [
    FlagArm("P0_blocks", BASE, {}),
    FlagArm("P1_wgrad", BASE + ["--fuse-wgrad-accum"], {}),
    FlagArm("P2_graphs", BASE + ["--fuse-wgrad-accum", "--cuda-graphs"], {}),
    FlagArm("P3_selective", SELECTIVE + ["--fuse-wgrad-accum"], {}),
    FlagArm("P4_sel_bf16", SELECTIVE + ["--fuse-wgrad-accum"],
            {"NSL_MATMUL_BF16": "1"}),
]

GRAPH_RE = re.compile(
    r"\[cuda-graph\] regions=(\d+) captured=(\d+) replays=(\d+) taints=(\d+) "
    r"mismatches=(\d+) repaired_ops=(\d+) eager=(\d+)"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--warmup-opt-steps", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    gpu_guard.acquire_or_refuse("posture_bench_1b")

    work = args.out.resolve()
    work.mkdir(parents=True, exist_ok=True)
    args.nsl = args.nsl.resolve()
    tokens_path = work / "bench_tokens.bin"
    n_tokens = args.opt_steps * rb.TOKENS_PER_UPDATE + 8 * rb.SEQ
    if not tokens_path.exists() or tokens_path.stat().st_size != n_tokens * 2:
        rb.gen_tokens.generate(tokens_path, n_tokens, rb.VOCAB, rb.TOKENS_PER_UPDATE)

    # P4 shares P3's artifact: build unique flag-sets once.
    binaries: dict[str, Path] = {}
    build_key = {a.key: " ".join(a.flag_list) for a in ARMS}
    built: dict[str, Path] = {}
    for a in ARMS:
        sig = build_key[a.key]
        if sig in built:
            binaries[a.key] = built[sig]
            print(f"[build] {a.key}: reusing {built[sig].parent.name}", flush=True)
            continue
        print(f"[build] {a.key}: {sig}", flush=True)
        binaries[a.key] = built[sig] = rb.build_arm(
            a, work, tokens_path, args.opt_steps, args.nsl
        )

    results: dict[str, dict] = {
        a.key: {"flags": a.flag_list, "run_env": a.run_env, "rounds": [],
                "graph": [], "peak_bytes": None, "surfaces": {}, "oom": False}
        for a in ARMS
    }

    for rnd in range(args.rounds):
        for a in ARMS:
            r = results[a.key]
            if r["oom"]:
                continue
            log = work / f"{a.key}_r{rnd}.stderr"
            print(f"[run] round {rnd} {a.key}", flush=True)
            rc, lines = rb.run_binary(binaries[a.key], a.run_env, args.timeout, log)
            stderr_text = log.read_text()
            if rc != 0:
                if any(m in stderr_text for m in rb.OOM_MARKERS):
                    r["oom"] = True
                    r["oom_dump"] = stderr_text[-1500:]
                    print(f"      {a.key}: OOM — arm retired", flush=True)
                    continue
                raise RuntimeError(f"{a.key} r{rnd}: rc={rc}\n{stderr_text[-1500:]}")
            timing = rb.parse_timing(lines, a, args.warmup_opt_steps)
            if timing is None:
                raise RuntimeError(f"{a.key} r{rnd}: too few OPT_STEP stamps")
            r["rounds"].append(dataclasses.asdict(timing))
            print(f"      {timing.tok_s:,.0f} tok/s over {timing.opt_steps_measured} steps",
                  flush=True)
            res_obj = rb.ArmResult(arm=None)
            rb.parse_epilogue(lines, res_obj)
            r["peak_bytes"] = res_obj.peak_bytes
            r["surfaces"] = res_obj.surfaces
            m = GRAPH_RE.search(stderr_text)
            if m:
                g = dict(zip(
                    ("regions", "captured", "replays", "taints", "mismatches",
                     "repaired_ops", "eager"), map(int, m.groups())))
                r["graph"].append(g)
                print(f"      graph: {g}", flush=True)
            elif "--cuda-graphs" in a.flag_list:
                r["graph"].append(None)
                print(f"      graph: NO SUMMARY LINE — graphs did not report",
                      flush=True)

    # ── report ──────────────────────────────────────────────────────────
    print("\n== posture bench: best rounds ==")
    for a in ARMS:
        r = results[a.key]
        if r["oom"]:
            print(f"{a.key:14s} OOM")
            continue
        best = max(x["tok_s"] for x in r["rounds"])
        peak = (r["peak_bytes"] or 0) / 2**30
        print(f"{a.key:14s} {best:8,.0f} tok/s   peak {peak:5.2f} GiB")
    out = work / "results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
