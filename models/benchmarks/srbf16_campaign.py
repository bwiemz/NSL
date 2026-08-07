"""Item-16 SR-BF16 trajectory certification campaign.

Compares --param-dtype bf16-sr against an f32 control on the IDENTICAL
CSLA + weight-stream residency schedule (the only delta between arms is the
authoritative parameter storage + the SR fused step), equal tokens, equal
seeds. Then certifies checkpoint-save/reload continuation under bf16-sr.

Arms per scale:
  f32_s<seed>     — --checkpoint-blocks --layerwise-accum --weight-stream
  bf16sr_s<seed>  — same + --param-dtype bf16-sr
  bf16sr_continue — reload bf16sr_s<seed0>'s checkpoint, train more steps

Anti-vacuity (the wgrad-fusion lesson: a silent fallback makes a parity
gate meaningless): every bf16sr arm ASSERTS the `[sr-bf16] teardown:`
counters report params > 0 and SR steps > 0, and every f32 arm asserts the
marker is ABSENT. A bf16sr arm that silently ran f32 fails the campaign.

Raw logs land under models/benchmarks/srbf16_logs/<arm>/ (same layout as
p0_campaign). Summary prints as markdown for the results doc.

Usage:
  python3 srbf16_campaign.py 50m  --steps 250 --seeds 1 2 3
  python3 srbf16_campaign.py 500m --steps 150 --seeds 1
  python3 srbf16_campaign.py 50m  --steps 250 --seeds 1 --continue-steps 60
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from p0_campaign import (
    HEADER,
    REPO,
    RunResult,
    gen_tokens,
    run_arm,
    summarize,
    tokens_per_step_for,
)

LOGS = REPO / "models" / "benchmarks" / "srbf16_logs"
TOKENS_DIR = LOGS / "tokens"

BASE_FLAGS = [
    "--source-ad",
    "--deterministic",
    "--checkpoint-blocks",
    "--layerwise-accum",
    "--weight-stream",
]

SR_TEARDOWN_RE = re.compile(
    r"\[sr-bf16\] teardown: (\d+) bf16-authoritative param\(s\), "
    r"(\d+) SR optimizer step\(s\), (\d+) widen-upload\(s\)"
)


def sr_teardown(arm: str) -> tuple[int, int, int] | None:
    """Parse the sr-bf16 teardown counters from an arm's preserved stderr."""
    log = LOGS / arm / "stderr.log"
    if not log.exists():
        return None
    m = SR_TEARDOWN_RE.search(log.read_text())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def assert_arm_honest(name: str, res: RunResult, expect_sr: bool) -> None:
    """Fail the campaign if an arm silently ran the wrong storage path."""
    if not res.ok:
        raise SystemExit(f"{name}: run FAILED — see {LOGS / name}/stderr.log")
    counters = sr_teardown(name)
    if expect_sr:
        if counters is None:
            raise SystemExit(
                f"{name}: no [sr-bf16] teardown counters — the bf16-sr arm "
                "silently ran without the mirror schedule (vacuous)"
            )
        params, steps, uploads = counters
        if params == 0 or steps == 0 or uploads == 0:
            raise SystemExit(
                f"{name}: vacuous SR run (params={params} steps={steps} "
                f"uploads={uploads})"
            )
    elif counters is not None:
        raise SystemExit(f"{name}: f32 control unexpectedly ran SR: {counters}")
    if not res.losses:
        raise SystemExit(f"{name}: empty loss stream")
    vals = [v for _, v in res.losses]
    if not all(v == v and abs(v) != float("inf") for v in vals):
        raise SystemExit(f"{name}: non-finite loss in stream")


def curve_delta(a: RunResult, b: RunResult) -> tuple[int, float, float]:
    """Pairwise |Δloss| over the common prefix: (n, mean, max)."""
    n = min(len(a.losses), len(b.losses))
    if n == 0:
        return 0, float("nan"), float("nan")
    deltas = [abs(a.losses[i][1] - b.losses[i][1]) for i in range(n)]
    return n, sum(deltas) / n, max(deltas)


SCALES = {
    # scale: (model dir, batch, seq, accum-in-program, per-micro budget s)
    "50m": ("coder50m", 1, 1024, 2, 4.0),
    "500m": ("coder500m", 1, 512, 8, 10.0),
    "1b": ("coder1b", 1, 512, 8, 20.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scale", choices=sorted(SCALES))
    parser.add_argument("--steps", type=int, required=True,
                        help="optimizer steps per arm")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--continue-steps", type=int, default=60,
                        help="reload-continuation optimizer steps (0 skips)")
    parser.add_argument("--skip-f32", action="store_true",
                        help="only run the bf16-sr arms")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="REAL u16 token corpus to slice instead of the "
                             "synthetic gen_tokens stream (item 7's "
                             "real-corpus trajectory arm)")
    parser.add_argument("--tag", default="",
                        help="suffix appended to every arm name so a real-"
                             "corpus campaign does not overwrite the "
                             "synthetic one's logs")
    parser.add_argument("--sr-hist", action="store_true",
                        help="run the bf16-sr arms under NSL_SR_HIST=1 "
                             "(update-magnitude/stall histograms in stderr)")
    args = parser.parse_args()

    model_dir, batch, seq, accum, per_micro_s = SCALES[args.scale]
    program = REPO / "models" / model_dir / "pretrain_srbf16_cert.nsl"
    cont_program = REPO / "models" / model_dir / "pretrain_srbf16_continue.nsl"
    if args.scale != "50m" and not program.exists():
        raise SystemExit(
            f"{program} missing — copy the coder50m cert/continue programs "
            "and adjust batch/seq/accum for this scale first"
        )

    micro_per_step = accum
    need = args.steps * tokens_per_step_for(batch, seq, micro_per_step)
    kind = "real" if args.corpus is not None else "syn"
    if args.corpus is not None:
        # Slice the REAL corpus to length. Refusing a short corpus beats
        # silently wrapping it (which would train on repeats and report a
        # memorization curve as a real-data curve).
        raw = args.corpus.read_bytes()
        if len(raw) < need * 2:
            raise SystemExit(
                f"corpus {args.corpus} has {len(raw) // 2} tokens, need {need}")
        # The kind is in the FILENAME, not just the tag: gen_tokens has a
        # size-check cache, so a same-named synthetic run would silently
        # reuse the corpus bytes as its "synthetic" stream.
        tok = TOKENS_DIR / f"srbf16_{args.scale}{args.tag}_{kind}_{args.steps}.bin"
        tok.parent.mkdir(parents=True, exist_ok=True)
        tok.write_bytes(raw[: need * 2])
    else:
        tok = TOKENS_DIR / f"srbf16_{args.scale}{args.tag}_{kind}_{args.steps}.bin"
        gen_tokens(tok, need)
    timeout_s = int(args.steps * micro_per_step * per_micro_s * 1.6) + 900

    env = {"NSL_WS_COUNTER": "1"}
    # run_arm writes under p0's LOGS constant — repoint it at ours.
    import p0_campaign as p0

    # Per-SCALE log namespace: arm names carry only tag+seed, so without
    # this a 1b run overwrites a 500m run's per-arm dirs and summary (it
    # did, on 2026-08-06 — the 500m-synthetic stderr logs were lost and
    # only the captured stdout tables survive).
    global LOGS
    LOGS = LOGS / args.scale
    p0.LOGS = LOGS
    LOGS.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    results: dict[str, RunResult] = {}
    save_paths: dict[str, Path] = {}

    for seed in args.seeds:
        arm_pairs = [] if args.skip_f32 else [
            (f"f32{args.tag}_s{seed}", BASE_FLAGS)
        ]
        arm_pairs.append(
            (f"bf16sr{args.tag}_s{seed}", [*BASE_FLAGS, "--param-dtype", "bf16-sr"])
        )
        for name, flags in arm_pairs:
            save = LOGS / name / "final.nslm"
            save.parent.mkdir(parents=True, exist_ok=True)
            print(f"=== {name}: {' '.join(flags)}")
            arm_env = dict(env)
            if args.sr_hist and "bf16sr" in name:
                arm_env["NSL_SR_HIST"] = "1"
            res = run_arm(
                name,
                program,
                tok,
                [*flags, "--seed", str(seed)],
                arm_env,
                rewrites=[("CERT_SAVE_PATH", save.as_posix())],
                timeout_s=timeout_s,
            )
            assert_arm_honest(name, res, expect_sr="bf16sr" in name)
            results[name] = res
            save_paths[name] = save
            rows.append(summarize(res, batch * seq))
            print(rows[-1])

    # Reload-continuation under bf16-sr from the first seed's checkpoint.
    if args.continue_steps > 0:
        parent = f"bf16sr{args.tag}_s{args.seeds[0]}"
        parent_save = save_paths.get(parent)
        if parent_save is None or not parent_save.exists():
            raise SystemExit(f"continuation parent checkpoint missing: {parent}")
        cont_need = args.continue_steps * tokens_per_step_for(
            batch, seq, micro_per_step)
        cont_tok = TOKENS_DIR / (
            f"srbf16_{args.scale}{args.tag}_{kind}_cont_{args.continue_steps}.bin")
        if args.corpus is not None:
            # FRESH real data from past the parent's consumed range — a
            # continuation that replays the parent's tokens would grade
            # memorization, and one on the synthetic stream would grade a
            # different distribution than the checkpoint was trained on.
            raw = args.corpus.read_bytes()
            if len(raw) < (need + cont_need) * 2:
                raise SystemExit(
                    f"corpus too short for continuation: have {len(raw) // 2} "
                    f"tokens, need {need + cont_need}")
            cont_tok.write_bytes(raw[need * 2 : (need + cont_need) * 2])
        else:
            gen_tokens(cont_tok, cont_need)
        name = f"bf16sr{args.tag}_continue"
        save = LOGS / name / "final.nslm"
        save.parent.mkdir(parents=True, exist_ok=True)
        print(f"=== {name}: reload {parent_save}")
        cont_env = dict(env)
        if args.sr_hist:
            cont_env["NSL_SR_HIST"] = "1"
        res = run_arm(
            name,
            cont_program,
            cont_tok,
            [*BASE_FLAGS, "--param-dtype", "bf16-sr", "--seed",
             str(args.seeds[0])],
            cont_env,
            rewrites=[
                ("CERT_LOAD_PATH", parent_save.as_posix()),
                ("CERT_SAVE_PATH", save.as_posix()),
            ],
            timeout_s=int(args.continue_steps * micro_per_step * per_micro_s * 1.6)
            + 900,
        )
        assert_arm_honest(name, res, expect_sr=True)
        results[name] = res
        rows.append(summarize(res, batch * seq))
        print(rows[-1])
        # The continuation must RESUME, not re-warm: its first losses stay
        # near the parent's final level rather than the parent's cold start.
        parent_res = results.get(parent)
        if parent_res and parent_res.losses and res.losses:
            parent_first = parent_res.losses[0][1]
            parent_last = parent_res.losses[-1][1]
            cont_first = res.losses[0][1]
            midpoint = (parent_first + parent_last) / 2.0
            verdict = "RESUMED" if cont_first < midpoint else "RE-WARMED (BAD)"
            print(
                f"continuation check: parent first={parent_first:.4f} "
                f"last={parent_last:.4f} cont first={cont_first:.4f} -> {verdict}"
            )
            if cont_first >= midpoint:
                raise SystemExit(
                    f"{name}: continuation started at {cont_first:.4f}, closer "
                    f"to the parent's cold start than its final loss — the "
                    "checkpoint did not carry the trained bf16 weights"
                )

    print(f"\n## SR-BF16 trajectory {args.scale} ({args.steps} steps)\n")
    print(HEADER)
    print("\n".join(rows))

    # Paired equal-token comparison per seed.
    if not args.skip_f32:
        print("\n## FP32-vs-SR-BF16 paired curves (equal tokens, equal seed)\n")
        print("| seed | steps | mean |dL| | max |dL| | f32 final | bf16sr final |")
        print("|---|---|---|---|---|---|")
        for seed in args.seeds:
            a = results.get(f"f32{args.tag}_s{seed}")
            b = results.get(f"bf16sr{args.tag}_s{seed}")
            if not a or not b:
                continue
            n, mean_d, max_d = curve_delta(a, b)
            print(
                f"| {seed} | {n} | {mean_d:.5f} | {max_d:.5f} "
                f"| {a.losses[-1][1]:.4f} | {b.losses[-1][1]:.4f} |"
            )

    # Seed spread within each arm family (final losses).
    for fam in ("f32", "bf16sr"):
        finals = [
            results[f"{fam}{args.tag}_s{s}"].losses[-1][1]
            for s in args.seeds
            if f"{fam}{args.tag}_s{s}" in results
            and results[f"{fam}{args.tag}_s{s}"].losses
        ]
        if len(finals) > 1:
            print(
                f"{fam} final-loss seed spread: min={min(finals):.4f} "
                f"max={max(finals):.4f} range={max(finals) - min(finals):.4f}"
            )

    (LOGS / f"campaign_summary{args.tag}_{kind}.json").write_text(
        json.dumps(
            {
                name: {
                    "ok": r.ok,
                    "n_losses": len(r.losses),
                    "first": r.losses[0][1] if r.losses else None,
                    "last": r.losses[-1][1] if r.losses else None,
                    "stream_wall_s": r.stream_wall_s,
                    "sr_teardown": sr_teardown(name),
                }
                for name, r in results.items()
            },
            indent=1,
        )
    )
    print(f"\nraw logs: {LOGS}")


if __name__ == "__main__":
    # p0_campaign resolves NSL from REPO/target/release; allow override so
    # the campaign can run against a worktree build.
    nsl_override = os.environ.get("NSL_BIN")
    if nsl_override:
        import p0_campaign as p0

        p0.NSL = Path(nsl_override)
    main()
