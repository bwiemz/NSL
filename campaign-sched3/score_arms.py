#!/usr/bin/env python3
"""Build the schedule-campaign trajectory table FROM LINEAGE RECORDS.

The 1B campaign produced a wrong conclusion from two errors that a table of
numbers cannot show and a schema can refuse outright:

  * slopes fitted across intervals of DIFFERENT WIDTHS, which mechanically
    flattens a rate and looked like deceleration;
  * a point from a PROBE ARM tabled beside points from the main chain.

So this tool does not take filenames. It ingests lineage records and REFUSES:

  * a series whose points disagree on `arm`              -> LineageError
  * intervals of unequal token width (unless --allow-ragged)
  * a checkpoint whose bytes no longer match its record  -> LineageError
  * points whose execution fingerprint differs within an arm
  * a duplicate micro_step

Subcommands
  verify   ingest + check the invariants, print the series (no GPU)
  splice   turn each snapshot into a scoreable full model (CPU)
  score    evaluate held-out for each spliced point (GPU)
  report   fit slopes with error bars and apply the acceptance criteria
"""
from __future__ import annotations
import argparse, json, math, pathlib, re, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lineage as L

CAMP = pathlib.Path(__file__).resolve().parent
WT = CAMP.parent
TOKENS_PER_MICRO = L.TOKENS_PER_MICRO


class LineageError(RuntimeError):
    """A schema/lineage failure. Never downgraded to a warning: the whole
    point is that a bad row cannot become a plausible-looking number."""


def load_series(arm: str, *, check_hash: bool, allow_ragged: bool) -> list[dict]:
    snap = CAMP / f"snap_{arm}"
    recs = []
    for f in sorted(snap.glob("m*.lineage.json")):
        r = json.loads(f.read_text())
        # THE FILENAME IS A CLAIM; THE RECORD IS THE EVIDENCE. They must agree.
        # A file called m88000 whose record says micro_step 80000 is precisely
        # the mislabeling that put a probe-arm point into the chain's table --
        # and it is the only way a duplicate step can reach this series at all,
        # since the filename otherwise makes duplicates unrepresentable.
        claimed = int(re.match(r"m(\d+)\.lineage\.json$", f.name).group(1))
        if claimed != r["micro_step"]:
            raise LineageError(
                f"arm {arm}: {f.name} claims micro {claimed} but its record says "
                f"{r['micro_step']} — the filename and the record disagree")
        r["_theta"] = f.with_name(f.name.replace(".lineage.json", ".nslm"))
        recs.append(r)
    if not recs:
        raise LineageError(f"arm {arm}: no lineage records in {snap}")
    recs.sort(key=lambda r: r["micro_step"])

    arms = {r["arm"] for r in recs}
    if arms != {arm}:
        raise LineageError(
            f"arm {arm}: series contains points from {sorted(arms)} — "
            "this is the exact error that produced the retracted 1B conclusion")

    steps = [r["micro_step"] for r in recs]
    if len(set(steps)) != len(steps):
        raise LineageError(f"arm {arm}: duplicate micro_step in {steps}")

    fps = {r["execution_fingerprint"] for r in recs}
    if len(fps) != 1:
        raise LineageError(f"arm {arm}: {len(fps)} distinct execution fingerprints; "
                           "the arm changed arithmetic mid-run")

    widths = {b - a for a, b in zip(steps, steps[1:])}
    if len(widths) > 1 and not allow_ragged:
        raise LineageError(
            f"arm {arm}: unequal interval widths {sorted(widths)} micro-steps. "
            "Fitting a rate across unequal intervals flattens it mechanically. "
            "Pass --allow-ragged only if you have a reason and will say so.")

    if check_hash:
        for r in recs:
            got = L.sha256_file(r["_theta"])
            if got != r["model_checkpoint_sha256"]:
                raise LineageError(
                    f"arm {arm} micro {r['micro_step']}: theta bytes changed since "
                    f"the record was written ({got[:12]} != "
                    f"{r['model_checkpoint_sha256'][:12]})")
    return recs


def fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and its standard error. Returns (nan, nan) if <3
    points: a two-point 'trend' is the thing that started all this."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    return b, math.sqrt(s2 / den)


def cmd_verify(args):
    for arm in args.arms:
        recs = load_series(arm, check_hash=args.check_hash, allow_ragged=args.allow_ragged)
        w = {b["micro_step"] - a["micro_step"] for a, b in zip(recs, recs[1:])}
        print(f"arm {arm}: {len(recs)} points, interval width "
              f"{w.pop() if len(w)==1 else sorted(w)} micro "
              f"({'uniform' if len(w)<=1 else 'RAGGED'})")
        for r in recs:
            print(f"    micro {r['micro_step']:>6}  {r['tokens_seen']:>13,} tok  "
                  f"lr {r['current_lr']:.6e}  int={r['interruption_count']} "
                  f"replay={r['replayed_microsteps']}  "
                  f"{'dirty' if r['git_dirty'] else 'clean'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "splice", "score", "report"])
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--allow-ragged", action="store_true",
                    help="permit unequal interval widths; say why in the write-up")
    ap.add_argument("--check-hash", action="store_true",
                    help="re-hash each theta and compare against its record")
    args = ap.parse_args()
    if args.cmd == "verify":
        return cmd_verify(args)
    print(f"{args.cmd}: not yet wired (arms still running)", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
