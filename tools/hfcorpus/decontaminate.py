#!/usr/bin/env python3
"""Benchmark decontamination scan for the v2 corpus (next-roadmap item 5).

Method — EXACT NORMALIZED-LINE matching, stated precisely because a
decontamination claim is only as strong as its method:

  * Every line of every benchmark field (HumanEval: prompt,
    canonical_solution, test; MBPP full+sanitized: text/prompt, code,
    test_list) is normalized (lowercase, whitespace runs collapsed to one
    space, stripped) and kept if >= MIN_LINE_CHARS after normalization —
    short lines ("return x", "import os") occur everywhere and would drown
    the signal in false positives.
  * Every corpus document is scanned line-by-line under the same
    normalization. A document is CONTAMINATED when it contains
    >= DOC_LINE_THRESHOLD distinct benchmark lines, or any single matched
    line of >= STRONG_LINE_CHARS — one long verbatim solution line is
    conclusive on its own.

This catches verbatim inclusion — the realistic contamination mode for
HumanEval/MBPP inside a code corpus scraped from the same GitHub the
benchmarks leaked into. It does NOT catch paraphrase or near-duplication;
the manifest records the method string so nobody mistakes the claim.

It also cannot see a benchmark problem whose lines are all short. The length
floor removes every line under MIN_LINE_CHARS from the benchmark set, so a
solution written as a few short statements leaves nothing to match, and a
verbatim copy of it in a corpus is counted as clean. The report's `coverage`
section says how many problems and how many solutions that is, per
benchmark, by running the rule itself on each one: a "0 contaminated
documents" line is a statement about the detectable ones only.

Output: a JSON report (default data/text/decontamination.json) that
manifest.py folds into CORPUS_MANIFEST_v2.json. Exit is nonzero if any set
named in --fail-on has a contaminated document — validation sets must be
clean by construction (re-cut them), while training-set contamination is
reported as a rate, not an error (the corpus is shipped; the record is
what makes its evaluation honest).

Usage (from the repo root, with the hfcorpus venv):
    python tools/hfcorpus/decontaminate.py \
        --scan val-stack val-web val-sft --report-only stack-train web-train sft-train \
        --fail-on val-stack val-web val-sft
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MIN_LINE_CHARS = 30
STRONG_LINE_CHARS = 60
DOC_LINE_THRESHOLD = 2
DOC_SEP = "\x00"

METHOD = (
    f"exact normalized-line matching (lowercase, whitespace-collapsed); "
    f"benchmark lines >= {MIN_LINE_CHARS} chars, import/from lines excluded "
    f"(boilerplate, not benchmark identity — 'from collections import "
    f"counter' flagged an innocent repo on first contact); a document is "
    f"contaminated at >= {DOC_LINE_THRESHOLD} distinct matched lines or one "
    f"matched line >= {STRONG_LINE_CHARS} chars; catches verbatim inclusion "
    f"only, not paraphrase, and not a benchmark problem whose lines are all "
    f"under the length floor (counted in the report's coverage section)"
)

# The benchmark columns whose text is matched, and the column that holds a
# problem's reference solution.
HUMANEVAL_COLUMNS = ("prompt", "canonical_solution", "test")
HUMANEVAL_SOLUTION = "canonical_solution"
MBPP_COLUMNS = ("text", "prompt", "code", "test_list", "test_imports",
                "challenge_test_list", "test_setup_code")
MBPP_SOLUTION = "code"


def normalize(line: str) -> str:
    return " ".join(line.lower().split())


def benchmark_lines() -> tuple[dict[str, int], set[str]]:
    """(per-benchmark source line counts, the normalized line set)."""
    import pyarrow.parquet as pq

    lines: set[str] = set()
    counts: dict[str, int] = {}

    def add(name: str, texts: list[str]) -> None:
        # DISTINCT lines per benchmark, so the per-benchmark counts sum to
        # benchmark_line_count (occurrence counts read as an inconsistency
        # next to it — review finding).
        fresh: set[str] = set()
        for t in texts:
            for raw in t.splitlines():
                norm = normalize(raw)
                if len(norm) < MIN_LINE_CHARS:
                    continue
                if norm.startswith(("import ", "from ")):
                    continue
                if norm not in lines:
                    fresh.add(norm)
        lines.update(fresh)
        counts[name] = counts.get(name, 0) + len(fresh)

    he = REPO / "data/hf/humaneval/openai_humaneval"
    he_files = sorted(he.glob("*.parquet"))
    if not he_files:
        sys.exit(f"no HumanEval parquet under {he} — run fetch.py --only benchmarks")
    for f in he_files:
        t = pq.read_table(f, columns=list(HUMANEVAL_COLUMNS))
        for col in t.column_names:
            add("humaneval", [v.as_py() or "" for v in t[col]])

    mbpp = REPO / "data/hf/mbpp"
    mbpp_files = sorted(mbpp.rglob("*.parquet"))
    if not mbpp_files:
        sys.exit(f"no MBPP parquet under {mbpp} — run fetch.py --only benchmarks")
    for f in mbpp_files:
        t = pq.read_table(f)
        for col in t.column_names:
            if col not in MBPP_COLUMNS:
                continue
            vals = []
            for v in t[col]:
                py = v.as_py()
                if py is None:
                    continue
                vals.append("\n".join(py) if isinstance(py, list) else str(py))
            add("mbpp", vals)

    return counts, lines


def doc_is_contaminated(doc: str, lines: set[str]) -> tuple[bool, set[str]]:
    """THE rule, in one place — extract.py's --drop-contaminated imports it,
    so the dropping filter and the verifying scan can never disagree."""
    hits: set[str] = set()
    strong = False
    for raw_line in doc.splitlines():
        norm = normalize(raw_line)
        if len(norm) >= MIN_LINE_CHARS and norm in lines:
            hits.add(norm)
            if len(norm) >= STRONG_LINE_CHARS:
                strong = True
    return (strong or len(hits) >= DOC_LINE_THRESHOLD, hits)


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def benchmark_problems() -> Iterator[tuple[str, str, str]]:
    """Every benchmark problem as `(benchmark, whole text, solution text)`.

    The whole text is every matched column of the row, one after another:
    what a corpus document holds if it includes the problem verbatim. The
    files and columns are the ones `benchmark_lines` reads.
    """
    import pyarrow.parquet as pq

    for f in sorted((REPO / "data/hf/humaneval/openai_humaneval").glob("*.parquet")):
        for row in pq.read_table(f, columns=list(HUMANEVAL_COLUMNS)).to_pylist():
            yield ("humaneval",
                   "\n".join(_cell_text(row[c]) for c in HUMANEVAL_COLUMNS),
                   _cell_text(row[HUMANEVAL_SOLUTION]))
    for f in sorted((REPO / "data/hf/mbpp").rglob("*.parquet")):
        table = pq.read_table(f)
        columns = [c for c in table.column_names if c in MBPP_COLUMNS]
        for row in table.to_pylist():
            yield ("mbpp",
                   "\n".join(_cell_text(row[c]) for c in columns),
                   _cell_text(row.get(MBPP_SOLUTION)))


def coverage(problems: Iterable[tuple[str, str, str]], lines: set[str]) -> dict[str, dict[str, int]]:
    """How much of each benchmark the rule can see at all.

    For every problem, `doc_is_contaminated` is run on the problem's own
    text and on its solution alone, against the same line set the corpus
    scan uses. A problem that does not flag itself has too few lines over
    the length floor: a corpus document containing it verbatim would pass
    the scan. Counted per benchmark:

      * `problems`: rows read;
      * `undetectable_problems`: rows whose whole text (prompt, solution and
        tests together) would not be flagged;
      * `undetectable_solutions`: rows whose solution, on its own, would not
        be flagged. This is the larger number and the more relevant one: a
        scraped repository holds the solution far more often than the
        benchmark's own prompt and tests.
    """
    out: dict[str, dict[str, int]] = {}
    for name, whole, solution in problems:
        entry = out.setdefault(name, {"problems": 0, "undetectable_problems": 0,
                                      "undetectable_solutions": 0})
        entry["problems"] += 1
        if not doc_is_contaminated(whole, lines)[0]:
            entry["undetectable_problems"] += 1
        if not doc_is_contaminated(solution, lines)[0]:
            entry["undetectable_solutions"] += 1
    return out


def scan_set(text_dir: Path, lines: set[str]) -> dict:
    docs = 0
    contaminated = 0
    examples: list[dict] = []
    scanned_bytes = 0
    for shard in sorted(text_dir.glob("*.txt")):
        raw = shard.read_text()
        scanned_bytes += len(raw)
        for local_index, doc in enumerate(raw.split(DOC_SEP)):
            docs += 1
            bad, hits = doc_is_contaminated(doc, lines)
            if bad:
                contaminated += 1
                if len(examples) < 5:
                    # Shard-LOCAL index, so the example reproduces by
                    # indexing into the named shard (review finding: the
                    # global counter paired with one shard's name did not).
                    examples.append(
                        {"shard": shard.name, "doc_index": local_index,
                         "matched_lines": len(hits),
                         "sample": sorted(hits)[0][:120]}
                    )
    return {
        "documents": docs,
        "contaminated_documents": contaminated,
        "scanned_bytes": scanned_bytes,
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", nargs="+", required=True,
                    help="data/text/<set> dirs whose result gates the exit")
    ap.add_argument("--report-only", nargs="*", default=[],
                    help="sets scanned and recorded but never fatal")
    ap.add_argument("--fail-on", nargs="*", default=None,
                    help="subset of --scan that must be clean (default: all of --scan)")
    ap.add_argument("--out", type=Path, default=REPO / "data/text/decontamination.json")
    args = ap.parse_args()
    fail_on = set(args.fail_on if args.fail_on is not None else args.scan)
    unknown = fail_on - set(args.scan)
    if unknown:
        sys.exit(f"--fail-on names sets not in --scan: {sorted(unknown)} — "
                 f"a typo here silently disarms the gate")

    t0 = time.time()
    counts, lines = benchmark_lines()
    print(f"[decontaminate] {len(lines)} benchmark lines "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))})",
          flush=True)

    blind = coverage(benchmark_problems(), lines)
    for name, entry in sorted(blind.items()):
        print(f"[decontaminate] {name}: of {entry['problems']} problems, "
              f"{entry['undetectable_problems']} could not be detected even "
              f"if copied whole, and {entry['undetectable_solutions']} have a "
              f"solution that could not be detected on its own "
              f"(all lines under {MIN_LINE_CHARS} chars)", flush=True)

    report = {
        "method": METHOD,
        "benchmarks": counts,
        "benchmark_line_count": len(lines),
        "coverage": blind,
        "sets": {},
    }
    dirty = []
    for name in list(args.scan) + list(args.report_only):
        d = REPO / "data/text" / name
        if not d.is_dir():
            sys.exit(f"no such text set: {d}")
        res = scan_set(d, lines)
        report["sets"][name] = res
        marker = ""
        if res["contaminated_documents"] and name in fail_on:
            dirty.append(name)
            marker = "  <-- FAIL"
        print(f"[decontaminate] {name}: {res['contaminated_documents']} of "
              f"{res['documents']} documents contaminated "
              f"({res['scanned_bytes'] / 1e6:.0f} MB){marker}", flush=True)

    args.out.write_text(json.dumps(report, indent=2))
    print(f"[decontaminate] report -> {args.out} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if dirty:
        print(f"decontamination FAILED: validation set(s) {dirty} contain "
              f"benchmark text — re-cut them (a contaminated val set scores "
              f"memorization, not generalization)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
