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
from pathlib import Path

import pyarrow.parquet as pq

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
    f"only, not paraphrase"
)


def normalize(line: str) -> str:
    return " ".join(line.lower().split())


def benchmark_lines() -> tuple[dict[str, int], set[str]]:
    """(per-benchmark source line counts, the normalized line set)."""
    lines: set[str] = set()
    counts: dict[str, int] = {}

    def add(name: str, texts: list[str]) -> None:
        n = 0
        for t in texts:
            for raw in t.splitlines():
                norm = normalize(raw)
                if len(norm) < MIN_LINE_CHARS:
                    continue
                if norm.startswith(("import ", "from ")):
                    continue
                lines.add(norm)
                n += 1
        counts[name] = counts.get(name, 0) + n

    he = REPO / "data/hf/humaneval/openai_humaneval"
    he_files = sorted(he.glob("*.parquet"))
    if not he_files:
        sys.exit(f"no HumanEval parquet under {he} — run fetch.py --only benchmarks")
    for f in he_files:
        t = pq.read_table(f, columns=["prompt", "canonical_solution", "test"])
        for col in t.column_names:
            add("humaneval", [v.as_py() or "" for v in t[col]])

    mbpp = REPO / "data/hf/mbpp"
    mbpp_files = sorted(mbpp.rglob("*.parquet"))
    if not mbpp_files:
        sys.exit(f"no MBPP parquet under {mbpp} — run fetch.py --only benchmarks")
    for f in mbpp_files:
        t = pq.read_table(f)
        for col in t.column_names:
            if col not in ("text", "prompt", "code", "test_list", "test_imports"):
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


def scan_set(text_dir: Path, lines: set[str]) -> dict:
    docs = 0
    contaminated = 0
    examples: list[dict] = []
    scanned_bytes = 0
    for shard in sorted(text_dir.glob("*.txt")):
        raw = shard.read_text()
        scanned_bytes += len(raw)
        for doc in raw.split(DOC_SEP):
            docs += 1
            bad, hits = doc_is_contaminated(doc, lines)
            if bad:
                contaminated += 1
                if len(examples) < 5:
                    examples.append(
                        {"shard": shard.name, "doc_index": docs - 1,
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

    t0 = time.time()
    counts, lines = benchmark_lines()
    print(f"[decontaminate] {len(lines)} benchmark lines "
          f"({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))})",
          flush=True)

    report = {
        "method": METHOD,
        "benchmarks": counts,
        "benchmark_line_count": len(lines),
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
