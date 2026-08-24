#!/usr/bin/env python3
"""Download the Hugging Face source datasets for the NSL pretraining corpus.

Two datasets, fetched into `data/hf/` (gitignored):

  * `HuggingFaceFW/fineweb`, config `sample-10BT` — 15 parquet files, 30.6 GB,
    14.9M deduplicated English web documents (~10B GPT-2-tokens). This is the
    bulk of the pretraining mix.
  * `HuggingFaceCode/stack-v3-train` — source code grouped by repository, with
    file contents inline. The full train subset is 3.54 TB / ~3.6T tokens across
    8,192 shards; `--stack-shards N` takes a strided sample (~0.55B tokens per
    shard).
  * `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`, config `canonical` —
    57,937 multi-teacher chat traces with full audit columns. Small; it feeds
    a separate SFT/anneal corpus, NOT the base pretraining mix.

`canonical/` contains TWO overlapping shard series in the same directory
(`train-*-of-00005` and `train-*-of-00006`) whose byte totals differ, so they
are not a reshard of identical content. Globbing `train-*.parquet` — which is
what `datasets` does by default, since the README declares config names with no
`data_files` — would silently concatenate both and double-count most traces.
Both series are downloaded here; `extract.py` picks the one whose row count
matches the card and refuses if neither does.

Downloads resume: re-running skips files already present with the right hash.

Usage (from the repo root):
    tools/hfcorpus/.venv/bin/python tools/hfcorpus/fetch.py [--only fineweb|distill]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "data/hf"

FINEWEB_REPO = "HuggingFaceFW/fineweb"
FINEWEB_PATTERNS = ["sample/10BT/*.parquet"]

# The Stack v3 train subset: 8,192 shards, 3.54 TB, ~3.6T tokens. Contents are
# inline under `files[].content`, so unlike The Stack v2 there is no separate
# blob fetch. Far more than any run here needs, so a bounded subset is taken --
# STRIDED across the full index rather than a prefix, because the shards come
# out of a Spark job whose partitioning order is not documented and a prefix
# could be correlated with repo size, language, or crawl order.
STACK_REPO = "HuggingFaceCode/stack-v3-train"
STACK_SUFFIX = "990b4288-3824-41ac-94a0-b6fd6fa23ffe-c000.snappy.parquet"
STACK_SHARDS = 8192


def stack_patterns(count: int) -> list[str]:
    stride = STACK_SHARDS // count
    return [f"data/part-{i * stride:05d}-{STACK_SUFFIX}" for i in range(count)]


DISTILL_REPO = "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation"
# The prose files are tiny and carry the provenance we cite in the corpus
# manifest, so they come along with the parquet.
DISTILL_PATTERNS = [
    "data/canonical/*.parquet",
    "*.md",
    "manifest.json",
    "curriculum_stages.json",
]


def fetch(repo_id: str, patterns: list[str], out: Path, workers: int) -> Path:
    print(f"[fetch] {repo_id} -> {out}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=patterns,
        max_workers=workers,
    )
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"[fetch] {repo_id}: {total / 1e9:.2f} GB on disk", flush=True)
    return Path(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fineweb", "distill", "stack"], default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stack-shards", type=int, default=20,
                    help="how many of the 8192 Stack v3 shards to take "
                    "(~0.55B tokens each); sampled with a stride, not a prefix")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "distill"):
        fetch(DISTILL_REPO, DISTILL_PATTERNS, DEST / "distill", args.workers)
    if args.only in (None, "fineweb"):
        fetch(FINEWEB_REPO, FINEWEB_PATTERNS, DEST / "fineweb", args.workers)
    if args.only in (None, "stack"):
        pats = stack_patterns(args.stack_shards)
        print(f"[fetch] stack: {len(pats)} of {STACK_SHARDS} shards, "
              f"stride {STACK_SHARDS // args.stack_shards}", flush=True)
        fetch(STACK_REPO, pats, DEST / "stack", args.workers)

    print("[fetch] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
