#!/usr/bin/env python3
"""Build a deterministic train/held-out corpus for tokenizer evaluation.

The corpus is drawn from this repository's own sources, which is the right
distribution for the NSL-Coder models: Rust (the compiler), NSL (the target
language), Markdown (docs and prose), Python and TOML (tooling).

Split is by SHA-1 of the repo-relative path, so a file always lands in the
same side of the split no matter what else is present.

Output (under --out):
    train.txt      documents joined by a single \\x00 byte
    heldout.txt    same format, disjoint from train.txt
    heldout.jsonl  one {"path", "bytes", "text"} record per held-out document
    manifest.json  counts and byte totals per extension per split
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

# Only these subtrees are corpus. An allowlist rather than a denylist because
# the repo also contains ~43k near-duplicate .rs files under .claude/worktrees
# and .worktrees; including those would both inflate the measured compression
# and leak train documents into the held-out split.
INCLUDE_ROOTS = (
    "crates",
    "stdlib",
    "models",
    "tools",
    "scripts",
    "docs",
    "spec",
    "tests",
    "examples",
)

# Build output nested inside an included root.
EXCLUDE_DIRS = frozenset({"target", "node_modules", ".nsl-cache", "checkpoints"})

EXTENSIONS = (".rs", ".nsl", ".md", ".py", ".toml")

# Documents are separated by NUL, which cannot appear in the source files
# themselves, so the boundary is unambiguous when we split it back apart.
DOC_SEP = "\x00"


@dataclass
class SplitStats:
    files: int = 0
    total_bytes: int = 0
    by_ext: dict[str, list[int]] = field(default_factory=dict)

    def add(self, ext: str, n_bytes: int) -> None:
        self.files += 1
        self.total_bytes += n_bytes
        slot = self.by_ext.setdefault(ext, [0, 0])
        slot[0] += 1
        slot[1] += n_bytes


def is_excluded(rel: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in rel.parts)


def heldout_bucket(rel: Path, heldout_pct: int) -> bool:
    digest = hashlib.sha1(str(rel).encode("utf-8")).digest()
    return (digest[0] * 256 + digest[1]) % 100 < heldout_pct


def iter_source_files(root: Path):
    for top in INCLUDE_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in EXTENSIONS:
                yield path
    for path in sorted(root.glob("*")):
        if path.is_file() and path.suffix in EXTENSIONS:
            yield path


def collect(root: Path, heldout_pct: int) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    train: list[tuple[Path, str]] = []
    heldout: list[tuple[Path, str]] = []
    # Exact-duplicate content is dropped repo-wide. Duplicates would otherwise
    # let a merge learned on a train copy score for free on its held-out twin.
    seen: set[str] = set()

    for path in iter_source_files(root):
        rel = path.relative_to(root)
        if is_excluded(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not text.strip() or DOC_SEP in text:
            continue
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        (heldout if heldout_bucket(rel, heldout_pct) else train).append((rel, text))

    return train, heldout


def write_split(docs: list[tuple[Path, str]], out_path: Path) -> SplitStats:
    stats = SplitStats()
    with out_path.open("w", encoding="utf-8") as handle:
        for index, (rel, text) in enumerate(docs):
            if index:
                handle.write(DOC_SEP)
            handle.write(text)
            stats.add(rel.suffix, len(text.encode("utf-8")))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heldout-pct", type=int, default=8)
    args = parser.parse_args()

    root: Path = args.root.resolve()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    train_docs, heldout_docs = collect(root, args.heldout_pct)
    train_stats = write_split(train_docs, out / "train.txt")
    heldout_stats = write_split(heldout_docs, out / "heldout.txt")

    with (out / "heldout.jsonl").open("w", encoding="utf-8") as handle:
        for rel, text in heldout_docs:
            record = {"path": str(rel), "bytes": len(text.encode("utf-8")), "text": text}
            handle.write(json.dumps(record) + "\n")

    manifest = {
        "root": str(root),
        "heldout_pct": args.heldout_pct,
        "extensions": list(EXTENSIONS),
        "train": {
            "files": train_stats.files,
            "bytes": train_stats.total_bytes,
            "by_ext": train_stats.by_ext,
        },
        "heldout": {
            "files": heldout_stats.files,
            "bytes": heldout_stats.total_bytes,
            "by_ext": heldout_stats.by_ext,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"train:   {train_stats.files:>5} files  {train_stats.total_bytes / 1e6:>8.2f} MB")
    print(f"heldout: {heldout_stats.files:>5} files  {heldout_stats.total_bytes / 1e6:>8.2f} MB")
    for ext in EXTENSIONS:
        tr = train_stats.by_ext.get(ext, [0, 0])
        ho = heldout_stats.by_ext.get(ext, [0, 0])
        print(f"  {ext:<6} train {tr[0]:>5}f/{tr[1] / 1e6:>7.2f}MB   heldout {ho[0]:>4}f/{ho[1] / 1e6:>6.2f}MB")


if __name__ == "__main__":
    main()
