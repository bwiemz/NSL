#!/usr/bin/env python3
"""Build training and evaluation corpora from unrelated local projects.

A tokenizer evaluated only on held-out files from the repository it was trained
on will flatter itself: it has learned that project's idioms, naming and line
shapes, and a chunking scheme aggressive enough to memorise whole lines scores
spectacularly without generalising at all.

So projects are partitioned at the *project* level:

  train projects   -> packed into other_train.txt, joined into the training mix
  holdout projects -> emitted whole as ood.jsonl, never seen during training

Files from train projects are additionally split by path hash, so
`other_heldout.jsonl` measures in-domain generalisation while `ood.jsonl`
measures transfer to unseen projects and languages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXCLUDE_DIRS = frozenset(
    {
        ".git",
        "target",
        "node_modules",
        "build",
        "dist",
        "venv",
        ".venv",
        "__pycache__",
        ".next",
        "vendor",
        "third_party",
        ".nsl-cache",
        "site-packages",
    }
)

EXTENSIONS = (
    ".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".go", ".rb", ".sh", ".sql", ".html", ".css", ".md", ".rs",
)

# Held out entirely. Chosen for language diversity: a Java desktop client, a
# Python/TypeScript service, a CAD tool, a security scanner and a static site.
DEFAULT_HOLDOUT = (
    "downlords-faf-client",
    "horizon-cad",
    "OpenMemory",
    "web vulnerability finder",
    "Portfolio website",
)

DOC_SEP = "\x00"
MAX_DOC_BYTES = 400_000


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS or (part.startswith(".") and len(part) > 1) for part in path.parts)


def collect_project(project: Path, projects_dir: Path, budget: int, seen: set[str]) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(project.rglob("*")):
        if budget <= 0:
            break
        if not path.is_file() or path.suffix not in EXTENSIONS:
            continue
        rel = path.relative_to(projects_dir)
        if is_excluded(rel):
            continue
        try:
            if path.stat().st_size > MAX_DOC_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not text.strip() or DOC_SEP in text:
            continue
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        n_bytes = len(text.encode("utf-8"))
        budget -= n_bytes
        docs.append({"path": str(rel), "bytes": n_bytes, "text": text})
    return docs


def heldout_bucket(path: str, pct: int) -> bool:
    digest = hashlib.sha1(path.encode("utf-8")).digest()
    return (digest[0] * 256 + digest[1]) % 100 < pct


def write_jsonl(docs: list[dict], path: Path) -> int:
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    return sum(int(d["bytes"]) for d in docs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-dir", type=Path, default=Path("/home/brandon/Projects"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude-name-prefix", action="append", default=["NSL"])
    parser.add_argument("--holdout-project", action="append", default=list(DEFAULT_HOLDOUT))
    parser.add_argument("--max-bytes-per-project", type=int, default=6_000_000)
    parser.add_argument("--heldout-pct", type=int, default=10)
    args = parser.parse_args()

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_names = set(args.holdout_project)

    seen: set[str] = set()
    train_docs: list[dict] = []
    train_heldout_docs: list[dict] = []
    ood_docs: list[dict] = []
    summary: dict[str, tuple[str, int]] = {}

    for project in sorted(p for p in args.projects_dir.iterdir() if p.is_dir()):
        if any(project.name.startswith(prefix) for prefix in args.exclude_name_prefix):
            continue
        docs = collect_project(project, args.projects_dir, args.max_bytes_per_project, seen)
        if not docs:
            continue
        n_bytes = sum(int(d["bytes"]) for d in docs)
        if project.name in holdout_names:
            ood_docs.extend(docs)
            summary[project.name] = ("holdout", n_bytes)
        else:
            for doc in docs:
                target = train_heldout_docs if heldout_bucket(str(doc["path"]), args.heldout_pct) else train_docs
                target.append(doc)
            summary[project.name] = ("train", n_bytes)

    packed = out_dir / "other_train.txt"
    packed.write_text(DOC_SEP.join(str(d["text"]) for d in train_docs), encoding="utf-8")
    train_bytes = sum(int(d["bytes"]) for d in train_docs)
    heldout_bytes = write_jsonl(train_heldout_docs, out_dir / "other_heldout.jsonl")
    ood_bytes = write_jsonl(ood_docs, out_dir / "ood.jsonl")

    print(f"other_train.txt      {len(train_docs):>5} docs  {train_bytes / 1e6:>7.2f} MB")
    print(f"other_heldout.jsonl  {len(train_heldout_docs):>5} docs  {heldout_bytes / 1e6:>7.2f} MB")
    print(f"ood.jsonl            {len(ood_docs):>5} docs  {ood_bytes / 1e6:>7.2f} MB  (unseen projects)")
    print()
    for name, (role, n_bytes) in sorted(summary.items(), key=lambda kv: -kv[1][1]):
        print(f"  {role:<8} {name:<32} {n_bytes / 1e6:>7.2f} MB")


if __name__ == "__main__":
    main()
