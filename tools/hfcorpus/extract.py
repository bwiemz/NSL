#!/usr/bin/env python3
"""Render the downloaded parquet into the NUL-separated text shards tokbench eats.

`tokbench pretokenize` splits its input corpus on U+0000 and encodes each piece
as one document, so that is the interchange format here. Output is written as
size-bounded shards because tokbench reads a whole corpus file into memory
(both backends; the legacy `hf` backend additionally holds the entire encoded
stream); a single 40 GB corpus file would not fit. Sharding also carries the
resume machinery in pretokenize.py. It is exact rather than approximate: the
encoder carries no state across documents, so concatenating per-shard token
streams gives byte-identical output to encoding the whole corpus at once.

Two renderers:

  fineweb  the `text` column, one document per row, in file-then-row order.
  distill  `messages_json` rendered to ChatML, one conversation per document.
  stack    `HuggingFaceCode/stack-v3-train`. Each parquet ROW is a repository and
           the code lives inline in a `files[]` array, so a row renders as one
           document of `<|file_sep|>path\ncontent` per file -- which is the point
           of the repo grouping: the model sees a project, not shuffled files.
           Repos beyond `--max-doc-bytes` are split across several documents
           rather than dropped, so one enormous monorepo cannot dominate a shard.
           The split falls BETWEEN files -- cutting inside one would hand the
           model half a function -- so a single oversized file still produces a
           document larger than the cap.
  code     `data/tokcorpus/combined_train.txt`, already NUL-separated, passed
           through the same sanitizer. It needs it more than the web does: that
           corpus contains machine-learning source, and a file that mentions
           `<|pad|>` in a comment would otherwise forge a boundary. Measured on
           the shipped corpus: 35 raw surfaces, against 9 in 45 GB of FineWeb.

SEPARATOR HYGIENE. Both renderers neutralize special-token surfaces that occur
inside document text. A `tokenizers` AddedToken is matched by surface at encode
time no matter what `add_special_tokens` is set to, so a crawled page whose body
contains the literal `<|endoftext|>` would otherwise inject a real end-of-document
id into the middle of the training stream — a document forging its own boundary.
The same applies to a distillation trace whose content contains `<|im_start|>`:
it would forge a turn. Neutralizing costs nothing and removes the whole class.
U+0000 is stripped for the same reason: it is the shard's document separator.

Usage (from the repo root):
    tools/hfcorpus/.venv/bin/python tools/hfcorpus/extract.py fineweb --out data/text/web
    tools/hfcorpus/.venv/bin/python tools/hfcorpus/extract.py distill --out data/text/sft
    # bounded sample for tokenizer training:
    ... extract.py fineweb --out data/text/tok-sample --stride 40 --max-bytes 1200000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]

# Reserved surfaces, loaded from the ONE committed record. extract.py must
# neutralize exactly the set train_tokenizer.py reserves — a surface reserved
# but not neutralized is a forgeable boundary — and two inline copies held
# equal by a comment is how that invariant eventually breaks. The gate
# (corpus_manifest_gate.rs) refuses an inline list in either file.
SPECIALS: list[str] = json.loads(
    (REPO / "models/tokenizers/special_tokens.json").read_text()
)["surfaces"]

# U+2063 INVISIBLE SEPARATOR breaks the surface match while keeping the text
# visually and semantically intact, so a page discussing these tokens still
# reads correctly to the model instead of being deleted or mangled.
_NEUTRALIZE = {s: s[:2] + "⁣" + s[2:] for s in SPECIALS}


def sanitize(text: str) -> str:
    """Strip the document separator and defuse special-token surfaces."""
    if "\x00" in text:
        text = text.replace("\x00", "")
    # `<|` is rare enough that this guard skips the replace loop on nearly
    # every document.
    if "<|" in text:
        for surface, safe in _NEUTRALIZE.items():
            if surface in text:
                text = text.replace(surface, safe)
    return text


class ShardWriter:
    """Writes NUL-separated documents into size-bounded shard files."""

    def __init__(self, out_dir: Path, prefix: str, target_bytes: int,
                 contamination_lines: set | None = None) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.target = target_bytes
        # --drop-contaminated: the benchmark line set; the rule itself is
        # decontaminate.doc_is_contaminated (ONE copy, so the dropping
        # filter and the verifying scan cannot disagree).
        self.contamination_lines = contamination_lines
        self.contaminated_dropped = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.fh = None
        self.written = 0
        self.docs_in_shard = 0
        self.digest = None
        self.shards: list[dict] = []
        self.total_bytes = 0
        self.total_docs = 0

    def _open(self) -> None:
        path = self.out_dir / f"{self.prefix}-{self.index:05d}.txt"
        self.fh = path.open("wb")
        self.path = path
        self.written = 0
        self.docs_in_shard = 0
        self.digest = hashlib.sha256()

    def _close(self) -> None:
        if self.fh is None:
            return
        self.fh.close()
        self.shards.append(
            {
                "file": self.path.name,
                "bytes": self.written,
                "documents": self.docs_in_shard,
                "sha256": self.digest.hexdigest(),
            }
        )
        print(
            f"[shard] {self.path.name}: {self.written / 1e9:.3f} GB, "
            f"{self.docs_in_shard:,} docs",
            flush=True,
        )
        self.index += 1
        self.fh = None

    def add(self, text: str) -> None:
        if self.contamination_lines is not None:
            from decontaminate import doc_is_contaminated

            bad, _ = doc_is_contaminated(text, self.contamination_lines)
            if bad:
                self.contaminated_dropped += 1
                return
        if self.fh is None:
            self._open()
        blob = text.encode("utf-8")
        # The separator precedes every document but the first in a shard, so a
        # shard never opens or closes with an empty document.
        if self.docs_in_shard:
            blob = b"\x00" + blob
        self.fh.write(blob)
        self.digest.update(blob)
        self.written += len(blob)
        self.docs_in_shard += 1
        self.total_bytes += len(blob)
        self.total_docs += 1
        if self.written >= self.target:
            self._close()

    def finish(self) -> None:
        self._close()


def render_distill(messages_json: str, tools_json: str | None) -> str | None:
    """Render one trace as ChatML, or None if it carries no usable turns."""
    try:
        messages = json.loads(messages_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(messages, list) or not messages:
        return None

    parts: list[str] = []

    if tools_json:
        try:
            tools = json.loads(tools_json)
        except (json.JSONDecodeError, TypeError):
            tools = None
        if tools:
            body = sanitize(json.dumps(tools, ensure_ascii=False, separators=(",", ":")))
            parts.append(f"<|im_start|>system\n<|tool_call|>{body}<|im_end|>\n")

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = sanitize(str(msg.get("role") or "user"))
        segments: list[str] = []

        reasoning = msg.get("reasoning_content")
        if reasoning:
            segments.append(sanitize(str(reasoning)))

        content = msg.get("content")
        if content:
            body = sanitize(str(content))
            # Mark a tool's return where it is emitted, so the model can tell an
            # observation from something it generated itself. Marking
            # `segments[-1]` after the fact would tag whichever segment happened
            # to land last -- the tool_call below, on any turn that has one.
            segments.append(f"<|tool_result|>{body}" if role == "tool" else body)

        calls = msg.get("tool_calls")
        if calls:
            body = sanitize(json.dumps(calls, ensure_ascii=False, separators=(",", ":")))
            segments.append(f"<|tool_call|>{body}")

        if not segments:
            continue
        parts.append(f"<|im_start|>{role}\n" + "\n".join(segments) + "<|im_end|>\n")

    if len(parts) < 2:
        return None
    return "".join(parts)


def iter_parquet(files: list[Path], columns: list[str]):
    """Yield row dicts in file-then-row order, one row group at a time."""
    for f in files:
        pf = pq.ParquetFile(f)
        for group in range(pf.num_row_groups):
            table = pf.read_row_group(group, columns=columns)
            cols = {c: table[c].to_pylist() for c in columns}
            for i in range(table.num_rows):
                yield {c: cols[c][i] for c in columns}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["fineweb", "distill", "code", "stack"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-bytes", type=int, default=1_000_000_000)
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="keep every Nth document; used to build a bounded tokenizer sample",
    )
    ap.add_argument("--max-bytes", type=int, default=0, help="0 = no cap")
    ap.add_argument("--drop-contaminated", action="store_true",
                    help="drop documents matching the benchmark line set "
                    "(decontaminate.doc_is_contaminated — the same rule the "
                    "verifying scan applies). Recommended for every val "
                    "extraction; the count lands in the manifest")
    ap.add_argument("--min-chars", type=int, default=1, help="drop shorter documents")
    ap.add_argument("--max-doc-bytes", type=int, default=4_000_000,
                    help="stack: start a new document once a repository passes this "
                    "size. The cut is BETWEEN files, never inside one, so a single "
                    "file larger than this still yields one larger document")
    ap.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="all",
        help="fineweb: val is the last parquet file, train is the rest; "
        "distill: selects the split's own parquet files by name",
    )
    args = ap.parse_args()

    contamination_lines = None
    if args.drop_contaminated:
        from decontaminate import benchmark_lines

        _, contamination_lines = benchmark_lines()
        print(f"[extract] --drop-contaminated armed: "
              f"{len(contamination_lines)} benchmark lines", flush=True)

    out = args.out if args.out.is_absolute() else REPO / args.out

    if args.source == "code":
        src = REPO / "data/tokcorpus/combined_train.txt"
        if not src.exists():
            print(f"missing {src}", file=sys.stderr)
            return 1
        writer = ShardWriter(out, "code", args.shard_bytes, contamination_lines)
        docs = src.read_text(encoding="utf-8", errors="replace").split("\x00")
        kept = 0
        for i, doc in enumerate(docs):
            if args.stride > 1 and i % args.stride:
                continue
            doc = sanitize(doc)
            if len(doc) < args.min_chars:
                continue
            writer.add(doc)
            kept += 1
        writer.finish()
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "source": "code",
                    "input": str(src.relative_to(REPO)),
                    "rows_seen": len(docs),
                    "documents": writer.total_docs,
                    "bytes": writer.total_bytes,
                    "contaminated_dropped": writer.contaminated_dropped,
                    "drop_contaminated_armed": args.drop_contaminated,
                    "specials_neutralized": SPECIALS,
                    "shards": writer.shards,
                },
                indent=2,
            )
        )
        print(f"[extract] {kept:,} documents, {writer.total_bytes / 1e9:.3f} GB -> {out}")
        return 0

    if args.source == "stack":
        # val reads the SEPARATE stack-val dir (a shard off the training
        # stride, fetched by fetch.py --only stack-val) so the split is
        # repository-disjoint by construction — and verifiably so:
        # verify.py intersects the repo_id columns of both dirs.
        subdir = "stack-val" if args.split == "val" else "stack"
        src = REPO / f"data/hf/{subdir}/data"
        files = sorted(src.glob("*.parquet"))
        if not files:
            hint = "--only stack-val" if args.split == "val" else "--only stack"
            print(f"no parquet under {src} — run fetch.py {hint}", file=sys.stderr)
            return 1
        print(f"[extract] stack: {len(files)} parquet files from {src}", flush=True)
        writer = ShardWriter(out, "stack", args.shard_bytes, contamination_lines)
        seen = kept = skipped = 0
        for row in iter_parquet(files, ["repo_path", "files"]):
            seen += 1
            if args.stride > 1 and (seen - 1) % args.stride:
                continue
            buf: list[str] = []
            size = 0
            emitted = False
            for f in row["files"] or ():
                content = f.get("content")
                if not content:
                    continue
                piece = (
                    f"<|file_sep|>{sanitize(str(f.get('file_path') or ''))}\n"
                    f"{sanitize(str(content))}"
                )
                buf.append(piece)
                size += len(piece)
                if size >= args.max_doc_bytes:
                    writer.add("".join(buf))
                    kept += 1
                    emitted = True
                    buf, size = [], 0
                    # Check the cap HERE too, not only per repository. One
                    # monorepo can emit hundreds of documents, and a per-repo
                    # check let a 700 MB request overshoot to 1.19 GB.
                    if args.max_bytes and writer.total_bytes >= args.max_bytes:
                        break
            if buf and size >= args.min_chars and not (
                args.max_bytes and writer.total_bytes >= args.max_bytes
            ):
                writer.add("".join(buf))
                kept += 1
                emitted = True
            if not emitted:
                skipped += 1
            if args.max_bytes and writer.total_bytes >= args.max_bytes:
                print(f"[extract] hit --max-bytes at {writer.total_bytes:,}", flush=True)
                break
            if kept and kept % 200_000 == 0:
                print(f"[extract] {kept:,} docs / {seen:,} repos, "
                      f"{writer.total_bytes / 1e9:.2f} GB", flush=True)
        writer.finish()
        (out / "manifest.json").write_text(json.dumps({
            "source": "stack", "split": args.split,
            "parquet_files": [f.name for f in files],
            "stride": args.stride, "max_bytes": args.max_bytes,
            "max_doc_bytes": args.max_doc_bytes,
            "rows_seen": seen, "documents": writer.total_docs, "skipped": skipped,
            "bytes": writer.total_bytes,
            "contaminated_dropped": writer.contaminated_dropped,
            "drop_contaminated_armed": args.drop_contaminated,
            "specials_neutralized": SPECIALS,
            "shards": writer.shards,
        }, indent=2))
        print(f"[extract] {writer.total_docs:,} documents from {seen:,} repos, "
              f"{writer.total_bytes / 1e9:.2f} GB in {len(writer.shards)} shards -> {out}",
              flush=True)
        return 0

    if args.source == "fineweb":
        src = REPO / "data/hf/fineweb/sample/10BT"
        files = sorted(src.glob("*.parquet"))
        # FineWeb's sample-10BT has no split of its own, so the split is cut at
        # FILE granularity: the last shard is held out whole. Cutting at a row
        # offset inside a shared file would be index-disjoint but not separated
        # -- adjacent FineWeb rows come from the same crawl segment and often
        # the same site -- and it would make the boundary depend on a row count
        # that changes if the dataset is ever re-exported.
        if args.split == "train":
            files = files[:-1]
        elif args.split == "val":
            files = files[-1:]
        columns = ["text"]
        prefix = "web"
    else:
        src = REPO / "data/hf/distill/data/canonical"
        # -of-00006 is the current export (52,205 train rows, matching the card's
        # 52,205/2,872/2,860 = 57,937). A stale -of-00005 series sits in the same
        # directory; globbing `train-*` would concatenate both and duplicate
        # 41,682 traces. Name the series explicitly.
        files = sorted(src.glob("train-*-of-00006.parquet"))
        files += sorted(src.glob("validation-*.parquet"))
        files += sorted(src.glob("test-*.parquet"))
        columns = ["messages_json", "tools_json"]
        prefix = "sft"
        if args.split == "train":
            files = sorted(src.glob("train-*-of-00006.parquet"))
        elif args.split == "val":
            files = sorted(src.glob("validation-*.parquet"))

    if not files:
        print(f"no parquet under {src} — run fetch.py first", file=sys.stderr)
        return 1

    print(f"[extract] {args.source}: {len(files)} parquet files from {src}", flush=True)

    writer = ShardWriter(out, prefix, args.shard_bytes, contamination_lines)
    seen = 0
    kept = 0
    skipped = 0

    for row in iter_parquet(files, columns):
        seen += 1
        if args.stride > 1 and (seen - 1) % args.stride:
            continue
        if args.source == "fineweb":
            text = row["text"]
            text = sanitize(text) if text else ""
        else:
            text = render_distill(row["messages_json"], row.get("tools_json"))
            if text is None:
                skipped += 1
                continue
        if len(text) < args.min_chars:
            skipped += 1
            continue
        writer.add(text)
        kept += 1
        if args.max_bytes and writer.total_bytes >= args.max_bytes:
            print(f"[extract] hit --max-bytes at {writer.total_bytes:,}", flush=True)
            break
        if kept % 500_000 == 0:
            print(
                f"[extract] {kept:,} kept / {seen:,} seen, "
                f"{writer.total_bytes / 1e9:.2f} GB",
                flush=True,
            )

    writer.finish()

    manifest = {
        "source": args.source,
        "split": args.split,
        "parquet_files": [f.name for f in files],
        "stride": args.stride,
        "max_bytes": args.max_bytes,
        "rows_seen": seen,
        "documents": writer.total_docs,
        "skipped": skipped,
        "bytes": writer.total_bytes,
        "contaminated_dropped": writer.contaminated_dropped,
        "drop_contaminated_armed": args.drop_contaminated,
        "specials_neutralized": SPECIALS,
        "shards": writer.shards,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[extract] {writer.total_docs:,} documents, {writer.total_bytes / 1e9:.2f} GB "
        f"in {len(writer.shards)} shards -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
