#!/usr/bin/env python3
"""Encode text shards into the flat u16 token stream NSL's DataLoader mmaps.

The on-disk format is what `load_mmap(path, 3)` already expects and is not
changed here: little-endian u16 token ids, no header, no padding. Vocab 49152
plus 6 reserved specials stays inside the u16 id space.

Each shard is encoded independently and the results are concatenated. That is
exact, not approximate: `tokbench pretokenize` carries no encoder state across
documents, so shard-then-concatenate is byte-identical to encoding one giant
corpus file -- which is just as well, because tokbench builds the whole token
stream in memory and a 40 GB corpus file would not fit.

`--doc-sep-token` puts one end-of-document id between documents. v1 emitted no
boundary at all, so a window straddling two documents gave the model the end of
one page as context for predicting the start of an unrelated one.

Usage (from the repo root):
    tools/hfcorpus/.venv/bin/python tools/hfcorpus/pretokenize.py \
        --text data/text/web-train --out data/tokens/mix/web_train.bin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOKBENCH = REPO / "tools/tokbench/target/release/tokbench"
DEFAULT_TOKENIZER = REPO / "data/tokenizer/nsl_mix_v2_t40960_v49152.json"



def rel(path: Path) -> str:
    """Repo-relative when the path is inside the repo, absolute otherwise.

    Manifests are written for paths the caller chose, which may sit outside the
    checkout (a scratch directory, another disk). `relative_to` raises there, so
    recording provenance would crash after the corpus was already written.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)
def encode_shard(tokenizer: Path, shard: Path, out: Path, doc_sep: str | None) -> int:
    cmd = [str(TOKBENCH), "pretokenize", "--tokenizer", str(tokenizer),
           "--corpus", str(shard), "--out", str(out)]
    if doc_sep:
        cmd += ["--doc-sep-token", doc_sep]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"tokbench pretokenize failed on {shard.name}")
    print(f"[encode] {shard.name}: {proc.stdout.strip()}", flush=True)
    return out.stat().st_size // 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=Path, required=True, help="directory of *.txt shards")
    ap.add_argument("--out", type=Path, required=True, help="destination .bin")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--doc-sep-token", default="<|endoftext|>",
                    help="pass an empty string to emit no boundary")
    ap.add_argument("--reap", action="store_true",
                    help="delete each text shard once encoded, to bound peak disk")
    ap.add_argument("--clean-parts", action="store_true",
                    help="remove the per-shard .bin parts after concatenating. They "
                    "are kept by default because they make a 91-shard run resumable")
    args = ap.parse_args()

    text_dir = args.text if args.text.is_absolute() else REPO / args.text
    out = args.out if args.out.is_absolute() else REPO / args.out
    tokenizer = args.tokenizer if args.tokenizer.is_absolute() else REPO / args.tokenizer
    # Emptiness is judged AFTER the roster is resolved: a fully-reaped resume has
    # no .txt left but every part encoded, and refusing it here would make --reap
    # and resume mutually exclusive in the one case they are meant to cover.
    shards = sorted(text_dir.glob("*.txt"))
    if not tokenizer.exists():
        raise SystemExit(f"tokenizer not found: {tokenizer}")

    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.parent / f".{out.stem}.parts"
    scratch.mkdir(parents=True, exist_ok=True)

    # A cached part is only reusable if it was produced by the same tokenizer
    # and the same separator. Without this, rerunning after retraining the
    # tokenizer reuses stale bytes while the manifest records the NEW tokenizer
    # and a sha256 of the old stream -- a corpus that lies about its own
    # provenance.
    key = {
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        "doc_sep_token": args.doc_sep_token or None,
    }
    keyfile = scratch / "cache_key.json"
    stale_reason = None
    if keyfile.exists():
        if json.loads(keyfile.read_text()) != key:
            stale_reason = "tokenizer or separator changed"
    elif any(scratch.glob("*.bin")):
        # Parts with NO key were written before this check existed, or by a run
        # whose key was lost. Their provenance is unknown, which is exactly the
        # case the key is for -- accepting them because there is nothing to
        # compare against would make the guard unreachable for its own motivating
        # case.
        stale_reason = "cached parts carry no tokenizer key"
    if stale_reason:
        print(f"[encode] {stale_reason} — discarding cached parts", flush=True)
        for stale in scratch.glob("*.bin"):
            stale.unlink()
        (scratch / "roster.json").unlink(missing_ok=True)
    keyfile.write_text(json.dumps(key, indent=2))

    # The shard ROSTER is recorded on first use and reused thereafter. Deriving
    # it from glob() on every run is what makes --reap and resume incompatible:
    # a reaped shard vanishes from the glob, so a resumed run would concatenate
    # only the shards that had not been encoded yet and emit a manifest and a
    # plausible B/tok for a silently truncated corpus.
    roster_file = scratch / "roster.json"
    if roster_file.exists():
        roster = json.loads(roster_file.read_text())
        present = {s.name for s in shards}
        missing = [r for r in roster if r["shard"] not in present
                   and not (scratch / f"{Path(r['shard']).stem}.bin").exists()]
        if missing:
            raise SystemExit(
                f"{len(missing)} shard(s) from the recorded roster are gone and were "
                f"never encoded (first: {missing[0]['shard']}). Re-extract the text "
                f"or delete {scratch} to start over."
            )
    else:
        if not shards:
            raise SystemExit(f"no *.txt shards under {text_dir} and no recorded roster")
        roster = [{"shard": s.name, "bytes": s.stat().st_size} for s in shards]
        roster_file.write_text(json.dumps(roster, indent=2))

    started = time.time()
    parts: list[dict] = []
    text_bytes = 0
    for entry in roster:
        shard = text_dir / entry["shard"]
        text_bytes += entry["bytes"]
        part = scratch / f"{Path(entry['shard']).stem}.bin"
        if not part.exists():
            if not shard.exists():
                raise SystemExit(f"shard {shard} is missing and has no encoded part")
            n = encode_shard(tokenizer, shard, part, args.doc_sep_token or None)
        else:
            n = part.stat().st_size // 2
            print(f"[encode] {entry['shard']}: cached ({n:,} tokens)", flush=True)
        parts.append({"shard": entry["shard"], "tokens": n})
        if args.reap and shard.exists():
            shard.unlink()

    # Concatenate. Streamed rather than read-then-write so peak memory stays at
    # one buffer regardless of corpus size.
    digest = hashlib.sha256()
    total = 0
    with out.open("wb") as dst:
        for part in parts:
            src = scratch / f"{Path(part['shard']).stem}.bin"
            with src.open("rb") as fh:
                while chunk := fh.read(1 << 24):
                    dst.write(chunk)
                    digest.update(chunk)
            total += part["tokens"]

    # Recorded for the manifest, NOT as a fit check: this stream is read back as
    # <u2, so a value above 65535 is unrepresentable here and a check against it
    # could never fire. Truncation would already have happened upstream, in
    # tokbench's `as u16` cast, which is where the guard lives. At 10B tokens the
    # scan still has to be vectorized -- a Python-level max() takes minutes.
    import numpy as np

    stream = np.memmap(out, dtype="<u2", mode="r")
    max_id = 0
    for start in range(0, stream.size, 1 << 26):
        block = stream[start : start + (1 << 26)]
        if block.size:
            max_id = max(max_id, int(block.max()))
    del stream

    elapsed = time.time() - started
    manifest = {
        "output": rel(out),
        "tokenizer": rel(tokenizer),
        "doc_sep_token": args.doc_sep_token or None,
        "text_bytes": text_bytes,
        "tokens": total,
        "bytes_per_token": round(text_bytes / total, 4) if total else None,
        "max_token_id": max_id,
        "sha256": digest.hexdigest(),
        "shards": parts,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.clean_parts:
        for part in parts:
            (scratch / f"{Path(part['shard']).stem}.bin").unlink(missing_ok=True)
        keyfile.unlink(missing_ok=True)
        roster_file.unlink(missing_ok=True)
        scratch.rmdir()
    print(
        f"[done] {total:,} tokens ({out.stat().st_size / 1e9:.2f} GB) "
        f"from {text_bytes / 1e9:.2f} GB text = {text_bytes / total:.3f} B/tok, "
        f"max id {max_id}, {elapsed / 60:.1f} min -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
