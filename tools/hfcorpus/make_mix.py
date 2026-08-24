#!/usr/bin/env python3
"""Interleave tokenized corpora into one training stream at a chosen ratio.

NSL's `load_mmap(path, 3)` takes a single file, so a mixture has to be
materialized rather than expressed as a loader weight. Mixing happens HERE, on
token ids, rather than back at the text stage, so changing the ratio costs one
pass over the .bin files instead of a full retokenization.

Documents are located by the `<|endoftext|>` id that `pretokenize.py` writes
after each one, and are emitted whole. Interleaving is document-granular and
seeded: a mixture that appended the code corpus after the web corpus would give
the model no code at all until the end of the epoch, and one that cut at token
offsets would splice unrelated documents into single windows.

The minority corpus is repeated as needed to reach its target share. Repetition
is reported, not hidden: `data/tokcorpus/combined_train.txt` is ~9M tokens
against ~10B of web, so any code share above ~0.1% means seeing the code corpus
more than once, and a high share means seeing it many times.

Usage (from the repo root):
    tools/hfcorpus/.venv/bin/python tools/hfcorpus/make_mix.py \
        --base data/tokens/mix/web_train.bin \
        --add data/tokens/mix/code_train.bin:0.03 \
        --out data/tokens/mix/pretrain_train.bin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]



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
def load_specials() -> dict[str, int]:
    """Reserved ids, preferring the copy committed beside the tokenizer.

    `data/` is gitignored, so a fresh clone has the tokenizer but not the
    training scratch directory; falling back to it keeps an existing local run
    working without making the committed tokenizer unusable on its own.
    """
    for path in (REPO / "models/tokenizers/special_tokens.json",
                 REPO / "data/tokenizer/special_tokens.json"):
        if path.exists():
            return json.loads(path.read_text())["ids"]
    raise SystemExit("no special_tokens.json — run train_tokenizer.py first")


def document_offsets(tokens: np.ndarray, eos: int) -> np.ndarray:
    """Start index of each document, given a stream with a trailing eos per doc.

    Scanned in blocks: `tokens == eos` over a 10B-token stream would materialize
    a 10 GB boolean mask before flatnonzero ever saw it.
    """
    found = []
    block = 1 << 26
    for start in range(0, tokens.size, block):
        hits = np.flatnonzero(tokens[start : start + block] == eos)
        if hits.size:
            found.append(hits + start)
    ends = np.concatenate(found) if found else np.empty(0, dtype=np.int64)
    if ends.size == 0:
        raise SystemExit(
            "no end-of-document id in this stream — it was encoded without "
            "--doc-sep-token, so documents cannot be located"
        )
    # Documents are [start, eos] spans, so tokens past the FINAL eos belong to
    # no document and would be silently dropped from the mixture. The shipped
    # streams all end in eos (verified: the mix's base count equals
    # web_train.bin's exactly), so this refusal is currently vacuous — it
    # exists for the input that breaks the assumption, which is precisely when
    # a silent truncation would otherwise be unfindable.
    if ends[-1] != tokens.size - 1:
        raise SystemExit(
            f"stream does not end with the end-of-document id: "
            f"{tokens.size - 1 - ends[-1]} trailing token(s) after the last "
            "eos would be silently dropped. Re-encode with --doc-sep-token."
        )
    starts = np.empty(ends.size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    return np.stack([starts, ends + 1], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--add", action="append", default=[],
                    help="PATH:SHARE, e.g. data/tokens/mix/code_train.bin:0.03")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    eos = load_specials()["<|endoftext|>"]
    base_path = args.base if args.base.is_absolute() else REPO / args.base
    out = args.out if args.out.is_absolute() else REPO / args.out

    base = np.memmap(base_path, dtype="<u2", mode="r")
    base_docs = document_offsets(base, eos)
    print(f"[mix] base {base_path.name}: {base.size:,} tokens, {len(base_docs):,} docs", flush=True)

    # Each addition's share is of the FINAL total, so the base share is
    # 1 - sum(shares) and the base is never itself repeated.
    adds = []
    total_share = 0.0
    for spec in args.add:
        path_str, sep, share_str = spec.rpartition(":")
        if not sep or not path_str:
            raise SystemExit(f"--add wants PATH:SHARE, got {spec!r}")
        try:
            share = float(share_str)
        except ValueError:
            raise SystemExit(f"--add share is not a number: {spec!r}") from None
        if not 0 < share < 1:
            raise SystemExit(f"--add share must be in (0, 1), got {share}")
        total_share += share
        p = Path(path_str)
        p = p if p.is_absolute() else REPO / p
        arr = np.memmap(p, dtype="<u2", mode="r")
        docs = document_offsets(arr, eos)
        adds.append({"path": p, "tokens": arr, "docs": docs, "share": share})
        print(f"[mix] add  {p.name}: {arr.size:,} tokens, {len(docs):,} docs, target {share:.1%}", flush=True)

    if total_share >= 1.0:
        raise SystemExit(f"added shares sum to {total_share:.3f}; must be < 1.0")

    base_share = 1.0 - total_share
    final_total = int(base.size / base_share)

    for a in adds:
        want = int(final_total * a["share"])
        epochs = want / a["tokens"].size
        a["want"] = want
        a["epochs"] = epochs
        print(
            f"[mix] {a['path'].name}: {want:,} tokens of {a['tokens'].size:,} "
            f"= {epochs:.2f} epochs of that corpus",
            flush=True,
        )

    rng = random.Random(args.seed)

    # Build the emission order: base documents in order, with the additions'
    # documents spliced in at seeded random positions. Keeping the base in order
    # preserves FineWeb's own dedup locality; only the injections are random.
    schedule: list[tuple[int, int]] = [(-1, i) for i in range(len(base_docs))]
    for ai, a in enumerate(adds):
        emitted = 0
        picks = []
        di = 0
        while emitted < a["want"]:
            start, end = a["docs"][di % len(a["docs"])]
            picks.append(di % len(a["docs"]))
            emitted += end - start
            di += 1
        for p in picks:
            schedule.append((ai, p))
        print(f"[mix] {a['path'].name}: {len(picks):,} document emissions", flush=True)

    # Shuffle only the injected entries into the base sequence, preserving base
    # order: sort by a key that is the base index for base docs and a random
    # position for injections.
    base_positions = len(base_docs)
    keyed = []
    bi = 0
    for src, idx in schedule:
        if src == -1:
            keyed.append((float(idx), src, idx))
        else:
            keyed.append((rng.uniform(0, base_positions), src, idx))
    keyed.sort(key=lambda t: t[0])

    out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    for a in adds:
        a["key"] = rel(a["path"])
    if len({a["key"] for a in adds}) != len(adds):
        raise SystemExit("two --add entries resolve to the same path")
    counts = {"base": 0, **{a["key"]: 0 for a in adds}}
    with out.open("wb") as fh:
        buf = []
        buf_tokens = 0
        for _, src, idx in keyed:
            if src == -1:
                start, end = base_docs[idx]
                chunk = base[start:end]
                counts["base"] += int(end - start)
            else:
                a = adds[src]
                start, end = a["docs"][idx]
                chunk = a["tokens"][start:end]
                counts[a["key"]] += int(end - start)
            buf.append(chunk)
            buf_tokens += chunk.size
            if buf_tokens >= (1 << 23):
                blob = np.concatenate(buf).tobytes()
                fh.write(blob)
                digest.update(blob)
                written += buf_tokens
                buf, buf_tokens = [], 0
        if buf:
            blob = np.concatenate(buf).tobytes()
            fh.write(blob)
            digest.update(blob)
            written += buf_tokens

    manifest = {
        "output": rel(out),
        "seed": args.seed,
        "eos_id": int(eos),
        "tokens": int(written),
        "sha256": digest.hexdigest(),
        "composition": {
            k: {"tokens": int(v), "share": round(v / written, 5)}
            for k, v in counts.items()
        },
        "repetition": {
            a["key"]: round(a["epochs"], 3) for a in adds
        },
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[mix] {written:,} tokens -> {out} ({out.stat().st_size / 1e9:.2f} GB)")
    for k, v in counts.items():
        print(f"       {k:<24} {v:>14,}  {v / written:6.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
