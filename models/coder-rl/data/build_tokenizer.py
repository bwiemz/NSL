#!/usr/bin/env python3
"""Train a 4096-vocab NSL-aware BPE tokenizer.

Trains on all NSL sources (stdlib, examples, models, spec) so the BPE
merges learn both NSL code patterns and English prose patterns. This
ensures efficient encoding of English prompts during GRPO Phase 3.

Usage:
    py build_tokenizer.py
"""

import json
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
except ImportError:
    print("Install: py -m pip install tokenizers")
    sys.exit(1)

from vocab import build_fixed_vocab, VOCAB_SIZE

# Tokens 0-9 are truly special (pad, bos, eos, etc.) and should be skipped on decode.
# Tokens 10-399 are keywords/operators/builtins that appear in normal code — they must
# NOT be treated as special so that tok.decode() includes them in the output.
_TRULY_SPECIAL_CUTOFF = 10


def collect_corpus_files(repo_root: Path) -> list[str]:
    """Collect all training corpus files (all 4 sources)."""
    paths: list[str] = []
    for d in ["stdlib", "examples", "models"]:
        for f in sorted((repo_root / d).rglob("*.nsl")):
            paths.append(str(f))
    for f in sorted((repo_root / "spec").rglob("*.md")):
        paths.append(str(f))
    return paths


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_path = Path(__file__).resolve().parent / "tokenizer" / "nsl_coder_4k.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_vocab = build_fixed_vocab()
    print(f"Fixed vocabulary: {len(fixed_vocab)} tokens (0-{len(fixed_vocab)-1})")

    corpus_files = collect_corpus_files(repo_root)
    print(f"Training corpus: {len(corpus_files)} files")
    for f in corpus_files[:5]:
        print(f"  {f}")
    if len(corpus_files) > 5:
        print(f"  ... and {len(corpus_files) - 5} more")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=fixed_vocab,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train(corpus_files, trainer)

    tokenizer.save(str(out_path))

    # Post-process: mark tokens 10-399 as non-special so tok.decode() includes them.
    # BpeTrainer registers all special_tokens with special=True, but keywords/operators
    # must round-trip through decode() just like regular BPE tokens.
    data = json.loads(out_path.read_text(encoding="utf-8"))
    for entry in data["added_tokens"]:
        if entry["id"] >= _TRULY_SPECIAL_CUTOFF:
            entry["special"] = False
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved tokenizer to {out_path}")

    # Reload from disk so the in-memory object reflects the patched special flags.
    tokenizer = Tokenizer.from_file(str(out_path))
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    test_code = 'fn forward(self, x: Tensor) -> Tensor:\n    return relu(x @ self.w)'
    encoded = tokenizer.encode(test_code)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nTest encode/decode:")
    print(f"  Input:   {test_code!r}")
    print(f"  Tokens:  {len(encoded.ids)} ids")
    print(f"  Decoded: {decoded!r}")


if __name__ == "__main__":
    main()
