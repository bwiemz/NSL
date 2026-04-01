#!/usr/bin/env python3
"""Build pre-tokenized SFT corpus for coder-rl Phase 1 training.

Encodes pure NSL code (no English prose) into binary format:
  - sft_tokens.bin: u16 input token IDs, padded with PAD_ID (0)
  - sft_labels.bin: i32 target labels shifted by 1, padded with -100

Usage:
    py prepare_sft.py
"""

import re
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Install: py -m pip install tokenizers")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Install: py -m pip install numpy")
    sys.exit(1)

from vocab import BOS_ID, EOS_ID, PAD_ID

SEQ_LEN = 2048
LABEL_IGNORE = -100


def collect_nsl_files(repo_root: Path) -> list[Path]:
    """Collect all .nsl files from stdlib, examples, models."""
    files: list[Path] = []
    for d in ["stdlib", "examples", "models"]:
        files.extend(sorted((repo_root / d).rglob("*.nsl")))
    return files


def extract_nsl_blocks(markdown_text: str) -> str:
    """Extract NSL code blocks from markdown spec files.

    Uses forgiving whitespace boundaries to handle inconsistent formatting.
    Falls back to unfenced code blocks that contain NSL keywords.
    """
    # Primary: fenced ```nsl blocks
    blocks = re.findall(r"```nsl\s*(.*?)\s*```", markdown_text, re.DOTALL)

    # Fallback: unfenced ``` blocks containing NSL keywords
    if not blocks:
        generic = re.findall(r"```\s*(.*?)\s*```", markdown_text, re.DOTALL)
        nsl_keywords = {"fn ", "let ", "const ", "model ", "train(", "from "}
        blocks = [b for b in generic if any(kw in b for kw in nsl_keywords)]

    return "\n\n".join(blocks)


def collect_spec_code(repo_root: Path) -> list[tuple[str, str]]:
    """Extract NSL code blocks from spec markdown files."""
    results: list[tuple[str, str]] = []
    spec_dir = repo_root / "spec"
    if not spec_dir.exists():
        return results
    for f in sorted(spec_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8", errors="replace")
        code = extract_nsl_blocks(text)
        if code.strip():
            results.append((f"spec/{f.name}", code))
    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(__file__).resolve().parent
    tok_path = out_dir / "tokenizer" / "nsl_coder_4k.json"

    if not tok_path.exists():
        print(f"Tokenizer not found at {tok_path}")
        print("Run build_tokenizer.py first.")
        sys.exit(1)

    tok = Tokenizer.from_file(str(tok_path))

    # 1. Collect code-only documents
    documents: list[tuple[str, str]] = []
    for f in collect_nsl_files(repo_root):
        text = f.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            rel = f.relative_to(repo_root)
            documents.append((str(rel), text))

    spec_docs = collect_spec_code(repo_root)
    documents.extend(spec_docs)

    print(f"Collected {len(documents)} documents:")
    nsl_count = sum(1 for name, _ in documents if not name.startswith("spec/"))
    spec_count = sum(1 for name, _ in documents if name.startswith("spec/"))
    print(f"  .nsl files: {nsl_count}")
    print(f"  spec code blocks: {spec_count}")

    # 2. Encode each document, build input/label pairs with padding
    all_inputs: list[int] = []
    all_labels: list[int] = []
    total_real_tokens = 0
    total_pad_tokens = 0

    for name, text in documents:
        ids = [BOS_ID] + tok.encode(text).ids + [EOS_ID]
        total_real_tokens += len(ids)

        inp = ids[:-1]
        lbl = ids[1:]

        padded_len = ((len(inp) + SEQ_LEN - 1) // SEQ_LEN) * SEQ_LEN
        pad_count = padded_len - len(inp)
        total_pad_tokens += pad_count

        inp += [PAD_ID] * pad_count
        lbl += [LABEL_IGNORE] * pad_count

        all_inputs.extend(inp)
        all_labels.extend(lbl)

    # 3. Write binary files
    inputs_arr = np.array(all_inputs, dtype=np.uint16)
    labels_arr = np.array(all_labels, dtype=np.int32)

    inputs_path = out_dir / "sft_tokens.bin"
    labels_path = out_dir / "sft_labels.bin"

    inputs_arr.tofile(str(inputs_path))
    labels_arr.tofile(str(labels_path))

    n_seq = len(all_inputs) // SEQ_LEN
    pad_pct = total_pad_tokens / len(all_inputs) * 100

    print(f"\nCorpus statistics:")
    print(f"  Real tokens:     {total_real_tokens:,}")
    print(f"  Pad tokens:      {total_pad_tokens:,} ({pad_pct:.1f}%)")
    print(f"  Total tokens:    {len(all_inputs):,}")
    print(f"  Sequences:       {n_seq} (seq_len={SEQ_LEN})")
    print(f"  sft_tokens.bin:  {inputs_path.stat().st_size:,} bytes")
    print(f"  sft_labels.bin:  {labels_path.stat().st_size:,} bytes")
    print(f"\nAt 10 epochs: ~{n_seq * 10} forward+backward passes")


if __name__ == "__main__":
    main()
