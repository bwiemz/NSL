# Tokenizers of record

Committed deliberately. `v1` was previously written only into a Claude session
scratchpad under `/tmp`, which was wiped; `data/tokens/train_new.bin` was the
sole surviving evidence of it, and every corpus pinned to it would have been
unreproducible. Both files are ~1.7 MB.

| file | corpus it encodes | reserved specials |
|---|---|---|
| `nsl_code_v1_t40960_v49152.json` | `data/tokens/train_new.bin` and its slices | none |
| `nsl_mix_v2_t40960_v49152.json` | `data/tokens/mix/*.bin` (web + code + chat) | 6 |

Both are two-stage BPE, stage-1 `cl100k` relaxing to whole lines at
`transition=40960`, `vocab_size=49152`, `min_freq=2`. 49152 is load-bearing: the
`load_mmap(path, 3)` stream format is u16, so the id space must stay under
65536.

## v1 is reproducible, and that is verified

    tools/tokbench/target/release/tokbench train2 \
      --corpus data/tokcorpus/combined_train.txt \
      --out /tmp/v1.json --stage1 cl100k --relaxed line \
      --vocab-size 49152 --transition 40960 --min-freq 2 --max-token-bytes 0

    tools/tokbench/target/release/tokbench pretokenize \
      --tokenizer /tmp/v1.json --corpus data/tokcorpus/combined_train.txt \
      --out /tmp/v1.bin
    cmp /tmp/v1.bin data/tokens/train_new.bin

This reproduces 8,925,916 tokens at 5.403 B/tok and `cmp` reports no
difference — which recovered the file, and also establishes that
`train_two_stage` is deterministic given the same corpus and spec.

## v1 is not retired

`v1` remains the tokenizer of record for `data/tokens/train_new.bin`,
`prod_train_slice.bin`, `prod_val_slice.bin`, `sr_*_slice.bin`, and everything
pinned against them — `crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs`,
`models/coder{50m,500m,1b}/config.nsl`, and the 5.403 B/tok provenance
fingerprint in `models/benchmarks/TOKENIZER_AND_MFU_2026_07_29.md`. `v2` writes
to new paths and changes none of them.

## Reserved ids are data, not a convention

`v2`'s special-token ids are emitted at training time into
`data/tokenizer/special_tokens.json`. Read them from there. They are assigned by
`add_special_tokens` after the learned merges, so do not assume they are
contiguous, or first, or last.
