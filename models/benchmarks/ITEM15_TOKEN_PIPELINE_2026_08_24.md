# Item 15 closure — direct u16 corpus generation, no float token IDs (2026-08-24)

Roadmap row: "Dataset manifest + tokenizer pipeline — hashes; direct u16
corpus generation (no f64 token IDs)". The manifest/hashes half shipped in
PR #523. This records what the remainder turned out to be, what changed, and
what deliberately did not.

## What was already true (verified, not assumed)

- **The corpus IS direct u16 end to end.** `tokbench pretokenize` writes
  little-endian u16 with an `as u16` guard on the max vocab id
  (`tools/tokbench/src/main.rs`); every mix/concat step
  (`tools/hfcorpus/{pretokenize,make_mix}.py`) reads and writes `<u2` via
  numpy; `load_mmap(path, 3)` mmaps it zero-copy as `DTYPE_U16_TOKEN`
  (`data_source.rs`). The f64 token-ID era was in-memory only: the old
  dtype-3 arm expanded u16→f64 at load until `1592e4d3` made it zero-copy.
- **The non-packed CPU batch is already integer.** `build_simple_batch`
  materializes `DTYPE_I32` input_ids/labels; GPU migration widens u16→i32;
  embedding lookup and fused-CE label conversion both take i32/u16 directly.

## What this PR changed

1. **The reference encoder is committed.** The `pretokenize
   --doc-sep-token` / `train2 --special-token` flags that built the v2
   corpus existed only in a locally patched binary — the corpus of record
   could not be rebuilt from any committed tree. The patch was recovered
   from the build worktree, committed, and the rebuilt binary re-encoded two
   full corpus sets sha256-identical to the manifest (`nsl_train`
   11,413,038 tokens; `sft_train` 41,020,908 tokens with 212,849
   `<|im_start|>` surface extractions). See models/tokenizers/README.md.
2. **The last float token-ID carrier in the training path is gone.** The
   packed batch builder (`packing.rs::packed_batch_to_dict`) wrote
   input_ids/labels as f32 tensors; they are now `DTYPE_I32`, matching the
   non-packed contract. (The f32 was lossless — ids < 2^24 — this is the
   pipeline-consistency half of the item, not a correctness fix.)
3. **Gates.** `tokbench_reference_encoder_gate.rs`: the committed
   tokenizer's added-token extraction semantics (the corpus depends on it —
   `<|file_sep|>` arrives as literal renderer text at 388-419/M tokens) runs
   in every CI build; the cert lane's multiproc tier builds tokbench from
   the tree and diffs its u16 stream against crate-computed ids, including
   the max-id refusal. Found while writing it: the `tokenizers` crate
   silently re-assigns an added token's declared id on load (70000 →
   next-free 49150), so an added-token id can never overflow u16 — only a
   model-vocab id can, and that is what the refusal arm plants.

## Deliberate non-goals (documented, not silent)

- **The runtime inference tokenizer** (`tokenizer.rs`:
  `nsl_tokenizer_encode/decode/encode_batch`) still moves ids through f64
  tensors. It is the inference/generate surface, not corpus generation;
  retiring it means changing what `generate.nsl` consumers do arithmetic
  on. Left for the item-19 docs pass to record as a known wart, or a
  dedicated interop item.
- **`load_mmap(path, 2)`** still expands i32→f64 at load (4x). Sole user:
  coder-rl's `sft_labels.bin`. Accepting i32 sources natively means
  widening the DataLoader's accepted-dtype list; not worth the churn for a
  small legacy path — noted here instead.
- **muon f32 validation bins** (`muon_data.py` writes `<f` val slices).
  Legacy benchmark inputs, exact for ids < 2^24, loaded via
  `load_mmap(..,1)`. Regenerating them would invalidate banked results for
  zero behavioral gain.
- **segment_ids / doc_starts / position_ids f32 backing** in the packed
  dict: not token ids; each has GPU-kernel consumers with their own
  documented promotion TODOs (see packing.rs).
