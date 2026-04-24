# PCA Backward CUDA Handoff

Status: in progress

## Confirmed fixed

- Original masked backward illegal-address fault is fixed.
  - Root cause was uninitialized `%k_start` / `%k_max` in backward prelude.
- PCA backward launcher ABI drift is fixed.
  - `dx_norm_ptr` is now passed in the correct slot.
- Test harness uploads `wv_dev` now.
- The original masked-no-op fixture now passes end-to-end:
  - `cargo test -p nsl-codegen --features cuda --test pca_tier_a_backward_correctness tier_a_backward_single_segment_matches_unmasked_baseline -- --ignored --nocapture --test-threads=1`

## Current kernel/runtime shape

- Backward runtime launch is serialized over q-blocks in `nsl_flash_attention_csha_backward`.
  - `grid_x` is forced to `1`.
  - The per-launch q-block base is carried through the otherwise-unused backward `seq_lens_ptr` slot and added into `%q_start` in the backward prelude.
- `dK` / `dV` finalize no longer use the original plain overwrite semantics from the multi-block launch path.
- The single-segment masked fixture is green with these changes.

## Current failing fixtures

The full ignored fixture file still fails:

- `tier_a_backward_two_equal_segments_matches_unpacked_reference`
- `tier_a_backward_unequal_segments_matches_unpacked_reference`

Most recent command:

```powershell
cargo test -p nsl-codegen --features cuda --test pca_tier_a_backward_correctness -- --ignored --nocapture --test-threads=1
```

## Latest observed failures

- Fixture 2 still fails in the reference side for `seq_len=64` unmasked backward.
  - Last probe showed: `dk` non-finite at flat index `160`, value `inf`.
- Fixture 3 still fails in the short reference segments (`seq_len=38`, likely also `26`).
  - Last probe showed `dq`, `dk`, and `dv` becoming `NaN` from index `0` on the short unmasked reference launch.

## What was tried after the original fix

- Zeroing the backward `P` tile: not the root cause.
- Half-precision atomic accumulation for `dK` / `dV`: improved the original masked-vs-unmasked drift, but was not enough for the broader reference cases.
- Deterministic serialized q-block launches: this is what made Fixture 1 pass.
- `kv_load` out-of-range K/V zero-fill and `k_global >= seq_len` masking in `ds_compute`: landed, but did not resolve the short unmasked reference failures by themselves.
- `finalize` tail guards and deterministic read-add-write updates for serialized `dK` / `dV`: landed, but did not resolve the remaining reference failures.
- A temporary CPU reference path built from saved forward activations was attempted in the test file and did not match the packed path; that attempt should be treated as experimental and re-evaluated before relying on it.

## Likely remaining root causes

Two remaining problems appear to be independent:

1. Short unmasked reference launches (`seq_len=38` / `26`) still have an invalid-row issue somewhere outside the already-patched slices.
   - Strong suspects: remaining q-row-dependent hooks/phases in fused backward, especially CSHA hook code paths that still assume a full `block_q` tile.

2. The `seq_len=64` unmasked reference still overflows / diverges in `dK` on the second q-block.
   - This is likely still tied to the reduced-output accumulation path or another q-block interaction in the short unmasked kernel.

## Best next steps

1. Revert or isolate the experimental CPU-reference helper in `pca_tier_a_backward_correctness.rs` before further debugging, unless you intend to finish that reference path properly.
2. Audit the fused backward hook phases for last-partial-q-block guards.
   - Focus on `crates/nsl-codegen/src/flash_attention_v2/phases/backward/csha_hooks_backward.rs`.
3. Reproduce the `seq_len=64` unmasked reference alone and instrument first non-finite production inside the kernel-local path.
   - The first known bad output is `dk[160] = inf`.
4. Keep the serialized q-block runtime path while debugging.
   - It is what made the original masked-no-op fixture deterministic and green.

## Files touched in this session

- `crates/nsl-codegen/tests/pca_tier_a_backward_correctness.rs`
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs`
- `crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs`
- `crates/nsl-codegen/src/flash_attention_v2/phases/backward/finalize.rs`
- `crates/nsl-codegen/src/flash_attention_v2/phases/backward/kv_load.rs`
- `crates/nsl-codegen/src/flash_attention_v2/phases/backward/ds_compute.rs`
- `crates/nsl-codegen/src/flash_attention_v2/phases/backward/dv_accum.rs`
- `crates/nsl-runtime/src/flash_attention.rs`
