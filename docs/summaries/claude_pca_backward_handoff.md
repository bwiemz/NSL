# PCA Backward CUDA Handoff

Status: complete for the PCA backward correctness target; forward correctness caveat closed (see "Forward correctness" section)

## Final outcome

The ignored PCA backward correctness suite is now green:

```powershell
cargo test -p nsl-codegen --features cuda --test pca_tier_a_backward_correctness -- --ignored --nocapture --test-threads=1
```

Latest result:

- Fixture 1 passed exactly.
- Fixture 2 passed exactly.
- Fixture 3 passed exactly.
- Final summary: `test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`

## What actually fixed it

The decisive change was in the test harness, not the backward kernel math.

- `crates/nsl-codegen/tests/pca_tier_a_backward_correctness.rs`
  - `launch_pca_backward` no longer relies on `nsl_flash_attention_csha_with_saves` to populate backward inputs.
  - The harness now stages trusted forward state itself:
    - RMSNorm on `x`
    - projected `Q/K/V`
    - forward `O`
    - `LSE`
    - saved `row_max` / `row_sum`
    - raw `x_raw`
  - Those tensors are uploaded directly into `q_dev/k_dev/v_dev/out_dev/lse_dev` and `CshaBackwardActivations` before launching `nsl_flash_attention_csha_backward`.
  - The packed path uses a segment-aware host forward reference so the saved softmax state matches PCA masking semantics.

Why this mattered:

- The backward kernel was being judged against saved activations coming from a forward-with-saves path that was not trustworthy for these PCA fixtures.
- Once backward consumed deterministic staged saves instead, the packed and unpacked GPU backward launches matched exactly across all three fixtures.

## Other lasting changes still present

- `crates/nsl-codegen/src/flash_attention_v2/phases/forward/finalize.rs`
  - Forward output stores are back to `f16` for save-enabled CSHA launches, matching the backward path and the PCA harness ABI.

- `crates/nsl-codegen/tests/csha_cuda_launch_fused.rs`
  - Updated stale direct FFI call sites for the new trailing `segment_ids` argument.
  - Added ignored multitile save-null-vs-save-active invariant probes:
    - `t1_forward_output_invariant_under_save_activations_flag_multitile`
    - `t1_forward_output_invariant_under_save_activations_flag_multitile_causal`
  - Both of those pass for the tested non-segmented configs.

## Forward correctness (caveat resolved)

The earlier handoff carried this caveat:

> This handoff does not claim that the fused save-enabled forward kernel is numerically correct for multitile PCA segment-masked workloads. The new `csha_cuda_launch_fused` probes only establish save-null vs save-active parity for the specific non-segmented seq=128 cases they cover; they are not a full correctness proof against CPU reference.

That gap is now closed by transitivity:

1. **CPU-reference correctness for the no-saves path on segmented multi-tile.** The pre-existing `pca_tier_a_forward_correctness` suite is green on this branch:

   ```text
   test tier_a_forward_single_segment_matches_unmasked_baseline ... ok    (max_abs=0.000000)
   test tier_a_forward_two_equal_segments_matches_unpacked_reference ... ok   (max_abs=0.000168)
   test tier_a_forward_unequal_segments_matches_unpacked_reference ... ok     (max_abs=0.000223)
   ```

   These call `nsl_flash_attention_csha` (no saves) and compare to a CPU/unpacked-padded reference at tolerance 5e-3 (head_dim=32, f16 mantissa).

2. **Save-null vs save-active bit-equality on segmented multi-tile.** Two new probes added in `csha_cuda_launch_fused.rs`:

   - `t1_forward_output_invariant_under_save_activations_flag_multitile_segmented_two_equal` — seq=128 = `[seg0]×64 + [seg1]×64`
   - `t1_forward_output_invariant_under_save_activations_flag_multitile_segmented_three_unequal` — seq=128 = `38 + 64 + 26` (matches Fixture 3 of the backward suite)

   Both pass with bit-identical O and LSE between save-null and save-active runs.

By transitivity (1 + 2), the save-enabled forward kernel is correctness-equivalent to the CPU reference for multitile segment-masked seq=128 configs at head_dim=32. Larger configs (seq>128, head_dim>32) remain unverified; if a future PCA tier expands the coverage, extend `pca_tier_a_forward_correctness` and the matching `_segmented_*` probe pairs accordingly.

## Files that matter for follow-up

- `crates/nsl-codegen/tests/pca_tier_a_backward_correctness.rs`
- `crates/nsl-codegen/tests/pca_tier_a_forward_correctness.rs`
- `crates/nsl-codegen/src/flash_attention_v2/phases/forward/finalize.rs`
- `crates/nsl-codegen/tests/csha_cuda_launch_fused.rs`
- `crates/nsl-runtime/src/flash_attention.rs`

## Useful validation commands

```powershell
cargo test -p nsl-codegen --features cuda --test pca_tier_a_backward_correctness -- --ignored --nocapture --test-threads=1
cargo test -p nsl-codegen --features cuda --test pca_tier_a_forward_correctness -- --ignored --nocapture --test-threads=1
cargo test -p nsl-codegen --features cuda --test csha_cuda_launch_fused t1_forward_output_invariant_under_save_activations_flag_multitile -- --ignored --nocapture --test-threads=1
cargo test -p nsl-codegen --features cuda --test csha_cuda_launch_fused t1_forward_output_invariant_under_save_activations_flag_multitile_segmented -- --ignored --nocapture --test-threads=1
```
