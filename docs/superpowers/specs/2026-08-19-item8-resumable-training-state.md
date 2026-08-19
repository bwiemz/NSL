# Item 8 — True resumable training state

**Roadmap criterion:** *Loader cursor/order + RNG + optimizer/model/step restore
automatically.*

## What exists before this change

`train(checkpoint_save=…, checkpoint_every=N)` writes θ (`.nslm`) plus an
`.optim` sidecar (v1) holding AdamW `m`/`v` and the micro-batch step counter,
committed atomically (tmp + rename, with a `model_sig` pairing check).
`checkpoint_load=…` restores all three.

Three things it does **not** restore, each of which silently changes what the
resumed run computes:

1. **Data order.** The epoch loop restarts at epoch 0 and the DataLoader at
   batch 0. A run interrupted 80% through epoch 3 re-reads epoch 0 from the
   top — the model sees the same tokens again while the step counter (and
   therefore the LR schedule and bias correction) says otherwise.
2. **The shuffle itself is not even reproducible.** The single-rank
   (non-sharded) path shuffles with entropy — `rand::rng()` — so "resume to
   the same order" is not expressible. Only the multi-rank sharded path is
   seeded.
3. **RNG state.** Dropout draws from the thread-local sampling RNG (CPU) and
   from a process-global `DROPOUT_SEED` counter (GPU, hard-coded to start at
   42 and **not** seeded by `--seed` at all). Neither is captured, so masks
   after a resume are a different stream.

## Deliverable

A resumed run is indistinguishable from one that never stopped — the same
property the Milestone-B gate pins for θ/m/v/step, now extended to the data
stream and the RNG.

### 1. Reproducible data order (`dataloader.rs`)

- The non-sharded shuffle uses `StdRng::seed_from_u64(seed ^ epoch)`, the same
  keying the sharded path already uses. `seed` defaults to the global `--seed`
  store when one was set, else the existing constant.
- Position is the loader's own **delivery slot** (`expected_batch_id`), not a
  count of yielded batches: the ragged-packed-tail sentinel makes those two
  differ, and only the slot indexes the permutation.
- `nsl_dataloader_resume_to(dl, epoch, slot)` arms a *pending resume* consumed
  by the next `reset()`: that reset sets `epoch = resume_epoch` instead of
  incrementing, and starts the cursor at `slot` rather than 0. Workers claim
  from the cursor, so the first `slot` entries of the permutation are never
  produced — an O(1) skip, not a replay.
- `nsl_dataloader_identity(dl)` — FNV-1a over the loader geometry
  (batch/seq/flags/world/seed/total_batches) and a corpus fingerprint (length,
  dtype, first and last MiB). Recorded at save, compared at load.

### 2. Capturable RNG (`rng_state.rs`, `sampling.rs`)

- The sampling thread-local becomes `rand_chacha::ChaCha12Rng` — the concrete
  type `StdRng` already is in rand 0.9, so **the stream is unchanged
  bit-for-bit** — chosen because it exposes `get_word_pos`/`set_word_pos`,
  making snapshot/restore O(1) instead of an O(draws) replay.
- The GPU dropout counter moves out of `gpu_dropout_f32`'s function-local
  `static` into `rng_state`, so it is (a) snapshot-able from CPU-only builds
  and (b) seeded by `nsl_rng_seed` — `--seed` now actually reaches GPU dropout.
- **SR-BF16's `step` needs nothing added, but its `seed` does.** The dither is
  `mix64(seed ^ opt_step·SALT, param_base + i)`. `opt_step` is derived in
  codegen from `step_count_var`, which the checkpoint already restores (the
  process-global `SRBF16_STEPS` is telemetry only — the teardown line and
  `nsl_sr_bf16_step_count` — not the stream key). But `seed` is the `--seed`
  **scalar** from `deterministic_ops::RNG_SEED`, read live by `sr_bf16.rs` and
  by the composed ZeRO-3 slice update — a second, independent RNG input that
  the sampling stream's ChaCha key says nothing about. It lives on the command
  line rather than in the recipe, so "re-run the recipe unchanged" makes
  dropping it a routine slip. The sidecar records it (`global_seed`, plus
  `global_seed_set` because `--seed 42` is indistinguishable from the default
  by value) and a mismatch refuses.

  *This was caught in review.* The first version of this section claimed
  SR-BF16 needed nothing at all — true of the `step` half, which is what had
  actually been checked.

### 3. Sidecar v2 (`checkpoint.rs`)

`OPTIM_VERSION` 1 → 2. The header gains a `resume` object:

```json
"resume":{"train_epoch":E,"loader_epoch":LE,"loader_slot":S,"loader_id":H,
          "rng_seed":"<64 hex>","rng_pos_hi":A,"rng_pos_lo":B,"gpu_dropout_ctr":C}
```

- A **v1 sidecar still loads** (θ/m/v/step are honestly restored) but emits a
  loud warning naming exactly what is not restored. Refusing would strand
  every checkpoint written before this change.
- Identity mismatches **refuse**: a different corpus or loader geometry, or a
  checkpoint saved with a loader resumed without one (or vice versa), aborts
  rather than resuming into a data stream that is not the one that was saved.

### 4. Codegen wiring (`stmt.rs`, `builtins.rs`)

- `nsl_train_checkpoint_save` takes `dl_handle` and the training epoch
  counter; `nsl_train_checkpoint_load` takes `dl_handle` and additionally
  publishes the restored training epoch through `nsl_train_resume_epoch()`.
- The epoch loop seeds `epoch_counter_var` from that value instead of 0, so a
  resume continues the remaining epochs rather than re-running all of them.

## Gates

| Gate | What it pins |
|---|---|
| `resume_replays_the_uninterrupted_data_order` (CLI) | Interrupted-vs-control **bit-exactness with a DataLoader in the loop** — the criterion, and the one the v1 sidecar cannot pass |
| `dataloader_resume_yields_the_exact_tail` (unit) | Slot skip delivers the same batch ids the uninterrupted loader would have |
| `shuffle_is_reproducible_across_loaders` (unit) | Same seed+epoch ⇒ same permutation; different epoch ⇒ different |
| `rng_snapshot_restores_the_stream` (unit) | Draw *k*, snapshot, draw *n*, restore, redraw *n* ⇒ identical |
| `resume_refuses_a_foreign_corpus` (CLI) | Identity mismatch aborts instead of silently reordering |
| `v1_sidecar_warns_about_data_order` (CLI) | Old checkpoints load, loudly |
| `resume_is_bit_exact_under_grad_accumulation` (CLI) | The save cadence (and so the recorded slot) is `every x accum` |
| `resume_tail_is_exact_with_a_padded_final_batch` (unit) | `drop_last=false` keeps a real final slot |
| `resume_tail_is_exact_under_packing_with_a_ragged_tail` (unit) | Slots ≠ delivered batches; only the slot indexes the permutation |
| `sidecar_v2_header_carries_the_whole_resume_block` (CLI) | The wire format itself — the bit-exactness gates would pass if a field were written and read under the same *wrong* key |
| `resume_of_a_finished_run_refuses_instead_of_training_nothing` (CLI) | **Review finding.** See below |
| `resume_refuses_a_different_global_seed` (CLI) | The `--seed` scalar is a second RNG input, and it lives on the command line rather than in the recipe |
| `pipelined_train_path_refuses_checkpoint_config` (CLI) | The Milestone B `@pipeline` refusal, which had no test |

## What review caught

**An exhausted epoch was unrepresentable, so the budget refusal could not fire
for the case the feature is about.** With a DataLoader, codegen records the
*in-progress* epoch, so a run's own final checkpoint always has `train_epoch <=
epochs - 1` and `train_epoch >= epochs` was structurally unreachable. A
checkpoint landing on an epoch's last delivery slot then passed every guard,
drew an empty epoch, and exited 0 having run no optimizer step — exactly the
silent no-op the refusal exists to prevent, in the exact shape (`epochs = 1`
plus a DataLoader) that `models/coder1b/pretrain_1b2048.nsl` uses. Two
independent refuters failed to refute it; one traced the reaching path into
`endurance_1b.py`, where any `--steps ≡ 48 (mod 50)` puts the final save on the
epoch's last slot and phase 3 would train nothing and exit 0.

The fix normalizes at *save* time: `(epoch E, slot total_batches)` and
`(epoch E+1, slot 0)` are the same position, but only the second lets both the
loader's bookkeeping and the budget refusal see that E is finished.

Six other findings were raised and refuted. The most instructive: the claim
that a `seed + 42` GPU dropout base makes adjacent seeds produce *shifted
copies* of one mask tape. The refuter reimplemented the PTX mixer and measured
same-position mask agreement at 0.8204 for offsets 1..4096 against an
independent baseline of 0.8200 — adjacent counters decorrelate, which is the
defining property of a counter-based RNG. The seed is still mixed rather than
added, but only to stop `--seed 0` landing on the unseeded default.
