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
- **SR-BF16 needs nothing added.** Its dither is the pure function
  `mix64(seed ^ opt_step·SALT, param_base + i)`, and `opt_step` is derived in
  codegen from `step_count_var` — which the checkpoint already restores. The
  process-global `SRBF16_STEPS` is telemetry only (the teardown line and
  `nsl_sr_bf16_step_count`), not the stream key. Checked rather than assumed:
  a stochastic-rounding stream that silently repeated from step 0 on every
  resume would correlate rounding error with the earlier part of training.

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
