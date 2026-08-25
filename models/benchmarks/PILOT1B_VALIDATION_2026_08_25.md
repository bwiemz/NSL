# The corpus-v2 1B pilot — 98.3M tokens (new-roadmap item 6)

**Program** `models/coder1b/pretrain_pilot.nsl` — the production recipe
(`pretrain_prod.nsl`) scaled to a 98,320,384-token prefix of the v2 mixture:
batch 2 × seq 2048, grad_accumulation 4, AdamW lr 1e-4 → cosine to 3e-5 over
24,000 micro-steps (warmup 2,400), grad_clip 1.0, checkpoint_every 500
optimizer steps, resident posture, `--source-ad --checkpoint-blocks
--fuse-rmsnorm-backward`, fused-lce active (`route=gemm` in build stderr).
RTX PRO 4500 Blackwell 32 GB. 2026-08-25, 05:23:26 → 14:52 wall.
(The stdout banner says "6000 opt steps" — the binary predates the header
fix that states the honest 24,004-micro/6,001-update derivation; cosmetic.)

The roadmap's charge: validate the tokenizer/data path, a working scheduler,
the LR range, checkpointing, throughput, and held-out curves BEFORE
committing GPU-months to the 22.6B corpus.

## Verdict against the five targets

| target | verdict |
|---|---|
| u16 corpus-v2 data path, end to end | **VALIDATED** — 24,004 micro-steps, zero CPU fallbacks, zero data errors |
| scheduler over a real warmup+cosine | **VALIDATED** — warmup descent visible in-loss; route behavior separately proven by the item-3 gates |
| checkpointing | **save VALIDATED** (12 rolling cadence saves, 12.9 GB pair each); resume probed separately below |
| throughput at production geometry | **VALIDATED** — 2,878 tok/s gross ≈ the item-1 bench's 2,914 steady-state |
| held-out curves / LR range | **THE ACTIONABLE RESULT** — the curve is flat after warmup, and the trained θ was subsequently lost to an operator error (below), so no VAL numbers exist for it. The flatness alone decides the next step. |

## Training: what 10 hours measured

- **Completed fully.** All 24,004 micro-steps = 6,001 optimizer updates
  (the 6,001st is the past-total_steps min_lr update the header documents).
  The crash below happened AFTER the train block and its epilogue.
- **Throughput** 98,320,384 tokens / 34,158 s = **2,878 tok/s gross**,
  including compile and twelve 12.9 GB checkpoint writes — within ~1.2% of
  the item-1 bench's 2,914 tok/s steady-state (RESIDENT_1B_THROUGHPUT_2026_08_25.md).
  The production data path costs ≈ nothing over the synthetic stream.
- **Memory** (allocator, exact-byte): peak 20,906,110,976 B = 19.47 GiB —
  byte-class identical to the bench's resident B2/A4 figure. Surfaces at
  peak: weights 4.02, optim_m/v/m_partial 4.00 each, activations 3.45 GiB.
  48,008 `gpu_mem_step` events banked in `pilot-out/events.jsonl`.
- **The loss curve** (300 prints, one per 80 micro-steps; per-print spread
  is huge — 5.4 to 11.1 — because the mixture's code/prose/NSL bins have
  very different CE, so windows are the unit):

  | window (micro-steps) | mean |
  |---|---|
  | 80–1,600 (warmup) | 9.130 |
  | 1,680–3,200 | 8.563 |
  | 3,280–4,800 | 8.838 |
  | 4,880–6,400 | 8.758 |
  | 6,480–8,000 | 8.655 |
  | 8,080–9,600 | 8.648 |
  | 9,680–11,200 | 9.047 |
  | 11,280–12,800 | 8.673 |
  | 12,880–14,400 | 8.823 |
  | quarters | 8.843 / 8.760 / 8.755 / 8.809 |
  | post-warmup halves | **8.759 / 8.790** |

  First print 11.081 (the ln(49152)=10.803 neighborhood); fast descent
  through warmup to ~8.6; then **no detectable slope for 21,600
  micro-steps** (the second post-warmup half is nominally WORSE; ±0.03 on
  these window sizes is noise around zero). This is the item-10 signature —
  but produced on machinery that is now PROVEN working (scheduler gated per
  route by item 3, causal mask fixed, grads checksummed by earlier
  campaigns, this run's own warmup responds to lr), so it is no longer a
  tooling suspicion. It is a hyperparameter result:
  **lr=1e-4 (width-scaled, never measured at 1B) does not descend past the
  token-statistics floor at 0.09 tokens/param.**

## What the pilot answers for the roadmap

**Do not start the 1–2B intermediate run on this recipe.** The pilot's
whole purpose was to answer "would GPU-months on this configuration be
wasted?" and the answer for lr=1e-4 as-shipped is: the post-warmup slope is
zero, so yes. The next 10-hour units of GPU time buy the most as a
**controlled LR pair (or 2×2) at pilot budget** — same program, same data,
lr varied — each arm of which produces its own held-out numbers. The config
prose (rewritten this campaign) already frames exactly this follow-up.

## Finding 1 — runtime: plain-loop inference dies on the display watchdog at 1B

After the train block, the first held-out forward aborted:

    cuMemcpyHtoD_v2(16777216 bytes) failed: CUDA_ERROR_LAUNCH_TIMEOUT
    [context: nsl_mul_scalar_f32]
    ... nsl_tensor_to_device <- nsl_tensor_add <- __nsl_model_GroupedQueryAttention_forward

A plain `for batch in loader:` loop never stages the batch to the model's
device. Each op reconciles ad hoc — host operands flowing into GPU ops,
per-op transfers, f64 fallback kernels (GPU→CPU migration converts to f64
by ABI) — and at 1B dims one such kernel exceeded the ~2 s display watchdog
on this desktop-attached card; the sticky context error then surfaced on
the next H2D copy. **The 500M recipe's val loop is the same shape** and its
recorded VAL numbers were produced by this same mixed path, evidently under
the watchdog at those dims. The 89-day production run would have lost its
val pass to this after the full epoch.

Follow-up (runtime): plain-loop inference against a GPU model must either
stage loader batches to the model's device or refuse the mixed graph — the
same doctrine item 2 (#531) applies to train-block Input leaves, extended
to the inference path. Recipe-side idiom until then: `batch.input_ids.to(cuda)`
before the forward (committed in pilot_finish.nsl's val loops; the probe
scripts named below are local artifacts — `models/*/*_probe.nsl` is
gitignored by policy).

## Finding 2 — operator error: the trained θ was lost to a save-armed "resume"

The crash above killed the run before its final `model_save`, leaving the
opt-6000 rolling checkpoint as the ONLY copy of the trained state. A
salvage script then made a fatal API misreading: **`checkpoint_save=` is
write-only — it never resumes.** Resume is armed by the separate
`checkpoint_load=` kwarg (stmt.rs emits `nsl_train_checkpoint_load` and
seeds the restored step counter from it). The salvage script passed only
`checkpoint_save`, trained from scratch (first loss print 11.081 — the
from-scratch signature), and its first checkpoint_every cadence save rolled
over the pilot's checkpoint at micro 2,000. The trained θ is unrecoverable;
with it went the trained-state VAL numbers and the in-place resume test.

Prevention (banked to memory as the rule this incident earned):
- copy the `.nslm` + `.optim` pair aside on real disk before ANY
  experimental run whose `checkpoint_save` can reach it;
- after launching a resume, verify it ENGAGED before letting it run — the
  first on_step print must CONTINUE the counter, and a first loss at ~10.8
  means from-scratch. By the first cadence write it is too late.

An earlier salvage attempt also aborted on GPU OOM with this process's
allocator holding a coherent 17.6 GiB — the missing ~12 GiB belonged to a
concurrent session's guarded fusion campaign (`cuMemGetInfo` counts the
whole card). That attempt had been launched WITHOUT gpu-guard.sh; the
guard, used correctly, refuses exactly this. Both directions of the lesson
were then demonstrated within the hour: the guard later refused THIS
session's probe while the other campaign held the lock.

## Probes (post-incident)

- `pilot_resume_probe.nsl` — WRITE-FREE resume-mechanics probe at 1B
  resident: `checkpoint_load=` from a backup copy of the (post-overwrite,
  from-scratch step-2000) checkpoint, no checkpoint_save. PASS = counter
  continues at 2,080 and losses resume in the trained band. RESULT: see
  addendum below.
- `pilot_val_staged_probe.nsl` — the staged-input val loop on a fresh
  model: doubles as a null control (expected loss ≈ ln(49152) = 10.803)
  and as the watchdog-crash counter-evidence. RESULT: see addendum below.

## Artifacts

- `pilot-out/pilot.stdout` (300 loss prints, peak epilogue),
  `pilot-out/pilot.stderr` (clean until the val abort; zero CPU fallbacks),
  `pilot-out/events.jsonl` (48,008 exact-byte memory events) — the training
  run, intact.
- `models/coder1b/checkpoints_backup/` — the post-overwrite step-2000 pair
  (probe input; NOT the pilot's trained state).
- Slice provenance: `head -c` commands in the pilot header; parents are the
  manifest-pinned v2 bins (CORPUS_MANIFEST_v2.json).
