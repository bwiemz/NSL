# CSLA Stage-2 (D1a + D1b): the window-buffered, layer-major schedule

**`--layerwise-accum` — buffering an entire FASE-Deferred accumulation window,
replaying its backward layer-by-layer, and updating each layer's parameters
the moment its accumulation completes — bit-exact with the interleaved
baseline, with the accumulator surface shrunk from `4·P` to
`max(one layer) + epilogue globals`.**

Companion to `CSLA-compiler-scheduled-layerwise-accumulation.md` (the design
paper); this note documents what Stage-2 actually built: §§1-5 the D1a
window-buffering slice, §6 the D1b layer-major slice on top of it.

---

## 1. The structural fact the paper's §7 glossed over

The paper's schedule ("for micro_batch in window(N): forward… then for layer
in reverse…") reads as if micro-batches were structural units. They are not:
`compile_train_block_inner` emits ONE Cranelift function whose batch loop is a
runtime loop (`nsl_dataloader_next_batch` + NULL-check), and the accumulation
window is a runtime modulo on a global step counter
(`(step_count+1) % N == 0`, stmt.rs). Micro-batches do not exist in the IR,
windows straddle epoch boundaries, the trailing partial window simply never
fires the optimizer, and the DataLoader destructively pops batches (no
re-fetch; `nsl_dataloader_next_batch` advances a monotonic counter).

So Stage-2 is a *loop-emission restructure*: the tape stays compiled once
("one tape, N executions" — unrolling N forwards into one tape would break
CCR's first-use anchor segmentation), and per-micro-batch state moves through
runtime-indexed buffers.

## 2. What D1a emits

Per batch iteration (unchanged runtime loop):

1. **Forward** exactly as today (FBIP-suppress bracketed).
2. **CCR early-free** of block interiors exactly as today.
3. **Save phase** (replaces the inline adjoint lowering): push every
   *adjoint-read primal value* into this micro-batch's slot list
   (`nsl_list_push`), push the slot list into `saves_outer`, push the batch
   dict into `dicts`, and DEFER their frees. Loss consumers, scheduler,
   `step_count`, `on_step`, `on_epoch` all run per iteration as today — loss
   values are forward-only, so the printed stream is identical by
   construction.
4. **Window backward** (new region, gated on the same `should_step` value,
   emitted just before the optimizer gate): a runtime `b`-loop over the N
   buffered micro-batches. Each pass seeds a fresh `VarMap` — the step
   iteration's forward map (params/constants are loop-invariant: FASE
   Deferred only updates θ at window boundaries, *after* this loop) with the
   buffered imports overridden from `slots[b]` — re-installs micro-batch b's
   packing metadata (`segment_ids`/`doc_starts` are thread-local per-batch
   state read at @flash_attention launch time), and lowers the SAME adjoint
   tape through `compile_wengert_ops` with the standard FASE hook. After each
   pass: adjoint-owned bulk free, then the per-b frees of owned buffered
   slots and that micro-batch's dict values.
5. **Optimizer region**: byte-for-byte the existing emission (monolithic
   Deferred loop → `fase_emit_final_step`, fused #388 kernel and all) —
   it consumes a bit-identical `m_partial`.

Teardown sweeps the trailing partial window (buffered saves + dicts of a
window that never fired — the baseline discards the same tail's `m_partial`,
so parity holds; the sweep is purely against leaks).

## 3. Why it is bit-exact

- **Same values in**: a buffered import IS the value the baseline's backward
  read (same tensor pointer, same content — nothing mutates activations
  between save and replay; θ updates happen only after the window loop).
- **Same kernels, same order per parameter**: the FASE hook fires per
  parameter inside each replay in micro-batch-ascending order — exactly the
  baseline's per-parameter accumulation sequence `m_partial[p] += g_b/N`,
  b = 0..N-1. f32 non-associativity never sees a different order.
- **Same recompute**: CCR's clones are part of the (shared) adjoint tape;
  each replay recomputes from that micro-batch's boundary residuals — CCR's
  proven bit-exactness carries over per micro-batch.
- **Same window semantics**: the global modulo counter is untouched, so
  epoch-straddling windows and the never-stepping partial tail behave
  identically.

The import set is `layerwise::adjoint_primal_imports`: adjoint-read ∩
primal-produced, minus `Param`/`Constant` leaves (loop-invariant), computed on
the FINAL adjoint (post dead-grad elimination, post CCR splice + last-use
frees) so recompute-victim reads — remapped to clones — are correctly absent.
Slots store i64 pointers/integers directly and f64 scalars via same-width
bitcast; buffering is deliberately *un-analyzed* beyond that (no
batch-taint analysis): buffering a batch-invariant value is merely a few
redundant list slots, never a correctness risk.

## 4. Admission (all refusals are loud compile errors)

Requires `--source-ad` (hard error on tape fallback too), `--checkpoint-blocks`
with a live CCR plan (without recompute, "buffer what the adjoint reads"
degenerates to N full activation sets), and a FASE-Deferred plan (AdamW/Adam,
`grad_accumulation >= 2`). Refuses: `grad_clip` (two-phase clip needs the
global L2 over the *complete* window before any update), WGGO per-param mode
tables, `--optim-state-offload` (the P3 staging path this work obsoletes),
`--checkpoint-compress` (not bit-exact), and the pipelined train path.

The adversarial review added three refusals for **compile-time SSA
side-channels** — state that travels by Cranelift `Value` instead of the tape,
which the replay's freshly-loaded values can never see:

- **CSHA-claimed fused backward** (`csha_forward_saves`, keyed by layer name,
  consumed + freed once): inside the b-loop this would mean stale saves for
  b&lt;N-1 plus an N-fold double-free;
- **fused SDPA dispatch** (`flash_attn_aux` saved-logsumexp, keyed by the
  forward output's Value + a `u32::MAX`-sentinel owned entry): a replay miss
  silently substitutes the reference backward for the fused phase-2 kernel
  whenever the fused forward actually launched. NOTE the dispatch is emitted
  for EVERY SDPA op on a CUDA target (the variant table covers head_dims
  {32,64,128} with runtime selection), so in D1a this refusal means
  **attention models train under `--layerwise-accum` on CPU only**; the
  packed-GQA parity gate runs on CPU and the GPU arm is a refusal case;
- **`@fused_lm_ce` / distill** (`fused_ce_fwd_casts` / `fused_kl_ce_bwd_cache`
  Value-keyed cast caches): a replay miss re-emits fresh casts per window with
  no free.

D1b (or a follow-up) lifts these by tape-carrying the side-band values
(routing the LSE and cast tensors through real tape VarIds so the import
machinery buffers them per micro-batch like everything else).

## 5. Validation

`crates/nsl-cli/tests/csla_layerwise_gate.rs`, cloning the CCR parity-gate
discipline (baseline-twice determinism probe, anti-vacuity, bit-exact loss
stream + `.nslm` bytes). Anti-vacuity is the `NSL_CSLA_COUNTER=1` atexit
report asserted to the EXACT expected window count on the layerwise arm and 0
on the baseline. Green on 2026-07-17 (RTX 5070 Ti, sm_120, `--target` default,
`--deterministic` for the GPU byte-compares):

| gate | schedule facts exercised | result |
|---|---|---|
| `csla_parity_ffn_cpu` | 13 micro-batches, N=2 → 6 windows + partial tail | bit-exact |
| `csla_parity_ffn_cpu_loss_read_by_adjoint` | `tanh(loss)` → the adjoint reads the loss tensor → loss_slot machinery (b&lt;N-1 skip + conditional per-iteration free) | bit-exact |
| `csla_parity_ffn_cpu_epoch_straddle` | epochs=2 → 13 windows, one spanning the epoch boundary | bit-exact |
| `csla_parity_ffn_gpu` | GPU twin (attention-free — SDPA on CUDA is refused in D1a) | bit-exact |
| `csla_parity_packed_gqa_cpu` | packed GQA on CPU: buffered packed dict state + per-b packing-registry re-install + packed EmbeddingBackward from buffered input_ids | bit-exact |
| `csla_refusals` | grad_clip / missing `--checkpoint-blocks` / SDPA-on-CUDA (cuda feature) / missing `--source-ad` | all refuse loudly |
| `csla_mpartial_surface_shrinks_gpu` | D1b memory win: MPartial surface peak under the flag vs baseline | ratio ≤ 0.7 (projected 0.556) |

All parity gates additionally assert the exact
`[csla] layer-major schedule` line (D1b): the FFN fixture must produce
`3 ranges, 6 layer-grouped params, 2 epilogue params` and the packed-GQA
fixture `3 ranges, 16 layer-grouped params, 2 epilogue params`.

### 5b. The LSE tape-carry and the measured 1B boundary

The follow-up slice lifted the fused-SDPA refusal: the save phase buffers
every `flash_attn_aux` entry as an extra window slot, and the replay
re-binds the aux map per micro-batch before each consuming range's lowering
(`flash_attn_aux[seed[fwd_out]] = lse[b]`), so the emitted backward consumes
the SAME per-batch logsumexp the baseline did — real tensor when the fused
launch fired, the runtime-0 decline sentinel otherwise. Gates:
`csla_parity_packed_gqa_gpu` (dispatch emitted, GQA strides decline at
runtime — parity of the decline plumbing) and
`csla_parity_packed_mha_gpu_fused` (contiguous MHA K/V — the fused forward
FIRES, `sdpa_fused_launch_count(0)` asserted > 0 and equal on both arms, and
the carried LSE feeds the fused phase-2 backward bit-exactly).

With attention unblocked, the paper §7.5's "confirm no-offload 1B fits
16 GiB" was RUN (`models/coder1b/pretrain_layerwise_fit.nsl`, RTX 5070 Ti).
**Measured answer: it does not fit — and m_partial is no longer the reason.**
The schedule engaged at scale (`17 ranges, 144 layer-grouped params, 2
epilogue params`), a full window backward + fused optimizer step completed
(loss 11.7769 ≈ ln(vocab)), and the allocator report at the OOM shows
`m_partial at-global-peak = 0 B` — the 3.86 GiB full-model window this
schedule replaced is gone, with only the 384 MiB tied-embedding epilogue
accumulator live. The binding wall is 12 GiB of RESIDENT f32 weights + m/v
plus the tied-embedding gradient chain's ~1.2 GiB transient spike: batch 2 /
seq 512 OOMs in the first forward; batch 1 at seq 512 and even seq 256 /
accum 2 (with and without `NSL_ASYNC_ALLOC=1`) OOM 384 MiB short in the
second window. The fix stack is orthogonal to CSLA: fp16 moments (P0.3,
frees ~4.1 GiB) and/or D2 weight streaming. A pre-existing runtime bug
surfaced on the way: the GPU-OOM CPU-fallback path
(`cpu_fallback_binary` → `nsl_tensor_to_device`) aborts with "panic in a
function that cannot unwind" instead of recovering.

Known limitations (review LOW findings, accepted for D1a): `NSL_PHASE_TIMING`
attributes the backward phase to nothing under the flag (the bwd region it
brackets is empty; the window backward is untimed); an early `return` out of
a step body leaks the current window's buffers (exit path only); the flag on
a program with no train block is a silent no-op, consistent with
`--checkpoint-blocks`.

Full regression battery unchanged-green with the flag off: 855 CPU tests, CCR
parity ×4 (incl. production tolerance), 48-step long-run drift, fused-step /
stream-migration / deferred-free / mem-accounting GPU gates.

## 6. D1b: the layer-major schedule (the paper §3 realized)

D1b generalizes the single backward range to the k ranges of
`layerwise::partition_ranges`: positional boundaries at each layer's
`adjoint_first` in backward-visit order; range 0 is the prologue (loss /
LM-head / final-norm adjoints), the last range swallows the
embedding-backward epilogue ops. Emission per window:

```
alloc epilogue-global accumulators (window-lived)
for range R in backward order:
    alloc layer(R)'s accumulators            # the shrunk m_partial surface
    for b in 0..N:                           # same replay loop as D1a
        seed = forward values + R's primal slots + R's carried adjoints
        lower slice R with the FASE hook     # accumulates layer(R)'s grads
        export R's boundary adjoints to the per-b carry list
        free slots/carries whose last reader is R
    per-layer update: fase_emit_final_step for layer(R)'s local params
    free layer(R)'s accumulators
epilogue update: globals, tied/cross-layer, dead layers, unseen params
```

Key mechanics:

- **Carry buffers**: boundary adjoints (`d(residual-after-L)`) are
  adjoint-produced — NOT in CCR's `escaping` set (primal values only). A
  per-micro-batch slot list carries them between ranges; exports are
  excluded from the producing range's bulk free, and their frees are the
  consuming ranges' existing FreeTensor markers (belt: an explicit free
  after the last consuming range for marker-less Tensor carries).
  Ghost-skipped exports are recorded at compile time so later ranges skip
  seeding them — consumers ghost-skip identically to the baseline.
- **Update groups**: a layer's param updates after its range iff its
  gradient op sits positionally inside that range (attribution slop demotes
  to the epilogue group — always correct, merely later). Dead params (no
  adjoint) update with their layer on zero accumulators — weight/moment
  decay still fire, exactly like the baseline's unconditional loop. Every
  `param_paths` slot lands in exactly one group; per-param updates are
  order-independent (no cross-param state in `fase_emit_final_step`, one
  shared `bc1_inv/bc2_inv` pair computed from the same `step_count`
  expression as the — now bypassed — optimizer site), so bit-exactness
  survives the reordering.
- **Update guard**: the CrossLayer classification runs on the FINAL adjoint
  with the last-use analysis *extended through NslList membership*
  (`ccr::extend_last_use_through_lists`, hoisted and now shared — Stage-1's
  scan lacked it, which per-layer in-place θ updates made load-bearing), and
  a one-time runtime pointer-alias abort
  (`nsl_csla_assert_params_unaliased`) covers pointer-tied weights the
  compile-time analysis cannot see.
- **Accumulator lifecycle**: setup pushes NULL accum slots (no full-model
  window, ever); each window allocates fresh `zeros_like` per group under
  the MPartial surface and frees after the group's update — identical
  initial bytes to the baseline's zeroed persistent buffers. The teardown
  loop skips per-slot frees (slots are NULL between windows).
- **Anti-vacuity**: the compile-time
  `[csla] layer-major schedule: k ranges, X layer-grouped params, Y epilogue
  params` line is asserted EXACTLY in the gates (a degenerate all-epilogue
  schedule would replicate D1a and pass the parity gates vacuously), and the
  GPU surface gate asserts `gpu_surface_peak_bytes(MPartial)` under the flag
  is ≤ 0.7× the baseline's (the FFN fixture projects 0.556×).

The D1a b-loop, buffer lists, seed-override mechanism, packing re-install
(now per range — any range may hold attention backward ops), partial-tail
semantics, refusals, and the parity harness all carry over unchanged.
