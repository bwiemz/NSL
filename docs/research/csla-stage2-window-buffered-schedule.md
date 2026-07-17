# CSLA Stage-2, slice 1 (D1a): the window-buffered schedule

**`--layerwise-accum` — buffering an entire FASE-Deferred accumulation window
and replaying its backward as one runtime loop, bit-exact with the
interleaved baseline.**

Companion to `CSLA-compiler-scheduled-layerwise-accumulation.md` (the design
paper); this note documents what Stage-2's first slice actually built, the
structural facts that shaped it, and the seam the layer-major slice (D1b)
plugs into.

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
| `csla_parity_ffn_cpu_epoch_straddle` | epochs=2 → 13 windows, one spanning the epoch boundary | bit-exact |
| `csla_parity_ffn_gpu` | GPU twin | bit-exact |
| `csla_parity_packed_gqa_gpu` | packed GQA + per-b packing-metadata re-install | bit-exact |
| `csla_refusals` | grad_clip / missing `--checkpoint-blocks` / missing `--source-ad` | all refuse loudly |

Full regression battery unchanged-green with the flag off: 855 CPU tests, CCR
parity ×4 (incl. production tolerance), 48-step long-run drift, fused-step /
stream-migration / deferred-free / mem-accounting GPU gates.

## 6. What D1a deliberately does NOT deliver, and the D1b seam

D1a's memory profile is *worse* than the baseline (N× the boundary-residual
surface, N batch dicts) — it is the enabling seam, default-off. The paper's
actual win (killing the full-model `m_partial`) is D1b:

- generalize the single backward range to the k ranges of
  `layerwise::analyze` (prologue / per-layer / epilogue partition of the
  adjoint by `adjoint_first` boundaries);
- per range, the same import/export machinery buffers the *boundary
  adjoints* between ranges (they are adjoint-produced — NOT in CCR's
  `escaping` set, which holds primal forward values only);
- fire `fase_emit_final_step` per layer for layer-local params (it is already
  strictly per-parameter; `bc1_inv`/`bc2_inv` are shared per window),
  guarded by the last-backward-use analysis *extended with the NslList
  lifetime fixpoint* (hoist ccr.rs's; layerwise.rs's Stage-1 scan lacks it)
  and a one-time runtime pointer-alias assert (pointer-tied weights are
  invisible to the compile-time analysis);
- shrink the accumulator from full-model to per-layer alloc/step/free.

The b-loop, buffer lists, seed-override mechanism, packing re-install,
partial-tail semantics, and the parity harness all carry over unchanged.
