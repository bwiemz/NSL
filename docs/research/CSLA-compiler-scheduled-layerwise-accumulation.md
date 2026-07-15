# Compiler-Scheduled Layerwise Accumulation (CSLA)

**Eliminating the full-model gradient window in static training graphs.**

Status: design + Stage-1 analysis (this document + `crates/nsl-codegen/src/layerwise.rs`).
Codegen (Stage 2) is scoped in §7. Milestone B of the post-#374 optimization roadmap.

---

## 1. Problem

After #374, NSL trains a 1.037B model on a 16 GiB card by offloading the AdamW
moments `m`, `v`, and the FASE window accumulator `m_partial` to host memory
(P0–P3). Once `m` and `v` are offloaded, **`m_partial` is the last
parameter-sized surface still touching the device** on the exact-fidelity
path. It is allocated once, per parameter, full-model-sized, before the epoch
loop:

```
// crates/nsl-codegen/src/stmt.rs:4472-4520
let accum_list = /* one nsl_tensor_zeros_like(param) per param, surface = m_partial */;
```

and it must stay live across the whole `grad_accumulation = N` micro-batch
window, because gradient accumulation sums each micro-batch's gradient into it
before the optimizer consumes it (`stmt_fase.rs:652` `fase_emit_accumulate`,
`stmt_fase.rs:60` `fase_emit_final_step`). Its device (or host) footprint is
**4·P bytes** (f32, exact-windowed — it can never be reduced-precision without
breaking the deferred-AdamW recipe):

| Model | Full f32 `m_partial` |
|---|---|
| 500M | ≈ 1.88 GiB |
| 1.037B | ≈ 3.86 GiB |
| 7.18B | ≈ 26.75 GiB |

The current mitigation (P3) stages `m_partial` host→device→host for **every
parameter on every micro-batch**. That is memory-cheap but throughput-hostile.
CSLA removes the full-model window entirely, so 1B + CCR can fit on 16 GiB
**without** optimizer-state or accumulator offload — and the 500M path stops
paying the per-parameter staging tax.

## 2. Why NSL can do this and a dynamic framework cannot

CSLA reorders the training program so aggressively that a dynamic (eager)
framework could not do it safely. NSL already has every prerequisite as a
static, inspectable object:

- a statically-extracted source-AD tape — the adjoint is a plain
  `Vec<WengertOp>` (`source_ad.rs:736`), freely partitioned and reordered;
- block identities shared with WGGO — `wggo_graph::layer_prefix` maps a
  `PrimalOp::Param("m.blocks.3.w_up")` to the bare layer key `blocks.3`
  (`wggo_graph.rs:94`);
- CCR's contiguous-block invariant — the extractor inlines each block's methods
  as a strictly sequential primal range, verified by CCR's monotonic-anchor
  check (`ccr.rs:470`);
- explicit last-use freeing already implemented over the adjoint list
  (`ccr.rs:317`);
- compile-time parameter ownership — each parameter is a **single memoized
  leaf VarId** (`source_ad.rs:2595`), so every read across the forward, a loop,
  or a micro-batch resolves to the same VarId and its gradient auto-accumulates
  into one `gen.adjoint_of(vid)` (`source_ad.rs:1421`);
- a compiler-generated optimizer schedule whose per-parameter update is already
  independent (`stmt_fase.rs:60`, one `theta/m/v/m_partial` tuple per param).

## 3. The schedule

Today (micro-batch-major):

```
for micro_batch in window(N):          # DataLoader loop + modulo counter
    forward all layers
    backward all layers                # one monolithic adjoint sweep
    for each param: m_partial[p] += grad[p] / N
optimizer step: for each param: consume m_partial[p]; m_partial[p] = 0
```

CSLA (layer-major, micro-batch-minor):

```
for micro_batch in window(N):
    forward all layers, SAVE block-boundary residuals[micro_batch][layer]
                                       # CCR boundaries; interiors freed

for layer L in reverse:
    accL = 0                           # ONE layer's accumulator
    for micro_batch in window(N):
        recompute layer L interior from residuals[micro_batch][L]   # CCR
        backward layer L               # produces this layer's param grads
        accL += grad(L) / N
    materialize the boundary adjoint into layer L-1 for each micro_batch
    optimizer-update layer L from accL      # only once accL is complete
    free accL and layer L's scratch
```

Live accumulator drops from `4·P` (all params) to

```
max over layers L of ( 4·bytes(params in L) + boundary adjoints live at L )
```

For a homogeneous n-layer transformer that is roughly `4·P/n + epilogue`,
i.e. a factor-of-n reduction in the accumulator surface.

## 4. Correctness — the last-backward-use guard

A per-layer update overwrites `theta[p]` with `theta[p] - lr·f(accL)`. This is
only legal once **no remaining backward operation reads `p`'s old value.** For
a stacked transformer with untied weights this is automatic (a block's weights
are read only inside that block's adjoint range). It is *not* automatic for:

- **tied embeddings / shared weights** — read in the embedding forward (early)
  and the output projection (late), so the gradient accumulates from two
  points and the update must wait for the later one;
- **adapters, cross-layer reuse, pipeline/recurrent structure** — any weight
  read from more than one layer.

The guard is a **last-backward-use dependency graph**, not `blocks.N` order.
For each parameter leaf `P` we compute

```
last_use(P) = adjoint.ops.rposition(|op| op.inputs.contains(&P))
```

(the exact pattern CCR uses at `ccr.rs:930`), extended by the NslList
lifetime-chain fixpoint (`ccr.rs:324-349`) so a param that flows into a
list-building op (e.g. RoPE `tensor_cat`) is not freed early. A parameter is
**layer-local** iff every adjoint op that reads it lies within its own layer's
adjoint range; otherwise it is **global-scope** and its update is deferred to
its true `last_use` (typically the very end of the backward).

Two tie representations exist in NSL (there is no `@tie_weights` construct):

1. **Structural tie** — the same compound field is read twice. Memoization
   collapses it to one leaf (`source_ad.rs:2595`); the last-use guard covers it
   with no extra machinery.
2. **Pointer tie** — two distinct fields alias one storage. The graph sees two
   leaves; the aliasing is reconciled only by the runtime pointer-scan
   (`stmt.rs:6526-6551`). CSLA must therefore treat any parameter reachable
   from more than one layer prefix (or flagged by the frontend) as global-scope
   and exclude it from layer-local update. The frozen-teacher subset
   (`is_frozen_compound`, `source_ad.rs:2376`) is the precedent for
   special-casing a subset of leaves.

The reordered adjoint is validated by the existing `ccr::validate`
(`ccr.rs:974`), which already asserts "no op after a FreeTensor reads its
victim" — repurposed here as "no adjoint op after a per-layer update reads that
layer's old weight."

## 5. Stage-1 analysis (implemented)

`crates/nsl-codegen/src/layerwise.rs` computes the schedule as a pure analysis
over `(primal, adjoint, named_params)`:

- `param → layer_key` via `layer_prefix` over `PrimalOp::Param` names;
- `param → last_backward_use` via `rposition` + the list-chain fixpoint;
- per-layer adjoint range `[first, last]` over the layer's param VarIds;
- **global-scope set**: params whose last use escapes their layer's range, plus
  epilogue params (embed / final norm / LM head) that carry no `blocks.N` key;
- a `MemoryProjection`: full-window elements (Σ over all params) vs the
  layerwise peak (max single layer + global), and the reduction factor.

It changes no codegen; it is the oracle the Stage-2 emitter and the WGGO memory
model consume, and its last-use graph is directly reusable by the Milestone-C
transient arena (priority 2) and the unified peak-memory scheduler (priority 6).

## 6. Reported metrics (`--training-report`)

For each train block the report prints, per layer: parameter count and element
total, the adjoint range, whether the layer is fully local; the global-scope
parameter list with the reason (epilogue / tied / cross-layer); and the
projected accumulator reduction (`full_window_elems` → `layerwise_peak_elems`,
factor). This makes the m_partial win auditable before any codegen fires.

## 7. Stage-2 codegen plan (follow-on)

1. **Buffer the window.** Replace the DataLoader-driven micro-batch loop with a
   compile-time-N gather: fetch N batches, run N forwards saving only the CCR
   block-boundary residuals per micro-batch, free interiors.
2. **Layer-major backward.** For each layer in reverse adjoint order (already
   the natural order of the reverse walk), for each buffered micro-batch:
   recompute the interior (reuse CCR's clone-splice, `ccr.rs:867`), run the
   layer's adjoint sub-range, accumulate into the layer accumulator.
3. **Per-layer update + free.** After the last micro-batch of a layer, fire the
   existing per-param optimizer step (`fase_emit_final_step`) for that layer's
   *layer-local* params only, guarded by the last-use analysis, then free the
   accumulator. Global-scope params fall through to a final epilogue update.
4. **Compose with CCR.** The reorder and CCR's splice both mutate `adjoint.ops`
   positionally and share the `last_use`/`rposition` machinery; insert after
   `eliminate_dead_gradients` (`stmt.rs:6047`) and before `compile_wengert_ops`.
5. **Gate + validate.** Behind `--layerwise-accum`; assert bit-exactness vs the
   full-window path (loss stream + model bytes) via the CCR parity harness and
   the long-run drift gate, and confirm no-offload 1B training fits 16 GiB.

Boundary-adjoint materialization (step 3's "materialize all required boundary
adjoints") is the subtlety: layer L's backward needs `d(residual after L)` for
each micro-batch, which layer L+1's backward produced. Those N boundary
adjoints must survive from L+1 to L — the CCR escape set (`BlockSegment.escaping`,
`ccr.rs:74`) already identifies exactly these tensors.

## 8. Relationship to the roadmap

CSLA is Milestone B (priority 1). It composes with — and is a prerequisite for
the exact-fidelity path of — Milestone C's arena (which colors the now-shorter
live intervals) and the fused multi-tensor optimizer (which applies the
per-layer update in one launch). It is orthogonal to Milestone D (multi-GPU
CPDT ZeRO), which shards what remains after the single-GPU accumulator is gone.
