# Compile-Time Transient-Memory Arena (Milestone C · p2)

Stage-1 analysis: `crates/nsl-codegen/src/transient_arena.rs`.
Gate: `crates/nsl-cli/tests/transient_arena_gpu_gate.rs`.
Reuses the M36 engine (`memory_planner.rs`); shares its last-use graph with the
CSLA analysis (`layerwise.rs`).

## 1. Problem

Training memory is dominated by *transients*: the forward activations kept for
the backward, the adjoint (gradient) temporaries, and the attention/flash
workspaces. In NSL today none of these are planned at compile time. They are
allocated and freed one-at-a-time through the runtime caching allocator
(`caching_allocator.rs`, a PyTorch-style block-splitting/coalescing pool). The
one compile-time arena that *does* exist — the M36 slab (`memory_planner.rs` →
`nsl_gpu_slab_*`) — walks only the top-level AST and records only `main`-level
`let` tensors built by a literal `zeros|ones|rand|randn|zeros_like`. It never
marks a value `saved_for_backward`, never looks at the Wengert adjoint tape, and
never sees a single backward or attention temporary. Precisely the memory that
matters for training gets **zero** compile-time planning.

This note is the Stage-1 planner that closes that gap on the analysis side: it
brings the M36 interference-graph + best-fit-decreasing coloring to the
forward+adjoint Wengert tape, so the transient surface can be measured, bounded,
and — in Stage-2 — placed.

## 2. The interval model

The forward list and the adjoint list are concatenated into one timeline:
forward ops at indices `0..F`, adjoint ops at `F..F+A`. Program point = position
in that concatenated sequence. Over it:

- A transient's **birth** is the index of the op that produces it.
- Its **death** is the highest index whose op reads it (`op.inputs`), including a
  CCR `FreeTensor` marker. A value produced but never read dies where it was
  born.
- A `saved_for_backward` forward value is read in the adjoint region, so its
  interval automatically stretches across the `F` boundary — the single fact the
  AST-only M36 planner cannot represent.
- `FreeTensor` markers spliced into the **adjoint** (the adjoint-region last-use
  frees, `ccr.rs`) count as reads, so those intervals end exactly where the pass
  freed the value. CCR *primal* early-frees are compiled from a separate
  free-list not in the analyzed forward list, so a primal value freed early is
  modeled as living to its last use — a conservative over-estimate, never an
  under-count.

**What counts as a transient allocation.** Leaves (`Input`/`Param`/`Constant`)
are persistent, not transient. `FreeTensor` is a marker with no result.
Non-tensor results (shape/index/list scalars — `WengertType != Tensor`) never
touch device memory. Exactly one op is excluded as an aliasing view:
`Transpose`, whose lowering (`nsl_tensor_transpose` → `new_view_i64`) is a
verified zero-copy stride view. The other shape ops that superficially look like
views actually **materialize** on the tape this pass analyzes — `Reshape` and
`Broadcast` lower to `nsl_tensor_clone`, `Slice` to a fresh `alloc_managed` +
kernel, `Select` to `nsl_tensor_where` — and are heavily emitted in the backward
(norm-layer backward clones several broadcasts; every Relu/Clamp/Abs/Where
backward emits a Select). Treating those as allocations is the **safe
(never-undercount)** direction and reflects real device memory; excluding them
would understate the very surface this pass exists to measure and, in Stage-2,
leave a live tensor unplaced. Everything else that yields a device tensor is an
allocation (`op_allocates`).

## 3. Why max-concurrency, not bytes, is the headline

NSL tensor shapes are largely runtime values — the same reason the CSLA analysis
projects in element counts, not bytes. Most transients here have no static
shape, so byte sizing is usually unavailable at this stage. The honest,
always-available result is therefore **max concurrency**: the maximum number of
transients ever simultaneously live, computed by an endpoint sweep over the
occupancy intervals `[birth, last_use + 1)`. The `+ 1` is memory-correct, not
padding: a value's last use is an *input* to the op at that index, which also
*writes* whatever it produces there, so the dying and newborn values coexist in
memory during that op. Encoding the end as `last_use + 1` makes both this sweep
and M36's half-open `intervals_overlap` (used by the byte path) count that
coexistence identically, so the two paths never disagree on adjacent intervals.
For the temporal interval graph that peak is the minimum slot count (its
chromatic number); `saved_for_backward` force-interference in the byte path can
require more. On the packed-GQA gate fixture it reports (checkpointed): **463
transient tensors collapse to a peak of 71 concurrently-live slots (6.5×)**;
plain: **349 → 60 (5.8×)**.

When *every* transient's element count is known (the CPU unit tests, and any
statically-shaped program), the quantified path fires: it builds
`memory_planner::TensorAlloc`s and runs the real `InterferenceGraph::build` +
`plan_slab` (BFD) to report arena bytes, slot count, savings, and fragmentation
— the identical engine the M36 slab uses. `saved_for_backward` force-interference
(gradient-key uniqueness) is honored there, so two disjoint saved tensors are
never coalesced.

## 4. Honest baseline — what an arena actually buys

The runtime caching allocator already recycles blocks well (best-fit free-lists,
O(1) coalescing, same-pool bias). So an arena's marginal win over it on *peak
bytes* is modest. The real wins are elsewhere, and this analysis is the oracle
that unlocks them:

1. **Churn / latency.** Hundreds of individual alloc/free round-trips per step
   collapse to a handful of fixed offsets — no per-transient pool lookup on the
   hot path.
2. **Fragmentation.** Fixed interval-colored offsets pack tighter than
   online best-fit under a shifting allocation order.
3. **Stable pointers → CUDA-graph replay (p8).** Graph replay requires the same
   device addresses every iteration. A caching allocator returns different
   pointers as the free-list shifts; a compile-time arena hands out the *same*
   offset every step. The arena is therefore a **prerequisite** for p8, and this
   plan is what p8 (and the p6 unified peak-memory scheduler) consume.
4. **Determinism.** A fixed layout removes an allocation-order source of run-to-run
   variation.

## 5. The CCR nuance (measured, not hidden)

On the fixture, enabling `--checkpoint-blocks` *raises* the transient count
(349 → 463) and peak concurrency (60 → 71). That is expected and correct:
checkpointing frees the large activations but adds many small recompute-clone
temporaries during the backward, and the size-agnostic concurrency metric counts
objects, not bytes. CCR's win is in **bytes** (freeing big tensors), which the
count cannot see. Byte quantification (Stage-2) is what would show CCR's peak-byte
reduction; the concurrency number is an honest proxy for allocator *churn*, not
for resident bytes. The gate asserts the invariant (peak < transients, ≥2×
collapse) rather than any exact number.

## 6. Validation

- **CPU unit tests** (`transient_arena.rs`): empty-safe; leaves/views excluded;
  a saved forward value's interval spans into the backward; `FreeTensor` bounds
  liveness; concurrency equals the interval-graph optimum; the quantified path
  reuses BFD and beats naive; saved tensors never share a slot.
- **GPU gate** (`transient_arena_gpu_gate.rs`, `#[ignore]`): compiles a real
  packed-GQA LM under source-AD + CCR on the GPU, asserts a well-formed report
  with a ≥2× collapse, and proves the analysis is **loss-neutral** — the loss
  stream is bit-identical with the report on and off (pure analysis, no codegen
  change).

## 7. Stage-2 plan (follow-on)

Stage-1 changes no codegen and hands out no offsets. Stage-2, in order of
increasing risk:

1. **Byte quantification.** Thread the type checker's shape info to `elems_of`
   so the quantified BFD path fires on real models, giving arena *bytes* and
   surfacing CCR's byte-peak win. Report-only; zero runtime risk.
2. **Runtime wiring.** Two options: (a) extend the existing, already-wired
   `nsl_gpu_slab_*` arena to the backward transients by having `wengert_lower`
   allocate planned results from the slab at planned offsets; or (b) add a
   per-tensor offset channel to the caching allocator. Either must respect a
   **correctness minefield**: a fixed offset assumes the plan's liveness is
   exact, but views hold refcounts, CCR splices recompute clones, and
   saved-for-backward addresses are gradient-map keys. A liveness error →
   silent memory corruption. This is exactly why Stage-1 lands the analysis and
   validation first; the runtime placement follows only behind byte
   quantification and an exhaustive equivalence gate.
3. **Feed p6/p8.** Expose the plan to the unified peak-memory scheduler (p6) and
   emit stable offsets for CUDA-graph capture (p8).

## 8. Relationship to the roadmap

p2 is the first Milestone-C item. It deliberately *extends* the M36 engine
rather than duplicating it (the roadmap's explicit instruction), reuses the CSLA
last-backward-use graph (Milestone B), and produces the stable-offset oracle that
p8 (CUDA graphs) and p6 (unified scheduler) depend on. The staged
foundation-first shape mirrors A1–A3 and B1: measure and validate the model on
real GPU graphs before any placement change that could corrupt memory.
