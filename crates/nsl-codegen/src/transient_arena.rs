//! Milestone C · p2 — compile-time transient-memory arena (Stage-1 planner).
//!
//! The M36 slab planner (`memory_planner.rs`) already computes an
//! interference-graph + best-fit-decreasing offset assignment and is wired to a
//! real runtime arena (`nsl_gpu_slab_*`). But its coverage is narrow: it walks
//! only the top-level AST and only records `main`-level `let` tensors built by a
//! literal `zeros|ones|rand|randn|zeros_like` — i.e. the *forward* prologue. It
//! never marks a tensor `saved_for_backward`, never sees the Wengert adjoint
//! tape, and never models the attention/backward temporaries that dominate
//! training memory. Those all flow through the runtime caching allocator with
//! *zero* compile-time planning.
//!
//! This module extends the M36 engine to that missing surface. It does **not**
//! duplicate the coloring — it reuses `memory_planner::{InterferenceGraph,
//! plan_slab, SizeKind, TensorAlloc}` verbatim, feeding them from the
//! concatenated `[forward ; adjoint]` Wengert timeline instead of the AST. On
//! that unified timeline a `saved_for_backward` forward value's live interval
//! correctly stretches from its forward definition to its last *backward* use.
//! CCR `FreeTensor` markers spliced into the adjoint (the adjoint-region
//! last-use frees) bound liveness where the checkpointing pass freed a value;
//! CCR *primal* early-frees are compiled from a separate free-list not present
//! in the analyzed forward list, so a primal value freed early is modeled as
//! living to its last use — a conservative over-estimate, never an under-count.
//!
//! Because NSL tensor shapes are largely runtime values (the same reason
//! [`crate::layerwise`] projects in element counts, not bytes), the headline
//! result is the **max concurrency** — the maximum number of transient tensors
//! ever simultaneously live. That is the minimum slot count for the temporal
//! interval graph (its chromatic number); `saved_for_backward` force-interference
//! — gradient-key uniqueness, honored by the byte-level plan — can push the
//! quantified slot count above it. It is the honest measure of what
//! a static arena buys over the caching allocator: hundreds of individual
//! alloc/free round-trips collapse to a handful of fixed, *stable* offsets.
//! Stable offsets are also the prerequisite for CUDA-graph replay (Milestone
//! C · p8), so this plan is the oracle p8 and the unified peak-memory scheduler
//! (p6) consume.
//!
//! Stage-1 is pure analysis — it changes no codegen and hands out no offsets.
//! The runtime wiring (routing backward transients through a planned arena) is
//! Stage-2; see `docs/research/transient-memory-arena.md`. Its last-use graph
//! is shared with [`crate::layerwise`] (both key off `op.inputs` positions).

use std::collections::HashMap;

use crate::memory_planner::{
    align_up, plan_slab, InterferenceGraph, SizeKind, SlabPlan, TensorAlloc, SLAB_ALIGNMENT,
};
use crate::wengert::{PrimalOp, VarId, WengertList, WengertType};

/// Which half of the timeline defined a transient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Forward,
    Backward,
}

/// One transient tensor's arena facts, on the concatenated timeline.
#[derive(Debug, Clone)]
pub struct Transient {
    pub var: VarId,
    /// Producing-op index on the concatenated `[forward ; adjoint]` timeline.
    pub birth: u32,
    /// Last index (fwd or bwd, incl. a `FreeTensor` marker) that reads `var`.
    /// Equals `birth` when the value is produced but never consumed.
    pub death: u32,
    /// Element count when a shape was available; `None` when runtime-shaped.
    pub elems: Option<u64>,
    /// Set for a forward value still read in the backward — its interval spans
    /// the fwd/bwd boundary and it may never share a slot (gradient-key safety,
    /// mirrored from M36's `InterferenceGraph`).
    pub saved_for_backward: bool,
    pub region: Region,
}

impl Transient {
    /// Length of this transient's live interval on the concatenated timeline.
    pub fn live_range(&self) -> u32 {
        self.death.saturating_sub(self.birth)
    }
}

/// The transient-arena plan for one train block.
#[derive(Debug, Clone)]
pub struct ArenaPlan {
    pub transients: Vec<Transient>,
    /// Max number of transients simultaneously live — the minimum slot count
    /// for the temporal interval graph (its chromatic number, size-agnostic).
    /// The quantified byte plan can exceed it when `saved_for_backward`
    /// force-interference forbids otherwise-legal slot sharing. Headline result
    /// when shapes are unknown.
    pub max_concurrency: usize,
    /// Program point at which `max_concurrency` is reached (first such point).
    pub peak_at: u32,
    /// Byte-level BFD packing over the SIZED subset of transients (Stage-2A
    /// partial quantification; `slab_transients` maps its dense ids back to
    /// `transients` positions). `elem_bytes` was assumed for the dtype.
    /// `None` only when nothing at all could be sized.
    pub slab: Option<SlabPlan>,
    /// Σ of live-interval lengths — a proxy for the alloc/free churn the
    /// caching allocator performs that a static arena removes.
    pub total_alloc_events: usize,
    /// Bytes assumed per element for the quantified path (GPU training = f32).
    pub elem_bytes: u64,
    /// Transients with NO static element count (Stage-2A partial
    /// quantification): they are outside `slab` and make every byte figure a
    /// lower bound. Zero means the slab covers the whole tape.
    pub unsized_count: usize,
    /// Dense alloc id -> position in `transients`, for the sized subset the
    /// slab was planned over (`slab.assignments` is keyed by the dense id).
    /// Identity permutation when `unsized_count == 0`.
    pub slab_transients: Vec<usize>,
    /// Transients pinned live-to-tape-end because something AFTER the tape
    /// reads them (gradient accumulators, the loss). They can never share a
    /// slot with each other, so their arena contribution is exactly their
    /// sum — the honest floor, not a packing win.
    pub escaping_count: usize,
}

/// True when `op` materializes a fresh device tensor that a transient arena
/// would place. Leaves (`Input`/`Param`/`Constant`) are persistent, not
/// transient; `FreeTensor` is a marker with no result; non-tensor results
/// (shape/index/list scalars — `WengertType != Tensor`) never touch device
/// memory.
///
/// Only `Transpose` is excluded as an aliasing op: `nsl_tensor_transpose` is a
/// *verified* zero-copy stride view (`NslTensor::new_view_i64`,
/// `tensor/shape_ops.rs`). The other shape ops that *look* like views actually
/// COPY on the exact tape this pass analyzes — `Reshape` and `Broadcast` lower
/// to `nsl_tensor_clone` (`wengert_lower.rs`), `Slice` to a fresh
/// `alloc_managed` + slice kernel (`cuda/mod.rs`), and `Select` to
/// `nsl_tensor_where` (`ad_ops.rs`). They are real allocations — and heavily
/// emitted in the backward (RMSNorm/LayerNorm backward alone clones ~6
/// broadcasts each; every Relu/Clamp/Abs/Where backward emits a Select) — so
/// they are counted. Counting a genuine allocation is the safe
/// (never-undercount) direction; excluding one would understate the very
/// surface this pass exists to measure and, in Stage-2, leave a live tensor
/// unplaced.
fn op_allocates(op: &PrimalOp, result_ty: WengertType) -> bool {
    if result_ty != WengertType::Tensor {
        return false;
    }
    !matches!(
        op,
        PrimalOp::Input(_)
            | PrimalOp::Param(_)
            | PrimalOp::Constant(_)
            | PrimalOp::FreeTensor
            | PrimalOp::Transpose { .. }
    )
}

/// The Cranelift-level type of a produced VarId, defaulting to the op's natural
/// type when the list carries no explicit override (matches the lowerer).
fn result_type(list: &WengertList, op: &crate::wengert::WengertOp) -> WengertType {
    list.var_types
        .get(&op.result)
        .copied()
        .unwrap_or_else(|| crate::wengert::type_for_op(&op.op))
}

/// Analyze the transient-memory arena over the forward + adjoint tapes.
///
/// `elems_of(var)` returns a transient's element count when a static shape is
/// known (`|_| None` yields the size-agnostic concurrency result — the common
/// case, since most transients are runtime-shaped). `elem_bytes` is the assumed
/// dtype width for the quantified path (4 for GPU f32 training).
/// `tape_escaping` names values consumed AFTER the tape — param-gradient
/// accumulators read by the optimizer emission, the loss read by callbacks.
/// The tape itself never reads them, so pure last-use liveness would give
/// them `death == birth` and BFD would happily time-share every gradient in
/// one slot — an illegal aliasing that inflated the byte savings until the
/// review caught it. Escaping values are pinned live to tape end instead.
pub fn analyze(
    forward: &WengertList,
    adjoint: &WengertList,
    elems_of: &dyn Fn(VarId) -> Option<u64>,
    tape_escaping: &std::collections::HashSet<VarId>,
    elem_bytes: u64,
) -> ArenaPlan {
    // Concatenated timeline: forward ops first, then adjoint. A saved forward
    // value read in the backward therefore has death >= forward.len().
    let fwd_n = forward.ops.len();
    let tape_end = (fwd_n + adjoint.ops.len()) as u32;

    // last_use[var] = highest concatenated index whose op reads `var`.
    let mut last_use: HashMap<VarId, u32> = HashMap::new();
    let mut record_uses = |ops: &[crate::wengert::WengertOp], base: usize| {
        for (i, op) in ops.iter().enumerate() {
            let idx = (base + i) as u32;
            for &input in &op.inputs {
                last_use.insert(input, idx); // monotonic -> ends at the last read
            }
        }
    };
    record_uses(&forward.ops, 0);
    record_uses(&adjoint.ops, fwd_n);

    // Build one transient per allocating op result.
    let mut transients: Vec<Transient> = Vec::new();
    let push_from = |ops: &[crate::wengert::WengertOp],
                     list: &WengertList,
                     base: usize,
                     region: Region,
                     transients: &mut Vec<Transient>| {
        for (i, op) in ops.iter().enumerate() {
            if !op_allocates(&op.op, result_type(list, op)) {
                continue;
            }
            let birth = (base + i) as u32;
            // Never-used values die where they were born (allocated then
            // freed) — unless they ESCAPE the tape, in which case they live
            // to its end (nothing on the tape reads a value the optimizer
            // or a callback consumes afterwards).
            let mut death = last_use.get(&op.result).copied().unwrap_or(birth).max(birth);
            if tape_escaping.contains(&op.result) {
                death = death.max(tape_end.saturating_sub(1));
            }
            transients.push(Transient {
                var: op.result,
                birth,
                death,
                elems: elems_of(op.result),
                saved_for_backward: op.saved_for_backward,
                region,
            });
        }
    };
    push_from(&forward.ops, forward, 0, Region::Forward, &mut transients);
    push_from(&adjoint.ops, adjoint, fwd_n, Region::Backward, &mut transients);

    // Max concurrency via an endpoint sweep over half-open [birth, death)
    // intervals. This is the size-agnostic slot optimum for the temporal
    // (non-saved) part; saved-for-backward force-interference is honored by the
    // byte-level plan below when sizes are known.
    let (max_concurrency, peak_at) = sweep_max_concurrency(&transients);

    let total_alloc_events = transients.len();

    // Quantified path (Stage-2A): PARTIAL quantification. The old gate was
    // all-or-nothing (`all(|t| t.elems.is_some())`), which on real programs
    // meant the byte path never fired at all — one runtime-shaped transient
    // starved the whole plan. The slab is now planned over the SIZED subset;
    // unsized transients are counted and reported, never silently absorbed.
    // The resulting byte total is therefore a LOWER BOUND on the full arena
    // (Stage-2 placement can only place what it can size), and the report
    // says so whenever the unsized count is nonzero.
    // The M36 engine indexes its interference/offset arrays by alloc id, so
    // ids must be DENSE 0..n over the sized subset; `slab_transients` maps
    // each dense id back to its position in `transients` (identity when
    // everything is sized, i.e. the pre-2A behavior).
    let slab_transients: Vec<usize> = transients
        .iter()
        .enumerate()
        .filter(|(_, t)| t.elems.is_some())
        .map(|(i, _)| i)
        .collect();
    let unsized_count = transients.len() - slab_transients.len();
    let slab = if !slab_transients.is_empty() {
        let allocs: Vec<TensorAlloc> = slab_transients
            .iter()
            .enumerate()
            .map(|(dense, &ti)| {
                let t = &transients[ti];
                let bytes = t.elems.unwrap().saturating_mul(elem_bytes);
                TensorAlloc {
                    id: dense as u32,
                    name: format!("t{}", t.var),
                    size_bytes: bytes,
                    birth: t.birth,
                    // Occupancy-exclusive end (`last_use + 1`), so M36's
                    // half-open `intervals_overlap` counts the same
                    // coexistence-during-the-consuming-op the sweep does.
                    death: occupancy_end(t),
                    source_loc: String::new(),
                    size_kind: SizeKind::Static(bytes),
                    saved_for_backward: t.saved_for_backward,
                }
            })
            .collect();
        let graph = InterferenceGraph::build(&allocs);
        Some(plan_slab(&allocs, &graph))
    } else {
        None
    };

    let escaping_count = transients
        .iter()
        .filter(|t| tape_escaping.contains(&t.var))
        .count();

    ArenaPlan {
        transients,
        max_concurrency,
        peak_at,
        slab,
        slab_transients,
        total_alloc_events,
        elem_bytes,
        unsized_count,
        escaping_count,
    }
}

// ---------------------------------------------------------------------------
// Stage-2B: element-count propagation
// ---------------------------------------------------------------------------

/// Propagate element counts forward through a tape from a seed.
///
/// # Why this had to exist before placement was worth anything
///
/// Measured on `models/coder50m/pretrain_cert.nsl`, the Stage-2A hint bridge
/// sized **1 of 1321** transients: it derives counts from model-field dims and
/// mirrors them onto param-gradient accumulators, and nothing else on the tape
/// has an AST node the type checker gave a concrete shape. So every byte
/// figure the arena reported was one transient's worth, and Stage-2B admitted
/// nothing at all — not because the admission rules were too strict, but
/// because 843 of 844 backward temporaries had no size to pack.
///
/// A pass that places nothing is indistinguishable from a pass that is off.
///
/// # What is propagated, and what deliberately is not
///
/// Only the ops whose output element count is determined by their inputs
/// WITHOUT knowing the shapes:
///
/// * unary elementwise — numel is preserved exactly;
/// * binary elementwise where both operands have the SAME numel — no
///   broadcast is possible, so the result matches.
///
/// A binary op whose operands differ in numel broadcasts, and the result's
/// numel is then a function of the shapes, which the tape does not carry — so
/// it stops. Matmul, reductions, reshape, concat and split all likewise stop:
/// each would need real shapes, and guessing one produces an undersized slot,
/// which is a buffer overrun rather than a bad statistic.
///
/// Stopping is free: an unsized transient is simply refused admission and
/// stays on the caching allocator.
pub fn propagate_elems(
    forward: &WengertList,
    adjoint: &WengertList,
    seed: &dyn Fn(VarId) -> Option<u64>,
) -> HashMap<VarId, u64> {
    let mut known: HashMap<VarId, u64> = HashMap::new();
    let note = |v: VarId, known: &mut HashMap<VarId, u64>| {
        if let Some(n) = seed(v) {
            known.insert(v, n);
        }
    };
    // Seed leaves and anything the caller already sized.
    for list in [forward, adjoint] {
        for op in &list.ops {
            note(op.result, &mut known);
            for &i in &op.inputs {
                note(i, &mut known);
            }
        }
    }
    // One forward sweep over the concatenated timeline. The tape is in
    // topological order by construction, so a single pass reaches every
    // derivable value; iterating to a fixpoint would buy nothing.
    for list in [forward, adjoint] {
        for op in &list.ops {
            if known.contains_key(&op.result) {
                continue;
            }
            let derived = match &op.op {
                PrimalOp::Neg
                | PrimalOp::Relu
                | PrimalOp::Gelu
                | PrimalOp::Silu
                | PrimalOp::Sigmoid
                | PrimalOp::Tanh
                | PrimalOp::Exp
                | PrimalOp::Log
                | PrimalOp::Sqrt => op.inputs.first().and_then(|i| known.get(i).copied()),
                PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div => {
                    match (op.inputs.first(), op.inputs.get(1)) {
                        (Some(a), Some(b)) => {
                            match (known.get(a).copied(), known.get(b).copied()) {
                                // Equal numel ⇒ no broadcast ⇒ result matches.
                                // Unequal ⇒ a broadcast whose result depends on
                                // the shapes, which the tape does not carry.
                                (Some(x), Some(y)) if x == y => Some(x),
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(n) = derived {
                known.insert(op.result, n);
            }
        }
    }
    known
}

// ---------------------------------------------------------------------------
// Stage-2B: shape propagation
// ---------------------------------------------------------------------------

/// What is statically known about one tape value's size.
///
/// Two levels rather than one because the shipped models run every batch
/// through a runtime-shaped `reshape` (`logits.reshape([ls[0]*ls[1], ls[2]])`,
/// `input_ids` flattened for the rank-1 embedding lookup): the dims are lost
/// there, but the element count is not — and a matmul against a rank-2
/// weight of known shape can carry the COUNT across (`numel_out =
/// numel_in / k * n`, exact because a mismatched inner dim is a runtime
/// abort, so any program that runs satisfies it). Numel-only propagation
/// with no shape seeds could do neither, which is why it sized 0 of 1321
/// transients on coder50m.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SizeInfo {
    /// Exact dims, every entry >= 1.
    Shape(Vec<i64>),
    /// Element count only — the dims died at a runtime-shaped op, the count
    /// survived. Enough to size a slot; not enough to cross a dim-dependent
    /// op (reductions, slices, non-rank-2 matmuls).
    Numel(u64),
}

impl SizeInfo {
    pub fn numel(&self) -> Option<u64> {
        match self {
            SizeInfo::Shape(dims) => shape_numel(dims),
            SizeInfo::Numel(n) => Some(*n),
        }
    }
}

/// A compile-time-constant VALUE the propagation tracks alongside sizes.
///
/// The shipped attention stack routes its shape arithmetic through runtime
/// values: `let s = x.shape; let batch = s[0]`, `let nh =
/// int(self._n_heads.item())`, `q.reshape([batch, seq_len, nh, hd])`. Every
/// one of those is a compile-time constant on a proven-loader tape — but
/// only if `shape`-of-a-known-var folds to a list, `subscript` of that list
/// folds to an int, integer arithmetic folds, and `item()` of a 1-element
/// config tensor folds to its constructor-folded value. Without this
/// lattice the forward chain died at the first `q.reshape(...)` and the
/// arena placed 8 of 1321 transients.
#[derive(Debug, Clone, PartialEq)]
enum ConstEntry {
    Int(i64),
    Num(f64),
    List(Vec<i64>),
}

impl ConstEntry {
    fn as_int(&self) -> Option<i64> {
        match self {
            ConstEntry::Int(n) => Some(*n),
            // An integral float IS the int for dim purposes ("item" yields
            // f64; `full([1], float(8))` round-trips through Num).
            ConstEntry::Num(f) if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.0e15 => {
                Some(*f as i64)
            }
            _ => None,
        }
    }
}

/// Checked product of validated-positive dims; empty (rank 0) = 1 element.
/// Bounded by `i64::MAX` — beyond that the byte math downstream (`pack`
/// multiplies by `elem_bytes` unchecked) could wrap, and no real tensor is
/// within 2^20 of the bound anyway.
fn shape_numel(dims: &[i64]) -> Option<u64> {
    dims.iter()
        .try_fold(1u64, |acc, &d| {
            if d <= 0 {
                None
            } else {
                acc.checked_mul(d as u64)
            }
        })
        .filter(|&n| n <= i64::MAX as u64)
}

/// NumPy right-aligned broadcast: pad the shorter shape with leading 1s;
/// per dim the sizes must be equal or one must be 1 (else None — the runtime
/// aborts on exactly that condition, `cpu.rs` `tensor_elementwise_op`).
fn numpy_broadcast(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let da = if i < n - a.len() { 1 } else { a[i - (n - a.len())] };
        let db = if i < n - b.len() { 1 } else { b[i - (n - b.len())] };
        if da == db || db == 1 {
            out.push(da);
        } else if da == 1 {
            out.push(db);
        } else {
            return None;
        }
    }
    Some(out)
}

/// Propagate [`SizeInfo`] through the forward tape, mirror onto adjoint
/// accumulators, then propagate through the adjoint.
///
/// `seed(var)` supplies dims for leaves (semantic-typed values, model-field
/// initializer dims, DataLoader-proven batch fields). `scalar_seed(var)`
/// supplies the VALUE of a 1-element config tensor (constructor-folded
/// `full([1], v)` fields), which is what lets `int(self._n_heads.item())`
/// fold. `mirror` is `(primal, adjoint_accumulator)` pairs restricted by the
/// caller to UNIQUE preimages — an adjoint has its primal's shape by
/// construction, but a shared accumulator (Add/Sub alias the output's
/// adjoint onto both operands) carries the OUTPUT's shape and must not be
/// mirrored.
///
/// Every rule was traced to the lowering + runtime kernel it describes; an
/// op with no rule, an out-of-range dim, an overflowing product, or a
/// device-divergent case (equal-numel differing-shape binaries, `Gather`,
/// `ScatterAdd`) derives nothing. Stopping is free — an unsized transient is
/// refused admission and stays on the caching allocator — while a wrong size
/// can never place a tensor (the runtime pin fires only on an EXACT size
/// match and the bind/placement reconciliation reports the miss), so the
/// failure direction of any rule error is loud and benign, not corruption.
pub fn propagate_size_info(
    forward: &WengertList,
    adjoint: &WengertList,
    seed: &dyn Fn(VarId) -> Option<Vec<i64>>,
    scalar_seed: &dyn Fn(VarId) -> Option<f64>,
    mirror: &[(VarId, VarId)],
) -> HashMap<VarId, SizeInfo> {
    let mut prop = SizeProp {
        known: HashMap::new(),
        consts: HashMap::new(),
        op_of: HashMap::new(),
        ty_of: HashMap::new(),
    };
    // Producing-op + type lookup across BOTH lists (adjoint ops reference
    // forward vars; the tape is topological so a lookup only ever chases an
    // already-defined var).
    for list in [forward, adjoint] {
        for op in &list.ops {
            prop.op_of.entry(op.result).or_insert(op);
            prop.ty_of
                .entry(op.result)
                .or_insert_with(|| result_type(list, op));
        }
    }
    for list in [forward, adjoint] {
        for op in &list.ops {
            for v in op.inputs.iter().chain(std::iter::once(&op.result)) {
                if !prop.known.contains_key(v) {
                    if let Some(dims) = seed(*v) {
                        if shape_numel(&dims).is_some() {
                            prop.known.insert(*v, SizeInfo::Shape(dims));
                        }
                    }
                }
                if !prop.consts.contains_key(v) {
                    if let Some(val) = scalar_seed(*v) {
                        prop.consts.insert(*v, ConstEntry::Num(val));
                    }
                }
            }
        }
    }

    prop.sweep(forward);
    for (p, a) in mirror {
        if let Some(info) = prop.known.get(p).cloned() {
            prop.known.entry(*a).or_insert(info);
        }
    }
    prop.sweep(adjoint);
    prop.known
}

struct SizeProp<'a> {
    known: HashMap<VarId, SizeInfo>,
    consts: HashMap<VarId, ConstEntry>,
    op_of: HashMap<VarId, &'a crate::wengert::WengertOp>,
    ty_of: HashMap<VarId, WengertType>,
}

impl<'a> SizeProp<'a> {
    fn sweep(&mut self, list: &'a WengertList) {
        for op in &list.ops {
            // Constant values first: they feed the very size rules below
            // (a `shape` read of a sized var -> the reshape list that sizes
            // the next var).
            if !self.consts.contains_key(&op.result) {
                if let Some(c) = self.derive_const(op) {
                    self.consts.insert(op.result, c);
                }
            }
            if self.known.contains_key(&op.result) {
                continue;
            }
            // Only tensor results occupy device memory; a List/Scalar/
            // Integer result must never be given a byte size.
            if self.ty_of.get(&op.result) != Some(&WengertType::Tensor) {
                continue;
            }
            if let Some(info) = self.derive_size(op) {
                if info.numel().is_some() {
                    self.known.insert(op.result, info);
                }
            }
        }
    }

    /// The known size of an op INPUT. A Scalar/Integer-typed operand is
    /// promoted to a 0-dim tensor by the lowerer (`promote_to_tensor` ->
    /// `nsl_tensor_scalar`), so it participates in broadcast as rank 0.
    fn input_info(&self, op: &crate::wengert::WengertOp, i: usize) -> Option<SizeInfo> {
        let v = *op.inputs.get(i)?;
        if let Some(info) = self.known.get(&v) {
            return Some(info.clone());
        }
        match self.ty_of.get(&v) {
            Some(WengertType::Scalar) | Some(WengertType::Integer) => {
                Some(SizeInfo::Shape(vec![]))
            }
            _ => None,
        }
    }

    fn const_of(&self, v: VarId) -> Option<&ConstEntry> {
        self.consts.get(&v)
    }

    fn const_int(&self, v: VarId) -> Option<i64> {
        self.const_of(v)?.as_int()
    }

    /// Statically-recoverable dims of a runtime shape LIST: an item-4
    /// `const_shape:` op (dims baked into the op name), a `list`
    /// passthrough whose every element folded, or a folded `shape` read.
    fn static_list_dims(&self, v: VarId) -> Option<Vec<i64>> {
        if let Some(ConstEntry::List(dims)) = self.const_of(v) {
            if !dims.is_empty() && dims.iter().all(|&d| d > 0) {
                return Some(dims.clone());
            }
        }
        None
    }

    /// Constant-value rules. Every arm mirrors the lowering it names.
    fn derive_const(&self, op: &crate::wengert::WengertOp) -> Option<ConstEntry> {
        use ConstEntry::{Int, List, Num};
        match &op.op {
            PrimalOp::Constant(f) => {
                if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.0e15 {
                    Some(Int(*f as i64))
                } else {
                    Some(Num(*f))
                }
            }
            // Arithmetic on folded scalar values. NOT gated on the result's
            // Wengert type for Add/Sub/Mul: `batch_size * seq_len` defaults
            // to Tensor in `type_for_op` (the Integer-vs-tensor choice
            // happens at lowering, by INPUT types, with no override
            // recorded), and those three agree between `imul` and a 0-dim
            // tensor op on in-range integers. DIVISION does not: the
            // lowering emits truncating `sdiv` only when BOTH operands are
            // Integer-typed, and `nsl_tensor_div` otherwise divides in
            // floating point — `100 / item(_n_heads)` is 12 one way and
            // 12.5 the other. So Div folds as integer only under the same
            // both-Integer condition the lowering checks; every other
            // typing folds as f64, matching `nsl_tensor_div` exactly (and
            // an inexact quotient then refuses `as_int`, so downstream
            // dims stop instead of claiming falsely).
            PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div => {
                let a = self.const_of(*op.inputs.first()?)?;
                let b = self.const_of(*op.inputs.get(1)?)?;
                let both_integer_typed = op.inputs.len() >= 2
                    && op.inputs.iter().take(2).all(|v| {
                        self.ty_of.get(v) == Some(&WengertType::Integer)
                    });
                match (a.as_int(), b.as_int()) {
                    (Some(x), Some(y))
                        if !matches!(op.op, PrimalOp::Div) || both_integer_typed =>
                    {
                        match op.op {
                            PrimalOp::Add => x.checked_add(y).map(Int),
                            PrimalOp::Sub => x.checked_sub(y).map(Int),
                            PrimalOp::Mul => x.checked_mul(y).map(Int),
                            // Cranelift `sdiv` truncates toward zero.
                            PrimalOp::Div => x.checked_div(y).map(Int),
                            _ => unreachable!(),
                        }
                    }
                    _ => {
                        let to_f = |c: &ConstEntry| match c {
                            Int(n) => Some(*n as f64),
                            Num(f) => Some(*f),
                            List(_) => None,
                        };
                        let (x, y) = (to_f(a)?, to_f(b)?);
                        let r = match op.op {
                            PrimalOp::Add => x + y,
                            PrimalOp::Sub => x - y,
                            PrimalOp::Mul => x * y,
                            PrimalOp::Div => x / y,
                            _ => unreachable!(),
                        };
                        Some(Num(r))
                    }
                }
            }
            PrimalOp::Passthrough(name) => match name.as_str() {
                // The value of a 1-element tensor: only present when the
                // tensor's value was seeded (constructor-folded config
                // fields) or itself derived.
                "item" => match self.const_of(*op.inputs.first()?)? {
                    Num(f) => Some(Num(*f)),
                    Int(n) => Some(Num(*n as f64)),
                    List(_) => None,
                },
                "int" => self.const_int(*op.inputs.first()?).map(Int),
                "float" => match self.const_of(*op.inputs.first()?)? {
                    Int(n) => Some(Num(*n as f64)),
                    Num(f) => Some(Num(*f)),
                    List(_) => None,
                },
                // A shape READ of a var whose dims are known IS a constant
                // list. This is the bridge that turns `let s = x.shape;
                // s[0]` into a foldable int.
                "shape" => {
                    let v = *op.inputs.first()?;
                    match self.known.get(&v)? {
                        SizeInfo::Shape(dims) => Some(List(dims.clone())),
                        SizeInfo::Numel(_) => None,
                    }
                }
                "subscript" => {
                    let list = match self.const_of(*op.inputs.first()?)? {
                        List(l) => l.clone(),
                        _ => return None,
                    };
                    let idx = self.const_int(*op.inputs.get(1)?)?;
                    let n = list.len() as i64;
                    let idx = if idx < 0 { n + idx } else { idx };
                    if (0..n).contains(&idx) {
                        Some(Int(list[idx as usize]))
                    } else {
                        None
                    }
                }
                "list" => {
                    let dims: Option<Vec<i64>> = op
                        .inputs
                        .iter()
                        .map(|&i| self.const_int(i))
                        .collect();
                    Some(List(dims?))
                }
                n if n.starts_with(crate::lm_head_inference::CONST_SHAPE_PREFIX) => {
                    let dims = crate::lm_head_inference::parse_const_shape_dims(n);
                    if dims.is_empty() {
                        None
                    } else {
                        Some(List(dims))
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Elementwise-binary result size (`Add/Sub/Mul/Div` semantics).
    fn binary_size(&self, a: SizeInfo, b: SizeInfo) -> Option<SizeInfo> {
        use SizeInfo::{Numel, Shape};
        match (a, b) {
            (Shape(sa), Shape(sb)) => {
                if sa == sb {
                    return Some(Shape(sa));
                }
                let na = shape_numel(&sa)?;
                let nb = shape_numel(&sb)?;
                // Equal-numel differing-shape pairs diverge by device: the
                // GPU flat path skips broadcasting and stamps A's shape,
                // the CPU path broadcasts (possibly to something LARGER).
                // Refuse rather than pick a device. Numel-1 sides broadcast
                // identically everywhere and fall through to NumPy.
                if na == nb && na > 1 {
                    return None;
                }
                numpy_broadcast(&sa, &sb).map(Shape)
            }
            (Shape(s), Numel(n)) | (Numel(n), Shape(s)) => {
                let ns = shape_numel(&s)?;
                if n == 1 {
                    // The count is exact; the RANK of the numel-1 side is
                    // unknown and can exceed the shaped side's, so dims are
                    // not claimable.
                    Some(Numel(ns))
                } else if ns == 1 {
                    Some(Numel(n))
                } else {
                    None
                }
            }
            (Numel(x), Numel(y)) => {
                if x == 1 {
                    Some(Numel(y))
                } else if y == 1 {
                    Some(Numel(x))
                } else {
                    None
                }
            }
        }
    }

    /// Matmul result size. `[..b, m, k] x [..b', k, n] -> broadcast(b, b')
    /// ++ [m, n]` when both shapes are known; when one side is numel-only
    /// the count still crosses if the OTHER side is rank-2 (no batch dims
    /// to broadcast, and the inner-dim match is a runtime abort
    /// precondition, so any program that runs satisfies it).
    fn matmul_size(&self, a: SizeInfo, b: SizeInfo) -> Option<SizeInfo> {
        use SizeInfo::{Numel, Shape};
        match (a, b) {
            (Shape(sa), Shape(sb)) if sa.len() >= 2 && sb.len() >= 2 => {
                let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
                let (k2, n) = (sb[sb.len() - 2], sb[sb.len() - 1]);
                if k != k2 {
                    return None;
                }
                let mut out = numpy_broadcast(&sa[..sa.len() - 2], &sb[..sb.len() - 2])?;
                out.push(m);
                out.push(n);
                Some(Shape(out))
            }
            // a: [batch.., m, k] with only numel known; b: [k, n].
            // numel_out = prod(batch) * m * n = na / k * n.
            (Numel(na), Shape(sb)) if sb.len() == 2 => {
                let (k, n) = (sb[0] as u64, sb[1] as u64);
                if k == 0 || na % k != 0 {
                    None
                } else {
                    (na / k).checked_mul(n).map(Numel)
                }
            }
            // a: [m, k]; b: [batch.., k, n] with only numel known.
            // numel_out = prod(batch) * m * n = nb * m / k  (n cancels).
            (Shape(sa), Numel(nb)) if sa.len() == 2 => {
                let (m, k) = (sa[0] as u64, sa[1] as u64);
                let prod = nb.checked_mul(m)?;
                if k == 0 || prod % k != 0 {
                    None
                } else {
                    Some(Numel(prod / k))
                }
            }
            _ => None,
        }
    }

    /// One op's result size, or None to stop. The catch-all refuses
    /// anything untraced (Concat's per-input extents are invisible behind
    /// its list var, Gather is device-divergent at dim 0, ScatterAdd's
    /// leading dim is `max(index)+1` — data-dependent, the conv/pool
    /// family has no transformer callers to validate against).
    fn derive_size(&self, op: &'a crate::wengert::WengertOp) -> Option<SizeInfo> {
        use SizeInfo::{Numel, Shape};
        match &op.op {
            // Shape-preserving: unary elementwise; softmax family; the
            // norms (result is x-shaped, gamma/beta are separate inputs);
            // Dropout; RoPE/RoPEInverse (clone / same-shape rotation); the
            // AD-internal Reshape/Broadcast/Split whose lowerings are
            // `nsl_tensor_clone`; Condition (mask stamped with lhs shape).
            PrimalOp::Relu
            | PrimalOp::Sigmoid
            | PrimalOp::Tanh
            | PrimalOp::Gelu
            | PrimalOp::Silu
            | PrimalOp::Exp
            | PrimalOp::Log
            | PrimalOp::Sqrt
            | PrimalOp::Abs
            | PrimalOp::Neg
            | PrimalOp::Clamp { .. }
            | PrimalOp::Softmax { .. }
            | PrimalOp::LogSoftmax { .. }
            | PrimalOp::LayerNorm { .. }
            | PrimalOp::RMSNorm { .. }
            | PrimalOp::BatchNorm { .. }
            | PrimalOp::Dropout { .. }
            | PrimalOp::DropoutMask { .. }
            | PrimalOp::RoPE { .. }
            | PrimalOp::RoPEInverse { .. }
            | PrimalOp::Reshape { .. }
            | PrimalOp::Broadcast
            | PrimalOp::Split { .. }
            | PrimalOp::Condition(_) => self.input_info(op, 0),

            PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div => {
                self.binary_size(self.input_info(op, 0)?, self.input_info(op, 1)?)
            }

            PrimalOp::Matmul => {
                self.matmul_size(self.input_info(op, 0)?, self.input_info(op, 1)?)
            }

            PrimalOp::Transpose { dim0, dim1 } => match self.input_info(op, 0)? {
                Shape(mut s) => {
                    if *dim0 < s.len() && *dim1 < s.len() {
                        s.swap(*dim0, *dim1);
                        Some(Shape(s))
                    } else {
                        None
                    }
                }
                n @ Numel(_) => Some(n),
            },

            PrimalOp::Sum { dim } | PrimalOp::Mean { dim } => match dim {
                // Global reduction: 1 element (0-dim on CPU, [1] on GPU —
                // the count is what sizing needs and it is 1 on both).
                None => Some(Shape(vec![1])),
                // keepdim=0 everywhere on this tape: the dim is REMOVED.
                Some(d) => match self.input_info(op, 0)? {
                    Shape(s) => {
                        let nd = s.len() as i64;
                        let dd = if *d < 0 { nd + d } else { *d };
                        if (0..nd).contains(&dd) {
                            let mut out = s;
                            out.remove(dd as usize);
                            Some(Shape(out))
                        } else {
                            None
                        }
                    }
                    Numel(_) => None,
                },
            },

            PrimalOp::CrossEntropyLoss
            | PrimalOp::MSELoss
            | PrimalOp::L1Loss
            | PrimalOp::FusedLinearCe { .. }
            | PrimalOp::FusedKlCe { .. } => Some(Shape(vec![1])),

            // The fused-CE backward bakes its shapes into the op itself.
            PrimalOp::FusedLinearCeBackwardExtract {
                component,
                vocab_size,
                hidden_size,
                batch_size,
                seq_len,
                x_rank3,
                ..
            } => {
                let (v, h) = (*vocab_size as i64, *hidden_size as i64);
                let (b, s) = (*batch_size as i64, *seq_len as i64);
                match component {
                    0 if *x_rank3 => Some(Shape(vec![b, s, h])),
                    0 => Some(Shape(vec![b.checked_mul(s)?, h])),
                    1 => Some(Shape(vec![v, h])),
                    2 => Some(Shape(vec![v])),
                    _ => None,
                }
            }
            PrimalOp::FusedKlCeBackwardExtract {
                component,
                vocab_size,
                student_hidden,
                batch_size,
                seq_len,
                ..
            } => {
                let (v, h) = (*vocab_size as i64, *student_hidden as i64);
                let (b, s) = (*batch_size as i64, *seq_len as i64);
                match component {
                    // dx_s is ALWAYS rank-2 [B*S, H_s] — no x_rank3 here.
                    0 => Some(Shape(vec![b.checked_mul(s)?, h])),
                    1 => Some(Shape(vec![v, h])),
                    2 => Some(Shape(vec![v])),
                    _ => None,
                }
            }

            // out = q's leading dims + v's last dim, valid only in the
            // regime every lowering path enforces (k exactly q-shaped, v
            // agreeing in the leading dims); anything looser hits matmul
            // batch-broadcast in the decomposed path and is refused here.
            PrimalOp::ScaledDotProductAttention { .. }
            | PrimalOp::ScaledDotProductAttentionPacked => {
                match (
                    self.input_info(op, 0)?,
                    self.input_info(op, 1)?,
                    self.input_info(op, 2)?,
                ) {
                    (Shape(sq), Shape(sk), Shape(sv))
                        if sq.len() >= 2
                            && sk == sq
                            && sv.len() == sq.len()
                            && sv[..sv.len() - 1] == sq[..sq.len() - 1] =>
                    {
                        let mut out = sq;
                        let last = out.len() - 1;
                        out[last] = sv[sv.len() - 1];
                        Some(Shape(out))
                    }
                    _ => None,
                }
            }

            // inputs = [dout, q, k, v, fwd_out, ...]: dQ is q-shaped, dK
            // k-shaped, dV v-shaped.
            PrimalOp::FlashAttentionBackwardExtract { component, .. }
            | PrimalOp::FlashAttentionBackwardExtractPacked { component, .. } => {
                self.input_info(op, 1 + *component as usize)
            }

            // Placeholder result ([1] zeros, never consumed as data).
            PrimalOp::FusedCshaBackward { .. } => Some(Shape(vec![1])),

            PrimalOp::Slice { dim, start, end, .. } => match self.input_info(op, 0)? {
                Shape(s) => {
                    let nd = s.len() as i64;
                    let dd = if *dim < 0 { nd + dim } else { *dim };
                    if !(0..nd).contains(&dd) {
                        return None;
                    }
                    let size = s[dd as usize];
                    // Exact replica of the runtime clamp (`shape_ops.rs`).
                    let sc = if *start < 0 {
                        (size + start).max(0)
                    } else {
                        (*start).min(size)
                    };
                    let ec = if *end < 0 {
                        (size + end).max(0)
                    } else {
                        (*end).min(size)
                    };
                    let extent = (ec - sc).max(0);
                    if extent <= 0 {
                        return None; // 0-element result: legal, never sized
                    }
                    let mut out = s;
                    out[dd as usize] = extent;
                    Some(Shape(out))
                }
                Numel(_) => None,
            },

            PrimalOp::PadZero { dim, pad_before, pad_after } => {
                match self.input_info(op, 0)? {
                    Shape(s) => {
                        let nd = s.len() as i64;
                        let dd = if *dim < 0 { nd + dim } else { *dim };
                        if !(0..nd).contains(&dd) || *pad_before < 0 || *pad_after < 0 {
                            return None;
                        }
                        let mut out = s;
                        out[dd as usize] = out[dd as usize]
                            .checked_add(*pad_before)?
                            .checked_add(*pad_after)?;
                        Some(Shape(out))
                    }
                    Numel(_) => None,
                }
            }

            // 2-D spatial upsample: BOTH trailing dims scale by `kernel`.
            PrimalOp::Repeat { kernel } => match self.input_info(op, 0)? {
                Shape(mut s) if s.len() >= 2 && *kernel > 0 => {
                    let n = s.len();
                    s[n - 2] = s[n - 2].checked_mul(*kernel as i64)?;
                    s[n - 1] = s[n - 1].checked_mul(*kernel as i64)?;
                    Some(Shape(s))
                }
                _ => None,
            },

            // inputs = [table, indices]; indices must be rank-1 at runtime
            // (anything else aborts), so out = [numel(indices), cols].
            PrimalOp::Embedding => {
                let table = match self.input_info(op, 0)? {
                    Shape(t) if t.len() == 2 => t,
                    _ => return None,
                };
                let n = self.input_info(op, 1)?.numel()?;
                if n > i64::MAX as u64 {
                    return None;
                }
                Some(Shape(vec![n as i64, table[1]]))
            }

            // Declared shape is true_val's, but the BUFFER is max(len)
            // over all three operands (`nsl_tensor_where`) — so equal
            // shapes are the only case where shape and buffer agree;
            // otherwise only the count is claimable, as the max.
            PrimalOp::Select => {
                let (c, t, f) = (
                    self.input_info(op, 0)?,
                    self.input_info(op, 1)?,
                    self.input_info(op, 2)?,
                );
                match (&c, &t, &f) {
                    (Shape(sc), Shape(st), Shape(sf)) if sc == st && st == sf => Some(t),
                    _ => {
                        let m = c.numel()?.max(t.numel()?).max(f.numel()?);
                        Some(Numel(m))
                    }
                }
            }

            // y = x @ W (+ adapter terms that never change the shape):
            // [..., k] x [k, n] -> [..., n].
            PrimalOp::FusedGatedLoraMatmul { .. }
            | PrimalOp::FusedLoraMatmul { .. }
            | PrimalOp::FusedIa3Matmul { .. } => {
                let w = match self.input_info(op, 1)? {
                    Shape(w) if w.len() == 2 => w,
                    _ => return None,
                };
                match self.input_info(op, 0)? {
                    Shape(x) if !x.is_empty() && x[x.len() - 1] == w[0] => {
                        let mut out = x;
                        let last = out.len() - 1;
                        out[last] = w[1];
                        Some(Shape(out))
                    }
                    Numel(nx) => {
                        let (k, n) = (w[0] as u64, w[1] as u64);
                        if k == 0 || nx % k != 0 {
                            None
                        } else {
                            (nx / k).checked_mul(n).map(Numel)
                        }
                    }
                    _ => None,
                }
            }

            PrimalOp::Passthrough(name) => match name.as_str() {
                // Shape-preserving passthroughs.
                "contiguous" | "cos" | "sin" | "rotate_half" | "rotate_half_neg"
                | "zeros_like" | "ones_like" | "silu_backward" | "gelu_backward"
                | "sigmoid_backward" | "tanh_backward" | "swiglu_gate_backward" => {
                    self.input_info(op, 0)
                }
                n if n.starts_with("ccr_cast_") => self.input_info(op, 0),
                // Backward FFIs whose result is a NON-FIRST input's shape.
                n if n.starts_with("rmsnorm_dx_backward") => self.input_info(op, 1),
                n if n.starts_with("rmsnorm_dgamma_backward") => self.input_info(op, 2),
                "embedding_backward" => self.input_info(op, 2),
                "cross_entropy_backward" | "mse_backward" | "l1_backward" => {
                    self.input_info(op, 1)
                }
                "reduce_to_shape" => self.input_info(op, 1),
                // Real reshape: the runtime ABORTS unless the target list's
                // product equals the input count, so a static target IS the
                // result shape for any program that runs — even when the
                // input is unsized (that is what recovers the RoPE
                // cos/sin path, whose table read is unsized but whose
                // `[1, 1, seq, d]` target is folded). A sized input that
                // CONTRADICTS the static target means the static reading
                // is wrong; refuse.
                "reshape" => {
                    if op.inputs.len() != 2 {
                        return None;
                    }
                    let n_in = self.input_info(op, 0).and_then(|i| i.numel());
                    if let Some(dims) = self.static_list_dims(op.inputs[1]) {
                        let n_out = shape_numel(&dims)?;
                        return match n_in {
                            Some(n) if n != n_out => None,
                            _ => Some(Shape(dims)),
                        };
                    }
                    n_in.map(Numel)
                }
                "expand" => {
                    let dims = self.static_list_dims(*op.inputs.get(1)?)?;
                    Some(Shape(dims))
                }
                // Rank changes with a preserved count.
                "squeeze" | "unsqueeze" => {
                    Some(Numel(self.input_info(op, 0)?.numel()?))
                }
                "item" => Some(Shape(vec![])),
                "mean_keepdim_last" | "sum_keepdim_last" => match self.input_info(op, 0)? {
                    Shape(mut s) if !s.is_empty() => {
                        let last = s.len() - 1;
                        s[last] = 1;
                        Some(Shape(s))
                    }
                    _ => None,
                },
                "transpose" => {
                    let d0 = self.const_int(*op.inputs.get(1)?)?;
                    let d1 = self.const_int(*op.inputs.get(2)?)?;
                    match self.input_info(op, 0)? {
                        Shape(mut s) => {
                            let nd = s.len() as i64;
                            let n0 = if d0 < 0 { nd + d0 } else { d0 };
                            let n1 = if d1 < 0 { nd + d1 } else { d1 };
                            if (0..nd).contains(&n0) && (0..nd).contains(&n1) {
                                s.swap(n0 as usize, n1 as usize);
                                Some(Shape(s))
                            } else {
                                None
                            }
                        }
                        n @ Numel(_) => Some(n),
                    }
                }
                "zeros" | "ones" | "full" | "randn" => {
                    let dims = self.static_list_dims(*op.inputs.first()?)?;
                    Some(Shape(dims))
                }
                // Scalar-immediate rewrites preserve their tensor operand's
                // shape (MFU campaign C3).
                n if n.starts_with("mul_scalar_rhs:")
                    || n.starts_with("add_scalar_rhs:")
                    || n.starts_with("div_scalar_rhs:")
                    || n.starts_with("sub_scalar_rhs:") =>
                {
                    self.input_info(op, 0)
                }
                // Fused elementwise chains are DELIBERATELY unsized: a
                // replay-fallback result may not match input 0's shape (a
                // genuinely-reducing rts tail), and an unsized transient is
                // refused arena admission (loud-benign) — never mis-placed.
                n if n.starts_with("fused_ew:") => None,
                // dict_get is seeded from DataLoader facts by the caller,
                // never derived; everything else ("arange", "list",
                // "shape", ...) has a data-dependent or non-tensor result.
                _ => None,
            },

            // Constant lowers to a 0-dim scalar tensor when tensor-typed.
            PrimalOp::Constant(_) => Some(Shape(vec![])),

            // Leaves are seeded, never derived; markers and everything
            // untraced (Concat behind its list var, device-divergent
            // Gather, the data-dependent ScatterAdd, conv/pool,
            // CshaFusedBackwardExtract) derive nothing.
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Stage-2B: admission
// ---------------------------------------------------------------------------

/// Why a transient was refused a planned address.
///
/// Enumerated rather than a bool so `--arena-report` can say which rule is
/// costing coverage. "Nothing was admitted" and "everything was aliasing" are
/// different findings and only one of them is a reason to widen the rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusedBecause {
    /// Forward-region value. Stage-2B is backward temporaries only.
    NotBackward,
    /// No static element count, so no slot size.
    Unsized,
    /// Read by the backward — its address is a gradient-map key.
    SavedForBackward,
    /// Read after the tape (optimizer, callbacks); the tape's last-use is not
    /// its real death.
    EscapesTape,
    /// The producing op returns storage aliased to an input, so a "slot" for
    /// it would be a second name for someone else's memory.
    Aliasing,
    /// Produced by an op whose lowering is not known to make exactly one
    /// device allocation.
    NotSingleAllocation,
    /// The op's mutation-candidate operand DIES at this op, so the runtime's
    /// FBIP arm (`can_mutate_inplace_gpu`: uniquely-owned input -> mutate in
    /// place, return the input pointer) can serve the result with NO
    /// allocation. Measured, not theoretical: every admitted Sqrt/Neg on
    /// coder50m leaked its bind through exactly this arm.
    InPlaceReuse,
    /// A binary whose operand shapes are not proven EQUAL (broadcast), or
    /// whose operand comes from a view (non-contiguous storage): both leave
    /// the GPU flat path, and the broadcast/fallback routes allocate
    /// differently (or on the host) — the single-straight-line-allocation
    /// claim does not hold.
    RuntimePathVaries,
}

/// One admitted transient and the slot it was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub var: VarId,
    /// Dense slot index, also the number of leading guards to skip.
    pub slot_index: u32,
    /// Byte offset of the slot's payload within the packed plan.
    pub offset: u64,
    pub bytes: u64,
}

/// Ops whose `wengert_lower` arm makes exactly ONE device allocation.
///
/// The pin is single-shot and size-exact, so an op that allocates a scratch
/// buffer before its result would either consume the pin with the scratch (if
/// the sizes happened to match) or leave it armed for something later. Both
/// are wrong in ways nothing downstream would notice, so admission is an
/// ALLOWLIST rather than a denylist: an op nobody has checked is refused.
///
/// Kept to the elementwise and reduction family on purpose. Matmul goes
/// through cuBLAS with its own workspace, attention allocates several buffers,
/// and the fused kernels allocate loss + LSE — none belong here.
fn is_single_allocation_op(op: &PrimalOp) -> bool {
    matches!(
        op,
        PrimalOp::Add
            | PrimalOp::Sub
            | PrimalOp::Mul
            | PrimalOp::Div
            | PrimalOp::Neg
            | PrimalOp::Relu
            | PrimalOp::Gelu
            | PrimalOp::Silu
            | PrimalOp::Sigmoid
            | PrimalOp::Tanh
            | PrimalOp::Exp
            | PrimalOp::Log
            | PrimalOp::Sqrt
    )
}

/// True when `op`'s result may alias an input's storage.
///
/// Deliberately the UNION of two predicates that disagree. This module's
/// `op_allocates` argues `Reshape`/`Broadcast`/`Slice`/`Select` genuinely copy
/// on this tape and so must be COUNTED (undercounting a byte figure is the
/// unsafe direction for a measurement), while `wengert::is_view_producing_op`
/// lists the user-facing `reshape`/`contiguous`/`expand`/`squeeze`/`unsqueeze`
/// passthroughs as runtime view ops.
///
/// For a byte count the disagreement is academic. For PLACEMENT it is not:
/// giving a fixed address to something that turns out to alias an input hands
/// two logical values one buffer. So placement takes the union and refuses
/// anything either predicate calls a view.
fn may_alias(op: &PrimalOp) -> bool {
    crate::wengert::is_view_producing_op(op)
        || matches!(
            op,
            PrimalOp::Transpose { .. }
                | PrimalOp::Reshape { .. }
                | PrimalOp::Broadcast
                | PrimalOp::Slice { .. }
        )
}

/// Stage-2B admission: which transients may be given a fixed address.
///
/// Conservative by construction — see [`RefusedBecause`] for every rule. The
/// returned placements are packed by [`pack`]; refusals are returned alongside
/// so the report can say what coverage was lost and to which rule.
pub fn admit(
    plan: &ArenaPlan,
    forward: &WengertList,
    adjoint: &WengertList,
    tape_escaping: &std::collections::HashSet<VarId>,
    sizes: &HashMap<VarId, SizeInfo>,
) -> (Vec<VarId>, Vec<(VarId, RefusedBecause)>) {
    // Concatenated last-use, same construction as `analyze` — the FBIP
    // in-place arm fires exactly when the mutated operand has no later
    // reader, which on the tape means its last use IS this op.
    let fwd_n = forward.ops.len();
    let mut last_use: HashMap<VarId, u32> = HashMap::new();
    for (base, list) in [(0usize, forward), (fwd_n, adjoint)] {
        for (i, op) in list.ops.iter().enumerate() {
            for &input in &op.inputs {
                last_use.insert(input, (base + i) as u32);
            }
        }
    }
    // Producers across BOTH lists, for the view check on binary operands.
    let mut producer: HashMap<VarId, &crate::wengert::WengertOp> = HashMap::new();
    for list in [forward, adjoint] {
        for op in &list.ops {
            producer.entry(op.result).or_insert(op);
        }
    }
    let is_view_output = |v: VarId| -> bool {
        producer.get(&v).is_some_and(|o| {
            crate::wengert::is_view_producing_op(&o.op)
                || matches!(o.op, PrimalOp::Transpose { .. })
        })
    };

    let mut admitted = Vec::new();
    let mut refused = Vec::new();
    for t in &plan.transients {
        let why = if t.region != Region::Backward {
            Some(RefusedBecause::NotBackward)
        } else if t.elems.is_none() {
            Some(RefusedBecause::Unsized)
        } else if t.saved_for_backward {
            Some(RefusedBecause::SavedForBackward)
        } else if tape_escaping.contains(&t.var) {
            Some(RefusedBecause::EscapesTape)
        } else {
            match adjoint.ops.iter().find(|o| o.result == t.var) {
                Some(op) if may_alias(&op.op) => Some(RefusedBecause::Aliasing),
                Some(op) if !is_single_allocation_op(&op.op) => {
                    Some(RefusedBecause::NotSingleAllocation)
                }
                // A transient with no producing op in the adjoint is not one
                // this pass can reason about at all.
                None => Some(RefusedBecause::NotSingleAllocation),
                Some(op) => {
                    let dies_here = |v: VarId| last_use.get(&v) == Some(&t.birth);
                    let is_binary = matches!(
                        op.op,
                        PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div
                    );
                    if is_binary {
                        // The GPU flat path — the only arm whose single
                        // exact-size allocation the pin can match — requires
                        // equal element counts and contiguous operands. A
                        // broadcast or a view operand takes a different
                        // route (possibly through the host).
                        let shapes_equal = match (
                            op.inputs.first().and_then(|v| sizes.get(v)),
                            op.inputs.get(1).and_then(|v| sizes.get(v)),
                        ) {
                            (Some(SizeInfo::Shape(a)), Some(SizeInfo::Shape(b))) => a == b,
                            _ => false,
                        };
                        let any_view = op.inputs.iter().take(2).any(|&v| is_view_output(v));
                        if !shapes_equal || any_view {
                            Some(RefusedBecause::RuntimePathVaries)
                        } else {
                            None
                        }
                    } else {
                        // Unary elementwise: two runtime arms to model.
                        // (a) `can_mutate_inplace_gpu` mutates a dying,
                        // uniquely-owned input in place — no allocation;
                        // a dying operand is the static over-approximation.
                        // (b) a NON-CONTIGUOUS operand is materialized
                        // first (`nsl_tensor_contiguous`), and because
                        // unaries preserve numel that clone is EXACTLY
                        // output-sized — it would consume the pin and put
                        // an op-local scratch at the planned stable
                        // address with every reconciliation counter still
                        // green. Same view refusal as the binary arm.
                        if op.inputs.first().copied().is_some_and(dies_here) {
                            Some(RefusedBecause::InPlaceReuse)
                        } else if op.inputs.first().copied().is_some_and(is_view_output) {
                            Some(RefusedBecause::RuntimePathVaries)
                        } else {
                            None
                        }
                    }
                }
            }
        };
        match why {
            Some(r) => refused.push((t.var, r)),
            None => admitted.push(t.var),
        }
    }
    (admitted, refused)
}

/// Pack the admitted transients: every placement gets its OWN disjoint
/// region, laid out sequentially in dense-index order.
///
/// Deliberately NOT the M36 BFD time-sharing engine. BFD assigns two
/// disjoint-lifetime placements the SAME byte offset, but the runtime
/// layout is `payload = base + offset + REDZONE * (idx + 1)` with `idx`
/// the dense placement index — under shared offsets, payload regions from
/// DIFFERENT slots bleed into each other by `REDZONE * Δidx` bytes while
/// both are live. That is silent device-memory corruption, and it was
/// caught only by the planned-vs-unplanned byte-identity bisection (slot 8
/// on coder50m — losses diverged at step 1). Time-sharing can return later
/// with a layout model that carries per-slot guard geometry; the point of
/// placement is stable addresses, not byte savings, and the sequential
/// layout's cost on the real model is ~11 MiB.
///
/// With offsets assigned in dense order, payload `k` ends at
/// `offset_k + REDZONE*(k+1) + bytes_k = offset_{k+1} + REDZONE*(k+1)`,
/// which is exactly one red zone before payload `k+1` begins — every pair
/// of adjacent payloads is separated by REDZONE poison bytes, matching
/// what `nsl_arena_init` sizes.
pub fn pack(plan: &ArenaPlan, admitted: &[VarId]) -> (Vec<Placement>, u64) {
    let wanted: std::collections::HashSet<VarId> = admitted.iter().copied().collect();
    let chosen: Vec<&Transient> = plan
        .transients
        .iter()
        .filter(|t| wanted.contains(&t.var))
        .collect();
    if chosen.is_empty() {
        return (Vec::new(), 0);
    }
    let mut placements = Vec::with_capacity(chosen.len());
    let mut offset = 0u64;
    for (dense, t) in chosen.iter().enumerate() {
        let exact = t.elems.expect("admission required a size") * plan.elem_bytes;
        // The PIN carries the EXACT byte count — the runtime allocation
        // asks for `numel * 4` unaligned, and the pin's exact-match rule
        // would otherwise never fire for any tensor whose bytes are not a
        // 256 multiple (e.g. every [50257]-vocab gradient). The LAYOUT
        // still strides by the aligned size, so every payload keeps its
        // 256-byte alignment and the slack between `exact` and the next
        // guard stays poison — an overrun of even one byte past the exact
        // payload lands in the checked window.
        placements.push(Placement {
            var: t.var,
            slot_index: dense as u32,
            offset,
            bytes: exact,
        });
        offset += align_up(exact, SLAB_ALIGNMENT);
    }
    (placements, offset)
}

/// Occupancy interval of a transient, half-open: `[birth, last_use + 1)`. The
/// `+ 1` is load-bearing and memory-correct, not padding: a value's last use is,
/// by definition, an *input* to the op at that index, and that same op *writes*
/// whatever it produces there — so the dying value and the newborn value both
/// occupy device memory during that op and cannot share storage. Encoding the
/// end as `last_use + 1` makes both the concurrency sweep here and M36's
/// half-open `intervals_overlap` (used by `plan_slab`) count that coexistence
/// identically, so the two paths never disagree on adjacent intervals. A
/// never-read value occupies exactly one point, `[birth, birth + 1)`.
fn occupancy_end(t: &Transient) -> u32 {
    t.death.max(t.birth).saturating_add(1)
}

/// Sweep the occupancy intervals and return (max simultaneously-live, first
/// point at which that peak occurs). O(n log n) via a +1/-1 endpoint event list.
fn sweep_max_concurrency(transients: &[Transient]) -> (usize, u32) {
    if transients.is_empty() {
        return (0, 0);
    }
    let mut events: Vec<(u32, i32)> = Vec::with_capacity(transients.len() * 2);
    for t in transients {
        events.push((t.birth, 1)); // start
        events.push((occupancy_end(t), -1)); // one past the last-use op
    }
    // Standard half-open sweep: at equal points, apply ends (-1) before starts
    // (+1) so two genuinely-adjacent intervals `[.,p)` and `[p,.)` are not
    // double-counted. The coexistence-during-the-consuming-op case is already
    // captured by the `+1` in `occupancy_end`, which pushes the end past the
    // shared point — so it registers as a real overlap here, not a tie.
    events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut live: i32 = 0;
    let mut best: i32 = 0;
    let mut best_at: u32 = 0;
    for (point, delta) in events {
        live += delta;
        if live > best {
            best = live;
            best_at = point;
        }
    }
    (best.max(0) as usize, best_at)
}

impl ArenaPlan {
    /// Total planned arena bytes on the quantified path (0 when unquantified).
    pub fn arena_bytes(&self) -> u64 {
        self.slab.as_ref().map(|s| s.total_bytes).unwrap_or(0)
    }

    /// Naive bytes (Σ of all transient sizes) on the quantified path.
    pub fn naive_bytes(&self) -> u64 {
        self.slab.as_ref().map(|s| s.naive_total).unwrap_or(0)
    }

    /// Byte reduction factor (naive / arena) on the quantified path; 1.0 when
    /// unquantified or nothing to gain.
    pub fn byte_reduction(&self) -> f64 {
        match &self.slab {
            Some(s) if s.total_bytes > 0 => s.naive_total as f64 / s.total_bytes as f64,
            _ => 1.0,
        }
    }

    /// A human-readable arena projection for `--memory-report`.
    pub fn render_report(&self, indent: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{indent}Transient arena (backward+forward tape, Milestone C·p2):\n\
             {indent}  {} transient tensor(s) over the fwd+bwd tape\n\
             {indent}  peak concurrency: {} live at once (arena needs {} slot(s))\n",
            self.total_alloc_events, self.max_concurrency, self.max_concurrency,
        ));
        if let Some(slab) = &self.slab {
            let sized = self.total_alloc_events - self.unsized_count;
            out.push_str(&format!(
                "{indent}  arena bytes: {} vs naive {} ({:.2}x smaller, {:.1}% saved), \
                 {} slot(s), frag {:.1}%\n",
                crate::memory_planner::format_bytes(slab.total_bytes),
                crate::memory_planner::format_bytes(slab.naive_total),
                self.byte_reduction(),
                slab.savings_fraction() * 100.0,
                slab.slots.len(),
                slab.fragmentation_ratio() * 100.0,
            ));
            if self.unsized_count > 0 {
                out.push_str(&format!(
                    "{indent}  partial quantification: bytes cover {sized} of {} \
                     transients ({} unsized stay on the caching allocator) — \
                     the byte figures are a LOWER BOUND on the full arena\n",
                    self.total_alloc_events, self.unsized_count,
                ));
            }
            if self.escaping_count > 0 {
                out.push_str(&format!(
                    "{indent}  {} tape-escaping value(s) (gradient accumulators \
                     / loss) pinned live to tape end — resident, never \
                     slot-shared\n",
                    self.escaping_count,
                ));
            }
        } else {
            // Unquantified: report the concurrency win in alloc-count terms.
            let churn_reduction = if self.max_concurrency > 0 {
                self.total_alloc_events as f64 / self.max_concurrency as f64
            } else {
                1.0
            };
            out.push_str(&format!(
                "{indent}  shapes runtime-valued: {} individual alloc/free round-trips collapse \
                 to {} stable arena slot(s) ({:.1}x fewer live objects at peak)\n\
                 {indent}  (byte sizing deferred to Stage-2 shape wiring; see \
                 docs/research/transient-memory-arena.md)\n",
                self.total_alloc_events, self.max_concurrency, churn_reduction,
            ));
        }
        out
    }

    /// Padding-aligned arena bytes, exposed for the VRAM-budget path.
    pub fn aligned_arena_bytes(&self) -> u64 {
        align_up(self.arena_bytes(), SLAB_ALIGNMENT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};

    fn op(id: u32, result: VarId, primal: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result,
            op: primal,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    fn saved(mut o: WengertOp) -> WengertOp {
        o.saved_for_backward = true;
        o
    }

    fn list(ops: Vec<WengertOp>) -> WengertList {
        WengertList {
            ops,
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    fn none_elems() -> impl Fn(VarId) -> Option<u64> {
        |_| None
    }

    #[test]
    fn empty_is_safe() {
        let plan = analyze(&list(vec![]), &list(vec![]), &none_elems(), &Default::default(), 4);
        assert_eq!(plan.max_concurrency, 0);
        assert!(plan.transients.is_empty());
        assert!(plan.slab.is_none());
        assert_eq!(plan.byte_reduction(), 1.0);
    }

    #[test]
    fn leaves_and_transpose_excluded_reshape_and_select_allocate() {
        // x=input, w=param, c=const (leaves, never counted).
        // t=transpose(x): zero-copy stride view (new_view_i64) -> excluded.
        // r=reshape(x): lowers to nsl_tensor_clone -> ALLOCATES.
        // s=select(x): lowers to nsl_tensor_where -> ALLOCATES.
        // z=relu(x): allocates.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("w".into()), vec![]),
            op(2, 2, PrimalOp::Constant(1.0), vec![]),
            op(3, 3, PrimalOp::Transpose { dim0: 0, dim1: 1 }, vec![0]),
            op(4, 4, PrimalOp::Reshape { target_ndim: 2 }, vec![0]),
            op(5, 5, PrimalOp::Select, vec![0]),
            op(6, 6, PrimalOp::Relu, vec![0]),
        ]);
        let plan = analyze(&fwd, &list(vec![]), &none_elems(), &Default::default(), 4);
        let mut vars: Vec<VarId> = plan.transients.iter().map(|t| t.var).collect();
        vars.sort();
        // Reshape (4), Select (5), Relu (6) all materialize; transpose (3) and
        // the leaves (0,1,2) do not.
        assert_eq!(vars, vec![4, 5, 6], "reshape/select/relu allocate; transpose is a view");
    }

    #[test]
    fn saved_forward_value_spans_into_backward() {
        // Forward: h = relu(x) [saved]. Backward reads h late.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            saved(op(1, 1, PrimalOp::Relu, vec![0])), // h, saved for backward
            op(2, 2, PrimalOp::Matmul, vec![1]),      // uses h in forward
        ]);
        // Backward tape (indices 3,4 on the concatenated timeline).
        let bwd = list(vec![
            op(10, 100, PrimalOp::Constant(1.0), vec![]), // seed
            op(11, 101, PrimalOp::Mul, vec![1, 100]),     // reads h (var 1) in backward
        ]);
        let plan = analyze(&fwd, &bwd, &none_elems(), &Default::default(), 4);
        let h = plan.transients.iter().find(|t| t.var == 1).unwrap();
        assert!(h.saved_for_backward);
        // Born at index 1, last used at concatenated index 4 (fwd_n=3 + 1).
        assert_eq!(h.birth, 1);
        assert_eq!(h.death, 4);
        assert!(h.live_range() >= 3);
    }

    #[test]
    fn free_tensor_marker_bounds_liveness() {
        // h produced at 1, freed at 3 (before its would-be later reads).
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Relu, vec![0]),
            op(2, 2, PrimalOp::Relu, vec![1]), // reads h at 2
            op(3, 3, PrimalOp::FreeTensor, vec![1]), // frees h at 3
        ]);
        let plan = analyze(&fwd, &list(vec![]), &none_elems(), &Default::default(), 4);
        let h = plan.transients.iter().find(|t| t.var == 1).unwrap();
        // FreeTensor counts as the last use -> death 3, not extended past it.
        assert_eq!(h.death, 3);
        // FreeTensor is not itself a transient (no tensor result).
        assert!(plan.transients.iter().all(|t| t.var != 3));
    }

    #[test]
    fn concurrency_is_interval_graph_optimum() {
        // Three relus with staggered, overlapping lifetimes:
        //   a: [0..3), b: [1..4), c: [2..5)  -> at point 2, all three live.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Relu, vec![0]), // a born 1
            op(2, 2, PrimalOp::Relu, vec![0]), // b born 2
            op(3, 3, PrimalOp::Relu, vec![0]), // c born 3
            op(4, 4, PrimalOp::Add, vec![1, 2]), // uses a,b at 4
            op(5, 5, PrimalOp::Add, vec![2, 3]), // uses b,c at 5
            op(6, 6, PrimalOp::Add, vec![3, 4]), // uses c at 6
        ]);
        let plan = analyze(&fwd, &list(vec![]), &none_elems(), &Default::default(), 4);
        // a:[1..4) b:[2..5) c:[3..6) plus the Add results 4,5,6.
        // Peak overlap of the relu chain is 3 (a,b,c) around point 3-4.
        assert!(plan.max_concurrency >= 3, "got {}", plan.max_concurrency);
    }

    #[test]
    fn quantified_path_reuses_bfd_and_reports_bytes() {
        // Two non-overlapping transients of equal size share one slot -> 2x.
        // a: relu(x) used at 2; b: relu(a) used at 4 (a dead by then).
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Relu, vec![0]), // a born 1
            op(2, 2, PrimalOp::Relu, vec![1]), // b born 2, last use of a
            op(3, 3, PrimalOp::Relu, vec![2]), // c born 3, last use of b
            op(4, 4, PrimalOp::Neg, vec![3]),  // last use of c
        ]);
        // All same element count -> quantified path fires.
        let elems = |_v: VarId| Some(256u64);
        let plan = analyze(&fwd, &list(vec![]), &elems, &Default::default(), 4);
        let slab = plan.slab.as_ref().expect("all shapes known -> quantified");
        // The chain a->b->c->d is a path graph under the occupancy model (each
        // value coexists with its immediate consumer during the consuming op),
        // so its chromatic number is 2. With no saved tensors, the size-agnostic
        // sweep and the byte-level BFD MUST agree on the slot count — the
        // consistency the `last_use + 1` occupancy convention guarantees.
        assert_eq!(plan.max_concurrency, 2);
        assert_eq!(
            plan.max_concurrency,
            slab.slots.len(),
            "sweep and plan_slab must agree on slot count (no saved tensors)"
        );
        // arena packs 2 slots; naive is 4 * 256 * 4.
        assert!(slab.total_bytes < slab.naive_total, "arena must beat naive");
        assert!(plan.byte_reduction() > 1.0);
    }

    #[test]
    fn partial_hints_quantify_the_sized_subset_only() {
        // Stage-2A: one unsized transient must no longer starve the byte
        // plan. Sizes known for vars 1 and 3 only; var 2 stays unsized.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Relu, vec![0]), // sized
            op(2, 2, PrimalOp::Relu, vec![1]), // UNSIZED
            op(3, 3, PrimalOp::Relu, vec![2]), // sized
            op(4, 4, PrimalOp::Neg, vec![3]),
        ]);
        let elems = |v: VarId| (v != 2).then_some(64u64);
        let plan = analyze(&fwd, &list(vec![]), &elems, &Default::default(), 4);
        assert_eq!(plan.unsized_count, 1);
        let slab = plan.slab.as_ref().expect("sized subset must be planned");
        // The sized transients are vars 1, 3 and 4 (Neg allocates too), so
        // the slab's naive total is 3 * 64 * 4 and the dense ids map back
        // to transient positions through `slab_transients`.
        assert_eq!(slab.naive_total, 3 * 64 * 4);
        assert_eq!(plan.slab_transients.len(), 3);
        // Var 1 is dead before var 3; vars 3 and 4 coexist -> two slots.
        assert_eq!(slab.slots.len(), 2);
        let report = plan.render_report("");
        assert!(
            report.contains("LOWER BOUND") && report.contains("1 unsized"),
            "partial quantification must be called out:\n{report}"
        );
        // Concurrency stays computed over ALL transients, sized or not.
        assert_eq!(plan.max_concurrency, 2);
    }

    #[test]
    fn escaping_values_live_to_tape_end_and_never_share() {
        // Two gradient-accumulator-shaped values: born on the tape, never
        // READ by any tape op (the optimizer consumes them afterwards).
        // Pure last-use liveness gives them point intervals — BFD would
        // time-share every gradient in one slot, which is exactly the
        // illegal aliasing the escape pin exists to prevent.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Relu, vec![0]), // g1: escapes, no tape reader
            op(2, 2, PrimalOp::Relu, vec![0]), // t: ordinary transient
            op(3, 3, PrimalOp::Relu, vec![2]), // g2: escapes, no tape reader
        ]);
        let escaping: std::collections::HashSet<VarId> = [1, 3].into_iter().collect();
        let plan = analyze(&fwd, &list(vec![]), &|_| Some(50u64), &escaping, 4);
        assert_eq!(plan.escaping_count, 2);
        let g1 = plan.transients.iter().find(|t| t.var == 1).unwrap();
        let g2 = plan.transients.iter().find(|t| t.var == 3).unwrap();
        assert_eq!(g1.death, 3, "escaping value must be pinned to tape end");
        assert_eq!(g2.death, 3);
        let slab = plan.slab.as_ref().unwrap();
        // g1/g2/t all overlap at the tape tail -> three slots, ZERO packing
        // win (slot alignment padding can push the arena total slightly
        // above raw naive; what matters is that nothing was shared).
        assert_eq!(slab.slots.len(), 3);
        assert!(slab.total_bytes >= slab.naive_total);
        let report = plan.render_report("");
        assert!(
            report.contains("tape-escaping"),
            "escaping pin must be visible in the report:\n{report}"
        );
    }

    #[test]
    fn full_hints_report_carries_no_partial_caveat() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Relu, vec![0]),
            op(2, 2, PrimalOp::Relu, vec![1]),
        ]);
        let plan = analyze(&fwd, &list(vec![]), &|_| Some(8u64), &Default::default(), 4);
        assert_eq!(plan.unsized_count, 0);
        let report = plan.render_report("");
        assert!(
            !report.contains("LOWER BOUND"),
            "fully-sized plan must not claim partiality:\n{report}"
        );
    }

    #[test]
    fn saved_tensors_never_share_a_slot() {
        // Two saved-for-backward tensors with disjoint intervals must NOT be
        // coalesced (gradient-key uniqueness), so the arena keeps 2 slots even
        // though their lifetimes don't overlap.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            saved(op(1, 1, PrimalOp::Relu, vec![0])), // s1, saved
            op(2, 2, PrimalOp::Relu, vec![1]),        // last use of s1
            saved(op(3, 3, PrimalOp::Relu, vec![2])), // s2, saved, disjoint
            op(4, 4, PrimalOp::Neg, vec![3]),         // last use of s2
        ]);
        let elems = |_v: VarId| Some(100u64);
        let plan = analyze(&fwd, &list(vec![]), &elems, &Default::default(), 4);
        let slab = plan.slab.as_ref().unwrap();
        // s1 (var 1) and s2 (var 3) are disjoint in time but both saved ->
        // forced to distinct slots. `assignments` is keyed by TensorAllocId
        // (position in `plan.transients`), so map VarId -> id first.
        let id_of = |var: VarId| {
            plan.transients.iter().position(|t| t.var == var).unwrap() as u32
        };
        let s1_slot = slab.assignments.get(&id_of(1)).map(|(s, _)| *s);
        let s2_slot = slab.assignments.get(&id_of(3)).map(|(s, _)| *s);
        assert!(s1_slot.is_some() && s2_slot.is_some());
        assert_ne!(s1_slot, s2_slot, "saved tensors must not share a slot");
    }

    // -----------------------------------------------------------------
    // Stage-2B admission
    // -----------------------------------------------------------------

    /// The admission set is the whole safety argument, so each rule gets a
    /// case that would be ADMITTED if that one rule were dropped.
    fn admit_of(
        fwd: Vec<WengertOp>,
        adj: Vec<WengertOp>,
        escaping: &[VarId],
    ) -> (Vec<VarId>, Vec<(VarId, RefusedBecause)>) {
        let esc: std::collections::HashSet<VarId> = escaping.iter().copied().collect();
        let forward = list(fwd);
        let adjoint = list(adj);
        let elems = |_v: VarId| Some(64u64);
        let plan = analyze(&forward, &adjoint, &elems, &esc, 4);
        // Every var the tapes mention gets the SAME shape, so the binary
        // equal-shape rule admits; tests for the broadcast/view refusals
        // build their own sizes maps.
        let mut sizes: HashMap<VarId, SizeInfo> = HashMap::new();
        for l in [&forward, &adjoint] {
            for o in &l.ops {
                for v in o.inputs.iter().chain(std::iter::once(&o.result)) {
                    sizes.insert(*v, SizeInfo::Shape(vec![64]));
                }
            }
        }
        admit(&plan, &forward, &adjoint, &esc, &sizes)
    }

    #[test]
    fn a_plain_backward_elementwise_temporary_is_admitted() {
        // Var 1 is read AGAIN after the Relu, so the runtime cannot mutate
        // it in place and the result genuinely allocates.
        let (ok, _) = admit_of(
            vec![],
            vec![
                op(0, 10, PrimalOp::Relu, vec![1]),
                op(1, 11, PrimalOp::Gelu, vec![1]),
            ],
            &[],
        );
        assert_eq!(ok, vec![10], "the one case the whole stage exists for");
    }

    /// FBIP: a dying, uniquely-owned input is mutated IN PLACE by the
    /// runtime (`can_mutate_inplace_gpu`) — no allocation ever reaches the
    /// pin. Dropping this rule admitted 34 Sqrt ops on coder50m and every
    /// one leaked its bind.
    #[test]
    fn an_in_place_candidate_is_refused() {
        let (ok, no) = admit_of(vec![], vec![op(0, 10, PrimalOp::Relu, vec![1])], &[]);
        assert!(ok.is_empty());
        assert_eq!(no, vec![(10, RefusedBecause::InPlaceReuse)]);
    }

    /// A broadcast (unequal shapes) or a view operand leaves the GPU flat
    /// path; those routes do not make one straight-line allocation of the
    /// result's size.
    #[test]
    fn broadcast_and_view_operand_binaries_are_refused() {
        // Unequal shapes -> broadcast.
        let forward = list(vec![]);
        let adjoint = list(vec![
            op(0, 10, PrimalOp::Add, vec![1, 2]),
            op(1, 11, PrimalOp::Gelu, vec![1]),
            op(2, 12, PrimalOp::Gelu, vec![2]),
        ]);
        let elems = |_v: VarId| Some(64u64);
        let plan = analyze(&forward, &adjoint, &elems, &Default::default(), 4);
        let mut sizes: HashMap<VarId, SizeInfo> = HashMap::new();
        sizes.insert(1, SizeInfo::Shape(vec![64]));
        sizes.insert(2, SizeInfo::Shape(vec![8, 64]));
        let (ok, no) = admit(&plan, &forward, &adjoint, &Default::default(), &sizes);
        assert!(!ok.contains(&10));
        assert_eq!(
            no.iter().find(|(v, _)| *v == 10).map(|(_, r)| *r),
            Some(RefusedBecause::RuntimePathVaries)
        );

        // A transpose-view operand -> non-contiguous -> off the flat path.
        let forward2 = list(vec![
            op(0, 3, PrimalOp::Transpose { dim0: 0, dim1: 1 }, vec![1]),
        ]);
        let adjoint2 = list(vec![
            op(0, 10, PrimalOp::Add, vec![3, 2]),
            op(1, 11, PrimalOp::Gelu, vec![3]),
            op(2, 12, PrimalOp::Gelu, vec![2]),
        ]);
        let plan2 = analyze(&forward2, &adjoint2, &elems, &Default::default(), 4);
        let mut sizes2: HashMap<VarId, SizeInfo> = HashMap::new();
        for v in [1, 2, 3] {
            sizes2.insert(v, SizeInfo::Shape(vec![64]));
        }
        let (ok2, no2) = admit(&plan2, &forward2, &adjoint2, &Default::default(), &sizes2);
        assert!(!ok2.contains(&10));
        assert_eq!(
            no2.iter().find(|(v, _)| *v == 10).map(|(_, r)| *r),
            Some(RefusedBecause::RuntimePathVaries)
        );
    }

    #[test]
    fn forward_temporaries_are_refused() {
        let (ok, no) = admit_of(vec![op(0, 10, PrimalOp::Relu, vec![1])], vec![], &[]);
        assert!(ok.is_empty());
        assert_eq!(no, vec![(10, RefusedBecause::NotBackward)]);
    }

    #[test]
    fn an_unsized_temporary_is_refused() {
        // Same tape as the admitted case, but with no static element count:
        // no size, no slot.
        let adjoint = list(vec![op(0, 10, PrimalOp::Relu, vec![1])]);
        let forward = list(vec![]);
        let plan = analyze(&forward, &adjoint, &none_elems(), &Default::default(), 4);
        let (ok, no) = admit(&plan, &forward, &adjoint, &Default::default(), &HashMap::new());
        assert!(ok.is_empty());
        assert_eq!(no, vec![(10, RefusedBecause::Unsized)]);
    }

    #[test]
    fn a_saved_temporary_is_refused() {
        let (ok, no) = admit_of(vec![], vec![saved(op(0, 10, PrimalOp::Relu, vec![1]))], &[]);
        assert!(ok.is_empty(), "a saved tensor's ADDRESS is a gradient-map key");
        assert_eq!(no, vec![(10, RefusedBecause::SavedForBackward)]);
    }

    #[test]
    fn a_tape_escaping_temporary_is_refused() {
        let (ok, no) = admit_of(vec![], vec![op(0, 10, PrimalOp::Relu, vec![1])], &[10]);
        assert!(ok.is_empty(), "the optimizer reads this after the tape ends");
        assert_eq!(no, vec![(10, RefusedBecause::EscapesTape)]);
    }

    /// Aliasing is the rule that would corrupt memory rather than merely
    /// mis-measure, so it takes the UNION of the two view predicates this
    /// crate carries — which disagree about reshape.
    #[test]
    fn aliasing_producers_are_refused_under_either_predicate() {
        for o in [
            op(0, 10, PrimalOp::Passthrough("reshape".into()), vec![1]),
            op(0, 10, PrimalOp::Passthrough("contiguous".into()), vec![1]),
            op(0, 10, PrimalOp::Passthrough("expand".into()), vec![1]),
            op(0, 10, PrimalOp::Broadcast, vec![1]),
        ] {
            let name = format!("{:?}", o.op);
            let (ok, no) = admit_of(vec![], vec![o], &[]);
            assert!(ok.is_empty(), "{name} may alias its input and must be refused");
            assert_eq!(no.first().map(|(_, r)| *r), Some(RefusedBecause::Aliasing),
                       "{name}");
        }
    }

    /// The allowlist is what keeps the size-exact pin honest: an op that
    /// allocates a scratch before its result would hand the pin to the wrong
    /// buffer.
    #[test]
    fn multi_allocation_ops_are_refused() {
        for o in [
            op(0, 10, PrimalOp::Matmul, vec![1, 2]),
            op(0, 10, PrimalOp::Softmax { dim: -1 }, vec![1]),
        ] {
            let name = format!("{:?}", o.op);
            let (ok, no) = admit_of(vec![], vec![o], &[]);
            assert!(ok.is_empty(), "{name} is not a proven single allocation");
            assert_eq!(no.first().map(|(_, r)| *r),
                       Some(RefusedBecause::NotSingleAllocation), "{name}");
        }
    }

    /// Packing must give distinct addresses to transients that are live at the
    /// same time — the property whose violation is silent corruption rather
    /// than a wrong number.
    #[test]
    fn overlapping_admitted_transients_never_share_an_offset() {
        // t10 is born at 0 and read at 2, so it is live across t11's whole
        // life. They must not land on the same offset.
        let adj = vec![
            op(0, 10, PrimalOp::Relu, vec![1]),
            op(1, 11, PrimalOp::Relu, vec![2]),
            op(2, 12, PrimalOp::Add, vec![10, 11]),
            // Keep 1 and 2 alive past their Relus, or the FBIP rule
            // (correctly) refuses both.
            op(3, 13, PrimalOp::Gelu, vec![1]),
            op(4, 14, PrimalOp::Gelu, vec![2]),
        ];
        let forward = list(vec![]);
        let adjoint = list(adj.clone());
        let elems = |_v: VarId| Some(64u64);
        let plan = analyze(&forward, &adjoint, &elems, &Default::default(), 4);
        let mut sizes: HashMap<VarId, SizeInfo> = HashMap::new();
        for v in [1, 2, 10, 11, 12, 13, 14] {
            sizes.insert(v, SizeInfo::Shape(vec![64]));
        }
        let (ok, _) = admit(&plan, &forward, &adjoint, &Default::default(), &sizes);
        assert!(ok.contains(&10) && ok.contains(&11), "both should be admitted: {ok:?}");
        let (placements, total) = pack(&plan, &ok);
        assert!(total > 0);
        let a = placements.iter().find(|p| p.var == 10).unwrap();
        let b = placements.iter().find(|p| p.var == 11).unwrap();
        assert_ne!(a.offset, b.offset,
                   "simultaneously-live transients were given the same address");
        assert_ne!(a.slot_index, b.slot_index);
    }

    // -----------------------------------------------------------------
    // Element-count propagation
    // -----------------------------------------------------------------

    #[test]
    fn unary_elementwise_preserves_numel_down_a_chain() {
        let adj = list(vec![
            op(0, 10, PrimalOp::Relu, vec![1]),
            op(1, 11, PrimalOp::Gelu, vec![10]),
            op(2, 12, PrimalOp::Tanh, vec![11]),
        ]);
        let seed = |v: VarId| (v == 1).then_some(512u64);
        let got = propagate_elems(&list(vec![]), &adj, &seed);
        for v in [10, 11, 12] {
            assert_eq!(got.get(&v), Some(&512), "numel must survive var {v}");
        }
    }

    #[test]
    fn equal_numel_binaries_propagate_and_unequal_ones_stop() {
        let adj = list(vec![
            op(0, 10, PrimalOp::Add, vec![1, 2]), // 64 + 64 -> 64
            op(1, 11, PrimalOp::Mul, vec![1, 3]), // 64 * 8  -> broadcast, unknown
        ]);
        let seed = |v: VarId| match v {
            1 | 2 => Some(64u64),
            3 => Some(8u64),
            _ => None,
        };
        let got = propagate_elems(&list(vec![]), &adj, &seed);
        assert_eq!(got.get(&10), Some(&64));
        assert_eq!(
            got.get(&11), None,
            "a broadcasting binary's numel depends on shapes the tape does not \
             carry; guessing it would undersize a slot, which is a buffer \
             overrun rather than a bad statistic"
        );
    }

    #[test]
    fn propagation_stops_at_ops_that_need_real_shapes() {
        // Matmul, reductions and reshape all change numel in ways only the
        // shapes determine. Each must leave its result unsized — and, because
        // the sweep is single-pass over a topological tape, everything
        // downstream of it too.
        for o in [
            op(0, 10, PrimalOp::Matmul, vec![1, 2]),
            op(0, 10, PrimalOp::Sum { dim: None }, vec![1]),
            op(0, 10, PrimalOp::Passthrough("reshape".into()), vec![1, 2]),
        ] {
            let name = format!("{:?}", o.op);
            let adj = list(vec![o, op(1, 11, PrimalOp::Relu, vec![10])]);
            let seed = |v: VarId| (v == 1 || v == 2).then_some(64u64);
            let got = propagate_elems(&list(vec![]), &adj, &seed);
            assert_eq!(got.get(&10), None, "{name} must not derive a numel");
            assert_eq!(got.get(&11), None, "nothing downstream of {name} either");
        }
    }

    #[test]
    fn a_seed_always_wins_over_derivation() {
        // The caller's hint is authoritative: it came from a declared shape,
        // while derivation is an inference over op semantics.
        let adj = list(vec![op(0, 10, PrimalOp::Relu, vec![1])]);
        let seed = |v: VarId| match v {
            1 => Some(64u64),
            10 => Some(999u64),
            _ => None,
        };
        let got = propagate_elems(&list(vec![]), &adj, &seed);
        assert_eq!(got.get(&10), Some(&999));
    }

    /// Propagation is what makes admission non-vacuous, so pin the two
    /// together: the same tape admits nothing on bare seeds and something
    /// once counts are propagated.
    #[test]
    fn propagation_is_what_makes_admission_non_vacuous() {
        // Three readers of var 1, then one more read of 1 at the end, so
        // none of the unaries is an in-place candidate.
        let adj_ops = vec![
            op(0, 10, PrimalOp::Relu, vec![1]),
            op(1, 11, PrimalOp::Gelu, vec![1]),
            op(2, 12, PrimalOp::Tanh, vec![1]),
            op(3, 13, PrimalOp::Sigmoid, vec![1]),
        ];
        let forward = list(vec![]);
        let adjoint = list(adj_ops);
        let bare = |v: VarId| (v == 1).then_some(512u64);
        let plan0 = analyze(&forward, &adjoint, &bare, &Default::default(), 4);
        let (ok0, _) = admit(&plan0, &forward, &adjoint, &Default::default(), &HashMap::new());
        assert!(ok0.is_empty(), "without propagation nothing is sized, so nothing is placed");

        let hints = propagate_elems(&forward, &adjoint, &bare);
        let plan1 = analyze(&forward, &adjoint,
                            &|v| hints.get(&v).copied(), &Default::default(), 4);
        let (ok1, _) = admit(&plan1, &forward, &adjoint, &Default::default(), &HashMap::new());
        assert_eq!(
            ok1.len(),
            3,
            "the three non-final readers become placeable (the final one is \
             an in-place candidate): {ok1:?}"
        );
    }

    #[test]
    fn packing_nothing_is_zero_bytes() {
        let plan = analyze(&list(vec![]), &list(vec![]), &none_elems(), &Default::default(), 4);
        let (p, total) = pack(&plan, &[]);
        assert!(p.is_empty());
        assert_eq!(total, 0);
    }
}

#[cfg(test)]
mod shape_prop_tests {
    use super::SizeInfo::{Numel, Shape};
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};

    fn op(id: u32, result: VarId, primal: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result,
            op: primal,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    fn list(ops: Vec<WengertOp>) -> WengertList {
        WengertList {
            ops,
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    fn run(
        fwd: &WengertList,
        bwd: &WengertList,
        seeds: &[(VarId, Vec<i64>)],
        mirror: &[(VarId, VarId)],
    ) -> HashMap<VarId, SizeInfo> {
        let m: HashMap<VarId, Vec<i64>> = seeds.iter().cloned().collect();
        propagate_size_info(fwd, bwd, &|v| m.get(&v).cloned(), &|_| None, mirror)
    }

    #[test]
    fn matmul_batched_shapes_cross() {
        // x:[2,3,8,16] @ w:[16,32] -> [2,3,8,32]; then relu keeps it.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("w".into()), vec![]),
            op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            op(3, 3, PrimalOp::Relu, vec![2]),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![2, 3, 8, 16]), (1, vec![16, 32])], &[]);
        assert_eq!(k.get(&2), Some(&Shape(vec![2, 3, 8, 32])));
        assert_eq!(k.get(&3), Some(&Shape(vec![2, 3, 8, 32])));
    }

    #[test]
    fn matmul_inner_dim_mismatch_refuses() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("w".into()), vec![]),
            op(2, 2, PrimalOp::Matmul, vec![0, 1]),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![4, 8]), (1, vec![16, 32])], &[]);
        assert_eq!(k.get(&2), None);
    }

    #[test]
    fn matmul_carries_numel_across_a_rank2_weight() {
        // The load-bearing rule: input_ids reshaped at runtime (dims lost,
        // count kept) still crosses `x @ W[16,32]`: 128 elems / 16 * 32 = 256.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("shape_list".into()), vec![]),
            op(2, 2, PrimalOp::Passthrough("reshape".into()), vec![0, 1]),
            op(3, 3, PrimalOp::Param("w".into()), vec![]),
            op(4, 4, PrimalOp::Matmul, vec![2, 3]),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![8, 16]), (3, vec![16, 32])], &[]);
        assert_eq!(k.get(&2), Some(&Numel(128)), "reshape keeps the count");
        assert_eq!(k.get(&4), Some(&Numel(256)), "numel crosses a rank-2 matmul");
    }

    #[test]
    fn matmul_numel_left_operand_rank2() {
        // a:[4,16] known, b numel-only 16*32=512 -> out = 512*4/16 = 128.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("a".into()), vec![]),
            op(1, 1, PrimalOp::Input("b0".into()), vec![]),
            op(2, 2, PrimalOp::Param("l".into()), vec![]),
            op(3, 3, PrimalOp::Passthrough("reshape".into()), vec![1, 2]),
            op(4, 4, PrimalOp::Matmul, vec![0, 3]),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![4, 16]), (1, vec![16, 32])], &[]);
        assert_eq!(k.get(&4), Some(&Numel(128)));
    }

    #[test]
    fn binary_broadcast_and_the_equal_numel_refusal() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("a".into()), vec![]),
            op(1, 1, PrimalOp::Input("b".into()), vec![]),
            op(2, 2, PrimalOp::Add, vec![0, 1]), // [2,3,4] + [4] -> [2,3,4]
            op(3, 3, PrimalOp::Input("c".into()), vec![]),
            op(4, 4, PrimalOp::Input("d".into()), vec![]),
            op(5, 5, PrimalOp::Mul, vec![3, 4]), // [3,1] * [1,3]: equal numel, differing shape
            op(6, 6, PrimalOp::Input("e".into()), vec![]),
            op(7, 7, PrimalOp::Sub, vec![0, 6]), // [2,3,4] - [2,1,4] -> [2,3,4]
        ]);
        let k = run(
            &fwd,
            &list(vec![]),
            &[
                (0, vec![2, 3, 4]),
                (1, vec![4]),
                (3, vec![3, 1]),
                (4, vec![1, 3]),
                (6, vec![2, 1, 4]),
            ],
            &[],
        );
        assert_eq!(k.get(&2), Some(&Shape(vec![2, 3, 4])));
        assert_eq!(
            k.get(&5),
            None,
            "equal-numel differing-shape binaries diverge by device and must refuse"
        );
        assert_eq!(k.get(&7), Some(&Shape(vec![2, 3, 4])));
    }

    #[test]
    fn scalar_constant_broadcasts_without_losing_the_shape() {
        // x * 0.02 (the coder init idiom): Constant is a 0-dim tensor.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Constant(0.02), vec![]),
            op(2, 2, PrimalOp::Mul, vec![0, 1]),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![49152, 512])], &[]);
        assert_eq!(k.get(&2), Some(&Shape(vec![49152, 512])));
    }

    #[test]
    fn reductions_remove_the_dim_and_support_negative_indexing() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Sum { dim: Some(-1) }, vec![0]),
            op(2, 2, PrimalOp::Mean { dim: Some(0) }, vec![0]),
            op(3, 3, PrimalOp::Sum { dim: None }, vec![0]),
            op(4, 4, PrimalOp::Softmax { dim: -1 }, vec![0]),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![2, 3, 4])], &[]);
        assert_eq!(k.get(&1), Some(&Shape(vec![2, 3])));
        assert_eq!(k.get(&2), Some(&Shape(vec![3, 4])));
        assert_eq!(k.get(&3), Some(&Shape(vec![1])));
        assert_eq!(k.get(&4), Some(&Shape(vec![2, 3, 4])));
    }

    #[test]
    fn embedding_takes_indices_numel_and_table_cols() {
        // Real forward entry: dict_get [2,1024] -> runtime flatten (numel
        // survives) -> embedding vs table [49152, 512] -> [2048, 512].
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("batch".into()), vec![]),
            op(1, 1, PrimalOp::Passthrough("dict_get:input_ids".into()), vec![0]),
            op(2, 2, PrimalOp::Param("l".into()), vec![]),
            op(3, 3, PrimalOp::Passthrough("reshape".into()), vec![1, 2]),
            op(4, 4, PrimalOp::Param("embed".into()), vec![]),
            op(5, 5, PrimalOp::Embedding, vec![4, 3]),
        ]);
        let k = run(
            &fwd,
            &list(vec![]),
            &[(1, vec![2, 1024]), (4, vec![49152, 512])],
            &[],
        );
        assert_eq!(k.get(&5), Some(&Shape(vec![2048, 512])));
    }

    #[test]
    fn static_reshape_recovers_dims_from_constant_lists_and_const_shape() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Constant(6.0), vec![]),
            op(2, 2, PrimalOp::Constant(4.0), vec![]),
            op(3, 3, PrimalOp::Passthrough("list".into()), vec![1, 2]),
            op(4, 4, PrimalOp::Passthrough("reshape".into()), vec![0, 3]),
            op(
                5,
                5,
                PrimalOp::Passthrough(crate::lm_head_inference::const_shape_name(&[2, 12])),
                vec![],
            ),
            op(6, 6, PrimalOp::Passthrough("reshape".into()), vec![0, 5]),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![2, 3, 4])], &[]);
        assert_eq!(k.get(&4), Some(&Shape(vec![6, 4])));
        assert_eq!(k.get(&6), Some(&Shape(vec![2, 12])));
    }

    #[test]
    fn static_reshape_contradicting_the_input_count_refuses() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Constant(7.0), vec![]),
            op(2, 2, PrimalOp::Constant(4.0), vec![]),
            op(3, 3, PrimalOp::Passthrough("list".into()), vec![1, 2]),
            op(4, 4, PrimalOp::Passthrough("reshape".into()), vec![0, 3]),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![2, 3, 4])], &[]);
        assert_eq!(k.get(&4), None, "24 elems cannot become 28");
    }

    #[test]
    fn reduce_to_shape_takes_the_target() {
        let bwd = list(vec![
            op(0, 10, PrimalOp::Constant(1.0), vec![]),
            op(1, 11, PrimalOp::Passthrough("reduce_to_shape".into()), vec![10, 0]),
        ]);
        let fwd = list(vec![op(0, 0, PrimalOp::Param("gamma".into()), vec![])]);
        let k = run(&fwd, &bwd, &[(0, vec![512])], &[]);
        assert_eq!(k.get(&11), Some(&Shape(vec![512])));
    }

    #[test]
    fn mirror_seeds_the_adjoint_between_sweeps() {
        // Forward sizes h; the mirror hands h's shape to its accumulator;
        // the adjoint sweep then sizes an op computed FROM the accumulator.
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Relu, vec![0]),
        ]);
        let bwd = list(vec![
            op(0, 100, PrimalOp::Constant(1.0), vec![]),
            op(1, 101, PrimalOp::Mul, vec![100, 50]), // 50 = h's adjoint accumulator
        ]);
        let k = run(&fwd, &bwd, &[(0, vec![8, 16])], &[(1, 50)]);
        assert_eq!(k.get(&50), Some(&Shape(vec![8, 16])));
        assert_eq!(k.get(&101), Some(&Shape(vec![8, 16])));
    }

    #[test]
    fn data_dependent_and_device_divergent_ops_refuse() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Input("idx".into()), vec![]),
            op(2, 2, PrimalOp::ScatterAdd { dim: 0 }, vec![0, 1]),
            op(3, 3, PrimalOp::Gather { dim: 0 }, vec![0, 1]),
            op(4, 4, PrimalOp::Concat { dim: 0 }, vec![0]),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![8, 16]), (1, vec![8])], &[]);
        assert_eq!(k.get(&2), None, "ScatterAdd's leading dim is max(index)+1");
        assert_eq!(k.get(&3), None, "Gather dim-0 semantics differ by device");
        assert_eq!(k.get(&4), None, "Concat extents hide behind its list var");
    }

    #[test]
    fn fused_ce_backward_shapes_come_from_the_op_fields() {
        let mk = |component: u8, x_rank3: bool| PrimalOp::FusedLinearCeBackwardExtract {
            component,
            vocab_size: 49152,
            hidden_size: 512,
            batch_size: 2,
            seq_len: 1024,
            vocab_tile: 1024,
            ignore_index: -100,
            has_bias: false,
            x_rank3,
        };
        let bwd = list(vec![
            op(0, 100, PrimalOp::Constant(1.0), vec![]),
            op(1, 101, mk(0, true), vec![100]),
            op(2, 102, mk(0, false), vec![100]),
            op(3, 103, mk(1, false), vec![100]),
        ]);
        let k = run(&list(vec![]), &bwd, &[], &[]);
        assert_eq!(k.get(&101), Some(&Shape(vec![2, 1024, 512])));
        assert_eq!(k.get(&102), Some(&Shape(vec![2048, 512])));
        assert_eq!(k.get(&103), Some(&Shape(vec![49152, 512])));
    }

    #[test]
    fn slice_replicates_the_runtime_clamp_and_padzero_restores() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(
                1,
                1,
                PrimalOp::Slice { dim: 1, start: 2, end: 99, orig_dim_size: 10 },
                vec![0],
            ),
            op(
                2,
                2,
                PrimalOp::PadZero { dim: 1, pad_before: 2, pad_after: 0 },
                vec![1],
            ),
        ]);
        let k = run(&fwd, &list(vec![]), &[(0, vec![4, 10])], &[]);
        assert_eq!(k.get(&1), Some(&Shape(vec![4, 8])), "end clamps to dim size");
        assert_eq!(k.get(&2), Some(&Shape(vec![4, 10])));
    }

    #[test]
    fn sdpa_requires_the_uniform_regime() {
        let sdpa = |r, q, k_, v| {
            op(
                r,
                r,
                PrimalOp::ScaledDotProductAttention { causal: true },
                vec![q, k_, v],
            )
        };
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("q".into()), vec![]),
            op(1, 1, PrimalOp::Input("k".into()), vec![]),
            op(2, 2, PrimalOp::Input("v".into()), vec![]),
            sdpa(3, 0, 1, 2),
            op(4, 4, PrimalOp::Input("k2".into()), vec![]),
            sdpa(5, 0, 4, 2),
        ]);
        let k = run(
            &fwd,
            &list(vec![]),
            &[
                (0, vec![2, 8, 128, 64]),
                (1, vec![2, 8, 128, 64]),
                (2, vec![2, 8, 128, 64]),
                (4, vec![2, 4, 128, 64]), // GQA-unexpanded K: refuse
            ],
            &[],
        );
        assert_eq!(k.get(&3), Some(&Shape(vec![2, 8, 128, 64])));
        assert_eq!(k.get(&5), None, "K not q-shaped -> outside the proven regime");
    }

    #[test]
    fn flash_backward_extract_mirrors_q_k_v() {
        let ext = |r, c| {
            op(
                r,
                r,
                PrimalOp::FlashAttentionBackwardExtract { causal: true, component: c },
                vec![10, 0, 1, 2, 11],
            )
        };
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("q".into()), vec![]),
            op(1, 1, PrimalOp::Input("k".into()), vec![]),
            op(2, 2, PrimalOp::Input("v".into()), vec![]),
        ]);
        let bwd = list(vec![ext(20, 0), ext(21, 1), ext(22, 2)]);
        let k = run(
            &fwd,
            &bwd,
            &[
                (0, vec![2, 8, 128, 64]),
                (1, vec![2, 4, 128, 64]),
                (2, vec![2, 4, 128, 64]),
            ],
            &[],
        );
        assert_eq!(k.get(&20), Some(&Shape(vec![2, 8, 128, 64])), "dQ is Q-shaped");
        assert_eq!(k.get(&21), Some(&Shape(vec![2, 4, 128, 64])), "dK is K-shaped under GQA");
        assert_eq!(k.get(&22), Some(&Shape(vec![2, 4, 128, 64])), "dV is V-shaped under GQA");
    }

    #[test]
    fn norm_backward_ffi_shapes_follow_the_non_first_inputs() {
        let bwd = list(vec![
            op(0, 100, PrimalOp::Constant(1.0), vec![]),
            op(
                1,
                101,
                PrimalOp::Passthrough("rmsnorm_dgamma_backward:4562254508917369340".into()),
                vec![100, 0, 1],
            ),
            op(
                2,
                102,
                PrimalOp::Passthrough("rmsnorm_dx_backward:4562254508917369340".into()),
                vec![100, 0, 1],
            ),
            op(3, 103, PrimalOp::Passthrough("embedding_backward".into()), vec![100, 2, 3]),
            op(4, 104, PrimalOp::Passthrough("cross_entropy_backward".into()), vec![100, 4, 5]),
        ]);
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("gamma".into()), vec![]),
            op(2, 2, PrimalOp::Input("idx".into()), vec![]),
            op(3, 3, PrimalOp::Param("table".into()), vec![]),
            op(4, 4, PrimalOp::Input("logits".into()), vec![]),
            op(5, 5, PrimalOp::Input("targets".into()), vec![]),
        ]);
        let k = run(
            &fwd,
            &bwd,
            &[
                (0, vec![2, 128, 512]),
                (1, vec![512]),
                (3, vec![49152, 512]),
                (4, vec![256, 49152]),
            ],
            &[],
        );
        assert_eq!(k.get(&101), Some(&Shape(vec![512])), "dgamma is gamma-shaped");
        assert_eq!(k.get(&102), Some(&Shape(vec![2, 128, 512])), "dx is x-shaped");
        assert_eq!(k.get(&103), Some(&Shape(vec![49152, 512])), "d(table) is table-shaped");
        assert_eq!(k.get(&104), Some(&Shape(vec![256, 49152])), "dlogits is logits-shaped");
    }

    /// The const lattice end to end — the mechanism that recovered the
    /// attention chain. A tape shaped like the GQA forward: a config tensor
    /// whose VALUE is seeded (ctor-folded `full([1], 8.0)`), read back via
    /// `item()` -> `int()`, multiplied into a reshape target list alongside
    /// subscripts of a shape read. If any lattice link drops, the reshape
    /// falls to Numel and this fails.
    #[test]
    fn the_scalar_value_lattice_folds_an_attention_style_reshape() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("_n_heads".into()), vec![]),
            // let s = x.shape; b = s[0]; sl = s[1]
            op(2, 2, PrimalOp::Passthrough("shape".into()), vec![0]),
            op(3, 3, PrimalOp::Constant(0.0), vec![]),
            op(4, 4, PrimalOp::Passthrough("subscript".into()), vec![2, 3]),
            op(5, 5, PrimalOp::Constant(1.0), vec![]),
            op(6, 6, PrimalOp::Passthrough("subscript".into()), vec![2, 5]),
            // nh = int(_n_heads.item()); hd = 64
            op(7, 7, PrimalOp::Passthrough("item".into()), vec![1]),
            op(8, 8, PrimalOp::Passthrough("int".into()), vec![7]),
            op(9, 9, PrimalOp::Constant(64.0), vec![]),
            // list [b, sl, nh, hd] -> q.reshape(...)
            op(10, 10, PrimalOp::Passthrough("list".into()), vec![4, 6, 8, 9]),
            op(11, 11, PrimalOp::Passthrough("reshape".into()), vec![0, 10]),
        ]);
        let m: HashMap<VarId, Vec<i64>> = [(0, vec![2, 128, 512])].into();
        let scalars: HashMap<VarId, f64> = [(1, 8.0)].into();
        let known = propagate_size_info(
            &fwd,
            &list(vec![]),
            &|v| m.get(&v).cloned(),
            &|v| scalars.get(&v).copied(),
            &[],
        );
        assert_eq!(
            known.get(&11),
            Some(&Shape(vec![2, 128, 8, 64])),
            "the item()/int()/shape-subscript lattice must fold the reshape"
        );
    }

    #[test]
    fn division_folds_as_f64_unless_both_operands_are_integer_typed() {
        // d / nh where nh came through item() (Tensor-typed): the lowering
        // divides in floating point, so trunc(100/8)=12 would be a false
        // claim — the fold must yield 12.5 and downstream int-consumers
        // must refuse.
        let mut fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("_n".into()), vec![]),
            op(2, 2, PrimalOp::Constant(100.0), vec![]),
            op(3, 3, PrimalOp::Passthrough("item".into()), vec![1]),
            op(4, 4, PrimalOp::Div, vec![2, 3]),
            op(5, 5, PrimalOp::Passthrough("list".into()), vec![4]),
            op(6, 6, PrimalOp::Passthrough("reshape".into()), vec![0, 5]),
        ]);
        // Mark the Div result Tensor-typed (the default; explicit for the
        // test's clarity).
        fwd.var_types.insert(4, crate::wengert::WengertType::Tensor);
        let scalars: HashMap<VarId, f64> = [(1, 8.0)].into();
        let known = propagate_size_info(
            &fwd,
            &list(vec![]),
            &|_| None,
            &|v| scalars.get(&v).copied(),
            &[],
        );
        assert_eq!(
            known.get(&6),
            None,
            "100/8.0 is 12.5 at runtime; folding it to 12 and claiming a \
             12-element reshape would be a false compile-time shape"
        );
    }

    #[test]
    fn overflow_and_nonpositive_dims_refuse() {
        let fwd = list(vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Relu, vec![0]),
        ]);
        // Overflowing product: i64::MAX * 2 as u64 product overflows.
        let k = run(&fwd, &list(vec![]), &[(0, vec![i64::MAX, 2])], &[]);
        assert_eq!(k.get(&0), None, "an overflowing seed is not a size");
        assert_eq!(k.get(&1), None);
        let k2 = run(&fwd, &list(vec![]), &[(0, vec![4, 0])], &[]);
        assert_eq!(k2.get(&1), None, "a zero dim is a sentinel, not a size");
    }
}
