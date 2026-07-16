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
    /// Byte-level BFD packing, present only when *every* transient's element
    /// count was known (the quantified path). `elem_bytes` was assumed for the
    /// dtype. `None` when any transient is runtime-shaped.
    pub slab: Option<SlabPlan>,
    /// Σ of live-interval lengths — a proxy for the alloc/free churn the
    /// caching allocator performs that a static arena removes.
    pub total_alloc_events: usize,
    /// Bytes assumed per element for the quantified path (GPU training = f32).
    pub elem_bytes: u64,
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
pub fn analyze(
    forward: &WengertList,
    adjoint: &WengertList,
    elems_of: &dyn Fn(VarId) -> Option<u64>,
    elem_bytes: u64,
) -> ArenaPlan {
    // Concatenated timeline: forward ops first, then adjoint. A saved forward
    // value read in the backward therefore has death >= forward.len().
    let fwd_n = forward.ops.len();

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
            // Never-used values die where they were born (allocated then freed).
            let death = last_use.get(&op.result).copied().unwrap_or(birth).max(birth);
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

    // Quantified path: only when *every* transient has a known element count.
    let all_known = !transients.is_empty() && transients.iter().all(|t| t.elems.is_some());
    let slab = if all_known {
        let allocs: Vec<TensorAlloc> = transients
            .iter()
            .enumerate()
            .map(|(id, t)| {
                let bytes = t.elems.unwrap().saturating_mul(elem_bytes);
                TensorAlloc {
                    id: id as u32,
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

    ArenaPlan {
        transients,
        max_concurrency,
        peak_at,
        slab,
        total_alloc_events,
        elem_bytes,
    }
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
        let plan = analyze(&list(vec![]), &list(vec![]), &none_elems(), 4);
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
        let plan = analyze(&fwd, &list(vec![]), &none_elems(), 4);
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
        let plan = analyze(&fwd, &bwd, &none_elems(), 4);
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
        let plan = analyze(&fwd, &list(vec![]), &none_elems(), 4);
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
        let plan = analyze(&fwd, &list(vec![]), &none_elems(), 4);
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
        let plan = analyze(&fwd, &list(vec![]), &elems, 4);
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
        let plan = analyze(&fwd, &list(vec![]), &elems, 4);
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
}
