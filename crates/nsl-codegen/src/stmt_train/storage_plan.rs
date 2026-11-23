//! `StoragePlanPass`: which values of a `precision="bf16"` train step are
//! stored bf16, and which ops the lowering must therefore emit through the
//! runtime's dtype-explicit entry points (`nsl_tensor_*_dt`).
//!
//! Scope of this first slice: the SwiGLU MLP of each transformer block,
//!
//! ```text
//! n = rmsnorm(h, gamma)          bf16 (the norm writes it bf16)
//! a = n @ W_gate                 bf16
//! s = silu(a)                    bf16
//! u = n @ W_up                   bf16
//! p = s * u                      bf16
//! d = p @ W_down                 bf16
//! r = h + d                      f32  (the residual stream stays f32)
//! ```
//!
//! (when `d` feeds a training dropout instead, whose kernel is f32-only, `d`
//! is stored f32 and the rest of the block as above)
//!
//! and in the backward: the gradients of `a`, `s`, `u`, `p` are bf16 (where
//! PyTorch autocast has them bf16); the gradient of `n` stays f32 (both its
//! dgrads are written f32 and summed in f32, then read by the f32 norm
//! backward); weight gradients are f32 (accumulated in f32 by the GEMM).
//! Every recompute clone CCR splices into the adjoint takes its original's
//! dtype, so a recomputed value equals the forward value its saved
//! neighbours were computed from.
//!
//! Rules, in order (design `bf16-design.md` 3.2, restricted to the slice):
//!
//! 1. A slice is admitted only when its values have exactly the consumers
//!    above in the forward; anything else leaves the block f32.
//! 2. A clone's dtype is its original's.
//! 3. Elementwise ops are bf16 when every input is bf16, else f32.
//! 4. An adjoint GEMM is bf16 when it produces the gradient of a value whose
//!    gradient is planned bf16 (its single reader is the `reduce_to_shape`
//!    onto that value); otherwise f32.
//!    4b. The RoPE adjoint of a bf16 gradient is bf16 when only GEMMs read
//!    its result (through views): the q/k projections' input and weight
//!    gradients, which take bf16 operands. (`NSL_ROPE_BWD_BF16=0`: f32.)
//! 5. A transpose of a bf16 value is a bf16 view; `reduce_to_shape` of a
//!    bf16 gradient onto a planned value of the same role is the identity.
//! 6. Verify: every op that reads a bf16 value is one the dtype-explicit
//!    lowering covers (or is dtype-agnostic: a free, a view, a shape read).
//!    Anything else is a compile error naming the op, not a silent widen.
//!
//! The plan is a pure function of the two lists and CCR's clone map; it is
//! installed on the compiler (`Compiler::storage_out`) for the lowering and
//! cleared with the train block, like the arena placements.

use std::collections::{HashMap, HashSet};

use crate::compiler::Compiler;
use crate::wengert::{PrimalOp, VarId, WengertList, WengertOp};

/// A value's storage dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Storage {
    F32,
    Bf16,
}

impl Storage {
    /// The runtime dtype tag (`nsl_abi::wire`: f32 = 1, bf16 = 3).
    pub fn tag(self) -> i64 {
        match self {
            Storage::F32 => 1,
            Storage::Bf16 => 3,
        }
    }
}

/// The plan for one train step.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct StoragePlan {
    /// For every op the lowering must emit through a dtype-explicit entry
    /// point (its result is bf16, or it reads a bf16 value): its result
    /// VarId and the output dtype it stores.
    pub out: HashMap<VarId, Storage>,
    /// Every value stored bf16, views included.
    pub bf16: HashSet<VarId>,
    /// SwiGLU slices admitted.
    pub slices: usize,
    /// Attention halves admitted (step 5).
    pub attention: usize,
    /// `param_dtype=`: the GEMMs planned only because they read a
    /// bf16-stored parameter (their other operand is an f32 value): lowered
    /// through `nsl_tensor_matmul_pdt`, which keeps the eager bf16 matmul
    /// mode's per-shape choice of bf16 or f32 operands.
    pub param_gemm: HashSet<VarId>,
}

/// One admitted SwiGLU slice (primal VarIds).
#[derive(Debug, Clone, Copy)]
struct Slice {
    n: VarId,
    a: VarId,
    /// `silu(a)`; `None` for the fused form `p = swiglu(a, u)`
    /// (`source_ad::fuse_swiglu_forward`), which stores no such value.
    s: Option<VarId>,
    u: VarId,
    p: VarId,
    d: VarId,
    /// `d` feeds a training dropout (whose kernel is f32-only) rather than
    /// the residual add: `d` is then stored f32 (the down GEMM rounds its
    /// f32 accumulator to f32, more precise than autocast's bf16 there).
    d_f32: bool,
}

fn producers(list: &WengertList) -> HashMap<VarId, &WengertOp> {
    list.ops.iter().map(|op| (op.result, op)).collect()
}

fn readers(list: &WengertList) -> HashMap<VarId, Vec<&WengertOp>> {
    let mut m: HashMap<VarId, Vec<&WengertOp>> = HashMap::new();
    for op in &list.ops {
        if matches!(op.op, PrimalOp::FreeTensor) {
            continue;
        }
        for &i in &op.inputs {
            m.entry(i).or_default().push(op);
        }
    }
    m
}

fn is_param(prod: &HashMap<VarId, &WengertOp>, v: VarId) -> bool {
    prod.get(&v).is_some_and(|op| matches!(op.op, PrimalOp::Param(_)))
}

/// `Matmul(x, W)` with `W` a parameter: `Some(x)`.
fn matmul_of_param(prod: &HashMap<VarId, &WengertOp>, v: VarId) -> Option<VarId> {
    let op = prod.get(&v)?;
    (matches!(op.op, PrimalOp::Matmul) && op.inputs.len() == 2 && is_param(prod, op.inputs[1])).then(|| op.inputs[0])
}

/// Whether `v` is read, directly or through transposes, reshapes and
/// copies of it, by at least one GEMM and by nothing else (a `shape` read
/// aside): rule 4b's GEMM-ready gradient.
fn feeds_only_gemms(rd: &HashMap<VarId, Vec<&WengertOp>>, v: VarId) -> bool {
    let mut stack = vec![v];
    let mut gemms = 0usize;
    while let Some(x) = stack.pop() {
        for r in rd.get(&x).map(Vec::as_slice).unwrap_or_default() {
            if matches!(r.op, PrimalOp::Matmul) {
                gemms += 1;
            } else if is_passthrough(&r.op, "shape") {
                continue;
            } else if (matches!(r.op, PrimalOp::Transpose { .. })
                || is_passthrough(&r.op, "reshape")
                || is_passthrough(&r.op, "contiguous"))
                && r.inputs.first() == Some(&x)
            {
                stack.push(r.result);
            } else {
                return false;
            }
        }
    }
    gemms > 0
}

/// Rule 4b's switch: `NSL_ROPE_BWD_BF16=0` at build time keeps the RoPE
/// adjoints f32 (the plan before rule 4b).
fn rope_grad_bf16_enabled() -> bool {
    std::env::var("NSL_ROPE_BWD_BF16").ok().as_deref() != Some("0")
}

fn only_reader<'a>(rd: &HashMap<VarId, Vec<&'a WengertOp>>, v: VarId) -> Option<&'a WengertOp> {
    match rd.get(&v).map(Vec::as_slice) {
        Some([one]) => Some(*one),
        _ => None,
    }
}

/// The SwiGLU slices of the forward whose values have exactly the slice's
/// consumers (rule 1).
fn find_slices(primal: &WengertList) -> Vec<Slice> {
    let prod = producers(primal);
    let rd = readers(primal);
    let mut out = Vec::new();
    for op in &primal.ops {
        // `s = silu(a); p = s * u`, or the fused `p = swiglu(a, u)`.
        let (a, s, u, p) = if matches!(op.op, PrimalOp::Silu) && op.inputs.len() == 1 {
            let (s, a) = (op.result, op.inputs[0]);
            let Some(p_op) = only_reader(&rd, s) else { continue };
            if !matches!(p_op.op, PrimalOp::Mul) || p_op.inputs.len() != 2 {
                continue;
            }
            let u = if p_op.inputs[0] == s { p_op.inputs[1] } else { p_op.inputs[0] };
            if u == s || !p_op.inputs.contains(&s) {
                continue;
            }
            (a, Some(s), u, p_op.result)
        } else if is_passthrough(&op.op, "swiglu") && op.inputs.len() == 2 {
            (op.inputs[0], None, op.inputs[1], op.result)
        } else {
            continue;
        };
        if a == u {
            continue;
        }
        let Some(n) = matmul_of_param(&prod, a) else { continue };
        if matmul_of_param(&prod, u) != Some(n) {
            continue;
        }
        let (Some(a_r), Some(u_r)) = (only_reader(&rd, a), only_reader(&rd, u)) else { continue };
        if a_r.result != s.unwrap_or(p) || u_r.result != p {
            continue;
        }
        let Some(d_op) = only_reader(&rd, p) else { continue };
        if matmul_of_param(&prod, d_op.result) != Some(p) {
            continue;
        }
        let d = d_op.result;
        if !prod.get(&n).is_some_and(|o| matches!(o.op, PrimalOp::RMSNorm { .. })) {
            continue;
        }
        let n_readers: HashSet<VarId> = rd.get(&n).map(|v| v.iter().map(|o| o.result).collect()).unwrap_or_default();
        let a_prod = prod[&a].result;
        let u_prod = prod[&u].result;
        if n_readers != HashSet::from([a_prod, u_prod]) || rd.get(&n).map_or(0, Vec::len) != 2 {
            continue;
        }
        // The down projection feeds the residual add(s), or a training
        // dropout (its mask and its apply, both reading `d`), and nothing else.
        let d_readers = rd.get(&d).map(Vec::as_slice).unwrap_or_default();
        let to_residual = !d_readers.is_empty() && d_readers.iter().all(|o| matches!(o.op, PrimalOp::Add));
        let to_dropout = d_readers.len() == 2
            && d_readers.iter().any(|o| matches!(o.op, PrimalOp::DropoutMask { .. }))
            && d_readers.iter().any(|o| matches!(o.op, PrimalOp::Dropout { .. }));
        if !to_residual && !to_dropout {
            continue;
        }
        out.push(Slice { n, a, s, u, p, d, d_f32: to_dropout });
    }
    out
}

fn is_unary_dt(op: &PrimalOp) -> bool {
    matches!(
        op,
        PrimalOp::Silu
            | PrimalOp::Relu
            | PrimalOp::Sigmoid
            | PrimalOp::Exp
            | PrimalOp::Log
            | PrimalOp::Sqrt
            | PrimalOp::Abs
            | PrimalOp::Neg
    )
}

fn is_binary(op: &PrimalOp) -> bool {
    matches!(op, PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div)
}

fn is_passthrough(op: &PrimalOp, name: &str) -> bool {
    matches!(op, PrimalOp::Passthrough(n) if n == name)
}

/// The op kinds the dtype-explicit lowering covers.
pub fn has_dt_lowering(op: &PrimalOp) -> bool {
    matches!(op, PrimalOp::Matmul | PrimalOp::RMSNorm { .. })
        || is_binary(op)
        || is_unary_dt(op)
        || is_passthrough(op, "swiglu_gate_backward")
        || is_passthrough(op, "swiglu")
        || is_passthrough(op, "swiglu_up_backward")
        || is_passthrough(op, "rotate_half")
        || is_passthrough(op, "rotate_half_neg")
        || is_passthrough(op, "reduce_to_shape")
        || is_passthrough(op, "rope")
        || is_passthrough(op, "rope_backward")
        || is_passthrough(op, "matmul_acc")
}

/// Ops whose result has their (first) input's storage and that the runtime
/// performs for either dtype: views and copies.
fn is_dtype_preserving(op: &PrimalOp) -> bool {
    matches!(op, PrimalOp::Transpose { .. })
        || is_passthrough(op, "reshape")
        || is_passthrough(op, "expand")
        || is_passthrough(op, "contiguous")
}

/// The attention half of a block (design step 5): the values to store bf16
/// at their producers (the attention norm, the q/k/v projections, the RoPE
/// results the attention reads, the output projection) for every
/// scaled-dot-product attention whose operands come from parameter GEMMs of
/// one RMSNorm output, through views, copies and RoPE.
fn find_attention_seeds(primal: &WengertList) -> (Vec<VarId>, Vec<VarId>) {
    let prod = producers(primal);
    let rd = readers(primal);
    // Walk back through views and copies to the value's real producer.
    let through = |mut v: VarId| -> VarId {
        while let Some(op) = prod.get(&v) {
            if is_dtype_preserving(&op.op) {
                v = op.inputs[0];
            } else {
                break;
            }
        }
        v
    };
    let mut seeds = Vec::new();
    let mut norms = Vec::new();
    for op in &primal.ops {
        if !matches!(op.op, PrimalOp::ScaledDotProductAttention { .. }) || op.inputs.len() < 3 {
            continue;
        }
        let mut gemms = Vec::new();
        let mut ropes = Vec::new();
        let mut ok = true;
        for &x in &op.inputs[..3] {
            let src = through(x);
            if matmul_of_param(&prod, src).is_some() {
                gemms.push(src);
                continue;
            }
            // The `rope` builtin of a view of a GEMM.
            if let Some(r) = prod.get(&src).filter(|o| is_passthrough(&o.op, "rope")) {
                let g = through(r.inputs[0]);
                if matmul_of_param(&prod, g).is_some() {
                    gemms.push(g);
                    ropes.push(src);
                    continue;
                }
                ok = false;
                break;
            }
            // RoPE: Add(Mul(t, cos), Mul(rotate_half(t), sin)) of a view of a GEMM.
            let Some(add) = prod.get(&src).filter(|o| matches!(o.op, PrimalOp::Add)) else {
                ok = false;
                break;
            };
            let gemm = add
                .inputs
                .iter()
                .filter_map(|&m| prod.get(&m).filter(|o| matches!(o.op, PrimalOp::Mul)))
                .flat_map(|m| m.inputs.iter().copied())
                .map(|t| {
                    let t = prod.get(&t).filter(|o| is_passthrough(&o.op, "rotate_half")).map_or(t, |o| o.inputs[0]);
                    through(t)
                })
                .find(|&g| matmul_of_param(&prod, g).is_some());
            match gemm {
                Some(g) => {
                    gemms.push(g);
                    ropes.push(src);
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || gemms.len() != 3 {
            continue;
        }
        // One RMSNorm output feeds the three projections and nothing else.
        let n = matmul_of_param(&prod, gemms[0]).expect("checked");
        if gemms.iter().any(|&g| matmul_of_param(&prod, g) != Some(n))
            || !prod.get(&n).is_some_and(|o| matches!(o.op, PrimalOp::RMSNorm { .. }))
        {
            continue;
        }
        let n_readers = rd.get(&n).map(Vec::as_slice).unwrap_or_default();
        if !n_readers.iter().all(|o| matches!(o.op, PrimalOp::Matmul) || is_passthrough(&o.op, "shape")) {
            continue;
        }
        // The output: views/copies of the attention result into a parameter GEMM.
        let mut out = op.result;
        while let Some([r]) = rd.get(&out).map(Vec::as_slice) {
            if is_dtype_preserving(&r.op) {
                out = r.result;
            } else {
                break;
            }
        }
        let wo = rd.get(&out).and_then(|rs| match rs.as_slice() {
            [r] if matches!(r.op, PrimalOp::Matmul) && matmul_of_param(&prod, r.result) == Some(out) => Some(r.result),
            _ => None,
        });
        let Some(wo) = wo else { continue };
        norms.push(n);
        seeds.push(n);
        seeds.extend(gemms.iter().copied());
        seeds.extend(ropes);
        // Stored bf16 when it feeds the residual add directly. Behind a
        // training dropout (f32-only kernel) the general rules write it f32,
        // as the MLP's down projection.
        if rd.get(&wo).is_some_and(|rs| !rs.is_empty() && rs.iter().all(|o| matches!(o.op, PrimalOp::Add))) {
            seeds.push(wo);
        }
    }
    seeds.sort_unstable();
    seeds.dedup();
    (seeds, norms)
}

/// `param_dtype=`: the parameters a step stores bf16. A `Param` of the step
/// qualifies when `index_of` maps its name to a position of the train
/// block's parameter list (it is trained, so the fused AdamW step updates
/// it) and every reader of it, looking through views (not copies), is a GEMM
/// or reads only its shape (`shape`, the shape operand of
/// `reduce_to_shape`). Those are the readers that already see a bf16 copy
/// of it under `precision="bf16"`: every storage GEMM reads its operands
/// bf16, and the eager bf16 matmul mode its parameter operand. Anything
/// else (an embedding gather, a norm, the fused LM head) keeps the
/// parameter f32. Returns `(VarId, list index)` pairs, every `Param` op of a
/// qualifying parameter (in either list) included, ordered by VarId.
pub fn select_bf16_params(
    primal: &WengertList,
    adjoint: &WengertList,
    index_of: &dyn Fn(&str) -> Option<usize>,
) -> Vec<(VarId, usize)> {
    let mut rd = readers(primal);
    for (k, v) in readers(adjoint) {
        rd.entry(k).or_default().extend(v);
    }
    let only_gemms = |v: VarId| -> bool {
        let mut stack = vec![v];
        let mut seen = HashSet::new();
        let mut gemm = false;
        while let Some(x) = stack.pop() {
            if !seen.insert(x) {
                continue;
            }
            for r in rd.get(&x).map(Vec::as_slice).unwrap_or_default() {
                match &r.op {
                    PrimalOp::Matmul => gemm = true,
                    // A summed input gradient's GEMM (`source_ad::fuse_dgrad_accumulate`)
                    // reads its two GEMM operands as a Matmul does; its accumulator
                    // (operand 2) is a gradient, never a parameter.
                    PrimalOp::Passthrough(n)
                        if n == "matmul_acc"
                            && r.inputs.iter().position(|&i| i == x).is_some_and(|k| k < 2) =>
                    {
                        gemm = true
                    }
                    // Views keep the parameter's data pointer, which is how
                    // the runtime finds its exact f32 master
                    // (`param_storage::exact_f32`); a `contiguous` copy
                    // would not, so it disqualifies.
                    o if is_dtype_preserving(o)
                        && !is_passthrough(o, "contiguous")
                        && r.inputs.first() == Some(&x) =>
                    {
                        stack.push(r.result)
                    }
                    PrimalOp::Passthrough(n) if n == "shape" => {}
                    PrimalOp::Passthrough(n)
                        if n == "reduce_to_shape" && r.inputs.first() != Some(&x) => {}
                    _ => return false,
                }
            }
        }
        gemm
    };
    // Every Param op of a name: a parameter is one storage however many
    // loads of it the lists hold, so it qualifies only if all of them do.
    let mut by_name: HashMap<&str, Vec<VarId>> = HashMap::new();
    for op in primal.ops.iter().chain(adjoint.ops.iter()) {
        if let PrimalOp::Param(name) = &op.op {
            by_name.entry(name.as_str()).or_default().push(op.result);
        }
    }
    let mut out: Vec<(VarId, usize)> = by_name
        .into_iter()
        .filter_map(|(name, vids)| Some((index_of(name)?, vids)))
        .filter(|(_, vids)| vids.iter().all(|&v| only_gemms(v)))
        .flat_map(|(idx, vids)| vids.into_iter().map(move |v| (v, idx)))
        .collect();
    out.sort_unstable();
    out
}

struct Planner<'a> {
    plan: StoragePlan,
    /// Norm outputs (and their clones): stored bf16, but their gradient is
    /// f32 (summed from several GEMMs, read by the f32 norm backward).
    norm_out: HashSet<VarId>,
    /// `param_dtype=`: the bf16-stored parameters and their views. A
    /// parameter's gradient is f32 (it is accumulated in f32 and read by the
    /// optimizer), whatever rule 6 says for values.
    param_view: HashSet<VarId>,
    clones: &'a HashMap<VarId, VarId>,
    /// Each recompute clone and each CCR shape stub, mapped to the primal
    /// value it stands for (see [`Planner::grad_is_bf16`]).
    stand_in: HashMap<VarId, VarId>,
    /// Rule 4b on (see [`plan_with_params`]).
    rope_grad_bf16: bool,
}

impl Planner<'_> {
    fn is_bf16(&self, v: VarId) -> bool {
        self.plan.bf16.contains(&v)
    }

    /// Rule 6: a gradient is stored as its value is, except a norm output's
    /// and a parameter's.
    ///
    /// A target that is a recompute clone or a shape stub is the value it
    /// stands for (rule 2), resolved through its original: the backward can
    /// reach the GEMM producing a value's gradient before the clone group
    /// that recomputes the value (CCR splices a block's clones at their
    /// first reader, and the down projection's dgrad reads none of them), and
    /// a stub is no op of either list. Asking the clone itself made such a
    /// GEMM, and the whole MLP backward after it, f32 in exactly the blocks
    /// whose clones were spliced late.
    fn grad_is_bf16(&self, target: VarId) -> bool {
        let t = self.stand_in.get(&target).copied().unwrap_or(target);
        self.is_bf16(t) && !self.norm_out.contains(&t) && !self.param_view.contains(&t)
    }

    fn set(&mut self, op: &WengertOp, st: Storage) {
        self.plan.out.insert(op.result, st);
        if st == Storage::Bf16 {
            self.plan.bf16.insert(op.result);
        }
    }

    fn refuse(op: &WengertOp, list: &str, why: &str) -> String {
        format!(
            "precision=\"bf16\": the storage plan cannot lower {list} op {:?} (result VarId {}, inputs {:?}): \
             {why}. This is a compiler limitation of the bf16 activation slice, not a program error; \
             use precision=\"f32\" for this model",
            op.op, op.result, op.inputs
        )
    }

    /// All of `vs` bf16, or none (`Err` names the op otherwise).
    fn all_or_none(&self, op: &WengertOp, list: &str, vs: &[VarId]) -> Result<bool, String> {
        let n = vs.iter().filter(|&&v| self.is_bf16(v)).count();
        if n != 0 && n != vs.len() {
            return Err(Self::refuse(op, list, "its tensor operands mix bf16 and f32 storage"));
        }
        Ok(n != 0)
    }

    /// Rules 2-6 for one op of the forward (`in_adjoint` false) or the
    /// backward.
    fn visit(&mut self, op: &WengertOp, in_adjoint: bool, rd_adj: &HashMap<VarId, Vec<&WengertOp>>) -> Result<(), String> {
        let list = if in_adjoint { "backward" } else { "forward" };
        if matches!(op.op, PrimalOp::FreeTensor) || self.plan.out.contains_key(&op.result) {
            return Ok(());
        }
        let reads_bf16 = op.inputs.iter().any(|&i| self.is_bf16(i));
        // Rule 2: a recompute clone stores what its original stores.
        if let Some(&orig) = self.clones.get(&op.result) {
            if self.norm_out.contains(&orig) {
                self.norm_out.insert(op.result);
            }
            if self.is_bf16(orig) && has_dt_lowering(&op.op) {
                self.set(op, Storage::Bf16);
                return Ok(());
            }
            if self.plan.out.get(&orig) == Some(&Storage::F32) && has_dt_lowering(&op.op) {
                self.set(op, Storage::F32);
                if self.plan.param_gemm.contains(&orig) {
                    self.plan.param_gemm.insert(op.result);
                }
                return Ok(());
            }
            // A view or copy clone: the general rules below, then checked.
            self.visit_general(op, in_adjoint, reads_bf16, list, rd_adj)?;
            if self.is_bf16(orig) != self.is_bf16(op.result) {
                return Err(Self::refuse(op, list, "a recompute clone whose storage would differ from its original's"));
            }
            return Ok(());
        }
        self.visit_general(op, in_adjoint, reads_bf16, list, rd_adj)
    }

    fn visit_general(
        &mut self,
        op: &WengertOp,
        in_adjoint: bool,
        reads_bf16: bool,
        list: &str,
        rd_adj: &HashMap<VarId, Vec<&WengertOp>>,
    ) -> Result<(), String> {
        // The single reader `reduce_to_shape(this, target)`: whose gradient this is.
        let target = || {
            only_reader(rd_adj, op.result)
                .filter(|r| is_passthrough(&r.op, "reduce_to_shape") && r.inputs[0] == op.result)
                .map(|r| r.inputs[1])
        };
        match &op.op {
            o if is_dtype_preserving(o) => {
                if op.inputs.first().is_some_and(|&i| self.is_bf16(i)) {
                    self.plan.bf16.insert(op.result);
                }
                if op
                    .inputs
                    .first()
                    .is_some_and(|i| self.param_view.contains(i))
                {
                    self.param_view.insert(op.result);
                }
                Ok(())
            }
            PrimalOp::Passthrough(name) if name == "shape" => Ok(()),
            PrimalOp::ScaledDotProductAttention { .. } => {
                // The attention writes O in its operands' storage (bf16: the
                // tensor-core kernels, option (a)); the log-sum-exp stays f32.
                if self.all_or_none(op, list, &op.inputs[..3])? {
                    self.plan.bf16.insert(op.result);
                }
                Ok(())
            }
            PrimalOp::FlashAttentionBackwardExtract { .. } => {
                // inputs: dO, q, k, v, O, scale. dQ/dK/dV in the operands' storage.
                if self.all_or_none(op, list, &op.inputs[..5])? {
                    self.plan.bf16.insert(op.result);
                }
                Ok(())
            }
            PrimalOp::Passthrough(name) if name == "reduce_to_shape" => {
                let (g, like) = (op.inputs[0], op.inputs[1]);
                if self.is_bf16(g) {
                    // Identity, or a sum back to the value's shape (GQA): the
                    // result is the gradient of `like`, stored as rule 6 says.
                    self.set(op, if self.grad_is_bf16(like) { Storage::Bf16 } else { Storage::F32 });
                }
                // An f32 gradient reduced onto a bf16 value reads only its shape.
                Ok(())
            }
            PrimalOp::Matmul => {
                // Rule 4: the gradient of a value whose gradient is bf16.
                if in_adjoint && target().is_some_and(|t| self.grad_is_bf16(t)) {
                    self.set(op, Storage::Bf16);
                } else if reads_bf16 {
                    // Every bf16 operand a stored parameter: without
                    // `param_dtype` this GEMM would read f32 operands and take
                    // the eager bf16 mode, whose per-shape rule it keeps.
                    if op
                        .inputs
                        .iter()
                        .all(|i| !self.is_bf16(*i) || self.param_view.contains(i))
                    {
                        self.plan.param_gemm.insert(op.result);
                    }
                    self.set(op, Storage::F32);
                }
                Ok(())
            }
            PrimalOp::Passthrough(name) if name == "matmul_acc" => {
                // A gradient summed from several GEMMs (`source_ad::fuse_dgrad_accumulate`),
                // stored as the sum it replaces was. Its accumulator `c` is the sum so far
                // for the same value, stored as rule 6 says for that value: f32 for a norm
                // output (the production blocks), where the unfused dgrads were f32 GEMM
                // outputs added in f32; bf16 where the value's gradient is bf16, where the
                // unfused dgrad (rule 4), its reduce and the add were all bf16 (the
                // lowering then replays exactly those three ops). An f32 sum is planned
                // when it reads a bf16 value; like a Matmul, through the eager rule when
                // its only bf16 operands are stored parameters.
                if self.is_bf16(op.inputs[2]) {
                    self.set(op, Storage::Bf16);
                } else if reads_bf16 {
                    if op.inputs[..2]
                        .iter()
                        .all(|i| !self.is_bf16(*i) || self.param_view.contains(i))
                    {
                        self.plan.param_gemm.insert(op.result);
                    }
                    self.set(op, Storage::F32);
                }
                Ok(())
            }
            PrimalOp::Passthrough(name) if name == "rope_backward" => {
                // The RoPE adjoint mixes in the f32 tables, but writes the
                // gradient of the head view: stored as rule 6 says (rule 4).
                // Rule 4b: a bf16 gradient whose result only GEMMs read
                // (through the head view's adjoint views: the q/k projections'
                // dgrad and wgrad) is written bf16, the operand those GEMMs
                // take, instead of f32 for each GEMM to round.
                if in_adjoint && target().is_some_and(|t| self.grad_is_bf16(t)) {
                    self.set(op, Storage::Bf16);
                } else if reads_bf16 && in_adjoint && self.rope_grad_bf16 && feeds_only_gemms(rd_adj, op.result) {
                    self.set(op, Storage::Bf16);
                } else if reads_bf16 {
                    self.set(op, Storage::F32);
                }
                Ok(())
            }
            o if is_binary(o)
                || is_unary_dt(o)
                || is_passthrough(o, "swiglu_gate_backward")
                || is_passthrough(o, "swiglu")
                || is_passthrough(o, "swiglu_up_backward")
                || is_passthrough(o, "rotate_half")
                || is_passthrough(o, "rotate_half_neg")
                || is_passthrough(o, "rope") =>
            {
                if reads_bf16 {
                    let all = op.inputs.iter().all(|&i| self.is_bf16(i));
                    self.set(op, if all { Storage::Bf16 } else { Storage::F32 });
                }
                Ok(())
            }
            _ if reads_bf16 => Err(Self::refuse(op, list, "it reads a bf16 value and has no dtype-explicit lowering")),
            _ => Ok(()),
        }
    }
}

/// Plan the step's storage. `clones` maps each CCR recompute clone in
/// `adjoint` to the primal value it recomputes (empty without CCR).
#[cfg(test)]
pub fn plan(primal: &WengertList, adjoint: &WengertList, clones: &HashMap<VarId, VarId>) -> Result<StoragePlan, String> {
    plan_with_params(primal, adjoint, clones, &[], &HashSet::new(), true)
}

/// [`plan`] with `params` (the `Param` VarIds of [`select_bf16_params`])
/// stored bf16: they and their views are bf16 values, their gradients f32,
/// and a GEMM whose only bf16 operand is one of them is in
/// [`StoragePlan::param_gemm`]. `stubs` are CCR's shape stubs
/// (`CcrPlan::shape_stubs`, `(original, stub)`): the backward reads a stub
/// in place of its original for that value's shape, so a gradient reduced
/// onto a stub is the original's gradient, stored as rule 6 says for the
/// original (as for a clone).
///
/// `rope_grad_bf16` stores a RoPE adjoint that reads a bf16 gradient in bf16
/// when every reader of its result, through views, is a GEMM (rule 4b);
/// `false` keeps it f32 (`NSL_ROPE_BWD_BF16=0`).
pub fn plan_with_params(
    primal: &WengertList,
    adjoint: &WengertList,
    clones: &HashMap<VarId, VarId>,
    stubs: &[(VarId, VarId)],
    params: &HashSet<VarId>,
    rope_grad_bf16: bool,
) -> Result<StoragePlan, String> {
    let slices = find_slices(primal);
    let (attn_seeds, attn_norms) = find_attention_seeds(primal);
    let stand_in: HashMap<VarId, VarId> =
        clones.iter().map(|(&c, &o)| (c, o)).chain(stubs.iter().map(|&(o, s)| (s, o))).collect();
    let mut pl = Planner {
        plan: StoragePlan { slices: slices.len(), attention: attn_norms.len(), ..Default::default() },
        norm_out: HashSet::new(),
        param_view: params.clone(),
        clones,
        stand_in,
        rope_grad_bf16,
    };
    pl.plan.bf16.extend(params.iter().copied());
    let prod = producers(primal);
    for sl in &slices {
        for v in [Some(sl.n), Some(sl.a), sl.s, Some(sl.u), Some(sl.p)].into_iter().flatten() {
            pl.set(prod[&v], Storage::Bf16);
        }
        pl.norm_out.insert(sl.n);
        pl.set(prod[&sl.d], if sl.d_f32 { Storage::F32 } else { Storage::Bf16 });
    }
    for v in attn_seeds {
        pl.set(prod[&v], Storage::Bf16);
    }
    pl.norm_out.extend(attn_norms);
    let rd_adj = readers(adjoint);
    for op in &primal.ops {
        pl.visit(op, false, &rd_adj)?;
    }
    for op in &adjoint.ops {
        pl.visit(op, true, &rd_adj)?;
    }
    // Rule 6 for the planned ops themselves.
    for op in primal.ops.iter().chain(adjoint.ops.iter()) {
        if pl.plan.out.contains_key(&op.result) && !has_dt_lowering(&op.op) {
            return Err(Planner::refuse(op, "this", "planned op without a dtype-explicit lowering"));
        }
    }
    Ok(pl.plan)
}

/// Every binding of `compile_train_block_inner` the storage plan reads.
pub(crate) struct StoragePlanInputs<'a> {
    pub(crate) adjoint: &'a WengertList,
    pub(crate) ccr_plan: &'a Option<crate::ccr::CcrPlan>,
    pub(crate) effective_primal: &'a WengertList,
    /// The train block's `precision=` resolved to bf16.
    pub(crate) precision_bf16: bool,
    /// The train block's `param_dtype=`, when not f32 (admitted already).
    pub(crate) param_dtype: Option<nsl_semantic::train_config::ParamDtype>,
    /// The train block's parameter list, in list order (`param_list`).
    pub(crate) param_paths: &'a [String],
}

impl Compiler<'_> {
    /// Plan bf16 activation storage for a `precision="bf16"` step and install
    /// it for the lowering (`storage_out`). A no-op for an f32 step. Under
    /// `param_dtype=`, also selects the parameters stored bf16 and returns
    /// their `param_list` indices, ascending (empty otherwise).
    pub(crate) fn plan_activation_storage(
        &mut self,
        inputs: StoragePlanInputs<'_>,
    ) -> Result<Vec<usize>, crate::CodegenError> {
        let StoragePlanInputs {
            adjoint,
            ccr_plan,
            effective_primal,
            precision_bf16,
            param_dtype,
            param_paths,
        } = inputs;
        if !precision_bf16 {
            return Ok(Vec::new());
        }
        let empty = HashMap::new();
        let clones = ccr_plan.as_ref().map_or(&empty, |p| &p.recompute_clones);
        let selected = match param_dtype {
            Some(pd) => {
                let index_of = |name: &str| param_paths.iter().position(|p| p == name);
                let sel = select_bf16_params(effective_primal, adjoint, &index_of);
                if sel.is_empty() {
                    return Err(crate::CodegenError::new(format!(
                        "param_dtype=\"{}\" found no parameter it can store in bf16: it stores the weight \
                         matrices whose every reader in the train step is a GEMM (through transposes and \
                         reshapes), and this step has none. Drop param_dtype=",
                        pd.as_str()
                    )));
                }
                sel
            }
            None => Vec::new(),
        };
        let params: HashSet<VarId> = selected.iter().map(|&(v, _)| v).collect();
        let stubs = ccr_plan.as_ref().map_or(&[][..], |p| &p.shape_stubs[..]);
        let planned = plan_with_params(effective_primal, adjoint, clones, stubs, &params, rope_grad_bf16_enabled())
            .map_err(crate::CodegenError::new)?;
        if planned.slices == 0 && planned.attention == 0 {
            // The program's fingerprint (`act=bf16`) and runtime banner were
            // decided from the key before `main`; with nothing planned they
            // would describe storage that does not exist.
            return Err(crate::CodegenError::new(
                "precision=\"bf16\" found no block it can store in bf16: this phase stores the \
                 SwiGLU MLP half of a pre-norm block (rmsnorm -> gate and up projections -> \
                 silu(gate) * up -> down projection -> residual add) and the attention half \
                 (rmsnorm -> q/k/v projections -> RoPE -> scaled-dot-product attention -> output \
                 projection), and this train step has neither. For bf16 GEMMs over f32 activations, build with --matmul-mode bf16 and \
                 leave precision unset",
            ));
        }
        if planned.attention > 0 && self.kernels.flash_attention_context.is_some() {
            // The attention half is stored for the tensor-core kernels with
            // bf16 I/O, which only the decorator-free SDPA dispatch reaches.
            // Under a module's `@flash_attention` context every SDPA lowers
            // through the decomposed f32 chain, or (a CSHA claim, which is
            // published only under that context) the fused CSHA forward:
            // neither reads bf16 operands.
            return Err(crate::CodegenError::new(
                "precision=\"bf16\" stores the attention half for the tensor-core attention \
                 kernels, but this module has a @flash_attention context, under which attention \
                 lowers through the f32 decomposition (or the CSHA fused forward) and cannot read \
                 bf16 operands. Remove the @flash_attention decorator (decorator-free attention \
                 takes the tensor-core kernels) or train with precision=\"f32\"",
            ));
        }
        {
            nsl_log::nsl_log!(INFO, "bf16-storage",
                "[bf16-storage] {} SwiGLU MLP block(s) stored bf16: {} values, {} ops lowered dtype-explicit",
                planned.slices,
                planned.bf16.len(),
                planned.out.len()
            );
            if planned.attention > 0 {
                nsl_log::nsl_log!(INFO, "bf16-storage",
                    "[bf16-storage] {} attention half(s) stored bf16 (tensor-core attention with bf16 I/O)",
                    planned.attention
                );
            }
        }
        let mut indices: Vec<usize> = selected.iter().map(|&(_, i)| i).collect();
        indices.sort_unstable();
        indices.dedup();
        if let Some(pd) = param_dtype {
            nsl_log::nsl_log!(
                INFO,
                "param-storage",
                "[param-storage] {} of {} parameters stored {} ({} of the GEMMs reading them keep the eager \
                 bf16 mode's per-shape operand choice)",
                indices.len(),
                param_paths.len(),
                pd.as_str(),
                planned.param_gemm.len()
            );
        }
        self.storage_param_gemm = planned.param_gemm;
        self.storage_out = planned.out;
        Ok(indices)
    }
}

/// The train blocks `main` runs directly (through decorators).
fn top_level_train_blocks<'a>(stmts: &'a [nsl_ast::stmt::Stmt], out: &mut Vec<&'a nsl_ast::block::TrainBlock>) {
    fn visit<'a>(s: &'a nsl_ast::stmt::Stmt, out: &mut Vec<&'a nsl_ast::block::TrainBlock>) {
        match &s.kind {
            nsl_ast::stmt::StmtKind::TrainBlock(t) => out.push(t),
            nsl_ast::stmt::StmtKind::Decorated { stmt, .. } => visit(stmt, out),
            _ => {}
        }
    }
    for s in stmts {
        visit(s, out);
    }
}

impl Compiler<'_> {
    /// Before `main` is emitted: does a train block ask for
    /// `precision="bf16"`? Then the program's GEMMs run in the bf16 matmul
    /// mode (an explicit `--matmul-mode` that disagrees, or
    /// `NSL_MATMUL_BF16=0`, is a compile error), and the execution
    /// fingerprint and the runtime banner say so. The train block's own
    /// compile re-checks its key (`admit_bf16_activation_storage`).
    pub(crate) fn resolve_program_precision(
        &mut self,
        stmts: &[nsl_ast::stmt::Stmt],
    ) -> Result<(), crate::CodegenError> {
        use nsl_semantic::train_config::{
            ParamDtype, Precision, TrainConfigPurpose, resolve_train_config,
        };
        let mut blocks = Vec::new();
        top_level_train_blocks(stmts, &mut blocks);
        // A block whose config does not resolve is reported by its own
        // compile; here it counts as the default.
        let configs: Vec<_> = blocks
            .iter()
            .map(|t| {
                resolve_train_config(
                    t,
                    &|sym| self.resolve_sym(sym).to_string(),
                    TrainConfigPurpose::UserTrainBlock,
                )
                .ok()
            })
            .collect();
        let precisions: Vec<Precision> = configs
            .iter()
            .map(|c| {
                c.as_ref()
                    .and_then(|c| c.precision)
                    .unwrap_or(Precision::F32)
            })
            .collect();
        // `param_dtype=` is arithmetic and runtime state, decided per program
        // like the precision: every train block of a program must agree.
        let param_dtypes: Vec<ParamDtype> = configs
            .iter()
            .map(|c| {
                c.as_ref()
                    .and_then(|c| c.param_dtype)
                    .unwrap_or(ParamDtype::F32)
            })
            .collect();
        if let Some(&pd) = param_dtypes.iter().find(|&&p| p != ParamDtype::F32) {
            if param_dtypes.iter().any(|&p| p != pd) {
                return Err(crate::CodegenError::new(format!(
                    "one train block asks for param_dtype=\"{}\" and another for a different parameter storage: \
                     the storage is recorded per program (its execution fingerprint and banner), so every train \
                     block of a program must agree",
                    pd.as_str()
                )));
            }
            // Setting the program's storage commits `main` to its banner and
            // fingerprint key, so the preconditions every lowering path
            // shares are checked here, not only in the block-level admission
            // (`admit_param_storage`): the pipelined path returns before that
            // admission, and with neither check a `@pipeline` program
            // declared bf16 parameters and trained f32 ones.
            if precisions
                .iter()
                .zip(&param_dtypes)
                .any(|(&p, &d)| d != ParamDtype::F32 && p != Precision::Bf16)
            {
                return Err(crate::CodegenError::new(format!(
                    "param_dtype=\"{}\" requires precision=\"bf16\": the bf16 storage plan's GEMMs are what read a \
                     bf16 parameter without a cast. Add precision=\"bf16\" or drop param_dtype=",
                    pd.as_str()
                )));
            }
            if self.features.pipeline_config.is_some() {
                return Err(crate::CodegenError::new(format!(
                    "param_dtype=\"{}\" is not supported on the pipelined train path (@pipeline): the parameter \
                     storage plan is not lowered there. Drop one",
                    pd.as_str()
                )));
            }
            self.param_storage = Some(pd);
        }
        if !precisions.contains(&Precision::Bf16) {
            return Ok(());
        }
        if precisions.contains(&Precision::F32) {
            return Err(crate::CodegenError::new(
                "one train block asks for precision=\"bf16\" and another for f32 storage: the GEMM \
                 mode bf16 storage needs is per program, so every train block of a program must \
                 agree. Write precision=\"bf16\" on each, or on none",
            ));
        }
        let mm = &mut self.compile_options.matmul;
        if mm.mode_explicit && mm.mode != crate::MatmulMode::Bf16 {
            return Err(crate::CodegenError::new(format!(
                "precision=\"bf16\" runs the program's GEMMs on bf16 operands, but --matmul-mode {} \
                 was given: drop the flag (precision=\"bf16\" sets the mode) or train with \
                 precision=\"f32\"",
                mm.mode.as_str()
            )));
        }
        if std::env::var("NSL_MATMUL_BF16").ok().as_deref() == Some("0") {
            return Err(crate::CodegenError::new(
                "precision=\"bf16\" runs the program's GEMMs on bf16 operands, but NSL_MATMUL_BF16=0 \
                 is set: unset it (precision=\"bf16\" sets the mode) or train with precision=\"f32\"",
            ));
        }
        mm.mode = crate::MatmulMode::Bf16;
        // The program states its arithmetic: the runtime's environment
        // overrides (NSL_MATMUL_TF32, NSL_MATMUL_PEDANTIC) do not apply, as
        // under an explicit --matmul-mode.
        mm.mode_explicit = true;
        self.activation_bf16 = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::NormEps;

    fn op(result: VarId, op: PrimalOp, inputs: &[VarId]) -> WengertOp {
        WengertOp { id: result, result, op, inputs: inputs.to_vec(), saved_for_backward: false, checkpointed: false }
    }

    fn list(ops: Vec<WengertOp>) -> WengertList {
        WengertList { ops, output: 0, var_names: HashMap::new(), var_types: HashMap::new() }
    }

    fn pt(name: &str) -> PrimalOp {
        PrimalOp::Passthrough(name.to_string())
    }

    const T: PrimalOp = PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX };

    /// One block exactly as the 50M recipe's step extracts it under
    /// `--checkpoint-blocks --checkpoint-selective --fuse-rmsnorm-backward`
    /// (VarIds and op order taken from a real compile's lists): the primal
    /// MLP, and its backward with CCR's clones of `h`, `n`, `s`, `p`.
    fn block() -> (WengertList, WengertList, HashMap<VarId, VarId>) {
        let rms = PrimalOp::RMSNorm { eps: NormEps::Var(131) };
        let primal = list(vec![
            op(18, pt("x_in"), &[]),
            op(126, PrimalOp::Param("wo".into()), &[]),
            op(125, pt("attn_flat"), &[]),
            op(127, PrimalOp::Matmul, &[125, 126]),
            op(129, PrimalOp::Add, &[18, 127]),
            op(130, PrimalOp::Param("ffn_norm.weight".into()), &[]),
            op(131, PrimalOp::Param("ffn_norm.eps".into()), &[]),
            op(132, rms.clone(), &[129, 130, 131]),
            op(133, PrimalOp::Param("w_gate".into()), &[]),
            op(134, PrimalOp::Matmul, &[132, 133]),
            op(135, PrimalOp::Silu, &[134]),
            op(136, PrimalOp::Param("w_up".into()), &[]),
            op(137, PrimalOp::Matmul, &[132, 136]),
            op(138, PrimalOp::Mul, &[135, 137]),
            op(139, PrimalOp::Param("w_down".into()), &[]),
            op(140, PrimalOp::Matmul, &[138, 139]),
            op(144, PrimalOp::Add, &[129, 140]),
        ]);
        let adjoint = list(vec![
            op(750, PrimalOp::Add, &[18, 127]),
            op(751, rms, &[750, 130, 131]),
            op(752, PrimalOp::Silu, &[134]),
            op(753, PrimalOp::Mul, &[752, 137]),
            op(496, pt("reduce_to_shape"), &[493, 750]),
            op(497, pt("reduce_to_shape"), &[493, 809]),
            op(500, T, &[139]),
            op(501, PrimalOp::Matmul, &[497, 500]),
            op(960, PrimalOp::FreeTensor, &[500]),
            op(502, pt("reduce_to_shape"), &[501, 753]),
            op(503, T, &[753]),
            op(504, PrimalOp::Matmul, &[503, 497]),
            op(505, pt("reduce_to_shape"), &[504, 139]),
            op(508, PrimalOp::Mul, &[502, 752]),
            op(509, pt("reduce_to_shape"), &[508, 137]),
            op(510, T, &[136]),
            op(511, PrimalOp::Matmul, &[509, 510]),
            op(512, pt("reduce_to_shape"), &[511, 751]),
            op(513, T, &[751]),
            op(514, PrimalOp::Matmul, &[513, 509]),
            op(515, pt("reduce_to_shape"), &[514, 136]),
            op(516, pt("swiglu_gate_backward"), &[502, 137, 134]),
            op(517, T, &[133]),
            op(518, PrimalOp::Matmul, &[516, 517]),
            op(519, pt("reduce_to_shape"), &[518, 751]),
            op(520, PrimalOp::Add, &[512, 519]),
            op(521, T, &[751]),
            op(522, PrimalOp::Matmul, &[521, 516]),
            op(523, pt("reduce_to_shape"), &[522, 133]),
            op(525, pt("rmsnorm_dx_backward_add:var"), &[520, 750, 130, 496, 131]),
            op(526, pt("rmsnorm_dgamma_backward:var"), &[520, 750, 130, 131]),
        ]);
        let clones = HashMap::from([(750, 129), (751, 132), (752, 135), (753, 138)]);
        (primal, adjoint, clones)
    }

    /// CCR splices a block's clone group at its first reader, which can be
    /// AFTER the down projection's dgrad (that GEMM reads none of the
    /// victims; its reduce onto the clone of `p` does). The dgrad's storage
    /// is decided from its target's (rule 4), the clone of `p`, which must
    /// count as `p` even though the clone has not been visited yet: the plan
    /// is the same as with the clones spliced first.
    #[test]
    fn a_dgrad_before_its_targets_clone_is_planned_as_after_it() {
        let (primal, adjoint, clones) = block();
        let want = plan(&primal, &adjoint, &clones).expect("plans");
        let mut late = adjoint.clone();
        let pos = |l: &WengertList, v: VarId| l.ops.iter().position(|o| o.result == v).unwrap();
        // Move the clones of `s` and `p` (752, 753) to right before the
        // reduce onto `p`'s clone (502), i.e. after the dgrad (501).
        let s_clone = late.ops.remove(pos(&late, 752));
        let p_clone = late.ops.remove(pos(&late, 753));
        let at = pos(&late, 502);
        late.ops.splice(at..at, [s_clone, p_clone]);
        assert!(pos(&late, 501) < pos(&late, 753), "the fixture must put the dgrad first");
        let got = plan(&primal, &late, &clones).expect("plans");
        assert_eq!(got.out.get(&501), Some(&Storage::Bf16), "the dgrad of p's gradient");
        assert_eq!(got.out, want.out);
        assert_eq!(got.bf16, want.bf16);
    }

    /// [`block`] after the SwiGLU fusion: the forward `s = silu(a); p = s*u`
    /// is one `p = swiglu(a, u)`; the backward's up adjoint reads `a`
    /// instead of the recomputed `s`; and, with the recompute folded into
    /// the backward (`fold`), CCR clones nothing for `p` (its shape is read
    /// through a stub) and the down projection's weight gradient reads the
    /// backward's own `h = swiglu(a, u)`.
    fn fused_block(fold: bool) -> (WengertList, WengertList, HashMap<VarId, VarId>, Vec<(VarId, VarId)>) {
        let (mut primal, _, _) = block();
        primal.ops.retain(|o| o.result != 135);
        for o in primal.ops.iter_mut().filter(|o| o.result == 138) {
            o.op = pt("swiglu");
            o.inputs = vec![134, 137];
        }
        let rms = PrimalOp::RMSNorm { eps: NormEps::Var(131) };
        // `p`'s value in the backward: the clone (753) or the fold's `h` (900);
        // its shape: the clone or the stub (809).
        let (value, shape) = if fold { (900, 809) } else { (753, 753) };
        let mut ops = vec![op(750, PrimalOp::Add, &[18, 127]), op(751, rms, &[750, 130, 131])];
        if !fold {
            ops.push(op(753, pt("swiglu"), &[134, 137]));
        }
        ops.extend([
            op(496, pt("reduce_to_shape"), &[493, 750]),
            op(497, pt("reduce_to_shape"), &[493, 808]),
            op(500, T, &[139]),
            op(501, PrimalOp::Matmul, &[497, 500]),
            op(502, pt("reduce_to_shape"), &[501, shape]),
        ]);
        let wgrad = [
            op(503, T, &[value]),
            op(504, PrimalOp::Matmul, &[503, 497]),
            op(505, pt("reduce_to_shape"), &[504, 139]),
        ];
        if !fold {
            ops.extend(wgrad.clone());
        }
        ops.extend([
            op(516, pt("swiglu_gate_backward"), &[502, 137, 134]),
            op(508, pt("swiglu_up_backward"), &[502, 134]),
        ]);
        if fold {
            ops.push(op(900, pt("swiglu"), &[134, 137]));
            ops.extend(wgrad);
        }
        ops.extend([
            op(509, pt("reduce_to_shape"), &[508, 137]),
            op(510, T, &[136]),
            op(511, PrimalOp::Matmul, &[509, 510]),
            op(512, pt("reduce_to_shape"), &[511, 751]),
            op(513, T, &[751]),
            op(514, PrimalOp::Matmul, &[513, 509]),
            op(515, pt("reduce_to_shape"), &[514, 136]),
            op(517, T, &[133]),
            op(518, PrimalOp::Matmul, &[516, 517]),
            op(519, pt("reduce_to_shape"), &[518, 751]),
            op(520, PrimalOp::Add, &[512, 519]),
            op(521, T, &[751]),
            op(522, PrimalOp::Matmul, &[521, 516]),
            op(523, pt("reduce_to_shape"), &[522, 133]),
            op(525, pt("rmsnorm_dx_backward_add:var"), &[520, 750, 130, 496, 131]),
            op(526, pt("rmsnorm_dgamma_backward:var"), &[520, 750, 130, 131]),
        ]);
        let mut clones = HashMap::from([(750, 129), (751, 132)]);
        if !fold {
            clones.insert(753, 138);
        }
        let stubs = if fold { vec![(138, 809)] } else { vec![] };
        (primal, list(ops), clones, stubs)
    }

    /// The fused block stores what the unfused one does, minus `silu(a)`:
    /// the fused op, its clone or the backward's `h`, both adjoints and the
    /// down projection's dgrad bf16, the weight gradients f32. With the fold,
    /// the dgrad's target is `p`'s shape stub, which must count as `p`.
    #[test]
    fn the_fused_mlp_block_stores_what_the_unfused_one_does() {
        use Storage::{Bf16, F32};
        for fold in [false, true] {
            let (primal, adjoint, clones, stubs) = fused_block(fold);
            let got = plan_with_params(&primal, &adjoint, &clones, &stubs, &HashSet::new(), true).expect("plans");
            assert_eq!(got.slices, 1, "fold={fold}");
            let value = if fold { 900 } else { 753 };
            let want_out: HashMap<VarId, Storage> = HashMap::from([
                (132, Bf16), (134, Bf16), (137, Bf16), (138, Bf16), (140, Bf16), (144, F32),
                (751, Bf16), (value, Bf16),
                (501, Bf16), (504, F32), (502, Bf16),
                (516, Bf16), (508, Bf16), (509, Bf16), (511, F32), (514, F32), (518, F32), (522, F32),
            ]);
            assert_eq!(got.out, want_out, "fold={fold}");
            for v in [520, 512, 519, 496, 497, 525] {
                assert!(!got.out.contains_key(&v) && !got.bf16.contains(&v), "fold={fold}: VarId {v} must stay f32");
            }
        }
        // Without the stub's mapping the dgrad (and with it both adjoints)
        // would fall back to f32: the mapping is what keeps the fold from
        // changing the plan.
        let (primal, adjoint, clones, _) = fused_block(true);
        let got = plan_with_params(&primal, &adjoint, &clones, &[], &HashSet::new(), true).expect("plans");
        assert!(!got.bf16.contains(&501) && got.out.get(&516) == Some(&F32), "{:?}", got.out);
    }

    /// Every value's dtype and every dtype-explicit op, derived by hand from
    /// the design's table (3.2) for this block, then compared whole.
    #[test]
    fn the_mlp_block_stores_what_autocast_stores_and_keeps_the_norm_gradient_f32() {
        let (primal, adjoint, clones) = block();
        let got = plan(&primal, &adjoint, &clones).expect("plans");
        assert_eq!(got.slices, 1);
        use Storage::{Bf16, F32};
        let want_out: HashMap<VarId, Storage> = HashMap::from([
            // forward: n, gate, silu, up, silu*up, down bf16; residual add f32
            (132, Bf16), (134, Bf16), (135, Bf16), (137, Bf16), (138, Bf16), (140, Bf16), (144, F32),
            // recompute clones: as their originals (h stays f32: not in the plan)
            (751, Bf16), (752, Bf16), (753, Bf16),
            // dgrad of p bf16 (its gradient is bf16), wgrad of W_down f32
            (501, Bf16), (504, F32),
            // reduce_to_shape of a bf16 gradient: dtype-explicit (identity here)
            (502, Bf16), (509, Bf16),
            // d(up) = dp * s bf16; its dgrad into n f32, W_up's wgrad f32
            (508, Bf16), (511, F32), (514, F32),
            // d(gate) bf16; its dgrad into n f32, W_gate's wgrad f32
            (516, Bf16), (518, F32), (522, F32),
        ]);
        assert_eq!(got.out, want_out);
        let want_bf16: HashSet<VarId> = [
            132, 134, 135, 137, 138, 140, 751, 752, 753, 501, 502, 503, 508, 509, 513, 516, 521,
        ]
        .into_iter()
        .collect();
        assert_eq!(got.bf16, want_bf16);
        // The norm's gradient (520) and the residual gradients stay f32 and
        // keep their ordinary lowering (no entry).
        for v in [520, 512, 519, 496, 497, 525] {
            assert!(!got.out.contains_key(&v) && !got.bf16.contains(&v), "VarId {v} must stay f32");
        }
    }

    /// [`block`] with the gate/up input-gradient sum accumulated by the gate dgrad's
    /// GEMM (`source_ad::fuse_dgrad_accumulate`: `520 = matmul_acc(516, 517, 512)`,
    /// the gate dgrad 518 and its reduce 519 gone).
    fn block_with_dgrad_accumulation() -> (WengertList, WengertList, HashMap<VarId, VarId>) {
        let (primal, mut adjoint, clones) = block();
        adjoint.ops.retain(|o| o.result != 518 && o.result != 519);
        for o in adjoint.ops.iter_mut().filter(|o| o.result == 520) {
            o.op = pt("matmul_acc");
            o.inputs = vec![516, 517, 512];
        }
        (primal, adjoint, clones)
    }

    /// The summed gradient of the norm output stays f32: the accumulation reads a bf16
    /// gradient and is planned f32 (dtype-explicit), the first term's dgrad as before;
    /// everything else is planned as without the accumulation. And a matrix whose
    /// gradient GEMM became an accumulation is still a GEMM-only parameter.
    #[test]
    fn a_dgrad_accumulation_is_planned_f32_and_its_weight_stays_a_gemm_parameter() {
        use Storage::F32;
        let (primal, adjoint, clones) = block();
        let base = plan(&primal, &adjoint, &clones).expect("plans");
        let (primal, adjoint, clones) = block_with_dgrad_accumulation();
        let got = plan(&primal, &adjoint, &clones).expect("plans");
        assert_eq!(got.out.get(&520), Some(&F32));
        assert!(!got.bf16.contains(&520) && !got.param_gemm.contains(&520));
        let mut want = base.out.clone();
        want.remove(&518);
        want.insert(520, F32);
        assert_eq!(got.out, want);
        assert_eq!(
            got.out.get(&511),
            Some(&F32),
            "the first term's dgrad is unchanged"
        );
        let paths = ["wo", "ffn_norm.weight", "w_gate", "w_up", "w_down"];
        let index_of = |n: &str| paths.iter().position(|p| *p == n);
        let sel = select_bf16_params(&primal, &adjoint, &index_of);
        assert_eq!(sel, vec![(126, 0), (133, 2), (136, 3), (139, 4)]);
        // A value whose gradient is bf16: its first term's dgrad and reduce are bf16
        // (rules 4-6), so the accumulation is planned bf16 as the unfused sum was.
        let (primal, mut adjoint, clones) = block_with_dgrad_accumulation();
        let bf16_sum = op(990, pt("matmul_acc"), &[508, 510, 509]);
        adjoint.ops.push(bf16_sum);
        let got_b = plan(&primal, &adjoint, &clones).expect("plans");
        assert_eq!(got_b.out.get(&509), Some(&Storage::Bf16), "the fixture's accumulator is bf16");
        assert_eq!(got_b.out.get(&990), Some(&Storage::Bf16));
        assert!(got_b.bf16.contains(&990) && !got_b.param_gemm.contains(&990));
        // With the parameters stored, the accumulation reads a bf16 gradient as well as
        // the stored weight: the storage route, not the eager rule.
        let params: HashSet<VarId> = [126, 133, 136, 139].into_iter().collect();
        let got = plan_with_params(&primal, &adjoint, &clones, &[], &params, true).expect("plans");
        assert_eq!(got.out.get(&520), Some(&F32));
        assert!(!got.param_gemm.contains(&520));
    }

    /// `param_dtype=`: the four matrices of the MLP block (`wo` too: its only
    /// reader is a GEMM) qualify; the norm gain (an RMSNorm reads it) and a
    /// field outside the parameter list (`ffn_norm.eps`) do not.
    #[test]
    fn the_parameters_only_gemms_read_are_selected() {
        let (primal, adjoint, _) = block();
        let paths = ["wo", "ffn_norm.weight", "w_gate", "w_up", "w_down"];
        let index_of = |n: &str| paths.iter().position(|p| *p == n);
        let sel = select_bf16_params(&primal, &adjoint, &index_of);
        assert_eq!(sel, vec![(126, 0), (133, 2), (136, 3), (139, 4)]);
        // An embedding gather of a matrix keeps it f32.
        let (mut primal, adjoint, _) = block();
        primal.ops.push(op(950, PrimalOp::Embedding, &[133, 18]));
        let sel = select_bf16_params(&primal, &adjoint, &index_of);
        assert!(sel.iter().all(|&(v, _)| v != 133), "{sel:?}");
        assert_eq!(sel.len(), 3);
    }

    /// The plan with the selected parameters stored bf16: they and their
    /// transposes are bf16 values; every weight gradient stays f32 (a
    /// parameter's gradient is f32 whatever rule 6 says, so dropping that
    /// exception plans the wgrads bf16: mutation M7); the GEMMs the
    /// activation plan already lowered keep their storage; and `wo`'s GEMM,
    /// whose other operand is an f32 value, is planned f32 in `param_gemm`
    /// (the eager mode's per-shape rule). Nothing else changes.
    #[test]
    fn stored_parameters_keep_f32_gradients_and_route_new_gemms_through_the_eager_rule() {
        let (primal, adjoint, clones) = block();
        let base = plan(&primal, &adjoint, &clones).expect("plans");
        let params: HashSet<VarId> = [126, 133, 136, 139].into_iter().collect();
        let got = plan_with_params(&primal, &adjoint, &clones, &[], &params, true).expect("plans");
        use Storage::{Bf16, F32};
        for (v, st) in [
            (504, F32),
            (514, F32),
            (522, F32),
            (501, Bf16),
            (511, F32),
            (518, F32),
            (134, Bf16),
            (140, Bf16),
        ] {
            assert_eq!(got.out.get(&v), Some(&st), "VarId {v}");
        }
        assert_eq!(
            got.out.get(&127),
            Some(&F32),
            "wo's GEMM reads a stored parameter"
        );
        assert_eq!(
            got.param_gemm,
            HashSet::from([127]),
            "only wo's GEMM has no bf16 activation operand"
        );
        let mut want_out = base.out.clone();
        want_out.insert(127, F32);
        assert_eq!(
            got.out, want_out,
            "storing the parameters changes no other op"
        );
        let mut want_bf16 = base.bf16.clone();
        want_bf16.extend([126, 133, 136, 139, 500, 510, 517]);
        assert_eq!(got.bf16, want_bf16);
        assert!(base.param_gemm.is_empty());
    }

    /// The 500M recipe's block: `dropout(d)` between the down projection and
    /// the residual add. `d` is written f32 (the dropout kernel reads f32);
    /// everything before it as in the dropout-free block.
    #[test]
    fn a_dropout_tail_keeps_the_down_output_f32_and_the_rest_bf16() {
        let (mut primal, adjoint, clones) = block();
        let add = primal.ops.pop().expect("residual add");
        assert_eq!(add.result, 144);
        primal.ops.push(op(141, PrimalOp::DropoutMask { p: 0.1 }, &[140]));
        primal.ops.push(op(143, PrimalOp::Dropout { p: 0.1 }, &[140, 141]));
        primal.ops.push(op(144, PrimalOp::Add, &[129, 143]));
        let got = plan(&primal, &adjoint, &clones).expect("plans");
        assert_eq!(got.slices, 1);
        assert_eq!(got.out.get(&140), Some(&Storage::F32), "the down GEMM writes f32 from bf16 p");
        assert!(!got.bf16.contains(&140) && !got.bf16.contains(&143));
        for v in [132, 134, 135, 137, 138, 501, 508, 516] {
            assert_eq!(got.out.get(&v), Some(&Storage::Bf16), "VarId {v}");
        }
        // The residual add reads two f32 values: ordinary lowering.
        assert_eq!(got.out.get(&144), None);
    }

    #[test]
    fn a_block_whose_norm_output_has_another_reader_stays_f32() {
        let (mut primal, adjoint, clones) = block();
        primal.ops.push(op(900, PrimalOp::Relu, &[132]));
        let got = plan(&primal, &adjoint, &clones).expect("plans");
        assert_eq!(got.slices, 0);
        assert!(got.out.is_empty() && got.bf16.is_empty());
    }

    #[test]
    fn an_unlowerable_reader_of_a_bf16_value_is_a_compile_error() {
        let (primal, mut adjoint, clones) = block();
        adjoint.ops.push(op(901, PrimalOp::Softmax { dim: -1 }, &[503]));
        let err = plan(&primal, &adjoint, &clones).expect_err("must refuse");
        assert!(err.contains("Softmax") && err.contains("901"), "{err}");
    }

    #[test]
    fn a_clone_takes_its_originals_dtype_even_when_its_inputs_would_say_f32() {
        // A full-block recompute clones the norm from the (f32) residual: by
        // its inputs it would be f32; it must be bf16 like the original.
        let (primal, adjoint, clones) = block();
        let got = plan(&primal, &adjoint, &clones).expect("plans");
        assert_eq!(got.out.get(&751), Some(&Storage::Bf16));
        let mut no_clone_map = clones.clone();
        no_clone_map.remove(&751);
        let got = plan(&primal, &adjoint, &no_clone_map).expect("plans");
        assert_eq!(got.out.get(&751), None, "without the clone map the norm clone would not be narrowed");
    }

    #[test]
    fn an_f32_program_plans_nothing() {
        let (primal, adjoint, _) = block();
        let mut primal = primal;
        // Break the slice: silu's output read twice.
        primal.ops.push(op(902, PrimalOp::Add, &[135, 135]));
        let got = plan(&primal, &adjoint, &HashMap::new()).expect("plans");
        assert_eq!(got, StoragePlan::default());
    }

    /// One attention half exactly as `nsl.nn.gqa.GroupedQueryAttention`
    /// extracts under the production flags (`--source-ad --checkpoint-blocks
    /// --checkpoint-selective --fuse-rmsnorm-backward --fuse-wgrad-accum`;
    /// VarIds and op order taken from a real compile's lists, d_model 128,
    /// two query heads over one KV head, head_dim 64, the fused `rope`):
    ///
    /// ```text
    /// x (2) -> attn_norm (6) -> q/k/v GEMMs (31, 33, 35) -> head views
    ///   -> rope (52 q, 58 k) -> GQA reshape/expand/contiguous/reshape (66 k, 74 v)
    ///   -> SDPA (80) -> transpose/reshape (84) -> Wo (86) -> residual add (88)
    /// ```
    ///
    /// and its backward. This step has no recompute clones (CCR clones only
    /// the looped blocks; the MLP test above covers clone dtypes).
    fn attention_block() -> (WengertList, WengertList) {
        let rms = PrimalOp::RMSNorm { eps: NormEps::Var(5) };
        let primal = list(vec![
            op(0, PrimalOp::Input("batch".into()), &[]),
            op(1, PrimalOp::Input("m".into()), &[]),
            op(2, PrimalOp::Input("x".into()), &[]),
            op(3, PrimalOp::Input("y".into()), &[]),
            op(4, PrimalOp::Param("m.attn_norm.weight".into()), &[]),
            op(5, PrimalOp::Param("m.attn_norm.eps".into()), &[]),
            op(6, rms.clone(), &[2, 4, 5]),
            op(7, PrimalOp::Constant(1.0), &[]),
            op(8, pt("shape"), &[6]),
            op(9, PrimalOp::Constant(0.0), &[]),
            op(10, pt("subscript"), &[8, 9]),
            op(11, PrimalOp::Constant(1.0), &[]),
            op(12, pt("subscript"), &[8, 11]),
            op(13, PrimalOp::Param("m.attn._d_model".into()), &[]),
            op(14, PrimalOp::Constant(128.0), &[]),
            op(15, pt("int"), &[14]),
            op(16, PrimalOp::Param("m.attn._n_heads".into()), &[]),
            op(17, PrimalOp::Constant(2.0), &[]),
            op(18, pt("int"), &[17]),
            op(19, PrimalOp::Param("m.attn._n_kv_heads".into()), &[]),
            op(20, PrimalOp::Constant(1.0), &[]),
            op(21, pt("int"), &[20]),
            op(22, PrimalOp::Param("m.attn._head_dim".into()), &[]),
            op(23, PrimalOp::Constant(64.0), &[]),
            op(24, pt("int"), &[23]),
            op(25, PrimalOp::Param("m.attn._n_rep".into()), &[]),
            op(26, PrimalOp::Constant(2.0), &[]),
            op(27, pt("int"), &[26]),
            op(28, PrimalOp::Param("m.attn._dropout_p".into()), &[]),
            op(29, PrimalOp::Constant(0.0), &[]),
            op(30, PrimalOp::Param("m.attn.wq".into()), &[]),
            op(31, PrimalOp::Matmul, &[6, 30]),
            op(32, PrimalOp::Param("m.attn.wk".into()), &[]),
            op(33, PrimalOp::Matmul, &[6, 32]),
            op(34, PrimalOp::Param("m.attn.wv".into()), &[]),
            op(35, PrimalOp::Matmul, &[6, 34]),
            op(36, pt("list"), &[10, 12, 18, 24]),
            op(37, pt("reshape"), &[31, 36]),
            op(38, PrimalOp::Transpose { dim0: 1, dim1: 2 }, &[37]),
            op(39, pt("list"), &[10, 12, 21, 24]),
            op(40, pt("reshape"), &[33, 39]),
            op(41, PrimalOp::Transpose { dim0: 1, dim1: 2 }, &[40]),
            op(42, pt("list"), &[10, 12, 21, 24]),
            op(43, pt("reshape"), &[35, 42]),
            op(44, PrimalOp::Transpose { dim0: 1, dim1: 2 }, &[43]),
            op(45, PrimalOp::Constant(0.0), &[]),
            op(46, pt("float"), &[12]),
            op(47, pt("arange"), &[45, 46]),
            op(48, PrimalOp::Param("m.attn.rope._cos_table".into()), &[]),
            op(49, PrimalOp::Embedding, &[48, 47]),
            op(50, PrimalOp::Param("m.attn.rope._sin_table".into()), &[]),
            op(51, PrimalOp::Embedding, &[50, 47]),
            op(52, pt("rope"), &[38, 49, 51]),
            op(53, PrimalOp::Constant(0.0), &[]),
            op(54, pt("float"), &[12]),
            op(55, pt("arange"), &[53, 54]),
            op(56, PrimalOp::Embedding, &[48, 55]),
            op(57, PrimalOp::Embedding, &[50, 55]),
            op(58, pt("rope"), &[41, 56, 57]),
            op(59, PrimalOp::Constant(1.0), &[]),
            op(60, pt("list"), &[10, 21, 59, 12, 24]),
            op(61, pt("reshape"), &[58, 60]),
            op(62, pt("list"), &[10, 21, 27, 12, 24]),
            op(63, pt("expand"), &[61, 62]),
            op(64, pt("contiguous"), &[63]),
            op(65, pt("list"), &[10, 18, 12, 24]),
            op(66, pt("reshape"), &[64, 65]),
            op(67, PrimalOp::Constant(1.0), &[]),
            op(68, pt("list"), &[10, 21, 67, 12, 24]),
            op(69, pt("reshape"), &[44, 68]),
            op(70, pt("list"), &[10, 21, 27, 12, 24]),
            op(71, pt("expand"), &[69, 70]),
            op(72, pt("contiguous"), &[71]),
            op(73, pt("list"), &[10, 18, 12, 24]),
            op(74, pt("reshape"), &[72, 73]),
            op(75, PrimalOp::Constant(1.0), &[]),
            op(76, pt("float"), &[24]),
            op(77, PrimalOp::Sqrt, &[76]),
            op(78, PrimalOp::Div, &[75, 77]),
            op(79, PrimalOp::Constant(1.0), &[]),
            op(80, PrimalOp::ScaledDotProductAttention { causal: true }, &[52, 66, 74, 78, 79]),
            op(81, PrimalOp::Transpose { dim0: 1, dim1: 2 }, &[80]),
            op(82, PrimalOp::Mul, &[18, 24]),
            op(83, pt("list"), &[10, 12, 82]),
            op(84, pt("reshape"), &[81, 83]),
            op(85, PrimalOp::Param("m.attn.wo".into()), &[]),
            op(86, PrimalOp::Matmul, &[84, 85]),
            op(88, PrimalOp::Add, &[2, 86]),
            op(89, PrimalOp::MSELoss, &[88, 3]),
        ]);
        let adjoint = list(vec![
            op(90, PrimalOp::Constant(1.0), &[]),
            op(91, pt("mse_backward"), &[90, 88, 3]),
            op(92, pt("reduce_to_shape"), &[91, 88]),
            op(97, pt("reduce_to_shape"), &[92, 86]),
            op(98, T, &[85]),
            op(99, PrimalOp::Matmul, &[97, 98]),
            op(100, pt("reduce_to_shape"), &[99, 84]),
            op(101, T, &[84]),
            op(102, PrimalOp::Matmul, &[101, 97]),
            op(103, pt("reduce_to_shape"), &[102, 85]),
            op(104, pt("shape"), &[81]),
            op(105, pt("reshape"), &[100, 104]),
            op(110, PrimalOp::Transpose { dim0: 1, dim1: 2 }, &[105]),
            op(111, PrimalOp::FlashAttentionBackwardExtract { causal: true, component: 0 }, &[110, 52, 66, 74, 80, 78]),
            op(112, PrimalOp::FlashAttentionBackwardExtract { causal: true, component: 1 }, &[110, 52, 66, 74, 80, 78]),
            op(113, PrimalOp::FlashAttentionBackwardExtract { causal: true, component: 2 }, &[110, 52, 66, 74, 80, 78]),
            op(126, pt("shape"), &[72]),
            op(127, pt("reshape"), &[113, 126]),
            op(129, pt("reduce_to_shape"), &[127, 69]),
            op(131, pt("shape"), &[44]),
            op(132, pt("reshape"), &[129, 131]),
            op(135, pt("shape"), &[64]),
            op(136, pt("reshape"), &[112, 135]),
            op(138, pt("reduce_to_shape"), &[136, 61]),
            op(140, pt("shape"), &[58]),
            op(141, pt("reshape"), &[138, 140]),
            op(144, pt("rope_backward"), &[141, 56, 57]),
            op(152, pt("rope_backward"), &[111, 49, 51]),
            op(162, PrimalOp::Transpose { dim0: 1, dim1: 2 }, &[132]),
            op(163, pt("shape"), &[35]),
            op(164, pt("reshape"), &[162, 163]),
            op(166, PrimalOp::Transpose { dim0: 1, dim1: 2 }, &[144]),
            op(167, pt("shape"), &[33]),
            op(168, pt("reshape"), &[166, 167]),
            op(170, PrimalOp::Transpose { dim0: 1, dim1: 2 }, &[152]),
            op(171, pt("shape"), &[31]),
            op(172, pt("reshape"), &[170, 171]),
            op(174, T, &[34]),
            op(175, PrimalOp::Matmul, &[164, 174]),
            op(176, pt("reduce_to_shape"), &[175, 6]),
            op(177, T, &[6]),
            op(178, PrimalOp::Matmul, &[177, 164]),
            op(179, pt("reduce_to_shape"), &[178, 34]),
            op(180, T, &[32]),
            op(181, PrimalOp::Matmul, &[168, 180]),
            op(182, pt("reduce_to_shape"), &[181, 6]),
            op(184, T, &[6]),
            op(185, PrimalOp::Matmul, &[184, 168]),
            op(186, pt("reduce_to_shape"), &[185, 32]),
            op(187, T, &[30]),
            op(188, PrimalOp::Matmul, &[172, 187]),
            op(189, pt("reduce_to_shape"), &[188, 6]),
            op(190, pt("fused_ew:v1:add(i0,i1);add(p0,i2)"), &[176, 182, 189]),
            op(191, T, &[6]),
            op(192, PrimalOp::Matmul, &[191, 172]),
            op(193, pt("reduce_to_shape"), &[192, 30]),
            op(217, pt("rmsnorm_dgamma_backward:var"), &[190, 2, 4, 5]),
        ]);
        (primal, adjoint)
    }

    /// Every value's dtype and every dtype-explicit op of the attention half,
    /// derived by hand from the design (step 5, option (a)) and compared whole.
    #[test]
    fn the_attention_half_stores_its_operands_bf16_and_keeps_the_norm_gradient_f32() {
        let (primal, adjoint) = attention_block();
        let got = plan(&primal, &adjoint, &HashMap::new()).expect("plans");
        assert_eq!((got.slices, got.attention), (0, 1));
        use Storage::{Bf16, F32};
        let want_out: HashMap<VarId, Storage> = HashMap::from([
            // forward seeds: the norm, the q/k/v GEMMs, both RoPE results and
            // Wo (it feeds the residual add directly)
            (6, Bf16), (31, Bf16), (33, Bf16), (35, Bf16), (52, Bf16), (58, Bf16), (86, Bf16),
            // the residual add reads f32 x and bf16 Wo: written f32
            (88, F32),
            // dgrad into the attention output bf16 (its gradient is bf16), Wo's wgrad f32
            (99, Bf16), (102, F32),
            // reduce_to_shape of bf16 gradients: the identity onto the
            // attention output, GQA's group sums onto the KV head views
            (100, Bf16), (129, Bf16), (138, Bf16),
            // rule 4b: the RoPE adjoints read bf16 dQ/dK and only the k and q
            // projections' GEMMs read their results (through the head
            // views' adjoints): written bf16
            (144, Bf16), (152, Bf16),
            // dgrads into the norm output f32 (summed in f32 and read by the
            // f32 norm backward), each reading a bf16 gradient; the four
            // wgrads f32
            (175, F32), (178, F32), (181, F32), (185, F32), (188, F32), (192, F32),
        ]);
        assert_eq!(got.out, want_out);
        let want_bf16: HashSet<VarId> = [
            // forward: seeds, then views and copies of them, and SDPA's O
            6, 31, 33, 35, 37, 38, 40, 41, 43, 44, 52, 58, 61, 63, 64, 66, 69, 71, 72, 74, 80, 81, 84, 86,
            // backward: dO and its views, dQ/dK/dV and their views and
            // sums, the RoPE adjoints and their views, and the transposed
            // norm output the wgrads read
            99, 100, 101, 105, 110, 111, 112, 113, 127, 129, 132, 136, 138, 141, 144, 152, 162, 164, 166, 168, 170,
            172, 177, 184, 191,
        ]
        .into_iter()
        .collect();
        assert_eq!(got.bf16, want_bf16);
        // The norm output's gradient: three f32 dgrads, their identity
        // reductions and the fused sum the dgamma kernel reads. SDPA's
        // log-sum-exp is not a list value (the kernel keeps it f32 in the
        // side channel the backward reads); the gradient arriving from the
        // residual stays f32 until the Wo dgrad.
        for v in [175, 176, 181, 182, 188, 189, 190, 217, 91, 92, 97] {
            assert!(!got.bf16.contains(&v), "VarId {v} must stay f32");
        }
    }

    /// `NSL_ROPE_BWD_BF16=0` (rule 4b off): the RoPE adjoints are written
    /// f32 and the q/k projections' GEMMs read them as f32 values (their
    /// ordinary lowering), the plan before rule 4b; everything else as
    /// above. A RoPE adjoint that something other than a GEMM reads stays f32
    /// with the rule on.
    #[test]
    fn rule_4b_off_or_a_non_gemm_reader_keeps_the_rope_adjoint_f32() {
        let (primal, adjoint) = attention_block();
        let on = plan(&primal, &adjoint, &HashMap::new()).expect("plans");
        let off = plan_with_params(&primal, &adjoint, &HashMap::new(), &[], &HashSet::new(), false).expect("plans");
        use Storage::{Bf16, F32};
        for v in [144, 152] {
            assert_eq!((on.out.get(&v), off.out.get(&v)), (Some(&Bf16), Some(&F32)), "VarId {v}");
        }
        // q's and k's GEMMs read only f32 values without rule 4b: no
        // dtype-explicit lowering.
        for v in [181, 188] {
            assert!(!off.out.contains_key(&v), "VarId {v}");
        }
        for v in [144, 152, 166, 168, 170, 172] {
            assert!(!off.bf16.contains(&v), "VarId {v} must be f32 with rule 4b off");
        }
        // Everything else is the same plan.
        let strip = |p: &StoragePlan| -> (Vec<(VarId, Storage)>, Vec<VarId>) {
            let skip = [144, 152, 166, 168, 170, 172, 181, 188];
            let mut o: Vec<_> = p.out.iter().filter(|(v, _)| !skip.contains(v)).map(|(&v, &s)| (v, s)).collect();
            let mut b: Vec<_> = p.bf16.iter().filter(|v| !skip.contains(v)).copied().collect();
            o.sort_unstable_by_key(|x| x.0);
            b.sort_unstable();
            (o, b)
        };
        assert_eq!(strip(&on), strip(&off));
        // k's adjoint also read by a non-GEMM (an elementwise op): f32.
        let mut adj2 = adjoint.clone();
        adj2.ops.push(op(300, PrimalOp::Mul, &[166, 166]));
        let p2 = plan(&primal, &adj2, &HashMap::new()).expect("plans");
        assert_eq!(p2.out.get(&144), Some(&F32));
        assert_eq!(p2.out.get(&152), Some(&Bf16));
    }

    /// The 500M recipe's attention: `dropout(Wo)` before the residual add.
    /// Wo is written f32 (the dropout kernel reads f32), the dropout's
    /// adjoint multiplies f32, and the rest of the half is as without it.
    #[test]
    fn an_attention_dropout_tail_keeps_the_output_projection_f32_and_the_rest_bf16() {
        let (mut primal, mut adjoint) = attention_block();
        let mse = primal.ops.pop().expect("loss");
        let add = primal.ops.pop().expect("residual add");
        assert_eq!((add.result, mse.result), (88, 89));
        primal.ops.push(op(900, PrimalOp::DropoutMask { p: 0.1 }, &[86]));
        primal.ops.push(op(901, PrimalOp::Dropout { p: 0.1 }, &[86, 900]));
        primal.ops.push(op(88, PrimalOp::Add, &[2, 901]));
        primal.ops.push(mse);
        // The real backward (as a compile emits it): the gradient reduced
        // onto the dropout output, scaled by the mask, then into Wo's GEMMs.
        let at = adjoint.ops.iter().position(|o| o.result == 97).expect("dY");
        adjoint.ops[at] = op(97, pt("reduce_to_shape"), &[92, 901]);
        adjoint.ops.insert(at + 1, op(902, pt("fused_ew:v1:mul(i0,i1);mul(p0,x3f8e38e4)"), &[97, 900]));
        for o in adjoint.ops.iter_mut().filter(|o| matches!(o.result, 99 | 102)) {
            for i in o.inputs.iter_mut().filter(|i| **i == 97) {
                *i = 902;
            }
        }
        let got = plan(&primal, &adjoint, &HashMap::new()).expect("plans");
        assert_eq!((got.slices, got.attention), (0, 1));
        use Storage::{Bf16, F32};
        let want_out: HashMap<VarId, Storage> = HashMap::from([
            (6, Bf16), (31, Bf16), (33, Bf16), (35, Bf16), (52, Bf16), (58, Bf16),
            // Wo: bf16 operand, f32 output; the mask, the dropout and the
            // residual add read f32 only (ordinary lowering)
            (86, F32),
            (99, Bf16), (102, F32), (100, Bf16), (129, Bf16), (138, Bf16),
            (144, Bf16), (152, Bf16), (175, F32), (178, F32), (181, F32), (185, F32), (188, F32), (192, F32),
        ]);
        assert_eq!(got.out, want_out);
        let want_bf16: HashSet<VarId> = [
            6, 31, 33, 35, 37, 38, 40, 41, 43, 44, 52, 58, 61, 63, 64, 66, 69, 71, 72, 74, 80, 81, 84,
            99, 100, 101, 105, 110, 111, 112, 113, 127, 129, 132, 136, 138, 141, 144, 152, 162, 164, 166, 168, 170,
            172, 177, 184, 191,
        ]
        .into_iter()
        .collect();
        assert_eq!(got.bf16, want_bf16);
        for v in [86, 900, 901, 88, 97, 902] {
            assert!(!got.bf16.contains(&v), "VarId {v} must stay f32");
        }

        // The same half with the dropout backward that redraws its mask from
        // the key (`NSL_DROPOUT_REGEN`, the default) in place of the fused
        // multiply chain: one op reading the f32 gradient and the key, f32
        // out. The plan is identical.
        let regen = crate::source_ad::dropout_backward_name(0.1);
        let at = adjoint
            .ops
            .iter()
            .position(|o| o.result == 902)
            .expect("the dropout backward");
        adjoint.ops[at] = op(902, pt(&regen), &[97, 900]);
        let with_regen = plan(&primal, &adjoint, &HashMap::new()).expect("plans");
        assert_eq!(with_regen, got);
    }

    /// The dropout backward reads an f32 gradient; one that reads a bf16
    /// value is a compile error (rule 6), not a silent widen.
    #[test]
    fn a_dropout_backward_reading_a_bf16_value_is_a_compile_error() {
        let (primal, mut adjoint, clones) = block();
        let regen = crate::source_ad::dropout_backward_name(0.1);
        adjoint.ops.push(op(903, pt(&regen), &[503, 132]));
        let err = plan(&primal, &adjoint, &clones).expect_err("must refuse");
        assert!(
            err.contains("dropout_backward") && err.contains("903"),
            "{err}"
        );
    }
}
