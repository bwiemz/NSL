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
}

/// One admitted SwiGLU slice (primal VarIds).
#[derive(Debug, Clone, Copy)]
struct Slice {
    n: VarId,
    a: VarId,
    s: VarId,
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
        if !matches!(op.op, PrimalOp::Silu) || op.inputs.len() != 1 {
            continue;
        }
        let (s, a) = (op.result, op.inputs[0]);
        let Some(n) = matmul_of_param(&prod, a) else { continue };
        let Some(p_op) = only_reader(&rd, s) else { continue };
        if !matches!(p_op.op, PrimalOp::Mul) || p_op.inputs.len() != 2 {
            continue;
        }
        let u = if p_op.inputs[0] == s { p_op.inputs[1] } else { p_op.inputs[0] };
        if u == s || !p_op.inputs.contains(&s) || matmul_of_param(&prod, u) != Some(n) {
            continue;
        }
        let p = p_op.result;
        let (Some(a_r), Some(u_r)) = (only_reader(&rd, a), only_reader(&rd, u)) else { continue };
        if a_r.result != s || u_r.result != p {
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
}

struct Planner<'a> {
    plan: StoragePlan,
    /// Values whose GRADIENT is planned bf16 (and their clones).
    grad_bf16: HashSet<VarId>,
    clones: &'a HashMap<VarId, VarId>,
}

impl Planner<'_> {
    fn is_bf16(&self, v: VarId) -> bool {
        self.plan.bf16.contains(&v)
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

    /// Rules 2-6 for one op of the forward (`in_adjoint` false) or the
    /// backward.
    fn visit(&mut self, op: &WengertOp, in_adjoint: bool, rd_adj: &HashMap<VarId, Vec<&WengertOp>>) -> Result<(), String> {
        let list = if in_adjoint { "backward" } else { "forward" };
        if matches!(op.op, PrimalOp::FreeTensor) || self.plan.out.contains_key(&op.result) {
            return Ok(());
        }
        let reads_bf16 = op.inputs.iter().any(|&i| self.is_bf16(i));
        // Rule 2: a recompute clone stores what its original stores.
        if let Some(orig) = self.clones.get(&op.result) {
            if self.grad_bf16.contains(orig) {
                self.grad_bf16.insert(op.result);
            }
            if self.is_bf16(*orig) {
                if !has_dt_lowering(&op.op) {
                    return Err(Self::refuse(op, list, "a recompute clone of a bf16 value whose op has no bf16 lowering"));
                }
                self.set(op, Storage::Bf16);
                return Ok(());
            }
        }
        match &op.op {
            PrimalOp::Transpose { .. } => {
                if reads_bf16 {
                    self.plan.bf16.insert(op.result);
                }
                Ok(())
            }
            PrimalOp::Passthrough(name) if name == "shape" => Ok(()),
            PrimalOp::Passthrough(name) if name == "reduce_to_shape" => {
                let (g, like) = (op.inputs[0], op.inputs[1]);
                if self.is_bf16(g) {
                    // Identity only: the gradient of a planned value onto that value.
                    if !self.is_bf16(like) {
                        return Err(Self::refuse(op, list, "a bf16 gradient reduced onto a value that is not a planned bf16 activation"));
                    }
                    self.plan.bf16.insert(op.result);
                }
                // An f32 gradient reduced onto a bf16 value reads only its shape.
                Ok(())
            }
            PrimalOp::Matmul => {
                // Rule 4: the gradient of a value whose gradient is bf16.
                let target = only_reader(rd_adj, op.result)
                    .filter(|r| is_passthrough(&r.op, "reduce_to_shape") && r.inputs[0] == op.result)
                    .map(|r| r.inputs[1]);
                if in_adjoint && target.is_some_and(|t| self.grad_bf16.contains(&t)) {
                    self.set(op, Storage::Bf16);
                } else if reads_bf16 {
                    self.set(op, Storage::F32);
                }
                Ok(())
            }
            o if is_binary(o) || is_unary_dt(o) || is_passthrough(o, "swiglu_gate_backward") => {
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
pub fn plan(primal: &WengertList, adjoint: &WengertList, clones: &HashMap<VarId, VarId>) -> Result<StoragePlan, String> {
    let slices = find_slices(primal);
    let mut pl = Planner { plan: StoragePlan { slices: slices.len(), ..Default::default() }, grad_bf16: HashSet::new(), clones };
    let prod = producers(primal);
    for sl in &slices {
        for v in [sl.n, sl.a, sl.s, sl.u, sl.p] {
            pl.set(prod[&v], Storage::Bf16);
        }
        pl.grad_bf16.extend([sl.a, sl.s, sl.u, sl.p]);
        if sl.d_f32 {
            pl.set(prod[&sl.d], Storage::F32);
        } else {
            pl.set(prod[&sl.d], Storage::Bf16);
            pl.grad_bf16.insert(sl.d);
        }
    }
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
}

impl Compiler<'_> {
    /// Plan bf16 activation storage for a `precision="bf16"` step and install
    /// it for the lowering (`storage_out`). A no-op for an f32 step.
    pub(crate) fn plan_activation_storage(&mut self, inputs: StoragePlanInputs<'_>) -> Result<(), crate::CodegenError> {
        let StoragePlanInputs { adjoint, ccr_plan, effective_primal, precision_bf16 } = inputs;
        if !precision_bf16 {
            return Ok(());
        }
        let empty = HashMap::new();
        let clones = ccr_plan.as_ref().map_or(&empty, |p| &p.recompute_clones);
        let planned = plan(effective_primal, adjoint, clones).map_err(crate::CodegenError::new)?;
        if planned.slices == 0 {
            nsl_log::nsl_log!(WARN, "bf16-storage",
                "[bf16-storage] precision=\"bf16\": no SwiGLU MLP block (rmsnorm -> gate/up -> silu * up -> \
                 down -> residual add) was found, so every activation stays f32; only the GEMMs run in bf16"
            );
        } else {
            nsl_log::nsl_log!(INFO, "bf16-storage",
                "[bf16-storage] {} SwiGLU MLP block(s) stored bf16: {} values, {} ops lowered dtype-explicit",
                planned.slices,
                planned.bf16.len(),
                planned.out.len()
            );
        }
        self.storage_out = planned.out;
        Ok(())
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
    pub(crate) fn resolve_program_precision(&mut self, stmts: &[nsl_ast::stmt::Stmt]) -> Result<(), crate::CodegenError> {
        use nsl_semantic::train_config::{resolve_train_config, Precision, TrainConfigPurpose};
        let mut blocks = Vec::new();
        top_level_train_blocks(stmts, &mut blocks);
        // A block whose config does not resolve is reported by its own
        // compile; here it counts as the default.
        let precisions: Vec<Precision> = blocks
            .iter()
            .map(|t| {
                resolve_train_config(t, &|sym| self.resolve_sym(sym).to_string(), TrainConfigPurpose::UserTrainBlock)
                    .ok()
                    .and_then(|c| c.precision)
                    .unwrap_or(Precision::F32)
            })
            .collect();
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
        adjoint.ops.push(op(901, pt("contiguous"), &[503]));
        let err = plan(&primal, &adjoint, &clones).expect_err("must refuse");
        assert!(err.contains("contiguous") && err.contains("901"), "{err}");
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
}
