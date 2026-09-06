// crates/nsl-codegen/src/kir_verify.rs
//! The KIR verifier (roadmap A2, step 2): the structural and typing rules
//! every `KernelIR` must satisfy before a backend prints it.
//!
//! The roadmap's finding was that the hand-PTX path has no verifier, so a
//! wrong register or offset ships as a silent numerical bug. KIR is the
//! sanctioned path new kernels take, and until now its only check was
//! "every block has a terminator" (`KernelIR::is_well_formed`). This
//! module checks what an SSA IR promises:
//!
//!   1. **Shape.** At least one block; block ids are their positions;
//!      every block has a terminator; every branch target exists.
//!   2. **SSA.** Every `VarId` is defined at most once — by a parameter or
//!      by exactly one op's destination.
//!   3. **Def-before-use.** Every operand is a parameter, or is defined
//!      earlier in the same block, or is defined in a block that strictly
//!      dominates the using block (dominators computed over the block
//!      CFG from block 0, the entry).
//!   4. **Typing.** Where `var_types` records both sides, an op's operands
//!      agree with its destination: arithmetic and math ops are
//!      homogeneous, `Cast` lands in its target type, `Load`/`Store` move
//!      the pointee type through a `Ptr`, `Cmp` produces `Bool`, `Select`
//!      picks between operands of the destination's type from a `Bool`,
//!      `Const` matches its literal, thread-index ops produce `U32`,
//!      `PtrOffset` keeps the base's pointer type.
//!
//! Untyped variables (`KirBuilder::new_var`) are exempt from rule 4 only;
//! the FPGA-only structured ops (`Matmul`, `ElementwiseAdd`, `Relu`) carry
//! their dtypes inline and are checked for rules 1–3 only.
//!
//! `kernel_lower::lower_kernel_to_ir` — the AST → KIR front door — runs
//! the verifier and refuses a kernel that fails it with a `CodegenError`
//! listing every violation, so a lowering bug surfaces at compile time
//! rather than as a wrong answer. `KernelIR::verify` is the same check for
//! any other producer.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::kernel_ir::{BlockId, KernelIR, KirConst, KirOp, KirTerminator, KirType, VarId};

/// One violation. `block` / `op_index` locate the offending op (the
/// terminator is reported with `op_index == usize::MAX`).
#[derive(Debug, Clone, PartialEq)]
pub enum KirVerifyError {
    /// The kernel has no blocks at all.
    NoBlocks,
    /// `blocks[i].id != i` — the builder's invariant, on which branch
    /// targets and dominance rely.
    BlockIdMismatch { index: usize, id: BlockId },
    MissingTerminator { block: BlockId },
    BadBranchTarget { block: BlockId, target: BlockId },
    /// A `VarId` defined twice (parameter and op, or two ops).
    Redefined { var: VarId, block: BlockId, op_index: usize },
    /// An operand that is never defined anywhere.
    Undefined { var: VarId, block: BlockId, op_index: usize },
    /// An operand defined later in the same block, or in a block that does
    /// not dominate the use.
    UseBeforeDef { var: VarId, block: BlockId, op_index: usize },
    /// A typed operand disagrees with the op's typing rule. `role` names
    /// the operand (`"dst"`, `"a"`, `"pointee"`, ...).
    TypeMismatch {
        var: VarId,
        block: BlockId,
        op_index: usize,
        role: &'static str,
        expected: KirType,
        found: KirType,
    },
    /// A memory op's address operand is typed but is not a pointer.
    NotAPointer { var: VarId, block: BlockId, op_index: usize, found: KirType },
}

impl fmt::Display for KirVerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KirVerifyError::NoBlocks => write!(f, "kernel has no blocks"),
            KirVerifyError::BlockIdMismatch { index, id } => {
                write!(f, "block at index {index} carries id {id}")
            }
            KirVerifyError::MissingTerminator { block } => {
                write!(f, "block {block} has no terminator")
            }
            KirVerifyError::BadBranchTarget { block, target } => {
                write!(f, "block {block} branches to non-existent block {target}")
            }
            KirVerifyError::Redefined { var, block, op_index } => {
                write!(f, "v{var} is defined again at block {block} op {op_index}")
            }
            KirVerifyError::Undefined { var, block, op_index } => {
                write!(f, "v{var} used at block {block} op {op_index} is never defined")
            }
            KirVerifyError::UseBeforeDef { var, block, op_index } => write!(
                f,
                "v{var} used at block {block} op {op_index} before its definition reaches it"
            ),
            KirVerifyError::TypeMismatch { var, block, op_index, role, expected, found } => {
                write!(
                    f,
                    "v{var} ({role}) at block {block} op {op_index}: expected {expected:?}, found {found:?}"
                )
            }
            KirVerifyError::NotAPointer { var, block, op_index, found } => write!(
                f,
                "v{var} at block {block} op {op_index} is used as an address but has type {found:?}"
            ),
        }
    }
}

/// The op index reported for a violation in a block's terminator.
pub const TERMINATOR_INDEX: usize = usize::MAX;

/// Verify `ir` against the rules in the module header. `Ok(())` or every
/// violation found, in block/op order.
pub fn verify(ir: &KernelIR) -> Result<(), Vec<KirVerifyError>> {
    let mut errors = Vec::new();

    // ── 1. Shape ─────────────────────────────────────────────────────
    if ir.blocks.is_empty() {
        return Err(vec![KirVerifyError::NoBlocks]);
    }
    for (index, block) in ir.blocks.iter().enumerate() {
        if block.id as usize != index {
            errors.push(KirVerifyError::BlockIdMismatch { index, id: block.id });
        }
    }
    if !errors.is_empty() {
        // Dominance and branch targets index blocks by id; nothing below
        // is meaningful until the ids are their positions.
        return Err(errors);
    }
    let n = ir.blocks.len();
    for block in &ir.blocks {
        match &block.terminator {
            None => errors.push(KirVerifyError::MissingTerminator { block: block.id }),
            Some(term) => {
                for target in terminator_targets(term) {
                    if target as usize >= n {
                        errors.push(KirVerifyError::BadBranchTarget { block: block.id, target });
                    }
                }
            }
        }
    }

    // ── 2. SSA: one definition per VarId ─────────────────────────────
    // def_site: var -> (block, op_index) of the op that defines it.
    // Parameters are defined "before the entry block" and kept apart in
    // `params`, so a use of one reaches from anywhere.
    let mut def_site: HashMap<VarId, (BlockId, usize)> = HashMap::new();
    let mut params: HashSet<VarId> = HashSet::new();
    for p in &ir.params {
        if !params.insert(p.id) {
            errors.push(KirVerifyError::Redefined { var: p.id, block: 0, op_index: 0 });
        }
    }
    for block in &ir.blocks {
        for (op_index, op) in block.ops.iter().enumerate() {
            if let Some(dst) = op_dst(op)
                && (params.contains(&dst) || def_site.insert(dst, (block.id, op_index)).is_some())
            {
                errors.push(KirVerifyError::Redefined { var: dst, block: block.id, op_index });
            }
        }
    }

    // ── 3. Def-before-use under dominance ────────────────────────────
    let dom = dominators(ir);
    let reaches = |var: VarId, use_block: BlockId, use_index: usize| -> Result<(), bool> {
        // Ok(()) = reaches; Err(true) = undefined; Err(false) = defined but
        // does not reach.
        if params.contains(&var) {
            return Ok(());
        }
        match def_site.get(&var) {
            None => Err(true),
            Some(&(def_block, def_index)) => {
                if def_block == use_block {
                    if def_index < use_index {
                        Ok(())
                    } else {
                        Err(false)
                    }
                } else if dom[use_block as usize].contains(&def_block) {
                    Ok(())
                } else {
                    Err(false)
                }
            }
        }
    };
    let report_use = |var: VarId, block: BlockId, op_index: usize, errors: &mut Vec<KirVerifyError>| {
        match reaches(var, block, op_index) {
            Ok(()) => {}
            Err(true) => errors.push(KirVerifyError::Undefined { var, block, op_index }),
            Err(false) => errors.push(KirVerifyError::UseBeforeDef { var, block, op_index }),
        }
    };
    for block in &ir.blocks {
        for (op_index, op) in block.ops.iter().enumerate() {
            for var in op_uses(op) {
                report_use(var, block.id, op_index, &mut errors);
            }
        }
        if let Some(KirTerminator::CondBranch(cond, _, _)) = &block.terminator {
            report_use(*cond, block.id, TERMINATOR_INDEX, &mut errors);
        }
    }

    // ── 4. Typing ────────────────────────────────────────────────────
    for block in &ir.blocks {
        for (op_index, op) in block.ops.iter().enumerate() {
            check_types(ir, op, block.id, op_index, &mut errors);
        }
        if let Some(KirTerminator::CondBranch(cond, _, _)) = &block.terminator
            && let Some(found) = ir.var_types.get(cond)
            && *found != KirType::Bool
        {
            errors.push(KirVerifyError::TypeMismatch {
                var: *cond,
                block: block.id,
                op_index: TERMINATOR_INDEX,
                role: "cond",
                expected: KirType::Bool,
                found: found.clone(),
            });
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Render a verifier failure as one line per violation, for a
/// `CodegenError` message.
pub fn render_errors(errors: &[KirVerifyError]) -> String {
    errors.iter().map(|e| format!("  - {e}")).collect::<Vec<_>>().join("\n")
}

fn terminator_targets(term: &KirTerminator) -> Vec<BlockId> {
    match term {
        KirTerminator::Branch(t) => vec![*t],
        KirTerminator::CondBranch(_, t, e) => vec![*t, *e],
        KirTerminator::Return => vec![],
    }
}

/// The `VarId` an op defines, if any. `Store`, `AtomicAdd`, `Barrier` and
/// `SharedMemFence` define nothing (`AtomicAdd`'s PTX lowering reuses the
/// value register for the returned old value, a backend detail, not an IR
/// definition).
pub fn op_dst(op: &KirOp) -> Option<VarId> {
    match op {
        KirOp::Add(d, _, _)
        | KirOp::Sub(d, _, _)
        | KirOp::Mul(d, _, _)
        | KirOp::Div(d, _, _)
        | KirOp::Pow(d, _, _)
        | KirOp::Fma(d, _, _, _)
        | KirOp::Neg(d, _)
        | KirOp::Abs(d, _)
        | KirOp::Sqrt(d, _)
        | KirOp::Exp(d, _)
        | KirOp::Log(d, _)
        | KirOp::Sin(d, _)
        | KirOp::Cos(d, _)
        | KirOp::Tanh(d, _)
        | KirOp::Cast(d, _, _)
        | KirOp::Load(d, _, _)
        | KirOp::ThreadId(d, _)
        | KirOp::BlockIdx(d, _)
        | KirOp::BlockDim(d, _)
        | KirOp::GridDim(d, _)
        | KirOp::GlobalId(d, _)
        | KirOp::WarpShuffle(d, _, _)
        | KirOp::Cmp(d, _, _, _)
        | KirOp::Select(d, _, _, _)
        | KirOp::Const(d, _)
        | KirOp::PtrOffset(d, _, _) => Some(*d),
        KirOp::Store(_, _, _) | KirOp::AtomicAdd(_, _, _) | KirOp::Barrier | KirOp::SharedMemFence => None,
        KirOp::Matmul { out, .. } | KirOp::ElementwiseAdd { out, .. } | KirOp::Relu { out, .. } => {
            Some(*out)
        }
    }
}

/// The `VarId`s an op reads.
pub fn op_uses(op: &KirOp) -> Vec<VarId> {
    match op {
        KirOp::Add(_, a, b)
        | KirOp::Sub(_, a, b)
        | KirOp::Mul(_, a, b)
        | KirOp::Div(_, a, b)
        | KirOp::Pow(_, a, b)
        | KirOp::WarpShuffle(_, a, b)
        | KirOp::Cmp(_, a, b, _)
        | KirOp::PtrOffset(_, a, b) => vec![*a, *b],
        KirOp::Fma(_, a, b, c) | KirOp::Select(_, a, b, c) => vec![*a, *b, *c],
        KirOp::Neg(_, s)
        | KirOp::Abs(_, s)
        | KirOp::Sqrt(_, s)
        | KirOp::Exp(_, s)
        | KirOp::Log(_, s)
        | KirOp::Sin(_, s)
        | KirOp::Cos(_, s)
        | KirOp::Tanh(_, s)
        | KirOp::Cast(_, s, _)
        | KirOp::Load(_, s, _) => vec![*s],
        KirOp::Store(p, v, _) | KirOp::AtomicAdd(p, v, _) => vec![*p, *v],
        KirOp::ThreadId(_, _)
        | KirOp::BlockIdx(_, _)
        | KirOp::BlockDim(_, _)
        | KirOp::GridDim(_, _)
        | KirOp::GlobalId(_, _)
        | KirOp::Const(_, _)
        | KirOp::Barrier
        | KirOp::SharedMemFence => vec![],
        KirOp::Matmul { a, b, .. } | KirOp::ElementwiseAdd { a, b, .. } => vec![*a, *b],
        KirOp::Relu { a, .. } => vec![*a],
    }
}

/// Dominator sets over the block CFG (entry = block 0), by the standard
/// iterative dataflow: `dom(entry) = {entry}`, `dom(b) = {b} ∪ ⋂ dom(p)`
/// over predecessors. Blocks unreachable from the entry keep the full set
/// (they never execute; nothing they use can be wrong at runtime).
fn dominators(ir: &KernelIR) -> Vec<HashSet<BlockId>> {
    let n = ir.blocks.len();
    let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    for block in &ir.blocks {
        if let Some(term) = &block.terminator {
            for t in terminator_targets(term) {
                if (t as usize) < n {
                    preds[t as usize].push(block.id);
                }
            }
        }
    }
    let all: HashSet<BlockId> = (0..n as BlockId).collect();
    let mut dom: Vec<HashSet<BlockId>> = vec![all.clone(); n];
    dom[0] = HashSet::from([0]);
    let mut changed = true;
    while changed {
        changed = false;
        for b in 1..n {
            let mut new: Option<HashSet<BlockId>> = None;
            for &p in &preds[b] {
                new = Some(match new {
                    None => dom[p as usize].clone(),
                    Some(acc) => acc.intersection(&dom[p as usize]).copied().collect(),
                });
            }
            let mut new = new.unwrap_or_else(|| all.clone());
            new.insert(b as BlockId);
            if new != dom[b] {
                dom[b] = new;
                changed = true;
            }
        }
    }
    dom
}

fn check_types(
    ir: &KernelIR,
    op: &KirOp,
    block: BlockId,
    op_index: usize,
    errors: &mut Vec<KirVerifyError>,
) {
    let ty = |v: &VarId| ir.var_types.get(v).cloned();
    let mut expect = |var: VarId, role: &'static str, expected: &KirType| {
        if let Some(found) = ty(&var)
            && found != *expected
        {
            errors.push(KirVerifyError::TypeMismatch {
                var,
                block,
                op_index,
                role,
                expected: expected.clone(),
                found,
            });
        }
    };
    match op {
        // Homogeneous arithmetic: every operand carries the destination's type.
        KirOp::Add(d, a, b)
        | KirOp::Sub(d, a, b)
        | KirOp::Mul(d, a, b)
        | KirOp::Div(d, a, b)
        | KirOp::Pow(d, a, b) => {
            if let Some(dt) = ty(d) {
                expect(*a, "a", &dt);
                expect(*b, "b", &dt);
            }
        }
        KirOp::Fma(d, a, b, c) => {
            if let Some(dt) = ty(d) {
                expect(*a, "a", &dt);
                expect(*b, "b", &dt);
                expect(*c, "c", &dt);
            }
        }
        KirOp::Neg(d, s)
        | KirOp::Abs(d, s)
        | KirOp::Sqrt(d, s)
        | KirOp::Exp(d, s)
        | KirOp::Log(d, s)
        | KirOp::Sin(d, s)
        | KirOp::Cos(d, s)
        | KirOp::Tanh(d, s) => {
            if let Some(dt) = ty(d) {
                expect(*s, "src", &dt);
            }
        }
        KirOp::WarpShuffle(d, v, _) => {
            if let Some(dt) = ty(d) {
                expect(*v, "val", &dt);
            }
        }
        KirOp::Cast(d, _, target) => expect(*d, "dst", target),
        KirOp::Load(d, p, _) => match ty(p) {
            Some(KirType::Ptr(pointee, _)) => expect(*d, "dst", &pointee),
            Some(found) => errors.push(KirVerifyError::NotAPointer { var: *p, block, op_index, found }),
            None => {}
        },
        KirOp::Store(p, v, _) | KirOp::AtomicAdd(p, v, _) => match ty(p) {
            Some(KirType::Ptr(pointee, _)) => expect(*v, "value", &pointee),
            Some(found) => errors.push(KirVerifyError::NotAPointer { var: *p, block, op_index, found }),
            None => {}
        },
        KirOp::ThreadId(d, _)
        | KirOp::BlockIdx(d, _)
        | KirOp::BlockDim(d, _)
        | KirOp::GridDim(d, _)
        | KirOp::GlobalId(d, _) => expect(*d, "dst", &KirType::U32),
        KirOp::Cmp(d, a, b, _) => {
            expect(*d, "dst", &KirType::Bool);
            if let Some(at) = ty(a) {
                expect(*b, "b", &at);
            }
        }
        KirOp::Select(d, c, t, f) => {
            expect(*c, "cond", &KirType::Bool);
            if let Some(dt) = ty(d) {
                expect(*t, "true_val", &dt);
                expect(*f, "false_val", &dt);
            }
        }
        KirOp::Const(d, KirConst { ty: kty, .. }) => expect(*d, "dst", kty),
        KirOp::PtrOffset(d, base, _) => match ty(base) {
            Some(bt @ KirType::Ptr(_, _)) => expect(*d, "dst", &bt),
            Some(found) => {
                errors.push(KirVerifyError::NotAPointer { var: *base, block, op_index, found })
            }
            None => {}
        },
        KirOp::Barrier
        | KirOp::SharedMemFence
        | KirOp::Matmul { .. }
        | KirOp::ElementwiseAdd { .. }
        | KirOp::Relu { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KirBuilder};

    fn f32_ptr() -> KirType {
        KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
    }

    /// The `builder_creates_valid_ir` kernel from `kernel_ir.rs`: a bounds
    /// check, a load-add-store body, an exit block.
    fn simple_add_kernel() -> KernelIR {
        let mut b = KirBuilder::new("test_add");
        let a_ptr = b.add_param("a", f32_ptr(), AddressSpace::Global);
        let b_ptr = b.add_param("b", f32_ptr(), AddressSpace::Global);
        let out_ptr = b.add_param("out", f32_ptr(), AddressSpace::Global);
        let len = b.add_param("len", KirType::U32, AddressSpace::Local);
        let entry = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        b.set_block(entry);
        let tid = b.new_typed_var(KirType::U32);
        b.emit(KirOp::GlobalId(tid, 0));
        let in_bounds = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(in_bounds, tid, len, CmpOp::Lt));
        b.terminate(KirTerminator::CondBranch(in_bounds, body, exit));
        b.set_block(body);
        let a_addr = b.new_typed_var(f32_ptr());
        b.emit(KirOp::PtrOffset(a_addr, a_ptr, tid));
        let a_val = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(a_val, a_addr, AddressSpace::Global));
        let b_addr = b.new_typed_var(f32_ptr());
        b.emit(KirOp::PtrOffset(b_addr, b_ptr, tid));
        let b_val = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(b_val, b_addr, AddressSpace::Global));
        let sum = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Add(sum, a_val, b_val));
        let out_addr = b.new_typed_var(f32_ptr());
        b.emit(KirOp::PtrOffset(out_addr, out_ptr, tid));
        b.emit(KirOp::Store(out_addr, sum, AddressSpace::Global));
        b.terminate(KirTerminator::Branch(exit));
        b.set_block(exit);
        b.terminate(KirTerminator::Return);
        b.finalize()
    }

    #[test]
    fn a_well_formed_kernel_verifies() {
        let ir = simple_add_kernel();
        assert_eq!(verify(&ir), Ok(()));
        assert!(ir.is_well_formed());
    }

    #[test]
    fn no_blocks_is_refused() {
        let ir = KirBuilder::new("empty").finalize();
        assert_eq!(verify(&ir), Err(vec![KirVerifyError::NoBlocks]));
        assert!(!ir.is_well_formed());
    }

    #[test]
    fn a_missing_terminator_is_reported_per_block() {
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.set_block(entry);
        let ir = b.finalize();
        assert_eq!(verify(&ir), Err(vec![KirVerifyError::MissingTerminator { block: 0 }]));
    }

    #[test]
    fn a_branch_to_a_missing_block_is_reported() {
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.set_block(entry);
        b.terminate(KirTerminator::Branch(7));
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::BadBranchTarget { block: 0, target: 7 }])
        );
    }

    #[test]
    fn a_second_definition_breaks_ssa() {
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.set_block(entry);
        let x = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(x, 0));
        b.emit(KirOp::BlockIdx(x, 0));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::Redefined { var: x, block: 0, op_index: 1 }])
        );
    }

    #[test]
    fn redefining_a_parameter_breaks_ssa() {
        let mut b = KirBuilder::new("t");
        let len = b.add_param("len", KirType::U32, AddressSpace::Local);
        let entry = b.new_block();
        b.set_block(entry);
        b.emit(KirOp::ThreadId(len, 0));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::Redefined { var: len, block: 0, op_index: 0 }])
        );
    }

    #[test]
    fn an_operand_that_is_never_defined_is_reported() {
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.set_block(entry);
        let d = b.new_typed_var(KirType::F32);
        let ghost: VarId = 99;
        b.emit(KirOp::Neg(d, ghost));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::Undefined { var: ghost, block: 0, op_index: 0 }])
        );
    }

    #[test]
    fn a_use_ahead_of_its_definition_in_the_same_block_is_reported() {
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.set_block(entry);
        let x = b.new_typed_var(KirType::U32);
        let y = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Add(y, x, x)); // op 0 uses x, defined at op 1
        b.emit(KirOp::ThreadId(x, 0));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![
                KirVerifyError::UseBeforeDef { var: x, block: 0, op_index: 0 },
                KirVerifyError::UseBeforeDef { var: x, block: 0, op_index: 0 },
            ])
        );
    }

    #[test]
    fn a_definition_in_a_non_dominating_block_does_not_reach() {
        // entry --cond--> left | right ; both --> join. `x` defined in
        // `left` is used in `join`: `left` does not dominate `join`.
        let mut b = KirBuilder::new("t");
        let flag = b.add_param("flag", KirType::Bool, AddressSpace::Local);
        let entry = b.new_block();
        let left = b.new_block();
        let right = b.new_block();
        let join = b.new_block();
        b.set_block(entry);
        b.terminate(KirTerminator::CondBranch(flag, left, right));
        b.set_block(left);
        let x = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(x, 0));
        b.terminate(KirTerminator::Branch(join));
        b.set_block(right);
        b.terminate(KirTerminator::Branch(join));
        b.set_block(join);
        let y = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Neg(y, x));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::UseBeforeDef { var: x, block: join, op_index: 0 }])
        );
    }

    #[test]
    fn a_definition_in_the_entry_reaches_every_branch() {
        let mut b = KirBuilder::new("t");
        let flag = b.add_param("flag", KirType::Bool, AddressSpace::Local);
        let entry = b.new_block();
        let left = b.new_block();
        let right = b.new_block();
        b.set_block(entry);
        let x = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(x, 0));
        b.terminate(KirTerminator::CondBranch(flag, left, right));
        for blk in [left, right] {
            b.set_block(blk);
            let y = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Neg(y, x));
            b.terminate(KirTerminator::Return);
        }
        let ir = b.finalize();
        assert_eq!(verify(&ir), Ok(()));
    }

    #[test]
    fn a_loop_back_edge_keeps_the_header_dominating_its_body() {
        // entry -> header ; header --cond--> body | exit ; body -> header.
        // `i` defined in entry reaches body and exit; `t` defined in body
        // does not reach header (body does not dominate header).
        let mut b = KirBuilder::new("t");
        let flag = b.add_param("flag", KirType::Bool, AddressSpace::Local);
        let entry = b.new_block();
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        b.set_block(entry);
        let i = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(i, 0));
        b.terminate(KirTerminator::Branch(header));
        b.set_block(header);
        let t: VarId = 50; // defined in body below
        let h = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Neg(h, t));
        b.terminate(KirTerminator::CondBranch(flag, body, exit));
        b.set_block(body);
        b.emit(KirOp::Neg(t, i));
        b.terminate(KirTerminator::Branch(header));
        b.set_block(exit);
        let e = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Neg(e, i));
        b.terminate(KirTerminator::Return);
        let mut ir = b.finalize();
        ir.var_types.insert(t, KirType::U32);
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::UseBeforeDef { var: t, block: header, op_index: 0 }])
        );
    }

    #[test]
    fn heterogeneous_arithmetic_is_a_type_error() {
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.set_block(entry);
        let x = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(x, KirConst { ty: KirType::F32, value: ConstValue::F32(1.0) }));
        let n = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(n, 0));
        let d = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Add(d, x, n));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::TypeMismatch {
                var: n,
                block: 0,
                op_index: 2,
                role: "b",
                expected: KirType::F32,
                found: KirType::U32,
            }])
        );
    }

    #[test]
    fn loads_and_stores_move_the_pointee_type() {
        let mut b = KirBuilder::new("t");
        let p = b.add_param("p", f32_ptr(), AddressSpace::Global);
        let entry = b.new_block();
        b.set_block(entry);
        let d = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Load(d, p, AddressSpace::Global));
        b.emit(KirOp::Store(p, d, AddressSpace::Global));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![
                KirVerifyError::TypeMismatch {
                    var: d,
                    block: 0,
                    op_index: 0,
                    role: "dst",
                    expected: KirType::F32,
                    found: KirType::U32,
                },
                KirVerifyError::TypeMismatch {
                    var: d,
                    block: 0,
                    op_index: 1,
                    role: "value",
                    expected: KirType::F32,
                    found: KirType::U32,
                },
            ])
        );
    }

    #[test]
    fn a_non_pointer_address_is_reported() {
        let mut b = KirBuilder::new("t");
        let n = b.add_param("n", KirType::U32, AddressSpace::Local);
        let entry = b.new_block();
        b.set_block(entry);
        let d = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(d, n, AddressSpace::Global));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::NotAPointer { var: n, block: 0, op_index: 0, found: KirType::U32 }])
        );
    }

    #[test]
    fn casts_land_in_their_target_type_and_compares_produce_bool() {
        let mut b = KirBuilder::new("t");
        let n = b.add_param("n", KirType::U32, AddressSpace::Local);
        let entry = b.new_block();
        b.set_block(entry);
        let f = b.new_typed_var(KirType::F64);
        b.emit(KirOp::Cast(f, n, KirType::F32));
        let c = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Cmp(c, n, n, CmpOp::Eq));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![
                KirVerifyError::TypeMismatch {
                    var: f,
                    block: 0,
                    op_index: 0,
                    role: "dst",
                    expected: KirType::F32,
                    found: KirType::F64,
                },
                KirVerifyError::TypeMismatch {
                    var: c,
                    block: 0,
                    op_index: 1,
                    role: "dst",
                    expected: KirType::Bool,
                    found: KirType::U32,
                },
            ])
        );
    }

    #[test]
    fn a_non_bool_branch_condition_is_reported_on_the_terminator() {
        let mut b = KirBuilder::new("t");
        let n = b.add_param("n", KirType::U32, AddressSpace::Local);
        let entry = b.new_block();
        let exit = b.new_block();
        b.set_block(entry);
        b.terminate(KirTerminator::CondBranch(n, exit, exit));
        b.set_block(exit);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::TypeMismatch {
                var: n,
                block: 0,
                op_index: TERMINATOR_INDEX,
                role: "cond",
                expected: KirType::Bool,
                found: KirType::U32,
            }])
        );
    }

    #[test]
    fn untyped_variables_are_exempt_from_typing_only() {
        // `new_var` records no type: rule 4 has nothing to say, rules 2–3
        // still apply.
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.set_block(entry);
        let x = b.new_var();
        let y = b.new_var();
        b.emit(KirOp::ThreadId(x, 0));
        b.emit(KirOp::Add(y, x, x));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(verify(&ir), Ok(()));
    }

    #[test]
    fn errors_render_one_per_line() {
        let errs = vec![
            KirVerifyError::MissingTerminator { block: 2 },
            KirVerifyError::Undefined { var: 4, block: 1, op_index: 3 },
        ];
        assert_eq!(
            render_errors(&errs),
            "  - block 2 has no terminator\n  - v4 used at block 1 op 3 is never defined"
        );
    }
}
