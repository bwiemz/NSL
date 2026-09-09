// crates/nsl-kir/src/kir_verify.rs
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
//!   5. **Async-copy discipline.** Within a block, every `CpAsync` is
//!      committed (`CpAsyncCommit`) before a `CpAsyncWait` and before the
//!      block ends, and a `CpAsyncWait { pending }` never names more
//!      groups than have been committed. `CpAsync` copies 4, 8 or 16 bytes
//!      from a `Ptr(_, Global)` into a `Ptr(_, Shared)`.
//!
//!   6. **Fragment typing.** `LdMatrixX4` loads from a `Ptr(_, Shared)`
//!      into four `Vec(F16, 2)` registers; `MmaF16M16N8K16` takes four
//!      `Vec(F16, 2)` A registers, two `Vec(F16, 2)` B registers and four
//!      `F32` accumulators in and out.
//!
//!   4b. **The scalar ISA (roadmap A2 step 4).** `And`/`Or`/`Xor`/`Not`
//!      are homogeneous on integers or `Bool`s, `Rem` on integers, `Min`/
//!      `Max` on any numeric type, `Rcp`/`Rsqrt` on `F32`/`F64`; `Shl`/
//!      `Shr` keep the operand's type and take a `U32` amount;
//!      `CastRounded` lands in its target; `LoadVec`/`StoreVec` move 2 or
//!      4 pointee-typed values through a `Ptr` (`BadVectorWidth`); `Vote`
//!      reads a `Bool` and produces a `Bool` (`Any`/`All`) or a `U32`
//!      (`Ballot`); `LaneId`/`WarpId` produce `U32`.
//!   8. **Predication.** `Predicated { pred, op }` reads a `Bool` and may
//!      wrap only an op with no destination (`PredicatedValueOp`): a
//!      predicated definition would be partial, which SSA forbids. The
//!      wrapped op is checked like any other.
//!   7. **Edges.** Every edge passes exactly one argument per parameter of
//!      its target block (`EdgeArityMismatch`), each argument reaches the
//!      terminator like any other use (rules 2–3 treat a block parameter as
//!      a definition at its block's entry), and a typed argument carries the
//!      parameter's type (`TypeMismatch { role: "arg" }`). The entry block
//!      has no parameters (`EntryBlockHasParams`). Roadmap A2 step 2: this
//!      is how a loop-carried value is written without phi nodes.
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

use crate::kernel_ir::{MmaShape, 
    AddressSpace, BlockId, KernelIR, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId,
    VoteMode,
};

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
    /// A typed pointer operand lives in the wrong state space (`CpAsync`
    /// copies global → shared, nothing else).
    AddressSpaceMismatch {
        var: VarId,
        block: BlockId,
        op_index: usize,
        role: &'static str,
        expected: AddressSpace,
        found: AddressSpace,
    },
    /// `CpAsync { bytes }` with a width `cp.async` does not have.
    BadAsyncCopyWidth { block: BlockId, op_index: usize, bytes: u8 },
    /// A `CpAsync` that is still uncommitted at a `CpAsyncWait` (the wait
    /// cannot cover it) or at the block's end (`op_index == TERMINATOR_INDEX`).
    UncommittedAsyncCopy { block: BlockId, op_index: usize },
    /// `CpAsyncWait { pending }` naming more groups than the block committed.
    AsyncWaitExceedsGroups { block: BlockId, op_index: usize, pending: u8, committed: u8 },
    /// Block 0 lists parameters; nothing enters the kernel with arguments.
    EntryBlockHasParams { count: usize },
    /// An edge from `block` to `target` passes `found` arguments for
    /// `expected` parameters.
    EdgeArityMismatch { block: BlockId, target: BlockId, expected: usize, found: usize },
    /// A `Predicated` wrapping an op that defines a value (or another
    /// `Predicated`).
    PredicatedValueOp { block: BlockId, op_index: usize },
    /// A `LoadVec`/`StoreVec` of `width` values; PTX vectors are 2 or 4 wide.
    BadVectorWidth { block: BlockId, op_index: usize, width: usize },
    /// Rule 6: `ldmatrix` has `.x1`, `.x2` and `.x4` and no other form.
    BadLdMatrixCount { block: BlockId, op_index: usize, count: usize },
    /// Rule 6: an `Mma` operand vector whose arity does not match its shape.
    BadMmaFragmentCount {
        block: BlockId,
        op_index: usize,
        shape: MmaShape,
        role: &'static str,
        expected: usize,
        found: usize,
    },
    /// Rule 8: `SharedRegion` names a region the layout does not have.
    SmemRegionOutOfRange { block: BlockId, op_index: usize, region: u32, regions: usize },
    /// Rule 8: two regions of the layout occupy overlapping bytes.
    SmemRegionsOverlap { earlier: String, later: String, at: u32 },
    /// Rule 8: the layout does not fit the budget for its `dynamic` flag.
    SmemBudgetExceeded { total: u32, budget: u32, dynamic: bool },
    /// Rule 8: a kernel declares both a region layout and the flat
    /// `shared_mem_bytes` block. Mixing a static `.shared` declaration with
    /// an `extern` one is the sm_120 illegal-address finding.
    SmemLayoutAndFlatBlock { regions: usize, flat_bytes: u32 },
    /// Rule 8: an `ldmatrix` or 16-byte `cp.async` address comes from a
    /// region that is not 16-byte aligned.
    SmemRegionUnaligned { block: BlockId, op_index: usize, region: String, align: u32 },
    /// Rule 9: a `Barrier` under a predicate. A barrier the whole warp does
    /// not reach is a hang, not a skipped instruction.
    PredicatedBarrier { block: BlockId, op_index: usize },
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
            KirVerifyError::AddressSpaceMismatch { var, block, op_index, role, expected, found } => {
                write!(
                    f,
                    "v{var} ({role}) at block {block} op {op_index}: expected a {expected:?} pointer, found {found:?}"
                )
            }
            KirVerifyError::BadAsyncCopyWidth { block, op_index, bytes } => write!(
                f,
                "CpAsync at block {block} op {op_index} copies {bytes} bytes; cp.async copies 4, 8 or 16"
            ),
            KirVerifyError::UncommittedAsyncCopy { block, op_index } => {
                if *op_index == TERMINATOR_INDEX {
                    write!(f, "block {block} ends with an uncommitted CpAsync")
                } else {
                    write!(f, "CpAsyncWait at block {block} op {op_index} cannot cover an uncommitted CpAsync")
                }
            }
            KirVerifyError::AsyncWaitExceedsGroups { block, op_index, pending, committed } => write!(
                f,
                "CpAsyncWait at block {block} op {op_index} allows {pending} pending group(s) but only {committed} were committed"
            ),
            KirVerifyError::EntryBlockHasParams { count } => {
                write!(f, "the entry block lists {count} parameter(s); nothing passes it arguments")
            }
            KirVerifyError::EdgeArityMismatch { block, target, expected, found } => write!(
                f,
                "the edge from block {block} to block {target} passes {found} argument(s) for {expected} parameter(s)"
            ),
            KirVerifyError::PredicatedValueOp { block, op_index } => write!(
                f,
                "Predicated at block {block} op {op_index} wraps an op that defines a value; predicate only side effects"
            ),
            KirVerifyError::BadLdMatrixCount { block, op_index, count } => write!(
                f,
                "block {block} op {op_index}: ldmatrix loads {count} matrices; \
                 PTX has .x1, .x2 and .x4 only"
            ),
            KirVerifyError::BadMmaFragmentCount { block, op_index, shape, role, expected, found } => write!(
                f,
                "block {block} op {op_index}: mma.{} takes {expected} {role} register(s), found {found}",
                shape.ptx_shape()
            ),
            KirVerifyError::SmemRegionOutOfRange { block, op_index, region, regions } => write!(
                f,
                "block {block} op {op_index}: shared region {region} but the layout has {regions}"
            ),
            KirVerifyError::SmemRegionsOverlap { earlier, later, at } => write!(
                f,
                "shared regions '{earlier}' and '{later}' overlap at byte {at}"
            ),
            KirVerifyError::SmemBudgetExceeded { total, budget, dynamic } => write!(
                f,
                "shared layout needs {total} bytes; the {} budget is {budget}",
                if *dynamic { "dynamic (extern .shared)" } else { "static .shared" }
            ),
            KirVerifyError::SmemLayoutAndFlatBlock { regions, flat_bytes } => write!(
                f,
                "kernel declares {regions} shared region(s) AND a flat {flat_bytes}-byte block; \
                 a kernel is static or dynamic, never both"
            ),
            KirVerifyError::SmemRegionUnaligned { block, op_index, region, align } => write!(
                f,
                "block {block} op {op_index}: address comes from shared region '{region}' \
                 aligned to {align}; ldmatrix and 16-byte cp.async need 16"
            ),
            KirVerifyError::PredicatedBarrier { block, op_index } => write!(
                f,
                "block {block} op {op_index}: a Barrier must not be predicated"
            ),
            KirVerifyError::BadVectorWidth { block, op_index, width } => write!(
                f,
                "vector memory op at block {block} op {op_index} moves {width} values; PTX vectors are 2 or 4 wide"
            ),
        }
    }
}

/// The op index reported for a violation in a block's terminator.
pub const TERMINATOR_INDEX: usize = usize::MAX;

/// The op index reported for a violation on a block parameter (a
/// definition at the block's entry, before op 0).
pub const BLOCK_PARAM_INDEX: usize = usize::MAX - 1;

/// Verify `ir` against the rules in the module header. `Ok(())` or every
/// violation found, in block/op order.
/// Rule 8, the whole-kernel half (roadmap A2 step 6).
///
/// Overlap, budget and the static/dynamic exclusivity are properties of the
/// layout rather than of any op, so they are checked once. The per-op halves
/// — an in-range region index, and a 16-aligned region behind an `ldmatrix`
/// or a 16-byte `cp.async` — live with their ops.
fn check_smem_layout(ir: &KernelIR, errors: &mut Vec<KirVerifyError>) {
    let layout = &ir.smem_layout;
    if layout.regions.is_empty() {
        return;
    }

    // A kernel is static or dynamic, never both. Mixing a static `.shared`
    // declaration with an `extern` one is the sm_120 illegal-address
    // finding the design spec records.
    if ir.shared_mem_bytes > 0 {
        errors.push(KirVerifyError::SmemLayoutAndFlatBlock {
            regions: layout.regions.len(),
            flat_bytes: ir.shared_mem_bytes,
        });
    }

    // Overlap. `offset_of` packs in declaration order, so a region can only
    // overlap its predecessor — but say which pair and where, because the
    // report is what a caller fixes.
    let mut prev_end: Option<(usize, u32)> = None;
    for index in 0..layout.regions.len() {
        let Some(start) = layout.offset_of(index) else { continue };
        if let Some((prev, end)) = prev_end
            && start < end
        {
            errors.push(KirVerifyError::SmemRegionsOverlap {
                earlier: layout.regions[prev].name.clone(),
                later: layout.regions[index].name.clone(),
                at: start,
            });
        }
        prev_end = Some((index, start.saturating_add(layout.regions[index].bytes)));
    }

    // Alignment. `ldmatrix` and a 16-byte `cp.async` require a 16-byte
    // aligned shared address. We can only say so for an address that comes
    // STRAIGHT from a `SharedRegion` — once it has been through arithmetic
    // the offset is a runtime value and this would need a range analysis.
    // That is the conservative direction: a missed case is a `ptxas` or
    // runtime error as it is today, whereas guessing would reject the
    // indexing every real kernel does.
    let mut region_of: HashMap<VarId, u32> = HashMap::new();
    for block in &ir.blocks {
        for op in &block.ops {
            if let KirOp::SharedRegion { dst, region } = op {
                region_of.insert(*dst, *region);
            }
        }
    }
    if !region_of.is_empty() {
        for block in &ir.blocks {
            for (op_index, op) in block.ops.iter().enumerate() {
                let addr = match op {
                    KirOp::LdMatrix { addr, .. } => Some(*addr),
                    KirOp::CpAsync { dst, bytes: 16, .. } => Some(*dst),
                    _ => None,
                };
                let Some(addr) = addr else { continue };
                let Some(index) = region_of.get(&addr) else { continue };
                let Some(region) = layout.regions.get(*index as usize) else { continue };
                if region.align < 16 {
                    errors.push(KirVerifyError::SmemRegionUnaligned {
                        block: block.id,
                        op_index,
                        region: region.name.clone(),
                        align: region.align,
                    });
                }
            }
        }
    }

    // Budget. `total_bytes` includes the alignment padding, which is the
    // number the hardware actually has to find.
    match layout.total_bytes() {
        Some(total) if total <= layout.budget() => {}
        Some(total) => errors.push(KirVerifyError::SmemBudgetExceeded {
            total,
            budget: layout.budget(),
            dynamic: layout.dynamic,
        }),
        // An overflowing layout cannot fit any budget.
        None => errors.push(KirVerifyError::SmemBudgetExceeded {
            total: u32::MAX,
            budget: layout.budget(),
            dynamic: layout.dynamic,
        }),
    }
}

pub fn verify(ir: &KernelIR) -> Result<(), Vec<KirVerifyError>> {
    let mut errors = Vec::new();

    // ── 0. Shared-memory layout (rule 8, roadmap A2 step 6) ──────────
    // Checked before anything else: these are properties of the kernel's
    // declaration, not of any one op, and a bad layout makes every
    // `SharedRegion` offset below meaningless.
    check_smem_layout(ir, &mut errors);

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
    if !ir.blocks[0].params.is_empty() {
        errors.push(KirVerifyError::EntryBlockHasParams { count: ir.blocks[0].params.len() });
    }

    // ── 7. Edges: one argument per target parameter ──────────────────
    // (Checked here, with the shape, so the typing pass below can pair
    // arguments with parameters by position.)
    for block in &ir.blocks {
        let Some(term) = &block.terminator else { continue };
        for edge in term.edges() {
            let Some(target) = ir.blocks.get(edge.target as usize) else { continue };
            if edge.args.len() != target.params.len() {
                errors.push(KirVerifyError::EdgeArityMismatch {
                    block: block.id,
                    target: edge.target,
                    expected: target.params.len(),
                    found: edge.args.len(),
                });
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
    // A block parameter is defined at its block's entry (rule 7).
    let mut block_params: HashMap<VarId, BlockId> = HashMap::new();
    for block in &ir.blocks {
        for p in &block.params {
            if params.contains(&p.id) || block_params.insert(p.id, block.id).is_some() {
                errors.push(KirVerifyError::Redefined {
                    var: p.id,
                    block: block.id,
                    op_index: BLOCK_PARAM_INDEX,
                });
            }
        }
    }
    for block in &ir.blocks {
        for (op_index, op) in block.ops.iter().enumerate() {
            for dst in op_dsts(op) {
                if params.contains(&dst)
                    || block_params.contains_key(&dst)
                    || def_site.insert(dst, (block.id, op_index)).is_some()
                {
                    errors.push(KirVerifyError::Redefined { var: dst, block: block.id, op_index });
                }
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
        if let Some(&def_block) = block_params.get(&var) {
            // Defined at `def_block`'s entry: reaches everything in that
            // block (its terminator included) and every block it dominates.
            return if def_block == use_block || dom[use_block as usize].contains(&def_block) {
                Ok(())
            } else {
                Err(false)
            };
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
        if let Some(term) = &block.terminator {
            for var in terminator_uses(term) {
                report_use(var, block.id, TERMINATOR_INDEX, &mut errors);
            }
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
        // Rule 7, typing half: a typed argument carries its parameter's type.
        if let Some(term) = &block.terminator {
            for edge in term.edges() {
                let Some(target) = ir.blocks.get(edge.target as usize) else { continue };
                for (arg, param) in edge.args.iter().zip(&target.params) {
                    if let Some(found) = ir.var_types.get(arg)
                        && *found != param.ty
                    {
                        errors.push(KirVerifyError::TypeMismatch {
                            var: *arg,
                            block: block.id,
                            op_index: TERMINATOR_INDEX,
                            role: "arg",
                            expected: param.ty.clone(),
                            found: found.clone(),
                        });
                    }
                }
            }
        }
    }

    // ── 5. Async-copy discipline ─────────────────────────────────────
    for block in &ir.blocks {
        let mut uncommitted: u32 = 0;
        let mut committed: u8 = 0;
        for (op_index, op) in block.ops.iter().enumerate() {
            match op {
                KirOp::CpAsync { bytes, .. } => {
                    if !matches!(bytes, 4 | 8 | 16) {
                        errors.push(KirVerifyError::BadAsyncCopyWidth {
                            block: block.id,
                            op_index,
                            bytes: *bytes,
                        });
                    }
                    uncommitted += 1;
                }
                KirOp::CpAsyncCommit => {
                    committed = committed.saturating_add(1);
                    uncommitted = 0;
                }
                KirOp::CpAsyncWait { pending } => {
                    if uncommitted > 0 {
                        errors.push(KirVerifyError::UncommittedAsyncCopy { block: block.id, op_index });
                    }
                    if *pending > committed {
                        errors.push(KirVerifyError::AsyncWaitExceedsGroups {
                            block: block.id,
                            op_index,
                            pending: *pending,
                            committed,
                        });
                    }
                    committed = committed.min(*pending);
                }
                _ => {}
            }
        }
        if uncommitted > 0 {
            errors.push(KirVerifyError::UncommittedAsyncCopy {
                block: block.id,
                op_index: TERMINATOR_INDEX,
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

fn is_float(ty: &KirType) -> bool {
    matches!(ty, KirType::F16 | KirType::Bf16 | KirType::F32 | KirType::F64)
}

fn is_integer(ty: &KirType) -> bool {
    matches!(
        ty,
        KirType::U32 | KirType::I32 | KirType::U64 | KirType::I64 | KirType::I8 | KirType::I16
    )
}

/// The packed f16x2 fragment register type the tensor-core ops carry.
fn f16x2() -> KirType {
    KirType::Vec(Box::new(KirType::F16), 2)
}

fn terminator_targets(term: &KirTerminator) -> Vec<BlockId> {
    term.edges().iter().map(|e: &&KirEdge| e.target).collect()
}

/// The `VarId`s a terminator reads: the condition of a `CondBranch` and
/// every edge argument.
pub fn terminator_uses(term: &KirTerminator) -> Vec<VarId> {
    let mut uses = Vec::new();
    if let KirTerminator::CondBranch(cond, _, _) = term {
        uses.push(*cond);
    }
    for edge in term.edges() {
        uses.extend_from_slice(&edge.args);
    }
    uses
}

/// The `VarId`s an op defines: one for most ops, four for the
/// warp-collective loads and MACs, none for `Store`, `AtomicAdd`, the
/// barriers and the async-copy group (`AtomicAdd`'s PTX lowering reuses the
/// value register for the returned old value, a backend detail, not an IR
/// definition).
pub fn op_dsts(op: &KirOp) -> Vec<VarId> {
    match op {
        KirOp::LdMatrix { dst, .. } => dst.clone(),
        KirOp::Mma { d, .. } => d.clone(),
        KirOp::LoadVec { dsts, .. } => dsts.clone(),
        KirOp::Predicated { op, .. } => op_dsts(op),
        _ => op_dst(op).into_iter().collect(),
    }
}

/// The single `VarId` an op defines, if it defines exactly one. The
/// multi-destination ops answer `None` here; use [`op_dsts`].
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
        | KirOp::And(d, _, _)
        | KirOp::Or(d, _, _)
        | KirOp::Xor(d, _, _)
        | KirOp::Not(d, _)
        | KirOp::Shl(d, _, _)
        | KirOp::Shr(d, _, _)
        | KirOp::Rem(d, _, _)
        | KirOp::Min(d, _, _)
        | KirOp::Max(d, _, _)
        | KirOp::Rcp(d, _)
        | KirOp::Rsqrt(d, _)
        | KirOp::CastRounded { dst: d, .. }
        | KirOp::Vote { dst: d, .. }
        | KirOp::LaneId(d)
        | KirOp::WarpId(d)
        | KirOp::WarpShuffle { dst: d, .. }
        | KirOp::Cmp(d, _, _, _)
        | KirOp::Select(d, _, _, _)
        | KirOp::Const(d, _)
        | KirOp::PtrOffset(d, _, _)
        | KirOp::SharedBase(d) => Some(*d),
        KirOp::SharedRegion { dst, .. } => Some(*dst),
        KirOp::Store(_, _, _)
        | KirOp::AtomicAdd(_, _, _)
        | KirOp::Barrier
        | KirOp::SharedMemFence
        | KirOp::CpAsync { .. }
        | KirOp::CpAsyncCommit
        | KirOp::CpAsyncWait { .. }
        | KirOp::LdMatrix { .. }
        | KirOp::Mma { .. }
        | KirOp::LoadVec { .. }
        | KirOp::StoreVec { .. }
        | KirOp::Predicated { .. } => None,
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
        | KirOp::And(_, a, b)
        | KirOp::Or(_, a, b)
        | KirOp::Xor(_, a, b)
        | KirOp::Shl(_, a, b)
        | KirOp::Shr(_, a, b)
        | KirOp::Rem(_, a, b)
        | KirOp::Min(_, a, b)
        | KirOp::Max(_, a, b)
        | KirOp::WarpShuffle { val: a, lane: b, .. }
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
        | KirOp::Not(_, s)
        | KirOp::Rcp(_, s)
        | KirOp::Rsqrt(_, s)
        | KirOp::Cast(_, s, _)
        | KirOp::CastRounded { src: s, .. }
        | KirOp::LoadVec { ptr: s, .. }
        | KirOp::Vote { pred: s, .. }
        | KirOp::Load(_, s, _) => vec![*s],
        KirOp::StoreVec { ptr, vals, .. } => {
            let mut v = vec![*ptr];
            v.extend_from_slice(vals);
            v
        }
        KirOp::Predicated { pred, op, .. } => {
            let mut v = vec![*pred];
            v.extend(op_uses(op));
            v
        }
        KirOp::Store(p, v, _) | KirOp::AtomicAdd(p, v, _) => vec![*p, *v],
        KirOp::CpAsync { dst, src, .. } => vec![*dst, *src],
        KirOp::LdMatrix { addr, .. } => vec![*addr],
        KirOp::Mma { a, b, c, .. } => {
            let mut v = a.clone();
            v.extend_from_slice(b);
            v.extend_from_slice(c);
            v
        }
        KirOp::SharedBase(_)
        | KirOp::SharedRegion { .. }
        | KirOp::CpAsyncCommit
        | KirOp::CpAsyncWait { .. }
        | KirOp::ThreadId(_, _)
        | KirOp::BlockIdx(_, _)
        | KirOp::BlockDim(_, _)
        | KirOp::GridDim(_, _)
        | KirOp::GlobalId(_, _)
        | KirOp::LaneId(_)
        | KirOp::WarpId(_)
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
        KirOp::WarpShuffle { dst: d, val: v, lane, .. } => {
            if let Some(dt) = ty(d) {
                expect(*v, "val", &dt);
            }
            expect(*lane, "lane", &KirType::U32);
        }
        // Roadmap A2 step 4: the scalar ISA.
        KirOp::And(d, a, b) | KirOp::Or(d, a, b) | KirOp::Xor(d, a, b) => {
            if let Some(dt) = ty(d) {
                expect(*a, "a", &dt);
                expect(*b, "b", &dt);
                if is_float(&dt) {
                    errors.push(KirVerifyError::TypeMismatch {
                        var: *d,
                        block,
                        op_index,
                        role: "dst (bitwise ops take integers or Bool)",
                        expected: KirType::U32,
                        found: dt,
                    });
                }
            }
        }
        KirOp::Not(d, s) => {
            if let Some(dt) = ty(d) {
                expect(*s, "src", &dt);
            }
        }
        KirOp::Rem(d, a, b) => {
            if let Some(dt) = ty(d) {
                expect(*a, "a", &dt);
                expect(*b, "b", &dt);
                if !is_integer(&dt) {
                    errors.push(KirVerifyError::TypeMismatch {
                        var: *d,
                        block,
                        op_index,
                        role: "dst (Rem takes integers)",
                        expected: KirType::U32,
                        found: dt,
                    });
                }
            }
        }
        KirOp::Min(d, a, b) | KirOp::Max(d, a, b) => {
            if let Some(dt) = ty(d) {
                expect(*a, "a", &dt);
                expect(*b, "b", &dt);
            }
        }
        KirOp::Shl(d, a, amount) | KirOp::Shr(d, a, amount) => {
            if let Some(dt) = ty(d) {
                expect(*a, "a", &dt);
            }
            expect(*amount, "amount", &KirType::U32);
        }
        KirOp::Rcp(d, s) | KirOp::Rsqrt(d, s) => {
            if let Some(dt) = ty(d) {
                expect(*s, "src", &dt);
                if !is_float(&dt) {
                    errors.push(KirVerifyError::TypeMismatch {
                        var: *d,
                        block,
                        op_index,
                        role: "dst (Rcp/Rsqrt take F32 or F64)",
                        expected: KirType::F32,
                        found: dt,
                    });
                }
            }
        }
        KirOp::CastRounded { dst: d, ty: target, .. } => expect(*d, "dst", target),
        // (The width check pushes directly, so it runs after the last use
        // of the `expect` closure in the arm.)
        KirOp::LoadVec { dsts, ptr, .. } => {
            match ty(ptr) {
                Some(KirType::Ptr(pointee, _)) => {
                    for d in dsts {
                        expect(*d, "dst", &pointee);
                    }
                }
                Some(found) => {
                    errors.push(KirVerifyError::NotAPointer { var: *ptr, block, op_index, found })
                }
                None => {}
            }
            if !matches!(dsts.len(), 2 | 4) {
                errors.push(KirVerifyError::BadVectorWidth { block, op_index, width: dsts.len() });
            }
        }
        KirOp::StoreVec { ptr, vals, .. } => {
            match ty(ptr) {
                Some(KirType::Ptr(pointee, _)) => {
                    for v in vals {
                        expect(*v, "value", &pointee);
                    }
                }
                Some(found) => {
                    errors.push(KirVerifyError::NotAPointer { var: *ptr, block, op_index, found })
                }
                None => {}
            }
            if !matches!(vals.len(), 2 | 4) {
                errors.push(KirVerifyError::BadVectorWidth { block, op_index, width: vals.len() });
            }
        }
        KirOp::Vote { dst: d, pred, mode } => {
            expect(*pred, "pred", &KirType::Bool);
            let out = match mode {
                VoteMode::Any | VoteMode::All => KirType::Bool,
                VoteMode::Ballot => KirType::U32,
            };
            expect(*d, "dst", &out);
        }
        KirOp::LaneId(d) | KirOp::WarpId(d) => expect(*d, "dst", &KirType::U32),
        KirOp::Predicated { pred, op: inner, .. } => {
            expect(*pred, "pred", &KirType::Bool);
            if !op_dsts(inner).is_empty() || matches!(**inner, KirOp::Predicated { .. }) {
                errors.push(KirVerifyError::PredicatedValueOp { block, op_index });
            }
            // Rule 9 (roadmap A2 step 6). `Barrier` has no destination, so
            // the value-op check above lets it through, but a `bar.sync`
            // that only part of the warp reaches does not skip — it hangs.
            // Every one of the estate's 248 barrier sites is an unpredicated
            // `bar.sync 0`, and this keeps it that way.
            if matches!(**inner, KirOp::Barrier) {
                errors.push(KirVerifyError::PredicatedBarrier { block, op_index });
            }
            check_types(ir, inner, block, op_index, errors);
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
        KirOp::SharedBase(d) => match ty(d) {
            Some(KirType::Ptr(_, AddressSpace::Shared)) | None => {}
            Some(KirType::Ptr(_, found)) => errors.push(KirVerifyError::AddressSpaceMismatch {
                var: *d,
                block,
                op_index,
                role: "dst",
                expected: AddressSpace::Shared,
                found,
            }),
            Some(found) => errors.push(KirVerifyError::NotAPointer { var: *d, block, op_index, found }),
        },
        KirOp::SharedRegion { dst, region } => {
            // Rule 8, index half: an out-of-range region has no offset, so
            // the printer would silently address byte 0 of the block.
            match ir.smem_layout.regions.get(*region as usize) {
                None => errors.push(KirVerifyError::SmemRegionOutOfRange {
                    block,
                    op_index,
                    region: *region,
                    regions: ir.smem_layout.regions.len(),
                }),
                Some(r) => {
                    // The destination carries the region's declared element
                    // type, in the shared space.
                    let want = KirType::Ptr(Box::new(r.elem.clone()), AddressSpace::Shared);
                    expect(*dst, "dst", &want);
                }
            }
        }
        KirOp::CpAsync { dst, src, .. } => {
            for (var, role, expected) in
                [(*dst, "dst", AddressSpace::Shared), (*src, "src", AddressSpace::Global)]
            {
                match ty(&var) {
                    None => {}
                    Some(KirType::Ptr(_, found)) if found != expected => {
                        errors.push(KirVerifyError::AddressSpaceMismatch {
                            var,
                            block,
                            op_index,
                            role,
                            expected,
                            found,
                        });
                    }
                    Some(KirType::Ptr(_, _)) => {}
                    Some(found) => {
                        errors.push(KirVerifyError::NotAPointer { var, block, op_index, found })
                    }
                }
            }
        }
        KirOp::LdMatrix { dst, addr, .. } => {
            // The address check pushes directly, so it runs after the last
            // use of the `expect` closure (which holds `errors` mutably).
            let frag = f16x2();
            for v in dst {
                expect(*v, "fragment", &frag);
            }
            // Rule 6, count half: PTX has `.x1`, `.x2` and `.x4` and no
            // other form, so an arity outside that set has no instruction
            // to print and is caught here rather than by `ptxas`.
            if !matches!(dst.len(), 1 | 2 | 4) {
                errors.push(KirVerifyError::BadLdMatrixCount {
                    block,
                    op_index,
                    count: dst.len(),
                });
            }
            match ty(addr) {
                None | Some(KirType::Ptr(_, AddressSpace::Shared)) => {}
                Some(KirType::Ptr(_, found)) => errors.push(KirVerifyError::AddressSpaceMismatch {
                    var: *addr,
                    block,
                    op_index,
                    role: "addr",
                    expected: AddressSpace::Shared,
                    found,
                }),
                Some(found) => {
                    errors.push(KirVerifyError::NotAPointer { var: *addr, block, op_index, found })
                }
            }
        }
        KirOp::Mma { shape, a_ty, d, a, b, c } => {
            // Rule 6: the A/B fragments carry the packed pair type for
            // `a_ty` — a bf16 tile wired with f16 registers is an error
            // here, not a silently wrong `mma` — and the arities follow the
            // shape.
            let frag = a_ty.fragment_ty();
            for v in a.iter().chain(b.iter()) {
                expect(*v, "fragment", &frag);
            }
            for v in c.iter().chain(d.iter()) {
                expect(*v, "accumulator", &KirType::F32);
            }
            let (want_a, want_b, want_cd) = shape.fragment_counts();
            for (role, found, expected) in [
                ("a", a.len(), want_a),
                ("b", b.len(), want_b),
                ("c", c.len(), want_cd),
                ("d", d.len(), want_cd),
            ] {
                if found != expected {
                    errors.push(KirVerifyError::BadMmaFragmentCount {
                        block,
                        op_index,
                        shape: *shape,
                        role,
                        expected,
                        found,
                    });
                }
            }
        }
        KirOp::CpAsyncCommit
        | KirOp::CpAsyncWait { .. }
        | KirOp::Barrier
        | KirOp::SharedMemFence
        | KirOp::Matmul { .. }
        | KirOp::ElementwiseAdd { .. }
        | KirOp::Relu { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KirBuilder, KirEdge};
    use crate::kernel_ir::{MmaOperandTy, SmemLayout, SmemRegion};
    use crate::kernel_ir::VoteMode;

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
        b.terminate(KirTerminator::CondBranch(in_bounds, body.into(), exit.into()));
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
        b.terminate(KirTerminator::Branch(exit.into()));
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
        b.terminate(KirTerminator::Branch(7.into()));
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
        b.terminate(KirTerminator::CondBranch(flag, left.into(), right.into()));
        b.set_block(left);
        let x = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(x, 0));
        b.terminate(KirTerminator::Branch(join.into()));
        b.set_block(right);
        b.terminate(KirTerminator::Branch(join.into()));
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
        b.terminate(KirTerminator::CondBranch(flag, left.into(), right.into()));
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
        b.terminate(KirTerminator::Branch(header.into()));
        b.set_block(header);
        let t: VarId = 50; // defined in body below
        let h = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Neg(h, t));
        b.terminate(KirTerminator::CondBranch(flag, body.into(), exit.into()));
        b.set_block(body);
        b.emit(KirOp::Neg(t, i));
        b.terminate(KirTerminator::Branch(header.into()));
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

    // ── Rule 7: block parameters and edges (roadmap A2 step 2) ──────

    /// The grid-stride loop every element-wise kernel is made of: the
    /// header takes `idx` as a parameter, the entry edge passes the thread's
    /// first index, the back edge passes `idx + stride`. Before block
    /// parameters this shape was unexpressible (see the back-edge test
    /// above, which pins that a plain redefinition is still refused).
    fn grid_stride_loop() -> (KirBuilder, BlockId, BlockId, VarId, VarId) {
        let mut b = KirBuilder::new("grid_stride");
        let n = b.add_param("n", KirType::U32, AddressSpace::Local);
        let entry = b.new_block();
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let idx = b.add_block_param(header, KirType::U32);
        b.set_block(entry);
        let start = b.new_typed_var(KirType::U32);
        b.emit(KirOp::GlobalId(start, 0));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![start])));
        b.set_block(header);
        let more = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(more, idx, n, CmpOp::Lt));
        b.terminate(KirTerminator::CondBranch(more, body.into(), exit.into()));
        b.set_block(body);
        let stride = b.new_typed_var(KirType::U32);
        b.emit(KirOp::BlockDim(stride, 0));
        let next = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Add(next, idx, stride));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![next])));
        b.set_block(exit);
        b.terminate(KirTerminator::Return);
        (b, header, body, idx, next)
    }

    #[test]
    fn a_loop_carried_value_is_a_block_parameter() {
        let (b, _, _, _, _) = grid_stride_loop();
        assert_eq!(verify(&b.finalize()), Ok(()));
    }

    #[test]
    fn an_edge_passes_one_argument_per_parameter() {
        let (mut b, header, body, _, _) = grid_stride_loop();
        b.set_block(body);
        b.terminate(KirTerminator::Branch(header.into()));
        assert_eq!(
            verify(&b.finalize()),
            Err(vec![KirVerifyError::EdgeArityMismatch {
                block: body,
                target: header,
                expected: 1,
                found: 0
            }])
        );
    }

    #[test]
    fn an_argument_carries_its_parameters_type() {
        let (mut b, header, body, _, _) = grid_stride_loop();
        b.set_block(body);
        let wrong = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(wrong, KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) }));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![wrong])));
        assert_eq!(
            verify(&b.finalize()),
            Err(vec![KirVerifyError::TypeMismatch {
                var: wrong,
                block: body,
                op_index: TERMINATOR_INDEX,
                role: "arg",
                expected: KirType::U32,
                found: KirType::F32,
            }])
        );
    }

    #[test]
    fn an_argument_is_a_use_at_the_terminator() {
        // The back edge passes a value defined in the exit block, which
        // does not dominate the body.
        let (mut b, header, body, _, _) = grid_stride_loop();
        let exit = 3;
        b.set_block(exit);
        let late = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(late, 0));
        b.terminate(KirTerminator::Return);
        b.set_block(body);
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![late])));
        assert_eq!(
            verify(&b.finalize()),
            Err(vec![KirVerifyError::UseBeforeDef { var: late, block: body, op_index: TERMINATOR_INDEX }])
        );
    }

    #[test]
    fn a_block_parameter_reaches_only_what_its_block_dominates() {
        // `idx` is the header's parameter; the entry block does not lie
        // below the header, so it cannot read it.
        let (mut b, header, _, idx, _) = grid_stride_loop();
        b.set_block(0);
        let start = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Neg(start, idx));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![start])));
        let ir = b.finalize();
        let errs = verify(&ir).unwrap_err();
        assert!(
            errs.contains(&KirVerifyError::UseBeforeDef { var: idx, block: 0, op_index: 1 }),
            "{errs:?}"
        );
    }

    #[test]
    fn a_block_parameter_is_a_definition() {
        let (mut b, _, body, idx, _) = grid_stride_loop();
        b.set_block(body);
        b.emit(KirOp::ThreadId(idx, 0));
        let ir = b.finalize();
        let errs = verify(&ir).unwrap_err();
        assert!(
            errs.contains(&KirVerifyError::Redefined { var: idx, block: body, op_index: 2 }),
            "{errs:?}"
        );
    }

    #[test]
    fn the_entry_block_takes_no_parameters() {
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.add_block_param(entry, KirType::U32);
        b.set_block(entry);
        b.terminate(KirTerminator::Return);
        assert_eq!(verify(&b.finalize()), Err(vec![KirVerifyError::EntryBlockHasParams { count: 1 }]));
    }

    // ── Rule 4b / 8: the scalar ISA and predication (roadmap A2 step 4) ──

    fn one_block() -> (KirBuilder, VarId) {
        let mut b = KirBuilder::new("isa");
        let p = b.add_param("p", f32_ptr(), AddressSpace::Global);
        let e = b.new_block();
        b.set_block(e);
        (b, p)
    }

    // ── Roadmap A2 step 6: SmemLayout and the tensor-core set ────────

    fn region(name: &str, bytes: u32, align: u32) -> SmemRegion {
        SmemRegion { name: name.to_string(), bytes, align, elem: KirType::F32 }
    }

    /// Offsets pack in declaration order and round each start up to the
    /// region's own alignment. This is the computation the 74 FA v2
    /// accessors derive from, so it is stated as a table rather than as a
    /// property.
    #[test]
    fn region_offsets_pack_in_order_and_respect_alignment() {
        let layout = SmemLayout {
            regions: vec![region("q", 100, 16), region("kv", 8, 16), region("tail", 4, 4)],
            dynamic: false,
        };
        assert_eq!(layout.offset_of(0), Some(0));
        // 100 is not a multiple of 16, so "kv" starts at 112, not 100.
        assert_eq!(layout.offset_of(1), Some(112));
        assert_eq!(layout.offset_of(2), Some(120));
        assert_eq!(layout.total_bytes(), Some(124));
        assert_eq!(layout.offset_of(3), None);
        assert_eq!(layout.index_of("kv"), Some(1));
    }

    /// Rule 8: a layout and the flat block are alternatives. Mixing a
    /// static `.shared` declaration with an `extern` one is the sm_120
    /// illegal-address finding.
    #[test]
    fn a_kernel_is_static_or_dynamic_never_both() {
        let (mut b, _) = one_block();
        b.set_smem_layout(SmemLayout { regions: vec![region("q", 64, 16)], dynamic: false });
        b.set_shared_mem(256);
        b.terminate(KirTerminator::Return);
        let errors = verify(&b.finalize()).unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(e, KirVerifyError::SmemLayoutAndFlatBlock { .. })),
            "expected the static/dynamic exclusivity error, got {errors:?}"
        );
    }

    /// Rule 8: the budget follows `dynamic` — the same layout that
    /// overflows a static declaration fits the `extern` opt-in.
    #[test]
    fn the_budget_follows_the_dynamic_flag() {
        let big = |dynamic| {
            let (mut b, _) = one_block();
            b.set_smem_layout(SmemLayout {
                regions: vec![region("big", 64 * 1024, 16)],
                dynamic,
            });
            b.terminate(KirTerminator::Return);
            verify(&b.finalize())
        };
        // 64 KiB overflows the 48 KiB static declaration ...
        let errors = big(false).unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(e, KirVerifyError::SmemBudgetExceeded { .. })),
            "64 KiB must not fit the static budget, got {errors:?}"
        );
        // ... and fits the 99 KiB dynamic one.
        assert!(big(true).is_ok(), "64 KiB must fit the dynamic budget");
    }

    /// Rule 8: `ldmatrix` needs a 16-aligned region. The check only fires
    /// for an address taken STRAIGHT from a `SharedRegion`; anything that
    /// has been through arithmetic is left alone deliberately.
    #[test]
    fn ldmatrix_refuses_an_underaligned_region() {
        let (mut b, _) = one_block();
        b.set_smem_layout(SmemLayout {
            regions: vec![SmemRegion {
                name: "frag".to_string(),
                bytes: 256,
                align: 4,
                elem: KirType::F16,
            }],
            dynamic: false,
        });
        let addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F16), AddressSpace::Shared));
        b.emit(KirOp::SharedRegion { dst: addr, region: 0 });
        let dst: Vec<VarId> = (0..4)
            .map(|_| b.new_typed_var(KirType::Vec(Box::new(KirType::F16), 2)))
            .collect();
        b.emit(KirOp::LdMatrix { dst, addr, trans: false });
        b.terminate(KirTerminator::Return);
        let errors = verify(&b.finalize()).unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(e, KirVerifyError::SmemRegionUnaligned { .. })),
            "expected the alignment error, got {errors:?}"
        );
    }

    /// Rule 8: `SharedRegion` naming a region the layout does not have.
    #[test]
    fn a_shared_region_index_is_bounds_checked() {
        let (mut b, _) = one_block();
        b.set_smem_layout(SmemLayout { regions: vec![region("only", 64, 16)], dynamic: false });
        let p = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Shared));
        b.emit(KirOp::SharedRegion { dst: p, region: 7 });
        b.terminate(KirTerminator::Return);
        let errors = verify(&b.finalize()).unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(e, KirVerifyError::SmemRegionOutOfRange { .. })),
            "expected the out-of-range error, got {errors:?}"
        );
    }

    /// Rule 6: the fragment arities follow the shape. An m16n8k8 tile wired
    /// with m16n8k16's four A registers is caught here, not by `ptxas`.
    #[test]
    fn mma_fragment_counts_follow_the_shape() {
        let (mut b, _) = one_block();
        let frag = |b: &mut KirBuilder| b.new_typed_var(KirType::Vec(Box::new(KirType::F16), 2));
        let acc = |b: &mut KirBuilder| b.new_typed_var(KirType::F32);
        let a: Vec<VarId> = (0..4).map(|_| frag(&mut b)).collect();
        let bb: Vec<VarId> = (0..1).map(|_| frag(&mut b)).collect();
        let c: Vec<VarId> = (0..4).map(|_| acc(&mut b)).collect();
        let d: Vec<VarId> = (0..4).map(|_| acc(&mut b)).collect();
        for v in a.iter().chain(bb.iter()) {
            b.emit(KirOp::Const(
                *v,
                KirConst {
                    ty: KirType::Vec(Box::new(KirType::F16), 2),
                    value: ConstValue::U32(0),
                },
            ));
        }
        for v in c.iter() {
            b.emit(KirOp::Const(*v, KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) }));
        }
        // m16n8k8 takes TWO a registers; this passes four.
        b.emit(KirOp::Mma { shape: MmaShape::M16N8K8, a_ty: MmaOperandTy::F16, d, a, b: bb, c });
        b.terminate(KirTerminator::Return);
        let errors = verify(&b.finalize()).unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                KirVerifyError::BadMmaFragmentCount { role: "a", expected: 2, found: 4, .. }
            )),
            "expected the a-arity error, got {errors:?}"
        );
    }

    /// Rule 9: a barrier the whole warp does not reach hangs; it does not
    /// skip. `Barrier` has no destination, so the value-op check lets it
    /// through and this is what refuses it.
    #[test]
    fn a_barrier_must_not_be_predicated() {
        let (mut b, _) = one_block();
        let t = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(t, 0));
        let p = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(p, t, t, CmpOp::Eq));
        b.emit(KirOp::Predicated { pred: p, negate: false, op: Box::new(KirOp::Barrier) });
        b.terminate(KirTerminator::Return);
        let errors = verify(&b.finalize()).unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(e, KirVerifyError::PredicatedBarrier { .. })),
            "expected the predicated-barrier error, got {errors:?}"
        );
    }

    #[test]
    fn rem_takes_integers() {
        let (mut b, _) = one_block();
        let x = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(x, KirConst { ty: KirType::F32, value: ConstValue::F32(1.0) }));
        let r = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Rem(r, x, x));
        b.terminate(KirTerminator::Return);
        let errs = verify(&b.finalize()).unwrap_err();
        assert!(matches!(errs[..], [KirVerifyError::TypeMismatch { var, .. }] if var == r), "{errs:?}");
    }

    #[test]
    fn a_shift_amount_is_u32() {
        let (mut b, _) = one_block();
        let x = b.new_typed_var(KirType::U64);
        b.emit(KirOp::Const(x, KirConst { ty: KirType::U64, value: ConstValue::U64(8) }));
        let d = b.new_typed_var(KirType::U64);
        b.emit(KirOp::Shl(d, x, x));
        b.terminate(KirTerminator::Return);
        assert_eq!(
            verify(&b.finalize()),
            Err(vec![KirVerifyError::TypeMismatch {
                var: x,
                block: 0,
                op_index: 1,
                role: "amount",
                expected: KirType::U32,
                found: KirType::U64,
            }])
        );
    }

    #[test]
    fn a_vector_load_is_two_or_four_wide() {
        let (mut b, p) = one_block();
        let d: Vec<VarId> = (0..3).map(|_| b.new_typed_var(KirType::F32)).collect();
        b.emit(KirOp::LoadVec { dsts: d, ptr: p, space: AddressSpace::Global });
        b.terminate(KirTerminator::Return);
        assert_eq!(
            verify(&b.finalize()),
            Err(vec![KirVerifyError::BadVectorWidth { block: 0, op_index: 0, width: 3 }])
        );
    }

    #[test]
    fn a_vector_load_is_pointee_typed() {
        let (mut b, p) = one_block();
        let d0 = b.new_typed_var(KirType::F32);
        let d1 = b.new_typed_var(KirType::U32);
        b.emit(KirOp::LoadVec { dsts: vec![d0, d1], ptr: p, space: AddressSpace::Global });
        b.terminate(KirTerminator::Return);
        assert_eq!(
            verify(&b.finalize()),
            Err(vec![KirVerifyError::TypeMismatch {
                var: d1,
                block: 0,
                op_index: 0,
                role: "dst",
                expected: KirType::F32,
                found: KirType::U32,
            }])
        );
    }

    #[test]
    fn a_ballot_is_u32_and_any_is_bool() {
        let (mut b, _) = one_block();
        let flag = b.new_typed_var(KirType::Bool);
        let t = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(t, 0));
        b.emit(KirOp::Cmp(flag, t, t, CmpOp::Eq));
        let bits = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Vote { dst: bits, pred: flag, mode: VoteMode::Ballot });
        let any = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Vote { dst: any, pred: flag, mode: VoteMode::Any });
        let wrong = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Vote { dst: wrong, pred: flag, mode: VoteMode::All });
        b.terminate(KirTerminator::Return);
        assert_eq!(
            verify(&b.finalize()),
            Err(vec![KirVerifyError::TypeMismatch {
                var: wrong,
                block: 0,
                op_index: 4,
                role: "dst",
                expected: KirType::Bool,
                found: KirType::U32,
            }])
        );
    }

    #[test]
    fn a_predicated_store_verifies_and_a_predicated_definition_does_not() {
        let (mut b, p) = one_block();
        let t = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(t, 0));
        let flag = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(flag, t, t, CmpOp::Eq));
        let v = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(v, KirConst { ty: KirType::F32, value: ConstValue::F32(1.0) }));
        b.emit(KirOp::Predicated {
            pred: flag,
            negate: false,
            op: Box::new(KirOp::Store(p, v, AddressSpace::Global)),
        });
        b.terminate(KirTerminator::Return);
        assert_eq!(verify(&b.finalize()), Ok(()));

        let (mut b, p) = one_block();
        let t = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(t, 0));
        let flag = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(flag, t, t, CmpOp::Eq));
        let v = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Predicated {
            pred: flag,
            negate: true,
            op: Box::new(KirOp::Load(v, p, AddressSpace::Global)),
        });
        b.terminate(KirTerminator::Return);
        assert_eq!(
            verify(&b.finalize()),
            Err(vec![KirVerifyError::PredicatedValueOp { block: 0, op_index: 2 }])
        );
    }

    #[test]
    fn a_predicated_op_is_still_type_checked() {
        let (mut b, p) = one_block();
        let t = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(t, 0));
        let flag = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(flag, t, t, CmpOp::Eq));
        b.emit(KirOp::Predicated {
            pred: flag,
            negate: false,
            op: Box::new(KirOp::Store(p, t, AddressSpace::Global)),
        });
        b.terminate(KirTerminator::Return);
        assert_eq!(
            verify(&b.finalize()),
            Err(vec![KirVerifyError::TypeMismatch {
                var: t,
                block: 0,
                op_index: 2,
                role: "value",
                expected: KirType::F32,
                found: KirType::U32,
            }])
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
        b.terminate(KirTerminator::CondBranch(n, exit.into(), exit.into()));
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

    fn shared_u8_ptr() -> KirType {
        KirType::Ptr(Box::new(KirType::I8), AddressSpace::Shared)
    }

    /// One 16-byte async copy, committed and waited, then a barrier: the
    /// shape every cp.async pipeline stage has.
    fn async_copy_stage() -> KirBuilder {
        let mut b = KirBuilder::new("t");
        let src = b.add_param("src", KirType::Ptr(Box::new(KirType::I8), AddressSpace::Global), AddressSpace::Global);
        let entry = b.new_block();
        b.set_block(entry);
        let smem = b.new_typed_var(shared_u8_ptr());
        b.emit(KirOp::SharedBase(smem));
        b.emit(KirOp::CpAsync { dst: smem, src, bytes: 16 });
        b.emit(KirOp::CpAsyncCommit);
        b.emit(KirOp::CpAsyncWait { pending: 0 });
        b.emit(KirOp::Barrier);
        b
    }

    #[test]
    fn a_committed_and_waited_async_copy_verifies_and_requires_the_feature() {
        let mut b = async_copy_stage();
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(verify(&ir), Ok(()));
        assert!(ir.required_features.contains(crate::FeatureSet::ASYNC_COPY));
        assert!(ir.required_features.contains(crate::FeatureSet::SHARED_MEMORY));
    }

    #[test]
    fn an_async_copy_left_uncommitted_at_block_end_is_reported() {
        let mut b = KirBuilder::new("t");
        let src = b.add_param("src", KirType::Ptr(Box::new(KirType::I8), AddressSpace::Global), AddressSpace::Global);
        let entry = b.new_block();
        b.set_block(entry);
        let smem = b.new_typed_var(shared_u8_ptr());
        b.emit(KirOp::SharedBase(smem));
        b.emit(KirOp::CpAsync { dst: smem, src, bytes: 8 });
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::UncommittedAsyncCopy { block: 0, op_index: TERMINATOR_INDEX }])
        );
    }

    #[test]
    fn a_wait_cannot_cover_an_uncommitted_copy() {
        let mut b = KirBuilder::new("t");
        let src = b.add_param("src", KirType::Ptr(Box::new(KirType::I8), AddressSpace::Global), AddressSpace::Global);
        let entry = b.new_block();
        b.set_block(entry);
        let smem = b.new_typed_var(shared_u8_ptr());
        b.emit(KirOp::SharedBase(smem));
        b.emit(KirOp::CpAsync { dst: smem, src, bytes: 4 });
        b.emit(KirOp::CpAsyncWait { pending: 0 });
        b.emit(KirOp::CpAsyncCommit);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::UncommittedAsyncCopy { block: 0, op_index: 2 }])
        );
    }

    #[test]
    fn a_wait_naming_more_groups_than_committed_is_reported() {
        let mut b = async_copy_stage();
        b.emit(KirOp::CpAsyncWait { pending: 2 }); // one group was committed, and the earlier wait drained it
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::AsyncWaitExceedsGroups {
                block: 0,
                op_index: 5,
                pending: 2,
                committed: 0,
            }])
        );
    }

    #[test]
    fn a_two_stage_pipeline_may_keep_one_group_in_flight() {
        let mut b = KirBuilder::new("t");
        let src = b.add_param("src", KirType::Ptr(Box::new(KirType::I8), AddressSpace::Global), AddressSpace::Global);
        let entry = b.new_block();
        b.set_block(entry);
        let smem = b.new_typed_var(shared_u8_ptr());
        b.emit(KirOp::SharedBase(smem));
        b.emit(KirOp::CpAsync { dst: smem, src, bytes: 16 });
        b.emit(KirOp::CpAsyncCommit);
        b.emit(KirOp::CpAsync { dst: smem, src, bytes: 16 });
        b.emit(KirOp::CpAsyncCommit);
        b.emit(KirOp::CpAsyncWait { pending: 1 }); // the first group has landed, the second may not have
        b.emit(KirOp::Barrier);
        b.emit(KirOp::CpAsyncWait { pending: 0 });
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(verify(&ir), Ok(()));
    }

    #[test]
    fn an_async_copy_width_cp_async_does_not_have_is_reported() {
        let mut b = KirBuilder::new("t");
        let src = b.add_param("src", KirType::Ptr(Box::new(KirType::I8), AddressSpace::Global), AddressSpace::Global);
        let entry = b.new_block();
        b.set_block(entry);
        let smem = b.new_typed_var(shared_u8_ptr());
        b.emit(KirOp::SharedBase(smem));
        b.emit(KirOp::CpAsync { dst: smem, src, bytes: 12 });
        b.emit(KirOp::CpAsyncCommit);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::BadAsyncCopyWidth { block: 0, op_index: 1, bytes: 12 }])
        );
    }

    #[test]
    fn an_async_copy_between_the_wrong_state_spaces_is_reported() {
        // dst is a global pointer and src a shared one: both roles wrong.
        let mut b = KirBuilder::new("t");
        let g = b.add_param("g", KirType::Ptr(Box::new(KirType::I8), AddressSpace::Global), AddressSpace::Global);
        let entry = b.new_block();
        b.set_block(entry);
        let smem = b.new_typed_var(shared_u8_ptr());
        b.emit(KirOp::SharedBase(smem));
        b.emit(KirOp::CpAsync { dst: g, src: smem, bytes: 16 });
        b.emit(KirOp::CpAsyncCommit);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![
                KirVerifyError::AddressSpaceMismatch {
                    var: g,
                    block: 0,
                    op_index: 1,
                    role: "dst",
                    expected: AddressSpace::Shared,
                    found: AddressSpace::Global,
                },
                KirVerifyError::AddressSpaceMismatch {
                    var: smem,
                    block: 0,
                    op_index: 1,
                    role: "src",
                    expected: AddressSpace::Global,
                    found: AddressSpace::Shared,
                },
            ])
        );
    }

    #[test]
    fn a_shared_base_typed_as_a_global_pointer_is_reported() {
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.set_block(entry);
        let p = b.new_typed_var(f32_ptr());
        b.emit(KirOp::SharedBase(p));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::AddressSpaceMismatch {
                var: p,
                block: 0,
                op_index: 0,
                role: "dst",
                expected: AddressSpace::Shared,
                found: AddressSpace::Global,
            }])
        );
    }

    fn frag() -> KirType {
        KirType::Vec(Box::new(KirType::F16), 2)
    }

    fn f16_shared_ptr() -> KirType {
        KirType::Ptr(Box::new(KirType::F16), AddressSpace::Shared)
    }

    /// ldmatrix A and B fragments from shared memory, an all-zero
    /// accumulator, one mma: the tile every tensor-core loop is made of.
    fn one_tile() -> (KirBuilder, [VarId; 4], [VarId; 4], [VarId; 4], [VarId; 4]) {
        let mut b = KirBuilder::new("t");
        let entry = b.new_block();
        b.set_block(entry);
        let smem = b.new_typed_var(f16_shared_ptr());
        b.emit(KirOp::SharedBase(smem));
        let a = [b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag())];
        b.emit(KirOp::LdMatrix { dst: a.to_vec(), addr: smem, trans: false });
        let bb = [b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag())];
        b.emit(KirOp::LdMatrix { dst: bb.to_vec(), addr: smem, trans: true });
        let mut c = [0; 4];
        for slot in &mut c {
            *slot = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Const(*slot, KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) }));
        }
        let d = [b.new_typed_var(KirType::F32), b.new_typed_var(KirType::F32), b.new_typed_var(KirType::F32), b.new_typed_var(KirType::F32)];
        (b, a, bb, c, d)
    }

    #[test]
    fn a_tensor_core_tile_verifies_and_requires_the_feature() {
        let (mut b, a, bb, c, d) = one_tile();
        b.emit(KirOp::Mma { shape: MmaShape::M16N8K16, a_ty: MmaOperandTy::F16, d: d.to_vec(), a: a.to_vec(), b: vec![bb[0], bb[1]], c: c.to_vec() });
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(verify(&ir), Ok(()));
        assert!(ir.required_features.contains(crate::FeatureSet::TENSOR_CORES));
    }

    #[test]
    fn an_accumulator_fragment_typed_as_a_packed_register_is_reported() {
        let (mut b, a, bb, mut c, d) = one_tile();
        let wrong = b.new_typed_var(frag());
        b.emit(KirOp::Const(wrong, KirConst { ty: frag(), value: ConstValue::U32(0) }));
        c[2] = wrong;
        b.emit(KirOp::Mma { shape: MmaShape::M16N8K16, a_ty: MmaOperandTy::F16, d: d.to_vec(), a: a.to_vec(), b: vec![bb[0], bb[1]], c: c.to_vec() });
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::TypeMismatch {
                var: wrong,
                block: 0,
                op_index: 8,
                role: "accumulator",
                expected: KirType::F32,
                found: frag(),
            }])
        );
    }

    #[test]
    fn a_b_fragment_typed_as_f32_is_reported() {
        let (mut b, a, bb, c, d) = one_tile();
        b.emit(KirOp::Mma { shape: MmaShape::M16N8K16, a_ty: MmaOperandTy::F16, d: d.to_vec(), a: a.to_vec(), b: vec![bb[0], c[0]], c: c.to_vec() });
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::TypeMismatch {
                var: c[0],
                block: 0,
                op_index: 7,
                role: "fragment",
                expected: frag(),
                found: KirType::F32,
            }])
        );
    }

    #[test]
    fn ldmatrix_from_a_global_pointer_is_reported() {
        let mut b = KirBuilder::new("t");
        let g = b.add_param("g", KirType::Ptr(Box::new(KirType::F16), AddressSpace::Global), AddressSpace::Global);
        let entry = b.new_block();
        b.set_block(entry);
        let a = [b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag())];
        b.emit(KirOp::LdMatrix { dst: a.to_vec(), addr: g, trans: false });
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::AddressSpaceMismatch {
                var: g,
                block: 0,
                op_index: 0,
                role: "addr",
                expected: AddressSpace::Shared,
                found: AddressSpace::Global,
            }])
        );
    }

    #[test]
    fn a_fragment_register_listed_twice_as_a_destination_breaks_ssa() {
        let (mut b, a, bb, c, mut d) = one_tile();
        d[3] = d[0];
        b.emit(KirOp::Mma { shape: MmaShape::M16N8K16, a_ty: MmaOperandTy::F16, d: d.to_vec(), a: a.to_vec(), b: vec![bb[0], bb[1]], c: c.to_vec() });
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(
            verify(&ir),
            Err(vec![KirVerifyError::Redefined { var: d[0], block: 0, op_index: 7 }])
        );
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
