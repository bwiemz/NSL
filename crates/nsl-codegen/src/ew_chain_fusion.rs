//! Elementwise-chain fusion over the source-AD adjoint tape.
//!
//! The generated backward decomposes into long runs of single-consumer
//! elementwise ops (Mul/Add/Sub/Div/Neg, interleaved with `reduce_to_shape`
//! bookkeeping) — each lowered as one FFI call = one kernel launch = one
//! full-size intermediate through the caching allocator. At 500M that is
//! ~800 elementwise launches per micro-batch. This pass rewrites each
//! maximal single-reader run into ONE `Passthrough("fused_ew:v1:...")` op;
//! the lowerer synthesizes a chain-specific PTX kernel (one load per input,
//! register-to-register `.rn` arithmetic in exact tape order, one store)
//! and the runtime FFI (`nsl_fused_ew_chain`) falls back to a decomposed
//! replay of the original ops when the uniform-shape/device/dtype fast-path
//! gate fails — so the transform is BIT-EXACT on every path by construction.
//!
//! Discipline copied from `fuse_swiglu_gate_backward` / `fuse_rmsnorm_dx_residual`
//! (the litigated adjoint-peephole precedents):
//! - runs at the stmt.rs adjoint window: after `eliminate_dead_gradients`,
//!   BEFORE the CCR splice, FreeTensor insertion, and the arena analyze;
//! - `assert_ids_in_space(Adjoint)` on entry; renumber ONLY via
//!   `renumber_adjoint_ops`, once, at exit;
//! - interiors require exactly one reader AND not-in-`needed`; the tail keeps
//!   its VarId so downstream readers / FASE hook keys / CCR scans are untouched;
//! - DELETION-ONLY + rewrite-in-place-at-tail: nothing is ever inserted, so
//!   the wgrad planner's 3-op adjacency can only be preserved, never broken.
//!
//! Not a registered pass — a lowering peephole, same considered line as
//! `wgrad_fusion` (see NOT_A_PASS in pass_registry_drift.rs).
//!
//! Kill-switches (compile-time env, `fuse_rmsnorm_dx_residual` precedent):
//! `NSL_FUSE_ELEMENTWISE_BWD=0` disables the chain fuser,
//! `NSL_FUSE_SCALAR_IMM=0` disables the standalone scalar-immediate sweep.
//!
//! v1 scope notes: skipped entirely under `--layerwise-accum` (the CSLA range
//! partition is positional over this tape; a follow-up can enable it with a
//! range-boundary-aware differential gate) and not applied to the `@backward`
//! grad-block window. `reduce_to_shape` is a chain BARRIER (measured: on the
//! prod tape every chain-reachable rts was a real GQA reduce that refused
//! the uniform-shape gate and replayed — see the barrier comment in
//! `try_join`; the RtsCheck wire machinery is retained for the
//! shape-filtered revival). A `Constant` feeding a LEFT operand slot is a chain
//! BARRIER and excluded from the sweep: the baseline reconciles the right
//! operand onto the (CPU-resident) constant's device and computes the chain
//! in host f64, which a device-f32 kernel does not reproduce bit-for-bit once
//! more than one op runs before the narrowing (D2b in the campaign record).

use std::collections::{HashMap, HashSet};

use crate::wengert::{PrimalOp, VarId, WengertOp, WengertType};

/// Fused-chain step opcodes. u8-stable: byte values are the descriptor v1
/// wire format shared with the runtime (`nsl_fused_ew_chain`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EwOpcode {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Neg = 4,
    /// `reduce_to_shape` pass-through member: identity on the fast path
    /// (the like-ref's shape is checked equal to the uniform shape by the
    /// runtime gate), a real reduce on the replay path.
    RtsCheck = 5,
}

/// A step operand. Encoded kinds (descriptor v1): Input=0, Prev=1, Imm=2,
/// absent=255 (unary rhs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// External tensor input, by slot index (first-use order, < n_inputs).
    Input(u8),
    /// Result of an earlier step in this chain.
    Prev(u8),
    /// Compile-time f32 immediate (bits). Only ever a RIGHT operand — a
    /// constant LEFT operand is a barrier (see module docs).
    Imm(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainStep {
    pub op: EwOpcode,
    pub lhs: Operand,
    /// None for unary (Neg).
    pub rhs: Option<Operand>,
}

/// The complete, order-preserving signature of one fused chain. The encoded
/// name IS the Passthrough op name — deterministic, TapeDigest-visible, and
/// parseable at lowering time (no side tables for the CCR remap to miss).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSig {
    pub steps: Vec<ChainStep>,
    pub n_inputs: u8,
}

pub const FUSED_EW_PREFIX: &str = "fused_ew:v1:";

/// Matcher caps. MAX_INPUTS doubles as the FFI arity (6 handle slots) and
/// the profitability ceiling; LOOKAHEAD bounds how far an external input's
/// liveness can be extended toward the tail (tape positions).
pub const MAX_INPUTS: usize = 6;
pub const MAX_ARITH_STEPS: usize = 6;
pub const MAX_IMMS: usize = 4;
pub const LOOKAHEAD: usize = 8;
pub const MIN_ARITH_STEPS: usize = 2;

impl EwOpcode {
    fn mnemonic(self) -> &'static str {
        match self {
            EwOpcode::Add => "add",
            EwOpcode::Sub => "sub",
            EwOpcode::Mul => "mul",
            EwOpcode::Div => "div",
            EwOpcode::Neg => "neg",
            EwOpcode::RtsCheck => "rts",
        }
    }

    fn from_mnemonic(s: &str) -> Option<Self> {
        Some(match s {
            "add" => EwOpcode::Add,
            "sub" => EwOpcode::Sub,
            "mul" => EwOpcode::Mul,
            "div" => EwOpcode::Div,
            "neg" => EwOpcode::Neg,
            "rts" => EwOpcode::RtsCheck,
            _ => return None,
        })
    }
}

fn operand_token(op: Operand) -> String {
    match op {
        Operand::Input(i) => format!("i{i}"),
        Operand::Prev(k) => format!("p{k}"),
        Operand::Imm(bits) => format!("x{bits:08x}"),
    }
}

fn parse_operand(tok: &str) -> Option<Operand> {
    let (kind, rest) = tok.split_at(1);
    match kind {
        "i" => rest.parse::<u8>().ok().map(Operand::Input),
        "p" => rest.parse::<u8>().ok().map(Operand::Prev),
        "x" => u32::from_str_radix(rest, 16).ok().map(Operand::Imm),
        _ => None,
    }
}

impl ChainSig {
    /// Render the Passthrough name, e.g.
    /// `fused_ew:v1:mul(i0,i1);rts(p0,i2);add(p1,i3)`.
    pub fn encode_name(&self) -> String {
        debug_assert_eq!(self.n_inputs, self.derived_n_inputs());
        let steps: Vec<String> = self
            .steps
            .iter()
            .map(|s| match s.rhs {
                Some(r) => format!(
                    "{}({},{})",
                    s.op.mnemonic(),
                    operand_token(s.lhs),
                    operand_token(r)
                ),
                None => format!("{}({})", s.op.mnemonic(), operand_token(s.lhs)),
            })
            .collect();
        format!("{FUSED_EW_PREFIX}{}", steps.join(";"))
    }

    /// Strict inverse of [`encode_name`]. Returns None for anything that is
    /// not a well-formed v1 signature.
    pub fn parse(name: &str) -> Option<ChainSig> {
        let body = name.strip_prefix(FUSED_EW_PREFIX)?;
        if body.is_empty() {
            return None;
        }
        let mut steps = Vec::new();
        for part in body.split(';') {
            let open = part.find('(')?;
            let close = part.strip_suffix(')')?;
            let op = EwOpcode::from_mnemonic(&part[..open])?;
            let args = &close[open + 1..];
            let mut it = args.split(',');
            let lhs = parse_operand(it.next()?)?;
            let rhs = match it.next() {
                Some(tok) => Some(parse_operand(tok)?),
                None => None,
            };
            if it.next().is_some() {
                return None;
            }
            // An immediate LEFT operand is outside the contract: the wire
            // format's single imm field carries only the RHS (the matcher
            // hard-barriers const-LEFT for host-f64 semantics), and the
            // runtime rejects such a record. Refuse it here too so a
            // malformed name can never lower (review finding N3).
            if matches!(lhs, Operand::Imm(_)) {
                return None;
            }
            match op {
                EwOpcode::Neg => {
                    if rhs.is_some() {
                        return None;
                    }
                }
                _ => {
                    rhs?;
                }
            }
            steps.push(ChainStep { op, lhs, rhs });
        }
        let sig = ChainSig {
            n_inputs: 0,
            steps,
        };
        let n = sig.derived_n_inputs();
        if n as usize > MAX_INPUTS {
            return None;
        }
        Some(ChainSig {
            n_inputs: n,
            steps: sig.steps,
        })
    }

    fn derived_n_inputs(&self) -> u8 {
        let mut max = 0u8;
        let mut any = false;
        for s in &self.steps {
            for o in std::iter::once(s.lhs).chain(s.rhs) {
                if let Operand::Input(i) = o {
                    any = true;
                    max = max.max(i);
                }
            }
        }
        if any {
            max + 1
        } else {
            0
        }
    }

    /// Kernel entry name: `nsl_fused_ew_<fnv1a64 of encode_name, 16 hex>`.
    /// Content-derived so identical per-layer chains share one PTX blob.
    pub fn kernel_name(&self) -> String {
        // One FNV-1a definition per crate (review): reuse c_wrapper's.
        let h = crate::c_wrapper::fnv1a_hash(&self.encode_name());
        format!("nsl_fused_ew_{h:016x}")
    }

    /// Descriptor v1 wire bytes (shared contract with `nsl_fused_ew_chain`):
    /// `[ver=1][n_steps][n_inputs][flags=0]` then per step
    /// `[opcode][lhs_kind][lhs_idx][rhs_kind][rhs_idx][imm_bits u32 LE]`.
    /// Operand kinds: Input=0, Prev=1, Imm=2, absent=255. The imm slot is
    /// only ever the RIGHT operand, so one `imm_bits` per step suffices.
    pub fn descriptor_bytes(&self) -> Vec<u8> {
        fn enc(op: Option<Operand>) -> (u8, u8, u32) {
            match op {
                Some(Operand::Input(i)) => (0, i, 0),
                Some(Operand::Prev(k)) => (1, k, 0),
                Some(Operand::Imm(bits)) => (2, 0, bits),
                None => (255, 0, 0),
            }
        }
        let mut out = Vec::with_capacity(4 + self.steps.len() * 9);
        out.push(1u8);
        out.push(self.steps.len() as u8);
        out.push(self.n_inputs);
        out.push(0u8);
        for s in &self.steps {
            // Real assert, not debug_assert: CI ships release, and a left-Imm
            // record silently corrupts the replay's operand resolution (the
            // step's single imm field is rhs-only). Compile-time cost only.
            assert!(
                !matches!(s.lhs, Operand::Imm(_)),
                "an immediate LEFT operand is a barrier, never encoded"
            );
            let (lk, li, limm) = enc(Some(s.lhs));
            let (rk, ri, rimm) = enc(s.rhs);
            assert_eq!(limm, 0);
            out.push(s.op as u8);
            out.push(lk);
            out.push(li);
            out.push(rk);
            out.push(ri);
            out.extend_from_slice(&rimm.to_le_bytes());
        }
        out
    }
}

/// Anti-vacuity counters printed by the `[fuse]` marker at the call site.
#[derive(Debug, Default, Clone, Copy)]
pub struct FuseStats {
    pub chains: usize,
    /// Standalone kernel launches removed (fusable arithmetic members minus
    /// the one fused launch that replaces each chain).
    pub device_ops_elided: usize,
    /// `reduce_to_shape` FFI calls absorbed as RtsCheck members.
    pub reduces_absorbed: usize,
    /// `Constant` ops absorbed as PTX immediates.
    pub imms_baked: usize,
}

fn is_fusable_arith(op: &PrimalOp) -> bool {
    matches!(
        op,
        PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div | PrimalOp::Neg
    )
}

fn arith_opcode(op: &PrimalOp) -> EwOpcode {
    match op {
        PrimalOp::Add => EwOpcode::Add,
        PrimalOp::Sub => EwOpcode::Sub,
        PrimalOp::Mul => EwOpcode::Mul,
        PrimalOp::Div => EwOpcode::Div,
        PrimalOp::Neg => EwOpcode::Neg,
        _ => unreachable!("arith_opcode on non-fusable op"),
    }
}

fn is_rts(op: &WengertOp) -> bool {
    matches!(&op.op, PrimalOp::Passthrough(n) if n == "reduce_to_shape") && op.inputs.len() == 2
}

/// A chain under construction / accepted.
struct Chain {
    /// Tape indices of the member ops, ascending; last is the tail.
    members: Vec<usize>,
    steps: Vec<ChainStep>,
    /// External tensor VarIds in first-use order (Input slot order).
    externals: Vec<VarId>,
    /// Tape indices of absorbed `Constant` ops (deleted with the members).
    const_members: Vec<usize>,
    imms: usize,
    arith_steps: usize,
}

impl Chain {
    fn input_slot(&mut self, v: VarId) -> Option<u8> {
        if let Some(pos) = self.externals.iter().position(|&e| e == v) {
            return Some(pos as u8);
        }
        if self.externals.len() >= MAX_INPUTS {
            return None;
        }
        self.externals.push(v);
        Some((self.externals.len() - 1) as u8)
    }
}

/// Fuse maximal single-reader elementwise runs in the adjoint tape.
///
/// See the module docs for the full contract. Returns anti-vacuity stats;
/// `NSL_FUSE_ELEMENTWISE_BWD=0` (compile-time env) disables and returns zeros.
pub fn run_backward_ew_fusion(
    ops: &mut Vec<WengertOp>,
    needed: &HashSet<VarId>,
    var_types: &HashMap<VarId, WengertType>,
) -> FuseStats {
    if std::env::var("NSL_FUSE_ELEMENTWISE_BWD").ok().as_deref() == Some("0") {
        return FuseStats::default();
    }
    // These take a BARE `&mut Vec<WengertOp>`, and both `adjoint.ops` and
    // `primal.ops` have that type — passing the primal here would renumber
    // primal ids into the adjoint half with nothing to say so (see the
    // sibling peepholes for the full account).
    crate::wengert::assert_ids_in_space(
        ops,
        crate::wengert::IdSpace::Adjoint,
        "ew_chain_fusion::run_backward_ew_fusion (argument)",
    );
    debug_assert!(
        !ops.iter().any(|op| matches!(op.op, PrimalOp::FreeTensor)),
        "run_backward_ew_fusion must run before free insertion"
    );

    let mut reads: HashMap<VarId, usize> = HashMap::new();
    let mut producer: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for inp in &op.inputs {
            *reads.entry(*inp).or_default() += 1;
        }
        producer.insert(op.result, i);
    }
    // The baseline binary lowering dispatches on var_types (Integer+Integer
    // folds to Cranelift ALU, Scalar promotes to a tensor); the fused/scalar
    // paths pass raw values as tensor handles. Adjoint arithmetic is
    // Tensor-typed today, but nothing upstream guarantees it — so a
    // non-Tensor-typed operand or result is a hard barrier (review finding
    // N2). Missing entries default to Tensor, matching the lowering's own
    // `unwrap_or(WengertType::Tensor)` convention.
    let is_tensor = |v: VarId| -> bool {
        matches!(
            var_types.get(&v).copied().unwrap_or(WengertType::Tensor),
            WengertType::Tensor
        )
    };

    let mut claimed = vec![false; ops.len()];
    let mut chains: Vec<Chain> = Vec::new();

    for start in 0..ops.len() {
        if claimed[start] || !is_fusable_arith(&ops[start].op) {
            continue;
        }
        // Build the candidate chain starting at `start`.
        let mut chain = Chain {
            members: Vec::new(),
            steps: Vec::new(),
            externals: Vec::new(),
            const_members: Vec::new(),
            imms: 0,
            arith_steps: 0,
        };
        // `flowing` = the VarId produced by the last accepted member; its
        // step index is steps.len()-1 (an RtsCheck aliases its lhs register
        // in the kernel, but is still a step for Prev addressing).
        let mut flowing: Option<VarId> = None;

        let mut idx = start;
        loop {
            if !is_tensor(ops[idx].result)
                || !ops[idx].inputs.iter().all(|&v| is_tensor(v))
            {
                break;
            }
            let accepted = try_join(
                &mut chain, ops, idx, flowing, needed, &reads, &producer, &claimed,
            );
            if !accepted {
                break;
            }
            chain.members.push(idx);
            flowing = Some(ops[idx].result);
            let cur = ops[idx].result;
            // Stop growing if the flowing value cannot be an interior:
            // multi-reader, needed, or no consumer within the lookahead.
            if reads.get(&cur).copied() != Some(1) || needed.contains(&cur) {
                break;
            }
            let Some(next) = find_unique_consumer(ops, idx, cur) else {
                break;
            };
            if claimed[next] {
                break;
            }
            idx = next;
        }

        if chain.arith_steps >= MIN_ARITH_STEPS
            && chain.arith_steps <= MAX_ARITH_STEPS
            && register_belt_ok(&chain)
        {
            for &m in &chain.members {
                claimed[m] = true;
            }
            for &c in &chain.const_members {
                claimed[c] = true;
            }
            chains.push(chain);
        }
    }

    if chains.is_empty() {
        return FuseStats::default();
    }

    // Rewrite: tail op becomes the fused Passthrough (result VarId kept —
    // downstream readers, `needed` membership, FASE hook keys, and the CCR
    // consumer scan are all untouched); every other member + absorbed
    // Constant is deleted. Deletion-only: nothing is inserted.
    let mut stats = FuseStats::default();
    let mut removed: HashSet<usize> = HashSet::new();
    for chain in &chains {
        let &tail = chain.members.last().expect("accepted chain has members");
        let sig = ChainSig {
            n_inputs: chain.externals.len() as u8,
            steps: chain.steps.clone(),
        };
        ops[tail].op = PrimalOp::Passthrough(sig.encode_name());
        ops[tail].inputs = chain.externals.clone();
        for &m in &chain.members {
            if m != tail {
                removed.insert(m);
            }
        }
        for &c in &chain.const_members {
            removed.insert(c);
        }
        stats.chains += 1;
        stats.device_ops_elided += chain.arith_steps - 1;
        stats.reduces_absorbed += chain.steps.len() - chain.arith_steps;
        stats.imms_baked += chain.imms;
    }
    let mut idx = 0;
    ops.retain(|_| {
        let keep = !removed.contains(&idx);
        idx += 1;
        keep
    });
    crate::wengert::renumber_adjoint_ops(ops);
    stats
}

/// Locate the single consumer of `v` after `from`, within LOOKAHEAD tape
/// positions. The caller has already established `reads[v] == 1`, so the
/// first hit is the only one; beyond the lookahead the chain simply ends
/// (bounds the liveness extension of external inputs).
fn find_unique_consumer(ops: &[WengertOp], from: usize, v: VarId) -> Option<usize> {
    let end = (from + 1 + LOOKAHEAD).min(ops.len());
    (from + 1..end).find(|&j| ops[j].inputs.contains(&v))
}

/// Attempt to append `ops[idx]` to the chain. TRANSACTIONAL: on rejection
/// the chain is rolled back to its entry state, so a partially-resolved op
/// can never leave phantom externals/imms behind (the fused op's input list
/// and the encoded signature must correspond exactly).
// The argument list mirrors the matcher's full context (tape, maps, claim
// state) — bundling them into a struct would be a struct used by exactly one
// fn; the lint's advice does not improve this call site.
#[allow(clippy::too_many_arguments)]
fn try_join(
    chain: &mut Chain,
    ops: &[WengertOp],
    idx: usize,
    flowing: Option<VarId>,
    needed: &HashSet<VarId>,
    reads: &HashMap<VarId, usize>,
    producer: &HashMap<VarId, usize>,
    claimed: &[bool],
) -> bool {
    if claimed[idx] {
        return false;
    }
    // Transaction snapshot.
    let snap_externals = chain.externals.len();
    let snap_consts = chain.const_members.len();
    let snap_imms = chain.imms;
    let rollback = |chain: &mut Chain| {
        chain.externals.truncate(snap_externals);
        chain.const_members.truncate(snap_consts);
        chain.imms = snap_imms;
        false
    };

    let op = &ops[idx];
    let flowing_operand = chain
        .steps
        .len()
        .checked_sub(1)
        .map(|k| Operand::Prev(k as u8));
    let const_info = |v: VarId| -> Option<(usize, f64)> {
        let &j = producer.get(&v)?;
        match ops[j].op {
            PrimalOp::Constant(c) => Some((j, c)),
            _ => None,
        }
    };

    // Map one tape operand to a chain Operand. The flowing value maps to
    // Prev(last); a RIGHT-slot Constant with one reader (and not needed)
    // bakes as an Imm (and its tape op is claimed for deletion); anything
    // else registers as an external Input.
    let resolve = |v: VarId, is_rhs: bool, chain: &mut Chain| -> Option<Operand> {
        if flowing == Some(v) {
            return flowing_operand;
        }
        if let Some((ci, c)) = const_info(v) {
            if is_rhs
                && chain.imms < MAX_IMMS
                && reads.get(&v).copied() == Some(1)
                && !needed.contains(&v)
            {
                chain.imms += 1;
                chain.const_members.push(ci);
                return Some(Operand::Imm((c as f32).to_bits()));
            }
            if !is_rhs {
                // Constant LEFT operand: the baseline computes this op (and
                // everything downstream of it) in host f64 — a device-f32
                // kernel is not bit-exact here. Hard barrier.
                return None;
            }
            // Multi-reader / needed constant on the right: treat as an
            // ordinary external (the runtime replay reproduces it), unless
            // slots are exhausted.
        }
        chain.input_slot(v).map(Operand::Input)
    };

    if is_rts(op) {
        // reduce_to_shape is a BARRIER (measured decision, 2026-08-25): rts
        // chain membership was tried — same-shape like-refs would ride free
        // as identities — but on the 500M prod tape EVERY absorbed rts was a
        // REAL reduce (48/micro-batch, the GQA K/V expand backward), so all
        // 48 chains refused the uniform-shape gate and replayed every step:
        // 11,712 fallbacks per 30-update round, pure overhead, zero fused
        // launches. Until a compile-time shape filter exists (the
        // transient-arena size-info follow-up recorded in the campaign
        // notes), an rts ends the chain before it. The RtsCheck descriptor/
        // kernel/replay machinery stays — the wire contract is unchanged and
        // the shape-filtered revival will reuse it.
        return false;
    }

    if !is_fusable_arith(&op.op) || chain.arith_steps >= MAX_ARITH_STEPS {
        return false;
    }
    match op.op {
        PrimalOp::Neg => {
            if op.inputs.len() != 1 {
                return false;
            }
            let Some(lhs) = resolve(op.inputs[0], false, chain) else {
                return rollback(chain);
            };
            chain.steps.push(ChainStep {
                op: EwOpcode::Neg,
                lhs,
                rhs: None,
            });
        }
        _ => {
            if op.inputs.len() != 2 {
                return false;
            }
            // A continuation must consume the flowing value in some slot
            // (the start op's operands are all external/imm).
            if let Some(f) = flowing
                && op.inputs[0] != f && op.inputs[1] != f
            {
                return false;
            }
            let Some(lhs) = resolve(op.inputs[0], false, chain) else {
                return rollback(chain);
            };
            let Some(rhs) = resolve(op.inputs[1], true, chain) else {
                return rollback(chain);
            };
            chain.steps.push(ChainStep {
                op: arith_opcode(&op.op),
                lhs,
                rhs: Some(rhs),
            });
        }
    }
    chain.arith_steps += 1;
    true
}

/// Mirror of the PTX emitter's register accounting; with the v1 caps this is
/// always far below `cost_model::MAX_FUSED_REGISTERS` — kept as a belt.
fn register_belt_ok(chain: &Chain) -> bool {
    let est = chain.externals.len() + chain.steps.len() * 2 + 8;
    est <= crate::cost_model::MAX_FUSED_REGISTERS as usize
}

/// Rewrite standalone `x OP const` binaries (RIGHT-operand constants only)
/// into scalar-immediate Passthrough ops: one launch, no per-step CPU scalar
/// alloc, no synchronous HtoD, no full-size broadcast materialization.
/// Bit-exact: the baseline narrows the f64 constant to f32 once at
/// `nsl_tensor_scalar` creation and runs the same single `.rn` f32 kernel op
/// over identical values; the scalar FFI performs the identical narrowing at
/// launch. Runs AFTER the chain fuser (chains absorb constants first).
/// `NSL_FUSE_SCALAR_IMM=0` (compile-time env) disables.
pub fn rewrite_scalar_immediates(
    ops: &mut Vec<WengertOp>,
    needed: &HashSet<VarId>,
    var_types: &HashMap<VarId, WengertType>,
) -> usize {
    if std::env::var("NSL_FUSE_SCALAR_IMM").ok().as_deref() == Some("0") {
        return 0;
    }
    crate::wengert::assert_ids_in_space(
        ops,
        crate::wengert::IdSpace::Adjoint,
        "ew_chain_fusion::rewrite_scalar_immediates (argument)",
    );
    debug_assert!(
        !ops.iter().any(|op| matches!(op.op, PrimalOp::FreeTensor)),
        "rewrite_scalar_immediates must run before free insertion"
    );

    let mut reads: HashMap<VarId, usize> = HashMap::new();
    let mut producer: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for inp in &op.inputs {
            *reads.entry(*inp).or_default() += 1;
        }
        producer.insert(op.result, i);
    }

    let mut rewrites: Vec<(usize, usize, String, VarId)> = Vec::new();
    for i in 0..ops.len() {
        let name = match ops[i].op {
            PrimalOp::Mul => "mul_scalar_rhs",
            PrimalOp::Add => "add_scalar_rhs",
            PrimalOp::Div => "div_scalar_rhs",
            PrimalOp::Sub => "sub_scalar_rhs",
            _ => continue,
        };
        if ops[i].inputs.len() != 2 {
            continue;
        }
        let (x, c) = (ops[i].inputs[0], ops[i].inputs[1]);
        // Same Tensor-type guard as the chain fuser (review finding N2).
        let tensor_typed = |v: VarId| {
            matches!(
                var_types.get(&v).copied().unwrap_or(WengertType::Tensor),
                WengertType::Tensor
            )
        };
        if !tensor_typed(x) || !tensor_typed(ops[i].result) {
            continue;
        }
        let Some(&j) = producer.get(&c) else { continue };
        let PrimalOp::Constant(v) = ops[j].op else {
            continue;
        };
        // The LEFT operand must not itself be a constant (host-math barrier;
        // also Constant(c) OP Constant(c) is a fold, not a rewrite).
        if producer
            .get(&x)
            .is_some_and(|&k| matches!(ops[k].op, PrimalOp::Constant(_)))
        {
            continue;
        }
        if reads.get(&c).copied() != Some(1) || needed.contains(&c) {
            continue;
        }
        if rewrites.iter().any(|&(_, cj, _, _)| cj == j) {
            continue;
        }
        rewrites.push((i, j, format!("{name}:{}", v.to_bits()), x));
    }
    if rewrites.is_empty() {
        return 0;
    }
    let count = rewrites.len();
    let mut removed: HashSet<usize> = HashSet::new();
    for (i, j, name, x) in rewrites {
        ops[i].op = PrimalOp::Passthrough(name);
        ops[i].inputs = vec![x];
        removed.insert(j);
    }
    let mut idx = 0;
    ops.retain(|_| {
        let keep = !removed.contains(&idx);
        idx += 1;
        keep
    });
    crate::wengert::renumber_adjoint_ops(ops);
    count
}

/// Op-kind histogram of an adjoint list (shared by the pre-/post-fusion
/// `[adjoint-profile]` prints in stmt.rs).
pub fn histogram(ops: &[WengertOp]) -> Vec<(String, usize)> {
    use std::collections::BTreeMap;
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for op in ops {
        let key = match &op.op {
            PrimalOp::Passthrough(n) => format!("Passthrough({n})"),
            other => format!("{other:?}")
                .split(['(', ' ', '{'])
                .next()
                .unwrap_or("?")
                .to_string(),
        };
        *hist.entry(key).or_default() += 1;
    }
    let mut rows: Vec<_> = hist.into_iter().collect();
    rows.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    rows
}

/// Count binary ops whose LEFT operand is a Constant — the D2b population
/// (host-f64 pull-down sites the v1 fuser must skip). Printed under
/// `NSL_PROFILE_ADJOINT` so the campaign record can size the deferred fix.
pub fn const_left_binary_sites(ops: &[WengertOp]) -> usize {
    let mut const_results: HashSet<VarId> = HashSet::new();
    for op in ops {
        if matches!(op.op, PrimalOp::Constant(_)) {
            const_results.insert(op.result);
        }
    }
    ops.iter()
        .filter(|op| {
            matches!(
                op.op,
                PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div
            ) && op.inputs.len() == 2
                && const_results.contains(&op.inputs[0])
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::renumber_adjoint_ops;
    use std::sync::Mutex;

    /// The kill-switch envs are process-global and the fuser reads them on
    /// entry, so every test that calls the fuser (or sets an env) serializes
    /// on this lock — a parallel test seeing a sibling's env window would
    /// spuriously observe zero chains.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn op(result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id: 0,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    fn adjoint(mut ops: Vec<WengertOp>) -> Vec<WengertOp> {
        renumber_adjoint_ops(&mut ops);
        ops
    }

    /// Empty map = every var defaults to Tensor (the lowering's convention).
    fn no_types() -> HashMap<VarId, WengertType> {
        HashMap::new()
    }

    #[test]
    fn chain_sig_name_roundtrip() {
        let sig = ChainSig {
            n_inputs: 4,
            steps: vec![
                ChainStep {
                    op: EwOpcode::Mul,
                    lhs: Operand::Input(0),
                    rhs: Some(Operand::Input(1)),
                },
                ChainStep {
                    op: EwOpcode::RtsCheck,
                    lhs: Operand::Prev(0),
                    rhs: Some(Operand::Input(2)),
                },
                ChainStep {
                    op: EwOpcode::Add,
                    lhs: Operand::Prev(1),
                    rhs: Some(Operand::Input(3)),
                },
                ChainStep {
                    op: EwOpcode::Neg,
                    lhs: Operand::Prev(2),
                    rhs: None,
                },
                ChainStep {
                    op: EwOpcode::Mul,
                    lhs: Operand::Prev(3),
                    rhs: Some(Operand::Imm(0x3f00_0000)),
                },
            ],
        };
        let name = sig.encode_name();
        assert_eq!(
            name,
            "fused_ew:v1:mul(i0,i1);rts(p0,i2);add(p1,i3);neg(p2);mul(p3,x3f000000)"
        );
        let parsed = ChainSig::parse(&name).expect("roundtrip parse");
        assert_eq!(parsed, sig);
        assert!(sig.kernel_name().starts_with("nsl_fused_ew_"));
        assert_eq!(sig.kernel_name().len(), "nsl_fused_ew_".len() + 16);
    }

    #[test]
    fn descriptor_bytes_roundtrip_layout() {
        let sig = ChainSig {
            n_inputs: 2,
            steps: vec![
                ChainStep {
                    op: EwOpcode::Mul,
                    lhs: Operand::Input(0),
                    rhs: Some(Operand::Input(1)),
                },
                ChainStep {
                    op: EwOpcode::Add,
                    lhs: Operand::Prev(0),
                    rhs: Some(Operand::Imm(0x4000_0000)),
                },
            ],
        };
        let d = sig.descriptor_bytes();
        assert_eq!(d.len(), 4 + 2 * 9);
        assert_eq!(&d[..4], &[1, 2, 2, 0]);
        // step 0: mul, Input(0), Input(1)
        assert_eq!(&d[4..13], &[2, 0, 0, 0, 1, 0, 0, 0, 0]);
        // step 1: add, Prev(0), Imm(2.0f32)
        assert_eq!(&d[13..18], &[0, 1, 0, 2, 0]);
        assert_eq!(&d[18..22], &0x4000_0000u32.to_le_bytes());
    }

    #[test]
    fn parse_rejects_malformed() {
        assert!(ChainSig::parse("fused_ew:v1:").is_none());
        assert!(ChainSig::parse("fused_ew:v1:mul(i0)").is_none()); // binary needs rhs
        assert!(ChainSig::parse("fused_ew:v1:neg(i0,i1)").is_none()); // unary no rhs
        assert!(ChainSig::parse("fused_ew:v1:pow(i0,i1)").is_none()); // unknown op
        assert!(ChainSig::parse("fused_ew:v2:mul(i0,i1)").is_none()); // wrong version
        assert!(ChainSig::parse("mul(i0,i1)").is_none());
    }

    #[test]
    fn fuses_simple_mul_add_run() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // v10 = Mul(v0, v1); v11 = Add(v10, v2)  ->  one fused op
        let mut ops = adjoint(vec![
            op(10, PrimalOp::Mul, vec![0, 1]),
            op(11, PrimalOp::Add, vec![10, 2]),
        ]);
        let needed: HashSet<VarId> = [11].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        assert_eq!(stats.chains, 1);
        assert_eq!(stats.device_ops_elided, 1);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].result, 11);
        assert_eq!(ops[0].inputs, vec![0, 1, 2]);
        let PrimalOp::Passthrough(name) = &ops[0].op else {
            panic!("expected fused passthrough");
        };
        assert_eq!(name, "fused_ew:v1:mul(i0,i1);add(p0,i2)");
    }

    #[test]
    fn matcher_stops_at_multi_reader_interior() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // v10 read twice -> not an interior; no chain (single ops don't fuse).
        let mut ops = adjoint(vec![
            op(10, PrimalOp::Mul, vec![0, 1]),
            op(11, PrimalOp::Add, vec![10, 2]),
            op(12, PrimalOp::Add, vec![10, 3]),
        ]);
        let needed: HashSet<VarId> = [11, 12].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        assert_eq!(stats.chains, 0);
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn matcher_stops_at_needed_interior() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // v10 is itself a needed gradient -> must stay materialized.
        let mut ops = adjoint(vec![
            op(10, PrimalOp::Mul, vec![0, 1]),
            op(11, PrimalOp::Add, vec![10, 2]),
        ]);
        let needed: HashSet<VarId> = [10, 11].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        assert_eq!(stats.chains, 0);
    }

    #[test]
    fn needed_allowed_at_tail_only() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Tail result in `needed` is fine — its VarId is preserved.
        let mut ops = adjoint(vec![
            op(10, PrimalOp::Mul, vec![0, 1]),
            op(11, PrimalOp::Neg, vec![10]),
        ]);
        let needed: HashSet<VarId> = [11].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        assert_eq!(stats.chains, 1);
        assert_eq!(ops[0].result, 11);
    }

    #[test]
    fn const_right_bakes_as_imm_and_const_deleted() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut ops = adjoint(vec![
            op(9, PrimalOp::Constant(0.5), vec![]),
            op(10, PrimalOp::Mul, vec![0, 9]),
            op(11, PrimalOp::Add, vec![10, 1]),
        ]);
        let needed: HashSet<VarId> = [11].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        assert_eq!(stats.chains, 1);
        assert_eq!(stats.imms_baked, 1);
        assert_eq!(ops.len(), 1);
        let PrimalOp::Passthrough(name) = &ops[0].op else {
            panic!()
        };
        assert_eq!(
            name,
            &format!("fused_ew:v1:mul(i0,x{:08x});add(p0,i1)", 0.5f32.to_bits())
        );
        assert_eq!(ops[0].inputs, vec![0, 1]);
    }

    #[test]
    fn const_left_is_a_barrier() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Div(const, x) runs on host f64 in the baseline — never a member.
        let mut ops = adjoint(vec![
            op(9, PrimalOp::Constant(1.0), vec![]),
            op(10, PrimalOp::Div, vec![9, 0]),
            op(11, PrimalOp::Mul, vec![10, 1]),
            op(12, PrimalOp::Add, vec![11, 2]),
        ]);
        let needed: HashSet<VarId> = [12].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        // The Div can't join; Mul->Add still fuses.
        assert_eq!(stats.chains, 1);
        assert!(ops.iter().any(|o| matches!(o.op, PrimalOp::Div)));
        assert!(ops.iter().any(|o| matches!(o.op, PrimalOp::Constant(_))));
    }

    #[test]
    fn rts_is_a_barrier_and_never_joins() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Mul -> reduce_to_shape -> Add: on the 500M prod tape every rts
        // reachable from a chain was a REAL reduce (GQA expand backward), so
        // absorbing them produced fallback-only chains (11,712 replays per
        // 30-update round). Measured decision: rts ends the chain, and a
        // 1-arith remnant dissolves — the tape lowers exactly as baseline.
        let mut ops = adjoint(vec![
            op(10, PrimalOp::Mul, vec![0, 1]),
            op(
                11,
                PrimalOp::Passthrough("reduce_to_shape".into()),
                vec![10, 2],
            ),
            op(12, PrimalOp::Add, vec![11, 3]),
        ]);
        let needed: HashSet<VarId> = [12].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        assert_eq!(stats.chains, 0);
        assert_eq!(stats.reduces_absorbed, 0);
        assert_eq!(ops.len(), 3, "the tape must be untouched");
    }

    #[test]
    fn rts_never_starts_a_chain_and_matmul_reduce_untouched() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Transpose -> Matmul -> reduce_to_shape (the wgrad triple) must
        // survive byte-identical: rts can only join by following a fusable
        // member's flow.
        let mut ops = adjoint(vec![
            op(
                10,
                PrimalOp::Transpose { dim0: 0, dim1: 1 },
                vec![0],
            ),
            op(11, PrimalOp::Matmul, vec![10, 1]),
            op(
                12,
                PrimalOp::Passthrough("reduce_to_shape".into()),
                vec![11, 2],
            ),
        ]);
        let needed: HashSet<VarId> = [12].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        assert_eq!(stats.chains, 0);
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn caps_respected() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // 8 chained Adds with fresh externals -> capped at 6 arith steps.
        let mut v = 100;
        let mut ops_v = vec![op(v, PrimalOp::Add, vec![0, 1])];
        for k in 2..9 {
            let next = v + 1;
            ops_v.push(op(next, PrimalOp::Add, vec![v, k]));
            v = next;
        }
        let mut ops = adjoint(ops_v);
        let needed: HashSet<VarId> = [v].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        assert!(stats.chains >= 1);
        for o in &ops {
            if let PrimalOp::Passthrough(name) = &o.op {
                let sig = ChainSig::parse(name).expect("valid sig");
                assert!(sig.steps.len() <= MAX_ARITH_STEPS + MAX_INPUTS);
                assert!(
                    sig.steps
                        .iter()
                        .filter(|s| s.op != EwOpcode::RtsCheck)
                        .count()
                        <= MAX_ARITH_STEPS
                );
                assert!((sig.n_inputs as usize) <= MAX_INPUTS);
            }
        }
    }

    #[test]
    fn deletion_only_no_insertions() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let ops_before = adjoint(vec![
            op(10, PrimalOp::Mul, vec![0, 1]),
            op(11, PrimalOp::Add, vec![10, 2]),
            op(12, PrimalOp::Matmul, vec![3, 4]),
        ]);
        let mut ops = ops_before.clone();
        let needed: HashSet<VarId> = [11, 12].into_iter().collect();
        run_backward_ew_fusion(&mut ops, &needed, &no_types());
        assert!(ops.len() < ops_before.len());
        // Every surviving result VarId existed before.
        let before: HashSet<VarId> = ops_before.iter().map(|o| o.result).collect();
        assert!(ops.iter().all(|o| before.contains(&o.result)));
    }

    #[test]
    fn kill_switch_disables() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("NSL_FUSE_ELEMENTWISE_BWD", "0") };
        let mut ops = adjoint(vec![
            op(10, PrimalOp::Mul, vec![0, 1]),
            op(11, PrimalOp::Add, vec![10, 2]),
        ]);
        let needed: HashSet<VarId> = [11].into_iter().collect();
        let stats = run_backward_ew_fusion(&mut ops, &needed, &no_types());
        unsafe { std::env::remove_var("NSL_FUSE_ELEMENTWISE_BWD") };
        assert_eq!(stats.chains, 0);
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn scalar_immediate_sweep_rewrites_const_right() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut ops = adjoint(vec![
            op(9, PrimalOp::Constant(2.0), vec![]),
            op(10, PrimalOp::Mul, vec![0, 9]),
        ]);
        let needed: HashSet<VarId> = [10].into_iter().collect();
        let n = rewrite_scalar_immediates(&mut ops, &needed, &no_types());
        assert_eq!(n, 1);
        assert_eq!(ops.len(), 1);
        let PrimalOp::Passthrough(name) = &ops[0].op else {
            panic!()
        };
        assert_eq!(name, &format!("mul_scalar_rhs:{}", 2.0f64.to_bits()));
        assert_eq!(ops[0].inputs, vec![0]);
        assert_eq!(ops[0].result, 10);
    }

    #[test]
    fn scalar_sweep_skips_const_left_and_shared_consts() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut ops = adjoint(vec![
            op(9, PrimalOp::Constant(2.0), vec![]),
            op(10, PrimalOp::Div, vec![9, 0]), // const LEFT: skip
            op(8, PrimalOp::Constant(3.0), vec![]),
            op(11, PrimalOp::Mul, vec![1, 8]),
            op(12, PrimalOp::Mul, vec![2, 8]), // 8 read twice: skip both
        ]);
        let needed: HashSet<VarId> = [10, 11, 12].into_iter().collect();
        let n = rewrite_scalar_immediates(&mut ops, &needed, &no_types());
        assert_eq!(n, 0);
        assert_eq!(ops.len(), 5);
    }

    #[test]
    fn const_left_counter_counts() {
        let ops = adjoint(vec![
            op(9, PrimalOp::Constant(1.0), vec![]),
            op(10, PrimalOp::Div, vec![9, 0]),
            op(11, PrimalOp::Mul, vec![0, 1]),
        ]);
        assert_eq!(const_left_binary_sites(&ops), 1);
    }
}
