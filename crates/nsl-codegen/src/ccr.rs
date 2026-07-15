//! CCR — Compiler-Chosen Recomputation (`docs/research/CCR.pdf`).
//!
//! P1.a of the pretraining memory-reduction plan: block-granular activation
//! checkpointing on the decorator-free source-AD path. The source-AD lowerer
//! retains every primal intermediate until end-of-backward (there is no
//! last-use freeing — see `free_wengert_owned_values` in stmt.rs), so the
//! activation wall is O(layers x per-layer intermediates). This pass converts
//! it to O(layers x boundary tensors + ONE block's intermediates):
//!
//!   1. **Segment** the flat inlined primal tape into per-transformer-block
//!      ranges using the same `blocks.N` param-name prefixes WGGO uses
//!      (`wggo_graph::layer_prefix`). The extractor inlines block methods
//!      strictly sequentially, so each block's ops form a contiguous range
//!      anchored at the first op consuming one of its params.
//!   2. **Classify** each in-segment result: `escaping` (consumed by a later
//!      primal op — the residual stream and anything else that crosses the
//!      block boundary) stays SAVED; `interior` (consumed only inside the
//!      segment and/or by the adjoint) becomes RECOMPUTE.
//!   3. **Early-free** the original interiors right after the forward (a
//!      tiny `FreeTensor`-only list lowered after the primal — see stmt.rs).
//!   4. **Splice** clones of the interior-producing ops (fresh VarIds,
//!      inputs remapped) into the adjoint immediately before the first
//!      adjoint op that consumes them, and remap those consumers.
//!   5. **Free** the recomputed tensors right after the last adjoint op
//!      that consumes them (`FreeTensor` markers), so at any moment during
//!      the backward only ~one block's recomputed interiors are live.
//!
//! Recompute clones run the same kernels in the same order on the same
//! inputs, so the transform is bit-exact (CCR section 5.4 test 1). Ops with
//! non-replayable semantics (Dropout — RNG) are force-saved; segments owned
//! by a CSHA backward claim are exempted (the claim table is keyed by
//! primal `OpId`, which a clone cannot satisfy).

use std::collections::{HashMap, HashSet};

use crate::wengert::{PrimalOp, VarId, WengertList, WengertOp, WengertType, type_for_op};
use crate::CodegenError;

/// Recompute policy (P1.a / P1.b of the memory-reduction plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcrPolicy {
    /// Recompute every (owned, replayable) block interior — maximum memory
    /// reduction, pays ~one extra forward of block-interior compute.
    Block,
    /// Megatron-style selective: NEVER recompute matmul-class ops (their
    /// outputs stay saved); replay only cheap bandwidth-bound work — norms,
    /// RoPE, elementwise activations, softmax, reshapes. Saves less memory
    /// than `Block` at near-zero recompute cost.
    Selective,
}

/// Matmul-class ops whose outputs are SAVED under `CcrPolicy::Selective`.
fn selective_saves(op: &PrimalOp) -> bool {
    matches!(
        op,
        PrimalOp::Matmul
            | PrimalOp::Conv2d { .. }
            | PrimalOp::ScaledDotProductAttention { .. }
            | PrimalOp::ScaledDotProductAttentionPacked
            | PrimalOp::FusedLinearCe { .. }
            | PrimalOp::FusedKlCe { .. }
            | PrimalOp::FusedGatedLoraMatmul { .. }
            | PrimalOp::FusedLoraMatmul { .. }
            | PrimalOp::FusedIa3Matmul { .. }
    )
}

/// One transformer block's contiguous slice of the primal tape.
#[derive(Debug)]
pub struct BlockSegment {
    /// The `blocks.N` layer key (shared vocabulary with WGGO).
    pub layer_key: String,
    /// Half-open op-index range `[start, end)` into the primal list.
    pub start: usize,
    pub end: usize,
    /// Results produced in-range and consumed by a later primal op —
    /// always saved (residual stream and friends).
    pub escaping: Vec<VarId>,
    /// Results produced in-range with no later primal consumer — the
    /// recompute candidates (minus the force-saved set).
    pub interior: Vec<VarId>,
}

/// The whole-tape checkpointing decision.
#[derive(Debug)]
pub struct CcrPlan {
    pub segments: Vec<BlockSegment>,
    /// Final RECOMPUTE set (union over segments of interior minus
    /// force-saved). Everything else keeps today's lifetime.
    pub recompute: HashSet<VarId>,
    /// Per-segment recompute victims in primal order (used both for the
    /// post-forward early-free list and the adjoint splice).
    pub per_segment_recompute: Vec<Vec<VarId>>,
    /// The subset of `recompute` that may be FREED (owned Tensor values).
    /// List-typed vars are cloned for structural consistency — an NslList
    /// holds raw element pointers with no refcount, so consumers of a list
    /// over freed elements must get a freshly-built list — but the stale
    /// original and the clone are both left to the end-of-backward bulk
    /// free (`nsl_tensor_free` is the wrong free function for a list, and
    /// lists are pointer-array sized — nothing worth reclaiming early).
    /// Populated by `restrict_to_owned`.
    pub free_eligible: HashSet<VarId>,
}

impl CcrPlan {
    /// Freeable recompute victims in primal order — the post-forward
    /// early-free list. Tensor-typed only (see `free_eligible`).
    pub fn early_free_vars(&self) -> Vec<VarId> {
        let mut v: Vec<VarId> = self
            .per_segment_recompute
            .iter()
            .flatten()
            .copied()
            .filter(|v| self.free_eligible.contains(v))
            .collect();
        v.sort_unstable();
        v
    }

    /// Restrict the recompute set to what the PRIMAL LOWERING itself
    /// classified as owned values, and record which of those are Tensors
    /// (free-eligible) vs Lists (clone-only).
    ///
    /// The tape-level type default ("Tensor") over-claims for scalar
    /// arithmetic (which lowers to raw f64 SSA values): cloning those and
    /// remapping their consumers feeds f64s into pointer-typed call sites
    /// (Cranelift verifier error), and "freeing" them is meaningless —
    /// they hold no tensor storage. Dropping them here means the adjoint
    /// keeps consuming the ORIGINAL scalar SSA values, which are always
    /// available (never freed) in the shared var_map. Called by stmt.rs
    /// between the primal lowering and the early-free list build.
    ///
    /// Returns false when nothing recomputable remains.
    pub fn restrict_to_owned(&mut self, owned: &HashMap<VarId, WengertType>) -> bool {
        let keep = |v: &VarId| {
            matches!(
                owned.get(v),
                Some(WengertType::Tensor) | Some(WengertType::List)
            )
        };
        for seg in self.per_segment_recompute.iter_mut() {
            seg.retain(keep);
        }
        self.recompute.retain(keep);
        self.free_eligible = self
            .recompute
            .iter()
            .copied()
            .filter(|v| matches!(owned.get(v), Some(WengertType::Tensor)))
            .collect();
        !self.free_eligible.is_empty()
    }
}

/// P1.c — budget-driven per-tensor SAVE/RECOMPUTE arbitration (the CCR
/// paper's MILP, section 3.3).
///
/// Structural simplification that makes the optimum EXACT: this transform
/// recomputes each victim exactly once per backward, from always-available
/// inputs (saved boundaries / params / earlier clones), so per-tensor
/// costs are independent — no recompute-chain multiplicities, no DAG
/// cycles (the paper's F-15 is impossible by construction). The MILP
/// therefore reduces to a 0/1 knapsack: choose the SAVE subset that
/// maximizes avoided recompute cost subject to saved-interior bytes <=
/// budget, solved optimally by DP over a quantized budget axis.
pub struct CcrBudget {
    /// Bytes per VarId where the type map had a concrete shape. Vars
    /// WITHOUT a size hint cannot be budgeted and stay RECOMPUTE.
    pub sizes: HashMap<VarId, u64>,
    /// Allowed SAVED-interior bytes across all segments (after any C-01
    /// credit the caller applied).
    pub budget_bytes: u64,
}

/// Relative cost of recomputing one output byte of this op — a bandwidth /
/// arithmetic-intensity proxy, not a calibrated model. Matmul-class ops
/// dominate; pure data movement is nearly free.
fn recompute_cost_weight(op: &PrimalOp) -> u64 {
    if selective_saves(op) {
        25 // matmul / attention class: arithmetic-intensity-heavy
    } else {
        match op {
            PrimalOp::Transpose { .. } | PrimalOp::Passthrough(_) => 1,
            _ => 4, // norms, RoPE, elementwise, softmax: bandwidth-bound
        }
    }
}

/// Knapsack granularity: 256 KiB per DP cell keeps the state table small
/// (a 16 GiB budget = 65 536 cells) while activation tensors of interest
/// are MiB-scale.
const BUDGET_QUANTUM: u64 = 256 * 1024;

/// Refine a `Block`-policy plan under a byte budget: pick the SAVE subset
/// (removed from `recompute`) that maximizes avoided recompute cost while
/// keeping saved-interior bytes within `budget.budget_bytes`. Returns the
/// number of tensors flipped to SAVE.
pub fn apply_budget(plan: &mut CcrPlan, primal: &WengertList, budget: &CcrBudget) -> usize {
    // Candidate items: recompute victims with a known size.
    let mut items: Vec<(VarId, u64, u64)> = Vec::new(); // (var, bytes, value)
    for victims in &plan.per_segment_recompute {
        for &v in victims {
            let Some(&bytes) = budget.sizes.get(&v) else {
                continue;
            };
            if bytes == 0 {
                continue;
            }
            let weight = primal
                .find_producer(v)
                .map(|op| recompute_cost_weight(&op.op))
                .unwrap_or(1);
            items.push((v, bytes, bytes.saturating_mul(weight)));
        }
    }
    if items.is_empty() || budget.budget_bytes < BUDGET_QUANTUM {
        return 0;
    }

    let cap = (budget.budget_bytes / BUDGET_QUANTUM) as usize;
    // dp[w] = best total value using <= w quanta; choice tracking per item.
    let mut dp = vec![0u64; cap + 1];
    let mut take = vec![vec![false; cap + 1]; items.len()];
    for (i, (_, bytes, value)) in items.iter().enumerate() {
        let w = (bytes.div_ceil(BUDGET_QUANTUM)) as usize;
        if w > cap {
            continue;
        }
        for c in (w..=cap).rev() {
            let cand = dp[c - w] + value;
            if cand > dp[c] {
                dp[c] = cand;
                take[i][c] = true;
            }
        }
    }
    // Walk back the chosen set.
    let mut saved: HashSet<VarId> = HashSet::new();
    let mut c = cap;
    for i in (0..items.len()).rev() {
        if take[i][c] {
            let (v, bytes, _) = items[i];
            saved.insert(v);
            c -= (bytes.div_ceil(BUDGET_QUANTUM)) as usize;
        }
    }
    if saved.is_empty() {
        return 0;
    }
    for victims in plan.per_segment_recompute.iter_mut() {
        victims.retain(|v| !saved.contains(v));
    }
    plan.recompute.retain(|v| !saved.contains(v));
    plan.free_eligible.retain(|v| !saved.contains(v));
    saved.len()
}

/// P1 completion — adjoint-region last-use freeing.
///
/// The 500M/seq-1024 OOM decomposition (per-surface accounting) showed the
/// binding wall is ADJOINT intermediates: every dx-chain temporary lives
/// until the end-of-backward bulk free (hundreds of `nsl_add_f32` /
/// `sum_dim_f32` blocks live simultaneously). This pass inserts a
/// `FreeTensor` marker immediately after each adjoint-produced var's last
/// use inside the adjoint list.
///
/// Safety rails:
/// - `protect` (param-gradient adjoints — consumed by the FASE hook or by
///   post-lowering grad collection — plus anything the caller retains) is
///   never freed;
/// - vars already freed by an existing `FreeTensor` (the recompute-clone
///   frees) are skipped;
/// - NslList hazard: a var consumed by a List-building op is kept alive
///   until the LIST's own last use (lists hold raw, un-refcounted element
///   pointers — the tape's `cat` consumes the list var, not the elements),
///   propagated transitively for nested lists;
/// - vars with no use after their producer (dead results) are freed
///   immediately after production;
/// - the lowering-side FreeTensor guard (I64 + Tensor-typed only) remains
///   the final authority on what actually emits a free.
///
/// Returns the number of markers inserted.
pub fn insert_adjoint_last_use_frees(
    adjoint: &mut WengertList,
    protect: &HashSet<VarId>,
    fresh: &mut VarId,
) -> usize {
    // Vars produced in the adjoint region (only these are candidates —
    // primal values are owned by the primal cleanup path).
    let produced: HashMap<VarId, usize> = adjoint
        .ops
        .iter()
        .enumerate()
        .map(|(i, op)| (op.result, i))
        .collect();
    let already_freed: HashSet<VarId> = adjoint
        .ops
        .iter()
        .filter(|op| matches!(op.op, PrimalOp::FreeTensor))
        .map(|op| op.inputs[0])
        .collect();

    // Last use per var (tape-level).
    let mut last_use: HashMap<VarId, usize> = HashMap::new();
    for (idx, op) in adjoint.ops.iter().enumerate() {
        for input in &op.inputs {
            last_use.insert(*input, idx);
        }
    }
    // List-chain propagation: an element of a list lives as long as the
    // list itself (and lists of lists, to fixpoint — bounded by nesting).
    loop {
        let mut changed = false;
        for op in adjoint.ops.iter() {
            let is_list = matches!(
                adjoint.var_types.get(&op.result),
                Some(WengertType::List)
            ) || matches!(&op.op, PrimalOp::Passthrough(n) if n == "list");
            if !is_list {
                continue;
            }
            let Some(&list_last) = last_use.get(&op.result) else {
                continue;
            };
            for input in &op.inputs {
                let e = last_use.entry(*input).or_insert(list_last);
                if *e < list_last {
                    *e = list_last;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Collect (insert_after_idx, victim) — free at last use, or right
    // after production for dead results.
    let mut pending: Vec<(usize, VarId)> = Vec::new();
    for (var, &prod_idx) in &produced {
        if protect.contains(var) || already_freed.contains(var) {
            continue;
        }
        if matches!(adjoint.ops[prod_idx].op, PrimalOp::FreeTensor) {
            continue;
        }
        // Only Tensor-typed (or untyped-defaulting-to-tensor) results; the
        // lowering guard re-checks against the actual SSA value type.
        if let Some(ty) = adjoint.var_types.get(var) {
            if !matches!(ty, WengertType::Tensor) {
                continue;
            }
        }
        let at = last_use.get(var).copied().unwrap_or(prod_idx).max(prod_idx);
        pending.push((at, *var));
    }
    // Insert from the back so earlier indices stay valid.
    pending.sort_by_key(|p| std::cmp::Reverse(p.0));
    let inserted = pending.len();
    for (at, victim) in pending {
        let result = *fresh;
        *fresh += 1;
        adjoint.ops.insert(
            at + 1,
            WengertOp {
                id: 0,
                result,
                op: PrimalOp::FreeTensor,
                inputs: vec![victim],
                saved_for_backward: false,
                checkpointed: false,
            },
        );
    }
    for (i, op) in adjoint.ops.iter_mut().enumerate() {
        op.id = i as u32;
    }
    inserted
}

/// Effective Wengert type of a result (falls back to the op default).
fn effective_type(list: &WengertList, op: &WengertOp) -> WengertType {
    list.var_types
        .get(&op.result)
        .copied()
        .unwrap_or_else(|| type_for_op(&op.op))
}

/// Parse the block ordinal out of a `blocks.N`-style layer key.
fn block_ordinal(key: &str) -> Option<u64> {
    key.rsplit('.').next()?.parse().ok()
}

/// Segment the primal tape and decide the recompute set.
///
/// Returns `None` (with a loud stderr note) when the tape has no
/// recognizable repeated-block structure or the anchors are not strictly
/// increasing — the caller then runs exactly as before. `csha_claimed_ops`
/// carries the primal `OpId`s owned by a CSHA backward claim; any segment
/// containing one is exempted (saved, not recomputed) because the claim
/// table cannot recognize a recompute clone's fresh `OpId`.
pub fn plan(
    primal: &WengertList,
    csha_claimed_ops: Option<&HashSet<u32>>,
    policy: CcrPolicy,
) -> Option<CcrPlan> {
    // ---- 1. Param vars per block key --------------------------------
    let mut param_block: HashMap<VarId, String> = HashMap::new();
    let mut block_keys: Vec<String> = Vec::new();
    for op in &primal.ops {
        if let PrimalOp::Param(name) = &op.op {
            if let Some(key) = crate::wggo_graph::layer_prefix(name) {
                if block_ordinal(&key).is_some() {
                    if !block_keys.contains(&key) {
                        block_keys.push(key.clone());
                    }
                    param_block.insert(op.result, key);
                }
            }
        }
    }
    if block_keys.is_empty() {
        eprintln!(
            "[ccr] --checkpoint-blocks requested but the tape has no \
             `blocks.N`-style parameters; running without checkpointing"
        );
        return None;
    }
    block_keys.sort_by_key(|k| block_ordinal(k).unwrap_or(u64::MAX));

    // ---- 2. Anchors: first op consuming a block-k param -------------
    let mut anchor: HashMap<&str, usize> = HashMap::new();
    for (idx, op) in primal.ops.iter().enumerate() {
        for input in &op.inputs {
            if let Some(key) = param_block.get(input) {
                anchor.entry(key.as_str()).or_insert(idx);
            }
        }
    }
    let mut anchors: Vec<(String, usize)> = Vec::new();
    for key in &block_keys {
        match anchor.get(key.as_str()) {
            Some(&idx) => anchors.push((key.clone(), idx)),
            None => {
                // Params registered but never consumed (fully pruned
                // block) — skip the key rather than refusing outright.
                continue;
            }
        }
    }
    if anchors.is_empty() {
        eprintln!("[ccr] no block params are consumed; running without checkpointing");
        return None;
    }
    for w in anchors.windows(2) {
        if w[1].1 <= w[0].1 {
            eprintln!(
                "[ccr] block anchors are not strictly increasing ({} at op {} \
                 vs {} at op {}) — the tape is not sequentially inlined as \
                 expected; refusing to checkpoint (running unchanged)",
                w[0].0, w[0].1, w[1].0, w[1].1
            );
            return None;
        }
    }

    // ---- 3. End of the last block: first later op consuming a
    //         non-block param (final norm / LM head), else end-of-tape.
    let last_anchor = anchors.last().unwrap().1;
    let mut epilogue_start = primal.ops.len();
    'outer: for (idx, op) in primal.ops.iter().enumerate().skip(last_anchor + 1) {
        for input in &op.inputs {
            if let Some(producer) = primal.find_producer(*input) {
                if let PrimalOp::Param(name) = &producer.op {
                    if crate::wggo_graph::layer_prefix(name).is_none() {
                        epilogue_start = idx;
                        break 'outer;
                    }
                }
            }
        }
    }

    // ---- 4. Build segments + escape analysis ------------------------
    // consumers[v] = primal op indices that read v.
    let mut last_primal_use: HashMap<VarId, usize> = HashMap::new();
    for (idx, op) in primal.ops.iter().enumerate() {
        for input in &op.inputs {
            last_primal_use.insert(*input, idx);
        }
    }

    let claimed = csha_claimed_ops.cloned().unwrap_or_default();
    let mut segments = Vec::new();
    let mut per_segment_recompute = Vec::new();
    let mut recompute = HashSet::new();

    for (i, (key, start)) in anchors.iter().enumerate() {
        let end = if i + 1 < anchors.len() {
            anchors[i + 1].1
        } else {
            epilogue_start
        };
        let mut escaping = Vec::new();
        let mut interior = Vec::new();
        let mut seg_recompute = Vec::new();

        // CSHA-claimed attention chains: the claim table is keyed by primal
        // OpId, which a recompute clone cannot satisfy, and the fused
        // backward reads pointer side-channel saves registered by the
        // claimed forward. Exemption is OP-level, not segment-level: the
        // claimed chain (rmsnorm -> projections -> RoPE -> attention) plus
        // its direct in-segment inputs stay SAVED, while the block's FFN /
        // second-norm interiors — the largest per-block tensors, e.g.
        // [b, s, 4d] — remain recomputable. Anything this under-covers is
        // caught loudly by `validate()` (a claim-machinery adjoint op
        // referencing a recomputed var fails the compile).
        let mut claim_saved: HashSet<VarId> = HashSet::new();
        for idx in *start..end {
            let op = &primal.ops[idx];
            if claimed.contains(&op.id) {
                claim_saved.insert(op.result);
                for input in &op.inputs {
                    claim_saved.insert(*input);
                }
            }
        }
        if !claim_saved.is_empty() {
            eprintln!(
                "[ccr] segment {key}: CSHA-claimed attention chain — force-saving \
                 {} chain tensors, recompute continues for the rest of the block",
                claim_saved.len()
            );
        }

        for idx in *start..end {
            let op = &primal.ops[idx];
            match &op.op {
                // Leaves are never "produced" work.
                PrimalOp::Input(_) | PrimalOp::Param(_) | PrimalOp::Constant(_) => continue,
                _ => {}
            }
            let escapes = op.result == primal.output
                || last_primal_use
                    .get(&op.result)
                    .is_some_and(|&last| last >= end);
            if escapes {
                escaping.push(op.result);
                continue;
            }
            interior.push(op.result);

            if claim_saved.contains(&op.result) {
                continue;
            }
            // Force-save: non-tensor/list results (scalar SSA values are
            // neither freeable nor worth cloning), Dropout (RNG must not
            // be replayed), structural markers. Lists ARE recompute
            // candidates: an NslList holds raw element pointers, so a
            // consumer of a list over freed elements needs a fresh list
            // built from the clones.
            let ty = effective_type(primal, op);
            if !matches!(ty, WengertType::Tensor | WengertType::List) {
                continue;
            }
            match &op.op {
                PrimalOp::Dropout { .. } | PrimalOp::PrologueRecompute { .. } => continue,
                PrimalOp::FreeTensor => continue,
                _ => {}
            }
            if policy == CcrPolicy::Selective && selective_saves(&op.op) {
                continue;
            }
            seg_recompute.push(op.result);
            recompute.insert(op.result);
        }

        segments.push(BlockSegment {
            layer_key: key.clone(),
            start: *start,
            end,
            escaping,
            interior,
        });
        per_segment_recompute.push(seg_recompute);
    }

    if recompute.is_empty() {
        eprintln!("[ccr] nothing recomputable found; running without checkpointing");
        return None;
    }
    let free_eligible = recompute
        .iter()
        .copied()
        .filter(|v| {
            primal
                .find_producer(*v)
                .map(|op| matches!(effective_type(primal, op), WengertType::Tensor))
                .unwrap_or(false)
        })
        .collect();
    Some(CcrPlan {
        segments,
        recompute,
        per_segment_recompute,
        free_eligible,
    })
}

/// Build the tiny post-forward early-free list: one `FreeTensor` per
/// recompute victim. Lowered by stmt.rs immediately after the primal with
/// the primal's `var_map` as seed; `explicit_freed_vars` on its lowering
/// result feeds the bulk-free exclusion set.
///
/// The markers' dummy result ids descend from `u32::MAX - 1`: they exist
/// only inside the mini-list's local lowering `var_map` (discarded after)
/// and must not collide with real tape ids, which grow upward from 0.
/// This list is built BEFORE the adjoint exists, so a max-over-both-lists
/// fresh counter is not yet available.
pub fn build_early_free_list(plan: &CcrPlan) -> WengertList {
    let victims = plan.early_free_vars();
    assert!(
        victims.len() < (1 << 16),
        "[ccr] early-free list unreasonably large ({}); dummy-id range \
         would approach real VarIds",
        victims.len()
    );
    let mut ops = Vec::new();
    for (i, victim) in victims.into_iter().enumerate() {
        let result = u32::MAX - 1 - i as u32;
        ops.push(WengertOp {
            id: i as u32,
            result,
            op: PrimalOp::FreeTensor,
            inputs: vec![victim],
            saved_for_backward: false,
            checkpointed: false,
        });
    }
    WengertList {
        ops,
        output: 0,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    }
}

/// Splice recompute clones + `FreeTensor` markers into the adjoint and
/// remap its references from original interior VarIds to the fresh clones.
///
/// `fresh` must start above every VarId in `primal` and `adjoint`; it is
/// advanced as clones are allocated.
pub fn apply_to_adjoint(
    primal: &WengertList,
    adjoint: &mut WengertList,
    plan: &CcrPlan,
    fresh: &mut VarId,
) -> Result<(), CodegenError> {
    // The adjoint lowering is seeded with the primal's VALUES (full_vars)
    // but its type map is the adjoint list's own — recompute clones that
    // reference primal SCALAR vars (e.g. the SDPA scale chain) would be
    // mistyped as tensors and lower raw f64 SSA values into pointer call
    // sites. Complete the seeding symmetrically: primal types flow in,
    // existing adjoint entries win.
    for (k, v) in &primal.var_types {
        adjoint.var_types.entry(*k).or_insert(*v);
    }

    // Global remap original-interior -> clone.
    let mut remap: HashMap<VarId, VarId> = HashMap::new();
    for victims in &plan.per_segment_recompute {
        for v in victims {
            remap.insert(*v, *fresh);
            *fresh += 1;
        }
    }

    // Per segment: the first adjoint index referencing any of its victims.
    // (The adjoint walks the primal in reverse, so later blocks splice
    // earlier; we insert in descending splice order so earlier insertions
    // never shift a pending position.)
    let mut splices: Vec<(usize, usize)> = Vec::new(); // (segment_idx, adjoint_idx)
    for (si, victims) in plan.per_segment_recompute.iter().enumerate() {
        if victims.is_empty() {
            continue;
        }
        let vset: HashSet<VarId> = victims.iter().copied().collect();
        let first = adjoint
            .ops
            .iter()
            .position(|op| op.inputs.iter().any(|i| vset.contains(i)));
        if let Some(idx) = first {
            splices.push((si, idx));
        }
        // No adjoint consumer: the victims are freed post-forward and the
        // backward never needs them — nothing to splice.
    }
    splices.sort_by_key(|s| std::cmp::Reverse(s.1));

    for (si, at) in splices {
        let seg = &plan.segments[si];
        let vset: HashSet<VarId> = plan.per_segment_recompute[si].iter().copied().collect();
        // Clone the producing ops in primal order, remapping inputs.
        let mut clones = Vec::new();
        for idx in seg.start..seg.end {
            let op = &primal.ops[idx];
            if !vset.contains(&op.result) {
                continue;
            }
            let inputs = op
                .inputs
                .iter()
                .map(|v| remap.get(v).copied().unwrap_or(*v))
                .collect();
            clones.push(WengertOp {
                id: 0, // renumbered below
                result: remap[&op.result],
                op: op.op.clone(),
                inputs,
                saved_for_backward: false,
                checkpointed: false,
            });
            // Clone types/names so downstream type inference matches.
            if let Some(ty) = primal.var_types.get(&op.result) {
                adjoint.var_types.insert(remap[&op.result], *ty);
            }
            if let Some(name) = primal.var_names.get(&op.result) {
                adjoint
                    .var_names
                    .insert(remap[&op.result], format!("{name}.ccr_recompute"));
            }
        }
        adjoint.ops.splice(at..at, clones);
    }

    // Remap every adjoint reference to a recomputed original.
    for op in adjoint.ops.iter_mut() {
        for input in op.inputs.iter_mut() {
            if let Some(fresh_id) = remap.get(input) {
                *input = *fresh_id;
            }
        }
    }

    // Per segment: free the clones right after their last adjoint use.
    // Process in descending last-use order so insertions don't shift
    // pending positions.
    let mut frees: Vec<(usize, Vec<VarId>)> = Vec::new(); // (last_idx, victims)
    for victims in &plan.per_segment_recompute {
        if victims.is_empty() {
            continue;
        }
        // Free only Tensor-typed clones; List clones stay for the bulk
        // cleanup (wrong free function + negligible bytes).
        let fresh_set: HashSet<VarId> = victims
            .iter()
            .filter(|v| plan.free_eligible.contains(v))
            .map(|v| remap[v])
            .collect();
        if fresh_set.is_empty() {
            continue;
        }
        let last = adjoint
            .ops
            .iter()
            .rposition(|op| op.inputs.iter().any(|i| fresh_set.contains(i)));
        if let Some(idx) = last {
            frees.push((idx, fresh_set.into_iter().collect()));
        }
    }
    frees.sort_by_key(|f| std::cmp::Reverse(f.0));
    for (last_idx, mut victims) in frees {
        victims.sort_unstable();
        let mut markers = Vec::new();
        for v in victims {
            let result = *fresh;
            *fresh += 1;
            markers.push(WengertOp {
                id: 0,
                result,
                op: PrimalOp::FreeTensor,
                inputs: vec![v],
                saved_for_backward: false,
                checkpointed: false,
            });
        }
        adjoint.ops.splice(last_idx + 1..last_idx + 1, markers);
    }

    // Renumber adjoint op ids sequentially (ids are positional metadata in
    // the adjoint list; nothing keys off them the way CSHA claims key off
    // primal ids).
    for (i, op) in adjoint.ops.iter_mut().enumerate() {
        op.id = i as u32;
    }

    validate(primal, adjoint, plan)
}

/// Static post-transform validation — loud failure over silent skip.
///
/// (1) every adjoint input is defined (primal result, primal leaf, earlier
///     adjoint result, or clone);
/// (2) no adjoint op after a `FreeTensor` references its victim;
/// (3) no adjoint op references an early-freed ORIGINAL interior (they were
///     freed right after the forward).
fn validate(
    primal: &WengertList,
    adjoint: &WengertList,
    plan: &CcrPlan,
) -> Result<(), CodegenError> {
    let primal_defined: HashSet<VarId> = primal.ops.iter().map(|o| o.result).collect();
    let mut defined: HashSet<VarId> = primal_defined.clone();
    let mut freed: HashMap<VarId, usize> = HashMap::new();

    for (idx, op) in adjoint.ops.iter().enumerate() {
        for input in &op.inputs {
            if let Some(&fidx) = freed.get(input) {
                return Err(CodegenError::new(format!(
                    "[ccr] validation: adjoint op {idx} ({:?} -> v{}) reads \
                     v{input}, freed by the FreeTensor at index {fidx}",
                    op.op, op.result
                )));
            }
            if plan.recompute.contains(input) {
                return Err(CodegenError::new(format!(
                    "[ccr] validation: adjoint op {idx} ({:?} -> v{}) still \
                     references early-freed original v{input} (remap missed it)",
                    op.op, op.result
                )));
            }
            // Ghost adjoint VarIds (never-materialized gradients) are an
            // accepted pre-existing pattern — but a CLONE with an undefined
            // input is a transform bug.
            if !defined.contains(input) && op.result != 0 {
                // Only enforce for ops we spliced (clones + frees have
                // var_names tagged or are FreeTensor).
                let is_ours = matches!(op.op, PrimalOp::FreeTensor)
                    || adjoint
                        .var_names
                        .get(&op.result)
                        .is_some_and(|n| n.ends_with(".ccr_recompute"));
                if is_ours {
                    return Err(CodegenError::new(format!(
                        "[ccr] validation: spliced op {idx} ({:?} -> v{}) has \
                         undefined input v{input}",
                        op.op, op.result
                    )));
                }
            }
        }
        if let PrimalOp::FreeTensor = op.op {
            freed.insert(op.inputs[0], idx);
        }
        defined.insert(op.result);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(id: u32, result: VarId, p: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result,
            op: p,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    /// Two-block toy tape:
    ///   v0 = Input x
    ///   v1 = Param blocks.0.w      v2 = Param blocks.1.w
    ///   v3 = Matmul(v0, v1)        (block 0 interior)
    ///   v4 = Add(v3, v0)           (block 0 output — escapes)
    ///   v5 = Matmul(v4, v2)        (block 1 interior)
    ///   v6 = Add(v5, v4)           (block 1 output)
    ///   v7 = Param head.w
    ///   v8 = Matmul(v6, v7)        (epilogue)
    fn toy_primal() -> WengertList {
        let mut var_names = HashMap::new();
        var_names.insert(1, "m.blocks.0.w".to_string());
        var_names.insert(2, "m.blocks.1.w".to_string());
        var_names.insert(7, "m.head.w".to_string());
        WengertList {
            ops: vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Param("m.blocks.0.w".into()), vec![]),
                op(2, 2, PrimalOp::Param("m.blocks.1.w".into()), vec![]),
                op(3, 3, PrimalOp::Matmul, vec![0, 1]),
                op(4, 4, PrimalOp::Add, vec![3, 0]),
                op(5, 5, PrimalOp::Matmul, vec![4, 2]),
                op(6, 6, PrimalOp::Add, vec![5, 4]),
                op(7, 7, PrimalOp::Param("m.head.w".into()), vec![]),
                op(8, 8, PrimalOp::Matmul, vec![6, 7]),
            ],
            output: 8,
            var_names,
            var_types: HashMap::new(),
        }
    }

    #[test]
    fn segments_and_interiors() {
        let primal = toy_primal();
        let plan = plan(&primal, None, CcrPolicy::Block).expect("two blocks segment");
        assert_eq!(plan.segments.len(), 2);
        assert_eq!(plan.segments[0].layer_key, "blocks.0");
        // v3 is interior to block 0 (only consumed by v4 inside the block);
        // v4 escapes (consumed by v5/v6 in block 1).
        assert!(plan.recompute.contains(&3));
        assert!(!plan.recompute.contains(&4));
        // v5 interior to block 1; v6 escapes into the epilogue.
        assert!(plan.recompute.contains(&5));
        assert!(!plan.recompute.contains(&6));
    }

    #[test]
    fn adjoint_splice_remaps_and_frees() {
        let primal = toy_primal();
        let plan = plan(&primal, None, CcrPolicy::Block).unwrap();
        // Toy adjoint touching both interiors: block-1 grads first (reverse
        // order), then block-0 grads.
        let mut adjoint = WengertList {
            ops: vec![
                // d_v5-ish: consumes v5 and v4
                op(0, 100, PrimalOp::Mul, vec![5, 4]),
                // d_w1: consumes v4 only
                op(1, 101, PrimalOp::Mul, vec![4, 100]),
                // d_v3-ish: consumes v3 and v0
                op(2, 102, PrimalOp::Mul, vec![3, 0]),
            ],
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let mut fresh = 200;
        apply_to_adjoint(&primal, &mut adjoint, &plan, &mut fresh).unwrap();

        // No adjoint op may still reference the originals v3 / v5.
        for o in &adjoint.ops {
            if matches!(o.op, PrimalOp::FreeTensor) {
                continue;
            }
            assert!(!o.inputs.contains(&3), "v3 not remapped: {:?}", o);
            assert!(!o.inputs.contains(&5), "v5 not remapped: {:?}", o);
        }
        // Clones of the two Matmuls exist, and each fresh var is freed
        // after its last use.
        let clone_count = adjoint
            .ops
            .iter()
            .filter(|o| matches!(o.op, PrimalOp::Matmul))
            .count();
        assert_eq!(clone_count, 2);
        let free_count = adjoint
            .ops
            .iter()
            .filter(|o| matches!(o.op, PrimalOp::FreeTensor))
            .count();
        assert_eq!(free_count, 2);
        // Frees come after last uses (validate() already enforced this).
    }

    #[test]
    fn no_blocks_declines() {
        let primal = WengertList {
            ops: vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Param("m.w".into()), vec![]),
                op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            ],
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        assert!(plan(&primal, None, CcrPolicy::Block).is_none());
    }

    #[test]
    fn dropout_is_force_saved() {
        let mut primal = toy_primal();
        // Replace block-0's interior Matmul with a Dropout.
        primal.ops[3] = op(3, 3, PrimalOp::Dropout { p: 0.1 }, vec![0]);
        let plan = plan(&primal, None, CcrPolicy::Block).unwrap();
        assert!(!plan.recompute.contains(&3), "dropout must not be replayed");
    }

    #[test]
    fn selective_saves_matmuls_recomputes_cheap() {
        let mut primal = toy_primal();
        // Insert a Relu over block-0's Matmul so the block has one cheap
        // and one expensive interior: v3 = Matmul, v9 = Relu(v3), and the
        // residual add consumes the Relu instead.
        primal.ops[4] = op(4, 4, PrimalOp::Add, vec![9, 0]);
        primal.ops.insert(4, op(9, 9, PrimalOp::Relu, vec![3]));
        let plan = plan(&primal, None, CcrPolicy::Selective).unwrap();
        assert!(!plan.recompute.contains(&3), "matmul output must stay saved");
        assert!(plan.recompute.contains(&9), "relu is replayable");
    }

    #[test]
    fn csha_claimed_segment_exempted() {
        let primal = toy_primal();
        let mut claimed = HashSet::new();
        claimed.insert(3u32); // op id 3 = block 0's Matmul
        let plan = plan(&primal, Some(&claimed), CcrPolicy::Block).unwrap();
        assert!(!plan.recompute.contains(&3), "claimed segment exempt");
        assert!(plan.recompute.contains(&5), "unclaimed segment intact");
    }
}
