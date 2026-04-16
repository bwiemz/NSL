//! CSHA Level 1 — Boundary Fusion (paper §2.1).
//!
//! Detects the canonical pre-norm attention prologue in a Wengert list:
//!
//! ```text
//!   x    ──► RMSNorm ──► Matmul(Wq) ──► RoPE    (Q path)
//!                   ─► Matmul(Wk) ──► RoPE    (K path)
//!                   ─► Matmul(Wv)             (V path — no RoPE)
//! ```
//!
//! Each detected chain becomes a [`BoundaryChain`] that the compiler
//! rewrites into a single fused kernel (`matmul_with_norm_rope_kernel`).
//! This module is pattern-detection only — the actual PTX rewrite happens
//! downstream in `epilogue_fusion` / `flash_attention`.
//!
//! The detector is deliberately conservative: it only claims a chain when
//! the topology unambiguously matches the paper's Figure in §2.1.  That
//! gives downstream passes a strong guarantee that the fused kernel is
//! semantics-preserving.

use std::collections::HashMap;

use serde::Serialize;

use crate::wengert::{PrimalOp, VarId, WengertList};

/// Which projection this chain belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ProjKind {
    Q,
    K,
    V,
}

impl ProjKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ProjKind::Q => "Wq",
            ProjKind::K => "Wk",
            ProjKind::V => "Wv",
        }
    }

    /// Infer from the parameter name.  Accepts `wq`, `Wq`, `.wq.`, etc.
    pub fn from_param_name(name: &str) -> Option<Self> {
        let lower = name.to_ascii_lowercase();
        // Prefer the last component to avoid accidental prefix matches.
        let tail = lower.rsplit('.').next().unwrap_or(&lower);
        if tail.starts_with("wq") || tail == "q_proj" {
            Some(ProjKind::Q)
        } else if tail.starts_with("wk") || tail == "k_proj" {
            Some(ProjKind::K)
        } else if tail.starts_with("wv") || tail == "v_proj" {
            Some(ProjKind::V)
        } else {
            None
        }
    }
}

/// One detected boundary-fusion chain.
#[derive(Debug, Clone, Serialize)]
pub struct BoundaryChain {
    /// Layer prefix (e.g. `blocks.3`) — derived from the projection param
    /// name.  `None` for floating chains that do not belong to any
    /// recognised layer.
    pub layer: Option<String>,
    /// Which projection this chain realises.
    pub kind: ProjKind,
    /// `RMSNorm` op index in the Wengert list.
    pub norm_op: u32,
    /// `Matmul` op index.
    pub matmul_op: u32,
    /// `RoPE` op index — `None` for V (no RoPE).
    pub rope_op: Option<u32>,
    /// Parameter name of the projection weight (`blocks.3.attn.wq`).
    pub weight_param: String,
    /// Gap D.1: `ScaledDotProductAttention` op index that consumes this
    /// chain's output (matmul_op result for V, rope_op result for Q/K).
    /// `None` when no SDPA is found in the Wengert list — e.g. structural
    /// tests where the chain is cut off at the projection, or models whose
    /// attention is open-coded rather than using the fused SDPA primitive.
    ///
    /// When present, the AD reverse-walk dispatcher claims the SDPA op
    /// (not the projection matmul) as the first `EmitFused` trigger point.
    /// This is the correct site for fused-backward emission because by the
    /// time the reverse walk hits SDPA, the SDPA output's y_bar is already
    /// populated (downstream ops like the output projection ran earlier in
    /// reverse order), so `dO` is available for the fused kernel to consume.
    pub sdpa_op: Option<u32>,
}

impl BoundaryChain {
    /// Number of Wengert ops eliminated by fusion (1 = norm folded into
    /// matmul, 2 = norm + rope both folded).
    pub fn ops_eliminated(&self) -> u32 {
        1 + if self.rope_op.is_some() { 1 } else { 0 }
    }

    /// Estimated HBM-traffic reduction for one fusion in bytes, assuming
    /// `[B, S, D]` input with `dtype_bytes` per element.
    ///
    /// Conservative model: each eliminated op saves two HBM round-trips
    /// (one read + one write) of the intermediate tensor.
    pub fn hbm_bytes_saved(&self, batch: u64, seq: u64, d_model: u64, dtype_bytes: u64) -> u64 {
        let bytes_per_tensor = batch * seq * d_model * dtype_bytes;
        // Norm output avoided: +1 tensor. RoPE output avoided: +1 tensor.
        let avoided = 1 + if self.rope_op.is_some() { 1 } else { 0 };
        avoided * 2 * bytes_per_tensor
    }
}

/// Top-level result of scanning a Wengert list for Level-1 patterns.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BoundaryScan {
    pub chains: Vec<BoundaryChain>,
}

impl BoundaryScan {
    pub fn num_chains(&self) -> usize {
        self.chains.len()
    }

    /// Chains grouped by layer prefix, sorted by prefix then projection.
    pub fn by_layer(&self) -> Vec<(String, Vec<&BoundaryChain>)> {
        let mut buckets: HashMap<String, Vec<&BoundaryChain>> = HashMap::new();
        for c in &self.chains {
            let k = c.layer.clone().unwrap_or_else(|| "other".to_string());
            buckets.entry(k).or_default().push(c);
        }
        let mut out: Vec<_> = buckets.into_iter().collect();
        for (_, v) in out.iter_mut() {
            v.sort_by_key(|c| (c.kind as u8, c.matmul_op));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Count of Q/K/V chains found.
    pub fn count_kind(&self, kind: ProjKind) -> usize {
        self.chains.iter().filter(|c| c.kind == kind).count()
    }
}

// ---------------------------------------------------------------------------
// Pattern detection
// ---------------------------------------------------------------------------

/// Walk a Wengert list and report every `RMSNorm → Matmul(Wq|Wk|Wv) →
/// RoPE?` chain it contains.
///
/// The scan is O(N) in the number of ops: for each `Matmul` we check its
/// inputs and consumers.
pub fn scan(list: &WengertList) -> BoundaryScan {
    // Build `producer[v] = op_index_that_produced_v`.
    let mut producer: HashMap<VarId, u32> = HashMap::with_capacity(list.ops.len());
    for (idx, op) in list.ops.iter().enumerate() {
        producer.insert(op.result, idx as u32);
    }
    // Build `consumers[v] = [op_index, ...]`.
    let mut consumers: HashMap<VarId, Vec<u32>> = HashMap::new();
    for (idx, op) in list.ops.iter().enumerate() {
        for &inp in &op.inputs {
            consumers.entry(inp).or_default().push(idx as u32);
        }
    }

    let mut chains = Vec::new();

    for (mm_idx, mm_op) in list.ops.iter().enumerate() {
        if !matches!(mm_op.op, PrimalOp::Matmul) {
            continue;
        }
        // Matmul must have exactly two inputs: (x_norm, Wproj) or
        // (Wproj, x_norm) depending on how the model wrote the expression.
        if mm_op.inputs.len() != 2 {
            continue;
        }

        // Identify the weight-param input and the norm input.
        let (weight_param, norm_var) = match classify_matmul_inputs(list, mm_op.inputs[0], mm_op.inputs[1]) {
            Some(pair) => pair,
            None => continue,
        };

        let kind = match ProjKind::from_param_name(&weight_param) {
            Some(k) => k,
            None => continue,
        };

        // Verify the "norm" input is actually an RMSNorm op output.
        let norm_op_idx = match producer.get(&norm_var) {
            Some(&i) => i,
            None => continue,
        };
        if !matches!(list.ops[norm_op_idx as usize].op, PrimalOp::RMSNorm { .. }) {
            continue;
        }

        // Check whether the Matmul's output is consumed by a RoPE op.
        let rope_op_idx = consumers
            .get(&mm_op.result)
            .and_then(|cs| {
                cs.iter()
                    .find(|&&c| matches!(list.ops[c as usize].op, PrimalOp::RoPE { .. }))
                    .copied()
            });

        // V must NOT have RoPE; Q/K must have RoPE.  If the topology
        // contradicts this, skip the chain to stay conservative.
        match kind {
            ProjKind::V => {
                if rope_op_idx.is_some() {
                    continue;
                }
            }
            ProjKind::Q | ProjKind::K => {
                if rope_op_idx.is_none() {
                    // Still valid: sometimes RoPE is applied lazily inside
                    // the attention kernel.  Emit the chain — downstream
                    // will skip the RoPE-epilogue emit.
                }
            }
        }

        // Gap D.1: detect the SDPA op that consumes this chain's output.
        // Q/K chains feed SDPA through the RoPE output (when present),
        // else through the matmul output. V always feeds SDPA through the
        // matmul output directly.
        let chain_output_var = match rope_op_idx {
            Some(ri) => list.ops[ri as usize].result,
            None => mm_op.result,
        };
        let sdpa_op_idx = consumers.get(&chain_output_var).and_then(|cs| {
            cs.iter()
                .find(|&&c| {
                    matches!(
                        list.ops[c as usize].op,
                        PrimalOp::ScaledDotProductAttention { .. }
                    )
                })
                .copied()
        });

        chains.push(BoundaryChain {
            // A.2.1c: use the fallback-aware layer-key derivation so
            // single-class models (whose Wengert Param names are
            // `"Model.wq"` without a `blocks.N` prefix) still get a
            // non-None layer key and therefore still produce
            // `FusionMark`s in `csha_apply::bridge`.
            layer: crate::wggo_graph::layer_key_with_fallback(&weight_param),
            kind,
            norm_op: norm_op_idx,
            matmul_op: mm_idx as u32,
            rope_op: rope_op_idx,
            weight_param,
            sdpa_op: sdpa_op_idx,
        });
    }

    // Deterministic output order.
    chains.sort_by_key(|c| (c.layer.clone().unwrap_or_default(), c.kind as u8, c.matmul_op));
    BoundaryScan { chains }
}

/// Classify the two inputs of a Matmul into (weight_param_name,
/// other_input_var_id).  Returns `None` if exactly one side isn't a
/// `Param(...)` op.
fn classify_matmul_inputs(
    list: &WengertList,
    lhs: VarId,
    rhs: VarId,
) -> Option<(String, VarId)> {
    let lhs_param = param_name_of(list, lhs);
    let rhs_param = param_name_of(list, rhs);
    match (lhs_param, rhs_param) {
        (Some(n), None) => Some((n, rhs)),
        (None, Some(n)) => Some((n, lhs)),
        _ => None, // both or neither are params — not our pattern
    }
}

/// If `v` is produced by a `Param(name)` op, return the name.
fn param_name_of(list: &WengertList, v: VarId) -> Option<String> {
    list.ops
        .iter()
        .find(|op| op.result == v)
        .and_then(|op| match &op.op {
            PrimalOp::Param(n) => Some(n.clone()),
            _ => None,
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};
    use std::collections::HashMap;

    fn op(id: u32, result: u32, o: PrimalOp, inputs: Vec<u32>) -> WengertOp {
        WengertOp {
            id,
            result,
            op: o,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    /// Build a tiny two-layer Wengert list that contains a full
    /// norm → Wq+rope, norm → Wk+rope, norm → Wv chain on block 0.
    fn one_block_attention() -> WengertList {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
            op(2, 2, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
            op(3, 3, PrimalOp::Matmul, vec![1, 2]), // x_norm @ Wq
            op(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
            op(5, 5, PrimalOp::Param("blocks.0.attn.wk".into()), vec![]),
            op(6, 6, PrimalOp::Matmul, vec![1, 5]),
            op(7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
            op(8, 8, PrimalOp::Param("blocks.0.attn.wv".into()), vec![]),
            op(9, 9, PrimalOp::Matmul, vec![1, 8]),
        ];
        WengertList {
            ops,
            output: 9,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    #[test]
    fn scan_finds_qkv_chains() {
        let w = one_block_attention();
        let s = scan(&w);
        assert_eq!(s.num_chains(), 3);
        assert_eq!(s.count_kind(ProjKind::Q), 1);
        assert_eq!(s.count_kind(ProjKind::K), 1);
        assert_eq!(s.count_kind(ProjKind::V), 1);
    }

    #[test]
    fn v_chain_has_no_rope() {
        let w = one_block_attention();
        let s = scan(&w);
        let v = s.chains.iter().find(|c| c.kind == ProjKind::V).unwrap();
        assert!(v.rope_op.is_none());
        assert_eq!(v.ops_eliminated(), 1);
    }

    #[test]
    fn qk_chains_have_rope() {
        let w = one_block_attention();
        let s = scan(&w);
        for c in s.chains.iter().filter(|c| c.kind != ProjKind::V) {
            assert!(c.rope_op.is_some(), "{:?} should have a RoPE", c.kind);
            assert_eq!(c.ops_eliminated(), 2);
        }
    }

    #[test]
    fn layer_prefix_is_recovered() {
        let w = one_block_attention();
        let s = scan(&w);
        for c in &s.chains {
            assert_eq!(c.layer.as_deref(), Some("blocks.0"));
        }
    }

    #[test]
    fn hbm_bytes_saved_scales_with_shape() {
        let c = BoundaryChain {
            layer: Some("blocks.0".into()),
            kind: ProjKind::Q,
            norm_op: 0,
            matmul_op: 1,
            rope_op: Some(2),
            weight_param: "blocks.0.attn.wq".into(),
            sdpa_op: None,
        };
        let small = c.hbm_bytes_saved(1, 1024, 512, 2);
        let big = c.hbm_bytes_saved(4, 1024, 512, 2);
        assert_eq!(big, 4 * small);
    }

    #[test]
    fn non_attention_matmuls_are_ignored() {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("blocks.0.ffn.w1".into()), vec![]),
            op(2, 2, PrimalOp::Matmul, vec![0, 1]), // FFN matmul, no norm
        ];
        let w = WengertList {
            ops,
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let s = scan(&w);
        assert_eq!(s.num_chains(), 0);
    }

    #[test]
    fn matmul_without_norm_input_is_skipped() {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
            op(2, 2, PrimalOp::Matmul, vec![0, 1]), // NOT fed by RMSNorm
        ];
        let w = WengertList {
            ops,
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let s = scan(&w);
        assert_eq!(s.num_chains(), 0);
    }

    #[test]
    fn by_layer_groups_chains_together() {
        let mut ops = Vec::new();
        let mut next = 0u32;
        let mut push = |o: PrimalOp, inputs: Vec<u32>, ops: &mut Vec<WengertOp>, next: &mut u32| {
            let id = *next;
            ops.push(WengertOp {
                id,
                result: id,
                op: o,
                inputs,
                saved_for_backward: false,
                checkpointed: false,
            });
            *next += 1;
            id
        };
        let x = push(PrimalOp::Input("x".into()), vec![], &mut ops, &mut next);
        for i in 0..2 {
            let n = push(PrimalOp::RMSNorm { eps: 1e-5 }, vec![x], &mut ops, &mut next);
            let wq = push(
                PrimalOp::Param(format!("blocks.{}.attn.wq", i)),
                vec![],
                &mut ops,
                &mut next,
            );
            let q = push(PrimalOp::Matmul, vec![n, wq], &mut ops, &mut next);
            let _ = push(PrimalOp::RoPE { dim: 64 }, vec![q], &mut ops, &mut next);
        }
        let w = WengertList {
            ops,
            output: next - 1,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let s = scan(&w);
        let by = s.by_layer();
        assert_eq!(by.len(), 2);
        assert_eq!(by[0].0, "blocks.0");
        assert_eq!(by[1].0, "blocks.1");
    }

    #[test]
    fn proj_kind_recognises_common_naming_schemes() {
        assert_eq!(ProjKind::from_param_name("blocks.0.attn.wq"), Some(ProjKind::Q));
        assert_eq!(ProjKind::from_param_name("blocks.0.attn.wk"), Some(ProjKind::K));
        assert_eq!(ProjKind::from_param_name("blocks.0.attn.wv"), Some(ProjKind::V));
        assert_eq!(ProjKind::from_param_name("layers.3.q_proj"), Some(ProjKind::Q));
        assert_eq!(ProjKind::from_param_name("h.1.attn.wo"), None);
        assert_eq!(ProjKind::from_param_name("blocks.0.ffn.w1"), None);
    }
}
