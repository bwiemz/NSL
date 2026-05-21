//! Diagnostic-mode utility for backward-kernel localizability.
//!
//! Swap CPU-computed components in for hardware-computed ones to bisect
//! failures. When a parity test fails, the developer's first question is
//! "is the bug in the GPU kernel or in the upstream component?" Diagnostic
//! mode answers that by letting the test source any intermediate value from
//! either a GPU kernel or a CPU reference.
//!
//! Phase 2 instantiates for D-pre-pass + dQ-kernel.
//! Phase 3 extends for dK/dV-kernel.
//! Future-milestone backward work inherits the utility without redesigning
//! the localizability primitive.
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §6.3 + §7.3

/// Source for the D tensor in a diagnostic-mode parity test.
///
/// `B2PrePass` is the default for full integration tests.
/// `CpuNaive` lets the developer swap CPU-computed D in to bisect "is the bug
/// in D pre-pass or in dQ-kernel?"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DSource {
    B2PrePass,
    CpuNaive,
}

/// Compute D for use as a dQ-kernel input.
///
/// Shape: `[B, H, S]` f32, row-major. `shape` = `(batch, heads, seq, hd)`.
///
/// Source dispatch:
/// - `DSource::CpuNaive`: computes D = rowsum(dO * O) on CPU. No GPU required.
/// - `DSource::B2PrePass`: under `feature = "cuda"`, emits the B.2 D pre-pass PTX
///   and launches it on GPU; returns the D tensor. Stubbed via `unimplemented!()`
///   until the GPU launcher infrastructure is wired (consumers under `feature="cuda"`
///   provide the real launcher; CPU-only builds panic with a clear message).
pub fn compute_d_for_test(
    d_o: &[half::f16],
    output: &[half::f16],
    shape: (usize, usize, usize, usize),
    source: DSource,
) -> Vec<f32> {
    let (batch, heads, seq, hd) = shape;
    match source {
        DSource::CpuNaive => {
            let mut d = vec![0.0f32; batch * heads * seq];
            for b in 0..batch {
                for h in 0..heads {
                    for q in 0..seq {
                        let base = ((b * heads + h) * seq + q) * hd;
                        let row_idx = (b * heads + h) * seq + q;
                        d[row_idx] = (0..hd)
                            .map(|di| d_o[base + di].to_f32() * output[base + di].to_f32())
                            .sum();
                    }
                }
            }
            d
        }
        DSource::B2PrePass => {
            let _ = (batch, heads, seq, hd);
            #[cfg(feature = "cuda")]
            {
                unimplemented!(
                    "DSource::B2PrePass — wire GPU launcher in cuda builds during PR validation"
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("DSource::B2PrePass requires feature='cuda'");
            }
        }
    }
}

// ===== Phase 2.6 — FSource diagnostic surface + ForwardInputs carrier =====

/// Test-default tensor dimensions for the forward-path generators + dispatch.
/// Match the dQ-kernel test conventions (batch=1, heads=1). `seq` is threaded
/// as an explicit parameter (CpuNaive uses 64/128 for multi-q-tile coverage;
/// B1Forward is pinned to 32 by the single-block forward precondition). `D` and
/// `d_model` come from the config (`head_dim` / `csha.d_model`).
const TEST_DEFAULT_BATCH: usize = 1;
const TEST_DEFAULT_HEADS: usize = 1;

/// Sources the forward outputs for a dQ-kernel parity test. Independent of DSource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FSource {
    CpuNaive,
    B1Forward,
}

/// Path-keyed forward inputs. Variant must match the FSource passed to compute_forward_for_test.
pub enum ForwardInputs {
    CpuNaive {
        q: Vec<half::f16>,
        k: Vec<half::f16>,
        v: Vec<half::f16>,
    },
    B1Forward {
        x: Vec<half::f16>,
        wq: Vec<half::f16>,
        wk: Vec<half::f16>,
        wv: Vec<half::f16>,
        norm_weight: Vec<half::f16>,
    },
}

/// dO backward input -- path-independent, row-major [B,H,S,D] f16.
///
/// Deterministic (cosine-of-index) so parity tests are reproducible across
/// the CpuNaive and B1Forward forward paths. `D = cfg.head_dim`; `seq` is
/// passed explicitly so the CpuNaive sweep can use 64/128 while B1Forward uses 32.
pub fn generate_d_o(
    cfg: &nsl_codegen::flash_attention::FlashAttentionConfig,
    seq: usize,
) -> Vec<half::f16> {
    let (b, h, s, d) = (
        TEST_DEFAULT_BATCH,
        TEST_DEFAULT_HEADS,
        seq,
        cfg.head_dim as usize,
    );
    (0..b * h * s * d)
        .map(|i| half::f16::from_f32((i as f32 * 0.419).cos() * 0.1))
        .collect()
}

/// Generate path-appropriate deterministic forward inputs for `source`.
///
/// - `CpuNaive`: raw row-major `[B,H,S,D]` q/k/v consumed directly by
///   `cpu_naive_forward`.
/// - `B1Forward`: raw (un-normalized, un-chunkified) row-major inputs —
///   `x` is `[B,S,d_model]`, `wq/wk/wv` are `[d_model, H*D]`, `norm_weight`
///   is `[d_model]`. The launcher (`run_b1_forward_and_adapt`) performs
///   RMSNorm + narrow + chunkify before the GPU launch.
///
/// `seq` is passed explicitly: CpuNaive uses 64/128 for multi-q-tile coverage,
/// B1Forward is pinned to 32 by the single-block forward precondition.
pub fn generate_forward_inputs(
    cfg: &nsl_codegen::flash_attention::FlashAttentionConfig,
    source: FSource,
    seq: usize,
) -> ForwardInputs {
    let (b, h, s, d) = (
        TEST_DEFAULT_BATCH,
        TEST_DEFAULT_HEADS,
        seq,
        cfg.head_dim as usize,
    );
    match source {
        FSource::CpuNaive => {
            let q = (0..b * h * s * d)
                .map(|i| half::f16::from_f32((i as f32 * 0.137).sin() * 0.1))
                .collect();
            let k = (0..b * h * s * d)
                .map(|i| half::f16::from_f32((i as f32 * 0.211).cos() * 0.1))
                .collect();
            let v = (0..b * h * s * d)
                .map(|i| half::f16::from_f32((i as f32 * 0.317).sin() * 0.1))
                .collect();
            ForwardInputs::CpuNaive { q, k, v }
        }
        FSource::B1Forward => {
            let d_model = cfg.csha.as_ref().map(|c| c.d_model as usize).unwrap_or(128);
            // Raw (un-normalized, un-chunkified) row-major inputs; the launcher
            // (run_b1_forward_and_adapt) does RMSNorm + chunkify before launch.
            let x = (0..b * s * d_model)
                .map(|i| half::f16::from_f32((i as f32 * 0.077).sin() * 0.05))
                .collect();
            let wq = (0..d_model * h * d)
                .map(|i| half::f16::from_f32((i as f32 * 0.091).cos() * 0.05))
                .collect();
            let wk = (0..d_model * h * d)
                .map(|i| half::f16::from_f32((i as f32 * 0.113).sin() * 0.05))
                .collect();
            let wv = (0..d_model * h * d)
                .map(|i| half::f16::from_f32((i as f32 * 0.127).cos() * 0.05))
                .collect();
            let norm_weight = vec![half::f16::from_f32(1.0); d_model];
            ForwardInputs::B1Forward { x, wq, wk, wv, norm_weight }
        }
    }
}

/// Source the forward outputs. CpuNaive -> cpu_naive_forward; B1Forward ->
/// adapter (cuda). The `inputs` variant must match `source` or this panics.
///
/// `seq` must match the value passed to `generate_forward_inputs`. B1Forward is
/// limited to seq <= the launcher's single-block max (32 for the closure gate).
pub fn compute_forward_for_test(
    inputs: &ForwardInputs,
    cfg: &nsl_codegen::flash_attention::FlashAttentionConfig,
    source: FSource,
    seq: usize,
) -> crate::cpu_naive_forward::ForwardOutputs {
    match (inputs, source) {
        (ForwardInputs::CpuNaive { .. }, FSource::CpuNaive) => {}
        (ForwardInputs::B1Forward { .. }, FSource::B1Forward) => {}
        _ => panic!("ForwardInputs variant must match source FSource"),
    }
    let (b, h, s, d) = (
        TEST_DEFAULT_BATCH,
        TEST_DEFAULT_HEADS,
        seq,
        cfg.head_dim as usize,
    );
    match (inputs, source) {
        (ForwardInputs::CpuNaive { q, k, v }, FSource::CpuNaive) => {
            crate::cpu_naive_forward::cpu_naive_forward(q, k, v, b, h, s, d, cfg.causal)
        }
        (ForwardInputs::B1Forward { x, wq, wk, wv, norm_weight }, FSource::B1Forward) => {
            #[cfg(feature = "cuda")]
            {
                crate::b1_adapter::run_b1_forward_and_adapt(x, wq, wk, wv, norm_weight, cfg, seq)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (x, wq, wk, wv, norm_weight, cfg, seq);
                panic!("FSource::B1Forward requires feature='cuda'")
            }
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_naive_d_matches_handwritten_reduction() {
        // 1 batch x 1 head x 2 rows x 4 elements; D = rowsum(dO * O).
        let d_o: Vec<half::f16> = vec![
            half::f16::from_f32(0.1),
            half::f16::from_f32(0.2),
            half::f16::from_f32(0.3),
            half::f16::from_f32(0.4),
            half::f16::from_f32(0.5),
            half::f16::from_f32(0.6),
            half::f16::from_f32(0.7),
            half::f16::from_f32(0.8),
        ];
        let o = d_o.clone();
        let d = compute_d_for_test(&d_o, &o, (1, 1, 2, 4), DSource::CpuNaive);
        // Row 0: 0.01 + 0.04 + 0.09 + 0.16 = 0.30
        // Row 1: 0.25 + 0.36 + 0.49 + 0.64 = 1.74
        assert!((d[0] - 0.30).abs() < 1e-2);
        assert!((d[1] - 1.74).abs() < 1e-2);
    }

    #[test]
    fn d_source_enum_is_copy() {
        let s = DSource::CpuNaive;
        let s2 = s;
        assert_eq!(s, s2);
    }

    #[test]
    fn d_source_b2prepass_distinct_from_cpu_naive() {
        assert_ne!(DSource::B2PrePass, DSource::CpuNaive);
    }
}
