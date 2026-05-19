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
