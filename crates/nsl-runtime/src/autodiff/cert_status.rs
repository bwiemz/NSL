//! Tape-AD backward certification: coverage.
//!
//! The tape twin of `nsl_codegen::ad_rules::ad_cert_status`. Every [`TapeOp`]
//! variant states how its backward arm (`backward.rs`, `run_backward_core`) is
//! certified, through an exhaustive match with no wildcard arm, so a new
//! variant does not compile until it states one.
//!
//! The certificates are the rows of `crates/nsl-cli/tests/source_ad_rule_cert.rs`:
//! a grad block over one primitive, run as `nsl run` (tape AD), whose loss is
//! held to an independent f64 forward and whose raw gradients are held to
//! central differences of that forward. That file checks the claims here
//! mechanically: every certificate a status names exists and passes in tape
//! mode, and its tape run RECORDS the op (read from the tape's own
//! `[tape-trace] record <Variant>` lines under `NSL_DEBUG_MEM_TRACE=1`).
//!
//! A status names the certificates that TARGET the op, not every program that
//! happens to record it: every certificate's loss is `sum(EXPR * r)`, so all of
//! them record a `Mul` and a `SumReduce`, which certifies neither.
//!
//! Layout is part of the input space. A base certificate hands its op
//! contiguous tensors; its `_vgrad` variant transposes the op's output before
//! the loss (the arm receives a strided gradient view) and its `_vin` variant
//! feeds strided input views. Arms that index storage linearly pass the base
//! and fail the variants; those failures are the [`TapeCertStatus::Defective`]
//! entries.

use super::{TapeOp, TapeShape};

/// How a [`TapeOp`]'s backward arm is certified against an independent oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapeCertStatus {
    /// Certified by these certificates in `source_ad_rule_cert.rs`, each of
    /// which passes in tape mode and records the op there. They run on the
    /// CPU, so they certify the CPU record site and the backward arm; a GPU
    /// record site that saves different state is not covered by them.
    Certified(&'static [&'static str]),
    /// Certified by `certified` (as [`Certified`](Self::Certified)), but each
    /// of `defects` is a certificate whose TAPE run is a ratcheted known
    /// failure: the op gives a wrong result in that configuration (the
    /// certificate's `known` entry says how). This is the inventory's visible
    /// debt. A fix flips the ratchet, and the certificate then moves to
    /// `certified`.
    Defective {
        certified: &'static [&'static str],
        defects: &'static [&'static str],
    },
    /// Reachable on the CPU and differentiable, with no finite-difference
    /// certificate; the reason says why and what holds it instead.
    Uncertified(&'static str),
    /// Recorded only on a GPU path, so no CPU program records it and the CPU
    /// certificate table cannot reach it; the reason names the record site
    /// and what holds it instead.
    GpuOnly(&'static str),
    /// No NSL source spelling records it.
    Unreachable(&'static str),
}

impl TapeCertStatus {
    /// The certificates whose passing tape runs certify the op.
    pub fn certified(&self) -> &'static [&'static str] {
        match *self {
            TapeCertStatus::Certified(names) => names,
            TapeCertStatus::Defective { certified, .. } => certified,
            _ => &[],
        }
    }

    /// The certificates whose tape runs are ratcheted known failures.
    pub fn defects(&self) -> &'static [&'static str] {
        match *self {
            TapeCertStatus::Defective { defects, .. } => defects,
            _ => &[],
        }
    }
}

/// The certification status of every [`TapeOp`] variant.
pub fn tape_cert_status(op: &TapeOp) -> TapeCertStatus {
    use TapeCertStatus::*;
    match op {
        TapeOp::Add { .. } => Certified(&[
            "add_same",
            "add_row",
            "add_col",
            "add_one",
            "add_row_vgrad",
            "add_row_vin",
        ]),
        TapeOp::Sub { .. } => Certified(&["sub_row", "sub_col", "sub_row_vgrad", "sub_row_vin"]),
        TapeOp::Mul { .. } => Certified(&["mul_row", "mul_col", "mul_col_vgrad", "mul_col_vin"]),
        TapeOp::Div { .. } => Certified(&[
            "div_row",
            "div_col",
            "div_numerator_broadcast",
            "div_row_vgrad",
            "div_row_vin",
        ]),
        TapeOp::MatMul { .. } => Certified(&[
            "matmul_2d",
            "matmul_3d_2d",
            "matmul_2d_3d",
            "matmul_3d_3d",
            "matmul_2d_vgrad",
            "matmul_2d_vin",
        ]),
        TapeOp::Fp8MatMul { .. } => Uncertified(
            "recorded on the CPU through an `@fp8_compute` fn in a tape-mode grad or train \
             block (refused under --source-ad). The forward is the plain f32 matmul, but the \
             backward rounds g and both saved operands to E5M2 on purpose (up to 2^-3 relative \
             error per element), so no finite-difference oracle over random operands applies. \
             The arm is held instead by fp8.rs \
             `tape_fp8_backward_gradients_use_the_logical_transpose`: E5M2-exact operands and \
             a ones seed, where the rounding is the identity, against the analytic gradient \
             (its wiring, at one uniform seed). fp8_matmul_backward.rs holds the reference fn \
             `fp8_matmul_e5m2_backward` to a same-rounding oracle",
        ),
        TapeOp::Neg { .. } => Certified(&["neg", "neg_vgrad", "neg_vin"]),
        TapeOp::Cast { .. } => Certified(&[
            "cast_f64_round_trip",
            "cast_f64_square",
            "cast_f64_round_trip_vgrad",
            "cast_f64_round_trip_vin",
        ]),
        TapeOp::MulScalar { .. } => {
            Certified(&["mul_literal", "mul_literal_vgrad", "mul_literal_vin"])
        }
        TapeOp::AddScalar { .. } => {
            Certified(&["add_literal", "add_literal_vgrad", "add_literal_vin"])
        }
        TapeOp::Transpose { .. } => Certified(&[
            "transpose",
            "transpose_3d",
            "contiguous",
            "transpose_3d_vgrad",
            "transpose_3d_vin",
        ]),
        TapeOp::SumReduce { .. } => Defective {
            certified: &[
                "sum_all",
                "sum_dim",
                "sum_dim_neg",
                "sum_dim_keepdim",
                "sum_dim_last",
                "sum_dim_keepdim_mid",
                "sum_dim_vin",
            ],
            defects: &["sum_dim_keepdim_mid_vgrad"],
        },
        TapeOp::MeanReduce { .. } => Defective {
            certified: &[
                "mean_all",
                "mean_dim",
                "mean_dim_last",
                "mean_dim_keepdim_mid",
                "mean_dim_vin",
            ],
            defects: &["mean_dim_keepdim_mid_vgrad"],
        },
        // CPU only. The GPU record site (`nsl_tensor_reduce_max`, cuda arm)
        // saves an all-zero argmax, so on the GPU this arm routes every
        // output's gradient to index 0 of the reduced dim.
        TapeOp::ReduceMax { .. } => Defective {
            certified: &[
                "reduce_max_dim1",
                "reduce_max_dim0",
                "reduce_max_keepdim_last",
                "reduce_max_mid",
                "reduce_max_dim1_vin",
            ],
            defects: &["reduce_max_keepdim_mid", "reduce_max_keepdim_last_vgrad"],
        },
        TapeOp::Gather { .. } => Certified(&[
            "gather",
            "gather_neg",
            "gather_dim0",
            "gather_mid",
            "gather_vin",
        ]),
        TapeOp::Exp { .. } => Certified(&["exp", "exp_vgrad", "exp_vin"]),
        TapeOp::Log { .. } => Certified(&["log", "log_vgrad", "log_vin"]),
        TapeOp::Sqrt { .. } => Certified(&["sqrt", "sqrt_vgrad", "sqrt_vin"]),
        TapeOp::Abs { .. } => Certified(&["abs", "abs_vgrad", "abs_vin"]),
        TapeOp::Clamp { .. } => Defective {
            certified: &["clamp"],
            defects: &["clamp_vgrad", "clamp_vin"],
        },
        TapeOp::ReLU { .. } => Defective {
            certified: &["relu"],
            defects: &["relu_vgrad", "relu_vin"],
        },
        TapeOp::GELU { .. } => Defective {
            certified: &["gelu"],
            defects: &["gelu_vgrad", "gelu_vin"],
        },
        TapeOp::SiLU { .. } => Defective {
            certified: &["silu"],
            defects: &["silu_vgrad", "silu_vin"],
        },
        TapeOp::Sin { .. } => Certified(&["sin", "sin_vgrad", "sin_vin"]),
        TapeOp::Cos { .. } => Certified(&["cos", "cos_vgrad", "cos_vin"]),
        TapeOp::Sigmoid { .. } => Defective {
            certified: &["sigmoid", "sigmoid_vin"],
            defects: &["sigmoid_vgrad"],
        },
        TapeOp::Tanh { .. } => Defective {
            certified: &["tanh", "tanh_vin"],
            defects: &["tanh_vgrad"],
        },
        TapeOp::Softmax { .. } => Defective {
            certified: &[
                "softmax_last",
                "softmax_dim0",
                "softmax_mid",
                "softmax_last_vin",
            ],
            defects: &["softmax_last_vgrad"],
        },
        TapeOp::LogSoftmax { .. } => Defective {
            certified: &[
                "log_softmax_last",
                "log_softmax_dim0",
                "log_softmax_mid",
                "log_softmax_last_vin",
            ],
            defects: &["log_softmax_last_vgrad"],
        },
        TapeOp::Slice { .. } => Defective {
            certified: &[
                "slice_dim1",
                "slice_dim0",
                "slice_neg",
                "slice_mid",
                "slice_dim1_vin",
            ],
            defects: &["slice_dim1_vgrad"],
        },
        TapeOp::Reshape { .. } => Certified(&["reshape", "reshape_vgrad", "reshape_vin"]),
        TapeOp::Cat { .. } => Certified(&[
            "cat_dim0",
            "cat_dim1",
            "cat_three",
            "cat_neg",
            "cat_dim1_vgrad",
            "cat_dim1_vin",
        ]),
        TapeOp::EmbeddingLookup { .. } => Defective {
            certified: &["embedding"],
            defects: &["embedding_vgrad", "embedding_vin"],
        },
        TapeOp::LayerNorm { .. } => Defective {
            certified: &[
                "layernorm",
                "layernorm_eps",
                "layernorm_3d",
                "layernorm_field_eps",
            ],
            defects: &["layernorm_vgrad", "layernorm_vin"],
        },
        TapeOp::RMSNorm { .. } => Defective {
            certified: &[
                "rmsnorm",
                "rmsnorm_eps",
                "rmsnorm_field_eps",
                "rmsnorm_field_eps_reused",
            ],
            defects: &["rmsnorm_vgrad", "rmsnorm_vin"],
        },
        TapeOp::Dropout { .. } => Defective {
            certified: &["dropout_seeded"],
            defects: &["dropout_seeded_vgrad", "dropout_seeded_vin"],
        },
        TapeOp::Conv2d { .. } => Defective {
            certified: &["conv2d"],
            defects: &["conv2d_vgrad", "conv2d_vin"],
        },
        TapeOp::MaxPool2d { .. } => Defective {
            certified: &["maxpool2d", "maxpool2d_overlap", "maxpool2d_pad"],
            defects: &["maxpool2d_vgrad", "maxpool2d_vin"],
        },
        TapeOp::RotateHalf { .. } => {
            Certified(&["rotate_half", "rotate_half_vgrad", "rotate_half_vin"])
        }
        TapeOp::BiasAdd { .. } => Defective {
            certified: &["bias_add"],
            defects: &["bias_add_vgrad", "bias_add_vin"],
        },
        TapeOp::Unsqueeze { .. } => Defective {
            certified: &["unsqueeze", "unsqueeze_vin"],
            defects: &["unsqueeze_vgrad"],
        },
        TapeOp::Expand { .. } => Certified(&["expand", "expand_vgrad"]),
        TapeOp::Stack { .. } => Certified(&[
            "stack_dim0",
            "stack_dim1",
            "stack_neg",
            "stack_dim0_vgrad",
            "stack_dim0_vin",
        ]),
        TapeOp::FlashAttention { .. } => GpuOnly(
            "recorded only by `nsl_flash_attention` / `nsl_flash_attention_csha` after a CUDA \
             kernel launch (the `@flash_attention` path); the CPU `scaled_dot_product_attention` \
             lowers to the decomposed chain, so no CPU program records it. The arm calls \
             `nsl_flash_attention_backward` without PTX (its CPU reference); \
             flash_attention_backward_gpu.rs holds that function's kernels to the reference, \
             but no finite-difference certificate drives this arm",
        ),
        TapeOp::Checkpoint { .. } => Unreachable(
            "its only record site, `nsl_checkpoint_record`, is exported in the ABI table but no \
             codegen path calls it: `@checkpoint` on a grad-block `let` records the callee's own \
             ops (tests/test_checkpoint_grad.nsl). The arm is a no-op, so a recorded Checkpoint \
             would pass NO gradient to its arguments",
        ),
    }
}

/// Every [`TapeOp`] variant (one sample each) with its status, keyed by
/// [`TapeOp::variant_name`]. The coverage gate in `source_ad_rule_cert.rs`
/// reads this; a unit test holds it to the enum's definition.
pub fn tape_cert_inventory() -> Vec<(&'static str, TapeCertStatus)> {
    tape_cert_samples()
        .iter()
        .map(|op| (op.variant_name(), tape_cert_status(op)))
        .collect()
}

/// One sample of every [`TapeOp`] variant. Plain data: nothing here is a live
/// tensor, and nothing reads the fields.
fn tape_cert_samples() -> Vec<TapeOp> {
    let s = TapeShape::new;
    vec![
        TapeOp::Add {
            a: 0,
            b: 0,
            out: 0,
            a_shape: s(),
            b_shape: s(),
        },
        TapeOp::Sub {
            a: 0,
            b: 0,
            out: 0,
            a_shape: s(),
            b_shape: s(),
        },
        TapeOp::Mul {
            a: 0,
            b: 0,
            out: 0,
            saved_a: 0,
            saved_b: 0,
            a_shape: s(),
            b_shape: s(),
        },
        TapeOp::Div {
            a: 0,
            b: 0,
            out: 0,
            saved_a: 0,
            saved_b: 0,
            a_shape: s(),
            b_shape: s(),
        },
        TapeOp::MatMul {
            a: 0,
            b: 0,
            out: 0,
            saved_a: 0,
            saved_b: 0,
        },
        TapeOp::Fp8MatMul {
            a: 0,
            b: 0,
            out: 0,
            saved_a: 0,
            saved_b: 0,
            scale_a: 1.0,
            scale_b: 1.0,
            k_dim: 0,
            device: 0,
        },
        TapeOp::Neg { a: 0, out: 0 },
        TapeOp::Cast {
            a: 0,
            out: 0,
            src_dtype: 0,
        },
        TapeOp::MulScalar {
            a: 0,
            scalar: 0.0,
            out: 0,
        },
        TapeOp::AddScalar { a: 0, out: 0 },
        TapeOp::Transpose {
            a: 0,
            out: 0,
            dim0: 0,
            dim1: 0,
        },
        TapeOp::SumReduce {
            a: 0,
            out: 0,
            dim: 0,
            keepdim: false,
            input_shape: s(),
        },
        TapeOp::MeanReduce {
            a: 0,
            out: 0,
            dim: 0,
            keepdim: false,
            num_elements: 0,
            input_shape: s(),
        },
        TapeOp::ReduceMax {
            a: 0,
            out: 0,
            dim: 0,
            keepdim: false,
            saved_argmax: Vec::new(),
            input_shape: s(),
        },
        TapeOp::Gather {
            a: 0,
            out: 0,
            dim: 0,
            indices_ptr: 0,
            input_shape: s(),
        },
        TapeOp::Exp {
            a: 0,
            out: 0,
            saved_out: 0,
        },
        TapeOp::Log {
            a: 0,
            out: 0,
            saved_a: 0,
        },
        TapeOp::Sqrt {
            a: 0,
            out: 0,
            saved_out: 0,
        },
        TapeOp::Abs {
            a: 0,
            out: 0,
            saved_a: 0,
        },
        TapeOp::Clamp {
            a: 0,
            out: 0,
            saved_a: 0,
            min_val: 0.0,
            max_val: 0.0,
        },
        TapeOp::ReLU {
            a: 0,
            out: 0,
            saved_a: 0,
        },
        TapeOp::GELU {
            a: 0,
            out: 0,
            saved_a: 0,
        },
        TapeOp::SiLU {
            a: 0,
            out: 0,
            saved_a: 0,
        },
        TapeOp::Sin {
            a: 0,
            out: 0,
            saved_a: 0,
        },
        TapeOp::Cos {
            a: 0,
            out: 0,
            saved_a: 0,
        },
        TapeOp::Sigmoid {
            a: 0,
            out: 0,
            saved_out: 0,
        },
        TapeOp::Tanh {
            a: 0,
            out: 0,
            saved_out: 0,
        },
        TapeOp::Softmax {
            a: 0,
            out: 0,
            saved_out: 0,
            dim: 0,
        },
        TapeOp::LogSoftmax {
            a: 0,
            out: 0,
            saved_out: 0,
            dim: 0,
        },
        TapeOp::Slice {
            a: 0,
            out: 0,
            dim: 0,
            start: 0,
            input_shape: s(),
        },
        TapeOp::Reshape {
            a: 0,
            out: 0,
            input_shape: s(),
        },
        TapeOp::Cat {
            inputs: Vec::new(),
            out: 0,
            dim: 0,
            split_sizes: Vec::new(),
        },
        TapeOp::EmbeddingLookup {
            weight: 0,
            indices: 0,
            out: 0,
            saved_weight: 0,
            saved_indices: 0,
        },
        TapeOp::LayerNorm {
            input: 0,
            weight: 0,
            bias: 0,
            out: 0,
            saved_input: 0,
            saved_mean: 0,
            saved_inv_std: 0,
            saved_weight: 0,
        },
        TapeOp::RMSNorm {
            input: 0,
            weight: 0,
            out: 0,
            saved_input: 0,
            saved_rms: 0,
            saved_weight: 0,
        },
        TapeOp::Dropout {
            a: 0,
            out: 0,
            saved_mask: 0,
            scale: 1.0,
        },
        TapeOp::Conv2d {
            input: 0,
            weight: 0,
            bias: 0,
            out: 0,
            saved_input: 0,
            saved_weight: 0,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        },
        TapeOp::MaxPool2d {
            a: 0,
            out: 0,
            saved_argmax: Vec::new(),
            input_shape: s(),
        },
        TapeOp::RotateHalf { a: 0, out: 0 },
        TapeOp::BiasAdd {
            tensor: 0,
            bias: 0,
            out: 0,
        },
        TapeOp::Unsqueeze {
            input: 0,
            out: 0,
            input_shape: s(),
        },
        TapeOp::Expand {
            input: 0,
            out: 0,
            original_shape: s(),
        },
        TapeOp::Stack {
            inputs: Vec::new(),
            out: 0,
            dim: 0,
        },
        TapeOp::FlashAttention {
            q: 0,
            k: 0,
            v: 0,
            out: 0,
            logsumexp: 0,
            scale: 1.0,
            batch: 0,
            heads: 0,
            seq_len: 0,
            head_dim: 0,
            causal: false,
            saved_q: 0,
            saved_k: 0,
            saved_v: 0,
        },
        TapeOp::Checkpoint {
            fn_ptr: 0,
            all_args: Vec::new(),
            tensor_arg_indices: Vec::new(),
            output: 0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The variant names `TapeOp` declares in `mod.rs`, read from its text.
    fn declared_variants() -> Vec<String> {
        let src = include_str!("mod.rs");
        let body = &src[src.find("pub enum TapeOp {").expect("TapeOp")..];
        let body = &body[..body.find("\n}\n").expect("end of TapeOp")];
        body.lines()
            .skip(1)
            .filter_map(|l| {
                let rest = l.strip_prefix("    ")?;
                if !rest.starts_with(|c: char| c.is_ascii_uppercase()) {
                    return None;
                }
                Some(rest.chars().take_while(|c| c.is_alphanumeric()).collect())
            })
            .collect()
    }

    /// Every `TapeOp` variant in `mod.rs` has exactly one entry in
    /// [`tape_cert_inventory`] (and nothing else does), so the coverage gate
    /// in `source_ad_rule_cert.rs` sees every certificate name. The exhaustive
    /// matches in `variant_name` and `tape_cert_status` force an arm for a new
    /// variant; this forces a sample, and pins `variant_name`'s spellings.
    #[test]
    fn tape_cert_inventory_covers_every_variant() {
        let mut declared = declared_variants();
        declared.sort();
        assert!(
            declared.len() > 40,
            "the scan found only {} TapeOp variants",
            declared.len()
        );
        let mut listed: Vec<String> = tape_cert_inventory()
            .into_iter()
            .map(|(n, _)| n.to_string())
            .collect();
        listed.sort();
        assert_eq!(
            declared, listed,
            "tape_cert_inventory is out of step with TapeOp (a missing, extra or duplicated \
             sample, or a variant_name spelling that is not the variant's)"
        );
    }

    /// Statuses are well formed: a certified op names at least one
    /// certificate and no name twice, a defective one at least one defect,
    /// and every other status gives a reason.
    #[test]
    fn tape_cert_statuses_are_well_formed() {
        for (op, status) in tape_cert_inventory() {
            let names: Vec<&str> = match status {
                TapeCertStatus::Certified(names) => names.to_vec(),
                TapeCertStatus::Defective { certified, defects } => {
                    assert!(
                        !defects.is_empty(),
                        "{op} is Defective with no defect: say Certified"
                    );
                    certified.iter().chain(defects).copied().collect()
                }
                TapeCertStatus::Uncertified(_)
                | TapeCertStatus::GpuOnly(_)
                | TapeCertStatus::Unreachable(_) => Vec::new(),
            };
            match status {
                TapeCertStatus::Certified(_) | TapeCertStatus::Defective { .. } => {
                    assert!(!names.is_empty(), "{op} is certified by no certificate");
                    let mut sorted = names.clone();
                    sorted.sort_unstable();
                    sorted.dedup();
                    assert_eq!(sorted.len(), names.len(), "{op} names a certificate twice");
                    assert!(
                        names.iter().all(|n| !n.is_empty()),
                        "{op} names an empty certificate"
                    );
                }
                TapeCertStatus::Uncertified(why)
                | TapeCertStatus::GpuOnly(why)
                | TapeCertStatus::Unreachable(why) => {
                    assert!(
                        why.len() > 20,
                        "{op}: a status reason must say why ({why:?})"
                    );
                }
            }
        }
    }

    /// The record trace names every op by its variant: the certificate
    /// harness parses `[tape-trace] record <Variant>` and nothing else.
    #[test]
    fn tape_op_trace_starts_with_the_variant_name() {
        for op in tape_cert_samples() {
            let line = super::super::tape_op_trace(&op);
            let first = line.split_whitespace().next().unwrap_or("");
            assert_eq!(first, op.variant_name(), "trace line {line:?}");
        }
    }
}
