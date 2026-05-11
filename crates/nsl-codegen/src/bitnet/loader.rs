//! HF safetensors checkpoint loader for BitNet b1.58.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §6.
//!
//! Microsoft has no 3B b1.58 release; the canonical reproduction is
//! `1bitLLM/bitnet_b1_58-3B` (PI.2). Weights on disk are FP16; the loader
//! pre-quantizes to packed ternary at load time using the b1.58 BitLinear
//! per-tensor absmean math:
//!
//!     weight_scale = mean(|w|)
//!     w_ternary    = round(clip(w / weight_scale, -1, +1))
//!     packed       = pack_trit_slice(w_ternary)  (Task 2)
//!
//! Per spec §1.3 escalation criteria, future inference must hit ≥80% of
//! the paper's claimed speedup; this pre-quantize-at-load step is a
//! one-time cost amortized across all forward passes.

use crate::bitnet::pack::pack_trit_slice;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::path::Path;

/// One quantized ternary weight tensor with its per-tensor scale.
///
/// **TODO (M35.1 Linux follow-on, before un-ignoring `bitnet_logit_match`):**
/// the `scale` field captured here is the BitLinear b1.58 per-tensor absmean
/// scale used during forward dequantization. Phase 1's `finalize.rs` epilogue
/// currently computes `Y = (act_scale / 127) * acc` and **omits the weight_scale
/// factor**, so the emitted kernel produces results off by a per-layer constant.
/// The Linux follow-on that fills in the full inference harness must:
///   1. Pass each layer's `scale` as a kernel parameter (or burn it as a
///      compile-time immediate in the synthesized PTX).
///   2. Multiply the FP32 accumulator by `scale` inside `finalize.rs::emit`
///      (or factor it into the activation-scale division to save one mul).
/// Until that lands, `bitnet_logit_match`'s logit comparison is intentionally
/// `#[ignore]`'d; pre-flight assertions about prompt-set shape and reference-
/// logits file size are the only currently active gates.
#[derive(Debug)]
pub struct LoadedTernaryWeight {
    /// Tensor name from the safetensors header (e.g., "model.layers.0.self_attn.q_proj.weight").
    pub name: String,
    /// Packed ternary bytes (4 trits per byte, high-bits-first per PACKED_BYTE_LAYOUT.md).
    pub packed_bytes: Vec<u8>,
    /// Original shape from the safetensors header (FP16 source shape).
    pub shape: Vec<usize>,
    /// Per-tensor absmean scale (b1.58 BitLinear convention).
    pub scale: f32,
}

/// Errors from the loader.
#[derive(Debug)]
pub enum LoaderError {
    /// std::io error (file missing, permission denied, etc.).
    Io(std::io::Error),
    /// safetensors crate failed to parse the header or tensor data.
    Safetensors(String),
    /// A tensor's element count is not a multiple of 4 (required by 4-trits-per-byte packing).
    AlignmentError { name: String, numel: usize },
    /// Tensor dtype is unexpected (e.g., not F16).
    UnsupportedDtype { name: String, dtype: String },
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderError::Io(e) => write!(f, "io: {e}"),
            LoaderError::Safetensors(s) => write!(f, "safetensors: {s}"),
            LoaderError::AlignmentError { name, numel } => write!(
                f,
                "tensor {name} has {numel} elements, not a multiple of 4 (cannot pack)"
            ),
            LoaderError::UnsupportedDtype { name, dtype } => write!(
                f,
                "tensor {name} has unsupported dtype {dtype} (expected F16)"
            ),
        }
    }
}

impl std::error::Error for LoaderError {}

impl From<std::io::Error> for LoaderError {
    fn from(e: std::io::Error) -> Self {
        LoaderError::Io(e)
    }
}

/// Whether a tensor name pattern indicates a BitLinear weight that should be quantized.
/// BitNet b1.58 quantizes attention + FFN projection weights (not embeddings or
/// norm parameters). Names like `*.self_attn.{q,k,v,o}_proj.weight` and
/// `*.mlp.{gate,up,down}_proj.weight` match this pattern.
fn is_bitlinear_weight(name: &str) -> bool {
    let bitlinear_suffixes = [
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ];
    bitlinear_suffixes.iter().any(|suf| name.ends_with(suf))
}

/// Quantize one FP16 tensor to packed ternary using b1.58's per-tensor absmean math.
/// Returns (packed_bytes, scale).
fn quantize_fp16_to_ternary(
    name: &str,
    view: TensorView<'_>,
) -> Result<(Vec<u8>, f32), LoaderError> {
    let dtype = view.dtype();
    if dtype != safetensors::Dtype::F16 {
        return Err(LoaderError::UnsupportedDtype {
            name: name.to_string(),
            dtype: format!("{dtype:?}"),
        });
    }
    let bytes = view.data();
    // Convert FP16 → FP32 once.
    let numel = bytes.len() / 2;
    if numel % 4 != 0 {
        return Err(LoaderError::AlignmentError {
            name: name.to_string(),
            numel,
        });
    }
    let mut fp32: Vec<f32> = Vec::with_capacity(numel);
    for chunk in bytes.chunks_exact(2) {
        let h = half::f16::from_le_bytes([chunk[0], chunk[1]]);
        fp32.push(h.to_f32());
    }
    // Per-tensor absmean scale (BitLinear b1.58 convention).
    let abs_sum: f32 = fp32.iter().map(|x| x.abs()).sum();
    let scale = abs_sum / (numel as f32);
    // Guard against degenerate (all-zero) weight tensors.
    let effective_scale = if scale > 0.0 { scale } else { 1.0 };
    // Quantize: round(clip(w / scale, -1, +1)) -> {-1, 0, +1}.
    let trits: Vec<i8> = fp32
        .iter()
        .map(|&w| {
            let scaled = w / effective_scale;
            let clipped = scaled.clamp(-1.0, 1.0);
            // round-half-away-from-zero in f32. b1.58 reference uses banker's rounding
            // but the merge gate already accounts for sub-int8 rounding differences
            // (FP16-ULP tolerance). For pre-quantize-at-load we accept f32 round semantics.
            let rounded = clipped.round();
            rounded as i8
        })
        .collect();
    let packed = pack_trit_slice(&trits);
    Ok((packed, scale))
}

/// Load + pre-quantize all BitLinear weights from a BitNet b1.58 safetensors file.
///
/// Tensors that match `is_bitlinear_weight` are FP16-quantized to packed ternary.
/// Other tensors (embeddings, norms, lm_head if untied) are not returned here;
/// the inference harness handles them as FP16 directly.
pub fn load_bitnet_b158_safetensors(path: &Path) -> Result<Vec<LoadedTernaryWeight>, LoaderError> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|e| LoaderError::Safetensors(format!("{e}")))?;

    let mut out = Vec::new();
    for (name, view) in tensors.tensors() {
        if !is_bitlinear_weight(&name) {
            continue;
        }
        let shape: Vec<usize> = view.shape().to_vec();
        let (packed_bytes, scale) = quantize_fp16_to_ternary(&name, view)?;
        out.push(LoadedTernaryWeight {
            name,
            packed_bytes,
            shape,
            scale,
        });
    }
    Ok(out)
}

/// Read the pinned (model_id, revision_sha) tuple from
/// `tests/fixtures/bitnet_b158_3b_revision.txt`.
pub fn read_pinned_revision(repo_root: &Path) -> std::io::Result<(String, String)> {
    let path = repo_root.join("tests/fixtures/bitnet_b158_3b_revision.txt");
    let content = std::fs::read_to_string(&path)?;
    let mut model_id = String::new();
    let mut revision = String::new();
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("model_id=") {
            model_id = line.trim_start_matches("model_id=").to_string();
        } else if line.starts_with("revision=") {
            revision = line.trim_start_matches("revision=").to_string();
        }
    }
    Ok((model_id, revision))
}
