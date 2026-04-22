//! WGGO — `.nslweights` checkpoint adapter.
//!
//! Loads an `.nslweights` sidecar file from disk and exposes its tensors
//! to the WGGO Stage-3 analyzer via [`crate::wggo_weight_analysis::
//! WeightProvider`].  The analyzer expects `&[f32]`, so per-tensor
//! buffers are converted from the checkpoint's native dtype (f32 / f64
//! / f16 / bf16) once at construction; subsequent lookups are O(1).
//!
//! The `.nslweights` layout is:
//!
//! ```text
//! [0..4]   magic "NSLW"
//! [4..8]   version u32 LE
//! [8..16]  header_size u64 LE
//! [16..]   JSON header (header_size bytes)
//! <pad to 64-byte alignment>
//! <raw tensor data>
//! ```
//!
//! The JSON header's `params` array contains
//! `{name, dtype, shape, offset, nbytes}` records; offsets are relative
//! to the start of the raw-data region.
//!
//! The parser is intentionally duplicated from `nsl-runtime`'s
//! standalone loader because that loader is a C FFI entry point with
//! `std::process::abort` error handling — not what a compile-time
//! analyzer wants.  This version returns `Result` and never aborts.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

use half::{bf16, f16};

use crate::wggo_weight_analysis::WeightProvider;

const NSLW_MAGIC: &[u8; 4] = b"NSLW";
const NSLW_VERSION: u32 = 1;

/// Errors returned while building a checkpoint adapter.
#[derive(Debug)]
pub enum NslWeightsError {
    Io(io::Error),
    TooSmall { need: usize, got: usize },
    BadMagic,
    UnsupportedVersion { got: u32 },
    TruncatedHeader { need: usize, got: usize },
    BadUtf8,
    BadJson(String),
    MissingParamsArray,
    MissingField { param: String, field: &'static str },
    BadShape(String),
    UnknownDtype { param: String, dtype: String },
    TensorOutOfBounds { param: String, offset: usize, nbytes: usize, total: usize },
    DtypeSizeMismatch { param: String, dtype: String, nbytes: usize },
}

impl std::fmt::Display for NslWeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::TooSmall { need, got } => write!(f, "blob too small ({got} bytes, need {need})"),
            Self::BadMagic => write!(f, "invalid .nslweights blob (bad magic)"),
            Self::UnsupportedVersion { got } => {
                write!(f, "unsupported version {got} (expected {NSLW_VERSION})")
            }
            Self::TruncatedHeader { need, got } => {
                write!(f, "header truncated (need {need} bytes, have {got})")
            }
            Self::BadUtf8 => write!(f, "header is not valid UTF-8"),
            Self::BadJson(e) => write!(f, "header JSON parse error: {e}"),
            Self::MissingParamsArray => write!(f, "header missing 'params' array"),
            Self::MissingField { param, field } => {
                write!(f, "tensor '{param}' missing '{field}'")
            }
            Self::BadShape(p) => write!(f, "tensor '{p}' has invalid shape"),
            Self::UnknownDtype { param, dtype } => {
                write!(f, "tensor '{param}' has unsupported dtype '{dtype}'")
            }
            Self::TensorOutOfBounds { param, offset, nbytes, total } => write!(
                f,
                "tensor '{param}' at offset={offset} nbytes={nbytes} exceeds blob size {total}"
            ),
            Self::DtypeSizeMismatch { param, dtype, nbytes } => write!(
                f,
                "tensor '{param}' dtype='{dtype}' but nbytes={nbytes} is not a multiple of the dtype size"
            ),
        }
    }
}

impl std::error::Error for NslWeightsError {}
impl From<io::Error> for NslWeightsError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// In-memory adapter around a parsed `.nslweights` file.
pub struct NslWeightsCheckpoint {
    /// Parameter name → converted f32 buffer.
    tensors: HashMap<String, Vec<f32>>,
    /// Parameter name → shape (u64 copy so the trait's `&[u64]` borrows
    /// outlive the `get_shape` call).
    shapes: HashMap<String, Vec<u64>>,
}

impl NslWeightsCheckpoint {
    /// Load and parse an `.nslweights` file from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, NslWeightsError> {
        let raw = fs::read(path)?;
        Self::from_bytes(&raw)
    }

    /// Parse an already-loaded `.nslweights` byte blob.
    pub fn from_bytes(raw: &[u8]) -> Result<Self, NslWeightsError> {
        let (metas, data_start) = parse_header(raw)?;
        let data = &raw[data_start..];
        let mut tensors = HashMap::with_capacity(metas.len());
        let mut shapes = HashMap::with_capacity(metas.len());
        for m in metas {
            if m.offset + m.nbytes > data.len() {
                return Err(NslWeightsError::TensorOutOfBounds {
                    param: m.name,
                    offset: m.offset,
                    nbytes: m.nbytes,
                    total: data.len(),
                });
            }
            let bytes = &data[m.offset..m.offset + m.nbytes];
            let f32_buf = convert_to_f32(&m.name, &m.dtype, bytes)?;
            shapes.insert(
                m.name.clone(),
                m.shape.iter().map(|&s| s.max(0) as u64).collect(),
            );
            tensors.insert(m.name, f32_buf);
        }
        Ok(Self { tensors, shapes })
    }

    /// Number of parameters stored.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the checkpoint has no parameters.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl std::fmt::Debug for NslWeightsCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NslWeightsCheckpoint")
            .field("num_params", &self.tensors.len())
            .finish()
    }
}

impl WeightProvider for NslWeightsCheckpoint {
    fn get(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|v| v.as_slice())
    }
    fn shape(&self, name: &str) -> Option<&[u64]> {
        self.shapes.get(name).map(|v| v.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

struct TensorMeta {
    name: String,
    shape: Vec<i64>,
    dtype: String,
    offset: usize,
    nbytes: usize,
}

fn parse_header(raw: &[u8]) -> Result<(Vec<TensorMeta>, usize), NslWeightsError> {
    if raw.len() < 16 {
        return Err(NslWeightsError::TooSmall { need: 16, got: raw.len() });
    }
    if &raw[0..4] != NSLW_MAGIC {
        return Err(NslWeightsError::BadMagic);
    }
    let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    if version != NSLW_VERSION {
        return Err(NslWeightsError::UnsupportedVersion { got: version });
    }
    let header_size = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
    let header_end = 16 + header_size;
    if raw.len() < header_end {
        return Err(NslWeightsError::TruncatedHeader {
            need: header_end,
            got: raw.len(),
        });
    }
    let json_bytes = &raw[16..header_end];
    let json_str = std::str::from_utf8(json_bytes).map_err(|_| NslWeightsError::BadUtf8)?;
    let root: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| NslWeightsError::BadJson(e.to_string()))?;
    let params = root["params"]
        .as_array()
        .ok_or(NslWeightsError::MissingParamsArray)?;

    let mut metas = Vec::with_capacity(params.len());
    for p in params {
        let name = p["name"]
            .as_str()
            .ok_or(NslWeightsError::MissingField {
                param: "?".to_string(),
                field: "name",
            })?
            .to_string();
        let dtype = p["dtype"]
            .as_str()
            .unwrap_or("f64")
            .to_string();
        let offset = p["offset"]
            .as_u64()
            .ok_or_else(|| NslWeightsError::MissingField {
                param: name.clone(),
                field: "offset",
            })? as usize;
        let nbytes = p["nbytes"]
            .as_u64()
            .ok_or_else(|| NslWeightsError::MissingField {
                param: name.clone(),
                field: "nbytes",
            })? as usize;
        let shape: Vec<i64> = p["shape"]
            .as_array()
            .ok_or_else(|| NslWeightsError::BadShape(name.clone()))?
            .iter()
            .map(|v| v.as_i64().unwrap_or(0))
            .collect();
        metas.push(TensorMeta { name, shape, dtype, offset, nbytes });
    }

    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;
    Ok((metas, data_start))
}

fn convert_to_f32(param: &str, dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, NslWeightsError> {
    let dsize = dtype_size(dtype).ok_or_else(|| NslWeightsError::UnknownDtype {
        param: param.to_string(),
        dtype: dtype.to_string(),
    })?;
    if !bytes.len().is_multiple_of(dsize) {
        return Err(NslWeightsError::DtypeSizeMismatch {
            param: param.to_string(),
            dtype: dtype.to_string(),
            nbytes: bytes.len(),
        });
    }
    let n = bytes.len() / dsize;
    let mut out = Vec::with_capacity(n);
    match dtype {
        "f32" => {
            for i in 0..n {
                let v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                out.push(v);
            }
        }
        "f64" => {
            for i in 0..n {
                let v = f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
                out.push(v as f32);
            }
        }
        "f16" => {
            for i in 0..n {
                let v = f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(v.to_f32());
            }
        }
        "bf16" => {
            for i in 0..n {
                let v = bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(v.to_f32());
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}

fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "f32" => Some(4),
        "f64" => Some(8),
        "f16" | "bf16" => Some(2),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a minimal in-memory .nslweights blob for testing.
    fn build_blob(params: &[(&str, &str, Vec<i64>, Vec<u8>)]) -> Vec<u8> {
        // Assign offsets sequentially with no gaps.
        let mut offset = 0usize;
        let mut param_json = Vec::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in params {
            param_json.push(json!({
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "offset": offset,
                "nbytes": bytes.len(),
            }));
            offset += bytes.len();
            data.extend_from_slice(bytes);
        }
        let header_json = json!({ "params": param_json }).to_string();
        let header_bytes = header_json.into_bytes();
        let header_size = header_bytes.len() as u64;

        let mut blob = Vec::new();
        blob.extend_from_slice(NSLW_MAGIC);
        blob.extend_from_slice(&NSLW_VERSION.to_le_bytes());
        blob.extend_from_slice(&header_size.to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        // Pad to 64-byte alignment.
        let total_header = 16 + header_bytes.len();
        let padding = (64 - (total_header % 64)) % 64;
        blob.extend(std::iter::repeat(0u8).take(padding));
        blob.extend_from_slice(&data);
        blob
    }

    #[test]
    fn parses_single_f32_tensor() {
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let blob = build_blob(&[("blocks.0.attn.wq", "f32", vec![2, 2], data)]);
        let ck = NslWeightsCheckpoint::from_bytes(&blob).expect("parse");
        assert_eq!(ck.len(), 1);
        let w = ck.get("blocks.0.attn.wq").expect("present");
        assert_eq!(w, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ck.shape("blocks.0.attn.wq"), Some([2u64, 2].as_slice()));
    }

    #[test]
    fn converts_f64_to_f32() {
        let data: Vec<u8> = [1.5f64, -2.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let blob = build_blob(&[("x", "f64", vec![2], data)]);
        let ck = NslWeightsCheckpoint::from_bytes(&blob).unwrap();
        assert_eq!(ck.get("x"), Some(&[1.5f32, -2.25][..]));
    }

    #[test]
    fn converts_f16_to_f32() {
        let v0 = f16::from_f32(3.5);
        let v1 = f16::from_f32(-1.0);
        let mut data = Vec::new();
        data.extend_from_slice(&v0.to_le_bytes());
        data.extend_from_slice(&v1.to_le_bytes());
        let blob = build_blob(&[("y", "f16", vec![2], data)]);
        let ck = NslWeightsCheckpoint::from_bytes(&blob).unwrap();
        let w = ck.get("y").unwrap();
        assert!((w[0] - 3.5).abs() < 1e-3);
        assert!((w[1] - -1.0).abs() < 1e-3);
    }

    #[test]
    fn converts_bf16_to_f32() {
        let v0 = bf16::from_f32(2.0);
        let v1 = bf16::from_f32(0.5);
        let mut data = Vec::new();
        data.extend_from_slice(&v0.to_le_bytes());
        data.extend_from_slice(&v1.to_le_bytes());
        let blob = build_blob(&[("z", "bf16", vec![2], data)]);
        let ck = NslWeightsCheckpoint::from_bytes(&blob).unwrap();
        let w = ck.get("z").unwrap();
        assert!((w[0] - 2.0).abs() < 1e-2);
        assert!((w[1] - 0.5).abs() < 1e-2);
    }

    #[test]
    fn missing_tensor_returns_none() {
        let data: Vec<u8> = [1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let blob = build_blob(&[("a", "f32", vec![1], data)]);
        let ck = NslWeightsCheckpoint::from_bytes(&blob).unwrap();
        assert!(ck.get("nonexistent").is_none());
        assert!(ck.shape("nonexistent").is_none());
    }

    #[test]
    fn bad_magic_returns_error() {
        let bad = vec![0u8; 64];
        let err = NslWeightsCheckpoint::from_bytes(&bad).unwrap_err();
        assert!(matches!(err, NslWeightsError::BadMagic));
    }

    #[test]
    fn too_small_returns_error() {
        let err = NslWeightsCheckpoint::from_bytes(&[0u8; 4]).unwrap_err();
        assert!(matches!(err, NslWeightsError::TooSmall { .. }));
    }

    #[test]
    fn unknown_dtype_returns_error() {
        let blob = build_blob(&[("w", "int8", vec![4], vec![0u8; 4])]);
        let err = NslWeightsCheckpoint::from_bytes(&blob).unwrap_err();
        assert!(matches!(err, NslWeightsError::UnknownDtype { .. }));
    }

    #[test]
    fn implements_weight_provider_trait() {
        // Round-trip through the trait to verify it works polymorphically.
        let data: Vec<u8> = [7.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let blob = build_blob(&[("p", "f32", vec![1], data)]);
        let ck = NslWeightsCheckpoint::from_bytes(&blob).unwrap();
        let dyn_ref: &dyn WeightProvider = &ck;
        assert_eq!(dyn_ref.get("p"), Some(&[7.0f32][..]));
        assert_eq!(dyn_ref.shape("p"), Some([1u64].as_slice()));
    }

    #[test]
    fn end_to_end_with_analyzer() {
        // Build a blob with a Wq that has one high-norm head, then run
        // the Stage-3 analyzer through the trait object.
        use crate::wengert::{PrimalOp, WengertList, WengertOp};
        use crate::wggo_cost::LayerShape;
        use crate::wggo_graph::build as build_graph;
        use crate::wggo_weight_analysis::{analyze, AnalysisConfig};
        use std::collections::HashMap;

        // 4 heads, head_dim=4, d_model=16 → Wq shape [16, 16]; boost head 2.
        let mut wq = vec![1.0f32; 16 * 16];
        for row in 0..16 {
            for col in 8..12 {
                wq[row * 16 + col] = 10.0;
            }
        }
        let data: Vec<u8> = wq.iter().flat_map(|v| v.to_le_bytes()).collect();
        let blob = build_blob(&[("blocks.0.attn.wq", "f32", vec![16, 16], data)]);
        let ck = NslWeightsCheckpoint::from_bytes(&blob).unwrap();

        let ops = vec![
            WengertOp {
                id: 0,
                result: 0,
                op: PrimalOp::Param("blocks.0.attn.wq".into()),
                inputs: vec![],
                saved_for_backward: false,
                checkpointed: false,
            },
        ];
        let wl = WengertList {
            ops,
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let g = build_graph(&wl);

        let shape = LayerShape {
            batch: 1,
            seq: 64,
            d_model: 16,
            head_dim: 4,
            n_kv_heads: 4,
            dtype_bytes: 2,
        };
        let rep = analyze(&g, &shape, 4, &ck, &AnalysisConfig::default());
        let idx = g
            .layers
            .iter()
            .position(|l| l.name == "blocks.0")
            .unwrap();
        let s = &rep.per_layer[idx].head_scores;
        assert!(s[2] > s[0] && s[2] > s[1] && s[2] > s[3]);
    }
}
