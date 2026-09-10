//! The calibration corpus's shape, read at compile time without the
//! runtime (roadmap A3).
//!
//! The harness needs `(count, seq)` of the calibration batches before it
//! emits the calibration binary, and `compile_and_calibrate` needs it to
//! size the forward wrapper. The runtime's loader
//! (`nsl_runtime::calibration_data`) used to be called for this, which
//! made the whole runtime a build dependency of the compiler for one
//! header read. This module reads only the header:
//!
//! * `.bin` — the NSL-native format; the header layout is
//!   `nsl_abi::wire::calibration_bin`, the same declaration the runtime's
//!   loader parses, so the two cannot disagree.
//! * `.safetensors` — the standard 8-byte length + JSON header; the
//!   `"calibration"` tensor's `shape` and `dtype` are read from that JSON
//!   (the payload is never touched — the runtime's loader validates it
//!   when it loads the data).
//!
//! Both readers report the same conditions the runtime's loader does
//! (missing `"calibration"` tensor, a non-F32 dtype, a zero-rank tensor),
//! so a corpus this peek accepts is one the harness can load.

use std::path::Path;

use nsl_abi::wire::calibration_bin::{self, BinHeaderError};

/// Why the corpus's shape could not be read.
#[derive(Debug)]
pub enum DataShapeError {
    /// Neither `.bin` nor `.safetensors`.
    UnsupportedExt(String),
    Io(std::io::Error),
    /// The `.bin` header (`nsl_abi::wire::calibration_bin`).
    BinHeader(BinHeaderError),
    /// The safetensors header.
    Safetensors(String),
    /// The corpus is not rank 3 (`[count, seq, hidden]`).
    Rank(usize),
}

impl std::fmt::Display for DataShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedExt(e) => write!(f, "unsupported extension: {e}"),
            Self::Io(e) => write!(f, "io: {e}"),
            Self::BinHeader(e) => write!(f, "bin header: {e}"),
            Self::Safetensors(e) => write!(f, "safetensors: {e}"),
            Self::Rank(r) => write!(f, "bin header: expected rank-3 calibration tensor, got rank {r}"),
        }
    }
}

impl std::error::Error for DataShapeError {}

impl From<std::io::Error> for DataShapeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// The full shape `[count, dim1, dim2, ...]` of the corpus at `path`,
/// from its header alone.
pub fn peek_shape(path: &Path) -> Result<Vec<u32>, DataShapeError> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("safetensors") => peek_safetensors_shape(path),
        Some("bin") => peek_bin_shape(path),
        Some(other) => Err(DataShapeError::UnsupportedExt(other.to_string())),
        None => Err(DataShapeError::UnsupportedExt(String::new())),
    }
}

/// For rank-3 corpora `[count, seq, hidden]`, `(count, seq)`.
pub fn peek_batch_seq(path: &Path) -> Result<(u32, u32), DataShapeError> {
    let shape = peek_shape(path)?;
    if shape.len() != 3 {
        return Err(DataShapeError::Rank(shape.len()));
    }
    Ok((shape[0], shape[1]))
}

/// Read up to `n` bytes from the start of `path`.
fn read_prefix(path: &Path, n: usize) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::with_capacity(n);
    f.by_ref().take(n as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

fn peek_bin_shape(path: &Path) -> Result<Vec<u32>, DataShapeError> {
    // The fixed header names the rank; the dims follow it. Two small
    // reads instead of the whole corpus.
    let fixed = read_prefix(path, calibration_bin::FIXED_HEADER_LEN)?;
    let rank = match calibration_bin::parse_header(&fixed) {
        // A fixed header with rank >= 1 always reports its dims truncated.
        Err(BinHeaderError::TruncatedDims { rank, .. }) => rank,
        Err(e) => return Err(DataShapeError::BinHeader(e)),
        Ok((shape, _)) => return Ok(shape),
    };
    let header = read_prefix(path, calibration_bin::FIXED_HEADER_LEN + rank * 4)?;
    calibration_bin::parse_header(&header)
        .map(|(shape, _)| shape)
        .map_err(DataShapeError::BinHeader)
}

fn peek_safetensors_shape(path: &Path) -> Result<Vec<u32>, DataShapeError> {
    let st = |e: String| DataShapeError::Safetensors(e);
    let len = read_prefix(path, 8)?;
    if len.len() < 8 {
        return Err(st("file shorter than the 8-byte header length".into()));
    }
    let n = u64::from_le_bytes(len[..8].try_into().expect("8 bytes")) as usize;
    let header = read_prefix(path, 8 + n)?;
    if header.len() < 8 + n {
        return Err(st(format!("header length {n} runs past the end of the file")));
    }
    let json: serde_json::Value = serde_json::from_slice(&header[8..8 + n])
        .map_err(|e| st(format!("header is not JSON: {e}")))?;
    let tensor = json
        .get("calibration")
        .ok_or_else(|| st("missing 'calibration' tensor".into()))?;
    let dtype = tensor.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
    if dtype != "F32" {
        return Err(st(format!("expected F32 tensor, got {dtype}")));
    }
    let dims: Vec<u32> = tensor
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or_else(|| st("'calibration' tensor has no shape".into()))?
        .iter()
        .map(|d| d.as_u64().map(|d| d as u32).ok_or_else(|| st("non-integer dimension".into())))
        .collect::<Result<_, _>>()?;
    if dims.is_empty() {
        return Err(st("zero-rank tensor".into()));
    }
    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin_file(dir: &tempfile::TempDir, name: &str, shape: &[u32], payload_f32s: usize) -> std::path::PathBuf {
        let p = dir.path().join(name);
        let mut bytes = calibration_bin::encode_header(shape);
        bytes.extend(std::iter::repeat_n(0u8, payload_f32s * 4));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn bin_shape_comes_from_the_header_alone() {
        let dir = tempfile::tempdir().unwrap();
        let p = bin_file(&dir, "c.bin", &[2, 4, 4], 32);
        assert_eq!(peek_shape(&p).unwrap(), vec![2, 4, 4]);
        assert_eq!(peek_batch_seq(&p).unwrap(), (2, 4));
        // No payload at all: the header still answers (the loader, not the
        // peek, checks the payload length).
        let p = bin_file(&dir, "h.bin", &[3, 5, 7], 0);
        assert_eq!(peek_batch_seq(&p).unwrap(), (3, 5));
    }

    #[test]
    fn bin_errors_match_the_runtime_loader_wording() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("short.bin");
        std::fs::write(&p, b"NSLB").unwrap();
        assert_eq!(peek_shape(&p).unwrap_err().to_string(), "bin header: too short");
        let p = dir.path().join("magic.bin");
        std::fs::write(&p, b"NOPE\x03\0\0\0").unwrap();
        assert_eq!(peek_shape(&p).unwrap_err().to_string(), "bin header: bad magic");
        let p = dir.path().join("rank0.bin");
        std::fs::write(&p, calibration_bin::encode_header(&[])).unwrap();
        assert_eq!(peek_shape(&p).unwrap_err().to_string(), "bin header: rank must be >= 1");
        let p = dir.path().join("dims.bin");
        std::fs::write(&p, &calibration_bin::encode_header(&[1, 2, 3])[..15]).unwrap();
        assert_eq!(peek_shape(&p).unwrap_err().to_string(), "bin header: truncated dims");
        let p = bin_file(&dir, "rank2.bin", &[2, 4], 8);
        assert_eq!(
            peek_batch_seq(&p).unwrap_err().to_string(),
            "bin header: expected rank-3 calibration tensor, got rank 2"
        );
        let p = dir.path().join("c.npy");
        std::fs::write(&p, b"").unwrap();
        assert_eq!(peek_shape(&p).unwrap_err().to_string(), "unsupported extension: npy");
    }

    #[test]
    fn safetensors_shape_comes_from_the_json_header() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.safetensors");
        let data = vec![0u8; 2 * 4 * 8 * 4];
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 4, 8], &data).unwrap();
        safetensors::serialize_to_file([("calibration", view)], None, &p).unwrap();
        assert_eq!(peek_shape(&p).unwrap(), vec![2, 4, 8]);
        assert_eq!(peek_batch_seq(&p).unwrap(), (2, 4));

        // Wrong tensor name and wrong dtype are refused as the loader would.
        let p = dir.path().join("name.safetensors");
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 4, 8], &data).unwrap();
        safetensors::serialize_to_file([("weights", view)], None, &p).unwrap();
        assert_eq!(
            peek_shape(&p).unwrap_err().to_string(),
            "safetensors: missing 'calibration' tensor"
        );
        let p = dir.path().join("dtype.safetensors");
        let half = vec![0u8; 2 * 4 * 8 * 2];
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2, 4, 8], &half).unwrap();
        safetensors::serialize_to_file([("calibration", view)], None, &p).unwrap();
        assert_eq!(peek_shape(&p).unwrap_err().to_string(), "safetensors: expected F32 tensor, got F16");
        // A header length past the end of the file is an error, not a read
        // of garbage.
        let p = dir.path().join("liar.safetensors");
        std::fs::write(&p, 1000u64.to_le_bytes()).unwrap();
        assert!(peek_shape(&p).unwrap_err().to_string().contains("runs past the end"));
    }
}
