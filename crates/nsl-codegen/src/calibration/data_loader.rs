//! Calibration dataset loader.  Supports `.bin` (tokenized LLM input)
//! and `.safetensors` (any pre-tensorised modality) per spec §5.
//!
//! Task 4 adds `.bin` support; Task 5 adds `.safetensors`.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum CalibrationDataError {
    Io(std::io::Error),
    UnsupportedExtension { path: PathBuf, extension: Option<String> },
    Truncated { path: PathBuf, claimed_bytes: usize, actual_bytes: usize },
    BadHeader { path: PathBuf, reason: String },
}

impl std::fmt::Display for CalibrationDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::UnsupportedExtension { path, extension } => write!(
                f,
                "calibration data {} has unsupported extension {:?} (expected .bin or .safetensors)",
                path.display(),
                extension
            ),
            Self::Truncated { path, claimed_bytes, actual_bytes } => write!(
                f,
                "calibration data {} is truncated: header claims {} bytes, got {}",
                path.display(),
                claimed_bytes,
                actual_bytes
            ),
            Self::BadHeader { path, reason } => write!(
                f,
                "calibration data {} has bad header: {reason}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for CalibrationDataError {}

impl From<std::io::Error> for CalibrationDataError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[derive(Debug)]
pub struct CalibrationDataset {
    kind: DatasetKind,
    num_samples: u32,
    seq_len: u32,
    tokens: Vec<i32>,
}

#[derive(Debug)]
enum DatasetKind {
    Bin,
    Safetensors,
}

impl CalibrationDataset {
    pub fn open(path: &Path) -> Result<Self, CalibrationDataError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase());
        match ext.as_deref() {
            Some("bin") => Self::open_bin(path),
            Some("safetensors") => Self::open_safetensors(path),
            other => Err(CalibrationDataError::UnsupportedExtension {
                path: path.to_path_buf(),
                extension: other.map(str::to_string),
            }),
        }
    }

    fn open_bin(path: &Path) -> Result<Self, CalibrationDataError> {
        let bytes = fs::read(path)?;
        if bytes.len() < 8 {
            return Err(CalibrationDataError::BadHeader {
                path: path.to_path_buf(),
                reason: format!(".bin too small ({} bytes, need >= 8)", bytes.len()),
            });
        }
        let num_samples = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let seq_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let expected_tokens = (num_samples as usize)
            .checked_mul(seq_len as usize)
            .ok_or_else(|| CalibrationDataError::BadHeader {
                path: path.to_path_buf(),
                reason: "num_samples * seq_len overflows usize".into(),
            })?;
        let expected_bytes = 8 + expected_tokens * 4;
        if bytes.len() < expected_bytes {
            return Err(CalibrationDataError::Truncated {
                path: path.to_path_buf(),
                claimed_bytes: expected_bytes,
                actual_bytes: bytes.len(),
            });
        }
        let mut tokens = Vec::with_capacity(expected_tokens);
        for i in 0..expected_tokens {
            let off = 8 + i * 4;
            tokens.push(i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
        }
        Ok(Self {
            kind: DatasetKind::Bin,
            num_samples,
            seq_len,
            tokens,
        })
    }

    fn open_safetensors(path: &Path) -> Result<Self, CalibrationDataError> {
        let bytes = fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| {
            CalibrationDataError::BadHeader {
                path: path.to_path_buf(),
                reason: format!("safetensors parse: {e}"),
            }
        })?;
        // Prefer `input_ids`, fall back to `inputs`.
        let name = ["input_ids", "inputs"]
            .iter()
            .find(|k| st.tensor(k).is_ok())
            .ok_or_else(|| CalibrationDataError::BadHeader {
                path: path.to_path_buf(),
                reason: "no tensor named 'input_ids' or 'inputs' in safetensors file".into(),
            })?;
        let tv = st.tensor(name).map_err(|e| CalibrationDataError::BadHeader {
            path: path.to_path_buf(),
            reason: format!("tensor {name} not retrievable: {e}"),
        })?;
        if tv.dtype() != safetensors::Dtype::I32 {
            return Err(CalibrationDataError::BadHeader {
                path: path.to_path_buf(),
                reason: format!("tensor {name} dtype is {:?}, expected I32", tv.dtype()),
            });
        }
        let shape = tv.shape();
        if shape.len() != 2 {
            return Err(CalibrationDataError::BadHeader {
                path: path.to_path_buf(),
                reason: format!("tensor {name} has rank {}, expected 2", shape.len()),
            });
        }
        let num_samples = shape[0] as u32;
        let seq_len = shape[1] as u32;
        let data = tv.data();
        let expected_bytes = (num_samples as usize) * (seq_len as usize) * 4;
        if data.len() < expected_bytes {
            return Err(CalibrationDataError::Truncated {
                path: path.to_path_buf(),
                claimed_bytes: expected_bytes,
                actual_bytes: data.len(),
            });
        }
        let mut tokens = Vec::with_capacity(expected_bytes / 4);
        for i in 0..(expected_bytes / 4) {
            let off = i * 4;
            tokens.push(i32::from_le_bytes(data[off..off + 4].try_into().unwrap()));
        }
        Ok(Self {
            kind: DatasetKind::Safetensors,
            num_samples,
            seq_len,
            tokens,
        })
    }

    pub fn num_samples(&self) -> u32 {
        self.num_samples
    }

    pub fn seq_len(&self) -> u32 {
        self.seq_len
    }

    pub fn sample_tokens(&self, idx: u32) -> Vec<i32> {
        assert!(idx < self.num_samples, "sample index {idx} >= {}", self.num_samples);
        let s = (idx as usize) * (self.seq_len as usize);
        let e = s + (self.seq_len as usize);
        self.tokens[s..e].to_vec()
    }

    pub fn kind_is_bin(&self) -> bool {
        matches!(self.kind, DatasetKind::Bin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    fn tmp(ext: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut p = std::env::temp_dir();
        p.push(format!("nsl-calib-data-{nanos}-{n}.{ext}"));
        p
    }

    fn write_bin(path: &std::path::Path, samples: u32, seq_len: u32, tokens: &[i32]) {
        let mut f = fs::File::create(path).unwrap();
        f.write_all(&samples.to_le_bytes()).unwrap();
        f.write_all(&seq_len.to_le_bytes()).unwrap();
        for t in tokens {
            f.write_all(&t.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn opens_bin_and_reports_sample_count() {
        let p = tmp("bin");
        let tokens: Vec<i32> = (0..32).collect();
        write_bin(&p, 4, 8, &tokens);
        let ds = CalibrationDataset::open(&p).unwrap();
        assert_eq!(ds.num_samples(), 4);
        assert_eq!(ds.seq_len(), 8);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn reads_sample_tokens() {
        let p = tmp("bin");
        let tokens: Vec<i32> = (0..12).collect();
        write_bin(&p, 3, 4, &tokens);
        let ds = CalibrationDataset::open(&p).unwrap();
        assert_eq!(ds.sample_tokens(0), vec![0, 1, 2, 3]);
        assert_eq!(ds.sample_tokens(1), vec![4, 5, 6, 7]);
        assert_eq!(ds.sample_tokens(2), vec![8, 9, 10, 11]);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn rejects_bad_extension() {
        let p = tmp("txt");
        fs::write(&p, b"whatever").unwrap();
        match CalibrationDataset::open(&p) {
            Err(CalibrationDataError::UnsupportedExtension { .. }) => {}
            other => panic!("expected UnsupportedExtension, got {other:?}"),
        }
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn rejects_truncated_bin() {
        let p = tmp("bin");
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(&10u32.to_le_bytes()).unwrap();
        f.write_all(&8u32.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
        drop(f);
        match CalibrationDataset::open(&p) {
            Err(CalibrationDataError::Truncated { .. }) => {}
            other => panic!("expected Truncated, got {other:?}"),
        }
        let _ = fs::remove_file(&p);
    }

    fn write_safetensors_input_ids(path: &std::path::Path, samples: u64, seq_len: u64) {
        // Minimal safetensors blob: one tensor named "input_ids", shape
        // [samples, seq_len], dtype I32.  Header is JSON:
        //   {"input_ids":{"dtype":"I32","shape":[samples,seq_len],"data_offsets":[0,nbytes]}}
        let nbytes = (samples * seq_len * 4) as usize;
        let header_json = format!(
            r#"{{"input_ids":{{"dtype":"I32","shape":[{samples},{seq_len}],"data_offsets":[0,{nbytes}]}}}}"#
        );
        let header_bytes = header_json.into_bytes();
        let header_len = header_bytes.len() as u64;

        let mut blob = Vec::new();
        blob.extend_from_slice(&header_len.to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        for i in 0..(samples * seq_len) as i32 {
            blob.extend_from_slice(&i.to_le_bytes());
        }
        fs::write(path, blob).unwrap();
    }

    #[test]
    fn opens_safetensors_and_reports_sample_count() {
        let p = tmp("safetensors");
        write_safetensors_input_ids(&p, 3, 5);
        let ds = CalibrationDataset::open(&p).unwrap();
        assert_eq!(ds.num_samples(), 3);
        assert_eq!(ds.seq_len(), 5);
        assert!(!ds.kind_is_bin());
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn safetensors_sample_tokens_match_sequential_fill() {
        let p = tmp("safetensors");
        write_safetensors_input_ids(&p, 2, 4);
        let ds = CalibrationDataset::open(&p).unwrap();
        assert_eq!(ds.sample_tokens(0), vec![0, 1, 2, 3]);
        assert_eq!(ds.sample_tokens(1), vec![4, 5, 6, 7]);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn safetensors_missing_input_ids_errors() {
        let p = tmp("safetensors");
        let header = r#"{"other":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let hb = header.as_bytes();
        let mut blob = (hb.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(hb);
        blob.extend_from_slice(&[0u8; 16]);
        fs::write(&p, blob).unwrap();
        match CalibrationDataset::open(&p) {
            Err(CalibrationDataError::BadHeader { reason, .. }) => {
                assert!(reason.contains("input_ids") || reason.contains("inputs"));
            }
            other => panic!("expected BadHeader, got {other:?}"),
        }
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn out_of_bounds_sample_index_panics_in_debug() {
        let p = tmp("bin");
        let tokens: Vec<i32> = vec![0, 1, 2, 3];
        write_bin(&p, 1, 4, &tokens);
        let ds = CalibrationDataset::open(&p).unwrap();
        let result = std::panic::catch_unwind(|| ds.sample_tokens(99));
        assert!(result.is_err());
        let _ = fs::remove_file(&p);
    }
}
