//! NSLI binary log format (mirror of NSLM checkpoint).
//!
//! Layout:
//!   [0..4]   magic = b"NSLI"
//!   [4..8]   version: u32 (LE)
//!   [8..16]  header_len: u64 (LE)
//!   [16..16+header_len]  JSON header (UTF-8)
//!   [aligned to 64-byte boundary]  raw tensor bytes (full dumps only)

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

const MAGIC: &[u8; 4] = b"NSLI";
const VERSION: u32 = 1;
const ALIGN: u64 = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsHeader {
    pub step: u64,
    pub tensor_name: String,
    pub kind: String,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub nan_count: u64,
    pub inf_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullHeader {
    pub step: u64,
    pub tensor_name: String,
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub stats: StatsHeader,
}

pub fn write_stats(path: &Path, step: u64, name: &str, stats: &[f64]) -> std::io::Result<()> {
    if stats.len() != 6 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "stats slice must contain 6 values: mean, std, min, max, nan_count, inf_count",
        ));
    }
    let header = StatsHeader {
        step,
        tensor_name: name.into(),
        kind: "stats".into(),
        mean: stats[0],
        std: stats[1],
        min: stats[2],
        max: stats[3],
        nan_count: stats[4] as u64,
        inf_count: stats[5] as u64,
    };
    let json = serde_json::to_vec(&header).map_err(std::io::Error::other)?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&(json.len() as u64).to_le_bytes())?;
    f.write_all(&json)?;
    Ok(())
}

pub fn write_full(path: &Path, header: &FullHeader, data: &[u8]) -> std::io::Result<()> {
    let json = serde_json::to_vec(header).map_err(std::io::Error::other)?;
    let header_len = json.len() as u64;
    let aligned = (16 + header_len).div_ceil(ALIGN) * ALIGN;
    let pad = aligned - (16 + header_len);
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(&json)?;
    f.write_all(&vec![0u8; pad as usize])?;
    f.write_all(data)?;
    Ok(())
}
