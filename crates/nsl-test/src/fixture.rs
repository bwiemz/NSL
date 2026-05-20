//! Fixture binary format per M57 v1 spec §6.2 — custom NSL-defined.
//! 12-byte file header + N × 48-byte block headers + raw little-endian data.
//! Format version 1; rank ≤ 4.

use sha2::{Digest, Sha256};
use thiserror::Error;

pub const MAGIC: [u8; 4] = *b"NSLF";
pub const FORMAT_VERSION: u32 = 1;
pub const MAX_RANK: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BlockKind {
    Weight = 0,
    Bias = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BlockDtype {
    I8 = 0,
    I16 = 1,
    I32 = 2,
    I64 = 3,
}

impl BlockDtype {
    pub fn size_bytes(self) -> usize {
        match self {
            BlockDtype::I8 => 1,
            BlockDtype::I16 => 2,
            BlockDtype::I32 => 4,
            BlockDtype::I64 => 8,
        }
    }

    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::I8),
            1 => Some(Self::I16),
            2 => Some(Self::I32),
            3 => Some(Self::I64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FixtureBlock {
    pub kind: BlockKind,
    pub dtype: BlockDtype,
    pub layer: u32,
    pub rank: u32,
    /// Zero-padded for unused dims.
    pub shape: [u64; MAX_RANK],
    pub data: Vec<u8>,
}

impl FixtureBlock {
    pub fn shape_used(&self) -> &[u64] {
        &self.shape[..self.rank as usize]
    }

    pub fn element_count(&self) -> u64 {
        self.shape_used().iter().product()
    }

    pub fn as_i8(&self) -> Vec<i8> {
        assert_eq!(self.dtype, BlockDtype::I8);
        self.data.iter().map(|&b| b as i8).collect()
    }

    pub fn as_i32(&self) -> Vec<i32> {
        assert_eq!(self.dtype, BlockDtype::I32);
        self.data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub fn as_i64(&self) -> Vec<i64> {
        assert_eq!(self.dtype, BlockDtype::I64);
        self.data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct FixtureFile {
    pub format_version: u32,
    pub blocks: Vec<FixtureBlock>,
}

#[derive(Debug, Error)]
pub enum FixtureError {
    #[error("bad magic: expected NSLF, got {0:?}")]
    BadMagic([u8; 4]),
    #[error(
        "unsupported fixture format version {0}; this nsl-cli expects format version {}",
        FORMAT_VERSION
    )]
    UnsupportedFormatVersion(u32),
    #[error("rank {0} exceeds MAX_RANK={}", MAX_RANK)]
    RankTooLarge(u32),
    #[error("hash mismatch: expected {expected}, got {got}")]
    HashMismatch { expected: String, got: String },
    #[error("truncated file at offset {0}")]
    Truncated(usize),
    #[error("invalid dtype code {0}")]
    InvalidDtype(u32),
    #[error("invalid kind code {0}")]
    InvalidKind(u32),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Compute the SHA-256 hash of bytes, hex-encoded.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let h = Sha256::digest(bytes);
    format!("{:x}", h)
}

/// Verify the bytes match the expected hex hash.
pub fn verify_hash(bytes: &[u8], expected_hex: &str) -> Result<(), FixtureError> {
    let got = sha256_hex(bytes);
    if got == expected_hex {
        Ok(())
    } else {
        Err(FixtureError::HashMismatch {
            expected: expected_hex.to_string(),
            got,
        })
    }
}

/// Parse fixture bytes into a FixtureFile. Assumes hash already verified.
pub fn parse(bytes: &[u8]) -> Result<FixtureFile, FixtureError> {
    if bytes.len() < 12 {
        return Err(FixtureError::Truncated(0));
    }

    let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
    if magic != MAGIC {
        return Err(FixtureError::BadMagic(magic));
    }
    let format_version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if format_version != FORMAT_VERSION {
        return Err(FixtureError::UnsupportedFormatVersion(format_version));
    }
    let n_blocks = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let mut offset = 12usize;

    let mut blocks = Vec::with_capacity(n_blocks as usize);
    for _ in 0..n_blocks {
        if bytes.len() < offset + 48 {
            return Err(FixtureError::Truncated(offset));
        }
        let kind_raw = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let dtype_raw = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
        let layer = u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap());
        let rank = u32::from_le_bytes(bytes[offset + 12..offset + 16].try_into().unwrap());

        if rank as usize > MAX_RANK {
            return Err(FixtureError::RankTooLarge(rank));
        }
        let dtype = BlockDtype::from_u32(dtype_raw)
            .ok_or(FixtureError::InvalidDtype(dtype_raw))?;
        let kind = match kind_raw {
            0 => BlockKind::Weight,
            1 => BlockKind::Bias,
            _ => return Err(FixtureError::InvalidKind(kind_raw)),
        };

        let mut shape = [0u64; MAX_RANK];
        for i in 0..MAX_RANK {
            shape[i] = u64::from_le_bytes(
                bytes[offset + 16 + 8 * i..offset + 24 + 8 * i]
                    .try_into()
                    .unwrap(),
            );
        }
        offset += 48;

        let used_shape = &shape[..rank as usize];
        // Defensive: parse() is pub and `Assumes hash already verified` is a
        // soft precondition. If a caller bypasses hash verification, an
        // adversarial shape (u64 dims that overflow usize on 32-bit, or
        // n_elements*size_bytes that overflows usize anywhere) must not
        // silently produce a truncated FixtureBlock.
        let n_elements: u64 = used_shape.iter().product();
        let size_bytes_u64 = dtype.size_bytes() as u64;
        let data_bytes_u64 = n_elements
            .checked_mul(size_bytes_u64)
            .ok_or(FixtureError::Truncated(offset))?;
        let data_bytes = usize::try_from(data_bytes_u64)
            .map_err(|_| FixtureError::Truncated(offset))?;
        if bytes.len() < offset + data_bytes {
            return Err(FixtureError::Truncated(offset));
        }
        let data = bytes[offset..offset + data_bytes].to_vec();
        offset += data_bytes;

        blocks.push(FixtureBlock {
            kind,
            dtype,
            layer,
            rank,
            shape,
            data,
        });
    }

    Ok(FixtureFile {
        format_version,
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_bad_magic() {
        let bytes = b"BADMx\x00\x00\x00\x00\x00\x00\x00";
        match parse(bytes) {
            Err(FixtureError::BadMagic(_)) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_unsupported_format_version() {
        let mut bytes = vec![];
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&2u32.to_le_bytes()); // version 2
        bytes.extend_from_slice(&0u32.to_le_bytes()); // 0 blocks
        match parse(&bytes) {
            Err(FixtureError::UnsupportedFormatVersion(2)) => {}
            other => panic!("expected UnsupportedFormatVersion(2), got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_truncated_file() {
        let bytes = b"NSL"; // only 3 bytes
        match parse(bytes) {
            Err(FixtureError::Truncated(0)) => {}
            other => panic!("expected Truncated(0), got {other:?}"),
        }
    }

    #[test]
    fn dtype_size_bytes() {
        assert_eq!(BlockDtype::I8.size_bytes(), 1);
        assert_eq!(BlockDtype::I16.size_bytes(), 2);
        assert_eq!(BlockDtype::I32.size_bytes(), 4);
        assert_eq!(BlockDtype::I64.size_bytes(), 8);
    }

    #[test]
    fn sha256_hex_is_deterministic() {
        let a = sha256_hex(b"hello");
        let b = sha256_hex(b"hello");
        assert_eq!(a, b);
        // 32 bytes * 2 hex chars = 64
        assert_eq!(a.len(), 64);
        // known SHA-256 of "hello"
        assert_eq!(
            a,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }
}
