//! Binary shard format and deterministic distributed shard assignment.
//!
//! Each shard is a contiguous binary file containing variable-length samples.
//! Format: [ShardHeader] [sample_0] [sample_1] ... [sample_N] [index_table]
//!
//! Shard assignment is deterministic per (epoch, rank, world_size) — no global
//! coordination needed. Each rank can independently compute its shard set.

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Shard format constants
// ---------------------------------------------------------------------------

/// Magic bytes: "NSLD" (NeuralScript Data).
pub const SHARD_MAGIC: [u8; 4] = *b"NSLD";

/// Current shard format version.
pub const SHARD_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Shard header
// ---------------------------------------------------------------------------

/// Header for a binary shard file.
#[derive(Debug, Clone)]
pub struct ShardHeader {
    /// Magic bytes (must be SHARD_MAGIC).
    pub magic: [u8; 4],
    /// Format version.
    pub version: u32,
    /// Number of samples in this shard.
    pub num_samples: u64,
    /// Byte offset where the index table starts (end of sample data).
    pub index_offset: u64,
}

impl ShardHeader {
    /// Serialized size in bytes.
    pub const SIZE: usize = 4 + 4 + 8 + 8; // 24 bytes

    /// Write header to a writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.num_samples.to_le_bytes())?;
        w.write_all(&self.index_offset.to_le_bytes())?;
        Ok(())
    }

    /// Read header from a reader.
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != SHARD_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid shard magic"));
        }
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);

        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let num_samples = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let index_offset = u64::from_le_bytes(buf8);

        Ok(ShardHeader { magic, version, num_samples, index_offset })
    }
}

// ---------------------------------------------------------------------------
// Shard writer
// ---------------------------------------------------------------------------

/// Writes samples to a binary shard file.
pub struct ShardWriter<W: Write + Seek> {
    writer: W,
    offsets: Vec<u64>,
    current_offset: u64,
}

impl<W: Write + Seek> ShardWriter<W> {
    /// Create a new shard writer. Writes a placeholder header.
    pub fn new(mut writer: W) -> io::Result<Self> {
        // Write placeholder header (will be updated on finalize)
        let header = ShardHeader {
            magic: SHARD_MAGIC,
            version: SHARD_VERSION,
            num_samples: 0,
            index_offset: 0,
        };
        header.write_to(&mut writer)?;
        Ok(ShardWriter {
            writer,
            offsets: Vec::new(),
            current_offset: ShardHeader::SIZE as u64,
        })
    }

    /// Write a sample (variable-length bytes). Returns the sample index.
    pub fn write_sample(&mut self, data: &[u8]) -> io::Result<usize> {
        let idx = self.offsets.len();
        self.offsets.push(self.current_offset);

        // Write length prefix (u32) + data
        let len = data.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(data)?;
        self.current_offset += 4 + data.len() as u64;

        Ok(idx)
    }

    /// Finalize the shard: write index table and update header.
    pub fn finalize(mut self) -> io::Result<()> {
        let index_offset = self.current_offset;

        // Write index table: [offset_0, offset_1, ..., offset_N]
        for &offset in &self.offsets {
            self.writer.write_all(&offset.to_le_bytes())?;
        }

        // Seek back and update header
        self.writer.seek(SeekFrom::Start(0))?;
        let header = ShardHeader {
            magic: SHARD_MAGIC,
            version: SHARD_VERSION,
            num_samples: self.offsets.len() as u64,
            index_offset,
        };
        header.write_to(&mut self.writer)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shard reader
// ---------------------------------------------------------------------------

/// Reads samples from a binary shard file.
pub struct ShardReader<R: Read + Seek> {
    reader: R,
    pub header: ShardHeader,
    /// Byte offset of each sample (loaded from index table).
    offsets: Vec<u64>,
}

impl<R: Read + Seek> ShardReader<R> {
    /// Open a shard for reading.
    pub fn open(mut reader: R) -> io::Result<Self> {
        let header = ShardHeader::read_from(&mut reader)?;

        // Read index table
        reader.seek(SeekFrom::Start(header.index_offset))?;
        let n = header.num_samples as usize;
        let mut offsets = Vec::with_capacity(n);
        let mut buf8 = [0u8; 8];
        for _ in 0..n {
            reader.read_exact(&mut buf8)?;
            offsets.push(u64::from_le_bytes(buf8));
        }

        Ok(ShardReader { reader, header, offsets })
    }

    /// Number of samples in this shard.
    pub fn num_samples(&self) -> usize {
        self.header.num_samples as usize
    }

    /// Read sample at the given index. Returns the raw bytes.
    pub fn read_sample(&mut self, index: usize) -> io::Result<Vec<u8>> {
        if index >= self.offsets.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "sample index out of bounds"));
        }
        self.reader.seek(SeekFrom::Start(self.offsets[index]))?;

        // Read length prefix
        let mut buf4 = [0u8; 4];
        self.reader.read_exact(&mut buf4)?;
        let len = u32::from_le_bytes(buf4) as usize;

        // Read sample data
        let mut data = vec![0u8; len];
        self.reader.read_exact(&mut data)?;
        Ok(data)
    }

    /// Iterate over all samples sequentially (cache-friendly sequential I/O).
    pub fn iter_samples(&mut self) -> ShardIterator<'_, R> {
        ShardIterator { reader: self, next_idx: 0 }
    }
}

/// Sequential iterator over shard samples.
pub struct ShardIterator<'a, R: Read + Seek> {
    reader: &'a mut ShardReader<R>,
    next_idx: usize,
}

impl<'a, R: Read + Seek> Iterator for ShardIterator<'a, R> {
    type Item = io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_idx >= self.reader.num_samples() {
            return None;
        }
        let result = self.reader.read_sample(self.next_idx);
        self.next_idx += 1;
        Some(result)
    }
}

// ---------------------------------------------------------------------------
// Deterministic shard assignment
// ---------------------------------------------------------------------------

/// Deterministic shard assignment for distributed training.
///
/// Given a (rank, world_size, num_shards, epoch), returns the shard IDs
/// assigned to this rank. The assignment is:
///   1. Seed a PRNG with `epoch`
///   2. Shuffle shard IDs [0..num_shards)
///   3. Each rank takes every `world_size`-th shard starting at `rank`
///
/// This is deterministic — same inputs always produce the same assignment,
/// with no inter-rank communication.
pub fn assign_shards(rank: u32, world_size: u32, num_shards: u32, epoch: u64) -> Vec<u32> {
    if world_size == 0 || num_shards == 0 {
        return Vec::new();
    }

    // Simple deterministic shuffle using epoch as seed
    let mut shard_ids: Vec<u32> = (0..num_shards).collect();
    let mut seed = epoch.wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);

    // Fisher-Yates shuffle with deterministic PRNG
    for i in (1..shard_ids.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (seed >> 33) as usize % (i + 1);
        shard_ids.swap(i, j);
    }

    // Each rank gets every world_size-th shard
    shard_ids.into_iter()
        .skip(rank as usize)
        .step_by(world_size as usize)
        .collect()
}

// ---------------------------------------------------------------------------
// Data checkpoint for exact resumption
// ---------------------------------------------------------------------------

/// Checkpoint state for the data pipeline.
/// Saved alongside model checkpoints for exact resumption.
#[derive(Debug, Clone)]
pub struct DataCheckpoint {
    /// Current shard ID being processed.
    pub shard_id: u32,
    /// Sample offset within the current shard.
    pub sample_offset: u64,
    /// Current epoch number.
    pub epoch: u64,
    /// PRNG state for deterministic resumption.
    pub rng_state: [u8; 32],
}

impl DataCheckpoint {
    /// Serialized size in bytes.
    pub const SIZE: usize = 4 + 8 + 8 + 32; // 52 bytes

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::SIZE);
        buf.extend_from_slice(&self.shard_id.to_le_bytes());
        buf.extend_from_slice(&self.sample_offset.to_le_bytes());
        buf.extend_from_slice(&self.epoch.to_le_bytes());
        buf.extend_from_slice(&self.rng_state);
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE { return None; }
        Some(DataCheckpoint {
            shard_id: u32::from_le_bytes(buf[0..4].try_into().ok()?),
            sample_offset: u64::from_le_bytes(buf[4..12].try_into().ok()?),
            epoch: u64::from_le_bytes(buf[12..20].try_into().ok()?),
            rng_state: buf[20..52].try_into().ok()?,
        })
    }
}

// ---------------------------------------------------------------------------
// Shard directory (for discovering shards on disk)
// ---------------------------------------------------------------------------

/// Discover all .nsld shard files in a directory.
pub fn discover_shards(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut shards = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "nsld") {
                shards.push(path);
            }
        }
    }
    shards.sort(); // deterministic ordering
    Ok(shards)
}

// ---------------------------------------------------------------------------
// FFI entry points
// ---------------------------------------------------------------------------

/// Assign shards for a rank. Returns a packed i64 array: [num_shards, shard_0, shard_1, ...].
#[no_mangle]
pub extern "C" fn nsl_data_assign_shards(
    rank: i64,
    world_size: i64,
    num_shards: i64,
    epoch: i64,
) -> i64 {
    let shards = assign_shards(rank as u32, world_size as u32, num_shards as u32, epoch as u64);
    let n = shards.len();
    let buf = crate::memory::checked_alloc((n + 1) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *buf = n as i64;
        for (i, &s) in shards.iter().enumerate() {
            *buf.add(1 + i) = s as i64;
        }
    }
    buf as i64
}

/// Free the buffer returned by nsl_data_assign_shards.
/// `ptr`: pointer returned by nsl_data_assign_shards.
#[no_mangle]
pub extern "C" fn nsl_data_free_shards(ptr: i64) -> i64 {
    if ptr == 0 { return -1; }
    let buf = ptr as *mut i64;
    let n = unsafe { *buf } as usize;
    unsafe {
        crate::memory::checked_free(
            buf as *mut u8,
            (n + 1) * std::mem::size_of::<i64>(),
        );
    }
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_shard_write_read_roundtrip() {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = ShardWriter::new(&mut buf).unwrap();
            writer.write_sample(b"hello world").unwrap();
            writer.write_sample(b"second sample").unwrap();
            writer.write_sample(b"").unwrap(); // empty sample
            writer.finalize().unwrap();
        }

        buf.seek(SeekFrom::Start(0)).unwrap();
        let mut reader = ShardReader::open(&mut buf).unwrap();
        assert_eq!(reader.num_samples(), 3);

        assert_eq!(reader.read_sample(0).unwrap(), b"hello world");
        assert_eq!(reader.read_sample(1).unwrap(), b"second sample");
        assert_eq!(reader.read_sample(2).unwrap(), b"");
    }

    #[test]
    fn test_shard_sequential_iteration() {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = ShardWriter::new(&mut buf).unwrap();
            for i in 0..5 {
                writer.write_sample(format!("sample_{i}").as_bytes()).unwrap();
            }
            writer.finalize().unwrap();
        }

        buf.seek(SeekFrom::Start(0)).unwrap();
        let mut reader = ShardReader::open(&mut buf).unwrap();
        let samples: Vec<_> = reader.iter_samples().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0], b"sample_0");
        assert_eq!(samples[4], b"sample_4");
    }

    #[test]
    fn test_shard_header_magic_validation() {
        let bad_data = vec![0u8; 24]; // zero magic
        let mut cursor = Cursor::new(bad_data);
        let result = ShardHeader::read_from(&mut cursor);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid shard magic"));
    }

    #[test]
    fn test_assign_shards_deterministic() {
        let a1 = assign_shards(0, 4, 100, 42);
        let a2 = assign_shards(0, 4, 100, 42);
        assert_eq!(a1, a2, "same inputs must produce same assignment");
    }

    #[test]
    fn test_assign_shards_no_overlap() {
        let world_size = 4;
        let num_shards = 20;
        let epoch = 7;

        let mut all_shards: Vec<u32> = Vec::new();
        for rank in 0..world_size {
            let assigned = assign_shards(rank, world_size, num_shards, epoch);
            all_shards.extend(assigned);
        }

        // Every shard should appear exactly once
        all_shards.sort();
        let expected: Vec<u32> = (0..num_shards).collect();
        assert_eq!(all_shards, expected, "all shards must be assigned exactly once");
    }

    #[test]
    fn test_assign_shards_different_epochs() {
        let a1 = assign_shards(0, 2, 10, 0);
        let a2 = assign_shards(0, 2, 10, 1);
        assert_ne!(a1, a2, "different epochs should produce different orderings");
    }

    #[test]
    fn test_assign_shards_single_rank() {
        let assigned = assign_shards(0, 1, 5, 0);
        assert_eq!(assigned.len(), 5, "single rank gets all shards");
    }

    #[test]
    fn test_assign_shards_empty() {
        assert_eq!(assign_shards(0, 1, 0, 0), Vec::<u32>::new());
        assert_eq!(assign_shards(0, 0, 10, 0), Vec::<u32>::new());
    }

    #[test]
    fn test_data_checkpoint_roundtrip() {
        let ckpt = DataCheckpoint {
            shard_id: 42,
            sample_offset: 1000,
            epoch: 5,
            rng_state: [0xAB; 32],
        };
        let bytes = ckpt.to_bytes();
        let restored = DataCheckpoint::from_bytes(&bytes).unwrap();
        assert_eq!(restored.shard_id, 42);
        assert_eq!(restored.sample_offset, 1000);
        assert_eq!(restored.epoch, 5);
        assert_eq!(restored.rng_state, [0xAB; 32]);
    }

    #[test]
    fn test_large_shard_many_samples() {
        let mut buf = Cursor::new(Vec::new());
        let n = 1000;
        {
            let mut writer = ShardWriter::new(&mut buf).unwrap();
            for i in 0..n {
                let sample = format!("sample_{i:05}");
                writer.write_sample(sample.as_bytes()).unwrap();
            }
            writer.finalize().unwrap();
        }

        buf.seek(SeekFrom::Start(0)).unwrap();
        let mut reader = ShardReader::open(&mut buf).unwrap();
        assert_eq!(reader.num_samples(), n);

        // Random access
        let s500 = reader.read_sample(500).unwrap();
        assert_eq!(s500, b"sample_00500");
        let s999 = reader.read_sample(999).unwrap();
        assert_eq!(s999, b"sample_00999");
    }
}
