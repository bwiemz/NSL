//! Standalone weight provider: embedded or memory-mapped sidecar .nslweights file.
//!
//! Initialized once per process via `nsl_standalone_init_embedded` or
//! `nsl_standalone_init_sidecar`. After init, `nsl_model_load` checks here first
//! and skips file I/O entirely when a provider is active.

use std::collections::HashMap;
use std::sync::OnceLock;

// NSLW file format constants
const NSLW_MAGIC: &[u8; 4] = b"NSLW";
const NSLW_VERSION: u32 = 1;

// ── Data structures ──────────────────────────────────────────────────────────

/// Metadata for a single tensor stored in a .nslweights blob.
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
    /// Byte offset into the raw data region (already adjusted to be absolute
    /// within the blob for mmap mode, relative to data_start for embedded mode).
    pub offset: usize,
    pub nbytes: usize,
}

/// The underlying byte source — either a static slice baked into the binary, or
/// a memory-mapped file.
enum WeightData {
    Embedded(&'static [u8]),
    Mmap(memmap2::Mmap),
}

impl WeightData {
    fn as_slice(&self) -> &[u8] {
        match self {
            WeightData::Embedded(s) => s,
            WeightData::Mmap(m) => m.as_ref(),
        }
    }
}

// SAFETY: the embedded slice is &'static, and Mmap is effectively read-only
// shared memory — both are safe to share across threads.
unsafe impl Send for WeightData {}
unsafe impl Sync for WeightData {}

/// Holds the parsed header index and the byte source.
pub struct WeightProvider {
    data: WeightData,
    /// Tensor name → metadata (offsets are absolute within `data`).
    pub index: HashMap<String, TensorMeta>,
    /// Ordered tensor names (same order as the JSON header).
    pub ordered_names: Vec<String>,
}

// ── Global slot ──────────────────────────────────────────────────────────────

static WEIGHT_PROVIDER: OnceLock<WeightProvider> = OnceLock::new();

// ── Header parsing ───────────────────────────────────────────────────────────

/// Parse a .nslweights blob header.
///
/// Layout:
///   [0..4]  magic "NSLW"
///   [4..8]  version u32 LE
///   [8..16] header_size u64 LE
///   [16..]  JSON header (header_size bytes)
///   <pad to 64-byte alignment>
///   <tensor data>
///
/// Returns `(metas, data_start_offset)` where `data_start_offset` is the byte
/// index at which the raw tensor data begins inside `raw`.
fn parse_nslweights_header(raw: &[u8]) -> (Vec<TensorMeta>, usize) {
    if raw.len() < 16 {
        eprintln!(
            "nsl: weight_provider: blob too small ({} bytes, need at least 16)",
            raw.len()
        );
        std::process::abort();
    }
    if &raw[0..4] != NSLW_MAGIC {
        eprintln!("nsl: weight_provider: invalid .nslweights blob (bad magic)");
        std::process::abort();
    }
    let version = u32::from_le_bytes(raw[4..8].try_into().unwrap_or_else(|_| std::process::abort()));
    if version != NSLW_VERSION {
        eprintln!(
            "nsl: weight_provider: unsupported version {} (expected {})",
            version, NSLW_VERSION
        );
        std::process::abort();
    }
    let header_size = u64::from_le_bytes(raw[8..16].try_into().unwrap_or_else(|_| std::process::abort())) as usize;

    let header_end = 16 + header_size;
    if raw.len() < header_end {
        eprintln!(
            "nsl: weight_provider: blob truncated (need {} bytes for header, have {})",
            header_end,
            raw.len()
        );
        std::process::abort();
    }

    // Parse JSON
    let json_bytes = &raw[16..header_end];
    let json_str = std::str::from_utf8(json_bytes).unwrap_or_else(|_| {
        eprintln!("nsl: weight_provider: header is not valid UTF-8");
        std::process::abort();
    });
    let root: serde_json::Value = serde_json::from_str(json_str).unwrap_or_else(|e| {
        eprintln!("nsl: weight_provider: header JSON parse error: {}", e);
        std::process::abort();
    });

    let params = root["params"].as_array().unwrap_or_else(|| {
        eprintln!("nsl: weight_provider: header missing 'params' array");
        std::process::abort();
    });

    let metas: Vec<TensorMeta> = params
        .iter()
        .map(|p| {
            let name = p["name"].as_str().unwrap_or("?").to_string();
            let dtype = p["dtype"].as_str().unwrap_or("f64").to_string();
            let offset = p["offset"].as_u64().unwrap_or_else(|| {
                eprintln!("nsl: weight_provider: tensor '{}' missing 'offset'", name);
                std::process::abort();
            }) as usize;
            let nbytes = p["nbytes"].as_u64().unwrap_or_else(|| {
                eprintln!("nsl: weight_provider: tensor '{}' missing 'nbytes'", name);
                std::process::abort();
            }) as usize;
            let shape: Vec<i64> = p["shape"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_i64().unwrap_or(0))
                .collect();
            TensorMeta { name, shape, dtype, offset, nbytes }
        })
        .collect();

    // data_start is right after the JSON header, padded to 64-byte alignment
    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;

    (metas, data_start)
}

// ── FFI init functions ───────────────────────────────────────────────────────

/// Initialize the weight provider from an embedded byte slice.
///
/// `data_ptr` points to `.rodata` (lives for the process lifetime).
/// `data_len` is the total byte length of the .nslweights blob.
#[no_mangle]
pub extern "C" fn nsl_standalone_init_embedded(data_ptr: i64, data_len: i64) {
    let raw: &'static [u8] =
        unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len as usize) };

    let (mut metas, data_start) = parse_nslweights_header(raw);

    // Adjust offsets: they are relative to the tensor-data region, so add
    // data_start so they become absolute within `raw`.
    for m in &mut metas {
        m.offset += data_start;
    }

    // The data slice is the full embedded blob (offsets are now absolute into it).
    let provider = build_provider(WeightData::Embedded(raw), metas);
    let _ = WEIGHT_PROVIDER.set(provider); // ignore if already set
}

/// Initialize the weight provider from a .nslweights sidecar file.
///
/// Search order:
///   1. Same directory as the compiled binary path given by the caller
///   2. Next to the current executable (`std::env::current_exe`)
///   3. `$NSL_WEIGHTS_PATH` environment variable
#[no_mangle]
pub extern "C" fn nsl_standalone_init_sidecar(compiled_path_ptr: i64, compiled_path_len: i64) {
    let compiled_path_str = unsafe {
        let slice = std::slice::from_raw_parts(
            compiled_path_ptr as *const u8,
            compiled_path_len as usize,
        );
        std::str::from_utf8(slice).unwrap_or("")
    };

    // Build candidate paths
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // 1. Same directory as the compiled binary
    if !compiled_path_str.is_empty() {
        let p = std::path::Path::new(compiled_path_str);
        let stem = p.file_stem().map(|s| {
            let mut pb = p.with_file_name(s);
            pb.set_extension("nslweights");
            pb
        });
        if let Some(path) = stem {
            candidates.push(path);
        }
    }

    // 2. Next to current executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // Try matching the compiled binary stem first, else the exe name
            let stem = if !compiled_path_str.is_empty() {
                std::path::Path::new(compiled_path_str)
                    .file_stem()
                    .map(|s| s.to_owned())
            } else {
                exe.file_stem().map(|s| s.to_owned())
            };
            if let Some(stem) = stem {
                let mut pb = dir.join(&stem);
                pb.set_extension("nslweights");
                candidates.push(pb);
            }
        }
    }

    // 3. $NSL_WEIGHTS_PATH env var
    if let Ok(env_path) = std::env::var("NSL_WEIGHTS_PATH") {
        candidates.push(std::path::PathBuf::from(env_path));
    }

    // Try each candidate
    let file = candidates.iter().find_map(|p| std::fs::File::open(p).ok());
    let file = match file {
        Some(f) => f,
        None => {
            eprintln!(
                "nsl: weight_provider: could not find .nslweights sidecar (searched: {:?})",
                candidates
            );
            std::process::abort();
        }
    };

    let mmap = unsafe {
        memmap2::Mmap::map(&file).unwrap_or_else(|e| {
            eprintln!("nsl: weight_provider: mmap failed: {}", e);
            std::process::abort();
        })
    };

    let (mut metas, data_start) = parse_nslweights_header(mmap.as_ref());

    // Adjust offsets to be absolute within the mmap'd buffer.
    for m in &mut metas {
        m.offset += data_start;
    }

    let provider = build_provider(WeightData::Mmap(mmap), metas);
    let _ = WEIGHT_PROVIDER.set(provider);
}

/// Returns 1 if the weight provider has been initialized, 0 otherwise.
#[no_mangle]
pub extern "C" fn nsl_standalone_has_weights() -> i64 {
    if WEIGHT_PROVIDER.get().is_some() { 1 } else { 0 }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn build_provider(data: WeightData, metas: Vec<TensorMeta>) -> WeightProvider {
    let mut index = HashMap::with_capacity(metas.len());
    let mut ordered_names = Vec::with_capacity(metas.len());
    for m in metas {
        ordered_names.push(m.name.clone());
        index.insert(m.name.clone(), m);
    }
    WeightProvider { data, index, ordered_names }
}

// ── Integration with nsl_model_load ─────────────────────────────────────────

/// If a weight provider is active, copy tensor data into the tensors in `list`
/// (by index order, matching `ordered_names`) and return `true`.
/// Returns `false` if no provider is set.
pub fn try_load_from_provider(tensors: &crate::list::NslList) -> bool {
    let provider = match WEIGHT_PROVIDER.get() {
        Some(p) => p,
        None => return false,
    };

    let raw = provider.data.as_slice();

    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = crate::tensor::NslTensor::from_ptr(tensor_ptr);
        let byte_count = (tensor.len as usize) * tensor.element_size();

        // Look up by index order
        let meta = provider.ordered_names.get(i).and_then(|n| provider.index.get(n));
        let meta = match meta {
            Some(m) => m,
            None => {
                eprintln!(
                    "nsl: weight_provider: no weight entry for tensor index {}",
                    i
                );
                std::process::abort();
            }
        };

        if meta.nbytes < byte_count {
            eprintln!(
                "nsl: weight_provider: tensor '{}' stored size {} < required {} bytes",
                meta.name, meta.nbytes, byte_count
            );
            std::process::abort();
        }

        let src_offset = meta.offset;
        if src_offset + byte_count > raw.len() {
            eprintln!(
                "nsl: weight_provider: tensor '{}' offset {} + {} exceeds blob size {}",
                meta.name, src_offset, byte_count, raw.len()
            );
            std::process::abort();
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                raw[src_offset..].as_ptr(),
                tensor.data as *mut u8,
                byte_count,
            );
        }
    }

    true
}
