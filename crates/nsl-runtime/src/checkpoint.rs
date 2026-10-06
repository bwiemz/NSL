use crate::list::NslList;
use crate::tensor::{
    DTYPE_BF16, DTYPE_CUSTOM_START, DTYPE_F32, DTYPE_F64, DTYPE_FP16, DTYPE_FP8E4M3,
    DTYPE_FP8E5M2, DTYPE_I32, DTYPE_INT8, DTYPE_INT8_BLOCKWISE, DTYPE_U16_SEGMENT,
    DTYPE_U16_TOKEN, NslTensor,
};
use std::io::Write;

const MAGIC: &[u8; 4] = b"NSLM";
const VERSION: u32 = 1;

/// Write helper: aborts on I/O error instead of panicking across extern "C".
fn write_or_abort(file: &mut std::fs::File, buf: &[u8], context: &str) {
    if let Err(e) = file.write_all(buf) {
        crate::nsl_log!(ERROR, "nsl", "nsl: model_save: {}: {}", context, e);
        std::process::abort();
    }
}

/// The `dtype` string an `.nslm` header records for a tensor tag.
///
/// `model_load` compares these strings for equality, so every tag needs its
/// own name. Until C5 step 2a every tag other than f32 was written as "f64":
/// an fp16 checkpoint then loaded into a bf16 model without complaint and its
/// bytes were reinterpreted. "f32" and "f64" keep their spelling so files
/// saved before the change still load into f32/f64 models. The dtype-refusal
/// messages (`fatal::mixed_dtypes`) name tags with it too.
pub(crate) fn checkpoint_dtype_name(dtype: u16) -> String {
    let name = match dtype {
        DTYPE_F64 => "f64",
        DTYPE_F32 => "f32",
        DTYPE_FP16 => "fp16",
        DTYPE_BF16 => "bf16",
        DTYPE_INT8 => "int8",
        DTYPE_FP8E4M3 => "fp8e4m3",
        DTYPE_FP8E5M2 => "fp8e5m2",
        DTYPE_U16_TOKEN => "u16_token",
        DTYPE_U16_SEGMENT => "u16_segment",
        DTYPE_I32 => "i32",
        DTYPE_INT8_BLOCKWISE => "int8_blockwise",
        id if id >= DTYPE_CUSTOM_START => return format!("custom{id}"),
        id => return format!("tag{id}"),
    };
    name.to_string()
}

/// Save model parameters to .nslm binary format.
/// path_ptr/path_len: string pointer and length for file path
/// param_names_ptr: NslList of string pointers
/// param_tensors_ptr: NslList of tensor pointers
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_save(
    path_ptr: i64,
    path_len: i64,
    param_names_ptr: i64,
    param_tensors_ptr: i64,
) {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    let names = NslList::from_ptr(param_names_ptr);
    let tensors = NslList::from_ptr(param_tensors_ptr);
    if names.len != tensors.len {
        crate::nsl_log!(ERROR, "nsl", "nsl: model_save: name/tensor count mismatch ({} names, {} tensors)",
            names.len, tensors.len
        );
        std::process::abort();
    }

    // Build JSON header
    let mut params_json = Vec::new();
    let mut data_offset: u64 = 0;
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        check_tensor_contiguous(tensor, i);
        let elem_size = tensor.element_size();
        let nbytes = (tensor.len as u64) * (elem_size as u64);
        let shape: Vec<i64> = (0..tensor.ndim as usize)
            .map(|d| unsafe { *tensor.shape.add(d) })
            .collect();
        let name_ptr = unsafe { *names.data.add(i) };
        let name = unsafe {
            std::ffi::CStr::from_ptr(name_ptr as *const std::os::raw::c_char)
        }.to_str().unwrap_or("?");
        let dtype_str = checkpoint_dtype_name(tensor.dtype);
        params_json.push(format!(
            r#"{{"name":"{}","shape":{:?},"dtype":"{}","offset":{},"nbytes":{}}}"#,
            name, shape, dtype_str, data_offset, nbytes
        ));
        data_offset += nbytes;
    }
    let header = format!(r#"{{"params":[{}]}}"#, params_json.join(","));
    let header_bytes = header.as_bytes();

    // Written to a temporary and committed by rename, so an interrupted save
    // leaves the previous file intact: `File::create` on the final path used
    // to truncate it before the first byte was written (external review
    // 2026-10-06). The temporary is unique to this process and call: two
    // processes saving the same path (a test running one fixture twice in
    // parallel) would otherwise share it, and one's rename would take the
    // other's file away mid-commit. A crash leaves the temporary behind.
    static SAVE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SAVE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = format!("{path}.tmp.{}.{seq}", std::process::id());
    let mut file = match std::fs::File::create(&tmp) {
        Ok(f) => f,
        Err(e) => {
            crate::nsl_log!(ERROR, "nsl", "nsl: model_save: cannot create file '{}': {}", tmp, e);
            std::process::abort();
        }
    };
    write_or_abort(&mut file, MAGIC, "write magic");
    write_or_abort(&mut file, &VERSION.to_le_bytes(), "write version");
    write_or_abort(
        &mut file,
        &(header_bytes.len() as u64).to_le_bytes(),
        "write header size",
    );
    write_or_abort(&mut file, header_bytes, "write header");

    // Pad to 64-byte alignment
    let total_header = 4 + 4 + 8 + header_bytes.len();
    let padding = (64 - (total_header % 64)) % 64;
    let pad_buf = [0u8; 64];
    write_or_abort(&mut file, &pad_buf[..padding], "write padding");

    // Item 12: a mid-loop `model_save` under `--weight-stream` sees streamed
    // params EVICTED — `t.data == null` while `t.device` stays set, so the
    // GPU staging branch below would `nsl_tensor_to_device` a null source and
    // crash loudly (#395's documented hazard). Materialize each
    // evicted-but-registered param from its pinned host mirror for the
    // duration of the serialization read, then restore the evicted state.
    // This makes `model_save` safe wherever it is called — a callback, mid
    // training loop, or teardown — without the caller forcing residency
    // first. The header loop above only reads intact metadata (shape / len /
    // dtype), so residency is needed for the DATA loop alone.
    let mut materialized: Vec<i64> = Vec::new();
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if tensor.data.is_null()
            && crate::weight_stream::nsl_weight_stream_is_registered(tensor_ptr) != 0
        {
            crate::weight_stream::nsl_weight_stream_upload(tensor_ptr);
            materialized.push(tensor_ptr);
        }
    }

    // Raw tensor data (little-endian, dtype-aware).
    // GPU tensors are transferred to CPU before reading data.
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        let byte_count = (tensor.len as usize) * tensor.element_size();

        if tensor.device > 0 {
            // GPU tensor: copy to CPU staging buffer before writing.
            //
            // The header above was built from the GPU tensor, so the staged
            // bytes must carry the same tag. The download preserves it (C5
            // step 2a). Before that it widened f32 to f64, and this loop had to
            // narrow the staging buffer back: writing it raw once serialized
            // interleaved f64 halves as "f32" data (the roadmap-4.3
            // FASE-parity gate saw |max| ~ 3.7e19 in freshly-initialized
            // weights). The mismatch arm stays as a guard, not a fallback.
            let cpu_ptr = crate::tensor::nsl_tensor_to_device(tensor_ptr, 0);
            let cpu_tensor = NslTensor::from_ptr(cpu_ptr);
            if cpu_tensor.dtype == tensor.dtype {
                let data_slice = unsafe {
                    std::slice::from_raw_parts(cpu_tensor.data as *const u8, byte_count)
                };
                write_or_abort(&mut file, data_slice, "write tensor data (GPU->CPU)");
            } else {
                crate::nsl_log!(ERROR, "nsl", "nsl: model_save: unsupported dtype transition in GPU staging \
                     (device dtype {} -> staged dtype {}) for tensor #{}",
                    tensor.dtype, cpu_tensor.dtype, i
                );
                std::process::abort();
            }
            crate::tensor::nsl_tensor_free(cpu_ptr);
        } else {
            let data_slice = unsafe {
                std::slice::from_raw_parts(tensor.data as *const u8, byte_count)
            };
            write_or_abort(&mut file, data_slice, "write tensor data");
        }
    }

    // Restore the streamed (evicted) state for every param materialized
    // above — read-only, so no writeback (model_save never mutates θ). If we
    // materialized nothing (no streaming, or all params already resident)
    // this is an empty loop.
    for &ptr in &materialized {
        crate::weight_stream::nsl_weight_stream_evict(ptr, 0);
    }
    sync_or_abort(&file, &tmp);
    drop(file);
    commit_rename(&tmp, path, "model_save");
}

/// fsync `file`: its bytes are on the device before a rename publishes it.
/// Without this a crash can leave a renamed but empty or partial file -- the
/// hazard `awq.rs::write_atomic` documents.
fn sync_or_abort(file: &std::fs::File, path: &str) {
    if let Err(e) = file.sync_all() {
        crate::nsl_log!(ERROR, "nsl", "nsl: checkpoint: fsync '{path}': {e}");
        std::process::abort();
    }
}

/// Atomically replace `to` with `from`, then fsync the directory so the
/// rename itself survives a crash.
fn commit_rename(from: &str, to: &str, what: &str) {
    if let Err(e) = std::fs::rename(from, to) {
        crate::nsl_log!(ERROR, "nsl", "nsl: {what}: rename '{from}' -> '{to}': {e}");
        std::process::abort();
    }
    sync_parent_dir(to);
}

/// fsync the directory holding `path` (a rename is a directory update).
///
/// A failure aborts: the train save relies on the model's rename being
/// durable before the sidecar's, and losing that order silently could leave
/// a new sidecar beside an old model after a power cut. A filesystem that
/// does not support syncing a directory (EINVAL / ENOTSUP) is tolerated, and
/// Windows, which cannot open a directory as a file, skips it.
fn sync_parent_dir(path: &str) {
    let dir = std::path::Path::new(path)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    #[cfg(unix)]
    {
        use std::io::ErrorKind;
        let r = std::fs::File::open(dir).and_then(|d| d.sync_all());
        if let Err(e) = r
            && !matches!(e.kind(), ErrorKind::InvalidInput | ErrorKind::Unsupported)
        {
            crate::nsl_log!(ERROR, "nsl", "nsl: checkpoint: fsync of directory '{}': {e}", dir.display());
            std::process::abort();
        }
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// SHA-256 of a whole file, as lowercase hex: the `.optim` sidecar's
/// `model_sha256`, which ties it to the exact `.nslm` it was saved with.
pub(crate) fn file_sha256_hex(path: &str) -> std::io::Result<String> {
    use sha2::Digest;
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// One `params` entry of an `.nslm` header.
#[derive(Debug, serde::Deserialize)]
struct NslmEntry {
    name: String,
    shape: Vec<i64>,
    dtype: String,
    offset: u64,
    nbytes: u64,
}

#[derive(serde::Deserialize)]
struct NslmHeader {
    params: Vec<NslmEntry>,
}

/// A parsed `.nslm` file: its entries, and where the data section starts.
#[derive(Debug)]
struct NslmLayout {
    entries: Vec<NslmEntry>,
    data_start: usize,
}

/// Parse and structurally validate an `.nslm` file before anything reads its
/// data: the declared header fits the file, the header is the JSON table the
/// writer emits, and the entries tile the data section exactly, in order,
/// with no gap, overlap or trailing bytes (external review 2026-10-06; the
/// loader used to slice an unbounded `header_size`, count entries by
/// substring, and ignore offsets).
fn parse_nslm(data: &[u8]) -> Result<NslmLayout, String> {
    if data.len() < 16 {
        return Err(format!("file too small ({} bytes, need at least 16)", data.len()));
    }
    if &data[0..4] != MAGIC {
        return Err("invalid .nslm file (bad magic)".into());
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != VERSION {
        return Err(format!("unsupported version {version} (expected {VERSION})"));
    }
    let mut size = [0u8; 8];
    size.copy_from_slice(&data[8..16]);
    let header_size = u64::from_le_bytes(size);
    let header_end = 16u64
        .checked_add(header_size)
        .filter(|&end| end <= data.len() as u64)
        .ok_or_else(|| {
            format!("the header claims {header_size} bytes but the file is {} bytes", data.len())
        })? as usize;
    let header: NslmHeader = serde_json::from_slice(&data[16..header_end])
        .map_err(|e| format!("the header is not a valid parameter table: {e}"))?;
    let data_start = header_end + (64 - header_end % 64) % 64;
    let mut next = 0u64;
    for (i, e) in header.params.iter().enumerate() {
        if e.offset != next {
            return Err(format!(
                "entry #{i} '{}' starts at data offset {} but the previous entry ends at {next}",
                e.name, e.offset
            ));
        }
        next = next
            .checked_add(e.nbytes)
            .ok_or_else(|| format!("entry #{i} '{}' has an impossible size {}", e.name, e.nbytes))?;
    }
    let have = (data.len() as u64).checked_sub(data_start as u64);
    if have != Some(next) {
        return Err(format!(
            "the header describes {next} data bytes but the file holds {} after the header",
            have.map_or_else(|| "none".to_string(), |h| h.to_string())
        ));
    }
    Ok(NslmLayout { entries: header.params, data_start })
}

/// `blocks.0.attn.wq` and `blocks[0].attn.wq` are the same path: the train
/// block names checkpoint entries with dotted indices, `model_save` with
/// brackets.
fn canonical_param_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, seg) in name.split('.').enumerate() {
        if i > 0 && !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()) {
            out.push('[');
            out.push_str(seg);
            out.push(']');
        } else {
            if i > 0 {
                out.push('.');
            }
            out.push_str(seg);
        }
    }
    out
}

/// Whether a file entry names the live parameter at the same position. A
/// train-block checkpoint prefixes the model VARIABLE (`m.blocks.0.w`) where
/// `model_save` writes the field path (`blocks[0].w`), and
/// `model_load` of a checkpoint is the documented weights-only warm start,
/// so that one leading segment is allowed to differ.
fn param_names_match(file: &str, live: &str) -> bool {
    let (file, live) = (canonical_param_name(file), canonical_param_name(live));
    file == live || file.split_once('.').is_some_and(|(_, rest)| rest == live)
}

fn live_param_name(names: &NslList, i: usize) -> String {
    let ptr = unsafe { *names.data.add(i) };
    if ptr == 0 {
        return "?".into();
    }
    unsafe { std::ffi::CStr::from_ptr(ptr as *const std::os::raw::c_char) }
        .to_string_lossy()
        .into_owned()
}

fn live_shape(tensor: &NslTensor) -> Vec<i64> {
    (0..tensor.ndim as usize).map(|d| unsafe { *tensor.shape.add(d) }).collect()
}

/// Check every entry against the live parameter at its position — count,
/// name (when the caller has names), dtype, shape and byte size — so a
/// refused load leaves the model untouched. The copy is positional; this is
/// what makes a reordered, resized or transposed parameter a refusal instead
/// of bytes read from the wrong place.
fn check_nslm_against_live(
    layout: &NslmLayout,
    tensors: &NslList,
    names: Option<&NslList>,
) -> Result<(), String> {
    let live = tensors.len as usize;
    if layout.entries.len() != live {
        return Err(format!(
            "the file has {} tensors and the model has {live}. A train-block checkpoint holds \
             only the trained parameters; tools/nslm_splice.py merges one into a full model \
             file by name.",
            layout.entries.len()
        ));
    }
    if let Some(n) = names
        && n.len as usize != live
    {
        return Err(format!("{} names for {live} tensors", n.len));
    }
    for (i, e) in layout.entries.iter().enumerate() {
        let tensor = NslTensor::from_ptr(unsafe { *tensors.data.add(i) });
        if let Some(n) = names {
            let live_name = live_param_name(n, i);
            if !param_names_match(&e.name, &live_name) {
                return Err(format!(
                    "tensor #{i} is '{}' in the file but '{live_name}' in the model -- the \
                     parameters were added, removed or reordered since the save",
                    e.name
                ));
            }
        }
        let live_dtype = checkpoint_dtype_name(tensor.dtype);
        if e.dtype != live_dtype {
            return Err(format!(
                "dtype mismatch for tensor #{i}: file has {}, model expects {live_dtype} \
                 (entry '{}') -- a raw byte copy would corrupt it. Convert the model's \
                 parameters with `.to(dtype)` or re-save the checkpoint from a model with \
                 the same dtypes.",
                e.dtype, e.name
            ));
        }
        let shape = live_shape(tensor);
        let bytes = (tensor.len as u64) * (tensor.element_size() as u64);
        if e.shape != shape || e.nbytes != bytes {
            return Err(format!(
                "tensor #{i} '{}' is {:?} ({} bytes) in the file but {shape:?} ({bytes} bytes) \
                 in the model",
                e.name, e.shape, e.nbytes
            ));
        }
    }
    Ok(())
}

/// Read and fully validate `path` for loading into `tensors`. Nothing is
/// mutated; an `Err` names what is wrong.
fn read_validated_nslm(
    path: &str,
    tensors: &NslList,
    names: Option<&NslList>,
) -> Result<(Vec<u8>, NslmLayout), String> {
    let data = std::fs::read(path).map_err(|e| format!("cannot read file '{path}': {e}"))?;
    let layout = parse_nslm(&data).map_err(|e| format!("'{path}': {e}"))?;
    check_nslm_against_live(&layout, tensors, names).map_err(|e| format!("'{path}': {e}"))?;
    Ok((data, layout))
}

/// Copy a validated file's tensors into the model. Every check has run
/// already ([`read_validated_nslm`]).
fn copy_nslm_into(data: &[u8], layout: &NslmLayout, tensors: &NslList) {
    for (i, e) in layout.entries.iter().enumerate() {
        let tensor = NslTensor::from_ptr(unsafe { *tensors.data.add(i) });
        let start = layout.data_start + e.offset as usize;
        let src = &data[start..start + e.nbytes as usize];
        if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::inner::memcpy_htod(
                    tensor.data,
                    src.as_ptr() as *const std::ffi::c_void,
                    src.len(),
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                crate::nsl_log!(ERROR, "nsl", "nsl: checkpoint load: tensor {i} is on GPU but CUDA not compiled");
                std::process::abort();
            }
        } else {
            // `model_save` writes the tensors' bytes as they are in memory,
            // so the copy back is byte for byte.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), tensor.data as *mut u8, src.len());
            }
        }
    }
}

fn model_load_impl(path: &str, tensors: &NslList, names: Option<&NslList>) {
    match read_validated_nslm(path, tensors, names) {
        Ok((data, layout)) => copy_nslm_into(&data, &layout, tensors),
        Err(e) => {
            crate::nsl_log!(ERROR, "nsl", "nsl: model_load: {e}");
            std::process::abort();
        }
    }
}

/// Load model parameters from .nslm binary format into existing tensors,
/// positionally. Checks count, dtype, shape and size of every entry before
/// copying any; without names it cannot see a reorder of same-shaped
/// parameters ([`nsl_model_load_named`] can, and is what `model_load` emits).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_load(path_ptr: i64, path_len: i64, param_tensors_ptr: i64) {
    let tensors = NslList::from_ptr(param_tensors_ptr);
    if crate::weight_provider::try_load_from_provider(tensors) {
        return;
    }
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    model_load_impl(path, tensors, None);
}

/// [`nsl_model_load`] that also requires each file entry to name the live
/// parameter at its position (`param_names_ptr`: NslList of C strings, the
/// same list `model_save` writes).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_load_named(
    path_ptr: i64,
    path_len: i64,
    param_names_ptr: i64,
    param_tensors_ptr: i64,
) {
    let tensors = NslList::from_ptr(param_tensors_ptr);
    if crate::weight_provider::try_load_from_provider(tensors) {
        return;
    }
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    model_load_impl(path, tensors, Some(NslList::from_ptr(param_names_ptr)));
}

// ---------------------------------------------------------------------------
// Milestone B: full training-state checkpoint (θ + optimizer moments + step)
// ---------------------------------------------------------------------------

const OPTIM_MAGIC: &[u8; 4] = b"NSLO";
/// v1: θ + m/v + micro-batch step counter.
/// v2 (item 8): adds the `resume` header block — training/loader epoch, the
/// loader's delivery slot, a corpus+geometry fingerprint, and every RNG
/// stream's state. v1 sidecars still load (see [`nsl_train_checkpoint_load`]).
const OPTIM_VERSION: u32 = 2;

/// Item 8: the training epoch restored by the last successful
/// [`nsl_train_checkpoint_load`], published to codegen through
/// [`nsl_train_resume_epoch`].
///
/// A return value rather than a global would be cleaner, but Cranelift call
/// lowering here returns a single i64 and the step counter already owns it;
/// threading an out-pointer through the train-block emitter for one scalar is
/// more moving parts than a value written and read on the same thread,
/// microseconds apart, in the train block's own prologue.
static RESUME_TRAIN_EPOCH: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(0);

/// The training epoch to start the epoch loop at — 0 unless a v2 checkpoint
/// was just loaded.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_train_resume_epoch() -> i64 {
    RESUME_TRAIN_EPOCH.load(std::sync::atomic::Ordering::SeqCst)
}

/// A cheap signature tying the `.optim` sidecar to the exact `.nslm` it was
/// saved with: FNV-1a over (file size, first MiB, last MiB). θ changes every
/// optimizer step, so a stale pair — model@N next to moments@N−k after a
/// crash between the two renames — mismatches with overwhelming probability
/// and the loader can refuse instead of silently resuming mixed state. Not
/// a cryptographic integrity check; a pairing check.
fn model_file_sig(path: &str) -> u64 {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            crate::nsl_log!(ERROR, "nsl", "nsl: checkpoint: cannot open '{path}' for signature: {e}");
            std::process::abort();
        }
    };
    let size = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    mix(&size.to_le_bytes());
    const CHUNK: u64 = 1 << 20;
    let mut buf = vec![0u8; CHUNK as usize];
    let n = f.read(&mut buf).unwrap_or(0);
    mix(&buf[..n]);
    if size > CHUNK {
        let _ = f.seek(SeekFrom::Start(size.saturating_sub(CHUNK)));
        let n = f.read(&mut buf).unwrap_or(0);
        mix(&buf[..n]);
    }
    h
}

/// Whether a sidecar header names `model_path` as the model it was saved with.
pub(crate) enum Pairing {
    Paired,
    Mismatch(String),
    /// Neither `model_sha256` nor `model_sig`: not a sidecar this runtime wrote.
    NoRecord,
}

/// The pairing check: the whole-file SHA-256 when the sidecar has one
/// (`model_sha256`, written since 2026-10-06), else the legacy sampled
/// `model_sig`, which cannot see a same-size change in the middle of θ.
pub(crate) fn sidecar_pairs_with(header: &[u8], model_path: &str) -> Pairing {
    if let Some(saved) = scan_header_string(header, b"\"model_sha256\":") {
        let saved = String::from_utf8_lossy(&saved).into_owned();
        return match file_sha256_hex(model_path) {
            Ok(live) if live == saved => Pairing::Paired,
            Ok(live) => Pairing::Mismatch(format!("model sha256 {live} vs sidecar {saved}")),
            Err(e) => Pairing::Mismatch(format!("cannot hash the model: {e}")),
        };
    }
    match scan_header_numbers(header, b"\"model_sig\":").first() {
        Some(&saved) => {
            let live = model_file_sig(model_path);
            if live == saved {
                Pairing::Paired
            } else {
                Pairing::Mismatch(format!("model_sig {live} vs sidecar {saved}"))
            }
        }
        None => Pairing::NoRecord,
    }
}

/// The JSON header of a sidecar file, if `path` is a readable NSLO sidecar
/// whose declared header fits the file.
fn read_sidecar_header(path: &str) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut fixed = [0u8; 16];
    f.read_exact(&mut fixed).ok()?;
    if &fixed[..4] != OPTIM_MAGIC {
        return None;
    }
    let header_size = u64::from_le_bytes(fixed[8..16].try_into().ok()?);
    let file_len = f.metadata().ok()?.len();
    if header_size > file_len.saturating_sub(16) {
        return None;
    }
    let mut header = vec![0u8; header_size as usize];
    f.read_exact(&mut header).ok()?;
    Some(header)
}

/// Finish a save that was interrupted between its two commit renames.
///
/// `nsl_train_checkpoint_save` writes and fsyncs both temporaries, then
/// renames the model and the sidecar in that order, so a crash between them
/// leaves the NEW model beside the OLD sidecar, with the new sidecar complete
/// as `<path>.optim.tmp`. If the current pair does not match and the
/// temporary does, the commit is completed here and the resume continues from
/// the new generation. A temporary that does not pair is left alone, and a
/// stale one beside a matching pair is ignored.
fn recover_interrupted_commit(path: &str, optim_path: &str) {
    let optim_tmp = format!("{optim_path}.tmp");
    if !std::path::Path::new(&optim_tmp).exists() {
        return;
    }
    if let Some(h) = read_sidecar_header(optim_path)
        && matches!(sidecar_pairs_with(&h, path), Pairing::Paired)
    {
        return;
    }
    if let Some(h) = read_sidecar_header(&optim_tmp)
        && matches!(sidecar_pairs_with(&h, path), Pairing::Paired)
    {
        crate::nsl_log!(WARN, "checkpoint",
            "[checkpoint] completing an interrupted save: '{optim_tmp}' pairs with \
             '{path}' and '{optim_path}' does not -- renaming it into place"
        );
        commit_rename(&optim_tmp, optim_path, "train_checkpoint_load");
    }
}

/// In-order needle scan of the sidecar header for one numeric field — the
/// same no-JSON-parser style as `nsl_model_load`'s dtype guard. Returns the
/// raw digit strings in header order.
fn scan_header_numbers(header: &[u8], needle: &[u8]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + needle.len() <= header.len() {
        if &header[pos..pos + needle.len()] == needle {
            let start = pos + needle.len();
            let end = header[start..]
                .iter()
                .position(|b| !b.is_ascii_digit())
                .unwrap_or(header.len() - start);
            if let Some(v) = std::str::from_utf8(&header[start..start + end])
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
            {
                out.push(v);
            }
            pos = start + end;
        } else {
            pos += 1;
        }
    }
    out
}

/// The first `"<needle>"` string value in the header, verbatim (no escape
/// handling — every string this runtime writes into the header is hex or a
/// tensor name).
fn scan_header_string(header: &[u8], needle: &[u8]) -> Option<Vec<u8>> {
    let pos = header
        .windows(needle.len())
        .position(|w| w == needle)?;
    let start = pos + needle.len();
    let rest = &header[start..];
    // Value starts at the opening quote right after `"key":`.
    if rest.first() != Some(&b'"') {
        return None;
    }
    let body = &rest[1..];
    let end = body.iter().position(|&b| b == b'"')?;
    Some(body[..end].to_vec())
}

/// The sidecar's `params` table. The header as a whole is not parsed as
/// JSON (its `env`/`exec` records are written verbatim), so this reads only
/// the array after the last `"params":`, which the save writes last.
fn sidecar_param_entries(header: &[u8]) -> Result<Vec<NslmEntry>, String> {
    let needle: &[u8] = b"\"params\":";
    let pos = header
        .windows(needle.len())
        .rposition(|w| w == needle)
        .ok_or("the sidecar header has no params table")?;
    serde_json::Deserializer::from_slice(&header[pos + needle.len()..])
        .into_iter::<Vec<NslmEntry>>()
        .next()
        .ok_or("the sidecar's params table is empty")?
        .map_err(|e| format!("the sidecar's params table is malformed: {e}"))
}

/// Read one moment tensor's raw f32 bytes into `buf` (device tensors are
/// staged D2H; host-resident tensors — `--optim-state-offload` — are read in
/// place). Aborts on the compositions the checkpoint contract refuses:
/// null slots (ZeRO placeholders) and non-f32 moments (CPDT precision) are
/// refused at codegen, so hitting one here means the refusal drifted — abort
/// loudly rather than serialize garbage.
fn read_moment_bytes(tensor_ptr: i64, which: &str, idx: usize, buf: &mut Vec<u8>) -> usize {
    if tensor_ptr == 0 {
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_save: {which}[{idx}] is a null moment slot \
             (ZeRO placeholder?) — full-state checkpointing does not compose \
             with sharded/owner-gated moments"
        );
        std::process::abort();
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.dtype != 1 {
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_save: {which}[{idx}] has dtype {} — only \
             plain f32 moments are checkpointable (CPDT moment precision is \
             refused at compile time)",
            tensor.dtype
        );
        std::process::abort();
    }
    check_tensor_contiguous(tensor, idx);
    let byte_count = (tensor.len as usize) * tensor.element_size();
    if tensor.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let start = buf.len();
            buf.resize(start + byte_count, 0);
            crate::cuda::inner::memcpy_dtoh(
                buf[start..].as_mut_ptr() as *mut std::ffi::c_void,
                tensor.data,
                byte_count,
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_save: {which}[{idx}] is on GPU but CUDA \
                 not compiled"
            );
            std::process::abort();
        }
    } else {
        let slice =
            unsafe { std::slice::from_raw_parts(tensor.data as *const u8, byte_count) };
        buf.extend_from_slice(slice);
    }
    byte_count
}

/// Save the FULL training state: θ as a normal `.nslm` (via
/// [`nsl_model_save`], so the streamed/bf16-sr materialization logic is
/// shared) plus a `<path>.optim` sidecar holding the AdamW moments and the
/// micro-batch step counter. Both files are written to a `.tmp`, fsynced, and
/// renamed model first: a crash before the first rename leaves the previous
/// pair intact, and one between the renames leaves the new pair complete,
/// which `recover_interrupted_commit` finishes at the next load. No crash
/// leaves a half-written file or a mixed pair that loads.
///
/// The sidecar extends the checkpoint WITHOUT touching `.nslm` version 1:
/// `nsl_model_load` hard-aborts on any unknown version, so a v2 container
/// would strand every existing consumer; a sidecar keeps the model file
/// loadable by plain `model_load` (weights-only restart stays possible).
///
/// Sidecar format (mirrors `.nslm` deliberately): magic `NSLO`, u32 LE
/// version, u64 LE header size, JSON header
/// `{"step_count":N,"model_sig":S,"model_sha256":"<hex>","resume":{…},"params":[{name,shape,dtype,offset,nbytes}...]}`
/// (`model_sha256` pairs the sidecar with the exact model file; `model_sig`,
/// the older sampled signature, is still written for older readers)
/// (all m entries in param order, then all v entries), zero-pad to 64, raw
/// little-endian f32 data back to back.
///
/// Item 8 — the `resume` block. θ/m/v/step alone do not describe a training
/// run: two more things determine what the next step computes, and both used
/// to restart from scratch on resume.
///
/// * **Where in the data we are** — `loader_epoch` + `loader_slot`, plus
///   `loader_id` fingerprinting the corpus and geometry they index into.
///   `dl_ptr` may be 0 (a train block with no DataLoader), which records a
///   loader-less checkpoint; mixing the two refuses at load.
/// * **The RNG streams** — dropout masks on both CPU and GPU. Captured on the
///   training thread, which is where the sampling RNG's thread-local lives.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_train_checkpoint_save(
    path_ptr: i64,
    path_len: i64,
    param_names_ptr: i64,
    param_tensors_ptr: i64,
    state1_ptr: i64,
    state2_ptr: i64,
    step_count: i64,
    dl_ptr: i64,
    train_epoch: i64,
) {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    let names = NslList::from_ptr(param_names_ptr);
    let m_list = NslList::from_ptr(state1_ptr);
    let v_list = NslList::from_ptr(state2_ptr);
    if m_list.len != v_list.len || m_list.len != names.len {
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_save: list length mismatch ({} names, {} m, {} v)",
            names.len, m_list.len, v_list.len
        );
        std::process::abort();
    }

    // θ first, through the existing save path (handles evicted/bf16-sr
    // params), into a tmp. The rename is DEFERRED until the sidecar tmp is
    // also fully written, so the two commits land back-to-back — a crash
    // window of microseconds instead of the seconds an 8 GB moment write
    // takes. The residual window (between the two renames) is covered by
    // the model signature echoed into the sidecar header: a stale pair
    // fails the loader's pairing check instead of silently resuming θ@N
    // with moments@N−k.
    let model_tmp = format!("{path}.tmp");
    nsl_model_save(
        model_tmp.as_ptr() as i64,
        model_tmp.len() as i64,
        param_names_ptr,
        param_tensors_ptr,
    );
    // The legacy sampled signature (first/last MiB) stays for older readers;
    // the pairing check uses the whole-file hash, which also sees a same-size
    // change in the middle of θ.
    let sig = model_file_sig(&model_tmp);
    let model_sha = match file_sha256_hex(&model_tmp) {
        Ok(h) => h,
        Err(e) => {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_save: hashing '{model_tmp}': {e}");
            std::process::abort();
        }
    };

    // Sidecar: header first (metadata reads need no residency), then data.
    let mut params_json = Vec::new();
    let mut data_offset: u64 = 0;
    for (prefix, list) in [("m", &m_list), ("v", &v_list)] {
        for i in 0..list.len as usize {
            let tensor_ptr = unsafe { *list.data.add(i) };
            if tensor_ptr == 0 {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_save: {prefix}[{i}] is a null moment \
                     slot — refused composition (see codegen checkpoint rules)"
                );
                std::process::abort();
            }
            let tensor = NslTensor::from_ptr(tensor_ptr);
            let nbytes = (tensor.len as u64) * (tensor.element_size() as u64);
            let shape: Vec<i64> = (0..tensor.ndim as usize)
                .map(|d| unsafe { *tensor.shape.add(d) })
                .collect();
            let name_ptr = unsafe { *names.data.add(i) };
            let name = unsafe {
                std::ffi::CStr::from_ptr(name_ptr as *const std::os::raw::c_char)
            }
            .to_str()
            .unwrap_or("?");
            params_json.push(format!(
                r#"{{"name":"{prefix}:{name}","shape":{shape:?},"dtype":"f32","offset":{data_offset},"nbytes":{nbytes}}}"#,
            ));
            data_offset += nbytes;
        }
    }
    // Item 8: the resume block. Every field is a non-negative integer or a
    // hex string, because the header parsers are needle scanners over ASCII
    // digits — a '-' would silently truncate the value being read.
    let rng = crate::rng_state::RngSnapshot::capture();
    let mut train_epoch = train_epoch.max(0) as u64;
    let (mut loader_epoch, mut loader_slot, loader_id) = if dl_ptr != 0 {
        (
            crate::dataloader::nsl_dataloader_epoch(dl_ptr).max(0) as u64,
            crate::dataloader::nsl_dataloader_slot(dl_ptr).max(0) as u64,
            crate::dataloader::nsl_dataloader_identity(dl_ptr) as u64,
        )
    } else {
        (0, 0, 0)
    };
    // NORMALIZE an exhausted epoch to the start of the next one. What the
    // resume block must name is "the next thing to do", and `(epoch E, slot
    // total_batches)` and `(epoch E+1, slot 0)` are the same position — but
    // only the second one lets the loader's own bookkeeping AND the
    // epochs-budget refusal at load see that epoch E is finished.
    //
    // Without this, a checkpoint that fires on an epoch's final delivery slot
    // records `train_epoch = E`, which is `< epochs` for the last epoch of the
    // run, so the "this budget is spent" refusal cannot fire. The resume then
    // passed every guard, drew an empty epoch, and exited 0 having trained
    // NOTHING — the exact silent no-op the refusal exists to prevent, in the
    // shape (`epochs = 1`, a DataLoader) that the 1B recipe uses.
    if dl_ptr != 0 {
        let per_epoch = crate::dataloader::nsl_dataloader_total_batches(dl_ptr).max(0) as u64;
        if loader_slot >= per_epoch {
            train_epoch += 1;
            loader_epoch += 1;
            loader_slot = 0;
        }
    }
    let resume = format!(
        r#""resume":{{"train_epoch":{train_epoch},"has_loader":{hl},"loader_epoch":{loader_epoch},"loader_slot":{loader_slot},"loader_id":{loader_id},"rng_seed":"{seed_hex}","rng_pos_hi":{hi},"rng_pos_lo":{lo},"gpu_dropout_ctr":{ctr},"bf16_sr_ctr":{srctr},"global_seed":{gseed},"global_seed_set":{gset},"exec":"{exec_fp}","train_cfg":"{train_cfg}","env":"{env_rec}"}}"#,
        hl = (dl_ptr != 0) as u64,
        // The compile-flag record installed by main(). Empty for a program
        // built before the fingerprint existed; the loader treats empty as
        // "unknown" and skips the comparison rather than refusing.
        exec_fp = crate::exec_fingerprint::exec_fingerprint(),
        // The resolved train/optimizer/scheduler record installed at
        // train-block entry (item 4). Same tolerance as `exec`: empty for
        // a build predating it; the loader says the check is skipped.
        train_cfg = crate::train_config_record::train_config_record(),
        // The runtime-read behavior-tier NSL_* variables that are SET
        // (roadmap A5). Empty means "nothing exported", which is the common
        // case; the loader tells that apart from "predates the record" by
        // the key's presence, not its value.
        env_rec = crate::env_record::env_record(),
        seed_hex = rng.seed_hex(),
        hi = (rng.sampling_pos >> 64) as u64,
        lo = rng.sampling_pos as u64,
        ctr = rng.gpu_dropout_ctr,
        // The bf16 operand-cast stochastic-rounding stream
        // (`--bf16-rounding sr`): a resume used to restart it at 0 and reuse
        // the dither windows of the run's first steps (external review
        // 2026-10-06).
        srctr = rng.bf16_sr_ctr,
        // The `--seed` SCALAR, which is a live training-RNG input in its own
        // right: SR-BF16's dither is `mix64(seed ^ step*SALT, ...)` and the
        // composed ZeRO-3 slice update reads the same global. Recording only
        // the sampling stream's ChaCha key would let a resume under a
        // different `--seed` silently switch every parameter's
        // stochastic-rounding stream mid-run.
        gseed = crate::deterministic_ops::get_rng_seed(),
        gset = crate::deterministic_ops::explicit_rng_seed().is_some() as u64,
    );
    let header = format!(
        r#"{{"step_count":{step_count},"model_sig":{sig},"model_sha256":"{model_sha}",{resume},"params":[{}]}}"#,
        params_json.join(",")
    );
    let header_bytes = header.as_bytes();

    let optim_path = format!("{path}.optim");
    let optim_tmp = format!("{optim_path}.tmp");
    let mut file = match std::fs::File::create(&optim_tmp) {
        Ok(f) => f,
        Err(e) => {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_save: cannot create '{optim_tmp}': {e}");
            std::process::abort();
        }
    };
    write_or_abort(&mut file, OPTIM_MAGIC, "write optim magic");
    write_or_abort(&mut file, &OPTIM_VERSION.to_le_bytes(), "write optim version");
    write_or_abort(
        &mut file,
        &(header_bytes.len() as u64).to_le_bytes(),
        "write optim header size",
    );
    write_or_abort(&mut file, header_bytes, "write optim header");
    let total_header = 4 + 4 + 8 + header_bytes.len();
    let padding = (64 - (total_header % 64)) % 64;
    let pad_buf = [0u8; 64];
    write_or_abort(&mut file, &pad_buf[..padding], "write optim padding");

    // Moment data: staged through one reusable buffer per tensor.
    let mut buf: Vec<u8> = Vec::new();
    for (which, list) in [("m", &m_list), ("v", &v_list)] {
        for i in 0..list.len as usize {
            let tensor_ptr = unsafe { *list.data.add(i) };
            buf.clear();
            read_moment_bytes(tensor_ptr, which, i, &mut buf);
            write_or_abort(&mut file, &buf, "write moment data");
        }
    }
    sync_or_abort(&file, &optim_tmp);
    drop(file);
    // Both tmps are complete and durable — commit the pair, model first,
    // syncing the directory after EACH rename so they reach the disk in this
    // order. The only mixed state a crash can then leave is the new model
    // beside the old sidecar, with the new sidecar complete as `.optim.tmp`;
    // `recover_interrupted_commit` finishes that commit at the next load
    // (external review 2026-10-06: the mixed state used to be unrecoverable,
    // and the previous model was already gone).
    commit_rename(&model_tmp, path, "train_checkpoint_save");
    commit_rename(&optim_tmp, &optim_path, "train_checkpoint_save");
    if dl_ptr != 0 {
        crate::nsl_log!(INFO, "checkpoint", 
            "[checkpoint] saved: {path} (+.optim) at micro-batch step \
             {step_count} ({} params, epoch {train_epoch} loader slot \
             {loader_slot})",
            m_list.len
        );
    } else {
        crate::nsl_log!(INFO, "checkpoint", 
            "[checkpoint] saved: {path} (+.optim) at micro-batch step {step_count} \
             ({} params)",
            m_list.len
        );
    }
}

/// Restore the FULL training state saved by [`nsl_train_checkpoint_save`]:
/// θ from the `.nslm` (positional, via [`nsl_model_load`]), moments from the
/// `<path>.optim` sidecar, the RNG streams and the loader position (item 8),
/// and return the saved micro-batch step counter for codegen to seed
/// `step_count_var` with. The restored training epoch is published through
/// [`nsl_train_resume_epoch`]. Aborts loudly on a missing or mismatched
/// checkpoint — a resume that silently starts fresh (or restores half a
/// state) is worse than no resume.
///
/// `epochs` is the train block's declared epoch count. Item 8 fixes its
/// meaning under resume: it is the **total** for the run, not "how many
/// more". A recipe says `epochs = 40`, crashes at epoch 12, and is re-run
/// unchanged with `checkpoint_load` — it then trains epochs 12..40. The
/// alternative ("N more") makes an unedited re-run train 2N epochs and
/// forces the author to hand-compute the remainder from a step counter.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_train_checkpoint_load(
    path_ptr: i64,
    path_len: i64,
    param_names_ptr: i64,
    param_tensors_ptr: i64,
    state1_ptr: i64,
    state2_ptr: i64,
    dl_ptr: i64,
    epochs: i64,
) -> i64 {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };

    // Refuse an armed standalone weight provider up front: nsl_model_load's
    // first move is `try_load_from_provider`, which would silently restore
    // θ from the EMBEDDED weights while this function restores moments and
    // the step counter from the sidecar — exactly the mixed half-state this
    // function's contract forbids.
    if crate::weight_provider::provider_is_set() {
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: a standalone weight provider is \
             armed — θ would come from the embedded weights while moments \
             and the step counter come from '{path}.optim' (mixed state). \
             Drop checkpoint_load in the embedded-weights build, or build \
             without the provider to resume from disk."
        );
        std::process::abort();
    }

    // VALIDATE-BEFORE-MUTATE: every check below runs before any live tensor
    // is touched, so a refused resume leaves the freshly-initialized train
    // state fully intact (same doctrine as nsl_model_load's dtype pre-pass).
    let optim_path = format!("{path}.optim");
    recover_interrupted_commit(path, &optim_path);
    let param_tensors = NslList::from_ptr(param_tensors_ptr);
    let param_names = NslList::from_ptr(param_names_ptr);
    let (model_data, model_layout) = read_validated_nslm(path, param_tensors, Some(param_names))
        .unwrap_or_else(|e| {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: {e}");
            std::process::abort();
        });
    let data = match std::fs::read(&optim_path) {
        Ok(d) => d,
        Err(e) => {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: cannot read '{optim_path}': {e} — \
                 resuming without optimizer state would silently re-warm \
                 AdamW; aborting instead"
            );
            std::process::abort();
        }
    };
    if data.len() < 16 || &data[0..4] != OPTIM_MAGIC {
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: '{optim_path}' is not an NSLO sidecar");
        std::process::abort();
    }
    let version = u32::from_le_bytes(
        data[4..8].try_into().unwrap_or_else(|_| std::process::abort()),
    );
    // v1 sidecars still load: θ, moments and the step counter are all
    // honestly present in them, and refusing would strand every checkpoint
    // written before item 8. What a v1 file CANNOT carry is the data
    // position and RNG state — so say exactly that, loudly, instead of
    // resuming into a silently different data order.
    if version != OPTIM_VERSION && version != 1 {
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: unsupported sidecar version {version} \
             (this runtime writes {OPTIM_VERSION} and reads 1..={OPTIM_VERSION})"
        );
        std::process::abort();
    }
    let is_v1 = version == 1;
    let header_size = u64::from_le_bytes(
        data[8..16].try_into().unwrap_or_else(|_| std::process::abort()),
    ) as usize;
    if header_size > data.len() - 16 {
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: sidecar header overruns the file");
        std::process::abort();
    }
    let header_bytes = &data[16..16 + header_size];

    // step_count: lightweight needle parse, same no-JSON-parser style as
    // nsl_model_load's checks.
    let step_count = {
        let needle = b"\"step_count\":";
        let pos = header_bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .unwrap_or_else(|| {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: sidecar header has no step_count");
                std::process::abort();
            });
        let digits = &header_bytes[pos + needle.len()..];
        let end = digits
            .iter()
            .position(|b| !b.is_ascii_digit())
            .unwrap_or(digits.len());
        std::str::from_utf8(&digits[..end])
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or_else(|| {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: unparsable step_count");
                std::process::abort();
            })
    };

    // Pairing check: the sidecar remembers the exact model file it was
    // saved next to. A crash between the two commit renames (or a hand-
    // mixed directory) leaves θ@N beside moments@N−k — same architecture,
    // same counts, silently divergent training. θ changes every optimizer
    // step, so the signature separates the pair reliably.
    match sidecar_pairs_with(header_bytes, path) {
        Pairing::Paired => {}
        Pairing::Mismatch(detail) => {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: '{optim_path}' was not saved \
                 with '{path}' ({detail}) — the pair is from different \
                 checkpoints (crash between commits, or mixed files). Refusing \
                 the mixed-state resume."
            );
            std::process::abort();
        }
        Pairing::NoRecord => {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: sidecar header has neither \
                 model_sha256 nor model_sig — not a checkpoint this runtime wrote"
            );
            std::process::abort();
        }
    }

    // ── Item 8: the resume block, parsed and VALIDATED here; applied only
    // after every other check has passed (validate-before-mutate).
    let resume = if is_v1 {
        crate::nsl_log!(WARN, "nsl", 
            "[nsl] WARNING: '{optim_path}' is a v1 sidecar — it carries θ, \
             AdamW moments and the step counter, but NOT the data position \
             or the RNG state. This resume restarts the DataLoader at epoch \
             0 batch 0 and re-draws dropout masks from a fresh stream, so it \
             is not a continuation of the interrupted run.\n\
             [nsl] WARNING: it also has no epoch to restore, so the epoch \
             loop starts at 0 and `epochs` keeps its PRE-item-8 meaning of \
             'how many more' — a recipe edited to the documented total \
             semantics will over-train by the epochs already completed, and \
             the 'budget already spent' refusal cannot fire. Re-save a v2 \
             checkpoint to get the documented behavior."
        );
        None
    } else {
        let num = |needle: &[u8], what: &str| -> u64 {
            match scan_header_numbers(header_bytes, needle).first() {
                Some(&v) => v,
                None => {
                    crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: v2 sidecar header is \
                         missing '{what}' — malformed checkpoint"
                    );
                    std::process::abort();
                }
            }
        };
        // Counters that end up as i64 (epoch/slot bounds, the published
        // resume epoch) must be range-checked HERE. `v as i64` on a value
        // above i64::MAX wraps NEGATIVE, and a negative epoch start slips
        // past the `>= epochs` budget refusal and seeds the epoch loop with
        // i64::MIN — an effectively unbounded run out of a corrupt field,
        // where every other malformed field in this block aborts.
        let counter = |needle: &[u8], what: &str| -> u64 {
            let v = num(needle, what);
            if v > i64::MAX as u64 {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: '{what}' is {v}, past the \
                     signed range every consumer of it uses — refusing rather \
                     than wrapping to a negative counter"
                );
                std::process::abort();
            }
            v
        };
        let seed_hex = scan_header_string(header_bytes, b"\"rng_seed\":")
            .unwrap_or_else(|| {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: v2 sidecar header is missing \
                     'rng_seed' — malformed checkpoint"
                );
                std::process::abort();
            });
        let sampling_seed = crate::rng_state::RngSnapshot::seed_from_hex(&seed_hex)
            .unwrap_or_else(|| {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: 'rng_seed' is not 64 hex \
                     digits — a partially-parsed seed would restore a \
                     DIFFERENT random stream while looking successful"
                );
                std::process::abort();
            });
        let hi = num(b"\"rng_pos_hi\":", "rng_pos_hi");
        let lo = num(b"\"rng_pos_lo\":", "rng_pos_lo");
        Some(ResumeState {
            train_epoch: counter(b"\"train_epoch\":", "train_epoch"),
            had_loader: num(b"\"has_loader\":", "has_loader") != 0,
            loader_epoch: counter(b"\"loader_epoch\":", "loader_epoch"),
            loader_slot: counter(b"\"loader_slot\":", "loader_slot"),
            loader_id: num(b"\"loader_id\":", "loader_id"),
            global_seed: num(b"\"global_seed\":", "global_seed"),
            global_seed_set: num(b"\"global_seed_set\":", "global_seed_set") != 0,
            exec: match scan_header_string(header_bytes, b"\"exec\":") {
                None => String::new(),
                Some(raw) => String::from_utf8(raw).unwrap_or_else(|_| {
                    crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: the sidecar's 'exec' \
                         record is not valid UTF-8. Treating it as absent \
                         would silently disable the compile-flag check, so \
                         this refuses instead — the checkpoint is corrupt."
                    );
                    std::process::abort();
                }),
            },
            train_cfg: match scan_header_string(header_bytes, b"\"train_cfg\":") {
                None => String::new(),
                Some(raw) => String::from_utf8(raw).unwrap_or_else(|_| {
                    crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: the sidecar's \
                         'train_cfg' record is not valid UTF-8. Treating it \
                         as absent would silently disable the config check, \
                         so this refuses instead — the checkpoint is corrupt."
                    );
                    std::process::abort();
                }),
            },
            env: scan_header_string(header_bytes, b"\"env\":").map(|raw| {
                String::from_utf8(raw).unwrap_or_else(|_| {
                    crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: the sidecar's 'env' \
                         record is not valid UTF-8. Treating it as absent \
                         would silently disable the environment check, so \
                         this refuses instead — the checkpoint is corrupt."
                    );
                    std::process::abort();
                })
            }),
            rng: crate::rng_state::RngSnapshot {
                sampling_seed,
                sampling_pos: ((hi as u128) << 64) | (lo as u128),
                gpu_dropout_ctr: num(b"\"gpu_dropout_ctr\":", "gpu_dropout_ctr"),
                // Absent in a sidecar written before it was recorded: the
                // stream then restarts at 0, as every resume used to.
                bf16_sr_ctr: scan_header_numbers(header_bytes, b"\"bf16_sr_ctr\":")
                    .first()
                    .copied()
                    .unwrap_or(0),
            },
        })
    };

    // The loader the checkpoint describes must be the loader we are resuming
    // into. Each direction of this mismatch silently changes what the resumed
    // run trains on, which is exactly what item 8 exists to prevent.
    if let Some(r) = &resume {
        // The `--seed` scalar is not just provenance: SR-BF16 rounds every
        // parameter with `mix64(seed ^ step*SALT, param_base + i)` and the
        // composed ZeRO-3 slice update reads the same global. Restoring θ,
        // the moments and the sampling stream while this silently changed
        // would switch every parameter's stochastic-rounding stream mid-run —
        // a resume that is not a continuation, with no diagnostic. Reachable
        // from a routine operator slip, because the resume is documented as
        // "re-run the recipe unchanged" and the seed lives on the command
        // line, not in the recipe.
        let live_seed_set = crate::deterministic_ops::explicit_rng_seed().is_some();
        let live_seed = crate::deterministic_ops::get_rng_seed();
        if live_seed_set != r.global_seed_set || live_seed != r.global_seed {
            let fmt = |set: bool, v: u64| {
                if set { format!("--seed {v}") } else { format!("no --seed (default {v})") }
            };
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: this run has {} but '{optim_path}' \
                 was saved with {} — the seed keys the stochastic-rounding and \
                 ZeRO dither streams directly, so resuming would continue θ and \
                 the moments while switching those streams. Re-run with the \
                 saved seed, or start a new run.",
                fmt(live_seed_set, live_seed),
                fmt(r.global_seed_set, r.global_seed),
            );
            std::process::abort();
        }

        // The compile flags that decided the arithmetic. `--seed` above is
        // guarded because it is a live RNG input; these are guarded for the
        // same reason one level up — they decide what the step COMPUTES, and
        // they live on the command line, not in the recipe. Dropping
        // `--source-ad` or `--deterministic` on a resume continues theta and
        // the moments under different arithmetic, with no diagnostic.
        //
        // An empty record on EITHER side means "built before this existed".
        // Refusing then would make every pre-existing checkpoint unresumable
        // to enforce a property those builds never claimed, so it warns once
        // and continues.
        let live_exec = crate::exec_fingerprint::exec_fingerprint();
        if r.exec.is_empty() || live_exec.is_empty() {
            let which = if r.exec.is_empty() { "checkpoint" } else { "this run" };
            crate::nsl_log!(WARN, "nsl", "nsl: train_checkpoint_load: no execution fingerprint in {which} \
                 — the compile-flag check is SKIPPED for this resume. A build \
                 predating the fingerprint cannot prove it used the same AD \
                 mode, dtype or fusion settings; re-save from a current build \
                 to restore the check."
            );
        } else {
            let arith = crate::exec_fingerprint::arithmetic_diff(&r.exec, &live_exec);
            if !arith.is_empty() {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: '{optim_path}' was written by a \
                     build whose ARITHMETIC differs from this one:\n{}\n  \
                     Resuming would continue theta and the optimizer moments \
                     while changing what a step computes, which is not a \
                     continuation of that run. Re-run with the saved flags, or \
                     start a new run.",
                    crate::exec_fingerprint::render(&arith)
                );
                std::process::abort();
            }
            // Placement-class changes are legitimate — the production 1B
            // recipe resumes with --optim-state-offload toggled, and the
            // arena carries its own byte-identity gate. Say so anyway: the
            // operator asked for a continuation and should know the shape of
            // the run changed under them.
            let placement = crate::exec_fingerprint::placement_diff(&r.exec, &live_exec);
            if !placement.is_empty() {
                crate::nsl_log!(WARN, "nsl", "nsl: train_checkpoint_load: resuming with different memory \
                     placement than '{optim_path}' was saved under:\n{}\n  \
                     These are value-neutral by construction, so the resume \
                     continues.",
                    crate::exec_fingerprint::render(&placement)
                );
            }
        }

        // The resolved train/optimizer/scheduler configuration (item 4).
        // Moment-meaning drift aborts with no escape; trajectory drift
        // (lr/schedule/clip) aborts naming the acknowledgment env. Policy
        // and messages live in `train_config_record::check_on_resume`.
        crate::train_config_record::check_on_resume(&r.train_cfg);

        // The runtime-read behavior-tier environment (roadmap A5). Same
        // doctrine as the two checks above: it is arithmetic, so it refuses,
        // with `NSL_RESUME_ALLOW_ENV_DRIFT=1` as the acknowledged escape.
        crate::env_record::check_on_resume(r.env.as_deref());

        // `epochs` is the run TOTAL (see this function's doc). A checkpoint
        // at or past it leaves the epoch loop with nothing to do: zero steps,
        // no output, exit 0 — the silent-no-op anti-pattern this codebase
        // refuses elsewhere (see the DataLoader's per-rank floor check).
        if (r.train_epoch as i64) >= epochs {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: '{optim_path}' is at epoch {} but \
                 this train block declares epochs = {epochs}, so the epoch \
                 loop would run ZERO steps and exit 0. `epochs` is the TOTAL \
                 for the run, not a count of additional epochs: raise it past \
                 {} to continue training, or drop checkpoint_load to start a \
                 new run.",
                r.train_epoch, r.train_epoch
            );
            std::process::abort();
        }
        if r.had_loader && dl_ptr == 0 {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: '{optim_path}' was saved from a \
                 DataLoader-driven run (epoch {}, slot {}) but this train \
                 block has no DataLoader — the saved data position cannot be \
                 restored. Use model_load(...) for a weights-only warm start.",
                r.loader_epoch, r.loader_slot
            );
            std::process::abort();
        }
        if !r.had_loader && dl_ptr != 0 {
            crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: '{optim_path}' was saved from a \
                 run with no DataLoader, but this train block has one — there \
                 is no saved data position to continue from, so the loader \
                 would silently start at epoch 0 batch 0. Use \
                 model_load(...) for a weights-only warm start."
            );
            std::process::abort();
        }
        if r.had_loader && dl_ptr != 0 {
            let live_id = crate::dataloader::nsl_dataloader_identity(dl_ptr) as u64;
            if live_id != r.loader_id {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: the DataLoader does not match \
                     the one '{optim_path}' was saved from (loader_id {live_id} \
                     vs sidecar {}) — a different corpus, batch geometry, or \
                     shuffle seed. Slot {} of a different permutation is \
                     different data; refusing the resume.",
                    r.loader_id, r.loader_slot
                );
                std::process::abort();
            }
        }
    }

    let m_list = NslList::from_ptr(state1_ptr);
    let v_list = NslList::from_ptr(state2_ptr);
    let expected = (m_list.len + v_list.len) as usize;
    let saved = sidecar_param_entries(header_bytes).unwrap_or_else(|e| {
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: '{optim_path}': {e}");
        std::process::abort();
    });
    if saved.len() != expected {
        // A hard abort, not a warning: mismatched moments positionally
        // restored into the wrong buffers is a silent training corruption,
        // and the caller explicitly asked for a full resume.
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: sidecar has {} moment tensors, \
             live train state expects {expected} — model/optimizer shape drift \
             between save and resume",
            saved.len()
        );
        std::process::abort();
    }

    // Per-entry validation, in order, before a single byte moves: each saved
    // entry names the live parameter its moment belongs to, is f32 like the
    // live moment, has its shape and size, and the entries tile the data
    // section exactly. The count check alone admits every same-count drift
    // (hidden-size change, transposed or reordered layer), which the copy
    // below would restore as garbage read from wrong offsets; and a
    // truncated sidecar used to be found mid-copy, after θ was overwritten.
    let data_start = {
        let total_header = 16 + header_size;
        total_header + (64 - total_header % 64) % 64
    };
    let mut next = 0u64;
    let mut entry = 0usize;
    for (which, list) in [("m", &m_list), ("v", &v_list)] {
        for i in 0..list.len as usize {
            let e = &saved[entry];
            let tensor_ptr = unsafe { *list.data.add(i) };
            if tensor_ptr == 0 {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: {which}[{i}] is a null \
                     moment slot — refused composition"
                );
                std::process::abort();
            }
            let tensor = NslTensor::from_ptr(tensor_ptr);
            let live_name = live_param_name(param_names, i);
            let saved_name = e.name.strip_prefix(which).and_then(|n| n.strip_prefix(':'));
            if !saved_name.is_some_and(|n| param_names_match(n, &live_name)) {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: sidecar entry #{entry} is '{}' \
                     but the live train state has '{which}:{live_name}' there — the \
                     parameters were added, removed or reordered since the save",
                    e.name
                );
                std::process::abort();
            }
            if tensor.dtype != DTYPE_F32 || e.dtype != "f32" {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: {which}[{i}] is {} in the \
                     sidecar and {} live — only plain f32 moments are restorable",
                    e.dtype,
                    checkpoint_dtype_name(tensor.dtype)
                );
                std::process::abort();
            }
            let shape = live_shape(tensor);
            let live_bytes = (tensor.len as u64) * (tensor.element_size() as u64);
            if e.nbytes != live_bytes || e.shape != shape {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: {which}[{i}] drifted between \
                     save and resume: sidecar has shape {:?} ({} bytes), live \
                     tensor is {shape:?} ({live_bytes} bytes) — a positional restore would \
                     read from the wrong offsets. Re-save from the current \
                     model configuration.",
                    e.shape, e.nbytes
                );
                std::process::abort();
            }
            if e.offset != next {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: sidecar entry #{entry} starts at \
                     data offset {} but the previous one ends at {next} — malformed header",
                    e.offset
                );
                std::process::abort();
            }
            next += e.nbytes;
            entry += 1;
        }
    }
    if (data.len() as u64).checked_sub(data_start as u64) != Some(next) {
        crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: '{optim_path}' holds {} data bytes but \
             its header describes {next} — truncated or padded sidecar",
            data.len().saturating_sub(data_start)
        );
        std::process::abort();
    }

    // All checks passed — NOW mutate: θ first, then moments.
    copy_nslm_into(&model_data, &model_layout, param_tensors);
    for (k, e) in saved.iter().enumerate() {
        let (list, i) = if k < m_list.len as usize {
            (&m_list, k)
        } else {
            (&v_list, k - m_list.len as usize)
        };
        let tensor = NslTensor::from_ptr(unsafe { *list.data.add(i) });
        let start = data_start + e.offset as usize;
        let src = &data[start..start + e.nbytes as usize];
        if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            crate::cuda::inner::memcpy_htod(
                tensor.data,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
            );
            #[cfg(not(feature = "cuda"))]
            {
                crate::nsl_log!(ERROR, "nsl", "nsl: train_checkpoint_load: moment #{k} is on GPU but \
                     CUDA not compiled"
                );
                std::process::abort();
            }
        } else {
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), tensor.data as *mut u8, src.len());
            }
        }
    }
    // Item 8: apply the resume block LAST — after every byte of θ and m/v is
    // in place, so an abort in the walk above cannot leave the loader armed
    // for a position the weights never reached.
    match &resume {
        Some(r) => {
            r.rng.restore();
            RESUME_TRAIN_EPOCH.store(r.train_epoch as i64, std::sync::atomic::Ordering::SeqCst);
            if dl_ptr != 0 {
                crate::dataloader::nsl_dataloader_resume_to(
                    dl_ptr,
                    r.loader_epoch as i64,
                    r.loader_slot as i64,
                );
            }
            crate::nsl_log!(INFO, "checkpoint", 
                "[checkpoint] resumed: {path} (+.optim) at micro-batch step \
                 {step_count} ({} params, epoch {} loader slot {})",
                m_list.len, r.train_epoch, r.loader_slot
            );
        }
        None => {
            // v1: no data position and no RNG state to restore. Explicitly
            // reset the published epoch so a v1 load after a v2 load in the
            // same process cannot inherit the v2 value.
            RESUME_TRAIN_EPOCH.store(0, std::sync::atomic::Ordering::SeqCst);
            crate::nsl_log!(INFO, "checkpoint", 
                "[checkpoint] resumed: {path} (+.optim) at micro-batch step {step_count} \
                 ({} params)",
                m_list.len
            );
        }
    }
    step_count
}

/// The item-8 half of a checkpoint: everything beyond θ/m/v/step that
/// determines what the next training step computes.
struct ResumeState {
    train_epoch: u64,
    /// The compile-flag record (`k=v,k=v`) installed by the SAVING run's
    /// `main()`. Empty when that program predates the fingerprint — which
    /// is why the comparison skips rather than refuses on empty: every
    /// checkpoint written before this feature would otherwise become
    /// unresumable.
    exec: String,
    /// The resolved train/optimizer/scheduler record (`k=v,k=v`) installed
    /// at the SAVING run's train-block entry (item 4). Same empty-means-
    /// predates-the-feature tolerance as `exec`.
    train_cfg: String,
    /// The runtime-read behavior-tier `NSL_*` variables the SAVING run had
    /// set (`NAME=v,NAME=v`; roadmap A5). `None` when the sidecar predates
    /// the record — distinct from `Some("")`, which is a run that had none
    /// set, so the check runs and an exported variable on THIS side is a
    /// difference.
    env: Option<String>,
    /// Whether the SAVING run had a DataLoader. Distinguishes "loader at
    /// epoch 0 slot 0" from "no loader at all" — without it, a loader-less
    /// checkpoint and a loader checkpoint saved before its first batch are
    /// the same three zeros.
    had_loader: bool,
    loader_epoch: u64,
    loader_slot: u64,
    loader_id: u64,
    /// The `--seed` SCALAR at save time (not the sampling stream's ChaCha
    /// key). A separate, live training-RNG input: SR-BF16's per-element
    /// dither and the composed ZeRO-3 slice update both read it directly.
    global_seed: u64,
    /// Whether that seed was set explicitly — `--seed 42` is indistinguishable
    /// from the default by value alone.
    global_seed_set: bool,
    rng: crate::rng_state::RngSnapshot,
}

fn check_tensor_contiguous(tensor: &NslTensor, idx: usize) {
    if tensor.ndim <= 1 {
        return;
    }
    let mut expected_stride = 1i64;
    for d in (0..tensor.ndim as usize).rev() {
        let actual = unsafe { *tensor.strides.add(d) };
        if actual != expected_stride {
            crate::nsl_log!(ERROR, "nsl", "nsl: model_save: parameter {} is not contiguous (dim {} stride {} expected {})",
                idx, d, actual, expected_stride
            );
            std::process::abort();
        }
        expected_stride *= unsafe { *tensor.shape.add(d) };
    }
}

#[cfg(test)]
mod pairing_tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("nsl_ckpt_pairing_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// An NSLO file: magic, 4 reserved bytes, the header length, the header.
    fn sidecar(header: &str) -> Vec<u8> {
        let mut v = OPTIM_MAGIC.to_vec();
        v.extend_from_slice(&[0u8; 4]);
        v.extend_from_slice(&(header.len() as u64).to_le_bytes());
        v.extend_from_slice(header.as_bytes());
        v
    }

    fn sha_header(model: &std::path::Path) -> String {
        format!("{{\"model_sha256\":\"{}\"}}", file_sha256_hex(model.to_str().unwrap()).unwrap())
    }

    #[test]
    fn file_sha256_matches_the_standard_vector() {
        let d = scratch("vector");
        let p = d.join("abc");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            file_sha256_hex(p.to_str().unwrap()).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn pairing_prefers_the_hash_and_falls_back_to_the_sampled_sig() {
        let d = scratch("pairs");
        let (m, other) = (d.join("m.nslm"), d.join("o.nslm"));
        std::fs::write(&m, b"model bytes").unwrap();
        std::fs::write(&other, b"model bytez").unwrap();
        let ms = m.to_str().unwrap();
        let h = sha_header(&m);
        assert!(matches!(sidecar_pairs_with(h.as_bytes(), ms), Pairing::Paired));
        assert!(matches!(sidecar_pairs_with(h.as_bytes(), other.to_str().unwrap()), Pairing::Mismatch(_)));
        // A hash that disagrees wins over a sampled sig that agrees: the sig
        // is only consulted for sidecars written before the hash existed.
        let both = format!("{{\"model_sig\":{},\"model_sha256\":\"{}\"}}", model_file_sig(ms), "0".repeat(64));
        assert!(matches!(sidecar_pairs_with(both.as_bytes(), ms), Pairing::Mismatch(_)));
        let legacy = format!("{{\"model_sig\":{}}}", model_file_sig(ms));
        assert!(matches!(sidecar_pairs_with(legacy.as_bytes(), ms), Pairing::Paired));
        assert!(matches!(sidecar_pairs_with(b"{}", ms), Pairing::NoRecord));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_sidecar_header_longer_than_the_file_is_not_read() {
        let d = scratch("short");
        let p = d.join("x.optim");
        let mut bytes = sidecar("{}");
        bytes[8..16].copy_from_slice(&(1u64 << 40).to_le_bytes());
        std::fs::write(&p, &bytes).unwrap();
        assert!(read_sidecar_header(p.to_str().unwrap()).is_none());
        std::fs::write(&p, sidecar("{\"k\":1}")).unwrap();
        assert_eq!(read_sidecar_header(p.to_str().unwrap()).as_deref(), Some(&b"{\"k\":1}"[..]));
        let _ = std::fs::remove_dir_all(d);
    }

    /// The three states a `.optim.tmp` can be found in at load time.
    #[test]
    fn an_interrupted_commit_is_completed_only_when_the_temporary_pairs() {
        let d = scratch("recover");
        let (m, opt, tmp) = (d.join("m.nslm"), d.join("m.nslm.optim"), d.join("m.nslm.optim.tmp"));
        let (ms, os) = (m.to_str().unwrap().to_owned(), opt.to_str().unwrap().to_owned());
        std::fs::write(&m, b"generation 2").unwrap();
        let stale = sidecar("{\"model_sha256\":\"00\"}");
        let fresh = sidecar(&sha_header(&m));

        // New model, old sidecar, new sidecar as the temporary: completed.
        std::fs::write(&opt, &stale).unwrap();
        std::fs::write(&tmp, &fresh).unwrap();
        recover_interrupted_commit(&ms, &os);
        assert_eq!(std::fs::read(&opt).unwrap(), fresh);
        assert!(!tmp.exists());

        // A leftover temporary beside a pair that already matches: ignored.
        std::fs::write(&tmp, &stale).unwrap();
        recover_interrupted_commit(&ms, &os);
        assert_eq!(std::fs::read(&opt).unwrap(), fresh);
        assert_eq!(std::fs::read(&tmp).unwrap(), stale);

        // Neither pairs: nothing is moved, and the load's own check refuses.
        std::fs::write(&opt, &stale).unwrap();
        recover_interrupted_commit(&ms, &os);
        assert_eq!(std::fs::read(&opt).unwrap(), stale);
        assert!(tmp.exists());
        let _ = std::fs::remove_dir_all(d);
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    /// An `.nslm` image with `header` and `data_len` data bytes.
    fn nslm(header: &str, data_len: usize) -> Vec<u8> {
        let mut v = MAGIC.to_vec();
        v.extend_from_slice(&VERSION.to_le_bytes());
        v.extend_from_slice(&(header.len() as u64).to_le_bytes());
        v.extend_from_slice(header.as_bytes());
        v.resize(v.len() + (64 - v.len() % 64) % 64, 0);
        v.resize(v.len() + data_len, 7);
        v
    }

    fn entry(name: &str, offset: u64, nbytes: u64) -> String {
        format!(r#"{{"name":"{name}","shape":[{}],"dtype":"f32","offset":{offset},"nbytes":{nbytes}}}"#, nbytes / 4)
    }

    #[test]
    fn a_tiled_data_section_parses() {
        let h = format!(r#"{{"params":[{},{}]}}"#, entry("a", 0, 8), entry("b", 8, 4));
        let layout = parse_nslm(&nslm(&h, 12)).unwrap();
        assert_eq!(layout.entries.len(), 2);
        assert_eq!(layout.data_start % 64, 0);
    }

    #[test]
    fn gaps_overlaps_and_trailing_bytes_are_refused() {
        let gap = format!(r#"{{"params":[{},{}]}}"#, entry("a", 0, 8), entry("b", 12, 4));
        assert!(parse_nslm(&nslm(&gap, 16)).unwrap_err().contains("starts at data offset 12"));
        let overlap = format!(r#"{{"params":[{},{}]}}"#, entry("a", 0, 8), entry("b", 4, 4));
        assert!(parse_nslm(&nslm(&overlap, 12)).unwrap_err().contains("starts at data offset 4"));
        let one = format!(r#"{{"params":[{}]}}"#, entry("a", 0, 8));
        assert!(parse_nslm(&nslm(&one, 12)).unwrap_err().contains("describes 8 data bytes"));
        assert!(parse_nslm(&nslm(&one, 4)).unwrap_err().contains("describes 8 data bytes"));
        let huge = format!(r#"{{"params":[{}]}}"#, entry("a", 0, u64::MAX));
        assert!(parse_nslm(&nslm(&huge, 4)).is_err());
        assert!(parse_nslm(&nslm(r#"{"params":[{"name":"a"}]}"#, 0)).unwrap_err().contains("not a valid parameter table"));
    }

    #[test]
    fn param_names_match_across_the_two_naming_schemes() {
        assert_eq!(canonical_param_name("m.blocks.0.attn.wq"), "m.blocks[0].attn.wq");
        assert_eq!(canonical_param_name("layers.12"), "layers[12]");
        assert!(param_names_match("blocks[0].w", "blocks[0].w"));
        assert!(param_names_match("blocks.0.w", "blocks[0].w"));
        assert!(param_names_match("m.blocks.0.w", "blocks[0].w"));
        assert!(!param_names_match("m.blocks.1.w", "blocks[0].w"));
        assert!(!param_names_match("a", "b"));
        // Only the FILE side may carry the extra variable segment.
        assert!(!param_names_match("w", "m.w"));
    }

    #[test]
    fn the_sidecar_params_table_is_read_past_verbatim_records() {
        let h = format!(
            r#"{{"step_count":3,"resume":{{"env":"A=b c"}},"params":[{},{}]}}"#,
            entry("m:w", 0, 4),
            entry("v:w", 4, 4)
        );
        let e = sidecar_param_entries(h.as_bytes()).unwrap();
        assert_eq!(e.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["m:w", "v:w"]);
        assert!(sidecar_param_entries(br#"{"step_count":3}"#).is_err());
    }
}
