use crate::list::NslList;
use crate::tensor::NslTensor;
use std::io::Write;

const MAGIC: &[u8; 4] = b"NSLM";
const VERSION: u32 = 1;

/// Write helper: aborts on I/O error instead of panicking across extern "C".
fn write_or_abort(file: &mut std::fs::File, buf: &[u8], context: &str) {
    if let Err(e) = file.write_all(buf) {
        eprintln!("nsl: model_save: {}: {}", context, e);
        std::process::abort();
    }
}

/// Save model parameters to .nslm binary format.
/// path_ptr/path_len: string pointer and length for file path
/// param_names_ptr: NslList of string pointers
/// param_tensors_ptr: NslList of tensor pointers
#[no_mangle]
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
        eprintln!(
            "nsl: model_save: name/tensor count mismatch ({} names, {} tensors)",
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
        let dtype_str = if tensor.dtype == 1 { "f32" } else { "f64" };
        params_json.push(format!(
            r#"{{"name":"{}","shape":{:?},"dtype":"{}","offset":{},"nbytes":{}}}"#,
            name, shape, dtype_str, data_offset, nbytes
        ));
        data_offset += nbytes;
    }
    let header = format!(r#"{{"params":[{}]}}"#, params_json.join(","));
    let header_bytes = header.as_bytes();

    let mut file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("nsl: model_save: cannot create file '{}': {}", path, e);
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
            // CRITICAL dtype trap: `nsl_tensor_to_device(_, 0)` follows the
            // runtime convention CPU=f64 / GPU=f32 and CONVERTS the staged
            // buffer to f64 (len*8 bytes). The header above was built from
            // the GPU tensor (dtype "f32", nbytes = len*4), so writing
            // `byte_count` raw bytes from the f64 staging buffer serialized
            // interleaved f64 halves as "f32" data — every checkpoint saved
            // from a GPU-resident model was garbage (found by the roadmap-4.3
            // FASE-parity gate: |max| ~ 3.7e19 in freshly-initialized
            // weights). Downcast the staging buffer element-wise so the bytes
            // match the declared dtype; fall through to a raw write only when
            // the staging preserved the dtype.
            let cpu_ptr = crate::tensor::nsl_tensor_to_device(tensor_ptr, 0);
            let cpu_tensor = NslTensor::from_ptr(cpu_ptr);
            if cpu_tensor.dtype == tensor.dtype {
                let data_slice = unsafe {
                    std::slice::from_raw_parts(cpu_tensor.data as *const u8, byte_count)
                };
                write_or_abort(&mut file, data_slice, "write tensor data (GPU->CPU)");
            } else if tensor.dtype == 1 && cpu_tensor.dtype == 0 {
                // GPU f32 declared in the header; staging is f64 — downcast.
                let n = tensor.len as usize;
                let src = unsafe {
                    std::slice::from_raw_parts(cpu_tensor.data as *const f64, n)
                };
                let mut buf = Vec::with_capacity(n * 4);
                for v in src {
                    buf.extend_from_slice(&(*v as f32).to_le_bytes());
                }
                write_or_abort(&mut file, &buf, "write tensor data (GPU f32 via f64 staging)");
            } else {
                eprintln!(
                    "nsl: model_save: unsupported dtype transition in GPU staging \
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
}

/// Load model parameters from .nslm binary format into existing tensors.
#[no_mangle]
pub extern "C" fn nsl_model_load(path_ptr: i64, path_len: i64, param_tensors_ptr: i64) {
    let tensors = NslList::from_ptr(param_tensors_ptr);
    if crate::weight_provider::try_load_from_provider(tensors) {
        return;
    }

    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("nsl: model_load: cannot read file '{}': {}", path, e);
            std::process::abort();
        }
    };

    if data.len() < 16 {
        eprintln!(
            "nsl: model_load: file too small ({} bytes, need at least 16)",
            data.len()
        );
        std::process::abort();
    }
    if &data[0..4] != MAGIC {
        eprintln!("nsl: model_load: invalid .nslm file (bad magic)");
        std::process::abort();
    }
    let version = u32::from_le_bytes(
        data[4..8]
            .try_into()
            .unwrap_or_else(|_| std::process::abort()),
    );
    if version != VERSION {
        eprintln!(
            "nsl: model_load: unsupported version {} (expected {})",
            version, VERSION
        );
        std::process::abort();
    }
    let header_size = u64::from_le_bytes(
        data[8..16]
            .try_into()
            .unwrap_or_else(|_| std::process::abort()),
    ) as usize;

    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;

    // Count saved params by counting "name": occurrences in the JSON header.
    // This is a lightweight check that avoids pulling in a full JSON parser.
    {
        let header_bytes = &data[16..16 + header_size];
        let needle = b"\"name\":";
        let saved_param_count = header_bytes
            .windows(needle.len())
            .filter(|w| *w == needle)
            .count();
        if saved_param_count != tensors.len as usize {
            eprintln!(
                "[nsl] WARNING: checkpoint has {} params but model expects {} params; \
                 weights may be mismatched",
                saved_param_count, tensors.len
            );
        }
    }

    // In-order dtype guard (same lightweight no-JSON-parser style as the
    // count check above): this loader walks the data section by the LIVE
    // tensor's element size and raw-copies bytes, so a dtype mismatch
    // between a file entry and the destination tensor (e.g. a CPU-saved
    // f64 checkpoint loaded into a GPU-resident f32 model, or vice versa)
    // would silently reinterpret bytes AND misalign every subsequent
    // tensor. Refuse loudly instead — found while fixing the model_save
    // GPU-staging dtype bug (f64 staging serialized under an f32 header).
    let file_dtypes: Vec<&[u8]> = {
        let header_bytes = &data[16..16 + header_size];
        let needle: &[u8] = b"\"dtype\":\"";
        let mut out = Vec::new();
        let mut pos = 0;
        while pos + needle.len() <= header_bytes.len() {
            if &header_bytes[pos..pos + needle.len()] == needle {
                let start = pos + needle.len();
                if let Some(end) = header_bytes[start..].iter().position(|&b| b == b'"') {
                    out.push(&header_bytes[start..start + end]);
                    pos = start + end;
                    continue;
                }
            }
            pos += 1;
        }
        out
    };

    // Pre-pass: validate EVERY entry's dtype before copying ANY bytes, so a
    // mismatch can never leave the model partially overwritten (tensors
    // 0..i-1 already mutated when the guard fires at i).
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if let Some(file_dtype) = file_dtypes.get(i) {
            let live_dtype: &[u8] = if tensor.dtype == 1 { b"f32" } else { b"f64" };
            if *file_dtype != live_dtype {
                eprintln!(
                    "nsl: model_load: dtype mismatch for tensor #{}: file has {}, \
                     model expects {} — raw byte copy would corrupt this tensor and \
                     misalign all subsequent ones. Re-save the checkpoint from a \
                     model on the same device convention (CPU=f64, GPU=f32).",
                    i,
                    String::from_utf8_lossy(file_dtype),
                    String::from_utf8_lossy(live_dtype),
                );
                std::process::abort();
            }
        }
    }

    let mut offset = data_start;
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        let byte_count = (tensor.len as usize) * tensor.element_size();
        if offset + byte_count > data.len() {
            eprintln!(
                "nsl: model_load: unexpected end of file at offset {} (tensor {}, need {} bytes, have {})",
                offset, i, byte_count, data.len() - offset
            );
            std::process::abort();
        }

        if tensor.device > 0 {
            // GPU tensor: load data into CPU staging buffer, then memcpy to device
            #[cfg(feature = "cuda")]
            {
                let staging = crate::memory::checked_alloc(byte_count);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data[offset..].as_ptr(),
                        staging,
                        byte_count,
                    );
                }
                crate::cuda::inner::memcpy_htod(
                    tensor.data,
                    staging as *const std::ffi::c_void,
                    byte_count,
                );
                unsafe { crate::memory::checked_free(staging, byte_count); }
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("nsl: model_load: tensor {} is on GPU but CUDA not compiled", i);
                std::process::abort();
            }
        } else {
            #[cfg(target_endian = "little")]
            {
                // Fast path: bulk copy on little-endian hardware (x86, ARM, etc.)
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data[offset..].as_ptr(),
                        tensor.data as *mut u8,
                        byte_count,
                    );
                }
            }

            #[cfg(not(target_endian = "little"))]
            {
                let elem_size = tensor.element_size();
                for j in 0..tensor.len as usize {
                    let start = offset + j * elem_size;
                    if tensor.dtype == 1 {
                        let val = f32::from_le_bytes(
                            data[start..start + 4]
                                .try_into()
                                .unwrap_or_else(|_| std::process::abort()),
                        );
                        unsafe { *tensor.data_f32().add(j) = val; }
                    } else {
                        let val = f64::from_le_bytes(
                            data[start..start + 8]
                                .try_into()
                                .unwrap_or_else(|_| std::process::abort()),
                        );
                        unsafe { *tensor.data_f64().add(j) = val; }
                    }
                }
            }
        }

        offset += byte_count;
    }
}

// ---------------------------------------------------------------------------
// Milestone B: full training-state checkpoint (θ + optimizer moments + step)
// ---------------------------------------------------------------------------

const OPTIM_MAGIC: &[u8; 4] = b"NSLO";
const OPTIM_VERSION: u32 = 1;

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
            eprintln!("nsl: checkpoint: cannot open '{path}' for signature: {e}");
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

/// In-order scan of `"shape":[...]` bodies (the bracketed text, verbatim).
fn scan_header_shapes(header: &[u8]) -> Vec<String> {
    let needle: &[u8] = b"\"shape\":[";
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + needle.len() <= header.len() {
        if &header[pos..pos + needle.len()] == needle {
            let start = pos + needle.len();
            if let Some(end) = header[start..].iter().position(|&b| b == b']') {
                out.push(String::from_utf8_lossy(&header[start..start + end]).to_string());
                pos = start + end;
                continue;
            }
        }
        pos += 1;
    }
    out
}

/// Read one moment tensor's raw f32 bytes into `buf` (device tensors are
/// staged D2H; host-resident tensors — `--optim-state-offload` — are read in
/// place). Aborts on the compositions the checkpoint contract refuses:
/// null slots (ZeRO placeholders) and non-f32 moments (CPDT precision) are
/// refused at codegen, so hitting one here means the refusal drifted — abort
/// loudly rather than serialize garbage.
fn read_moment_bytes(tensor_ptr: i64, which: &str, idx: usize, buf: &mut Vec<u8>) -> usize {
    if tensor_ptr == 0 {
        eprintln!(
            "nsl: train_checkpoint_save: {which}[{idx}] is a null moment slot \
             (ZeRO placeholder?) — full-state checkpointing does not compose \
             with sharded/owner-gated moments"
        );
        std::process::abort();
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.dtype != 1 {
        eprintln!(
            "nsl: train_checkpoint_save: {which}[{idx}] has dtype {} — only \
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
            eprintln!(
                "nsl: train_checkpoint_save: {which}[{idx}] is on GPU but CUDA \
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
/// micro-batch step counter. Both files are written to a `.tmp` and renamed,
/// so a crash mid-save leaves the previous checkpoint intact — "clean
/// checkpoint" means the on-disk state is never half-written.
///
/// The sidecar extends the checkpoint WITHOUT touching `.nslm` version 1:
/// `nsl_model_load` hard-aborts on any unknown version, so a v2 container
/// would strand every existing consumer; a sidecar keeps the model file
/// loadable by plain `model_load` (weights-only restart stays possible).
///
/// Sidecar format (mirrors `.nslm` deliberately): magic `NSLO`, u32 LE
/// version, u64 LE header size, JSON header
/// `{"step_count":N,"params":[{name,shape,dtype,offset,nbytes}...]}` (all m
/// entries in param order, then all v entries), zero-pad to 64, raw
/// little-endian f32 data back to back.
#[no_mangle]
pub extern "C" fn nsl_train_checkpoint_save(
    path_ptr: i64,
    path_len: i64,
    param_names_ptr: i64,
    param_tensors_ptr: i64,
    state1_ptr: i64,
    state2_ptr: i64,
    step_count: i64,
) {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    let names = NslList::from_ptr(param_names_ptr);
    let m_list = NslList::from_ptr(state1_ptr);
    let v_list = NslList::from_ptr(state2_ptr);
    if m_list.len != v_list.len || m_list.len != names.len {
        eprintln!(
            "nsl: train_checkpoint_save: list length mismatch ({} names, {} m, {} v)",
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
    let sig = model_file_sig(&model_tmp);

    // Sidecar: header first (metadata reads need no residency), then data.
    let mut params_json = Vec::new();
    let mut data_offset: u64 = 0;
    for (prefix, list) in [("m", &m_list), ("v", &v_list)] {
        for i in 0..list.len as usize {
            let tensor_ptr = unsafe { *list.data.add(i) };
            if tensor_ptr == 0 {
                eprintln!(
                    "nsl: train_checkpoint_save: {prefix}[{i}] is a null moment \
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
    let header = format!(
        r#"{{"step_count":{step_count},"model_sig":{sig},"params":[{}]}}"#,
        params_json.join(",")
    );
    let header_bytes = header.as_bytes();

    let optim_path = format!("{path}.optim");
    let optim_tmp = format!("{optim_path}.tmp");
    let mut file = match std::fs::File::create(&optim_tmp) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("nsl: train_checkpoint_save: cannot create '{optim_tmp}': {e}");
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
    drop(file);
    // Both tmps are complete — commit the pair back-to-back.
    if let Err(e) = std::fs::rename(&model_tmp, path) {
        eprintln!("nsl: train_checkpoint_save: rename '{model_tmp}' -> '{path}': {e}");
        std::process::abort();
    }
    if let Err(e) = std::fs::rename(&optim_tmp, &optim_path) {
        eprintln!("nsl: train_checkpoint_save: rename '{optim_tmp}' -> '{optim_path}': {e}");
        std::process::abort();
    }
    eprintln!(
        "[checkpoint] saved: {path} (+.optim) at micro-batch step {step_count} \
         ({} params)",
        m_list.len
    );
}

/// Restore the FULL training state saved by [`nsl_train_checkpoint_save`]:
/// θ from the `.nslm` (positional, via [`nsl_model_load`]), moments from the
/// `<path>.optim` sidecar, and return the saved micro-batch step counter for
/// codegen to seed `step_count_var` with. Aborts loudly on a missing or
/// mismatched checkpoint — a resume that silently starts fresh (or restores
/// half a state) is worse than no resume.
#[no_mangle]
pub extern "C" fn nsl_train_checkpoint_load(
    path_ptr: i64,
    path_len: i64,
    param_tensors_ptr: i64,
    state1_ptr: i64,
    state2_ptr: i64,
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
        eprintln!(
            "nsl: train_checkpoint_load: a standalone weight provider is \
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
    let data = match std::fs::read(&optim_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "nsl: train_checkpoint_load: cannot read '{optim_path}': {e} — \
                 resuming without optimizer state would silently re-warm \
                 AdamW; aborting instead"
            );
            std::process::abort();
        }
    };
    if data.len() < 16 || &data[0..4] != OPTIM_MAGIC {
        eprintln!("nsl: train_checkpoint_load: '{optim_path}' is not an NSLO sidecar");
        std::process::abort();
    }
    let version = u32::from_le_bytes(
        data[4..8].try_into().unwrap_or_else(|_| std::process::abort()),
    );
    if version != OPTIM_VERSION {
        eprintln!(
            "nsl: train_checkpoint_load: unsupported sidecar version {version} \
             (expected {OPTIM_VERSION})"
        );
        std::process::abort();
    }
    let header_size = u64::from_le_bytes(
        data[8..16].try_into().unwrap_or_else(|_| std::process::abort()),
    ) as usize;
    if 16 + header_size > data.len() {
        eprintln!("nsl: train_checkpoint_load: sidecar header overruns the file");
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
                eprintln!("nsl: train_checkpoint_load: sidecar header has no step_count");
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
                eprintln!("nsl: train_checkpoint_load: unparsable step_count");
                std::process::abort();
            })
    };

    // Pairing check: the sidecar remembers the exact model file it was
    // saved next to. A crash between the two commit renames (or a hand-
    // mixed directory) leaves θ@N beside moments@N−k — same architecture,
    // same counts, silently divergent training. θ changes every optimizer
    // step, so the signature separates the pair reliably.
    match scan_header_numbers(header_bytes, b"\"model_sig\":").first() {
        Some(&saved_sig) => {
            let live_sig = model_file_sig(path);
            if saved_sig != live_sig {
                eprintln!(
                    "nsl: train_checkpoint_load: '{optim_path}' was not saved \
                     with '{path}' (model_sig {live_sig} vs sidecar \
                     {saved_sig}) — the pair is from different checkpoints \
                     (crash between commits, or mixed files). Refusing the \
                     mixed-state resume."
                );
                std::process::abort();
            }
        }
        None => {
            eprintln!(
                "nsl: train_checkpoint_load: sidecar header has no model_sig \
                 — not a checkpoint this runtime wrote"
            );
            std::process::abort();
        }
    }

    let m_list = NslList::from_ptr(state1_ptr);
    let v_list = NslList::from_ptr(state2_ptr);
    let expected = (m_list.len + v_list.len) as usize;
    let saved = {
        let needle = b"\"name\":";
        header_bytes
            .windows(needle.len())
            .filter(|w| *w == needle)
            .count()
    };
    if saved != expected {
        // A hard abort, not the model loader's warning: mismatched moments
        // positionally restored into the wrong buffers is a silent training
        // corruption, and the caller explicitly asked for a full resume.
        eprintln!(
            "nsl: train_checkpoint_load: sidecar has {saved} moment tensors, \
             live train state expects {expected} — model/optimizer shape drift \
             between save and resume"
        );
        std::process::abort();
    }

    // Per-entry validation: the count check alone admits every same-count
    // drift (hidden-size change, transposed layer), which the walk below
    // would restore as garbage read from wrong offsets. Compare each saved
    // entry's nbytes AND shape against the live tensor, in order, before a
    // single byte moves.
    let saved_nbytes = scan_header_numbers(header_bytes, b"\"nbytes\":");
    let saved_shapes = scan_header_shapes(header_bytes);
    if saved_nbytes.len() != expected || saved_shapes.len() != expected {
        eprintln!(
            "nsl: train_checkpoint_load: sidecar header lists {} nbytes / {} \
             shape entries for {expected} tensors — malformed header",
            saved_nbytes.len(),
            saved_shapes.len()
        );
        std::process::abort();
    }
    let mut entry = 0usize;
    for (which, list) in [("m", &m_list), ("v", &v_list)] {
        for i in 0..list.len as usize {
            let tensor_ptr = unsafe { *list.data.add(i) };
            if tensor_ptr == 0 {
                eprintln!(
                    "nsl: train_checkpoint_load: {which}[{i}] is a null \
                     moment slot — refused composition"
                );
                std::process::abort();
            }
            let tensor = NslTensor::from_ptr(tensor_ptr);
            let live_bytes = (tensor.len as u64) * (tensor.element_size() as u64);
            let live_shape = {
                let s: Vec<i64> = (0..tensor.ndim as usize)
                    .map(|d| unsafe { *tensor.shape.add(d) })
                    .collect();
                let rendered = format!("{s:?}");
                rendered[1..rendered.len() - 1].to_string()
            };
            if saved_nbytes[entry] != live_bytes || saved_shapes[entry] != live_shape {
                eprintln!(
                    "nsl: train_checkpoint_load: {which}[{i}] drifted between \
                     save and resume: sidecar has shape [{}] ({} bytes), live \
                     tensor is [{}] ({} bytes) — a positional restore would \
                     read from the wrong offsets. Re-save from the current \
                     model configuration.",
                    saved_shapes[entry], saved_nbytes[entry], live_shape, live_bytes
                );
                std::process::abort();
            }
            entry += 1;
        }
    }

    // All checks passed — NOW mutate: θ first, then moments.
    nsl_model_load(path_ptr, path_len, param_tensors_ptr);

    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let mut offset = total_header + padding;
    for (which, list) in [("m", &m_list), ("v", &v_list)] {
        for i in 0..list.len as usize {
            let tensor_ptr = unsafe { *list.data.add(i) };
            if tensor_ptr == 0 {
                eprintln!(
                    "nsl: train_checkpoint_load: {which}[{i}] is a null moment \
                     slot — refused composition"
                );
                std::process::abort();
            }
            let tensor = NslTensor::from_ptr(tensor_ptr);
            if tensor.dtype != 1 {
                eprintln!(
                    "nsl: train_checkpoint_load: {which}[{i}] has dtype {} — \
                     only plain f32 moments are restorable",
                    tensor.dtype
                );
                std::process::abort();
            }
            let byte_count = (tensor.len as usize) * tensor.element_size();
            if offset + byte_count > data.len() {
                eprintln!(
                    "nsl: train_checkpoint_load: sidecar ended early at {which}[{i}] \
                     (need {byte_count} bytes at offset {offset}, have {})",
                    data.len().saturating_sub(offset)
                );
                std::process::abort();
            }
            if tensor.device > 0 {
                #[cfg(feature = "cuda")]
                {
                    let staging = crate::memory::checked_alloc(byte_count);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data[offset..].as_ptr(),
                            staging,
                            byte_count,
                        );
                    }
                    crate::cuda::inner::memcpy_htod(
                        tensor.data,
                        staging as *const std::ffi::c_void,
                        byte_count,
                    );
                    unsafe {
                        crate::memory::checked_free(staging, byte_count);
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!(
                        "nsl: train_checkpoint_load: {which}[{i}] is on GPU but \
                         CUDA not compiled"
                    );
                    std::process::abort();
                }
            } else {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data[offset..].as_ptr(),
                        tensor.data as *mut u8,
                        byte_count,
                    );
                }
            }
            offset += byte_count;
        }
    }
    // Full-consumption check: the per-entry validation above makes this
    // unreachable in practice, but a walk that ends short of the data
    // section would mean the header lied — refuse rather than trust it.
    if offset != data.len() {
        eprintln!(
            "nsl: train_checkpoint_load: sidecar has {} unconsumed trailing \
             bytes after the last moment — header/data drift",
            data.len().saturating_sub(offset)
        );
        std::process::abort();
    }
    eprintln!(
        "[checkpoint] resumed: {path} (+.optim) at micro-batch step {step_count} \
         ({} params)",
        m_list.len
    );
    step_count
}

fn check_tensor_contiguous(tensor: &NslTensor, idx: usize) {
    if tensor.ndim <= 1 {
        return;
    }
    let mut expected_stride = 1i64;
    for d in (0..tensor.ndim as usize).rev() {
        let actual = unsafe { *tensor.strides.add(d) };
        if actual != expected_stride {
            eprintln!(
                "nsl: model_save: parameter {} is not contiguous (dim {} stride {} expected {})",
                idx, d, actual, expected_stride
            );
            std::process::abort();
        }
        expected_stride *= unsafe { *tensor.shape.add(d) };
    }
}
