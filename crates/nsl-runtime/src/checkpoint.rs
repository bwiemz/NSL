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
