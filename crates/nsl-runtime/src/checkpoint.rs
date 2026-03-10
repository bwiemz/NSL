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
        let nbytes = (tensor.len as u64) * 8;
        let shape: Vec<i64> = (0..tensor.ndim as usize)
            .map(|d| unsafe { *tensor.shape.add(d) })
            .collect();
        params_json.push(format!(
            r#"{{"name":"param_{}","shape":{:?},"dtype":"f64","offset":{},"nbytes":{}}}"#,
            i, shape, data_offset, nbytes
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

    // Raw tensor data (little-endian f64)
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        for j in 0..tensor.len as usize {
            let val = unsafe { *tensor.data.add(j) };
            write_or_abort(&mut file, &val.to_le_bytes(), "write tensor data");
        }
    }
}

/// Load model parameters from .nslm binary format into existing tensors.
#[no_mangle]
pub extern "C" fn nsl_model_load(path_ptr: i64, path_len: i64, param_tensors_ptr: i64) {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    let tensors = NslList::from_ptr(param_tensors_ptr);
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

    let mut offset = data_start;
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        let byte_count = (tensor.len as usize) * 8;
        if offset + byte_count > data.len() {
            eprintln!(
                "nsl: model_load: unexpected end of file at offset {} (tensor {}, need {} bytes, have {})",
                offset, i, byte_count, data.len() - offset
            );
            std::process::abort();
        }

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
            for j in 0..tensor.len as usize {
                let val = f64::from_le_bytes(
                    data[offset + j * 8..offset + j * 8 + 8]
                        .try_into()
                        .unwrap_or_else(|_| std::process::abort()),
                );
                unsafe {
                    *tensor.data.add(j) = val;
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
