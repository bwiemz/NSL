// M18b: Safetensors read/write FFI

use std::ffi::c_void;

use crate::dict::{nsl_dict_keys, nsl_dict_new, nsl_dict_set_str};
use crate::list::NslList;
use crate::memory::checked_alloc;
use crate::tensor::NslTensor;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: allocate a NUL-terminated C string
// ─────────────────────────────────────────────────────────────────────────────
pub(crate) fn alloc_c_string(s: &str) -> *mut u8 {
    let bytes = s.as_bytes();
    let ptr = checked_alloc(bytes.len() + 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        *ptr.add(bytes.len()) = 0;
    }
    ptr
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: convert safetensors dtype bytes → Vec<f32>
// ─────────────────────────────────────────────────────────────────────────────
pub(crate) fn convert_to_f32(dtype: safetensors::Dtype, data: &[u8]) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => {
            // Interpret bytes directly as f32
            let count = data.len() / 4;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into().unwrap();
                out.push(f32::from_le_bytes(bytes));
            }
            out
        }
        safetensors::Dtype::F64 => {
            let count = data.len() / 8;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 8] = data[i * 8..(i + 1) * 8].try_into().unwrap();
                let v = f64::from_le_bytes(bytes);
                out.push(v as f32);
            }
            out
        }
        safetensors::Dtype::F16 => {
            let count = data.len() / 2;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 2] = data[i * 2..(i + 1) * 2].try_into().unwrap();
                let v = half::f16::from_le_bytes(bytes);
                out.push(f32::from(v));
            }
            out
        }
        safetensors::Dtype::BF16 => {
            let count = data.len() / 2;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 2] = data[i * 2..(i + 1) * 2].try_into().unwrap();
                let v = half::bf16::from_le_bytes(bytes);
                out.push(f32::from(v));
            }
            out
        }
        safetensors::Dtype::I32 => {
            let count = data.len() / 4;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into().unwrap();
                let v = i32::from_le_bytes(bytes);
                out.push(v as f32);
            }
            out
        }
        safetensors::Dtype::I64 => {
            let count = data.len() / 8;
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 8] = data[i * 8..(i + 1) * 8].try_into().unwrap();
                let v = i64::from_le_bytes(bytes);
                out.push(v as f32);
            }
            out
        }
        other => {
            eprintln!("[nsl] safetensors_load: unsupported dtype {:?}, treating as zeros", other);
            Vec::new()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build an NslTensor (f32, device) from a Vec<f32> and shape
// ─────────────────────────────────────────────────────────────────────────────
pub(crate) fn allocate_f32_tensor(
    f32_data: &[f32],
    shape: &[usize],
    ndim: i64,
    len: i64,
    device: u8,
) -> NslTensor {
    // Allocate and copy shape
    let shape_ptr = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &d) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = d as i64 };
    }

    let strides_ptr = NslTensor::compute_strides(shape_ptr, ndim);

    let data_ptr: *mut c_void = if device == 0 {
        // CPU path
        let raw = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(f32_data.as_ptr(), raw, len as usize);
        }
        raw as *mut c_void
    } else {
        // GPU path (cuda feature)
        #[cfg(feature = "cuda")]
        {
            let raw = crate::cuda::inner::alloc_managed((len as usize) * std::mem::size_of::<f32>());
            // alloc_managed returns *mut c_void (unified memory); we can write from CPU
            let f32_raw = raw as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(f32_data.as_ptr(), f32_raw, len as usize);
            }
            raw
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("[nsl] safetensors_load: device>0 requires CUDA feature");
            std::process::abort();
        }
    };

    NslTensor {
        data: data_ptr,
        shape: shape_ptr,
        strides: strides_ptr,
        ndim,
        len,
        refcount: 1,
        device,
        dtype: 1, // f32
        owns_data: 1,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FFI: nsl_safetensors_load
// ─────────────────────────────────────────────────────────────────────────────

/// Load a .safetensors file and return an NslDict (i64 ptr) of name → NslTensor.
/// All tensors are loaded as f32 (dtype=1).
/// `device`: 0 = CPU, 1+ = CUDA device
#[no_mangle]
pub extern "C" fn nsl_safetensors_load(path_ptr: i64, path_len: i64, device: i64) -> i64 {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[nsl] safetensors_load: cannot read '{}': {}", path, e);
            std::process::abort();
        }
    };

    let tensors = match safetensors::SafeTensors::deserialize(&bytes) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[nsl] safetensors_load: parse error in '{}': {}", path, e);
            std::process::abort();
        }
    };

    let dict = nsl_dict_new();

    for (name, view) in tensors.tensors() {
        let shape_usize: Vec<usize> = view.shape().to_vec();
        let ndim = shape_usize.len() as i64;
        let len: i64 = if shape_usize.is_empty() {
            1
        } else {
            shape_usize.iter().map(|&d| d as i64).product()
        };

        let f32_data = convert_to_f32(view.dtype(), view.data());

        // Pad / handle empty tensors gracefully
        let f32_data = if f32_data.is_empty() && len > 0 {
            vec![0f32; len as usize]
        } else {
            f32_data
        };

        let nsl_tensor = allocate_f32_tensor(&f32_data, &shape_usize, ndim, len, device as u8);

        // Box and leak into i64 pointer
        let tensor_ptr = Box::into_raw(Box::new(nsl_tensor)) as i64;

        // Insert into dict with a C-string key
        let key_ptr = alloc_c_string(&name) as i64;
        nsl_dict_set_str(dict, key_ptr, tensor_ptr);
    }

    dict
}

// ─────────────────────────────────────────────────────────────────────────────
// FFI: nsl_safetensors_save
// ─────────────────────────────────────────────────────────────────────────────

/// Save an NslDict of name → NslTensor to a .safetensors file.
/// All tensors are written as F32 regardless of original dtype.
#[no_mangle]
pub extern "C" fn nsl_safetensors_save(dict_ptr: i64, path_ptr: i64, path_len: i64) {
    use std::collections::HashMap;

    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };

    // Collect key list from dict
    let keys_list_ptr = nsl_dict_keys(dict_ptr);
    let keys_list = NslList::from_ptr(keys_list_ptr);

    // Build the Vec<(name, TensorView)> that safetensors::serialize_to_file expects.
    // We collect data into owned Vecs first, then pass views.
    let mut owned: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

    for i in 0..keys_list.len as usize {
        let key_i64 = unsafe { *keys_list.data.add(i) };

        let name = unsafe {
            std::ffi::CStr::from_ptr(key_i64 as *const std::os::raw::c_char)
                .to_str()
                .unwrap_or("?")
                .to_owned()
        };

        // Get tensor pointer from dict
        let tensor_ptr = crate::dict::nsl_dict_get_str(dict_ptr, key_i64);
        let tensor = NslTensor::from_ptr(tensor_ptr);

        let len = tensor.len as usize;
        let shape: Vec<usize> = (0..tensor.ndim as usize)
            .map(|d| unsafe { *tensor.shape.add(d) as usize })
            .collect();

        // Convert to f32 bytes
        let f32_bytes: Vec<u8> = match tensor.dtype {
            1 => {
                // Already f32
                let slice =
                    unsafe { std::slice::from_raw_parts(tensor.data_f32(), len) };
                slice.iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            0 => {
                // f64 → f32
                let slice =
                    unsafe { std::slice::from_raw_parts(tensor.data_f64(), len) };
                slice
                    .iter()
                    .flat_map(|v| (*v as f32).to_le_bytes())
                    .collect()
            }
            other => {
                eprintln!("[nsl] safetensors_save: unknown dtype {other}");
                std::process::abort();
            }
        };

        owned.push((name, f32_bytes, shape));
    }

    // Build the metadata HashMap<String, TensorView> for safetensors
    // safetensors::serialize_to_file takes Iterator<Item = (&str, &TensorView)>
    // We use the low-level serialize + write approach instead.
    let data: HashMap<String, safetensors::tensor::TensorView<'_>> = owned
        .iter()
        .map(|(name, bytes, shape)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                shape.clone(),
                bytes.as_slice(),
            )
            .unwrap_or_else(|e| {
                eprintln!("[nsl] safetensors_save: TensorView error for '{}': {}", name, e);
                std::process::abort();
            });
            (name.clone(), view)
        })
        .collect();

    // Serialize to bytes and write file manually for better error messages
    let serialized = match safetensors::tensor::serialize(&data, &None) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[nsl] safetensors_save: serialize error: {}", e);
            std::process::abort();
        }
    };

    if let Err(e) = std::fs::write(path, &serialized) {
        eprintln!("[nsl] safetensors_save: write error for '{}': {}", path, e);
        std::process::abort();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;

    // Helper: write a minimal safetensors file to a temp path and return the path
    fn write_temp_safetensors(
        name: &str,
        shape: &[usize],
        data_f32: &[f32],
    ) -> tempfile::TempPath {
        let bytes: Vec<u8> = data_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.to_vec(),
            &bytes,
        )
        .unwrap();
        let mut map = HashMap::new();
        map.insert(name.to_string(), view);
        let serialized = safetensors::tensor::serialize(&map, &None).unwrap();

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&serialized).unwrap();
        tmp.into_temp_path()
    }

    fn write_temp_safetensors_f16(
        name: &str,
        shape: &[usize],
        data_f16: &[half::f16],
    ) -> tempfile::TempPath {
        let bytes: Vec<u8> = data_f16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F16,
            shape.to_vec(),
            &bytes,
        )
        .unwrap();
        let mut map = HashMap::new();
        map.insert(name.to_string(), view);
        let serialized = safetensors::tensor::serialize(&map, &None).unwrap();

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&serialized).unwrap();
        tmp.into_temp_path()
    }

    // ── test 1 ──────────────────────────────────────────────────────────────
    #[test]
    fn test_load_safetensors_single_tensor() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let path = write_temp_safetensors("weight", &[2, 2], &data);
        let path_str = path.to_str().unwrap();

        let dict = nsl_safetensors_load(
            path_str.as_ptr() as i64,
            path_str.len() as i64,
            0, // CPU
        );

        let count = crate::dict::nsl_dict_len(dict);
        assert_eq!(count, 1, "expected 1 entry in dict");

        // Retrieve the tensor by name
        let key = alloc_c_string("weight");
        let tensor_ptr = crate::dict::nsl_dict_get_str(dict, key as i64);
        let tensor = NslTensor::from_ptr(tensor_ptr);

        assert_eq!(tensor.dtype, 1, "dtype should be f32 (1)");
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.len, 4);
        unsafe {
            assert_eq!(*tensor.data_f32().add(0), 1.0f32);
            assert_eq!(*tensor.data_f32().add(3), 4.0f32);
        }
    }

    // ── test 2 ──────────────────────────────────────────────────────────────
    #[test]
    fn test_load_safetensors_f16_conversion() {
        let data_f16: Vec<half::f16> = vec![
            half::f16::from_f32(1.5),
            half::f16::from_f32(2.5),
            half::f16::from_f32(3.5),
        ];
        let path = write_temp_safetensors_f16("bias", &[3], &data_f16);
        let path_str = path.to_str().unwrap();

        let dict = nsl_safetensors_load(
            path_str.as_ptr() as i64,
            path_str.len() as i64,
            0,
        );

        let key = alloc_c_string("bias");
        let tensor_ptr = crate::dict::nsl_dict_get_str(dict, key as i64);
        let tensor = NslTensor::from_ptr(tensor_ptr);

        assert_eq!(tensor.dtype, 1, "dtype should be f32 (1)");
        assert_eq!(tensor.len, 3);

        let v0 = unsafe { *tensor.data_f32().add(0) };
        let v2 = unsafe { *tensor.data_f32().add(2) };
        assert!((v0 - 1.5f32).abs() < 0.01, "v0={v0}");
        assert!((v2 - 3.5f32).abs() < 0.01, "v2={v2}");
    }

    // ── test 3 ──────────────────────────────────────────────────────────────
    #[test]
    fn test_safetensors_round_trip() {
        // Create a dict with two tensors, save, reload, verify count matches.
        let data_a = vec![1.0f32, 2.0, 3.0];
        let data_b = vec![4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0];

        let path_a = write_temp_safetensors("a", &[3], &data_a);
        let path_b = write_temp_safetensors("b", &[2, 3], &data_b);

        // Load both into a single dict by loading separately and combining
        let path_a_str = path_a.to_str().unwrap();
        let path_b_str = path_b.to_str().unwrap();

        let dict_a = nsl_safetensors_load(
            path_a_str.as_ptr() as i64,
            path_a_str.len() as i64,
            0,
        );
        let dict_b = nsl_safetensors_load(
            path_b_str.as_ptr() as i64,
            path_b_str.len() as i64,
            0,
        );

        // Build a combined safetensors file with both tensors for round-trip
        let key_a_ptr = alloc_c_string("a");
        let key_b_ptr = alloc_c_string("b");
        let tensor_a_ptr = crate::dict::nsl_dict_get_str(dict_a, key_a_ptr as i64);
        let tensor_b_ptr = crate::dict::nsl_dict_get_str(dict_b, key_b_ptr as i64);

        // Create combined dict
        let combined = nsl_dict_new();
        nsl_dict_set_str(combined, key_a_ptr as i64, tensor_a_ptr);
        nsl_dict_set_str(combined, key_b_ptr as i64, tensor_b_ptr);

        assert_eq!(crate::dict::nsl_dict_len(combined), 2);

        // Save to temp file
        let out_tmp = tempfile::NamedTempFile::new().unwrap();
        let out_path = out_tmp.path().to_str().unwrap().to_owned();
        nsl_safetensors_save(
            combined,
            out_path.as_ptr() as i64,
            out_path.len() as i64,
        );

        // Reload and verify count
        let reloaded = nsl_safetensors_load(
            out_path.as_ptr() as i64,
            out_path.len() as i64,
            0,
        );
        let reloaded_count = crate::dict::nsl_dict_len(reloaded);
        assert_eq!(reloaded_count, 2, "round-trip should have 2 entries");

        // Verify values survived
        let key_a2 = alloc_c_string("a");
        let ta2 = NslTensor::from_ptr(crate::dict::nsl_dict_get_str(reloaded, key_a2 as i64));
        assert_eq!(ta2.len, 3);
        let v = unsafe { *ta2.data_f32().add(0) };
        assert!((v - 1.0f32).abs() < 1e-5, "a[0] should be 1.0, got {v}");
    }
}
