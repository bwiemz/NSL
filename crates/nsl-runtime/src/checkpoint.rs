use crate::list::NslList;
use crate::tensor::NslTensor;
use std::io::Write;

const MAGIC: &[u8; 4] = b"NSLM";
const VERSION: u32 = 1;

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
    assert_eq!(
        names.len, tensors.len,
        "model_save: name/tensor count mismatch"
    );

    // Build JSON header
    let mut params_json = Vec::new();
    let mut data_offset: u64 = 0;
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        assert_tensor_contiguous(tensor, i);
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

    let mut file = std::fs::File::create(path).expect("model_save: cannot create file");
    file.write_all(MAGIC).unwrap();
    file.write_all(&VERSION.to_le_bytes()).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header_bytes).unwrap();

    // Pad to 64-byte alignment
    let total_header = 4 + 4 + 8 + header_bytes.len();
    let padding = (64 - (total_header % 64)) % 64;
    file.write_all(&vec![0u8; padding]).unwrap();

    // Raw tensor data (little-endian f64)
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        for j in 0..tensor.len as usize {
            let val = unsafe { *tensor.data.add(j) };
            file.write_all(&val.to_le_bytes()).unwrap();
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
    let data = std::fs::read(path).expect("model_load: cannot read file");

    assert_eq!(&data[0..4], MAGIC, "model_load: invalid .nslm file");
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    assert_eq!(version, VERSION, "model_load: unsupported version");
    let header_size = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;

    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;

    let mut offset = data_start;
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        for j in 0..tensor.len as usize {
            let val = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            unsafe {
                *tensor.data.add(j) = val;
            }
            offset += 8;
        }
    }
}

fn assert_tensor_contiguous(tensor: &NslTensor, idx: usize) {
    if tensor.ndim <= 1 {
        return;
    }
    let mut expected_stride = 1i64;
    for d in (0..tensor.ndim as usize).rev() {
        let actual = unsafe { *tensor.strides.add(d) };
        assert_eq!(
            actual, expected_stride,
            "model_save: parameter {} is not contiguous (dim {} stride {} expected {})",
            idx, d, actual, expected_stride
        );
        expected_stride *= unsafe { *tensor.shape.add(d) };
    }
}
