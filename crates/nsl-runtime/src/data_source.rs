//! Data source readers: JSONL, CSV, and memory-mapped binary files.
//!
//! These are small-data utilities for loading datasets into NSL programs.
//! For large-scale training data, prefer the streaming DataLoader pipeline.

use std::ffi::c_void;
use std::fs;
use std::io::{BufRead, BufReader};

use crate::list::{nsl_list_new, nsl_list_push};
use crate::memory::checked_alloc;
use crate::string::nsl_str_from_rust;
use crate::tensor::NslTensor;

/// Convert a (ptr, len) pair from the NSL ABI into a Rust `String`.
unsafe fn str_from_ptr_len(ptr: i64, len: i64) -> String {
    let slice = std::slice::from_raw_parts(ptr as *const u8, len as usize);
    String::from_utf8_lossy(slice).into_owned()
}

/// Create an NslTensor with manually specified data pointer, length, dtype, and ownership.
fn create_mmap_tensor(data: *mut c_void, len: i64, dtype: u16, owns_data: u8) -> i64 {
    let shape_ptr = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape_ptr = len };
    let strides = NslTensor::compute_strides(shape_ptr, 1);

    let tensor = Box::new(NslTensor {
        data,
        shape: shape_ptr,
        strides,
        ndim: 1,
        len,
        refcount: 1,
        device: 0,
        dtype,
        owns_data,
    });
    Box::into_raw(tensor) as i64
}

// ---------------------------------------------------------------------------
// JSONL reader
// ---------------------------------------------------------------------------

/// Load a JSONL file, extracting a named field from each line as a string.
///
/// Returns an NslList of NSL string pointers.
#[no_mangle]
pub extern "C" fn nsl_load_jsonl(
    path_ptr: i64,
    path_len: i64,
    field_ptr: i64,
    field_len: i64,
) -> i64 {
    let path = unsafe { str_from_ptr_len(path_ptr, path_len) };
    let field = unsafe { str_from_ptr_len(field_ptr, field_len) };

    let file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("nsl: nsl_load_jsonl: cannot open '{}': {}", path, e);
            return nsl_list_new();
        }
    };

    let list = nsl_list_new();
    let reader = BufReader::new(file);

    for (line_no, line_result) in reader.lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("nsl: nsl_load_jsonl: read error at line {}: {}", line_no + 1, e);
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let obj: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "nsl: nsl_load_jsonl: malformed JSON at line {}: {}",
                    line_no + 1,
                    e
                );
                continue;
            }
        };
        if let Some(val) = obj.get(&field) {
            let s = match val {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let nsl_s = nsl_str_from_rust(&s);
            nsl_list_push(list, nsl_s);
        }
    }

    list
}

// ---------------------------------------------------------------------------
// CSV reader
// ---------------------------------------------------------------------------

/// Parse a single CSV line, handling double-quoted fields and escaped quotes.
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    // escaped quote
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == ',' {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

/// Load a CSV file, extracting a column by index.
///
/// `has_header`: 1 = skip first line, 0 = no header.
/// Returns an NslList of NSL string pointers.
#[no_mangle]
pub extern "C" fn nsl_load_csv(
    path_ptr: i64,
    path_len: i64,
    col_idx: i64,
    has_header: i64,
) -> i64 {
    let path = unsafe { str_from_ptr_len(path_ptr, path_len) };

    let file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("nsl: nsl_load_csv: cannot open '{}': {}", path, e);
            return nsl_list_new();
        }
    };

    let list = nsl_list_new();
    let reader = BufReader::new(file);
    let col = col_idx as usize;

    for (line_no, line_result) in reader.lines().enumerate() {
        // Skip header row
        if has_header == 1 && line_no == 0 {
            // consume and discard
            let _ = line_result;
            continue;
        }
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("nsl: nsl_load_csv: read error at line {}: {}", line_no + 1, e);
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields = parse_csv_line(trimmed);
        if col < fields.len() {
            let nsl_s = nsl_str_from_rust(&fields[col]);
            nsl_list_push(list, nsl_s);
        }
    }

    list
}

// ---------------------------------------------------------------------------
// Memory-mapped binary reader
// ---------------------------------------------------------------------------

/// Memory-map a binary file as a flat 1D tensor.
///
/// `dtype`: 0 = f64, 1 = f32, 2 = i32, 3 = u16.
///
/// For f64 and f32, the tensor data points directly into the mmap region (zero-copy).
/// For i32 and u16, values are converted to f64 and stored in a heap-allocated buffer.
#[no_mangle]
pub extern "C" fn nsl_load_mmap(path_ptr: i64, path_len: i64, dtype: i64) -> i64 {
    let path = unsafe { str_from_ptr_len(path_ptr, path_len) };

    let file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("nsl: nsl_load_mmap: cannot open '{}': {}", path, e);
            std::process::abort();
        }
    };

    let mmap = match unsafe { memmap2::Mmap::map(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("nsl: nsl_load_mmap: mmap failed for '{}': {}", path, e);
            std::process::abort();
        }
    };

    let byte_len = mmap.len();

    match dtype {
        0 => {
            // f64: zero-copy
            let elem_size = std::mem::size_of::<f64>();
            let n_elements = byte_len / elem_size;
            let data_ptr = mmap.as_ptr() as *mut c_void;
            // Leak the mmap handle so the memory stays valid for the process lifetime.
            let _ = Box::into_raw(Box::new(mmap));
            create_mmap_tensor(data_ptr, n_elements as i64, 0, 0)
        }
        1 => {
            // f32: zero-copy
            let elem_size = std::mem::size_of::<f32>();
            let n_elements = byte_len / elem_size;
            let data_ptr = mmap.as_ptr() as *mut c_void;
            let _ = Box::into_raw(Box::new(mmap));
            create_mmap_tensor(data_ptr, n_elements as i64, 1, 0)
        }
        2 => {
            // i32 -> f64 conversion
            let elem_size = std::mem::size_of::<i32>();
            let n_elements = byte_len / elem_size;
            let out_buf = checked_alloc(n_elements * std::mem::size_of::<f64>()) as *mut f64;
            let src = mmap.as_ptr() as *const i32;
            for i in 0..n_elements {
                unsafe {
                    *out_buf.add(i) = *src.add(i) as f64;
                }
            }
            create_mmap_tensor(out_buf as *mut c_void, n_elements as i64, 0, 1)
        }
        3 => {
            // u16 -> f64 conversion (pre-tokenized LLM datasets)
            let elem_size = std::mem::size_of::<u16>();
            let n_elements = byte_len / elem_size;
            let out_buf = checked_alloc(n_elements * std::mem::size_of::<f64>()) as *mut f64;
            let src = mmap.as_ptr() as *const u16;
            for i in 0..n_elements {
                unsafe {
                    *out_buf.add(i) = *src.add(i) as f64;
                }
            }
            create_mmap_tensor(out_buf as *mut c_void, n_elements as i64, 0, 1)
        }
        _ => {
            eprintln!("nsl: nsl_load_mmap: unsupported dtype {}", dtype);
            std::process::abort();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_jsonl() {
        let dir = std::env::temp_dir().join("nsl_test_jsonl");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test.jsonl");
        {
            let mut f = fs::File::create(&path).unwrap();
            writeln!(f, r#"{{"text": "hello", "label": 1}}"#).unwrap();
            writeln!(f, r#"{{"text": "world", "label": 2}}"#).unwrap();
            writeln!(f, r#"{{"text": "foo", "label": 3}}"#).unwrap();
        }

        let path_str = path.to_str().unwrap();
        let field = "text";
        let list = nsl_load_jsonl(
            path_str.as_ptr() as i64,
            path_str.len() as i64,
            field.as_ptr() as i64,
            field.len() as i64,
        );
        assert_eq!(crate::list::nsl_list_len(list), 3);
    }

    #[test]
    fn test_load_csv() {
        let dir = std::env::temp_dir().join("nsl_test_csv");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test.csv");
        {
            let mut f = fs::File::create(&path).unwrap();
            writeln!(f, "name,age,city").unwrap();
            writeln!(f, "Alice,30,NYC").unwrap();
            writeln!(f, "Bob,25,LA").unwrap();
        }

        let path_str = path.to_str().unwrap();
        let list = nsl_load_csv(
            path_str.as_ptr() as i64,
            path_str.len() as i64,
            0, // column 0 = name
            1, // has header
        );
        assert_eq!(crate::list::nsl_list_len(list), 2);
    }

    #[test]
    fn test_load_mmap_f64() {
        let dir = std::env::temp_dir().join("nsl_test_mmap");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_f64.bin");
        {
            let data: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, 4 * std::mem::size_of::<f64>())
            };
            fs::write(&path, bytes).unwrap();
        }

        let path_str = path.to_str().unwrap();
        let tensor_ptr = nsl_load_mmap(
            path_str.as_ptr() as i64,
            path_str.len() as i64,
            0, // dtype = f64
        );
        let tensor = NslTensor::from_ptr(tensor_ptr);
        assert_eq!(tensor.len, 4);
        assert_eq!(tensor.owns_data, 0);
        assert_eq!(tensor.dtype, 0);
    }

    #[test]
    fn test_load_mmap_u16() {
        let dir = std::env::temp_dir().join("nsl_test_mmap");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_u16.bin");
        {
            let data: [u16; 4] = [100, 200, 50256, 42];
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    4 * std::mem::size_of::<u16>(),
                )
            };
            fs::write(&path, bytes).unwrap();
        }

        let path_str = path.to_str().unwrap();
        let tensor_ptr = nsl_load_mmap(
            path_str.as_ptr() as i64,
            path_str.len() as i64,
            3, // dtype = u16
        );
        let tensor = NslTensor::from_ptr(tensor_ptr);
        assert_eq!(tensor.len, 4);
        assert_eq!(tensor.owns_data, 1);
        assert_eq!(tensor.dtype, 0); // converted to f64
        unsafe {
            let data = tensor.data_f64();
            assert_eq!(*data.add(2), 50256.0);
        }
    }
}
