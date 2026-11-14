//! `-w`: the weights file a program was compiled against, opened at startup.
//!
//! A program built with `-w weights.safetensors` runs with that file's
//! values: `main` calls [`nsl_weights_bind`] before any user code, and every
//! model constructor calls [`nsl_weights_load_field`] for each field the file
//! names. The compiler may have rewritten code on the strength of some of
//! those values (a matmul by an identity weight dropped, a product of two
//! weights folded into a constant), so the bind call carries the content
//! hash of every tensor it was allowed to do that with, and refuses a file
//! in which one of them differs: an edited file is an error, not a wrong
//! answer.
//!
//! The hash is the one `nsl_codegen::weight_aware::tensor_content_hash`
//! computes: SHA-256 over the dtype as the file spells it, a zero byte, each
//! dimension as a little-endian u64, and the stored bytes.

use std::collections::HashMap;
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

use crate::tensor::NslTensor;

/// One tensor of the file: where its bytes are.
#[derive(Debug, Clone, PartialEq)]
struct Stored {
    dtype: String,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

/// The opened file.
struct Bound {
    path: String,
    bytes: memmap2::Mmap,
    tensors: HashMap<String, Stored>,
}

static BOUND: OnceLock<Bound> = OnceLock::new();

#[derive(serde::Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Parse a safetensors file's header: an 8-byte little-endian length, then
/// that many bytes of JSON mapping each tensor name to its dtype, shape and
/// byte range within the data section that follows.
fn parse_header(bytes: &[u8]) -> Result<HashMap<String, Stored>, String> {
    let len_bytes: [u8; 8] = bytes
        .get(..8)
        .and_then(|b| b.try_into().ok())
        .ok_or("the file is shorter than a safetensors header")?;
    let header_len = u64::from_le_bytes(len_bytes);
    let data_start = 8u64
        .checked_add(header_len)
        .filter(|end| *end <= bytes.len() as u64)
        .ok_or("the header claims more bytes than the file holds")? as usize;
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&bytes[8..data_start])
        .map_err(|e| format!("the header is not valid JSON: {e}"))?;
    let mut tensors = HashMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let entry: HeaderEntry = serde_json::from_value(value)
            .map_err(|e| format!("the header entry for '{name}' is malformed: {e}"))?;
        let [begin, end] = entry.data_offsets;
        let (begin, end) = (
            data_start.saturating_add(begin),
            data_start.saturating_add(end),
        );
        if begin > end || end > bytes.len() {
            return Err(format!("the data range of '{name}' lies outside the file"));
        }
        tensors.insert(
            name,
            Stored {
                dtype: entry.dtype,
                shape: entry.shape,
                begin,
                end,
            },
        );
    }
    Ok(tensors)
}

fn content_hash(stored: &Stored, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(stored.dtype.as_bytes());
    hasher.update([0u8]);
    for dim in &stored.shape {
        hasher.update((*dim as u64).to_le_bytes());
    }
    hasher.update(&bytes[stored.begin..stored.end]);
    hasher.finalize().into()
}

/// The `(name, hash)` pairs of an expectation table: for each a
/// little-endian u32 name length, the name, and 32 bytes of hash.
fn decode_expectations(mut table: &[u8]) -> Result<Vec<(String, [u8; 32])>, String> {
    let mut out = Vec::new();
    while !table.is_empty() {
        let len_bytes: [u8; 4] = table
            .get(..4)
            .and_then(|b| b.try_into().ok())
            .ok_or("truncated name length")?;
        let len = u32::from_le_bytes(len_bytes) as usize;
        let name = table.get(4..4 + len).ok_or("truncated name")?;
        let hash: [u8; 32] = table
            .get(4 + len..4 + len + 32)
            .and_then(|b| b.try_into().ok())
            .ok_or("truncated hash")?;
        out.push((String::from_utf8_lossy(name).into_owned(), hash));
        table = &table[4 + len + 32..];
    }
    Ok(out)
}

/// The first expected tensor the file no longer matches, as the message to
/// refuse with.
fn first_mismatch(
    path: &str,
    bytes: &[u8],
    tensors: &HashMap<String, Stored>,
    expected: &[(String, [u8; 32])],
) -> Option<String> {
    for (name, hash) in expected {
        let what = match tensors.get(name) {
            None => "is no longer in the file",
            Some(stored) if content_hash(stored, bytes) != *hash => "has different contents",
            Some(_) => continue,
        };
        return Some(format!(
            "the weights file '{path}' has changed since this program was compiled with it: \
             parameter '{name}' {what}. The compiled code depends on that parameter's values, \
             so it would compute wrong results with this file. Recompile with the file as it \
             is now."
        ));
    }
    None
}

fn refuse(message: &str) -> ! {
    crate::nsl_log!(ERROR, "nsl", "nsl: -w: {message}");
    std::process::abort();
}

/// Open the weights file the program was compiled with and check every
/// tensor the compiler may have specialized on.
///
/// `table` is the expectation table (`decode_expectations`); it may be empty.
///
/// # Safety
/// `path_ptr` must point to `path_len` bytes of UTF-8 and `table_ptr` to
/// `table_len` bytes (or be null with `table_len` zero).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsl_weights_bind(
    path_ptr: i64,
    path_len: i64,
    table_ptr: i64,
    table_len: i64,
) {
    let path = String::from_utf8_lossy(unsafe {
        std::slice::from_raw_parts(path_ptr as *const u8, path_len.max(0) as usize)
    })
    .into_owned();
    let table: &[u8] = if table_ptr == 0 || table_len <= 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(table_ptr as *const u8, table_len as usize) }
    };
    if BOUND.get().is_some() {
        return;
    }
    let file = std::fs::File::open(&path).unwrap_or_else(|e| {
        refuse(&format!(
            "cannot open the weights file '{path}' this program was compiled with: {e}"
        ))
    });
    // Mapped rather than read: the tensors are copied out of it one model
    // constructor at a time, for as long as the program constructs models.
    // SAFETY: the map is read-only and private to this process; the file is
    // one the user handed to the compiler, and a program that truncates its
    // own weights file while running it is outside what this can protect.
    let bytes = unsafe { memmap2::Mmap::map(&file) }
        .unwrap_or_else(|e| refuse(&format!("cannot map the weights file '{path}': {e}")));
    let tensors = parse_header(&bytes)
        .unwrap_or_else(|e| refuse(&format!("'{path}' is not a safetensors file: {e}")));
    let expected = decode_expectations(table)
        .unwrap_or_else(|e| refuse(&format!("the compiled expectation table is corrupt: {e}")));
    if let Some(message) = first_mismatch(&path, &bytes, &tensors, &expected) {
        refuse(&message);
    }
    let _ = BOUND.set(Bound {
        path,
        bytes,
        tensors,
    });
}

/// Read element `i` of a stored tensor as f64.
fn read_element(dtype: &str, bytes: &[u8], i: usize) -> Option<f64> {
    Some(match dtype {
        "F32" => f32::from_le_bytes(bytes.get(4 * i..4 * i + 4)?.try_into().ok()?) as f64,
        "F64" => f64::from_le_bytes(bytes.get(8 * i..8 * i + 8)?.try_into().ok()?),
        "F16" => half::f16::from_le_bytes(bytes.get(2 * i..2 * i + 2)?.try_into().ok()?).to_f64(),
        "BF16" => half::bf16::from_le_bytes(bytes.get(2 * i..2 * i + 2)?.try_into().ok()?).to_f64(),
        _ => return None,
    })
}

/// The file's tensor `name`, converted to the element type of a model field
/// (`dtype` 0 is f64, 1 is f32), as the bytes to copy into it. `shape` is the
/// field's shape.
fn field_bytes(
    stored: &Stored,
    bytes: &[u8],
    name: &str,
    shape: &[i64],
    dtype: u16,
) -> Result<Vec<u8>, String> {
    let stored_shape: Vec<i64> = stored.shape.iter().map(|d| *d as i64).collect();
    if stored_shape != shape {
        return Err(format!(
            "'{name}' has shape {stored_shape:?} in the file but {shape:?} in the model"
        ));
    }
    let count: usize = stored.shape.iter().product();
    let data = &bytes[stored.begin..stored.end];
    let mut out = Vec::with_capacity(
        count
            * if dtype == crate::tensor::DTYPE_F32 {
                4
            } else {
                8
            },
    );
    for i in 0..count {
        let value = read_element(&stored.dtype, data, i).ok_or_else(|| {
            format!(
                "'{name}' is stored as {} ({} bytes for {count} elements), which cannot be \
                 loaded into a float tensor",
                stored.dtype,
                data.len()
            )
        })?;
        match dtype {
            crate::tensor::DTYPE_F32 => out.extend_from_slice(&(value as f32).to_le_bytes()),
            crate::tensor::DTYPE_F64 => out.extend_from_slice(&value.to_le_bytes()),
            other => {
                return Err(format!(
                    "'{name}': the model field has dtype tag {other}, and only f32 and f64 \
                     fields are loaded from a weights file"
                ));
            }
        }
    }
    Ok(out)
}

/// Copy the file's tensor `name` into the model field `tensor`, in place.
///
/// With `strict` zero a tensor whose shape is not the field's is left
/// alone: the compiler passes that for a bare name more than one parameter
/// could claim, where the shape is what says whose it is.
///
/// # Safety
/// `tensor` must be a live tensor handle and `name_ptr` must point to
/// `name_len` bytes of UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsl_weights_load_field(
    tensor: i64,
    name_ptr: i64,
    name_len: i64,
    strict: i64,
) {
    let name = String::from_utf8_lossy(unsafe {
        std::slice::from_raw_parts(name_ptr as *const u8, name_len.max(0) as usize)
    })
    .into_owned();
    let Some(bound) = BOUND.get() else {
        refuse(&format!(
            "a model built with -w was constructed before the weights file was opened \
             (parameter '{name}'); this program has no `main` that opens it"
        ));
    };
    let Some(stored) = bound.tensors.get(&name) else {
        refuse(&format!(
            "the weights file '{}' no longer has the parameter '{name}' this program loads \
             from it",
            bound.path
        ));
    };
    let field = NslTensor::from_ptr(tensor);
    if !field.is_contiguous() {
        refuse(&format!(
            "'{name}': the model field is a strided view and cannot be loaded into"
        ));
    }
    let shape: Vec<i64> = (0..field.ndim as usize)
        .map(|d| unsafe { *field.shape.add(d) })
        .collect();
    if strict == 0
        && stored
            .shape
            .iter()
            .map(|d| *d as i64)
            .ne(shape.iter().copied())
    {
        return;
    }
    let data = field_bytes(stored, &bound.bytes, &name, &shape, field.dtype)
        .unwrap_or_else(|e| refuse(&format!("loading '{}': {e}", bound.path)));
    if field.device > 0 {
        #[cfg(feature = "cuda")]
        crate::cuda::inner::memcpy_htod(
            field.data,
            data.as_ptr() as *const std::ffi::c_void,
            data.len(),
        );
        #[cfg(not(feature = "cuda"))]
        refuse(&format!(
            "'{name}' is on a GPU but this runtime was built without CUDA"
        ));
    } else {
        // SAFETY: the field is contiguous, its shape equals the stored one,
        // and `data` holds exactly `len` elements of its dtype.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), field.data as *mut u8, data.len()) };
    }
}

/// A constant the compiler folded from two weights (an f64 host tensor),
/// as the operation it replaced would have produced it: in the dtype of
/// `like` (one of that operation's operands) and on its device. Returns an
/// owned reference; `constant` is left to its owner.
///
/// The compiler cannot know either at compile time: the parameters' dtype is
/// the model's (a BF16 file loads into f32 fields), and the model may have
/// been moved to a GPU.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_weights_const_like(constant: i64, like: i64) -> i64 {
    use crate::tensor::{DTYPE_BF16, DTYPE_F32, DTYPE_F64, DTYPE_FP16};
    let target = NslTensor::from_ptr_ref(like).dtype;
    let source = NslTensor::from_ptr_ref(constant).dtype;
    if !matches!(target, DTYPE_F64 | DTYPE_F32 | DTYPE_FP16 | DTYPE_BF16) {
        refuse(&format!(
            "a constant folded from the weights file cannot be given the dtype (tag {target}) \
             of the parameters it replaces; recompile with --no-constant-fold"
        ));
    }
    let converted = if source == target {
        constant
    } else {
        // Not taped: the fold is made only when no parameter is
        // differentiated or written.
        crate::tensor::precision_cast::convert_untaped(constant, target)
    };
    let placed = crate::tensor::nsl_tensor_to_device_like(converted, like);
    if converted != constant {
        crate::tensor::nsl_tensor_free(converted);
    }
    placed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A safetensors file with the given tensors, written out by hand.
    fn file(tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut data: Vec<u8> = Vec::new();
        for (i, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let dims = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{dims}],\"data_offsets\":[{},{}]}}",
                data.len(),
                data.len() + bytes.len()
            ));
            data.extend_from_slice(bytes);
        }
        header.push('}');
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        out
    }

    fn f32s(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn the_header_gives_each_tensor_its_bytes() {
        let bytes = file(&[
            ("a", "F32", &[2], f32s(&[1.0, 2.0])),
            ("b", "F32", &[1, 3], f32s(&[3.0, 4.0, 5.0])),
        ]);
        let tensors = parse_header(&bytes).unwrap();
        let b = &tensors["b"];
        assert_eq!(
            (b.dtype.as_str(), b.shape.as_slice()),
            ("F32", &[1usize, 3][..])
        );
        assert_eq!(&bytes[b.begin..b.end], f32s(&[3.0, 4.0, 5.0]).as_slice());
        assert_eq!(tensors["a"].end - tensors["a"].begin, 8);
    }

    #[test]
    fn a_range_outside_the_file_is_refused() {
        let mut bytes = file(&[("a", "F32", &[2], f32s(&[1.0, 2.0]))]);
        bytes.truncate(bytes.len() - 1);
        assert!(
            parse_header(&bytes)
                .unwrap_err()
                .contains("outside the file")
        );
        assert!(parse_header(&[1, 2, 3]).is_err());
    }

    /// The hash of `[1.0f32, 2.0]` named F32 with shape [2], computed
    /// independently with Python's hashlib:
    ///   sha256(b"F32" + b"\0" + (2).to_bytes(8, "little")
    ///          + struct.pack("<2f", 1.0, 2.0))
    #[test]
    fn the_content_hash_is_the_documented_one() {
        let bytes = file(&[("a", "F32", &[2], f32s(&[1.0, 2.0]))]);
        let tensors = parse_header(&bytes).unwrap();
        let hex: String = content_hash(&tensors["a"], &bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            hex,
            "6774455a26a851dd86ea91ff83c714320b9746d476d63204492fb74fa11d32b3"
        );
    }

    #[test]
    fn a_changed_or_missing_tensor_is_named() {
        let original = file(&[
            ("w", "F32", &[2], f32s(&[1.0, 2.0])),
            ("v", "F32", &[1], f32s(&[7.0])),
        ]);
        let tensors = parse_header(&original).unwrap();
        let expected = vec![
            ("v".to_string(), content_hash(&tensors["v"], &original)),
            ("w".to_string(), content_hash(&tensors["w"], &original)),
        ];
        assert_eq!(first_mismatch("f", &original, &tensors, &expected), None);

        // One value of `w` edited: `w` is named, `v` is not.
        let edited = file(&[
            ("w", "F32", &[2], f32s(&[1.0, 2.5])),
            ("v", "F32", &[1], f32s(&[7.0])),
        ]);
        let message =
            first_mismatch("f", &edited, &parse_header(&edited).unwrap(), &expected).unwrap();
        assert!(
            message.contains("parameter 'w' has different contents"),
            "{message}"
        );

        // The same bytes under another shape are a different tensor.
        let reshaped = file(&[
            ("w", "F32", &[1, 2], f32s(&[1.0, 2.0])),
            ("v", "F32", &[1], f32s(&[7.0])),
        ]);
        assert!(
            first_mismatch("f", &reshaped, &parse_header(&reshaped).unwrap(), &expected).is_some()
        );

        let without = file(&[("w", "F32", &[2], f32s(&[1.0, 2.0]))]);
        let message =
            first_mismatch("f", &without, &parse_header(&without).unwrap(), &expected).unwrap();
        assert!(
            message.contains("parameter 'v' is no longer in the file"),
            "{message}"
        );
    }

    #[test]
    fn the_expectation_table_round_trips() {
        let mut table = Vec::new();
        for (name, fill) in [("Toy.w", 7u8), ("b", 9u8)] {
            table.extend_from_slice(&(name.len() as u32).to_le_bytes());
            table.extend_from_slice(name.as_bytes());
            table.extend_from_slice(&[fill; 32]);
        }
        assert_eq!(
            decode_expectations(&table).unwrap(),
            vec![
                ("Toy.w".to_string(), [7u8; 32]),
                ("b".to_string(), [9u8; 32])
            ]
        );
        assert!(decode_expectations(&table[..table.len() - 1]).is_err());
    }

    /// The folded constant takes the operand's dtype: an f64 [1.5, -2.25]
    /// against an f32 operand comes back f32 with the same values, and
    /// against an f64 operand comes back as the constant itself.
    #[test]
    fn a_folded_constant_takes_the_operand_s_dtype() {
        let shape = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape, 2);
        let constant = crate::tensor::creation::nsl_tensor_zeros_dtype(
            shape,
            i64::from(crate::tensor::DTYPE_F64),
        );
        let like32 = crate::tensor::creation::nsl_tensor_zeros_dtype(
            shape,
            i64::from(crate::tensor::DTYPE_F32),
        );
        let like64 = crate::tensor::creation::nsl_tensor_zeros_dtype(
            shape,
            i64::from(crate::tensor::DTYPE_F64),
        );
        let c = NslTensor::from_ptr(constant);
        unsafe {
            *(c.data as *mut f64) = 1.5;
            *(c.data as *mut f64).add(1) = -2.25;
        }
        let out = nsl_weights_const_like(constant, like32);
        let o = NslTensor::from_ptr(out);
        assert_eq!(o.dtype, crate::tensor::DTYPE_F32);
        let got = unsafe { [*(o.data as *const f32), *(o.data as *const f32).add(1)] };
        assert_eq!(got, [1.5f32, -2.25]);
        assert_eq!(nsl_weights_const_like(constant, like64), constant);
        for t in [out, constant, constant, like32, like64] {
            crate::tensor::nsl_tensor_free(t);
        }
        crate::list::nsl_list_free(shape);
    }

    #[test]
    fn a_stored_tensor_converts_to_the_field_s_float_width() {
        let bytes = file(&[
            ("w", "F32", &[2], f32s(&[1.5, -2.25])),
            ("h", "F16", &[2], [0x00, 0x3c, 0x00, 0xc0].to_vec()), // 1.0, -2.0
            ("i", "I32", &[1], 5i32.to_le_bytes().to_vec()),
        ]);
        let tensors = parse_header(&bytes).unwrap();
        let as_f64 =
            field_bytes(&tensors["w"], &bytes, "w", &[2], crate::tensor::DTYPE_F64).unwrap();
        assert_eq!(
            as_f64,
            [1.5f64.to_le_bytes(), (-2.25f64).to_le_bytes()].concat()
        );
        let as_f32 =
            field_bytes(&tensors["h"], &bytes, "h", &[2], crate::tensor::DTYPE_F32).unwrap();
        assert_eq!(as_f32, f32s(&[1.0, -2.0]));
        // Wrong shape and a non-float dtype are refused with the names.
        let err = field_bytes(
            &tensors["w"],
            &bytes,
            "w",
            &[1, 2],
            crate::tensor::DTYPE_F32,
        )
        .unwrap_err();
        assert!(
            err.contains("[2] in the file but [1, 2] in the model"),
            "{err}"
        );
        let err =
            field_bytes(&tensors["i"], &bytes, "i", &[1], crate::tensor::DTYPE_F32).unwrap_err();
        assert!(err.contains("stored as I32"), "{err}");
    }
}
