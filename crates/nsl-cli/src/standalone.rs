use std::path::Path;

pub enum EmbedMode {
    Auto,
    Always,
    Never,
}

// ---------------------------------------------------------------------------
// Weight tensor types
// ---------------------------------------------------------------------------

pub struct WeightTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
    /// Raw bytes in little-endian f64 format (8 bytes per element).
    pub data: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub enum WeightDtype {
    F32,
    F64,
}

// ---------------------------------------------------------------------------
// Task 2: build-time safetensors reading
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 half-precision (F16) bit pattern to f64.
fn f16_bits_to_f64(bits: u16) -> f64 {
    let sign: i32 = if bits >> 15 != 0 { -1 } else { 1 };
    let exponent = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x03FF) as i32;

    if exponent == 0x1F {
        // Inf or NaN
        if mantissa == 0 {
            return if sign > 0 { f64::INFINITY } else { f64::NEG_INFINITY };
        } else {
            return f64::NAN;
        }
    }

    let value: f64 = if exponent == 0 {
        // Subnormal
        (mantissa as f64) * 2.0_f64.powi(-24)
    } else {
        // Normal
        (1.0 + (mantissa as f64) * 2.0_f64.powi(-10)) * 2.0_f64.powi(exponent - 15)
    };

    (sign as f64) * value
}

/// Convert a BF16 bit pattern (upper 16 bits of f32) to f64.
fn bf16_bits_to_f64(bits: u16) -> f64 {
    // BF16 is simply f32 with the lower 16 bits zeroed out.
    let f32_bits: u32 = (bits as u32) << 16;
    f32::from_bits(f32_bits) as f64
}

/// Read a .safetensors file and return all tensors converted to f64 LE bytes,
/// sorted by name for deterministic ordering.
pub fn read_safetensors(path: &Path) -> Result<Vec<WeightTensor>, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("read_safetensors: cannot read {:?}: {}", path, e))?;

    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| format!("read_safetensors: parse error for {:?}: {}", path, e))?;

    let mut result: Vec<WeightTensor> = Vec::new();

    for (name, view) in tensors.tensors() {
        use safetensors::Dtype;

        let raw = view.data();
        let shape: Vec<usize> = view.shape().to_vec();
        let n_elems: usize = shape.iter().product::<usize>().max(1);

        let (dtype, data_bytes): (WeightDtype, Vec<u8>) = match view.dtype() {
            Dtype::F64 => {
                // Already in target format; just copy.
                if raw.len() != n_elems * 8 {
                    return Err(format!(
                        "read_safetensors: tensor '{}' F64 byte length mismatch ({} vs {}*8)",
                        name,
                        raw.len(),
                        n_elems
                    ));
                }
                (WeightDtype::F64, raw.to_vec())
            }
            Dtype::F32 => {
                if raw.len() != n_elems * 4 {
                    return Err(format!(
                        "read_safetensors: tensor '{}' F32 byte length mismatch ({} vs {}*4)",
                        name,
                        raw.len(),
                        n_elems
                    ));
                }
                let mut out = Vec::with_capacity(n_elems * 8);
                for chunk in raw.chunks_exact(4) {
                    let v = f32::from_le_bytes(chunk.try_into().unwrap()) as f64;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                (WeightDtype::F64, out)
            }
            Dtype::F16 => {
                if raw.len() != n_elems * 2 {
                    return Err(format!(
                        "read_safetensors: tensor '{}' F16 byte length mismatch ({} vs {}*2)",
                        name,
                        raw.len(),
                        n_elems
                    ));
                }
                let mut out = Vec::with_capacity(n_elems * 8);
                for chunk in raw.chunks_exact(2) {
                    let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                    let v = f16_bits_to_f64(bits);
                    out.extend_from_slice(&v.to_le_bytes());
                }
                (WeightDtype::F64, out)
            }
            Dtype::BF16 => {
                if raw.len() != n_elems * 2 {
                    return Err(format!(
                        "read_safetensors: tensor '{}' BF16 byte length mismatch ({} vs {}*2)",
                        name,
                        raw.len(),
                        n_elems
                    ));
                }
                let mut out = Vec::with_capacity(n_elems * 8);
                for chunk in raw.chunks_exact(2) {
                    let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                    let v = bf16_bits_to_f64(bits);
                    out.extend_from_slice(&v.to_le_bytes());
                }
                (WeightDtype::F64, out)
            }
            other => {
                return Err(format!(
                    "read_safetensors: tensor '{}' has unsupported dtype {:?}",
                    name, other
                ));
            }
        };

        result.push(WeightTensor {
            name: name.to_string(),
            shape,
            dtype,
            data: data_bytes,
        });
    }

    // Sort by name for deterministic ordering.
    result.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(result)
}

// ---------------------------------------------------------------------------
// Task 3: .nslweights format serialization
// ---------------------------------------------------------------------------

const NSLW_MAGIC: &[u8; 4] = b"NSLW";
const NSLW_VERSION: u32 = 1;

/// Serialize a list of weight tensors into the .nslweights binary format.
///
/// Layout:
///   [magic: 4 bytes] [version: u32 LE] [header_size: u64 LE]
///   [JSON header: header_size bytes] [padding to 64-byte alignment]
///   [raw tensor data concatenated]
///
/// JSON header format: `{"tensors":[{"name":"...","shape":[...],"dtype":"f64","offset":0,"nbytes":...},...]}`
pub fn serialize_nslweights(tensors: &[WeightTensor]) -> Vec<u8> {
    // Build JSON header entries while tracking per-tensor byte offsets.
    let mut entries: Vec<String> = Vec::with_capacity(tensors.len());
    let mut data_offset: u64 = 0;

    for t in tensors {
        let nbytes = t.data.len() as u64;
        let dtype_str = match t.dtype {
            WeightDtype::F32 => "f32",
            WeightDtype::F64 => "f64",
        };
        let shape_json = {
            let parts: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
            format!("[{}]", parts.join(","))
        };
        entries.push(format!(
            r#"{{"name":"{}","shape":{},"dtype":"{}","offset":{},"nbytes":{}}}"#,
            t.name, shape_json, dtype_str, data_offset, nbytes
        ));
        data_offset += nbytes;
    }

    let header = format!(r#"{{"params":[{}]}}"#, entries.join(","));
    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    // Fixed prefix: 4 (magic) + 4 (version) + 8 (header_size) = 16 bytes.
    let total_before_pad = 16 + header_bytes.len();
    let padding = (64 - (total_before_pad % 64)) % 64;

    let total_data: usize = tensors.iter().map(|t| t.data.len()).sum();
    let mut out = Vec::with_capacity(total_before_pad + padding + total_data);

    out.extend_from_slice(NSLW_MAGIC);
    out.extend_from_slice(&NSLW_VERSION.to_le_bytes());
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);

    // Padding zeros to reach 64-byte alignment.
    out.resize(out.len() + padding, 0u8);

    // Raw tensor data concatenated in header order.
    for t in tensors {
        out.extend_from_slice(&t.data);
    }

    out
}

/// Write raw .nslweights bytes to a sidecar file at the given path.
pub fn write_nslweights_sidecar_raw(data: &[u8], path: &Path) -> Result<(), String> {
    std::fs::write(path, data)
        .map_err(|e| format!("write_nslweights_sidecar_raw: cannot write {:?}: {}", path, e))
}
