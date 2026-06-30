//! `nsl convert` — model weight format conversion between NSLM and safetensors.
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use std::process;

pub(crate) fn convert_nslm_to_safetensors(input_path: &std::path::Path, output_path: &std::path::Path) -> Result<(), String> {
    use std::collections::HashMap;

    let data = std::fs::read(input_path)
        .map_err(|e| format!("cannot read '{}': {}", input_path.display(), e))?;

    if data.len() < 16 {
        return Err(format!(
            "'{}' is too small to be a valid NSLM file ({} bytes)",
            input_path.display(), data.len()
        ));
    }
    if &data[0..4] != b"NSLM" {
        return Err(format!("'{}' is not a valid NSLM file (bad magic)", input_path.display()));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 1 {
        return Err(format!(
            "unsupported NSLM version {} in '{}' (expected 1)",
            version, input_path.display()
        ));
    }
    let header_size = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    if 16 + header_size > data.len() {
        return Err(format!(
            "NSLM header claims {} bytes but file is only {} bytes",
            header_size, data.len()
        ));
    }
    let header_json: serde_json::Value = serde_json::from_slice(&data[16..16 + header_size])
        .map_err(|e| format!("NSLM header JSON parse error: {}", e))?;

    let params = header_json["params"]
        .as_array()
        .ok_or_else(|| "NSLM header missing 'params' array".to_string())?;

    // Compute start of raw data (64-byte aligned from byte 16+header_size)
    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;

    // Build safetensors entries: convert f64 → f32
    let mut owned: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::with_capacity(params.len());

    for param in params {
        let name = param["name"]
            .as_str()
            .ok_or_else(|| "NSLM param missing 'name'".to_string())?
            .to_owned();
        let dtype_str = param["dtype"]
            .as_str()
            .ok_or_else(|| format!("NSLM param '{}' missing 'dtype'", name))?;
        let offset = param["offset"]
            .as_u64()
            .ok_or_else(|| format!("NSLM param '{}' missing 'offset'", name))? as usize;
        let nbytes = param["nbytes"]
            .as_u64()
            .ok_or_else(|| format!("NSLM param '{}' missing 'nbytes'", name))? as usize;
        let shape: Vec<usize> = param["shape"]
            .as_array()
            .ok_or_else(|| format!("NSLM param '{}' missing 'shape'", name))?
            .iter()
            .map(|v| v.as_i64().unwrap_or(0) as usize)
            .collect();

        let abs_start = data_start + offset;
        let abs_end = abs_start + nbytes;
        if abs_end > data.len() {
            return Err(format!(
                "NSLM tensor '{}' data [{}..{}] exceeds file size {}",
                name, abs_start, abs_end, data.len()
            ));
        }
        let raw = &data[abs_start..abs_end];

        // Convert to f32 LE bytes for safetensors
        let f32_bytes: Vec<u8> = match dtype_str {
            "f64" => {
                let count = raw.len() / 8;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 8] = raw[i * 8..(i + 1) * 8].try_into().unwrap();
                    let v = f64::from_le_bytes(b) as f32;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            "f32" => raw.to_vec(),
            other => {
                return Err(format!("NSLM tensor '{}' has unsupported dtype '{}'", name, other));
            }
        };

        owned.push((name, f32_bytes, shape));
    }

    // Serialize to safetensors
    let st_data: HashMap<String, safetensors::tensor::TensorView<'_>> = owned
        .iter()
        .map(|(name, bytes, shape)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                shape.clone(),
                bytes.as_slice(),
            )
            .map_err(|e| format!("safetensors TensorView error for '{}': {}", name, e))?;
            Ok((name.clone(), view))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;

    let serialized = safetensors::tensor::serialize(&st_data, &None)
        .map_err(|e| format!("safetensors serialize error: {}", e))?;

    std::fs::write(output_path, &serialized)
        .map_err(|e| format!("cannot write '{}': {}", output_path.display(), e))?;

    Ok(())
}

/// Shard of metadata used when writing NSLM header.
#[derive(Debug)]
struct NslmParamMeta {
    name: String,
    shape: Vec<i64>,
    dtype: &'static str,
    offset: u64,
    nbytes: u64,
}

fn convert_safetensors_to_nslm(input_path: &std::path::Path, output_path: &std::path::Path) -> Result<(), String> {
    use std::io::Write;

    let bytes = std::fs::read(input_path)
        .map_err(|e| format!("cannot read '{}': {}", input_path.display(), e))?;

    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| format!("safetensors parse error in '{}': {}", input_path.display(), e))?;

    // Collect tensors in iteration order
    let mut metas: Vec<NslmParamMeta> = Vec::new();
    let mut data_blocks: Vec<Vec<u8>> = Vec::new();
    let mut data_offset: u64 = 0;

    for (name, view) in st.tensors() {
        let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();

        // Convert incoming dtype to f32 LE bytes (NSLM will store as f32)
        let f32_bytes: Vec<u8> = match view.dtype() {
            safetensors::Dtype::F32 => view.data().to_vec(),
            safetensors::Dtype::F64 => {
                let raw = view.data();
                let count = raw.len() / 8;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 8] = raw[i * 8..(i + 1) * 8].try_into().unwrap();
                    let v = f64::from_le_bytes(b) as f32;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            safetensors::Dtype::F16 => {
                let raw = view.data();
                let count = raw.len() / 2;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 2] = raw[i * 2..(i + 1) * 2].try_into().unwrap();
                    let v = f32::from(half::f16::from_le_bytes(b));
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            safetensors::Dtype::BF16 => {
                let raw = view.data();
                let count = raw.len() / 2;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 2] = raw[i * 2..(i + 1) * 2].try_into().unwrap();
                    let v = f32::from(half::bf16::from_le_bytes(b));
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            safetensors::Dtype::I32 => {
                let raw = view.data();
                let count = raw.len() / 4;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 4] = raw[i * 4..(i + 1) * 4].try_into().unwrap();
                    let v = i32::from_le_bytes(b) as f32;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            safetensors::Dtype::I64 => {
                let raw = view.data();
                let count = raw.len() / 8;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 8] = raw[i * 8..(i + 1) * 8].try_into().unwrap();
                    let v = i64::from_le_bytes(b) as f32;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            other => {
                return Err(format!(
                    "safetensors tensor '{}' has unsupported dtype {:?}",
                    name, other
                ));
            }
        };

        let nbytes = f32_bytes.len() as u64;
        metas.push(NslmParamMeta {
            name: name.to_owned(),
            shape,
            dtype: "f32",
            offset: data_offset,
            nbytes,
        });
        data_offset += nbytes;
        data_blocks.push(f32_bytes);
    }

    // Build NSLM JSON header
    let params_json: Vec<String> = metas
        .iter()
        .map(|m| {
            format!(
                r#"{{"name":"{}","shape":{:?},"dtype":"{}","offset":{},"nbytes":{}}}"#,
                m.name, m.shape, m.dtype, m.offset, m.nbytes
            )
        })
        .collect();
    let header = format!(r#"{{"params":[{}]}}"#, params_json.join(","));
    let header_bytes = header.as_bytes();

    // Write NSLM file
    let mut file = std::fs::File::create(output_path)
        .map_err(|e| format!("cannot create '{}': {}", output_path.display(), e))?;

    let magic: &[u8; 4] = b"NSLM";
    let version: u32 = 1;
    let header_size = header_bytes.len() as u64;

    file.write_all(magic)
        .map_err(|e| format!("write error (magic): {}", e))?;
    file.write_all(&version.to_le_bytes())
        .map_err(|e| format!("write error (version): {}", e))?;
    file.write_all(&header_size.to_le_bytes())
        .map_err(|e| format!("write error (header_size): {}", e))?;
    file.write_all(header_bytes)
        .map_err(|e| format!("write error (header): {}", e))?;

    // Pad to 64-byte alignment (measured from byte 0)
    let total_header = 4 + 4 + 8 + header_bytes.len(); // magic + version + header_size + header
    let padding = (64 - (total_header % 64)) % 64;
    let pad_buf = [0u8; 64];
    file.write_all(&pad_buf[..padding])
        .map_err(|e| format!("write error (padding): {}", e))?;

    // Raw tensor data
    for block in &data_blocks {
        file.write_all(block)
            .map_err(|e| format!("write error (tensor data): {}", e))?;
    }

    Ok(())
}

pub(crate) fn run_convert(input: &std::path::Path, output: &std::path::Path) {
    let in_ext = input.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let out_ext = output.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    match (in_ext.as_str(), out_ext.as_str()) {
        ("nslm", "safetensors") => {
            match convert_nslm_to_safetensors(input, output) {
                Ok(()) => println!("Converted {} → {}", input.display(), output.display()),
                Err(e) => {
                    eprintln!("error: {}", e);
                    process::exit(1);
                }
            }
        }
        ("safetensors", "nslm") => {
            match convert_safetensors_to_nslm(input, output) {
                Ok(()) => println!("Converted {} → {}", input.display(), output.display()),
                Err(e) => {
                    eprintln!("error: {}", e);
                    process::exit(1);
                }
            }
        }
        _ => {
            eprintln!(
                "error: unsupported conversion '{}.{}' → '{}.{}'.\n\
                 Supported: .nslm → .safetensors, .safetensors → .nslm",
                input.display(), in_ext, output.display(), out_ext
            );
            process::exit(1);
        }
    }
}
