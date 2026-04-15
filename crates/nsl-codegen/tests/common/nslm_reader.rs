//! Minimal `.nslm` parser for integration tests.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
struct Entry {
    name: String,
    dtype: String,
    offset: usize,
    nbytes: usize,
}

pub fn read_nslm(path: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut f = File::open(path).map_err(|e| format!("open {:?}: {}", path, e))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| format!("read: {}", e))?;

    if buf.len() < 16 || &buf[0..4] != b"NSLM" {
        return Err("bad magic (expected NSLM)".into());
    }
    let header_size =
        u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
    let header_end = 16 + header_size;
    if buf.len() < header_end {
        return Err("truncated header".into());
    }
    let header_json = std::str::from_utf8(&buf[16..header_end])
        .map_err(|e| format!("header utf8: {}", e))?;

    let entries = parse_entries(header_json)?;

    // 64-byte alignment from start of file.
    let header_total = header_end;
    let pad = (64 - (header_total % 64)) % 64;
    let data_start = header_total + pad;

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    for e in entries {
        let begin = data_start + e.offset;
        let end = begin + e.nbytes;
        if buf.len() < end {
            return Err(format!("truncated data for {}", e.name));
        }
        let slab = &buf[begin..end];
        let values = match e.dtype.as_str() {
            "f32" => slab
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "f64" => slab
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            other => return Err(format!("unsupported dtype: {}", other)),
        };
        out.insert(e.name, values);
    }
    Ok(out)
}

fn parse_entries(json: &str) -> Result<Vec<Entry>, String> {
    let mut entries = Vec::new();
    let mut cursor = 0;
    while let Some(name_idx) = json[cursor..].find("\"name\":\"") {
        let abs = cursor + name_idx + 8;
        let name_end = json[abs..]
            .find('"')
            .ok_or("malformed name field")?;
        let name = json[abs..abs + name_end].to_string();
        cursor = abs + name_end;

        let dtype = extract_str_field(json, cursor, "dtype")?;
        cursor = find_after(json, cursor, "dtype")?;
        let offset: usize = extract_int_field(json, cursor, "offset")?;
        cursor = find_after(json, cursor, "offset")?;
        let nbytes: usize = extract_int_field(json, cursor, "nbytes")?;
        cursor = find_after(json, cursor, "nbytes")?;

        entries.push(Entry {
            name,
            dtype,
            offset,
            nbytes,
        });
    }
    Ok(entries)
}

fn extract_str_field(json: &str, from: usize, key: &str) -> Result<String, String> {
    let pattern = format!("\"{}\":\"", key);
    let idx = json[from..]
        .find(&pattern)
        .ok_or_else(|| format!("{} field missing", key))?;
    let abs = from + idx + pattern.len();
    let end = json[abs..]
        .find('"')
        .ok_or_else(|| format!("{} field unterminated", key))?;
    Ok(json[abs..abs + end].to_string())
}

fn extract_int_field(json: &str, from: usize, key: &str) -> Result<usize, String> {
    let pattern = format!("\"{}\":", key);
    let idx = json[from..]
        .find(&pattern)
        .ok_or_else(|| format!("{} field missing", key))?;
    let abs = from + idx + pattern.len();
    let end = json[abs..]
        .find(|c: char| c == ',' || c == '}')
        .ok_or_else(|| format!("{} value unterminated", key))?;
    json[abs..abs + end]
        .trim()
        .parse::<usize>()
        .map_err(|e| format!("{}: {}", key, e))
}

fn find_after(json: &str, from: usize, key: &str) -> Result<usize, String> {
    let pattern = format!("\"{}\":", key);
    let idx = json[from..]
        .find(&pattern)
        .ok_or_else(|| format!("{} field missing", key))?;
    Ok(from + idx + pattern.len())
}
