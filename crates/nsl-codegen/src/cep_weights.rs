//! CEP — checkpoint names, ordered loading, and `.nslm` output.
//!
//! NSL writes the same per-layer parameter under two spellings: `model_save`
//! and `nsl export`/`nsl convert` write `blocks[0].attn.wq`, a train-block
//! checkpoint writes `blocks.0.attn.wq` (sometimes behind the model variable,
//! `m.blocks.0.attn.wq`). Every CEP stage that looks a tensor up goes through
//! [`BlockIndex`], so all of them read the same file the same way, and the
//! slicer writes names back in the spelling it was given.
//!
//! `.nslm` is read and written here as well as `.safetensors`, because it is
//! the only format that records the model's parameter ORDER, and `model_load`
//! copies positionally. A `.safetensors` file is sorted by its writer, so the
//! order cannot be recovered from one.

use std::collections::HashMap;
use std::path::Path;

use crate::weight_aware::{WeightDType, WeightEntry, WeightMap};

/// A parsed per-layer tensor name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockName<'a> {
    /// Everything before `blocks` (empty, or a model variable and its dot).
    pub prefix: &'a str,
    /// The layer index.
    pub layer: u32,
    /// `blocks[N]` (true) or `blocks.N` (false).
    pub bracket: bool,
    /// The path inside the block, e.g. `attn.wq` or `attn_norm.weight`.
    pub rest: &'a str,
}

impl BlockName<'_> {
    /// The same tensor's name in layer `layer`, in the spelling it was read in.
    pub fn with_layer(&self, layer: u32) -> String {
        if self.bracket {
            format!("{}blocks[{layer}].{}", self.prefix, self.rest)
        } else {
            format!("{}blocks.{layer}.{}", self.prefix, self.rest)
        }
    }
}

/// Parse `[<var>.]blocks[N].<rest>` or `[<var>.]blocks.N.<rest>`. Returns
/// `None` for a tensor that does not belong to a block (embedding, final norm).
pub fn parse_block_name(name: &str) -> Option<BlockName<'_>> {
    const KEY: &str = "blocks";
    let start = if name.starts_with(KEY) {
        0
    } else {
        name.find(".blocks").map(|i| i + 1)?
    };
    let prefix = &name[..start];
    // The prefix is at most one path segment: the model variable.
    if prefix.trim_end_matches('.').contains('.') {
        return None;
    }
    let after = &name[start + KEY.len()..];
    let (digits, rest, bracket) = if let Some(a) = after.strip_prefix('[') {
        let end = a.find(']')?;
        (&a[..end], a[end + 1..].strip_prefix('.')?, true)
    } else {
        let a = after.strip_prefix('.')?;
        let end = a.find('.')?;
        (&a[..end], &a[end + 1..], false)
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) || rest.is_empty() {
        return None;
    }
    Some(BlockName {
        prefix,
        layer: digits.parse().ok()?,
        bracket,
        rest,
    })
}

/// Per-layer lookup over a [`WeightMap`], independent of name spelling.
#[derive(Debug, Clone, Default)]
pub struct BlockIndex {
    by_layer: HashMap<(u32, String), String>,
}

impl BlockIndex {
    /// Index every per-layer tensor. A file that holds the same tensor under
    /// two spellings is refused: there is no telling which one is current.
    pub fn build(wm: &WeightMap) -> Result<Self, String> {
        let mut by_layer: HashMap<(u32, String), String> = HashMap::new();
        let mut names: Vec<&str> = wm.names().collect();
        names.sort_unstable();
        for name in names {
            let Some(b) = parse_block_name(name) else {
                continue;
            };
            if let Some(prev) = by_layer.insert((b.layer, b.rest.to_string()), name.to_string()) {
                return Err(format!(
                    "CEP: the weights file names the same tensor twice ('{prev}' and '{name}')"
                ));
            }
        }
        Ok(Self { by_layer })
    }

    /// The file's own name for `blocks[layer].<rest>`.
    pub fn name(&self, layer: u32, rest: &str) -> Option<&str> {
        self.by_layer
            .get(&(layer, rest.to_string()))
            .map(String::as_str)
    }

    /// The tensor at `blocks[layer].<rest>`.
    pub fn get<'a>(&self, wm: &'a WeightMap, layer: u32, rest: &str) -> Option<&'a WeightEntry> {
        self.name(layer, rest).and_then(|n| wm.get(n))
    }

    /// Whether the file has any per-layer tensor at all.
    pub fn is_empty(&self) -> bool {
        self.by_layer.is_empty()
    }
}

/// The on-disk format a weights file was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Safetensors,
    Nslm,
}

/// Weights as CEP reads them: the tensors, and the order they were stored in.
#[derive(Debug, Clone)]
pub struct CepWeights {
    pub map: WeightMap,
    /// Tensor names in file order. For `.nslm` this is the model's parameter
    /// order; for `.safetensors` it is name order, which means nothing.
    pub order: Vec<String>,
    pub format: WeightFormat,
}

impl CepWeights {
    /// Wrap an in-memory map (tests, and callers that built one themselves).
    pub fn from_map(map: WeightMap) -> Self {
        let mut order: Vec<String> = map.names().map(str::to_string).collect();
        order.sort_unstable();
        Self {
            map,
            order,
            format: WeightFormat::Safetensors,
        }
    }
}

fn is_nslm(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("nslm"))
}

/// Load `.safetensors` or `.nslm` (chosen by extension).
pub fn load_weights(path: &Path) -> Result<CepWeights, String> {
    if is_nslm(path) {
        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
        let entries = parse_nslm(&bytes).map_err(|e| format!("'{}': {e}", path.display()))?;
        let order: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
        let mut map: HashMap<String, WeightEntry> = HashMap::with_capacity(entries.len());
        for e in entries {
            if let Some(dup) = map.insert(e.name.clone(), e) {
                return Err(format!(
                    "'{}': tensor '{}' appears twice",
                    path.display(),
                    dup.name
                ));
            }
        }
        Ok(CepWeights {
            map: WeightMap::from_entries(map),
            order,
            format: WeightFormat::Nslm,
        })
    } else {
        let map = WeightMap::load(path).map_err(|e| e.to_string())?;
        let mut cw = CepWeights::from_map(map);
        cw.format = WeightFormat::Safetensors;
        Ok(cw)
    }
}

const NSLM_MAGIC: &[u8; 4] = b"NSLM";
const NSLM_VERSION: u32 = 1;

/// The dtype names `model_save` writes (`checkpoint_dtype_name` in the runtime).
fn nslm_dtype(name: &str) -> Option<WeightDType> {
    Some(match name {
        "f32" => WeightDType::F32,
        "f64" => WeightDType::F64,
        "fp16" => WeightDType::F16,
        "bf16" => WeightDType::BF16,
        "int8" => WeightDType::I8,
        "i32" => WeightDType::I32,
        "fp8e4m3" => WeightDType::F8E4M3,
        "fp8e5m2" => WeightDType::F8E5M2,
        _ => return None,
    })
}

fn nslm_dtype_name(dt: WeightDType) -> &'static str {
    match dt {
        WeightDType::F32 => "f32",
        WeightDType::F64 => "f64",
        WeightDType::F16 => "fp16",
        WeightDType::BF16 => "bf16",
        WeightDType::I8 => "int8",
        WeightDType::I32 => "i32",
        WeightDType::F8E4M3 => "fp8e4m3",
        WeightDType::F8E5M2 => "fp8e5m2",
    }
}

#[derive(serde::Deserialize)]
struct NslmParam {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    offset: u64,
    nbytes: u64,
}

#[derive(serde::Deserialize)]
struct NslmHeader {
    params: Vec<NslmParam>,
}

/// Parse an `.nslm` file (magic, version, JSON parameter table, 64-byte
/// aligned data) into entries in file order. Every size is checked against
/// the file before any slice is taken.
fn parse_nslm(data: &[u8]) -> Result<Vec<WeightEntry>, String> {
    if data.len() < 16 || &data[..4] != NSLM_MAGIC {
        return Err("not an .nslm file (bad magic)".into());
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != NSLM_VERSION {
        return Err(format!(
            "unsupported .nslm version {version} (expected {NSLM_VERSION})"
        ));
    }
    let mut size = [0u8; 8];
    size.copy_from_slice(&data[8..16]);
    let header_size = u64::from_le_bytes(size);
    let header_end = 16u64
        .checked_add(header_size)
        .filter(|&end| end <= data.len() as u64)
        .ok_or_else(|| {
            format!(
                "the header claims {header_size} bytes but the file is {} bytes",
                data.len()
            )
        })? as usize;
    let header: NslmHeader = serde_json::from_slice(&data[16..header_end])
        .map_err(|e| format!("the header is not a valid parameter table: {e}"))?;
    let data_start = header_end + (64 - header_end % 64) % 64;
    let mut out = Vec::with_capacity(header.params.len());
    for p in header.params {
        let dtype = nslm_dtype(&p.dtype).ok_or_else(|| {
            format!(
                "tensor '{}' has dtype '{}', which CEP cannot read",
                p.name, p.dtype
            )
        })?;
        let shape: Vec<usize> = p.shape.iter().map(|&d| d as usize).collect();
        let elems = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        let start = (data_start as u64).checked_add(p.offset);
        let end = start.and_then(|s| s.checked_add(p.nbytes));
        let (Some(elems), Some(start), Some(end)) = (elems, start, end) else {
            return Err(format!("tensor '{}' has an impossible size", p.name));
        };
        if end > data.len() as u64
            || elems.checked_mul(dtype.byte_width()) != Some(p.nbytes as usize)
        {
            return Err(format!(
                "tensor '{}' is {:?} {} ({} bytes) at offset {}, which does not fit the file",
                p.name, shape, p.dtype, p.nbytes, p.offset
            ));
        }
        out.push(WeightEntry::new(
            p.name,
            data[start as usize..end as usize].to_vec(),
            shape,
            dtype,
        ));
    }
    Ok(out)
}

/// Write `entries`, in the order given, as an `.nslm` file that `model_load`
/// reads back. Written to a temporary and renamed, so a failed write leaves
/// no partial file under the final name.
pub fn write_nslm(entries: &[&WeightEntry], path: &Path) -> std::io::Result<()> {
    let mut params = Vec::with_capacity(entries.len());
    let mut offset = 0u64;
    for e in entries {
        let shape: Vec<String> = e.shape.iter().map(|d| d.to_string()).collect();
        params.push(format!(
            r#"{{"name":{},"shape":[{}],"dtype":"{}","offset":{},"nbytes":{}}}"#,
            serde_json::to_string(&e.name)?,
            shape.join(", "),
            nslm_dtype_name(e.dtype),
            offset,
            e.data.len()
        ));
        offset += e.data.len() as u64;
    }
    let header = format!(r#"{{"params":[{}]}}"#, params.join(","));
    let mut bytes = Vec::with_capacity(16 + header.len() + 64 + offset as usize);
    bytes.extend_from_slice(NSLM_MAGIC);
    bytes.extend_from_slice(&NSLM_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    let pad = (64 - bytes.len() % 64) % 64;
    bytes.resize(bytes.len() + pad, 0);
    for e in entries {
        bytes.extend_from_slice(&e.data);
    }
    write_atomically(path, &bytes)
}

/// Write `bytes` to `path` through a sibling temporary and a rename.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".tmp.{}", std::process::id()));
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Whether `path` asks for `.nslm` output.
pub fn wants_nslm(path: &Path) -> bool {
    is_nslm(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_both_spellings_and_a_model_variable_prefix() {
        let b = parse_block_name("blocks[3].attn.wq").unwrap();
        assert_eq!(
            (b.prefix, b.layer, b.bracket, b.rest),
            ("", 3, true, "attn.wq")
        );
        let d = parse_block_name("blocks.12.attn_norm.weight").unwrap();
        assert_eq!(
            (d.prefix, d.layer, d.bracket, d.rest),
            ("", 12, false, "attn_norm.weight")
        );
        let m = parse_block_name("m.blocks.0.ffn.w_gate").unwrap();
        assert_eq!(
            (m.prefix, m.layer, m.bracket, m.rest),
            ("m.", 0, false, "ffn.w_gate")
        );
    }

    #[test]
    fn rejects_names_that_are_not_block_tensors() {
        for name in [
            "embed",
            "norm.weight",
            "blocks",
            "blocks.0",
            "blocks[0]",
            "blocks.x.w",
            "blocks[].w",
            "a.b.blocks.0.w",
            "blockss.0.w",
        ] {
            assert_eq!(parse_block_name(name), None, "{name}");
        }
    }

    #[test]
    fn renumbering_keeps_the_spelling() {
        assert_eq!(
            parse_block_name("blocks[3].attn.wq").unwrap().with_layer(1),
            "blocks[1].attn.wq"
        );
        assert_eq!(
            parse_block_name("m.blocks.3.attn.wq")
                .unwrap()
                .with_layer(1),
            "m.blocks.1.attn.wq"
        );
    }

    fn entry(name: &str, shape: Vec<usize>, vals: &[f32]) -> WeightEntry {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        WeightEntry::new(name.to_string(), data, shape, WeightDType::F32)
    }

    #[test]
    fn index_refuses_a_tensor_stored_under_both_spellings() {
        let mut m = HashMap::new();
        m.insert(
            "blocks.0.attn.wq".to_string(),
            entry("blocks.0.attn.wq", vec![1], &[1.0]),
        );
        m.insert(
            "blocks[0].attn.wq".to_string(),
            entry("blocks[0].attn.wq", vec![1], &[2.0]),
        );
        let err = BlockIndex::build(&WeightMap::from_entries(m)).unwrap_err();
        assert!(err.contains("twice"), "{err}");
    }

    /// The byte layout is written out by hand here, from the format `model_save`
    /// documents, so the writer is checked against the format and not against
    /// the reader beside it.
    #[test]
    fn nslm_bytes_match_the_documented_layout() {
        let a = entry("blocks[0].w", vec![2, 1], &[1.5, -2.0]);
        let b = entry("embed", vec![1], &[4.0]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nslm");
        write_nslm(&[&a, &b], &path).unwrap();
        let got = std::fs::read(&path).unwrap();

        let header = concat!(
            r#"{"params":[{"name":"blocks[0].w","shape":[2, 1],"dtype":"f32","offset":0,"nbytes":8},"#,
            r#"{"name":"embed","shape":[1],"dtype":"f32","offset":8,"nbytes":4}]}"#
        );
        let mut want = Vec::new();
        want.extend_from_slice(b"NSLM");
        want.extend_from_slice(&1u32.to_le_bytes());
        want.extend_from_slice(&(header.len() as u64).to_le_bytes());
        want.extend_from_slice(header.as_bytes());
        while want.len() % 64 != 0 {
            want.push(0);
        }
        want.extend_from_slice(&1.5f32.to_le_bytes());
        want.extend_from_slice(&(-2.0f32).to_le_bytes());
        want.extend_from_slice(&4.0f32.to_le_bytes());
        assert_eq!(got, want);

        // And it reads back in file order, not name order.
        let w = load_weights(&path).unwrap();
        assert_eq!(w.format, WeightFormat::Nslm);
        assert_eq!(
            w.order,
            vec!["blocks[0].w".to_string(), "embed".to_string()]
        );
        assert_eq!(w.map.get("embed").unwrap().data, 4.0f32.to_le_bytes());
    }

    #[test]
    fn nslm_with_a_size_that_does_not_fit_is_refused() {
        let header =
            r#"{"params":[{"name":"w","shape":[4],"dtype":"f32","offset":0,"nbytes":16}]}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"NSLM");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        while bytes.len() % 64 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0u8; 8]); // 8 of the 16 promised bytes
        let err = parse_nslm(&bytes).unwrap_err();
        assert!(err.contains("does not fit"), "{err}");
    }
}
