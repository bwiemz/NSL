//! CEP Option 2 — SP1: weight slicing (pure data transform).

use std::collections::HashMap;

use crate::cep_oracle::ModelSpec;
use crate::cep_rewrite::{LayerDelta, PruneDelta};
use crate::weight_aware::{WeightDType, WeightEntry, WeightMap};

#[derive(Debug, Clone, PartialEq)]
pub enum CepSliceError {
    HeadIndexOutOfRange { layer: u32, head: u32, n_heads: u32 },
    ShapeMismatch { tensor: String, expected: String, found: String },
    MissingTensor { tensor: String },
    InconsistentFfn { layer: u32, detail: String },
    UnitOutOfRange { tensor: String, unit: u32, n_units: u32 },
}

impl std::fmt::Display for CepSliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CepSliceError::HeadIndexOutOfRange { layer, head, n_heads } =>
                write!(f, "CEP slice: layer {layer} head {head} out of range (n_heads={n_heads})"),
            CepSliceError::ShapeMismatch { tensor, expected, found } =>
                write!(f, "CEP slice: '{tensor}' shape mismatch\n  expected: {expected}\n  found:    {found}"),
            CepSliceError::MissingTensor { tensor } =>
                write!(f, "CEP slice: expected tensor '{tensor}' not found in weights"),
            CepSliceError::InconsistentFfn { layer, detail } =>
                write!(f, "CEP slice: layer {layer} FFN inconsistency: {detail}"),
            CepSliceError::UnitOutOfRange { tensor, unit, n_units } =>
                write!(f, "CEP slice: '{tensor}' survivor unit {unit} out of range (n_units={n_units})"),
        }
    }
}

/// Gather survivor blocks along `axis` (0 = rows, 1 = cols) of a row-major 2-D tensor.
/// A block is `block_width` consecutive rows/cols; `survivor_units` are 0-based block
/// indices (**strictly ascending** (enforced)). Returns a new contiguous WeightEntry; dtype-agnostic (byte copy).
pub fn slice_blocks(
    entry: &WeightEntry,
    axis: usize,
    n_units: u32,
    block_width: u32,
    survivor_units: &[u32],
) -> Result<WeightEntry, CepSliceError> {
    if entry.shape.len() != 2 {
        return Err(CepSliceError::ShapeMismatch {
            tensor: entry.name.clone(),
            expected: "2-D tensor".into(),
            found: format!("{:?}", entry.shape),
        });
    }
    let rows = entry.shape[0];
    let cols = entry.shape[1];
    let bw = entry.dtype.byte_width();
    // Data must be exactly rows*cols*bw bytes, else the byte gather below would index-panic.
    let expected_bytes = rows * cols * bw;
    if entry.data.len() != expected_bytes {
        return Err(CepSliceError::ShapeMismatch {
            tensor: entry.name.clone(),
            expected: format!("{expected_bytes} bytes ({rows}x{cols}x{bw})"),
            found: format!("{} bytes", entry.data.len()),
        });
    }
    // survivor_units must be strictly ascending: the gather preserves their order, and a
    // non-ascending (or duplicated) list would silently reorder/duplicate weight blocks.
    if survivor_units.windows(2).any(|w| w[0] >= w[1]) {
        return Err(CepSliceError::ShapeMismatch {
            tensor: entry.name.clone(),
            expected: "strictly ascending survivor_units".into(),
            found: format!("{survivor_units:?}"),
        });
    }
    let block = block_width as usize;
    // Invariant: n_units * block_width must exactly equal the tensor dimension on the sliced
    // axis. Without this, the final unit's byte-range access can silently exceed the row/col
    // boundary and panic. The per-unit `u < n_units` check below is a necessary but not
    // sufficient condition — it only rules out index overflow, not shape/spec mismatch.
    let axis_dim = if axis == 0 { rows } else { cols };
    let declared_dim = (n_units as usize).saturating_mul(block);
    if declared_dim != axis_dim {
        return Err(CepSliceError::ShapeMismatch {
            tensor: entry.name.clone(),
            expected: format!(
                "n_units ({n_units}) * block_width ({block_width}) = {declared_dim} \
                 to equal axis-{axis} dimension {axis_dim}"
            ),
            found: format!("{axis_dim}"),
        });
    }
    for &u in survivor_units {
        if u >= n_units {
            return Err(CepSliceError::UnitOutOfRange { tensor: entry.name.clone(), unit: u, n_units });
        }
    }
    let n_surv = survivor_units.len();
    let mut out = Vec::new();
    let (new_rows, new_cols) = match axis {
        0 => {
            // Keep survivor row-blocks; each row is `cols * bw` bytes.
            out.reserve(survivor_units.len() * block * cols * bw);
            let row_bytes = cols * bw;
            for &u in survivor_units {
                let start = (u as usize) * block;
                for r in start..start + block {
                    let off = r * row_bytes;
                    out.extend_from_slice(&entry.data[off..off + row_bytes]);
                }
            }
            (n_surv * block, cols)
        }
        1 => {
            // Keep survivor col-blocks; gather per row.
            out.reserve(rows * survivor_units.len() * block * bw);
            for r in 0..rows {
                let row_off = r * cols * bw;
                for &u in survivor_units {
                    let c0 = (u as usize) * block;
                    let off = row_off + c0 * bw;
                    out.extend_from_slice(&entry.data[off..off + block * bw]);
                }
            }
            (rows, n_surv * block)
        }
        _ => {
            return Err(CepSliceError::ShapeMismatch {
                tensor: entry.name.clone(),
                expected: "axis 0 or 1".into(),
                found: format!("axis {axis}"),
            })
        }
    };
    let num_elements = new_rows * new_cols;
    Ok(WeightEntry {
        name: entry.name.clone(),
        data: out,
        shape: vec![new_rows, new_cols],
        dtype: entry.dtype,
        num_elements,
        sparsity: None,
        eliminated: false,
    })
}

/// Rank d_ff neurons by column L2 norm of w_gate (`[d_model, d_ff]`, neurons on axis-1);
/// return the top `new_d_ff` survivor indices in ascending order. dtype-agnostic via to_f64.
/// Returns `Ok(empty)` when `new_d_ff == 0` (or `d_ff == 0`); never panics on a
/// well-formed `WeightEntry` (a data-length mismatch yields `ShapeMismatch`).
pub fn ffn_survivors_by_magnitude(
    w_gate: &WeightEntry,
    d_ff: u32,
    new_d_ff: u32,
) -> Result<Vec<u32>, CepSliceError> {
    if w_gate.shape.len() != 2 || w_gate.shape[1] != d_ff as usize {
        return Err(CepSliceError::ShapeMismatch {
            tensor: w_gate.name.clone(),
            expected: format!("[d_model, {d_ff}]"),
            found: format!("{:?}", w_gate.shape),
        });
    }
    let rows = w_gate.shape[0];
    let cols = d_ff as usize;
    let bw = w_gate.dtype.byte_width();
    // Data must cover the claimed shape, else the per-column reads below would index-panic
    // (parity with slice_blocks).
    let expected_bytes = rows * cols * bw;
    if w_gate.data.len() != expected_bytes {
        return Err(CepSliceError::ShapeMismatch {
            tensor: w_gate.name.clone(),
            expected: format!("{expected_bytes} bytes ({rows}x{cols}x{bw})"),
            found: format!("{} bytes", w_gate.data.len()),
        });
    }
    let keep = (new_d_ff as usize).min(cols);

    let mut norms: Vec<(u32, f64)> = (0..cols)
        .map(|c| {
            let mut sumsq = 0.0_f64;
            for r in 0..rows {
                let off = (r * cols + c) * bw;
                let v = w_gate.dtype.to_f64(&w_gate.data[off..off + bw]);
                sumsq += v * v;
            }
            (c as u32, sumsq) // sqrt is monotonic; ranking on sumsq is equivalent
        })
        .collect();
    // Highest magnitude first; tie-break by lower index for determinism.
    norms.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
    });
    let mut survivors: Vec<u32> = norms.into_iter().take(keep).map(|(c, _)| c).collect();
    survivors.sort_unstable();
    Ok(survivors)
}

/// Map our dtype to the safetensors crate's dtype for re-serialization.
pub(crate) fn to_safetensors_dtype(dt: WeightDType) -> safetensors::Dtype {
    match dt {
        WeightDType::F16 => safetensors::Dtype::F16,
        WeightDType::BF16 => safetensors::Dtype::BF16,
        WeightDType::F32 => safetensors::Dtype::F32,
        WeightDType::F64 => safetensors::Dtype::F64,
        WeightDType::F8E4M3 => safetensors::Dtype::F8_E4M3,
        WeightDType::F8E5M2 => safetensors::Dtype::F8_E5M2,
        WeightDType::I8 => safetensors::Dtype::I8,
        WeightDType::I32 => safetensors::Dtype::I32,
    }
}

/// Parse "blocks.{l}.{attn|ffn}.{name}" into (layer, "attn"|"ffn", leaf). Returns None
/// for non-per-layer tensors (embed, final norm, lm_head) — copied through unchanged.
/// Only `blocks.N.*` naming is recognized; `layers.N.*` / `h.N.*` conventions return None
/// here and pass through unsliced (NSL canonical models, the only ones CEP extracts, use
/// `blocks.*`).
fn parse_layer_tensor(name: &str) -> Option<(u32, &'static str, String)> {
    let prefix = crate::wggo_graph::layer_prefix(name)?; // "blocks.0"
    let layer: u32 = prefix.strip_prefix("blocks.")?.parse().ok()?;
    let rest = name.strip_prefix(&prefix)?.strip_prefix('.')?; // "attn.wq" | "ffn.w_gate"
    let (sect, leaf) = rest.split_once('.')?;
    match sect {
        "attn" => Some((layer, "attn", leaf.to_string())),
        "ffn" => Some((layer, "ffn", leaf.to_string())),
        _ => None,
    }
}

/// Slice a model's weights to a PruneDelta. Returns the sliced entries (a smaller set).
/// Pure data transform — drops pruned heads' blocks (GQA-group-aligned), narrows FFN by
/// magnitude survivors (consistent across w_gate/w_up/w_down), drops + re-keys dropped layers.
///
/// LIMITATION (SP1 scope): only `attn.*` and `ffn.*` per-layer tensors are sliced/re-keyed.
/// Other intra-layer tensors (e.g. RMSNorm scales `blocks.{l}.norm.*`) take the passthrough
/// path with their ORIGINAL key. That is correct for head/FFN pruning (which never renumbers
/// layers), but it is NOT drop-aware: if `drop_layer` is ever set, such tensors are neither
/// omitted for dropped layers nor renumbered for survivors. The greedy planner never sets
/// `drop_layer` (it is always false today), so this is latent; full layer-drop support that
/// also handles norm/other intra-layer tensors is deferred (SP2).
pub fn apply_prune_delta_to_weights(
    wm: &WeightMap,
    spec: &ModelSpec,
    delta: &PruneDelta,
) -> Result<HashMap<String, WeightEntry>, CepSliceError> {
    // Per-layer drop flags + the old->new layer re-key map (skip dropped layers).
    let mut drop = vec![false; spec.n_layers as usize];
    for ld in &delta.per_layer {
        if (ld.layer as usize) < drop.len() && ld.drop_layer {
            drop[ld.layer as usize] = true;
        }
    }
    let mut rekey = vec![None; spec.n_layers as usize];
    let mut next = 0u32;
    for (l, d) in drop.iter().enumerate() {
        if !*d {
            rekey[l] = Some(next);
            next += 1;
        }
    }
    // Index the delta by layer for quick lookup.
    let by_layer: HashMap<u32, &LayerDelta> =
        delta.per_layer.iter().map(|ld| (ld.layer, ld)).collect();

    // Precompute per-layer FFN survivor sets (one ranking per layer, from w_gate).
    let mut ffn_survivors: HashMap<u32, Vec<u32>> = HashMap::new();
    for ld in &delta.per_layer {
        if let Some(new_ff) = ld.new_d_ff {
            let l = ld.layer;
            let gate_name = format!("blocks.{l}.ffn.w_gate");
            let gate = wm
                .get(&gate_name)
                .ok_or(CepSliceError::MissingTensor { tensor: gate_name })?;
            let surv = ffn_survivors_by_magnitude(gate, spec.d_ff[l as usize], new_ff)?;
            ffn_survivors.insert(l, surv);
        }
    }

    let mut out: HashMap<String, WeightEntry> = HashMap::new();
    for (name, entry) in wm.entries() {
        match parse_layer_tensor(name) {
            None => {
                // Passthrough (embed, final norm, lm_head, etc.).
                out.insert(name.clone(), entry.clone());
            }
            Some((l, sect, leaf)) => {
                if (l as usize) >= drop.len() || drop[l as usize] {
                    continue; // dropped layer — omit
                }
                let new_l = rekey[l as usize].expect("surviving layer has a rekey");
                let ld = by_layer.get(&l);
                let hd = spec.head_dim[l as usize];

                let mut sliced = if let Some(ld) = ld {
                    slice_one(entry, sect, &leaf, l, spec, ld, hd, &ffn_survivors)?
                } else {
                    entry.clone()
                };
                sliced.name = format!("blocks.{new_l}.{sect}.{leaf}");
                out.insert(sliced.name.clone(), sliced);
            }
        }
    }
    Ok(out)
}

/// Slice a single per-layer tensor according to its role + the layer's delta.
#[allow(clippy::too_many_arguments)]
fn slice_one(
    entry: &WeightEntry,
    sect: &str,
    leaf: &str,
    layer: u32,
    spec: &ModelSpec,
    ld: &LayerDelta,
    hd: u32,
    ffn_survivors: &HashMap<u32, Vec<u32>>,
) -> Result<WeightEntry, CepSliceError> {
    let l = layer as usize;
    // Attention head survivors (group-aligned: pruned_heads are whole groups).
    let n_heads = spec.n_heads[l];
    let n_kv = spec.n_kv_heads[l].max(1);
    let group = (n_heads / n_kv).max(1);
    // Invariant (upheld by plan_to_prune_delta): pruned_heads are whole GQA groups, so the
    // KV-survivor test below — checking each group's first head as representative — is sound.
    // Guard it in debug builds against a hand-edited/future-planner delta that violates
    // group-alignment, which would otherwise desync Q survivors from KV survivors.
    debug_assert!(
        ld.pruned_heads.iter().all(|&h| {
            let base = (h / group) * group;
            (base..base + group).all(|x| ld.pruned_heads.contains(&x))
        }),
        "slice_one: layer {layer} pruned_heads {:?} are not whole GQA groups (group={group})",
        ld.pruned_heads
    );
    let survivor_q: Vec<u32> = (0..n_heads).filter(|h| !ld.pruned_heads.contains(h)).collect();
    // KV survivors: a group survives iff its query heads survive.
    let survivor_kv: Vec<u32> = (0..n_kv)
        .filter(|&g| !ld.pruned_heads.contains(&(g * group)))
        .collect();

    match (sect, leaf) {
        ("attn", "wq") if !ld.pruned_heads.is_empty() => {
            slice_blocks(entry, 1, n_heads, hd, &survivor_q)
        }
        ("attn", "wo") if !ld.pruned_heads.is_empty() => {
            slice_blocks(entry, 0, n_heads, hd, &survivor_q)
        }
        ("attn", "wk") | ("attn", "wv") if !ld.pruned_heads.is_empty() => {
            slice_blocks(entry, 1, n_kv, hd, &survivor_kv)
        }
        ("ffn", "w_gate") | ("ffn", "w_up") => {
            if let Some(surv) = ffn_survivors.get(&layer) {
                slice_blocks(entry, 1, spec.d_ff[l], 1, surv)
            } else {
                Ok(entry.clone())
            }
        }
        ("ffn", "w_down") => {
            if let Some(surv) = ffn_survivors.get(&layer) {
                slice_blocks(entry, 0, spec.d_ff[l], 1, surv)
            } else {
                Ok(entry.clone())
            }
        }
        _ => Ok(entry.clone()), // norm scales etc. inside a layer — passthrough
    }
}

/// Re-serialize sliced entries to a smaller .safetensors at `path`.
pub fn write_sliced_weights(
    entries: &HashMap<String, WeightEntry>,
    path: &std::path::Path,
) -> std::io::Result<()> {
    use safetensors::tensor::TensorView;
    // Build views, propagating a shape/data-length inconsistency as an io::Error rather than
    // panicking at the serialization boundary (`?` can't be used inside a `.map` closure).
    let mut views: Vec<(String, TensorView)> = Vec::with_capacity(entries.len());
    for (name, e) in entries {
        let v = TensorView::new(to_safetensors_dtype(e.dtype), e.shape.clone(), &e.data)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))?;
        views.push((name.clone(), v));
    }
    let bytes = safetensors::serialize(views, &None)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    std::fs::write(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cep_oracle::{Activation, ModelSpec, NormType};
    use crate::cep_rewrite::{LayerDelta, PruneDelta};
    use crate::weight_aware::{WeightDType, WeightEntry, WeightMap};

    fn small_spec() -> ModelSpec {
        // d_model=8, 2 layers, 4 heads, 2 kv (group=2), head_dim=2, d_ff=6.
        ModelSpec {
            d_model: 8, n_layers: 2,
            n_heads: vec![4, 4], n_kv_heads: vec![2, 2], head_dim: vec![2, 2], d_ff: vec![6, 6],
            vocab: 16, max_seq: 32, batch: 1,
            activation: Activation::SwiGlu, norm: NormType::RmsNorm, dtype_bytes: 4,
        }
    }

    // Build a WeightMap with per-layer attn+ffn tensors (ramp values) + an embed passthrough.
    fn small_weights(spec: &ModelSpec) -> WeightMap {
        let mut entries = std::collections::HashMap::new();
        // Each layer gets a distinct base offset so a tensor's layer identity is recoverable
        // from its data — this lets the drop/re-key test assert the surviving tensor holds the
        // ORIGINAL layer's bytes (not a different layer's). Layer 0 uses base 0, so the
        // value-identity assertions in the head/FFN slicing test are unaffected.
        let mut ramp = |name: &str, shape: Vec<usize>, base: f32| {
            let n: usize = shape.iter().product();
            let vals: Vec<f32> = (0..n).map(|i| base + i as f32).collect();
            entries.insert(name.to_string(), f32_entry(name, shape, &vals));
        };
        let dm = spec.d_model as usize;
        for l in 0..spec.n_layers as usize {
            let nh = spec.n_heads[l] as usize;
            let nkv = spec.n_kv_heads[l] as usize;
            let hd = spec.head_dim[l] as usize;
            let dff = spec.d_ff[l] as usize;
            let base = (l as f32) * 10_000.0;
            ramp(&format!("blocks.{l}.attn.wq"), vec![dm, nh * hd], base);
            ramp(&format!("blocks.{l}.attn.wk"), vec![dm, nkv * hd], base);
            ramp(&format!("blocks.{l}.attn.wv"), vec![dm, nkv * hd], base);
            ramp(&format!("blocks.{l}.attn.wo"), vec![nh * hd, dm], base);
            ramp(&format!("blocks.{l}.ffn.w_gate"), vec![dm, dff], base);
            ramp(&format!("blocks.{l}.ffn.w_up"), vec![dm, dff], base);
            ramp(&format!("blocks.{l}.ffn.w_down"), vec![dff, dm], base);
        }
        ramp("embed", vec![spec.vocab as usize, dm], 0.0);
        WeightMap::from_entries(entries)
    }

    #[test]
    fn orchestrator_slices_heads_ffn_and_passes_through() {
        let spec = small_spec();
        let wm = small_weights(&spec);
        // Drop group 0 (heads 0,1) of layer 0 -> 2 surviving heads {2,3}, 1 kv head {1}.
        // Narrow layer 0 FFN to 4 (drop 2 lowest-magnitude neurons). Layer 1 untouched.
        let delta = PruneDelta {
            per_layer: vec![
                LayerDelta { layer: 0, pruned_heads: vec![0, 1], new_d_ff: Some(4), drop_layer: false },
                LayerDelta { layer: 1, pruned_heads: vec![], new_d_ff: None, drop_layer: false },
            ],
        };
        let out = apply_prune_delta_to_weights(&wm, &spec, &delta).unwrap();

        // wq [8, 8] -> survivor heads {2,3} -> cols {4,5,6,7} -> [8, 4]
        let wq = &out["blocks.0.attn.wq"];
        assert_eq!(wq.shape, vec![8, 4]);
        // identity: row 0 of sliced == original row 0 cols 4..8 = [4,5,6,7]
        let wq_f = read_f32(wq);
        assert_eq!(&wq_f[0..4], &[4., 5., 6., 7.]);
        // wk [8,4] -> survivor kv head {1} -> cols {2,3} -> [8,2]
        assert_eq!(out["blocks.0.attn.wk"].shape, vec![8, 2]);
        // wo [8,8] -> survivor heads {2,3} -> rows {4,5,6,7} -> [4,8]
        assert_eq!(out["blocks.0.attn.wo"].shape, vec![4, 8]);
        // FFN narrowed to 4 across all three (consistent)
        assert_eq!(out["blocks.0.ffn.w_gate"].shape, vec![8, 4]);
        assert_eq!(out["blocks.0.ffn.w_up"].shape, vec![8, 4]);
        assert_eq!(out["blocks.0.ffn.w_down"].shape, vec![4, 8]);
        // layer 1 untouched
        assert_eq!(out["blocks.1.attn.wq"].shape, vec![8, 8]);
        // passthrough
        assert_eq!(out["embed"].shape, vec![16, 8]);
    }

    #[test]
    fn orchestrator_drops_and_rekeys_layers() {
        let spec = small_spec();
        let wm = small_weights(&spec);
        // Drop layer 0 entirely -> layer 1 re-keyed to blocks.0.
        let delta = PruneDelta {
            per_layer: vec![
                LayerDelta { layer: 0, pruned_heads: vec![], new_d_ff: None, drop_layer: true },
                LayerDelta { layer: 1, pruned_heads: vec![], new_d_ff: None, drop_layer: false },
            ],
        };
        let out = apply_prune_delta_to_weights(&wm, &spec, &delta).unwrap();
        assert!(!out.contains_key("blocks.1.attn.wq"), "old layer-1 name must be re-keyed");
        // original layer-1 wq now lives at blocks.0.attn.wq
        assert!(out.contains_key("blocks.0.attn.wq"));
        assert_eq!(out["blocks.0.attn.wq"].shape, vec![8, 8]);
        // non-vacuous: it must hold the ORIGINAL layer-1 bytes (base 10000), not layer-0's
        // (base 0) — proves the re-key carried the surviving layer's data, not a stale copy.
        let wq_f = read_f32(&out["blocks.0.attn.wq"]);
        assert_eq!(wq_f[0], 10_000.0);
        assert_eq!(wq_f[63], 10_063.0);
        assert!(out.contains_key("embed"));
    }

    fn f32_entry(name: &str, shape: Vec<usize>, vals: &[f32]) -> WeightEntry {
        let mut data = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        WeightEntry {
            name: name.to_string(),
            data,
            shape,
            dtype: WeightDType::F32,
            num_elements: vals.len(),
            sparsity: None,
            eliminated: false,
        }
    }

    fn read_f32(e: &WeightEntry) -> Vec<f32> {
        e.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    #[test]
    fn slice_blocks_axis1_keeps_survivor_columns_by_identity() {
        // 2x4 row-major: rows [0,1,2,3] and [10,11,12,13]. block_width=1, keep cols {0,2,3}.
        let e = f32_entry("t", vec![2, 4], &[0., 1., 2., 3., 10., 11., 12., 13.]);
        let out = slice_blocks(&e, 1, 4, 1, &[0, 2, 3]).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(read_f32(&out), vec![0., 2., 3., 10., 12., 13.]);
    }

    #[test]
    fn slice_blocks_axis1_keeps_survivor_head_blocks() {
        // 1x4, block_width=2 (2 "heads" of width 2). Keep head 1 -> cols {2,3}.
        let e = f32_entry("t", vec![1, 4], &[0., 1., 2., 3.]);
        let out = slice_blocks(&e, 1, 2, 2, &[1]).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(read_f32(&out), vec![2., 3.]);
    }

    #[test]
    fn slice_blocks_axis0_keeps_survivor_rows() {
        // 4x2 row-major. block_width=1, keep rows {1,3}.
        let e = f32_entry("t", vec![4, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let out = slice_blocks(&e, 0, 4, 1, &[1, 3]).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(read_f32(&out), vec![2., 3., 6., 7.]);
    }

    #[test]
    fn slice_blocks_rejects_out_of_range_unit() {
        let e = f32_entry("t", vec![2, 4], &[0.; 8]);
        assert!(slice_blocks(&e, 1, 4, 1, &[4]).is_err());
    }

    #[test]
    fn slice_blocks_rejects_non_ascending_units() {
        let e = f32_entry("t", vec![2, 4], &[0.; 8]);
        assert!(slice_blocks(&e, 1, 4, 1, &[2, 0]).is_err());
        // duplicates are also rejected (not strictly ascending)
        assert!(slice_blocks(&e, 1, 4, 1, &[1, 1]).is_err());
    }

    #[test]
    fn slice_blocks_rejects_truncated_data() {
        // shape claims 2x4 (32 bytes) but only 6 floats (24 bytes) of data.
        let e = f32_entry("t", vec![2, 4], &[0., 1., 2., 3., 4., 5.]);
        assert!(slice_blocks(&e, 1, 4, 1, &[0, 1]).is_err());
    }

    #[test]
    fn ffn_survivors_rejects_truncated_data() {
        // shape claims [2,4] (32 bytes) but only 6 floats (24 bytes) of data.
        let w_gate = f32_entry("ffn.w_gate", vec![2, 4], &[0., 1., 2., 3., 4., 5.]);
        assert!(ffn_survivors_by_magnitude(&w_gate, 4, 2).is_err());
    }

    #[test]
    fn ffn_survivors_keep_highest_magnitude_columns() {
        // w_gate [2, 4]: column L2 norms ~ [sqrt(0+0), sqrt(1+1), sqrt(9+9), sqrt(4+4)]
        // = col0=0, col1=1.41, col2=4.24, col3=2.83. Keep top 2 -> cols {2, 3} (ascending).
        let w_gate = f32_entry("ffn.w_gate", vec![2, 4], &[
            0., 1., 3., 2.,
            0., 1., 3., 2.,
        ]);
        let surv = ffn_survivors_by_magnitude(&w_gate, 4, 2).unwrap();
        assert_eq!(surv, vec![2, 3]);
    }

    #[test]
    fn ffn_survivors_full_width_is_identity() {
        let w_gate = f32_entry("ffn.w_gate", vec![2, 3], &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(ffn_survivors_by_magnitude(&w_gate, 3, 3).unwrap(), vec![0, 1, 2]);
    }

    // Regression: slice_blocks must return ShapeMismatch (not panic) when
    // n_units * block_width != tensor dimension on the sliced axis.
    // Before the fix this would panic with an index-out-of-bounds when the
    // last unit's byte range exceeded the row boundary.
    #[test]
    fn slice_blocks_rejects_inconsistent_n_units_axis1() {
        // 2×10 tensor; caller claims n_units=3, block_width=4 → 3*4=12 ≠ 10.
        let e = f32_entry("t", vec![2, 10], &[0.0f32; 20]);
        let err = slice_blocks(&e, 1, 3, 4, &[0, 1]);
        assert!(
            matches!(err, Err(CepSliceError::ShapeMismatch { .. })),
            "expected ShapeMismatch for n_units*block_width != cols; got {:?}", err
        );
    }

    #[test]
    fn slice_blocks_rejects_inconsistent_n_units_axis0() {
        // 10×2 tensor; caller claims n_units=3, block_width=4 → 3*4=12 ≠ 10.
        let e = f32_entry("t", vec![10, 2], &[0.0f32; 20]);
        let err = slice_blocks(&e, 0, 3, 4, &[0, 1]);
        assert!(
            matches!(err, Err(CepSliceError::ShapeMismatch { .. })),
            "expected ShapeMismatch for n_units*block_width != rows; got {:?}", err
        );
    }

    #[test]
    fn slice_blocks_accepts_exact_n_units_axis1() {
        // 2×8 tensor, n_units=4, block_width=2 → 4*2=8=cols. Keeps units {1,3}.
        let e = f32_entry("t", vec![2, 8], &[0., 1., 2., 3., 4., 5., 6., 7.,
                                             10., 11., 12., 13., 14., 15., 16., 17.]);
        let out = slice_blocks(&e, 1, 4, 2, &[1, 3]).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
        // row 0: unit 1 → cols 2,3 → [2,3]; unit 3 → cols 6,7 → [6,7]
        assert_eq!(read_f32(&out), vec![2., 3., 6., 7., 12., 13., 16., 17.]);
    }
}
