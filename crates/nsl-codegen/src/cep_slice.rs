//! CEP — weight slicing: apply a prune delta to a checkpoint.
//!
//! A pure data transform. Pruned heads lose their columns of `wq` and rows of
//! `wo`, and their key/value group its columns of `wk`/`wv`; a narrowed FFN
//! keeps the same neurons in `w_gate`, `w_up` and `w_down`; a dropped layer
//! loses every tensor it owns and the layers after it are renumbered. The
//! head counts that `GroupedQueryAttention` stores as tensors (`_n_heads`,
//! `_n_kv_heads`, `_n_rep`) are rewritten to match, because its forward
//! reshapes by them. Tensors come out in the order they went in.

use std::collections::HashMap;

use crate::cep_importance::{ffn_neuron_magnitudes, ffn_survivors};
use crate::cep_oracle::ModelSpec;
use crate::cep_rewrite::{LayerDelta, PruneDelta};
use crate::cep_weights::{BlockIndex, CepWeights, parse_block_name};
use crate::weight_aware::{WeightDType, WeightEntry, write_f64_as_dtype};

#[derive(Debug, Clone, PartialEq)]
pub enum CepSliceError {
    HeadIndexOutOfRange {
        layer: u32,
        head: u32,
        n_heads: u32,
    },
    ShapeMismatch {
        tensor: String,
        expected: String,
        found: String,
    },
    MissingTensor {
        tensor: String,
    },
    InconsistentFfn {
        layer: u32,
        detail: String,
    },
    UnitOutOfRange {
        tensor: String,
        unit: u32,
        n_units: u32,
    },
    /// A tensor or a delta entry names a layer the model does not have.
    UnknownLayer {
        what: String,
        layer: u32,
        n_layers: u32,
    },
    /// The delta asks for something the slicer cannot do consistently.
    BadDelta {
        layer: u32,
        detail: String,
    },
    /// A stored configuration scalar disagrees with the model source.
    StaleScalar {
        tensor: String,
        expected: u32,
        found: f64,
    },
    /// The weights file itself is inconsistent.
    BadWeights(String),
}

impl std::fmt::Display for CepSliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CepSliceError::HeadIndexOutOfRange {
                layer,
                head,
                n_heads,
            } => write!(
                f,
                "CEP slice: layer {layer} head {head} out of range (n_heads={n_heads})"
            ),
            CepSliceError::ShapeMismatch {
                tensor,
                expected,
                found,
            } => write!(
                f,
                "CEP slice: '{tensor}' shape mismatch\n  expected: {expected}\n  found:    {found}"
            ),
            CepSliceError::MissingTensor { tensor } => write!(
                f,
                "CEP slice: expected tensor '{tensor}' not found in weights"
            ),
            CepSliceError::InconsistentFfn { layer, detail } => {
                write!(f, "CEP slice: layer {layer} FFN inconsistency: {detail}")
            }
            CepSliceError::UnitOutOfRange {
                tensor,
                unit,
                n_units,
            } => write!(
                f,
                "CEP slice: '{tensor}' survivor unit {unit} out of range (n_units={n_units})"
            ),
            CepSliceError::UnknownLayer {
                what,
                layer,
                n_layers,
            } => write!(
                f,
                "CEP slice: {what} names layer {layer}, but the model has {n_layers} layers"
            ),
            CepSliceError::BadDelta { layer, detail } => {
                write!(f, "CEP slice: layer {layer}: {detail}")
            }
            CepSliceError::StaleScalar {
                tensor,
                expected,
                found,
            } => write!(
                f,
                "CEP slice: '{tensor}' stores {found}, but the model source says {expected} \
                           (the weights were not saved from this model)"
            ),
            CepSliceError::BadWeights(m) => write!(f, "{m}"),
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
    // The function's contract is that the sliced axis is exactly `n_units` blocks
    // of `block_width` consecutive rows/cols, i.e. `n_units * block_width == axis_dim`.
    // The `u < n_units` check below rules out unit-index overflow but is NOT
    // sufficient: if the model spec says `n_heads * head_dim` but the stored
    // tensor was shaped differently (or a future caller passes a stale `n_units`),
    // the last unit's byte range silently runs past the row/col boundary and the
    // gather panics with index-out-of-bounds. Convert that latent panic into
    // ShapeMismatch up-front.
    let axis_dim = match axis {
        0 => rows,
        1 => cols,
        _ => {
            return Err(CepSliceError::ShapeMismatch {
                tensor: entry.name.clone(),
                expected: "axis 0 or 1".into(),
                found: format!("axis {axis}"),
            });
        }
    };
    let span =
        (n_units as usize)
            .checked_mul(block)
            .ok_or_else(|| CepSliceError::ShapeMismatch {
                tensor: entry.name.clone(),
                expected: format!("n_units({n_units}) * block_width({block}) fits in usize"),
                found: "multiplication overflow".into(),
            })?;
    if span != axis_dim {
        return Err(CepSliceError::ShapeMismatch {
            tensor: entry.name.clone(),
            expected: format!("axis {axis} dim = n_units * block_width = {span}"),
            found: format!("axis {axis} dim = {axis_dim}"),
        });
    }
    for &u in survivor_units {
        if u >= n_units {
            return Err(CepSliceError::UnitOutOfRange {
                tensor: entry.name.clone(),
                unit: u,
                n_units,
            });
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
            });
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

/// Slice a model's weights to a [`PruneDelta`]. Returns the surviving tensors
/// in the order of `weights.order`.
///
/// Every tensor that belongs to a block is handled, whatever its section: a
/// dropped layer loses all of them (attention, FFN, norms, RoPE tables,
/// stored scalars) and later layers are renumbered as a whole. Names keep the
/// spelling they were read in (`blocks[0].…` or `blocks.0.…`).
pub fn apply_prune_delta_to_weights(
    weights: &CepWeights,
    spec: &ModelSpec,
    delta: &PruneDelta,
) -> Result<Vec<WeightEntry>, CepSliceError> {
    let wm = &weights.map;
    let n_layers = spec.n_layers;
    let index = BlockIndex::build(wm).map_err(CepSliceError::BadWeights)?;

    let mut by_layer: HashMap<u32, &LayerDelta> = HashMap::new();
    for ld in &delta.per_layer {
        if ld.layer >= n_layers {
            return Err(CepSliceError::UnknownLayer {
                what: "the prune delta".into(),
                layer: ld.layer,
                n_layers,
            });
        }
        by_layer.insert(ld.layer, ld);
    }
    let dropped = |l: u32| by_layer.get(&l).is_some_and(|ld| ld.drop_layer);
    // Old layer index -> new layer index, skipping dropped layers.
    let mut rekey: Vec<Option<u32>> = Vec::with_capacity(n_layers as usize);
    let mut next = 0u32;
    for l in 0..n_layers {
        if dropped(l) {
            rekey.push(None);
        } else {
            rekey.push(Some(next));
            next += 1;
        }
    }

    // Per surviving layer: which heads, key/value groups and FFN neurons stay.
    let mut plans: HashMap<u32, LayerPlan> = HashMap::new();
    for l in 0..n_layers {
        if dropped(l) {
            continue;
        }
        let Some(ld) = by_layer.get(&l) else { continue };
        plans.insert(l, LayerPlan::build(wm, &index, spec, ld)?);
    }

    let mut out = Vec::with_capacity(weights.order.len());
    for name in &weights.order {
        let entry = wm.get(name).ok_or_else(|| CepSliceError::MissingTensor {
            tensor: name.clone(),
        })?;
        let Some(block) = parse_block_name(name) else {
            out.push(entry.clone()); // embedding, final norm, ...
            continue;
        };
        if block.layer >= n_layers {
            return Err(CepSliceError::UnknownLayer {
                what: format!("tensor '{name}'"),
                layer: block.layer,
                n_layers,
            });
        }
        let Some(new_layer) = rekey[block.layer as usize] else {
            continue;
        };
        let mut sliced = match plans.get(&block.layer) {
            Some(plan) => plan.slice(entry, block.rest)?,
            None => entry.clone(),
        };
        sliced.name = block.with_layer(new_layer);
        out.push(sliced);
    }
    Ok(out)
}

/// What survives in one layer.
struct LayerPlan {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    d_ff: u32,
    /// Surviving query heads and key/value heads, each ascending. `None`
    /// when no head is pruned.
    heads: Option<(Vec<u32>, Vec<u32>)>,
    /// Surviving FFN neurons, ascending. `None` when the FFN keeps its width.
    neurons: Option<Vec<u32>>,
}

impl LayerPlan {
    fn build(
        wm: &crate::weight_aware::WeightMap,
        index: &BlockIndex,
        spec: &ModelSpec,
        ld: &LayerDelta,
    ) -> Result<Self, CepSliceError> {
        let layer = ld.layer;
        let l = layer as usize;
        let n_heads = spec.n_heads[l];
        let n_kv = spec.n_kv_heads[l].max(1);
        let group = (n_heads / n_kv).max(1);

        let heads = if ld.pruned_heads.is_empty() {
            None
        } else {
            if let Some(&h) = ld.pruned_heads.iter().find(|&&h| h >= n_heads) {
                return Err(CepSliceError::HeadIndexOutOfRange {
                    layer,
                    head: h,
                    n_heads,
                });
            }
            // A key/value head serves a whole group of query heads, so the
            // group goes or stays as one.
            let whole_groups = ld.pruned_heads.iter().all(|&h| {
                let base = (h / group) * group;
                (base..base + group).all(|x| ld.pruned_heads.contains(&x))
            });
            if !whole_groups {
                return Err(CepSliceError::BadDelta {
                    layer,
                    detail: format!(
                        "pruned heads {:?} are not whole key/value groups of {group}",
                        ld.pruned_heads
                    ),
                });
            }
            let q: Vec<u32> = (0..n_heads)
                .filter(|h| !ld.pruned_heads.contains(h))
                .collect();
            let kv: Vec<u32> = (0..n_kv)
                .filter(|&g| !ld.pruned_heads.contains(&(g * group)))
                .collect();
            if q.is_empty() {
                return Err(CepSliceError::BadDelta {
                    layer,
                    detail: "every head is pruned; drop the layer instead".into(),
                });
            }
            Some((q, kv))
        };

        let neurons = match ld.new_d_ff {
            Some(new_ff) if new_ff < spec.d_ff[l] => {
                if new_ff == 0 {
                    return Err(CepSliceError::BadDelta {
                        layer,
                        detail: "an FFN width of 0 was requested; drop the layer instead".into(),
                    });
                }
                let fetch = |rest: &str| {
                    index
                        .get(wm, layer, rest)
                        .ok_or_else(|| CepSliceError::MissingTensor {
                            tensor: format!("blocks[{layer}].{rest}"),
                        })
                };
                let mags = ffn_neuron_magnitudes(
                    fetch("ffn.w_gate")?,
                    fetch("ffn.w_up")?,
                    fetch("ffn.w_down")?,
                    spec.d_model,
                    spec.d_ff[l],
                )
                .map_err(|detail| CepSliceError::InconsistentFfn { layer, detail })?;
                Some(ffn_survivors(&mags, new_ff as usize))
            }
            Some(new_ff) if new_ff > spec.d_ff[l] => {
                return Err(CepSliceError::BadDelta {
                    layer,
                    detail: format!(
                        "FFN width {new_ff} is wider than the model's {}",
                        spec.d_ff[l]
                    ),
                });
            }
            _ => None,
        };

        Ok(Self {
            n_heads,
            n_kv,
            head_dim: spec.head_dim[l],
            d_ff: spec.d_ff[l],
            heads,
            neurons,
        })
    }

    /// Slice one of this layer's tensors, named by its path inside the block.
    fn slice(&self, entry: &WeightEntry, rest: &str) -> Result<WeightEntry, CepSliceError> {
        match (rest, &self.heads, &self.neurons) {
            ("attn.wq", Some((q, _)), _) => slice_blocks(entry, 1, self.n_heads, self.head_dim, q),
            ("attn.wo", Some((q, _)), _) => slice_blocks(entry, 0, self.n_heads, self.head_dim, q),
            ("attn.wk" | "attn.wv", Some((_, kv)), _) => {
                slice_blocks(entry, 1, self.n_kv, self.head_dim, kv)
            }
            // `GroupedQueryAttention` stores its head counts as tensors and
            // reshapes by them, so they must follow the projections.
            ("attn._n_heads", heads, _) => {
                let new = heads.as_ref().map_or(self.n_heads, |(q, _)| q.len() as u32);
                rewrite_scalar(entry, self.n_heads, new)
            }
            ("attn._n_kv_heads", heads, _) => {
                let new = heads.as_ref().map_or(self.n_kv, |(_, kv)| kv.len() as u32);
                rewrite_scalar(entry, self.n_kv, new)
            }
            ("attn._n_rep", _, _) => {
                // Whole groups go, so heads per key/value head is unchanged.
                let rep = self.n_heads / self.n_kv;
                rewrite_scalar(entry, rep, rep)
            }
            ("ffn.w_gate" | "ffn.w_up", _, Some(n)) => slice_blocks(entry, 1, self.d_ff, 1, n),
            ("ffn.w_down", _, Some(n)) => slice_blocks(entry, 0, self.d_ff, 1, n),
            _ => Ok(entry.clone()), // norms, RoPE tables, other scalars
        }
    }
}

/// Replace a one-element configuration tensor's value, after checking that it
/// held what the model source says it should.
fn rewrite_scalar(
    entry: &WeightEntry,
    expected: u32,
    new: u32,
) -> Result<WeightEntry, CepSliceError> {
    let bw = entry.dtype.byte_width();
    if entry.num_elements != 1 || entry.data.len() != bw {
        return Err(CepSliceError::ShapeMismatch {
            tensor: entry.name.clone(),
            expected: "a one-element tensor".into(),
            found: format!("{:?}", entry.shape),
        });
    }
    let found = entry.dtype.to_f64(&entry.data);
    if found != expected as f64 {
        return Err(CepSliceError::StaleScalar {
            tensor: entry.name.clone(),
            expected,
            found,
        });
    }
    let mut out = entry.clone();
    write_f64_as_dtype(new as f64, entry.dtype, &mut out.data);
    Ok(out)
}

/// Serialise sliced entries to `path`: `.nslm` (in the order given, which is
/// what `model_load` needs) or `.safetensors` (which its format sorts).
pub fn write_sliced_weights(
    entries: &[WeightEntry],
    path: &std::path::Path,
) -> std::io::Result<()> {
    if crate::cep_weights::wants_nslm(path) {
        let refs: Vec<&WeightEntry> = entries.iter().collect();
        return crate::cep_weights::write_nslm(&refs, path);
    }
    crate::cep_weights::write_atomically(path, &serialize_safetensors(entries)?)
}

/// The bytes of a `.safetensors` file holding `entries`.
pub fn serialize_safetensors(entries: &[WeightEntry]) -> std::io::Result<Vec<u8>> {
    use safetensors::tensor::TensorView;
    let invalid = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let mut views: Vec<(String, TensorView)> = Vec::with_capacity(entries.len());
    for e in entries {
        let v = TensorView::new(to_safetensors_dtype(e.dtype), e.shape.clone(), &e.data)
            .map_err(|err| invalid(err.to_string()))?;
        views.push((e.name.clone(), v));
    }
    safetensors::serialize(views, None).map_err(|e| invalid(e.to_string()))
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
            d_model: 8,
            n_layers: 2,
            n_heads: vec![4, 4],
            n_kv_heads: vec![2, 2],
            head_dim: vec![2, 2],
            d_ff: vec![6, 6],
            vocab: 16,
            max_seq: 32,
            batch: 1,
            activation: Activation::SwiGlu,
            norm: NormType::RmsNorm,
            dtype_bytes: 4,
            tied_lm_head: true,
        }
    }

    /// A checkpoint shaped like one `model_save` writes for `small_spec`, in
    /// model order: the embedding, then per layer the norm, the attention's
    /// stored scalars and projections, the second norm and the FFN, then the
    /// final norm. Values are ramps offset by 10000 per layer, so a tensor's
    /// layer can be read off its data.
    fn small_weights(spec: &ModelSpec, bracket: bool) -> CepWeights {
        let mut order = Vec::new();
        let mut entries = std::collections::HashMap::new();
        let mut put = |name: String, shape: Vec<usize>, vals: Vec<f32>| {
            entries.insert(name.clone(), f32_entry(&name, shape, &vals));
            order.push(name);
        };
        let ramp = |n: usize, base: f32| -> Vec<f32> { (0..n).map(|i| base + i as f32).collect() };
        let dm = spec.d_model as usize;
        put(
            "embed".into(),
            vec![spec.vocab as usize, dm],
            ramp(spec.vocab as usize * dm, 0.0),
        );
        for l in 0..spec.n_layers as usize {
            let nh = spec.n_heads[l] as usize;
            let nkv = spec.n_kv_heads[l] as usize;
            let hd = spec.head_dim[l] as usize;
            let dff = spec.d_ff[l] as usize;
            let base = (l as f32) * 10_000.0;
            let p = |rest: &str| {
                if bracket {
                    format!("blocks[{l}].{rest}")
                } else {
                    format!("blocks.{l}.{rest}")
                }
            };
            put(p("attn_norm.weight"), vec![dm], vec![1.0 + l as f32; dm]);
            put(p("attn._n_heads"), vec![1], vec![nh as f32]);
            put(p("attn._n_kv_heads"), vec![1], vec![nkv as f32]);
            put(p("attn._head_dim"), vec![1], vec![hd as f32]);
            put(p("attn._n_rep"), vec![1], vec![(nh / nkv) as f32]);
            put(p("attn.wq"), vec![dm, nh * hd], ramp(dm * nh * hd, base));
            put(p("attn.wk"), vec![dm, nkv * hd], ramp(dm * nkv * hd, base));
            put(p("attn.wv"), vec![dm, nkv * hd], ramp(dm * nkv * hd, base));
            put(p("attn.wo"), vec![nh * hd, dm], ramp(nh * hd * dm, base));
            put(p("ffn_norm.weight"), vec![dm], vec![1.5 + l as f32; dm]);
            put(p("ffn.w_gate"), vec![dm, dff], ramp(dm * dff, base));
            put(p("ffn.w_up"), vec![dm, dff], ramp(dm * dff, base));
            put(p("ffn.w_down"), vec![dff, dm], ramp(dff * dm, base));
        }
        put("norm.weight".into(), vec![dm], vec![9.0; dm]);
        CepWeights {
            map: WeightMap::from_entries(entries),
            order,
            format: crate::cep_weights::WeightFormat::Nslm,
        }
    }

    fn find<'a>(out: &'a [WeightEntry], name: &str) -> &'a WeightEntry {
        out.iter().find(|e| e.name == name).unwrap_or_else(|| {
            panic!(
                "no tensor '{name}' in {:?}",
                out.iter().map(|e| &e.name).collect::<Vec<_>>()
            )
        })
    }

    #[test]
    fn orchestrator_slices_heads_ffn_and_passes_through() {
        let spec = small_spec();
        let w = small_weights(&spec, false);
        // Drop group 0 (heads 0,1) of layer 0 -> 2 surviving heads {2,3}, 1 kv head {1}.
        // Narrow layer 0 FFN to 4. Layer 1 untouched.
        let delta = PruneDelta {
            per_layer: vec![
                LayerDelta {
                    layer: 0,
                    pruned_heads: vec![0, 1],
                    new_d_ff: Some(4),
                    drop_layer: false,
                },
                LayerDelta {
                    layer: 1,
                    pruned_heads: vec![],
                    new_d_ff: None,
                    drop_layer: false,
                },
            ],
        };
        let out = apply_prune_delta_to_weights(&w, &spec, &delta).unwrap();

        // wq [8, 8] -> survivor heads {2,3} -> cols {4,5,6,7} -> [8, 4]
        let wq = find(&out, "blocks.0.attn.wq");
        assert_eq!(wq.shape, vec![8, 4]);
        // row 0 of the original is 0..8; columns 4..8 of it are 4,5,6,7.
        assert_eq!(&read_f32(wq)[0..4], &[4., 5., 6., 7.]);
        // wk [8,4] -> survivor kv head {1} -> cols {2,3} -> [8,2]; row 0 was 0..4.
        let wk = find(&out, "blocks.0.attn.wk");
        assert_eq!(wk.shape, vec![8, 2]);
        assert_eq!(&read_f32(wk)[0..2], &[2., 3.]);
        // wo [8,8] -> survivor heads {2,3} -> rows {4,5,6,7} -> [4,8]; row 4 starts at 32.
        let wo = find(&out, "blocks.0.attn.wo");
        assert_eq!(wo.shape, vec![4, 8]);
        assert_eq!(read_f32(wo)[0], 32.0);
        // FFN narrowed to 4 across all three, keeping the same neurons in each.
        // With ramp values the last columns/rows are the largest, so neurons
        // 2..6 survive: w_gate row 0 was 0..6 -> 2,3,4,5; w_down row 2 starts at 16.
        let gate = find(&out, "blocks.0.ffn.w_gate");
        assert_eq!(gate.shape, vec![8, 4]);
        assert_eq!(&read_f32(gate)[0..4], &[2., 3., 4., 5.]);
        assert_eq!(find(&out, "blocks.0.ffn.w_up").shape, vec![8, 4]);
        let down = find(&out, "blocks.0.ffn.w_down");
        assert_eq!(down.shape, vec![4, 8]);
        assert_eq!(read_f32(down)[0], 16.0);
        // layer 1 untouched
        assert_eq!(find(&out, "blocks.1.attn.wq").shape, vec![8, 8]);
        // passthrough
        assert_eq!(find(&out, "embed").shape, vec![16, 8]);
    }

    /// `GroupedQueryAttention.forward` reshapes by the head counts it stores
    /// as tensors, so a sliced `wq` beside the old `_n_heads` cannot run.
    #[test]
    fn stored_head_counts_follow_the_projections() {
        let spec = small_spec();
        let w = small_weights(&spec, true);
        let delta = PruneDelta {
            per_layer: vec![
                LayerDelta {
                    layer: 0,
                    pruned_heads: vec![2, 3],
                    new_d_ff: None,
                    drop_layer: false,
                },
                LayerDelta {
                    layer: 1,
                    pruned_heads: vec![],
                    new_d_ff: None,
                    drop_layer: false,
                },
            ],
        };
        let out = apply_prune_delta_to_weights(&w, &spec, &delta).unwrap();
        assert_eq!(read_f32(find(&out, "blocks[0].attn._n_heads")), vec![2.0]);
        assert_eq!(
            read_f32(find(&out, "blocks[0].attn._n_kv_heads")),
            vec![1.0]
        );
        assert_eq!(read_f32(find(&out, "blocks[0].attn._n_rep")), vec![2.0]);
        assert_eq!(read_f32(find(&out, "blocks[0].attn._head_dim")), vec![2.0]);
        // The untouched layer keeps its counts.
        assert_eq!(read_f32(find(&out, "blocks[1].attn._n_heads")), vec![4.0]);
        assert_eq!(
            read_f32(find(&out, "blocks[1].attn._n_kv_heads")),
            vec![2.0]
        );
    }

    #[test]
    fn a_stored_head_count_that_disagrees_with_the_source_is_refused() {
        let spec = small_spec();
        let mut w = small_weights(&spec, true);
        w.map
            .insert(f32_entry("blocks[0].attn._n_heads", vec![1], &[8.0]));
        let delta = PruneDelta {
            per_layer: vec![LayerDelta {
                layer: 0,
                pruned_heads: vec![0, 1],
                new_d_ff: None,
                drop_layer: false,
            }],
        };
        let err = apply_prune_delta_to_weights(&w, &spec, &delta).unwrap_err();
        assert!(
            matches!(err, CepSliceError::StaleScalar { expected: 4, .. }),
            "{err}"
        );
    }

    /// Tensors come out in the order they went in: `model_load` copies by
    /// position.
    #[test]
    fn output_keeps_the_input_order() {
        let spec = small_spec();
        let w = small_weights(&spec, true);
        let delta = PruneDelta {
            per_layer: vec![
                LayerDelta {
                    layer: 0,
                    pruned_heads: vec![0, 1],
                    new_d_ff: Some(4),
                    drop_layer: false,
                },
                LayerDelta {
                    layer: 1,
                    pruned_heads: vec![0, 1],
                    new_d_ff: Some(4),
                    drop_layer: false,
                },
            ],
        };
        let out = apply_prune_delta_to_weights(&w, &spec, &delta).unwrap();
        let names: Vec<&str> = out.iter().map(|e| e.name.as_str()).collect();
        let want: Vec<&str> = w.order.iter().map(String::as_str).collect();
        assert_eq!(names, want);
    }

    #[test]
    fn dropping_a_layer_moves_every_tensor_of_the_layers_after_it() {
        let spec = small_spec();
        for bracket in [true, false] {
            let w = small_weights(&spec, bracket);
            let name = |l: u32, rest: &str| {
                if bracket {
                    format!("blocks[{l}].{rest}")
                } else {
                    format!("blocks.{l}.{rest}")
                }
            };
            // Drop layer 0 entirely -> layer 1 becomes layer 0.
            let delta = PruneDelta {
                per_layer: vec![
                    LayerDelta {
                        layer: 0,
                        pruned_heads: vec![],
                        new_d_ff: None,
                        drop_layer: true,
                    },
                    LayerDelta {
                        layer: 1,
                        pruned_heads: vec![],
                        new_d_ff: None,
                        drop_layer: false,
                    },
                ],
            };
            let out = apply_prune_delta_to_weights(&w, &spec, &delta).unwrap();
            assert!(
                out.iter()
                    .all(|e| !e.name.contains("blocks[1]") && !e.name.contains("blocks.1.")),
                "nothing may be left under the old layer-1 name"
            );
            // The projection holds the ORIGINAL layer 1's values (base 10000)...
            let wq = read_f32(find(&out, &name(0, "attn.wq")));
            assert_eq!((wq[0], wq[63]), (10_000.0, 10_063.0));
            // ...and so do the norms, which the layer owns just as much: layer
            // 1's scales were 2.0 and 2.5, the dropped layer's 1.0 and 1.5.
            assert_eq!(
                read_f32(find(&out, &name(0, "attn_norm.weight"))),
                vec![2.0; 8]
            );
            assert_eq!(
                read_f32(find(&out, &name(0, "ffn_norm.weight"))),
                vec![2.5; 8]
            );
            // 13 per-layer tensors + embed + final norm.
            assert_eq!(out.len(), 15);
            assert_eq!(read_f32(find(&out, "norm.weight")), vec![9.0; 8]);
        }
    }

    #[test]
    fn a_tensor_for_a_layer_the_model_lacks_is_refused() {
        let spec = small_spec();
        let mut w = small_weights(&spec, false);
        w.map
            .insert(f32_entry("blocks.99.ffn.w_gate", vec![8, 6], &[0.0; 48]));
        w.order.push("blocks.99.ffn.w_gate".into());
        let delta = PruneDelta::for_spec(&spec);
        let err = apply_prune_delta_to_weights(&w, &spec, &delta).unwrap_err();
        assert!(
            matches!(
                err,
                CepSliceError::UnknownLayer {
                    layer: 99,
                    n_layers: 2,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn a_delta_for_a_layer_the_model_lacks_is_refused() {
        let spec = small_spec();
        let w = small_weights(&spec, false);
        let delta = PruneDelta {
            per_layer: vec![LayerDelta {
                layer: 99,
                pruned_heads: vec![],
                new_d_ff: Some(2),
                drop_layer: false,
            }],
        };
        let err = apply_prune_delta_to_weights(&w, &spec, &delta).unwrap_err();
        assert!(
            matches!(err, CepSliceError::UnknownLayer { layer: 99, .. }),
            "{err}"
        );
    }

    #[test]
    fn half_a_key_value_group_is_refused() {
        let spec = small_spec();
        let w = small_weights(&spec, false);
        let delta = PruneDelta {
            per_layer: vec![LayerDelta {
                layer: 0,
                pruned_heads: vec![0],
                new_d_ff: None,
                drop_layer: false,
            }],
        };
        let err = apply_prune_delta_to_weights(&w, &spec, &delta).unwrap_err();
        assert!(
            matches!(err, CepSliceError::BadDelta { layer: 0, .. }),
            "{err}"
        );
    }

    /// 64 neurons with a large gate column and a zero `w_down` row contribute
    /// nothing; the other 64 have a gate ten times smaller and live `w_up` and
    /// `w_down`. Narrowing to 64 must keep the live ones. (Ranking by the gate
    /// column alone keeps exactly the dead ones.)
    #[test]
    fn ffn_narrowing_keeps_live_neurons_over_dead_ones_with_large_gates() {
        let spec = ModelSpec {
            d_model: 2,
            n_layers: 1,
            n_heads: vec![1],
            n_kv_heads: vec![1],
            head_dim: vec![2],
            d_ff: vec![128],
            vocab: 4,
            max_seq: 8,
            batch: 1,
            activation: Activation::SwiGlu,
            norm: NormType::RmsNorm,
            dtype_bytes: 4,
            tied_lm_head: true,
        };
        let mut entries = std::collections::HashMap::new();
        let mut order = Vec::new();
        let mut put = |name: &str, shape: Vec<usize>, vals: Vec<f32>| {
            entries.insert(name.to_string(), f32_entry(name, shape, &vals));
            order.push(name.to_string());
        };
        put("blocks[0].attn.wq", vec![2, 2], vec![1.0; 4]);
        put("blocks[0].attn.wk", vec![2, 2], vec![1.0; 4]);
        put("blocks[0].attn.wv", vec![2, 2], vec![1.0; 4]);
        put("blocks[0].attn.wo", vec![2, 2], vec![1.0; 4]);
        // Columns 0..64: gate 10; columns 64..128: gate 1.
        let gate: Vec<f32> = (0..2 * 128)
            .map(|i| if i % 128 < 64 { 10.0 } else { 1.0 })
            .collect();
        put("blocks[0].ffn.w_gate", vec![2, 128], gate);
        put("blocks[0].ffn.w_up", vec![2, 128], vec![1.0; 256]);
        // Rows 0..64 zero; row j >= 64 holds the value j.
        let down: Vec<f32> = (0..128 * 2)
            .map(|i| if i / 2 < 64 { 0.0 } else { (i / 2) as f32 })
            .collect();
        put("blocks[0].ffn.w_down", vec![128, 2], down);
        let w = CepWeights {
            map: WeightMap::from_entries(entries),
            order,
            format: crate::cep_weights::WeightFormat::Nslm,
        };
        let delta = PruneDelta {
            per_layer: vec![LayerDelta {
                layer: 0,
                pruned_heads: vec![],
                new_d_ff: Some(64),
                drop_layer: false,
            }],
        };
        let out = apply_prune_delta_to_weights(&w, &spec, &delta).unwrap();
        let down = read_f32(find(&out, "blocks[0].ffn.w_down"));
        // Surviving rows are the original rows 64..128, in order.
        let want: Vec<f32> = (64..128).flat_map(|j| [j as f32, j as f32]).collect();
        assert_eq!(down, want);
        assert!(
            read_f32(find(&out, "blocks[0].ffn.w_gate"))
                .iter()
                .all(|&v| v == 1.0)
        );
    }

    #[test]
    fn safetensors_and_nslm_outputs_hold_the_same_tensors() {
        let spec = small_spec();
        let w = small_weights(&spec, true);
        let out = apply_prune_delta_to_weights(&w, &spec, &PruneDelta::for_spec(&spec)).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let st = dir.path().join("o.safetensors");
        let nslm = dir.path().join("o.nslm");
        write_sliced_weights(&out, &st).unwrap();
        write_sliced_weights(&out, &nslm).unwrap();
        let a = crate::cep_weights::load_weights(&st).unwrap();
        let b = crate::cep_weights::load_weights(&nslm).unwrap();
        // .nslm keeps the model's order; both hold identical bytes per name.
        assert_eq!(b.order, w.order);
        assert_eq!(a.map.len(), b.map.len());
        for name in &w.order {
            assert_eq!(
                a.map.get(name).unwrap().data,
                b.map.get(name).unwrap().data,
                "{name}"
            );
            assert_eq!(
                a.map.get(name).unwrap().shape,
                b.map.get(name).unwrap().shape,
                "{name}"
            );
        }
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
        e.data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
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

    // Regression: prior to the span check, callers passing `n_units * block_width`
    // larger or smaller than the axis dim could reach the gather loop and either
    // panic (span > axis_dim — last block reads past data.len()) or silently
    // succeed with a wrong-shape result (span < axis_dim — trailing data unused).
    // Both spec/shape mismatches now surface as ShapeMismatch up-front.

    #[test]
    fn slice_blocks_rejects_axis1_span_exceeds_cols() {
        // shape 2x4, axis=1, n_units=3, block_width=2 -> span=6 > cols=4.
        // Without the guard: gather index off=row_off+4*bw runs past row in last unit.
        let e = f32_entry("t", vec![2, 4], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let err = slice_blocks(&e, 1, 3, 2, &[0, 1, 2]).unwrap_err();
        match err {
            CepSliceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn slice_blocks_rejects_axis0_span_exceeds_rows() {
        // shape 4x2, axis=0, n_units=3, block_width=2 -> span=6 > rows=4.
        // Without the guard: start=4 in last unit -> r in 4..6 -> off=4*row_bytes
        // runs past data.len() and panics.
        let e = f32_entry("t", vec![4, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let err = slice_blocks(&e, 0, 3, 2, &[0, 1, 2]).unwrap_err();
        match err {
            CepSliceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn slice_blocks_rejects_span_smaller_than_axis_dim() {
        // shape 2x4, axis=1, n_units=2, block_width=1 -> span=2 < cols=4.
        // Without the guard: gather succeeds but produces wrong-shape output
        // (would silently drop trailing cols 2,3 from the spec's view).
        let e = f32_entry("t", vec![2, 4], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let err = slice_blocks(&e, 1, 2, 1, &[0, 1]).unwrap_err();
        match err {
            CepSliceError::ShapeMismatch { .. } => {}
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }
}
