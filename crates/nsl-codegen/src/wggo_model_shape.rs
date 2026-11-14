//! WGGO — the shapes of the model being compiled.
//!
//! Every time, memory and budget figure WGGO produces is a function of each
//! layer's shape: how many parameters it has, how wide its matrix products
//! are, how many attention heads of what width it runs, and how many tokens
//! pass through it per micro-batch. This module reads those from the train
//! block's own forward tape.
//!
//! The source is the same one the transient arena uses
//! (`stmt_train/transient_arena_projection.rs`): the dims of the model's
//! tensor fields (literal initialisers and constructor-folded ones), the
//! values of its one-element configuration fields (`_n_heads`), the semantic
//! types of the step's inputs, and the batch shape a proven `DataLoader`
//! yields, propagated through the tape by
//! [`crate::transient_arena::propagate_size_info`]. Nothing is assumed: a
//! trainable parameter, a product with a weight, a normalisation or an
//! attention query whose shape the propagation cannot prove makes
//! [`derive`] return [`ShapeUnknown`], and the caller then declines to plan
//! instead of substituting a layer that is not the model's.
//!
//! What is read, and what is not:
//!
//! * Parameters are the trainable ones (`stmt::is_trainable_param_leaf_name`):
//!   they are what the optimizer holds moments for. Buffers and one-element
//!   config fields (`_cos_table`, `_n_heads`) are on the tape as parameters
//!   but are not counted.
//! * A matrix product is recorded when one operand is a weight (a parameter,
//!   possibly through a transpose or reshape). The products of two
//!   activations inside attention (QKᵀ, AV) are not part of the cost model
//!   and need no shape here.
//! * Attention is found as a scaled-dot-product-attention op, or, for the
//!   decomposed masked form, as the softmax over a product of two
//!   activations; its geometry is the query operand's shape.
//! * Activation bytes are the sized results; results of unknown size are
//!   counted in [`LayerGeometry::unsized_ops`] and make the figure a lower
//!   bound.
//!
//! Parameter roles (Q/K/V/O projection, FFN projection) come from
//! [`crate::wggo_ilp::proj_role`], the name table WRGA's placement filter
//! already uses. They decide which products shrink when a head is dropped;
//! a model that names its projections differently still gets exact sizes for
//! the configuration it is written in, and offers no head decision.

use std::collections::{HashMap, HashSet};

use crate::transient_arena::SizeInfo;
use crate::wengert::{PrimalOp, VarId, WengertList, WengertType, is_view_producing_op};
use crate::wggo_cost::LayerShape;
use crate::wggo_graph::OptGraph;
use crate::wggo_ilp::{ProjRole, proj_role};

/// The geometry of a layer's attention call, read off the `[batch, heads,
/// seq, head_dim]` query operand of its scaled-dot-product-attention op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionGeometry {
    pub batch: u64,
    pub n_heads: u64,
    /// Key/value heads before grouped-query expansion: the K projection's
    /// output width over `head_dim`. Equal to `n_heads` without GQA.
    pub n_kv_heads: u64,
    pub seq: u64,
    pub head_dim: u64,
    /// Rows of the Q projection's weight: the residual-stream width.
    pub d_model: u64,
}

impl AttentionGeometry {
    /// Query heads per key/value head.
    pub fn group(&self) -> u64 {
        (self.n_heads / self.n_kv_heads.max(1)).max(1)
    }
}

/// One matrix product `[rows, in_dim] x [in_dim, out_dim]` of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulGeometry {
    /// Product of every dimension of the left operand but the last: the
    /// number of row vectors multiplied (tokens per micro-batch, for a
    /// projection applied to `[batch, seq, in_dim]`).
    pub rows: u64,
    pub in_dim: u64,
    pub out_dim: u64,
    /// Role of the weight operand, when it is a parameter with a recognised
    /// projection name.
    pub role: Option<ProjRole>,
}

/// What WGGO knows about one layer of the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGeometry {
    pub name: String,
    /// Elements of every parameter the layer owns. One-element
    /// configuration fields are included; they are parameters of the tape.
    pub param_elements: u64,
    /// Elements of the layer's Q / K / V / O projection weights. The cost
    /// model scales these with the number of heads a candidate keeps.
    pub q_elements: u64,
    pub kv_elements: u64,
    pub o_elements: u64,
    pub attention: Option<AttentionGeometry>,
    pub matmuls: Vec<MatmulGeometry>,
    /// `(elements, width)` of each normalisation op's input.
    pub norms: Vec<(u64, u64)>,
    /// Elements of the layer's forward activations: the results of its
    /// tensor ops that allocate (views and parameters excluded).
    pub activation_elements: u64,
    /// Tensor ops counted in `activation_elements`, and tensor ops whose
    /// size the propagation could not prove (not counted).
    pub sized_ops: u32,
    pub unsized_ops: u32,
    /// Inner width of the layer's FFN: the largest output width among its
    /// FFN-role projections, 0 when it has none.
    pub ffn_width: u64,
    /// Last dimension of the tensors the layer reads from other layers and
    /// of the tensors other layers read from it, when each is unique.
    pub input_width: Option<u64>,
    pub output_width: Option<u64>,
}

impl LayerGeometry {
    /// Query-head count, 0 for a layer without an attention call.
    pub fn n_heads(&self) -> u64 {
        self.attention.map_or(0, |a| a.n_heads)
    }

    /// Parameter elements when `heads` of the layer's query heads are kept:
    /// the Q and O projections shrink in proportion, the K and V projections
    /// by whole key/value groups.
    pub fn param_elements_at(&self, heads: u64) -> u64 {
        let Some(a) = self.attention else {
            return self.param_elements;
        };
        let h = heads.min(a.n_heads);
        if h == a.n_heads {
            return self.param_elements;
        }
        let kv_kept = h.div_ceil(a.group()).min(a.n_kv_heads);
        let fixed = self
            .param_elements
            .saturating_sub(self.q_elements + self.kv_elements + self.o_elements);
        fixed
            + (self.q_elements + self.o_elements) / a.n_heads.max(1) * h
            + self.kv_elements / a.n_kv_heads.max(1) * kv_kept
    }

    /// The nominal-shape record the head scorer and CSHA take.
    pub fn layer_shape(&self, dtype_bytes: u64) -> Option<LayerShape> {
        let a = self.attention?;
        Some(LayerShape {
            batch: a.batch,
            seq: a.seq,
            d_model: a.d_model,
            head_dim: a.head_dim,
            n_kv_heads: a.n_kv_heads,
            dtype_bytes,
        })
    }
}

/// Per-layer geometry of the model, parallel to [`OptGraph::layers`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelGeometry {
    pub layers: Vec<LayerGeometry>,
    /// Bytes per element of the model's tensors on the compile target.
    pub dtype_bytes: u64,
}

impl ModelGeometry {
    pub fn total_param_elements(&self) -> u64 {
        self.layers.iter().map(|l| l.param_elements).sum()
    }

    /// The attention shape shared by every attention layer, or `None` when
    /// the model has no attention call or its attention layers differ.
    pub fn uniform_attention(&self) -> Option<AttentionGeometry> {
        let mut it = self.layers.iter().filter_map(|l| l.attention);
        let first = it.next()?;
        it.all(|a| a == first).then_some(first)
    }
}

/// Why the model's shapes could not be determined. Each reason names the
/// tape value involved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeUnknown {
    pub reasons: Vec<String>,
}

impl ShapeUnknown {
    /// The first few reasons, for a one-line diagnostic.
    pub fn summary(&self) -> String {
        const SHOWN: usize = 6;
        let mut s = self
            .reasons
            .iter()
            .take(SHOWN)
            .cloned()
            .collect::<Vec<_>>()
            .join("; ");
        if self.reasons.len() > SHOWN {
            s.push_str(&format!("; and {} more", self.reasons.len() - SHOWN));
        }
        s
    }
}

impl std::fmt::Display for ShapeUnknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

fn dims_u64(info: &SizeInfo) -> Option<Vec<u64>> {
    match info {
        SizeInfo::Shape(d) => Some(d.iter().map(|&x| x as u64).collect()),
        SizeInfo::Numel(_) => None,
    }
}

fn is_attention(op: &PrimalOp) -> bool {
    matches!(
        op,
        PrimalOp::ScaledDotProductAttention { .. } | PrimalOp::ScaledDotProductAttentionPacked
    )
}

/// True when the train step consumes a packed batch: its tape holds a
/// packed attention call, or reads the batch's `attention_mask` or
/// `segment_ids` (the masked form of attention is decomposed into plain ops,
/// so the read of the mask is what marks it). Only then is there something
/// for a packing decision to apply to.
pub fn tape_has_packed_attention(list: &WengertList) -> bool {
    list.ops.iter().any(|op| match &op.op {
        PrimalOp::ScaledDotProductAttentionPacked => true,
        PrimalOp::Passthrough(name) => {
            matches!(
                name.as_str(),
                "dict_get:attention_mask" | "dict_get:segment_ids"
            )
        }
        _ => false,
    })
}

/// Read every layer's geometry from the forward tape.
///
/// `sizes` is the result of [`crate::transient_arena::propagate_size_info`]
/// over `list`; `graph` is [`crate::wggo_graph::build`] of the same list.
pub fn derive(
    list: &WengertList,
    graph: &OptGraph,
    sizes: &HashMap<VarId, SizeInfo>,
    dtype_bytes: u64,
) -> Result<ModelGeometry, ShapeUnknown> {
    derive_for(list, graph, sizes, dtype_bytes, None)
}

/// [`derive`], told which of the tape's `Param` ops are the model's trained
/// tensor parameters.
///
/// The tape registers every model field a step reads as a `Param`: the
/// weights, but also buffers (`_cos_table`), one-element config tensors
/// (`_n_heads`) and plain `float` fields (a norm's `eps`). Only the first
/// kind is a parameter of the model. `trained` is the train block's own
/// parameter list (`Compiler::enumerate_model_tensor_paths`); when it is
/// `None`, or names nothing on this tape, a `Param` counts when its leaf
/// name is a trainable one and the tape types it as a tensor.
pub fn derive_for(
    list: &WengertList,
    graph: &OptGraph,
    sizes: &HashMap<VarId, SizeInfo>,
    dtype_bytes: u64,
    trained: Option<&HashSet<String>>,
) -> Result<ModelGeometry, ShapeUnknown> {
    let trained = trained.filter(|set| {
        list.ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Param(n) if set.contains(n)))
    });
    let mut reasons: Vec<String> = Vec::new();
    let param_name: HashMap<VarId, &str> = list
        .ops
        .iter()
        .filter_map(|op| match &op.op {
            PrimalOp::Param(n) => Some((op.result, n.as_str())),
            _ => None,
        })
        .collect();
    let op_layer = attribute_ops(list, graph);
    let layer_of_var: HashMap<VarId, u32> = list
        .ops
        .iter()
        .zip(op_layer.iter())
        .filter_map(|(op, l)| l.map(|l| (op.result, l)))
        .collect();
    let result_ty = |op: &crate::wengert::WengertOp| {
        list.var_types
            .get(&op.result)
            .copied()
            .unwrap_or_else(|| crate::wengert::type_for_op(&op.op))
    };
    let last_dim =
        |v: VarId| -> Option<u64> { dims_u64(sizes.get(&v)?).and_then(|d| d.last().copied()) };
    let producer: HashMap<VarId, &crate::wengert::WengertOp> =
        list.ops.iter().map(|op| (op.result, op)).collect();
    // The parameter a value is, or is a view of (a transposed or reshaped
    // weight is still that weight).
    let weight_of = |mut v: VarId| -> Option<&str> {
        for _ in 0..8 {
            if let Some(name) = param_name.get(&v) {
                return Some(name);
            }
            let op = producer.get(&v)?;
            if !is_view_producing_op(&op.op) {
                return None;
            }
            v = *op.inputs.first()?;
        }
        None
    };

    let mut layers = Vec::with_capacity(graph.layers.len());
    for layer in &graph.layers {
        let mut g = LayerGeometry {
            name: layer.name.clone(),
            param_elements: 0,
            q_elements: 0,
            kv_elements: 0,
            o_elements: 0,
            attention: None,
            matmuls: Vec::new(),
            norms: Vec::new(),
            activation_elements: 0,
            sized_ops: 0,
            unsized_ops: 0,
            ffn_width: 0,
            input_width: None,
            output_width: None,
        };
        let mut k_cols: Option<u64> = None;
        let mut q_rows: Option<u64> = None;
        // (tape op, query var) of each attention call in the layer.
        let mut attn_q: Vec<(u32, VarId)> = Vec::new();

        for (op, _) in list
            .ops
            .iter()
            .zip(op_layer.iter())
            .filter(|(_, l)| **l == Some(layer.index))
        {
            match &op.op {
                PrimalOp::Param(name) => {
                    // Buffers and config fields are not optimized: no
                    // gradient, no moments, and often no literal shape (a
                    // computed table). They are not model parameters here.
                    let is_model_parameter = match trained {
                        Some(set) => set.contains(name),
                        None => {
                            crate::stmt::is_trainable_param_leaf_name(name)
                                && result_ty(op) == WengertType::Tensor
                        }
                    };
                    if !is_model_parameter {
                        continue;
                    }
                    let Some(n) = sizes.get(&op.result).and_then(SizeInfo::numel) else {
                        reasons.push(format!("parameter `{name}` has no compile-time shape"));
                        continue;
                    };
                    g.param_elements += n;
                    let dims = sizes.get(&op.result).and_then(dims_u64);
                    match proj_role(name) {
                        Some(ProjRole::Q) => {
                            g.q_elements += n;
                            q_rows = dims.as_ref().and_then(|d| d.first().copied());
                        }
                        Some(ProjRole::K) => {
                            g.kv_elements += n;
                            k_cols = dims.as_ref().and_then(|d| d.last().copied());
                        }
                        Some(ProjRole::V) => g.kv_elements += n,
                        Some(ProjRole::O) => g.o_elements += n,
                        Some(ProjRole::Ffn) => {
                            // The inner width is the dimension an FFN weight
                            // does not share with the residual stream; the
                            // up/gate projections carry it as their output,
                            // the down projection as its input.
                            if let Some(d) = dims.as_ref().filter(|d| d.len() == 2) {
                                g.ffn_width = g.ffn_width.max(d[0].max(d[1]));
                            }
                        }
                        None => {}
                    }
                    continue;
                }
                PrimalOp::Matmul => {
                    // A product with a weight. Products of two activations
                    // (QKᵀ and AV inside attention) are not costed.
                    let weight = op
                        .inputs
                        .get(1)
                        .and_then(|v| weight_of(*v))
                        .or_else(|| op.inputs.first().and_then(|v| weight_of(*v)));
                    if let Some(weight) = weight {
                        let a = op.inputs.first().and_then(|v| sizes.get(v));
                        let b = op.inputs.get(1).and_then(|v| sizes.get(v));
                        let b_dims = b.and_then(dims_u64).filter(|d| d.len() >= 2);
                        let a_numel = a.and_then(SizeInfo::numel);
                        match (a_numel, b_dims) {
                            (Some(a_numel), Some(bd)) => {
                                let (k, n) = (bd[bd.len() - 2], bd[bd.len() - 1]);
                                if k == 0 || !a_numel.is_multiple_of(k) {
                                    reasons.push(format!(
                                        "matrix product at tape op {} in `{}` has operands of \
                                         inconsistent size",
                                        op.id, layer.name
                                    ));
                                } else {
                                    g.matmuls.push(MatmulGeometry {
                                        rows: a_numel / k,
                                        in_dim: k,
                                        out_dim: n,
                                        role: proj_role(weight),
                                    });
                                }
                            }
                            _ => reasons.push(format!(
                                "matrix product with `{weight}` at tape op {} in `{}` has an \
                                 operand of unknown shape",
                                op.id, layer.name
                            )),
                        }
                    }
                }
                o if is_attention(o) => match op.inputs.first() {
                    Some(q) => attn_q.push((op.id, *q)),
                    None => reasons.push(format!(
                        "attention at tape op {} in `{}` has no query operand",
                        op.id, layer.name
                    )),
                },
                // Decomposed attention: softmax over a product of two
                // activations, reached through the scale and mask
                // arithmetic. The query is that product's left operand.
                PrimalOp::Softmax { .. } => {
                    let mut v = op.inputs.first().copied();
                    for _ in 0..6 {
                        let Some(p) = v.and_then(|v| producer.get(&v)) else {
                            break;
                        };
                        match p.op {
                            PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div => {
                                v = p.inputs.first().copied();
                            }
                            PrimalOp::Matmul => {
                                if p.inputs.iter().all(|i| weight_of(*i).is_none())
                                    && let Some(q) = p.inputs.first()
                                {
                                    attn_q.push((op.id, *q));
                                }
                                break;
                            }
                            _ => break,
                        }
                    }
                }
                PrimalOp::RMSNorm { .. } | PrimalOp::LayerNorm { .. } => {
                    let inp = op.inputs.first().copied();
                    match inp.and_then(|v| sizes.get(&v)).and_then(SizeInfo::numel) {
                        Some(n) => g
                            .norms
                            .push((n, inp.and_then(last_dim).unwrap_or(1).max(1))),
                        None => reasons.push(format!(
                            "normalisation at tape op {} in `{}` has an input of unknown shape",
                            op.id, layer.name
                        )),
                    }
                }
                _ => {}
            }

            // Forward activations: tensor results that allocate.
            if result_ty(op) == WengertType::Tensor
                && !is_view_producing_op(&op.op)
                && !matches!(op.op, PrimalOp::Input(_) | PrimalOp::Constant(_))
            {
                match sizes.get(&op.result).and_then(SizeInfo::numel) {
                    Some(n) => {
                        g.activation_elements += n;
                        g.sized_ops += 1;
                    }
                    None => g.unsized_ops += 1,
                }
            }
        }

        // Attention geometry: the query is `[batch, heads, seq, head_dim]`,
        // or `[batch, seq, head_dim]` / `[seq, head_dim]` for one head.
        // Every attention call of the layer must agree.
        let mut queries: Vec<(u32, [u64; 4])> = Vec::new();
        for (op_id, q) in &attn_q {
            match sizes.get(q).and_then(dims_u64).as_deref() {
                Some(&[b, h, s, d]) => queries.push((*op_id, [b, h, s, d])),
                Some(&[b, s, d]) => queries.push((*op_id, [b, 1, s, d])),
                Some(&[s, d]) => queries.push((*op_id, [1, 1, s, d])),
                _ => reasons.push(format!(
                    "attention at tape op {op_id} in `{}` has a query of unknown shape",
                    layer.name
                )),
            }
        }
        if let Some((first_id, q)) = queries.first() {
            if queries.iter().any(|(_, d)| d != q) {
                reasons.push(format!(
                    "`{}` has attention calls of differing shapes (first at tape op {first_id})",
                    layer.name
                ));
            } else {
                let [batch, n_heads, seq, head_dim] = *q;
                let n_kv_heads = match k_cols {
                    Some(c) if head_dim > 0 && c.is_multiple_of(head_dim) && c > 0 => c / head_dim,
                    _ => n_heads,
                };
                if n_heads == 0 || head_dim == 0 || !n_heads.is_multiple_of(n_kv_heads.max(1)) {
                    reasons.push(format!(
                        "`{}` has an attention shape WGGO cannot plan ({n_heads} heads of \
                         {head_dim}, {n_kv_heads} key/value heads)",
                        layer.name
                    ));
                } else {
                    g.attention = Some(AttentionGeometry {
                        batch,
                        n_heads,
                        n_kv_heads,
                        seq,
                        head_dim,
                        d_model: q_rows.unwrap_or(n_heads * head_dim),
                    });
                }
            }
        }

        layers.push(g);
    }

    // Stream widths: the last dimension of every tensor that crosses a layer
    // boundary, seen from both sides. A step input counts as coming from
    // outside; parameters, literals and scalars are not streams. A tensor of
    // unknown shape makes the width unknown.
    let ty_of: HashMap<VarId, WengertType> = list
        .ops
        .iter()
        .map(|op| (op.result, result_ty(op)))
        .collect();
    let is_literal: HashSet<VarId> = list
        .ops
        .iter()
        .filter(|op| matches!(op.op, PrimalOp::Constant(_)))
        .map(|op| op.result)
        .collect();
    let mut in_widths: Vec<HashSet<Option<u64>>> = vec![HashSet::new(); layers.len()];
    let mut out_widths: Vec<HashSet<Option<u64>>> = vec![HashSet::new(); layers.len()];
    for (op, consumer) in list.ops.iter().zip(op_layer.iter()) {
        for v in &op.inputs {
            let producer = layer_of_var.get(v).copied();
            if producer == *consumer
                || param_name.contains_key(v)
                || is_literal.contains(v)
                || ty_of.get(v) != Some(&WengertType::Tensor)
                || matches!(sizes.get(v), Some(SizeInfo::Shape(d)) if d.is_empty())
            {
                continue;
            }
            if let Some(c) = consumer {
                in_widths[*c as usize].insert(last_dim(*v));
            }
            if let Some(p) = producer {
                out_widths[p as usize].insert(last_dim(*v));
            }
        }
    }
    for (i, g) in layers.iter_mut().enumerate() {
        g.input_width = unique_known(&in_widths[i]);
        g.output_width = unique_known(&out_widths[i]);
    }

    if reasons.is_empty() {
        Ok(ModelGeometry {
            layers,
            dtype_bytes,
        })
    } else {
        Err(ShapeUnknown { reasons })
    }
}

/// Which graph layer each tape op belongs to, by position in `list.ops`.
///
/// A parameter belongs to the layer [`crate::wggo_graph::build`] put it in.
/// Any other op belongs to the layer of a parameter it consumes directly
/// (`h @ self.w` is the work of the layer that owns `w`); failing that, to
/// the layer of its most recently computed operand, so the reshapes,
/// attention call and residual add that follow a block's projections stay
/// with that block. An op with no attributed operand (a step input, a
/// literal) has no layer.
fn attribute_ops(list: &WengertList, graph: &OptGraph) -> Vec<Option<u32>> {
    let mut out: Vec<Option<u32>> = vec![None; list.ops.len()];
    let mut param_pos: HashMap<VarId, usize> = HashMap::new();
    for layer in &graph.layers {
        for &i in &layer.op_indices {
            if let Some(op) = list.ops.get(i as usize)
                && matches!(op.op, PrimalOp::Param(_))
            {
                out[i as usize] = Some(layer.index);
                param_pos.insert(op.result, i as usize);
            }
        }
    }
    let pos_of: HashMap<VarId, usize> = list
        .ops
        .iter()
        .enumerate()
        .map(|(i, op)| (op.result, i))
        .collect();
    for (i, op) in list.ops.iter().enumerate() {
        if out[i].is_some() {
            continue;
        }
        let from_param = op
            .inputs
            .iter()
            .find_map(|v| param_pos.get(v).and_then(|&p| out[p]));
        out[i] = from_param.or_else(|| {
            op.inputs
                .iter()
                .filter_map(|v| pos_of.get(v).copied())
                .filter(|&p| p < i && out[p].is_some())
                .max()
                .and_then(|p| out[p])
        });
    }
    out
}

/// Bytes per element of the model's tensors: every tensor constructor
/// (`randn`, `ones`, `zeros`, `full`) creates f32, on the CPU and on a GPU.
pub const TENSOR_ELEMENT_BYTES: u64 = 4;

impl crate::compiler::Compiler<'_> {
    /// The geometry of the model a train block trains, from the forward
    /// tape `extractor` holds.
    ///
    /// The dimension seeds are the ones the transient arena uses
    /// (`stmt_train/transient_arena_projection.rs`), restricted to the
    /// forward tape: semantic types of the step's values, the model's field
    /// dims resolved through the owning model type, uniquely named field
    /// dims as a fallback, and the `[batch, seq]` shape of a proven
    /// `DataLoader`'s batch fields.
    ///
    /// `trained` is the train block's parameter list when the caller has it
    /// (see [`derive_for`]).
    pub(crate) fn wggo_model_geometry(
        &self,
        extractor: &crate::source_ad::WengertExtractor<'_>,
        trained: Option<&[String]>,
    ) -> Result<ModelGeometry, ShapeUnknown> {
        let trained: Option<HashSet<String>> = trained.map(|p| p.iter().cloned().collect());
        let list = extractor.wengert_list();
        let mut dim_seeds: HashMap<VarId, Vec<i64>> =
            crate::profiling::captures::dim_hints_from_var_nodes(
                extractor.var_nodes(),
                self.type_map,
            );
        for (vid, d) in extractor.known_param_dims() {
            dim_seeds.entry(*vid).or_insert_with(|| d.clone());
        }
        let field_dims = self.models.unique_field_dims();
        for (name, vid) in extractor.named_param_var_ids() {
            if dim_seeds.contains_key(vid) {
                continue;
            }
            let leaf = name.rsplit('.').next().unwrap_or(name);
            if let Some(d) = field_dims.get(leaf) {
                dim_seeds.insert(*vid, d.clone());
            }
        }
        if let Some(facts) = self.lm_head_loader_scan.facts() {
            let input_leaves: HashSet<VarId> = list
                .ops
                .iter()
                .filter(|o| matches!(o.op, PrimalOp::Input(_)))
                .map(|o| o.result)
                .collect();
            for op in &list.ops {
                let PrimalOp::Passthrough(n) = &op.op else {
                    continue;
                };
                let Some(field) = n.strip_prefix("dict_get:") else {
                    continue;
                };
                if matches!(
                    field,
                    "input_ids" | "labels" | "segment_ids" | "position_ids"
                ) && op.inputs.len() == 1
                    && input_leaves.contains(&op.inputs[0])
                {
                    dim_seeds
                        .entry(op.result)
                        .or_insert_with(|| vec![facts.batch_size as i64, facts.seq_len as i64]);
                }
            }
        }
        let scalar_seeds = extractor.known_param_scalar_values();
        let no_adjoint = WengertList {
            ops: Vec::new(),
            output: list.output,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let sizes = crate::transient_arena::propagate_size_info(
            list,
            &no_adjoint,
            &|v| dim_seeds.get(&v).cloned(),
            &|v| scalar_seeds.get(&v).copied(),
            &[],
        );
        let graph = crate::wggo_graph::build(list);
        derive_for(list, &graph, &sizes, TENSOR_ELEMENT_BYTES, trained.as_ref())
    }
}

fn unique_known(widths: &HashSet<Option<u64>>) -> Option<u64> {
    if widths.len() == 1 {
        widths.iter().next().copied().flatten()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Hand-built transformer tapes with every shape written out, for tests of
/// the planner that need a model whose sizes are known on paper.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use crate::wengert::WengertOp;

    /// The shape of a fixture model.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct Dims {
        pub blocks: u32,
        pub d_model: i64,
        pub n_heads: i64,
        pub n_kv_heads: i64,
        pub head_dim: i64,
        pub ffn: i64,
        pub batch: i64,
        pub seq: i64,
        pub vocab: i64,
    }

    impl Dims {
        /// 12 heads of 64 over a 768-wide stream, 4 key/value heads, an FFN
        /// of 3072, batch 2, 16 tokens.
        pub(crate) fn gqa12(blocks: u32) -> Self {
            Self {
                blocks,
                d_model: 768,
                n_heads: 12,
                n_kv_heads: 4,
                head_dim: 64,
                ffn: 3072,
                batch: 2,
                seq: 16,
                vocab: 100,
            }
        }

        /// 8 heads of 64 over a 512-wide stream, 4 key/value heads, an FFN
        /// of 2048, batch 1, 1024 tokens.
        pub(crate) fn base512(blocks: u32) -> Self {
            Self {
                blocks,
                d_model: 512,
                n_heads: 8,
                n_kv_heads: 4,
                head_dim: 64,
                ffn: 2048,
                batch: 1,
                seq: 1024,
                vocab: 100,
            }
        }

        /// Parameter elements of one block: Q and O projections, K and V
        /// projections, the two FFN matrices and the norm gain.
        pub(crate) fn block_param_elements(&self) -> u64 {
            let d = self.d_model as u64;
            let q = d * (self.n_heads * self.head_dim) as u64;
            let kv = d * (self.n_kv_heads * self.head_dim) as u64;
            2 * q + 2 * kv + 2 * d * self.ffn as u64 + d
        }
    }

    struct Builder {
        ops: Vec<WengertOp>,
        sizes: HashMap<VarId, SizeInfo>,
    }

    impl Builder {
        fn push(&mut self, op: PrimalOp, inputs: Vec<VarId>, shape: &[i64]) -> VarId {
            let id = self.ops.len() as u32;
            self.ops.push(WengertOp {
                id,
                result: id,
                op,
                inputs,
                saved_for_backward: false,
                checkpointed: false,
            });
            self.sizes.insert(id, SizeInfo::Shape(shape.to_vec()));
            id
        }
        fn param(&mut self, name: String, shape: &[i64]) -> VarId {
            self.push(PrimalOp::Param(name), vec![], shape)
        }
    }

    /// `dims.blocks` pre-norm attention + FFN blocks on a residual stream,
    /// named `m.blocks.N.…`, then an output projection `m.head`. `packed`
    /// selects the packed attention op.
    pub(crate) fn transformer(dims: Dims, packed: bool) -> (WengertList, HashMap<VarId, SizeInfo>) {
        let Dims {
            d_model: d,
            n_heads: h,
            n_kv_heads: kv,
            head_dim: hd,
            ffn,
            batch: b,
            seq: s,
            vocab,
            ..
        } = dims;
        let mut t = Builder {
            ops: Vec::new(),
            sizes: HashMap::new(),
        };
        let stream = [b, s, d];
        let mut x = t.push(PrimalOp::Input("x".into()), vec![], &stream);
        let seg = t.push(PrimalOp::Input("segment_ids".into()), vec![], &[b, s]);
        let scale = t.push(PrimalOp::Constant(0.125), vec![], &[]);
        for i in 0..dims.blocks {
            let n = |f: &str| format!("m.blocks.{i}.{f}");
            let norm = t.param(n("norm_a"), &[d]);
            let wq = t.param(n("attn.wq"), &[d, h * hd]);
            let wk = t.param(n("attn.wk"), &[d, kv * hd]);
            let wv = t.param(n("attn.wv"), &[d, kv * hd]);
            let wo = t.param(n("attn.wo"), &[h * hd, d]);
            let w_up = t.param(n("w_up"), &[d, ffn]);
            let w_down = t.param(n("w_down"), &[ffn, d]);
            let eps = crate::wengert::NormEps::Const(1e-5);
            let xn = t.push(PrimalOp::RMSNorm { eps }, vec![x, norm], &stream);
            let q = t.push(PrimalOp::Matmul, vec![xn, wq], &[b, s, h * hd]);
            let k = t.push(PrimalOp::Matmul, vec![xn, wk], &[b, s, kv * hd]);
            let v = t.push(PrimalOp::Matmul, vec![xn, wv], &[b, s, kv * hd]);
            let heads = [b, h, s, hd];
            let reshape = || PrimalOp::Passthrough("reshape".into());
            let q4 = t.push(reshape(), vec![q], &heads);
            let k4 = t.push(reshape(), vec![k], &heads);
            let v4 = t.push(reshape(), vec![v], &heads);
            let attn = if packed {
                t.push(
                    PrimalOp::ScaledDotProductAttentionPacked,
                    vec![q4, k4, v4, scale, seg],
                    &heads,
                )
            } else {
                t.push(
                    PrimalOp::ScaledDotProductAttention { causal: true },
                    vec![q4, k4, v4],
                    &heads,
                )
            };
            let flat = t.push(reshape(), vec![attn], &[b, s, h * hd]);
            let o = t.push(PrimalOp::Matmul, vec![flat, wo], &stream);
            let h1 = t.push(PrimalOp::Add, vec![x, o], &stream);
            let up = t.push(PrimalOp::Matmul, vec![h1, w_up], &[b, s, ffn]);
            let down = t.push(PrimalOp::Matmul, vec![up, w_down], &stream);
            x = t.push(PrimalOp::Add, vec![h1, down], &stream);
        }
        let head = t.param("m.head".into(), &[d, vocab]);
        let logits = t.push(PrimalOp::Matmul, vec![x, head], &[b, s, vocab]);
        let list = WengertList {
            ops: t.ops,
            output: logits,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        (list, t.sizes)
    }

    /// The geometry of [`transformer`] at f32.
    pub(crate) fn geometry(list: &WengertList, sizes: &HashMap<VarId, SizeInfo>) -> ModelGeometry {
        derive(
            list,
            &crate::wggo_graph::build(list),
            sizes,
            TENSOR_ELEMENT_BYTES,
        )
        .expect("every shape of the fixture is supplied")
    }

    /// The tape var of the parameter called `name`.
    pub(crate) fn param_var(list: &WengertList, name: &str) -> VarId {
        list.ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Param(n) if n == name))
            .map(|op| op.result)
            .unwrap_or_else(|| panic!("no parameter `{name}` in the fixture"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::fixtures::{Dims, geometry, param_var, transformer};
    use super::*;
    use crate::wggo_graph::build as build_graph;

    /// One grouped-query block reads back the shape it was written with.
    /// Expected figures are worked out by hand from the fixture's dims,
    /// independently of the code under test.
    #[test]
    fn a_grouped_query_block_reads_back_its_own_shape() {
        let (list, sizes) = transformer(Dims::gqa12(1), false);
        let geo = geometry(&list, &sizes);
        let block = geo.layers.iter().find(|l| l.name == "blocks.0").unwrap();

        // wq 768x768 + wk 768x256 + wv 768x256 + wo 768x768 + w_up 768x3072
        // + w_down 3072x768 + norm 768
        //   = 589824 + 196608 + 196608 + 589824 + 2359296 + 2359296 + 768.
        assert_eq!(block.param_elements, 6_292_224);
        assert_eq!(
            block.attention,
            Some(AttentionGeometry {
                batch: 2,
                n_heads: 12,
                n_kv_heads: 4,
                seq: 16,
                head_dim: 64,
                d_model: 768,
            })
        );
        assert_eq!(block.ffn_width, 3072);
        // Q, K, V, O, up, down: six products, each over 2 x 16 = 32 rows.
        assert_eq!(block.matmuls.len(), 6);
        assert!(block.matmuls.iter().all(|m| m.rows == 32));
        assert_eq!(block.norms, vec![(2 * 16 * 768, 768)]);
        // Allocating results: norm, Q, O, both residual adds and the FFN
        // output at 2x16x768 = 24576 (six of them), K and V at 2x16x256 =
        // 8192, attention at 2x12x16x64 = 24576, FFN inner at 2x16x3072 =
        // 98304. Reshapes are views.
        assert_eq!(block.activation_elements, 7 * 24_576 + 2 * 8_192 + 98_304);
        assert_eq!(block.unsized_ops, 0);
        // The block reads a 768-wide stream and hands one on.
        assert_eq!(block.input_width, Some(768));
        assert_eq!(block.output_width, Some(768));

        // Dropping to 9 of 12 heads keeps 3 of 4 key/value heads: Q and O
        // lose a quarter (2 x 147456), K and V lose a quarter (2 x 49152).
        assert_eq!(block.param_elements_at(9), 6_292_224 - 294_912 - 98_304);
        assert_eq!(block.param_elements_at(12), 6_292_224);

        // The output projection is its own layer, with no attention.
        let other = geo.layers.iter().find(|l| l.name == "other").unwrap();
        assert_eq!(other.param_elements, 768 * 100);
        assert_eq!(other.attention, None);
        assert_eq!(geo.total_param_elements(), 6_292_224 + 76_800);
    }

    #[test]
    fn every_block_of_a_deeper_model_is_sized_alike() {
        let dims = Dims::base512(6);
        let (list, sizes) = transformer(dims, false);
        let geo = geometry(&list, &sizes);
        let blocks: Vec<_> = geo
            .layers
            .iter()
            .filter(|l| l.name.starts_with("blocks."))
            .collect();
        assert_eq!(blocks.len(), 6);
        for b in &blocks {
            // 2x512x512 + 2x512x256 + 2x512x2048 + 512 = 2,884,096.
            assert_eq!(b.param_elements, 2_884_096);
            assert_eq!(b.param_elements, dims.block_param_elements());
            assert_eq!(b.n_heads(), 8);
        }
        assert_eq!(
            geo.uniform_attention().map(|a| (a.n_heads, a.n_kv_heads)),
            Some((8, 4))
        );
    }

    #[test]
    fn a_parameter_without_a_shape_is_reported_not_guessed() {
        let (list, mut sizes) = transformer(Dims::gqa12(1), false);
        sizes.remove(&param_var(&list, "m.blocks.0.w_up"));
        let graph = build_graph(&list);
        let err = derive(&list, &graph, &sizes, 4).unwrap_err();
        assert!(
            err.reasons.iter().any(|r| r.contains("m.blocks.0.w_up")),
            "{err}"
        );
    }

    #[test]
    fn an_attention_call_of_unknown_shape_is_reported() {
        let (list, mut sizes) = transformer(Dims::gqa12(1), false);
        let attn = list.ops.iter().find(|op| is_attention(&op.op)).unwrap();
        sizes.insert(attn.inputs[0], SizeInfo::Numel(2 * 12 * 16 * 64));
        let graph = build_graph(&list);
        let err = derive(&list, &graph, &sizes, 4).unwrap_err();
        assert!(err.reasons.iter().any(|r| r.contains("attention")), "{err}");
    }

    /// Buffers and config fields sit on the tape as parameters but are not
    /// trained; an unknown shape for one must not stop the plan, and a known
    /// one is not counted as a parameter.
    #[test]
    fn untrained_buffers_are_not_model_parameters() {
        let (mut list, mut sizes) = transformer(Dims::gqa12(1), false);
        let table = list.ops.len() as u32;
        list.ops.push(crate::wengert::WengertOp {
            id: table,
            result: table,
            op: PrimalOp::Param("m.blocks.0.attn.rope._cos_table".into()),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
        let geo = derive(&list, &build_graph(&list), &sizes, 4).expect("an unsized buffer is fine");
        let block = geo.layers.iter().find(|l| l.name == "blocks.0").unwrap();
        assert_eq!(block.param_elements, 6_292_224);
        sizes.insert(table, SizeInfo::Shape(vec![16, 32]));
        let geo = derive(&list, &build_graph(&list), &sizes, 4).unwrap();
        let block = geo.layers.iter().find(|l| l.name == "blocks.0").unwrap();
        assert_eq!(block.param_elements, 6_292_224);
    }

    /// A `float` field read by the step (a norm's `eps`) is a `Param` on the
    /// tape with no shape. It is not one of the model's trained tensors, so
    /// with the train block's parameter list it is ignored; without the
    /// list it is ignored when the tape types it as a scalar, and reported
    /// when nothing says what it is.
    #[test]
    fn a_scalar_field_is_not_a_parameter() {
        let (mut list, sizes) = transformer(Dims::gqa12(1), false);
        let trained: HashSet<String> = list
            .ops
            .iter()
            .filter_map(|op| match &op.op {
                PrimalOp::Param(n) => Some(n.clone()),
                _ => None,
            })
            .collect();
        let eps = list.ops.len() as u32;
        list.ops.push(crate::wengert::WengertOp {
            id: eps,
            result: eps,
            op: PrimalOp::Param("m.blocks.0.norm.eps".into()),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
        let graph = build_graph(&list);
        let geo = derive_for(&list, &graph, &sizes, 4, Some(&trained)).expect("eps is not trained");
        let block = geo.layers.iter().find(|l| l.name == "blocks.0").unwrap();
        assert_eq!(block.param_elements, 6_292_224);

        let err = derive(&list, &graph, &sizes, 4).unwrap_err();
        assert!(
            err.reasons
                .iter()
                .any(|r| r.contains("m.blocks.0.norm.eps")),
            "{err}"
        );
        list.var_types.insert(eps, WengertType::Scalar);
        assert!(derive(&list, &graph, &sizes, 4).is_ok());
    }

    /// The decomposed (masked) form of attention has no attention op: it is
    /// a softmax over `q @ kᵀ` scaled and masked. Its geometry is read from
    /// the query all the same, and the product of two activations needs no
    /// shape of its own.
    #[test]
    fn decomposed_attention_is_recognised_by_its_softmax() {
        let (mut list, mut sizes) = transformer(Dims::gqa12(1), false);
        let attn_pos = list.ops.iter().position(|op| is_attention(&op.op)).unwrap();
        let (q, k) = (list.ops[attn_pos].inputs[0], list.ops[attn_pos].inputs[1]);
        let attn_var = list.ops[attn_pos].result;
        // scores = q @ k (shape unknown), softmax(scores) in place of the
        // attention op's result.
        let scores = list.ops.len() as u32;
        list.ops.push(crate::wengert::WengertOp {
            id: scores,
            result: scores,
            op: PrimalOp::Matmul,
            inputs: vec![q, k],
            saved_for_backward: false,
            checkpointed: false,
        });
        list.ops[attn_pos].op = PrimalOp::Softmax { dim: -1 };
        list.ops[attn_pos].inputs = vec![scores];
        // Keep the tape topological: move the product before the softmax.
        let product = list.ops.pop().unwrap();
        list.ops.insert(attn_pos, product);
        sizes.remove(&scores);
        let _ = attn_var;
        let geo = derive(&list, &build_graph(&list), &sizes, 4).expect("decomposed attention");
        let block = geo.layers.iter().find(|l| l.name == "blocks.0").unwrap();
        assert_eq!(
            block
                .attention
                .map(|a| (a.n_heads, a.n_kv_heads, a.head_dim)),
            Some((12, 4, 64))
        );
        assert_eq!(
            block.matmuls.len(),
            6,
            "the activation product is not a weight product"
        );
        assert_eq!(
            block.unsized_ops, 1,
            "and its unknown size is counted, not guessed"
        );
    }

    /// A single-head attention over `[seq, head_dim]` operands is one head.
    #[test]
    fn a_rank_2_query_is_one_head() {
        let (list, mut sizes) = transformer(Dims::gqa12(1), false);
        let attn = list.ops.iter().find(|op| is_attention(&op.op)).unwrap();
        sizes.insert(attn.inputs[0], SizeInfo::Shape(vec![16, 256]));
        let geo = derive(&list, &build_graph(&list), &sizes, 4).unwrap();
        let block = geo.layers.iter().find(|l| l.name == "blocks.0").unwrap();
        // K projection is 256 wide: one key/value head of 256.
        assert_eq!(
            block
                .attention
                .map(|a| (a.batch, a.n_heads, a.n_kv_heads, a.seq, a.head_dim)),
            Some((1, 1, 1, 16, 256))
        );
    }

    #[test]
    fn packed_attention_is_found_on_the_tape_that_has_it() {
        assert!(tape_has_packed_attention(
            &transformer(Dims::gqa12(1), true).0
        ));
        let (mut unpacked, _) = transformer(Dims::gqa12(1), false);
        assert!(!tape_has_packed_attention(&unpacked));
        // The masked form is decomposed; the step's read of the batch's
        // mask is what shows it.
        let id = unpacked.ops.len() as u32;
        unpacked.ops.push(crate::wengert::WengertOp {
            id,
            result: id,
            op: PrimalOp::Passthrough("dict_get:attention_mask".into()),
            inputs: vec![0],
            saved_for_backward: false,
            checkpointed: false,
        });
        assert!(tape_has_packed_attention(&unpacked));
    }
}
