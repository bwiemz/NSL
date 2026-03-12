// M18b: TraceGraph -> ONNX protobuf conversion
//
// Converts a recorded TraceGraph (see trace.rs) into a serialised ONNX model
// that can be consumed by any ONNX-compatible runtime (ONNX Runtime, PyTorch,
// etc.).

use std::collections::HashMap;
use std::io::Write as _;

use prost::Message as _;

use crate::onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeDimProto, TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
    ATTRIBUTE_FLOAT, ATTRIBUTE_INT, ATTRIBUTE_INTS, DOUBLE, FLOAT,
};
use crate::tensor::NslTensor;
use crate::trace::{AttrValue, OpType, TraceGraph};

// ─────────────────────────────────────────────────────────────────────────────
// OpType → ONNX op-name mapping
// ─────────────────────────────────────────────────────────────────────────────

fn op_type_to_onnx(op: OpType) -> &'static str {
    match op {
        OpType::Add => "Add",
        OpType::Sub => "Sub",
        OpType::Mul => "Mul",
        OpType::Div => "Div",
        OpType::MatMul => "MatMul",
        OpType::Transpose => "Transpose",
        OpType::Relu => "Relu",
        OpType::Sigmoid => "Sigmoid",
        OpType::Tanh => "Tanh",
        OpType::Softmax => "Softmax",
        OpType::LayerNorm => "LayerNormalization",
        OpType::Reshape => "Reshape",
        OpType::Unsqueeze => "Unsqueeze",
        OpType::Expand => "Expand",
        OpType::Concat => "Concat",
        OpType::Gather => "Gather",
        OpType::Neg => "Neg",
        OpType::Exp => "Exp",
        OpType::Log => "Log",
        OpType::Sqrt => "Sqrt",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a ValueInfoProto for a named tensor with a known shape/dtype
// ─────────────────────────────────────────────────────────────────────────────

fn make_value_info(name: &str, elem_type: i32, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&d| TensorShapeDimProto { dim_value: d })
                        .collect(),
                }),
            }),
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: read raw bytes from a live NslTensor (CPU only)
// ─────────────────────────────────────────────────────────────────────────────

/// Read all tensor data as little-endian bytes.
/// Returns (raw_bytes, onnx_data_type).
unsafe fn tensor_raw_bytes(t: &NslTensor) -> (Vec<u8>, i32) {
    let len = t.len as usize;
    match t.dtype {
        0 => {
            // f64 CPU
            let ptr = t.data as *const f64;
            let slice = std::slice::from_raw_parts(ptr, len);
            let bytes: Vec<u8> = slice
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            (bytes, DOUBLE)
        }
        1 => {
            // f32 GPU or CPU
            let ptr = t.data as *const f32;
            let slice = std::slice::from_raw_parts(ptr, len);
            let bytes: Vec<u8> = slice
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            (bytes, FLOAT)
        }
        _ => {
            // Unknown dtype — emit empty raw_data with FLOAT placeholder
            (Vec::new(), FLOAT)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core builder
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a `TraceGraph` into an ONNX `ModelProto`.
///
/// For each `TraceOp` the inputs are resolved as follows:
/// 1. Registered graph input → use the registered name.
/// 2. A prior op's output (present in `ptr_to_node`) → `"node_N_out"`.
/// 3. Neither → treat as a static weight / initializer.  The tensor is read
///    from memory and stored as a `TensorProto` initializer.
pub fn build_onnx_model(graph: &TraceGraph) -> ModelProto {
    // Map: ptr → resolved name (populated incrementally)
    let mut name_map: HashMap<i64, String> = HashMap::new();

    // Initializers (weights that aren't computed ops or named inputs)
    let mut initializers: Vec<TensorProto> = Vec::new();
    let mut init_counter: usize = 0;

    // Register the named graph inputs up-front
    let mut graph_inputs: Vec<ValueInfoProto> = Vec::new();
    for (ptr, name) in &graph.inputs {
        name_map.insert(*ptr, name.clone());
        // We don't have the shape/dtype here without reading the tensor, so we
        // use unknown shape with FLOAT as a safe default.  If the tensor is
        // live we could read it, but inputs are generally provided at runtime
        // so we emit a shape-free ValueInfoProto for the graph input list.
        graph_inputs.push(ValueInfoProto {
            name: name.clone(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: FLOAT,
                    shape: Some(TensorShapeProto { dim: vec![] }),
                }),
            }),
        });
    }

    // Build nodes
    let mut nodes: Vec<NodeProto> = Vec::new();

    for (idx, op) in graph.ops.iter().enumerate() {
        let output_name = format!("node_{}_out", idx);

        // Resolve each input pointer
        let input_names: Vec<String> = op
            .input_ptrs
            .iter()
            .map(|&ptr| {
                if let Some(n) = name_map.get(&ptr) {
                    // Already known: registered input or prior node output
                    return n.clone();
                }
                // Neither a registered input nor a computed node — treat as initializer
                let init_name = format!("initializer_{}", init_counter);
                init_counter += 1;

                // Try to read the tensor from memory
                let (raw_bytes, data_type, dims) = if ptr != 0 {
                    let t = NslTensor::from_ptr(ptr);
                    let shape: Vec<i64> = if t.ndim > 0 && !t.shape.is_null() {
                        unsafe {
                            std::slice::from_raw_parts(t.shape, t.ndim as usize).to_vec()
                        }
                    } else {
                        vec![]
                    };
                    let (bytes, dt) = unsafe { tensor_raw_bytes(t) };
                    (bytes, dt, shape)
                } else {
                    (Vec::new(), FLOAT, Vec::new())
                };

                initializers.push(TensorProto {
                    dims,
                    data_type,
                    name: init_name.clone(),
                    raw_data: raw_bytes,
                });

                name_map.insert(ptr, init_name.clone());
                init_name
            })
            .collect();

        // Convert attributes
        let attributes: Vec<AttributeProto> = op
            .attributes
            .iter()
            .map(|(name, val)| match val {
                AttrValue::Float(f) => AttributeProto {
                    name: name.clone(),
                    r#type: ATTRIBUTE_FLOAT,
                    f: *f as f32,
                    i: 0,
                    ints: vec![],
                },
                AttrValue::Int(i) => AttributeProto {
                    name: name.clone(),
                    r#type: ATTRIBUTE_INT,
                    f: 0.0,
                    i: *i,
                    ints: vec![],
                },
                AttrValue::Ints(v) => AttributeProto {
                    name: name.clone(),
                    r#type: ATTRIBUTE_INTS,
                    f: 0.0,
                    i: 0,
                    ints: v.clone(),
                },
            })
            .collect();

        nodes.push(NodeProto {
            input: input_names,
            output: vec![output_name.clone()],
            name: format!("node_{}", idx),
            op_type: op_type_to_onnx(op.op_type).to_string(),
            domain: String::new(),
            attribute: attributes,
        });

        // Register this op's output so subsequent ops can reference it
        name_map.insert(op.output_ptr, output_name);
    }

    // Remap output names: rename the producing node's output to the registered
    // graph-output name so ONNX runtimes can match them.  Without this the
    // nodes produce "node_N_out" but the graph output list references "output_0"
    // (or whatever name was registered), which is an invalid graph.
    for (ptr, name) in &graph.outputs {
        if let Some(&node_id) = graph.ptr_to_node.get(ptr) {
            if node_id < nodes.len() {
                nodes[node_id].output = vec![name.clone()];
            }
        }
    }

    // Build graph outputs
    let graph_outputs: Vec<ValueInfoProto> = graph
        .outputs
        .iter()
        .map(|(ptr, name)| {
            // Find the op that produced this pointer to get shape/dtype
            if let Some(&node_idx) = graph.ptr_to_node.get(ptr) {
                let op = &graph.ops[node_idx];
                let elem_type = match op.output_dtype {
                    0 => DOUBLE,
                    1 => FLOAT,
                    _ => FLOAT,
                };
                make_value_info(name, elem_type, &op.output_shape)
            } else {
                // Fallback: no shape info
                ValueInfoProto {
                    name: name.clone(),
                    r#type: Some(TypeProto {
                        tensor_type: Some(TensorTypeProto {
                            elem_type: FLOAT,
                            shape: Some(TensorShapeProto { dim: vec![] }),
                        }),
                    }),
                }
            }
        })
        .collect();

    ModelProto {
        ir_version: 8,
        producer_name: "nsl".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(GraphProto {
            name: "nsl_graph".to_string(),
            node: nodes,
            initializer: initializers,
            input: graph_inputs,
            output: graph_outputs,
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FFI
// ─────────────────────────────────────────────────────────────────────────────

/// Export a traced graph to an ONNX file.
///
/// # Arguments
/// * `trace_ptr`  — raw pointer (i64) returned by `nsl_trace_stop()`
/// * `path_ptr`   — pointer to UTF-8 bytes of the output path
/// * `path_len`   — byte length of the path (not NUL-terminated)
#[no_mangle]
pub extern "C" fn nsl_onnx_export(trace_ptr: i64, path_ptr: i64, path_len: i64) {
    if trace_ptr == 0 || path_ptr == 0 || path_len <= 0 {
        eprintln!("[NSL] nsl_onnx_export: invalid arguments");
        return;
    }

    let graph = unsafe { &*(trace_ptr as *const TraceGraph) };

    let path = unsafe {
        let bytes = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_owned(),
            Err(e) => {
                eprintln!("[NSL] nsl_onnx_export: invalid UTF-8 path: {}", e);
                return;
            }
        }
    };

    let model = build_onnx_model(graph);

    let mut buf = Vec::new();
    if let Err(e) = model.encode(&mut buf) {
        eprintln!("[NSL] nsl_onnx_export: encode error: {}", e);
        return;
    }

    match std::fs::File::create(&path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(&buf) {
                eprintln!("[NSL] nsl_onnx_export: write error: {}", e);
            }
        }
        Err(e) => {
            eprintln!("[NSL] nsl_onnx_export: create file '{}' error: {}", path, e);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{AttrValue, OpType, TraceGraph, TraceOp};
    use std::collections::HashMap;

    /// Build a TraceGraph manually: input → Add(input, weight) → Relu → output.
    fn make_test_graph() -> TraceGraph {
        // Use ptr=0 for the weight so build_onnx_model skips reading tensor memory
        // (real programs pass live NslTensor pointers; tests use 0 as sentinel).
        let input_ptr: i64 = 0x1000;
        let weight_ptr: i64 = 0; // 0 → skip tensor read, emit empty initializer
        let add_out_ptr: i64 = 0x3000;
        let relu_out_ptr: i64 = 0x4000;

        let mut ptr_to_node: HashMap<i64, usize> = HashMap::new();
        ptr_to_node.insert(add_out_ptr, 0);
        ptr_to_node.insert(relu_out_ptr, 1);

        TraceGraph {
            ops: vec![
                TraceOp {
                    op_type: OpType::Add,
                    input_ptrs: vec![input_ptr, weight_ptr],
                    output_ptr: add_out_ptr,
                    output_shape: vec![2, 4],
                    output_dtype: 0,
                    attributes: vec![],
                },
                TraceOp {
                    op_type: OpType::Relu,
                    input_ptrs: vec![add_out_ptr],
                    output_ptr: relu_out_ptr,
                    output_shape: vec![2, 4],
                    output_dtype: 0,
                    attributes: vec![],
                },
            ],
            inputs: vec![(input_ptr, "input".to_string())],
            outputs: vec![(relu_out_ptr, "output".to_string())],
            ptr_to_node,
        }
    }

    #[test]
    fn test_build_onnx_model_structure() {
        let graph = make_test_graph();
        let model = build_onnx_model(&graph);

        // Opset
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.producer_name, "nsl");
        assert_eq!(model.opset_import.len(), 1);
        assert_eq!(model.opset_import[0].version, 17);

        let g = model.graph.as_ref().expect("graph should be present");

        // 2 nodes: Add and Relu
        assert_eq!(g.node.len(), 2, "expected 2 nodes");
        assert_eq!(g.node[0].op_type, "Add");
        assert_eq!(g.node[1].op_type, "Relu");

        // 1 named input
        assert_eq!(g.input.len(), 1);
        assert_eq!(g.input[0].name, "input");

        // 1 named output
        assert_eq!(g.output.len(), 1);
        assert_eq!(g.output[0].name, "output");

        // The weight (0x2000) should appear as an initializer
        assert_eq!(g.initializer.len(), 1);
        // Its name should match the Add node's second input
        let add_node = &g.node[0];
        assert_eq!(add_node.input.len(), 2);
        assert_eq!(add_node.input[0], "input");
        assert_eq!(add_node.input[1], g.initializer[0].name);

        // The Relu node takes the Add output
        assert_eq!(g.node[1].input[0], g.node[0].output[0]);
    }

    #[test]
    fn test_build_onnx_model_encodes() {
        let graph = make_test_graph();
        let model = build_onnx_model(&graph);

        let mut buf = Vec::new();
        model.encode(&mut buf).expect("encode should succeed");
        assert!(!buf.is_empty());

        // Decode and spot-check
        let decoded = ModelProto::decode(buf.as_slice()).expect("decode should succeed");
        let g = decoded.graph.unwrap();
        assert_eq!(g.node.len(), 2);
        assert_eq!(g.node[0].op_type, "Add");
        assert_eq!(g.node[1].op_type, "Relu");
    }

    #[test]
    fn test_attributes_preserved() {
        let input_ptr: i64 = 0x1000;
        let out_ptr: i64 = 0x2000;

        let mut ptr_to_node = HashMap::new();
        ptr_to_node.insert(out_ptr, 0_usize);

        let graph = TraceGraph {
            ops: vec![TraceOp {
                op_type: OpType::Transpose,
                input_ptrs: vec![input_ptr],
                output_ptr: out_ptr,
                output_shape: vec![4, 2],
                output_dtype: 0,
                attributes: vec![
                    ("perm".to_string(), AttrValue::Ints(vec![0, 2, 1])),
                    ("axis".to_string(), AttrValue::Int(-1)),
                    ("alpha".to_string(), AttrValue::Float(0.5)),
                ],
            }],
            inputs: vec![(input_ptr, "x".to_string())],
            outputs: vec![(out_ptr, "y".to_string())],
            ptr_to_node,
        };

        let model = build_onnx_model(&graph);
        let g = model.graph.unwrap();
        assert_eq!(g.node.len(), 1);
        let node = &g.node[0];
        assert_eq!(node.attribute.len(), 3);

        let perm_attr = node.attribute.iter().find(|a| a.name == "perm").unwrap();
        assert_eq!(perm_attr.r#type, ATTRIBUTE_INTS);
        assert_eq!(perm_attr.ints, vec![0, 2, 1]);

        let axis_attr = node.attribute.iter().find(|a| a.name == "axis").unwrap();
        assert_eq!(axis_attr.r#type, ATTRIBUTE_INT);
        assert_eq!(axis_attr.i, -1);

        let alpha_attr = node.attribute.iter().find(|a| a.name == "alpha").unwrap();
        assert_eq!(alpha_attr.r#type, ATTRIBUTE_FLOAT);
        assert!((alpha_attr.f - 0.5_f32).abs() < 1e-6);
    }

    #[test]
    fn test_nsl_onnx_export_writes_file() {
        use tempfile::NamedTempFile;

        let graph = make_test_graph();
        let model = build_onnx_model(&graph);

        // Encode manually to verify what the FFI should produce
        let mut expected = Vec::new();
        model.encode(&mut expected).unwrap();

        let tmp = NamedTempFile::new().expect("tempfile");
        let path = tmp.path().to_str().unwrap().to_owned();
        let path_bytes = path.as_bytes();

        // Box the graph so we can get a stable pointer
        let boxed = Box::new(make_test_graph());
        let trace_ptr = Box::into_raw(boxed) as i64;

        nsl_onnx_export(
            trace_ptr,
            path_bytes.as_ptr() as i64,
            path_bytes.len() as i64,
        );

        // Reclaim the box to avoid leak
        let _ = unsafe { Box::from_raw(trace_ptr as *mut TraceGraph) };

        let written = std::fs::read(&path).expect("output file should exist");
        assert!(!written.is_empty(), "exported file should be non-empty");

        // Must be valid ONNX protobuf
        let decoded = ModelProto::decode(written.as_slice()).expect("must decode");
        let g = decoded.graph.unwrap();
        assert_eq!(g.node.len(), 2);
    }
}
