// M18b: Pre-generated ONNX protobuf structs
//
// Hand-crafted minimal ONNX protobuf types that map to the official ONNX spec.
// Uses `prost::Message` derive for serialization — no .proto compilation needed.

use prost::Message;

// ─────────────────────────────────────────────────────────────────────────────
// Top-level model
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, repeated, tag = "2")]
    pub node: Vec<NodeProto>,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Nodes and attributes
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(string, tag = "7")]
    pub domain: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    /// AttributeType enum: FLOAT=1, INT=2, INTS=7
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    #[prost(float, tag = "4")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensors and type info
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "13")]
    pub raw_data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TensorTypeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<TensorShapeDimProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeDimProto {
    #[prost(int64, tag = "1")]
    pub dim_value: i64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX data-type constants (TensorProto.DataType enum)
// ─────────────────────────────────────────────────────────────────────────────

pub const FLOAT: i32 = 1;
pub const INT64: i32 = 7;
pub const DOUBLE: i32 = 11;
pub const FLOAT16: i32 = 10;

// ─────────────────────────────────────────────────────────────────────────────
// AttributeProto.AttributeType constants
// ─────────────────────────────────────────────────────────────────────────────

pub const ATTRIBUTE_FLOAT: i32 = 1;
pub const ATTRIBUTE_INT: i32 = 2;
pub const ATTRIBUTE_INTS: i32 = 7;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    /// Encode a simple ModelProto with one Relu node, decode it back, verify fields.
    #[test]
    fn test_roundtrip_relu_model() {
        let model = ModelProto {
            ir_version: 8,
            producer_name: "nsl".to_string(),
            producer_version: "0.1".to_string(),
            opset_import: vec![OperatorSetIdProto {
                domain: "".to_string(),
                version: 17,
            }],
            graph: Some(GraphProto {
                name: "test_graph".to_string(),
                node: vec![NodeProto {
                    input: vec!["input_0".to_string()],
                    output: vec!["output_0".to_string()],
                    name: "relu_0".to_string(),
                    op_type: "Relu".to_string(),
                    domain: "".to_string(),
                    attribute: vec![],
                }],
                initializer: vec![],
                input: vec![ValueInfoProto {
                    name: "input_0".to_string(),
                    r#type: Some(TypeProto {
                        tensor_type: Some(TensorTypeProto {
                            elem_type: FLOAT,
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    TensorShapeDimProto { dim_value: 1 },
                                    TensorShapeDimProto { dim_value: 4 },
                                ],
                            }),
                        }),
                    }),
                }],
                output: vec![ValueInfoProto {
                    name: "output_0".to_string(),
                    r#type: Some(TypeProto {
                        tensor_type: Some(TensorTypeProto {
                            elem_type: FLOAT,
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    TensorShapeDimProto { dim_value: 1 },
                                    TensorShapeDimProto { dim_value: 4 },
                                ],
                            }),
                        }),
                    }),
                }],
            }),
        };

        // Encode
        let mut buf = Vec::new();
        model.encode(&mut buf).expect("encode should succeed");
        assert!(!buf.is_empty(), "encoded bytes should be non-empty");

        // Decode
        let decoded = ModelProto::decode(buf.as_slice()).expect("decode should succeed");

        // Verify top-level fields
        assert_eq!(decoded.ir_version, 8);
        assert_eq!(decoded.producer_name, "nsl");
        assert_eq!(decoded.producer_version, "0.1");
        assert_eq!(decoded.opset_import.len(), 1);
        assert_eq!(decoded.opset_import[0].version, 17);

        // Verify graph
        let graph = decoded.graph.as_ref().expect("graph should be present");
        assert_eq!(graph.name, "test_graph");
        assert_eq!(graph.node.len(), 1);

        // Verify the Relu node
        let node = &graph.node[0];
        assert_eq!(node.op_type, "Relu");
        assert_eq!(node.name, "relu_0");
        assert_eq!(node.input, vec!["input_0"]);
        assert_eq!(node.output, vec!["output_0"]);

        // Verify inputs/outputs
        assert_eq!(graph.input.len(), 1);
        assert_eq!(graph.input[0].name, "input_0");
        assert_eq!(graph.output.len(), 1);
        assert_eq!(graph.output[0].name, "output_0");

        // Verify shape
        let tensor_type = graph.input[0]
            .r#type
            .as_ref()
            .unwrap()
            .tensor_type
            .as_ref()
            .unwrap();
        assert_eq!(tensor_type.elem_type, FLOAT);
        let dims: Vec<i64> = tensor_type
            .shape
            .as_ref()
            .unwrap()
            .dim
            .iter()
            .map(|d| d.dim_value)
            .collect();
        assert_eq!(dims, vec![1, 4]);
    }

    /// Verify that attribute encoding preserves float, int, and ints fields.
    #[test]
    fn test_attribute_roundtrip() {
        let attr_float = AttributeProto {
            name: "alpha".to_string(),
            r#type: ATTRIBUTE_FLOAT,
            f: 0.01_f32,
            i: 0,
            ints: vec![],
        };
        let attr_int = AttributeProto {
            name: "axis".to_string(),
            r#type: ATTRIBUTE_INT,
            f: 0.0,
            i: -1,
            ints: vec![],
        };
        let attr_ints = AttributeProto {
            name: "perm".to_string(),
            r#type: ATTRIBUTE_INTS,
            f: 0.0,
            i: 0,
            ints: vec![0, 2, 1],
        };

        for attr in &[attr_float, attr_int, attr_ints] {
            let mut buf = Vec::new();
            attr.encode(&mut buf).unwrap();
            let decoded = AttributeProto::decode(buf.as_slice()).unwrap();
            assert_eq!(decoded.name, attr.name);
            assert_eq!(decoded.r#type, attr.r#type);
            assert!((decoded.f - attr.f).abs() < 1e-6);
            assert_eq!(decoded.i, attr.i);
            assert_eq!(decoded.ints, attr.ints);
        }
    }
}
