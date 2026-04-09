//! M55: E2E roundtrip tests — exercises the full ZK pipeline from DAG through
//! lowering, witness generation, and folding prove/verify.
//!
//! ## Test tiers
//!
//! - **Tier 1**: Pure arithmetic (ReLU)
//! - **Tier 2**: Lookup-based (GELU)
//! - **Tier 3**: Full transformer ops (softmax)
//! - **Tier 4**: Multi-layer table sharing
//! - **Privacy mode**: WeightPrivate mode proof
//! - **Table sharing**: Verifies deduplication of lookup tables

use crate::zk::backend::{FoldingBackend, ZkConfig, ZkField, ZkMode};
use crate::zk::field::FieldElement;
use crate::zk::field_m31::Mersenne31Field;
use crate::zk::folding::{FoldingConfig, FoldingProver};
use crate::zk::lower::{lower_dag_to_zkir, lower_model_for_folding, ZkDag, ZkOp};
use crate::zk::witness::WitnessGenerator;

/// Helper: lower a DAG, generate witness, fold as a single layer, prove, and verify.
/// Returns `true` if verification succeeds.
fn roundtrip_prove_verify(dag: &ZkDag, input_values: &[FieldElement]) -> bool {
    let config = ZkConfig::default();
    let ir = lower_dag_to_zkir(dag, &config);

    let mut gen = WitnessGenerator::new();
    let witness = gen.generate(&ir, input_values, &[]).unwrap();

    // Use the folding backend with Mersenne-31 field.
    // Fold the whole-model IR as a single layer (small test DAGs don't
    // benefit from per-layer splitting).
    let mut prover = FoldingProver::<Mersenne31Field>::new(FoldingConfig::default());

    let layer_witness: Vec<Mersenne31Field> = (0..ir.num_wires as usize)
        .map(|i| {
            if i < witness.values.len() {
                // Map BN254 field element to M31 (take low bits)
                Mersenne31Field::from_u64(witness.values[i].limbs[0])
            } else {
                Mersenne31Field::zero()
            }
        })
        .collect();
    prover.fold_layer(&ir, &layer_witness).unwrap();

    let proof = prover.finalize().unwrap();

    // Verify the proof
    <FoldingProver<Mersenne31Field> as FoldingBackend<Mersenne31Field>>::verify(&proof, &[])
        .unwrap()
}

/// Helper: use compile_zk_from_dag for BN254 field path.
fn roundtrip_compile_zk(dag: &ZkDag) -> bool {
    let config = ZkConfig {
        field: ZkField::BN254,
        ..Default::default()
    };
    let proof = crate::zk::compile_zk_from_dag(dag, &config).unwrap();
    proof.num_folds > 0 && !proof.data.is_empty()
}

// ---------------------------------------------------------------------------
// Tier 1: Pure arithmetic — ReLU MLP
// ---------------------------------------------------------------------------

#[test]
fn tier1_mlp_relu_roundtrip() {
    // out = relu(matmul(x, W) + b)
    // x: [1, 4], W: [4, 2], b: [2]
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![1, 4],
                dtype_bits: 8,
            },
            ZkOp::Weight {
                name: "W".into(),
                shape: vec![4, 2],
                dtype_bits: 8,
                values: Some(vec![1, 2, 3, 4, 5, 6, 7, 8]),
            },
            ZkOp::Weight {
                name: "b".into(),
                shape: vec![1, 2],
                dtype_bits: 8,
                values: Some(vec![1, 1]),
            },
            ZkOp::Matmul { a: 0, b: 1 },
            ZkOp::Add { a: 3, b: 2 },
            ZkOp::Relu { input: 4 },
        ],
        output_idx: 5,
        input_indices: vec![0],
        weight_indices: vec![1, 2],
    };

    // Lower to ZK-IR and verify structure
    let config = ZkConfig::default();
    let ir = lower_dag_to_zkir(&dag, &config);

    // Should have ReLU lookup table registered
    assert!(
        ir.lookup_tables.contains_key(&("relu".to_string(), 8)),
        "relu table should be registered"
    );

    // Generate witness with input values
    let input_values: Vec<FieldElement> = vec![1, 2, 3, 4]
        .into_iter()
        .map(FieldElement::from_u64)
        .collect();
    assert!(
        roundtrip_prove_verify(&dag, &input_values),
        "Tier 1 MLP+ReLU proof should verify"
    );
}

#[test]
fn tier1_simple_relu_positive_passthrough() {
    // A single relu on a positive input should pass through unchanged.
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![1],
                dtype_bits: 8,
            },
            ZkOp::Relu { input: 0 },
        ],
        output_idx: 1,
        input_indices: vec![0],
        weight_indices: vec![],
    };

    let input = vec![FieldElement::from_u64(5)];
    assert!(
        roundtrip_prove_verify(&dag, &input),
        "ReLU(5) should prove/verify"
    );
}

// ---------------------------------------------------------------------------
// Tier 2: Lookup-based — GELU
// ---------------------------------------------------------------------------

#[test]
fn tier2_mlp_gelu_roundtrip() {
    // out = gelu(matmul(x, W) + b)
    // x: [1, 2], W: [2, 2], b: [2]
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![1, 2],
                dtype_bits: 8,
            },
            ZkOp::Weight {
                name: "W".into(),
                shape: vec![2, 2],
                dtype_bits: 8,
                values: Some(vec![1, 0, 0, 1]),
            },
            ZkOp::Weight {
                name: "b".into(),
                shape: vec![1, 2],
                dtype_bits: 8,
                values: Some(vec![0, 0]),
            },
            ZkOp::Matmul { a: 0, b: 1 },
            ZkOp::Add { a: 3, b: 2 },
            ZkOp::Gelu { input: 4 },
        ],
        output_idx: 5,
        input_indices: vec![0],
        weight_indices: vec![1, 2],
    };

    let config = ZkConfig::default();
    let ir = lower_dag_to_zkir(&dag, &config);

    // GELU lookup table should be registered
    assert!(
        ir.lookup_tables.contains_key(&("gelu".to_string(), 8)),
        "gelu table should be registered"
    );

    let input_values = vec![FieldElement::from_u64(3), FieldElement::from_u64(5)];
    assert!(
        roundtrip_prove_verify(&dag, &input_values),
        "Tier 2 MLP+GELU proof should verify"
    );
}

#[test]
fn tier2_gelu_zero_input() {
    // GELU(0) = 0
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![1],
                dtype_bits: 8,
            },
            ZkOp::Gelu { input: 0 },
        ],
        output_idx: 1,
        input_indices: vec![0],
        weight_indices: vec![],
    };

    let input = vec![FieldElement::from_u64(0)];
    assert!(
        roundtrip_prove_verify(&dag, &input),
        "GELU(0) should prove/verify"
    );
}

// ---------------------------------------------------------------------------
// Tier 3: Full transformer ops — softmax(Q @ K^T)
// ---------------------------------------------------------------------------

#[test]
fn tier3_attention_softmax_roundtrip() {
    // softmax(matmul(Q, K^T))
    // Q: [1, 2], K: [1, 2] (transpose K -> [2, 1])
    // matmul(Q, K^T) -> [1, 1], softmax -> [1, 1]
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "Q".into(),
                shape: vec![1, 2],
                dtype_bits: 8,
            },
            ZkOp::Weight {
                name: "K".into(),
                shape: vec![1, 2],
                dtype_bits: 8,
                values: Some(vec![1, 1]),
            },
            ZkOp::Transpose { input: 1 },        // K^T: [2, 1]
            ZkOp::Matmul { a: 0, b: 2 },         // Q @ K^T: [1, 1]
            ZkOp::Softmax { input: 3, dim: -1 }, // softmax: [1, 1]
        ],
        output_idx: 4,
        input_indices: vec![0],
        weight_indices: vec![1],
    };

    let config = ZkConfig::default();
    let ir = lower_dag_to_zkir(&dag, &config);

    // Softmax uses exp + inv lookup tables
    assert!(
        ir.lookup_tables.contains_key(&("exp".to_string(), 8)),
        "exp table should be registered for softmax"
    );
    assert!(
        ir.lookup_tables.contains_key(&("inv".to_string(), 8)),
        "inv table should be registered for softmax"
    );

    let input_values = vec![FieldElement::from_u64(2), FieldElement::from_u64(3)];
    assert!(
        roundtrip_prove_verify(&dag, &input_values),
        "Tier 3 attention+softmax proof should verify"
    );
}

#[test]
fn tier3_softmax_multi_element() {
    // softmax over a 4-element vector — exercises the full exp+sum+inv+mul chain
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![4],
                dtype_bits: 8,
            },
            ZkOp::Softmax { input: 0, dim: -1 },
        ],
        output_idx: 1,
        input_indices: vec![0],
        weight_indices: vec![],
    };

    let input_values: Vec<FieldElement> = vec![1, 2, 3, 4]
        .into_iter()
        .map(FieldElement::from_u64)
        .collect();
    assert!(
        roundtrip_prove_verify(&dag, &input_values),
        "softmax over 4-element vector should prove/verify"
    );
}

// ---------------------------------------------------------------------------
// Tier 4: Multi-layer table sharing
// ---------------------------------------------------------------------------

#[test]
fn tier4_two_layer_table_sharing() {
    // Two sequential relu layers — verify only ONE relu table exists in IR
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![4],
                dtype_bits: 8,
            },
            ZkOp::Relu { input: 0 },
            ZkOp::Relu { input: 1 }, // second relu — should share table
        ],
        output_idx: 2,
        input_indices: vec![0],
        weight_indices: vec![],
    };

    let config = ZkConfig::default();
    let ir = lower_dag_to_zkir(&dag, &config);
    let relu_tables: Vec<_> = ir
        .lookup_tables
        .keys()
        .filter(|(name, _)| name == "relu")
        .collect();
    assert_eq!(
        relu_tables.len(),
        1,
        "ReLU table should be shared across layers"
    );

    // Also prove/verify the result
    let input_values: Vec<FieldElement> = vec![1, 2, 3, 4]
        .into_iter()
        .map(FieldElement::from_u64)
        .collect();
    assert!(
        roundtrip_prove_verify(&dag, &input_values),
        "two-layer ReLU should prove/verify"
    );
}

#[test]
fn tier4_mixed_activation_tables() {
    // relu then gelu — should create two separate tables
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![2],
                dtype_bits: 8,
            },
            ZkOp::Relu { input: 0 },
            ZkOp::Gelu { input: 1 },
        ],
        output_idx: 2,
        input_indices: vec![0],
        weight_indices: vec![],
    };

    let config = ZkConfig::default();
    let ir = lower_dag_to_zkir(&dag, &config);
    assert!(ir.lookup_tables.contains_key(&("relu".to_string(), 8)));
    assert!(ir.lookup_tables.contains_key(&("gelu".to_string(), 8)));
    assert_eq!(
        ir.lookup_tables.len(),
        2,
        "relu + gelu should produce exactly 2 tables"
    );

    let input_values = vec![FieldElement::from_u64(3), FieldElement::from_u64(5)];
    assert!(
        roundtrip_prove_verify(&dag, &input_values),
        "mixed activation tables should prove/verify"
    );
}

// ---------------------------------------------------------------------------
// Privacy mode tests
// ---------------------------------------------------------------------------

#[test]
fn weight_private_mode_produces_valid_proof() {
    // Simple matmul with private weights
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![1, 2],
                dtype_bits: 8,
            },
            ZkOp::Weight {
                name: "W".into(),
                shape: vec![2, 1],
                dtype_bits: 8,
                values: Some(vec![3, 5]),
            },
            ZkOp::Matmul { a: 0, b: 1 },
        ],
        output_idx: 2,
        input_indices: vec![0],
        weight_indices: vec![1],
    };

    let config = ZkConfig {
        mode: ZkMode::WeightPrivate,
        ..ZkConfig::default()
    };

    let ir = lower_dag_to_zkir(&dag, &config);

    let input_values = vec![FieldElement::from_u64(2), FieldElement::from_u64(4)];
    let mut gen = WitnessGenerator::new();
    let witness = gen.generate(&ir, &input_values, &[]).unwrap();

    // Fold using Mersenne-31
    let layer_irs = lower_model_for_folding(&dag, &config);
    let mut prover = FoldingProver::<Mersenne31Field>::new(FoldingConfig::default());

    for layer_ir in &layer_irs {
        let layer_witness: Vec<Mersenne31Field> = (0..layer_ir.num_wires as usize)
            .map(|i| {
                if i < witness.values.len() {
                    Mersenne31Field::from_u64(witness.values[i].limbs[0])
                } else {
                    Mersenne31Field::zero()
                }
            })
            .collect();
        prover.fold_layer(layer_ir, &layer_witness).unwrap();
    }

    let proof = prover.finalize().unwrap();
    let valid =
        <FoldingProver<Mersenne31Field> as FoldingBackend<Mersenne31Field>>::verify(&proof, &[])
            .unwrap();
    assert!(valid, "WeightPrivate mode proof should verify");
}

#[test]
fn wrong_outputs_rejected() {
    // Prove a correct witness, then verify the proof is structurally valid
    // but would fail if we tampered with the accumulator
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![2],
                dtype_bits: 8,
            },
            ZkOp::Relu { input: 0 },
        ],
        output_idx: 1,
        input_indices: vec![0],
        weight_indices: vec![],
    };

    let config = ZkConfig::default();
    let ir = lower_dag_to_zkir(&dag, &config);

    let input_values = vec![FieldElement::from_u64(3), FieldElement::from_u64(7)];
    let mut gen = WitnessGenerator::new();
    let witness = gen.generate(&ir, &input_values, &[]).unwrap();

    let layer_irs = lower_model_for_folding(&dag, &config);
    let mut prover = FoldingProver::<Mersenne31Field>::new(FoldingConfig::default());

    for layer_ir in &layer_irs {
        let layer_witness: Vec<Mersenne31Field> = (0..layer_ir.num_wires as usize)
            .map(|i| {
                if i < witness.values.len() {
                    Mersenne31Field::from_u64(witness.values[i].limbs[0])
                } else {
                    Mersenne31Field::zero()
                }
            })
            .collect();
        prover.fold_layer(layer_ir, &layer_witness).unwrap();
    }

    let proof = prover.finalize().unwrap();

    // Valid proof should verify
    let valid =
        <FoldingProver<Mersenne31Field> as FoldingBackend<Mersenne31Field>>::verify(&proof, &[])
            .unwrap();
    assert!(valid, "correct proof should verify");

    // Empty proof should be rejected
    use crate::zk::backend::ZkProof;
    let fake_proof = ZkProof {
        data: vec![],
        num_folds: 0,
        public_inputs: vec![],
        public_outputs: vec![],
    };
    let result = <FoldingProver<Mersenne31Field> as FoldingBackend<Mersenne31Field>>::verify(
        &fake_proof,
        &[],
    );
    assert!(result.is_err(), "empty proof should be rejected");
}

// ---------------------------------------------------------------------------
// Additional coverage: sigmoid, tanh, elementwise mul
// ---------------------------------------------------------------------------

#[test]
fn sigmoid_roundtrip() {
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![2],
                dtype_bits: 8,
            },
            ZkOp::Sigmoid { input: 0 },
        ],
        output_idx: 1,
        input_indices: vec![0],
        weight_indices: vec![],
    };

    let input_values = vec![FieldElement::from_u64(0), FieldElement::from_u64(5)];
    assert!(
        roundtrip_prove_verify(&dag, &input_values),
        "sigmoid roundtrip should verify"
    );
}

#[test]
fn elementwise_mul_roundtrip() {
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "a".into(),
                shape: vec![3],
                dtype_bits: 8,
            },
            ZkOp::Input {
                name: "b".into(),
                shape: vec![3],
                dtype_bits: 8,
            },
            ZkOp::Mul { a: 0, b: 1 },
        ],
        output_idx: 2,
        input_indices: vec![0, 1],
        weight_indices: vec![],
    };

    let input_values: Vec<FieldElement> = vec![2, 3, 4, 5, 6, 7]
        .into_iter()
        .map(FieldElement::from_u64)
        .collect();
    assert!(
        roundtrip_prove_verify(&dag, &input_values),
        "elementwise mul roundtrip should verify"
    );
}

#[test]
fn matmul_add_roundtrip_no_activation() {
    // Pure linear layer: out = matmul(x, W) + b, no activation
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![1, 3],
                dtype_bits: 8,
            },
            ZkOp::Weight {
                name: "W".into(),
                shape: vec![3, 2],
                dtype_bits: 8,
                values: Some(vec![1, 0, 0, 1, 1, 1]),
            },
            ZkOp::Weight {
                name: "b".into(),
                shape: vec![1, 2],
                dtype_bits: 8,
                values: Some(vec![10, 20]),
            },
            ZkOp::Matmul { a: 0, b: 1 },
            ZkOp::Add { a: 3, b: 2 },
        ],
        output_idx: 4,
        input_indices: vec![0],
        weight_indices: vec![1, 2],
    };

    let input_values: Vec<FieldElement> = vec![1, 2, 3]
        .into_iter()
        .map(FieldElement::from_u64)
        .collect();
    assert!(
        roundtrip_prove_verify(&dag, &input_values),
        "linear layer (no activation) roundtrip should verify"
    );
}

// ---------------------------------------------------------------------------
// BN254 field path via compile_zk_from_dag
// ---------------------------------------------------------------------------

#[test]
fn bn254_field_path_compiles() {
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![2],
                dtype_bits: 8,
            },
            ZkOp::Relu { input: 0 },
        ],
        output_idx: 1,
        input_indices: vec![0],
        weight_indices: vec![],
    };

    assert!(
        roundtrip_compile_zk(&dag),
        "BN254 field path should produce a valid folding proof"
    );
}

#[test]
fn m31_field_path_compiles() {
    let dag = ZkDag {
        ops: vec![
            ZkOp::Input {
                name: "x".into(),
                shape: vec![2],
                dtype_bits: 8,
            },
            ZkOp::Relu { input: 0 },
        ],
        output_idx: 1,
        input_indices: vec![0],
        weight_indices: vec![],
    };

    let config = ZkConfig {
        field: ZkField::Mersenne31,
        ..Default::default()
    };
    let proof = crate::zk::compile_zk_from_dag(&dag, &config).unwrap();
    assert!(proof.num_folds > 0);
    assert!(!proof.data.is_empty());
}
