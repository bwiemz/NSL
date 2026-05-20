//! Snapshot tests for the KIR → HIR pass (M57 v1 spec §2.3).
//! Each snapshot captures the structural HIR shape for human audit.
//!
//! NOTE: WireId and GenvarId use process-global atomic counters, making exact
//! Debug output non-deterministic across test runs. Snapshots use a structural
//! summary (node kinds + widths) that strips the non-deterministic ID values.

use insta::assert_snapshot;
use nsl_codegen::hir::{HirModule, HirNode, KirToHirPass};
use nsl_codegen::kernel_ir::*;

/// Build a KernelIR from a list of ops using the real KirBuilder API.
fn kir_with_ops(name: &str, ops: Vec<KirOp>) -> KernelIR {
    let mut b = KirBuilder::new(name);
    let blk = b.new_block();
    b.set_block(blk);
    for op in ops {
        b.emit(op);
    }
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// Produce a stable structural summary of an HirModule for snapshot testing.
/// Strips non-deterministic WireId/GenvarId/RegisterId values; keeps node
/// kinds, widths, and port names which are the auditable content.
fn structural_summary(module: &HirModule) -> String {
    let mut out = String::new();
    out.push_str(&format!("module: {}\n", module.name));
    out.push_str(&format!("test_taps: {}\n", module.test_taps));
    out.push_str(&format!("ports: {}\n", module.ports.len()));
    out.push_str(&format!("nodes: {}\n", module.nodes().len()));
    for (i, node) in module.nodes().iter().enumerate() {
        summarize_node(&mut out, node, i, 0);
    }
    out
}

fn summarize_node(out: &mut String, node: &HirNode, idx: usize, depth: usize) {
    let indent = "  ".repeat(depth);
    match node {
        HirNode::GenerateFor(g) => {
            out.push_str(&format!("{indent}[{idx}] GenerateFor lo={} hi={} body={{\n",
                g.lo, g.hi));
            for (j, child) in g.body.iter().enumerate() {
                summarize_node(out, child, j, depth + 1);
            }
            out.push_str(&format!("{indent}}}\n"));
        }
        HirNode::Mul(m) => {
            out.push_str(&format!(
                "{indent}[{idx}] Mul a_width={} b_width={} out_width={} a={:?} b={:?}\n",
                m.a_width, m.b_width, m.out_width,
                format_sigref(&m.a), format_sigref(&m.b),
            ));
        }
        HirNode::Add(a) => {
            out.push_str(&format!(
                "{indent}[{idx}] Add width={} a={:?} b={:?}\n",
                a.width, format_sigref(&a.a), format_sigref(&a.b),
            ));
        }
        HirNode::Max0(m) => {
            out.push_str(&format!(
                "{indent}[{idx}] Max0 width={} a={:?}\n",
                m.width, format_sigref(&m.a),
            ));
        }
        HirNode::SignExtend(s) => {
            out.push_str(&format!(
                "{indent}[{idx}] SignExtend src_width={} dst_width={}\n",
                s.src_width, s.dst_width,
            ));
        }
        HirNode::Wire(w) => {
            out.push_str(&format!("{indent}[{idx}] Wire width={}\n", w.width));
        }
        HirNode::Register(r) => {
            out.push_str(&format!("{indent}[{idx}] Register width={}\n", r.width));
        }
        HirNode::GenerateIf(g) => {
            out.push_str(&format!("{indent}[{idx}] GenerateIf cond={:?}\n", g.cond));
        }
    }
}

fn format_sigref(s: &nsl_codegen::hir::SignalRef) -> String {
    match s {
        nsl_codegen::hir::SignalRef::Port(name) => format!("Port({name})"),
        nsl_codegen::hir::SignalRef::Wire(_) => "Wire(<id>)".to_string(),
        nsl_codegen::hir::SignalRef::Register(_) => "Register(<id>)".to_string(),
        nsl_codegen::hir::SignalRef::LocalParam(name) => format!("LocalParam({name})"),
    }
}

// ---------------------------------------------------------------------------
// Task 2.5: Matmul decomposition
// ---------------------------------------------------------------------------

#[test]
fn matmul_i8_decomposition() {
    // Layer 1: i8×i8→i32, shape [1,784]×[784,128]
    // prod_width = 8+8 = 16 < acc_width = 32 → 3-node MAC chain (Mul+SignExtend+Add)
    let kir = kir_with_ops(
        "matmul_i8_test",
        vec![KirOp::Matmul {
            a: 1, b: 2, out: 3,
            a_dtype: KirType::I8, b_dtype: KirType::I8, out_dtype: KirType::I32,
            a_shape: [1, 784], b_shape: [784, 128],
        }],
    );
    let pass = KirToHirPass::new(true);
    let module = pass.lower(&kir, "matmul_i8_test").unwrap();
    assert_snapshot!(structural_summary(&module));
}

#[test]
fn matmul_i32_decomposition_collapses_mac_chain() {
    // Layer 2: i32×i32→i64, shape [1,128]×[128,10]
    // prod_width = 32+32 = 64 == acc_width = 64 → 2-node MAC chain (Mul+Add, no SignExtend)
    let kir = kir_with_ops(
        "matmul_i32_test",
        vec![KirOp::Matmul {
            a: 1, b: 2, out: 3,
            a_dtype: KirType::I32, b_dtype: KirType::I32, out_dtype: KirType::I64,
            a_shape: [1, 128], b_shape: [128, 10],
        }],
    );
    let pass = KirToHirPass::new(true);
    let module = pass.lower(&kir, "matmul_i32_test").unwrap();
    assert_snapshot!(structural_summary(&module));
}

// ---------------------------------------------------------------------------
// Task 2.6: ElementwiseAdd decomposition
// ---------------------------------------------------------------------------

#[test]
fn elementwise_add_i32() {
    let kir = kir_with_ops(
        "bias_add_test",
        vec![KirOp::ElementwiseAdd {
            a: 1, b: 2, out: 3,
            dtype: KirType::I32, shape: [128],
        }],
    );
    let pass = KirToHirPass::new(true);
    let module = pass.lower(&kir, "bias_add_test").unwrap();
    assert_snapshot!(structural_summary(&module));
}

#[test]
fn elementwise_add_i64() {
    let kir = kir_with_ops(
        "bias_add_i64_test",
        vec![KirOp::ElementwiseAdd {
            a: 1, b: 2, out: 3,
            dtype: KirType::I64, shape: [10],
        }],
    );
    let pass = KirToHirPass::new(true);
    let module = pass.lower(&kir, "bias_add_i64_test").unwrap();
    assert_snapshot!(structural_summary(&module));
}

// ---------------------------------------------------------------------------
// Task 2.7: Relu decomposition
// ---------------------------------------------------------------------------

#[test]
fn relu_i32() {
    let kir = kir_with_ops(
        "relu_test",
        vec![KirOp::Relu {
            a: 1, out: 2,
            dtype: KirType::I32, shape: [128],
        }],
    );
    let pass = KirToHirPass::new(true);
    let module = pass.lower(&kir, "relu_test").unwrap();
    assert_snapshot!(structural_summary(&module));
}

// ---------------------------------------------------------------------------
// Task 2.8: Full v1 MLP composition
// ---------------------------------------------------------------------------

#[test]
fn full_v1_mlp_composition() {
    let kir = kir_with_ops(
        "tiny_mlp",
        vec![
            // Layer 1: 784 → 128 (i8×i8→i32)
            KirOp::Matmul {
                a: 1, b: 2, out: 3,
                a_dtype: KirType::I8, b_dtype: KirType::I8, out_dtype: KirType::I32,
                a_shape: [1, 784], b_shape: [784, 128],
            },
            KirOp::ElementwiseAdd {
                a: 3, b: 4, out: 5,
                dtype: KirType::I32, shape: [128],
            },
            KirOp::Relu {
                a: 5, out: 6,
                dtype: KirType::I32, shape: [128],
            },
            // Layer 2: 128 → 10 (i32×i32→i64)
            KirOp::Matmul {
                a: 6, b: 7, out: 8,
                a_dtype: KirType::I32, b_dtype: KirType::I32, out_dtype: KirType::I64,
                a_shape: [1, 128], b_shape: [128, 10],
            },
            KirOp::ElementwiseAdd {
                a: 8, b: 9, out: 10,
                dtype: KirType::I64, shape: [10],
            },
            KirOp::Relu {
                a: 10, out: 11,
                dtype: KirType::I64, shape: [10],
            },
        ],
    );

    let pass = KirToHirPass::new(true);
    let module = pass.lower(&kir, "tiny_mlp").unwrap();
    assert_snapshot!(structural_summary(&module));

    // Sanity: module has 6 GenerateFor bodies (one per op)
    assert_eq!(module.nodes().len(), 6);
    assert_eq!(module.name, "tiny_mlp");
    assert!(module.test_taps);
}
