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
///
/// M57.1 §3.4 (Task 3.2): summary also enumerates declared Port::Input names +
/// LocalParam names so the no-dangling-references invariant is auditable in
/// the snapshot. LocalParams are listed by name+width only (the baked `value`
/// field is Task 4.1's responsibility and lands as a separate snapshot diff).
fn structural_summary(module: &HirModule) -> String {
    use nsl_codegen::hir::Port;
    let mut out = String::new();
    out.push_str(&format!("module: {}\n", module.name));
    out.push_str(&format!("test_taps: {}\n", module.test_taps));
    out.push_str(&format!("ports: {}\n", module.ports.len()));
    for port in &module.ports {
        match port {
            Port::Input { name, width } => {
                out.push_str(&format!("  Port::Input {name} width={width}\n"));
            }
            Port::Output { name, width, .. } => {
                out.push_str(&format!("  Port::Output {name} width={width}\n"));
            }
        }
    }
    // Group LocalParams by leading-alpha prefix (e.g. "W1_…" and "b1_…") and
    // collapse each group to first + last element so v1 matmul snapshots with
    // ~100k weight elements remain human-auditable. Group cardinality (the
    // primary correctness signal) is still asserted exactly.
    out.push_str(&format!("local_params: {}\n", module.local_params().len()));
    summarize_local_params(&mut out, module.local_params());
    // M57.1 wire-array mini §3.2 (Task W5): also enumerate LocalParamArrays
    // (the fused-emitter shape for W<i>/b<i>). Names + dims + width are
    // deterministic; the bulk `values` field is elided to keep the snapshot
    // human-auditable. Emitted only when non-zero — standalone lowerer
    // snapshots stay byte-identical (Pin 2: M57 snapshot preservation).
    if !module.local_param_arrays().is_empty() {
        out.push_str(&format!("local_param_arrays: {}\n", module.local_param_arrays().len()));
        for lpa in module.local_param_arrays() {
            out.push_str(&format!(
                "  LocalParamArray {} dims={:?} width={}\n",
                lpa.name, lpa.dims, lpa.width,
            ));
        }
    }
    out.push_str(&format!("nodes: {}\n", module.nodes().len()));
    summarize_node_list(&mut out, module.nodes(), 0);
    out
}

/// Summarize a list of HIR nodes with run-length collapse for the
/// bias-seed and x-bus fan-out repetition patterns. M57.1 wire-array
/// realization's per-element AssignWireArrayElement runs (`acc_l<i>[o][0] =
/// IndexedLocalParam(b<i>[o])` for o ∈ [0, n_outputs) and `x_l<i>_a[k] =
/// PortBitSlice(x_l<i>[k*W +: W])` for k ∈ [0, k_dim)) would otherwise
/// dwarf the structural skeleton. Collapsing to first + last + count keeps
/// the snapshot human-auditable while still asserting per-element cardinality
/// exactly via the `nodes:` total.
fn summarize_node_list(out: &mut String, nodes: &[HirNode], depth: usize) {
    let mut i = 0;
    while i < nodes.len() {
        // Try to find a run of consecutive AssignWireArrayElement nodes that
        // form a "fan-out" or "bias-seed" pattern. The run extends as long
        // as the same array_name is being driven AND each consecutive entry
        // uses a Literal(o) at position 0 differing by exactly +1.
        let run_len = elidable_assign_run(nodes, i);
        if run_len >= 3 {
            summarize_node(out, &nodes[i], i, depth);
            let indent = "  ".repeat(depth);
            out.push_str(&format!(
                "{indent}…({} consecutive AssignWireArrayElement entries elided)…\n",
                run_len - 2,
            ));
            summarize_node(out, &nodes[i + run_len - 1], i + run_len - 1, depth);
            i += run_len;
        } else {
            summarize_node(out, &nodes[i], i, depth);
            i += 1;
        }
    }
}

/// Return the number of consecutive `AssignWireArrayElement` nodes starting
/// at `start` that drive the same array with sequentially-incrementing
/// Literal-only indices (i.e. the v1 MLP bias-seed / x-bus fan-out pattern).
/// Returns 0 when the run is empty or doesn't form a pattern.
fn elidable_assign_run(nodes: &[HirNode], start: usize) -> usize {
    use nsl_codegen::hir::nodes::IndexExpr;
    fn key(n: &HirNode) -> Option<(&str, &Vec<IndexExpr>)> {
        match n {
            HirNode::AssignWireArrayElement(a) => Some((a.array_name.as_str(), &a.indices)),
            _ => None,
        }
    }
    let (first_name, first_idx) = match key(&nodes[start]) {
        Some(k) => k,
        None => return 0,
    };
    // All indices must be Literal — fan-out / bias-seed patterns use only
    // compile-time integer indices.
    if first_idx.iter().any(|ix| !matches!(ix, IndexExpr::Literal(_))) {
        return 0;
    }
    let mut len = 1;
    let mut prev_idx0 = match &first_idx[0] {
        IndexExpr::Literal(n) => *n,
        _ => unreachable!(),
    };
    while start + len < nodes.len() {
        let next = match key(&nodes[start + len]) {
            Some(k) => k,
            None => break,
        };
        if next.0 != first_name { break; }
        if next.1.len() != first_idx.len() { break; }
        // All non-leading indices must match exactly; the leading index
        // must be Literal(prev + 1).
        let mut ok = true;
        for (k, ix) in next.1.iter().enumerate() {
            match (k, ix) {
                (0, IndexExpr::Literal(n)) => {
                    if *n != prev_idx0 + 1 { ok = false; break; }
                }
                (_, IndexExpr::Literal(n)) => {
                    match &first_idx[k] {
                        IndexExpr::Literal(f) if *n == *f => {}
                        _ => { ok = false; break; }
                    }
                }
                _ => { ok = false; break; }
            }
        }
        if !ok { break; }
        prev_idx0 += 1;
        len += 1;
    }
    if len >= 3 { len } else { 0 }
}

/// Group LocalParams by the prefix that precedes the first `_<digit>` segment
/// (e.g. `W1_0_0` and `W1_127_9` both group as `W1`; `b1_0` and `b1_9` as
/// `b1`). Within each contiguous run of same-prefix entries, emit the first +
/// last element only with an ellipsis showing the omitted middle count.
fn summarize_local_params(
    out: &mut String,
    lps: &[nsl_codegen::hir::LocalParam],
) {
    fn prefix(name: &str) -> &str {
        // Find the first `_` followed by an ASCII digit; everything before it
        // is the prefix. Anchoring on `_<digit>` lets `W1_0_0` group as `W1`
        // and `b1_0` group as `b1` without conflating with hypothetical names
        // like `Weights_…` (which would have no leading `_<digit>`).
        let bytes = name.as_bytes();
        let mut i = 0;
        while i + 1 < bytes.len() {
            if bytes[i] == b'_' && bytes[i + 1].is_ascii_digit() {
                return &name[..i];
            }
            i += 1;
        }
        name
    }

    let mut i = 0;
    while i < lps.len() {
        let p = prefix(&lps[i].name);
        let mut j = i + 1;
        while j < lps.len() && prefix(&lps[j].name) == p {
            j += 1;
        }
        let run = &lps[i..j];
        if run.len() == 1 {
            out.push_str(&format!(
                "  LocalParam {} width={}\n",
                run[0].name, run[0].width
            ));
        } else {
            out.push_str(&format!(
                "  LocalParam {} width={}\n",
                run[0].name, run[0].width
            ));
            if run.len() > 2 {
                out.push_str(&format!("  …(prefix={p:?}, {} elided)…\n", run.len() - 2));
            }
            out.push_str(&format!(
                "  LocalParam {} width={}\n",
                run[run.len() - 1].name, run[run.len() - 1].width
            ));
        }
        i = j;
    }
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
        // M57.1 wire-array realization (Task W1/W2): module-scope multi-dim
        // wire array. Names + dims + width are deterministic; no IDs.
        HirNode::WireArray(wa) => {
            out.push_str(&format!(
                "{indent}[{idx}] WireArray {} dims={:?} width={}\n",
                wa.name, wa.dims, wa.width,
            ));
        }
        // M57.1 wire-array realization (Task W4): drives one element of a
        // WireArray. Render index shape + src kind without leaking WireIds.
        HirNode::AssignWireArrayElement(a) => {
            let idx_str = format_index_exprs(&a.indices);
            out.push_str(&format!(
                "{indent}[{idx}] AssignWireArrayElement {}{} = {}\n",
                a.array_name, idx_str, format_sigref(&a.src),
            ));
        }
        // M57.2 (Task 3): combinational equality guard.
        HirNode::CmpEq(c) => {
            out.push_str(&format!(
                "{indent}[{idx}] CmpEq lhs={} rhs={}\n",
                format_sigref(&c.lhs), c.rhs,
            ));
        }
        // M57.2 (Task 4): combinational add-constant.
        HirNode::AddConst(a) => {
            out.push_str(&format!(
                "{indent}[{idx}] AddConst width={} k={} src={}\n",
                a.width, a.k, format_sigref(&a.src),
            ));
        }
        // M57.2 (Task 5): declaration-only scalar register.
        HirNode::RegDecl(d) => {
            out.push_str(&format!(
                "{indent}[{idx}] RegDecl width={}\n",
                d.width,
            ));
        }
        // M57.2 (Task 6): module-scope clocked register array.
        HirNode::RegArray(ra) => {
            out.push_str(&format!(
                "{indent}[{idx}] RegArray {} dims={:?} width={}\n",
                ra.name, ra.dims, ra.width,
            ));
        }
        // M57.2 (Task 9): clocked sequential process.
        HirNode::SeqProcess(sp) => {
            out.push_str(&format!(
                "{indent}[{idx}] SeqProcess (reset_arrays={}, body={} stmts)\n",
                sp.reset_arrays.len(), sp.body.len(),
            ));
        }
    }
}

fn format_index_exprs(indices: &[nsl_codegen::hir::nodes::IndexExpr]) -> String {
    use nsl_codegen::hir::nodes::IndexExpr;
    indices
        .iter()
        .map(|ix| match ix {
            IndexExpr::Literal(n) => format!("[{}]", n),
            IndexExpr::Genvar(name) => format!("[{}]", name),
            IndexExpr::GenvarPlus(name, k) => format!("[({} + {})]", name, k),
            IndexExpr::Reg(r) => format!("[_r{}]", r.0),
            IndexExpr::RegPlus(r, k) => format!("[(_r{} + {})]", r.0, k),
        })
        .collect()
}

fn format_sigref(s: &nsl_codegen::hir::SignalRef) -> String {
    match s {
        nsl_codegen::hir::SignalRef::Port(name) => format!("Port({name})"),
        nsl_codegen::hir::SignalRef::Wire(_) => "Wire(<id>)".to_string(),
        nsl_codegen::hir::SignalRef::Register(_) => "Register(<id>)".to_string(),
        nsl_codegen::hir::SignalRef::LocalParam(name) => format!("LocalParam({name})"),
        // M57.1 wire-array realization (Task W2): read of one element of a
        // module-scope wire array. Indices are deterministic; included verbatim.
        nsl_codegen::hir::SignalRef::WireArrayElement { array_name, indices } => {
            format!("WireArrayElement({array_name}{})", format_index_exprs(indices))
        }
        // M57.1 wire-array realization (Task W5): indexed LocalParamArray read.
        nsl_codegen::hir::SignalRef::IndexedLocalParam { array_name, indices } => {
            format!("IndexedLocalParam({array_name}{})", format_index_exprs(indices))
        }
        // M57.1 wire-array realization (Task W5): Verilog `name[base +: width]`.
        nsl_codegen::hir::SignalRef::PortBitSlice { name, base_bit, width } => {
            format!("PortBitSlice({name}[{base_bit} +: {width}])")
        }
        // M57.1 wire-array realization (Task W5): concat across array.
        nsl_codegen::hir::SignalRef::WireArrayConcat { array_name, n, fixed_index } => {
            match fixed_index {
                Some(fix) => format!("WireArrayConcat({array_name}, n={n}, fix={fix})"),
                None => format!("WireArrayConcat({array_name}, n={n})"),
            }
        }
        // M57.2: combinational reg-array element read — identical render shape to
        // WireArrayElement.
        nsl_codegen::hir::SignalRef::RegArrayElement { array_name, indices } => {
            format!("RegArrayElement({array_name}{})", format_index_exprs(indices))
        }
        // M57.2: concat across a reg-array (1D; no fixed_index) — drives `out`.
        nsl_codegen::hir::SignalRef::RegArrayConcat { array_name, n } => {
            format!("RegArrayConcat({array_name}, n={n})")
        }
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

    // M57.1 wire-array mini §3.1 + §3.2 (Task W5): module now contains the
    // fused wire-array shape. We just sanity-check the module identity +
    // tap flag; the bulk audit is in the snapshot.
    assert_eq!(module.name, "tiny_mlp");
    assert!(module.test_taps);

    // 2 LocalParamArrays per layer (W<i> + b<i>) × 2 layers = 4.
    assert_eq!(module.local_param_arrays().len(), 4);
    let lpa_names: Vec<&str> = module
        .local_param_arrays()
        .iter()
        .map(|lpa| lpa.name.as_str())
        .collect();
    assert!(lpa_names.contains(&"W1"));
    assert!(lpa_names.contains(&"b1"));
    assert!(lpa_names.contains(&"W2"));
    assert!(lpa_names.contains(&"b2"));

    // No `relu_in_l<i>` Port::Input anywhere — the fused emitter reads
    // directly from acc_l<i>[*][k_dim] via WireArrayElement.
    use nsl_codegen::hir::Port;
    let has_relu_in_port = module.ports.iter().any(|p| match p {
        Port::Input { name, .. } => name.starts_with("relu_in_l"),
        _ => false,
    });
    assert!(
        !has_relu_in_port,
        "fused v1 MLP must not declare relu_in_l<i> Port::Input"
    );

    // The final-layer Port::Output("out") is present.
    let has_out_port = module.ports.iter().any(|p| matches!(
        p, Port::Output { name, .. } if name == "out"
    ));
    assert!(has_out_port, "final layer must declare Port::Output(\"out\")");
}
