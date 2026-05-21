//! KIR → HIR pass per M57 v1 spec §2.3 (load-bearing pass).
//! Three concerns: op recognition + dispatch; HIR construction; dtype propagation.

use crate::fpga_error::{FpgaLoweringError, V1_SUPPORTED_OPS};
use crate::hir::module::HirModule;
use crate::kernel_ir::{KernelIR, KirOp, KirType, VarId};

pub struct KirToHirPass {
    pub test_taps: bool,
}

impl KirToHirPass {
    pub fn new(test_taps: bool) -> Self {
        Self { test_taps }
    }

    /// Lower a KernelIR (representing one NSL `model` block) to a single HirModule.
    pub fn lower(
        &self,
        kir: &KernelIR,
        module_name: &str,
    ) -> Result<HirModule, FpgaLoweringError> {
        // M57.1 wire-array mini §6 (snapshot stability): WireId / RegisterId /
        // GenvarId counters are process-global atomics. Without this reset, the
        // `_gv<N>` numbering in HIR snapshots depends on test discovery order
        // — running `full_v1_mlp_composition` alone produces different IDs
        // than running it as part of the full suite. Resetting at the entry
        // to `lower` makes per-module snapshots deterministic without coupling
        // to test ordering.
        crate::hir::ids::WireId::reset();
        crate::hir::ids::RegisterId::reset();
        crate::hir::ids::GenvarId::reset();

        let mut module = HirModule::new(module_name);
        module.test_taps = self.test_taps;

        // M57.1 §3.5 + wire-array mini §3.1: peekable layer-aware iteration.
        // The dispatch prefers the fused v1 MLP chain emitter when a contiguous
        // Matmul → ElementwiseAdd → Relu pattern is recognized; otherwise it
        // falls through to the standalone lowerers (preserving M57 snapshots,
        // Pin 2).
        let ops: Vec<&KirOp> = kir.ops().collect();
        let mut i = 0;
        while i < ops.len() {
            match ops[i] {
                KirOp::Matmul { out: matmul_out, .. } => {
                    // M57.1 wire-array mini §3.1: try the multi-layer chain
                    // first. Recognizer verifies SSA + dimensional handoff
                    // (Pin 1) before fusing; errors loudly on dimensional
                    // mismatch.
                    if let Some(chain_len) = try_recognize_v1_mlp_chain(&ops, i)? {
                        lower_v1_mlp_layer_chain(
                            &ops[i..i + chain_len],
                            &mut module,
                        )?;
                        i += chain_len;
                        continue;
                    }
                    // Single Matmul → optional ElementwiseAdd standalone path.
                    let bias_op = ops.get(i + 1).copied().filter(|next| {
                        matches!(
                            next,
                            KirOp::ElementwiseAdd { a, .. } if *a == *matmul_out
                        )
                    });
                    self.lower_matmul_with_optional_bias(ops[i], bias_op, &mut module)?;
                    i += if bias_op.is_some() { 2 } else { 1 };
                }
                KirOp::ElementwiseAdd { .. } => {
                    // Standalone ElementwiseAdd (not absorbed by a preceding
                    // Matmul). The v1 MLP path doesn't emit standalone Adds
                    // (every Add follows a Matmul), but the lowering remains
                    // correct if they appear.
                    self.lower_elementwise_add(ops[i], &mut module)?;
                    i += 1;
                }
                KirOp::Relu { .. } => {
                    self.lower_relu(ops[i], &mut module)?;
                    i += 1;
                }

                // Unsupported in v1: error at first unrecognized op with localized
                // op-kind name + the supported list (spec §2.5).
                other => {
                    return Err(FpgaLoweringError::UnsupportedKirOp {
                        op_kind: kir_op_kind_name(other),
                        supported: V1_SUPPORTED_OPS,
                    })
                }
            }
        }

        Ok(module)
    }

    /// M57.1 §3.5: wrapper that consumes an optional adjacent ElementwiseAdd(bias)
    /// for the ripple-chain bias-as-seed fold.
    ///
    /// Folds the bias into the MAC accumulator's initial value, replacing the
    /// post-MAC ElementwiseAdd. Per the §3.5 ripple chain:
    ///
    /// ```text
    ///   acc[o][0] = b<i>[o]                                   (bias-as-seed; Q3 Pin 3)
    ///   for k ∈ [0, k_dim):
    ///       prod[k] = sext( x[k] * W[k,o] )
    ///       acc[o][k+1] = acc[o][k] + prod[k]
    ///   result[o] = acc[o][k_dim]
    /// ```
    ///
    /// When `bias_op` is `None` (defensive path), the accumulator seeds from
    /// zero. v1 MLP always has bias; the no-bias path exists for completeness
    /// (e.g. a stand-alone `Matmul` op that didn't have an adjacent bias-add).
    ///
    /// HIR structural encoding (approach (a) from plan §3.4 Step 1):
    /// - Outer GenerateFor over n_outputs declares one `bias_seed_wire`
    ///   (acc_width). The wire's driver is conceptually the bias LocalParam
    ///   `b<i>_<o>` (or 0 when no bias); the Verilog emitter realizes the
    ///   cross-iteration wire-array threading in PR 3.
    /// - Inner GenerateFor over k_dim emits the per-iteration MAC body
    ///   (Mul + optional SignExtend + Add), exactly as in M57. The Add's
    ///   first operand is a layer-scoped accumulator placeholder
    ///   (`acc_l<i>_prev`); the Verilog emitter resolves the
    ///   cross-iteration wiring to acc[o][k].
    ///
    /// The separate ElementwiseAdd that previously implemented the bias is
    /// NOT emitted when `bias_op` is `Some` — the fold subsumes it.
    fn lower_matmul_with_optional_bias(
        &self,
        matmul_op: &KirOp,
        bias_op: Option<&KirOp>,
        module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        use crate::hir::control::GenerateFor;
        use crate::hir::module::HirNode;
        use crate::hir::nodes::{Add, LocalParam, Mul, Port, SignExtend, Wire};
        use crate::hir::signals::SignalRef;

        let (a_dtype, b_dtype, out_dtype, a_shape, b_shape) = match matmul_op {
            KirOp::Matmul {
                a_dtype,
                b_dtype,
                out_dtype,
                a_shape,
                b_shape,
                ..
            } => (a_dtype.clone(), b_dtype.clone(), out_dtype.clone(), *a_shape, *b_shape),
            _ => unreachable!("lower_matmul_with_optional_bias called with non-Matmul op"),
        };

        // Standard matmul C[r,c] = sum_k A[r,k] * B[k,c]:
        //   k_dim     = a_shape[1] == b_shape[0]  (the reduction axis — inner MAC loop)
        //   n_outputs = b_shape[1]                (output neuron count — outer loop)
        let k_dim = a_shape[1];
        let n_outputs = b_shape[1];
        assert_eq!(a_shape[1], b_shape[0], "Matmul inner dimensions must match");

        let a_width = kir_dtype_width(a_dtype);
        let b_width = kir_dtype_width(b_dtype);
        let acc_width = kir_dtype_width(out_dtype);
        let prod_width = a_width + b_width;
        let has_bias = bias_op.is_some();

        // ── M57.1 §3.4: Declare ports + LocalParams inline ──────────────────
        // Per the no-dangling-references invariant, every SignalRef::Port and
        // SignalRef::LocalParam emitted below must be backed by a declaration.
        //
        // x is the only true Port::Input (the stimulus passed by the testbench);
        // W and b are LocalParams (values baked at CLI time by Task 4.1's
        // bake_fixture_into_localparams).
        let layer_idx = next_layer_index(module);
        let x_port_name = format!("x_l{}", layer_idx);
        let w_lp_prefix = format!("W{}", layer_idx);
        let b_lp_prefix = format!("b{}", layer_idx);

        // Declare Port::Input for the layer input. Width is k_dim * a_width —
        // a single packed bus carrying all k input elements. Per-element
        // indexing is resolved by the Verilog emitter (PR 3) using genvars.
        module.add_port(Port::input(&x_port_name, k_dim * a_width));

        // Declare LocalParam for each W element (k_dim × n_outputs values).
        // Value is 0 placeholder; baked to fixture-derived value at CLI time
        // (Task 4.1). Width = b_width (matches the operand of Mul below).
        for o in 0..n_outputs {
            for k in 0..k_dim {
                let elem_name = format!("{}_{}_{}", w_lp_prefix, k, o);
                module.add_local_param(LocalParam::new(elem_name, b_width, 0));
            }
        }
        // M57.1 §3.5 Pin 3: bias LocalParams are declared only when we are
        // folding an adjacent ElementwiseAdd(bias). The defensive no-bias path
        // (zero-seeded accumulator) omits them so the structural HIR audit
        // (snapshot tests) reflects the dispatch decision.
        if has_bias {
            for o in 0..n_outputs {
                let elem_name = format!("{}_{}", b_lp_prefix, o);
                module.add_local_param(LocalParam::new(elem_name, acc_width, 0));
            }
        }

        // ── M57.1 §3.5: bias-as-seed combinational ripple chain ─────────────
        //
        // For each output o ∈ [0, n_outputs):
        //   acc[o][0] = b<i>[o]                  (bias-as-seed; Q3 Pin 3)
        //                  or 0 when has_bias=false  (defensive path)
        //   for k ∈ [0, k_dim):
        //       prod[k] = sext( x[k] * W[k,o] )
        //       acc[o][k+1] = acc[o][k] + prod[k]
        //   result[o] = acc[o][k_dim]
        //
        // HIR structural encoding (plan §3.4 approach (a), wire-array
        // conceptual): the outer GenerateFor body declares a single
        // `bias_seed_wire` (acc[o][0]); the inner GenerateFor body emits the
        // per-iteration MAC body. The Verilog emitter (PR 3) resolves the
        // cross-iteration wiring to a Verilog `wire [W-1:0] acc [K:0];` array.
        //
        // Tap-semantics note (§3.5 + §3 Concern #1): the matmul tap holds the
        // post-bias accumulator now (was pre-bias product sum + separate
        // bias-add). The historical name `tap_l<i>_matmul_out` is preserved
        // for the M57.1 closure PR; semantic rename to `tap_l<i>_mac_out` is
        // a deferred cosmetic cleanup.
        let mut outer_body: Vec<HirNode> = Vec::new();

        // acc[o][0]: bias-seed wire declared once per output. Driver
        // resolution to `b<i>_<o>` (or '0 for the no-bias path) is the
        // Verilog emitter's job (wire-array cross-iteration threading is
        // not representable directly at HIR-level today).
        //
        // We keep the SignalRef referencing the bias LocalParam (or a
        // zero-port placeholder for the defensive path) attached to the
        // ripple chain via the `acc_l<i>_prev` port name on the inner Add —
        // see below. The bias_seed_wire HirNode merely declares the
        // accumulator-array's slot at HIR level so structural audits can
        // verify it exists.
        let bias_seed_wire = Wire::new(acc_width);
        outer_body.push(HirNode::Wire(bias_seed_wire));

        // Inner generate-for body — one iteration of the MAC ripple.
        let mut inner_body: Vec<HirNode> = Vec::new();

        // Mul: x_l<i>[k] * W<i>_<k>_<o> → prod  (a_width × b_width → prod_width)
        let mul = Mul::new(
            SignalRef::port(&x_port_name),
            SignalRef::local_param(&w_lp_prefix),
            a_width,
            b_width,
            prod_width,
        );
        let prod_id = mul.out;
        inner_body.push(HirNode::Mul(mul));

        // SignExtend prod → acc_width if needed (3-node MAC chain when prod < acc)
        let add_input = if prod_width < acc_width {
            let se = SignExtend::new(SignalRef::wire(prod_id), prod_width, acc_width);
            let se_id = se.dst;
            inner_body.push(HirNode::SignExtend(se));
            SignalRef::wire(se_id)
        } else {
            // When prod_width == acc_width, MAC chain is 2-node: Mul → Add directly.
            SignalRef::wire(prod_id)
        };

        // Ripple step: acc[o][k+1] = acc[o][k] + product[k].
        // The first operand references `acc_l<i>_prev` — a layer-scoped
        // accumulator placeholder name. The Verilog emitter resolves this to
        // either the bias-seed wire (k=0) or the previous inner-iteration's
        // accumulator (k>0), realizing the wire-array semantics.
        //
        // Naming changed from M57's universal `__matmul_acc_prev_LEGACY`
        // (single-driver violation) to per-layer `acc_l<i>_prev` so multiple
        // matmul layers can coexist without colliding.
        let acc_prev_name = format!("acc_l{}_prev", layer_idx);
        let add = Add::new(
            SignalRef::port(&acc_prev_name),
            add_input,
            acc_width,
        );
        inner_body.push(HirNode::Add(add));

        outer_body.push(HirNode::GenerateFor(GenerateFor::new(
            0,
            k_dim as i64,
            inner_body,
        )));

        module.add_node(HirNode::GenerateFor(GenerateFor::new(
            0,
            n_outputs as i64,
            outer_body,
        )))?;

        // Note: bias_op is consumed via the peekable iteration in
        // KirToHirPass::lower (i += 2 when present). When bias_op is None,
        // the bias-seed wire is still emitted but the bias LocalParams are
        // omitted; the Verilog emitter drives acc[o][0] from '0 in that path.
        let _ = bias_op;

        Ok(())
    }

    fn lower_elementwise_add(
        &self,
        op: &KirOp,
        module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        use crate::hir::control::GenerateFor;
        use crate::hir::module::HirNode;
        use crate::hir::nodes::{Add, Port};
        use crate::hir::signals::SignalRef;

        let (dtype, shape) = match op {
            KirOp::ElementwiseAdd { dtype, shape, .. } => (dtype.clone(), *shape),
            _ => unreachable!("lower_elementwise_add called with non-ElementwiseAdd op"),
        };

        let width = kir_dtype_width(dtype);
        let n = shape[0];

        // M57.1 §3.4: declare Port::Input for both operands inline so the
        // no-dangling-references invariant holds on standalone ElementwiseAdd.
        // M57.1 §3.5 (Task 3.4) folds matmul+bias pairs into the MAC ripple's
        // accumulator seed, so this function is reached ONLY for non-bias
        // standalone Adds in v1 (the recognizer doesn't emit these on the v1
        // MLP path, but the lowering remains correct if they appear).
        //
        // Width is n * width — single packed bus carrying all n elements. The
        // Verilog emitter (PR 3) resolves per-element indexing via the
        // generate-for genvar.
        let layer_idx = next_port_layer_index(module, "eltadd_a_l");
        let a_name = format!("eltadd_a_l{}", layer_idx);
        let b_name = format!("eltadd_b_l{}", layer_idx);
        module.add_port(Port::input(&a_name, n * width));
        module.add_port(Port::input(&b_name, n * width));

        let body = vec![HirNode::Add(Add::new(
            SignalRef::port(&a_name),
            SignalRef::port(&b_name),
            width,
        ))];

        module.add_node(HirNode::GenerateFor(GenerateFor::new(0, n as i64, body)))?;

        Ok(())
    }

    fn lower_relu(
        &self,
        op: &KirOp,
        module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        use crate::hir::control::GenerateFor;
        use crate::hir::module::HirNode;
        use crate::hir::nodes::{Max0, Port};
        use crate::hir::signals::SignalRef;

        let (dtype, shape) = match op {
            KirOp::Relu { dtype, shape, .. } => (dtype.clone(), *shape),
            _ => unreachable!("lower_relu called with non-Relu op"),
        };

        let width = kir_dtype_width(dtype);
        let n = shape[0];

        // M57.1 §3.4: declare Port::Input for the ReLU input + Port::Output
        // for the layer's result. The output port doubles as the layer tap
        // for parity testing (M57 §2.6.2).
        //
        // The Max0 node's own `out` WireId is the driver of the output port —
        // no separate Wire node is needed (Mul/Add/Max0/SignExtend each carry
        // their own implicit output WireId; only stateful registers and
        // explicitly-named wires use HirNode::Wire). Construct the Max0 first
        // so we can hand its `out` WireId to Port::output as the driver.
        let layer_idx = next_port_layer_index(module, "relu_in_l");
        let in_name = format!("relu_in_l{}", layer_idx);
        let out_name = format!("relu_out_l{}", layer_idx);

        module.add_port(Port::input(&in_name, n * width));

        let max0 = Max0::new(SignalRef::port(&in_name), width);
        let max0_out_id = max0.out;

        let body = vec![HirNode::Max0(max0)];

        module.add_node(HirNode::GenerateFor(GenerateFor::new(0, n as i64, body)))?;

        // Output port driven by the Max0 wire (tap-enabled per M57 §2.6.2).
        module.add_port(Port::output(
            &out_name,
            n * width,
            SignalRef::wire(max0_out_id),
        ));

        Ok(())
    }
}

/// Derive the next 1-based layer index from already-declared LocalParams.
///
/// Scans `module.local_params()` for names matching `W<n>_…` and returns the
/// smallest 1-based index greater than the largest `<n>` seen (or 1 when no
/// matching LocalParams exist).
///
/// This is robust to layers having heterogeneous (k_dim × n_outputs) shapes
/// (e.g. MLP 784→128→10), unlike the alternative of dividing the W count by
/// a fixed per-layer element count. It also tolerates non-W LocalParams
/// (e.g. biases) being interleaved.
///
/// Complexity note: per-call cost is O(P) where P is the number of
/// LocalParams already declared. Called once per matmul layer, total cost
/// across the pass is O(L · P) ≈ O(P²) for L layers with P proportional to
/// layer width × depth. For v1 (≤ a few layers; the MLP fixture has 2),
/// this is practical and avoids carrying mutable counter state on the
/// pass. A v2 with arbitrary layer count can replace this with a
/// `&mut layer_counter` threaded through `KirToHirPass`.
fn next_layer_index(module: &HirModule) -> usize {
    let max_seen = module
        .local_params
        .iter()
        .filter_map(|lp| {
            // Match prefix exactly: "W" then ASCII digits then "_". Rejects
            // e.g. "Weight…" (would parse `eight…` as non-digit and bail).
            let rest = lp.name.strip_prefix('W')?;
            let (digits, tail) = rest
                .char_indices()
                .find(|(_, c)| !c.is_ascii_digit())
                .map(|(i, _)| rest.split_at(i))
                .unwrap_or((rest, ""));
            if !tail.starts_with('_') || digits.is_empty() {
                return None;
            }
            digits.parse::<usize>().ok()
        })
        .max()
        .unwrap_or(0);
    max_seen + 1
}

/// Derive the next 1-based layer index from already-declared ports whose name
/// starts with `prefix` followed by ASCII digits (and nothing more, OR digits
/// followed by an underscore — but in v1 the eltadd/relu port-naming scheme
/// terminates the layer-index at end-of-name, e.g. `eltadd_a_l1`, `relu_in_l2`).
///
/// Counterpart to `next_layer_index` (which scans LocalParams for matmul's
/// `W<n>_` prefix). `lower_elementwise_add` and `lower_relu` declare ports —
/// not LocalParams — so they need port-scanning disambiguation. Each lowering
/// function picks its own anchor prefix (`eltadd_a_l` / `relu_in_l`) which
/// keeps eltadd and relu indices independent (both can be "l1" without
/// colliding).
fn next_port_layer_index(module: &HirModule, prefix: &str) -> usize {
    use crate::hir::nodes::Port;
    let max_seen = module
        .ports
        .iter()
        .filter_map(|p| {
            let name = match p {
                Port::Input { name, .. } | Port::Output { name, .. } => name,
            };
            let rest = name.strip_prefix(prefix)?;
            if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            rest.parse::<usize>().ok()
        })
        .max()
        .unwrap_or(0);
    max_seen + 1
}

/// M57.1 wire-array mini §3.1: scan forward from `start` for a contiguous
/// v1 MLP chain (Matmul → ElementwiseAdd → Relu triples with verified SSA +
/// dimensional handoffs across layers).
///
/// Returns `Some(chain_len)` where `chain_len = layers_matched * 3` if at
/// least one layer matches the pattern. Returns `None` if the very first
/// triple at `start` isn't a complete Matmul/ElementwiseAdd/Relu pattern —
/// the dispatch then falls through to the standalone lowerers.
///
/// Errors with `FpgaLoweringError::UnsupportedV1Shape` if a multi-layer
/// chain is recognized but the dimensional handoff fails
/// (`prev_layer.n_outputs != this_layer.k_dim`). This is intentional fail-
/// loud per spec Pin 1 — silently emitting a mis-wired chain would produce
/// out-of-bounds WireArrayElement reads.
fn try_recognize_v1_mlp_chain(
    ops: &[&KirOp],
    start: usize,
) -> Result<Option<usize>, FpgaLoweringError> {
    let mut i = start;
    let mut prev_relu_out: Option<VarId> = None;
    let mut prev_n_outputs: Option<usize> = None;
    let mut layers_matched: usize = 0;

    while i + 2 < ops.len() {
        let matmul = ops[i];
        let eltadd = ops[i + 1];
        let relu = ops[i + 2];

        // Layer's matmul: extract a, out, a_shape, b_shape.
        let (m_a, m_out, m_a_shape, m_b_shape) = match matmul {
            KirOp::Matmul { a, out, a_shape, b_shape, .. } =>
                (*a, *out, *a_shape, *b_shape),
            _ => break,
        };

        // Inter-layer checks (Pin 1): SSA + dimensional handoff. The chain's
        // first layer has no prev; subsequent layers must connect cleanly.
        if let Some(prev_out) = prev_relu_out {
            if prev_out != m_a {
                break;
            }
            if let Some(prev_no) = prev_n_outputs {
                if prev_no != m_a_shape[1] {
                    return Err(FpgaLoweringError::UnsupportedV1Shape {
                        found: format!(
                            "layer {} k_dim={} != layer {} n_outputs={}",
                            layers_matched + 1,
                            m_a_shape[1],
                            layers_matched,
                            prev_no,
                        ),
                        expected: "prev_layer.n_outputs == this_layer.k_dim",
                    });
                }
            }
        }

        // Layer's ElementwiseAdd: SSA chain m_out -> e_a, capture e_out.
        let (e_a, e_out) = match eltadd {
            KirOp::ElementwiseAdd { a, out, .. } => (*a, *out),
            _ => break,
        };
        if e_a != m_out {
            break;
        }

        // Layer's Relu: SSA chain e_out -> r_a, capture r_out.
        let (r_a, r_out) = match relu {
            KirOp::Relu { a, out, .. } => (*a, *out),
            _ => break,
        };
        if r_a != e_out {
            break;
        }

        prev_relu_out = Some(r_out);
        prev_n_outputs = Some(m_b_shape[1]);
        layers_matched += 1;
        i += 3;
    }

    if layers_matched >= 1 {
        Ok(Some(layers_matched * 3))
    } else {
        Ok(None)
    }
}

/// M57.1 wire-array mini §3.2: emit a chain of v1 MLP layers as a unit.
/// Each layer except the first reads from the previous layer's `relu_l<i-1>`
/// WireArray directly (no inter-layer Port::Input). The final layer carries
/// the `Port::Output("out")` driven by `WireArrayConcat(relu_l<N>)`.
fn lower_v1_mlp_layer_chain(
    ops: &[&KirOp],
    module: &mut HirModule,
) -> Result<(), FpgaLoweringError> {
    assert_eq!(ops.len() % 3, 0, "v1 MLP chain ops must come in triples");
    let n_layers = ops.len() / 3;
    let mut prev_relu_array: Option<String> = None;
    for layer_idx in 1..=n_layers {
        let base = (layer_idx - 1) * 3;
        let matmul = ops[base];
        let eltadd = ops[base + 1];
        let relu = ops[base + 2];
        let is_final = layer_idx == n_layers;
        let prev_ref = prev_relu_array.as_deref();
        lower_v1_mlp_single_layer(
            matmul,
            Some(eltadd),
            relu,
            prev_ref,
            layer_idx,
            is_final,
            module,
        )?;
        prev_relu_array = Some(format!("relu_l{}", layer_idx));
    }
    Ok(())
}

/// M57.1 wire-array mini §3.2: fused emitter for a single v1 MLP layer.
/// Emits the full HIR shape for one layer of the v1 MLP chain:
///   - Layer 1 only: `Port::Input(x_l1)` + per-element fan-out into a
///     `x_l1_a` WireArray cell via `PortBitSlice`. Layers > 1: read from
///     `prev_relu_array_name` directly (no Port::Input).
///   - `LocalParamArray("W<i>", [k_dim, n_outputs])` (Task W5; replaces
///     per-element `W<i>_<k>_<o>` LocalParams) and
///     `LocalParamArray("b<i>", [n_outputs])` when bias is present.
///   - `WireArray(acc_l<i>, [n_outputs, k_dim+1])` +
///     `WireArray(relu_l<i>, [n_outputs])` declarations.
///   - n_outputs bias-seed assigns `acc[o][0] = b<i>[o]` (or zero LocalParam
///     when has_bias=false; defensive path).
///   - Outer `GenerateFor` over `_gv_o` containing inner `GenerateFor` over
///     `_gv_k`: Mul(x_l<i>_a[_gv_k] * W<i>[_gv_k][_gv_o]) → optional
///     SignExtend → Add(acc[_gv_o][_gv_k] + sext_prod) → AssignWireArrayElement
///     writing into `acc[_gv_o][_gv_k + 1]`.
///   - Final-column relu GenerateFor over `_gv_o`: max0(acc[_gv_o][k_dim]) →
///     relu_l<i>[_gv_o].
///   - Tap ports `tap_l<i>_matmul_out` (post-bias accumulator, fixed col
///     `k_dim`) and `tap_l<i>_relu_out` when `module.test_taps`.
///   - When `is_final_layer`: `Port::Output("out", ...)` driven by
///     `WireArrayConcat(relu_l<i>, n_outputs, None)`.
///
/// Inputs:
/// - `matmul` / `eltadd` / `relu`: the layer's three KIR ops. `eltadd=None`
///   triggers the zero-seed defensive path.
/// - `prev_relu_array_name`: when `Some`, this is layer i > 1 — reads from
///   the previous layer's relu WireArray instead of declaring `Port::Input`.
/// - `layer_idx`: 1-based layer index for naming.
/// - `is_final_layer`: when true, declares `Port::Output("out")`.
/// - `module`: HirModule to mutate.
pub(crate) fn lower_v1_mlp_single_layer(
    matmul: &KirOp,
    eltadd: Option<&KirOp>,
    relu: &KirOp,
    prev_relu_array_name: Option<&str>,
    layer_idx: usize,
    is_final_layer: bool,
    module: &mut HirModule,
) -> Result<(), FpgaLoweringError> {
    use crate::hir::control::GenerateFor as Gf;
    use crate::hir::ids::GenvarId;
    use crate::hir::module::HirNode;
    use crate::hir::nodes::{
        Add, AssignWireArrayElement, IndexExpr, LocalParam, LocalParamArray,
        Max0, Mul, Port, SignExtend, WireArray,
    };
    use crate::hir::signals::SignalRef;

    // ── Extract Matmul shapes + dtypes ─────────────────────────────────
    let (a_dtype, b_dtype, out_dtype, a_shape, b_shape) = match matmul {
        KirOp::Matmul {
            a_dtype,
            b_dtype,
            out_dtype,
            a_shape,
            b_shape,
            ..
        } => (
            a_dtype.clone(),
            b_dtype.clone(),
            out_dtype.clone(),
            *a_shape,
            *b_shape,
        ),
        _ => unreachable!("lower_v1_mlp_single_layer called with non-Matmul matmul"),
    };

    // Type-check relu — only ReLU is supported as the layer's activation.
    if !matches!(relu, KirOp::Relu { .. }) {
        unreachable!("lower_v1_mlp_single_layer called with non-Relu relu");
    }

    let k_dim = a_shape[1];
    let n_outputs = b_shape[1];
    assert_eq!(
        a_shape[1], b_shape[0],
        "Matmul inner dimensions must match"
    );

    let a_width = kir_dtype_width(a_dtype);
    let b_width = kir_dtype_width(b_dtype);
    let acc_width = kir_dtype_width(out_dtype);
    let prod_width = a_width + b_width;

    let has_bias = eltadd.is_some();
    let x_port_name = format!("x_l{}", layer_idx);
    let w_arr_name = format!("W{}", layer_idx);
    let b_arr_name = format!("b{}", layer_idx);
    let acc_array_name = format!("acc_l{}", layer_idx);
    let relu_array_name = format!("relu_l{}", layer_idx);
    let test_taps = module.test_taps;

    // ── Layer 1: declare Port::Input(x_l1) + x_l1_a WireArray fan-out.
    //    Layers 2+: skip — read from prev_relu_array_name directly. ────────
    let x_array_name: String = match prev_relu_array_name {
        Some(prev) => prev.to_string(),
        None => {
            // Layer 1: external stimulus port + per-element fan-out into a
            // `x_l<i>_a` WireArray (unified matmul-body read path — see §3.2).
            module.add_port(Port::input(&x_port_name, k_dim * a_width));
            let name = format!("x_l{}_a", layer_idx);
            module.add_node(HirNode::WireArray(WireArray {
                name: name.clone(),
                dims: vec![k_dim],
                width: a_width,
            }))?;
            for k in 0..k_dim {
                module.add_node(HirNode::AssignWireArrayElement(AssignWireArrayElement {
                    array_name: name.clone(),
                    indices: vec![IndexExpr::Literal(k)],
                    src: SignalRef::port_bit_slice(
                        x_port_name.clone(),
                        k * a_width,
                        a_width,
                    ),
                }))?;
            }
            name
        }
    };

    // ── LocalParamArrays: W<i>[k_dim][n_outputs] + b<i>[n_outputs] ────────
    // Values are zero-initialized; baked at CLI time (Task W6) by the
    // fixture loader walking `module.local_param_arrays_mut()`.
    module.add_local_param_array(LocalParamArray {
        name: w_arr_name.clone(),
        dims: vec![k_dim, n_outputs],
        width: b_width,
        values: vec![0; k_dim * n_outputs],
    });
    if has_bias {
        module.add_local_param_array(LocalParamArray {
            name: b_arr_name.clone(),
            dims: vec![n_outputs],
            width: acc_width,
            values: vec![0; n_outputs],
        });
    }

    // ── WireArrays: acc_l<i> [n_outputs][k_dim+1] + relu_l<i> [n_outputs] ─
    module.add_node(HirNode::WireArray(WireArray {
        name: acc_array_name.clone(),
        dims: vec![n_outputs, k_dim + 1],
        width: acc_width,
    }))?;
    module.add_node(HirNode::WireArray(WireArray {
        name: relu_array_name.clone(),
        dims: vec![n_outputs],
        width: acc_width,
    }))?;

    // ── Bias-seed assigns: acc[o][0] = b<i>[o] (or zero LocalParam) ──────
    //
    // Per-element assigns at module top — LocalParam access by genvar isn't
    // directly supported in standard Verilog generate blocks. (The W<i>
    // matrix uses LocalParamArray + IndexedLocalParam inside the ripple
    // body because Verilog DOES allow indexing into a `localparam ARRAY`
    // by genvar, just not into a family of scalars-by-name.)
    //
    // Defensive no-bias path: declare a single `zero_w<acc_width>` scalar
    // LocalParam (idempotent across multiple no-bias calls within a module)
    // and reference it as the seed source.
    let zero_lp_name = format!("zero_w{}", acc_width);
    if !has_bias
        && !module
            .local_params()
            .iter()
            .any(|lp| lp.name == zero_lp_name)
    {
        module.add_local_param(LocalParam::new(&zero_lp_name, acc_width, 0));
    }
    for o in 0..n_outputs {
        let src = if has_bias {
            // Per-element b<i>[o] read via IndexedLocalParam (literal index).
            SignalRef::indexed_local_param(
                b_arr_name.clone(),
                vec![IndexExpr::Literal(o)],
            )
        } else {
            SignalRef::local_param(&zero_lp_name)
        };
        module.add_node(HirNode::AssignWireArrayElement(AssignWireArrayElement {
            array_name: acc_array_name.clone(),
            indices: vec![IndexExpr::Literal(o), IndexExpr::Literal(0)],
            src,
        }))?;
    }

    // ── Outer GenerateFor over _gv_o, inner GenerateFor over _gv_k ──────
    let gv_o = GenvarId::fresh();
    let gv_o_name = format!("_gv{}", gv_o.0);
    let gv_k = GenvarId::fresh();
    let gv_k_name = format!("_gv{}", gv_k.0);

    // Inner-body construction: Mul → optional SignExtend → Add →
    // AssignWireArrayElement (writing into acc[_gv_o][_gv_k + 1]).
    let mut inner_body: Vec<HirNode> = Vec::new();

    // product = x[_gv_k] * W<i>[_gv_k][_gv_o]
    let prod = Mul::new(
        SignalRef::wire_array_element(
            x_array_name.clone(),
            vec![IndexExpr::Genvar(gv_k_name.clone())],
        ),
        SignalRef::indexed_local_param(
            w_arr_name.clone(),
            vec![
                IndexExpr::Genvar(gv_k_name.clone()),
                IndexExpr::Genvar(gv_o_name.clone()),
            ],
        ),
        a_width,
        b_width,
        prod_width,
    );
    let prod_id = prod.out;
    inner_body.push(HirNode::Mul(prod));

    // Optional SignExtend if prod_width < acc_width.
    let extended_src = if prod_width < acc_width {
        let se = SignExtend::new(SignalRef::wire(prod_id), prod_width, acc_width);
        let se_id = se.dst;
        inner_body.push(HirNode::SignExtend(se));
        SignalRef::wire(se_id)
    } else {
        SignalRef::wire(prod_id)
    };

    // acc[_gv_o][_gv_k + 1] = acc[_gv_o][_gv_k] + extended
    // Model: fresh Wire(sum) + Add(...) targeting sum + AssignWireArrayElement.
    let add = Add::new(
        SignalRef::wire_array_element(
            acc_array_name.clone(),
            vec![
                IndexExpr::Genvar(gv_o_name.clone()),
                IndexExpr::Genvar(gv_k_name.clone()),
            ],
        ),
        extended_src,
        acc_width,
    );
    let sum_id = add.out;
    inner_body.push(HirNode::Add(add));
    inner_body.push(HirNode::AssignWireArrayElement(AssignWireArrayElement {
        array_name: acc_array_name.clone(),
        indices: vec![
            IndexExpr::Genvar(gv_o_name.clone()),
            IndexExpr::GenvarPlus(gv_k_name.clone(), 1),
        ],
        src: SignalRef::wire(sum_id),
    }));

    let inner_gf = Gf {
        var: gv_k,
        lo: 0,
        hi: k_dim as i64,
        body: inner_body,
    };
    let outer_body = vec![HirNode::GenerateFor(inner_gf)];
    module.add_node(HirNode::GenerateFor(Gf {
        var: gv_o,
        lo: 0,
        hi: n_outputs as i64,
        body: outer_body,
    }))?;

    // ── Final-column relu GenerateFor: relu[_gv_o] = max0(acc[_gv_o][k_dim]) ─
    let gv_relu = GenvarId::fresh();
    let gv_relu_name = format!("_gv{}", gv_relu.0);
    let relu_max0 = Max0::new(
        SignalRef::wire_array_element(
            acc_array_name.clone(),
            vec![
                IndexExpr::Genvar(gv_relu_name.clone()),
                IndexExpr::Literal(k_dim),
            ],
        ),
        acc_width,
    );
    let relu_max0_id = relu_max0.out;
    let relu_body = vec![
        HirNode::Max0(relu_max0),
        HirNode::AssignWireArrayElement(AssignWireArrayElement {
            array_name: relu_array_name.clone(),
            indices: vec![IndexExpr::Genvar(gv_relu_name.clone())],
            src: SignalRef::wire(relu_max0_id),
        }),
    ];
    module.add_node(HirNode::GenerateFor(Gf {
        var: gv_relu,
        lo: 0,
        hi: n_outputs as i64,
        body: relu_body,
    }))?;

    // ── Tap ports (when test_taps) ──────────────────────────────────────
    // `tap_l<i>_matmul_out`: post-bias accumulator (concat of acc[o][k_dim]
    // over o ∈ [0, n_outputs)).
    // `tap_l<i>_relu_out`: layer output (concat of relu_l<i>[o]).
    if test_taps {
        module.add_port(Port::output(
            format!("tap_l{}_matmul_out", layer_idx),
            n_outputs * acc_width,
            SignalRef::wire_array_concat(
                acc_array_name.clone(),
                n_outputs,
                Some(k_dim),
            ),
        ));
        module.add_port(Port::output(
            format!("tap_l{}_relu_out", layer_idx),
            n_outputs * acc_width,
            SignalRef::wire_array_concat(
                relu_array_name.clone(),
                n_outputs,
                None,
            ),
        ));
    }

    // ── Final-layer output port ────────────────────────────────────────
    if is_final_layer {
        module.add_port(Port::output(
            "out",
            n_outputs * acc_width,
            SignalRef::wire_array_concat(relu_array_name, n_outputs, None),
        ));
    }

    // Silence unused-variable warning when neither relu's max0 nor sum_id
    // are referenced under specific cfg paths.
    let _ = relu;

    Ok(())
}

/// Map KirType to bit-width for FPGA HIR construction.
/// Panics on types not supported in v1 FPGA target.
pub(crate) fn kir_dtype_width(dtype: KirType) -> usize {
    match dtype {
        KirType::I8 => 8,
        KirType::I16 => 16,
        KirType::I32 => 32,
        KirType::I64 => 64,
        other => panic!("KirType {:?} unsupported in v1 FPGA target", other),
    }
}

/// Maps every KirOp variant to a stable string for the error message.
/// Tedious but necessary; KirOp doesn't derive a stable name function.
pub(crate) fn kir_op_kind_name(op: &KirOp) -> &'static str {
    match op {
        KirOp::Add(..) => "Add",
        KirOp::Sub(..) => "Sub",
        KirOp::Mul(..) => "Mul",
        KirOp::Div(..) => "Div",
        KirOp::Fma(..) => "Fma",
        KirOp::Neg(..) => "Neg",
        KirOp::Abs(..) => "Abs",
        KirOp::Sqrt(..) => "Sqrt",
        KirOp::Exp(..) => "Exp",
        KirOp::Log(..) => "Log",
        KirOp::Sin(..) => "Sin",
        KirOp::Cos(..) => "Cos",
        KirOp::Tanh(..) => "Tanh",
        KirOp::Pow(..) => "Pow",
        KirOp::Cast(..) => "Cast",
        KirOp::Load(..) => "Load",
        KirOp::Store(..) => "Store",
        KirOp::AtomicAdd(..) => "AtomicAdd",
        KirOp::ThreadId(..) => "ThreadId",
        KirOp::BlockIdx(..) => "BlockIdx",
        KirOp::BlockDim(..) => "BlockDim",
        KirOp::GridDim(..) => "GridDim",
        KirOp::GlobalId(..) => "GlobalId",
        KirOp::Barrier => "Barrier",
        KirOp::WarpShuffle(..) => "WarpShuffle",
        KirOp::Cmp(..) => "Cmp",
        KirOp::Select(..) => "Select",
        KirOp::Const(..) => "Const",
        KirOp::PtrOffset(..) => "PtrOffset",
        KirOp::SharedMemFence => "SharedMemFence",
        KirOp::Matmul { .. } => "Matmul",
        KirOp::ElementwiseAdd { .. } => "ElementwiseAdd",
        KirOp::Relu { .. } => "Relu",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::module::HirNode;
    use crate::hir::nodes::{Port, WireArray};

    /// M57.1 wire-array mini §3.2 / Task W5: smallest single-layer case
    /// (layer 1, with bias). Verifies the fused emitter shape:
    /// Port::Input + x_l1_a fan-out + LocalParamArrays + WireArrays.
    #[test]
    fn lower_v1_mlp_single_layer_w5_emits_full_shape() {
        // Smallest possible MLP layer: 2 inputs → 3 outputs, i8 × i8 → i32.
        let matmul = KirOp::Matmul {
            a: 0, b: 1, out: 2,
            a_dtype: KirType::I8, b_dtype: KirType::I8, out_dtype: KirType::I32,
            a_shape: [1, 2], b_shape: [2, 3],
        };
        let eltadd = KirOp::ElementwiseAdd {
            a: 2, b: 3, out: 4,
            dtype: KirType::I32, shape: [3],
        };
        let relu = KirOp::Relu {
            a: 4, out: 5,
            dtype: KirType::I32, shape: [3],
        };

        let mut module = HirModule::new("test");
        lower_v1_mlp_single_layer(
            &matmul, Some(&eltadd), &relu,
            None, 1, true, &mut module,
        ).expect("single-layer scaffold lowers");

        // Verify the WireArrays were declared. Layer 1 with full ripple
        // body declares: x_l1_a (k_dim=2), acc_l1 (n_outputs × k_dim+1),
        // and relu_l1 (n_outputs).
        let wire_arrays: Vec<&WireArray> = module.nodes().iter().filter_map(|n| match n {
            HirNode::WireArray(wa) => Some(wa),
            _ => None,
        }).collect();
        assert_eq!(wire_arrays.len(), 3, "x_l1_a + acc_l1 + relu_l1");
        let names: Vec<&str> = wire_arrays.iter().map(|wa| wa.name.as_str()).collect();
        assert!(names.contains(&"x_l1_a"));
        assert!(names.contains(&"acc_l1"));
        assert!(names.contains(&"relu_l1"));

        // Verify Port::Input(x_l1) declared (layer 1, prev_relu_array_name is None).
        let inputs: Vec<&str> = module.ports.iter().filter_map(|p| match p {
            Port::Input { name, .. } => Some(name.as_str()),
            _ => None,
        }).collect();
        assert!(inputs.contains(&"x_l1"), "x_l1 Port::Input must be declared");

        // LocalParamArrays: W1 [2,3] + b1 [3].
        assert_eq!(module.local_param_arrays().len(), 2, "W1 + b1");
        let w1 = &module.local_param_arrays()[0];
        assert_eq!(w1.name, "W1");
        assert_eq!(w1.dims, vec![2, 3]);
        assert_eq!(w1.width, 8);
        let b1 = &module.local_param_arrays()[1];
        assert_eq!(b1.name, "b1");
        assert_eq!(b1.dims, vec![3]);
        assert_eq!(b1.width, 32);
    }

    /// M57.1 wire-array mini §3.2 / Task W5: layer 2+ must NOT declare a
    /// Port::Input — its `x` is read from the previous layer's relu
    /// WireArray (`prev_relu_array_name`).
    #[test]
    fn lower_v1_mlp_single_layer_w5_layer_2_skips_port_input() {
        let matmul = KirOp::Matmul {
            a: 0, b: 1, out: 2,
            a_dtype: KirType::I32, b_dtype: KirType::I8, out_dtype: KirType::I64,
            a_shape: [1, 3], b_shape: [3, 2],
        };
        let eltadd = KirOp::ElementwiseAdd {
            a: 2, b: 3, out: 4,
            dtype: KirType::I64, shape: [2],
        };
        let relu = KirOp::Relu {
            a: 4, out: 5,
            dtype: KirType::I64, shape: [2],
        };

        let mut module = HirModule::new("test_l2");
        lower_v1_mlp_single_layer(
            &matmul, Some(&eltadd), &relu,
            Some("relu_l1"), 2, true, &mut module,
        ).expect("layer 2 lowers");

        // Layer 2 must NOT declare any Port::Input (taps + final out are
        // Port::Output; this test pre-asserts no Port::Input). Also: with
        // is_final_layer=true and test_taps=true, expect 3 Port::Output:
        // tap_l2_matmul_out, tap_l2_relu_out, out.
        let inputs: Vec<&str> = module.ports.iter().filter_map(|p| match p {
            Port::Input { name, .. } => Some(name.as_str()),
            _ => None,
        }).collect();
        assert!(
            inputs.is_empty(),
            "layer 2 must not declare Port::Input (reads from prev_relu_array)"
        );

        // Layer 2 declares acc_l2 + relu_l2 only (no x_l2_a — it reads
        // relu_l1 directly).
        let wire_arrays: Vec<&WireArray> = module.nodes().iter().filter_map(|n| match n {
            HirNode::WireArray(wa) => Some(wa),
            _ => None,
        }).collect();
        assert_eq!(wire_arrays.len(), 2, "acc_l2 + relu_l2 only (no x_l2_a)");
        assert_eq!(wire_arrays[0].name, "acc_l2");
        assert_eq!(wire_arrays[1].name, "relu_l2");
    }

    /// M57.1 wire-array mini §3.2 / Task W5: with-bias layer emits
    /// `n_outputs` per-element bias-seed `AssignWireArrayElement` nodes
    /// indexed via `IndexedLocalParam(b<i>[o])`.
    #[test]
    fn lower_v1_mlp_single_layer_w5_emits_bias_seed_assigns() {
        use crate::hir::nodes::{AssignWireArrayElement, IndexExpr};
        use crate::hir::signals::SignalRef;

        let matmul = KirOp::Matmul {
            a: 0, b: 1, out: 2,
            a_dtype: KirType::I8, b_dtype: KirType::I8, out_dtype: KirType::I32,
            a_shape: [1, 2], b_shape: [2, 3],
        };
        let eltadd = KirOp::ElementwiseAdd {
            a: 2, b: 3, out: 4,
            dtype: KirType::I32, shape: [3],
        };
        let relu = KirOp::Relu {
            a: 4, out: 5,
            dtype: KirType::I32, shape: [3],
        };

        let mut module = HirModule::new("test");
        lower_v1_mlp_single_layer(
            &matmul, Some(&eltadd), &relu,
            None, 1, true, &mut module,
        ).expect("scaffold lowers");

        // 3 bias-seed AssignWireArrayElement nodes — each at acc_l1[o][0]
        // reading from b1[o] via IndexedLocalParam.
        let bias_assigns: Vec<&AssignWireArrayElement> = module.nodes().iter().filter_map(|n| match n {
            HirNode::AssignWireArrayElement(a)
                if a.array_name == "acc_l1"
                    && a.indices.len() == 2
                    && matches!(&a.indices[1], IndexExpr::Literal(0)) =>
            {
                Some(a)
            }
            _ => None,
        }).collect();
        assert_eq!(bias_assigns.len(), 3, "3 bias-seed assigns");
        for (idx, a) in bias_assigns.iter().enumerate() {
            match &a.indices[0] {
                IndexExpr::Literal(n) => assert_eq!(*n, idx),
                _ => panic!("expected Literal(o) at indices[0]"),
            }
            match &a.src {
                SignalRef::IndexedLocalParam { array_name, indices } => {
                    assert_eq!(array_name, "b1");
                    assert_eq!(indices.len(), 1);
                    match &indices[0] {
                        IndexExpr::Literal(n) => assert_eq!(*n, idx),
                        _ => panic!("expected Literal index into b1"),
                    }
                }
                _ => panic!("expected IndexedLocalParam src, got {:?}", a.src),
            }
        }

        // With bias present, no zero_w<acc_width> scalar LocalParam.
        assert!(
            !module.local_params().iter().any(|lp| lp.name == "zero_w32"),
            "with-bias layer must NOT declare zero_w32 LocalParam"
        );
    }

    /// M57.1 wire-array mini §3.2 / Task W5: no-bias defensive path
    /// declares a single `zero_w<acc_width>` LocalParam and uses it as the
    /// accumulator seed.
    #[test]
    fn lower_v1_mlp_single_layer_w5_no_bias_uses_zero_localparam() {
        use crate::hir::nodes::{AssignWireArrayElement, IndexExpr};
        use crate::hir::signals::SignalRef;

        let matmul = KirOp::Matmul {
            a: 0, b: 1, out: 2,
            a_dtype: KirType::I8, b_dtype: KirType::I8, out_dtype: KirType::I32,
            a_shape: [1, 2], b_shape: [2, 3],
        };
        let relu = KirOp::Relu {
            a: 2, out: 3,
            dtype: KirType::I32, shape: [3],
        };

        let mut module = HirModule::new("test_no_bias");
        lower_v1_mlp_single_layer(
            &matmul, None, &relu,
            None, 1, true, &mut module,
        ).expect("scaffold lowers no-bias");

        let zero_count = module
            .local_params()
            .iter()
            .filter(|lp| lp.name == "zero_w32")
            .count();
        assert_eq!(zero_count, 1, "zero_w32 LocalParam must be declared once");

        let bias_assigns: Vec<&AssignWireArrayElement> = module.nodes().iter().filter_map(|n| match n {
            HirNode::AssignWireArrayElement(a)
                if a.array_name == "acc_l1"
                    && a.indices.len() == 2
                    && matches!(&a.indices[1], IndexExpr::Literal(0)) =>
            {
                Some(a)
            }
            _ => None,
        }).collect();
        assert_eq!(bias_assigns.len(), 3);
        for a in bias_assigns.iter() {
            match &a.src {
                SignalRef::LocalParam(name) => assert_eq!(name, "zero_w32"),
                _ => panic!("expected zero_w32 LocalParam, got {:?}", a.src),
            }
        }

        // No-bias still emits W1 LocalParamArray but no b1.
        assert_eq!(module.local_param_arrays().len(), 1, "W1 only (no b1)");
        assert_eq!(module.local_param_arrays()[0].name, "W1");
    }
}
