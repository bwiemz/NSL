//! KIR → HIR pass per M57 v1 spec §2.3 (load-bearing pass).
//! Three concerns: op recognition + dispatch; HIR construction; dtype propagation.

use crate::fpga_error::{FpgaLoweringError, V1_SUPPORTED_OPS};
use crate::hir::module::HirModule;
use crate::kernel_ir::{KirOp, KirType, KernelIR};

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
        let mut module = HirModule::new(module_name);
        module.test_taps = self.test_taps;

        // M57.1 §3.5: peekable layer-aware iteration. Index-based loop so that
        // when we see a Matmul, we can peek at the next op for an
        // ElementwiseAdd(bias) and fuse the pair for the bias-as-seed fold
        // (Task 3.4 fills in the bias-fold logic; Task 3.1 is no-op delegation).
        let ops: Vec<&KirOp> = kir.ops().collect();
        let mut i = 0;
        while i < ops.len() {
            match ops[i] {
                KirOp::Matmul { out: matmul_out, .. } => {
                    // Peek at the adjacent op. The bias-fold (Task 3.4) requires
                    // the ElementwiseAdd's `a` to reference this Matmul's `out`
                    // (i.e., bias is added to the matmul result, not the other
                    // way around). The v1 MLP recognizer emits ops in that order.
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

/// M57.1 wire-array mini §3.2: fused emitter for a single v1 MLP layer.
/// Emits acc_l<i> + relu_l<i> WireArrays at module scope, generate-for
/// bodies driving bias-as-seed ripple + final-column ReLU.
///
/// **W3 scope (this task)**: declarations only — `Port::Input(x_l<i>)`,
/// `LocalParam`s (`W<i>_<k>_<o>`, `b<i>_<o>`), `HirNode::WireArray(acc_l<i>)`,
/// `HirNode::WireArray(relu_l<i>)`. The generate-for ripple body,
/// final-column relu, tap outputs, and `is_final_layer` output port all
/// land in subsequent W-tasks (W4 = AssignWireArrayElement body; W5 =
/// full multi-layer dispatch).
///
/// Inputs:
/// - `matmul`: the `KirOp::Matmul` for this layer.
/// - `eltadd`: optional `KirOp::ElementwiseAdd` (bias). When `None`, no
///   bias LocalParams are emitted; the accumulator will seed from zero in
///   W4 (defensive path not exercised by v1 MLP).
/// - `relu`: the `KirOp::Relu` for this layer (used in W4/W5 for the
///   final-column max0; in W3 only consumed for type-checking).
/// - `prev_relu_array_name`: when `Some`, this is layer i > 1 — reads
///   from the previous layer's relu WireArray instead of declaring
///   `Port::Input`. When `None`, declares `Port::Input("x_l<i>", …)`.
/// - `layer_idx`: 1-based layer index for naming.
/// - `is_final_layer`: when true, future tasks will declare
///   `Port::Output("out")`. In W3 it is stored only for the signature
///   contract.
/// - `module`: HirModule to mutate.
//
// W3 lands the scaffold; production dispatch from `KirToHirPass::lower` is
// wired in W5 (multi-layer dispatch). Allow dead_code until then so the
// scaffold compiles clippy-clean.
#[allow(dead_code)]
pub(crate) fn lower_v1_mlp_single_layer(
    matmul: &KirOp,
    eltadd: Option<&KirOp>,
    relu: &KirOp,
    prev_relu_array_name: Option<&str>,
    layer_idx: usize,
    is_final_layer: bool,
    module: &mut HirModule,
) -> Result<(), FpgaLoweringError> {
    use crate::hir::module::HirNode;
    use crate::hir::nodes::{
        AssignWireArrayElement, IndexExpr, LocalParam, Port, WireArray,
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

    // Type-check relu (W3 scope: just confirm it is a Relu op; the Max0
    // body lands in W4).
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

    let has_bias = eltadd.is_some();
    let x_port_name = format!("x_l{}", layer_idx);
    let w_lp_prefix = format!("W{}", layer_idx);
    let b_lp_prefix = format!("b{}", layer_idx);
    let acc_array_name = format!("acc_l{}", layer_idx);
    let relu_array_name = format!("relu_l{}", layer_idx);

    // ── Layer 1: declare Port::Input(x_l1). Layers 2+: skip (read from
    //    prev_relu_array_name in W5). ────────────────────────────────────
    if prev_relu_array_name.is_none() {
        module.add_port(Port::input(&x_port_name, k_dim * a_width));
    }

    // ── LocalParams: W<i>_<k>_<o> ──────────────────────────────────────
    for o in 0..n_outputs {
        for k in 0..k_dim {
            module.add_local_param(LocalParam::new(
                format!("{}_{}_{}", w_lp_prefix, k, o),
                b_width,
                0,
            ));
        }
    }

    // ── LocalParams: b<i>_<o> (only when bias is present) ─────────────
    if has_bias {
        for o in 0..n_outputs {
            module.add_local_param(LocalParam::new(
                format!("{}_{}", b_lp_prefix, o),
                acc_width,
                0,
            ));
        }
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

    // ── Bias-seed assigns: acc[o][0] = b<i>_<o> (or zero LocalParam) ──────
    //
    // M57.1 wire-array mini §3.2 + §4: per-element bias-seed assigns emitted
    // at module top-level (NOT inside the GenerateFor body, because LocalParam
    // access by genvar isn't directly supported in standard Verilog). W5's
    // ripple body uses Genvar-indexed `acc[o][k+1]` writes via a separate
    // LocalParamArray + IndexedLocalParam mechanism.
    //
    // No-bias defensive path: declare a single `zero_w<acc_width>` LocalParam
    // and use it as the seed. Idempotent — only declared once even if the
    // function is called for multiple layers without bias.
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
            SignalRef::local_param(format!("{}_{}", b_lp_prefix, o))
        } else {
            SignalRef::local_param(&zero_lp_name)
        };
        module.add_node(HirNode::AssignWireArrayElement(AssignWireArrayElement {
            array_name: acc_array_name.clone(),
            indices: vec![IndexExpr::Literal(o), IndexExpr::Literal(0)],
            src,
        }))?;
    }

    // W4 scope: bias-seed assigns now emitted. Generate-for ripple body
    // + relu final column + tap outputs + final-layer Port::Output land in W5.
    let _ = is_final_layer;
    let _ = prev_relu_array_name;
    let _ = relu_array_name;

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

    /// M57.1 wire-array mini §3.2 / Task W3: smallest single-layer case
    /// (layer 1, with bias). Verifies the scaffold emits the expected
    /// Port::Input + LocalParams + WireArray declarations.
    #[test]
    fn lower_v1_mlp_single_layer_w3_scaffold_emits_decls() {
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

        // Verify the two WireArrays were declared.
        let wire_arrays: Vec<&WireArray> = module.nodes().iter().filter_map(|n| match n {
            HirNode::WireArray(wa) => Some(wa),
            _ => None,
        }).collect();
        assert_eq!(wire_arrays.len(), 2, "should declare acc_l1 + relu_l1");
        assert_eq!(wire_arrays[0].name, "acc_l1");
        assert_eq!(wire_arrays[0].dims, vec![3, 3]); // n_outputs=3, k_dim+1=3
        assert_eq!(wire_arrays[0].width, 32);
        assert_eq!(wire_arrays[1].name, "relu_l1");
        assert_eq!(wire_arrays[1].dims, vec![3]);
        assert_eq!(wire_arrays[1].width, 32);

        // Verify Port::Input(x_l1) declared (layer 1, prev_relu_array_name is None).
        let inputs: Vec<&str> = module.ports.iter().filter_map(|p| match p {
            Port::Input { name, .. } => Some(name.as_str()),
            _ => None,
        }).collect();
        assert!(inputs.contains(&"x_l1"), "x_l1 Port::Input must be declared");

        // LocalParams: 2*3 W + 3 b = 9
        assert_eq!(module.local_params().len(), 9, "expected 6 W + 3 b LocalParams");
    }

    /// M57.1 wire-array mini §3.2 / Task W3: layer 2+ must NOT declare a
    /// Port::Input — its `x` is read from the previous layer's relu
    /// WireArray (`prev_relu_array_name`).
    #[test]
    fn lower_v1_mlp_single_layer_w3_layer_2_skips_port_input() {
        // Layer 2: prev_relu_array_name is Some — no Port::Input declared.
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
        ).expect("single-layer scaffold lowers");

        // Layer 2 must NOT declare any Port::Input.
        let inputs: Vec<&str> = module.ports.iter().filter_map(|p| match p {
            Port::Input { name, .. } => Some(name.as_str()),
            _ => None,
        }).collect();
        assert!(
            inputs.is_empty(),
            "layer 2 must not declare Port::Input (reads from prev_relu_array)"
        );

        // But layer 2 still declares its own LocalParams + WireArrays.
        // 3*2 W + 2 b = 8 LocalParams.
        assert_eq!(module.local_params().len(), 3 * 2 + 2, "3*2 W + 2 b LocalParams");
        let wire_arrays: Vec<&WireArray> = module.nodes().iter().filter_map(|n| match n {
            HirNode::WireArray(wa) => Some(wa),
            _ => None,
        }).collect();
        assert_eq!(wire_arrays.len(), 2);
        assert_eq!(wire_arrays[0].name, "acc_l2");
        assert_eq!(wire_arrays[1].name, "relu_l2");
    }

    /// M57.1 wire-array mini §3.2 + §4 / Task W4: with-bias layer must emit
    /// `n_outputs` per-element bias-seed `AssignWireArrayElement` nodes that
    /// drive `acc_l<i>[o][0]` from the matching `b<i>_<o>` LocalParam.
    #[test]
    fn lower_v1_mlp_single_layer_w4_emits_bias_seed_assigns() {
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

        // n_outputs (3) per-element bias-seed AssignWireArrayElement nodes —
        // each at acc_l1[o][0] reading from b1_<o> LocalParam.
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
        assert_eq!(
            bias_assigns.len(),
            3,
            "should emit 3 bias-seed assigns (one per output)"
        );

        // Each bias_assign reads from b1_<o>; indices[0] is Literal(o).
        for (idx, a) in bias_assigns.iter().enumerate() {
            match &a.indices[0] {
                IndexExpr::Literal(n) => assert_eq!(*n, idx),
                _ => panic!("expected Literal(o) at indices[0]"),
            }
            match &a.src {
                SignalRef::LocalParam(name) => {
                    assert_eq!(name, &format!("b1_{}", idx));
                }
                _ => panic!("expected LocalParam src, got {:?}", a.src),
            }
        }

        // With bias present, no zero_w<acc_width> LocalParam should be emitted.
        assert!(
            !module.local_params().iter().any(|lp| lp.name == "zero_w32"),
            "with-bias layer must NOT declare zero_w32 LocalParam"
        );
    }

    /// M57.1 wire-array mini §3.2 + §4 / Task W4: no-bias defensive path
    /// declares a single `zero_w<acc_width>` LocalParam and uses it as the
    /// accumulator seed.
    #[test]
    fn lower_v1_mlp_single_layer_w4_no_bias_uses_zero_localparam() {
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

        // Zero LocalParam declared exactly once.
        let zero_count = module
            .local_params()
            .iter()
            .filter(|lp| lp.name == "zero_w32")
            .count();
        assert_eq!(zero_count, 1, "zero_w32 LocalParam must be declared once");

        // Bias-seed assigns reference zero_w32.
        let bias_assigns: Vec<&AssignWireArrayElement> = module.nodes().iter().filter_map(|n| match n {
            HirNode::AssignWireArrayElement(a)
                if a.array_name == "acc_l1"
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
    }
}
