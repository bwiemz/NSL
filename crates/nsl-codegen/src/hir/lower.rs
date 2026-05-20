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
    /// Task 3.1: behavior-preserving delegation. When `bias_op` is `Some`, the
    /// caller has consumed the next KIR op (i += 2); to keep M57's HIR output
    /// byte-identical we still emit a standalone ElementwiseAdd GenerateFor
    /// here. Task 3.4 replaces that with a bias-as-seed fold (seeds the MAC
    /// accumulator from `bias[o]` instead of zero, dropping the separate Add
    /// block). When `bias_op` is `None`, the accumulator seeds from zero
    /// (current M57 behavior).
    fn lower_matmul_with_optional_bias(
        &self,
        matmul_op: &KirOp,
        bias_op: Option<&KirOp>,
        module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        self.lower_matmul(matmul_op, module)?;
        // Task 3.4 will fold this into the Matmul's accumulator seed; for now
        // we re-emit the standalone Add so HIR snapshots stay byte-identical.
        if let Some(bias) = bias_op {
            self.lower_elementwise_add(bias, module)?;
        }
        Ok(())
    }

    fn lower_matmul(
        &self,
        op: &KirOp,
        module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        use crate::hir::control::GenerateFor;
        use crate::hir::module::HirNode;
        use crate::hir::nodes::{Add, LocalParam, Mul, Port, SignExtend};
        use crate::hir::signals::SignalRef;

        let (a_dtype, b_dtype, out_dtype, a_shape, b_shape) = match op {
            KirOp::Matmul {
                a_dtype,
                b_dtype,
                out_dtype,
                a_shape,
                b_shape,
                ..
            } => (a_dtype.clone(), b_dtype.clone(), out_dtype.clone(), *a_shape, *b_shape),
            _ => unreachable!("lower_matmul called with non-Matmul op"),
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

        // ── M57.1 §3.4: Declare ports + LocalParams inline ──────────────────
        // Per the no-dangling-references invariant, every SignalRef::Port and
        // SignalRef::LocalParam emitted below must be backed by a declaration.
        //
        // x is the only true Port::Input (the stimulus passed by the testbench);
        // W and b are LocalParams (values baked at CLI time by Task 4.1's
        // bake_fixture_into_localparams).
        //
        // Layer-disambiguating suffix: scan existing LocalParams for unique
        // `W<n>_` prefixes — robust to layers having different (k_dim, n_outputs)
        // pairs (e.g. MLP 784→128→10). Picking by max-existing-index keeps the
        // counter monotonic regardless of how many elements each prior layer
        // contributed. Alternative would be &mut state on the pass; the
        // count-derived index is implementation-light for v1.
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
        // Declare LocalParam for each bias element (n_outputs values).
        // Width = acc_width — bias is added at accumulator precision in the
        // bias-as-seed ripple chain (Task 3.4).
        for o in 0..n_outputs {
            let elem_name = format!("{}_{}", b_lp_prefix, o);
            module.add_local_param(LocalParam::new(elem_name, acc_width, 0));
        }

        // ── Build the MAC body ──────────────────────────────────────────────
        // The inner generate-for body emits one structural Mul + (optional)
        // SignExtend + Add. The Verilog emitter resolves per-iteration indexing
        // (`x_l<i>[k]`, `W<i>_<k>_<o>`) when it unrolls the genvars. SignalRef
        // here uses the bare prefix; the emitter substitutes the indices.
        let mut outer_body: Vec<HirNode> = Vec::new();
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

        // TEMPORARY (one task only): keep the M57-shape accumulator placeholder.
        // Task 3.4 replaces this with the bias-as-seed ripple chain — the very
        // first MAC iteration seeds the accumulator from `b<i>_<o>` instead of
        // referencing an external (and currently undeclared) acc_prev port.
        let add = Add::new(
            SignalRef::port("__matmul_acc_prev_LEGACY"),
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
        // no-dangling-references invariant holds on standalone ElementwiseAdd
        // (v1 MLP path absorbs the matmul+bias pair into a fold at Task 3.4;
        // until then Task 3.1's wrapper still re-emits the bias-add through
        // this function — so the layer-disambiguation must agree between the
        // two invocation contexts).
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
/// Scans `module.local_params` for names matching `W<n>_…` and returns the
/// smallest 1-based index greater than the largest `<n>` seen (or 1 when no
/// matching LocalParams exist).
///
/// This is robust to layers having heterogeneous (k_dim × n_outputs) shapes
/// (e.g. MLP 784→128→10), unlike the alternative of dividing the W count by
/// a fixed per-layer element count. It also tolerates non-W LocalParams
/// (e.g. biases) being interleaved.
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
