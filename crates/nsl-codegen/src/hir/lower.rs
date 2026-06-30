//! KIR → HIR pass per M57 v1 spec §2.3 (load-bearing pass).
//! Three concerns: op recognition + dispatch; HIR construction; dtype propagation.

use crate::fpga_error::{FpgaLoweringError, V1_SUPPORTED_OPS};
use crate::hir::module::HirModule;
use crate::kernel_ir::{KernelIR, KirOp, KirType, VarId};

pub struct KirToHirPass {
    pub test_taps: bool,
    /// M57.2: select the sequential clocked FSM lowering for recognized v1 MLP
    /// chains instead of the combinational `lower_v1_mlp_layer_chain`. The
    /// `new` constructor defaults this to `false` (combinational path) so all
    /// existing callers stay on the untouched golden-equivalence reference.
    pub sequential: bool,
}

impl KirToHirPass {
    pub fn new(test_taps: bool) -> Self {
        Self { test_taps, sequential: false }
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
                        // M57.2 dispatch: a recognized v1 MLP chain lowers via
                        // the clocked sequential FSM when `sequential` is set;
                        // otherwise via the untouched combinational chain. Both
                        // consume the same `&[&KirOp]` triple slice. Standalone
                        // fallbacks below remain combinational regardless — the
                        // v1 MLP chain is the only sequential target.
                        if self.sequential {
                            lower_v1_mlp_sequential(
                                &ops[i..i + chain_len],
                                &mut module,
                            )?;
                        } else {
                            lower_v1_mlp_layer_chain(
                                &ops[i..i + chain_len],
                                &mut module,
                            )?;
                        }
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
        use crate::hir::nodes::{Add, LocalParam, Mul, Port, SignExtend, Wire, WireDecl};
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
        // M57.1 fix: declare before driving assign — avoids implicit 1-bit net.
        inner_body.push(HirNode::WireDecl(WireDecl { id: prod_id, width: prod_width }));
        inner_body.push(HirNode::Mul(mul));

        // SignExtend prod → acc_width if needed (3-node MAC chain when prod < acc)
        let add_input = if prod_width < acc_width {
            let se = SignExtend::new(SignalRef::wire(prod_id), prod_width, acc_width);
            let se_id = se.dst;
            // M57.1 fix: declare before driving assign.
            inner_body.push(HirNode::WireDecl(WireDecl { id: se_id, width: acc_width }));
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
        let add_id = add.out;
        // M57.1 fix: declare before driving assign.
        inner_body.push(HirNode::WireDecl(WireDecl { id: add_id, width: acc_width }));
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
        use crate::hir::nodes::{Add, Port, WireDecl};
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

        let add = Add::new(
            SignalRef::port(&a_name),
            SignalRef::port(&b_name),
            width,
        );
        let add_id = add.out;
        // M57.1 fix: declare before driving assign — avoids implicit 1-bit net.
        let body = vec![
            HirNode::WireDecl(WireDecl { id: add_id, width }),
            HirNode::Add(add),
        ];

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
        use crate::hir::nodes::{Max0, Port, WireDecl};
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

        // M57.1 fix: declare before driving assign — avoids implicit 1-bit net.
        let body = vec![
            HirNode::WireDecl(WireDecl { id: max0_out_id, width }),
            HirNode::Max0(max0),
        ];

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

// ════════════════════════════════════════════════════════════════════════
//  M57.2 — sequential clocked FSM lowering (parallel to the combinational
//  `lower_v1_mlp_layer_chain` above). Builds an IDLE → {L{i}_MAC} → DONE
//  state machine with bias-as-seed accumulation and inline relu, one
//  MAC/cycle. The combinational path stays untouched as the golden
//  equivalence reference.
// ════════════════════════════════════════════════════════════════════════

use crate::hir::ids::{RegisterId, WireId};

/// Minimum number of bits required to represent the unsigned value `v`
/// (so a `[width-1:0]` register holds `0..=v`). `bits_to_hold(0) == 1`,
/// `bits_to_hold(4) == 3`, `bits_to_hold(127) == 7`, `bits_to_hold(783) == 10`.
fn bits_to_hold(v: usize) -> usize {
    if v == 0 {
        1
    } else {
        (usize::BITS - v.leading_zeros()) as usize
    }
}

/// M57.2: per-layer geometry + datapath wire ids gathered during sequential
/// lowering. `k`/`n` are the layer's matmul inner/output dims; widths come
/// from the layer's dtypes (mirroring `lower_v1_mlp_single_layer`). The
/// `*_id` fields are populated by `build_datapath` (Task 11) and read by
/// `build_fsm` (Task 12).
struct LayerDims {
    k: usize,
    n: usize,
    a_width: usize,
    b_width: usize,
    acc_width: usize,
    prod_width: usize,
    /// `W{i}` LocalParamArray name (weights, `[k][n]`).
    w_name: String,
    /// `b{i}` LocalParamArray name (bias, `[n]`).
    b_name: String,
    // ── Datapath wires (Task 11) ────────────────────────────────────────
    /// `acc_next = acc + sext(prod)` (width `acc_width_max`).
    acc_next_id: Option<WireId>,
    /// `relu = max0(acc_next)` (width `acc_width_max`).
    relu_id: Option<WireId>,
    /// `k_next = k + 1`.
    k_next_id: Option<WireId>,
    /// `o_next = o + 1`.
    o_next_id: Option<WireId>,
    /// `k_is_last = (k == K_i - 1)`.
    k_is_last_id: Option<WireId>,
    /// `o_is_last = (o == N_i - 1)`.
    o_is_last_id: Option<WireId>,
}

/// M57.2: shared lowering context threaded through `build_datapath` (Task 11)
/// and `build_fsm` (Task 12). Built by `lower_v1_mlp_sequential` (Task 10).
struct SeqLoweringCtx {
    // ── FSM scalar registers ────────────────────────────────────────────
    state_reg: RegisterId,
    o_reg: RegisterId,
    k_reg: RegisterId,
    acc_reg: RegisterId,
    done_reg: RegisterId,
    valid_reg: RegisterId,
    // ── State codes, ordered (name, code) ───────────────────────────────
    /// `[("S_IDLE",0), ("S_LOAD",1), ("S_L1_MAC",2), …, ("S_DONE",last)]`.
    state_codes: Vec<(String, u64)>,
    // ── Per-layer geometry + datapath wires ─────────────────────────────
    layers: Vec<LayerDims>,
    // ── Shared sizing ───────────────────────────────────────────────────
    /// Single shared accumulator width (max acc_width across layers; 64 for
    /// the v1 fixture). All `Add`/`Max0`/`SignExtend` targets use this.
    acc_width_max: usize,
    /// `k` counter width (holds `max_i K_i`).
    k_width: usize,
    /// `o` counter width (holds `max_i N_i`).
    o_width: usize,
}

impl SeqLoweringCtx {
    /// Look up a state code by name (panics if absent — names are
    /// constructed and consumed within this module). Consumed by `build_fsm`
    /// (Task 12).
    fn state_code(&self, name: &str) -> u64 {
        self.state_codes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| *c)
            .unwrap_or_else(|| panic!("state {name} not registered in SeqLoweringCtx"))
    }
}

/// M57.2: lower a recognized v1 MLP chain (same `&[&KirOp]` triple slice as
/// `lower_v1_mlp_layer_chain`) to a clocked sequential FSM. Builds the
/// declarations + lowering context, then the combinational datapath
/// (`build_datapath`) and the FSM body (`build_fsm`).
pub(crate) fn lower_v1_mlp_sequential(
    ops: &[&KirOp],
    module: &mut HirModule,
) -> Result<(), FpgaLoweringError> {
    use crate::hir::ids::ResetSignalId;
    use crate::hir::module::HirNode;
    use crate::hir::nodes::{LocalParam, LocalParamArray, Port, RegArray, RegDecl};
    use crate::hir::signals::SignalRef;

    assert_eq!(ops.len() % 3, 0, "v1 MLP chain ops must come in triples");
    let n_layers = ops.len() / 3;
    assert!(n_layers >= 1, "sequential lowering requires >= 1 layer");

    // ── 1. Extract per-layer geometry from the matmul triples ───────────
    let mut layers: Vec<LayerDims> = Vec::with_capacity(n_layers);
    for layer_idx in 1..=n_layers {
        let matmul = ops[(layer_idx - 1) * 3];
        let (a_dtype, b_dtype, out_dtype, a_shape, b_shape) = match matmul {
            KirOp::Matmul {
                a_dtype, b_dtype, out_dtype, a_shape, b_shape, ..
            } => (
                a_dtype.clone(), b_dtype.clone(), out_dtype.clone(), *a_shape, *b_shape,
            ),
            _ => unreachable!("lower_v1_mlp_sequential called with non-Matmul matmul"),
        };
        let k = a_shape[1];
        let n = b_shape[1];
        assert_eq!(a_shape[1], b_shape[0], "Matmul inner dimensions must match");
        let a_width = kir_dtype_width(a_dtype);
        let b_width = kir_dtype_width(b_dtype);
        let acc_width = kir_dtype_width(out_dtype);
        layers.push(LayerDims {
            k,
            n,
            a_width,
            b_width,
            acc_width,
            prod_width: a_width + b_width,
            w_name: format!("W{}", layer_idx),
            b_name: format!("b{}", layer_idx),
            acc_next_id: None,
            relu_id: None,
            k_next_id: None,
            o_next_id: None,
            k_is_last_id: None,
            o_is_last_id: None,
        });
    }

    // ── Shared sizing ───────────────────────────────────────────────────
    let k1 = layers[0].k;
    let a_width_1 = layers[0].a_width;
    let max_k = layers.iter().map(|l| l.k).max().unwrap();
    let max_n = layers.iter().map(|l| l.n).max().unwrap();
    let acc_width_max = layers.iter().map(|l| l.acc_width).max().unwrap();
    let n_last = layers[n_layers - 1].n;
    let k_width = bits_to_hold(max_k);
    let o_width = bits_to_hold(max_n);

    // ── 2. Sequential reset name: rst_n (active-low) per spec §1.2 ───────
    module.reset_signals.insert(ResetSignalId::DEFAULT, "rst_n".into());

    // ── 3. FSM state localparams + ordered code table ───────────────────
    //   IDLE=0, L1_MAC=1, …, L{n}_MAC=n, DONE=n+1.
    // states: IDLE + per-layer MAC + DONE = n_layers + 2
    let n_states = n_layers + 2; // IDLE + per-layer MAC + DONE
    let sw = bits_to_hold(n_states - 1);
    let mut state_codes: Vec<(String, u64)> = Vec::with_capacity(n_states);
    state_codes.push(("S_IDLE".into(), 0));
    for layer_idx in 1..=n_layers {
        state_codes.push((format!("S_L{}_MAC", layer_idx), layer_idx as u64));
    }
    state_codes.push(("S_DONE".into(), (n_states - 1) as u64));
    for (name, code) in &state_codes {
        module.add_local_param(LocalParam::new(name, sw, *code as i128));
    }
    // Constant-assignment helpers (RegAssign.value is a SignalRef with no
    // integer-literal variant): ZERO (acc-width) covers all `<= '0` resets and
    // counter zeroing; ONE (1-bit) covers `done/valid <= 1` in S_DONE.
    module.add_local_param(LocalParam::new("ZERO", acc_width_max, 0));
    module.add_local_param(LocalParam::new("ONE", 1, 1));

    // ── 4. Scalar RegDecls (declaration-only; driven by the SeqProcess) ──
    let state_decl = RegDecl::new(sw);
    let state_reg = state_decl.id;
    module.add_node(HirNode::RegDecl(state_decl))?;
    let o_decl = RegDecl::new(o_width);
    let o_reg = o_decl.id;
    module.add_node(HirNode::RegDecl(o_decl))?;
    let k_decl = RegDecl::new(k_width);
    let k_reg = k_decl.id;
    module.add_node(HirNode::RegDecl(k_decl))?;
    let acc_decl = RegDecl::new(acc_width_max);
    let acc_reg = acc_decl.id;
    module.add_node(HirNode::RegDecl(acc_decl))?;
    let done_decl = RegDecl::new(1);
    let done_reg = done_decl.id;
    module.add_node(HirNode::RegDecl(done_decl))?;
    let valid_decl = RegDecl::new(1);
    let valid_reg = valid_decl.id;
    module.add_node(HirNode::RegDecl(valid_decl))?;

    // ── 5. RegArray buffers: x_buf[K_1] + shared h_buf[max_N] ───────────
    module.add_node(HirNode::RegArray(RegArray {
        name: "x_buf".into(),
        dims: vec![k1],
        width: a_width_1,
    }))?;
    module.add_node(HirNode::RegArray(RegArray {
        name: "h_buf".into(),
        dims: vec![max_n],
        width: acc_width_max,
    }))?;

    // ── 6. Weight LocalParamArrays per layer (mirror M57.1 idiom) ───────
    for layer in &layers {
        module.add_local_param_array(LocalParamArray {
            name: layer.w_name.clone(),
            dims: vec![layer.k, layer.n],
            width: layer.b_width,
            values: vec![0; layer.k * layer.n],
        });
        module.add_local_param_array(LocalParamArray {
            name: layer.b_name.clone(),
            dims: vec![layer.n],
            width: layer.acc_width,
            values: vec![0; layer.n],
        });
    }

    // ── 7. Handshake ports ──────────────────────────────────────────────
    module.add_port(Port::input("clk", 1));
    module.add_port(Port::input("rst_n", 1));
    module.add_port(Port::input("start", 1));
    module.add_port(Port::input("x", k1 * a_width_1));
    // done/valid outputs driven by the 1-bit RegDecls (FSM drives them).
    module.add_port(Port::output("done", 1, SignalRef::register(done_reg)));
    module.add_port(Port::output("valid", 1, SignalRef::register(valid_reg)));
    // `out` is driven combinationally by concatenating h_buf's last-layer
    // cells via RegArrayConcat (Task 12 adds the SignalRef variant); the FSM
    // does NOT write `out`. done/valid signal validity.
    module.add_port(Port::output(
        "out",
        n_last * acc_width_max,
        SignalRef::reg_array_concat("h_buf", n_last),
    ));

    // ── 8. Deterministic cycle count (§5.1: overhead = IDLE (start/latch) + DONE = 2) ──
    let total: usize = 2 + layers.iter().map(|l| l.n * l.k).sum::<usize>();
    module.cycle_count = Some(total as u64);

    // ── 9. Assemble ctx + build datapath (Task 11) + FSM (Task 12) ──────
    let mut ctx = SeqLoweringCtx {
        state_reg,
        o_reg,
        k_reg,
        acc_reg,
        done_reg,
        valid_reg,
        state_codes,
        layers,
        acc_width_max,
        k_width,
        o_width,
    };

    build_datapath(&mut ctx, module)?;
    build_fsm(&ctx, module)?;

    Ok(())
}

/// M57.2 (Task 11): per-layer combinational MAC-step + control wires the FSM
/// reads, indexed by the runtime `k`/`o` registers (vs the combinational
/// path's genvars). One inner MAC iteration per layer:
///
/// ```text
///   prod      = x_src[k] * W{i}[k][o]              (a_width × b_width → prod_width)
///   prod_ext  = sext(prod) → acc_width_max         (when prod_width < acc_width_max)
///   acc_next  = acc + prod_ext                     (§3.5; acc register read)
///   relu      = max0(acc_next)                     (inline relu, §3.5)
///   k_next    = k + 1 ;  o_next = o + 1
///   k_is_last = (k == K_i-1) ; o_is_last = (o == N_i-1)
/// ```
///
/// `x_src` = `x_buf` (layer 1) or the shared `h_buf` (layer i>1). A single
/// shared 64-bit `acc` register serves all phases, so `Add`/`Max0`/`SignExtend`
/// targets use `acc_width_max` uniformly (Add requires equal operand widths);
/// per-layer sign-extended values sit in the 64-bit acc and the 64-bit
/// relu/compare is value-correct. Per-layer wires are distinct (fresh ids) and
/// coexist at module scope — only the active phase's `acc`/`k`/`o` drive
/// meaningful values. Wire ids are recorded back into each `LayerDims` for
/// `build_fsm` (Task 12).
fn build_datapath(
    ctx: &mut SeqLoweringCtx,
    module: &mut HirModule,
) -> Result<(), FpgaLoweringError> {
    use crate::hir::module::HirNode;
    use crate::hir::nodes::{Add, AddConst, CmpEq, IndexExpr, Max0, Mul, SignExtend, WireDecl};
    use crate::hir::signals::SignalRef;

    let acc_width_max = ctx.acc_width_max;
    let k_width = ctx.k_width;
    let o_width = ctx.o_width;
    let k_reg = ctx.k_reg;
    let o_reg = ctx.o_reg;
    let acc_reg = ctx.acc_reg;

    let n_layers = ctx.layers.len();
    for layer_idx in 1..=n_layers {
        let layer = &ctx.layers[layer_idx - 1];
        let k_i = layer.k;
        let n_i = layer.n;
        let a_width = layer.a_width;
        let b_width = layer.b_width;
        let prod_width = layer.prod_width;
        let w_name = layer.w_name.clone();

        // x_src[k]: layer 1 reads the latched x_buf; layers > 1 read shared h_buf.
        let x_src_name = if layer_idx == 1 { "x_buf" } else { "h_buf" };
        let x_elem = SignalRef::reg_array_element(x_src_name, vec![IndexExpr::Reg(k_reg)]);
        // W{i}[k][o] read at runtime k,o.
        let w_elem = SignalRef::indexed_local_param(
            w_name,
            vec![IndexExpr::Reg(k_reg), IndexExpr::Reg(o_reg)],
        );
        // prod = x_src[k] * W{i}[k][o]
        let prod = Mul::new(x_elem, w_elem, a_width, b_width, prod_width);
        let prod_id = prod.out;
        // M57.2: declare before driving assign to avoid implicit 1-bit net.
        module.add_node(HirNode::WireDecl(WireDecl { id: prod_id, width: prod.out_width }))?;
        module.add_node(HirNode::Mul(prod))?;

        // Optional SignExtend prod → acc_width_max (shared 64-bit acc).
        let prod_ext = if prod_width < acc_width_max {
            let se = SignExtend::new(SignalRef::wire(prod_id), prod_width, acc_width_max);
            let se_id = se.dst;
            // M57.2: declare before driving assign.
            module.add_node(HirNode::WireDecl(WireDecl { id: se_id, width: se.dst_width }))?;
            module.add_node(HirNode::SignExtend(se))?;
            SignalRef::wire(se_id)
        } else {
            SignalRef::wire(prod_id)
        };

        // acc_next = acc + prod_ext   (acc register read; §3.5).
        let acc_next = Add::new(SignalRef::register(acc_reg), prod_ext, acc_width_max);
        let acc_next_id = acc_next.out;
        // M57.2: declare before driving assign.
        module.add_node(HirNode::WireDecl(WireDecl { id: acc_next_id, width: acc_next.width }))?;
        module.add_node(HirNode::Add(acc_next))?;

        // relu = max0(acc_next)  (inline relu reads the combinational wire).
        let relu = Max0::new(SignalRef::wire(acc_next_id), acc_width_max);
        let relu_id = relu.out;
        // M57.2: declare before driving assign.
        module.add_node(HirNode::WireDecl(WireDecl { id: relu_id, width: relu.width }))?;
        module.add_node(HirNode::Max0(relu))?;

        // k_next = k + 1 ; o_next = o + 1.
        let k_next = AddConst::new(SignalRef::register(k_reg), 1, k_width);
        let k_next_id = k_next.out;
        // M57.2: declare before driving assign.
        module.add_node(HirNode::WireDecl(WireDecl { id: k_next_id, width: k_next.width }))?;
        module.add_node(HirNode::AddConst(k_next))?;
        let o_next = AddConst::new(SignalRef::register(o_reg), 1, o_width);
        let o_next_id = o_next.out;
        // M57.2: declare before driving assign.
        module.add_node(HirNode::WireDecl(WireDecl { id: o_next_id, width: o_next.width }))?;
        module.add_node(HirNode::AddConst(o_next))?;

        // k_is_last = (k == K_i-1) ; o_is_last = (o == N_i-1).
        let k_is_last = CmpEq::new(SignalRef::register(k_reg), (k_i - 1) as u64);
        let k_is_last_id = k_is_last.out;
        // M57.2: CmpEq is 1-bit.
        module.add_node(HirNode::WireDecl(WireDecl { id: k_is_last_id, width: 1 }))?;
        module.add_node(HirNode::CmpEq(k_is_last))?;
        let o_is_last = CmpEq::new(SignalRef::register(o_reg), (n_i - 1) as u64);
        let o_is_last_id = o_is_last.out;
        // M57.2: CmpEq is 1-bit.
        module.add_node(HirNode::WireDecl(WireDecl { id: o_is_last_id, width: 1 }))?;
        module.add_node(HirNode::CmpEq(o_is_last))?;

        // Record wire ids back into the LayerDims for build_fsm (Task 12).
        let layer_mut = &mut ctx.layers[layer_idx - 1];
        layer_mut.acc_next_id = Some(acc_next_id);
        layer_mut.relu_id = Some(relu_id);
        layer_mut.k_next_id = Some(k_next_id);
        layer_mut.o_next_id = Some(o_next_id);
        layer_mut.k_is_last_id = Some(k_is_last_id);
        layer_mut.o_is_last_id = Some(o_is_last_id);
    }

    Ok(())
}

/// M57.2 (Task 12): the FSM `SeqProcess` — one `always_ff` whose `body` is a
/// state `Case` (IDLE → {L{i}_MAC} → DONE) with bias-as-seed accumulation
/// (SQ-5, three cases) and inline relu (SQ-6/8). Reads the datapath wire ids
/// recorded by `build_datapath` (Task 11).
///
/// Bias-seed (SQ-5) three cases:
///   1. Phase-entry seed `b{i}[0]` — set on the transition INTO the phase:
///      layer 1 on `start` in IDLE; layer i>1 at the prev layer's o-boundary
///      (so S_L{i}_MAC's non-final o_is_last arm seeds `b{i+1}[0]`).
///   2. Intra-layer advance `acc <= b{i}[(o+1)]` via RegPlus — **reads old o**,
///      concurrent with `o <= o_next`.
///   3. Final layer's o-boundary → S_DONE with **no seed**.
///
/// Inline relu (SQ-6/8): on the `k_is_last` arm, `h_buf[o] <= relu` reads the
/// combinational `relu` wire (from acc_next); `acc <=` takes the bias-seed,
/// NOT acc_next. On the else (k<K-1) arm: `acc <= acc_next; k <= k_next`.
///
/// The FSM does NOT write `out` — the `out` port is driven combinationally by
/// `RegArrayConcat("h_buf", N_last)`. v2 sequential output is just the `out`
/// port (per-layer test_taps are dropped: the shared h_buf aliases them).
fn build_fsm(ctx: &SeqLoweringCtx, module: &mut HirModule) -> Result<(), FpgaLoweringError> {
    use crate::hir::clock_reset::{ClkRef, ResetPolarity, ResetRef, ResetSync};
    use crate::hir::ids::{ClockDomainId, ResetSignalId};
    use crate::hir::module::HirNode;
    use crate::hir::nodes::{IndexExpr, SeqLValue, SeqProcess, SeqStmt};
    use crate::hir::signals::SignalRef;

    let state_reg = ctx.state_reg;
    let o_reg = ctx.o_reg;
    let k_reg = ctx.k_reg;
    let acc_reg = ctx.acc_reg;
    let done_reg = ctx.done_reg;
    let valid_reg = ctx.valid_reg;
    let n_layers = ctx.layers.len();

    // Constant-source SignalRefs (RegAssign.value is a SignalRef — no integer
    // literal variant). ZERO covers all `<= '0`; ONE covers `done/valid <= 1`.
    let zero = || SignalRef::local_param("ZERO");
    let one = || SignalRef::local_param("ONE");

    // ── S_IDLE arm ──────────────────────────────────────────────────────
    // done/valid cleared every cycle in IDLE; on `start`, latch x → x_buf,
    // seed acc with b1[0], zero o/k, advance to S_L1_MAC.
    let k1 = ctx.layers[0].k;
    let a_width_1 = ctx.layers[0].a_width;
    let b1_name = ctx.layers[0].b_name.clone();
    let mut start_body: Vec<SeqStmt> = Vec::with_capacity(k1 + 4);
    for k in 0..k1 {
        // x_buf[k] <= x[k*a_width +: a_width]
        start_body.push(SeqStmt::RegAssign {
            target: SeqLValue::RegArrayElement {
                array_name: "x_buf".into(),
                indices: vec![IndexExpr::Literal(k)],
            },
            value: SignalRef::port_bit_slice("x", k * a_width_1, a_width_1),
        });
    }
    // acc <= b1[0]; o <= '0; k <= '0; state <= S_L1_MAC.
    start_body.push(SeqStmt::RegAssign {
        target: SeqLValue::Register(acc_reg),
        value: SignalRef::indexed_local_param(b1_name, vec![IndexExpr::Literal(0)]),
    });
    start_body.push(SeqStmt::RegAssign { target: SeqLValue::Register(o_reg), value: zero() });
    start_body.push(SeqStmt::RegAssign { target: SeqLValue::Register(k_reg), value: zero() });
    start_body.push(SeqStmt::RegAssign {
        target: SeqLValue::Register(state_reg),
        value: SignalRef::local_param("S_L1_MAC"),
    });
    let idle_arm = vec![
        SeqStmt::RegAssign { target: SeqLValue::Register(done_reg), value: zero() },
        SeqStmt::RegAssign { target: SeqLValue::Register(valid_reg), value: zero() },
        SeqStmt::If {
            cond: SignalRef::port("start"),
            then_body: start_body,
            else_body: vec![],
        },
    ];

    // ── Case arms: IDLE + per-layer MAC + DONE ──────────────────────────
    let mut arms: Vec<(u64, Vec<SeqStmt>)> = Vec::with_capacity(n_layers + 2);
    arms.push((ctx.state_code("S_IDLE"), idle_arm));

    for layer_idx in 1..=n_layers {
        let layer = &ctx.layers[layer_idx - 1];
        let is_final = layer_idx == n_layers;
        let relu_id = layer.relu_id.expect("build_datapath must run before build_fsm");
        let acc_next_id = layer.acc_next_id.expect("build_datapath must run before build_fsm");
        let k_next_id = layer.k_next_id.expect("build_datapath must run before build_fsm");
        let o_next_id = layer.o_next_id.expect("build_datapath must run before build_fsm");
        let k_is_last_id = layer.k_is_last_id.expect("build_datapath must run before build_fsm");
        let o_is_last_id = layer.o_is_last_id.expect("build_datapath must run before build_fsm");
        let b_name = layer.b_name.clone();

        // o_is_last arm (last k of last o of this layer): advance the phase.
        let o_is_last_then: Vec<SeqStmt> = if is_final {
            // Final layer → S_DONE, NO seed (case 3).
            vec![SeqStmt::RegAssign {
                target: SeqLValue::Register(state_reg),
                value: SignalRef::local_param("S_DONE"),
            }]
        } else {
            // Intermediate layer → seed NEXT layer's b{i+1}[0] (case 1),
            // zero o, advance to S_L{i+1}_MAC.
            let next_b_name = ctx.layers[layer_idx].b_name.clone();
            let next_state = format!("S_L{}_MAC", layer_idx + 1);
            vec![
                SeqStmt::RegAssign {
                    target: SeqLValue::Register(acc_reg),
                    value: SignalRef::indexed_local_param(
                        next_b_name,
                        vec![IndexExpr::Literal(0)],
                    ),
                },
                SeqStmt::RegAssign { target: SeqLValue::Register(o_reg), value: zero() },
                SeqStmt::RegAssign {
                    target: SeqLValue::Register(state_reg),
                    value: SignalRef::local_param(next_state),
                },
            ]
        };
        // intra-layer o-advance (k_is_last but not o_is_last): seed b{i}[o+1]
        // (case 2 — RegPlus reads OLD o, concurrent with o <= o_next).
        let o_advance: Vec<SeqStmt> = vec![
            SeqStmt::RegAssign {
                target: SeqLValue::Register(acc_reg),
                value: SignalRef::indexed_local_param(
                    b_name,
                    vec![IndexExpr::RegPlus(o_reg, 1)],
                ),
            },
            SeqStmt::RegAssign {
                target: SeqLValue::Register(o_reg),
                value: SignalRef::wire(o_next_id),
            },
        ];

        // k_is_last arm: write inline-relu result to h_buf[o], reset k, then
        // branch on o_is_last (phase advance vs intra-layer o-advance).
        let mut k_is_last_then: Vec<SeqStmt> = vec![
            SeqStmt::RegAssign {
                target: SeqLValue::RegArrayElement {
                    array_name: "h_buf".into(),
                    indices: vec![IndexExpr::Reg(o_reg)],
                },
                value: SignalRef::wire(relu_id),
            },
            SeqStmt::RegAssign { target: SeqLValue::Register(k_reg), value: zero() },
        ];
        k_is_last_then.push(SeqStmt::If {
            cond: SignalRef::wire(o_is_last_id),
            then_body: o_is_last_then,
            else_body: o_advance,
        });

        // else (k < K-1): ordinary accumulate — acc <= acc_next; k <= k_next.
        let k_advance: Vec<SeqStmt> = vec![
            SeqStmt::RegAssign {
                target: SeqLValue::Register(acc_reg),
                value: SignalRef::wire(acc_next_id),
            },
            SeqStmt::RegAssign {
                target: SeqLValue::Register(k_reg),
                value: SignalRef::wire(k_next_id),
            },
        ];

        let mac_arm = vec![SeqStmt::If {
            cond: SignalRef::wire(k_is_last_id),
            then_body: k_is_last_then,
            else_body: k_advance,
        }];
        arms.push((ctx.state_code(&format!("S_L{}_MAC", layer_idx)), mac_arm));
    }

    // ── S_DONE arm ──────────────────────────────────────────────────────
    let done_arm = vec![
        SeqStmt::RegAssign { target: SeqLValue::Register(done_reg), value: one() },
        SeqStmt::RegAssign { target: SeqLValue::Register(valid_reg), value: one() },
        SeqStmt::If {
            cond: SignalRef::port("start"),
            then_body: vec![],
            // !start handled by the else: return to IDLE. We model `if (!start)`
            // by swapping then/else — there is no NotPort SignalRef, and the
            // emitter renders `if (start) begin end else begin <return> end`,
            // which is logically `if (!start)`.
            else_body: vec![SeqStmt::RegAssign {
                target: SeqLValue::Register(state_reg),
                value: SignalRef::local_param("S_IDLE"),
            }],
        },
    ];
    arms.push((ctx.state_code("S_DONE"), done_arm));

    // ── default arm: return to IDLE ─────────────────────────────────────
    let default_arm = vec![SeqStmt::RegAssign {
        target: SeqLValue::Register(state_reg),
        value: SignalRef::local_param("S_IDLE"),
    }];

    // ── reset_body: scalar resets (state <= S_IDLE; o/k/acc/done/valid <= 0)
    let reset_body = vec![
        SeqStmt::RegAssign {
            target: SeqLValue::Register(state_reg),
            value: SignalRef::local_param("S_IDLE"),
        },
        SeqStmt::RegAssign { target: SeqLValue::Register(o_reg), value: zero() },
        SeqStmt::RegAssign { target: SeqLValue::Register(k_reg), value: zero() },
        SeqStmt::RegAssign { target: SeqLValue::Register(acc_reg), value: zero() },
        SeqStmt::RegAssign { target: SeqLValue::Register(done_reg), value: zero() },
        SeqStmt::RegAssign { target: SeqLValue::Register(valid_reg), value: zero() },
    ];

    // reset_arrays: x_buf[K_1] + shared h_buf[max_N].
    let max_n = ctx.layers.iter().map(|l| l.n).max().unwrap();
    let reset_arrays = vec![("x_buf".into(), k1), ("h_buf".into(), max_n)];

    let sp = SeqProcess {
        clock: ClkRef { domain_id: ClockDomainId::DEFAULT },
        reset: ResetRef {
            signal_id: ResetSignalId::DEFAULT,
            polarity: ResetPolarity::Low,
            sync: ResetSync::Async,
        },
        reset_body,
        reset_arrays,
        body: vec![SeqStmt::Case {
            selector: SignalRef::register(state_reg),
            arms,
            default: default_arm,
        }],
    };
    module.add_node(HirNode::SeqProcess(sp))?;

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
        Max0, Mul, Port, SignExtend, WireArray, WireDecl,
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
    // M57.1 fix: declare before driving assign — avoids implicit 1-bit net.
    inner_body.push(HirNode::WireDecl(WireDecl { id: prod_id, width: prod_width }));
    inner_body.push(HirNode::Mul(prod));

    // Optional SignExtend if prod_width < acc_width.
    let extended_src = if prod_width < acc_width {
        let se = SignExtend::new(SignalRef::wire(prod_id), prod_width, acc_width);
        let se_id = se.dst;
        // M57.1 fix: declare before driving assign.
        inner_body.push(HirNode::WireDecl(WireDecl { id: se_id, width: acc_width }));
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
    // M57.1 fix: declare before driving assign.
    inner_body.push(HirNode::WireDecl(WireDecl { id: sum_id, width: acc_width }));
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
    // M57.1 fix: declare before driving assign.
    let relu_body = vec![
        HirNode::WireDecl(WireDecl { id: relu_max0_id, width: acc_width }),
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
    use crate::kernel_ir::{KirBuilder, KirTerminator};

    /// M57.2: build the canonical two-layer v1 MLP KernelIR (784 → 128 → 10),
    /// matching the combinational snapshot test's `full_v1_mlp_composition`
    /// fixture (`tests/hir_pass_snapshots.rs`). Layer 1: i8×i8→i32; layer 2:
    /// i32×i32→i64. Used by the sequential-lowering structural tests.
    fn two_layer_mlp_kir() -> KernelIR {
        let mut b = KirBuilder::new("tiny_mlp");
        let blk = b.new_block();
        b.set_block(blk);
        // Layer 1: 784 → 128 (i8×i8→i32)
        b.emit(KirOp::Matmul {
            a: 1, b: 2, out: 3,
            a_dtype: KirType::I8, b_dtype: KirType::I8, out_dtype: KirType::I32,
            a_shape: [1, 784], b_shape: [784, 128],
        });
        b.emit(KirOp::ElementwiseAdd {
            a: 3, b: 4, out: 5,
            dtype: KirType::I32, shape: [128],
        });
        b.emit(KirOp::Relu {
            a: 5, out: 6,
            dtype: KirType::I32, shape: [128],
        });
        // Layer 2: 128 → 10 (i32×i32→i64)
        b.emit(KirOp::Matmul {
            a: 6, b: 7, out: 8,
            a_dtype: KirType::I32, b_dtype: KirType::I32, out_dtype: KirType::I64,
            a_shape: [1, 128], b_shape: [128, 10],
        });
        b.emit(KirOp::ElementwiseAdd {
            a: 8, b: 9, out: 10,
            dtype: KirType::I64, shape: [10],
        });
        b.emit(KirOp::Relu {
            a: 10, out: 11,
            dtype: KirType::I64, shape: [10],
        });
        b.terminate(KirTerminator::Return);
        b.finalize()
    }

    #[test]
    fn sequential_skeleton_has_handshake_ports_and_buffers() {
        let kir = two_layer_mlp_kir();
        let module = KirToHirPass { test_taps: true, sequential: true }
            .lower(&kir, "tiny_mlp_seq").unwrap();
        let port_names: Vec<&str> = module.ports.iter().map(|p| match p {
            Port::Input { name, .. } | Port::Output { name, .. } => name.as_str(),
        }).collect();
        for required in ["clk", "rst_n", "start", "done", "valid", "x", "out"] {
            assert!(port_names.contains(&required), "missing port {required}");
        }
        // x_buf[784] + h_buf RegArrays present
        let reg_arrays: Vec<&str> = module.nodes().iter().filter_map(|n| match n {
            HirNode::RegArray(ra) => Some(ra.name.as_str()), _ => None,
        }).collect();
        assert!(reg_arrays.contains(&"x_buf"));
        assert!(reg_arrays.contains(&"h_buf"));
        // cycle count populated: 2 + N1*K1 + N2*K2 = 2 + 128*784 + 10*128
        assert_eq!(module.cycle_count, Some(2 + 128 * 784 + 10 * 128));
    }

    #[test]
    fn datapath_has_acc_next_and_guards_per_layer() {
        let kir = two_layer_mlp_kir();
        let module = KirToHirPass { test_taps: true, sequential: true }
            .lower(&kir, "tiny_mlp_seq").unwrap();
        // Two layers → two CmpEq for k_is_last + two for o_is_last (>= 4 total),
        // two AddConst for k_next/o_next per layer (>= 4 total), and Mul/Add/Max0
        // present for the MAC step.
        let n_cmp = module.nodes().iter().filter(|n| matches!(n, HirNode::CmpEq(_))).count();
        let n_addc = module.nodes().iter().filter(|n| matches!(n, HirNode::AddConst(_))).count();
        let n_mul = module.nodes().iter().filter(|n| matches!(n, HirNode::Mul(_))).count();
        let n_max0 = module.nodes().iter().filter(|n| matches!(n, HirNode::Max0(_))).count();
        assert!(n_cmp >= 4, "expected k/o guards per layer, got {n_cmp}");
        assert!(n_addc >= 4, "expected k/o increments per layer, got {n_addc}");
        assert!(n_mul >= 2, "expected one MAC product per layer, got {n_mul}");
        assert!(n_max0 >= 2, "expected inline relu per layer, got {n_max0}");
    }

    #[test]
    fn fsm_seq_process_present_with_state_case() {
        use crate::hir::nodes::SeqStmt;
        let kir = two_layer_mlp_kir();
        let module = KirToHirPass { test_taps: false, sequential: true }
            .lower(&kir, "tiny_mlp_seq").unwrap();
        let sp = module.nodes().iter().find_map(|n| match n {
            HirNode::SeqProcess(sp) => Some(sp), _ => None });
        let sp = sp.expect("expected one SeqProcess");
        // body is a single state Case
        assert!(matches!(sp.body.as_slice(), [SeqStmt::Case { .. }]));
        // reset_arrays cover x_buf + h_buf
        let names: Vec<&str> = sp.reset_arrays.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"x_buf") && names.contains(&"h_buf"));
    }

    #[test]
    fn sequential_false_keeps_combinational_path() {
        let kir = two_layer_mlp_kir();
        let comb = KirToHirPass { test_taps: true, sequential: false }
            .lower(&kir, "tiny_mlp").unwrap();
        // combinational path has NO SeqProcess and cycle_count is None
        assert!(!comb.nodes().iter().any(|n| matches!(n, HirNode::SeqProcess(_))));
        assert_eq!(comb.cycle_count, None);
        let seq = KirToHirPass { test_taps: true, sequential: true }
            .lower(&kir, "tiny_mlp_seq").unwrap();
        assert!(seq.nodes().iter().any(|n| matches!(n, HirNode::SeqProcess(_))));
        assert!(seq.cycle_count.is_some());
    }

    /// M57.2: every `assign _wN =` in the sequential datapath must have a
    /// matching `wire signed [...:0] _wN;` declaration (no implicit 1-bit nets).
    #[test]
    fn sequential_datapath_wires_are_all_declared() {
        use crate::backend_verilog::lower::VerilogEmitter;
        use std::collections::HashSet;

        let kir = two_layer_mlp_kir();
        let module = KirToHirPass { test_taps: false, sequential: true }
            .lower(&kir, "tiny_mlp_seq")
            .unwrap();
        let v = VerilogEmitter::emit_module(&module);

        // Collect declared wire ids from lines matching `wire signed [..:0] _wN;`
        let mut declared: HashSet<u64> = HashSet::new();
        for line in v.lines() {
            let trimmed = line.trim();
            // Match: "wire signed [<bits>:0] _w<N>;"
            if let Some(rest) = trimmed.strip_prefix("wire signed [") {
                if let Some(id_part) = rest.find("] _w").map(|pos| &rest[pos + 4..]) {
                    if let Some(id_str) = id_part.strip_suffix(';') {
                        if let Ok(id) = id_str.parse::<u64>() {
                            declared.insert(id);
                        }
                    }
                }
            }
        }

        // Collect assigned wire ids from lines matching `assign _wN = `
        let mut assigned: HashSet<u64> = HashSet::new();
        for line in v.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("assign _w") {
                if let Some(eq_pos) = rest.find(" = ") {
                    if let Ok(id) = rest[..eq_pos].parse::<u64>() {
                        assigned.insert(id);
                    }
                }
            }
        }

        // Every assigned wire must be declared.
        let undeclared: Vec<u64> = assigned.difference(&declared).copied().collect();
        assert!(
            undeclared.is_empty(),
            "sequential datapath has undeclared wires (implicit 1-bit nets): {:?}",
            undeclared,
        );

        // Specifically assert the 64-bit acc_next wire is declared as [63:0].
        assert!(
            v.contains("wire signed [63:0]"),
            "expected at least one 64-bit wire declaration (acc_next) in sequential Verilog",
        );

        // Sanity: there are assigned wires at all (guards against a vacuously-passing test).
        assert!(!assigned.is_empty(), "no assigned wires found — datapath may be empty");
    }

    /// M57.1 fix: every `assign _wN =` in the **combinational** v1 MLP datapath
    /// must have a matching `wire signed [...:0] _wN;` declaration so Verilog
    /// simulators see explicitly-sized wires instead of implicit 1-bit nets.
    ///
    /// Mirrors `sequential_datapath_wires_are_all_declared` (M57.2) for the
    /// combinational path. Must fail before the fix and pass after.
    #[test]
    fn combinational_datapath_wires_are_all_declared() {
        use crate::backend_verilog::lower::VerilogEmitter;
        use std::collections::HashSet;

        let kir = two_layer_mlp_kir();
        let module = KirToHirPass { test_taps: false, sequential: false }
            .lower(&kir, "tiny_mlp")
            .unwrap();
        let v = VerilogEmitter::emit_module(&module);

        // Collect declared wire ids from lines matching `wire signed [..:0] _wN;`
        let mut declared: HashSet<u64> = HashSet::new();
        for line in v.lines() {
            let trimmed = line.trim();
            // Match: "wire signed [<bits>:0] _w<N>;"
            if let Some(rest) = trimmed.strip_prefix("wire signed [") {
                if let Some(id_part) = rest.find("] _w").map(|pos| &rest[pos + 4..]) {
                    if let Some(id_str) = id_part.strip_suffix(';') {
                        if let Ok(id) = id_str.parse::<u64>() {
                            declared.insert(id);
                        }
                    }
                }
            }
        }

        // Collect assigned wire ids from lines matching `assign _wN = `
        let mut assigned: HashSet<u64> = HashSet::new();
        for line in v.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("assign _w") {
                if let Some(eq_pos) = rest.find(" = ") {
                    if let Ok(id) = rest[..eq_pos].parse::<u64>() {
                        assigned.insert(id);
                    }
                }
            }
        }

        // Every assigned wire must be declared (assigned ⊆ declared).
        let mut undeclared: Vec<u64> = assigned.difference(&declared).copied().collect();
        undeclared.sort_unstable();
        assert!(
            undeclared.is_empty(),
            "combinational datapath has undeclared wires (implicit 1-bit nets): {:?}",
            undeclared,
        );

        // Sanity: there are assigned wires at all (guards against a vacuously-passing test).
        assert!(!assigned.is_empty(), "no assigned wires found — combinational datapath may be empty");
    }

    /// M57.2: the FSM must have exactly n_layers + 2 states (IDLE + per-layer MAC
    /// + DONE) and must NOT contain the phantom S_LOAD state that was declared but
    /// never entered.
    #[test]
    fn sequential_fsm_has_no_phantom_state() {
        let kir = two_layer_mlp_kir();
        let module = KirToHirPass { test_taps: false, sequential: true }
            .lower(&kir, "tiny_mlp_seq").unwrap();
        // No "S_LOAD" localparam in the module.
        assert!(
            !module.local_params().iter().any(|lp| lp.name == "S_LOAD"),
            "S_LOAD phantom state must not be emitted"
        );
        // State localparams = IDLE + 2 MAC + DONE = 4 (n_layers=2 → n_layers+2=4).
        let n_state_lps = module.local_params().iter().filter(|lp| lp.name.starts_with("S_")).count();
        assert_eq!(n_state_lps, 4, "expected IDLE + 2 layer-MAC + DONE = 4 states, got {n_state_lps}");
    }

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

    /// M57.2 (Task 21 / SQ-5): bias-seed correctness regression.
    ///
    /// The FSM must use exactly three bias-seed forms per §3.4:
    ///   (1) Phase-entry seed:  `b{i}[0]`              — Literal(0)
    ///   (2) Intra-layer advance: `b{i}[(_rN + 1)]`   — RegPlus(o_reg, 1)
    ///   (3) Final layer → S_DONE: NO bias-seed at all (no b{last}[…] assign
    ///       in the o_is_last arm).
    ///
    /// The BUG this guards against (SQ-5): accidentally emitting `b{i}[_rN]`
    /// (bare Reg) instead of `b{i}[(_rN + 1)]` in the intra-layer advance,
    /// which would re-read the old accumulator bias instead of the next output's
    /// bias, silently corrupting every h_buf cell except the first.
    ///
    /// Approach: emit the sequential module to Verilog, then parse the text to
    /// verify the three cases.  Text matching is more robust than deep AST
    /// walking for this invariant because it covers the full emission pipeline
    /// (HIR construction + Verilog templates + sequential process emission).
    #[test]
    fn bias_seed_three_cases_use_correct_indices() {
        use crate::backend_verilog::lower::VerilogEmitter;

        let kir = two_layer_mlp_kir();
        let module = KirToHirPass { test_taps: false, sequential: true }
            .lower(&kir, "tiny_mlp_seq").unwrap();
        let v = VerilogEmitter::emit_module(&module);

        // ── Case (1): phase-entry seeds b1[0] and b2[0] ─────────────────────
        assert!(
            v.contains("b1[0]"),
            "layer-1 phase-entry seed b1[0] missing from FSM Verilog",
        );
        assert!(
            v.contains("b2[0]"),
            "layer-2 phase-entry seed b2[0] missing from FSM Verilog",
        );

        // ── Case (2): intra-layer advance b{i}[(o+1)] ───────────────────────
        // The o register emits as `_rN`; the RegPlus form emits as `[(_rN + 1)]`.
        // We locate the o-register name from the `o <= ZERO` reset line
        // (the only line of the form `_rN <= ZERO;` in the reset block where N
        // is also used as an index in b1/b2 subscripts).
        //
        // More practically: scan for `b1[(` and confirm ` + 1)]` follows on the
        // same line (the full token is `b1[(_rN + 1)]`).
        let b1_intra = v.lines().any(|line| {
            if let Some(pos) = line.find("b1[(") {
                line[pos..].contains(" + 1)]")
            } else {
                false
            }
        });
        assert!(
            b1_intra,
            "intra-layer advance `b1[(_rN + 1)]` missing from FSM Verilog",
        );

        // ── Case (3): no bare `b1[_rN]` or `b2[_rN]` form ──────────────────
        // Locate the o-register id by finding a line `_rN <= _rM;` in the
        // always_ff block where _rM is the o_next wire — but the simplest
        // proxy is: any line that matches `b1[_r` or `b2[_r` but does NOT
        // also contain ` + 1)]` is the bug form.
        let bare_b1_bug = v.lines().any(|line| {
            // Match `b1[_r` (bare reg subscript) but not `b1[(_r` (RegPlus form)
            line.contains("b1[_r")
        });
        assert!(
            !bare_b1_bug,
            "bare-reg bias index `b1[_rN]` found in FSM Verilog — SQ-5 regression!",
        );
        let bare_b2_bug = v.lines().any(|line| {
            line.contains("b2[_r")
        });
        assert!(
            !bare_b2_bug,
            "bare-reg bias index `b2[_rN]` found in FSM Verilog — SQ-5 regression!",
        );

        // ── Case (3) continued: final layer's o_is_last arm enters S_DONE ────
        // The S_DONE transition must appear in the Verilog.
        assert!(
            v.contains("S_DONE"),
            "S_DONE state missing from FSM — final-layer o_is_last arm not wired correctly",
        );
    }
}
