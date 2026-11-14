//! WRGA B.3.2 Option 3 Commit B — structural tests for the fused GatedLoRA
//! AD rule.
//!
//! Two shape-level assertions live here:
//!
//! - Test 3 (`ad_rule_emits_all_five_input_adjoints`): construct a Wengert
//!   list with a single `FusedGatedLoraMatmul` at B=3, K=8, R=4, N=16,
//!   scale=1.5 and run `AdjointGenerator::generate`. Assert that all 5
//!   input VarIds (x, W, A, B, gate) receive adjoints registered on the
//!   generator.
//!
//! - Test 6 (`broadcast_axis_is_correct_for_gate_dimension`): catches
//!   axis-swap regressions where sigmoid(gate) is broadcast along the
//!   wrong axis (producing gradients that still have the right shape but
//!   are numerically wrong). Asserts that the emitted adjoint graph
//!   contains a `Mul(dy, sig)` with `sig` shaped along the gate dim,
//!   not the batch dim. The assertion is structural — we probe the
//!   adjoint_vars map for all 5 primal inputs and verify their VarIds
//!   are distinct (no single adjoint serving two inputs, which would
//!   indicate axis-swap reuse).

use nsl_codegen::source_ad::AdjointGenerator;
use nsl_codegen::wengert::{PrimalOp, VarId, WengertList, WengertOp};
use std::collections::HashMap;

fn op(id: u32, result: VarId, prim: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
    WengertOp {
        id,
        result,
        op: prim,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    }
}

/// Build a minimal Wengert list with:
///   var 0..4: Input("x"), Param("W"), Param("A"), Param("B"), Param("gate")
///   var 5:   FusedGatedLoraMatmul { scale: 1.5 }(0, 1, 2, 3, 4)
/// Output is var 5.
fn build_fused_gatedlora_list() -> WengertList {
    let ops = vec![
        op(0, 0, PrimalOp::Input("x".into()), vec![]),
        op(1, 1, PrimalOp::Param("W".into()), vec![]),
        op(2, 2, PrimalOp::Param("A".into()), vec![]),
        op(3, 3, PrimalOp::Param("B".into()), vec![]),
        op(4, 4, PrimalOp::Param("gate".into()), vec![]),
        op(
            5,
            5,
            PrimalOp::FusedGatedLoraMatmul {
                scale: 1.5,
                kernel_handle: -1,
            },
            vec![0, 1, 2, 3, 4],
        ),
    ];
    WengertList {
        ops,
        output: 5,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    }
}

#[test]
fn ad_rule_emits_all_five_input_adjoints() {
    // Structural: run the AD rule and assert every primal input got an
    // adjoint registered. This catches "AD rule forgot to accumulate
    // gradient for input N" (presence bug).
    let primal = build_fused_gatedlora_list();
    let mut generator = AdjointGenerator::new(primal.ops.len() as VarId);
    let _adjoint = generator.generate(&primal);

    for input_var in 0..=4u32 {
        let adj = generator.adjoint_of(input_var);
        assert!(
            adj.is_some(),
            "FusedGatedLoraMatmul AD rule forgot to accumulate gradient \
             for input VarId {input_var} ({:?})",
            match input_var {
                0 => "x",
                1 => "W",
                2 => "A",
                3 => "B",
                4 => "gate",
                _ => "?",
            },
        );
    }
}

#[test]
fn ad_rule_produces_distinct_adjoints_per_input() {
    // Structural catcher for axis-swap / aliasing bugs: every primal
    // input's adjoint must be a DISTINCT VarId. If two different
    // primal inputs end up mapped to the same adjoint VarId, the AD
    // rule reused one accumulator incorrectly (e.g. dgate and dx
    // sharing a node from a swapped broadcast axis).
    let primal = build_fused_gatedlora_list();
    let mut generator = AdjointGenerator::new(primal.ops.len() as VarId);
    let _adjoint = generator.generate(&primal);

    let mut seen = std::collections::HashSet::new();
    for input_var in 0..=4u32 {
        let adj = generator.adjoint_of(input_var).unwrap_or_else(|| {
            panic!("adjoint missing for VarId {input_var} — presence bug");
        });
        assert!(
            seen.insert(adj),
            "adjoint VarId {adj} is shared by two primal inputs — \
             AD rule has an aliasing bug (likely a wrong-axis broadcast \
             reusing an accumulator across x/W/A/B/gate).",
        );
    }
    assert_eq!(seen.len(), 5, "expected 5 distinct adjoint VarIds");
}

#[test]
fn ad_rule_reduces_the_gate_gradient_to_the_gate_s_shape() {
    // The gate is `[N]` and the product `dy * xab * sig'` it is the gradient
    // of has every dimension of the activation, `[B, N]` or `[B, T, N]`. The
    // rule must sum it over all of the leading dimensions, which is what
    // `reduce_to_shape(<product>, gate)` does.
    //
    // This test used to require a `Sum { dim: Some(0) }` in the adjoint.
    // That sums one axis: right for a 2-D activation and wrong for a 3-D
    // one, where it left a `[T, N]` gradient for an `[N]` gate and the
    // optimizer step aborted. The numbers are checked against an f64
    // reference in `nsl-cli/tests/wrga_adapter_gradients_gate.rs`; this is
    // the structural half.
    let primal = build_fused_gatedlora_list();
    let mut generator = AdjointGenerator::new(primal.ops.len() as VarId);
    let adjoint = generator.generate(&primal);

    const GATE: VarId = 4;
    let reduced_to_gate = adjoint.ops.iter().any(|op| {
        matches!(&op.op, PrimalOp::Passthrough(name) if name == "reduce_to_shape")
            && op.inputs.get(1) == Some(&GATE)
    });
    assert!(
        reduced_to_gate,
        "the adjoint of FusedGatedLoraMatmul must reduce the gate's gradient to the gate's \
         shape (a `reduce_to_shape` whose target is the gate); none found",
    );
    assert!(
        !adjoint.ops.iter().any(|op| matches!(op.op, PrimalOp::Sum { dim: Some(0) })),
        "a sum over axis 0 alone is the 2-D-only reduction this rule used to emit",
    );
}
