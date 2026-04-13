//! Task 1 (B.1): eliminate_by_backward_live must translate primal VarIds
//! to adjoint VarIds via the AdjointGenerator map before filtering.

use nsl_codegen::{CompileOptions, FreezeDecoratorConfig, WrgaInputs};

const SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([16, 16])
    b: Tensor = zeros([16])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w + self.b

fn main():
    let m = Toy()
    let x = ones([4, 16])
    let y = zeros([4, 16])
    train(model = m, epochs = 1):
        optimizer: SGD(lr = 0.01)
        step(batch):
            let pred = m.forward(x)
            let loss = mse_loss(pred, y)
"#;

#[test]
fn backward_live_filter_drops_adjoint_ops_for_frozen_param() {
    let opts = CompileOptions {
        source_ad: true,
        wrga_inputs: Some(WrgaInputs {
            freeze: vec![FreezeDecoratorConfig {
                include: vec!["m.w".into()],
                exclude: vec![],
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(SRC, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        analysis
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "input must type-check: {:?}",
        analysis.diagnostics
    );

    let plan = nsl_codegen::debug_compile_and_return_plan(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &opts,
    )
    .expect("compile must succeed")
    .expect("wrga::run must fire");

    assert!(
        plan.prune.stats.frozen_params > 0,
        "sanity: at least one param should be frozen, stats={:?}",
        plan.prune.stats,
    );

    let dropped = nsl_codegen::debug_last_adjoint_ops_dropped()
        .expect("after a compile with freeze, the dropped-count side channel should be populated");
    assert!(
        dropped > 0,
        "expected >0 adjoint ops dropped by backward_live filter; got {}",
        dropped,
    );
}
