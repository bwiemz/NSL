//! Task 4: `wrga::run` is actually invoked when a @train block compiles
//! and WrgaInputs has at least one freeze config.

use nsl_codegen::{CompileOptions, FreezeDecoratorConfig, WrgaInputs};

const TRAIN_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([16, 16])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

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
fn wrga_run_fires_for_train_block_with_freeze() {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(TRAIN_SRC, nsl_errors::FileId(0), &mut interner);
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

    let opts = CompileOptions {
        source_ad: true,
        wrga_inputs: Some(WrgaInputs {
            freeze: vec![FreezeDecoratorConfig {
                include: vec!["w".into()],
                exclude: vec![],
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let plan = nsl_codegen::debug_compile_and_return_plan(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &opts,
    )
    .expect("compile must succeed");
    assert!(
        plan.is_some(),
        "wrga::run should have fired for a @train block with non-empty WrgaInputs"
    );
    let plan = plan.unwrap();
    assert!(
        plan.prune.stats.frozen_params > 0
            || plan.prune.stats.gradient_targets > 0,
        "wrga plan should reflect at least one param classification, stats={:?}",
        plan.prune.stats
    );
}
