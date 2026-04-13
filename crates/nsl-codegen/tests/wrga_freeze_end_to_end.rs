//! Task 5: End-to-end @freeze elimination test.
//!
//! Compiles a real NSL source through the full pipeline (lex -> parse ->
//! semantic -> codegen) with a WrgaInputs that freezes `m.w`, and asserts:
//! 1. The WrgaPlan's `prune.pruned.var_names` contains a VarId whose name
//!    is `m.w`.
//! 2. That VarId is NOT in `plan.prune.backward_live`.
//! 3. `plan.prune.stats.frozen_params >= 1`.

use nsl_codegen::wrga::WrgaPlan;
use nsl_codegen::{CompileOptions, FreezeDecoratorConfig, WrgaInputs};

const TRAIN_SRC: &str = r#"from nsl.nn.losses import mse_loss

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

fn compile_with(opts: CompileOptions) -> Option<WrgaPlan> {
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
    nsl_codegen::debug_compile_and_return_plan(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &opts,
    )
    .expect("compile must succeed")
}

#[test]
fn freeze_eliminates_frozen_param_from_backward_live() {
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
    let plan = compile_with(opts).expect("wrga::run must fire for @train");

    // Find the VarId(s) whose var_names entry matches the frozen param.
    let frozen_vids: Vec<_> = plan
        .prune
        .pruned
        .var_names
        .iter()
        .filter(|(_, name)| *name == "m.w")
        .map(|(vid, _)| *vid)
        .collect();
    assert!(
        !frozen_vids.is_empty(),
        "expected at least one `m.w` entry in var_names; got {:?}",
        plan.prune.pruned.var_names,
    );

    for vid in &frozen_vids {
        assert!(
            !plan.prune.backward_live.contains(vid),
            "@freeze(include=m.w) should remove {:?} from backward_live; live={:?}",
            vid,
            plan.prune.backward_live,
        );
    }

    assert!(
        plan.prune.stats.frozen_params >= frozen_vids.len(),
        "stats.frozen_params={} but {} `m.w` VarIds should be frozen",
        plan.prune.stats.frozen_params,
        frozen_vids.len(),
    );
}


/// B.1 Task 5: Sanity counterpart to `freeze_eliminates_frozen_param_from_backward_live`.
///
/// When `WrgaInputs` is empty, `invoke_wrga_if_enabled` must return None and
/// `last_wrga_plan` stays unset.  We use a train-block-free source to sidestep
/// the stdlib-optimizer resolution gap tracked separately.
#[test]
fn no_wrga_inputs_skips_wrga_run_on_simple_compile() {
    const SRC: &str = r#"
model Toy:
    w: Tensor = ones([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

fn main():
    let m = Toy()
    let x = ones([4, 4])
    let _y = m.forward(x)
"#;

    let opts = nsl_codegen::CompileOptions {
        wrga_inputs: Some(nsl_codegen::WrgaInputs::default()),
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
    .expect("compile must succeed");
    assert!(plan.is_none(), "empty WrgaInputs must skip wrga::run");
}
