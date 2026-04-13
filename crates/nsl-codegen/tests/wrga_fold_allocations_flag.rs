//! WRGA Milestone B.2 Task 3: `--wrga-fold-allocations` flag gates the
//! typed-key `consume_hints` path. Default-off preserves B.1's
//! observational behaviour; default-on bumps the `CONSUME_HINTS_CALLS`
//! side-channel counter.

use nsl_codegen::{CompileOptions, FreezeDecoratorConfig, WrgaInputs};

const SRC: &str = r#"from nsl.nn.losses import mse_loss

model Stack:
    w1: Tensor = ones([32, 32])
    w2: Tensor = ones([32, 32])
    w3: Tensor = ones([32, 32])

    fn forward(self, x: Tensor) -> Tensor:
        let a = x @ self.w1
        let b = a @ self.w2
        return b @ self.w3

fn main():
    let m = Stack()
    let x = ones([4, 32])
    let y = zeros([4, 32])
    train(model = m, epochs = 1):
        optimizer: SGD(lr = 0.01)
        step(batch):
            let pred = m.forward(x)
            let loss = mse_loss(pred, y)
"#;

fn compile_with_flag(flag: bool) -> Option<()> {
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            freeze: vec![FreezeDecoratorConfig {
                include: vec!["m.w1".into(), "m.w2".into()],
                exclude: vec![],
            }],
            ..Default::default()
        }),
        wrga_fold_allocations: flag,
        source_ad: true,
        ..Default::default()
    };
    let _plan = nsl_codegen::debug_compile_and_return_plan(SRC, &opts).ok()??;
    Some(())
}

#[test]
fn fold_flag_on_invokes_consume_hints() {
    nsl_codegen::debug_reset_consume_hints_calls();
    compile_with_flag(true).expect("compile with flag on must succeed");
    let calls = nsl_codegen::debug_last_consume_hints_calls();
    assert!(
        calls.unwrap_or(0) >= 1,
        "consume_hints must run when --wrga-fold-allocations is on (got {:?})",
        calls,
    );
}

#[test]
fn fold_flag_off_preserves_observational_path() {
    nsl_codegen::debug_reset_consume_hints_calls();
    compile_with_flag(false).expect("compile with flag off must succeed");
    // Flag-off must NOT invoke consume_hints; observational apply_wrga_hints
    // runs instead (verified by the existing wrga_memory_hints test).
    assert_eq!(
        nsl_codegen::debug_last_consume_hints_calls(),
        None,
        "consume_hints must not run when flag is off",
    );
}
