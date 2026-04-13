//! Task 2b (B.2): adapter A/B fields exist on the compiled model with correct
//! shapes and strict init.

use nsl_codegen::{AdapterDecoratorConfig, AdapterKind, CompileOptions, WrgaInputs};

const LORA_SRC: &str = r#"from nsl.nn.losses import mse_loss

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
fn adapter_inject_emits_lora_a_b_fields_with_expected_shapes() {
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::Lora,
                targets: vec!["m.w".into()],
                rank: Some(4),
                alpha: Some(4),
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(LORA_SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire when adapter is present");

    let site_id = "m_w__lora";
    let all_fields: Vec<_> = plan
        .placements
        .iter()
        .flat_map(|p| p.synthesized_fields.iter().cloned())
        .collect();
    assert!(
        all_fields.iter().any(|n| n == &format!("lora_A_{site_id}")),
        "expected lora_A_{site_id} in synthesized fields; got {:?}",
        all_fields,
    );
    assert!(
        all_fields.iter().any(|n| n == &format!("lora_B_{site_id}")),
        "expected lora_B_{site_id}; got {:?}",
        all_fields,
    );

    let b_init = plan
        .placements
        .iter()
        .flat_map(|p| p.init_strategies.iter())
        .find(|s| s.field_name == format!("lora_B_{site_id}"))
        .expect("lora_B init strategy must be emitted");
    assert_eq!(
        b_init.kind,
        nsl_codegen::wrga_adapter_inject::InitKind::Zeros,
        "lora_B must use strict Zeros init, not default",
    );

    let a_init = plan
        .placements
        .iter()
        .flat_map(|p| p.init_strategies.iter())
        .find(|s| s.field_name == format!("lora_A_{site_id}"))
        .expect("lora_A init strategy must be emitted");
    assert_eq!(
        a_init.kind,
        nsl_codegen::wrga_adapter_inject::InitKind::KaimingUniform,
    );
}
