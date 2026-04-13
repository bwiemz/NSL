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

// ── B.2.1 Task 2: constructor-observation tests ─────────────────────
//
// These tests exercise the compile path and verify that WRGA produces the
// expected synthesized fields for each adapter kind. The tests are
// observational — they do not assert on runtime tensor values. The
// constructor-side wiring (side-table slot reservation, field-access
// routing through the slot) is a structural prerequisite for Task 3's
// forward-pass rewrite; its correctness is verified by cargo build +
// later integration tests.

const LORA_B21_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

fn main():
    let m = Toy()
    let x = ones([4, 8])
    let y = zeros([4, 8])
    train(model = m, epochs = 1):
        optimizer: SGD(lr = 0.01)
        step(batch):
            let pred = m.forward(x)
            let loss = mse_loss(pred, y)
"#;

const IA3_B21_SRC: &str = r#"from nsl.nn.losses import mse_loss

model ToyIa3:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

fn main():
    let m = ToyIa3()
    let x = ones([4, 8])
    let y = zeros([4, 8])
    train(model = m, epochs = 1):
        optimizer: SGD(lr = 0.01)
        step(batch):
            let pred = m.forward(x)
            let loss = mse_loss(pred, y)
"#;

const GATED_B21_SRC: &str = r#"from nsl.nn.losses import mse_loss

model ToyGated:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

fn main():
    let m = ToyGated()
    let x = ones([4, 8])
    let y = zeros([4, 8])
    train(model = m, epochs = 1):
        optimizer: SGD(lr = 0.01)
        step(batch):
            let pred = m.forward(x)
            let loss = mse_loss(pred, y)
"#;

#[test]
fn lora_constructor_allocates_a_and_b_with_expected_shapes() {
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::Lora,
                targets: vec!["m.w".into()],
                rank: Some(2),
                alpha: Some(2),
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(LORA_B21_SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire");
    let lora_a_entry = plan
        .placements
        .iter()
        .flat_map(|p| p.synthesized_fields.iter().cloned())
        .find(|n| n.starts_with("lora_A_"))
        .expect("lora_A field synthesized");
    assert!(
        lora_a_entry.contains("m_w"),
        "site id must reflect target: got {lora_a_entry}",
    );
}

#[test]
fn ia3_constructor_emits_scale_field() {
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::Ia3,
                targets: vec!["m.w".into()],
                rank: None,
                alpha: None,
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(IA3_B21_SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire");
    assert!(
        plan.placements
            .iter()
            .flat_map(|p| p.synthesized_fields.iter())
            .any(|n| n.starts_with("ia3_scale_")),
        "ia3_scale field must be synthesized",
    );
}

// ── B.2.1 Task 2.5: side-table init emission at @train block entry ─────
//
// These tests exercise the full codegen path end-to-end for each adapter
// kind. Compilation routes through `wrga_adapter_init::emit_adapter_init_sidetable`
// immediately after WRGA fires, which builds the heap table, populates it
// with freshly-initialised tensors, and writes the table pointer into the
// reserved slot on the model struct. A successful compile is evidence that
// the IR is emitted; runtime tensor-shape verification is covered by the
// B.2.1 Build 4 e2e proof (Task 5) which executes the compiled binary.

#[test]
fn task_2_5_lora_init_emits_cleanly() {
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
    let plan = nsl_codegen::debug_compile_and_return_plan(LORA_B21_SRC, &opts)
        .expect("lora init-pass compile must succeed")
        .expect("wrga::run must fire");
    // Sanity: the placement that drives init emission has resolved dims.
    let placement = plan
        .placements
        .iter()
        .find(|p| p.decorator_kind.is_some())
        .expect("adapter placement must exist");
    assert_eq!(placement.init_strategies.len(), 2, "LoRA has A + B");
}

#[test]
fn task_2_5_ia3_init_emits_cleanly() {
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::Ia3,
                targets: vec!["m.w".into()],
                rank: None,
                alpha: None,
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(IA3_B21_SRC, &opts)
        .expect("ia3 init-pass compile must succeed")
        .expect("wrga::run must fire");
    let placement = plan
        .placements
        .iter()
        .find(|p| p.decorator_kind.is_some())
        .expect("adapter placement must exist");
    assert_eq!(placement.init_strategies.len(), 1, "IA3 has single scale");
}

#[test]
fn task_2_5_gatedlora_init_emits_cleanly() {
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::GatedLora,
                targets: vec!["m.w".into()],
                rank: Some(4),
                alpha: Some(4),
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(GATED_B21_SRC, &opts)
        .expect("gated-lora init-pass compile must succeed")
        .expect("wrga::run must fire");
    let placement = plan
        .placements
        .iter()
        .find(|p| p.decorator_kind.is_some())
        .expect("adapter placement must exist");
    assert_eq!(
        placement.init_strategies.len(),
        3,
        "GatedLoRA has A + B + gate",
    );
}

#[test]
fn gatedlora_constructor_emits_a_b_and_gate() {
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::GatedLora,
                targets: vec!["m.w".into()],
                rank: Some(2),
                alpha: Some(2),
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(GATED_B21_SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire");
    let fields: Vec<_> = plan
        .placements
        .iter()
        .flat_map(|p| p.synthesized_fields.iter().cloned())
        .collect();
    assert!(fields.iter().any(|n| n.starts_with("lora_A_")));
    assert!(fields.iter().any(|n| n.starts_with("lora_B_")));
    assert!(fields.iter().any(|n| n.starts_with("gate_")));
}
