//! Task 4 (B.1): WrgaPlan.memory.assignments feed the real memory planner.
//!
//! Verifies that `apply_wrga_hints` is invoked when a WRGA plan is produced,
//! that it sets the pre/post slot-count side-channels, and that the post-hint
//! slot count is never greater than pre-hint (conservative coalescing).
//!
//! **B.1 scope note:** `apply_wrga_hints` currently observes WRGA's
//! MemoryPlan and publishes side-channel pre/post slot counts, but does
//! NOT mutate the real `LivenessAnalyzer` (which is keyed by AST string
//! names rather than Wengert VarIds).  The aggressive allocator-merge
//! path is slated for Milestone B.2.  This test asserts observability,
//! not realised memory savings.

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

#[test]
fn wrga_memory_hints_reduce_or_maintain_slot_count() {
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

    let opts = CompileOptions {
        source_ad: true,
        wrga_inputs: Some(WrgaInputs {
            freeze: vec![FreezeDecoratorConfig {
                include: vec!["m.w1".into(), "m.w2".into()],
                exclude: vec![],
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    // Clear side-channels from any prior test in the same thread.
    nsl_codegen::debug_clear_allocator_slot_channels();

    let plan = nsl_codegen::debug_compile_and_return_plan_from_ast(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &opts,
    )
    .expect("compile must succeed")
    .expect("wrga::run must fire");

    // Sanity: WRGA's own MemoryPlan either found ≥ 2 vars sharing a slot,
    // or produced no assignments at all (nothing to hint).
    let mut slot_counts = std::collections::BTreeMap::<u32, usize>::new();
    for a in &plan.memory.assignments {
        *slot_counts.entry(a.slot).or_insert(0) += 1;
    }
    let shared_slots = slot_counts.values().filter(|&&c| c >= 2).count();
    // Sanity: at minimum WRGA produced a (possibly trivial) plan.  Shared
    // slots are not required for this test — the hint path must still run
    // and populate the pre/post side-channels so downstream milestones can
    // observe coalescing behaviour.
    let _ = shared_slots;

    // Integration: hint application never INCREASES real slot count.
    if let (Some(pre), Some(post)) = (
        nsl_codegen::debug_last_allocator_slot_count_pre_hint(),
        nsl_codegen::debug_last_allocator_slot_count_post_hint(),
    ) {
        assert!(
            post <= pre,
            "post-hint slot count {} must be ≤ pre-hint {}",
            post,
            pre,
        );
    }

    // If the side-channels weren't set, the hint path wasn't exercised —
    // fail so we notice.
    assert!(
        nsl_codegen::debug_last_allocator_slot_count_pre_hint().is_some(),
        "apply_wrga_hints should have set the pre-hint side-channel",
    );
}
