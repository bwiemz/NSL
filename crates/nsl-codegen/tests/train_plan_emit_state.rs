//! Roadmap A1, `TrainPlan` step 3: the late emitters take the train block's
//! setup handles through `EmitState`, not one by one.
//!
//! `docs/superpowers/specs/2026-09-08-a1-train-plan-ir-design.md` splits a
//! train block into facts (`TrainPlan`) and handles (`EmitState`). Before
//! step 3, `OptimizerStepInputs`, `CslaWindowInputs`, `SchedulerStepInputs`
//! and `HealthHooksInputs` each re-listed the same setup handles:
//! `param_list`, the optimizer state lists, the loop variables and so on.
//! Those now live in `EmitState` (`src/stmt_train/emit_state.rs`), and each
//! `Inputs` struct keeps only what one micro-batch produces.
//!
//! This gate reads `EmitState`'s field names and refuses any of them as a
//! field of a late emitter's `Inputs` struct, so a setup handle cannot creep
//! back in beside `emit`.

use std::path::Path;

/// The late emitters and their `Inputs` structs.
const LATE: &[(&str, &str)] = &[
    ("optimizer_step.rs", "OptimizerStepInputs"),
    ("csla_window.rs", "CslaWindowInputs"),
    ("scheduler_step.rs", "SchedulerStepInputs"),
    ("health_hooks.rs", "HealthHooksInputs"),
];

fn src(file: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/stmt_train").join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The `pub(crate) name:` fields of `struct <name>` in `text`, comments
/// skipped.
fn fields(text: &str, name: &str) -> Vec<String> {
    let start = text
        .lines()
        .position(|l| l.trim_start().starts_with(&format!("pub(crate) struct {name}")))
        .unwrap_or_else(|| panic!("struct {name} not found"));
    text.lines()
        .skip(start + 1)
        .take_while(|l| l.trim() != "}")
        .filter_map(|l| {
            let l = l.split("//").next().unwrap_or("").trim();
            let rest = l.strip_prefix("pub(crate) ")?;
            Some(rest.split(':').next()?.trim().to_string())
        })
        .collect()
}

#[test]
fn late_emitters_take_setup_handles_only_through_emit_state() {
    let setup = fields(&src("emit_state.rs"), "EmitState");
    assert!(setup.len() >= 15, "EmitState lost its fields? {setup:?}");
    let mut offences = Vec::new();
    for (file, name) in LATE {
        let own = fields(&src(file), name);
        assert!(own.iter().any(|f| f == "emit"), "{name} does not take `emit: &EmitState`");
        assert!(own.iter().any(|f| f == "plan"), "{name} does not take `plan: &TrainPlan`");
        for f in &own {
            if setup.contains(f) {
                offences.push(format!("{file}: {name}.{f} is an EmitState field"));
            }
        }
    }
    assert!(offences.is_empty(), "setup handles belong in EmitState (TrainPlan step 3):\n{}", offences.join("\n"));
}

/// The scanner is not vacuous: it reads a struct's fields, doc comments
/// aside, and it sees an `EmitState` field where one is declared.
#[test]
fn the_scanner_reads_fields() {
    let text = "pub(crate) struct Demo<'a> {\n    /// doc: param_list\n    pub(crate) param_list: Value,\n    pub(crate) plan: &'a TrainPlan,\n}\n";
    assert_eq!(fields(text, "Demo"), ["param_list", "plan"]);
    let setup = fields(&src("emit_state.rs"), "EmitState");
    for handle in ["param_list", "state_list_1", "step_count_var", "lr_var", "accum_list"] {
        assert!(setup.iter().any(|f| f == handle), "{handle}");
    }
}
