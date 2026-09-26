//! Roadmap A1, `TrainPlan` step 2: the train block's *planning* modules read
//! facts, never Cranelift handles.
//!
//! `docs/superpowers/specs/2026-09-08-a1-train-plan-ir-design.md` splits the
//! driver into planning (data in, data out: the passes that decide what to
//! emit) and emission (the modules that hold a `FunctionBuilder`). A planning
//! module that takes a `Value` can only run where that value exists, which is
//! what keeps planning welded to emission. `plan_wggo` was the last one: it
//! took Muon's mode-table base `Value` only to ask whether it existed, and now
//! takes that fact as a `bool`.
//!
//! This gate reads the planning modules' source and refuses any non-comment
//! line that names a Cranelift type or crate. A new planning module belongs
//! in `PLANNING`; a module that genuinely needs the builder is emission and
//! does not.

const PLANNING: &[&str] = &[
    "plan.rs",
    "plan_wggo.rs",
    "plan_ccr.rs",
    "plan_wrga_cpdt.rs",
    "csla_precompute.rs",
    "adjoint_tape_opt.rs",
    "ccr_adjoint_frees.rs",
];

/// Names that only emission code has any reason to spell.
const HANDLE_NAMES: &[&str] = &["cranelift", "FunctionBuilder", "InstBuilder", "Value", "Variable", "cl_types"];

fn is_word(line: &str, name: &str) -> bool {
    line.match_indices(name).any(|(at, _)| {
        let before = line[..at].chars().next_back();
        let after = line[at + name.len()..].chars().next();
        let ident = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        !ident(before) && !ident(after)
    })
}

#[test]
fn planning_modules_name_no_cranelift_handles() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/stmt_train");
    let mut offences = Vec::new();
    for file in PLANNING {
        let src = std::fs::read_to_string(dir.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        for (n, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for name in HANDLE_NAMES {
                if is_word(code, name) {
                    offences.push(format!("{file}:{}: `{name}` in {}", n + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        offences.is_empty(),
        "planning modules must take facts, not Cranelift handles (TrainPlan step 2):\n{}",
        offences.join("\n")
    );
}

/// The gate is not vacuous: every listed module exists, and the scanner
/// finds a handle where one is spelled.
#[test]
fn the_scanner_sees_a_handle() {
    assert!(is_word("pub(crate) mode_table_base: Option<Value>,", "Value"));
    assert!(!is_word("let values = ValueList::new();", "Value"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/stmt_train");
    for file in PLANNING {
        assert!(dir.join(file).exists(), "{file} is listed but missing");
    }
}
