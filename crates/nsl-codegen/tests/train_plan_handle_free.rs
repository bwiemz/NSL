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
//!
//! Step 5 splits the rows that took the builder into a planning function and
//! an emitting one. Where both stay in one file (the plan is tied to the
//! emitter it feeds), `PLANNING_FNS` holds the planning function to the same
//! rule, over its own body.

const PLANNING: &[&str] = &[
    "plan.rs",
    "plan_wggo.rs",
    "plan_ccr.rs",
    "plan_wrga_cpdt.rs",
    "plan_csha_prune.rs",
    "csla_precompute.rs",
    "adjoint_tape_opt.rs",
    "ccr_adjoint_frees.rs",
];

/// Planning functions that share a file with the emitter they feed:
/// `(file, fn name)`.
const PLANNING_FNS: &[(&str, &str)] = &[
    ("primal_vars.rs", "plan_primal_facts"),
    ("primal_vars.rs", "plan_cpkd_report"),
    ("transient_arena_projection.rs", "plan_transient_arena_projection"),
    ("adapter_sites.rs", "plan_wrga_adapter_loads"),
    // The field walks the VarMap planning is built from (step 5b), beside the
    // loaders that replay them.
    ("../stmt.rs", "plan_nested_field"),
    ("../stmt.rs", "plan_source_ad_named_param"),
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

/// The lines of the `impl` method `name` in `src`: from its `fn` line to the
/// closing brace at the method's four-space indent. `None` if it is absent,
/// or if that brace is not where the method ends: a nested block closing at
/// four spaces would stop the scan early and let the rest of the body pass
/// unread, so the next non-blank line must be back at method level.
fn method_lines<'a>(src: &'a str, name: &str) -> Option<Vec<(usize, &'a str)>> {
    let sig = format!("fn {name}(");
    let lines: Vec<&str> = src.lines().collect();
    let start = lines.iter().position(|l| l.trim_start().starts_with("pub(crate) fn ") && l.contains(&sig))?;
    let end = (start..lines.len()).find(|&i| lines[i] == "    }")?;
    let next = lines[end + 1..].iter().find(|l| !l.trim().is_empty());
    let at_method_level = next.is_none_or(|l| l.starts_with('}') || (l.starts_with("    ") && !l.starts_with("     ")));
    at_method_level.then(|| (start..=end).map(|i| (i, lines[i])).collect())
}

fn handle_offences<'a>(label: &str, lines: impl Iterator<Item = (usize, &'a str)>) -> Vec<String> {
    let mut offences = Vec::new();
    for (n, line) in lines {
        let code = line.split("//").next().unwrap_or("");
        for name in HANDLE_NAMES {
            if is_word(code, name) {
                offences.push(format!("{label}:{}: `{name}` in {}", n + 1, line.trim()));
            }
        }
    }
    offences
}

#[test]
fn planning_functions_name_no_cranelift_handles() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/stmt_train");
    let mut offences = Vec::new();
    for (file, name) in PLANNING_FNS {
        let src = std::fs::read_to_string(dir.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        let body = method_lines(&src, name).unwrap_or_else(|| panic!("{file}: no `fn {name}`"));
        offences.extend(handle_offences(&format!("{file} ({name})"), body.into_iter()));
    }
    assert!(
        offences.is_empty(),
        "planning functions must take facts, not Cranelift handles (TrainPlan step 5):\n{}",
        offences.join("\n")
    );
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
    // The method extractor finds a body and stops at its end, and a handle in
    // that body is an offence; the emitter beside it is not scanned.
    let src = "impl X {\n    pub(crate) fn plan_a(&self) {\n        let v: Value = x;\n    }\n\n    pub(crate) fn emit_a(&self, b: &mut FunctionBuilder) {\n    }\n}\n";
    let body = method_lines(src, "plan_a").expect("plan_a is found");
    assert_eq!(body.len(), 3, "the body ends at the method's closing brace");
    assert_eq!(handle_offences("t", body.into_iter()).len(), 1);
    assert!(method_lines(src, "plan_b").is_none());
    // A block closing at the method's indent is not the method's end.
    let early = "impl X {\n    pub(crate) fn plan_c(&self) {\n        if x {\n    }\n        let v: Value = x;\n    }\n}\n";
    assert!(method_lines(early, "plan_c").is_none(), "an early close must not be taken as the end");
    assert!(is_word("pub(crate) mode_table_base: Option<Value>,", "Value"));
    assert!(!is_word("let values = ValueList::new();", "Value"));
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/stmt_train");
    for file in PLANNING {
        assert!(dir.join(file).exists(), "{file} is listed but missing");
    }
}
