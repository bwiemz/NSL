//! TrainPlan step 4a (roadmap A1): `stmt_train::pass_order::PASS_ORDER`, the
//! train-block driver's planning steps as one table, held to the tree.
//!
//! The table is only worth having if it cannot drift from what the driver
//! does. Each test here pins one direction of that:
//!
//! - the driver makes every step's call, in table order, and each call is
//!   defined in the file the table names;
//! - the `PassScheduler::schedule` sites in the driver's files are exactly
//!   the table's `(module, pass)` pairs, and every registry pass that runs
//!   in the `TrainBlock` phase has a step;
//! - the table's order inverts none of the pass bus's `InvocationOrdered`
//!   edges, and the check bites when the order is reversed;
//! - the stages never go backwards, and no tape-reading pass comes before
//!   extraction.

use std::collections::BTreeSet;
use std::path::Path;

use nsl_codegen::pass_registry::{CompilePhase, TapeAccess, PASSES};
use nsl_codegen::stmt_train::pass_order::{
    bus_order_violations, bus_order_violations_in, first_invocations, TrainPassDecl, TrainStage, PASS_ORDER,
};

fn src(rel: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Drop `/* block */` and `//` line comments, so a commented-out call or
/// schedule site does not count as live.
fn code_only(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        rest = match rest[start + 2..].find("*/") {
            Some(end) => &rest[start + 2 + end + 2..],
            None => "",
        };
    }
    out.push_str(rest);
    out.lines()
        .map(|l| match l.find("//") {
            Some(i) if !l[..i].contains('"') => &l[..i],
            _ => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The call-site spelling of a step's entry: `name(` for a method or free
/// function, `Type::new(` for a constructor.
fn call(entry: &str) -> String {
    format!("{entry}(")
}

#[test]
fn the_driver_makes_every_call_in_table_order() {
    let driver = code_only(&src("stmt_train/driver.rs"));
    let mut at = 0;
    for s in PASS_ORDER {
        let needle = call(s.entry);
        let found = driver[at..].find(&needle).unwrap_or_else(|| {
            let anywhere = driver.find(&needle).map(|p| format!(" (it appears earlier, at byte {p})")).unwrap_or_default();
            panic!("{}: `{needle}` does not appear in driver.rs after the previous step{anywhere}", s.step)
        });
        at += found + needle.len();
    }
}

#[test]
fn every_entry_is_defined_where_the_table_says() {
    for s in PASS_ORDER {
        let module = code_only(&src(s.defined_in));
        let defined = match s.entry.split_once("::") {
            Some((ty, _)) => module.contains(&format!("pub struct {ty}")),
            None => module.contains(&format!("fn {}(", s.entry)) || module.contains(&format!("fn {}<", s.entry)),
        };
        assert!(defined, "{}: `{}` is not defined in {}", s.step, s.entry, s.defined_in);
    }
}

/// `(file, pass)` for every `.schedule("PASS"` call in `code`.
fn schedule_sites(file: &str, code: &str, out: &mut BTreeSet<(String, String)>) {
    let flat: String = code.split_whitespace().collect();
    let mut rest = flat.as_str();
    while let Some(i) = rest.find(".schedule(\"") {
        rest = &rest[i + ".schedule(\"".len()..];
        let end = rest.find('"').expect("closing quote");
        out.insert((file.to_string(), rest[..end].to_string()));
    }
}

/// The files the train-block driver's planning stretch lives in: the
/// driver's own directory and the admission module.
fn driver_files() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/stmt_train");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .expect("stmt_train/")
        .map(|e| e.expect("entry").file_name().into_string().expect("utf-8"))
        .filter(|n| n.ends_with(".rs"))
        .map(|n| format!("stmt_train/{n}"))
        .collect();
    files.push("stmt_admission.rs".to_string());
    files.sort();
    files
}

#[test]
fn the_scheduled_sites_are_exactly_the_tables_passes() {
    let mut in_tree = BTreeSet::new();
    for f in driver_files() {
        schedule_sites(&f, &code_only(&src(&f)), &mut in_tree);
    }
    let in_table: BTreeSet<(String, String)> =
        PASS_ORDER.iter().flat_map(|s| s.passes.iter().map(|p| (s.module.to_string(), p.to_string()))).collect();
    let unlisted: Vec<_> = in_tree.difference(&in_table).collect();
    let stale: Vec<_> = in_table.difference(&in_tree).collect();
    assert!(
        unlisted.is_empty() && stale.is_empty(),
        "scheduled in the tree but not in PASS_ORDER: {unlisted:?}; in PASS_ORDER but not scheduled there: {stale:?}"
    );
    // The pipelined train block keeps its own driver and schedules nothing.
    assert!(!in_tree.iter().any(|(f, _)| f == "stmt_train/pipelined.rs"));
}

#[test]
fn every_train_block_pass_has_a_step_and_every_step_pass_is_registered() {
    let named: BTreeSet<&str> = first_invocations(PASS_ORDER).into_iter().collect();
    for p in named.iter() {
        let d = PASSES.iter().find(|d| d.name == *p).unwrap_or_else(|| panic!("{p} is not in pass_registry::PASSES"));
        assert!(d.phases.contains(&CompilePhase::TrainBlock), "{p} runs in the train-block driver but does not declare TrainBlock");
    }
    for d in PASSES.iter().filter(|d| d.phases.contains(&CompilePhase::TrainBlock)) {
        assert!(named.contains(d.name), "{} declares the TrainBlock phase but no PASS_ORDER step runs it", d.name);
    }
}

#[test]
fn the_order_inverts_no_invocation_ordered_bus_edge() {
    assert!(bus_order_violations().is_empty(), "{:?}", bus_order_violations());
}

/// The bus check bites: reversed, the table puts CSHA and WRGA before WGGO,
/// their producer on the overrides channel.
#[test]
fn a_reversed_order_is_refused() {
    let reversed: Vec<TrainPassDecl> = PASS_ORDER.iter().rev().copied().collect();
    let v = bus_order_violations_in(&reversed);
    let pairs: BTreeSet<(&str, &str)> = v.iter().map(|(p, c, _)| (*p, *c)).collect();
    assert!(pairs.contains(&("WGGO", "CSHA")) && pairs.contains(&("WGGO", "WRGA")), "{v:?}");
}

#[test]
fn stages_never_go_backwards_and_the_tape_passes_follow_extraction() {
    for w in PASS_ORDER.windows(2) {
        assert!(w[0].stage <= w[1].stage, "{} ({:?}) after {} ({:?})", w[1].step, w[1].stage, w[0].step, w[0].stage);
    }
    let extract = PASS_ORDER.iter().position(|s| s.entry == "WengertExtractor::new").expect("extraction");
    let adjoint = PASS_ORDER.iter().position(|s| s.entry == "AdjointGenerator::new").expect("adjoint generation");
    assert_eq!(PASS_ORDER[extract].stage, TrainStage::Tape);
    assert!(PASS_ORDER[..extract].iter().all(|s| s.stage == TrainStage::PreTape));
    assert_eq!(PASS_ORDER[adjoint].stage, TrainStage::Adjoint);
    assert!(PASS_ORDER[..adjoint].iter().all(|s| s.stage != TrainStage::Adjoint));
    for s in PASS_ORDER.iter().filter(|s| s.stage == TrainStage::PreTape) {
        for p in s.passes {
            let d = PASSES.iter().find(|d| d.name == *p).expect("registered");
            assert!(matches!(d.tape, TapeAccess::None), "{} runs {p} before extraction, but {p} touches the tape", s.step);
        }
    }
}

#[test]
fn step_names_are_unique() {
    let names: BTreeSet<&str> = PASS_ORDER.iter().map(|s| s.step).collect();
    assert_eq!(names.len(), PASS_ORDER.len());
}
