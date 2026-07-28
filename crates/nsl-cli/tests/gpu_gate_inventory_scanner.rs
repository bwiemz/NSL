//! Edge-case gates for `scripts/gpu-gate-inventory.awk` — the classifier the
//! cert lane's whole inventory rides on.
//!
//! The `cpu-stub` class was reviewed with synthetic probe files, and the
//! probes found real holes the in-tree stubs don't exercise: a one-line
//! `#[ignore] #[cfg(not(feature = "cuda"))]` classified `gpu` (resurrecting
//! the permanent NOTFOUND the class exists to kill), and the mirror order
//! VANISHED from the inventory (the silent drop the scanner's header
//! forbids). Those probes are committed here so the holes stay closed; each
//! case names the review finding it pins.
//!
//! The probes live in `fixtures/gpu_gate_inventory/*.rs-probe` — NOT inline
//! raw strings, and NOT `.rs` files. The scanner is line-oriented with no
//! notion of Rust string literals, so inline probe text in this file was
//! itself inventoried as gates (two phantom `gpu` rows that could never
//! run — the exact NOTFOUND class this work removes), and the inventory
//! walks `find crates -name '*.rs'`, which the `.rs-probe` suffix evades.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Run the scanner exactly as `gpu-cert.sh inventory()` does on one probe
/// fixture and return the (fn, class) rows. `cuda_gated=1` mirrors the
/// shell's whole-file grep for files that mention `feature = "cuda"`
/// anywhere — which every file with a cfg-not stub does.
fn scan(fixture: &str) -> Vec<(String, String)> {
    let root = repo_root();
    let probe = root
        .join("crates/nsl-cli/tests/fixtures/gpu_gate_inventory")
        .join(fixture);
    assert!(probe.is_file(), "missing probe fixture {}", probe.display());
    let out = Command::new("awk")
        .arg("-v")
        .arg("cuda_gated=1")
        .arg("-f")
        .arg(root.join("scripts/gpu-gate-inventory.awk"))
        .arg(&probe)
        .output()
        .expect("awk is present on every machine that runs the cert lane");
    assert!(
        out.status.success(),
        "awk failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            assert!(cols.len() >= 3, "malformed row: {l}");
            (cols[1].to_string(), cols[2].to_string())
        })
        .collect()
}

fn class_of<'a>(rows: &'a [(String, String)], func: &str) -> &'a str {
    &rows
        .iter()
        .find(|(f, _)| f == func)
        .unwrap_or_else(|| panic!("fn {func} missing from inventory: {rows:?}"))
        .1
}

/// The two in-tree stub shapes: cfg-not above the attributes, with and
/// without a reason (the fp8_dispatcher and wrga forms).
#[test]
fn item_level_stubs_classify_cpu_stub() {
    let rows = scan("item_level_stubs.rs-probe");
    assert_eq!(class_of(&rows, "reasoned_stub"), "cpu-stub");
    assert_eq!(class_of(&rows, "bare_stub"), "cpu-stub");
}

/// Review finding M1: `#[ignore = "..."] #[cfg(not(...))]` on ONE line
/// classified `gpu` — the cfg was invisible to the start-anchored rule.
/// One-line stacks are inside the support envelope (finding F4 put them
/// there for `#[test] #[ignore]`).
#[test]
fn one_line_ignore_then_cfg_is_a_stub() {
    let rows = scan("one_line_ignore_then_cfg.rs-probe");
    assert_eq!(class_of(&rows, "one_liner"), "cpu-stub");
}

/// Review finding L3: the mirror order VANISHED — the cfg rule's `next`
/// swallowed the same-line `#[ignore]`, silently dropping the row. The
/// scanner's header forbids silent drops; the row must exist AND be a stub.
#[test]
fn one_line_cfg_then_ignore_does_not_vanish() {
    let rows = scan("one_line_cfg_then_ignore.rs-probe");
    assert_eq!(class_of(&rows, "mirror_one_liner"), "cpu-stub");
}

/// Containment, the direction that must NEVER break: a cfg-not stub earlier
/// in the file must not demote a later, genuinely runnable gate out of the
/// lane — that would silently delete coverage (the F1 defect inverted).
#[test]
fn the_flag_does_not_leak_onto_later_gates() {
    let rows = scan("flag_containment.rs-probe");
    assert_eq!(class_of(&rows, "stub"), "cpu-stub");
    assert_eq!(class_of(&rows, "real_gate"), "gpu");
}

/// The boundary rule clears on a non-ignored fn too: cfg-not gating a plain
/// (non-test) item must not mark the next ignored test.
#[test]
fn cfg_on_an_unignored_item_does_not_mark_the_next_gate() {
    let rows = scan("unignored_item.rs-probe");
    assert_eq!(class_of(&rows, "real_gate"), "gpu");
}

/// Review finding L4a: a multi-line block comment inside the attribute
/// stack must not clear the flag (only a real item boundary should).
#[test]
fn a_block_comment_in_the_stack_keeps_the_flag() {
    let rows = scan("block_comment.rs-probe");
    assert_eq!(class_of(&rows, "commented_stub"), "cpu-stub");
}

/// Review finding L4b: legal spacing variants of the cfg attribute. The
/// shell's whole-file grep still sets cuda_gated=1 for such a file, so a
/// missed variant classifies the stub into a RUN class -> NOTFOUND.
#[test]
fn spaced_cfg_variants_still_match() {
    let rows = scan("spaced_cfg.rs-probe");
    assert_eq!(class_of(&rows, "spaced_stub"), "cpu-stub");
    assert_eq!(class_of(&rows, "hash_spaced_stub"), "cpu-stub");
}
