//! Milestone A, @fase fix: the decorator's validated config must REACH the
//! FASE planner — historically `validate_fase_decorator`'s return value was
//! discarded at its only call site (nsl-semantic/checker/stmt.rs), while the
//! `@pca` arm twelve lines below captured its config. Both knobs (`mode`,
//! `v_approx`) were hard-coded in codegen's `FaseConfig` construction.
//!
//! Witness: the `[fase] @fase decorator applied:` stderr line, emitted only
//! when the decorator is present — the control build must NOT carry it.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

const TEMPLATE: &str = r#"
model Tiny:
    w: Tensor = randn([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = randn([2, 4])
__DEC__train(model = m, epochs = 1, grad_accumulation = 4):
    optimizer: __OPT__(lr = 0.01)
    step(batch):
        let y = m.forward(x)
        let loss = (y * y).sum()
"#;

struct BuildOut {
    ok: bool,
    stderr: String,
}

fn build(tag: &str, decorator: &str, optimizer: &str) -> BuildOut {
    let tmp = std::env::temp_dir().join(format!("nsl_fase_gate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    let dec = if decorator.is_empty() { String::new() } else { format!("{decorator}\n") };
    std::fs::write(
        &prog,
        TEMPLATE.replace("__DEC__", &dec).replace("__OPT__", optimizer),
    )
    .unwrap();
    let exe = tmp.join("out.exe");
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["build", "-o", exe.to_str().unwrap()])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .output()
        .expect("spawn nsl build");
    let r = BuildOut {
        ok: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    };
    std::fs::remove_dir_all(&tmp).ok();
    r
}

const WITNESS: &str = "[fase] @fase decorator applied:";

/// Control: no decorator, no witness line — pins the differential's meaning.
#[test]
fn control_carries_no_decorator_witness() {
    let r = build("ctl", "", "AdamW");
    assert!(r.ok, "control failed:\n{}", r.stderr);
    assert!(
        !r.stderr.contains(WITNESS),
        "control unexpectedly shows the @fase witness:\n{}",
        r.stderr
    );
}

/// `mode = off` on AdamW: the planner would derive Deferred; the decorator
/// must force FullBuffer and the witness must cite the decorator as why.
#[test]
fn mode_off_forces_full_buffer() {
    let r = build("off", "@fase(mode = off)", "AdamW");
    assert!(r.ok, "build failed:\n{}", r.stderr);
    assert!(
        r.stderr.contains(WITNESS)
            && r.stderr.contains("mode=FullBuffer")
            && r.stderr.contains("@fase(mode = off)"),
        "mode=off must reach the planner and force FullBuffer:\n{}",
        r.stderr
    );
}

/// `v_approx = false` on AdamW flips the planner's own derivation to
/// FullBuffer — the second knob reaching the planner, with the planner's own
/// rationale (not the forcing path).
#[test]
fn v_approx_false_reaches_the_planner() {
    let r = build("vapprox", "@fase(v_approx = false)", "AdamW");
    assert!(r.ok, "build failed:\n{}", r.stderr);
    assert!(
        r.stderr.contains("mode=FullBuffer v_approx=false")
            && r.stderr.contains("v_t approximation disabled by config"),
        "v_approx=false must flow into FaseConfig.allow_v_approx:\n{}",
        r.stderr
    );
}

/// `mode = deferred` where feasible: witness shows Deferred.
#[test]
fn mode_deferred_is_honoured_when_feasible() {
    let r = build("defok", "@fase(mode = deferred)", "AdamW");
    assert!(r.ok, "build failed:\n{}", r.stderr);
    assert!(
        r.stderr.contains("mode=Deferred"),
        "feasible deferred request must plan Deferred:\n{}",
        r.stderr
    );
}

/// `mode = deferred` on Lion is infeasible: the compile must REFUSE with
/// requested-vs-derived, not silently downgrade (the precondition-refusal
/// doctrine this repo promoted to a rule after three instances).
#[test]
fn mode_deferred_on_lion_refuses() {
    let r = build("deflion", "@fase(mode = deferred)", "Lion");
    assert!(!r.ok, "infeasible deferred must fail the build:\n{}", r.stderr);
    assert!(
        r.stderr.contains("@fase(mode = deferred) cannot be honoured")
            && r.stderr.contains("Lion"),
        "refusal must name request and reason:\n{}",
        r.stderr
    );
}

/// The two @cpdt options with no consumer must refuse at check time — the
/// Phase 4 'unused options either work or are rejected' criterion.
#[test]
fn dead_cpdt_options_refuse() {
    for (tag, dec, needle) in [
        ("prec", "@cpdt(mode = off, precision = mixed)", "`precision` argument has no consumer"),
        ("tmem", "@cpdt(mode = off, target_memory = \"80%\")", "`target_memory` argument has no consumer"),
    ] {
        let r = build(tag, dec, "AdamW");
        assert!(
            !r.ok && r.stderr.contains(needle),
            "[{tag}] dead option must refuse with its typed message:\n{}",
            r.stderr
        );
    }
}
