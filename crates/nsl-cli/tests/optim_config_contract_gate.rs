//! The optimizer/scheduler/callbacks section contract, end to end: the
//! section namespaces are CLOSED, and the refusal fires at `nsl check`
//! time AND on the run/build path (the loader emits the same semantic
//! diagnostics and refuses before codegen).
//!
//! The per-rule matrix (per-optimizer kwarg tables, literal/range rules,
//! scheduler names/kwargs/defaults, callback names, duplicate handling,
//! the Muon backfill) lives in nsl-semantic — this gate pins the CLI
//! surface: a typo'd optimizer kwarg that used to silently train at the
//! default lr now names itself on stderr, an unknown scheduler that used
//! to silently train at constant lr refuses, and a fully valid program
//! (int-literal scheduler steps included) still compiles, runs, and
//! actually applies the scheduler.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// `{sections}` is spliced (4-space indented) between the optimizer slot
/// and the step section.
fn fixture(sections: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([2, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = full([2, 2], 2.0)
let y = zeros([2, 2])
train(model = m, epochs = 2):
{sections}
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print("OPTIM_CONTRACT_DONE")
"#
    )
}

fn run_cmd(tag: &str, src: &str, args: &[&str]) -> (bool, String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_optimcfg_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl");
    let res = (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    );
    let _ = std::fs::remove_dir_all(&tmp);
    res
}

#[test]
fn a_typo_optimizer_kwarg_refuses_at_check_time_with_the_table_listed() {
    // The motivating case: AdamW(lrr=0.01) used to hit codegen's `_ => {}`
    // and silently train at the default lr=0.01.
    let src = fixture("    optimizer: AdamW(lrr = 0.01)");
    let (ok, _stdout, stderr) = run_cmd("typo_check", &src, &["check"]);
    assert!(!ok, "lrr=0.01 must fail nsl check:\n{stderr}");
    assert!(
        stderr.contains("unknown AdamW kwarg 'lrr'"),
        "the refusal must name the kwarg:\n{stderr}"
    );
    assert!(
        stderr.contains("AdamW accepts lr="),
        "the refusal must list the per-optimizer table:\n{stderr}"
    );
}

#[test]
fn a_typo_optimizer_kwarg_refuses_on_the_run_path_before_codegen() {
    let src = fixture("    optimizer: AdamW(lrr = 0.01)");
    let (ok, _stdout, stderr) = run_cmd("typo_run", &src, &["run", "--source-ad"]);
    assert!(!ok, "lrr=0.01 must fail nsl run:\n{stderr}");
    assert!(
        stderr.contains("unknown AdamW kwarg 'lrr'"),
        "the run path must carry the same refusal:\n{stderr}"
    );
    // The codegen backstop prefixes its span-less error with
    // "optimizer config refused:" — its absence pins that the semantic
    // layer refused FIRST ("refuses before codegen" is the property).
    assert!(
        !stderr.contains("optimizer config refused:"),
        "the codegen backstop fired — the semantic layer should have \
         refused first:\n{stderr}"
    );
}

#[test]
fn an_unknown_scheduler_name_refuses_instead_of_constant_lr() {
    // Previously: fell through the lowering's name map to
    // `_ => base_lr_val // fallback: no change` — compiled clean and
    // silently trained at CONSTANT lr.
    let src = fixture(
        "    optimizer: AdamW(lr = 0.001)\n    scheduler: CosineAnnealing(t_max = 100)",
    );
    let (ok, _stdout, stderr) = run_cmd("sched_name", &src, &["check"]);
    assert!(!ok, "unknown scheduler must fail nsl check:\n{stderr}");
    assert!(
        stderr.contains("unknown scheduler 'CosineAnnealing'"),
        "the refusal must name the scheduler:\n{stderr}"
    );
    assert!(
        stderr.contains("warmup_cosine"),
        "the refusal must list the accepted names:\n{stderr}"
    );
}

#[test]
fn a_typo_scheduler_kwarg_refuses_instead_of_defaulting() {
    // Previously: pushed into scheduler_args, never matched by any
    // `.find()`, hardcoded default silently applied.
    let src = fixture(
        "    optimizer: AdamW(lr = 0.001)\n    scheduler: warmup_cosine(warmup_step = 5, total_steps = 100)",
    );
    let (ok, _stdout, stderr) = run_cmd("sched_kwarg", &src, &["check"]);
    assert!(!ok, "typo'd scheduler kwarg must fail:\n{stderr}");
    assert!(
        stderr.contains("unknown warmup_cosine kwarg 'warmup_step'"),
        "the refusal must name the kwarg:\n{stderr}"
    );
}

#[test]
fn an_unknown_callback_name_refuses() {
    // Previously collected and silently never emitted.
    let src = fixture(concat!(
        "    optimizer: AdamW(lr = 0.001)\n",
        "    callbacks:\n",
        "        on_stpe(step, loss):\n",
        "            let s = step\n",
    ));
    let (ok, _stdout, stderr) = run_cmd("cb_name", &src, &["check"]);
    assert!(!ok, "unknown callback must fail:\n{stderr}");
    assert!(
        stderr.contains("unknown callback 'on_stpe'"),
        "the refusal must name the callback:\n{stderr}"
    );
}

#[test]
fn a_valid_program_with_scheduler_and_callbacks_still_runs() {
    // Anti-overbreadth control, corpus-shaped: every AdamW kwarg,
    // int-literal scheduler steps (every committed scheduler call passes
    // ints), and an on_step callback. The callback printing per-step is
    // also the witness that validated callbacks still fire.
    let src = fixture(concat!(
        "    optimizer: AdamW(lr = 0.0003, weight_decay = 0.1, beta1 = 0.9, ",
        "beta2 = 0.95, eps = 1e-8)\n",
        "    scheduler: warmup_cosine(warmup_steps = 2, total_steps = 10, min_lr = 0.00003)\n",
        "    callbacks:\n",
        "        on_step(step, loss):\n",
        "            print(\"CB_STEP\")\n",
    ));
    let (ok, stdout, stderr) = run_cmd("valid", &src, &["run", "--source-ad"]);
    assert!(ok, "valid program failed:\n{stderr}");
    assert!(
        stdout.contains("OPTIM_CONTRACT_DONE"),
        "fixture did not finish:\n{stdout}"
    );
    assert!(
        stdout.contains("CB_STEP"),
        "the validated on_step callback must still fire:\n{stdout}"
    );
    assert!(
        !stderr.contains("optimizer config refused")
            && !stderr.contains("unknown"),
        "no contract diagnostics may fire on a valid program:\n{stderr}"
    );
}

#[test]
fn scheduler_sections_refuse_under_pipeline_instead_of_silently_dropping() {
    // Pre-contract the pipelined section dispatch ended in `_ => {}`:
    // a scheduler: under @pipeline compiled clean and trained at constant
    // lr with zero diagnostics. (This is a codegen-level composition
    // refusal — @pipeline resolution happens at lowering, like the
    // existing eval:/distribute: refusals on that path.)
    let src = format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    @pipeline(stages = 2)
    w: Tensor = ones([2, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = full([2, 2], 2.0)
let y = zeros([2, 2])
train(model = m):
    optimizer: AdamW(lr = 0.001)
    scheduler: warmup_cosine(warmup_steps = 2, total_steps = 10)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#
    );
    let (ok, _stdout, stderr) = run_cmd("pipe_sched", &src, &["build"]);
    assert!(!ok, "scheduler under @pipeline must refuse:\n{stderr}");
    assert!(
        stderr.contains("not supported on the @pipeline train path"),
        "expected the pipelined scheduler refusal:\n{stderr}"
    );
}
