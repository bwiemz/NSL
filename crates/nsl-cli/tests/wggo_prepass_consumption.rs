//! Integration pins for the WGGO-before-kernel-synthesis restructure.
//!
//! The pre-pass solves each train block's plan BEFORE
//! `compile_flash_attention_kernels`, so (a) the @pca admission gate can
//! consume packing preferences at synthesis time and (b) FASE recipe
//! selection / the per-param mode table — which read `self.wggo_overrides`
//! before the in-place planning site — finally see the plan for the FIRST
//! train block, and never a previous block's stale overrides.
//!
//! These tests pin the consumption machinery itself:
//!   1. a planned train block CONSUMES its pre-plan (fingerprint match, no
//!      duplicate solve — exactly one summary line);
//!   2. two train blocks each consume THEIR OWN pre-plan (per-block keying,
//!      the stale-leak regression);
//!   3. `--wggo off`/absent leaves the pipeline byte-identical (no pre-pass
//!      chatter at all).
//!
//! The admission-flip semantics (`plan_prefers_segment_id`) are pinned by
//! unit tests in `wggo_prepass.rs` — the ILP's mode choice for any given
//! fixture is an implementation detail this file deliberately doesn't pin.

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

fn run_nsl(source: &str, tag: &str, wggo: Option<&str>) -> String {
    run_nsl_env(source, tag, wggo, &[])
}

/// `run_nsl` + extra child env (per-process — tests run on parallel
/// threads, so no `std::env::set_var`).
fn run_nsl_env(source: &str, tag: &str, wggo: Option<&str>, envs: &[(&str, &str)]) -> String {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_wggo_prepass_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run", "--source-ad"]);
    if let Some(mode) = wggo {
        cmd.args(["--wggo", mode]);
    }
    cmd.arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let output = cmd.output().expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    std::fs::remove_dir_all(&tmp).ok();
    stderr
}

const ONE_BLOCK: &str = r#"from nsl.nn.losses import mse_loss

model TinyMlp:
    w: Tensor = ones([16, 16])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = TinyMlp()
let x = ones([4, 16])
let y = zeros([4, 16])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)
"#;

const TWO_BLOCKS: &str = r#"from nsl.nn.losses import mse_loss

model TinyMlp:
    w: Tensor = ones([16, 16])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

model OtherMlp:
    w2: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w2

let m = TinyMlp()
let x = ones([4, 16])
let y = zeros([4, 16])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)

let m2 = OtherMlp()
let x2 = ones([2, 8])
let y2 = zeros([2, 8])

train(model = m2, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m2.forward(x2)
        let loss = mse_loss(out, y2)
"#;

/// One train block: the pre-plan must be consumed (fingerprint match), the
/// solver must run exactly once (one summary line), and nothing may be
/// rejected.
#[test]
fn preplan_consumed_single_block() {
    let stderr = run_nsl(ONE_BLOCK, "single", Some("greedy"));
    assert!(
        stderr.contains("[wggo] consumed pre-solved plan (graph fingerprint match)"),
        "pre-plan was not consumed — fingerprint machinery regressed?\nstderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("wggo-preplan-rejected"),
        "pre-plan was rejected on a program where extraction inputs are \
         fully mirrorable:\n{stderr}"
    );
    let summaries = stderr.matches("[wggo] wggo[greedy]").count();
    assert_eq!(
        summaries, 1,
        "expected exactly one [wggo] summary (one consumption, no duplicate \
         solve), got {summaries}:\n{stderr}"
    );
}

/// Two train blocks: per-block keying — each block consumes its own plan.
/// Before the restructure the second block would have seen the FIRST
/// block's stale overrides at FASE-selection time; now the wrapper installs
/// each block's own pre-plan (or None) explicitly.
#[test]
fn preplan_consumed_per_block_two_blocks() {
    let stderr = run_nsl(TWO_BLOCKS, "two", Some("greedy"));
    let consumed = stderr
        .matches("[wggo] consumed pre-solved plan (graph fingerprint match)")
        .count();
    assert_eq!(
        consumed, 2,
        "expected both train blocks to consume their own pre-plan, got \
         {consumed}:\n{stderr}"
    );
    assert!(
        !stderr.contains("wggo-preplan-rejected"),
        "unexpected pre-plan rejection:\n{stderr}"
    );
    let summaries = stderr.matches("[wggo] wggo[greedy]").count();
    assert_eq!(summaries, 2, "expected two summaries:\n{stderr}");
}

/// Review-finding regression: a top-level `for` loop before the train block.
/// The in-place path registers the loop variable as a FuncState variable and
/// registration EAGERLY emits an `Input` leaf per registered symbol — the
/// pre-pass prefix walk doesn't mirror loop vars, so before the fingerprint
/// learned to ignore unreferenced Input leaves this shape was a GUARANTEED
/// false `graph_fingerprint_mismatch` on a semantically identical graph.
const FOR_LOOP_BLOCK: &str = r#"from nsl.nn.losses import mse_loss

model TinyMlp:
    w: Tensor = ones([16, 16])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = TinyMlp()
let x = ones([4, 16])
let y = zeros([4, 16])

for i in range(3):
    print(i)

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)
"#;

#[test]
fn preplan_consumed_with_top_level_for_loop() {
    let stderr = run_nsl(FOR_LOOP_BLOCK, "forloop", Some("greedy"));
    assert!(
        stderr.contains("[wggo] consumed pre-solved plan (graph fingerprint match)"),
        "top-level for-loop registration noise rejected the pre-plan again — \
         the unreferenced-Input-leaf filter regressed:\n{stderr}"
    );
    assert!(
        !stderr.contains("wggo-preplan-rejected"),
        "false fingerprint rejection:\n{stderr}"
    );
}

/// Review-finding regression: the step param shadowing a top-level `let` of
/// the same name (same interned Symbol). The pre-pass used to register it
/// twice — a duplicate Input leaf and a guaranteed rejection.
#[test]
fn preplan_consumed_when_step_param_shadows_top_level_let() {
    let program = ONE_BLOCK.replace(
        "let y = zeros([4, 16])",
        "let y = zeros([4, 16])\nlet batch = ones([1])",
    );
    let stderr = run_nsl(&program, "shadow", Some("greedy"));
    assert!(
        stderr.contains("[wggo] consumed pre-solved plan (graph fingerprint match)"),
        "step-param shadow double-registration rejected the pre-plan:\n{stderr}"
    );
    assert!(
        !stderr.contains("wggo-preplan-rejected"),
        "false fingerprint rejection:\n{stderr}"
    );
}

/// Review-finding regression: `--wggo off` + `--calibration-data` must not
/// print the "pre-pass deferred" note — off means chatter-free, and nothing
/// is being deferred when nothing would ever plan.
#[test]
fn wggo_off_with_calibration_data_is_chatter_free() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_wggo_prepass_offcal_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, ONE_BLOCK).unwrap();
    let calib = tmp.join("calib.bin");
    std::fs::write(&calib, [0u8; 128]).unwrap();

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "build", "--source-ad", "--wggo", "off"])
        .arg("--calibration-data")
        .arg(&calib)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl build");
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Exit status deliberately not asserted (the dummy calibration corpus
    // may fail the harness in either direction); the pin is the chatter.
    assert!(
        !stderr.contains("pre-pass deferred"),
        "--wggo off must not print the calibrated-importance deferral note:\n{stderr}"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

/// Without --wggo the pre-pass must not run at all — no plan lines, no
/// pre-pass chatter, byte-identical to the pre-restructure pipeline.
#[test]
fn no_wggo_no_prepass() {
    let stderr = run_nsl(ONE_BLOCK, "off", None);
    for needle in [
        "consumed pre-solved plan",
        "wggo-preplan-rejected",
        "[wggo] wggo[",
    ] {
        assert!(
            !stderr.contains(needle),
            "`{needle}` appeared without --wggo:\n{stderr}"
        );
    }
}

/// AdamW + grad_accumulation >= 2: a per-param FASE mode table IS emitted
/// from the pre-plan's overrides. When the in-place replan diverges on
/// fase_fused (forced via the test knob), the compile must HARD-REFUSE —
/// P0 item 1: never execute a mode table that encodes a stale plan.
const ADAMW_ACCUM_BLOCK: &str = r#"from nsl.nn.losses import mse_loss

model TinyMlp:
    w: Tensor = ones([16, 16])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = TinyMlp()
let x = ones([4, 16])
let y = zeros([4, 16])

train(model = m, epochs = 1, grad_accumulation = 2):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)
"#;

#[test]
fn stale_mode_table_hard_refuses() {
    let stderr = run_nsl_env(
        ADAMW_ACCUM_BLOCK,
        "stale_refuse",
        Some("greedy"),
        &[("NSL_WGGO_FORCE_STALE_TABLE", "1")],
    );
    assert!(
        stderr.contains("refusing to execute a stale mode table"),
        "expected the stale-mode-table hard refusal:\n{stderr}"
    );
}

/// Same forced divergence but NO mode table (SGD → no per-param FASE
/// table): nothing stale can execute, so the compile proceeds with the
/// loud note instead of refusing.
#[test]
fn stale_divergence_without_table_notes_and_proceeds() {
    let stderr = run_nsl_env(
        ONE_BLOCK,
        "stale_note",
        Some("greedy"),
        &[("NSL_WGGO_FORCE_STALE_TABLE", "1")],
    );
    assert!(
        !stderr.contains("refusing to execute a stale mode table"),
        "SGD (no mode table) must not hard-refuse:\n{stderr}"
    );
    assert!(
        stderr.contains("no per-param FASE mode table was emitted"),
        "expected the no-table divergence note:\n{stderr}"
    );
    // The run itself proceeds: the WGGO summary still prints.
    assert!(
        stderr.contains("[wggo] wggo[greedy]"),
        "compile did not proceed to the WGGO summary:\n{stderr}"
    );
}
