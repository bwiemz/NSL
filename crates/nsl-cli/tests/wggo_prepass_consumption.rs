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
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_wggo_prepass_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run", "--source-ad"]);
    if let Some(mode) = wggo {
        cmd.args(["--wggo", mode]);
    }
    cmd.arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
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
