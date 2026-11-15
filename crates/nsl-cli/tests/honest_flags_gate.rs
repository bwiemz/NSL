//! Flags that measured as doing nothing, or as costing what their help did not
//! say, must say so at compile time (models/benchmarks/MEMORY_OPTIONS_2026_11.md).
//!
//! * `--wggo-moment-precision` acts only through WGGO's moment-bit decision,
//!   which WGGO makes only with sensitivity data or under a memory budget. On
//!   the production recipes the binary built with the flag was byte-identical
//!   to the one without it, while the activation report said "witnessed". A
//!   train block it cannot change is now a compile error naming the switch
//!   that does work.
//! * `--zero-stage` shards ACROSS ranks; with one device it shards nothing
//!   (0 GiB saved at Coder-1B and -500M). Compiling it for one device warns.
//! * `--pretrain-optimized` is the measured production flag set: block
//!   checkpointing on, WGGO off (its `greedy` plan cost 12-18%).

use assert_cmd::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// Two `blocks.N` layers and gradient accumulation, so the FASE-Deferred plan
/// (the only one WGGO's moment bits can reach) and CCR both apply.
const SRC: &str = r#"model Block:
    w: Tensor = ones([8, 8]) * 0.1

    fn forward(self, h: Tensor) -> Tensor:
        return relu(h @ self.w) + h

model Net:
    blocks: [Block; 2] = Block()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
let x = ones([4, 8])

train(model = m, epochs = 2, grad_accumulation = 2):
    optimizer: AdamW(lr = 0.001, weight_decay = 0.0)
    step(batch):
        let out = m.forward(x)
        let loss = sum(out * out)
"#;

fn build(flags: &[&str]) -> Output {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("t.nsl");
    std::fs::write(&src, SRC).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(tmp.path())
        .arg("build")
        .arg(&src)
        .args(["--target", "cpu", "-o"])
        .arg(tmp.path().join("t.bin"))
        .args(flags);
    cmd.output().expect("spawn nsl")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn an_inert_wggo_moment_precision_is_a_compile_error() {
    // WGGO plans this block (greedy), but with no sensitivity data and no
    // budget it decides no reduced-precision moment: the flag can change
    // nothing.
    for flags in [
        &["--source-ad", "--wggo", "greedy", "--wggo-moment-precision"][..],
        // Without WGGO at all there is no decision for the flag to lower.
        &["--source-ad", "--wggo-moment-precision"][..],
    ] {
        let o = build(flags);
        let err = stderr(&o);
        assert!(!o.status.success(), "{flags:?} built; stderr:\n{err}");
        assert!(
            err.contains("--wggo-moment-precision has no effect on this train block"),
            "{flags:?}: not the inert-flag refusal:\n{err}"
        );
        assert!(
            err.contains("first_moment_dtype=\"bf16\""),
            "{flags:?}: the refusal must name the switch that works:\n{err}"
        );
    }
    // The same program builds without the flag: the refusal is about the
    // flag, not the program.
    let o = build(&["--source-ad", "--wggo", "greedy"]);
    assert!(o.status.success(), "the program itself must build:\n{}", stderr(&o));
}

#[test]
fn zero_stage_on_one_device_warns_that_it_does_nothing() {
    let o = build(&["--source-ad", "--zero-stage", "1"]);
    let err = stderr(&o);
    assert!(o.status.success(), "a one-device ZeRO build is still legal:\n{err}");
    assert!(
        err.contains("[zero] --zero-stage 1 with one device does nothing"),
        "no single-device warning:\n{err}"
    );
    // With two devices it shards, and there is nothing to warn about.
    let o = build(&["--source-ad", "--zero-stage", "1", "--devices", "2"]);
    let err = stderr(&o);
    assert!(o.status.success(), "{err}");
    assert!(!err.contains("with one device does nothing"), "warned at --devices 2:\n{err}");
}

/// The `[ccr]` decision lines of a build: what CCR checkpointed and how.
fn ccr_lines(o: &Output) -> Vec<String> {
    stderr(o).lines().filter(|l| l.starts_with("[ccr]")).map(str::to_string).collect()
}

#[test]
fn pretrain_optimized_is_the_production_flag_set() {
    const PRODUCTION: &[&str] = &[
        "--source-ad",
        "--checkpoint-blocks",
        "--checkpoint-selective",
        "--fuse-rmsnorm-backward",
        "--fuse-wgrad-accum",
    ];
    let bundle = build(&["--pretrain-optimized"]);
    let explicit = build(PRODUCTION);
    let full_recompute = build(&["--pretrain-optimized", "--checkpoint-blocks"]);
    for (name, o) in [("bundle", &bundle), ("explicit", &explicit), ("full", &full_recompute)] {
        assert!(o.status.success(), "{name} build failed:\n{}", stderr(o));
    }
    // The bundle checkpoints exactly as the spelled-out production flags do,
    // and a typed `--checkpoint-blocks` (full recompute) is honoured. The
    // anti-vacuity half: the two postures really do produce different plans
    // on this program, so the equality is not two empty lists.
    assert!(!ccr_lines(&bundle).is_empty(), "the bundle ran no CCR:\n{}", stderr(&bundle));
    assert_eq!(ccr_lines(&bundle), ccr_lines(&explicit));
    assert_ne!(
        ccr_lines(&full_recompute),
        ccr_lines(&bundle),
        "selective and full recompute planned identically, so this test cannot tell them apart"
    );
    assert!(!stderr(&bundle).contains("[wggo]"), "the bundle planned with WGGO:\n{}", stderr(&bundle));
}
