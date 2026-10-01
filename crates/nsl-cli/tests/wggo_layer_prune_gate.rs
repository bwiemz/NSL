//! WGGO whole-block layer prune, end to end through `nsl run`.
//!
//! Until this gate, WGGO's layer prune was unreachable from any real
//! program: the DP only offers `Prune` below an importance floor that
//! production never fills, every planner layer is a `blocks.N` whole block
//! that `wggo_prune` refused (`WholeBlockUnsupported`), and its parameter
//! matcher could not see the `m.` model-variable prefix source-AD names
//! carry. `--wggo-prune-layers` / `--wggo-layer-prune-fraction` now force the
//! decision, and the v2 chain-collapse executes it.
//!
//! The property is exact, not "the loss moved": every block of the fixture
//! has the SAME deterministic init, so a 4-block model with one block
//! collapsed to an identity is the 3-block model, op for op. Its loss
//! trajectory and its surviving blocks' weights must be BIT-IDENTICAL
//! (decimal-string equality) to a 3-block run that was never pruned, and the
//! pruned block's weights must come out of training untouched — SGD with no
//! weight decay, so a parameter with no gradient cannot move. Anti-vacuity:
//! the unpruned 4-block run must differ, or the fixture could not tell a
//! prune from a no-op.
//!
//! The refusals matter as much as the rewrite: a prune request that cannot
//! be honored must fail the compile, never train the full model while the
//! user believes layers were removed.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const EPOCHS: usize = 4;

/// `{N}` = number of blocks. Every `Blk()` gets the same init (no RNG in the
/// model), and `x`/`y` are drawn after it, so they are the same tensors for
/// every block count. The weight sums print before AND after training.
const NET: &str = r#"
from nsl.nn.losses import mse_loss

model Blk:
    wa: Tensor = ones([4, 4]) * 0.05
    wb: Tensor = ones([4, 4]) * 0.05

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + (x @ self.wa)
        return h + (h @ self.wb)

model Net:
    blocks: [Blk; {N}] = Blk()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
let x = randn([4, 4]) * 0.5
let y = randn([4, 4]) * 0.5

for b in m.blocks:
    print(b.wa.sum().item())
    print(b.wb.sum().item())

train(model=m, epochs=4):
    optimizer: SGD(lr=0.1)
    step(batch):
        let loss = mse_loss(m.forward(x), y)
    callbacks:
        on_step(step, loss):
            print(loss)

for b in m.blocks:
    print(b.wa.sum().item())
    print(b.wb.sum().item())
"#;

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

/// Run the `n_blocks` fixture with `flags`. `files` are written next to the
/// fixture first (so a flag can name them by relative path).
fn run(n_blocks: usize, flags: &[&str], files: &[(&str, Vec<u8>)]) -> Run {
    let root = repo_root();
    let tmp = tempfile::TempDir::new().expect("scratch dir");
    let src = tmp.path().join("net.nsl");
    std::fs::write(&src, NET.replace("{N}", &n_blocks.to_string())).expect("write fixture");
    for (name, bytes) in files {
        std::fs::write(tmp.path().join(name), bytes).expect("write side file");
    }
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .args(flags)
        .arg(&src)
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The fixture's printed numbers, split by role. Kept as the decimal TEXT
/// the program printed, so equality is bit equality.
#[derive(Debug)]
struct Trace {
    /// Per block, `[wa_sum, wb_sum]` before training.
    init: Vec<[String; 2]>,
    losses: Vec<String>,
    /// Per block, `[wa_sum, wb_sum]` after training.
    trained: Vec<[String; 2]>,
}

fn trace(r: &Run, n_blocks: usize) -> Trace {
    assert!(r.ok, "run failed:\nstdout:\n{}\nstderr:\n{}", r.stdout, r.stderr);
    let nums: Vec<String> = r
        .stdout
        .lines()
        .map(str::trim)
        .filter(|l| l.parse::<f64>().is_ok())
        .map(str::to_string)
        .collect();
    assert_eq!(
        nums.len(),
        4 * n_blocks + EPOCHS,
        "expected {n_blocks} blocks x 2 sums before and after plus {EPOCHS} losses:\n{}",
        r.stdout
    );
    let pairs = |s: &[String]| -> Vec<[String; 2]> {
        s.chunks(2).map(|c| [c[0].clone(), c[1].clone()]).collect()
    };
    Trace {
        init: pairs(&nums[..2 * n_blocks]),
        losses: nums[2 * n_blocks..2 * n_blocks + EPOCHS].to_vec(),
        trained: pairs(&nums[2 * n_blocks + EPOCHS..]),
    }
}

fn prune_lines(r: &Run) -> Vec<&str> {
    r.stderr.lines().filter(|l| l.starts_with("[prune]")).collect()
}

/// Pruning the blocks in `pruned` out of 4 must equal the
/// `4 - pruned.len()`-block model exactly, and leave the pruned blocks'
/// weights untouched.
fn assert_equals_smaller_model(pruned_run: &Run, pruned: &[usize]) {
    let p = trace(pruned_run, 4);
    let kept = 4 - pruned.len();
    // Same flags minus the prune: the ONLY difference is the block count.
    let reference = trace(&run(kept, &["--source-ad", "--wggo", "greedy"], &[]), kept);

    assert_eq!(
        p.losses, reference.losses,
        "loss trajectory with blocks {pruned:?} pruned must be BIT-IDENTICAL to the \
         {kept}-block model"
    );
    let survivors: Vec<&[String; 2]> = p
        .trained
        .iter()
        .enumerate()
        .filter(|(i, _)| !pruned.contains(i))
        .map(|(_, w)| w)
        .collect();
    let reference_trained: Vec<&[String; 2]> = reference.trained.iter().collect();
    assert_eq!(
        survivors, reference_trained,
        "the surviving blocks must train exactly like the {kept}-block model's blocks"
    );

    for (i, (before, after)) in p.init.iter().zip(&p.trained).enumerate() {
        if pruned.contains(&i) {
            // No gradient reaches a pruned block: SGD leaves it at its init.
            assert_eq!(after, before, "blocks.{i} was pruned, so training must not touch it");
        } else {
            // ...while every other block moved (the run really trained).
            assert_ne!(before, after, "blocks.{i} did not train at all");
        }
    }

    // Anti-vacuity: the UNpruned 4-block model trains differently, so the
    // equality above could not hold if the prune had silently not happened.
    let full = trace(&run(4, &["--source-ad", "--wggo", "greedy"], &[]), 4);
    assert_ne!(
        full.losses, p.losses,
        "the unpruned 4-block run matched the pruned one — the fixture cannot detect a no-op prune"
    );
}

#[test]
fn prune_layers_collapses_a_whole_block_bit_exactly() {
    let r = run(4, &["--source-ad", "--wggo", "greedy", "--wggo-prune-layers", "blocks.1"], &[]);
    assert!(r.ok, "pruned run failed:\n{}", r.stderr);

    let lines = prune_lines(&r);
    assert_eq!(lines.len(), 1, "expected exactly one [prune] line:\n{}", r.stderr);
    assert!(
        lines[0].contains(" name=blocks.1 role=Block applied=true closure_size=4 ops_deleted=6 "),
        "unexpected [prune] line (2 params + 2 matmuls, plus both residual Adds): {}",
        lines[0]
    );
    assert!(
        r.stderr.contains("[wggo] layer-prune: forcing Prune for blocks.1"),
        "missing the forced-prune report:\n{}",
        r.stderr
    );

    assert_equals_smaller_model(&r, &[1]);
}

#[test]
fn adjacent_blocks_prune_together_bit_exactly() {
    // blocks.1's output IS blocks.2's input: the two collapses share a stream
    // value, and each was validated against the unpruned list. The commit has
    // to resolve blocks.2's input through blocks.1's collapse, or blocks.3
    // would read a deleted value.
    let r = run(4, &["--source-ad", "--wggo", "greedy", "--wggo-prune-layers", "blocks.1,blocks.2"], &[]);
    assert!(r.ok, "pruned run failed:\n{}", r.stderr);
    let lines = prune_lines(&r);
    assert_eq!(lines.len(), 2, "expected two [prune] lines:\n{}", r.stderr);
    assert!(lines[0].contains(" name=blocks.1 role=Block applied=true "), "{}", lines[0]);
    assert!(lines[1].contains(" name=blocks.2 role=Block applied=true "), "{}", lines[1]);

    assert_equals_smaller_model(&r, &[1, 2]);
}

/// `.nslweights` (see `wggo_weight_analysis_nslweights.rs`): magic, u32 LE
/// version 1, u64 LE header size, JSON header, zero padding to a 64-byte
/// boundary, then raw f32 LE tensors at the header's data-relative offsets.
fn nslweights(tensors: &[(String, Vec<f32>)]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut params = Vec::new();
    for (name, values) in tensors {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        params.push(format!(
            r#"{{"name":"{name}","dtype":"f32","shape":[4,4],"offset":{},"nbytes":{}}}"#,
            data.len(),
            bytes.len()
        ));
        data.extend_from_slice(&bytes);
    }
    let header = format!(r#"{{"params":[{}]}}"#, params.join(","));
    let mut out = Vec::new();
    out.extend_from_slice(b"NSLW");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    while out.len() % 64 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&data);
    out
}

#[test]
fn layer_prune_fraction_prunes_the_lowest_magnitude_block() {
    // blocks.2's weights are 50x smaller than the rest: it is the one
    // weight-RMS block. floor(0.25 x 4) = 1 layer.
    let tensors: Vec<(String, Vec<f32>)> = (0..4)
        .flat_map(|b| {
            let v = if b == 2 { 0.001 } else { 0.05 };
            ["wa", "wb"].map(|p| (format!("blocks.{b}.{p}"), vec![v; 16]))
        })
        .collect();
    let r = run(
        4,
        &[
            "--source-ad",
            "--wggo",
            "greedy",
            "--wggo-weights",
            "w.nslweights",
            "--wggo-layer-prune-fraction",
            "0.25",
        ],
        &[("w.nslweights", nslweights(&tensors))],
    );
    assert!(r.ok, "fraction run failed:\n{}", r.stderr);

    let lines = prune_lines(&r);
    assert_eq!(lines.len(), 1, "exactly one block must be pruned:\n{}", r.stderr);
    assert!(
        lines[0].contains(" name=blocks.2 role=Block applied=true "),
        "the lowest-magnitude block is blocks.2: {}",
        lines[0]
    );
    assert!(
        r.stderr.contains("forcing Prune for blocks.2 (fraction 0.25 of 4 block layer(s)")
            && r.stderr.contains("blocks.2=0.0200"),
        "missing the ranking report:\n{}",
        r.stderr
    );

    assert_equals_smaller_model(&r, &[2]);
}

// ── Refusals: a request that cannot be honored fails the compile ──────────

fn assert_refused(r: &Run, needles: &[&str]) {
    assert!(!r.ok, "expected the compile to be refused; it succeeded:\n{}", r.stderr);
    for n in needles {
        assert!(r.stderr.contains(n), "refusal must mention {n:?}:\n{}", r.stderr);
    }
    assert!(prune_lines(r).is_empty(), "a refused run must apply no prune:\n{}", r.stderr);
}

#[test]
fn prune_layers_without_source_ad_is_refused() {
    let r = run(4, &["--wggo", "greedy", "--wggo-prune-layers", "blocks.1"], &[]);
    assert_refused(
        &r,
        &["--wggo-prune-layers / --wggo-layer-prune-fraction requires --source-ad"],
    );
}

#[test]
fn prune_layers_without_wggo_is_refused() {
    let r = run(4, &["--source-ad", "--wggo-prune-layers", "blocks.1"], &[]);
    assert_refused(
        &r,
        &["--wggo-prune-layers / --wggo-layer-prune-fraction requires --wggo <full|greedy|auto>"],
    );
}

#[test]
fn prune_layers_unknown_name_lists_the_known_layers() {
    let r = run(4, &["--source-ad", "--wggo", "greedy", "--wggo-prune-layers", "blocks.7"], &[]);
    assert_refused(
        &r,
        &[
            "--wggo-prune-layers: unknown layer `blocks.7`",
            "blocks.0 (Block), blocks.1 (Block), blocks.2 (Block), blocks.3 (Block)",
        ],
    );
}

#[test]
fn layer_prune_fraction_without_weights_is_refused() {
    let r = run(
        4,
        &["--source-ad", "--wggo", "greedy", "--wggo-layer-prune-fraction", "0.25"],
        &[],
    );
    assert_refused(&r, &["--wggo-layer-prune-fraction requires --wggo-weights"]);
}
