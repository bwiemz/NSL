//! Item 8: periodic-checkpointing gate (`--checkpoint-stride`).
//!
//! The stride coalesces CCR block anchors into super-segments — saving only
//! every k-th block boundary and recomputing the span. Three guarantees:
//!   1. BIT-EXACT across strides on CPU (recompute is deterministic), with and
//!      without CSLA (`--layerwise-accum`).
//!   2. `auto` and `dp` actually search, and land on the stride / partition
//!      worked out by hand below from the fixture's literal shapes.
//!   3. On GPU, the Activations surface peak drops at the sweet-spot stride
//!      (the saved-boundary term, buffered per accumulation micro-batch, is the
//!      dominant surface). GPU tests are `#[ignore]`.
//!
//! # The numbers the search tests expect
//!
//! The fixture is 8 identical blocks at batch 2, sequence 64, width 32 and
//! hidden width 64, all f32. Per block:
//!
//!   - recomputed interiors: `n` (2·64·32·4 = 16 384 B), `u` (2·64·64·4 =
//!     32 768 B) and `f` (16 384 B): 65 536 B;
//!   - saved boundary: the block output, 16 384 B.
//!
//! The train block accumulates 4 micro-batches and `--layerwise-accum` buffers
//! the saved boundaries of all 4, so a partition into `m` spans whose longest
//! has `s` blocks projects to
//!
//!   peak = 4 · m · 16 384  +  s · 65 536 + (s − 1) · 16 384   bytes
//!
//! (the second term is one span's recompute: its interiors plus the `s − 1`
//! boundaries the span swallowed). For the uniform strides:
//!
//!   | stride | spans      | peak (bytes) | MiB  |
//!   |--------|------------|--------------|------|
//!   | 1      | 8 × 1      | 589 824      | 0.56 |
//!   | 2      | 4 × 2      | 409 600      | 0.39 |
//!   | 3      | 3, 3, 2    | 425 984      | 0.41 |
//!   | 4      | 2 × 4      | 442 368      | 0.42 |
//!   | 6      | 6, 2       | 606 208      | 0.58 |
//!   | 8      | 1 × 8      | 704 512      | 0.67 |
//!
//! Without a budget both searches minimise the peak. Over all partitions the
//! minimum is four spans of two (any `m` forces `s ≥ ⌈8/m⌉`; `m = 4, s = 2`
//! gives 409 600 and every other `m` gives more), so `auto` must choose
//! stride 2 and `dp` must keep anchors `[0, 2, 4, 6]`.
//!
//! With a 1 MiB budget every partition fits, and `dp` then minimises the
//! recompute cost: three launches per block plus one per swallowed boundary,
//! which is least when nothing is swallowed. It must keep all 8 anchors.
//! `auto` takes the smallest stride that fits, which is 1.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn repo_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn fixture() -> String {
    std::fs::read_to_string(
        repo_root().join("crates/nsl-cli/tests/fixtures/ccr_periodic_stride.nsl"),
    )
    .expect("read fixture")
}

/// Run `source` under source AD with the given extra args. `gpu` moves the
/// model to the device (the `# GPU_PLACEMENT` marker) and needs a `cuda`
/// build of the test binary.
fn run_source(source: &str, gpu: bool, extra: &[&str]) -> Run {
    let mut src = source.to_string();
    if gpu {
        assert!(src.contains("# GPU_PLACEMENT"));
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ccr_periodic_stride.nsl");
    std::fs::write(&path, &src).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg("--source-ad");
    if !gpu {
        cmd.args(["--target", "cpu"]);
    }
    cmd.args(extra).arg(&path);
    let out = cmd.output().expect("spawn nsl run");
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn run(gpu: bool, extra: &[&str]) -> Run {
    run_source(&fixture(), gpu, extra)
}

/// The loss lines between the LOSS_STREAM markers, joined — the parity key.
/// Panics on an empty stream or a non-finite loss: two empty streams, or two
/// streams of `NaN`, compare equal.
fn loss_stream(stdout: &str) -> String {
    let after = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .map(|(_, r)| r)
        .unwrap_or("");
    let inner = after
        .split_once("LOSS_STREAM_END")
        .map(|(l, _)| l)
        .unwrap_or("");
    let stream = inner
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!stream.is_empty(), "no loss stream produced:\n{stdout}");
    let lower = stream.to_ascii_lowercase();
    assert!(
        !lower.contains("nan") && !lower.contains("inf"),
        "non-finite loss in the stream:\n{stream}"
    );
    stream
}

fn act_peak(stdout: &str) -> Option<u64> {
    let after = stdout.split_once("ACT_PEAK_BEGIN").map(|(_, r)| r)?;
    let inner = after.split_once("ACT_PEAK_END").map(|(l, _)| l)?;
    inner
        .lines()
        .map(str::trim)
        .find_map(|l| l.parse::<u64>().ok())
}

/// The stride-1 reference under CSLA, which every search result must match
/// bit for bit.
fn stride_one_reference() -> String {
    let r = run(
        false,
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--checkpoint-stride",
            "1",
        ],
    );
    assert!(r.ok, "stride-1 run failed:\n{}", r.stderr);
    loss_stream(&r.stdout)
}

#[test]
fn periodic_stride_is_bit_exact_across_strides_cpu() {
    // CSLA active (the regime the stride targets): stride 1/2/4 must agree.
    let reference = stride_one_reference();
    for k in ["2", "4"] {
        let r = run(
            false,
            &[
                "--checkpoint-blocks",
                "--layerwise-accum",
                "--checkpoint-stride",
                k,
            ],
        );
        assert!(r.ok, "stride {k} run failed:\n{}", r.stderr);
        assert!(
            r.stderr
                .contains(&format!("periodic checkpointing: stride {k}")),
            "stride {k} did not coalesce, so the comparison is stride 1 with itself:\n{}",
            r.stderr
        );
        assert_eq!(
            loss_stream(&r.stdout),
            reference,
            "stride {k} diverged from stride 1 — recompute must be bit-exact"
        );
    }
}

#[test]
fn periodic_stride_emits_coalescing_note() {
    let r = run(false, &["--checkpoint-blocks", "--checkpoint-stride", "2"]);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    assert!(
        r.stderr
            .contains("periodic checkpointing: stride 2 → 4 CCR super-segment(s)"),
        "expected the coalescing note for 8 blocks at stride 2 on stderr:\n{}",
        r.stderr
    );
}

/// `auto` without a budget takes the stride with the smallest projected
/// peak. By the table in the module header that is stride 2, at 0.39 MiB
/// against 0.56 MiB for stride 1.
#[test]
fn auto_stride_chooses_the_minimum_peak_stride() {
    let r = run(
        false,
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--checkpoint-stride",
            "auto",
        ],
    );
    assert!(r.ok, "auto run failed:\n{}", r.stderr);
    assert!(
        r.stderr
            .contains("--checkpoint-stride auto: chose stride 2 "),
        "auto must choose stride 2:\n{}",
        r.stderr
    );
    for candidate in [
        "k=1:0.56MiB",
        "k=2:0.39MiB",
        "k=3:0.41MiB",
        "k=4:0.42MiB",
        "k=8:0.67MiB",
    ] {
        assert!(
            r.stderr.contains(candidate),
            "candidate `{candidate}` missing from the decision line:\n{}",
            r.stderr
        );
    }
    assert!(
        r.stderr
            .contains("periodic checkpointing: stride 2 → 4 CCR super-segment(s)"),
        "the chosen stride was not applied:\n{}",
        r.stderr
    );
    assert_eq!(
        loss_stream(&r.stdout),
        stride_one_reference(),
        "auto (stride 2) diverged from stride 1 — recompute must be bit-exact"
    );
}

/// With a budget `auto` takes the smallest stride that fits. Stride 1
/// projects to 0.56 MiB, so a 1 MiB budget keeps stride 1.
#[test]
fn auto_stride_takes_the_smallest_stride_that_fits_the_budget() {
    let r = run(
        false,
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--checkpoint-stride",
            "auto",
            "--checkpoint-budget-mib",
            "1",
        ],
    );
    assert!(r.ok, "auto run failed:\n{}", r.stderr);
    assert!(
        r.stderr
            .contains("--checkpoint-stride auto: chose stride 1 "),
        "auto must keep stride 1 under a budget stride 1 fits:\n{}",
        r.stderr
    );
    // Under `auto` the budget caps the projected peak and is not spent a
    // second time on flipping tensors back to SAVE.
    assert!(
        !r.stderr.contains("flipped back to SAVE"),
        "the knapsack ran after the stride search:\n{}",
        r.stderr
    );
}

/// P5 item 21: without a budget the partition DP minimises the projected
/// peak: four spans of two (see the module header). The run must stay
/// bit-exact with stride 1, under CSLA where the window weighting matters.
#[test]
fn dp_stride_keeps_the_minimum_peak_partition_and_is_bit_exact() {
    let dp = run(
        false,
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--checkpoint-stride",
            "dp",
        ],
    );
    assert!(dp.ok, "dp run failed:\n{}", dp.stderr);
    assert!(
        dp.stderr
            .contains("--checkpoint-stride dp: kept 4 of 8 block anchors [0, 2, 4, 6]"),
        "dp must keep anchors [0, 2, 4, 6] of 8:\n{}",
        dp.stderr
    );
    assert!(
        !dp.stderr.contains("fallback"),
        "dp fell back to the uniform search:\n{}",
        dp.stderr
    );
    assert_eq!(
        loss_stream(&dp.stdout),
        stride_one_reference(),
        "dp partition diverged from stride 1 — recompute must be bit-exact"
    );
}

/// With a budget every partition fits (1 MiB against at most 0.67 MiB), the
/// DP minimises recompute cost instead, and the cheapest partition is the
/// one that swallows no boundary: all 8 anchors.
#[test]
fn dp_stride_with_a_loose_budget_keeps_every_anchor() {
    let dp = run(
        false,
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--checkpoint-stride",
            "dp",
            "--checkpoint-budget-mib",
            "1",
        ],
    );
    assert!(dp.ok, "dp+budget run failed:\n{}", dp.stderr);
    assert!(
        dp.stderr
            .contains("--checkpoint-stride dp: kept 8 of 8 block anchors [0, 1, 2, 3, 4, 5, 6, 7]"),
        "dp must keep all 8 anchors when every partition fits:\n{}",
        dp.stderr
    );
    assert!(
        !dp.stderr.contains("flipped back to SAVE"),
        "the knapsack ran after the partition search:\n{}",
        dp.stderr
    );
    assert_eq!(
        loss_stream(&dp.stdout),
        stride_one_reference(),
        "dp+budget diverged"
    );
}

/// The same model with nothing a size can be read from: no shape
/// annotation, the batch's shape taken at run time, and a loader whose last
/// batch may be short, so that its `[batch_size, seq_len]` is not proven.
/// (With `drop_last=true` the loader's literals alone size the whole model,
/// annotated or not.) The searches have no sizes to project from and must
/// refuse, not print a decision over zero bytes and exit 0.
#[test]
fn a_stride_search_without_static_shapes_is_refused() {
    let symbolic = fixture()
        .replace(
            "fn forward(self, h: Tensor<[2, 64, 32], f32>) -> Tensor<[2, 64, 32], f32>:",
            "fn forward(self, h: Tensor) -> Tensor:",
        )
        .replace("let n: Tensor<[2, 64, 32], f32> =", "let n =")
        .replace("let u: Tensor<[2, 64, 64], f32> =", "let u =")
        .replace("let f: Tensor<[2, 64, 32], f32> =", "let f =")
        .replace(
            "let flat_ids = input_ids.reshape([128])",
            "let s = input_ids.shape\n        let flat_ids = input_ids.reshape([s[0] * s[1]])",
        )
        .replace(
            "let h: Tensor<[2, 64, 32], f32> = emb.reshape([2, 64, 32])",
            "let h = emb.reshape([s[0], s[1], 32])",
        )
        .replace("drop_last=true", "drop_last=false");
    assert!(
        !symbolic.contains("Tensor<[2, 64")
            && symbolic.contains("s[0] * s[1]")
            && symbolic.contains("drop_last=false"),
        "the fixture changed shape; update the replacements above"
    );
    // The model itself still trains with a fixed stride.
    let fixed = run_source(
        &symbolic,
        false,
        &["--checkpoint-blocks", "--checkpoint-stride", "2"],
    );
    assert!(
        fixed.ok,
        "the symbolic fixture must run at a fixed stride:\n{}",
        fixed.stderr
    );
    loss_stream(&fixed.stdout);

    for mode in ["auto", "dp"] {
        let r = run_source(
            &symbolic,
            false,
            &["--checkpoint-blocks", "--checkpoint-stride", mode],
        );
        assert!(
            !r.ok,
            "--checkpoint-stride {mode} on symbolic shapes must fail the build:\n{}",
            r.stderr
        );
        assert!(
            r.stderr
                .contains(&format!("--checkpoint-stride {mode} cannot be honoured"))
                && r.stderr.contains("no static size"),
            "expected the refusal to name the flag and the reason:\n{}",
            r.stderr
        );
    }
}

/// GPU: the DP's min-peak partition (`[0, 2, 4, 6]`, a different plan from
/// stride 1) must not exceed the per-block activation peak.
#[test]
#[ignore = "requires CUDA GPU"]
fn dp_stride_activation_peak_not_worse_gpu() {
    let base = ["--checkpoint-blocks", "--layerwise-accum"];
    let run_with = |extra: &[&str]| -> Run {
        let mut args = base.to_vec();
        args.extend_from_slice(extra);
        let r = run(true, &args);
        assert!(r.ok, "gpu run {extra:?} failed:\n{}", r.stderr);
        r
    };
    let r1 = run_with(&["--checkpoint-stride", "1"]);
    let rdp = run_with(&["--checkpoint-stride", "dp"]);
    assert!(
        rdp.stderr.contains("kept 4 of 8 block anchors"),
        "dp did not choose a coalesced partition, so this compares stride 1 with itself:\n{}",
        rdp.stderr
    );
    let p1 = act_peak(&r1.stdout).unwrap_or_else(|| panic!("no ACT_PEAK:\n{}", r1.stdout));
    let pdp = act_peak(&rdp.stdout).unwrap_or_else(|| panic!("no ACT_PEAK:\n{}", rdp.stdout));
    assert!(
        pdp <= p1,
        "dp partition must not raise the activation peak: dp={pdp} vs stride1={p1}"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn periodic_stride_reduces_activation_peak_gpu() {
    let base = ["--checkpoint-blocks", "--layerwise-accum"];
    let peak_at = |k: &str| -> u64 {
        let mut args = base.to_vec();
        args.push("--checkpoint-stride");
        args.push(k);
        let r = run(true, &args);
        assert!(r.ok, "gpu stride {k} failed:\n{}", r.stderr);
        act_peak(&r.stdout).unwrap_or_else(|| panic!("no ACT_PEAK for stride {k}:\n{}", r.stdout))
    };
    let (p1, p2) = (peak_at("1"), peak_at("2"));
    // Stride 2 is the sweet spot on this 8-block / G=4 model: fewer saved
    // boundaries (× the accumulation window) outweigh the larger recompute span.
    assert!(
        p2 < p1,
        "stride 2 must lower the activation peak: stride2={p2} vs stride1={p1}"
    );
}
