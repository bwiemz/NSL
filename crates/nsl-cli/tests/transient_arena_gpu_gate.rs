//! Milestone C · p2 — transient-memory arena gate.
//!
//! The arena planner (`nsl_codegen::transient_arena`) is a compile-time
//! analysis: over the final forward+adjoint Wengert tape it computes the peak
//! number of transient tensors ever simultaneously live — the minimum arena
//! slot count, and the surface the M36 slab planner (AST, forward-only) never
//! sees. This gate compiles a real packed-GQA LM on the GPU with the arena
//! report enabled and asserts:
//!
//!   1. The analysis runs end-to-end on a real source-AD + CCR training graph
//!      and emits a well-formed `[arena]` report.
//!   2. The report is internally consistent: many transient tensors collapse to
//!      a strictly smaller peak-concurrency slot count (a real reduction).
//!   3. Enabling the report is loss-neutral — it is pure analysis and must not
//!      perturb the compiled program (bit-identical loss stream on/off).
//!
//! (1)+(3) are the load-bearing guarantees; (2) is asserted as an invariant
//! (peak < transients), not a brittle exact count, since the numbers move as
//! the compiler evolves.
//!
//! GPU-only, `#[ignore]`. Run:
//!   cargo test -p nsl-cli --features cuda --test transient_arena_gpu_gate \
//!     -- --ignored --test-threads=1

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

/// Run the fixture; return (stdout, stderr, success). `arena_report` toggles
/// `NSL_ARENA_REPORT=1`.
fn run(arena_report: bool) -> (String, String, bool) {
    let root = repo_root();
    let fixture = root.join("crates/nsl-cli/tests/fixtures/long_run_drift.nsl");
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--",
            "run",
            "--source-ad",
            "--deterministic",
            "--checkpoint-blocks",
        ])
        .arg(&fixture)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if arena_report {
        cmd.env("NSL_ARENA_REPORT", "1");
    }
    let out = cmd.output().expect("spawn nsl run");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

fn loss_stream(stdout: &str) -> String {
    let mut s = String::new();
    let mut inside = false;
    for line in stdout.lines() {
        match line.trim() {
            "LOSS_STREAM_BEGIN" => inside = true,
            "LOSS_STREAM_END" => inside = false,
            l if inside => {
                s.push_str(l);
                s.push('\n');
            }
            _ => {}
        }
    }
    s
}

/// Pull the first integer that follows `key` on the line containing `key`.
fn first_int_after(stderr: &str, key: &str) -> Option<u64> {
    let line = stderr.lines().find(|l| l.contains(key))?;
    let idx = line.find(key)? + key.len();
    parse_leading_int(&line[idx..])
}

/// Pull the first integer anywhere on the line containing `key`.
fn first_int_on_line(stderr: &str, key: &str) -> Option<u64> {
    let line = stderr.lines().find(|l| l.contains(key))?;
    parse_leading_int(line)
}

fn parse_leading_int(s: &str) -> Option<u64> {
    s.chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

#[test]
#[ignore = "requires CUDA GPU (~2 runs of the packed-GQA fixture)"]
fn arena_report_is_consistent_and_loss_neutral() {
    let (out_on, err_on, ok_on) = run(true);
    assert!(ok_on, "run with arena report failed:\n{err_on}");

    // (1) A well-formed report was emitted on a real GPU training compile.
    assert!(
        err_on.contains("[arena]") && err_on.contains("Transient arena"),
        "arena report missing from stderr:\n{err_on}"
    );

    // (2) Internal consistency: transients >> peak concurrency (real reduction).
    let transients = first_int_on_line(&err_on, "transient tensor(s) over")
        .expect("could not parse transient count");
    let peak = first_int_after(&err_on, "peak concurrency:")
        .expect("could not parse peak concurrency");
    assert!(transients > 0, "no transients found");
    assert!(peak > 0, "peak concurrency should be positive");
    assert!(
        peak < transients,
        "arena must collapse {transients} transients to fewer than {transients} slots, got peak {peak}"
    );
    // The GQA LM churns hundreds of transients into a small resident set; assert
    // at least a 2x collapse as a floor (observed ~6.5x) without over-pinning.
    assert!(
        peak * 2 <= transients,
        "expected >=2x fewer live objects at peak: {transients} transients, peak {peak}"
    );

    // (3) Purity: the report must not change the compiled program.
    let (out_off, _err_off, ok_off) = run(false);
    assert!(ok_off, "run without arena report failed");
    assert_eq!(
        loss_stream(&out_on),
        loss_stream(&out_off),
        "enabling NSL_ARENA_REPORT changed the loss stream — the analysis is not pure"
    );

    eprintln!(
        "[arena-gate] OK: {transients} transients -> peak {peak} slots ({:.1}x), loss-neutral",
        transients as f64 / peak as f64
    );
}

/// Stage-2A: the hint plumbing (semantic types + initializer-derived model
/// field dims + adjoint mirroring) must produce a PARTIALLY QUANTIFIED
/// report on a plain CSLA fixture — real bytes for the sized subset, the
/// lower-bound caveat for the rest, and NOT the old "shapes runtime-valued"
/// fallback (which is exactly what reappears if any layer of the plumbing
/// silently breaks). CPU-only: the report is emitted at compile time.
#[test]
fn stage2a_report_quantifies_param_gradients_cpu() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_arena2a_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .unwrap()
    .replace(
        "CSLA_SAVE_PATH",
        &tmp.join("out.nslm").display().to_string().replace('\\', "/"),
    );
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args([
            "run",
            "--source-ad",
            "--deterministic",
            "--checkpoint-blocks",
            "--layerwise-accum",
        ])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_ARENA_REPORT", "1")
        .output()
        .expect("spawn nsl run");
    assert!(
        out.status.success(),
        "run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Everything AFTER the first tag, not just the first tagged chunk: the
    // Stage-2B provenance line ("element counts: ...") now precedes the
    // rendered report, and taking one chunk would grade the wrong text.
    let arena = stderr
        .splitn(2, "[arena]")
        .nth(1)
        .expect("no [arena] report in stderr");
    assert!(
        arena.contains("arena bytes:"),
        "no quantified byte line — the Stage-2A hint plumbing went dark:\n{arena}"
    );
    assert!(
        arena.contains("LOWER BOUND"),
        "partial quantification must be labeled a lower bound:\n{arena}"
    );
    assert!(
        !arena.contains("shapes runtime-valued"),
        "report fell back to the pre-2A unquantified branch:\n{arena}"
    );
    assert!(
        arena.contains("tape-escaping"),
        "gradient accumulators must be pinned live-to-tape-end (the escape \
         fix) and reported as resident:\n{arena}"
    );
    // "bytes cover N of M transients" — N must be positive and below M
    // (this fixture's params are sized via initializer dims + the adjoint
    // mirror; its activation transients remain runtime-shaped).
    let covered = first_int_after(arena, "bytes cover ")
        .expect("no coverage count in the partial-quantification line");
    let total = first_int_after(arena, " of ")
        .expect("no total count in the partial-quantification line");
    assert!(
        covered > 0 && covered < total,
        "expected 0 < covered < total, got {covered} of {total}:\n{arena}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Stage-2B/2C: the full byte-identity parity gate, as a cert-lane test.
///
/// Delegates to `scripts/arena-parity.sh` (the same script the flag's help
/// names) rather than reimplementing it: two builds of the deterministic
/// coder50m cert program (with/without `--transient-arena`), full runs,
/// byte-compared loss streams AND checkpoints, bind/placement
/// reconciliation (placements == binds, 0 guard failures, 0 misplaced),
/// and a per-step red-zone canary pass. This is the gate that caught the
/// allocator-entry pin bypass and the BFD cross-slot aliasing.
#[test]
#[ignore = "requires CUDA GPU (~5 runs of a 40-step coder50m program)"]
fn arena_parity_script_passes() {
    let root = repo_root();
    let out = Command::new("bash")
        .arg(root.join("scripts/arena-parity.sh"))
        .arg(env!("CARGO_BIN_EXE_nsl"))
        .current_dir(&root)
        .output()
        .expect("spawn arena-parity.sh");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains("arena-parity: PASS"),
        "arena parity failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
