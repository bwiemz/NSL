//! Item 3 gates — the compiled ParameterPlan and its runtime cross-check.
//!
//! The plan's whole value is that it *discriminates*: one derivation says,
//! per parameter, which of the three residency backends owns it, and the
//! runtime confirms the parameter landed there. A check that always says the
//! same thing would be worse than none — it would look like coverage. So
//! these gates never assert only "the marker appeared"; they assert the
//! marker's CONTENT changes with the flags, and that the plan report and the
//! independent `[sr-bf16]` teardown counter agree on how many parameters are
//! in bf16.
//!
//! GPU-only (`--weight-stream` aborts on CPU-resident parameters), so the
//! e2e tests are `#[ignore]`. The plan derivation itself is unit-tested on
//! the host in `nsl_codegen::parameter_plan`, and the verify logic in
//! `nsl_runtime::param_plan`.
//!
//! Run:
//!   cargo test -p nsl-cli --features cuda --test param_plan_gate \
//!     -- --ignored --test-threads=1

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
}

/// Compile+run the CSLA FFN fixture on the GPU with `extra` flags.
fn run(tag: &str, extra: &[&str], plan_report: bool) -> (Run, PathBuf) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_paramplan_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .expect("ffn fixture missing")
    .replace(
        "CSLA_SAVE_PATH",
        &tmp.join("out.nslm").display().to_string().replace('\\', "/"),
    )
    .replace("# GPU_PLACEMENT", "m.to(cuda)");
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic", "--checkpoint-blocks"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if plan_report {
        cmd.env("NSL_PARAM_PLAN_REPORT", "1");
    }
    let out = cmd.output().expect("spawn nsl run");
    (
        Run {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            ok: out.status.success(),
        },
        tmp,
    )
}

fn loss_stream(stdout: &str) -> String {
    stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default()
}

/// The integer immediately preceding `label` on `line`. Anchoring on the
/// label rather than "the Nth number on the line" matters: `bf16-sr` itself
/// contains digits, so positional parsing silently reads the wrong field.
fn num_before(line: &str, label: &str) -> Option<u64> {
    line[..line.find(label)?]
        .trim_end()
        .rsplit(|c: char| !c.is_ascii_digit())
        .next()
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
}

/// `[param-plan] verified N parameter(s): A resident, B host-mirrored, C bf16-sr, D sharded`
/// → `(N, A, B, C, D)`.
fn verified_line(stderr: &str) -> Option<(u64, u64, u64, u64, u64)> {
    let line = stderr
        .lines()
        .find(|l| l.contains("[param-plan] verified"))?;
    Some((
        num_before(line, "parameter(s)")?,
        num_before(line, "resident")?,
        num_before(line, "host-mirrored")?,
        num_before(line, "bf16-sr")?,
        num_before(line, "sharded")?,
    ))
}

fn count_lines(stderr: &str, needle: &str) -> usize {
    stderr.lines().filter(|l| l.contains(needle)).count()
}

const WS: &[&str] = &["--layerwise-accum", "--weight-stream"];
const WS_SR: &[&str] = &[
    "--layerwise-accum",
    "--weight-stream",
    "--param-dtype",
    "bf16-sr",
];

/// The core discrimination gate. The SAME fixture under two storage modes
/// must produce two DIFFERENT plans, each matching the backend that actually
/// took the parameters — host mirrors for plain `--weight-stream`, bf16-sr
/// mirrors when `--param-dtype bf16-sr` is added. A check that could not tell
/// these apart would pass vacuously on both.
#[test]
#[ignore = "requires CUDA GPU (2 runs of the CSLA FFN fixture)"]
fn the_plan_names_the_backend_that_actually_took_each_parameter_gpu() {
    let (plain, t1) = run("plain", WS, false);
    assert!(plain.ok, "plain --weight-stream run failed:\n{}", plain.stderr);
    let (total_p, resident_p, host_p, bf16_p, shard_p) = verified_line(&plain.stderr)
        .unwrap_or_else(|| panic!("no [param-plan] verified line:\n{}", plain.stderr));
    assert!(total_p > 0, "plan verified zero parameters — vacuous");
    assert_eq!(
        (host_p, bf16_p, shard_p),
        (total_p, 0, 0),
        "plain --weight-stream must put every declared param in the host-mirror \
         table:\n{}",
        plain.stderr
    );
    assert_eq!(resident_p, 0, "only streamed params are declared");

    let (sr, t2) = run("sr", WS_SR, false);
    assert!(sr.ok, "bf16-sr run failed:\n{}", sr.stderr);
    let (total_s, _, host_s, bf16_s, shard_s) = verified_line(&sr.stderr)
        .unwrap_or_else(|| panic!("no [param-plan] verified line:\n{}", sr.stderr));
    assert_eq!(
        (host_s, bf16_s, shard_s),
        (0, total_s, 0),
        "--param-dtype bf16-sr must put every declared param in the SR table, \
         NOT the host-mirror table:\n{}",
        sr.stderr
    );
    // Same fixture, same parameter count — only the backend moved.
    assert_eq!(total_p, total_s, "parameter count changed between modes");

    // Cross-check against an INDEPENDENT counter: the sr-bf16 teardown line
    // reports how many params the SR backend actually holds. If the plan and
    // that counter disagree, one of them is lying.
    let td = sr
        .stderr
        .lines()
        .find(|l| l.contains("[sr-bf16] teardown:"))
        .unwrap_or_else(|| panic!("no sr-bf16 teardown line:\n{}", sr.stderr));
    let sr_params: u64 = td
        .split_whitespace()
        .find_map(|w| w.parse().ok())
        .expect("no count in sr-bf16 teardown line");
    assert_eq!(
        bf16_s, sr_params,
        "plan says {bf16_s} bf16-sr params, the SR backend's own teardown says \
         {sr_params}:\n{}",
        sr.stderr
    );

    let _ = std::fs::remove_dir_all(&t1);
    let _ = std::fs::remove_dir_all(&t2);
}

/// The third backend. `--zero-stage 3` redirects the same registration
/// sites to the broadcast backend, and the plan must say so — sharded, with
/// zero host mirrors and zero bf16 mirrors. Together with the test above
/// this pins all three arms of `nsl_weight_stream_register`'s dispatch
/// chain, which is the only reason the cross-check can fail informatively.
///
/// Runs under `--collectives sim-gpu` (2 ranks on one device); each rank
/// reports its own plan, so the assertions read the first line.
#[test]
#[ignore = "requires CUDA GPU (sim-gpu collectives, 2 ranks on 1 device)"]
fn the_plan_names_the_sharded_backend_under_zero3_gpu() {
    let (r, tmp) = run(
        "z3",
        &[
            "--layerwise-accum",
            "--weight-stream",
            "--zero-stage",
            "3",
            "--devices",
            "2",
            "--collectives",
            "sim-gpu",
        ],
        false,
    );
    assert!(r.ok, "zero3 run failed:\n{}", r.stderr);
    let (total, resident, host, bf16, sharded) = verified_line(&r.stderr)
        .unwrap_or_else(|| panic!("no [param-plan] verified line:\n{}", r.stderr));
    assert!(total > 0, "plan verified zero parameters — vacuous");
    assert_eq!(
        (resident, host, bf16, sharded),
        (0, 0, 0, total),
        "--zero-stage 3 must put every declared param in the sharded backend:\n{}",
        r.stderr
    );
    // Anti-vacuity against the sharding actually happening, not just being
    // declared: the zero3 backend's own banner must be present.
    assert!(
        r.stderr.contains("[zero3] tensor-granular parameter sharding enabled"),
        "zero3 backend never armed — the plan would be describing nothing:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// The verify runs every micro-batch (so a table that drifts mid-run is
/// caught) but reports once. Both halves matter: reporting per-step would
/// bury a 500-step run's stderr, and checking once would miss drift.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_check_reports_once_per_train_block_gpu() {
    let (r, tmp) = run("once", WS_SR, false);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    assert_eq!(
        count_lines(&r.stderr, "[param-plan] verified"),
        1,
        "expected exactly one verified line per train block:\n{}",
        r.stderr
    );
    assert_eq!(
        count_lines(&r.stderr, "[param-plan] FATAL"),
        0,
        "unexpected plan mismatch:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// `NSL_PARAM_PLAN_REPORT=1` prints the per-parameter plan, and printing it
/// is pure — the compiled program is byte-for-byte the same decision
/// sequence, so the loss stream must not move.
#[test]
#[ignore = "requires CUDA GPU (2 runs)"]
fn the_per_parameter_report_is_opt_in_and_loss_neutral_gpu() {
    let (off, t1) = run("rep_off", WS_SR, false);
    assert!(off.ok, "run failed:\n{}", off.stderr);
    assert!(
        !off.stderr.contains("[param-plan] param["),
        "per-parameter lines leaked without NSL_PARAM_PLAN_REPORT:\n{}",
        off.stderr
    );

    let (on, t2) = run("rep_on", WS_SR, true);
    assert!(on.ok, "run failed:\n{}", on.stderr);
    let per_param = count_lines(&on.stderr, "[param-plan] param[");
    assert!(
        per_param > 0,
        "NSL_PARAM_PLAN_REPORT=1 produced no per-parameter lines:\n{}",
        on.stderr
    );
    // The header counts every parameter, streamed or not; the verified line
    // counts only the declared (streamed) ones. The report is the wider view,
    // which is the point of having it.
    let (declared, ..) = verified_line(&on.stderr).expect("no verified line");
    assert!(
        per_param as u64 >= declared,
        "report lists {per_param} params but {declared} were declared:\n{}",
        on.stderr
    );

    assert_eq!(
        loss_stream(&off.stdout),
        loss_stream(&on.stdout),
        "NSL_PARAM_PLAN_REPORT changed the loss stream — the report is not pure"
    );
    assert!(!loss_stream(&on.stdout).is_empty(), "no losses captured");

    let _ = std::fs::remove_dir_all(&t1);
    let _ = std::fs::remove_dir_all(&t2);
}

/// Without a residency backend there is nothing to cross-check, and the
/// compiler emits no declare/verify calls at all — the feature costs
/// literally nothing on the default path.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_run_with_no_residency_backend_emits_no_check_gpu() {
    let (r, tmp) = run("none", &["--layerwise-accum"], true);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    assert_eq!(
        count_lines(&r.stderr, "[param-plan] verified"),
        0,
        "a run with no streaming emitted a plan check:\n{}",
        r.stderr
    );
    // The report still describes the (all-resident) plan when asked.
    assert!(
        r.stderr.contains("[param-plan]") && r.stderr.contains("resident"),
        "NSL_PARAM_PLAN_REPORT should still describe an all-resident plan:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// A parameter plan that cannot hold is refused at compile time with an
/// actionable message rather than producing a binary whose storage does not
/// match its certification. (CPU: the refusal precedes any GPU work.)
#[test]
fn an_impossible_plan_is_refused_before_codegen() {
    // bf16-sr without streaming: the CLI's composition matrix refuses first,
    // which is the behavior users see. The plan's own restatement of the same
    // invariant is unit-tested in nsl_codegen::parameter_plan; here we only
    // pin that SOME layer refuses, because a silently-accepted run would
    // train f32 weights while reporting bf16-sr storage.
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_paramplan_refuse_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .expect("ffn fixture missing")
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
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--param-dtype",
            "bf16-sr",
        ])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(
        !out.status.success(),
        "bf16-sr without --weight-stream was accepted"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("requires --weight-stream"),
        "refusal did not name the missing flag:\n{err}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// The parser the GPU gates depend on. Pinned because the obvious
/// implementation ("split on non-digits, take the Nth") reads `16` out of
/// `bf16-sr` and silently mis-attributes every field — a broken parser here
/// would make the gates above fail confusingly or, worse, pass on the wrong
/// numbers.
#[test]
fn the_verified_line_parser_is_not_confused_by_digits_in_labels() {
    let line = "[param-plan] verified 6 parameter(s): 0 resident, 6 host-mirrored, \
                0 bf16-sr, 0 sharded";
    assert_eq!(verified_line(line), Some((6, 0, 6, 0, 0)));
    let sr = "[param-plan] verified 12 parameter(s): 1 resident, 0 host-mirrored, \
              11 bf16-sr, 0 sharded";
    assert_eq!(verified_line(sr), Some((12, 1, 0, 11, 0)));
    let z3 = "[param-plan] verified 8 parameter(s): 0 resident, 0 host-mirrored, \
              0 bf16-sr, 8 sharded";
    assert_eq!(verified_line(z3), Some((8, 0, 0, 0, 8)));
    assert_eq!(verified_line("no marker here"), None);
}

/// Compile-time only: the fixture path the other gates depend on exists.
/// Cheap guard against a rename silently turning every `#[ignore]` GPU gate
/// above into a panic nobody runs.
#[test]
fn the_fixture_the_gpu_gates_use_exists() {
    let p: &Path = &repo_root().join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl");
    assert!(p.exists(), "missing fixture: {}", p.display());
    let src = std::fs::read_to_string(p).unwrap();
    assert!(src.contains("CSLA_SAVE_PATH"), "fixture lost its save marker");
    assert!(
        src.contains("# GPU_PLACEMENT"),
        "fixture lost its GPU placement marker"
    );
}
