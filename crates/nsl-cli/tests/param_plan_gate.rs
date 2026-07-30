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
    run_env(tag, extra, plan_report, &[])
}

fn run_env(tag: &str, extra: &[&str], plan_report: bool, env: &[(&str, &str)]) -> (Run, PathBuf) {
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
    for (k, v) in env {
        cmd.env(k, v);
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
    assert!(host_p > 0, "no parameter reached the host-mirror table");
    assert_eq!(
        (bf16_p, shard_p),
        (0, 0),
        "plain --weight-stream must not put anything in the bf16-sr or sharded \
         backends:\n{}",
        plain.stderr
    );
    // The fixture's tied embedding roots a buffered `transpose` view, so the
    // schedule deliberately excludes it (the #397 hazard) — residents are a
    // real, checked population here, not an empty set that makes the
    // "registered nowhere" expectation vacuous.
    assert!(
        resident_p > 0,
        "expected some view-rooted/unclaimed params to stay resident and be \
         checked as such:\n{}",
        plain.stderr
    );
    assert_eq!(resident_p + host_p, total_p, "counts do not partition");

    let (sr, t2) = run("sr", WS_SR, false);
    assert!(sr.ok, "bf16-sr run failed:\n{}", sr.stderr);
    let (total_s, resident_s, host_s, bf16_s, shard_s) = verified_line(&sr.stderr)
        .unwrap_or_else(|| panic!("no [param-plan] verified line:\n{}", sr.stderr));
    assert_eq!(
        (host_s, shard_s),
        (0, 0),
        "--param-dtype bf16-sr must put its params in the SR table, NOT the \
         host-mirror table:\n{}",
        sr.stderr
    );
    assert!(bf16_s > 0, "no parameter reached the SR backend");
    // Same fixture: the same parameters move, only the backend changes.
    assert_eq!(
        (total_p, resident_p, host_p),
        (total_s, resident_s, bf16_s),
        "the streamed/resident split moved between storage modes; only the \
         BACKEND should have"
    );

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
    assert!(sharded > 0, "no parameter reached the sharded backend");
    assert_eq!(
        (host, bf16),
        (0, 0),
        "--zero-stage 3 must not leave anything in the host-mirror or bf16-sr \
         backends:\n{}",
        r.stderr
    );
    assert_eq!(resident + sharded, total, "counts do not partition");
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
    // The report still describes the plan when asked — and it must say the
    // plan is ALL-resident. Asserting `contains("resident")` alone would pass
    // on an all-streamed plan too: "resident" appears in the per-parameter
    // mode column and in unrelated [weight-stream] chatter.
    let header = r
        .stderr
        .lines()
        .find(|l| l.contains("[param-plan]") && l.contains("parameter(s):"))
        .unwrap_or_else(|| panic!("no [param-plan] header:\n{}", r.stderr));
    assert!(
        header.contains("0 streamed"),
        "with no residency backend every parameter must be resident:\n{header}"
    );
    assert!(
        num_before(header, "parameter(s)").is_some_and(|n| n > 0),
        "report covers no parameters — vacuous:\n{header}"
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

/// **The gate that makes every other gate here mean something.**
///
/// `nsl_param_plan_verify`'s abort arm cannot be reached by a correct
/// compiler — producing a real mismatch requires the bug the check exists to
/// catch. So without fault injection, replacing `observe()` with `expected()`
/// (i.e. deleting the entire cross-check and keeping only the bookkeeping)
/// leaves all the assertions above passing: they read counts derived from the
/// plan, not from the tables. This drives `NSL_PARAM_PLAN_FAULT`, which
/// corrupts exactly one declared entry, and requires the process to die
/// naming the parameter.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_corrupted_plan_entry_aborts_the_run_gpu() {
    let (r, tmp) = run_env("fault", WS_SR, false, &[("NSL_PARAM_PLAN_FAULT", "0")]);
    assert!(
        !r.ok,
        "a parameter whose declared plan disagrees with its actual backend did \
         NOT abort the run — the cross-check is decorative:\n{}",
        r.stderr
    );
    assert!(
        r.stderr.contains("[param-plan] FATAL"),
        "aborted without the plan diagnostic (something else failed?):\n{}",
        r.stderr
    );
    assert!(
        r.stderr.contains("param 0"),
        "the diagnostic must name the offending parameter:\n{}",
        r.stderr
    );
    // Exactly one parameter was corrupted, so exactly one must be reported —
    // proving the check is per-parameter and not an all-or-nothing flag.
    assert!(
        r.stderr.contains("FATAL: 1 of "),
        "expected exactly one mismatched parameter:\n{}",
        r.stderr
    );
    // And the injection announces itself, so a stray env var in CI cannot
    // quietly turn a real run into a corrupted one.
    assert!(
        r.stderr.contains("FAULT INJECTION ACTIVE"),
        "fault injection ran without announcing itself:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Same fixture, no fault: proves the gate above fails for the reason it
/// claims (the corruption) rather than because this configuration is broken.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_same_run_without_fault_injection_succeeds_gpu() {
    let (r, tmp) = run("nofault", WS_SR, false);
    assert!(r.ok, "control run failed:\n{}", r.stderr);
    assert!(
        !r.stderr.contains("[param-plan] FATAL"),
        "control run reported a mismatch:\n{}",
        r.stderr
    );
    assert!(
        !r.stderr.contains("FAULT INJECTION"),
        "fault injection leaked into the control run:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// `nsl_param_plan_teardown` is load-bearing and nothing else covers it:
/// delete its emission and a single-train-block program still passes, while
/// a two-block program aborts because block 2 verifies block 1's pointers
/// after `teardown_mirrors` already dropped them from `MIRRORS`.
#[test]
#[ignore = "requires CUDA GPU (2 train blocks over 2 models, one process)"]
fn two_train_blocks_in_one_process_each_verify_their_own_plan_gpu() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_paramplan_two_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    // ONE program, TWO models, TWO train blocks. Block 1's teardown drops its
    // parameters from MIRRORS; if the declared plan outlived the block, block
    // 2's verify would find block 1's six pointers registered nowhere while
    // their entries still say host-mirrored, and abort on staleness.
    let base = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .expect("ffn fixture missing");
    let second = r#"
let m2 = TinyLM()
m2.to(cuda)

print("SECOND_BLOCK_BEGIN")
train(model=m2, epochs=1, grad_accumulation=2):
    optimizer: AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)
    step(batch):
        let logits2 = m2.forward_train(batch.input_ids)
        let ls2 = logits2.shape
        let flat_logits2 = logits2.reshape([ls2[0] * ls2[1], ls2[2]])
        let flat_labels2 = batch.labels.reshape([ls2[0] * ls2[1]])
        let loss = cross_entropy(flat_logits2, flat_labels2)
    callbacks:
        on_step(step, loss):
            print(loss)
print("SECOND_BLOCK_END")

"#;
    let src = base
        .replace(
            "CSLA_SAVE_PATH",
            &tmp.join("a.nslm").display().to_string().replace('\\', "/"),
        )
        .replace("# GPU_PLACEMENT", "m.to(cuda)")
        .replace("model_save(m, \"", &format!("{second}model_save(m, \""));
    assert!(
        src.contains("SECOND_BLOCK_BEGIN"),
        "fixture lost its model_save anchor — the second train block was never \
         spliced in, which would make this gate vacuous"
    );
    let prog = tmp.join("two.nsl");
    std::fs::write(&prog, &src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args([
            "run",
            "--source-ad",
            "--deterministic",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--weight-stream",
        ])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let err = String::from_utf8_lossy(&out.stderr);
    let sout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "two-block program failed:\n{err}");
    assert!(
        sout.contains("SECOND_BLOCK_BEGIN") && sout.contains("SECOND_BLOCK_END"),
        "the second train block never ran — the gate would be vacuous:\n{sout}"
    );
    assert!(
        !err.contains("[param-plan] FATAL"),
        "stale-plan mismatch: the first block's plan outlived its teardown:\n{err}"
    );
    assert_eq!(
        count_lines(&err, "[param-plan] verified"),
        2,
        "expected one verified line per train block:\n{err}"
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
