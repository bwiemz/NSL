//! Roadmap item 2 step 4 — end-to-end gates for the pass bus.
//!
//! `pass_bus_unit.rs` drives [`nsl_codegen::pass_bus::PassBus`] directly, which
//! proves the counters and the finding rules. It cannot prove the accessors are
//! wired into a real compile — a bus whose publish sites were all deleted would
//! satisfy every unit test in that file and report nothing here.
//!
//! So these run the compiler. CPU-only on the small FFN fixture, no GPU, so
//! they belong in ordinary CI rather than the hardware lane.
//!
//! Both halves everywhere: a channel shows traffic when its pass runs AND is
//! silent when it does not. A gate that only checked the first would pass just
//! as well against a hard-coded report.

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

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
}

fn run(tag: &str, extra: &[&str], trace: bool) -> Run {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_passbus_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src =
        std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"))
            .expect("ffn fixture missing")
            .replace(
                "CSLA_SAVE_PATH",
                &tmp.join("out.nslm").display().to_string().replace('\\', "/"),
            );
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if trace {
        cmd.env("NSL_PASS_TRACE", "1");
    }
    let out = cmd.output().expect("spawn nsl run");
    let r = Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        ok: out.status.success(),
    };
    let _ = std::fs::remove_dir_all(&tmp);
    r
}

/// `(publishes, reads_full, reads_empty)` for `channel`, from its report line.
fn counts(stderr: &str, channel: &str) -> Option<(u64, u64, u64)> {
    let line = stderr
        .lines()
        .find(|l| l.starts_with(&format!("[pass-bus] {channel}: ")))?;
    // "published 1x by CSHA, read 1x full, 0x empty" — anchored on the label
    // that follows each number, never positionally. A positional parser on a
    // sibling gate read "16" out of `bf16-sr`; the labels do not move.
    let num_before = |label: &str| -> Option<u64> {
        line[..line.find(label)?]
            .trim_end()
            .trim_end_matches('x')
            .rsplit(|c: char| !c.is_ascii_digit())
            .next()
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse().ok())
    };
    Some((
        num_before(" by ")?,
        num_before(" full")?,
        num_before(" empty")?,
    ))
}

fn loss_stream(stdout: &str) -> String {
    stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default()
}

/// The parser is load-bearing for every other test here, so it is pinned on a
/// literal line rather than only exercised through them.
#[test]
fn the_count_parser_reads_the_labels_not_the_positions() {
    let l = "[pass-bus] csha_bridge: published 12x by CSHA, read 3x full, 40x empty";
    assert_eq!(counts(l, "csha_bridge"), Some((12, 3, 40)));
    assert_eq!(counts(l, "wrga_plan"), None);
    let zero = "[pass-bus] cpdt_plan: published 0x by CPDT, read 0x full, 1x empty";
    assert_eq!(counts(zero, "cpdt_plan"), Some((0, 0, 1)));
}

/// The bus is opt-in and pure. It is compiler-side atomics — nothing is
/// emitted into the program — so any divergence means the accessors moved more
/// than the fields did.
#[test]
fn the_bus_is_opt_in_and_cannot_change_the_program() {
    let off = run("off", &["--csha", "auto"], false);
    assert!(off.ok, "run failed:\n{}", off.stderr);
    assert!(
        !off.stderr.contains("[pass-bus]"),
        "the bus report printed without NSL_PASS_TRACE=1:\n{}",
        off.stderr
    );

    let on = run("on", &["--csha", "auto"], true);
    assert!(on.ok, "run failed:\n{}", on.stderr);
    assert!(on.stderr.contains("[pass-bus]"), "no bus report with the env set");

    assert!(
        !loss_stream(&off.stdout).is_empty(),
        "no losses captured — the purity check would be vacuous"
    );
    assert_eq!(
        loss_stream(&off.stdout),
        loss_stream(&on.stdout),
        "NSL_PASS_TRACE changed the loss stream — the bus is not pure"
    );
}

/// A channel carries traffic when its producing pass runs, and none when it
/// does not. Both directions on the one channel a CPU fixture can drive.
#[test]
fn a_channel_is_published_only_when_its_producer_runs() {
    let with = run("csha_on", &["--csha", "auto"], true);
    assert!(with.ok, "run failed:\n{}", with.stderr);
    let (pubs, full, _) = counts(&with.stderr, "csha_bridge")
        .unwrap_or_else(|| panic!("no csha_bridge line:\n{}", with.stderr));
    assert_eq!(pubs, 1, "CSHA ran but published no bridge:\n{}", with.stderr);
    assert!(
        full >= 1,
        "csha_bridge was published and never read — the accessor is not wired \
         into the consumers:\n{}",
        with.stderr
    );

    let without = run("csha_off", &[], true);
    assert!(without.ok, "run failed:\n{}", without.stderr);
    assert!(
        counts(&without.stderr, "csha_bridge").is_none(),
        "csha_bridge showed traffic with CSHA off, so the positive half above \
         proves nothing:\n{}",
        without.stderr
    );
}

/// Reads are counted even when the channel is empty. This is what makes
/// `SILENT DEFAULT` possible at all: without empty reads, a consumer taking
/// its `None` branch would be indistinguishable from no consumer existing.
#[test]
fn empty_reads_are_counted_on_a_plain_compile() {
    let r = run("empty", &[], true);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    for channel in ["wrga_plan", "cpdt_plan", "cfie_plan"] {
        let (pubs, full, empty) = counts(&r.stderr, channel)
            .unwrap_or_else(|| panic!("no {channel} line:\n{}", r.stderr));
        assert_eq!(
            (pubs, full),
            (0, 0),
            "{channel} should be unpublished on a plain compile:\n{}",
            r.stderr
        );
        assert!(
            empty >= 1,
            "{channel} recorded no empty reads, so its consumers are not \
             going through the accessor:\n{}",
            r.stderr
        );
    }
}

/// The standard CPU configurations must produce NO findings.
///
/// Both findings describe a contradiction, not a configuration, so a clean
/// fixture producing one means either a real defect or a rule that cries wolf
/// — and the first version of the `SilentDefault` rule did exactly that, firing
/// on every compile where CSHA ran and declined. Pinning zero here is what
/// stops that regressing quietly into noise nobody reads.
#[test]
fn the_standard_configurations_report_no_findings() {
    for (tag, flags) in [
        ("plain", &[][..]),
        ("csha", &["--csha", "auto"][..]),
        ("ccr", &["--checkpoint-blocks", "--layerwise-accum"][..]),
        ("wggo", &["--wggo", "full"][..]),
    ] {
        let r = run(tag, flags, true);
        assert!(r.ok, "{tag} run failed:\n{}", r.stderr);
        let findings: Vec<&str> = r
            .stderr
            .lines()
            .filter(|l| l.contains("DEAD OUTPUT") || l.contains("SILENT DEFAULT"))
            .collect();
        assert!(
            findings.is_empty(),
            "{tag} produced bus findings: {findings:#?}\nEither a real defect \
             appeared, or a finding rule became too broad. Do not silence it \
             without deciding which."
        );
        // Anti-vacuity: the run must actually have exercised the bus, or
        // "no findings" is just "no data".
        assert!(
            r.stderr.contains("[pass-bus] "),
            "{tag} printed no bus lines at all:\n{}",
            r.stderr
        );
    }
}

/// The bus report and the pass trace must arrive together. `SILENT DEFAULT` is
/// computed from both, so a terminal path that emitted one without the other
/// would show counts nobody could interpret.
#[test]
fn the_bus_report_accompanies_the_pass_trace() {
    let r = run("together", &["--csha", "auto"], true);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    let trace_at = r.stderr.find("[pass-trace] ").expect("no pass trace");
    let bus_at = r.stderr.find("[pass-bus] ").expect("no bus report");
    assert!(
        trace_at < bus_at,
        "the bus report printed before the trace it is read against"
    );
}

/// Guards the fixture the tests above depend on, so a rename turns into one
/// clear failure rather than six confusing ones.
#[test]
fn the_fixture_these_gates_use_exists() {
    let p = repo_root().join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl");
    assert!(p.exists(), "missing fixture: {}", p.display());
    assert!(
        std::fs::read_to_string(&p).unwrap().contains("CSLA_SAVE_PATH"),
        "fixture lost its save marker"
    );
}
