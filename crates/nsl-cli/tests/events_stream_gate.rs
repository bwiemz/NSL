//! Item 17: the structured runtime event stream (`NSL_EVENTS=<path>`).
//!
//! Tests previously regex-parsed bracketed markers out of process stderr —
//! 47 test files, 4 python drivers, one of which pinned nine fields of one
//! line (order and punctuation included) in a single regex. `NSL_EVENTS`
//! gives every counter reporter a machine-readable JSONL twin built from the
//! SAME snapshot as the stderr line.
//!
//! What is pinned here, and why each guard is not vacuous on its own:
//!   1. the file is valid JSONL with the envelope (v/seq/kind/step/fields)
//!      and every counter kind a CPU run can produce actually appears —
//!      an empty file also "parses", so presence is asserted;
//!   2. every event of a registered kind carries its schema's REQUIRED
//!      fields (`exec_markers::EVENT_SCHEMAS`) — the format validation the
//!      marker registry's gates never did;
//!   3. `NSL_EVENTS` alone keeps stderr SILENT — machine output must not
//!      drag in human verbosity, or drivers scraping stderr see new lines;
//!   4. stderr and events AGREE on values — the two renderings come from one
//!      snapshot, and this is the end-to-end proof;
//!   5. the marker lines are byte-identical with and without `NSL_EVENTS` —
//!      the no-regression contract for the 47 files that still parse them;
//!   6. an unwritable events path warns once and the run still SUCCEEDS —
//!      instrumentation must never kill training.
//! The GPU half cross-validates `[gpu-mem]` stderr (MB) against
//! `gpu_mem_step` events (exact bytes) and pins that events keep flowing
//! past the stderr throttle at step 5.

use std::path::PathBuf;
use std::process::Command;

use nsl_cli::exec_markers::EVENT_SCHEMAS;

fn repo_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Marker prefixes of every counter reporter that registers under NSL_EVENTS.
/// Used both to prove silence (test 3) and byte-parity (test 5).
const COUNTER_MARKERS: &[&str] = &[
    "[zero] ",
    "[weight-stream]",
    "[csla] window",
    "[fase-fused]",
    "[wgrad-accum]",
    "[nsl-kernel-count]",
    "[nsl-gpu-launch-count]",
];

/// Counter env vars that gate the stderr lines (NOT the events).
const COUNTER_ENVS: &[&str] = &[
    "NSL_ZERO_COUNTER",
    "NSL_WS_COUNTER",
    "NSL_CSLA_COUNTER",
    "NSL_FASE_FUSED_COUNTER",
    "NSL_WGRAD_COUNTER",
    "NSL_KERNEL_LAUNCH_COUNTER",
    "NSL_WRGA_GPU_LAUNCH_COUNTER",
];

struct Run {
    ok: bool,
    stderr: String,
    events: Vec<serde_json::Value>,
}

/// Run the CPU grad-integrity fixture with the given env vars; `events_path`
/// (when Some) is exported as NSL_EVENTS and read back after exit.
fn run_cpu(tag: &str, envs: &[(&str, &str)], events_path: Option<&PathBuf>) -> Run {
    let root = repo_root();
    let fixture = root.join("crates/nsl-cli/tests/fixtures/grad_integrity_fullbuffer.nsl");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad"])
        .arg(&fixture)
        .current_dir(std::env::temp_dir())
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    // Scrub gates inherited from the ambient environment: a leaked
    // NSL_EVENTS or counter var would silently change what "silent" means.
    cmd.env_remove("NSL_EVENTS");
    for e in COUNTER_ENVS {
        cmd.env_remove(e);
    }
    cmd.env_remove("NSL_GRAD_INTEGRITY");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    if let Some(p) = events_path {
        let _ = std::fs::remove_file(p);
        cmd.env("NSL_EVENTS", p);
    }
    let out = cmd.output().unwrap_or_else(|e| panic!("spawn nsl run ({tag}): {e}"));
    let events = events_path
        .map(|p| {
            std::fs::read_to_string(p)
                .unwrap_or_default()
                .lines()
                .map(|l| {
                    serde_json::from_str(l).unwrap_or_else(|e| {
                        panic!("({tag}) events line is not valid JSON: {e}\n  line: {l}")
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Run {
        ok: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        events,
    }
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("nsl_events_gate_{}_{name}.jsonl", std::process::id()))
}

/// Kinds a plain CPU run of the fixture must produce (every counter atexit
/// registers under NSL_EVENTS; grad_integrity additionally needs arming).
const CPU_KINDS: &[&str] = &[
    "zero_counters",
    "weight_stream_counters",
    "csla_counters",
    "fase_fused_counters",
    "wgrad_counters",
    "kernel_launch_count",
    "gpu_launch_count",
];

#[test]
fn events_file_is_valid_jsonl_with_the_envelope_and_all_cpu_kinds() {
    let path = scratch("envelope");
    let r = run_cpu("envelope", &[("NSL_GRAD_INTEGRITY", "1")], Some(&path));
    assert!(r.ok, "fixture run failed:\n{}", r.stderr);
    assert!(!r.events.is_empty(), "NSL_EVENTS produced an empty file");
    // Strictly-increasing seq is a SINGLE-RANK property (each rank is its
    // own process with its own counter); this fixture is single-rank.
    let mut last_seq = -1i64;
    for ev in &r.events {
        assert_eq!(ev["v"], 1, "envelope version: {ev}");
        let seq = ev["seq"].as_i64().expect("seq is an integer");
        assert!(seq > last_seq, "seq must be strictly increasing: {ev}");
        last_seq = seq;
        assert_eq!(ev["rank"], 0, "single-rank run must carry rank 0: {ev}");
        assert!(ev["kind"].as_str().is_some_and(|k| !k.is_empty()), "kind: {ev}");
        assert!(ev.get("step").is_some(), "step present (may be null): {ev}");
        assert!(ev["fields"].is_object(), "fields is an object: {ev}");
    }
    let kinds: Vec<&str> = r.events.iter().filter_map(|e| e["kind"].as_str()).collect();
    for want in CPU_KINDS.iter().chain(["grad_integrity"].iter()) {
        assert!(
            kinds.contains(want),
            "kind '{want}' missing from a CPU run; got {kinds:?}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn every_event_carries_its_registered_schema_fields() {
    let path = scratch("schema");
    let r = run_cpu("schema", &[("NSL_GRAD_INTEGRITY", "1")], Some(&path));
    assert!(r.ok, "fixture run failed:\n{}", r.stderr);
    let mut checked = 0usize;
    for ev in &r.events {
        let kind = ev["kind"].as_str().unwrap();
        let Some(schema) = EVENT_SCHEMAS.iter().find(|s| s.kind == kind) else {
            panic!("event kind '{kind}' is not in exec_markers::EVENT_SCHEMAS — register it");
        };
        let fields = ev["fields"].as_object().unwrap();
        for required in schema.fields {
            assert!(
                fields.contains_key(*required),
                "kind '{kind}' is missing required field '{required}': {ev}"
            );
        }
        checked += 1;
    }
    // Anti-vacuity: the loop must have validated real events, and every
    // CPU-reachable schema must have been exercised (a schema no fixture can
    // reach is a schema this gate silently stops guarding).
    assert!(checked >= CPU_KINDS.len(), "only {checked} events validated");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_schema_registry_is_internally_consistent() {
    // Every kind unique; every schema's marker actually registered in
    // EXEC_MARKERS. The registry documents both properties (review finding:
    // documented-but-unvalidated is how the marker registry's own gaps
    // started).
    let mut kinds = std::collections::HashSet::new();
    for s in EVENT_SCHEMAS {
        assert!(kinds.insert(s.kind), "duplicate event kind '{}'", s.kind);
        assert!(
            nsl_cli::exec_markers::EXEC_MARKERS
                .iter()
                .any(|m| m.token == s.marker),
            "schema '{}' names marker '{}' which is not in EXEC_MARKERS",
            s.kind,
            s.marker
        );
        assert!(!s.fields.is_empty(), "schema '{}' has no fields", s.kind);
        let mut f = std::collections::HashSet::new();
        for field in s.fields {
            assert!(f.insert(field), "schema '{}' repeats field '{field}'", s.kind);
        }
    }
}

#[test]
fn events_alone_keep_stderr_silent() {
    let path = scratch("silent");
    let r = run_cpu("silent", &[], Some(&path));
    assert!(r.ok, "fixture run failed:\n{}", r.stderr);
    assert!(!r.events.is_empty(), "events file empty");
    for marker in COUNTER_MARKERS {
        assert!(
            !r.stderr.contains(marker),
            "NSL_EVENTS alone must not print '{marker}' to stderr — machine \
             output is decoupled from human verbosity:\n{}",
            r.stderr
        );
    }
}

#[test]
fn stderr_and_events_agree_on_values() {
    let path = scratch("agree");
    let r = run_cpu(
        "agree",
        &[("NSL_ZERO_COUNTER", "1"), ("NSL_WGRAD_COUNTER", "1")],
        Some(&path),
    );
    assert!(r.ok, "fixture run failed:\n{}", r.stderr);

    // [zero] line vs zero_counters event, all nine fields by name.
    let zline = r
        .stderr
        .lines()
        .find(|l| l.starts_with("[zero] "))
        .expect("NSL_ZERO_COUNTER=1 must print the [zero] line");
    let zev = r
        .events
        .iter()
        .find(|e| e["kind"] == "zero_counters")
        .expect("zero_counters event");
    for tok in zline.trim_start_matches("[zero] ").split_whitespace() {
        let (key, val) = tok.split_once('=').expect("key=value token");
        assert_eq!(
            zev["fields"][key].as_i64(),
            val.parse::<i64>().ok(),
            "[zero] field '{key}' disagrees between stderr and event:\n  line: {zline}\n  event: {zev}"
        );
    }

    // [wgrad-accum] line vs wgrad_counters event.
    let wline = r
        .stderr
        .lines()
        .find(|l| l.starts_with("[wgrad-accum]"))
        .expect("NSL_WGRAD_COUNTER=1 must print the [wgrad-accum] line");
    let wev = r
        .events
        .iter()
        .find(|e| e["kind"] == "wgrad_counters")
        .expect("wgrad_counters event");
    let fused: i64 = wline
        .split("fused GEMM: ")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.trim().parse().ok())
        .expect("parse fused GEMM");
    let fallback: i64 = wline
        .split("decomposed fallback: ")
        .nth(1)
        .and_then(|s| s.trim().parse().ok())
        .expect("parse fallback");
    assert_eq!(wev["fields"]["fused_gemm"].as_i64(), Some(fused));
    assert_eq!(wev["fields"]["decomposed_fallback"].as_i64(), Some(fallback));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn marker_lines_are_byte_identical_with_and_without_events() {
    let envs: Vec<(&str, &str)> = COUNTER_ENVS.iter().map(|e| (*e, "1")).collect();
    let without = run_cpu("parity_off", &envs, None);
    let path = scratch("parity_on");
    let with = run_cpu("parity_on", &envs, Some(&path));
    assert!(without.ok && with.ok, "fixture runs failed");
    let markers_of = |s: &str| -> Vec<String> {
        s.lines()
            .filter(|l| COUNTER_MARKERS.iter().any(|m| l.starts_with(m)))
            .map(str::to_owned)
            .collect()
    };
    let a = markers_of(&without.stderr);
    let b = markers_of(&with.stderr);
    assert!(!a.is_empty(), "no marker lines printed — the parity check is vacuous");
    assert_eq!(
        a, b,
        "counter marker lines must be BYTE-IDENTICAL with and without NSL_EVENTS \
         — 47 test files parse them"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn an_unwritable_events_path_never_kills_the_run() {
    let bogus = PathBuf::from("/nonexistent-events-dir/events.jsonl");
    let r = run_cpu("unwritable", &[("NSL_EVENTS", "/nonexistent-events-dir/events.jsonl")], None);
    assert!(
        r.ok,
        "an unwritable NSL_EVENTS path must not fail the run — instrumentation \
         never kills training:\n{}",
        r.stderr
    );
    assert!(
        r.stderr.contains("could not be opened"),
        "the failure must be reported once, not swallowed:\n{}",
        r.stderr
    );
    assert!(!bogus.exists());
}

/// GPU half: `[gpu-mem]` stderr (MB, throttled to step <= 5) vs
/// `gpu_mem_step` events (exact bytes, EVERY step). Reuses the cuda-graph
/// fixture like mse_leak_gate does.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_mem_events_agree_with_stderr_and_outlive_the_throttle() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_events_gpu_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/cuda_graph_gate.nsl"),
    )
    .unwrap();
    src = src.replace(
        "# GPU_PLACEMENT",
        "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)",
    );
    src = src.replace("m.forward_train(x)", "m.forward_train(xg)");
    src = src.replace("(pred, y)", "(pred, yg)");
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let events_path = tmp.join("events.jsonl");
    let _ = std::fs::remove_file(&events_path);

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    // Scrub the ambient environment like run_cpu does — an exported
    // NSL_DEBUG_MEM_ALL (routine during memory debugging on this box) would
    // print [gpu-mem] past step 5 and fail the throttle assertion below for
    // a reason unrelated to the code (review finding).
    cmd.env_remove("NSL_DEBUG_MEM_ALL");
    cmd.env_remove("NSL_GRAD_INTEGRITY");
    for e in COUNTER_ENVS {
        cmd.env_remove(e);
    }
    cmd.env("NSL_EVENTS", &events_path);
    let out = cmd.output().expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "GPU fixture failed:\n{stderr}");

    let events: Vec<serde_json::Value> = std::fs::read_to_string(&events_path)
        .expect("events file")
        .lines()
        .map(|l| serde_json::from_str(l).expect("valid JSON"))
        .collect();
    let mem: Vec<&serde_json::Value> =
        events.iter().filter(|e| e["kind"] == "gpu_mem_step").collect();
    assert!(!mem.is_empty(), "no gpu_mem_step events from a GPU run");

    // Decoupling: stderr throttles at step 5, events must not.
    let max_step = mem.iter().filter_map(|e| e["step"].as_i64()).max().unwrap();
    assert!(
        max_step > 5,
        "events must outlive the stderr throttle (max step {max_step}); the \
         fixture trains 8 epochs so later steps exist"
    );
    assert!(
        !stderr.lines().any(|l| l.starts_with("[gpu-mem] step=6")
            || l.starts_with("[gpu-mem] step=7")),
        "stderr throttle at step 5 must survive events being on"
    );

    // Cross-validation: for every [gpu-mem] stderr header line, the FIRST
    // event with the same step must agree (the second call per step happens
    // after cleanup and legitimately differs). MB fields are the event's
    // exact bytes integer-divided.
    let mut seen = std::collections::HashSet::new();
    let mut checked = 0;
    for line in stderr.lines() {
        let Some(rest) = line.strip_prefix("[gpu-mem] step=") else { continue };
        let step: i64 = rest.split_whitespace().next().unwrap().parse().unwrap();
        if !seen.insert(step) {
            continue;
        }
        let ev = mem
            .iter()
            .find(|e| e["step"].as_i64() == Some(step))
            .unwrap_or_else(|| panic!("no gpu_mem_step event for step {step}"));
        let field = |key: &str| -> i64 {
            rest.split(key)
                .nth(1)
                .and_then(|s| {
                    s.chars()
                        .take_while(char::is_ascii_digit)
                        .collect::<String>()
                        .parse()
                        .ok()
                })
                .unwrap_or_else(|| panic!("field {key} in: {line}"))
        };
        let mib = 1024 * 1024;
        assert_eq!(ev["fields"]["allocated_bytes"].as_i64().unwrap() / mib, field("alloc="));
        assert_eq!(ev["fields"]["reserved_bytes"].as_i64().unwrap() / mib, field("reserved="));
        assert_eq!(ev["fields"]["live_blocks"].as_i64().unwrap(), field("live_blocks="));
        assert_eq!(
            ev["fields"]["persistent_blocks"].as_i64().unwrap(),
            field("persistent_blocks=")
        );
        checked += 1;
    }
    assert!(checked >= 3, "cross-validated only {checked} step lines — vacuous");
}
