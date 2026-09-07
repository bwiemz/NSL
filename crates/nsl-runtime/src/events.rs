//! Structured runtime event stream (roadmap item 17).
//!
//! `NSL_EVENTS=<path>` makes the runtime append one JSON object per line to
//! `<path>` — machine-readable twins of the bracketed stderr markers that
//! tests and campaign drivers previously regex-parsed out of process output.
//! Before this, 47 test files and 4 python drivers hand-parsed lines like
//! `[weight-stream] uploads: 12 evicts: 3 …` (one python consumer pinned nine
//! fields, their order, and their punctuation in a single regex), and the
//! only thing protecting them was prose in the emitters saying "append-only,
//! new fields at the END".
//!
//! Contract:
//! - One event per line:
//!   `{"v":1,"seq":N,"rank":R,"kind":"...","step":S|null,"fields":{...}}`.
//! - `v` is the stream-format version; bump on any change to the ENVELOPE
//!   (per-kind field sets may grow — consumers must ignore unknown fields).
//! - `seq` is a PER-PROCESS monotonic counter, so consumers can order and
//!   de-duplicate within one rank (a step boundary emits `gpu_mem_step`
//!   twice: at step start and after cleanup). Under `--devices N` every
//!   rank is a separate process appending to the same file, each with its
//!   own seq starting at 0 — multi-rank consumers MUST key on (rank, seq),
//!   and a strictly-increasing-seq assertion is only valid single-rank.
//! - `rank` is the envelope's process identity (`NSL_LOCAL_RANK`, 0 when
//!   unset), so interleaved multi-rank series are attributable without
//!   every kind carrying its own rank field.
//! - Emission is best-effort and NEVER aborts or panics: a training run must
//!   not die because an events path is unwritable. The first failure prints
//!   one `[nsl] warning:` line and further emission is disabled.
//! - The writer is the compiled program (one process per rank). The `nsl`
//!   CLI, which passes `NSL_EVENTS` through to the program it spawns and
//!   whose own diagnostics also go through `nsl_log!`, opts itself out with
//!   [`opt_out_this_process`] so the file has one writer per rank.
//! - The stderr markers are UNCHANGED, byte for byte, and stay gated by
//!   their own env vars; `NSL_EVENTS` gates only this file. Both renderings
//!   are built from a single snapshot of the underlying counters at each
//!   call site, so the two cannot disagree about values.
//! - Appends are one `write(2)` per line on an `O_APPEND` fd, so concurrent
//!   multi-rank writers (`--devices N` under ZeRO) do not interleave bytes
//!   within a line; ranks carry a `rank` field where identity matters.
//!
//! The registry of event kinds and their field names lives with the marker
//! registry in `crates/nsl-cli/src/exec_markers.rs`, where the existing
//! rot-gates extend to validate FORMAT — which nothing did for the stderr
//! lines.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Stream-envelope version. Bump only for envelope changes; growing a kind's
/// field set is not a version bump (consumers ignore unknown fields).
pub const EVENTS_VERSION: u32 = 1;

static SINK: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
static SEQ: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicBool = AtomicBool::new(false);
static OPTED_OUT: AtomicBool = AtomicBool::new(false);

/// Keep this process out of the stream, whatever `NSL_EVENTS` says.
///
/// The stream belongs to the compiled program: `seq` is a per-process
/// counter and `rank` a process identity, so its consumers assume one
/// writer per rank. The `nsl` CLI calls this first thing in `main` — it
/// inherits the variable it passes to the program it spawns, and since its
/// compile-time diagnostics go through `nsl_log!` too, without this the
/// compiler's `log` events would land in the program's file with a second
/// `seq` sequence starting at 0 (the events gate pins this). The file is
/// not opened or created by an opted-out process.
pub fn opt_out_this_process() {
    OPTED_OUT.store(true, Ordering::Relaxed);
}

fn sink() -> &'static Option<Mutex<std::fs::File>> {
    if let Some(s) = SINK.get() {
        return s;
    }
    // The open happens OUTSIDE the `OnceLock` initializer, and so does the
    // warning: `nsl_log!` dispatches to the subscriber, which asks
    // `enabled()` → `sink()` whether to mirror the line into this stream,
    // and a `get_or_init` that re-enters itself from its own closure blocks
    // forever (a compiled program whose `NSL_EVENTS` points at an
    // unopenable path used to hang on its first diagnostic line). A racing
    // second initializer just drops its extra append-mode handle.
    let mut failure = None;
    let opened = match std::env::var("NSL_EVENTS").ok().filter(|p| !p.is_empty()) {
        None => None,
        Some(path) => match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => Some(Mutex::new(f)),
            Err(e) => {
                failure = Some((path, e));
                None
            }
        },
    };
    let first = SINK.set(opened).is_ok();
    let out = SINK.get().expect("SINK is set above");
    if let (true, Some((path, e))) = (first, failure) {
        crate::nsl_log!(WARN, "nsl", "[nsl] warning: NSL_EVENTS={path} could not be opened ({e}); events disabled");
    }
    out
}

/// True when `NSL_EVENTS` is set to a writable path. Callers use this to
/// decide whether to take a snapshot at all on hot paths.
pub fn enabled() -> bool {
    !OPTED_OUT.load(Ordering::Relaxed) && sink().is_some() && !FAILED.load(Ordering::Relaxed)
}

/// This process's rank for the event envelope: `NSL_LOCAL_RANK`, 0 when
/// unset — the same variable the multi-rank launcher exports per child.
fn rank() -> i64 {
    static RANK: OnceLock<i64> = OnceLock::new();
    *RANK.get_or_init(|| {
        std::env::var("NSL_LOCAL_RANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Append one event. `fields` are key/value pairs; values use
/// `serde_json::Value` so counters, strings and lists all fit.
/// Best-effort: errors disable the stream with one warning, never panic —
/// this runs inside `extern "C"` atexit hooks where unwinding aborts.
pub fn emit(kind: &str, step: Option<i64>, fields: &[(&str, serde_json::Value)]) {
    if OPTED_OUT.load(Ordering::Relaxed) {
        return;
    }
    let Some(file) = sink() else { return };
    if FAILED.load(Ordering::Relaxed) {
        return;
    }
    let mut map = serde_json::Map::with_capacity(fields.len());
    for (k, v) in fields {
        map.insert((*k).to_string(), v.clone());
    }
    // ONE write call per line (see O_APPEND note in the header). `to_string`
    // on a json! value cannot fail; the newline rides in the same buffer.
    // The seq is taken INSIDE the file mutex: allocated outside, two
    // concurrent emitters could commit their lines out of seq order and the
    // monotonicity consumers pin would be coincidental rather than
    // guaranteed (every call site is single-threaded today; the first
    // threaded emitter must not turn that accident into a flake).
    let write_failed = match file.lock() {
        Ok(mut f) => {
            let line = serde_json::json!({
                "v": EVENTS_VERSION,
                "seq": SEQ.fetch_add(1, Ordering::Relaxed),
                "rank": rank(),
                "kind": kind,
                "step": step,
                "fields": serde_json::Value::Object(map),
            });
            let buf = format!("{line}\n");
            f.write_all(buf.as_bytes()).is_err()
        }
        Err(_) => true,
    };
    if write_failed && !FAILED.swap(true, Ordering::Relaxed) {
        crate::nsl_log!(WARN, "nsl", "[nsl] warning: NSL_EVENTS write failed; events disabled for the rest of the run");
    }
}

/// `u64` counter field.
pub fn u(v: u64) -> serde_json::Value {
    serde_json::Value::from(v)
}

/// `i64` field.
pub fn i(v: i64) -> serde_json::Value {
    serde_json::Value::from(v)
}

/// List-of-integers field (e.g. missing parameter indices).
pub fn ulist(v: &[usize]) -> serde_json::Value {
    serde_json::Value::from(v.iter().map(|x| *x as u64).collect::<Vec<u64>>())
}

#[cfg(test)]
mod tests {
    // The sink is a process-global OnceLock keyed off the environment, so
    // exercising real emission here would race every other test in the
    // process and pin the first-observed env value. The end-to-end behaviour
    // (JSONL well-formedness, envelope fields, stderr agreement, unwritable
    // path never aborting) is gated in crates/nsl-cli/tests/
    // events_stream_gate.rs against real child processes instead.
    #[test]
    fn value_helpers_produce_json_numbers_and_lists() {
        assert_eq!(super::u(7).to_string(), "7");
        assert_eq!(super::i(-3).to_string(), "-3");
        assert_eq!(super::ulist(&[1, 4]).to_string(), "[1,4]");
    }
}
