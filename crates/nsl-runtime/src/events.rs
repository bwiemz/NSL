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
//! - One event per line: `{"v":1,"seq":N,"kind":"...","step":S|null,"fields":{...}}`.
//! - `v` is the stream-format version; bump on any change to the ENVELOPE
//!   (per-kind field sets may grow — consumers must ignore unknown fields).
//! - `seq` is a process-wide monotonic counter, so consumers can order and
//!   de-duplicate (a step boundary emits `gpu_mem_step` twice: at step start
//!   and after cleanup).
//! - Emission is best-effort and NEVER aborts or panics: a training run must
//!   not die because an events path is unwritable. The first failure prints
//!   one `[nsl] warning:` line and further emission is disabled.
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

fn sink() -> &'static Option<Mutex<std::fs::File>> {
    SINK.get_or_init(|| {
        let path = std::env::var("NSL_EVENTS").ok()?;
        if path.is_empty() {
            return None;
        }
        match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => Some(Mutex::new(f)),
            Err(e) => {
                eprintln!("[nsl] warning: NSL_EVENTS={path} could not be opened ({e}); events disabled");
                None
            }
        }
    })
}

/// True when `NSL_EVENTS` is set to a writable path. Callers use this to
/// decide whether to take a snapshot at all on hot paths.
pub fn enabled() -> bool {
    sink().is_some() && !FAILED.load(Ordering::Relaxed)
}

/// Append one event. `fields` are key/value pairs; values use
/// `serde_json::Value` so counters, strings and lists all fit.
/// Best-effort: errors disable the stream with one warning, never panic —
/// this runs inside `extern "C"` atexit hooks where unwinding aborts.
pub fn emit(kind: &str, step: Option<i64>, fields: &[(&str, serde_json::Value)]) {
    let Some(file) = sink() else { return };
    if FAILED.load(Ordering::Relaxed) {
        return;
    }
    let mut map = serde_json::Map::with_capacity(fields.len());
    for (k, v) in fields {
        map.insert((*k).to_string(), v.clone());
    }
    let line = serde_json::json!({
        "v": EVENTS_VERSION,
        "seq": SEQ.fetch_add(1, Ordering::Relaxed),
        "kind": kind,
        "step": step,
        "fields": serde_json::Value::Object(map),
    });
    // ONE write call per line (see O_APPEND note in the header). `to_string`
    // on a json! value cannot fail; the newline rides in the same buffer.
    let buf = format!("{line}\n");
    let write_failed = match file.lock() {
        Ok(mut f) => f.write_all(buf.as_bytes()).is_err(),
        Err(_) => true,
    };
    if write_failed && !FAILED.swap(true, Ordering::Relaxed) {
        eprintln!("[nsl] warning: NSL_EVENTS write failed; events disabled for the rest of the run");
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
