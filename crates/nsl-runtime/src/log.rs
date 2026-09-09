//! The runtime's logging front door (roadmap C3), now a wrapper around
//! the toolchain-wide `nsl-log` crate (roadmap A3).
//!
//! Every diagnostic line the runtime prints goes through
//! [`nsl_log!`](crate::nsl_log) — a `tracing` event with a `target` naming
//! the subsystem (`"zero3"`, `"cuda-graph"`, …) and the line as its
//! message — instead of a bare `eprintln!`. The macro, the subscriber that
//! renders each line to stderr byte-identically to the `eprintln!` it
//! replaced (`nsl_log::NslSubscriber`, re-exported here), and the
//! host-subscriber behaviour are `nsl-log`'s; see that crate's docs. What
//! this module adds is the runtime's own contribution to every line:
//!
//! When `NSL_EVENTS` is set, every line is also appended to the JSONL
//! stream as a `log` event (`fields: {level, target, message}`), so the
//! structured record and stderr cannot disagree (`src/events.rs`). The
//! stream is the runtime's, not the logger's, so it reaches the subscriber
//! as `nsl-log`'s [`EventsMirror`](nsl_log::EventsMirror) hook:
//! [`ensure_installed`] registers [`EventsStreamMirror`] (first
//! registration wins, so every call after the first is an atomic load) and
//! then installs the subscriber. The runtime's `nsl_log!` calls
//! `ensure_installed` before each line, so the mirror is registered before
//! the first runtime line; the `nsl` CLI calls it first thing in `main`
//! for the same reason (and opts itself out of the stream, since the
//! stream belongs to the program it spawns).
//!
//! Levels: `ERROR` for a line that precedes an abort or reports a lost
//! result (`FATAL`, a failed collective), `WARN` for a degraded-but-
//! continuing condition (a fallback, a refused knob, a corrupted guard
//! detected), `INFO` for the informational markers (counters, traces,
//! `disabled by …`). The level does not change what is printed.
//!
//! Migration status: every diagnostic `eprintln!` in `nsl-runtime` goes
//! through `nsl_log!` — the bracketed-marker family (`[zero3]`,
//! `[cuda-graph]`, `[weight-stream]`, `[arena]`, …), the `nsl: …` and
//! `[nsl] …` families (target `"nsl"`), and the per-subsystem lines
//! (`"cfie"` for the `CFIE: …` refusals, `"flash-attention"` /
//! `"flash-bwd"`, `"fused-linear-ce"`, `"cuda"`, `"tensor"`, `"huggingface"`,
//! …; a line that starts with its own `[marker]` uses the marker as its
//! target). Program output stays on `println!` — the `print` builtin
//! (`print.rs`), the tensor printer, the health JSON — since that is
//! stdout, not a diagnostic. nsl-codegen's compile-time diagnostics use
//! the same macro (`nsl_log::nsl_log!`, directly; target `"codegen"` for
//! the `warning:` / `error:` / `note:` lines, the subsystem otherwise —
//! `"autotune"`, `"ccr"`, `"source-ad"`, `"wggo"`, …), leaving only its
//! multi-line `eprint!` report dumps and the dev-tool binaries under
//! `src/bin/` on raw prints. nsl-cli's own lines (`error: …` before an
//! exit, `warning:` / `note:`, the `[nsl] …` launcher lines) use it too
//! (target `"cli"`, `"nsl"`, or the line's own marker), so every
//! diagnostic line the toolchain prints is a `tracing` event.

/// Emit one diagnostic line: `nsl_log!(LEVEL, "target", "format", args…)`.
///
/// `LEVEL` is a `tracing::Level` constant name (`ERROR`, `WARN`, `INFO`,
/// `DEBUG`, `TRACE`); the format string and arguments are exactly what the
/// `eprintln!` took. The line printed is the formatted message plus a
/// newline. This is `nsl_log::nsl_log!` with the runtime's
/// [`ensure_installed`](crate::log::ensure_installed) in front, so the
/// `NSL_EVENTS` mirror is registered before the line is emitted.
#[macro_export]
macro_rules! nsl_log {
    ($level:ident, $target:literal, $($arg:tt)+) => {{
        $crate::log::ensure_installed();
        $crate::log::tracing::event!(target: $target, $crate::log::tracing::Level::$level, $($arg)+);
    }};
}

/// Re-exported for [`nsl_log!`](crate::nsl_log): the macro reaches
/// `tracing` through this path instead of the crate needing its own
/// dependency on it.
#[doc(hidden)]
pub use nsl_log::tracing;

/// The toolchain's subscriber (see `nsl-log`): re-exported at its
/// historical path.
pub use nsl_log::NslSubscriber;

/// The runtime's [`EventsMirror`](nsl_log::EventsMirror): every logged
/// line becomes a `log` event on the `NSL_EVENTS` stream when the stream
/// is on. `enabled` is `events::enabled()`, so a process with no stream
/// (or one that opted out) never renders the message a second time.
pub struct EventsStreamMirror;

impl nsl_log::EventsMirror for EventsStreamMirror {
    fn enabled(&self) -> bool {
        crate::events::enabled()
    }

    fn emit(&self, level: &str, target: &str, message: &str) {
        crate::events::emit(
            "log",
            None,
            &[
                ("level", serde_json::Value::from(level)),
                ("target", serde_json::Value::from(target)),
                ("message", serde_json::Value::from(message)),
            ],
        );
    }
}

static MIRROR: EventsStreamMirror = EventsStreamMirror;

/// Register the runtime's `NSL_EVENTS` mirror (once) and install
/// [`NslSubscriber`] as the process's global subscriber (once). A host
/// that installed its own subscriber first keeps it; calling this again is
/// free.
pub fn ensure_installed() {
    let _ = nsl_log::set_events_mirror(&MIRROR);
    nsl_log::ensure_installed();
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing::{span, Event, Metadata, Subscriber};

    /// A scoped subscriber that records (level, target, message) through
    /// the same visitor the real one uses.
    struct Capture(Arc<Mutex<Vec<(String, String, String)>>>);

    impl Subscriber for Capture {
        fn enabled(&self, _m: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _s: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }
        fn record(&self, _s: &span::Id, _v: &span::Record<'_>) {}
        fn record_follows_from(&self, _s: &span::Id, _f: &span::Id) {}
        fn event(&self, event: &Event<'_>) {
            let mut v = nsl_log::MessageVisitor { message: String::new() };
            event.record(&mut v);
            self.0.lock().unwrap().push((
                event.metadata().level().as_str().to_string(),
                event.metadata().target().to_string(),
                v.message,
            ));
        }
        fn enter(&self, _s: &span::Id) {}
        fn exit(&self, _s: &span::Id) {}
    }

    #[test]
    fn the_runtime_macro_registers_the_events_mirror_and_still_dispatches() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let ptr = 0x1234_i64;
        tracing::subscriber::with_default(Capture(captured.clone()), || {
            crate::nsl_log!(ERROR, "zero3", "[zero3] FATAL: gather of untracked tensor {ptr}");
        });
        assert_eq!(
            captured.lock().unwrap().clone(),
            vec![("ERROR".to_string(), "zero3".to_string(), format!("[zero3] FATAL: gather of untracked tensor {ptr}"))]
        );
        // The line above went through `ensure_installed`, which registered
        // the runtime's mirror (the first registration wins, so this holds
        // whichever test in the binary logged first).
        let mirror = nsl_log::events_mirror().expect("the runtime registered its mirror");
        assert!(std::ptr::addr_eq(mirror as *const _, &super::MIRROR as *const _));
        // Without `NSL_EVENTS` (or opted out) the mirror is off, so the
        // subscriber never renders the message a second time.
        if std::env::var_os("NSL_EVENTS").is_none() {
            assert!(!nsl_log::EventsMirror::enabled(mirror));
        }
    }

    #[test]
    fn ensure_installed_is_idempotent() {
        super::ensure_installed();
        super::ensure_installed();
        let captured = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::with_default(Capture(captured.clone()), || {
            crate::nsl_log!(INFO, "probe", "[probe] after install");
        });
        assert_eq!(captured.lock().unwrap().len(), 1);
    }
}
