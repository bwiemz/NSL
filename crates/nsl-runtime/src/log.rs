//! The runtime's logging front door (roadmap C3).
//!
//! Every diagnostic line the runtime prints goes through
//! [`nsl_log!`](crate::nsl_log) — a `tracing` event with a `target` naming
//! the subsystem (`"zero3"`, `"cuda-graph"`, …) and the line as its
//! message — instead of a bare `eprintln!`. Two things follow:
//!
//! 1. **stderr is byte-identical.** [`NslSubscriber`] is the crate's own
//!    `tracing::Subscriber`: it writes the message plus one newline to
//!    stderr under the stderr lock, straight from the format arguments
//!    (no heap allocation on that path, so the `nsl: out of memory` line
//!    in `memory.rs` still prints), and nothing else — no timestamp,
//!    level, target or colour. The `[zero3] …`,
//!    `[cuda-graph] …` marker lines that `nsl-cli`'s gates compare byte for
//!    byte (`crates/nsl-cli/src/exec_markers.rs`) come out exactly as the
//!    `eprintln!` they replaced. The subscriber installs itself on the
//!    first `nsl_log!` ([`ensure_installed`]), so a compiled NSL program,
//!    a test binary and a foreign host all get the same lines without an
//!    init call.
//! 2. **A host can listen.** A process that has already installed its own
//!    global subscriber (a Python host with `tracing_subscriber`, say)
//!    keeps it: `ensure_installed` yields, and the runtime's lines reach
//!    that subscriber as ordinary events with their target and level. The
//!    stderr rendering is then the host's, which is the point.
//!
//! When `NSL_EVENTS` is set, every line is also appended to the JSONL
//! stream as a `log` event (`fields: {level, target, message}`), so the
//! structured record and stderr cannot disagree (`src/events.rs`).
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
//! the same macro (`nsl_runtime::nsl_log!`; target `"codegen"` for the
//! `warning:` / `error:` / `note:` lines, the subsystem otherwise —
//! `"autotune"`, `"ccr"`, `"source-ad"`, `"wggo"`, …), leaving only its
//! multi-line `eprint!` report dumps and the dev-tool binaries under
//! `src/bin/` on raw prints. Next: nsl-cli.

use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::OnceLock;

use tracing::field::{Field, Visit};
use tracing::{span, Event, Metadata, Subscriber};

/// Emit one diagnostic line: `nsl_log!(LEVEL, "target", "format", args…)`.
///
/// `LEVEL` is a `tracing::Level` constant name (`ERROR`, `WARN`, `INFO`,
/// `DEBUG`, `TRACE`); the format string and arguments are exactly what the
/// `eprintln!` took. The line printed is the formatted message plus a
/// newline — see the module header for what happens to it.
#[macro_export]
macro_rules! nsl_log {
    ($level:ident, $target:literal, $($arg:tt)+) => {{
        $crate::log::ensure_installed();
        $crate::log::tracing::event!(target: $target, $crate::log::tracing::Level::$level, $($arg)+);
    }};
}

/// Re-exported for [`nsl_log!`](crate::nsl_log): a crate that invokes the
/// macro (nsl-codegen, nsl-cli) reaches `tracing` through this path
/// instead of needing its own dependency on it.
#[doc(hidden)]
pub use tracing;

/// The runtime's subscriber: the event's `message` field, verbatim, to
/// stderr (and to the `NSL_EVENTS` stream when it is on). Spans are
/// accepted and ignored.
pub struct NslSubscriber;

/// Collects the `message` field of an event as the formatted string (the
/// events-stream mirror, and the tests).
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // `tracing` hands the format_args of `event!("…", args)` here as
        // `message`; `fmt::Arguments`' Debug is its Display, so this is the
        // text `eprintln!` would have produced.
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        }
    }
}

/// Writes the `message` field straight to a locked stderr, formatting
/// piece by piece with no intermediate `String` — the path an
/// out-of-memory diagnostic has to survive.
struct StderrVisitor<'a> {
    out: &'a mut std::io::StderrLock<'static>,
}

impl Visit for StderrVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            // A failed stderr write has nowhere to be reported; `eprintln!`
            // would panic here, which inside an `extern "C"` frame aborts.
            let _ = write!(self.out, "{value:?}");
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            let _ = self.out.write_all(value.as_bytes());
        }
    }
}

impl Subscriber for NslSubscriber {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }

    fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

    fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

    fn event(&self, event: &Event<'_>) {
        // stderr first, allocation-free, one lock for message + newline:
        // this is the whole of the stderr contract — the bytes are the
        // message's, nothing is added.
        {
            let stderr = std::io::stderr();
            let mut handle = stderr.lock();
            event.record(&mut StderrVisitor { out: &mut handle });
            let _ = handle.write_all(b"\n");
        }
        if crate::events::enabled() {
            let mut visitor = MessageVisitor { message: String::new() };
            event.record(&mut visitor);
            let metadata = event.metadata();
            crate::events::emit(
                "log",
                None,
                &[
                    ("level", serde_json::Value::from(metadata.level().as_str())),
                    ("target", serde_json::Value::from(metadata.target())),
                    ("message", serde_json::Value::from(visitor.message)),
                ],
            );
        }
    }

    fn enter(&self, _span: &span::Id) {}

    fn exit(&self, _span: &span::Id) {}
}

/// Install [`NslSubscriber`] as the process's global subscriber, once. A
/// host that installed its own first keeps it (see the module header);
/// calling this again is free.
pub fn ensure_installed() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        // Err = a global subscriber already exists: the host's wins.
        let _ = tracing::subscriber::set_global_default(NslSubscriber);
    });
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::{span, Event, Metadata, Subscriber};

    /// A scoped subscriber that records (level, target, message) through
    /// the same visitor the real one uses — so the test pins the text the
    /// stderr line would carry, without capturing stderr.
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
            let mut v = super::MessageVisitor { message: String::new() };
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

    struct Named(Vec<(String, String)>);
    impl Visit for Named {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0.push((field.name().to_string(), format!("{value:?}")));
        }
    }

    #[test]
    fn the_message_is_the_eprintln_text_verbatim() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let ptr = 0x1234_i64;
        let rc = -4;
        tracing::subscriber::with_default(Capture(captured.clone()), || {
            crate::nsl_log!(ERROR, "zero3", "[zero3] FATAL: gather of untracked tensor {ptr}");
            crate::nsl_log!(WARN, "nsl-tcp", "[nsl-tcp] Send failed to rank {}: {}", 3, rc);
            crate::nsl_log!(INFO, "scope", "[scope] tracked={total}, freed={freed}", total = 2, freed = 1);
        });
        let got = captured.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                ("ERROR".into(), "zero3".into(), format!("[zero3] FATAL: gather of untracked tensor {ptr}")),
                ("WARN".into(), "nsl-tcp".into(), format!("[nsl-tcp] Send failed to rank {}: {}", 3, rc)),
                ("INFO".into(), "scope".into(), "[scope] tracked=2, freed=1".to_string()),
            ]
        );
    }

    #[test]
    fn the_visitor_ignores_fields_other_than_message() {
        // A structured event with extra fields still renders only its
        // message: the extra fields ride in the event for a host subscriber,
        // never on stderr.
        let captured = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::with_default(Capture(captured.clone()), || {
            tracing::event!(target: "probe", tracing::Level::INFO, bytes = 42, "[probe] line");
        });
        let got = captured.lock().unwrap().clone();
        assert_eq!(got, vec![("INFO".into(), "probe".into(), "[probe] line".into())]);
        let _ = Named(Vec::new());
    }

    #[test]
    fn ensure_installed_is_idempotent_and_the_macro_still_dispatches() {
        super::ensure_installed();
        super::ensure_installed();
        // With the global subscriber installed (or a host's), the macro
        // must not panic and a scoped subscriber still takes precedence.
        let captured = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::with_default(Capture(captured.clone()), || {
            crate::nsl_log!(INFO, "probe", "[probe] after install");
        });
        assert_eq!(captured.lock().unwrap().len(), 1);
    }
}
