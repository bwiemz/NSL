//! Milestone A: CLI-side activation enforcement.
//!
//! `nsl_codegen::activation` supplies the pure half — the contract table and
//! the reconciler. This module supplies the policy half: which surfaces count
//! as "requested" on THIS invocation, and what an unsatisfied contract does
//! to the process.
//!
//! # What counts as requested
//!
//! - **Flags**: every long flag literally present on the command line, read
//!   from argv. clap's derive API offers no explicit-vs-defaulted signal
//!   without re-plumbing every dispatcher through `ArgMatches`, and argv is
//!   the ground truth anyway: a flag the user typed is a request; a default
//!   the parser filled in is not. Tokens after a bare `--` are program
//!   arguments (`nsl run ... -- --whatever`), never requests.
//! - **Decorators**: the entry module's decorator names, logged by the
//!   codegen entry points into `nsl_codegen::activation` during the compile.
//!
//! # Severity
//!
//! On `build` and `run`, an unsatisfied contract is a hard error (exit 1) —
//! the milestone's core rule: no supported request may disappear silently.
//! `--allow-inert-requests` demotes that to a warning for migration.
//! On `check`, reconciliation is report-only: check runs no codegen, so a
//! silent owner there is not evidence of inertness (the reconciler returns
//! OutOfScope for compile-scoped surfaces at check).

use nsl_codegen::activation::{self, OutcomeState, RequestedSurface};
use nsl_codegen::pass_registry::Subcommand;

/// Long flags explicitly present on the command line, kebab-case, without
/// the leading `--`. `--flag=value` yields `flag`; everything after a bare
/// `--` separator is skipped.
pub(crate) fn requested_long_flags() -> Vec<String> {
    requested_long_flags_from(std::env::args().skip(1))
}

fn requested_long_flags_from(args: impl Iterator<Item = String>) -> Vec<String> {
    // The CLI's only short aliases; a short flag is the same request as its
    // long form, and review caught `-w model.nslm` escaping the scan. Kept
    // in lockstep by the unit test below against args.rs's `short = '..'`
    // declarations.
    const SHORT_ALIASES: &[(&str, &str)] = &[("-w", "weights"), ("-o", "output")];
    let mut out = Vec::new();
    let push = |name: &str, out: &mut Vec<String>| {
        if !name.is_empty() && !out.iter().any(|e| e == name) {
            out.push(name.to_string());
        }
    };
    for a in args {
        if a == "--" {
            break;
        }
        if let Some(rest) = a.strip_prefix("--") {
            push(rest.split('=').next().unwrap_or(rest), &mut out);
        } else if let Some((_, long)) = SHORT_ALIASES.iter().find(|(s, _)| *s == a) {
            push(long, &mut out);
        }
    }
    out
}

/// `--allow-unknown-decorators` travels to nsl-semantic (and every module
/// the loader analyzes) as an env var — the plumbing `--grad-integrity` and
/// `--collectives` already use. One helper so the env-var name has one
/// spelling in the tree (review found three pasted copies).
pub(crate) fn apply_allow_unknown_decorators(enabled: bool) {
    if enabled {
        std::env::set_var("NSL_ALLOW_UNKNOWN_DECORATORS", "1");
    }
}

/// `--distribute` refusal (deferral-must-refuse): the M43 3D-parallelism
/// config has zero consumers anywhere in the tree; refusing beats compiling
/// a single-process binary that LOOKS distributed. One helper so the two
/// dispatchers cannot drift (review found the prose pasted verbatim twice).
pub(crate) fn refuse_unimplemented_distribute(distribute: &Option<String>) {
    if distribute.is_some() {
        eprintln!(
            "error: --distribute is not implemented (the M43 3D-parallelism \
             config has no consumer); use --zero-stage/--devices/--collectives \
             for multi-process training, or drop the flag"
        );
        std::process::exit(1);
    }
}

/// Reconcile every surface requested on this invocation against the compile
/// that just ran, honouring the report/escape-hatch flags. Called AFTER a
/// successful compile, BEFORE the artifact is handed to the user (or, on
/// `run`, before the program executes — an inert request must not produce a
/// run that looks observed).
///
/// The policy flags (`--activation-report`, `--allow-inert-requests`) are
/// read from argv like every other requested flag — clap has already
/// validated them as declared arguments by the time any dispatcher runs, so
/// this needs no signature plumbing through the four build flavors.
pub(crate) fn enforce_from_argv(entry: Subcommand) {
    let flags = requested_long_flags();
    let activation_report = flags.iter().any(|f| f == "activation-report");
    let allow_inert = flags.iter().any(|f| f == "allow-inert-requests");
    let mut requested: Vec<RequestedSurface> = flags
        .into_iter()
        // The meta-surfaces of this very mechanism are not optimization
        // requests; reconciling them would put two permanent "no contract"
        // lines in every report.
        .filter(|f| f != "activation-report" && f != "allow-inert-requests")
        .map(RequestedSurface::Flag)
        .collect();
    requested.extend(
        nsl_codegen::activation::requested_decorators()
            .into_iter()
            .map(RequestedSurface::Decorator),
    );

    let outcomes = activation::reconcile(&requested, entry);

    if activation_report {
        eprint!("{}", activation::render_report(&outcomes));
    }

    let unsat = activation::unsatisfied(&outcomes);
    if unsat.is_empty() {
        return;
    }

    // On check, compile-scoped contracts already reconcile to OutOfScope; an
    // Unsatisfied can only arise from a check-scoped Disposition contract
    // (e.g. WRGA's check-only analysis modes). Those stay report-only:
    // `nsl check` must keep answering "is this program well-formed", not
    // "did an optimizer fire".
    let hard = entry != Subcommand::Check && !allow_inert;

    eprintln!(
        "{}: {} requested feature(s) were silently inert:",
        if hard { "error" } else { "warning" },
        unsat.len(),
    );
    for o in &unsat {
        if let OutcomeState::Unsatisfied { owner } = &o.state {
            eprintln!("  {}: owner {} recorded no disposition", o.surface.render(), owner);
        }
    }
    eprintln!(
        "note: every supported request must produce Applied, an execution witness, \
         or a typed Declined(reason). Re-run with NSL_PASS_TRACE=1 to see what ran{}",
        if hard { ", or --allow-inert-requests to demote this error to a warning" } else { "" },
    );
    if hard {
        // The single-file build path emits its pass trace only after the
        // artifact is written, which this exit prevents — so when the trace
        // is on, print it here or the error's own NSL_PASS_TRACE suggestion
        // would be a dead end on exactly the failing path.
        if nsl_codegen::pass_trace::enabled() {
            eprint!("{}", nsl_codegen::pass_trace::report());
        }
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_scan_takes_long_flags_and_stops_at_separator() {
        let args = [
            "build", "prog.nsl", "--wggo", "full", "--vram-budget=8GiB", "-o", "out",
            "--", "--not-a-request",
        ]
        .into_iter()
        .map(String::from);
        let flags = requested_long_flags_from(args);
        assert_eq!(
            flags,
            vec!["wggo".to_string(), "vram-budget".to_string(), "output".to_string()],
            "-o is the output short alias and must surface as a request"
        );
    }

    #[test]
    fn short_aliases_map_to_their_long_forms() {
        let args = ["-w", "m.nslm", "-o", "out"].into_iter().map(String::from);
        assert_eq!(
            requested_long_flags_from(args),
            vec!["weights".to_string(), "output".to_string()]
        );
        // The alias list must cover every `short = '..'` args.rs declares —
        // a new short flag must be added here or its requests escape the
        // reconciler.
        let src = include_str!("args.rs");
        let shorts = src.matches("short = '").count();
        assert_eq!(
            shorts, 2,
            "args.rs declares {shorts} short flags; update SHORT_ALIASES in \
             requested_long_flags_from and this count"
        );
    }

    #[test]
    fn argv_scan_dedupes_repeated_flags() {
        let args = ["--wggo", "full", "--wggo", "off"].into_iter().map(String::from);
        assert_eq!(requested_long_flags_from(args), vec!["wggo".to_string()]);
    }
}
