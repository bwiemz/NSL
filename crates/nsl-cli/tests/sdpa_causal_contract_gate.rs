//! `scaled_dot_product_attention`'s causal flag means the same thing in
//! both AD modes, in every spelling.
//!
//! The bug this pins: the eager/tape lowering (`expr/calls.rs`) honoured
//! ONLY the named `causal=` spelling and defaulted to NON-causal, while
//! source AD (`source_ad.rs`) read the POSITIONAL slot and defaulted to
//! causal. The stdlib GQA passes the flag positionally
//! (`scaled_dot_product_attention(q, k, v, scale, true)` in
//! `nsl/nn/gqa.nsl`), so under tape AD every causal LM — the 50M coder
//! model included — silently trained with BIDIRECTIONAL attention: each
//! position could read its own future tokens. Label leakage lowers the
//! loss once training starts, so it reads as "the tape is doing better",
//! not as a bug; it was found by an AD differential whose residual
//! disagreement could not be explained by precision.
//!
//! Two properties, and BOTH are load-bearing:
//!   1. the modes agree exactly for every spelling — same program, same
//!      function;
//!   2. causal spellings differ from non-causal ones — otherwise the
//!      agreement in (1) could be satisfied by both modes ignoring the
//!      flag, which is the pre-fix state of exactly one of them.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Raw outcome of one `nsl run` on the fixture.
struct RunOut {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run_raw(spelling: &str, source_ad: bool, tmp: &std::path::Path, tag: &str) -> RunOut {
    let root = workspace_root();
    let template = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/sdpa_causal_contract.nsl"),
    )
    .expect("fixture");
    assert_eq!(
        template.matches("CAUSAL_SPELLING").count(),
        1,
        "the fixture must contain exactly one CAUSAL_SPELLING placeholder"
    );
    let program = template.replace("CAUSAL_SPELLING", spelling);
    let path = tmp.join(format!("sdpa_causal_{tag}.nsl"));
    std::fs::write(&path, program).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run");
    if source_ad {
        cmd.arg("--source-ad");
    }
    let out = cmd
        .arg(&path)
        .current_dir(tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    RunOut {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

/// Losses for one (spelling, mode) cell.
fn run_spelling(spelling: &str, source_ad: bool, tmp: &std::path::Path, tag: &str) -> Vec<f64> {
    let out = run_raw(spelling, source_ad, tmp, tag);
    assert!(
        out.ok && out.stdout.contains("SDPA_CAUSAL_CONTRACT_COMPLETE"),
        "run failed for spelling {spelling:?} (source_ad={source_ad}):\n{}",
        out.stderr
    );
    // The `--source-ad` arm must actually have TAKEN the source-AD path.
    // Codegen falls back to tape when extraction fails, printing this
    // marker — and a silent fallback would make every cross-mode assertion
    // below compare tape against tape, i.e. pass vacuously no matter how
    // far the two lowerings drift.
    if source_ad {
        assert!(
            !out.stderr.contains("source AD extraction failed"),
            "the --source-ad arm fell back to tape AD for spelling {spelling:?}; \
             every cross-mode comparison in this gate would be tape-vs-tape:\n{}",
            out.stderr
        );
    }
    let losses: Vec<f64> = out
        .stdout
        .lines()
        .skip_while(|l| !l.contains("LOSS_STREAM_BEGIN"))
        .take_while(|l| !l.contains("LOSS_STREAM_END"))
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert!(
        losses.len() >= 16,
        "expected 16 micro-step losses for {spelling:?}, got {}",
        losses.len()
    );
    losses
}

#[test]
fn sdpa_causal_flag_means_the_same_thing_in_both_ad_modes() {
    let tmp = tempfile::TempDir::new().unwrap();

    // (spelling, tag, is_causal). The absent spelling defaults to CAUSAL —
    // matching source AD's long-standing default and the documented
    // `@flash_attention` default.
    let cases = [
        (", true", "pos_true", true),
        (", causal=true", "named_true", true),
        ("", "absent", true),
        (", false", "pos_false", false),
        (", causal=false", "named_false", false),
    ];

    let mut causal_ref: Option<Vec<f64>> = None;
    let mut noncausal_ref: Option<Vec<f64>> = None;

    for (spelling, tag, is_causal) in cases {
        let tape = run_spelling(spelling, false, tmp.path(), &format!("{tag}_tape"));
        let src = run_spelling(spelling, true, tmp.path(), &format!("{tag}_src"));

        // (1) the two modes compiled the same function.
        assert_eq!(
            tape.len(),
            src.len(),
            "step counts differ for spelling {spelling:?}"
        );
        for (i, (t, s)) in tape.iter().zip(src.iter()).enumerate() {
            assert!(
                (t - s).abs() < 1e-9,
                "AD modes disagree for spelling {spelling:?} at step {i}: \
                 tape={t} source={s} — the causal flag is being read \
                 differently by the two lowerings"
            );
        }

        // All spellings of the same semantics must agree with each other
        // too (positional == named == default).
        let slot = if is_causal { &mut causal_ref } else { &mut noncausal_ref };
        match slot {
            None => *slot = Some(tape.clone()),
            Some(reference) => {
                for (i, (a, b)) in tape.iter().zip(reference.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-9,
                        "spelling {spelling:?} disagrees with the other \
                         {}causal spellings at step {i}: {a} vs {b}",
                        if is_causal { "" } else { "non-" }
                    );
                }
            }
        }
    }

    // (2) anti-vacuity: causal and non-causal must actually differ, or the
    // agreement above would also hold with the flag ignored entirely.
    let causal = causal_ref.expect("causal cases ran");
    let noncausal = noncausal_ref.expect("non-causal cases ran");
    let gap = causal
        .iter()
        .zip(noncausal.iter())
        .map(|(c, n)| (c - n).abs())
        .fold(0.0f64, f64::max);
    assert!(
        gap > 1e-6,
        "causal and non-causal attention produced identical losses \
         (max gap {gap:e}) — the flag is being ignored, so the \
         cross-mode agreement above is vacuous"
    );
}

/// What the eager path CANNOT resolve, it must refuse — not default.
///
/// `causal` selects the attention function at compile time, and source AD
/// resolves `let`-bound constants through the Wengert graph while this
/// path cannot. Silently defaulting would put the two modes back to
/// compiling different functions for one program (in the opposite
/// direction from the original bug: a variable-bound `false` would train
/// causally under tape and bidirectionally under source AD).
#[test]
fn sdpa_refuses_what_it_cannot_resolve_at_compile_time() {
    let tmp = tempfile::TempDir::new().unwrap();

    // (spelling, tag, expected needle in the refusal)
    let refusals = [
        // A variable is not a literal — this is the regression the review
        // caught in the first version of the fix.
        (
            ", use_causal",
            "var",
            "must be a bool literal",
        ),
        // An int literal is not a bool literal (source AD would read 0 as
        // non-causal; this path would silently keep the default).
        (", 0", "int", "must be a bool literal"),
        // Unknown keyword: source AD consumes slot 5 POSITIONALLY, so a
        // misspelling would mean different things in the two modes.
        (", causl=true", "typo_kwarg", "has no keyword argument"),
        // Arity: a 6th argument has no meaning here.
        (", true, true", "arity", "at most 5 arguments"),
    ];

    for (spelling, tag, needle) in refusals {
        let out = run_raw(spelling, false, tmp.path(), tag);
        assert!(
            !out.ok,
            "spelling {spelling:?} must be REFUSED, but the program compiled and ran"
        );
        assert!(
            out.stderr.contains(needle),
            "spelling {spelling:?} was refused, but not for the documented reason \
             (expected {needle:?}):\n{}",
            out.stderr
        );
    }
}
