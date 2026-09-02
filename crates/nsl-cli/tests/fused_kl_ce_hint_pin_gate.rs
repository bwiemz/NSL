//! `@fused_kl_ce` shape hints that disagree with the models must REFUSE, not
//! read off the end of a device allocation.
//!
//! The same defect class `fused_lm_ce_hint_pin_gate.rs` gates for
//! `@fused_lm_ce`, which the KL-CE lowerings carried unpinned for one more
//! release: `vocab_size` / `hidden_size` / `teacher_hidden` are hand-written
//! decorator literals, the kernels stride BOTH heads from them alone, and
//! the FFI receives only `nsl_tensor_data_ptr` results. `batch_size *
//! seq_len` was already pinned (`nsl_fused_lce_targets_i64_alloc`); the four
//! hidden/vocab extents were not.
//!
//! Two pins per lowering, under DISTINCT site codes, because KL-CE has two
//! (x, W) pairs and a refusal that names the wrong hint gives
//! followable-but-wrong advice: the student pair pins `hidden_size`, the
//! teacher pair pins `teacher_hidden` — and the fixture's dims are
//! asymmetric (HS=32, HT=64) precisely so a swapped attribution cannot pass
//! these arms.
//!
//! Subprocess arms for the same reason as the LM-CE gate: the pin aborts
//! (`extern "C"` — a returned error would be discarded by the lowering), so
//! `#[should_panic]` cannot observe it.
//!
//! On the negative assertions: `!contains("shape-hint pin")` in the control
//! needs no `NEGATIVE_NEEDLES` entry — the refusal arms assert the same
//! site strings positively, so a rename reddens those loudly rather than
//! quietly disarming the control. (The registry pins bracketed `[marker]`
//! tokens; these are plain strings.)

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    success: bool,
    stdout: String,
    stderr: String,
}

const STUDENT_SITE: &str = "@fused_kl_ce student shape-hint pin (hidden_size)";
const TEACHER_SITE: &str = "@fused_kl_ce teacher shape-hint pin (teacher_hidden)";

/// Run the KL-CE fixture on the GPU with `rewrites` applied — a hint rewrite
/// is the ONLY difference between the refusing arms and the control.
fn run_with(rewrites: &[(&str, &str)], tag: &str) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_klcepin_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/fused_klce_pin.nsl"),
    )
    .expect("fused_klce_pin.nsl fixture missing");
    assert!(src.contains("# GPU_PLACEMENT"));
    // Place the INPUTS as well as the models. A train-step input left on the
        // host while parameters are on the GPU is refused since acd2535c
        // (2026-08-25) -- correctly, because every op would otherwise reconcile
        // the WEIGHTS down to host f64. This fixture predates that guard and
        // had gone red unnoticed for eight days, which is what the first full
        // GPU certification run surfaced.
        src = src.replace(
            "# GPU_PLACEMENT",
            "teacher.to(cuda)\nstudent.to(cuda)\nlet x = x.to(cuda)\nlet targets = targets.to(cuda)",
        );
    for (from, to) in rewrites {
        // Exactly one — `replace` rewrites every occurrence, and a marker
        // that drifted to zero makes the arm vacuous.
        assert_eq!(
            src.matches(from).count(),
            1,
            "rewrite marker '{from}' must occur exactly once in the fixture"
        );
        src = src.replace(from, to);
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl run");
    RunOut {
        success: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

/// Assert the run refused at `site`, with a diagnostic naming the values,
/// and NOT at `other_site` — the asymmetric dims make a swapped attribution
/// detectable, so it must be detected.
fn assert_pin_refused(out: &RunOut, site: &str, other_site: &str, must_contain: &[&str]) {
    assert!(
        !out.success,
        "a wrong @fused_kl_ce hint must refuse.\nstdout:\n{}\nstderr:\n{}",
        out.stdout, out.stderr
    );
    assert!(
        out.stderr.contains(site),
        "the refusal must come from {site:?}.\nstderr:\n{}",
        out.stderr
    );
    assert!(
        !out.stderr.contains(other_site),
        "the OTHER pair's pin fired ({other_site:?}) — attribution swapped.\nstderr:\n{}",
        out.stderr
    );
    // Driver signatures of the pre-fix failure, not the constant the pin's
    // own message quotes (the self-collision the LM-CE gate shipped once).
    for signature in ["cuMemcpyHtoD_v2", "cannot unwind"] {
        assert!(
            !out.stderr.contains(signature),
            "the pin must fire BEFORE the launch; {signature:?} means the \
             wrong hint reached the hardware.\nstderr:\n{}",
            out.stderr
        );
    }
    for needle in must_contain {
        assert!(
            out.stderr.contains(needle),
            "the diagnostic must contain {needle:?} so the fix is mechanical.\
             \nstderr:\n{}",
            out.stderr
        );
    }
}

/// A wrong STUDENT `hidden_size` refuses at the student site, naming both
/// the wrong hint and the value the tensor implies.
#[test]
#[ignore = "requires CUDA GPU"]
fn student_hidden_size_disagreeing_is_refused_gpu() {
    let out = run_with(&[("hidden_size = 32", "hidden_size = 64")], "hs");
    assert_pin_refused(
        &out,
        STUDENT_SITE,
        TEACHER_SITE,
        &["hidden_size = 64", "imply hidden_size = 32"],
    );
}

/// A wrong `teacher_hidden` refuses at the TEACHER site. The student pair is
/// untouched and must pass its pin first — a refusal blaming the student
/// would be the misattribution the two site codes exist to prevent.
#[test]
#[ignore = "requires CUDA GPU"]
fn teacher_hidden_disagreeing_is_refused_gpu() {
    let out = run_with(&[("teacher_hidden = 64", "teacher_hidden = 32")], "ht");
    // The BODY names teacher_hidden, not hidden_size — a review found the
    // first version told the user to correct the wrong hint, which would
    // have broken the correct student pair. `h_name` threading fixed it;
    // these needles pin that it stays fixed.
    assert_pin_refused(
        &out,
        TEACHER_SITE,
        STUDENT_SITE,
        &["teacher_hidden = 32", "imply teacher_hidden = 64"],
    );
}

/// A wrong `vocab_size` refuses against the STUDENT head weight (the first
/// pair pinned), naming the vocabulary the weight implies. 512 keeps
/// `vocab_tile = 128` legal so the mutation survives to the pin.
#[test]
#[ignore = "requires CUDA GPU"]
fn vocab_size_disagreeing_is_refused_gpu() {
    let out = run_with(&[("vocab_size = 256", "vocab_size = 512")], "v");
    assert_pin_refused(
        &out,
        STUDENT_SITE,
        TEACHER_SITE,
        &["vocab_size = 512", "implying vocab_size = 256"],
    );
}

/// Control: the unmodified fixture still runs the FUSED path to completion —
/// the pins must be false-positive-free on the shape the substitution
/// actually produces, and the build report proves the fused kernel (not a
/// fallback) is what ran.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_unmodified_fixture_still_trains_gpu() {
    let out = run_with(&[], "ctrl");
    assert!(
        out.success,
        "control failed.\nstdout:\n{}\nstderr:\n{}",
        out.stdout, out.stderr
    );
    assert!(
        out.stderr.contains("Fused KL-CE: teacher+student logits never materialized"),
        "the fused path must be the one exercised.\nstderr:\n{}",
        out.stderr
    );
    assert!(
        !out.stderr.contains("shape-hint pin"),
        "no pin may fire on agreeing hints.\nstderr:\n{}",
        out.stderr
    );
}
