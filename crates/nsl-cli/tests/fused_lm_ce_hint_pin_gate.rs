//! `@fused_lm_ce` shape hints that disagree with the model must REFUSE, not
//! read off the end of a device allocation.
//!
//! # The defect this gates
//!
//! `vocab_size` / `hidden_size` / `batch_size` / `seq_len` are hand-written
//! decorator literals. Nothing derives them from the model, and the fused
//! kernels stride `x` as `[rows, hidden_size]` and `W` as
//! `[vocab_size, hidden_size]` from those literals alone — the FFI receives
//! raw device data pointers, so it has no shape to check them against.
//!
//! `batch_size * seq_len` has been pinned against the label tensor since
//! Sprint 2.5 (`nsl_fused_lce_targets_i64_alloc`). `hidden_size` and
//! `vocab_size` were not, and the two failure modes were not comparable.
//! Measured on Coder-50M before the fix, with `hidden_size = 1024` against a
//! `d_model = 512` head:
//!
//! ```text
//! panicked at crates/nsl-runtime/src/cuda/mod.rs:1079:17:
//! cuMemcpyHtoD_v2(4096 bytes) failed: CUDA_ERROR_ILLEGAL_ADDRESS
//!   [context: rmsnorm_f32]
//! ...
//! panic in a function that cannot unwind
//! ```
//!
//! — a diagnostic that names neither the decorator nor the hint, arriving
//! from a subsystem the user never touched. An overread landing inside
//! allocator slack produces no error at all, just a wrong loss.
//!
//! # Why these gates run a subprocess
//!
//! The pin calls `std::process::abort()`. It has to: the codegen call sites
//! in `wengert_lower.rs` discard these FFIs' return values, so a refusal that
//! returns is a refusal nobody hears. An abort cannot be observed by
//! `#[should_panic]` from inside the test process, so each arm asserts on a
//! child `nsl run`'s exit status AND its stderr.
//!
//! The arithmetic itself is unit-tested without a GPU in
//! `nsl-runtime::fused_linear_ce::hint_extent_tests`. What only an e2e run
//! can show is that the pin is REACHED on the production lowering — which is
//! the half that was missing.
//!
//! # On the negative assertions here
//!
//! `!stderr.contains("[fused-lm-ce]")` in the control arms is registered in
//! `exec_markers::NEGATIVE_NEEDLES` — a negative assertion cannot fail, so
//! renaming that prefix would otherwise leave it permanently, silently true.
//! `!stderr.contains("@fused_lm_ce shape-hint pin")` needs no entry: the
//! refusal arms assert the SAME string positively via `assert_pin_refused`,
//! so a rename reddens those loudly rather than quietly disarming these.

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

/// The repo's canonical fully-hinted `@fused_lm_ce` program: a REAL bias,
/// FLAT `[rows, V]` logits (`x_rank3 = false`), vocab 128, batch 2 x seq 64.
const FLAT_BIASED: &str = "csla_fused_lmce.nsl";

/// The shape `models/coder50m/pretrain.nsl` actually uses, and the opposite
/// of `FLAT_BIASED` on both axes the substitution branches on: a BIASLESS
/// weight-tied head and 3-D logits flattened at the loss (`x_rank3 = true`,
/// GEMM-chunked). A control arm on only one of these cannot show the pin is
/// false-positive-free on the other.
const RANK3_BIASLESS: &str = "fused_lmce_rank3_biasless.nsl";

/// Run `fixture` on the GPU with `rewrites` applied.
///
/// A hint rewrite is the ONLY difference between the refusing arms and the
/// controls below.
fn run_with(fixture: &str, rewrites: &[(&str, &str)], tag: &str) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_hintpin_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let mut src =
        std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures").join(fixture))
            .unwrap_or_else(|e| panic!("{fixture} fixture missing: {e}"));
    assert!(src.contains("# GPU_PLACEMENT"));
    src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    src = src.replace(
        "CSLA_SAVE_PATH",
        &tmp.join("out.nslm").display().to_string().replace('\\', "/"),
    );
    for (from, to) in rewrites {
        // Exactly one, not at least one: `replace` rewrites every occurrence,
        // so a second one appearing in the fixture later would silently
        // mutate more than the decorator hint this arm means to mutate.
        assert_eq!(
            src.matches(from).count(),
            1,
            "rewrite marker '{from}' must occur exactly once in the fixture — \
             it was edited out from under this gate (making the arm vacuous) \
             or duplicated (making the mutation wider than intended)"
        );
        src = src.replace(from, to);
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--deterministic"])
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

/// Assert the run refused, and that its diagnostic is the hint pin's rather
/// than a downstream shape error or an ILLEGAL_ADDRESS abort.
fn assert_pin_refused(out: &RunOut, must_contain: &[&str]) {
    assert!(
        !out.success,
        "a wrong @fused_lm_ce hint must refuse.\nstdout:\n{}\nstderr:\n{}",
        out.stdout, out.stderr
    );
    assert!(
        out.stderr.contains("@fused_lm_ce shape-hint pin"),
        "the refusal must come from the hint pin, not from whatever the \
         kernel happened to trip over downstream.\nstderr:\n{}",
        out.stderr
    );
    // Match the DRIVER's signatures for the pre-fix failure, not the string
    // "CUDA_ERROR_ILLEGAL_ADDRESS" — the pin's own message quotes that name
    // while explaining what it prevents, so excluding it could never hold.
    // (This gate failed on exactly that collision when first written.)
    for signature in ["cuMemcpyHtoD_v2", "cannot unwind"] {
        assert!(
            !out.stderr.contains(signature),
            "the pin must fire BEFORE the launch; {signature:?} in stderr \
             means the wrong hint still reached the hardware.\nstderr:\n{}",
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

/// The measured production failure: a `hidden_size` LARGER than the model's,
/// which used to reach the kernel and read past the activation allocation.
#[test]
#[ignore = "requires CUDA GPU"]
fn hidden_size_larger_than_the_model_is_refused_gpu() {
    let out = run_with(FLAT_BIASED, &[("hidden_size=64", "hidden_size=128")], "h_big");
    assert_pin_refused(
        &out,
        &["hidden_size = 128", "imply hidden_size = 64"],
    );
}

/// The other direction. It surfaces downstream as a shape mismatch even
/// without the pin, but relying on that would leave the guard asymmetric —
/// and the message here names the value to use, which the shape error does
/// not.
#[test]
#[ignore = "requires CUDA GPU"]
fn hidden_size_smaller_than_the_model_is_refused_gpu() {
    let out = run_with(FLAT_BIASED, &[("hidden_size=64", "hidden_size=32")], "h_small");
    assert_pin_refused(&out, &["hidden_size = 32", "imply hidden_size = 64"]);
}

/// `vocab_size` is pinned against the LM-head weight. 256 keeps
/// `vocab_tile = 128` legal (a multiple of 128 and <= vocab_size), so this
/// arm reaches the pin rather than tripping the emitter's tile validation
/// first — the mutation has to survive long enough to test what it claims to.
#[test]
#[ignore = "requires CUDA GPU"]
fn vocab_size_disagreeing_with_the_head_is_refused_gpu() {
    let out = run_with(FLAT_BIASED, &[("vocab_size=128", "vocab_size=256")], "v_big");
    assert_pin_refused(&out, &["vocab_size = 256", "implying vocab_size = 128"]);
}

/// REVIEW FINDING. Element counts alone cannot see a `batch_size`/`seq_len`
/// SWAP: the forward FFI collapses the pair into `rows = b * s`, so 2 x 64
/// and 64 x 2 are indistinguishable there. The BACKWARD does not collapse it
/// — on the rank-3 path it allocates `dx` as
/// `[batch_size, seq_len, hidden_size]`, so a swapped pair yields a `dx` with
/// the right rank and element count and the wrong dims, flowing into the
/// shape-sensitive norm/residual backward.
///
/// Run on the rank-3 fixture because that is the path where the swap has
/// consequences.
#[test]
#[ignore = "requires CUDA GPU"]
fn transposed_batch_and_seq_is_refused_gpu() {
    // The `vocab_tile` suffix keeps this to the DECORATOR: the bare
    // `batch_size=2, seq_len=64` also matches the DataLoader line, and
    // rewriting both would change the actual shape instead of just the hints
    // — which is the mutation this arm is NOT trying to make. The uniqueness
    // assertion in `run_with` caught that.
    let out = run_with(
        RANK3_BIASLESS,
        &[(
            "batch_size=2, seq_len=64, vocab_tile=128",
            "batch_size=64, seq_len=2, vocab_tile=128",
        )],
        "bs_swap",
    );
    assert_pin_refused(&out, &["batch_size = 64", "seq_len = 2"]);
}

/// ANTI-VACUITY for the path this change actually ships. `pretrain.nsl` is
/// biasless, rank-3 and GEMM-chunked; the control below runs the flat/biased
/// v1-PTX fixture, so on its own it cannot show the pin leaves the shipped
/// combination alone.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_rank3_biasless_fixture_still_trains_gpu() {
    let out = run_with(RANK3_BIASLESS, &[], "control_rank3");
    assert!(
        out.success,
        "the rank-3 biasless head's own hints are correct and must not be \
         refused.\nstderr:\n{}",
        out.stderr
    );
    assert!(
        !out.stderr.contains("@fused_lm_ce shape-hint pin"),
        "no pin diagnostic may fire on agreeing hints.\nstderr:\n{}",
        out.stderr
    );
    // A decline would mean the composite path ran and the pin was never
    // reached, making the arm vacuous as a false-positive control.
    assert!(
        !out.stderr.contains("[fused-lm-ce]"),
        "the fusion must actually engage on this fixture.\nstderr:\n{}",
        out.stderr
    );
    let losses = out
        .stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.lines().filter(|l| l.contains("tensor([")).count())
        .unwrap_or(0);
    assert!(losses > 0, "control arm did not train.\nstdout:\n{}", out.stdout);
}

/// ANTI-VACUITY. Every assertion above is on a FAILING run, so all three
/// would pass against a pin that refused unconditionally — including one
/// that broke every correct `@fused_lm_ce` program in the repo. (That is not
/// hypothetical: the first version of this pin read the kernel's raw device
/// pointer as an `NslTensor` and killed the correct run too.)
///
/// The unmodified fixture must still train, and must not print the pin's
/// diagnostic.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_unmodified_fixture_still_trains_gpu() {
    let out = run_with(FLAT_BIASED, &[], "control");
    assert!(
        out.success,
        "the fixture's own hints are correct and must not be refused.\
         \nstderr:\n{}",
        out.stderr
    );
    assert!(
        !out.stderr.contains("@fused_lm_ce shape-hint pin"),
        "no pin diagnostic may fire on agreeing hints.\nstderr:\n{}",
        out.stderr
    );
    // The fixture prints its loss stream between these markers; an empty one
    // would mean the run "succeeded" without training.
    let losses = out
        .stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.lines().filter(|l| l.contains("tensor([")).count())
        .unwrap_or(0);
    assert!(
        losses > 0,
        "control arm produced no losses — it did not actually train.\
         \nstdout:\n{}",
        out.stdout
    );
}
