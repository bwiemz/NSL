//! Which fused linear-CE kernel family a compile picks, and the one bad
//! combination it used to pick silently.
//!
//! # What was unguarded
//!
//! `use_gemm` (wengert_lower.rs, forward and backward) decides between the
//! GEMM-chunked runtime path and the v1 per-row-CTA PTX kernels. At
//! production shape the gap is 335x forward and 504x backward (486 / 2222 ms
//! against 1.45 / 4.41 ms at rows=1024, V=49152, H=512 — measured in
//! `fused_linear_ce_perf_probe.rs`). If either predicate inverted, coder50m's
//! loss head would become three orders of magnitude slower and **every gate
//! in this repo would have stayed green**:
//!
//! * `fused_linear_ce_gemm_numerical.rs` and `fused_linear_ce_perf_probe.rs`
//!   call the FFI directly and are `#[ignore]`d;
//! * the only end-to-end fixture that reads the launch counters
//!   (`csla_fused_lmce.nsl`) exercises the small-vocab v1 path, and its
//!   consuming test asserts nothing about them.
//!
//! So the compile-time routing decision had no coverage at all. It now names
//! itself on stderr (`[fused-lce] <phase> route=…`) and this gate pins it.
//!
//! # The silent combination
//!
//! A 16-bit `dtype=` hint at large vocab compiles clean and takes the slow
//! kernels. That combination is SUPPORTED — `fused_linear_ce_precision_cast_
//! lowering.rs` pins the large-vocab 16-bit cast insertion as a contract — so
//! the fix is to make the cost impossible to miss, not to refuse it. The arms
//! below pin the warning and the route, and the control at vocab 128 shows the
//! warning is scoped to the threshold rather than to 16-bit storage.

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

struct BuildOut {
    success: bool,
    stderr: String,
}

/// Compile a fixture to an object file. `--emit-obj` stops before linking and
/// before any GPU is needed, and the routing decision is made during lowering,
/// so this reaches it on a CPU-only runner.
fn build_fixture(fixture_dir: &str, fixture: &str, rewrites: &[(&str, &str)], tag: &str) -> BuildOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_lceroute_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let mut src = std::fs::read_to_string(root.join(fixture_dir).join(fixture))
        .unwrap_or_else(|e| panic!("{fixture} fixture missing: {e}"));
    for (from, to) in rewrites {
        assert_eq!(
            src.matches(from).count(),
            1,
            "rewrite marker '{from}' must occur exactly once in {fixture} — it was \
             edited out from under this gate (making the arm vacuous) or duplicated"
        );
        src = src.replace(from, to);
    }
    src = src.replace("# GPU_PLACEMENT", "");
    src = src.replace(
        "CSLA_SAVE_PATH",
        &tmp.join("out.nslm").display().to_string().replace('\\', "/"),
    );
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    // `-o` explicitly: `nsl build` with no output path writes the artifact
    // beside the SOURCE, which for a fixture path means dropping a binary
    // into the repo's fixtures directory.
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["build", "--source-ad", "--emit-obj"])
        .arg("-o")
        .arg(tmp.join("prog.o"))
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl build");
    BuildOut {
        success: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

/// Every `[fused-lce]` route line, as `(phase, route)`.
fn routes(stderr: &str) -> Vec<(String, String)> {
    stderr
        .lines()
        .filter_map(|l| l.trim().strip_prefix("[fused-lce] "))
        .filter_map(|rest| {
            let mut it = rest.split_whitespace();
            let phase = it.next()?.to_string();
            let route = it.find_map(|t| t.strip_prefix("route="))?.to_string();
            Some((phase, route))
        })
        .collect()
}

/// The production shape — biasless, 3-D logits — must take the GEMM path in
/// BOTH directions, and say so.
#[test]
fn the_biasless_production_shape_routes_to_gemm() {
    let out = build_fixture(
        "crates/nsl-cli/tests/fixtures",
        "fused_lmce_rank3_biasless.nsl",
        &[],
        "gemm",
    );
    assert!(out.success, "biasless build failed:\n{}", out.stderr);

    let r = routes(&out.stderr);
    assert!(
        !r.is_empty(),
        "no [fused-lce] route line — the substitution did not happen at all, so \
         this gate would be asserting about a program that never fused:\n{}",
        out.stderr
    );
    for phase in ["forward", "backward"] {
        let got = r
            .iter()
            .find(|(p, _)| p == phase)
            .unwrap_or_else(|| panic!("no [fused-lce] line for {phase}:\n{}", out.stderr));
        assert_eq!(
            got.1, "gemm",
            "the biasless production shape took route={} in {phase} — the v1 kernels \
             are 335x/504x slower here, and they cannot even serve a biasless head",
            got.1
        );
    }
}

/// The kill switch is load-bearing: with the GEMM path disabled, the same
/// program must REFUSE rather than quietly take kernels that read a bias it
/// does not have. This is what proves the assertion above is not a constant.
#[test]
fn disabling_the_gemm_path_refuses_the_biasless_head() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_lceroute_off_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/fused_lmce_rank3_biasless.nsl"),
    )
    .unwrap()
    .replace("# GPU_PLACEMENT", "")
    .replace(
        "CSLA_SAVE_PATH",
        &tmp.join("out.nslm").display().to_string().replace('\\', "/"),
    );
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["build", "--source-ad", "--emit-obj"])
        .arg("-o")
        .arg(tmp.join("prog.o"))
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_FUSED_LCE_GEMM", "0")
        .output()
        .expect("spawn nsl build");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        !out.status.success(),
        "NSL_FUSED_LCE_GEMM=0 on a biasless head must refuse:\n{stderr}"
    );
    assert!(
        stderr.contains("biasless head requires the GEMM path"),
        "wrong refusal — expected the biasless-head message:\n{stderr}"
    );
}

/// 16-bit storage at large vocab: the route choice worth three orders of
/// magnitude that a user could not see coming.
///
/// It must COMPILE — `fused_linear_ce_precision_cast_lowering.rs` pins
/// "Large-vocab forward path -> same three-cast insertion" as a contract and
/// Sprint v6 built that support on purpose — and it must say, loudly, what it
/// just chose. (An earlier draft of this gate asserted a refusal here. That
/// was a performance opinion breaking a working, numerically-certified
/// configuration; every sibling refusal at these sites is a correctness one.)
///
/// This is also the only arm that produces `route=v1-large`, so it is what
/// keeps the third route observable at all.
#[test]
fn sixteen_bit_storage_at_large_vocab_warns_and_takes_v1_large() {
    let out = build_fixture(
        "crates/nsl-cli/tests/fixtures",
        "fused_lmce_bf16_large_vocab.nsl",
        &[],
        "bf16large",
    );
    assert!(
        out.success,
        "16-bit storage above the threshold must still COMPILE — it is a pinned \
         contract, not a defect:\n{}",
        out.stderr
    );

    let r = routes(&out.stderr);
    assert!(
        !r.is_empty(),
        "no [fused-lce] route line — this fixture did not fuse, so it says nothing \
         about routing:\n{}",
        out.stderr
    );
    for (phase, route) in &r {
        assert_eq!(
            route, "v1-large",
            "16-bit storage at vocab 16384 should take the large-vocab v1 kernels in \
             {phase}, got route={route}"
        );
    }

    // Anchored on tokens that survive rewrapping, not on the wrapped sentence:
    // the message is a multi-line Rust literal, so matching its line breaks
    // would make a reflow look like a missing warning.
    for needle in [
        "[fused-lce] warning:",
        "16-bit `dtype=` hint",
        "335x",
        "504x",
    ] {
        assert!(
            out.stderr.contains(needle),
            "the warning must carry {needle:?} so a reader can weigh the cost:\n{}",
            out.stderr
        );
    }
}

/// Control: the SAME 16-bit hint below the threshold still compiles. Without
/// this, the refusal above could be a blanket ban on `dtype=` and nobody
/// would notice.
#[test]
fn sixteen_bit_storage_below_the_threshold_still_compiles() {
    // The SAME fixture family as the refusing arm — the parent it was derived
    // from — with only the dtype hint flipped, so the vocabulary is the only
    // difference between this arm and that one. (`fused_lm_ce_e2e_bf16.nsl`
    // looks like the obvious control and is not: its `bias_add` head is not
    // source-AD-extractable, so it never fuses and never routes.)
    let out = build_fixture(
        "crates/nsl-cli/tests/fixtures",
        "csla_fused_lmce.nsl",
        &[(r#"dtype="f32""#, r#"dtype="bf16""#)],
        "bf16small",
    );
    assert!(
        out.success,
        "16-bit storage at vocab 4096 (below the 8192 threshold) must still \
         compile — the v1 kernels are its supported path:\n{}",
        out.stderr
    );
    let r = routes(&out.stderr);
    // NOT just "every route we saw was v1": the first version of this arm said
    // exactly that and passed on a program whose step body source-AD could not
    // extract, so NO routing decision was ever made and the loop ran zero
    // times. A control arm that is vacuous when the fixture stops fusing is
    // worse than none — it reports the refusal is well-scoped when nothing was
    // scoped at all.
    assert!(
        !r.is_empty(),
        "no [fused-lce] route line — this fixture did not fuse, so it says \
         nothing about whether the refusal is correctly scoped:\n{}",
        out.stderr
    );
    for (phase, route) in &r {
        assert_eq!(
            route, "v1",
            "small-vocab 16-bit storage should take the v1 kernels in {phase}, got \
             route={route}"
        );
    }
}

/// The `is_large` half of `use_gemm` — the case the 335x/504x numbers are
/// actually about, and the one no other arm reached.
///
/// The three arms above all turn on `!has_bias` or on the dtype tag; a
/// BIASED f32 head above the threshold is the production large-vocab shape,
/// and it must route to gemm on vocabulary alone. Same fixture as the refusing
/// arm with the dtype hint flipped back to f32, so vocabulary is held constant
/// and the dtype is the only difference between "refuses" and "routes to
/// gemm" — which is also what shows the refusal is dtype-scoped rather than a
/// ban on large vocabularies.
#[test]
fn a_large_vocab_f32_head_routes_to_gemm_on_size_alone() {
    let out = build_fixture(
        "crates/nsl-cli/tests/fixtures",
        "fused_lmce_bf16_large_vocab.nsl",
        &[(r#"dtype="bf16""#, r#"dtype="f32""#)],
        "gemmlarge",
    );
    assert!(
        out.success,
        "a biased f32 head above the large-vocab threshold must compile:\n{}",
        out.stderr
    );
    let r = routes(&out.stderr);
    assert!(
        !r.is_empty(),
        "no [fused-lce] route line — this fixture did not fuse:\n{}",
        out.stderr
    );
    for (phase, route) in &r {
        assert_eq!(
            route, "gemm",
            "a large-vocab f32 head took route={route} in {phase} — this is the \
             shape the 335x/504x measurement was taken on"
        );
    }
}
