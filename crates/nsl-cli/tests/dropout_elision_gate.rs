//! Gates for the source-AD dropout p=0 elision, the scalar resolver behind
//! it, and the unresolved-p refusal.
//!
//! History: `dropout(x, self._dropout_p.item(), training)` — the house
//! pattern — reached extraction as a non-literal p, and the old
//! `.unwrap_or(0.1)` default silently ran REAL p=0.1 dropout on models whose
//! config said 0.0 (coder1b among them): ~1.5 GB of force-saved
//! outputs+masks at the 1B@2048 microbatch-2 peak, and semantics the source
//! never asked for. Milestone B fixed the resolver (it follows the
//! ctor-folded config scalar through the `.item()` passthrough and elides
//! the call at p=0); the dropout-mask campaign then REMOVED the 0.1 default
//! entirely — a genuinely unresolvable p now refuses the compile instead of
//! silently training a different model than the config describes.
//!
//! The resolver WHITELISTS value-preserving passthroughs. The adversarial
//! case that forced it: `full([1], 0.3).item()` — a constructor's inputs[0]
//! is its SHAPE LIST, so a naive walk resolves p to 1.0 (the shape dim). With
//! the whitelist the constructor is not walked, p stays unresolved, and the
//! refusal fires. The two failure modes have DISTINCT messages ("could not
//! be resolved" vs the out-of-range "outside [0, 1)"), so a whitelist
//! regression that resolves p=1.0 is still caught here.
//!
//! Elision witness: a p=0 dropout that SURVIVED extraction would hit the
//! DropoutMask lowering guard (p outside (0,1) is a hard CodegenError), so a
//! successful run IS the elision proof.
//!
//! Backward correctness (source-vs-tape mask parity) lives in
//! `dropout_backward_parity_gate.rs`.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// A tiny trainable model whose forward routes through dropout with `p_expr`
/// as the probability argument.
fn fixture(p_field: &str, p_expr: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([2, 2])
{p_field}

    fn forward(self, x: Tensor, training: bool) -> Tensor:
        let h = x @ self.w
        return dropout(h, {p_expr}, training)

let m = Tiny()

let x = full([2, 2], 2.0)
let y = zeros([2, 2])
train(model = m, epochs = 2):
    optimizer: AdamW(lr = 0.01)
    step(batch):
        let pred = m.forward(x, true)
        let loss = mse_loss(pred, y)

print("DROPOUT_GATE_DONE")
"#
    )
}

fn run_fixture(tag: &str, src: &str) -> (bool, String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_dropgate_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let res = (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    );
    let _ = std::fs::remove_dir_all(&tmp);
    res
}

/// The pre-fix once-per-process warning. It must never come back: the
/// backward now consumes the exact forward mask, and anything the compiler
/// cannot make correct refuses instead of warning.
const OLD_WRONG_BACKWARD_WARNING: &str = "[source-ad] WARNING: dropout with p=";
/// The unresolved-p refusal (extraction; see source_ad.rs dropout arm).
const UNRESOLVED_REFUSAL: &str = "could not be resolved to a compile-time constant";
/// The out-of-range refusal — distinct from UNRESOLVED so a resolver
/// regression that walks a ctor to its shape list (p=1.0) is still caught.
const OUT_OF_RANGE_REFUSAL: &str = "outside [0, 1)";

#[test]
fn config_scalar_p_zero_is_elided_through_item() {
    // The house pattern at p=0: ctor-folded field, .item() hop. Elided —
    // the run completes (a surviving p=0 op would refuse at lowering).
    let (ok, stdout, stderr) = run_fixture(
        "cfgzero",
        &fixture(
            "    _dropout_p: Tensor = full([1], 0.0)",
            "self._dropout_p.item()",
        ),
    );
    assert!(ok, "p=0 config-scalar run failed:\n{stderr}");
    assert!(stdout.contains("DROPOUT_GATE_DONE"), "fixture did not finish");
    assert!(
        !stderr.contains(OLD_WRONG_BACKWARD_WARNING),
        "the pre-fix wrong-backward warning is back:\n{stderr}"
    );
}

#[test]
fn literal_p_zero_is_elided() {
    let (ok, _stdout, stderr) = run_fixture("litzero", &fixture("", "0.0"));
    assert!(ok, "p=0 literal run failed:\n{stderr}");
}

#[test]
fn nonzero_p_compiles_and_trains_without_the_old_warning() {
    // p=0.1 survives to lowering and now trains with the mask-correct
    // backward — no warning, no refusal. (Numerical parity with tape AD is
    // gated in dropout_backward_parity_gate.rs.)
    let (ok, stdout, stderr) = run_fixture("litnonzero", &fixture("", "0.1"));
    assert!(ok, "p=0.1 must compile and run:\n{stderr}");
    assert!(stdout.contains("DROPOUT_GATE_DONE"), "fixture did not finish");
    assert!(
        !stderr.contains(OLD_WRONG_BACKWARD_WARNING),
        "the wrong-backward warning must be gone — the backward consumes \
         the exact forward mask now:\n{stderr}"
    );
    assert!(
        !stderr.contains("falling back to tape-based AD"),
        "dropout must lower on the source-AD path, not silently fall back:\n{stderr}"
    );
}

#[test]
fn inline_constructor_p_refuses_instead_of_defaulting() {
    // full([1], 0.3).item(): a naive resolver walks the constructor to its
    // SHAPE LIST and reads p=1.0; the whitelist leaves it unresolved. The
    // old behavior silently defaulted to 0.1 — now it must REFUSE, with the
    // unresolved message (NOT the out-of-range one, which would mean the
    // resolver "resolved" the shape dim 1.0 and the whitelist regressed).
    let (ok, _stdout, stderr) = run_fixture(
        "inlinector",
        &fixture("", "full([1], 0.3).item()"),
    );
    assert!(
        !ok,
        "unresolvable p must refuse the compile, not default:\n{stderr}"
    );
    assert!(
        stderr.contains(UNRESOLVED_REFUSAL),
        "expected the unresolved-p refusal:\n{stderr}"
    );
    assert!(
        !stderr.contains(OUT_OF_RANGE_REFUSAL),
        "resolver walked a constructor to its shape list (p=1.0) — the \
         whitelist regressed:\n{stderr}"
    );
    assert!(
        !stderr.contains("falling back to tape-based AD"),
        "a refusal must abort the compile, not fall back to tape AD:\n{stderr}"
    );
}

#[test]
fn p_one_refuses_out_of_range() {
    // 1/(1-p) is undefined at p=1; the refusal must fire at extraction
    // with the out-of-range message.
    let (ok, _stdout, stderr) = run_fixture("pone", &fixture("", "1.0"));
    assert!(!ok, "p=1.0 must refuse:\n{stderr}");
    assert!(
        stderr.contains(OUT_OF_RANGE_REFUSAL),
        "expected the out-of-range refusal:\n{stderr}"
    );
}
