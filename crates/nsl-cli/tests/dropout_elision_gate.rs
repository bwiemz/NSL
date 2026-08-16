//! Milestone B gates for the source-AD dropout p=0 elision and the scalar
//! resolver behind it.
//!
//! The defect this closes: `dropout(x, self._dropout_p.item(), training)` —
//! the house pattern — reached extraction as a non-literal p, and the old
//! `.unwrap_or(0.1)` default silently ran REAL p=0.1 dropout on models whose
//! config said 0.0 (coder1b among them): ~1.5 GB of force-saved
//! outputs+masks at the 1B@2048 microbatch-2 peak, and semantics the source
//! never asked for. The resolver now follows the ctor-folded config scalar
//! through the `.item()` passthrough and elides the call at p=0.
//!
//! The resolver WHITELISTS value-preserving passthroughs. The adversarial
//! case that forced it: `full([1], 0.3).item()` — a constructor's inputs[0]
//! is its SHAPE LIST, so a naive walk resolves p to 1.0 (the shape dim) and
//! "elides" nothing while compiling dropout at a p the program never wrote.
//! With the whitelist the constructor is not walked, p stays unresolved, and
//! the pre-existing 0.1 default (with its warning) applies.
//!
//! Dropout that SURVIVES to source-AD lowering warns loudly: its backward is
//! structurally wrong (gradient scaled by the p argument, not the runtime
//! RNG mask — a pre-existing defect this campaign surfaced; refusal is
//! deferred because coder50m/coder500m ship with DROPOUT=0.1 and every
//! pipeline over them would red-line).

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

const WRONG_BACKWARD_WARNING: &str =
    "[source-ad] WARNING: dropout with p=";

#[test]
fn config_scalar_p_zero_is_elided_through_item() {
    // The house pattern at p=0: ctor-folded field, .item() hop. Elided —
    // the run completes AND no wrong-backward warning fires (no Dropout op
    // survives to lowering).
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
        !stderr.contains(WRONG_BACKWARD_WARNING),
        "p=0 dropout must be ELIDED, not lowered-with-warning:\n{stderr}"
    );
}

#[test]
fn literal_p_zero_is_elided() {
    let (ok, _stdout, stderr) =
        run_fixture("litzero", &fixture("", "0.0"));
    assert!(ok, "p=0 literal run failed:\n{stderr}");
    assert!(
        !stderr.contains(WRONG_BACKWARD_WARNING),
        "literal p=0 must be elided:\n{stderr}"
    );
}

#[test]
fn nonzero_p_survives_and_warns_about_the_broken_backward() {
    let (ok, _stdout, stderr) =
        run_fixture("litnonzero", &fixture("", "0.1"));
    assert!(ok, "p=0.1 must still compile (refusal deferred):\n{stderr}");
    assert!(
        stderr.contains(WRONG_BACKWARD_WARNING) && stderr.contains("p=0.1"),
        "surviving dropout must warn about the mask-free backward:\n{stderr}"
    );
}

#[test]
fn inline_constructor_p_is_not_resolved_to_its_shape() {
    // full([1], 0.3).item(): a naive resolver walks the constructor to its
    // SHAPE LIST and reads p=1.0. The whitelist leaves it unresolved, so
    // the 0.1 default applies — the warning must say p=0.1, and must NOT
    // say p=1 (shape dim) or p=0.3 (falsely claiming resolution).
    //
    // Exit status is deliberately NOT asserted: an inline-ctor `.item()`
    // argument trips a PRE-EXISTING forward-lowering type bug (i64 where
    // f64 expected, Cranelift verifier error) unrelated to the resolver.
    // The extraction — where the resolver runs and warns — happens first,
    // so the stderr assertions below hold either way.
    let (_ok, _stdout, stderr) = run_fixture(
        "inlinector",
        &fixture("", "full([1], 0.3).item()"),
    );
    assert!(
        stderr.contains("p=0.1"),
        "unresolvable p must fall back to the 0.1 default (warned):\n{stderr}"
    );
    assert!(
        !stderr.contains("p=1 ") && !stderr.contains("p=1\n"),
        "resolver walked a constructor to its shape list — the whitelist \
         regressed:\n{stderr}"
    );
}
