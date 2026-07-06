//! M34 v1 — `@context_parallel` single-node behavior.
//!
//! Supersedes `cpdt_part3_v2_22_context_parallel_naive_fallback.rs`. The
//! v2.22 warning ("M34 in progress; falling through to naive attention")
//! was a blanket "not implemented" message. M34 v1 splits the behavior by
//! ring_size to reflect the actual state of the runtime layer:
//!
//! * The runtime `run_ring_attention_full` composer at
//!   `crates/nsl-runtime/src/context_parallel/attention.rs` is verified
//!   correct against `naive_attention` on a matrix of shapes and ring
//!   sizes (2/4/8-way ring, causal + non-causal). Multi-device
//!   distribution (NCCL send/recv, IPC) is what remains.
//!
//! * `ring_size == 1` is a semantic identity — no warning fires. The
//!   decorator carries planning intent for the CPDT pipeline but doesn't
//!   change forward semantics on a 1-rank ring.
//!
//! * `ring_size >= 2` fires a refined warning that names the actual
//!   deferral (multi-device distribution) rather than the blanket
//!   "M34 in progress" v2.22 text.
//!
//! ## What this test verifies
//!
//! - Build succeeds for all three fixtures (single ring_size=2,
//!   ring_size=1 identity, plain no-decorator control).
//! - `ring_size=2` fires the new "single-device ring math is verified"
//!   message with `multi-device distribution` in the body.
//! - `ring_size=1` fires NO `@context_parallel` warning at all
//!   (semantic identity — new in M34 v1).
//! - No-decorator control fires no warning (negative control).
//! - Multi-model composition (two independent `@context_parallel(ring_size=N)`
//!   decorators on different models) still routes each config to its own
//!   call site — v2.19 lookup fix + v2.20 helper still in effect.
//!
//! Runtime execution correctness is inherited from the runtime crate's
//! matrix regression test; this test does not link or run the emitted
//! object.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Model whose forward calls `scaled_dot_product_attention` with an
/// `@context_parallel(ring_size=2)` decorator on a sentinel field.
const SINGLE_MODEL_CP_SRC: &str = r#"model Attn:
    @context_parallel(ring_size=2)
    ring_marker: int = 2
    wq: Tensor = ones([8, 8])
    wk: Tensor = ones([8, 8])
    wv: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        let q = x @ self.wq
        let k = x @ self.wk
        let v = x @ self.wv
        return scaled_dot_product_attention(q, k, v, 0.125)

let m = Attn()
let x = ones([2, 8])
let y = m.forward(x)
"#;

/// M34 v1 addition: `ring_size=1` is a semantic identity. The decorator
/// still parses and routes through the resolver, but the codegen skips
/// the warning entirely.
const SINGLE_MODEL_RING1_SRC: &str = r#"model Attn:
    @context_parallel(ring_size=1)
    ring_marker: int = 1
    wq: Tensor = ones([8, 8])
    wk: Tensor = ones([8, 8])
    wv: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        let q = x @ self.wq
        let k = x @ self.wk
        let v = x @ self.wv
        return scaled_dot_product_attention(q, k, v, 0.125)

let m = Attn()
let x = ones([2, 8])
let y = m.forward(x)
"#;

/// Same shape as `SINGLE_MODEL_CP_SRC` but WITHOUT the `@context_parallel`
/// decorator. Used as the negative control for the "no warning leak"
/// assertion.
const PLAIN_MODEL_SRC: &str = r#"model Attn:
    wq: Tensor = ones([8, 8])
    wk: Tensor = ones([8, 8])
    wv: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        let q = x @ self.wq
        let k = x @ self.wk
        let v = x @ self.wv
        return scaled_dot_product_attention(q, k, v, 0.125)

let m = Attn()
let x = ones([2, 8])
let y = m.forward(x)
"#;

#[test]
fn context_parallel_ring_size_2_emits_multi_device_deferral_warning() {
    // ring_size=2 → M34 v1 refined warning naming multi-device
    // distribution as the actual scope gap (vs v2.22's blanket "M34 in
    // progress"). Build succeeds via the naive path.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("cp.nsl");
    fs::write(&src, SINGLE_MODEL_CP_SRC).unwrap();
    let out = tmp.path().join("cp.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out);

    cmd.assert().success().stderr(
        predicate::str::contains("@context_parallel(ring_size=2)")
            .and(predicate::str::contains("single-device ring math is verified"))
            .and(predicate::str::contains("multi-device distribution"))
            // v2.22's blanket "M34 in progress" text is gone.
            .and(predicate::str::contains("M34 in progress").not()),
    );
    assert!(out.exists(), "naive path must still emit a valid object file");
}

#[test]
fn context_parallel_ring_size_1_is_identity_no_warning() {
    // ring_size=1 → semantic identity. Decorator is recognized and routed
    // through the resolver (proven by the fact that v2.19-v2.22 tests all
    // exercised this path for ring_size=2), but M34 v1 silences the
    // warning entirely for 1-rank rings.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("cp_r1.nsl");
    fs::write(&src, SINGLE_MODEL_RING1_SRC).unwrap();
    let out = tmp.path().join("cp_r1.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out);

    cmd.assert().success().stderr(
        predicate::str::contains("@context_parallel")
            .not()
            .and(predicate::str::contains("multi-device distribution").not())
            .and(predicate::str::contains("single-device ring math").not()),
    );
    assert!(out.exists());
}

#[test]
fn plain_model_without_context_parallel_emits_no_warning() {
    // NEGATIVE CONTROL: no `@context_parallel` decorator → no warning.
    // Proves the codegen didn't turn the warning into an always-on
    // eprintln and that the resolver correctly discriminates on
    // decorator presence.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("plain.nsl");
    fs::write(&src, PLAIN_MODEL_SRC).unwrap();
    let out = tmp.path().join("plain.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out);

    cmd.assert().success().stderr(
        predicate::str::contains("@context_parallel")
            .not()
            .and(predicate::str::contains("ring_size").not()),
    );
    assert!(out.exists());
}

/// Two models, each with its own `@context_parallel(ring_size=N)` on
/// a sentinel field, each forward calling
/// `scaled_dot_product_attention`. Post-M34 v1, both forwards compile
/// via the naive path and stderr surfaces two independent M34 v1
/// warnings with each model's own ring size — proving v2.19 multi-model
/// lookup + v2.20 helper generalization still route correctly after
/// M34 v1's warning refinement.
const TWO_MODEL_CP_SRC: &str = r#"model AttnA:
    @context_parallel(ring_size=2)
    ring_marker: int = 2
    wq: Tensor = ones([8, 8])
    wk: Tensor = ones([8, 8])
    wv: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        let q = x @ self.wq
        let k = x @ self.wk
        let v = x @ self.wv
        return scaled_dot_product_attention(q, k, v, 0.125)

model AttnB:
    @context_parallel(ring_size=4)
    ring_marker: int = 4
    wq: Tensor = ones([8, 8])
    wk: Tensor = ones([8, 8])
    wv: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        let q = x @ self.wq
        let k = x @ self.wk
        let v = x @ self.wv
        return scaled_dot_product_attention(q, k, v, 0.125)

let a = AttnA()
let b = AttnB()
let x = ones([2, 8])
let y0 = a.forward(x)
let y1 = b.forward(x)
"#;

#[test]
fn multi_model_context_parallel_each_forward_gets_its_own_warning() {
    // Two independent `@context_parallel(ring_size=N)` decorators on two
    // different models. Each model's forward emits an M34 v1 warning
    // with its own ring size. Proves v2.19 lookup + v2.20 helper
    // generalization + M34 v1 warning refinement compose end-to-end for
    // the multi-model case.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("cp_multi.nsl");
    fs::write(&src, TWO_MODEL_CP_SRC).unwrap();
    let out = tmp.path().join("cp_multi.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out);

    cmd.assert().success().stderr(
        predicate::str::contains("@context_parallel(ring_size=2)")
            .and(predicate::str::contains("@context_parallel(ring_size=4)"))
            .and(predicate::str::contains("multi-device distribution"))
            // v2.22 blanket text absent from both warnings.
            .and(predicate::str::contains("M34 in progress").not()),
    );
    assert!(
        out.exists(),
        "multi-model naive-attention path must emit a valid object file"
    );
}
