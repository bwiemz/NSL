//! CPDT Part III v2.22 — `@context_parallel` naive-attention fallback + M34 warning.
//!
//! Supersedes the v2.21 test `cpdt_part3_v2_21_context_parallel_scale_bitcast.rs`
//! (deleted in this commit). v2.21 fixed the F64/I64 scale bitcast so the
//! ring-attention FFI chain emitted verifier-valid Cranelift IR, but the
//! runtime was still a stub returning 0 (v2.20 bug #3) AND the codegen /
//! FFI positional layout only aligned at position 0 (v2.20 bug #2) —
//! meaning every `@context_parallel`-decorated method silently produced
//! zero-tensor output at runtime.
//!
//! v2.22 closes both remaining deferrals pragmatically: the buggy
//! ring-attention emission (`nsl_cp_init` + `nsl_sequence_partition` +
//! `nsl_ring_attention` + `nsl_sequence_gather` + `nsl_cp_destroy`) is
//! removed from `expr/calls.rs` and the `@context_parallel` codepath
//! falls through to the naive attention path (the same path that runs
//! when the decorator is absent). The decorator becomes advisory — the
//! build surfaces a WARNING so the user knows ring attention is not
//! yet distributed, but the computed output is mathematically correct.
//!
//! When M34 (`crates/nsl-runtime/src/context_parallel/*`) lands with a
//! finalized runtime FFI shape, the codegen emission gets rewritten
//! against that shape here and the warning + fallback go away together.
//!
//! ## What v2.22 verifies
//!
//! - Build succeeds (naive path is valid IR — was already tested
//!   throughout the NSL suite before v2.19 ever reached this codepath).
//! - The M34 warning fires from stderr with the exact ring_size the user
//!   supplied — proves the v2.19 multi-model lookup + v2.20 helper
//!   generalization still route the config to the right call site.
//! - No `@context_parallel` warning fires when the decorator is absent —
//!   proves v2.22 didn't turn the warning into an always-on eprintln.
//! - Multi-model composition: two independent `@context_parallel(ring_size=N)`
//!   decorators each get their own warning + own ring size in the
//!   stderr — proves multi-model resolution still works and both
//!   forwards' naive paths compile (was blocked pre-v2.21).
//!
//! Runtime execution correctness is inherited from the pre-existing
//! naive-attention codepath; this test does not link or run the emitted
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
fn context_parallel_falls_through_to_naive_with_m34_warning() {
    // POST-v2.22: `@context_parallel(ring_size=2)` still resolves via
    // the v2.19 lookup fix + v2.20 helper generalization, but the
    // buggy ring-attention emission has been deleted. Build succeeds
    // via the naive attention path (the same path used when the
    // decorator is absent) and stderr surfaces the M34 warning so the
    // user knows the decorator is currently advisory.
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
        predicate::str::contains(
            "@context_parallel(ring_size=2) recognized but the ring-attention runtime is incomplete",
        )
        .and(predicate::str::contains("falling through to naive attention"))
        .and(predicate::str::contains("M34 in progress")),
    );
    assert!(
        out.exists(),
        "naive-attention fallback must emit a valid object file"
    );
}

#[test]
fn plain_model_without_context_parallel_emits_no_warning() {
    // NEGATIVE CONTROL: no `@context_parallel` decorator → no warning
    // in stderr. Proves v2.22 didn't turn the warning into an
    // always-on `eprintln!` and that the lookup fix correctly
    // discriminates on the presence of the decorator.
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
            .and(predicate::str::contains("ring-attention runtime").not()),
    );
    assert!(out.exists());
}

/// Two models, each with its own `@context_parallel(ring_size=N)` on
/// a sentinel field, each forward calling
/// `scaled_dot_product_attention`. Post-v2.22, both forwards compile
/// via the naive path and stderr surfaces two independent warnings
/// with each model's own ring size — proving the v2.19 multi-model
/// lookup fix + v2.20 helper generalization still route correctly
/// after v2.22's emission rewrite.
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
    // Two independent `@context_parallel(ring_size=N)` decorators on
    // two different models. Each model's forward hits the naive
    // fallback with its own ring size in the warning message. Proves
    // v2.19 lookup + v2.20 helper generalization + v2.22 fallback
    // compose end-to-end for the multi-model case.
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
        predicate::str::contains(
            "@context_parallel(ring_size=2) recognized but the ring-attention runtime is incomplete",
        )
        .and(predicate::str::contains(
            "@context_parallel(ring_size=4) recognized but the ring-attention runtime is incomplete",
        )),
    );
    assert!(
        out.exists(),
        "multi-model naive-attention fallback must emit a valid object file"
    );
}
