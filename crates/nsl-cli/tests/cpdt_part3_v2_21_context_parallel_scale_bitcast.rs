//! CPDT Part III v2.21 — `@context_parallel` scale bitcast CLI gate.
//!
//! Supersedes the v2.20 test `cpdt_part3_v2_20_context_parallel_lookup_fix.rs`
//! (deleted in this commit). That test pinned a BUILD FAILURE — v2.20's
//! lookup fix reached the ring-attention emission for the first time and
//! surfaced a downstream Cranelift Verifier error because `scale_val`
//! (F64 for a `0.125` literal) landed on a builtins.rs-declared I64
//! slot in the `nsl_ring_attention` FFI call.
//!
//! v2.21 closes that specific bug (#1 in v2.20's memory) using the same
//! polymorphic scale-bit-cast pattern established in
//! `crates/nsl-codegen/src/wengert_lower.rs:355-388`:
//!
//! ```text
//! F64 → fdemote → F32 → bitcast → I32 → uextend → I64 (upper 32 zero)
//! F32 →                                bitcast → I32 → uextend → I64
//! tensor handle → nsl_tensor_item → F64 → fdemote → F32 → bitcast → I32 → uextend → I64
//! ```
//!
//! This matches the `scale_bits: i64` convention used by
//! `nsl_speculative_decode_step` and the F32-bit-pattern-in-I64
//! convention the runtime FFI declaration at
//! `crates/nsl-runtime/src/context_parallel/ffi.rs:29-45` implies via
//! the `_scale_bits` parameter name.
//!
//! ## Remaining v2.next deferrals (narrowed from v2.20's 3-bug list)
//!
//! v2.21 closes bug #1 only. Two independent bugs still surface AFTER
//! this cycle:
//!
//!   * **Bug #2 — codegen/FFI positional misalignment**: codegen at
//!     `expr/calls.rs` emits args in the order
//!     `(ctx, q_part, k_val, v_val, scale_bits, causal, null×7)` but
//!     the runtime FFI declares the parameter positions as
//!     `(ctx_handle, scale_bits, causal, block_table_ptr, k_pool_ptr,
//!     v_pool_ptr, block_size, output_ptr, ptx_ptr, name_ptr, block_q,
//!     block_kv, shared_mem_bytes)`. Only position 0 (`ctx_handle`)
//!     aligns. Post-v2.21 the correctly-bit-cast `scale_bits_i64`
//!     lands in the runtime's `k_pool_ptr` slot (codegen position 4 =
//!     runtime position 4), while `q_part` (a tensor pointer) lands
//!     in the runtime's `scale_bits` slot (runtime position 1), and
//!     `causal_flag` lands in `v_pool_ptr` (runtime position 5).
//!     Semantically wrong bytes reach wrong slots at every non-zero
//!     position, but the Cranelift verifier is silent because every
//!     arg is I64 in `builtins.rs`. The bitcast fix is thus a
//!     Cranelift-IR-only correctness improvement — it does NOT move
//!     scale into the right runtime slot. Whoever closes bug #2 must
//!     re-audit the whole call-arg vector against the runtime FFI
//!     signature; the v2.21 bitcast alone must not be read as
//!     "scale is now delivered correctly."
//!   * **Bug #3 — runtime impl is a stub**: `nsl_ring_attention`
//!     returns 0 unconditionally, so nothing observable happens at
//!     runtime regardless of arg alignment.
//!
//! These two are functionally invisible in test output — the build
//! succeeds, telemetry fires, and the returned value is 0 (which is
//! wrong output but doesn't crash). Both are M34 completion work,
//! independent of v2.21's Cranelift type-alignment fix.

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

#[test]
fn context_parallel_scale_bitcast_produces_verifier_valid_ir_only() {
    // V2.21 SCOPE: `scale_val` (F64 from `0.125`) gets fdemoted to
    // F32, bitcast to I32, and uextended to I64 before being passed
    // to `nsl_ring_attention` at codegen position 4. Cranelift's
    // verifier accepts the resulting IR, the build succeeds, and the
    // telemetry line (a compile-time eprintln) still fires — proving
    // the v2.19+v2.20 lookup fix is intact and the v2.21 IR-level
    // type fix works.
    //
    // NOT verified by this test:
    //   - Runtime semantic correctness — the .o file is never linked
    //     or executed, so the runtime stub (bug #3, returns 0
    //     unconditionally) never runs.
    //   - FFI arg alignment — bug #2 in the docstring above still
    //     places the correctly-bitcast scale in the runtime's
    //     `k_pool_ptr` slot rather than `scale_bits`. Cranelift is
    //     silent because every arg is I64.
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

    cmd.assert().success().stderr(predicate::str::contains(
        "Context parallelism active: ring_size=2",
    ));
    assert!(
        out.exists(),
        "IR type-check succeeds and object emit runs; runtime execution and FFI arg alignment NOT verified — see docstring bugs #2/#3"
    );
}

/// Two models, each with its own `@context_parallel(ring_size=N)` on
/// a sentinel field, each forward calling
/// `scaled_dot_product_attention`. Newly possible post-v2.21 (was
/// blocked before because the first model's ring-attention emission
/// hit the Cranelift verifier and halted compilation).
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
fn multi_model_context_parallel_each_forward_picks_its_own_ring_size() {
    // POST-v2.19 + v2.20 + v2.21: two models with independent
    // `@context_parallel(ring_size=N)` decorators each get their own
    // ring size honored. Both forwards compile (v2.21 IR fix) and both
    // telemetry lines fire (v2.19 lookup fix + v2.20 helper
    // generalization for the @context_parallel site).
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
        predicate::str::contains("Context parallelism active: ring_size=2").and(
            predicate::str::contains("Context parallelism active: ring_size=4"),
        ),
    );
    assert!(
        out.exists(),
        "both forwards' IR type-checks succeed and object emit runs; runtime execution and FFI arg alignment NOT verified — see docstring bugs #2/#3"
    );
}
