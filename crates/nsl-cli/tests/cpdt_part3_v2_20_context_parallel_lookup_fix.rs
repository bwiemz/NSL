//! CPDT Part III v2.20 — `@context_parallel` lookup-fix CLI gate.
//!
//! Closes one of the v2.19 deferrals. Pre-v2.20, the @context_parallel
//! resolution in `expr/calls.rs` carried the SAME buggy
//! `state.current_function_name.split("__").next()` pattern v2.19
//! fixed for `moe_dispatch{,_ffn,_swiglu}`:
//!
//!   1. `state.current_function_name` is `None` for model method
//!      bodies (only top-level `compile_fn` at func.rs:133 sets it).
//!   2. Even if set to the mangled `__nsl_model_X_method` name,
//!      `split("__").next()` returns the empty string before the
//!      leading `__`.
//!
//! Both failure modes meant that inside ANY model method body the
//! `@context_parallel` decorator was a SILENT no-op — the ring-
//! attention codepath at calls.rs:1224 never executed, and the
//! `[nsl] Context parallelism active: ring_size=N` telemetry never
//! fired.
//!
//! v2.20 generalizes the v2.19 helper to a free function
//! `crate::moe::resolve_decorator_config_for_call_site<T>` over any
//! `HashMap<String, T>` keyed by `"{model}.{field}"`, and applies it
//! to BOTH `@context_parallel` (this test) and `@speculative` (no
//! direct telemetry — see the `decorator_resolver_*` unit tests in
//! `crates/nsl-codegen/src/moe.rs` for multi-model resolver coverage
//! across all branches).
//!
//! ## Freshly-surfaced v2.next deferral: ring-attention IR emission
//!
//! Reaching the @context_parallel codepath revealed THREE independent
//! pre-existing bugs in the ring-attention IR emission at
//! calls.rs:1224-1357 that were INVISIBLE pre-v2.20 (the lookup
//! always returned None, so the emission code was never reached):
//!
//!   1. **Cranelift type rejection (what this test pins via "Verifier
//!      errors")**: `scale_val` is an F64 Cranelift Value, but
//!      `builtins.rs:1714-1732` declares all 13 args of
//!      `nsl_ring_attention` as I64. The verifier rejects the
//!      type mismatch and the build aborts before the runtime is
//!      ever consulted.
//!   2. **Codegen / FFI positional misalignment (independent of #1)**:
//!      codegen at calls.rs:1264-1278 emits args in the order
//!      `(ctx, q_part, k_val, v_val, scale_val, causal, null×7)`,
//!      but the runtime FFI in
//!      `crates/nsl-runtime/src/context_parallel/ffi.rs:29-45`
//!      declares the parameters as
//!      `(ctx_handle, scale_bits, causal, block_table_ptr, k_pool_ptr,
//!      v_pool_ptr, block_size, output_ptr, ptx_ptr, name_ptr,
//!      block_q, block_kv, shared_mem_bytes)`. Only position 0
//!      (`ctx_handle`) lines up — the codegen would land `q_part`
//!      where `scale_bits` is expected, `scale_val` where
//!      `k_pool_ptr` is expected, etc. Even if bug #1 were fixed via
//!      bitcast, this semantic misalignment would still produce
//!      wrong behaviour.
//!   3. **Runtime impl is a stub**: `nsl_ring_attention` in
//!      `context_parallel/ffi.rs:29-45` returns 0 unconditionally —
//!      no actual ring-attention happens regardless of how the args
//!      are wired.
//!
//! As a result the build now fails with `Compilation error: Verifier
//! errors` rather than silently no-op'ing. This is a strict
//! improvement (loud failure > silent wrong behaviour) but closing
//! the loop requires a v2.next cycle that picks one canonical
//! ring-attention shape and aligns the codegen + builtins.rs sig +
//! runtime impl together. The test below pins the v2.20 lookup fix
//! by asserting the telemetry line fires (proving the resolver
//! correctly picked up the per-model config) while accepting the
//! downstream verifier failure.

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
fn context_parallel_lookup_fires_telemetry_from_model_method_body() {
    // V2.20 LOOKUP FIX: pre-v2.20 the `@context_parallel(ring_size=N)`
    // on a model field was completely ignored when its forward method
    // called `scaled_dot_product_attention`. The lookup at
    // calls.rs:1218 now correctly resolves via
    // `self.current_method_model_name`, so the telemetry line
    // `[nsl] Context parallelism active: ring_size=2` fires — proving
    // the resolver fix reaches the ring-attention codepath.
    //
    // Build fails downstream with a Cranelift Verifier error from the
    // ring-attention IR emission (see file header — v2.next
    // deferral). The fact that the LOOKUP fires is the v2.20 fix; the
    // downstream emission bug is independent.
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

    cmd.assert()
        .failure()
        .stderr(
            predicate::str::contains("Context parallelism active: ring_size=2").and(
                predicate::str::contains("Verifier errors"),
            ),
        );
}
