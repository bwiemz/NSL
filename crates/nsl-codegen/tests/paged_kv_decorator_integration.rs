//! Sprint 2 cycle-3 — paper §3.2 paged-KV end-to-end decorator wiring.
//!
//! Mirrors `tree_mask_decorator_integration.rs` (Sprint 1 cycle-3).
//! This test compiles a minimal NSL fixture with `@flash_attention` +
//! `@paged_kv(block_size=32)` and asserts:
//!
//!   1. The compile context's `FlashAttentionConfig::paged` is `true`
//!      (proof the extraction-site arm in `compiler/kernel.rs:1005-1022`
//!      flipped the bool and threaded it through the config
//!      constructor).
//!
//!   2. The synthesized kernel name contains the `_p1_` variant tag
//!      (per `flash_attention.rs::flash_attention_kernel_name`'s
//!      format `flash_attn_p{paged}_r{rope}_...`). The runtime
//!      dispatcher picks the paged kernel by name, so the tag is the
//!      load-bearing wire between config and dispatch.
//!
//!   3. The synthesized PTX contains the paged-only block-table
//!      indirection block (`flash_attention.rs:1622-1649`). The
//!      PARAMETER `block_table_ptr` is declared unconditionally on the
//!      kernel (`flash_attention.rs:438`) and its `ld.param.u64` load
//!      is unconditional (`flash_attention.rs:622`) — same null-when-
//!      paged-false ABI-stability pattern as `dfs_enter_ptr` in
//!      Sprint 1. Only the indirection comment + `div.u64` /
//!      `mul.lo.u64 ... block_size` sequence inside `if config.paged`
//!      is paged-gated. Probing for the gated emission therefore
//!      gives the ground-truth proof that the paged code path fired
//!      — probing for the parameter name would be a false positive
//!      (the EXACT trap class fixed in Sprint 1 review-fix `0a987a73`).
//!
//! Together (1)+(2)+(3) close the loop: source decorator → semantic
//! checker → codegen extraction → config → kernel-name + PTX.
//!
//! Note on `block_size`: `@paged_kv(block_size=32)` exercises the
//! arg-parsing path in `compiler/kernel.rs:1007-1020`, but the parsed
//! value is consumed only for the `block_kv % block_size` alignment
//! check (`compiler/kernel.rs:1167`); it is not stored on
//! `FlashAttentionConfig`. So this test cannot pin `block_size==32`
//! through the compile context — that would require either a config
//! field (deferred follow-on) or runtime-launch inspection (out of
//! scope for a codegen-pin test). The arg-parsing path is still
//! exercised indirectly: a misspelled or rejected arg would cause
//! the extraction site to abort with a codegen error, which would
//! make `compile_flash_attention_kernels` return `Err` and the helper
//! panic before reaching the assertions below.

#![cfg(feature = "test-helpers")]

use nsl_codegen::test_helpers::flash_paged_kv_context_for_source;

const PAGED_KV_FIXTURE: &str = include_str!("fixtures/paged_kv_decorator.nsl");

#[test]
fn paged_kv_decorator_reaches_config_and_kernel_name_and_ptx() {
    let (ctx_set, paged_flag, kernel_name, ptx_has_paged_indirection) =
        flash_paged_kv_context_for_source(PAGED_KV_FIXTURE);

    assert!(
        ctx_set,
        "fixture's @flash_attention must build a compile context — \
         if this fails the decorator scanner regressed independent of \
         the paged_kv wiring"
    );

    assert_eq!(
        paged_flag,
        Some(true),
        "Sprint 2 cycle-3: @paged_kv must flip \
         FlashAttentionConfig::paged to true (extraction site at \
         compiler/kernel.rs:1005-1022 sets `paged = true` when the \
         decorator is present)"
    );

    let kernel_name = kernel_name.expect("context_set=true implies kernel_name=Some");
    assert!(
        kernel_name.contains("_p1_"),
        "Sprint 2 cycle-3: kernel name must contain `_p1_` variant tag \
         when paged=true (per flash_attention_kernel_name format \
         `flash_attn_p{{paged}}_r..._g..._c..._t..._q..._kv...`), got {}",
        kernel_name,
    );

    assert!(
        ptx_has_paged_indirection,
        "Sprint 2 cycle-3: synthesized PTX must contain the paged-only \
         block-table indirection comment when config.paged=true \
         (`flash_attention.rs:1622-1649`, gated on `if config.paged`). \
         The PARAMETER `block_table_ptr` is declared unconditionally \
         and the `ld.param.u64` for it is also unconditional; only \
         the indirection block is paged-gated. Probing for the gated \
         emission gives the ground-truth proof that the paged code \
         path fired — same false-positive trap class fixed in \
         Sprint 1 review fix 0a987a73."
    );
}
