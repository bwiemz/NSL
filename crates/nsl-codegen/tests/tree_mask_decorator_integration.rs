//! Sprint 1 cycle-3 — paper §4 tree-mask end-to-end decorator wiring.
//!
//! This test compiles a minimal NSL fixture with
//! `@flash_attention` + `@tree_mask` and asserts:
//!
//!   1. The compile context's `FlashAttentionConfig::tree_mask` is
//!      `true` (proof the new extraction-site arm in `kernel.rs`
//!      flipped the bool and threaded it through the config
//!      constructor — pre-Sprint-1 cycle-3, this was hardcoded `false`).
//!
//!   2. The synthesized kernel name contains the `_t1_` variant tag
//!      (paper §4: dispatcher picks the tree_mask kernel by name; the
//!      tag was already shipped in M33, but no source-level NSL program
//!      could reach it before this sprint).
//!
//!   3. The synthesized PTX loads the `%dfs_enter_base` register from
//!      `dfs_enter_ptr`. The PARAMETER `dfs_enter_ptr` is declared
//!      unconditionally on the kernel (`flash_attention.rs:449`); only
//!      the REGISTER load (`flash_attention.rs:584`/`635`) is gated on
//!      `config.tree_mask`. Probing for `%dfs_enter_base` therefore
//!      gives the ground-truth proof that the M33 ancestor-check code
//!      path fired — probing for the parameter name would be a false
//!      positive (caught by the holistic reviewer pre-cycle-3-close).
//!
//! Together (1)+(2)+(3) close the loop: source decorator → semantic
//! checker → codegen extraction → config → kernel-name + PTX.

#![cfg(feature = "test-helpers")]

use nsl_codegen::test_helpers::flash_tree_mask_context_for_source;

const TREE_MASK_FIXTURE: &str = include_str!("fixtures/tree_mask_decorator.nsl");

#[test]
fn tree_mask_decorator_reaches_config_and_kernel_name_and_ptx() {
    let (ctx_set, tree_mask_flag, kernel_name, ptx_has_dfs_enter_base) =
        flash_tree_mask_context_for_source(TREE_MASK_FIXTURE);

    assert!(
        ctx_set,
        "fixture's @flash_attention must build a compile context — \
         if this fails the decorator scanner regressed independent of \
         the tree_mask wiring"
    );

    assert_eq!(
        tree_mask_flag,
        Some(true),
        "Sprint 1 cycle-3 task A: @tree_mask must flip \
         FlashAttentionConfig::tree_mask to true (was hardcoded `false` \
         at the three construction sites in compiler/kernel.rs before \
         this sprint)"
    );

    let kernel_name = kernel_name.expect("context_set=true implies kernel_name=Some");
    assert!(
        kernel_name.contains("_t1_"),
        "Sprint 1 cycle-3: kernel name must contain `_t1_` variant tag \
         when tree_mask=true (matches flash_attention.rs \
         test_tree_mask_variant), got {}",
        kernel_name,
    );

    assert!(
        ptx_has_dfs_enter_base,
        "Sprint 1 cycle-3: synthesized PTX must load the \
         `%dfs_enter_base` register when config.tree_mask=true \
         (paper §4 DFS-enter/DFS-exit ancestor check, shipped in M33 \
         and gated on this flag). The PARAMETER `dfs_enter_ptr` is \
         declared unconditionally; only the register load proves the \
         M33 code path fired."
    );
}
