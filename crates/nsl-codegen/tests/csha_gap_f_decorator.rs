//! Gap F — source-language surface for reaching the CSHA fused
//! backward dispatcher.
//!
//! Gap E (PR #58) documented two pre-existing gaps that blocked any
//! real NSL source from reaching `nsl_flash_attention_csha_backward`:
//!
//!   F.1 (DOC-GAP A): `compile_flash_attention_kernels` scanned
//!       `ModelMember::LayerDecl` for `@flash_attention` but silently
//!       dropped the decorator when attached to a
//!       `ModelMember::Method`.  The natural idiom
//!       `@flash_attention fn forward(...)` inside a `model` block
//!       therefore had no effect.
//!
//!   F.2 (DOC-GAP B): `@flash_attention` baked in `head_dim=64` at
//!       compile time.  The Tier C fused-backward SMEM validator
//!       rejected every block_q/block_kv tuple at hd=64 (~713 KB >
//!       99 KB cap), and no source-level NSL program could request
//!       hd=32 where the validator accepts.
//!
//! This test pins both fixes:
//!
//!   1. `@flash_attention` on a `ModelMember::Method` now fires
//!      `compile_flash_attention_kernels` → `flash_attention_context`
//!      is populated.
//!   2. `@flash_attention(head_dim=32)` threads hd=32 into the
//!      synthesized `FlashAttentionConfig`; hd=64 is preserved as the
//!      default when the argument is omitted.
//!   3. Invalid `head_dim` values (outside `ALLOWED_HEAD_DIM`) yield a
//!      clear codegen error rather than silently proceeding.
//!   4. With `@train` + `head_dim=32` + a method-level decorator the
//!      fused backward PTX is embedded — previously impossible because
//!      F.1 blocked scanner entry and F.2 blocked SMEM acceptance.

#![cfg(feature = "test-helpers")]

use nsl_codegen::test_helpers::flash_gap_f_context_for_source;

/// F.1 regression: method-level `@flash_attention` (no training) builds
/// a context.  Pre-fix this returned `(false, None, false)`.
const METHOD_INFERENCE_SRC: &str = r#"
model TinyAttn:
    wq: Tensor = ones([32, 32])

    @flash_attention
    fn forward(self, x: Tensor) -> Tensor:
        return x

let m = TinyAttn()
"#;

/// F.2 regression: `@flash_attention(head_dim=32)` on a method threads
/// hd=32 into the config and (because hd=32 fits the SMEM budget)
/// embeds the fused backward PTX when combined with `@train`.
const METHOD_TRAINING_HD32_SRC: &str = r#"
model TinyAttn:
    wq: Tensor = ones([32, 32])

    @flash_attention(head_dim=32)
    fn forward(self, x: Tensor) -> Tensor:
        return x

let m = TinyAttn()

train(model = m, epochs = 1):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let y = m.wq
"#;

/// Default-head_dim reference: `@flash_attention` with no args on a
/// method must still default to hd=64.  Gap F also side-clamps the
/// *training* config's block_kv to 32 so the Tier C backward PTX
/// assembler accepts it — so the fused backward now embeds even at
/// hd=64, unlike pre-Gap-F where the (64, 64, 64) backward config
/// was rejected by the SMEM validator.
const METHOD_TRAINING_HD_DEFAULT_SRC: &str = r#"
model TinyAttn:
    wq: Tensor = ones([32, 32])

    @flash_attention
    fn forward(self, x: Tensor) -> Tensor:
        return x

let m = TinyAttn()

train(model = m, epochs = 1):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let y = m.wq
"#;

#[test]
fn method_decorator_inference_populates_context() {
    let (ctx_set, head_dim, _has_bwd) = flash_gap_f_context_for_source(METHOD_INFERENCE_SRC);
    assert!(
        ctx_set,
        "Gap F.1: `@flash_attention` on a `ModelMember::Method` must \
         populate `flash_attention_context`.  Pre-F.1, the scanner \
         only matched `ModelMember::LayerDecl` and silently dropped \
         method-level decorators."
    );
    assert_eq!(
        head_dim,
        Some(64),
        "Gap F.1: default head_dim must stay 64 when \
         `@flash_attention` has no args — byte-identical behaviour \
         guarantee for existing code."
    );
}

#[test]
fn method_decorator_training_hd32_embeds_backward_ptx() {
    let (ctx_set, head_dim, has_bwd) = flash_gap_f_context_for_source(METHOD_TRAINING_HD32_SRC);
    assert!(
        ctx_set,
        "Gap F.1 + F.2: `@flash_attention(head_dim=32)` on a \
         `ModelMember::Method` must populate the context"
    );
    assert_eq!(
        head_dim,
        Some(32),
        "Gap F.2: `@flash_attention(head_dim=32)` must thread hd=32 \
         through to the synthesized `FlashAttentionConfig`"
    );
    assert!(
        has_bwd,
        "Gap F (combined): with hd=32 + `@train`, the Tier C SMEM \
         validator must accept the backward config and embed the \
         fused-backward PTX.  If this regresses, the backward PTX \
         synthesizer is rejecting an hd=32 config that used to fit."
    );
}

#[test]
fn method_decorator_training_hd_default_embeds_backward_ptx() {
    let (ctx_set, head_dim, has_bwd) =
        flash_gap_f_context_for_source(METHOD_TRAINING_HD_DEFAULT_SRC);
    assert!(
        ctx_set,
        "Gap F.1: default-head_dim method decorator must still build \
         a context."
    );
    assert_eq!(
        head_dim,
        Some(64),
        "Gap F.2: default head_dim must be 64 when decorator has no \
         `head_dim=` argument — inference behaviour is byte-identical."
    );
    // Secondary side effect of Gap F: because the training config
    // clamps block_kv=32 (the only value the Tier C backward emitter
    // supports), the backward SMEM validator now accepts (64, 32, 64)
    // too.  Pre-Gap-F this was None because block_kv inherited 64
    // from the forward and the dK/dV tiles blew the budget.
    assert!(
        has_bwd,
        "Gap F post-fix: with block_kv clamped to 32 in the training \
         config, the Tier C backward validator should accept hd=64 \
         as well.  If this flips, the backward SMEM accounting \
         regressed at (64, 32, 64)."
    );
}

/// Parser/kwarg pattern smoke: the existing `@flash_attention(causal=true)`
/// kwarg still parses and applies alongside the new `head_dim=` arg.
const BOTH_ARGS_SRC: &str = r#"
model TinyAttn:
    wq: Tensor = ones([32, 32])

    @flash_attention(causal=false, head_dim=32)
    fn forward(self, x: Tensor) -> Tensor:
        return x

let m = TinyAttn()
"#;

/// Invalid `head_dim` must produce a clear codegen error.  The v2
/// emitter's ALLOWED_HEAD_DIM is `{32, 64, 128, 256}` — anything else
/// is either unsupported by the tile geometry or would silently land
/// on a PTX path that does the wrong thing.
#[test]
fn method_decorator_rejects_invalid_head_dim() {
    const INVALID_SRC: &str = r#"
model TinyAttn:
    @flash_attention(head_dim=48)
    fn forward(self, x: Tensor) -> Tensor:
        return x

let m = TinyAttn()
"#;
    // `flash_gap_f_context_for_source` calls `.expect` on the
    // inner `compile_flash_attention_kernels` result, so an invalid
    // decorator argument must panic with the validation message.
    let result = std::panic::catch_unwind(|| {
        let _ = flash_gap_f_context_for_source(INVALID_SRC);
    });
    assert!(
        result.is_err(),
        "Gap F.2: `@flash_attention(head_dim=48)` must fail codegen — \
         48 is not in ALLOWED_HEAD_DIM"
    );
}

#[test]
fn method_decorator_multiple_kwargs_combine() {
    let (ctx_set, head_dim, _has_bwd) = flash_gap_f_context_for_source(BOTH_ARGS_SRC);
    assert!(
        ctx_set,
        "mixing `causal=` + `head_dim=` kwargs on `@flash_attention` \
         must still populate the context"
    );
    assert_eq!(
        head_dim,
        Some(32),
        "head_dim= must be honoured alongside causal= — kwarg parsing \
         is not order-sensitive"
    );
}
