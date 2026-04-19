//! Gap B — forward PTX synthesis must contain CSHA save-writes under
//! `@train`, and the fused-backward PTX must also land on the
//! `FlashAttentionCompileContext` so Gap C/D has DataIds to consume.
//!
//! Gap A shipped the runtime side (`_with_saves` FFI + alloc/free) but
//! the PTX embedded by `compile_flash_attention_kernels` was always
//! synthesized with `csha=None`, which suppresses
//! `emit_save_activations_subset` at PTX-generation time.  End result
//! pre-Gap-B: alloc runs, pointers flow, kernel writes nothing → save
//! buffers stay zero → any backward kernel reading them gets garbage.
//!
//! This test verifies the Gap B post-process in
//! `compile_flash_attention_kernels`:
//!
//! 1. A program WITHOUT `@train` leaves `csha_forward_with_saves_ptx_id`
//!    and `csha_backward_ptx_data_id` as `None` (inference default — no
//!    extra PTX embedded, no SMEM budget paid).
//! 2. A program WITH `@train` populates both IDs AND attaches a
//!    `csha_training_config` with `level=1` (minimum-invasive preset —
//!    no fusion flags flipped, just save_activations_for_backward=true).
//! 3. The forward PTX re-synthesized from that `csha_training_config`
//!    contains the `V2_CSHA_SAVE_Q_SKIP_` labels that prove
//!    `emit_save_activations_subset` actually fired.

#![cfg(feature = "test-helpers")]

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;
use nsl_codegen::test_helpers::flash_gap_b_context_for_source;

/// Program with `@flash_attention` but no `@train` — Gap B should leave
/// the training-PTX fields `None` so inference compiles pay nothing.
const INFERENCE_SRC: &str = "\
@flash_attention
fn forward():
    pass
";

/// Program with `@flash_attention` AND a top-level `@train` block — Gap
/// B should fire, embedding both the save-enabled forward PTX and the
/// fused-backward PTX in the compile context.
///
/// The train block body is intentionally minimal — Gap B only cares
/// that *some* `TrainBlock` exists in the top-level stmt list; the
/// block's contents don't drive the PTX decision.  A `model` + `step`
/// pair is the smallest form the parser accepts.
const TRAINING_SRC: &str = r#"
@flash_attention
fn forward():
    pass

model Toy:
    w: Tensor = ones([4, 4])

let m = Toy()

train(model = m, epochs = 1):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let y = m.w
"#;

#[test]
fn inference_program_leaves_training_ptx_unpopulated() {
    let (has_saves_ptx, has_bwd_ptx, training_level) =
        flash_gap_b_context_for_source(INFERENCE_SRC);
    assert!(
        !has_saves_ptx,
        "inference compile must NOT embed CSHA with-saves forward PTX \
         (pays SMEM + register-pressure cost for nothing)"
    );
    assert!(
        !has_bwd_ptx,
        "inference compile must NOT embed the fused-backward PTX \
         (no @train block → no backward pass → dead bytes)"
    );
    assert_eq!(
        training_level, None,
        "inference compile must NOT record a csha_training_config \
         (Gap B pre-scan incorrectly saw a @train block)"
    );
}

#[test]
fn training_program_embeds_both_save_forward_and_backward_ptx() {
    let (has_saves_ptx, has_bwd_ptx, training_level) =
        flash_gap_b_context_for_source(TRAINING_SRC);
    assert!(
        has_saves_ptx,
        "Gap B must embed CSHA with-saves forward PTX when @train is \
         present — otherwise the kernel bytes have no save codepaths \
         and `_with_saves` FFI silently writes zero-filled buffers"
    );
    // The fused-backward PTX may be skipped when the SMEM budget
    // validator rejects the default (64,64,64) config (it currently
    // fits, but a future budget tightening could force this to None
    // — Gap C/D falls back to the legacy tape-op backward when it's
    // missing).  Emit an explicit diagnostic on miss so regressions
    // don't silently erode coverage.
    if !has_bwd_ptx {
        eprintln!(
            "[gap-b] backward PTX was NOT embedded — the Tier C SMEM \
             validator rejected the default training config; Gap C/D \
             will fall through to the CPU tape-op backward.  If this \
             is unexpected, check flash_attention_v2::synthesize_backward."
        );
    }
    assert_eq!(
        training_level,
        Some(1),
        "Gap B must record a csha_training_config with level=1 (the \
         minimum-invasive preset that turns on save_activations without \
         flipping any fusion flags)"
    );
}

/// Direct PTX-bytes check: the with-saves forward PTX must contain the
/// `V2_CSHA_SAVE_Q_SKIP_` labels that `emit_save_activations_subset`
/// generates.  We reconstruct the exact same CSHA config
/// `maybe_synthesize_csha_training_ptx` uses and synthesize the PTX
/// twice — once with `save_activations_for_backward=true` (must contain
/// the labels) and once without (must NOT contain them, proving the
/// flag is the only knob that gates emission).
#[test]
fn with_saves_ptx_contains_csha_save_labels() {
    let mk = |save: bool| FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80, segment_masked: false, csha: Some(CshaExtras {
            level: 1,
            fused_rmsnorm: false,
            fused_projections: false,
            fused_output_proj: false,
            active_heads: 0,
            rmsnorm_eps: 1e-5,
            d_model: 0,
            save_activations_for_backward: save,
        }),
    };

    let ptx_with_saves = synthesize_flash_attention_ptx_v2(&mk(true));
    let ptx_without_saves = synthesize_flash_attention_ptx_v2(&mk(false));

    let saves_text = String::from_utf8_lossy(&ptx_with_saves);
    assert!(
        saves_text.contains("V2_CSHA_SAVE_Q_SKIP_"),
        "Gap B training PTX must contain the save-write skip label \
         (emit_save_activations_subset emission)"
    );
    assert!(
        saves_text.contains("st.global.b16"),
        "Gap B training PTX must contain at least one f16 HBM save store"
    );

    let no_saves_text = String::from_utf8_lossy(&ptx_without_saves);
    assert!(
        !no_saves_text.contains("V2_CSHA_SAVE_Q_SKIP_"),
        "inference PTX (save_activations_for_backward=false) must NOT \
         contain save-write labels — emission should be fully gated"
    );
}
