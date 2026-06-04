//! Sprint 1 T1.1 — CSHA forward output O threaded through FusedCshaBackward.
//!
//! The Tier B.2 hybrid backward's D pre-pass (kernel 1) computes
//! `D = rowsum(dO * O)` and therefore needs the forward attention output `O`
//! as an input. Pre-T1.1, the `FusedCshaBackward` lowerer at
//! `wengert_lower.rs::PrimalOp::FusedCshaBackward` passed `null` for the
//! `out_ptr` slot in `nsl_flash_attention_csha_backward` — see the
//! PARITY-GATE NOTE in `crates/nsl-runtime/src/flash_attention.rs` near
//! line 1907. That null caused `csha_tensor_data_ptr(out_ptr)` to resolve
//! to 0 and the D pre-pass to silently produce `D == 0`, breaking the
//! hybrid path's §8 zero-output guard.
//!
//! T1.1's contract: thread the forward `out_val` through
//! `CshaSavePointers.out` so the backward emission can read it. This is
//! the simplest correct approach because the forward already produces an
//! `out_val` in the same `FunctionBuilder` scope as the save-pointer
//! stash (Approach A in the design — extend the existing per-layer
//! `CshaSavePointers` record).
//!
//! These tests pin two invariants:
//!
//! 1. **Structural:** `CshaSavePointers` exposes an `out` field of type
//!    `cranelift_codegen::ir::Value`. The forward lowering stores the
//!    Cranelift Value produced by `nsl_tensor_zeros_on` (the `out_val`
//!    fed to `nsl_flash_attention_csha_with_saves`) into this slot.
//!
//! 2. **Wiring:** the `wengert_lower.rs::FusedCshaBackward` source no
//!    longer passes `null` for `out_ptr`. Source-level grep is the
//!    load-bearing assertion — relocation scans cannot see the call's
//!    argument *values*, only the symbol identity. The source-level
//!    invariant is the only way to lock the fix in place without a full
//!    GPU integration run (which Phase 3 of the M35.2a closure will
//!    cover end-to-end).
//!
//! Companion runtime change is unnecessary: the runtime already reads
//! `out_ptr` via `csha_tensor_data_ptr(out_ptr)` at
//! `flash_attention.rs:1914`. Passing a real NslTensor handle is
//! sufficient.

use nsl_codegen::csha_apply::CshaSavePointers;
use cranelift_codegen::ir::Value;

/// Pin the struct layout: `CshaSavePointers` MUST carry an `out` field
/// of type `cranelift_codegen::ir::Value`. The forward lowering stores
/// the same Cranelift Value that was passed as `out_val` to
/// `nsl_flash_attention_csha_with_saves` so the backward emission can
/// read it without a separate cross-emit lookup.
///
/// This test fails to compile if the field is missing, which is the
/// failing-test condition for T1.1's struct extension.
#[test]
fn csha_save_pointers_carries_forward_out_value() {
    // Construct a fully-populated record using sentinel u32 ids — the
    // same pattern `tests/csha_gap_c_multiresult_primitive.rs` uses to
    // exercise the cache without a real Cranelift function body. The
    // line that names `out:` is the load-bearing assertion: it pins
    // the struct's API at compile time.
    let saves = CshaSavePointers {
        q_proj: Value::from_u32(400),
        k_proj: Value::from_u32(401),
        v_proj: Value::from_u32(402),
        row_max: Value::from_u32(403),
        row_sum: Value::from_u32(404),
        x_raw:   Value::from_u32(405),
        out:     Value::from_u32(406),
        // Sprint 1 cycle-2: RoPE cos/sin Values stashed alongside `out`
        // so the Tier B.2 hybrid backward's `emit_drope` (proj_backward
        // kernel) reads the same Values the forward FFI was handed.
        cos:     Value::from_u32(407),
        sin:     Value::from_u32(408),
        backward_ptx_data_id: None,
        backward_name_data_id: None,
    };

    // The `out` field MUST be addressable and round-trip the value we
    // stored. Trivial property, but it locks the field's presence and
    // type — accidental rename to `o_ptr` or signature change to
    // `Option<Value>` would fail here.
    assert_eq!(saves.out, Value::from_u32(406));
}

/// Sprint 1 cycle-2 structural pin for the new `cos` / `sin` fields on
/// `CshaSavePointers`. Same shape as the `out` round-trip above: pin the
/// API so an accidental rename, a switch to `Option<Value>`, or removal
/// of either field fails to compile here.
///
/// The forward CSHA call site stashes the same Cranelift Values it hands
/// to `nsl_flash_attention_csha_with_saves` for `rope_cos` / `rope_sin`.
/// The Tier B.2 hybrid backward's `emit_drope` (proj_backward kernel,
/// Sprint 10) MUST read identical Values so forward and backward agree
/// on the rotation — when both are null the in-kernel null-guard fires
/// on both sides and rotation is skipped self-consistently
/// (rope-effectively-off, the current production state). When future
/// work threads non-null cos/sin into the forward call site, backward
/// picks them up automatically with no additional edits.
#[test]
fn csha_save_pointers_carries_rope_cos_and_sin_values() {
    let saves = CshaSavePointers {
        q_proj: Value::from_u32(500),
        k_proj: Value::from_u32(501),
        v_proj: Value::from_u32(502),
        row_max: Value::from_u32(503),
        row_sum: Value::from_u32(504),
        x_raw:   Value::from_u32(505),
        out:     Value::from_u32(506),
        cos:     Value::from_u32(507),
        sin:     Value::from_u32(508),
        backward_ptx_data_id: None,
        backward_name_data_id: None,
    };

    // Distinct sentinels prove the two slots are independent fields, not
    // aliased to a single Value or accidentally collapsed at construction.
    assert_eq!(saves.cos, Value::from_u32(507));
    assert_eq!(saves.sin, Value::from_u32(508));
    assert_ne!(saves.cos, saves.sin);
}

/// Source-level invariant: the `FusedCshaBackward` lowering in
/// `wengert_lower.rs` MUST pass a non-null `out_ptr` slot to
/// `nsl_flash_attention_csha_backward`. The backward FFI signature
/// (see `nsl-runtime/src/flash_attention.rs` ~line 1290) places
/// `out_ptr` as the 4th positional arg. Pre-T1.1, the lowerer emitted
/// `null` here and the runtime's D pre-pass produced `D == 0`.
///
/// We grep the source file for the call-site contents because:
///   - the call is a 50-arg `nsl_flash_attention_csha_backward`
///     invocation; reconstructing it via a Cranelift unit test is
///     heavier than the test's value.
///   - the relocation-scan approach used by `csha_gap_d_emit_fused.rs`
///     sees only the *symbol* of the FFI, never its argument values —
///     so it cannot distinguish `null` from a real `saves.out`.
///   - the M35.2a sprint plan calls for a structural, source-level
///     pin (T1.1 description) ahead of the GPU end-to-end gate.
#[test]
fn wengert_lower_passes_saves_out_for_out_ptr() {
    let src = include_str!("../src/wengert_lower.rs");

    // Locate the `nsl_flash_attention_csha_backward` call body. We
    // restrict the search to ~80 lines after the FFI name to bound the
    // window — the call's arg list is densely packed and lives within
    // ~60 lines of the FFI-name token in `FusedCshaBackward`.
    let needle = "\"nsl_flash_attention_csha_backward\"";
    let idx = src
        .find(needle)
        .expect("wengert_lower.rs must contain the backward FFI call");
    let window_end = (idx + 4000).min(src.len());
    let window = &src[idx..window_end];

    // The arg list lays out as:
    //   q_ptr, k_ptr, v_ptr,
    //   <out_ptr slot>,
    //   <logsumexp_ptr slot>,
    //   ...
    //
    // Pre-T1.1 the slot was literally `null,                 // out_ptr`.
    // Post-T1.1 it must be `saves.out` (the field on `CshaSavePointers`
    // populated at forward emission). We assert both: the new
    // `saves.out` token is present, AND the old `null,` immediately
    // followed by the `// out_ptr` comment is gone.
    assert!(
        window.contains("saves.out"),
        "FusedCshaBackward must source `out_ptr` from the forward \
         save-pointer record (saves.out); call-window did not mention \
         it. T1.1 contract violated.\n\nwindow=\n{}",
        window
    );

    // The historical `null` slot was annotated with a trailing
    // `// out_ptr` comment on the same line. The post-fix call must
    // not have any line in the call-window that places `null` and
    // `out_ptr` together as the slot-4 entry. The marker we pin is
    // the very specific pre-fix spelling — a single regression that
    // re-introduces it will fail this test.
    let pre_fix_marker = "null,                 // out_ptr";
    assert!(
        !window.contains(pre_fix_marker),
        "FusedCshaBackward call site still contains the pre-T1.1 \
         `null` placeholder for out_ptr ('{}'). The D pre-pass kernel \
         reads O from this slot and produces zero D when it is null, \
         silently corrupting backward gradients.",
        pre_fix_marker
    );
}
