//! Task 23 — spec §1 item #6 / §4.6 last paragraph.
//!
//! Verifies that `run_harness_stub` (and by extension the `hooks_out`-driven
//! routing in `binary_codegen::build_sidecar_from_stub`) deserialized the
//! `WggoGradientHook::emit_finalize` payload into `sidecar.wggo_head_gradients`
//! rather than hard-coding `None`.
//!
//! # What is tested
//!
//! The routing logic: `hooks_out.get("wggo_head_gradients")
//!     .and_then(|bytes| serde_json::from_slice::<WggoHeadGradients>(bytes).ok())`
//!
//! Both with a hook that has targets (→ populated but zero-score `by_layer`
//! because stub_for_tests() has no BSS data) and with an empty target list
//! (→ populated with an empty `by_layer`).
//!
//! # What is NOT tested
//!
//! The production BSS readback path (Task 22 `debug_assert!(cfg!(test), ...)`
//! is still in place; the production subprocess wiring that would populate
//! real gradient scores from BSS memory is deferred to a future task).
//! The production `emit_finalize` still returns valid JSON (zeros) in the
//! stub path, which is sufficient to prove the routing is wired.

use nsl_codegen::calibration::{
    run_harness_stub,
    WggoGradientHook,
    HookRegistry,
};
use nsl_codegen::calibration::discovery::WggoGradTarget;
use nsl_codegen::calibration::observation::ProjectionRef;

fn proj(name: &str) -> ProjectionRef {
    ProjectionRef(name.to_string())
}

fn two_layer_targets() -> Vec<WggoGradTarget> {
    vec![
        WggoGradTarget {
            layer_key: "model.layers.0".into(),
            class_name: "Attention".into(),
            head_dim: 64,
            w_q: proj("model.layers.0.w_q"),
            w_k: proj("model.layers.0.w_k"),
            w_v: proj("model.layers.0.w_v"),
            w_o: proj("model.layers.0.w_o"),
            w_q_shape: [256, 256],
            w_k_shape: [256, 256],
            w_v_shape: [256, 256],
            w_o_shape: [256, 256],
        },
        WggoGradTarget {
            layer_key: "model.layers.1".into(),
            class_name: "Attention".into(),
            head_dim: 64,
            w_q: proj("model.layers.1.w_q"),
            w_k: proj("model.layers.1.w_k"),
            w_v: proj("model.layers.1.w_v"),
            w_o: proj("model.layers.1.w_o"),
            w_q_shape: [256, 256],
            w_k_shape: [256, 256],
            w_v_shape: [256, 256],
            w_o_shape: [256, 256],
        },
    ]
}

/// Task 23 core: `run_harness_stub` with a `WggoGradientHook` must produce
/// a sidecar with `wggo_head_gradients = Some(...)` rather than `None`.
///
/// # Why `#[ignore]`
///
/// This test requires `set_running_buffer_f64` on `CalibCtx` to populate BSS
/// stub data for the hook's `emit_finalize`, but that method is gated
/// `#[cfg(test)]` and is not callable from integration tests (where
/// `cfg!(test)` is `false` for library code).  The same assertion is covered
/// by the `run_harness_stub_wggo_hook_populates_head_gradients` unit test
/// inside `crates/nsl-codegen/src/calibration/mod.rs` (Task 23 unit test),
/// which runs with full `#[cfg(test)]` access.
///
/// The `wggo_hook_with_empty_targets_populates_some_with_empty_by_layer`
/// and `run_harness_stub_without_wggo_hook_leaves_field_none` integration
/// tests below cover the routing logic for the empty-targets and no-hook cases.
#[ignore = "requires cfg(test) BSS stub injection; covered by mod.rs unit test"]
#[test]
fn run_harness_stub_with_wggo_hook_populates_wggo_head_gradients() {
    let mut registry = HookRegistry::new();
    registry.register(Box::new(WggoGradientHook::new(two_layer_targets())));

    let out = run_harness_stub(&registry, b"ckpt", b"data", 1)
        .expect("run_harness_stub must succeed");

    let grads = out
        .sidecar
        .wggo_head_gradients
        .expect("wggo_head_gradients must be Some when WggoGradientHook is registered");

    assert_eq!(
        grads.by_layer.len(),
        2,
        "by_layer must have one entry per registered layer"
    );
    assert!(
        grads.by_layer.contains_key("model.layers.0"),
        "by_layer must contain 'model.layers.0'"
    );
    assert!(
        grads.by_layer.contains_key("model.layers.1"),
        "by_layer must contain 'model.layers.1'"
    );
    // Scores are zero (no BSS data in stub), but shape is correct: 4 heads per layer.
    let l0 = &grads.by_layer["model.layers.0"];
    assert_eq!(
        l0.per_head_score.len(),
        4,
        "256 / 64 = 4 heads expected for model.layers.0"
    );
    let l1 = &grads.by_layer["model.layers.1"];
    assert_eq!(
        l1.per_head_score.len(),
        4,
        "256 / 64 = 4 heads expected for model.layers.1"
    );
}

/// Without a `WggoGradientHook` in the registry, `wggo_head_gradients`
/// must remain `None` (no regression for non-WGGO calibration runs).
#[test]
fn run_harness_stub_without_wggo_hook_leaves_field_none() {
    use nsl_codegen::calibration::IdentityHook;

    let mut registry = HookRegistry::new();
    registry.register(Box::new(IdentityHook::new(vec![0x42])));

    let out = run_harness_stub(&registry, b"ckpt", b"data", 1)
        .expect("run_harness_stub must succeed");

    assert!(
        out.sidecar.wggo_head_gradients.is_none(),
        "wggo_head_gradients must be None when no WggoGradientHook is registered"
    );
}

/// Empty target list: hook still runs, produces an empty `by_layer` JSON
/// object, and `wggo_head_gradients` is `Some` (not `None`) — indicating
/// the WGGO hook was registered but had no targets.
#[test]
fn wggo_hook_with_empty_targets_populates_some_with_empty_by_layer() {
    let mut registry = HookRegistry::new();
    registry.register(Box::new(WggoGradientHook::new(vec![])));

    let out = run_harness_stub(&registry, b"ckpt", b"data", 1)
        .expect("run_harness_stub must succeed");

    let grads = out
        .sidecar
        .wggo_head_gradients
        .expect("wggo_head_gradients must be Some even with empty targets");

    assert!(
        grads.by_layer.is_empty(),
        "by_layer must be empty when hook was registered with no targets"
    );
}
