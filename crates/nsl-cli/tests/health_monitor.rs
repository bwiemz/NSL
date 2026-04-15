use nsl_cli::health_monitor::{HealthRenderer, group_by_layer_prefix};
use nsl_runtime::health::collector::HealthSnapshot;
use std::collections::HashMap;

fn snap_with_grad_norms(per_layer: &[(u32, f64)], nan: u64, slope: Option<f64>) -> HealthSnapshot {
    let mut s = HealthSnapshot::default();
    s.step = 100;
    s.loss = Some(4.23);
    s.grad_norm_total = Some(12.4);
    s.per_layer_grad_norm = per_layer.iter().copied().collect();
    s.nan_inf_count_window = nan;
    s.steps_in_window = 100;
    s.loss_ema_slope = slope;
    s
}

#[test]
fn renders_header_and_per_layer_block() {
    let snap = snap_with_grad_norms(&[(0, 12.1), (1, 13.4), (2, 18.7)], 0, Some(-0.032));
    let r = HealthRenderer::new();
    let body = r.format_block(&snap);
    assert!(body.contains("=== Training Health Monitor (live) ==="));
    assert!(body.contains("Step 100"));
    assert!(body.contains("Loss: 4.23"));
    assert!(body.contains("Grad norm: 12.4"));
    assert!(body.contains("Per-layer gradient norms:"));
    assert!(body.contains("L0:"));
    assert!(body.contains("L1:"));
    assert!(body.contains("L2:"));
    assert!(body.contains("18.7"));
    assert!(body.contains("⚠ elevated"), "L2 over threshold should be flagged: {}", body);
}

#[test]
fn nan_count_warning_renders() {
    let snap = snap_with_grad_norms(&[(0, 1.0)], 3, Some(-0.01));
    let r = HealthRenderer::new();
    let body = r.format_block(&snap);
    assert!(body.contains("⚠ 3 NaN/Inf"), "{}", body);
}

#[test]
fn loss_trend_classification() {
    let snap_decreasing = snap_with_grad_norms(&[], 0, Some(-0.032));
    let snap_increasing = snap_with_grad_norms(&[], 0, Some(0.05));
    let snap_flat = snap_with_grad_norms(&[], 0, Some(0.0));
    let snap_none = snap_with_grad_norms(&[], 0, None);
    let r = HealthRenderer::new();
    assert!(r.format_block(&snap_decreasing).contains("decreasing"));
    assert!(r.format_block(&snap_increasing).contains("increasing"));
    assert!(r.format_block(&snap_flat).contains("flat"));
    assert!(r.format_block(&snap_none).contains("insufficient"));
}

#[test]
fn group_by_layer_prefix_simple_stack() {
    let mut m = HashMap::new();
    m.insert("m.transformer.h.0.attn.wq".to_string(), 0.3);
    m.insert("m.transformer.h.0.attn.wk".to_string(), 0.2);
    m.insert("m.transformer.h.1.attn.wq".to_string(), 0.4);
    let groups = group_by_layer_prefix(&m);
    let labels: Vec<_> = groups.iter().map(|(l, _)| l.clone()).collect();
    assert!(labels.contains(&"L0".to_string()));
    assert!(labels.contains(&"L1".to_string()));
    let l0_entries = &groups.iter().find(|(l, _)| l == "L0").unwrap().1;
    assert_eq!(l0_entries.len(), 2);
}

#[test]
fn group_by_layer_prefix_encoder_decoder_split() {
    let mut m = HashMap::new();
    m.insert("m.encoder.blocks.0.attn.wq".to_string(), 0.1);
    m.insert("m.decoder.blocks.0.attn.wq".to_string(), 0.2);
    let groups = group_by_layer_prefix(&m);
    let labels: Vec<_> = groups.iter().map(|(l, _)| l.clone()).collect();
    assert!(labels.contains(&"Enc.L0".to_string()));
    assert!(labels.contains(&"Dec.L0".to_string()));
}

#[test]
fn group_by_layer_prefix_handles_no_numeric_segment() {
    let mut m = HashMap::new();
    m.insert("standalone_param".to_string(), 1.0);
    let groups = group_by_layer_prefix(&m);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].0, "misc");
}

#[test]
fn empty_grad_norms_skips_block() {
    let snap = HealthSnapshot::default();
    let r = HealthRenderer::new();
    let body = r.format_block(&snap);
    assert!(!body.contains("Per-layer gradient norms:"));
    assert!(body.contains("=== Training Health Monitor"));
    assert!(body.contains("NaN watch:"));
}
