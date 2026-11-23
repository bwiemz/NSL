use nsl_runtime::health::collector::HealthCollector;

#[test]
fn nan_loss_increments_counter_and_skips_ema() {
    let mut c = HealthCollector::new();
    c.record_loss(1, 1.0);
    c.record_loss(2, f64::NAN);
    c.record_loss(3, 2.0);
    let snap = c.snapshot();
    assert_eq!(snap.nan_inf_count_window, 1);
    assert!(snap.loss_ema.unwrap() > 1.0 && snap.loss_ema.unwrap() < 2.0);
}

#[test]
fn loss_history_capped_at_window() {
    let mut c = HealthCollector::new();
    for i in 0..150u64 {
        c.record_loss(i, i as f64);
    }
    let snap = c.snapshot();
    assert_eq!(snap.steps_in_window, 100);
    assert_eq!(snap.loss, Some(149.0));
}

#[test]
fn decreasing_loss_produces_negative_slope() {
    let mut c = HealthCollector::new();
    for i in 0..20u64 {
        c.record_loss(i, 100.0 - i as f64);
    }
    let snap = c.snapshot();
    let slope = snap.loss_ema_slope.expect("slope should exist with 20 samples");
    assert!(slope < -0.5, "expected strongly negative slope, got {}", slope);
}

#[test]
fn weight_init_then_update_produces_pct_delta() {
    let mut c = HealthCollector::new();
    c.record_weight_norm("m.l0.w", 100.0, true);
    c.record_weight_norm("m.l0.w", 105.0, false);
    let snap = c.snapshot();
    let pct = snap.per_tensor_weight_pct_delta.get("m.l0.w").copied().unwrap();
    assert!((pct - 5.0).abs() < 1e-6, "expected +5%, got {}", pct);
}

#[test]
fn grad_norm_total_is_root_sum_of_squares() {
    let mut c = HealthCollector::new();
    c.record_grad_norm("m.l0", 0, 3.0);
    c.record_grad_norm("m.l1", 1, 4.0);
    let snap = c.snapshot();
    assert!((snap.grad_norm_total.unwrap() - 5.0).abs() < 1e-6);
}

#[test]
fn should_flush_gates_at_intervals() {
    let mut c = HealthCollector::new();
    c.record_loss(0, 1.0);
    assert!(c.should_flush(), "step 0 should flush (init)");
    c.record_loss(1, 1.0);
    assert!(!c.should_flush(), "step 1 should not flush");
    c.record_loss(99, 1.0);
    assert!(!c.should_flush(), "step 99 should not flush");
    c.record_loss(100, 1.0);
    assert!(c.should_flush(), "step 100 should flush");
}

#[test]
fn snapshot_serializes_to_json() {
    let mut c = HealthCollector::new();
    c.record_loss(5, 4.2);
    c.record_grad_norm("m.l0", 0, 2.5);
    let snap = c.snapshot();
    let s = serde_json::to_string(&snap).unwrap();
    assert!(s.contains("\"step\":5"));
    assert!(s.contains("\"loss\":4.2"));
}

/// The totals are sums of squares over many tensors whose magnitudes differ by orders
/// of magnitude, so the f64 result depends on the summation order. Two processes (or
/// two collectors) recording the same norms must report the same bits: the monitor
/// gate compares runs, and a hash map's per-instance random order made it flake by
/// one ulp.
#[test]
fn totals_do_not_depend_on_recording_order() {
    let norms: Vec<(String, u32, f64)> = (0..64u32)
        .map(|i| {
            let mag = 10f64.powi((i % 9) as i32 - 4);
            (
                format!("layers.{i}.weight"),
                i,
                mag * (1.0 + f64::from(i) * 1.37e-3),
            )
        })
        .collect();
    let totals = |order: &[usize]| {
        let mut c = HealthCollector::new();
        for &k in order {
            let (path, layer, n) = &norms[k];
            c.record_weight_norm(path, *n, true);
            c.record_grad_norm(path, *layer, *n);
        }
        let s = c.snapshot();
        (
            s.weight_norm_total.expect("weights recorded").to_bits(),
            s.grad_norm_total.expect("grads recorded").to_bits(),
        )
    };
    let forward: Vec<usize> = (0..norms.len()).collect();
    let reference = totals(&forward);
    for shift in 1..24 {
        let mut order = forward.clone();
        order.rotate_left(shift * 5 % norms.len());
        if shift % 2 == 1 {
            order.reverse();
        }
        assert_eq!(
            totals(&order),
            reference,
            "recording order changed the totals (shift {shift})"
        );
    }
}
