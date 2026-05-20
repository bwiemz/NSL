use nsl_test::cpu_naive_forward::cpu_naive_forward;
use half::f16;

#[test]
fn cpu_naive_forward_small_identity_case() {
    // batch=1, heads=1, seq=2, hd=2, non-causal
    // q = k = identity [[1,0],[0,1]], v = [[1,2],[3,4]]
    // S = (1/sqrt(2)) * q @ k^T = (1/sqrt(2)) * [[1,0],[0,1]]
    //   = [[0.707, 0], [0, 0.707]]
    // row_max = [0.707, 0.707]
    // row_sum = [exp(0) + exp(-0.707), exp(-0.707) + exp(0)] = [1 + 0.493, ...]
    //         ≈ [1.493, 1.493]
    // P[0] = [exp(0)/1.493, exp(-0.707)/1.493] ≈ [0.670, 0.330]
    // O[0] = 0.670*[1,2] + 0.330*[3,4] ≈ [1.659, 2.659]
    // O[1] = 0.330*[1,2] + 0.670*[3,4] ≈ [2.341, 3.341]

    let q = vec![
        f16::from_f32(1.0), f16::from_f32(0.0),
        f16::from_f32(0.0), f16::from_f32(1.0),
    ];
    let k = q.clone();
    let v = vec![
        f16::from_f32(1.0), f16::from_f32(2.0),
        f16::from_f32(3.0), f16::from_f32(4.0),
    ];

    let out = cpu_naive_forward(&q, &k, &v, 1, 1, 2, 2, false);

    assert!((out.row_max[0] - 0.7071068).abs() < 1e-4,
        "row_max[0] = {} (expected ~0.7071068)", out.row_max[0]);
    assert!((out.row_max[1] - 0.7071068).abs() < 1e-4);
    assert!((out.row_sum[0] - 1.4931).abs() < 1e-3,
        "row_sum[0] = {} (expected ~1.4931)", out.row_sum[0]);
    assert!((out.o[0].to_f32() - 1.659).abs() < 5e-3);
    assert!((out.o[1].to_f32() - 2.659).abs() < 5e-3);
    assert!((out.o[2].to_f32() - 2.341).abs() < 5e-3);
    assert!((out.o[3].to_f32() - 3.341).abs() < 5e-3);
}

#[test]
fn cpu_naive_forward_row_sum_is_sum_not_reciprocal() {
    // Pin spec §2.1: row_sum is the SUM, not the reciprocal.
    let q = vec![f16::from_f32(1.0); 4];
    let k = vec![f16::from_f32(1.0); 4];
    let v = vec![f16::from_f32(1.0); 4];
    let out = cpu_naive_forward(&q, &k, &v, 1, 1, 2, 2, false);
    assert!(out.row_sum[0] > 1.0,
        "row_sum should be Σ exp(S - row_max) >= 1 when any S=row_max; got {}", out.row_sum[0]);
}

#[test]
fn cpu_naive_forward_causal_masks_upper_triangle() {
    let q = vec![f16::from_f32(1.0); 8]; // 1*1*4*2
    let k = vec![f16::from_f32(1.0); 8];
    let v = vec![f16::from_f32(1.0); 8];
    let out_causal = cpu_naive_forward(&q, &k, &v, 1, 1, 4, 2, true);
    let out_full = cpu_naive_forward(&q, &k, &v, 1, 1, 4, 2, false);
    // Causal: q=0 only sees k=0, so row_sum has 1 term; full sees 4 terms.
    assert_ne!(out_causal.row_sum[0], out_full.row_sum[0],
        "causal masking should affect row_sum for q=0 (1 key vs 4 keys)");
    // For q=3, causal sees k=0..3 (4 terms) = same as full case
    assert!((out_causal.row_sum[3] - out_full.row_sum[3]).abs() < 1e-4,
        "for q=last_row, causal == full");
}

#[test]
fn cpu_naive_forward_output_shape_matches_input_shape() {
    let q = vec![f16::from_f32(0.1); 2 * 2 * 4 * 8]; // B=2, H=2, S=4, D=8
    let k = q.clone();
    let v = q.clone();
    let out = cpu_naive_forward(&q, &k, &v, 2, 2, 4, 8, false);
    assert_eq!(out.row_max.len(), 2 * 2 * 4);
    assert_eq!(out.row_sum.len(), 2 * 2 * 4);
    assert_eq!(out.o.len(), 2 * 2 * 4 * 8);
}

#[test]
fn cpu_naive_forward_no_nan_in_outputs_for_finite_inputs() {
    let q = vec![f16::from_f32(0.5); 16]; // 1*1*4*4
    let k = q.clone();
    let v = q.clone();
    let out = cpu_naive_forward(&q, &k, &v, 1, 1, 4, 4, false);
    for &r in &out.row_max { assert!(r.is_finite()); }
    for &r in &out.row_sum { assert!(r.is_finite()); }
    for &o in &out.o { assert!(o.to_f32().is_finite()); }
}
