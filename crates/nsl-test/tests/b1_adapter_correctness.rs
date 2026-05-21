//! Tests for the B.1 -> row-major adapter (Phase 2.6 T6+T7).
//! B.1's retrofit writes row-major saves, so the adapter is identity passthrough.
use half::f16;
use nsl_test::b1_adapter::{B1Saves, reshape_b1_saves_to_row_major};

#[test]
fn reshape_is_identity_passthrough_for_row_major_saves() {
    let b = 1usize; let h = 1usize; let s = 4usize; let d = 4usize;
    let proj_len = b*h*s*d;
    let stat_len = b*h*s;
    let q_proj: Vec<f16> = (0..proj_len).map(|i| f16::from_f32(i as f32)).collect();
    let k_proj: Vec<f16> = (0..proj_len).map(|i| f16::from_f32((i + 100) as f32)).collect();
    let v_proj: Vec<f16> = (0..proj_len).map(|i| f16::from_f32((i + 200) as f32)).collect();
    let row_max: Vec<f32> = (0..stat_len).map(|i| i as f32).collect();
    let row_sum: Vec<f32> = (0..stat_len).map(|i| (i + 10) as f32).collect();
    let o: Vec<f16> = (0..proj_len).map(|i| f16::from_f32((i + 300) as f32)).collect();
    let saves = B1Saves {
        q_proj: q_proj.clone(), k_proj: k_proj.clone(), v_proj: v_proj.clone(),
        row_max: row_max.clone(), row_sum: row_sum.clone(), o: o.clone(),
    };
    let fwd = reshape_b1_saves_to_row_major(saves, b, h, s, d);
    assert_eq!(fwd.q_saved, q_proj);
    assert_eq!(fwd.k_saved, k_proj);
    assert_eq!(fwd.v_saved, v_proj);
    assert_eq!(fwd.row_max, row_max);
    assert_eq!(fwd.row_sum, row_sum);
    assert_eq!(fwd.o, o);
}

#[test]
#[should_panic(expected = "q_proj length")]
fn reshape_rejects_wrong_length() {
    let saves = B1Saves {
        q_proj: vec![f16::ZERO; 3], // wrong: should be 1*1*4*4=16
        k_proj: vec![f16::ZERO; 16], v_proj: vec![f16::ZERO; 16],
        row_max: vec![0.0; 4], row_sum: vec![0.0; 4], o: vec![f16::ZERO; 16],
    };
    let _ = reshape_b1_saves_to_row_major(saves, 1, 1, 4, 4);
}
