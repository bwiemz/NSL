//! CPKD fused KL-CE — GPU numerical validation against an f64 reference.
//!
//! Drives the SAME FFIs the compiled distill step dispatches
//! (`nsl_fused_kl_ce_forward` / `nsl_fused_kl_ce_backward`) with
//! deterministic seeded buffers and compares per-row losses, saved LSEs,
//! and all three student gradients against a reference written out in f64
//! in this file, from the definition of the loss:
//!
//! ```text
//! loss_row = alpha * CE(s, target) + (1 - alpha) * T^2 * KL(softmax(t/T) || softmax(s/T))
//! d loss / d s = [alpha * (softmax(s) - onehot) + (1 - alpha) * T * (softmax(s/T) - softmax(t/T))] / num_valid
//! ```
//!
//! ## Fixture
//! rows = 2*16 = 32, V=512, HS=64, HT=96 (asymmetric on purpose),
//! vocab_tile=128, alpha=0.4, T=2.5, ignore_index -100 at every 8th row.
//! Hidden states and head weights are U(-1, 1), so the logits have a
//! standard deviation near 2 and neither distribution is close to uniform.
//!
//! ## Tolerances
//! * forward per-row loss + LSEs: |gpu - ref| / max(|ref|, 1) < 2e-5
//! * backward dx_s/dW_s/dbias_s: max abs err < 1e-4 * max|ref|, per tensor
//! * skip identity: loss/lse rows and dx_s rows exactly 0 for -100 rows
//!
//! The gradient tolerance is relative to the tensor's own scale. It used to
//! be an absolute 5e-3 over inputs drawn from U(-0.3, 0.3): the reference
//! maxima of dx_s and dW_s were 2.3e-3 and 4.2e-3, so an all-zero gradient
//! passed, and swapping the KL direction moved the loss by less than the
//! forward tolerance. `the_reference_tells_the_kl_terms_apart` holds this
//! fixture to being able to see both.
//!
//! ## Running
//! ```bash
//! cargo test -p nsl-codegen --features cuda --test cpkd_fused_kl_ce_numerical \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_codegen::cpkd_fused_loss::{
    FusedKlCeConfig, synthesize_fused_kl_ce_backward_ptx, synthesize_fused_kl_ce_ptx,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_fused_kl_ce_backward, nsl_fused_kl_ce_forward, nsl_test_cuda_alloc,
    nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
};

const B: usize = 2;
const S: usize = 16;
const ROWS: usize = B * S;
const V: usize = 512;
const HS: usize = 64;
const HT: usize = 96;
const VOCAB_TILE: u32 = 128;
const IGNORE_INDEX: i64 = -100;
const ALPHA: f64 = 0.4;
const TEMP: f64 = 2.5;

fn fill_seeded(dst: &mut [f32], seed: u64) {
    let mut s = seed;
    for x in dst.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        *x = ((u as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }
}

/// The KL term of the reference. The loss is `Forward`; the other two exist
/// so the fixture can be shown to tell a wrong kernel from a right one.
#[derive(Clone, Copy, PartialEq)]
enum KlTerm {
    /// `KL(teacher || student)`.
    Forward,
    /// `KL(student || teacher)`. Only the loss is computed.
    Reverse,
    /// No KL term at all: the loss and gradients of `alpha * CE` alone.
    Absent,
}

/// The f64 reference: per-row loss, the three saved log-sum-exps, and the
/// gradients of the mean-over-valid-rows loss with respect to `x_s`, `W_s`
/// and `bias_s`.
struct Reference {
    loss: Vec<f64>,
    lse_s1: Vec<f64>,
    lse_st: Vec<f64>,
    lse_tt: Vec<f64>,
    dxs: Vec<f64>,
    dws: Vec<f64>,
    dbs: Vec<f64>,
}

struct Inputs<'a> {
    xs: &'a [f32],
    ws: &'a [f32],
    bs: &'a [f32],
    xt: &'a [f32],
    wt: &'a [f32],
    bt: &'a [f32],
    targets: &'a [i64],
}

fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    m + v.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

fn logits(x: &[f32], w: &[f32], b: &[f32], row: usize, hidden: usize) -> Vec<f64> {
    (0..V)
        .map(|v| {
            (0..hidden).fold(b[v] as f64, |acc, h| acc + x[row * hidden + h] as f64 * w[v * hidden + h] as f64)
        })
        .collect()
}

fn reference(input: &Inputs, kl_term: KlTerm) -> Reference {
    let num_valid = input.targets.iter().filter(|&&t| t != IGNORE_INDEX).count() as f64;
    let mut r = Reference {
        loss: vec![0.0; ROWS],
        lse_s1: vec![0.0; ROWS],
        lse_st: vec![0.0; ROWS],
        lse_tt: vec![0.0; ROWS],
        dxs: vec![0.0; ROWS * HS],
        dws: vec![0.0; V * HS],
        dbs: vec![0.0; V],
    };
    for row in 0..ROWS {
        let target = input.targets[row];
        if target == IGNORE_INDEX {
            continue;
        }
        let s = logits(input.xs, input.ws, input.bs, row, HS);
        let t = logits(input.xt, input.wt, input.bt, row, HT);
        let s_scaled: Vec<f64> = s.iter().map(|v| v / TEMP).collect();
        let t_scaled: Vec<f64> = t.iter().map(|v| v / TEMP).collect();
        let (lse_s1, lse_st, lse_tt) = (log_sum_exp(&s), log_sum_exp(&s_scaled), log_sum_exp(&t_scaled));
        let ce = lse_s1 - s[target as usize];
        let kl: f64 = (0..V)
            .map(|v| {
                let (log_ps, log_pt) = (s_scaled[v] - lse_st, t_scaled[v] - lse_tt);
                match kl_term {
                    KlTerm::Forward => log_pt.exp() * (log_pt - log_ps),
                    KlTerm::Reverse => log_ps.exp() * (log_ps - log_pt),
                    KlTerm::Absent => 0.0,
                }
            })
            .sum();
        r.loss[row] = ALPHA * ce + (1.0 - ALPHA) * TEMP * TEMP * kl;
        (r.lse_s1[row], r.lse_st[row], r.lse_tt[row]) = (lse_s1, lse_st, lse_tt);
        if kl_term == KlTerm::Reverse {
            continue; // only the loss is compared for the reversed term
        }
        for v in 0..V {
            let onehot = if v == target as usize { 1.0 } else { 0.0 };
            let ce_grad = (s[v] - lse_s1).exp() - onehot;
            let kl_grad = match kl_term {
                KlTerm::Absent => 0.0,
                _ => (s_scaled[v] - lse_st).exp() - (t_scaled[v] - lse_tt).exp(),
            };
            let g = (ALPHA * ce_grad + (1.0 - ALPHA) * TEMP * kl_grad) / num_valid;
            r.dbs[v] += g;
            for h in 0..HS {
                r.dxs[row * HS + h] += g * input.ws[v * HS + h] as f64;
                r.dws[v * HS + h] += g * input.xs[row * HS + h] as f64;
            }
        }
    }
    r
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0f64, |m, x| m.max(x.abs()))
}

struct Fixture {
    xs: Vec<f32>,
    ws: Vec<f32>,
    bs: Vec<f32>,
    xt: Vec<f32>,
    wt: Vec<f32>,
    bt: Vec<f32>,
    targets: Vec<i64>,
}

impl Fixture {
    fn new() -> Self {
        // Targets: -100 at every 8th row.
        let targets = (0..ROWS)
            .map(|i| if i % 8 == 0 { IGNORE_INDEX } else { ((i * 37 + 13) % V) as i64 })
            .collect();
        let seeded = |len: usize, seed: u64| {
            let mut v = vec![0f32; len];
            fill_seeded(&mut v, seed);
            v
        };
        Fixture {
            xs: seeded(ROWS * HS, 42),
            ws: seeded(V * HS, 137),
            bs: seeded(V, 7),
            xt: seeded(ROWS * HT, 1042),
            wt: seeded(V * HT, 1137),
            bt: seeded(V, 1007),
            targets,
        }
    }

    fn inputs(&self) -> Inputs<'_> {
        Inputs {
            xs: &self.xs,
            ws: &self.ws,
            bs: &self.bs,
            xt: &self.xt,
            wt: &self.wt,
            bt: &self.bt,
            targets: &self.targets,
        }
    }
}

/// Relative tolerance of the per-row loss and LSE comparison.
const FORWARD_TOL: f64 = 2e-5;
/// Tolerance of each gradient tensor, as a fraction of its largest entry.
const GRADIENT_TOL: f64 = 1e-4;

/// The fixture can see what the device test claims to check. It launches
/// nothing, so it runs wherever this file is built.
///
/// * Reversing the KL direction moves the per-row loss by hundreds of times
///   the forward tolerance.
/// * Dropping the KL term moves every gradient tensor by hundreds of times
///   its tolerance, so a kernel without the KL term does not pass. (An
///   all-zero gradient cannot pass either: each tolerance is a fraction of
///   the reference tensor's largest entry.)
#[test]
fn the_reference_tells_the_kl_terms_apart() {
    let fixture = Fixture::new();
    let forward = reference(&fixture.inputs(), KlTerm::Forward);
    let reverse = reference(&fixture.inputs(), KlTerm::Reverse);
    let absent = reference(&fixture.inputs(), KlTerm::Absent);

    let worst_row = (0..ROWS)
        .map(|r| (forward.loss[r] - reverse.loss[r]).abs() / forward.loss[r].abs().max(1.0))
        .fold(0f64, f64::max);
    assert!(
        worst_row > 100.0 * FORWARD_TOL,
        "reversing the KL direction moves the loss by only {worst_row:e} (tolerance {FORWARD_TOL:e})"
    );

    for (name, with, without) in [
        ("dx_s", &forward.dxs, &absent.dxs),
        ("dW_s", &forward.dws, &absent.dws),
        ("dbias_s", &forward.dbs, &absent.dbs),
    ] {
        let tolerance = GRADIENT_TOL * max_abs(with);
        let kl_share = with.iter().zip(without.iter()).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
        assert!(
            kl_share > 100.0 * tolerance,
            "{name}: the KL term moves the gradient by {kl_share:e}, tolerance {tolerance:e}"
        );
    }
}

#[allow(clippy::too_many_lines)]
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_kl_ce_gpu_forward_and_backward_match_reference() {
    if !cuda_available() {
        return;
    }

    let cfg = FusedKlCeConfig {
        vocab_size: V as u32,
        student_hidden: HS as u32,
        teacher_hidden: HT as u32,
        batch_size: B as u32,
        seq_len: S as u32,
        vocab_tile: VOCAB_TILE,
        gpu_sm: 80,
        ignore_index: IGNORE_INDEX,
    };
    cfg.validate().expect("config must be valid");

    let fixture = Fixture::new();
    let Fixture { xs, ws, bs, xt, wt, bt, targets } = &fixture;
    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count() as i64;

    // f64 reference, written out in this file.
    let want = reference(&fixture.inputs(), KlTerm::Forward);
    let (ref_loss, ref_lse_s1, ref_lse_st, ref_lse_tt) = (&want.loss, &want.lse_s1, &want.lse_st, &want.lse_tt);
    let (ref_dxs, ref_dws, ref_dbs) = (&want.dxs, &want.dws, &want.dbs);

    // PTX.
    let fwd_ptx = synthesize_fused_kl_ce_ptx(&cfg);
    let bwd_ptx = synthesize_fused_kl_ce_backward_ptx(&cfg);
    let fwd_name = format!("{}\0", cfg.kernel_name());
    let bwd_name = format!("{}\0", cfg.bwd_kernel_name());

    // Device buffers.
    let alloc = |bytes: usize| nsl_test_cuda_alloc(bytes as i64);
    let xs_dev = alloc(ROWS * HS * 4);
    let ws_dev = alloc(V * HS * 4);
    let bs_dev = alloc(V * 4);
    let xt_dev = alloc(ROWS * HT * 4);
    let wt_dev = alloc(V * HT * 4);
    let bt_dev = alloc(V * 4);
    let tgt_dev = alloc(ROWS * 8);
    let loss_dev = alloc(ROWS * 4);
    let lse_s1_dev = alloc(ROWS * 4);
    let lse_st_dev = alloc(ROWS * 4);
    let lse_tt_dev = alloc(ROWS * 4);
    let dxs_dev = alloc(ROWS * HS * 4);
    let dws_dev = alloc(V * HS * 4);
    let dbs_dev = alloc(V * 4);
    for d in [
        xs_dev, ws_dev, bs_dev, xt_dev, wt_dev, bt_dev, tgt_dev, loss_dev, lse_s1_dev,
        lse_st_dev, lse_tt_dev, dxs_dev, dws_dev, dbs_dev,
    ] {
        assert_ne!(d, 0, "device alloc failed");
    }

    let h2d = |dev: i64, host: *const u8, bytes: usize| {
        nsl_test_cuda_h2d(dev, host as i64, bytes as i64);
    };
    h2d(xs_dev, xs.as_ptr() as *const u8, ROWS * HS * 4);
    h2d(ws_dev, ws.as_ptr() as *const u8, V * HS * 4);
    h2d(bs_dev, bs.as_ptr() as *const u8, V * 4);
    h2d(xt_dev, xt.as_ptr() as *const u8, ROWS * HT * 4);
    h2d(wt_dev, wt.as_ptr() as *const u8, V * HT * 4);
    h2d(bt_dev, bt.as_ptr() as *const u8, V * 4);
    h2d(tgt_dev, targets.as_ptr() as *const u8, ROWS * 8);
    // Gradient outputs must be zero-filled (red.global.add accumulation).
    let zeros_dxs = vec![0f32; ROWS * HS];
    let zeros_dws = vec![0f32; V * HS];
    let zeros_dbs = vec![0f32; V];
    h2d(dxs_dev, zeros_dxs.as_ptr() as *const u8, ROWS * HS * 4);
    h2d(dws_dev, zeros_dws.as_ptr() as *const u8, V * HS * 4);
    h2d(dbs_dev, zeros_dbs.as_ptr() as *const u8, V * 4);

    // ── Forward ──────────────────────────────────────────────────────────
    let rc = nsl_fused_kl_ce_forward(
        fwd_ptx.as_ptr() as i64,
        fwd_name.as_ptr() as i64,
        xs_dev,
        ws_dev,
        bs_dev,
        xt_dev,
        wt_dev,
        bt_dev,
        tgt_dev,
        loss_dev,
        lse_s1_dev,
        lse_st_dev,
        lse_tt_dev,
        ROWS as i64,
        V as i64,
        HS as i64,
        HT as i64,
        f32_bits(ALPHA),
        f32_bits(TEMP),
        cfg.shared_mem_bytes() as i64,
    );
    assert_eq!(rc, 0, "forward launch failed rc={rc}");

    let d2h = |dev: i64, host: *mut u8, bytes: usize| {
        nsl_test_cuda_d2h(host as i64, dev, bytes as i64);
    };
    let mut gpu_loss = vec![0f32; ROWS];
    let mut gpu_lse_s1 = vec![0f32; ROWS];
    let mut gpu_lse_st = vec![0f32; ROWS];
    let mut gpu_lse_tt = vec![0f32; ROWS];
    d2h(loss_dev, gpu_loss.as_mut_ptr() as *mut u8, ROWS * 4);
    d2h(lse_s1_dev, gpu_lse_s1.as_mut_ptr() as *mut u8, ROWS * 4);
    d2h(lse_st_dev, gpu_lse_st.as_mut_ptr() as *mut u8, ROWS * 4);
    d2h(lse_tt_dev, gpu_lse_tt.as_mut_ptr() as *mut u8, ROWS * 4);

    let rel = |gpu: f32, r: f64| ((gpu as f64) - r).abs() / r.abs().max(1.0);
    for row in 0..ROWS {
        if targets[row] == IGNORE_INDEX {
            assert_eq!(gpu_loss[row], 0.0, "row {row}: skip row loss must be 0");
            assert_eq!(gpu_lse_s1[row], 0.0, "row {row}: skip row lse_s1 must be 0");
            continue;
        }
        assert!(
            rel(gpu_loss[row], ref_loss[row]) < FORWARD_TOL,
            "row {row}: loss gpu={} ref={}",
            gpu_loss[row],
            ref_loss[row]
        );
        assert!(
            rel(gpu_lse_s1[row], ref_lse_s1[row]) < FORWARD_TOL,
            "row {row}: lse_s1 gpu={} ref={}",
            gpu_lse_s1[row],
            ref_lse_s1[row]
        );
        assert!(
            rel(gpu_lse_st[row], ref_lse_st[row]) < FORWARD_TOL,
            "row {row}: lse_sT gpu={} ref={}",
            gpu_lse_st[row],
            ref_lse_st[row]
        );
        assert!(
            rel(gpu_lse_tt[row], ref_lse_tt[row]) < FORWARD_TOL,
            "row {row}: lse_tT gpu={} ref={}",
            gpu_lse_tt[row],
            ref_lse_tt[row]
        );
    }
    eprintln!("forward: {ROWS} rows match reference (rel < {FORWARD_TOL:e})");

    // ── Backward ─────────────────────────────────────────────────────────
    let rc = nsl_fused_kl_ce_backward(
        bwd_ptx.as_ptr() as i64,
        bwd_name.as_ptr() as i64,
        f32_bits(1.0),
        xs_dev,
        ws_dev,
        bs_dev,
        xt_dev,
        wt_dev,
        bt_dev,
        tgt_dev,
        lse_s1_dev,
        lse_st_dev,
        lse_tt_dev,
        dxs_dev,
        dws_dev,
        dbs_dev,
        ROWS as i64,
        V as i64,
        HS as i64,
        HT as i64,
        f32_bits(ALPHA),
        f32_bits(TEMP),
        num_valid,
    );
    assert_eq!(rc, 0, "backward launch failed rc={rc}");

    let mut gpu_dxs = vec![0f32; ROWS * HS];
    let mut gpu_dws = vec![0f32; V * HS];
    let mut gpu_dbs = vec![0f32; V];
    d2h(dxs_dev, gpu_dxs.as_mut_ptr() as *mut u8, ROWS * HS * 4);
    d2h(dws_dev, gpu_dws.as_mut_ptr() as *mut u8, V * HS * 4);
    d2h(dbs_dev, gpu_dbs.as_mut_ptr() as *mut u8, V * 4);

    let max_err = |name: &str, gpu: &[f32], r: &[f64]| {
        // `f64::max` returns the non-NaN operand, so the fold below drops a
        // NaN error and scores an all-NaN gradient 0.0. Refuse it first, on
        // both sides.
        if let Some(i) = gpu.iter().position(|v| !v.is_finite()) {
            panic!("{name}: non-finite entry in gpu at index {i} ({})", gpu[i]);
        }
        if let Some(i) = r.iter().position(|v| !v.is_finite()) {
            panic!("{name}: non-finite entry in reference at index {i} ({})", r[i]);
        }
        gpu.iter()
            .zip(r.iter())
            .map(|(&g, &rr)| ((g as f64) - rr).abs())
            .fold(0f64, f64::max)
    };
    for (name, gpu, expected) in [
        ("dx_s", &gpu_dxs, ref_dxs),
        ("dW_s", &gpu_dws, ref_dws),
        ("dbias_s", &gpu_dbs, ref_dbs),
    ] {
        let err = max_err(name, gpu, expected);
        let scale = max_abs(expected);
        eprintln!("backward {name}: max abs err {err:.2e}, largest reference entry {scale:.2e}");
        assert!(err < GRADIENT_TOL * scale, "{name}: max err {err:e} against a largest entry of {scale:e}");
    }

    // Skip identity: -100 rows have exactly-zero dx_s.
    for row in 0..ROWS {
        if targets[row] == IGNORE_INDEX {
            for h in 0..HS {
                assert_eq!(
                    gpu_dxs[row * HS + h],
                    0.0,
                    "row {row} h {h}: skip row dx_s must be exactly 0"
                );
            }
        }
    }

    for d in [
        xs_dev, ws_dev, bs_dev, xt_dev, wt_dev, bt_dev, tgt_dev, loss_dev, lse_s1_dev,
        lse_st_dev, lse_tt_dev, dxs_dev, dws_dev, dbs_dev,
    ] {
        nsl_test_cuda_free(d);
    }
}
