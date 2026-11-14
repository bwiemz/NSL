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
//! `the_device_comparison_catches_wrong_results` runs the same comparison
//! on results that are wrong (no backward; another temperature; no KL
//! term) and requires it to report them.
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

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {rc}");
        false
    } else {
        true
    }
}

fn f32_bits(v: f64) -> i64 {
    (v as f32).to_bits() as i64
}

fn config() -> FusedKlCeConfig {
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
    cfg
}

/// What the device produced for one forward (and, optionally, backward).
struct Device {
    loss: Vec<f32>,
    lse_s1: Vec<f32>,
    lse_st: Vec<f32>,
    lse_tt: Vec<f32>,
    dxs: Vec<f32>,
    dws: Vec<f32>,
    dbs: Vec<f32>,
}

fn upload<T>(host: &[T]) -> i64 {
    let bytes = std::mem::size_of_val(host) as i64;
    let dev = nsl_test_cuda_alloc(bytes);
    assert_ne!(dev, 0, "device alloc failed");
    nsl_test_cuda_h2d(dev, host.as_ptr() as i64, bytes);
    dev
}

fn download(dev: i64, len: usize) -> Vec<f32> {
    let mut host = vec![0f32; len];
    nsl_test_cuda_d2h(host.as_mut_ptr() as i64, dev, (len * 4) as i64);
    host
}

/// Launch the forward, then the backward unless `run_backward` is false
/// (the gradient buffers then stay as the caller zero-filled them), with
/// the given loss constants, and read everything back.
fn launch(fixture: &Fixture, alpha: f64, temperature: f64, run_backward: bool) -> Device {
    let cfg = config();
    let num_valid = fixture.targets.iter().filter(|&&t| t != IGNORE_INDEX).count() as i64;
    let fwd_ptx = synthesize_fused_kl_ce_ptx(&cfg);
    let bwd_ptx = synthesize_fused_kl_ce_backward_ptx(&cfg);
    let fwd_name = format!("{}\0", cfg.kernel_name());
    let bwd_name = format!("{}\0", cfg.bwd_kernel_name());

    let (xs, ws, bs) = (upload(&fixture.xs), upload(&fixture.ws), upload(&fixture.bs));
    let (xt, wt, bt) = (upload(&fixture.xt), upload(&fixture.wt), upload(&fixture.bt));
    let targets = upload(&fixture.targets);
    // Outputs. The gradient buffers must be zero-filled: the kernel
    // accumulates into them (`red.global.add`).
    let rows = vec![0f32; ROWS];
    let (loss, lse_s1, lse_st, lse_tt) = (upload(&rows), upload(&rows), upload(&rows), upload(&rows));
    let (dxs, dws, dbs) = (upload(&vec![0f32; ROWS * HS]), upload(&vec![0f32; V * HS]), upload(&vec![0f32; V]));

    let rc = nsl_fused_kl_ce_forward(
        fwd_ptx.as_ptr() as i64,
        fwd_name.as_ptr() as i64,
        xs,
        ws,
        bs,
        xt,
        wt,
        bt,
        targets,
        loss,
        lse_s1,
        lse_st,
        lse_tt,
        ROWS as i64,
        V as i64,
        HS as i64,
        HT as i64,
        f32_bits(alpha),
        f32_bits(temperature),
        cfg.shared_mem_bytes() as i64,
    );
    assert_eq!(rc, 0, "forward launch failed rc={rc}");
    if run_backward {
        let rc = nsl_fused_kl_ce_backward(
            bwd_ptx.as_ptr() as i64,
            bwd_name.as_ptr() as i64,
            f32_bits(1.0),
            xs,
            ws,
            bs,
            xt,
            wt,
            bt,
            targets,
            lse_s1,
            lse_st,
            lse_tt,
            dxs,
            dws,
            dbs,
            ROWS as i64,
            V as i64,
            HS as i64,
            HT as i64,
            f32_bits(alpha),
            f32_bits(temperature),
            num_valid,
        );
        assert_eq!(rc, 0, "backward launch failed rc={rc}");
    }

    let out = Device {
        loss: download(loss, ROWS),
        lse_s1: download(lse_s1, ROWS),
        lse_st: download(lse_st, ROWS),
        lse_tt: download(lse_tt, ROWS),
        dxs: download(dxs, ROWS * HS),
        dws: download(dws, V * HS),
        dbs: download(dbs, V),
    };
    for dev in [xs, ws, bs, xt, wt, bt, targets, loss, lse_s1, lse_st, lse_tt, dxs, dws, dbs] {
        nsl_test_cuda_free(dev);
    }
    out
}

/// Every way `got` departs from `want`, by name. Empty means the device
/// result is the reference within the tolerances at the top of this file.
fn mismatches(fixture: &Fixture, want: &Reference, got: &Device) -> Vec<String> {
    let mut found = Vec::new();
    for (name, gpu, expected) in [
        ("loss", &got.loss, &want.loss),
        ("lse_s1", &got.lse_s1, &want.lse_s1),
        ("lse_sT", &got.lse_st, &want.lse_st),
        ("lse_tT", &got.lse_tt, &want.lse_tt),
    ] {
        for row in 0..ROWS {
            let (g, w) = (gpu[row] as f64, expected[row]);
            let bad = if fixture.targets[row] == IGNORE_INDEX {
                // A skipped row is exactly zero.
                g != 0.0
            } else {
                !g.is_finite() || (g - w).abs() / w.abs().max(1.0) >= FORWARD_TOL
            };
            if bad {
                found.push(format!("{name}: row {row} is {g}, expected {w}"));
                break;
            }
        }
    }
    for (name, gpu, expected) in
        [("dx_s", &got.dxs, &want.dxs), ("dW_s", &got.dws, &want.dws), ("dbias_s", &got.dbs, &want.dbs)]
    {
        // `f64::max` returns the non-NaN operand, so a NaN error would
        // drop out of the fold below: refuse non-finite entries first.
        if let Some(i) = gpu.iter().position(|v| !v.is_finite()) {
            found.push(format!("{name}: non-finite entry at {i} ({})", gpu[i]));
            continue;
        }
        let err = gpu.iter().zip(expected.iter()).map(|(&g, &w)| (g as f64 - w).abs()).fold(0f64, f64::max);
        let scale = max_abs(expected);
        eprintln!("{name}: max abs err {err:.2e}, largest reference entry {scale:.2e}");
        if err >= GRADIENT_TOL * scale {
            found.push(format!("{name}: max err {err:e} against a largest entry of {scale:e}"));
        }
    }
    // Skip identity: an ignored row's dx_s is exactly zero.
    for row in (0..ROWS).filter(|&r| fixture.targets[r] == IGNORE_INDEX) {
        if got.dxs[row * HS..(row + 1) * HS].iter().any(|&g| g != 0.0) {
            found.push(format!("dx_s: ignored row {row} is not exactly zero"));
            break;
        }
    }
    found
}

#[test]
#[ignore = "requires CUDA GPU"]
fn fused_kl_ce_gpu_forward_and_backward_match_reference() {
    if !cuda_available() {
        return;
    }
    let fixture = Fixture::new();
    let want = reference(&fixture.inputs(), KlTerm::Forward);
    let got = launch(&fixture, ALPHA, TEMP, true);
    let found = mismatches(&fixture, &want, &got);
    assert!(found.is_empty(), "the device result is not the reference:\n{}", found.join("\n"));
}

/// The comparison above can fail, shown on the device with wrong results
/// obtained without touching the kernels:
///
/// * a backward that never ran (all-zero gradients) is reported for every
///   gradient tensor. This is the case the old absolute tolerance passed;
/// * kernels launched with temperature 1 instead of 2.5 compute a different
///   KL term, and are reported for the loss and for every gradient tensor;
/// * kernels launched with alpha 1 compute no KL term at all, likewise.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_device_comparison_catches_wrong_results() {
    if !cuda_available() {
        return;
    }
    let fixture = Fixture::new();
    let want = reference(&fixture.inputs(), KlTerm::Forward);
    let reported = |got: &Device, name: &str| mismatches(&fixture, &want, got).iter().any(|m| m.starts_with(name));

    let no_backward = launch(&fixture, ALPHA, TEMP, false);
    assert!(!reported(&no_backward, "loss"), "the forward is unchanged and must still match");
    for name in ["dx_s", "dW_s", "dbias_s"] {
        assert!(reported(&no_backward, name), "an all-zero {name} went unnoticed");
    }
    for (what, alpha, temperature) in [("temperature 1", ALPHA, 1.0), ("alpha 1 (no KL term)", 1.0, TEMP)] {
        let wrong = launch(&fixture, alpha, temperature, true);
        for name in ["loss", "dx_s", "dW_s", "dbias_s"] {
            assert!(reported(&wrong, name), "{what}: a wrong {name} went unnoticed");
        }
    }
}

/// A kernel launch that fails must end the process, not return.
///
/// The FFIs used to log and return a code, which the compiled step
/// dropped: the zero-filled loss buffer then read as a loss of 0 with zero
/// gradients, and training carried on. The launch is made to fail here by
/// asking the (valid) module for an entry it does not have. The call exits
/// the process, so it runs in a child: this test re-runs itself with
/// `KLCE_TEST_FAILED_LAUNCH_CHILD` set and reads the child's exit status,
/// which must be the runtime's CUDA-driver code, 13.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_failed_launch_exits_instead_of_returning() {
    const CHILD: &str = "KLCE_TEST_FAILED_LAUNCH_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let out = std::process::Command::new(std::env::current_exe().expect("test binary path"))
            .args(["--ignored", "--exact", "a_failed_launch_exits_instead_of_returning", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .expect("re-run the test binary");
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("skipping:") {
            eprintln!("{stderr}");
            return;
        }
        assert_eq!(out.status.code(), Some(13), "the child must exit with the CUDA-driver code:\n{stderr}");
        assert!(stderr.contains("the fused KL-CE kernel failed to launch"), "{stderr}");
        assert!(!stderr.contains("returned from a failed launch"), "{stderr}");
        return;
    }

    if !cuda_available() {
        return;
    }
    let cfg = config();
    let ptx = synthesize_fused_kl_ce_ptx(&cfg);
    let alloc = |bytes: usize| nsl_test_cuda_alloc(bytes as i64);
    let (xs, ws, bs) = (alloc(ROWS * HS * 4), alloc(V * HS * 4), alloc(V * 4));
    let (xt, wt, bt) = (alloc(ROWS * HT * 4), alloc(V * HT * 4), alloc(V * 4));
    let targets = alloc(ROWS * 8);
    let outs = [alloc(ROWS * 4), alloc(ROWS * 4), alloc(ROWS * 4), alloc(ROWS * 4)];
    let rc = nsl_fused_kl_ce_forward(
        ptx.as_ptr() as i64,
        c"nsl_no_such_kernel".as_ptr() as i64,
        xs,
        ws,
        bs,
        xt,
        wt,
        bt,
        targets,
        outs[0],
        outs[1],
        outs[2],
        outs[3],
        ROWS as i64,
        V as i64,
        HS as i64,
        HT as i64,
        f32_bits(ALPHA),
        f32_bits(TEMP),
        cfg.shared_mem_bytes() as i64,
    );
    // Not reached when the runtime is right. Exit 0 so the parent sees the
    // wrong status, and say why.
    eprintln!("nsl_fused_kl_ce_forward returned from a failed launch with rc={rc}");
    std::process::exit(0);
}
