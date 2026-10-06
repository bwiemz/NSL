//! Compiled-program certificates for the fused loss kernels' source-AD
//! wiring (external review 2026-10-06, finding 6).
//!
//! `FusedLinearCe` was `Uncertified` in `ad_rules::ad_cert_status`: its
//! kernels are held to an f64 reference (`nsl-codegen/tests/
//! fused_linear_ce_numerical.rs`), but nothing checked that a compiled
//! program's gradients through them are right -- the adjoint wiring, the
//! upstream-gradient scale, the ignore-index count, which operand gets
//! which gradient. The op is built only inside a `train` block and runs
//! only on the GPU, so `source_ad_rule_cert.rs` (CPU `grad` blocks) cannot
//! reach it.
//!
//! The certificate is one training step:
//!
//! * `x`, `W` and `b` are MODEL FIELDS, so every operand of the loss is a
//!   parameter and all three gradients show up as updates;
//! * the optimizer is plain `SGD` (momentum 0, no weight decay), so the
//!   update is exactly `lr * grad` and `(before - after) / lr` is the raw
//!   gradient -- no AdamW to normalise a wrong scale away;
//! * the loss is `0.37 * fused_linear_ce(...)`, so the backward must honour
//!   the upstream gradient, and two of the eight targets are the ignore
//!   index, so the mean must divide by the six valid rows;
//! * the oracle is an f64 central difference of the composite loss over
//!   every one of the 4480 parameter entries, computed here from the
//!   parameter values the program printed before the step.
//!
//! The fused path must provably have run: source AD engaged with no tape
//! fallback, and the fused forward and backward kernels each launched.

use std::path::{Path, PathBuf};
use std::process::Command;

const N: usize = 8; // rows = batch_size * seq_len
const H: usize = 32;
const V: usize = 128;
const LR: f64 = 4.0;
const SCALE: f64 = 0.37;
/// Two ignored rows: the mean is over the other six.
const TARGETS: [f64; N] = [3.0, 17.0, -100.0, 64.0, 127.0, -100.0, 0.0, 99.0];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn program() -> String {
    let targets = TARGETS
        .iter()
        .map(|t| format!("full([1], {t:.1})"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"from nsl.nn.losses import fused_linear_ce

model Head:
    x: Tensor = randn([{N}, {H}]) * 0.5
    w: Tensor = randn([{V}, {H}]) * 0.2
    b: Tensor = randn([{V}]) * 0.1

let m = Head()
m.to(cuda)
let t = tensor_cat([{targets}], 0).to(cuda)

print("X0_BEGIN")
print(m.x)
print("X0_END")
print("W0_BEGIN")
print(m.w)
print("W0_END")
print("B0_BEGIN")
print(m.b)
print("B0_END")

@fused_lm_ce(enabled=true, dtype="f32", vocab_size={V}, hidden_size={H}, batch_size=1, seq_len={N}, vocab_tile=128)
train(model=m, epochs=1):
    optimizer: SGD(lr={LR:.1})
    step(batch):
        let loss = {SCALE} * fused_linear_ce(m.x, m.w, m.b, t)

print("X1_BEGIN")
print(m.x)
print("X1_END")
print("W1_BEGIN")
print(m.w)
print("W1_END")
print("B1_BEGIN")
print(m.b)
print("B1_END")
print("FWD_BEGIN")
print(fused_lce_launch_count(0))
print("FWD_END")
print("BWD_BEGIN")
print(fused_lce_launch_count(2))
print("BWD_END")
"#
    )
}

/// The numbers printed between `<tag>_BEGIN` and `<tag>_END`. A non-finite
/// value is refused by name: a NaN parameter must not parse as a short list.
fn between(stdout: &str, tag: &str) -> Vec<f64> {
    let (b, e) = (format!("{tag}_BEGIN"), format!("{tag}_END"));
    let start = stdout.find(&b).unwrap_or_else(|| panic!("{b} not printed:\n{stdout}")) + b.len();
    let end = start + stdout[start..].find(&e).unwrap_or_else(|| panic!("{e} not printed"));
    let body = &stdout[start..end];
    let lower = body.to_ascii_lowercase();
    assert!(!lower.contains("nan") && !lower.contains("inf"), "{tag} is not finite:\n{body}");
    body.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
        .map(|t| t.parse::<f64>().unwrap_or_else(|e| panic!("{tag}: '{t}': {e}")))
        .collect()
}

/// `SCALE * mean over valid rows of (logsumexp(x_i W^T + b) - logit_i[t_i])`,
/// in f64: the composite `fused_linear_ce` (stdlib `nn/losses.nsl`).
fn loss(x: &[f64], w: &[f64], b: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut valid = 0usize;
    for (i, &t) in TARGETS.iter().enumerate() {
        if t < 0.0 {
            continue;
        }
        let logits: Vec<f64> = (0..V)
            .map(|v| (0..H).map(|h| x[i * H + h] * w[v * H + h]).sum::<f64>() + b[v])
            .collect();
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
        total += lse - logits[t as usize];
        valid += 1;
    }
    SCALE * total / valid as f64
}

/// The fused linear-CE loss over (x, W, b).
fn lce_loss(p: &[Vec<f64>]) -> f64 {
    loss(&p[0], &p[1], &p[2])
}
const LCE: fn(&[Vec<f64>]) -> f64 = lce_loss;

/// Central differences of `f` for every entry of every operand in `p`.
fn fd_grad(p: &[Vec<f64>], f: fn(&[Vec<f64>]) -> f64) -> Vec<Vec<f64>> {
    fd_grad_some(p, f, p.len())
}

/// Certificate `fused_linear_ce_step` (named by `ad_cert_status` for
/// `FusedLinearCe`; `source_ad_rule_cert.rs`'s coverage gate checks this fn
/// exists): dx, dW and db through a compiled step against f64 central
/// differences.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_linear_ce_step() {
    let Some((stdout, stderr)) = run(&program(), "lce") else { return };

    // The fused path ran, under source AD, with no tape fallback.
    assert!(stderr.contains("Using source-to-source AD for backward pass"), "{stderr}");
    assert!(!stderr.contains("falling back to tape-based AD"), "{stderr}");
    let fwd = between(&stdout, "FWD")[0];
    let bwd = between(&stdout, "BWD")[0];
    assert!(fwd >= 1.0 && bwd >= 1.0, "the fused kernels must have launched (fwd {fwd}, bwd {bwd})");

    let before = [between(&stdout, "X0"), between(&stdout, "W0"), between(&stdout, "B0")];
    let after = [between(&stdout, "X1"), between(&stdout, "W1"), between(&stdout, "B1")];
    for (k, n) in [N * H, V * H, V].into_iter().enumerate() {
        assert_eq!(before[k].len(), n, "operand {k}: printed {} values", before[k].len());
        assert_eq!(after[k].len(), n, "operand {k}: printed {} values", after[k].len());
    }

    compare(&before, &after, &["dx", "dW", "db"], |q| fd_grad(q, LCE));
}

/// Run `src` with `nsl run --source-ad` on the GPU. `None` when there is no
/// CUDA driver (the gate is GPU-only; the cert lane has one).
fn run(src: &str, tag: &str) -> Option<(String, String)> {
    run_env(src, tag, &[])
}

fn run_env(src: &str, tag: &str, env: &[(&str, &str)]) -> Option<(String, String)> {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_fused_cert_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("cert.nsl");
    std::fs::write(&prog, src).unwrap();
    // f32 arithmetic everywhere outside the fused kernel, so the only
    // reduced-precision step is the one being certified.
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--matmul-mode", "f32"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .envs(env.iter().copied())
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&tmp);
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    if stderr.contains("CUDA driver") && stderr.contains("not found") {
        eprintln!("SKIP: no CUDA driver");
        return None;
    }
    assert!(out.status.success(), "the program failed:\nstdout:\n{stdout}\nstderr:\n{stderr}");
    Some((stdout, stderr))
}

/// `(before - after) / LR` for each operand against `oracle`'s f64 gradient
/// of the same operand.
fn compare(before: &[Vec<f64>], after: &[Vec<f64>], names: &[&str], oracle: impl Fn(&[Vec<f64>]) -> Vec<Vec<f64>>) {
    let wants = oracle(before);
    let mut report = Vec::new();
    for (k, name) in names.iter().enumerate() {
        let want = &wants[k];
        let got: Vec<f64> = before[k].iter().zip(&after[k]).map(|(b, a)| (b - a) / LR).collect();
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale > 1e-4, "{name}: the oracle gradient is ~0 ({scale:e}), so the check would be vacuous");
        let (worst, at) = got
            .iter()
            .zip(want)
            .enumerate()
            .map(|(i, (g, w))| ((g - w).abs(), i))
            .fold((0.0f64, 0), |acc, x| if x.0 > acc.0 { x } else { acc });
        // f32 parameters: (before - after) carries ~1 ulp of each value
        // (~1.2e-7 at |p| ~ 1), divided by LR; the kernel's own f32
        // accumulation over H = 32 and V = 128 is below that.
        let tol = 1e-4 * scale + 1e-7;
        report.push(format!(
            "{name}: max |got - fd| = {worst:.3e} at {at} (got {:.6e}, fd {:.6e}), scale {scale:.3e}, tol {tol:.3e}",
            got[at], want[at]
        ));
        assert!(worst <= tol, "{name} disagrees with the f64 oracle:\n{}", report.join("\n"));
    }
    eprintln!("{}", report.join("\n"));
}

// ---------------------------------------------------------------------------
// FusedKlCe (CPKD distillation)
// ---------------------------------------------------------------------------

const HT: usize = 64; // teacher hidden; the kernel supports HS != HT
const ALPHA: f64 = 0.6;
const TEMP: f64 = 2.0;

fn kl_program() -> String {
    let targets = TARGETS
        .iter()
        .map(|t| format!("full([1], {t:.1})"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"from nsl.nn.losses import fused_kl_ce

model Teacher:
    xt: Tensor = randn([{N}, {HT}]) * 0.5
    wt: Tensor = randn([{V}, {HT}]) * 0.2
    bt: Tensor = randn([{V}]) * 0.1

model Student:
    xs: Tensor = randn([{N}, {H}]) * 0.5
    ws: Tensor = randn([{V}, {H}]) * 0.2
    bs: Tensor = randn([{V}]) * 0.1

let teacher = Teacher()
let student = Student()
teacher.to(cuda)
student.to(cuda)
let t = tensor_cat([{targets}], 0).to(cuda)

print("S0X_BEGIN")
print(student.xs)
print("S0X_END")
print("S0W_BEGIN")
print(student.ws)
print("S0W_END")
print("S0B_BEGIN")
print(student.bs)
print("S0B_END")
print("T0X_BEGIN")
print(teacher.xt)
print("T0X_END")
print("T0W_BEGIN")
print(teacher.wt)
print("T0W_END")
print("T0B_BEGIN")
print(teacher.bt)
print("T0B_END")

@fused_kl_ce(enabled = true, vocab_size = {V}, hidden_size = {H}, teacher_hidden = {HT}, batch_size = 1, seq_len = {N}, vocab_tile = 128)
distill(teacher = teacher, student = student, epochs = 1):
    optimizer: SGD(lr = {LR:.1})
    loss:
        alpha = {ALPHA}
        temperature = {TEMP:.1}
    step(batch):
        let loss = {SCALE} * fused_kl_ce(student.xs, student.ws, student.bs, teacher.xt, teacher.wt, teacher.bt, t, {ALPHA}, {TEMP:.1})

print("S1X_BEGIN")
print(student.xs)
print("S1X_END")
print("S1W_BEGIN")
print(student.ws)
print("S1W_END")
print("S1B_BEGIN")
print(student.bs)
print("S1B_END")
print("T1W_BEGIN")
print(teacher.wt)
print("T1W_END")
"#
    )
}

fn log_softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + z.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
    z.iter().map(|v| v - lse).collect()
}

fn head(x: &[f64], w: &[f64], b: &[f64], i: usize, hid: usize) -> Vec<f64> {
    (0..V).map(|v| (0..hid).map(|h| x[i * hid + h] * w[v * hid + h]).sum::<f64>() + b[v]).collect()
}

/// `SCALE * mean over valid rows of (ALPHA * CE(s) + (1 - ALPHA) * T^2 *
/// KL(softmax(t/T) || softmax(s/T)))`, the composite `fused_kl_ce` (stdlib
/// `nn/losses.nsl`), over the student (xs, ws, bs) with the teacher fixed.
fn kl_loss(p: &[Vec<f64>]) -> f64 {
    let (xs, ws, bs, xt, wt, bt) = (&p[0], &p[1], &p[2], &p[3], &p[4], &p[5]);
    let (mut ce, mut kl, mut valid) = (0.0, 0.0, 0usize);
    for (i, &t) in TARGETS.iter().enumerate() {
        if t < 0.0 {
            continue;
        }
        let s = head(xs, ws, bs, i, H);
        let te = head(xt, wt, bt, i, HT);
        ce -= log_softmax(&s)[t as usize];
        let ls = log_softmax(&s.iter().map(|v| v / TEMP).collect::<Vec<_>>());
        let lt = log_softmax(&te.iter().map(|v| v / TEMP).collect::<Vec<_>>());
        kl += lt.iter().zip(&ls).map(|(a, b)| a.exp() * (a - b)).sum::<f64>();
        valid += 1;
    }
    let n = valid as f64;
    SCALE * (ALPHA * ce / n + (1.0 - ALPHA) * TEMP * TEMP * kl / n)
}

/// Certificate `fused_kl_ce_step` (named by `ad_cert_status` for
/// `FusedKlCe`): the student's dx, dW and db through a compiled distill
/// step against f64 central differences, and the teacher left untouched.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_kl_ce_step() {
    let Some((stdout, stderr)) = run(&kl_program(), "klce") else { return };
    assert!(stderr.contains("Using source-to-source AD"), "{stderr}");
    assert!(!stderr.contains("falling back to tape-based AD"), "{stderr}");
    assert!(
        stderr.contains("Fused KL-CE"),
        "the fused KL-CE kernel must be what ran (build report line):\n{stderr}"
    );

    let teacher_w0 = between(&stdout, "T0W");
    assert_eq!(teacher_w0, between(&stdout, "T1W"), "the teacher is frozen: its head must not move");
    let before = [
        between(&stdout, "S0X"),
        between(&stdout, "S0W"),
        between(&stdout, "S0B"),
        between(&stdout, "T0X"),
        teacher_w0,
        between(&stdout, "T0B"),
    ];
    let after = [between(&stdout, "S1X"), between(&stdout, "S1W"), between(&stdout, "S1B")];
    for (k, n) in [N * H, V * H, V, N * HT, V * HT, V].into_iter().enumerate() {
        assert_eq!(before[k].len(), n, "operand {k}: printed {} values", before[k].len());
    }
    // Only the student's three operands are compared; the oracle varies just
    // those (the teacher's are inputs to it).
    compare(&before[..3], &after, &["dxs", "dWs", "dbs"], |q| {
        let mut full = q.to_vec();
        full.extend_from_slice(&before[3..]);
        let mut g = fd_grad_some(&full, kl_loss, 3);
        g.truncate(3);
        g
    });
}

/// [`fd_grad`] over only the first `k` operands of `p`.
fn fd_grad_some(p: &[Vec<f64>], f: fn(&[Vec<f64>]) -> f64, k: usize) -> Vec<Vec<f64>> {
    let h = 1e-6;
    let mut q = p.to_vec();
    (0..k)
        .map(|j| {
            (0..p[j].len())
                .map(|i| {
                    let orig = q[j][i];
                    q[j][i] = orig + h;
                    let up = f(&q);
                    q[j][i] = orig - h;
                    let down = f(&q);
                    q[j][i] = orig;
                    (up - down) / (2.0 * h)
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ScaledDotProductAttentionPacked on the GPU kernels
// ---------------------------------------------------------------------------

// The fused packed kernels need head_dim in {32, 64, 128} and the sequence a
// multiple of the 64-row tile. Three uneven documents.
const PB: usize = 1;
const PH: usize = 2;
const PS: usize = 64;
const PD: usize = 32;
const DOCS: [usize; 3] = [20, 30, 14];

fn packed_program() -> String {
    let scale = 1.0 / (PD as f64).sqrt();
    let seg = DOCS
        .iter()
        .enumerate()
        .map(|(d, n)| format!("full([1, {n}], {d}.0)"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"model Attn:
    q: Tensor = randn([{PB}, {PH}, {PS}, {PD}])
    k: Tensor = randn([{PB}, {PH}, {PS}, {PD}])
    v: Tensor = randn([{PB}, {PH}, {PS}, {PD}])

let m = Attn()
m.to(cuda)
let seg = tensor_cat([{seg}], 1).to(cuda)
let r = randn([{PB}, {PH}, {PS}, {PD}]).to(cuda)

print("R_BEGIN")
print(r)
print("R_END")
print("Q0_BEGIN")
print(m.q)
print("Q0_END")
print("K0_BEGIN")
print(m.k)
print("K0_END")
print("V0_BEGIN")
print(m.v)
print("V0_END")

train(model=m, epochs=1):
    optimizer: SGD(lr={LR:.1})
    step(batch):
        let loss = sum(scaled_dot_product_attention_packed(m.q, m.k, m.v, {scale}, seg) * r)

print("Q1_BEGIN")
print(m.q)
print("Q1_END")
print("K1_BEGIN")
print(m.k)
print("K1_END")
print("V1_BEGIN")
print(m.v)
print("V1_END")
print("FUSED_BEGIN")
print(sdpa_fused_launch_count(0) + sdpa_fused_launch_count(1))
print("FUSED_END")
"#
    )
}

fn doc_of(i: usize) -> usize {
    let mut end = 0;
    for (d, n) in DOCS.iter().enumerate() {
        end += n;
        if i < end {
            return d;
        }
    }
    unreachable!("position {i} past the packed sequence")
}

/// Exact f64 gradients of `sum(packed_attention(q, k, v) * r)` w.r.t. q, k
/// and v: the standard attention backward with dO = r, causal within each
/// document.
fn packed_grads(q: &[f64], k: &[f64], v: &[f64], r: &[f64]) -> [Vec<f64>; 3] {
    let scale = 1.0 / (PD as f64).sqrt();
    let (mut dq, mut dk, mut dv) = (vec![0.0; q.len()], vec![0.0; k.len()], vec![0.0; v.len()]);
    for bh in 0..PB * PH {
        let base = bh * PS * PD;
        let at = |t: &[f64], i: usize, c: usize| t[base + i * PD + c];
        for i in 0..PS {
            let vis: Vec<usize> = (0..=i).filter(|&j| doc_of(j) == doc_of(i)).collect();
            let scores: Vec<f64> =
                vis.iter().map(|&j| (0..PD).map(|c| at(q, i, c) * at(k, j, c)).sum::<f64>() * scale).collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = e.iter().sum();
            let p: Vec<f64> = e.iter().map(|x| x / z).collect();
            // dP_ij = dO_i . V_j; dS = P (dP - sum_k P_ik dP_ik).
            let dp: Vec<f64> = vis.iter().map(|&j| (0..PD).map(|c| at(r, i, c) * at(v, j, c)).sum()).collect();
            let pdp: f64 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
            for (n, &j) in vis.iter().enumerate() {
                let ds = p[n] * (dp[n] - pdp);
                for c in 0..PD {
                    dv[base + j * PD + c] += p[n] * at(r, i, c);
                    dq[base + i * PD + c] += scale * ds * at(k, j, c);
                    dk[base + j * PD + c] += scale * ds * at(q, i, c);
                }
            }
        }
    }
    [dq, dk, dv]
}

/// The forward of [`packed_grads`]' loss, for a finite-difference spot check
/// of the analytic oracle itself.
fn packed_loss(q: &[f64], k: &[f64], v: &[f64], r: &[f64]) -> f64 {
    let scale = 1.0 / (PD as f64).sqrt();
    let mut total = 0.0;
    for bh in 0..PB * PH {
        let base = bh * PS * PD;
        let at = |t: &[f64], i: usize, c: usize| t[base + i * PD + c];
        for i in 0..PS {
            let vis: Vec<usize> = (0..=i).filter(|&j| doc_of(j) == doc_of(i)).collect();
            let scores: Vec<f64> =
                vis.iter().map(|&j| (0..PD).map(|c| at(q, i, c) * at(k, j, c)).sum::<f64>() * scale).collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for c in 0..PD {
                let o: f64 = vis.iter().zip(&e).map(|(&j, w)| w / z * at(v, j, c)).sum();
                total += o * at(r, i, c);
            }
        }
    }
    total
}

/// Certificate `sdpa_packed_step` (named by `ad_cert_status` for
/// `ScaledDotProductAttentionPacked`, beside the CPU `grad` certificates):
/// q, k and v gradients through a compiled step on the FUSED kernels --
/// the forward's launch counter and the GPU backward's dispatch line prove
/// which path ran -- against exact f64 gradients. The kernels' MMA operands
/// are f16, so the bound is the packed parity gate's composed
/// forward-then-backward bound (1e-2 of the largest gradient), not f32
/// noise.
#[test]
#[ignore = "requires CUDA GPU"]
fn sdpa_packed_step() {
    let Some((stdout, stderr)) = run_env(&packed_program(), "packed", &[("NSL_FLASH_DEBUG", "1")]) else {
        return;
    };
    assert!(stderr.contains("Using source-to-source AD for backward pass"), "{stderr}");
    assert!(!stderr.contains("falling back to tape-based AD"), "{stderr}");
    assert!(between(&stdout, "FUSED")[0] >= 1.0, "the fused packed forward must have launched:\n{stderr}");
    assert!(
        stderr.contains("[flash-bwd] GPU backward dispatched"),
        "the GPU packed backward must have run, not the CPU reference:\n{stderr}"
    );

    let r = between(&stdout, "R");
    let before = [between(&stdout, "Q0"), between(&stdout, "K0"), between(&stdout, "V0")];
    let after = [between(&stdout, "Q1"), between(&stdout, "K1"), between(&stdout, "V1")];
    let n = PB * PH * PS * PD;
    assert_eq!(r.len(), n);
    for k in 0..3 {
        assert_eq!((before[k].len(), after[k].len()), (n, n), "operand {k}");
    }
    let exact = packed_grads(&before[0], &before[1], &before[2], &r);

    // The oracle itself, against central differences at a spread of entries.
    let h = 1e-6;
    for (k, stride) in [(0usize, 97usize), (1, 89), (2, 83)] {
        for i in (0..n).step_by(stride) {
            let mut p = before.clone();
            p[k][i] += h;
            let up = packed_loss(&p[0], &p[1], &p[2], &r);
            p[k][i] -= 2.0 * h;
            let down = packed_loss(&p[0], &p[1], &p[2], &r);
            let fd = (up - down) / (2.0 * h);
            assert!((fd - exact[k][i]).abs() <= 1e-6 * exact[k][i].abs().max(1.0), "oracle {k}[{i}]: {fd} vs {}", exact[k][i]);
        }
    }

    let mut report = Vec::new();
    for (k, name) in ["dq", "dk", "dv"].into_iter().enumerate() {
        let got: Vec<f64> = before[k].iter().zip(&after[k]).map(|(b, a)| (b - a) / LR).collect();
        let scale = exact[k].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(&exact[k]).fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
        report.push(format!("{name}: max |got - exact| = {worst:.3e}, scale {scale:.3e}, rel {:.3e}", worst / scale));
        assert!(worst <= 1e-2 * scale, "{name} disagrees with the exact gradient:\n{}", report.join("\n"));
    }
    eprintln!("{}", report.join("\n"));
}
