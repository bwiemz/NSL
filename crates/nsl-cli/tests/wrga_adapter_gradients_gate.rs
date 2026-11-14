//! Adapter gradients on a `[batch, seq, d]` activation, on every path that
//! computes them: tape AD, source AD, and source AD through the fused
//! adapter call (`NSL_WRGA_FUSED_CUDA=1` with a pinned target).
//!
//! The fused call has its own adjoint rules in the source-AD pass. They
//! computed the weight, A, B, gate and scale gradients as batched products
//! and never summed them over the activation's leading dimensions: for `x`
//! of shape `[2, 3, 4]` the gradient of a `[4, 6]` weight came out
//! `[2, 4, 6]`, and the optimizer step aborted on the length. (The scale
//! and gate gradients were summed over axis 0 only.)
//!
//! Each program trains one adapted `[4, 6]` weight for a few SGD steps on a
//! non-constant `[2, 3, 4]` input and prints every tensor. The reference is
//! computed here in f64: the forward pass written out element by element,
//! each gradient by central differences. B starts from a non-zero tensor so
//! that the gradients of A and of the gate are non-zero from the first step
//! (with B at zero they vanish, and a rule that returned zero for them
//! would pass).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// How a program is built and run.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Path_ {
    TapeAd,
    SourceAd,
    /// Source AD with the adapter lowered to the single fused call. The
    /// tensors are host tensors, so the call runs its ordinary-operations
    /// arithmetic; what differs is the adjoint rule.
    SourceAdFusedCall,
}

struct Run {
    stdout: String,
    /// `[nsl-kernel-count] N`: how many fused adapter calls ran.
    fused_calls: u64,
}

fn run(source: &str, path: Path_) -> Run {
    let root = repo_root();
    let tmp = tempfile::TempDir::new().unwrap();
    let file = tmp.path().join("prog.nsl");
    std::fs::write(&file, source).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run")
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_KERNEL_LAUNCH_COUNTER", "1")
        .env_remove("NSL_WRGA_FUSED_CUDA");
    match path {
        Path_::TapeAd => {}
        Path_::SourceAd => {
            cmd.arg("--source-ad");
        }
        Path_::SourceAdFusedCall => {
            cmd.args(["--source-ad", "--target", "cuda_sm80"]).env("NSL_WRGA_FUSED_CUDA", "1");
        }
    }
    let out = cmd.arg(&file).output().expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{path:?} failed:\n{stderr}");
    let fused_calls = stderr
        .lines()
        .find(|l| l.contains("[nsl-kernel-count]"))
        .and_then(|l| l.split_whitespace().next_back())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    Run { stdout: String::from_utf8_lossy(&out.stdout).into_owned(), fused_calls }
}

/// The numbers printed between the lines `<tag>_BEGIN` and `<tag>_END`.
fn section(stdout: &str, tag: &str) -> Vec<f64> {
    let (begin, end) = (format!("{tag}_BEGIN"), format!("{tag}_END"));
    let mut lines = stdout.lines().skip_while(|l| l.trim() != begin);
    assert!(lines.next().is_some(), "no line {begin} in:\n{stdout}");
    let inner: Vec<&str> = lines.take_while(|l| l.trim() != end).collect();
    inner
        .join(" ")
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Lora,
    GatedLora,
    Ia3,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Lora => "lora",
            Kind::GatedLora => "gatedlora",
            Kind::Ia3 => "ia3",
        }
    }
}

// x is [2, 3, 4]: six rows of width 4. w is [4, 6].
const ROWS: usize = 6;
const N_IN: usize = 4;
const N_OUT: usize = 6;
const RANK: usize = 2;
const ALPHA: f64 = 4.0;
const LR: f64 = 0.05;
const STEPS: usize = 4;

// The data, as the NSL source builds it.
fn x_data() -> Vec<f64> {
    (0..ROWS * N_IN).map(|i| i as f64 * 0.1 + 0.1).collect()
}
fn y_data() -> Vec<f64> {
    (0..ROWS * N_OUT).map(|i| i as f64 * 0.05 - 1.0).collect()
}
/// B's starting value: `arange(12).reshape([2, 6]) * 0.05 - 0.2`.
fn b_start() -> Vec<f64> {
    (0..RANK * N_OUT).map(|i| i as f64 * 0.05 - 0.2).collect()
}

/// The tensors a model of `kind` holds, in the order of the reference's
/// parameter vector, with the tag each prints under and its length.
fn tensors(kind: Kind) -> Vec<(&'static str, usize)> {
    let mut t = vec![("W", N_IN * N_OUT)];
    match kind {
        Kind::Lora => t.extend([("A", N_IN * RANK), ("B", RANK * N_OUT)]),
        Kind::GatedLora => t.extend([("A", N_IN * RANK), ("B", RANK * N_OUT), ("GATE", N_OUT)]),
        Kind::Ia3 => t.push(("SCALE", N_OUT)),
    }
    t
}

/// The adapted forward pass on the six rows, element by element:
///   LoRA:       x @ W + ((x @ A) @ B) * (alpha / rank)
///   gated LoRA: x @ W + sigmoid(gate) * ((x @ A) @ B) * (alpha / rank)
///   IA³:        (x @ W) * scale
fn forward(kind: Kind, x: &[f64], p: &[f64]) -> Vec<f64> {
    let w = &p[..N_IN * N_OUT];
    let rest = &p[N_IN * N_OUT..];
    let mut out = vec![0.0; ROWS * N_OUT];
    for r in 0..ROWS {
        for j in 0..N_OUT {
            let base: f64 = (0..N_IN).map(|k| x[N_IN * r + k] * w[N_OUT * k + j]).sum();
            out[N_OUT * r + j] = match kind {
                Kind::Ia3 => base * rest[j],
                Kind::Lora | Kind::GatedLora => {
                    let (a, b) = (&rest[..N_IN * RANK], &rest[N_IN * RANK..]);
                    let delta: f64 = (0..RANK)
                        .map(|q| {
                            let xa: f64 = (0..N_IN).map(|k| x[N_IN * r + k] * a[RANK * k + q]).sum();
                            xa * b[N_OUT * q + j]
                        })
                        .sum();
                    let gate = match kind {
                        Kind::GatedLora => 1.0 / (1.0 + (-b[RANK * N_OUT + j]).exp()),
                        _ => 1.0,
                    };
                    base + gate * delta * (ALPHA / RANK as f64)
                }
            };
        }
    }
    out
}

/// `mse_loss`: the mean squared error over every cell.
fn loss(kind: Kind, p: &[f64]) -> f64 {
    let (x, y) = (x_data(), y_data());
    let pred = forward(kind, &x, p);
    pred.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / pred.len() as f64
}

/// SGD from `p`, each gradient by central differences. With `frozen`, the
/// weight (the first `N_IN * N_OUT` entries) is not updated.
fn sgd_reference(kind: Kind, mut p: Vec<f64>, frozen: bool) -> Vec<f64> {
    for _ in 0..STEPS {
        let grad: Vec<f64> = (0..p.len())
            .map(|i| {
                let (mut up, mut down) = (p.clone(), p.clone());
                up[i] += 1e-6;
                down[i] -= 1e-6;
                (loss(kind, &up) - loss(kind, &down)) / 2e-6
            })
            .collect();
        for (i, (pi, gi)) in p.iter_mut().zip(&grad).enumerate() {
            if !(frozen && i < N_IN * N_OUT) {
                *pi -= LR * gi;
            }
        }
    }
    p
}

fn show(tag: &str, expr: &str) -> String {
    format!("let t_{tag}: Tensor = {expr}\nprint(\"{tag}_BEGIN\")\nprint(t_{tag})\nprint(\"{tag}_END\")\n")
}

fn program(kind: Kind, frozen: bool) -> String {
    let site = format!("Toy_w__{}", kind.name());
    let field = |tag: &str| -> String {
        match tag {
            "W" => "m.w".to_string(),
            "A" => format!("m.lora_A_{site}"),
            "B" => format!("m.lora_B_{site}"),
            "GATE" => format!("m.gate_{site}"),
            "SCALE" => format!("m.ia3_scale_{site}"),
            other => unreachable!("{other}"),
        }
    };
    let seed_b = if kind == Kind::Ia3 {
        String::new()
    } else {
        format!(
            "m.lora_B_{site} = (arange({n}).reshape([{RANK}, {N_OUT}])) * 0.05 - full([{RANK}, {N_OUT}], 0.2)\n",
            n = RANK * N_OUT
        )
    };
    let before: String =
        tensors(kind).iter().map(|(tag, _)| show(&format!("{tag}0"), &field(tag))).collect();
    let after: String = tensors(kind).iter().map(|(tag, _)| show(tag, &field(tag))).collect();
    format!(
        r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = randn([{N_IN}, {N_OUT}]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

{freeze}@adapter(type={kind}, target=["Toy.w"], rank={RANK}, alpha=4)
let m = Toy()
{seed_b}{before}
let x = (arange({nx}).reshape([2, 3, {N_IN}])) * 0.1 + full([2, 3, {N_IN}], 0.1)
let y = (arange({ny}).reshape([2, 3, {N_OUT}])) * 0.05 - full([2, 3, {N_OUT}], 1.0)

train(model = m, epochs = {STEPS}):
    optimizer: SGD(lr = {LR})
    step(batch):
        let loss = mse_loss(m.forward(x), y)

{after}"#,
        freeze = if frozen { "@freeze\n" } else { "" },
        kind = kind.name(),
        nx = ROWS * N_IN,
        ny = ROWS * N_OUT,
    )
}

fn check(kind: Kind, frozen: bool) {
    let src = program(kind, frozen);
    for path in [Path_::TapeAd, Path_::SourceAd, Path_::SourceAdFusedCall] {
        let label = format!("{} {}{path:?}", kind.name(), if frozen { "frozen base, " } else { "" });
        let r = run(&src, path);
        if path == Path_::SourceAdFusedCall {
            // One forward per step; without this the path would silently be
            // the ordinary one again.
            assert!(
                r.fused_calls >= STEPS as u64,
                "{label}: {} fused adapter calls ran, expected at least {STEPS}",
                r.fused_calls
            );
        } else {
            assert_eq!(r.fused_calls, 0, "{label}: the fused call is opt-in");
        }

        // The starting point, as the program printed it.
        let mut init: Vec<f64> = Vec::new();
        for (tag, len) in tensors(kind) {
            let got = section(&r.stdout, &format!("{tag}0"));
            assert_eq!(got.len(), len, "{label}: {tag} has {len} elements");
            init.extend(got);
        }
        if kind != Kind::Ia3 {
            let b = &init[N_IN * N_OUT + N_IN * RANK..][..RANK * N_OUT];
            for (got, want) in b.iter().zip(b_start()) {
                assert!((got - want).abs() < 1e-6, "{label}: B was seeded to {want}, reads {got}");
            }
        }
        let want = sgd_reference(kind, init.clone(), frozen);

        let mut offset = 0;
        for (tag, len) in tensors(kind) {
            let got = section(&r.stdout, tag);
            let (want, from) = (&want[offset..offset + len], &init[offset..offset + len]);
            offset += len;
            assert_eq!(got.len(), len, "{label}: {tag} has {len} elements");
            let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            let travel = want.iter().zip(from).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            eprintln!("{label} {tag}: max |Δ| from the f64 reference {worst:.2e}; it moves {travel:.3}");
            if frozen && tag == "W" {
                assert_eq!(got, from, "{label}: the frozen weight changed");
                continue;
            }
            // 500x the per-element tolerance below.
            assert!(travel > 5e-3, "{label} {tag}: the reference barely moves ({travel})");
            for (i, (a, b)) in got.iter().zip(want).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "{label} {tag}[{i}] = {a}, f64 reference {b} (|Δ|={:.2e})",
                    (a - b).abs()
                );
            }
        }
    }
}

#[test]
fn lora_gradients_match_the_reference_on_a_3d_activation() {
    check(Kind::Lora, false);
}

#[test]
fn lora_gradients_match_the_reference_on_a_3d_activation_with_a_frozen_base() {
    check(Kind::Lora, true);
}

#[test]
fn gated_lora_gradients_match_the_reference_on_a_3d_activation() {
    check(Kind::GatedLora, false);
}

#[test]
fn ia3_gradients_match_the_reference_on_a_3d_activation() {
    check(Kind::Ia3, false);
}

#[test]
fn ia3_gradients_match_the_reference_on_a_3d_activation_with_a_frozen_base() {
    check(Kind::Ia3, true);
}
