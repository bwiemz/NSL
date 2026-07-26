//! Item 7 (`--fuse-wgrad-accum`) end-to-end gate.
//!
//! The flag collapses source-AD's weight-gradient chain
//! `Transpose -> Matmul -> reduce_to_shape` AND the FASE-Deferred accumulate
//! into one cuBLAS call writing straight into `m_partial`. That deletes two
//! tape ops and leaves their results UNMAPPED, which is the risky part: the
//! P0.2 grad-integrity guard treats an unresolved input on a live adjoint op
//! as a compile error precisely because the pre-#396 behaviour (skip it) would
//! zero a real gradient. This gate exists to prove the elision produces the
//! same trained weights rather than a quietly missing one.
//!
//! Three layers, because each catches something the others cannot:
//!
//! * **CPU parity** (default features) — the runtime's precondition check
//!   rejects CPU tensors and falls back to the exact decomposed chain, so the
//!   two arms must come out BIT-IDENTICAL. This isolates the *codegen*: if the
//!   plan mis-identifies a chain or rewires the operands wrongly, the numbers
//!   move even though the arithmetic is unchanged.
//! * **GPU parity** (`#[ignore]`, `--features cuda`) — exercises the actual
//!   fused GEMM, which is NOT bit-exact (cuBLAS sums the `B*T` products in its
//!   own order instead of rounding each per-batch partial first). Held to the
//!   repo's f32 training tolerance.
//! * **Refusals** — the three compositions that would be silently wrong.
//!
//! Every parity assertion is paired with a non-vacuity check on the
//! `[wgrad-fusion] N chain(s) fused` line. Without it, a tape change that
//! stops matching the pattern turns this gate green while testing nothing:
//! flag-on would simply BE flag-off.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Two weight shapes on purpose, because the fusion's whole point is the
/// BATCHED case:
///
/// * `w1` is fed a 3-D `[B, T, d]` activation, so the weight adjoint builds a
///   `[B, d, o]` raw gradient — B times the parameter — that `reduce_to_shape`
///   then sums away. That temporary is what the fused GEMM removes.
/// * `w2` is fed the 2-D result, where the reduce is an identity. The fusion
///   still applies (N = T rather than B*T); covering it keeps the identity
///   path from regressing unnoticed.
///
/// `grad_accumulation=4` + AdamW + `--source-ad` is what selects the
/// FASE-Deferred path, which owns the accumulate the fusion folds into.
/// Deterministic `arange` init (not `randn`) so the two arms are comparable
/// rather than merely similar.
/// The ACTIVATIONS have to be device-resident, not just the model. `m.to(cuda)`
/// alone leaves the inputs on the host, the whole forward runs there, and the
/// fused GEMM's `x.device == m.device` precondition fails on every call — the
/// runtime then falls back to the exact decomposed chain and the GPU arm
/// silently becomes a second CPU arm. (That is not hypothetical: this gate was
/// green that way until the `[wgrad-accum]` counter exposed 16 of 16 calls
/// falling back.)
fn fixture_src(cuda: bool) -> String {
    let (to_device, xs, ys) = if cuda {
        ("m.to(cuda)\n", "x.to(cuda)", "y.to(cuda)")
    } else {
        ("", "x", "y")
    };
    format!(
        r#"from nsl.nn.losses import mse_loss

model Net:
    w1: Tensor = (arange(24).reshape([4, 6])) * 0.02
    w2: Tensor = (arange(18).reshape([6, 3])) * 0.03

    fn forward(self, x: Tensor) -> Tensor:
        let h = gelu(x @ self.w1)
        return h @ self.w2

let m = Net()
{to_device}let x = (arange(40).reshape([2, 5, 4])) * 0.01
let y = (arange(30).reshape([2, 5, 3])) * 0.01
let xd = {xs}
let yd = {ys}

train(model=m, epochs=8, grad_accumulation=4):
    optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    step(batch):
        let pred = m.forward(xd)
        let loss = mse_loss(pred, yd)

print("W1_BEGIN")
print(m.w1)
print("W1_END")
print("W2_BEGIN")
print(m.w2)
print("W2_END")
"#
    )
}

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
}

fn run(fixture: &Path, cuda: bool, extra: &[&str]) -> Run {
    let root = repo_root();
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli"]);
    if cuda {
        cmd.args(["--features", "cuda"]);
    }
    cmd.args(["--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["--", "run", "--source-ad"]);
    if cuda {
        // Bit-reproducible GPU kernels, so the only difference between the
        // arms is the fusion itself and not nondeterministic reduction order.
        cmd.args(["--deterministic", "--target", "sm_89"]);
    }
    cmd.args(extra)
        .arg(fixture)
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // Which path the RUNTIME took. The codegen line below only proves the
        // call was emitted; this proves the GEMM ran instead of the silent
        // decomposed fallback. Harmless in both arms.
        .env("NSL_WGRAD_COUNTER", "1");
    let out = cmd.output().expect("spawn nsl run");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        ok: out.status.success(),
    }
}

/// Total chains fused, summed over every adjoint lowering in the compile.
/// `None` when the flag was off (the line is only emitted when it is on).
fn fused_chains(stderr: &str) -> Option<u64> {
    let mut total = None;
    for line in stderr.lines() {
        if let Some(rest) = line.trim().strip_prefix("[wgrad-fusion] ") {
            if let Some(n) = rest
                .split_once(" chain(s)")
                .and_then(|(n, _)| n.trim().parse::<u64>().ok())
            {
                total = Some(total.unwrap_or(0) + n);
            }
        }
    }
    total
}

/// `(fused GEMM calls, decomposed fallback calls)` from the
/// `NSL_WGRAD_COUNTER=1` atexit report.
fn wgrad_runtime_counts(stderr: &str) -> Option<(u64, u64)> {
    let rest = stderr
        .lines()
        .find_map(|l| l.trim().strip_prefix("[wgrad-accum] fused GEMM: "))?;
    let (fused, tail) = rest.split_once(", decomposed fallback: ")?;
    Some((fused.trim().parse().ok()?, tail.trim().parse().ok()?))
}

fn parse_between(stdout: &str, begin: &str, end: &str) -> Vec<f64> {
    let after = stdout.split_once(begin).map(|(_, r)| r).unwrap_or("");
    let inner = after.split_once(end).map(|(l, _)| l).unwrap_or("");
    inner
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

fn weights(stdout: &str) -> (Vec<f64>, Vec<f64>) {
    (
        parse_between(stdout, "W1_BEGIN", "W1_END"),
        parse_between(stdout, "W2_BEGIN", "W2_END"),
    )
}

/// The trained weights must have MOVED off their `arange` init. Without this
/// a gate comparing two all-zero-gradient runs would pass while proving
/// nothing about gradients at all.
fn assert_weights_trained(w1: &[f64], w2: &[f64]) {
    assert_eq!(w1.len(), 24, "expected 24 w1 values, got {}", w1.len());
    assert_eq!(w2.len(), 18, "expected 18 w2 values, got {}", w2.len());
    // w1[0] and w2[0] both init to exactly 0.0.
    assert!(
        w1[0].abs() > 1e-4,
        "w1[0] never moved off its 0.0 init ({}) — no gradient flowed, so a \
         parity comparison here would be vacuous",
        w1[0]
    );
    assert!(
        w2[0].abs() > 1e-4,
        "w2[0] never moved off its 0.0 init ({})",
        w2[0]
    );
}

fn write_fixture(name: &str, cuda: bool) -> PathBuf {
    let p = std::env::temp_dir().join(name);
    std::fs::write(&p, fixture_src(cuda)).expect("write fixture");
    p
}

#[test]
fn cpu_fused_wgrad_is_bit_identical_to_the_unfused_chain() {
    let f = write_fixture("nsl_wgrad_accum_gate_cpu.nsl", false);

    let off = run(&f, false, &[]);
    assert!(off.ok, "flag-off run failed:\n{}", off.stderr);
    let on = run(&f, false, &["--fuse-wgrad-accum"]);
    assert!(on.ok, "flag-on run failed:\n{}", on.stderr);

    // NON-VACUITY: the fusion must actually have fired. Two trainable weights
    // ⇒ two chains in the adjoint tape that carries them.
    let fired = fused_chains(&on.stderr).unwrap_or(0);
    assert!(
        fired >= 2,
        "expected >= 2 fused chains (one per trainable weight), got {fired}. \
         Without this the parity below is vacuous — flag-on would simply be \
         flag-off.\nstderr:\n{}",
        on.stderr
    );
    assert!(
        fused_chains(&off.stderr).is_none(),
        "the fusion reported chains with the flag OFF — it must be opt-in"
    );

    // Mirror image of the GPU check: on CPU every call must take the
    // FALLBACK. This is what licenses the bit-identical assertion below —
    // and if a future change ever made the GEMM path CPU-reachable, that
    // assertion would start failing for a reason nobody could explain
    // without this line.
    let (gemm, fallback) = wgrad_runtime_counts(&on.stderr).unwrap_or_else(|| {
        panic!("no [wgrad-accum] counter line\nstderr:\n{}", on.stderr)
    });
    assert_eq!(gemm, 0, "the cuBLAS path fired in a non-cuda build");
    assert!(
        fallback >= 2,
        "expected the decomposed fallback to run at least once per weight, \
         got {fallback}"
    );

    let (on_w1, on_w2) = weights(&on.stdout);
    let (off_w1, off_w2) = weights(&off.stdout);
    assert_weights_trained(&on_w1, &on_w2);

    // On CPU the runtime rejects the GEMM preconditions and falls back to the
    // exact decomposed chain, so the arithmetic is unchanged and any drift is
    // a CODEGEN defect (mis-planned chain, mis-wired operand, dropped grad).
    assert_eq!(
        on_w1, off_w1,
        "w1 differs on CPU, where the fused path falls back to the exact \
         decomposed chain — this is a codegen defect, not a numerics one"
    );
    assert_eq!(on_w2, off_w2, "w2 differs on CPU (see above)");
}

#[test]
fn fuse_wgrad_accum_requires_source_ad() {
    // ANTI-VACUITY for the refusals below: prove the CLI rejects things at
    // all in this position, and specifically that the flag cannot be used on
    // the tape-AD path (which has no FASE accumulate hook to fold into).
    let f = write_fixture("nsl_wgrad_accum_gate_nosad.nsl", false);
    let root = repo_root();
    let out = Command::new(env!("CARGO"))
        .args([
            "run",
            "-q",
            "-p",
            "nsl-cli",
            "--manifest-path",
        ])
        .arg(root.join("Cargo.toml"))
        .args(["--", "run", "--fuse-wgrad-accum"])
        .arg(&f)
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(
        !out.status.success(),
        "--fuse-wgrad-accum without --source-ad must be refused"
    );
}

/// The three compositions that would be silently wrong rather than merely
/// slower. Each is refused by clap at parse time, so the check is cheap and
/// needs no device.
#[test]
fn incompatible_compositions_are_refused() {
    let f = write_fixture("nsl_wgrad_accum_gate_conflict.nsl", false);
    let root = repo_root();
    // (flag, why it cannot compose)
    let cases: [(&[&str], &str); 3] = [
        (
            &["--grad-integrity"],
            "must read the raw gradient the fusion never materializes",
        ),
        (
            &["--optim-state-offload"],
            "m_partial is host-resident; the device GEMM cannot write it",
        ),
        (
            &["--layerwise-accum"],
            "CSLA lowers pre-sliced tapes, defeating the single-reader proof",
        ),
    ];
    for (extra, why) in cases {
        let mut cmd = Command::new(env!("CARGO"));
        cmd.args(["run", "-q", "-p", "nsl-cli", "--manifest-path"])
            .arg(root.join("Cargo.toml"))
            .args(["--", "run", "--source-ad", "--fuse-wgrad-accum"])
            .args(extra)
            .arg(&f)
            .current_dir(&root)
            .env("NSL_STDLIB_PATH", root.join("stdlib"));
        let out = cmd.output().expect("spawn nsl run");
        assert!(
            !out.status.success(),
            "--fuse-wgrad-accum {extra:?} must be refused: {why}"
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU (2 FASE training runs)"]
fn gpu_fused_wgrad_matches_the_unfused_chain_within_f32_tolerance() {
    let f = write_fixture("nsl_wgrad_accum_gate_gpu.nsl", true);

    let off = run(&f, true, &[]);
    assert!(off.ok, "flag-off GPU run failed:\n{}", off.stderr);
    let on = run(&f, true, &["--fuse-wgrad-accum"]);
    assert!(on.ok, "flag-on GPU run failed:\n{}", on.stderr);

    let fired = fused_chains(&on.stderr).unwrap_or(0);
    assert!(
        fired >= 2,
        "expected >= 2 fused chains, got {fired}\nstderr:\n{}",
        on.stderr
    );
    // NON-VACUITY, the part that actually matters on GPU: the codegen line
    // above only proves the CALL was emitted. `nsl_tensor_wgrad_accum` falls
    // back to the exact decomposed chain — silently — whenever its
    // preconditions do not hold, and every such fallback would make this
    // tolerance comparison a test of the chain against itself.
    let (gemm, fallback) = wgrad_runtime_counts(&on.stderr).unwrap_or_else(|| {
        panic!(
            "no [wgrad-accum] counter line; cannot tell the fused GEMM from \
             the fallback\nstderr:\n{}",
            on.stderr
        )
    });
    assert!(
        gemm > 0,
        "every wgrad call took the DECOMPOSED FALLBACK (fused={gemm}, \
         fallback={fallback}) — this comparison would be vacuous. The GEMM \
         preconditions (contiguous, GPU-resident f32, matching leading dims) \
         are not being met on this shape."
    );
    assert_eq!(
        fallback, 0,
        "{fallback} of {} wgrad calls silently fell back; the arms are then \
         only partly comparable",
        gemm + fallback
    );

    let (on_w1, on_w2) = weights(&on.stdout);
    let (off_w1, off_w2) = weights(&off.stdout);
    assert_weights_trained(&on_w1, &on_w2);
    assert_weights_trained(&off_w1, &off_w2);

    // NOT bit-exact by construction: the fused GEMM sums the B*T products in
    // cuBLAS's order rather than rounding each per-batch partial before the
    // reduce. Measured against an f64 reference the fused form is the more
    // accurate of the two (nsl-runtime/tests/wgrad_accum_fused.rs), but the
    // contract here is only agreement to the repo's f32 training tolerance.
    const TOL: f64 = 2e-3;
    for (name, (a, b)) in [("w1", (&on_w1, &off_w1)), ("w2", (&on_w2, &off_w2))] {
        assert_eq!(a.len(), b.len(), "{name}: arm length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < TOL,
                "{name}[{i}] fused={x} unfused={y} (|Δ|={} >= {TOL})",
                (x - y).abs()
            );
        }
    }
}
