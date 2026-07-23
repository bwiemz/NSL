//! Numerical validation for FASE Deferred mode (CFTP item #2).
//!
//! This test compiles and runs a tiny NSL fixture with SGD + grad_accumulation=4,
//! reads the saved `.nslm` checkpoint, and compares the final `w` tensor against
//! an exact Rust SGD reference.

mod common {
    include!("common/mod.rs");
}

use common::nslm_reader::read_nslm;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn fixture(name: &str) -> PathBuf {
    fixture_path(name)
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

/// Workspace root: CARGO_MANIFEST_DIR/../..
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Run `nsl run <fixture>` with cwd set to `workdir`.
///
/// `cargo run` is invoked with `--manifest-path` pointing at the workspace
/// Cargo.toml so cargo can find the project from any cwd.  `workdir` is
/// passed as the cwd so that `model_save(m, "file.nslm")` lands there.
/// `NSL_STDLIB_PATH` is set to the workspace `stdlib/` so that imports like
/// `from nsl.nn.losses import mse_loss` resolve correctly.
///
/// Panics on non-zero exit.
fn nsl_run(fixture: &Path, workdir: &Path) {
    nsl_run_with_args(fixture, workdir, &[]);
}

/// Like `nsl_run` but appends `extra` CLI flags after `run <fixture>`.
///
/// Use this to pass flags such as `--source-ad` that exercise specific
/// compiler paths (e.g. the FASE consume-per-param hook in source-AD mode).
fn nsl_run_with_args(fixture: &Path, workdir: &Path, extra: &[&str]) {
    let root = workspace_root();
    let cargo_toml = root.join("Cargo.toml");
    let stdlib_path = root.join("stdlib");

    let mut cmd = std::process::Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(&cargo_toml)
        .args(["-p", "nsl-cli"]);
    if cfg!(feature = "cuda") {
        // Feature-compatibility guard: an unfeatured `cargo run -p nsl-cli`
        // REBUILDS AND REPLACES target/debug/nsl as a non-CUDA binary while
        // the workspace suite runs, breaking every concurrently-running test
        // that spawns that path (phantom "CUDA support not compiled"
        // failures inside their trained programs). Forward cuda so the
        // spawned build matches the suite's binary (and is usually a no-op).
        cmd.args(["--features", "cuda"]);
    }
    cmd.args(["--", "run"])
        .arg(fixture)
        .current_dir(workdir)
        .env("NSL_STDLIB_PATH", &stdlib_path);
    for a in extra {
        cmd.arg(a);
    }
    let status = cmd.status().expect("failed to spawn cargo run nsl-cli");

    assert!(
        status.success(),
        "nsl run failed on {:?} with extra args {:?}",
        fixture,
        extra
    );
}

/// Rust reference for plain SGD with grad_accumulation=4 and constant inputs
/// across all 4 micro-batches (no dataloader — step runs once per epoch, so
/// epochs=4 accumulates 4 micro-batches before the optimizer fires).
///
/// Fixture init values:
///   w = ones([2,1])  → [[1.0], [1.0]]
///   x = ones([4,2])  → all rows = [1.0, 1.0]
///   y = zeros([4,1]) → all targets = [0.0]
///
/// Forward:  pred = x @ w  → [[2.0], [2.0], [2.0], [2.0]]
/// Residual: r = pred - y  → [[2.0], [2.0], [2.0], [2.0]]
/// MSE grad: g = (2/N) * x.T @ r  where N=4
///           x.T @ r = [[8.0], [8.0]]  →  g = [[4.0], [4.0]]
/// SGD step: w_new = w - lr * g  → [[0.96], [0.96]]
///
/// Because all 4 micro-batches see the same (x, y), the FASE Deferred
/// mean(g_k) = g, so the accumulated result must exactly match a single
/// plain SGD step.
fn sgd_reference(
    w_init: &[f32; 2],
    x: &[[f32; 2]; 4],
    y: &[[f32; 1]; 4],
    lr: f32,
) -> [f32; 2] {
    // Forward: pred[i] = sum_j x[i][j] * w[j]
    let mut pred = [0.0_f32; 4];
    for i in 0..4 {
        for j in 0..2 {
            pred[i] += x[i][j] * w_init[j];
        }
    }

    // Residual: r[i] = pred[i] - y[i]
    let mut r = [0.0_f32; 4];
    for i in 0..4 {
        r[i] = pred[i] - y[i][0];
    }

    // MSE gradient wrt w: g[j] = (2/N) * sum_i x[i][j] * r[i]
    let n = 4.0_f32;
    let mut g = [0.0_f32; 2];
    for j in 0..2 {
        for i in 0..4 {
            g[j] += x[i][j] * r[i];
        }
        g[j] *= 2.0 / n;
    }

    // SGD parameter update
    [w_init[0] - lr * g[0], w_init[1] - lr * g[1]]
}

#[test]
fn sgd_exact_equivalence() {
    let tmp = TempDir::new().expect("tempdir");

    nsl_run(
        &fixture_path("fase_deferred_sgd_equivalence.nsl"),
        tmp.path(),
    );

    let checkpoint = tmp.path().join("sgd_out.nslm");
    assert!(
        checkpoint.exists(),
        "expected checkpoint at {:?} — model_save did not produce the file",
        checkpoint
    );

    let tensors = read_nslm(&checkpoint).expect("read nslm");

    let w_compiled = tensors
        .get("w")
        .or_else(|| tensors.get("m.w"))
        .unwrap_or_else(|| {
            panic!(
                "w tensor not in checkpoint; available keys: {:?}",
                tensors.keys().collect::<Vec<_>>()
            )
        });

    assert_eq!(
        w_compiled.len(),
        2,
        "w should have 2 elements (shape [2,1]), got {}",
        w_compiled.len()
    );

    // Reference inputs — must match the fixture exactly:
    //   w = ones([2,1]), x = ones([4,2]), y = zeros([4,1]), lr = 0.01
    let w_init = [1.0_f32, 1.0_f32];
    let x = [[1.0_f32, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
    let y = [[0.0_f32]; 4];
    let w_ref = sgd_reference(&w_init, &x, &y, 0.01);

    println!("SGD equivalence check:");
    println!("  compiled w[0]={} w[1]={}", w_compiled[0], w_compiled[1]);
    println!("  reference w[0]={} w[1]={}", w_ref[0], w_ref[1]);

    for i in 0..2 {
        let diff = (w_compiled[i] - w_ref[i]).abs();
        let scale = w_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "SGD θ[{}] diverged: compiled={} reference={} rel_err={}",
            i,
            w_compiled[i],
            w_ref[i],
            diff / scale
        );
    }
}

/// Rust reference for FASE-Deferred AdamW with W windows, N=4 micro-batches
/// each, constant inputs across every micro-batch.
///
/// Matches the `emit_adamw` recipe in `fase_optimizer.rs` exactly — note that
/// NSL's FASE AdamW does NOT apply Adam's standard bias-correction divisors
/// (1 - β^t).  The update rule is:
///
///   m_partial = mean(g_1..g_N) = g (constant inputs → constant g per window)
///   m  = β₁·m + (1-β₁)·m_partial
///   v  = β₂·v + (1-β₂)·m_partial²   (FASE approximation)
///   m_hat = m / (1 - β₁^t)
///   v_hat = v / (1 - β₂^t)
///   denom = sqrt(v_hat) + ε
///   tmp = m_hat / denom
///   θ -= lr · (tmp + wd·θ)
fn adamw_fase_deferred_reference(
    w_init: &[f32; 2],
    x: &[[f32; 2]; 4],
    y: &[[f32; 1]; 4],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    windows: u32,
) -> [f32; 2] {
    let mut w = *w_init;
    let mut m_state = [0.0_f32; 2];
    let mut v_state = [0.0_f32; 2];

    for step in 1..=windows {
        // Gradient with current w (constant across all 4 micro-batches in this
        // window, so m_partial = g).
        let mut pred = [0.0_f32; 4];
        for i in 0..4 {
            pred[i] = x[i][0] * w[0] + x[i][1] * w[1];
        }
        let mut r = [0.0_f32; 4];
        for i in 0..4 {
            r[i] = pred[i] - y[i][0];
        }
        let n = 4.0_f32;
        let mut g = [0.0_f32; 2];
        for j in 0..2 {
            for i in 0..4 {
                g[j] += x[i][j] * r[i];
            }
            g[j] *= 2.0 / n;
        }

        let m_partial = g;

        for j in 0..2 {
            // m = β₁·m + (1-β₁)·m_partial
            m_state[j] = beta1 * m_state[j] + (1.0 - beta1) * m_partial[j];
            // v = β₂·v + (1-β₂)·m_partial²  (exact standard windowed AdamW:
            // m_partial is the window-mean gradient)
            v_state[j] =
                beta2 * v_state[j] + (1.0 - beta2) * m_partial[j] * m_partial[j];
            // Bias correction
            let bc1 = 1.0 - beta1.powi(step as i32);
            let bc2 = 1.0 - beta2.powi(step as i32);
            let m_hat = m_state[j] / bc1;
            let v_hat = v_state[j] / bc2;
            // denom = sqrt(v_hat) + eps
            let denom = v_hat.sqrt() + eps;
            // tmp = m_hat / denom
            let tmp = m_hat / denom;
            // θ -= lr * (tmp + wd·θ)
            w[j] -= lr * (tmp + wd * w[j]);
        }
    }
    w
}

#[test]
fn adamw_fase_deferred_pipeline_equivalence() {
    let tmp = TempDir::new().expect("tempdir");
    nsl_run(
        &fixture("fase_deferred_adamw_equivalence.nsl"),
        tmp.path(),
    );

    let checkpoint = tmp.path().join("adamw_out.nslm");
    assert!(
        checkpoint.exists(),
        "expected checkpoint at {:?}",
        checkpoint
    );
    let tensors = read_nslm(&checkpoint).expect("read nslm");

    let w_compiled = tensors
        .get("w")
        .or_else(|| tensors.get("m.w"))
        .expect(&format!(
            "w tensor not in checkpoint; available: {:?}",
            tensors.keys().collect::<Vec<_>>()
        ));
    assert_eq!(w_compiled.len(), 2);

    // Match fixture init: ones([2,1]) → [1.0, 1.0]
    let w_init = [1.0_f32, 1.0_f32];
    // Match fixture data: ones([4,2]) and zeros([4,1]).
    let x = [[1.0, 1.0]; 4];
    let y = [[0.0]; 4];
    let w_ref = adamw_fase_deferred_reference(
        &w_init,
        &x,
        &y,
        /*lr=*/ 0.001,
        /*beta1=*/ 0.9,
        /*beta2=*/ 0.999,
        /*eps=*/ 1e-8,
        /*wd=*/ 0.01,
        /*windows=*/ 3,
    );

    println!("AdamW FASE-Deferred equivalence check:");
    println!("  compiled w[0]={} w[1]={}", w_compiled[0], w_compiled[1]);
    println!("  reference w[0]={} w[1]={}", w_ref[0], w_ref[1]);

    for i in 0..2 {
        let diff = (w_compiled[i] - w_ref[i]).abs();
        let scale = w_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "AdamW θ[{}] diverged: compiled={} reference={} rel_err={}",
            i,
            w_compiled[i],
            w_ref[i],
            diff / scale
        );
    }
}

/// Rust reference: FASE-Deferred AdamW with two-phase global-L2 grad clip.
/// Same fixture shape as `adamw_fase_deferred_reference`, plus clipping:
/// compute global L2 norm of m_partial across all parameters, scale by
/// clip_factor = min(1, τ / (norm + 1e-6)).
fn adamw_fase_deferred_clipped_reference(
    w_init: &[f32; 2],
    x: &[[f32; 2]; 4],
    y: &[[f32; 1]; 4],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    tau: f32,
    windows: u32,
) -> [f32; 2] {
    let mut w = *w_init;
    let mut m_state = [0.0_f32; 2];
    let mut v_state = [0.0_f32; 2];

    for step in 1..=windows {
        // Gradient (constant across window).
        let mut pred = [0.0_f32; 4];
        for i in 0..4 {
            pred[i] = x[i][0] * w[0] + x[i][1] * w[1];
        }
        let mut r = [0.0_f32; 4];
        for i in 0..4 {
            r[i] = pred[i] - y[i][0];
        }
        let n = 4.0_f32;
        let mut g = [0.0_f32; 2];
        for j in 0..2 {
            for i in 0..4 {
                g[j] += x[i][j] * r[i];
            }
            g[j] *= 2.0 / n;
        }

        // m_partial = mean(g) = g (constant inputs across micro-batches).
        let mut m_partial = g;

        // Phase A: global L2 norm of m_partial.
        let total_sq: f32 = m_partial.iter().map(|&v| v * v).sum();
        let norm = total_sq.sqrt();
        let clip_factor = 1.0_f32.min(tau / (norm + 1e-6));

        // Phase B: scale, then AdamW step with bias correction.
        for j in 0..2 {
            m_partial[j] *= clip_factor;
            m_state[j] = beta1 * m_state[j] + (1.0 - beta1) * m_partial[j];
            v_state[j] =
                beta2 * v_state[j] + (1.0 - beta2) * m_partial[j] * m_partial[j];
            let bc1 = 1.0 - beta1.powi(step as i32);
            let bc2 = 1.0 - beta2.powi(step as i32);
            let m_hat = m_state[j] / bc1;
            let v_hat = v_state[j] / bc2;
            w[j] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * w[j]);
        }
    }
    w
}

#[test]
fn adamw_deferred_with_grad_clip() {
    let tmp = TempDir::new().expect("tempdir");
    nsl_run(
        &fixture("fase_deferred_grad_accum_4_clipped.nsl"),
        tmp.path(),
    );

    let checkpoint = tmp.path().join("adamw_clipped_out.nslm");
    assert!(
        checkpoint.exists(),
        "expected checkpoint at {:?}",
        checkpoint
    );
    let tensors = read_nslm(&checkpoint).expect("read nslm");

    let w_compiled = tensors
        .get("w")
        .or_else(|| tensors.get("m.w"))
        .expect(&format!(
            "w tensor not in checkpoint; available: {:?}",
            tensors.keys().collect::<Vec<_>>()
        ));
    assert_eq!(w_compiled.len(), 2);

    let w_init = [1.0_f32, 1.0_f32];
    let x = [[1.0, 1.0]; 4];
    let y = [[0.0]; 4];
    let w_ref = adamw_fase_deferred_clipped_reference(
        &w_init,
        &x,
        &y,
        /*lr=*/ 0.001,
        /*beta1=*/ 0.9,
        /*beta2=*/ 0.999,
        /*eps=*/ 1e-8,
        /*wd=*/ 0.01,
        /*tau=*/ 0.01,
        /*windows=*/ 3,
    );

    println!("AdamW+clip FASE-Deferred equivalence check:");
    println!("  compiled w[0]={} w[1]={}", w_compiled[0], w_compiled[1]);
    println!("  reference w[0]={} w[1]={}", w_ref[0], w_ref[1]);

    for i in 0..2 {
        let diff = (w_compiled[i] - w_ref[i]).abs();
        let scale = w_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "AdamW+clip θ[{}] diverged: compiled={} reference={} rel_err={}",
            i,
            w_compiled[i],
            w_ref[i],
            diff / scale
        );
    }
}

#[test]
fn adamw_fase_deferred_source_ad_pipeline_equivalence() {
    // Item #4: exercises the consume-per-param hook.  Source-AD + FASE
    // Deferred invokes the callback per parameter gradient during
    // compile_wengert_ops.  Final parameter values must match the same
    // reference as the tape-AD variant.
    let tmp = TempDir::new().expect("tempdir");
    nsl_run_with_args(
        &fixture("fase_deferred_adamw_source_ad.nsl"),
        tmp.path(),
        &["--source-ad"],
    );

    let checkpoint = tmp.path().join("adamw_source_ad_out.nslm");
    assert!(
        checkpoint.exists(),
        "expected checkpoint at {:?}",
        checkpoint
    );
    let tensors = read_nslm(&checkpoint).expect("read nslm");
    let w_compiled = tensors
        .get("w")
        .or_else(|| tensors.get("m.w"))
        .expect(&format!(
            "w tensor not in checkpoint; available: {:?}",
            tensors.keys().collect::<Vec<_>>()
        ));
    assert_eq!(w_compiled.len(), 2);

    let w_init = [1.0_f32, 1.0_f32];
    let x = [[1.0, 1.0]; 4];
    let y = [[0.0]; 4];
    let w_ref = adamw_fase_deferred_reference(
        &w_init,
        &x,
        &y,
        /*lr=*/ 0.001,
        /*beta1=*/ 0.9,
        /*beta2=*/ 0.999,
        /*eps=*/ 1e-8,
        /*wd=*/ 0.01,
        /*windows=*/ 3,
    );

    println!("AdamW FASE-Deferred (source-AD) equivalence check:");
    println!("  compiled w[0]={} w[1]={}", w_compiled[0], w_compiled[1]);
    println!("  reference w[0]={} w[1]={}", w_ref[0], w_ref[1]);

    for i in 0..2 {
        let diff = (w_compiled[i] - w_ref[i]).abs();
        let scale = w_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "AdamW (source-AD) θ[{}] diverged: compiled={} reference={} rel_err={}",
            i,
            w_compiled[i],
            w_ref[i],
            diff / scale
        );
    }
}

// ─── v_t semantics discrimination (roadmap 2.1, rescoped) ───────────────────

/// Dual-reference AdamW for the `fase_vt_discrimination.nsl` fixture, where
/// micro-batch k (globally 0-based) sees `x = 1.25^k * ones(4,2)`, `y = 0`,
/// and params freeze within each accumulation window (Deferred semantics).
///
/// Returns `(theta_standard, theta_option_b)`:
///   standard:  v = β₂·v + (1-β₂)·(mean g)²   — exact windowed AdamW
///   option_b:  v = β₂·v + (1-β₂)·mean(g²)    — the CFTP §2.3 approximation
/// `m` uses the window-mean gradient in both.
#[allow(clippy::too_many_arguments)]
fn adamw_vt_dual_references(
    w_init: &[f32; 2],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    windows: u32,
    accum: u32,
    x_growth: f32,
) -> ([f32; 2], [f32; 2]) {
    // Gradient of mean((c·(w0+w1))²) over 4 rows w.r.t. each w_j is
    // 2·c²·(w0+w1) (identical for both components at this fixture's shapes).
    let run = |option_b: bool| -> [f32; 2] {
        let mut w = *w_init;
        let mut m = [0.0_f32; 2];
        let mut v = [0.0_f32; 2];
        let mut c = 1.0_f32;
        for step in 1..=windows {
            let mut mean_g = [0.0_f32; 2];
            let mut mean_g2 = [0.0_f32; 2];
            for _k in 0..accum {
                let s = w[0] + w[1]; // params frozen within the window
                for j in 0..2 {
                    let g = 2.0 * c * c * s;
                    mean_g[j] += g / accum as f32;
                    mean_g2[j] += (g * g) / accum as f32;
                }
                c *= x_growth;
            }
            for j in 0..2 {
                m[j] = beta1 * m[j] + (1.0 - beta1) * mean_g[j];
                let v_inc = if option_b {
                    mean_g2[j]
                } else {
                    mean_g[j] * mean_g[j]
                };
                v[j] = beta2 * v[j] + (1.0 - beta2) * v_inc;
                let bc1 = 1.0 - beta1.powi(step as i32);
                let bc2 = 1.0 - beta2.powi(step as i32);
                let m_hat = m[j] / bc1;
                let v_hat = v[j] / bc2;
                w[j] -= lr * (m_hat / (v_hat.sqrt() + eps));
            }
        }
        w
    };
    (run(false), run(true))
}

/// Pins WHICH second-moment formula the compiled FASE Deferred path emits.
///
/// Every other fixture uses constant micro-batches, where (mean g)² ==
/// mean(g²) and the two candidate formulas are indistinguishable — the
/// docs/pseudocode claimed the CFTP "Option B" approximation while the
/// emitted `SquaredAccumulate` over `MPartial` computes the EXACT standard
/// windowed AdamW. With intra-window gradient variation this test separates
/// the trajectories (Jensen strict inequality) and asserts the compiled
/// result matches the standard formula and NOT Option B.
#[test]
fn adamw_deferred_vt_is_exact_windowed_not_option_b() {
    let tmp = TempDir::new().expect("tempdir");
    nsl_run(&fixture("fase_vt_discrimination.nsl"), tmp.path());

    let checkpoint = tmp.path().join("vt_disc_out.nslm");
    assert!(checkpoint.exists(), "expected checkpoint at {:?}", checkpoint);
    let tensors = read_nslm(&checkpoint).expect("read nslm");
    let w_compiled = tensors
        .get("w")
        .or_else(|| tensors.get("m.w"))
        .expect("w tensor not in checkpoint");
    assert_eq!(w_compiled.len(), 2);

    let (w_std, w_ob) = adamw_vt_dual_references(
        &[1.0, 1.0],
        0.001,
        0.9,
        0.999,
        1e-8,
        /*windows=*/ 2,
        /*accum=*/ 4,
        /*x_growth=*/ 1.25,
    );

    // Vacuity guard: if the fixture's x-mutation didn't take effect, the
    // gradients are constant per window and the references coincide — the
    // test would prove nothing. Fail loudly instead of passing vacuously.
    let ref_sep = (w_std[0] - w_ob[0]).abs().max((w_std[1] - w_ob[1]).abs());
    assert!(
        ref_sep > 1e-6,
        "reference trajectories did not separate (sep={ref_sep:.3e}) — the \
         fixture's per-micro-batch gradient variation is not reaching the \
         step body; discrimination is vacuous"
    );

    let d_std = (w_compiled[0] - w_std[0]).abs().max((w_compiled[1] - w_std[1]).abs());
    let d_ob = (w_compiled[0] - w_ob[0]).abs().max((w_compiled[1] - w_ob[1]).abs());
    assert!(
        d_std < 1e-5 && d_ob > 10.0 * d_std.max(1e-7),
        "compiled θ does not discriminate to the standard formula: \
         |θ-standard|={d_std:.3e}, |θ-option_b|={d_ob:.3e}, ref_sep={ref_sep:.3e} \
         (compiled={w_compiled:?}, standard={w_std:?}, option_b={w_ob:?})"
    );
}
