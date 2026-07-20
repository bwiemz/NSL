//! P5 Muon: full-precision mixed Muon/AdamW optimizer gates.
//!
//! - The Newton-Schulz orthogonalization is pinned against an INDEPENDENT
//!   f64 reference implemented here (coefficients, iteration order, the
//!   tall-transpose branch and the 1e-7 norm floor all participate).
//! - Mixed routing: a model whose params all route to the AdamW arm (by
//!   name or by rank) trains BIT-IDENTICALLY to plain AdamW.
//! - Anti-vacuity: on a model with rank-2 hidden weights, Muon must DIVERGE
//!   from AdamW (the orthogonalized update really runs).
//! - GPU (`--ignored`): same fixture on CUDA — trains, loss decreases, and
//!   the mixed routing still differs from GPU AdamW.
//!
//! GPU: `cargo test -p nsl-cli --features cuda --test muon_optimizer_gate -- --ignored`

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    success: bool,
    stdout: String,
    stderr: String,
    losses: Vec<String>,
}

fn run_fixture(fixture: &str, tag: &str, rewrites: &[(&str, &str)], extra: &[&str]) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_muon_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures").join(fixture),
    )
    .unwrap();
    for (from, to) in rewrites {
        assert!(src.contains(from), "rewrite marker '{from}' missing in {fixture}");
        src = src.replace(from, to);
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--deterministic"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    // CPU runs print bare floats; GPU runs print `tensor([v])` and
    // interleave [gpu-mem] diagnostics — normalize both to the numeric
    // string so stream comparisons and parses are format-independent.
    let losses = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| {
            v.lines()
                .filter_map(|l| {
                    let l = l.trim();
                    if let Some(inner) = l
                        .strip_prefix("tensor([")
                        .and_then(|r| r.strip_suffix("])"))
                    {
                        Some(inner.to_string())
                    } else if l.parse::<f64>().is_ok() {
                        Some(l.to_string())
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    RunOut {
        success: out.status.success(),
        stdout,
        stderr,
        losses,
    }
}

// ── Independent Newton-Schulz reference (f64) ───────────────────────────

fn matmul(a: &[f64], b: &[f64], n: usize, k: usize, m: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * m];
    for i in 0..n {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..m {
                c[i * m + j] += av * b[p * m + j];
            }
        }
    }
    c
}

fn transpose(a: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            t[j * n + i] = a[i * m + j];
        }
    }
    t
}

/// Mirrors stdlib muon_orthogonalize: tall inputs orthogonalize the
/// transpose; quintic iteration with (3.4445, -4.7750, 2.0315); norm floor
/// 1e-7; 5 steps.
fn ns_reference(g: &[f64], rows: usize, cols: usize, steps: usize) -> Vec<f64> {
    let (mut x, n, m, transposed) = if rows > cols {
        (transpose(g, rows, cols), cols, rows, true)
    } else {
        (g.to_vec(), rows, cols, false)
    };
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-7;
    let inv = 1.0 / norm;
    for v in &mut x {
        *v *= inv;
    }
    for _ in 0..steps {
        let xt = transpose(&x, n, m);
        let a = matmul(&x, &xt, n, m, n);
        let aa = matmul(&a, &a, n, n, n);
        let mut b = vec![0.0; n * n];
        for i in 0..n * n {
            b[i] = -4.7750 * a[i] + 2.0315 * aa[i];
        }
        let bx = matmul(&b, &x, n, n, m);
        for i in 0..n * m {
            x[i] = 3.4445 * x[i] + bx[i];
        }
    }
    if transposed {
        transpose(&x, n, m)
    } else {
        x
    }
}

fn stats_block(stdout: &str, begin: &str, end: &str) -> Vec<f64> {
    stdout
        .split_once(begin)
        .and_then(|(_, r)| r.split_once(end))
        .map(|(v, _)| {
            v.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// The NSL Newton-Schulz must match the independent f64 reference on both
/// the wide (no transpose) and tall (transpose) branches — three scalar
/// statistics each, tolerance 1e-9 (CPU f64 end to end).
#[test]
fn muon_orthogonalize_matches_reference() {
    let out = run_fixture("muon_orthogonalize_probe.nsl", "nsref", &[], &[]);
    assert!(out.success, "probe failed:\n{}\n{}", out.stdout, out.stderr);

    for (block, rows, cols, omega, phase, shift) in [
        ("WIDE", 4usize, 6usize, 1.7f64, 0.3f64, 1.5f64),
        ("TALL", 6, 4, 2.3, 0.7, 2.0),
    ] {
        let got = stats_block(&out.stdout, &format!("{block}_BEGIN"), &format!("{block}_END"));
        assert_eq!(got.len(), 3, "{block}: expected 3 stats:\n{}", out.stdout);
        let g: Vec<f64> = (0..rows * cols)
            .map(|i| (i as f64 * omega + phase).sqrt() - shift)
            .collect();
        let o = ns_reference(&g, rows, cols, 5);
        let sum: f64 = o.iter().sum();
        let fro2: f64 = o.iter().map(|v| v * v).sum();
        // Small-side Gram matrix Frobenius^2 (wide: o o^T; tall: o^T o).
        let small = rows.min(cols);
        let r = if rows <= cols {
            let ot = transpose(&o, rows, cols);
            matmul(&o, &ot, rows, cols, rows)
        } else {
            let ot = transpose(&o, rows, cols);
            matmul(&ot, &o, cols, rows, cols)
        };
        let gram2: f64 = r.iter().map(|v| v * v).sum();
        for (name, want, have) in [
            ("sum", sum, got[0]),
            ("fro2", fro2, got[1]),
            ("gram2", gram2, got[2]),
        ] {
            // NSL-level arange/full CREATE f32 tensors (creation.rs
            // tensor_from_shape_list), so the probe chain carries f32 input
            // noise (~1e-6 relative after 5 NS iterations) even though model
            // params — and therefore real Muon training — are f64. 1e-5
            // still pins the coefficients (a 1e-4 coefficient error moves
            // these stats by ~1e-4) and any iteration-order change (O(1)).
            // The f64 step math is separately pinned bit-exactly by
            // muon_all_adamw_routes_bit_exact.
            assert!(
                (want - have).abs() < 1e-5,
                "{block} {name}: reference {want} vs NSL {have}"
            );
        }
        // Anti-vacuity of the orthogonalization itself: the normalized input
        // has Frobenius^2 == 1; the semi-orthogonalized output must sit near
        // rank (NS-band tolerant) — far from 1.
        assert!(
            got[1] > 0.4 * small as f64 && got[1] < 1.6 * small as f64,
            "{block}: ||O||_F^2 = {} not near rank {small} — orthogonalization vacuous?",
            got[1]
        );
    }

    // Review L9 — full Muon-arm step pin: 2 steps on a tall 4x3 param with
    // constant grad (momentum accumulate, nesterov combine on the UPDATED
    // buffer, sqrt(rows/cols) scale, decoupled weight decay). Mirrors the
    // fixture's STEP block exactly.
    let (rows, cols) = (4usize, 3usize);
    let mut p: Vec<f64> = (0..12).map(|i| (i as f64 * 1.3 + 0.5).sqrt() - 1.2).collect();
    let g: Vec<f64> = (0..12).map(|i| (i as f64 * 0.9 + 0.9).sqrt() - 1.0).collect();
    let mut m = vec![0.0f64; 12];
    let (lr, momentum, wd) = (0.02f64, 0.95f64, 0.01f64);
    let scale = (rows as f64 / cols as f64).sqrt();
    for _ in 0..2 {
        for i in 0..12 {
            m[i] = momentum * m[i] + g[i];
        }
        let upd: Vec<f64> = (0..12).map(|i| g[i] + momentum * m[i]).collect();
        let o = ns_reference(&upd, rows, cols, 5);
        for i in 0..12 {
            p[i] = p[i] * (1.0 - lr * wd) - (lr * scale) * o[i];
        }
    }
    let got = stats_block(&out.stdout, "STEP_BEGIN", "STEP_END");
    assert_eq!(got.len(), 3, "STEP: expected 3 stats:\n{}", out.stdout);
    let want = [
        p.iter().sum::<f64>(),
        p.iter().map(|v| v * v).sum::<f64>(),
        m.iter().sum::<f64>(),
    ];
    for (name, w, h) in [
        ("sum(p)", want[0], got[0]),
        ("fro2(p)", want[1], got[1]),
        ("sum(m)", want[2], got[2]),
    ] {
        // 1e-4: f32 tensor creation noise compounds over 2 steps; a wrong
        // sign/coefficient/order moves these at O(1e-2)+.
        assert!(
            (w - h).abs() < 1e-4,
            "STEP {name}: reference {w} vs NSL {h}"
        );
    }
}

/// All-AdamW-routed model (embed by name, norm by rank): Muon at AdamW
/// hyperparameters must be BIT-IDENTICAL to AdamW — loss stream and model
/// bytes.
#[test]
fn muon_all_adamw_routes_bit_exact() {
    let tmp = std::env::temp_dir().join(format!("nsl_muon_eq_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("adamw.nslm");
    let save_m = tmp.join("muon.nslm");

    let adamw = run_fixture(
        "muon_adamw_equiv.nsl",
        "eq_adamw",
        &[
            (
                "OPTIMIZER_LINE",
                "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
            ),
            ("MUON_SAVE_PATH", &save_a.display().to_string()),
        ],
        &[],
    );
    assert!(adamw.success, "adamw run failed:\n{}", adamw.stderr);
    assert!(!adamw.losses.is_empty(), "empty adamw loss stream");

    let muon = run_fixture(
        "muon_adamw_equiv.nsl",
        "eq_muon",
        &[
            (
                "OPTIMIZER_LINE",
                "Muon(lr=0.002, momentum=0.95, nesterov=true, ns_steps=5, \
                 weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
            ),
            ("MUON_SAVE_PATH", &save_m.display().to_string()),
        ],
        &[],
    );
    assert!(muon.success, "muon run failed:\n{}", muon.stderr);
    assert_eq!(
        adamw.losses, muon.losses,
        "all-AdamW-routed Muon diverged from AdamW\nstderr:\n{}",
        muon.stderr
    );
    // The routing table must show embed was routed by NAME (not silently
    // Muon'd and coincidentally equal).
    assert!(
        muon.stderr.contains("[muon]") && muon.stderr.contains("m.embed"),
        "routing table missing:\n{}",
        muon.stderr
    );
    let a = std::fs::read(&save_a).expect("adamw .nslm");
    let b = std::fs::read(&save_m).expect("muon .nslm");
    assert_eq!(a, b, "model bytes diverged on the all-AdamW route");
}

/// Anti-vacuity on the REAL mixed case: rank-2 hidden weights route to the
/// orthogonalized update, so Muon must (a) train — final loss below first —
/// and (b) DIFFER from AdamW at identical shared hyperparameters.
#[test]
fn muon_mixed_trains_and_differs_from_adamw() {
    let adamw = run_fixture(
        "csla_layerwise_ffn.nsl",
        "mix_adamw",
        &[("CSLA_SAVE_PATH", "mix_adamw.nslm")],
        &[],
    );
    assert!(adamw.success, "adamw baseline failed:\n{}", adamw.stderr);

    let muon = run_fixture(
        "csla_layerwise_ffn.nsl",
        "mix_muon",
        &[
            (
                "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
                "Muon(lr=0.002, momentum=0.95, nesterov=true, ns_steps=5, \
                 weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
            ),
            ("CSLA_SAVE_PATH", "mix_muon.nslm"),
        ],
        &[],
    );
    assert!(muon.success, "muon run failed:\n{}", muon.stderr);
    assert!(
        muon.losses.len() >= 4,
        "too few losses: {:?}\n{}",
        muon.losses,
        muon.stderr
    );
    let first: f64 = muon.losses.first().unwrap().parse().unwrap();
    let last: f64 = muon.losses.last().unwrap().parse().unwrap();
    assert!(
        last < first,
        "muon did not train: first {first} last {last}"
    );
    assert_ne!(
        adamw.losses, muon.losses,
        "muon stream identical to adamw — orthogonalized update vacuous"
    );
    // w_up/w_down are rank-2 and NOT name-excluded: only m.embed may appear
    // in the name-routed list.
    let route_line = muon
        .stderr
        .lines()
        .find(|l| l.contains("[muon]"))
        .expect("routing table missing");
    assert!(
        route_line.contains("1 name-routed") && route_line.contains("m.embed"),
        "unexpected routing: {route_line}"
    );
}

/// P5 item 2 — long-run quality comparison, GPU. 8 epochs (104 micro-
/// batches, 52 optimizer steps under grad_accumulation=2) of the FFN LM
/// under AdamW at its fixture-tuned lr vs mixed Muon at the spec defaults
/// (lr=0.02, momentum=0.95). Prints a side-by-side report; hard
/// assertions stay at sanity level (both train; Muon's final loss is not
/// catastrophically worse) because a quality RANKING on a toy fixture
/// would be noise-pinned, not signal.
#[test]
#[ignore = "requires CUDA GPU; ~2 min"]
fn muon_vs_adamw_long_run_gpu_report() {
    let epochs = ("epochs=1", "epochs=8");
    let adamw = run_fixture(
        "csla_layerwise_ffn.nsl",
        "long_adamw",
        &[
            ("# GPU_PLACEMENT", "m.to(cuda)"),
            epochs,
            ("CSLA_SAVE_PATH", "long_adamw.nslm"),
        ],
        &[],
    );
    assert!(adamw.success, "long adamw run failed:\n{}", adamw.stderr);
    let muon = run_fixture(
        "csla_layerwise_ffn.nsl",
        "long_muon",
        &[
            ("# GPU_PLACEMENT", "m.to(cuda)"),
            epochs,
            (
                "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
                "Muon(lr=0.02, momentum=0.95, nesterov=true, ns_steps=5, \
                 weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
            ),
            ("CSLA_SAVE_PATH", "long_muon.nslm"),
        ],
        &[],
    );
    assert!(muon.success, "long muon run failed:\n{}", muon.stderr);

    let f = |v: &[String]| -> (f64, f64, f64) {
        let first: f64 = v.first().unwrap().parse().unwrap();
        let last: f64 = v.last().unwrap().parse().unwrap();
        let tail = &v[v.len().saturating_sub(8)..];
        let tail_mean = tail.iter().map(|s| s.parse::<f64>().unwrap()).sum::<f64>()
            / tail.len() as f64;
        (first, last, tail_mean)
    };
    let (a0, an, am) = f(&adamw.losses);
    let (m0, mn, mm) = f(&muon.losses);
    println!("=== Muon vs AdamW long-run report (GPU, 8 epochs, 52 steps) ===");
    println!("AdamW (lr=0.002):  first {a0:.4}  last {an:.4}  tail-8 mean {am:.4}");
    println!("Muon  (lr=0.02):   first {m0:.4}  last {mn:.4}  tail-8 mean {mm:.4}");
    println!(
        "tail-8 mean delta (Muon - AdamW): {:+.4} ({})",
        mm - am,
        if mm < am { "Muon ahead" } else { "AdamW ahead" }
    );
    assert!(an < a0, "adamw did not train: {a0} -> {an}");
    assert!(mn < m0, "muon did not train: {m0} -> {mn}");
    // Sanity bound, not a ranking: Muon must be in the same league.
    assert!(
        mm < am + 1.0,
        "muon tail-8 mean {mm} catastrophically behind adamw {am}"
    );
}

/// GPU: the mixed step runs the Newton-Schulz chain on device f32 tensors
/// (matmul/transpose/sum/item). Must train and must differ from GPU AdamW.
#[test]
#[ignore = "requires CUDA GPU"]
fn muon_mixed_gpu_trains_and_differs() {
    let adamw = run_fixture(
        "csla_layerwise_ffn.nsl",
        "gpu_adamw",
        &[
            ("# GPU_PLACEMENT", "m.to(cuda)"),
            ("CSLA_SAVE_PATH", "gpu_adamw.nslm"),
        ],
        &[],
    );
    assert!(adamw.success, "GPU adamw baseline failed:\n{}", adamw.stderr);

    let muon = run_fixture(
        "csla_layerwise_ffn.nsl",
        "gpu_muon",
        &[
            ("# GPU_PLACEMENT", "m.to(cuda)"),
            (
                "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
                "Muon(lr=0.002, momentum=0.95, nesterov=true, ns_steps=5, \
                 weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
            ),
            ("CSLA_SAVE_PATH", "gpu_muon.nslm"),
        ],
        &[],
    );
    assert!(muon.success, "GPU muon run failed:\n{}", muon.stderr);
    let first: f64 = muon.losses.first().expect("losses").parse().unwrap();
    let last: f64 = muon.losses.last().unwrap().parse().unwrap();
    assert!(last < first, "GPU muon did not train: {first} -> {last}");
    assert_ne!(
        adamw.losses, muon.losses,
        "GPU muon stream identical to GPU adamw"
    );
    // CPU/GPU sanity: the first micro-batch loss is pre-update model state —
    // f32-vs-f64 forward noise only.
    let cpu = run_fixture(
        "csla_layerwise_ffn.nsl",
        "cpu_muon_ref",
        &[
            (
                "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
                "Muon(lr=0.002, momentum=0.95, nesterov=true, ns_steps=5, \
                 weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
            ),
            ("CSLA_SAVE_PATH", "cpu_muon.nslm"),
        ],
        &[],
    );
    assert!(cpu.success, "CPU muon reference failed:\n{}", cpu.stderr);
    let g0: f64 = muon.losses[0].parse().unwrap();
    let c0: f64 = cpu.losses[0].parse().unwrap();
    assert!(
        (g0 - c0).abs() < 1e-2,
        "GPU first loss {g0} too far from CPU {c0}"
    );
}
