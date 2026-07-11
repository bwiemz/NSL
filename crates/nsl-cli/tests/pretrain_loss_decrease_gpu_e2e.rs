//! Roadmap 4.2 + 4.3: end-to-end GPU pretraining gates for the production
//! composition (stdlib GQA/SDPA + FASE Deferred + two-phase grad clip).
//!
//! * 4.2 `pretrain_loss_decreases_on_gpu` — runs a downscaled copy of
//!   `models/coder-rl/pretrain.nsl` on the GPU via real `nsl run --source-ad`
//!   and asserts the loss stream is finite and strictly improving
//!   (mean of last 5 < mean of first 5, and final < uniform baseline).
//!   This is the first test in the repo that asserts *training makes progress*
//!   end-to-end, not just that gradients match an oracle.
//!
//! * 4.3 `fase_deferred_matches_plain_adamw_checkpoint` — trains the same
//!   downscaled model twice on the identical constant corpus: once with
//!   grad_accumulation=4 (FASE Deferred windowed AdamW) and once with
//!   grad_accumulation=1 (plain per-step AdamW), optimizer-step counts
//!   matched. With constant micro-batches the window-mean gradient equals
//!   each micro-batch gradient, so the two trajectories are mathematically
//!   identical up to accumulation arithmetic — the checkpoints must agree
//!   within a small tolerance. A silent FASE bug (wrong v_t window, dropped
//!   micro-batch, mis-scaled mean) breaks this equality.
//!
//! Both are `#[ignore]` + cuda-gated: they need the RTX GPU box
//! (`cargo test -p nsl-cli --features cuda --test pretrain_loss_decrease_gpu_e2e
//!  -- --ignored --test-threads=1`). Tape-AD is never used: it lowers SDPA
//! naively and leaves GPU params frozen (documented in
//! zero_grad_accum_gpu_e2e.rs), which would vacuously "pass" a loss check by
//! never training — `--source-ad` is load-bearing.

#![cfg(feature = "cuda")]

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

/// Rewrite the literal knobs + save path of pretrain.nsl so the e2e stays
/// minutes-cheap and writes into a temp dir. DataLoader/train arguments must
/// be compile-time literals in NSL (a `const` does not propagate into
/// DataLoader kwargs), so the program keeps them inline with unique spellings
/// and this helper rewrites them by exact string match — every `replace`
/// below asserts it actually fired so a reworded knob can't silently leave
/// the full-size config running in CI.
fn downscaled_program(save_path: &Path, accum: u32, token_count: u64) -> String {
    let src = std::fs::read_to_string(repo_root().join("models/coder-rl/pretrain.nsl"))
        .expect("models/coder-rl/pretrain.nsl must exist");
    let mut out = src;
    for (from, to) in [
        ("full([524288], 1.0)".to_string(), format!("full([{token_count}], 1.0)")),
        ("batch_size=4, seq_len=1024".to_string(), "batch_size=2, seq_len=256".to_string()),
        ("grad_accumulation=4".to_string(), format!("grad_accumulation={accum}")),
        (
            "model_save(m, \"models/coder-rl/checkpoints/coder_rl_pretrain.nslm\")".to_string(),
            format!("model_save(m, \"{}\")", save_path.display()),
        ),
    ] {
        assert!(out.contains(&from), "knob `{from}` not found in pretrain.nsl — resync harness");
        out = out.replace(&from, &to);
    }
    out
}

struct RunOutput {
    losses: Vec<f64>,
    stdout: String,
    stderr: String,
    success: bool,
}

fn run_program(source: &str, tag: &str) -> RunOutput {
    run_program_with_env(source, tag, &[])
}

fn run_program_with_env(source: &str, tag: &str, extra_env: &[(&str, &str)]) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_pretrain_e2e_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("pretrain_scaled.nsl");
    std::fs::write(&prog, source).unwrap();
    // `from model import NSLCoderRL` resolves relative to the program dir —
    // copy the model next to the scaled program.
    std::fs::copy(root.join("models/coder-rl/model.nsl"), tmp.join("model.nsl")).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run", "--source-ad"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let output = cmd.output().expect("failed to spawn nsl run");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    // Parse the bare-float loss stream between the markers.
    let mut losses = Vec::new();
    let mut in_stream = false;
    for line in stdout.lines() {
        let t = line.trim();
        match t {
            "LOSS_STREAM_BEGIN" => in_stream = true,
            "LOSS_STREAM_END" => in_stream = false,
            _ if in_stream => {
                if let Ok(v) = t.parse::<f64>() {
                    losses.push(v);
                }
            }
            _ => {}
        }
    }
    RunOutput { losses, stdout, stderr, success: output.status.success() }
}

fn gpu_present() -> bool {
    // Cheap driver probe; skip (not fail) on GPU-less machines even when the
    // cuda feature is compiled in.
    Command::new("nvidia-smi").output().map(|o| o.status.success()).unwrap_or(false)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// 4.2 — loss must go down. 65536 tokens = 128 micro-batches at 2x256
/// = 32 optimizer steps at ACCUM=4; constant tokens collapse fast, so the
/// signal is unambiguous within a couple of minutes of GPU time.
#[test]
#[ignore = "requires CUDA GPU (runs a real training loop)"]
fn pretrain_loss_decreases_on_gpu() {
    if !gpu_present() {
        eprintln!("skipping: no NVIDIA driver");
        return;
    }
    let tmp_ckpt = std::env::temp_dir().join(format!("pretrain_42_{}.nslm", std::process::id()));
    let out = run_program(&downscaled_program(&tmp_ckpt, 4, 65536), "loss42");

    assert!(
        out.success,
        "nsl run failed.\n--- stdout tail ---\n{}\n--- stderr tail ---\n{}",
        out.stdout.chars().rev().take(2000).collect::<String>().chars().rev().collect::<String>(),
        out.stderr.chars().rev().take(2000).collect::<String>().chars().rev().collect::<String>(),
    );
    assert!(
        out.losses.len() >= 20,
        "expected a loss stream (>=20 micro-steps), got {} entries",
        out.losses.len()
    );
    assert!(
        out.losses.iter().all(|v| v.is_finite()),
        "non-finite loss in stream: {:?}",
        out.losses
    );
    let first = mean(&out.losses[..5]);
    let last = mean(&out.losses[out.losses.len() - 5..]);
    assert!(
        last < first,
        "loss did not decrease: first5={first:.4} last5={last:.4}\nstream={:?}",
        out.losses
    );
    // Constant-token corpus: the model must beat the uniform-vocabulary
    // baseline decisively, not just twitch downward.
    assert!(
        last < 8.0,
        "final loss {last:.4} did not move below the ~8.32 uniform baseline"
    );
    std::fs::remove_file(&tmp_ckpt).ok();
}

// ─── 4.3 parity: FASE Deferred (accum=4) vs plain AdamW (accum=1) ───────────

mod nslm {
    //! Minimal .nslm reader — a faithful copy of the proven parser at
    //! crates/nsl-codegen/tests/common/nslm_reader.rs (NSLM magic + hand-parsed
    //! JSON header entries {name, dtype, offset, nbytes} + 64-byte-aligned
    //! payload; f64 entries downcast to f32 like the original). Copied rather
    //! than shared because test `common/` modules don't cross crate boundaries.
    use std::collections::HashMap;
    use std::path::Path;

    struct Entry {
        name: String,
        dtype: String,
        offset: usize,
        nbytes: usize,
    }

    pub fn read(path: &Path) -> HashMap<String, Vec<f32>> {
        let buf = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        assert!(buf.len() >= 16 && &buf[0..4] == b"NSLM", "bad magic in {path:?}");
        let header_size = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        let header_end = 16 + header_size;
        let json = std::str::from_utf8(&buf[16..header_end]).expect("header utf8");
        let pad = (64 - (header_end % 64)) % 64;
        let data_start = header_end + pad;

        let mut out = HashMap::new();
        for e in parse_entries(json) {
            let slab = &buf[data_start + e.offset..data_start + e.offset + e.nbytes];
            let values: Vec<f32> = match e.dtype.as_str() {
                "f32" => slab
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                "f64" => slab
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
                other => panic!("unsupported dtype {other} in {path:?}"),
            };
            out.insert(e.name, values);
        }
        out
    }

    fn parse_entries(json: &str) -> Vec<Entry> {
        let mut entries = Vec::new();
        let mut cursor = 0;
        while let Some(name_idx) = json[cursor..].find("\"name\":\"") {
            let abs = cursor + name_idx + 8;
            let name_end = json[abs..].find('"').expect("malformed name");
            let name = json[abs..abs + name_end].to_string();
            cursor = abs + name_end;
            let dtype = extract_str(json, cursor, "dtype");
            let offset = extract_int(json, cursor, "offset");
            let nbytes = extract_int(json, cursor, "nbytes");
            entries.push(Entry { name, dtype, offset, nbytes });
        }
        entries
    }

    fn extract_str(json: &str, from: usize, key: &str) -> String {
        let pattern = format!("\"{key}\":\"");
        let abs = from + json[from..].find(&pattern).expect(key) + pattern.len();
        let end = json[abs..].find('"').expect(key);
        json[abs..abs + end].to_string()
    }

    fn extract_int(json: &str, from: usize, key: &str) -> usize {
        let pattern = format!("\"{key}\":");
        let abs = from + json[from..].find(&pattern).expect(key) + pattern.len();
        let end = json[abs..].find(|c: char| c == ',' || c == '}').expect(key);
        json[abs..abs + end].trim().parse().expect(key)
    }
}

/// 4.3 — same corpus, same optimizer-step count: accum=4 over 65536 tokens
/// (32 optimizer steps of window-mean grads) vs accum=1 over 16384 tokens
/// (32 optimizer steps of per-batch grads). Constant data makes the window
/// mean equal the per-batch gradient, so the checkpoints must match.
#[test]
#[ignore = "requires CUDA GPU (runs two real training loops)"]
fn fase_deferred_matches_plain_adamw_checkpoint() {
    if !gpu_present() {
        eprintln!("skipping: no NVIDIA driver");
        return;
    }
    let pid = std::process::id();
    let ckpt_fase = std::env::temp_dir().join(format!("pretrain_43_fase_{pid}.nslm"));
    let ckpt_plain = std::env::temp_dir().join(format!("pretrain_43_plain_{pid}.nslm"));

    // Force the deterministic CPU attention backward in BOTH runs. Since the
    // decorator-free backward variant table landed, the GPU phase-2 kernel
    // serves these models — and its float-atomicAdd dK/dV accumulation is
    // scheduling-order nondeterministic, so the two runs would receive
    // slightly different attention gradients and drift past the 5e-3 gate
    // (measured 5.999e-3) for reasons unrelated to what this test pins:
    // FASE-deferred vs per-step AdamW OPTIMIZER equivalence. 4.2 keeps the
    // GPU backward in its path; this gate isolates the optimizer semantics.
    let det = &[("NSL_FLASH_BWD_CPU", "1")][..];
    let fase = run_program_with_env(&downscaled_program(&ckpt_fase, 4, 65536), "fase43", det);
    assert!(fase.success, "FASE run failed:\n{}", fase.stderr);
    let plain = run_program_with_env(&downscaled_program(&ckpt_plain, 1, 16384), "plain43", det);
    assert!(plain.success, "plain run failed:\n{}", plain.stderr);

    let a = nslm::read(&ckpt_fase);
    let b = nslm::read(&ckpt_plain);
    assert_eq!(a.len(), b.len(), "checkpoint tensor sets differ");
    assert!(!a.is_empty(), "empty checkpoint");

    // f32 GPU state + differing accumulation arithmetic: allow a small
    // absolute drift, but it must stay tiny relative to the ~0.02-magnitude
    // initialization scale after only 32 steps.
    let tol = 5e-3;
    for (name, va) in &a {
        let vb = &b[name];
        assert_eq!(va.len(), vb.len(), "shape mismatch in {name}");
        let max_diff = va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < tol,
            "{name}: FASE-deferred vs plain-AdamW checkpoint diverged \
             (max_abs_diff={max_diff:.3e} > {tol:.0e}) — windowed-accumulation \
             semantics drifted from per-step AdamW on constant data"
        );
    }
    std::fs::remove_file(&ckpt_fase).ok();
    std::fs::remove_file(&ckpt_plain).ok();
}
