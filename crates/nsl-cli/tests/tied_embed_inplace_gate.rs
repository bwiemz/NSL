//! A tied embedding's gradient accumulated in place (`nsl-codegen`
//! `tied_embed_fusion`): under the FASE hook, the fused LM head's dW GEMMs
//! add `1/N · dW` straight into the embedding's accumulation buffer (`beta =
//! 1`) and the embedding backward scatter-adds `1/N · dy` rows into it, so no
//! `[V, H]` scratch is zero-filled and no `[V, H]` add runs.
//!
//! The fixture is the native-GQA one (`fixtures/sdpa_native_gqa.nsl`) under
//! `@fused_lm_ce`: an embedding tied to its LM head, `grad_accumulation=2`, 4
//! updates. It is
//! trained with the fusion and with `NSL_TIED_EMBED_INPLACE=0` (the
//! materialised `m += (dW + dE) / N`) under the production flags. The two are
//! not bit-identical — the sum `m + (dW + dE)/N` becomes `(m + dW/N) + dE/N`
//! — so the gate checks that:
//!
//! - the fusion fired (the `[tied-embed]` line) and the kill switch keeps it
//!   off;
//! - the first micro-batch's loss (before any update) is the same bits;
//! - every later loss and every trained weight agree to rounding;
//! - under `--deterministic` the fused program is still bit-reproducible run
//!   to run (its embedding scatter then takes the deterministic route).

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOutput {
    stdout: String,
    stderr: String,
}

fn spawn_drain<R: Read + Send + 'static>(
    mut pipe: R,
    buf: Arc<Mutex<Vec<u8>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.lock().unwrap().extend_from_slice(&chunk[..n]),
            }
        }
    })
}

const PROD_FLAGS: [&str; 5] = [
    "--source-ad",
    "--checkpoint-blocks",
    "--checkpoint-selective",
    "--fuse-rmsnorm-backward",
    "--fuse-wgrad-accum",
];

fn run(tag: &str, bf16: bool, deterministic: bool, envs: &[(&str, &str)]) -> RunOutput {
    let root = repo_root();
    let mut src = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/sdpa_native_gqa.nsl"))
        .expect("sdpa_native_gqa fixture missing")
        .replace("attn: ATTN = ATTN(", "attn: GroupedQueryAttention = GroupedQueryAttention(")
        .replace("# GPU_PLACEMENT", "m.to(cuda)")
        // The production LM head: the fused linear cross-entropy, which
        // needs the logits' dims at compile time (no runtime shape read).
        .replace(
            "train(model=m, epochs=1,",
            "@fused_lm_ce(enabled=true, vocab_size=128, hidden_size=256, batch_size=2, seq_len=128, vocab_tile=128)\ntrain(model=m, epochs=1,",
        )
        .replace(
            "        let ls = logits.shape\n        let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])\n        let flat_labels = batch.labels.reshape([ls[0] * ls[1]])",
            "        let flat_logits = logits.reshape([256, 128])\n        let flat_labels = batch.labels.reshape([256])",
        );
    assert!(
        src.contains("logits.reshape([256, 128])"),
        "the fixture's step changed: update the rewrite above"
    );
    if bf16 {
        src = src.replace(
            "grad_accumulation=2):",
            "grad_accumulation=2, precision=\"bf16\"):",
        );
    }
    let tmp = std::env::temp_dir().join(format!("nsl_tied_embed_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("tied_embed.nsl");
    std::fs::write(&prog, &src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--seed", "5"]).args(PROD_FLAGS);
    if deterministic {
        cmd.arg("--deterministic");
    }
    cmd.arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .envs(envs.iter().copied())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn nsl run");
    let out_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let err_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let out_reader = spawn_drain(child.stdout.take().unwrap(), out_buf.clone());
    let err_reader = spawn_drain(child.stderr.take().unwrap(), err_buf.clone());
    let status = child.wait().expect("wait nsl run");
    let _ = out_reader.join();
    let _ = err_reader.join();
    let stdout = {
        let g = out_buf.lock().unwrap();
        String::from_utf8_lossy(&g).to_string()
    };
    let stderr = {
        let g = err_buf.lock().unwrap();
        String::from_utf8_lossy(&g).to_string()
    };
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
        status.success(),
        "{tag} failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    RunOutput { stdout, stderr }
}

fn section(stdout: &str, begin: &str, end: &str, n: usize) -> Vec<String> {
    let lines: Vec<String> = stdout
        .split(begin)
        .nth(1)
        .and_then(|s| s.split(end).next())
        .unwrap_or("")
        .lines()
        .map(str::trim)
        // A device scalar prints as `tensor([..])`, a host one as the bare number.
        .filter(|l| l.starts_with("tensor([") || l.parse::<f64>().is_ok())
        .map(str::to_string)
        .collect();
    assert_eq!(lines.len(), n, "{begin}: {n} values expected:\n{stdout}");
    for l in &lines {
        let lower = l.to_ascii_lowercase();
        assert!(
            !lower.contains("nan") && !lower.contains("inf"),
            "non-finite value: {l:?}"
        );
    }
    lines
}

fn value(line: &str) -> f64 {
    line.trim_start_matches("tensor([")
        .split(|c: char| c == ']' || c == ',')
        .next()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no value in {line:?}"))
}

const MARKER: &str = "[tied-embed]";

/// The two orders of the sum differ by rounding, and AdamW at the fixture's
/// lr 0.01 amplifies that over the updates as the loss falls towards 0.16
/// (as in `sdpa_native_gqa_gate`): measured, the last loss within 1.6e-4
/// (relative) under `--deterministic`, 7.6e-5 at bf16, and the weight sums
/// within 4e-4. A lost or mis-scaled contribution moves them by percents.
const LOSS_REL: f64 = 1e-3;
const WEIGHT_SUM_REL: f64 = 5e-3;

fn assert_close(tag: &str, what: &str, a: &[String], b: &[String], rel: f64) {
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (value(x), value(y));
        assert!(
            (x - y).abs() <= rel * y.abs().max(1e-3),
            "{tag} {what}: {x} vs {y}"
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_tied_embedding_gradient_is_accumulated_in_place() {
    for bf16 in [false, true] {
        let tag = format!("bf16_{bf16}");
        // Production routes on both arms (the embedding scatter's atomics
        // included), so the comparison is to a tolerance.
        let fused = run(&format!("{tag}_in"), bf16, false, &[]);
        let plain = run(
            &format!("{tag}_off"),
            bf16,
            false,
            &[("NSL_TIED_EMBED_INPLACE", "0")],
        );
        assert!(
            fused.stderr.contains(MARKER),
            "{tag}: the in-place accumulation did not fire:\n{}",
            fused.stderr
        );
        assert!(
            !plain.stderr.contains(MARKER),
            "{tag}: the kill switch did not keep it off:\n{}",
            plain.stderr
        );
        let (lf, lp) = (
            section(&fused.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END", 8),
            section(&plain.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END", 8),
        );
        assert_eq!(
            lf[0], lp[0],
            "{tag}: the first loss (no update yet) differs"
        );
        assert_close(&tag, "loss", &lf, &lp, LOSS_REL);
        let (wf, wp) = (
            section(&fused.stdout, "WEIGHTS_BEGIN", "WEIGHTS_END", 5),
            section(&plain.stdout, "WEIGHTS_BEGIN", "WEIGHTS_END", 5),
        );
        assert_close(&tag, "weight sum", &wf, &wp, WEIGHT_SUM_REL);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn in_place_accumulation_keeps_deterministic_runs_reproducible() {
    let a = run("det_a", false, true, &[("NSL_FLASH_BWD_CPU", "0")]);
    let b = run("det_b", false, true, &[("NSL_FLASH_BWD_CPU", "0")]);
    assert!(
        a.stderr.contains(MARKER),
        "the in-place accumulation did not fire:\n{}",
        a.stderr
    );
    let la = section(&a.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END", 8);
    let lb = section(&b.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END", 8);
    assert_eq!(la, lb, "--deterministic loss streams differ run to run");
    assert_eq!(
        section(&a.stdout, "WEIGHTS_BEGIN", "WEIGHTS_END", 5),
        section(&b.stdout, "WEIGHTS_BEGIN", "WEIGHTS_END", 5),
        "--deterministic trained weights differ run to run"
    );
    // And against the materialised sum, under --deterministic as well.
    let p = run(
        "det_off",
        false,
        true,
        &[("NSL_FLASH_BWD_CPU", "0"), ("NSL_TIED_EMBED_INPLACE", "0")],
    );
    let lp = section(&p.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END", 8);
    assert_eq!(la[0], lp[0], "the first loss (no update yet) differs");
    assert_close("det", "loss", &la, &lp, LOSS_REL);
}
