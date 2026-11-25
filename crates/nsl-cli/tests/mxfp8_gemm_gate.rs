//! `NSL_MXFP8_GEMMS=1`: the GEMMs that read a parameter run in MXFP8.
//!
//! The fixture (`fixtures/attn_layout.nsl`, two pre-norm blocks of
//! grouped-query attention and a SwiGLU MLP under a tied LM head) trains on
//! the production flag line with `precision="bf16"`, 8 micro-batches of
//! 2 × 128 tokens in 4 optimizer updates. Each block has seven projections
//! (q, k, v, o, gate, up, down), so every micro-batch runs 14 forward
//! projections, 14 input gradients and 14 weight gradients.
//!
//! - every one of them runs MXFP8 (the runtime's `[mxfp8]` counts), none
//!   declines, and a parameter's two weight images are quantized once per
//!   optimizer step and reused by the other micro-batch;
//! - two runs under `--deterministic` give the same loss stream and weights,
//!   bit for bit;
//! - the losses stay within MXFP8 distance of the bf16 run;
//! - without the variable nothing runs MXFP8.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const PROD: &[&str] = &[
    "--source-ad",
    "--checkpoint-blocks",
    "--checkpoint-selective",
    "--fuse-rmsnorm-backward",
    "--fuse-wgrad-accum",
];

struct Out {
    stdout: String,
    stderr: String,
}

fn run(tag: &str, mx: bool, deterministic: bool) -> Out {
    let root = repo_root();
    let src = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/attn_layout.nsl"))
        .expect("attn_layout fixture missing")
        .replace(
            "attn: ATTN = ATTN(",
            "attn: GroupedQueryAttention = GroupedQueryAttention(",
        )
        .replace("NH, NKV, 64, DROP,", "4, 2, 64, 0.0,")
        .replace("# GPU_PLACEMENT", "m.to(cuda)")
        .replace(
            "grad_accumulation=2):",
            "grad_accumulation=2, precision=\"bf16\"):",
        );
    let tmp = std::env::temp_dir().join(format!("nsl_mxfp8_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("p.nsl");
    std::fs::write(&prog, &src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--seed", "7"]).args(PROD);
    if deterministic {
        // Keep the attention backward on the tensor-core kernels, which are
        // deterministic themselves (one writer per output element). Before
        // #894 `--deterministic` sent it to the CPU reference, which reads no
        // bf16.
        cmd.arg("--deterministic").env("NSL_FLASH_BWD_CPU", "0");
    }
    cmd.arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_BF16_LT_COUNTER", "1")
        .env_remove("NSL_MXFP8_GEMMS");
    if mx {
        cmd.env("NSL_MXFP8_GEMMS", "1");
    }
    let o = cmd.output().expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&tmp);
    let (stdout, stderr) = (
        String::from_utf8_lossy(&o.stdout).to_string(),
        String::from_utf8_lossy(&o.stderr).to_string(),
    );
    assert!(
        o.status.success(),
        "{tag} failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    Out { stdout, stderr }
}

fn section(stdout: &str, begin: &str, end: &str) -> Vec<String> {
    stdout
        .split(begin)
        .nth(1)
        .and_then(|s| s.split(end).next())
        .unwrap_or("")
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("tensor([") || l.parse::<f64>().is_ok())
        .map(str::to_string)
        .collect()
}

fn value(line: &str) -> f64 {
    line.trim_start_matches("tensor([")
        .split([']', ','])
        .next()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no value in {line:?}"))
}

fn losses(o: &Out) -> Vec<String> {
    let l = section(&o.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END");
    assert_eq!(l.len(), 8, "8 micro-batch losses expected:\n{}", o.stdout);
    for x in &l {
        assert!(value(x).is_finite(), "non-finite loss {x}");
    }
    l
}

fn weights(o: &Out) -> Vec<String> {
    let w = section(&o.stdout, "WEIGHTS_BEGIN", "WEIGHTS_END");
    assert_eq!(w.len(), 5, "5 weight sums expected:\n{}", o.stdout);
    w
}

/// The `[mxfp8]` counter line: name -> value.
fn mx_counts(o: &Out) -> std::collections::HashMap<String, u64> {
    let line = o
        .stderr
        .lines()
        .find(|l| l.starts_with("[mxfp8] fwd="))
        .unwrap_or_else(|| panic!("no [mxfp8] counter line:\n{}", o.stderr));
    // `fwd=1 dgrad=2 ... weight-images quantized=3 reused=4 lt issued=5 tuned=6 declined=7`
    let mut out = std::collections::HashMap::new();
    let mut prefix = "";
    for tok in line.trim_start_matches("[mxfp8] ").split_whitespace() {
        match tok.split_once('=') {
            Some((k, v)) => {
                out.insert(
                    format!("{prefix}{k}"),
                    v.parse().unwrap_or_else(|_| panic!("{line}")),
                );
            }
            None => {
                prefix = if tok == "weight-images" {
                    "img-"
                } else {
                    "lt-"
                }
            }
        }
    }
    out
}

#[test]
#[ignore = "requires CUDA GPU"]
fn every_projection_gemm_runs_mxfp8_and_stays_near_bf16() {
    let mx = run("mx", true, false);
    let c = mx_counts(&mx);
    eprintln!("[mxfp8_gemm_gate] counts: {c:?}");
    // 8 micro-batches x 2 blocks x 7 projections.
    assert_eq!(c["fwd"], 112, "{c:?}");
    assert_eq!(c["wgrad"], 112, "{c:?}");
    assert_eq!(c["dgrad"] + c["dgrad-acc"], 112, "{c:?}");
    assert_eq!(c["declined"], 0, "{c:?}");
    assert_eq!(c["lt-declined"], 0, "{c:?}");
    assert!(
        c["img-reused"] > 0,
        "the weight images were never reused: {c:?}"
    );
    assert!(
        mx.stderr.contains("[mxfp8]") && mx.stderr.contains("lowered MXFP8"),
        "no compile-time [mxfp8] line:\n{}",
        mx.stderr
    );

    let bf = run("bf16", false, false);
    let cb = mx_counts(&bf);
    assert_eq!(
        cb["fwd"] + cb["dgrad"] + cb["dgrad-acc"] + cb["wgrad"],
        0,
        "bf16 run took MXFP8: {cb:?}"
    );
    let (lm, lb) = (losses(&mx), losses(&bf));
    let mut worst = 0f64;
    for (a, b) in lm.iter().zip(&lb) {
        let (a, b) = (value(a), value(b));
        worst = worst.max((a - b).abs() / b.abs());
    }
    eprintln!("[mxfp8_gemm_gate] worst relative loss difference vs bf16: {worst:e}");
    assert!(
        worst < 0.05,
        "MXFP8 losses drift {worst:e} from bf16:\n{lm:?}\n{lb:?}"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn mxfp8_training_is_bit_reproducible_under_deterministic() {
    let (a, b) = (run("det_a", true, true), run("det_b", true, true));
    assert_eq!(
        losses(&a),
        losses(&b),
        "loss streams differ between two --deterministic runs"
    );
    assert_eq!(
        weights(&a),
        weights(&b),
        "trained weights differ between two --deterministic runs"
    );
    let c = mx_counts(&a);
    assert_eq!(c["declined"], 0, "{c:?}");
    assert_eq!(c["fwd"], 112, "{c:?}");
}
