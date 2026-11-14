//! An `@flash_attention` method at a sequence length the kernel's tiles do
//! not divide is refused, and at one they do it computes attention.
//!
//! The method-body forward (`nsl_flash_attention`, and the CSHA launchers
//! beside it) runs the v2 kernel, which has no tail guard on the sequence:
//! each CTA stores all `block_q` (64) of its query rows. At a sequence of 32
//! the second 32 rows went past the end of the output buffer and overwrote
//! the 2 KB after it, which was another live tensor. In
//! `examples/csha_toy_pretrain_hd32.nsl` that tensor was the copy of Wq the
//! program keeps to measure the update, so the Wq update read 17.2 against
//! PyTorch's 0.0061 while every gradient was right. The launchers now
//! refuse such a sequence and the generated code stops on the refusal.
//!
//! The first test checks the refusal at a sequence of 32; the second checks
//! that the decorated method's output matches attention written out by hand
//! (`softmax(q k^T * scale) v`) at one and at two whole tiles. Both need a
//! GPU.
//!
//! ```text
//! cargo test -p nsl-cli --release --features cuda \
//!     --test flash_attention_ragged_seq_gate -- --ignored --test-threads=1
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nsl_fa_ragged_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A decorated, non-causal attention method called once outside any train
/// block, on `x` of shape `[1, 1, seq, 32]`, printed next to the same
/// attention written out by hand. The hand-written form uses no
/// `scaled_dot_product_attention`, which the decorator reroutes for the
/// whole module.
///
/// The weights are chosen so that the output does not cancel. With Wq and
/// Wk at amplitude 0.3 and Wv centred on zero the attention is close to a
/// uniform average of rows that sum to almost nothing (`|r|` is 1/100 of
/// `|v|`), and the f16 rounding of the kernel's operands alone puts the
/// output 3.4e-3 off (measured, and the same in a float64 simulation of
/// that rounding). Here the same simulation gives 4.1e-4 at 64 rows and
/// 3.3e-4 at 128, and attention with uniform weights would be 100% off.
fn program(seq: usize) -> String {
    let n = seq * 32;
    format!(
        r#"model Attn:
    wq: Tensor = tensor_sin(arange(0.0, 1024.0) * 0.37).reshape([32, 32])
    wk: Tensor = tensor_sin(arange(0.0, 1024.0) * 0.53 + 1.0).reshape([32, 32])
    wv: Tensor = (0.5 + 0.3 * tensor_sin(arange(0.0, 1024.0) * 0.71 + 2.0)).reshape([32, 32])

    @flash_attention(head_dim=32, causal=false)
    fn forward(self, x: Tensor) -> Tensor:
        let q = x @ self.wq
        let k = x @ self.wk
        let v = x @ self.wv
        let scale = 1.0 / sqrt(32.0)
        return scaled_dot_product_attention(q, k, v, scale, causal=false)

let m = Attn()
m.to(cuda)
let x = tensor_sin(arange(0.0, {n}.0) * 0.11).reshape([1, 1, {seq}, 32]).to(cuda)
let y = m.forward(x)
let q = x @ m.wq
let k = x @ m.wk
let v = x @ m.wv
let r = softmax((q @ k.transpose(2, 3)) * (1.0 / sqrt(32.0)), -1) @ v
print(sum((y - r) * (y - r)))
print(sum(r * r))
print("fa_ragged_seq_gate: DONE")
"#
    )
}

fn run(dir: &Path, seq: usize) -> Output {
    let root = repo_root();
    let prog = dir.join(format!("fa_seq{seq}.nsl"));
    std::fs::write(&prog, program(seq)).unwrap();
    Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--target", "cuda", "--matmul-mode", "f32"])
        .arg(&prog)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl")
}

/// The numbers the program prints, from lines `tensor([x])` or bare `x`.
fn numbers(stdout: &str) -> Vec<f64> {
    stdout
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            let inner = l
                .strip_prefix("tensor([")
                .and_then(|r| r.strip_suffix("])"))
                .unwrap_or(l);
            inner.parse::<f64>().ok()
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_sequence_shorter_than_the_tile_is_refused() {
    let dir = scratch_dir("seq32");
    let out = run(&dir, 32);
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !out.status.success() && !stdout.contains("fa_ragged_seq_gate: DONE"),
        "a decorated attention at seq_len 32 (block_q 64) ran to the end:\n{stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("seq_len=32 is not a multiple of the kernel's block_q=64"),
        "the run stopped, but not on the ragged-sequence refusal:\n{stderr}"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn whole_tiles_match_attention_written_out() {
    for seq in [64usize, 128] {
        let dir = scratch_dir(&format!("seq{seq}"));
        let out = run(&dir, seq);
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            out.status.success() && stdout.contains("fa_ragged_seq_gate: DONE"),
            "seq_len {seq} did not run:\n{stdout}\n{stderr}"
        );
        let v = numbers(&stdout);
        assert_eq!(
            v.len(),
            2,
            "seq_len {seq}: expected two numbers, got {v:?}\n{stdout}"
        );
        let rel = (v[0] / v[1]).sqrt();
        eprintln!("seq_len {seq}: |y - r| / |r| = {rel:.3e}");
        // f16 operands and output: 4.1e-4 and 3.3e-4 in simulation (see
        // `program`); the margin is for the kernel's approximate exp and
        // reciprocal. |r|^2 is about 800 and 1700.
        assert!(
            v[1] > 1.0 && rel < 2e-3,
            "seq_len {seq}: the decorated attention is {rel:.3e} off the written-out one \
             (|r|^2 = {})",
            v[1]
        );
    }
}
