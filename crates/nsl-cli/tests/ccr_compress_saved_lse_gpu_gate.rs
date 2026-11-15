//! `--checkpoint-compress bf16` must not make the attention backward recompute
//! the forward's logsumexp.
//!
//! The flash-attention forward saves its per-row logsumexp, and the backward
//! is handed it through a codegen side channel keyed by the forward OUTPUT's
//! value (`Compiler::flash_attn_aux`). Compression stores that output as bf16
//! at the end of its block and the backward reads a restored f32 copy, a
//! different value; the side channel did not follow it, so the backward got no
//! saved logsumexp and recomputed it with `nsl_flash_lse_*`, one thread per
//! row over the whole key range: 108 ms per call at Coder-1B, 6.9 s of a
//! 9.6 s update (Nsight Systems, November 2026). That was the "compress
//! triples the 1B step" of `models/benchmarks/NSL_VS_PYTORCH_2026_11.md`.
//!
//! The gate trains Coder-50M (the real model, synthetic tokens) on the GPU
//! with the production checkpointing flags, with and without compression,
//! under the kernel profiler, and asserts that the attention backward ran on
//! the device and the logsumexp recompute kernel did not, in both. It does so
//! for both attention backwards: the tensor-core kernels (`nsl_flash_tc_bwd_*`,
//! the default) and the previous ones (`flash_attn_bwd_*`, under
//! `NSL_TC_ATTENTION=0`). Each recomputes a missing logsumexp with its own
//! `nsl_flash_lse_*` kernel, which is what the gate looks for.
//!
//! GPU-only. Run:
//!   cargo test -p nsl-cli --features cuda --test ccr_compress_saved_lse_gpu_gate \
//!     -- --ignored --test-threads=1
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

fn gpu_present() -> bool {
    Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Two optimizer updates of batch 1 x 1024, accumulation 2: 4097 tokens and
/// some slack, ids drawn uniformly below the vocabulary.
fn write_tokens(path: &std::path::Path) {
    let n = 2 * 2 * 1024 + 64;
    let mut s: u64 = 0x5EED;
    let bytes: Vec<u8> = (0..n)
        .flat_map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) % 49152) as u16).to_le_bytes()
        })
        .collect();
    std::fs::write(path, bytes).unwrap();
}

fn program(tokens: &std::path::Path) -> String {
    format!(
        r#"from model import NSLCoder
from nsl.nn.losses import cross_entropy

let m = NSLCoder()
m.to(cuda)
let tokens = load_mmap("{}", 3)
let loader = DataLoader(tokens, batch_size=1, seq_len=1024, shuffle=false, drop_last=true)
print("LOSS_STREAM_BEGIN")
@fused_lm_ce(enabled=true, vocab_size=49152, hidden_size=512, batch_size=1, seq_len=1024)
train(model=m, epochs=1, grad_accumulation=2, schedule_unit="optimizer_step", grad_clip=1.0):
    optimizer: AdamW(lr=0.0003, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8)
    step(batch):
        let logits = m.forward_train(batch.input_ids, true)
        let flat_logits = logits.reshape([1024, 49152])
        let flat_labels = batch.labels.reshape([1024])
        let loss = cross_entropy(flat_logits, flat_labels)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")
"#,
        tokens.display()
    )
}

/// Run the program with `flags` under the kernel profiler, with
/// `NSL_TC_ATTENTION` set to `tc_attention`; returns (stderr, the profile's
/// JSON).
fn run_profiled(tag: &str, flags: &[&str], tc_attention: &str) -> (String, String) {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_ccr_lse_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["model.nsl", "config.nsl"] {
        std::fs::copy(root.join("models/coder50m").join(f), dir.join(f)).unwrap();
    }
    let tokens = dir.join("tokens.bin");
    write_tokens(&tokens);
    let prog = dir.join("prog.nsl");
    std::fs::write(&prog, program(&tokens)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .args(flags)
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_PROFILE_KERNELS", "1")
        .env("NSL_TC_ATTENTION", tc_attention)
        .output()
        .expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success() && stdout.contains("LOSS_STREAM_END"),
        "run failed (tag={tag}):\n{stderr}"
    );
    let json = std::fs::read_to_string(dir.join("kernel_profile.json"))
        .unwrap_or_else(|e| panic!("no kernel_profile.json (tag={tag}): {e}\n{stderr}"));
    let _ = std::fs::remove_dir_all(&dir);
    (stderr, json)
}

const PROD: [&str; 5] = [
    "--source-ad",
    "--checkpoint-blocks",
    "--checkpoint-selective",
    "--fuse-rmsnorm-backward",
    "--fuse-wgrad-accum",
];

#[test]
#[ignore = "requires CUDA GPU"]
fn compressed_saves_keep_the_forward_logsumexp() {
    if !gpu_present() {
        return;
    }
    let mut compress: Vec<&str> = PROD.to_vec();
    compress.extend(["--checkpoint-compress", "bf16"]);
    // (NSL_TC_ATTENTION, the backward kernels that serve the call under it)
    let paths = [("1", "nsl_flash_tc_bwd"), ("0", "flash_attn_bwd")];
    for (tc, bwd_kernel) in paths {
        for (save, flags) in [("plain", PROD.to_vec()), ("compress", compress.clone())] {
            let tag = format!("{save}_tc{tc}");
            let tag = tag.as_str();
            let (stderr, json) = run_profiled(tag, &flags, tc);
            if save == "compress" {
                assert!(
                    stderr.contains("compressed saves:"),
                    "compression did not engage, so this run proves nothing:\n{stderr}"
                );
            }
            assert!(
                json.contains(bwd_kernel),
                "{tag}: no {bwd_kernel}* kernel ran — the gate is not looking at the path \
             it guards"
            );
            assert!(
                !json.contains("nsl_flash_lse"),
                "{tag}: the attention backward recomputed the logsumexp (nsl_flash_lse_*) \
             instead of reading the forward's saved one"
            );
        }
    }
}
