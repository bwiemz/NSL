//! Decorator-free SDPA backward — per-head_dim variant-table wiring gates.
//!
//! # The gap this closes (PR #335 / #344 follow-up)
//!
//! `compile_flash_attention_kernels` only synthesizes backward PTX for
//! programs with an `@flash_attention` decorator. Every decorator-free
//! model — i.e. every plain stdlib transformer — lowered its SDPA
//! backward with four NULL PTX pointers, so `nsl_flash_attention_backward`
//! silently took the CPU reference path: five device-to-host copies, a
//! CPU triple loop, and three re-uploads per attention op per step, while
//! the forward ran on GPU. The #324-#328 GPU backward kernels were
//! unreachable from plain models.
//!
//! The fix embeds a per-head_dim backward variant table (hd 32/64/128,
//! synthesized for the op's static `causal` flag) and selects the matching
//! variant on the RUNTIME head_dim value; unmatched dims (and GQA ops,
//! which the classic kernel does not support) still select NULL and keep
//! the CPU fallback.
//!
//! # Gates
//!
//! 1. `variant_table_embedded_for_decorator_free_train` (host-only):
//!    the emitted object for a decorator-free SDPA train program must
//!    contain all three phase-2 entry names — the names encode the
//!    per-head_dim tile selection (q64_kv64 / q32_kv32 / q32_kv16), so
//!    their joint presence proves the whole table was embedded.
//! 2. `variant_table_not_embedded_without_sdpa` (host-only): laziness —
//!    a train program with no attention must carry none of it.
//! 3. `decorator_free_sdpa_backward_dispatches_on_gpu` (GPU, ignored):
//!    `nsl run` with NSL_FLASH_DEBUG=1 must print the
//!    `[flash-bwd] GPU backward dispatched` line for a decorator-free
//!    model AND all trainable params must move — the positive observable
//!    that was missing for a whole campaign (the GPU path used to be
//!    silent on success, so "no [flash-bwd] lines" was the only, purely
//!    negative, signal).
//!
//! Invocation (GPU gate — note `--features cuda`, the nsl-cli feature that
//! also makes compiled child programs link the CUDA-enabled runtime;
//! `nsl-runtime/cuda` alone only affects the compiler process):
//!   cargo test -p nsl-cli --test sdpa_decorator_free_gpu_backward \
//!     --features cuda -- --ignored --nocapture

use assert_cmd::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Decorator-free variant of the CSHA toy: same shape discipline
/// (`[batch=1, heads=1, seq, head_dim]` rank-4 input, `mse_loss` to dodge
/// the unrelated `sum(tensor)` source-AD lowering bug), but NO
/// `@flash_attention` decorator — this is the "plain stdlib model" shape
/// the variant table exists for.
///
/// seq=64 / head_dim=64 is deliberate: `select_backward_blocks(64)` picks
/// (32, 32) tiles and the runtime refuses seq not divisible by both tile
/// sizes (no ragged-tail guards in the phase-2 kernel), so 64 % 32 == 0 is
/// what makes the GPU path admissible at all.
const DECORATOR_FREE_SRC: &str = r#"from nsl.nn.losses import mse_loss

model TinyAttn:
    w_norm: Tensor = ones([64])
    wq: Tensor = ones([64, 64])
    wk: Tensor = ones([64, 64])
    wv: Tensor = ones([64, 64])

    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(64.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
m.to(cuda)
let x = ones([1, 1, 64, 64]).to(cuda)
let y = zeros([1, 1, 64, 64]).to(cuda)

print("BEFORE_wv")
print(sum(m.wv))
print("BEFORE_w_norm")
print(sum(m.w_norm))

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)

print("AFTER_wv")
print(sum(m.wv))
print("AFTER_w_norm")
print(sum(m.w_norm))
"#;

/// Control: a train program with no attention anywhere. The variant table
/// is synthesized lazily at the first SDPA-backward lowering, so this
/// object must not contain any backward entry names.
const NO_SDPA_SRC: &str = r#"from nsl.nn.losses import mse_loss

model TinyMlp:
    w: Tensor = ones([16, 16])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = TinyMlp()
let x = ones([4, 16])
let y = zeros([4, 16])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)
"#;

/// Compile `src` to object files with `nsl build --emit-obj` and return the
/// raw bytes of the ENTRY module's object.
///
/// Both fixtures import from the stdlib, which routes the build down the
/// multi-file path: per-module objects land in a `nsl_build_<pid>` temp dir
/// and the CLI prints one `Wrote <path>` line per module. The entry module's
/// object is the one whose file stem starts with `{tag}` — that's where the
/// train block (and thus the lazily-embedded backward variant table) lives.
fn build_obj_bytes(src: &str, tag: &str) -> Vec<u8> {
    let tmp = TempDir::new().expect("tempdir");
    let src_path = tmp.path().join(format!("{tag}.nsl"));
    fs::write(&src_path, src).expect("write source");

    let mut cmd = Command::cargo_bin("nsl").expect("locate nsl binary");
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .current_dir(tmp.path())
        .arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--target")
        .arg("sm_89")
        .arg("--emit-obj");
    let output = cmd.output().expect("spawn nsl build");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        output.status.success(),
        "[sdpa-bwd-table] `nsl build --emit-obj` failed for {tag}.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr),
    );
    let entry_obj = stdout
        .lines()
        .filter_map(|l| l.trim().strip_prefix("Wrote "))
        .find(|p| {
            std::path::Path::new(p)
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with(tag))
        })
        .unwrap_or_else(|| {
            panic!(
                "[sdpa-bwd-table] no `Wrote <...>/{tag}*.o` line in build \
                 stdout — emit-obj output convention changed?\nstdout:\n{stdout}"
            )
        })
        .to_string();
    fs::read(&entry_obj)
        .unwrap_or_else(|e| panic!("read emitted object {entry_obj}: {e}"))
}

fn contains_bytes(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|w| w == needle.as_bytes())
}

/// Gate 1: all three per-head_dim phase-2 variants embedded.
///
/// The entry names don't encode head_dim directly, but the tile selection
/// does — `backward_select_blocks`: hd32→(64,64), hd64→(32,32),
/// hd128→(32,16) — so the three names below are exactly the three
/// head_dim variants for a causal (c1) op. (SDPA's `causal` defaults to
/// true when the call site omits it, as this fixture does.)
#[test]
fn variant_table_embedded_for_decorator_free_train() {
    let obj = build_obj_bytes(DECORATOR_FREE_SRC, "decorator_free_attn");
    for name in [
        "flash_attn_bwd_main_c1_q64_kv64", // head_dim 32
        "flash_attn_bwd_main_c1_q32_kv32", // head_dim 64
        "flash_attn_bwd_main_c1_q32_kv16", // head_dim 128
    ] {
        assert!(
            contains_bytes(&obj, name),
            "[sdpa-bwd-table] emitted object is missing backward variant \
             entry `{name}` — the decorator-free SDPA backward variant \
             table was not (fully) embedded, so plain models regress to \
             the silent CPU backward"
        );
    }
}

/// Gate 2: laziness — no attention, no table.
#[test]
fn variant_table_not_embedded_without_sdpa() {
    let obj = build_obj_bytes(NO_SDPA_SRC, "no_sdpa_mlp");
    assert!(
        !contains_bytes(&obj, "flash_attn_bwd_main"),
        "[sdpa-bwd-table] a train program with no SDPA op embedded \
         backward variant PTX — the table is supposed to be synthesized \
         lazily at the first SDPA-backward lowering, not unconditionally"
    );
}

/// Gate 3 (GPU): the decorator-free model's backward actually dispatches
/// on the GPU, and training still updates the weights.
#[test]
#[ignore = "requires a CUDA-capable GPU"]
fn decorator_free_sdpa_backward_dispatches_on_gpu() {
    let tmp = TempDir::new().expect("tempdir");
    let src_path = tmp.path().join("decorator_free_gpu.nsl");
    fs::write(&src_path, DECORATOR_FREE_SRC).expect("write source");

    let mut cmd = Command::cargo_bin("nsl").expect("locate nsl binary");
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .env("NSL_FLASH_DEBUG", "1")
        .arg("run")
        .arg("--source-ad")
        .arg("--target")
        .arg("sm_89")
        .arg(&src_path);
    let output = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let driver_missing = stderr.contains("no CUDA devices")
            || stderr.contains("CUDA_ERROR_NOT_INITIALIZED")
            || stderr.contains("CUDA driver not found");
        if driver_missing {
            eprintln!("[sdpa-bwd-gpu] SKIP — no CUDA driver.");
            return;
        }
        panic!(
            "[sdpa-bwd-gpu] `nsl run` failed with exit {}.\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}",
            output.status
        );
    }

    // The positive dispatch observable: the runtime narrates the backward
    // dispatch under NSL_FLASH_DEBUG=1. head_dim=64 must have selected the
    // (32,32)-tile variant.
    assert!(
        stderr.contains("[flash-bwd] GPU backward dispatched")
            && stderr.contains("head_dim=64"),
        "[sdpa-bwd-gpu] the decorator-free SDPA backward did NOT dispatch \
         on the GPU — either the variant table wasn't wired (look for a \
         'no backward PTX provided' line below) or the launch failed and \
         fell back to CPU.\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
    );

    // And the step must still be a real training step: both bracketed
    // params moved. (Grad correctness at kernel level is pinned by the
    // #324-#328 suites; this is the end-to-end wiring bellwether.)
    let grab = |marker: &str| -> f64 {
        let mut lines = stdout.lines();
        while let Some(l) = lines.next() {
            if l.trim() == marker {
                for v in lines.by_ref() {
                    let t = v.trim();
                    if t.is_empty() || t.starts_with('[') {
                        continue;
                    }
                    // CPU sums print bare (`4096.0`), GPU sums print as a
                    // 1-element tensor (`tensor([4096.0])`) — accept both.
                    let t = t
                        .strip_prefix("tensor([")
                        .and_then(|s| s.strip_suffix("])"))
                        .unwrap_or(t);
                    return t.parse::<f64>().unwrap_or_else(|_| {
                        panic!("[sdpa-bwd-gpu] non-numeric line after {marker}: {t}")
                    });
                }
            }
        }
        panic!("[sdpa-bwd-gpu] marker {marker} not found in stdout:\n{stdout}");
    };
    // wv and w_norm are the two params with structurally non-zero gradients
    // in this all-ones toy (uniform attention makes the Q/K grads exactly
    // zero — d softmax is zero under uniform probabilities and a uniform
    // upstream, so wq/wk staying put is the fixture's math, not a bug).
    for name in ["wv", "w_norm"] {
        let before = grab(&format!("BEFORE_{name}"));
        let after = grab(&format!("AFTER_{name}"));
        assert!(
            before.is_finite() && after.is_finite(),
            "[sdpa-bwd-gpu] non-finite {name} sum (before={before}, after={after})"
        );
        assert!(
            (after - before).abs() > 1e-6,
            "[sdpa-bwd-gpu] param {name} did not move (before={before}, after={after}) — \
             the GPU backward dispatched but produced no effective gradient"
        );
    }
}
