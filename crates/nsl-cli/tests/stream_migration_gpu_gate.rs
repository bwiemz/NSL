//! Milestone C · p8 PR-A — compute-stream migration gate.
//!
//! All kernel launches (and cuBLAS via a per-call `cublasSetStream_v2`)
//! migrated from the legacy NULL stream to a dedicated per-thread BLOCKING
//! compute stream (`inner::current_stream()`). Blocking-stream semantics make
//! the migration ordering-neutral — the legacy NULL stream is an implicit
//! two-way barrier against every blocking stream, so the runtime's sync
//! memcpys and interleaved NULL-stream work keep the exact total order —
//! while making the stream capturable for CUDA graphs (p8 PR-B; the legacy
//! NULL stream cannot be captured).
//!
//! Differential proof (campaign pattern): the SAME CSHA training run under
//!   * `NSL_LEGACY_NULL_STREAM=1` (old NULL-stream launches),
//!   * default                     (compute-stream launches)
//! must produce **bit-identical** trained-parameter sums under
//! `--deterministic`. The CSHA fixture exercises kernels, cuBLAS matmuls,
//! sync HtoD/DtoH, the event-based deferred frees, and the caching allocator
//! in one run — every ordering assumption the migration touched.
//!
//! GPU-only, `#[ignore]`. Run:
//!   cargo test -p nsl-cli --features cuda --test stream_migration_gpu_gate \
//!     -- --ignored --nocapture --test-threads=1

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Same CSHA fixture as the deferred-free gate: @flash_attention training with
/// source-AD dispatches the fused CSHA backward (deferred frees + cuBLAS +
/// elementwise kernels interleaved), the densest stream-ordering workload the
/// test suite has.
const CSHA_TRAIN_SRC: &str = r#"from nsl.nn.losses import mse_loss

model TinyAttn:
    w_norm: Tensor = ones([32])
    wq: Tensor = ones([32, 32])
    wk: Tensor = ones([32, 32])
    wv: Tensor = ones([32, 32])

    @flash_attention(head_dim=32)
    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(32.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
let x = ones([1, 1, 32, 32])
let y = zeros([1, 1, 32, 32])

train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)

print("AFTER_wq")
print(sum(m.wq))
print("AFTER_wk")
print(sum(m.wk))
print("AFTER_wv")
print(sum(m.wv))
print("AFTER_w_norm")
print(sum(m.w_norm))
"#;

fn run(fixture: &std::path::Path, legacy: bool) -> (String, bool, String) {
    let root = repo_root();
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--",
            "run",
            "--source-ad",
            "--deterministic",
            "--target",
            "sm_89",
        ])
        .arg(fixture)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if legacy {
        cmd.env("NSL_LEGACY_NULL_STREAM", "1");
    }
    let out = cmd.output().expect("spawn nsl run");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    let mut block = String::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("AFTER_") {
            for next in lines.iter().skip(i + 1) {
                let n = next.trim();
                if n.is_empty() || n.starts_with('[') {
                    continue;
                }
                let num = n
                    .strip_prefix("tensor([")
                    .and_then(|r| r.strip_suffix("])"))
                    .unwrap_or(n);
                if num.parse::<f64>().is_ok() {
                    block.push_str(name);
                    block.push('=');
                    block.push_str(num);
                    block.push('\n');
                }
                break;
            }
        }
    }
    (
        block,
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
#[ignore = "requires a CUDA GPU (2 CSHA training runs)"]
fn compute_stream_matches_legacy_null_stream_bit_exact() {
    let tmp = std::env::temp_dir().join("nsl_stream_migration_gate.nsl");
    std::fs::write(&tmp, CSHA_TRAIN_SRC).expect("write fixture");

    // Legacy (old NULL-stream) first, so a cold compile doesn't skew things.
    let (legacy_sums, ok_legacy, err_legacy) = run(&tmp, true);
    assert!(ok_legacy, "legacy (NSL_LEGACY_NULL_STREAM=1) run failed:\n{err_legacy}");
    let (compute_sums, ok_compute, err_compute) = run(&tmp, false);
    assert!(ok_compute, "compute-stream (default) run failed:\n{err_compute}");

    let n = legacy_sums.lines().count();
    assert_eq!(n, 4, "expected 4 AFTER_ sums, got {n}:\nlegacy=\n{legacy_sums}");
    assert!(
        legacy_sums.contains("wv=") && !legacy_sums.contains("wv=1024\n"),
        "expected wv to move from its 1024.0 init (a real gradient step):\n{legacy_sums}"
    );

    assert_eq!(
        compute_sums, legacy_sums,
        "compute-stream launches diverged from legacy NULL-stream launches — an \
         ordering assumption broke (sync-memcpy barrier, deferred-free event, \
         cuBLAS stream, or transfer-stream wait). Re-run with \
         NSL_LEGACY_NULL_STREAM=1 to confirm, then audit `current_stream()` \
         call sites in cuda/mod.rs."
    );

    eprintln!("[stream-migration] OK: CSHA trained-param sums bit-exact (legacy == compute stream)\n{legacy_sums}");
}
