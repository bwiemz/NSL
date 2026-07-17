//! Milestone C · p3-remainder — stream-ordered deferred-free gate.
//!
//! The two raw-`cuMemFree` hot-path sites (`nsl_csha_free_backward_activations`
//! on the CSHA @train backward, and the Tier B.1 `PrepassScratch::drop` on the
//! CSHA inference path) previously blocked on a `cuCtxSynchronize` before every
//! physical free, to guarantee the kernels reading those buffers had finished.
//! p3-remainder replaces that host barrier with a CUDA event: the buffer is
//! freed physically only once a NULL-stream completion event (recorded *after*
//! the consuming kernels) fires — no CPU stall, same "return VRAM to the
//! driver" semantics.
//!
//! The correctness proof is the same differential test used for p3: the SAME
//! CSHA training run under
//!   * `NSL_CUDA_SYNC=1` (eager: `cuCtxSynchronize` then free — old behavior),
//!   * default            (stream-ordered deferred free)
//! must produce **bit-identical** trained-parameter sums. This fixture uses a
//! `@flash_attention` model trained with source-AD, which dispatches the CSHA
//! fused backward and therefore allocates + frees the six save buffers every
//! step via `nsl_csha_free_backward_activations`. If the deferred free freed or
//! recycled a buffer while its backward kernel was still reading it, the
//! gradients — and thus the parameter sums — would diverge from the eager run.
//! Under `--deterministic` the GPU path is bit-reproducible, so exact equality
//! is the right assertion, not a tolerance band.
//!
//! GPU-only, `#[ignore]`. Run:
//!   cargo test -p nsl-cli --features cuda --test deferred_free_gpu_gate \
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

/// A `@flash_attention` model trained for a few SGD steps with source-AD. The
/// backward dispatches the CSHA fused backward, whose save buffers are released
/// through the stream-ordered deferred-free path under test. Three steps give
/// the deferred free several alloc→(kernel)→free cycles to expose any race.
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

/// Run the CSHA training fixture. `eager` sets `NSL_CUDA_SYNC=1` (old
/// sync-then-free). Returns (after_sums_block, success, stderr).
fn run(fixture: &std::path::Path, eager: bool) -> (String, bool, String) {
    let root = repo_root();
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--",
            "run",
            // source-AD is required for the CSHA fused-backward dispatcher.
            "--source-ad",
            // bit-reproducible GPU path, so eager vs deferred is exact-comparable.
            "--deterministic",
            // sm_89 PTX JITs forward-compatibly onto any sm >= 89 device.
            "--target",
            "sm_89",
        ])
        .arg(fixture)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if eager {
        cmd.env("NSL_CUDA_SYNC", "1");
    }
    let out = cmd.output().expect("spawn nsl run");

    // Collect the `AFTER_<name>` marker lines and the bare numeric line that
    // follows each (skipping interleaved `[gpu-mem]` / `[nsl]` diagnostics).
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    let mut block = String::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("AFTER_") {
            // Find the first bare-number line after the marker.
            for next in lines.iter().skip(i + 1) {
                let n = next.trim();
                if n.is_empty() || n.starts_with('[') {
                    continue;
                }
                if n.parse::<f64>().is_ok() {
                    block.push_str(name);
                    block.push('=');
                    block.push_str(n);
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
fn deferred_free_matches_eager_bit_exact() {
    let tmp = std::env::temp_dir().join("nsl_deferred_free_gate.nsl");
    std::fs::write(&tmp, CSHA_TRAIN_SRC).expect("write fixture");

    // Eager (old sync-then-free) first, so a cold compile doesn't skew things.
    let (eager_sums, ok_eager, err_eager) = run(&tmp, true);
    assert!(ok_eager, "eager (NSL_CUDA_SYNC=1) run failed:\n{err_eager}");
    let (deferred_sums, ok_deferred, err_deferred) = run(&tmp, false);
    assert!(
        ok_deferred,
        "stream-ordered deferred-free (default) run failed:\n{err_deferred}"
    );

    // Sanity: all four parameters were captured, and at least one actually
    // moved — proves the CSHA backward (and thus the deferred-free path) really
    // ran, rather than a no-op that would make any two runs trivially equal.
    let n = eager_sums.lines().count();
    assert_eq!(
        n, 4,
        "expected 4 AFTER_ sums, got {n}:\neager=\n{eager_sums}"
    );
    assert!(
        eager_sums.contains("wv=") && !eager_sums.contains("wv=1024\n"),
        "expected wv to move from its 1024.0 init (a real gradient step):\n{eager_sums}"
    );

    assert_eq!(
        deferred_sums, eager_sums,
        "stream-ordered deferred free diverged from eager sync-then-free — a raw \
         `cuMemFree` ran (or a buffer was recycled) before its CSHA backward \
         kernel finished reading it. Re-run with NSL_CUDA_SYNC=1 to confirm, then \
         audit the event-record ordering in `defer_free_device`."
    );

    eprintln!("[deferred-free] OK: CSHA trained-param sums bit-exact (eager == deferred)\n{eager_sums}");
}
