//! A host element write cannot change a parameter behind the BF16 matmul
//! mode's cached weight image.
//!
//! Under `NSL_MATMUL_BF16=1` each parameter the fused AdamW step has stepped
//! keeps a persistent bf16 copy that later matmuls read instead of the f32
//! weights (`nsl-runtime/src/cuda/bf16_cast_cache.rs`). A write that changed
//! the f32 weights without evicting that copy would feed every later matmul
//! the old values. The element writes evict nothing because they never reach
//! device memory: `nsl_tensor_set` (and `_set_element`, `_slice_assign`)
//! refuse a GPU tensor before writing. This holds that with the cache warm:
//! after two training steps, `m.w[0, 0] = 100.0` stops the program with the
//! reason, and nothing computed after it is printed.
//!
//! Run (GPU required):
//!   cargo test -p nsl-cli --features cuda --test element_write_gpu_gate \
//!     -- --ignored --test-threads=1
#![cfg(feature = "cuda")]

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn an_element_write_to_a_cached_gpu_parameter_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let prog = dir.path().join("p.nsl");
    std::fs::write(
        &prog,
        r#"model M:
    w: Tensor = ones([16, 16])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = M()
m.to(cuda)
let x = ones([4, 16]).to(cuda)
train(model=m, epochs=2):
    optimizer: AdamW(lr=0.01)
    step(batch):
        let loss = sum(m.forward(x))
print("BEFORE_WRITE")
m.w[0, 0] = 100.0
print("AFTER_WRITE")
print(sum(m.forward(x)).item())
"#,
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .arg(&prog)
        .current_dir(dir.path())
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .env("NSL_MATMUL_BF16", "1")
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stdout.contains("BEFORE_WRITE"), "the training part did not run:\n{stdout}\n{stderr}");
    assert!(!out.status.success(), "the element write into a GPU parameter ran:\n{stdout}");
    assert!(stderr.contains("element write on a GPU tensor is not supported"), "{stderr}");
    assert!(!stdout.contains("AFTER_WRITE"), "the program went on after the write:\n{stdout}");
}
