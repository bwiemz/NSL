//! `--csha` on a model whose attention is not in an `@flash_attention` method
//! must say it did nothing.
//!
//! CSHA's kernels are emitted only for `@flash_attention` methods. Every
//! stock model calls `scaled_dot_product_attention` directly
//! (`stdlib/nsl/nn/gqa.nsl`), gets the same fused flash forward and backward
//! with or without the flag, and compiles to a byte-identical object. The
//! planner still ran, on a nominal layer shape, and its output was printed
//! as if it described the build: Coder-50M and Coder-1B both reported
//! "CSHA L2-pipeline ... 6.00x speedup ... Backward: Tier B.2" with a kernel
//! name that appears in no object, followed by four "CSHA fused backward
//! rejected" lines per layer, and the pass trace recorded
//! `CSHA: applied`.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const PROGRAM: &str = r#"from nsl.nn.norms import RMSNorm
from nsl.nn.gqa import GroupedQueryAttention
from nsl.nn.losses import mse_loss

model Block:
    norm: RMSNorm = RMSNorm(32)
    attn: GroupedQueryAttention = GroupedQueryAttention(32, 4, 2, 8, 0.0, 16, 10000.0, 1)

    fn forward(self, x: Tensor) -> Tensor:
        return x + self.attn.forward(self.norm.forward(x), true)

let m = Block()
let x = randn([1, 16, 32])
let target = zeros([1, 16, 32])

train(model = m, epochs = 2):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, target)
print("CSHA_REPORT_GATE_DONE")
"#;

fn run(tag: &str, flags: &[&str]) -> (String, String) {
    let dir = std::env::temp_dir().join(format!("nsl_cshareport_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(&prog, PROGRAM).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--target", "cpu", "--source-ad", "--seed", "3"])
        .args(flags)
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .env("NSL_PASS_TRACE", "1")
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        out.status.success() && stdout.contains("CSHA_REPORT_GATE_DONE"),
        "[{tag}] the program failed:\n{stdout}\n{stderr}"
    );
    (stdout, stderr)
}

#[test]
fn csha_on_decorator_free_attention_reports_that_it_did_nothing() {
    for (tag, flags) in [("summary", &["--csha", "auto"][..]), ("report", &["--csha", "auto", "--csha-report"][..])] {
        let (_, err) = run(tag, flags);
        // The planner did find the chains: this is not the "no attention" case.
        assert!(
            err.contains("[csha] not applied: 3 attention projection chain(s) found"),
            "[{tag}] expected the not-applied line naming the chains it found:\n{err}"
        );
        for claim in ["speedup", "CSHA L2-pipeline", "CSHA L1-boundary", "Backward: Tier", "CSHA fused backward rejected"] {
            assert!(!err.contains(claim), "[{tag}] the build printed {claim:?} for a kernel it does not emit:\n{err}");
        }
        assert!(
            err.contains("[pass-trace] CSHA: declined"),
            "[{tag}] the pass trace must record a decline, not an application:\n{err}"
        );
    }
}
