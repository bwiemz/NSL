//! P0 certification regression: `--monitor` on a FASE-Deferred train block
//! (AdamW + grad_accumulation >= 2) used to SIGSEGV — the health grad-norm
//! loop read per-batch gradients the source-AD hook had already consumed
//! into m_partial and freed. The fix skips grad-norm recording on that
//! path (loss/weight-norm/flush still record) with a loud note.

use std::process::Command;

const PROG: &str = r#"from nsl.nn.losses import mse_loss

model TinyMlp:
    w: Tensor = ones([16, 16])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = TinyMlp()
let x = ones([4, 16])
let y = zeros([4, 16])

train(model = m, epochs = 4, grad_accumulation = 2):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)
"#;

#[test]
fn monitor_on_fase_deferred_does_not_crash() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let tmp = std::env::temp_dir().join(format!("nsl_monitor_fase_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, PROG).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args([
            "run", "--source-ad", "--deterministic", "--monitor",
            "--health-interval", "1",
        ])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "--monitor on FASE-Deferred crashed (status {:?}):\n{stderr}",
        out.status
    );
    assert!(
        stderr.contains("gradient norms are not recorded under the FASE-Deferred hook"),
        "expected the loud grad-norm skip note:\n{stderr}"
    );
    // The health snapshot must still flush (loss/weight-norm recording live).
    assert!(
        tmp.join("prog.nsl.nsl-health.json").exists(),
        "health snapshot file missing — flush path broken:\n{stderr}"
    );
}
