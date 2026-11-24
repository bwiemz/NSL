//! Round-3 gate: an RMSNorm's input gradient and its gain gradient lowered as one pass
//! over the row (`nsl_rmsnorm_backward_fused`, kill switch `NSL_FUSE_NORM_DGAMMA=0`).
//!
//! The kernel's `dx` is the dx kernel's, bit for bit; its `dgamma` is the dgamma
//! kernels' arithmetic over a finer row chunking (`nsl_kir::kernels::rmsnorm_bwd`), so it
//! rounds differently in the last bits. Hence:
//!
//! - on the host (the runtime runs the two ops there) the compiler's grouping must fire
//!   and the losses must equal the kill switch's exactly;
//! - on the GPU the fused kernel must run, a second run must give the same bits
//!   (`--deterministic`), and the losses must follow the kill switch's to within a
//!   summation-order difference.
//!
//! Fixture: two pre-norm transformer blocks (stdlib grouped-query attention, SwiGLU MLP,
//! residual adds) in a `blocks` array, the production flag line, four micro-batches per
//! optimizer step. Four norms: three with a dx and a dgamma op in the backward, and the
//! first, which reads the model input, with only a dgamma op.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// The fixture: `gpu` places it on the device, `bf16` stores activations bf16.
fn fixture(gpu: bool, bf16: bool) -> String {
    let place = if gpu {
        "m.to(cuda)\nlet xd = x.to(cuda)\nlet yd = y.to(cuda)"
    } else {
        "let xd = x\nlet yd = y"
    };
    let precision = if bf16 { r#", precision="bf16""# } else { "" };
    format!(
        r#"from nsl.nn.norms import RMSNorm
from nsl.nn.gqa import GroupedQueryAttention
from nsl.nn.losses import mse_loss

model Block:
    attn_norm: RMSNorm = RMSNorm(128)
    attn: GroupedQueryAttention = GroupedQueryAttention(128, 2, 1, 64, 0.0, 64, 10000.0, 2)
    ffn_norm: RMSNorm = RMSNorm(128)
    w_gate: Tensor = ((arange(32768).reshape([128, 256])) * 0.0000035) - 0.06
    w_up: Tensor = ((arange(32768).reshape([128, 256])) * 0.0000025) - 0.045
    w_down: Tensor = ((arange(32768).reshape([256, 128])) * 0.000002) - 0.04

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + self.attn.forward(self.attn_norm.forward(x), true)
        let n = self.ffn_norm.forward(h)
        return h + (silu(n @ self.w_gate) * (n @ self.w_up)) @ self.w_down

model Net:
    blocks: [Block; 2] = Block()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
let x = ((arange(8192).reshape([2, 32, 128])) * 0.0003) - 1.2
let y = ((arange(8192).reshape([2, 32, 128])) * 0.0002) - 0.8
{place}

train(model=m, epochs=8, grad_accumulation=4{precision}):
    optimizer: AdamW(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-2, weight_decay=0.0)
    step(batch):
        let pred = m.forward(xd)
        let loss = mse_loss(pred, yd)
    callbacks:
        on_step(step, loss):
            print(loss)
"#
    )
}

const PROD: &[&str] = &[
    "--source-ad",
    "--checkpoint-blocks",
    "--checkpoint-selective",
    "--fuse-rmsnorm-backward",
    "--fuse-wgrad-accum",
];

const MARKER: &str = "[fuse] rmsnorm dx+dgamma passes:";

/// `nsl run --deterministic` of `src` with `flags` and `env`: (stdout, stderr,
/// kernel_profile.json, empty without `NSL_PROFILE_KERNELS`).
fn run(tag: &str, src: &str, flags: &[&str], env: &[(&str, &str)]) -> (String, String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_norm_bwd_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let profile = tmp.join("kernel_profile.json");
    let _ = std::fs::remove_file(&profile);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--deterministic"])
        .args(flags)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env_remove("NSL_FUSE_NORM_DGAMMA")
        .env_remove("NSL_MATMUL_BF16");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed (tag={tag}):\n{stderr}");
    let prof = std::fs::read_to_string(&profile).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&tmp);
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr,
        prof,
    )
}

/// The printed losses: a device scalar prints `tensor([x])`, a host one a bare float.
fn losses(out: &str) -> Vec<f64> {
    out.lines()
        .filter_map(|l| {
            let l = l.trim();
            let v = l
                .strip_prefix("tensor([")
                .and_then(|r| r.strip_suffix("])"))
                .unwrap_or(l);
            v.parse::<f64>().ok()
        })
        .inspect(|x| assert!(x.is_finite(), "non-finite loss {x}"))
        .collect()
}

/// The passes the compiler grouped, from its `[fuse]` line.
fn passes(stderr: &str) -> u64 {
    let rest = stderr
        .split_once(MARKER)
        .map(|(_, r)| r)
        .unwrap_or_else(|| panic!("no `{MARKER}` in:\n{stderr}"));
    rest.split(|c: char| !c.is_ascii_digit())
        .find(|s| !s.is_empty())
        .and_then(|n| n.parse().ok())
        .expect("a count")
}

/// The default arm and the kill switch: (losses on, losses off, stderr on, profile on).
fn both_arms(tag: &str, src: &str, env: &[(&str, &str)]) -> (Vec<f64>, Vec<f64>, String, String) {
    let with = |extra: &[(&'static str, &'static str)]| -> Vec<(&str, &str)> {
        env.iter()
            .copied()
            .chain(extra.iter().copied())
            .chain([("NSL_PROFILE_KERNELS", "1")])
            .collect()
    };
    let (on, on_err, on_prof) = run(&format!("{tag}_on"), src, PROD, &with(&[]));
    let (off, off_err, off_prof) = run(
        &format!("{tag}_off"),
        src,
        PROD,
        &with(&[("NSL_FUSE_NORM_DGAMMA", "0")]),
    );
    // Four norms; the first reads the model input, which needs no gradient, so it has
    // a gain gradient and no input gradient. The other three pair up.
    assert_eq!(passes(&on_err), 3, "{tag}: grouped passes:\n{on_err}");
    assert!(
        !off_err.contains(MARKER),
        "{tag}: the kill switch still grouped:\n{off_err}"
    );
    assert!(
        !off_prof.contains("nsl_rmsnorm_bwd_fused"),
        "{tag}: the kill switch still launched it"
    );
    let (l_on, l_off) = (losses(&on), losses(&off));
    assert_eq!(l_off.len(), 8, "{tag}: one loss per micro-batch:\n{off}");
    assert!(
        l_off[7] < l_off[0],
        "{tag}: the program did not train: {l_off:?}"
    );
    assert_eq!(l_on.len(), l_off.len(), "{tag}");
    (l_on, l_off, on_err, on_prof)
}

/// On the host the runtime runs the two ops, so the grouping changes nothing.
#[test]
fn norm_backward_pass_groups_and_is_exact_on_the_host() {
    let (on, off, _, _) = both_arms("cpu", &fixture(false, false), &[]);
    assert_eq!(on, off, "the host runs the same two ops either way");
}

/// On the GPU: the fused kernel runs, is deterministic, and changes the losses only by
/// the gain gradient's summation order.
fn gpu_case(tag: &str, bf16: bool, env: &[(&str, &str)]) {
    let src = fixture(true, bf16);
    let (on, off, _, prof) = both_arms(tag, &src, env);
    assert!(
        prof.contains("\"name\":\"nsl_rmsnorm_bwd_fused_add_f32_s1"),
        "{tag}: no fused launch:\n{prof}"
    );
    let with: Vec<(&str, &str)> = env.iter().copied().collect();
    let (again, _, _) = run(&format!("{tag}_again"), &src, PROD, &with);
    assert_eq!(
        losses(&again),
        on,
        "{tag}: two runs differ under --deterministic"
    );
    for (k, (a, b)) in on.iter().zip(&off).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5 * b.abs() + 1e-7,
            "{tag}: loss {k}: fused {a} vs two ops {b}"
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn norm_backward_pass_runs_and_tracks_the_two_ops_under_f32_storage_gpu() {
    gpu_case("f32", false, &[]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn norm_backward_pass_runs_and_tracks_the_two_ops_under_bf16_storage_gpu() {
    gpu_case("bf16", true, &[("NSL_FLASH_BWD_CPU", "0")]);
}
