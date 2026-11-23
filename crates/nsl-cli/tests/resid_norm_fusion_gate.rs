//! Round-2 perf item 6 gates: the residual add lowered together with the RMSNorm of its
//! sum (one launch, `NSL_FUSE_ADD_RMSNORM`).
//!
//! Claimed bit-exact: the fused kernel is the add kernel and the norm kernel's
//! arithmetic, so a program prints the same losses, to the last digit, with the kill
//! switch. Fixture: two pre-norm transformer blocks (stdlib grouped-query attention,
//! SwiGLU MLP, residual adds) in a `blocks` array, the production flag line,
//! `--deterministic`, four micro-batches per optimizer step. Compiled pairs: 3 forward
//! (each block's mid-block add, block 0's output into block 1's attention norm) and 2 in
//! CCR's recompute.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// The fixture: `gpu` places it on the device, `bf16` stores activations bf16, `dropout`
/// puts dropout 0.1 after the attention and MLP outputs (the 500M recipe's shape).
fn fixture(gpu: bool, bf16: bool, dropout: bool) -> String {
    let p = if dropout { "0.1" } else { "0.0" };
    let mlp = "(silu(n @ self.w_gate) * (n @ self.w_up)) @ self.w_down";
    let mlp = if dropout {
        format!("dropout({mlp}, 0.1, true)")
    } else {
        mlp.to_string()
    };
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
    attn: GroupedQueryAttention = GroupedQueryAttention(128, 2, 1, 64, {p}, 64, 10000.0, 2)
    ffn_norm: RMSNorm = RMSNorm(128)
    w_gate: Tensor = ((arange(32768).reshape([128, 256])) * 0.0000035) - 0.06
    w_up: Tensor = ((arange(32768).reshape([128, 256])) * 0.0000025) - 0.045
    w_down: Tensor = ((arange(32768).reshape([256, 128])) * 0.000002) - 0.04

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + self.attn.forward(self.attn_norm.forward(x), true)
        let n = self.ffn_norm.forward(h)
        return h + {mlp}

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

/// `nsl run --deterministic` of `src` with `flags` and `env`: (stdout, stderr,
/// kernel_profile.json, empty without `NSL_PROFILE_KERNELS`).
fn run(tag: &str, src: &str, flags: &[&str], env: &[(&str, &str)]) -> (String, String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_resid_norm_{tag}_{}", std::process::id()));
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
        .env_remove("NSL_FUSE_ADD_RMSNORM")
        .env_remove("NSL_MATMUL_BF16");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed (tag={tag}):\n{stderr}");
    let prof = std::fs::read_to_string(&profile).unwrap_or_default();
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

/// The value after `key` on the runtime's exit report line (`NSL_BF16_LT_COUNTER=1`),
/// e.g. `counter(err, "[add-rmsnorm] fused=")`.
fn counter(stderr: &str, key: &str) -> u64 {
    let rest = stderr
        .split_once(key)
        .map(|(_, r)| r)
        .unwrap_or_else(|| panic!("no `{key}` in:\n{stderr}"));
    rest.split(|c: char| !c.is_ascii_digit())
        .next()
        .and_then(|n| n.parse().ok())
        .expect("a count")
}

/// The GPU runs' exit report: every residual add + RMSNorm took the fused launch.
fn assert_fused_launches(tag: &str, err: &str) {
    assert!(
        counter(err, "[add-rmsnorm] fused=") > 0,
        "{tag}: no fused launch ran:\n{err}"
    );
    assert_eq!(
        counter(err, " pairs="),
        0,
        "{tag}: a launch fell back to the pair:
{err}"
    );
}

/// The two arms (the fusion and its kill switch) of one program: their losses are
/// identical; the marker fires only with the fusion. Returns the default arm's
/// (stderr, profile).
fn two_arms(
    tag: &str,
    src: &str,
    flags: &[&str],
    env: &[(&str, &str)],
    pairs: &str,
) -> (String, String) {
    let with = |extra: &[(&'static str, &'static str)]| -> Vec<(&str, &str)> {
        env.iter()
            .copied()
            .chain(extra.iter().copied())
            .chain([("NSL_PROFILE_KERNELS", "1")])
            .collect()
    };
    let (on, on_err, on_prof) = run(&format!("{tag}_on"), src, flags, &with(&[]));
    let (off, off_err, off_prof) = run(
        &format!("{tag}_off"),
        src,
        flags,
        &with(&[("NSL_FUSE_ADD_RMSNORM", "0")]),
    );
    assert!(on_err.contains(pairs), "{tag}: want `{pairs}`:\n{on_err}");
    assert!(
        !off_err.contains("[fuse] add+rmsnorm pairs:"),
        "{tag}:\n{off_err}"
    );
    let (l_on, l_off) = (losses(&on), losses(&off));
    assert_eq!(l_off.len(), 8, "{tag}: one loss per micro-batch:\n{off}");
    assert!(
        l_off[7] < l_off[0],
        "{tag}: the program did not train: {l_off:?}"
    );
    assert_eq!(l_on, l_off, "{tag}: the add+norm launch vs the kill switch");
    if !off_prof.is_empty() {
        assert!(
            !off_prof.contains("\"name\":\"nsl_add_rmsnorm"),
            "{tag}: the kill switch still launched it"
        );
    }
    (on_err, on_prof)
}

/// On the host the grouping runs (the runtime takes the two calls there), so the marker
/// and the losses check the compiler side in the default test run.
#[test]
fn resid_norm_fusions_fire_and_are_bit_exact_cpu() {
    two_arms(
        "cpu",
        &fixture(false, false, false),
        PROD,
        &[],
        "[fuse] add+rmsnorm pairs: 5 (3 forward, 2 backward)",
    );
}

/// f32 storage on the GPU: the fused kernel (`nsl_add_rmsnorm_f32`) runs, and the
/// program is bit-identical to the kill switches.
#[test]
#[ignore = "requires CUDA GPU"]
fn resid_norm_fusions_are_bit_exact_under_f32_storage_gpu() {
    let (err, prof) = two_arms(
        "f32",
        &fixture(true, false, false),
        PROD,
        &[],
        "[fuse] add+rmsnorm pairs: 5 (3 forward, 2 backward)",
    );
    assert!(
        prof.contains("\"name\":\"nsl_add_rmsnorm_f32"),
        "no fused launch:\n{prof}"
    );
    assert_fused_launches("f32", &err);
}

/// bf16 activation storage with the production flags (the tensor-core attention backward
/// kept on the device under `--deterministic`): the fused kernel reads the f32 residual
/// and the bf16 projection and writes the f32 sum and the bf16 norm output, with GEMMs
/// through GemmEx and, with `NSL_MATMUL_BF16_LT=1`, cublasLt. Bit-identical to the kill
/// switch.
#[test]
#[ignore = "requires CUDA GPU"]
fn resid_norm_fusions_are_bit_exact_under_bf16_storage_gpu() {
    for (tag, lt) in [("bf16", "0"), ("bf16lt", "1")] {
        let (err, prof) = two_arms(
            tag,
            &fixture(true, true, false),
            PROD,
            &[("NSL_FLASH_BWD_CPU", "0"), ("NSL_MATMUL_BF16_LT", lt)],
            "[fuse] add+rmsnorm pairs: 5 (3 forward, 2 backward)",
        );
        assert!(
            err.contains("[bf16-storage] 2 SwiGLU MLP block(s) stored bf16"),
            "{err}"
        );
        assert!(
            prof.contains("\"name\":\"nsl_add_rmsnorm_fb_fb"),
            "{tag}: no fused launch:\n{prof}"
        );
        assert_fused_launches(tag, &err);
    }
}

/// The 500M recipe's shape: dropout 0.1 after both block outputs, so the residual add
/// reads the f32 dropout output (`nsl_add_rmsnorm_ff_fb`); the key-based dropout is
/// untouched and the program stays bit-identical.
#[test]
#[ignore = "requires CUDA GPU"]
fn resid_norm_fusions_compose_with_dropout_under_bf16_storage_gpu() {
    let (err, prof) = two_arms(
        "bf16drop",
        &fixture(true, true, true),
        PROD,
        &[("NSL_FLASH_BWD_CPU", "0")],
        "[fuse] add+rmsnorm pairs: 5 (3 forward, 2 backward)",
    );
    assert!(
        prof.contains("\"name\":\"nsl_add_rmsnorm_ff_fb"),
        "no fused launch:\n{prof}"
    );
    assert_fused_launches("bf16drop", &err);
    assert!(
        prof.contains("\"name\":\"nsl_dropout_key_f32"),
        "the dropout must still run:\n{prof}"
    );
}
