//! Round-2 perf item 6 gates: the residual add lowered together with the RMSNorm of its
//! sum (one launch, `NSL_FUSE_ADD_RMSNORM`), and a value's input gradients from several
//! GEMMs summed by GEMM accumulation (`NSL_FUSE_DGRAD_ACCUM`).
//!
//! Both are claimed bit-exact: the fused kernel is the add kernel and the norm kernel's
//! arithmetic, and the accumulating GEMM runs the standalone product's plan with the
//! add's one rounding in its epilogue. So a program prints the same losses, to the last
//! digit, with either or both kill switches. Fixture: two pre-norm transformer blocks
//! (stdlib grouped-query attention, SwiGLU MLP, residual adds) in a `blocks` array, the
//! production flag line, `--deterministic`, four micro-batches per optimizer step.
//! Per micro-batch: 3 forward pairs (each block's mid-block add, block 0's output into
//! block 1's attention norm) and 2 in CCR's recompute; 3 accumulations per block (k and
//! q onto v, gate onto up).

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
        .env_remove("NSL_FUSE_DGRAD_ACCUM")
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

/// The four arms (both fusions, each alone, neither) of one program: their losses are
/// identical; the markers fire exactly in the arms that enable them. Returns the
/// default arm's (stderr, profile).
fn four_arms(
    tag: &str,
    src: &str,
    flags: &[&str],
    env: &[(&str, &str)],
    pairs: &str,
    accs: &str,
) -> (String, String) {
    let with = |extra: &[(&'static str, &'static str)]| -> Vec<(&str, &str)> {
        env.iter()
            .copied()
            .chain(extra.iter().copied())
            .chain([("NSL_PROFILE_KERNELS", "1"), ("NSL_BF16_LT_COUNTER", "1")])
            .collect()
    };
    let (on, on_err, on_prof) = run(&format!("{tag}_on"), src, flags, &with(&[]));
    let (no_a, no_a_err, no_a_prof) = run(
        &format!("{tag}_noA"),
        src,
        flags,
        &with(&[("NSL_FUSE_ADD_RMSNORM", "0")]),
    );
    let (no_b, no_b_err, _) = run(
        &format!("{tag}_noB"),
        src,
        flags,
        &with(&[("NSL_FUSE_DGRAD_ACCUM", "0")]),
    );
    let (off, off_err, _) = run(
        &format!("{tag}_off"),
        src,
        flags,
        &with(&[("NSL_FUSE_ADD_RMSNORM", "0"), ("NSL_FUSE_DGRAD_ACCUM", "0")]),
    );
    assert!(on_err.contains(pairs), "{tag}: want `{pairs}`:\n{on_err}");
    assert!(on_err.contains(accs), "{tag}: want `{accs}`:\n{on_err}");
    assert!(
        no_a_err.contains(accs) && !no_a_err.contains("[fuse] add+rmsnorm pairs:"),
        "{tag}:\n{no_a_err}"
    );
    assert!(
        no_b_err.contains(pairs) && !no_b_err.contains("[fuse] dgrad accumulations:"),
        "{tag}:\n{no_b_err}"
    );
    assert!(
        !off_err.contains("[fuse] add+rmsnorm pairs:")
            && !off_err.contains("[fuse] dgrad accumulations:"),
        "{tag}:\n{off_err}"
    );
    let (l_on, l_a, l_b, l_off) = (losses(&on), losses(&no_a), losses(&no_b), losses(&off));
    assert_eq!(l_off.len(), 8, "{tag}: one loss per micro-batch:\n{off}");
    assert!(
        l_off[7] < l_off[0],
        "{tag}: the program did not train: {l_off:?}"
    );
    assert_eq!(l_on, l_off, "{tag}: both fusions vs neither");
    assert_eq!(l_a, l_off, "{tag}: the accumulation alone vs neither");
    assert_eq!(l_b, l_off, "{tag}: the add+norm launch alone vs neither");
    if !no_a_prof.is_empty() {
        assert!(
            !no_a_prof.contains("\"name\":\"nsl_add_rmsnorm"),
            "{tag}: the kill switch still launched it"
        );
    }
    let _ = on_prof.len();
    (on_err, on_prof)
}

/// On the host both rewrites run (the runtime takes the decomposed calls there), so the
/// markers and the losses check the compiler side in the default test run.
#[test]
fn resid_norm_fusions_fire_and_are_bit_exact_cpu() {
    four_arms(
        "cpu",
        &fixture(false, false, false),
        PROD,
        &[],
        "[fuse] add+rmsnorm pairs: 5 (3 forward, 2 backward)",
        "[fuse] dgrad accumulations: 6",
    );
}

/// f32 storage on the GPU: the fused kernel (`nsl_add_rmsnorm_f32`) runs, and the
/// program is bit-identical to the kill switches.
#[test]
#[ignore = "requires CUDA GPU"]
fn resid_norm_fusions_are_bit_exact_under_f32_storage_gpu() {
    let (err, prof) = four_arms(
        "f32",
        &fixture(true, false, false),
        PROD,
        &[],
        "[fuse] add+rmsnorm pairs: 5 (3 forward, 2 backward)",
        "[fuse] dgrad accumulations: 6",
    );
    assert!(
        prof.contains("\"name\":\"nsl_add_rmsnorm_f32"),
        "no fused launch:\n{prof}"
    );
    assert_fused_launches("f32", &err);
    // An f32 program's products are TF32: its sums are the unfused calls by design.
    assert_eq!(counter(&err, "[dgrad-acc] one-gemm="), 0, "{err}");
    assert!(counter(&err, " decomposed=") > 0, "{err}");
}

/// bf16 activation storage with the production flags (the tensor-core attention backward
/// kept on the device under `--deterministic`): the fused kernel reads the f32 residual
/// and the bf16 projection and writes the f32 sum and the bf16 norm output; the
/// accumulating GEMMs take the bf16-storage route through GemmEx and, with
/// `NSL_MATMUL_BF16_LT=1`, through cublasLt. Bit-identical to the kill switches.
#[test]
#[ignore = "requires CUDA GPU"]
fn resid_norm_fusions_are_bit_exact_under_bf16_storage_gpu() {
    for (tag, lt) in [("bf16", "0"), ("bf16lt", "1")] {
        let (err, prof) = four_arms(
            tag,
            &fixture(true, true, false),
            PROD,
            &[("NSL_FLASH_BWD_CPU", "0"), ("NSL_MATMUL_BF16_LT", lt)],
            "[fuse] add+rmsnorm pairs: 5 (3 forward, 2 backward)",
            "[fuse] dgrad accumulations: 6",
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
        // The gate/up sums (bf16 gradients, the storage route) take one GEMM; the
        // fixture's q/k sums are below the eager bf16 mode's intensity gate (TF32).
        assert!(
            counter(&err, "[dgrad-acc] one-gemm=") > 0,
            "{tag}: no one-GEMM sum ran:\n{err}"
        );
    }
}

/// The 500M recipe's shape: dropout 0.1 after both block outputs, so the residual add
/// reads the f32 dropout output (`nsl_add_rmsnorm_ff_fb`); the key-based dropout is
/// untouched and the program stays bit-identical.
#[test]
#[ignore = "requires CUDA GPU"]
fn resid_norm_fusions_compose_with_dropout_under_bf16_storage_gpu() {
    let (err, prof) = four_arms(
        "bf16drop",
        &fixture(true, true, true),
        PROD,
        &[("NSL_FLASH_BWD_CPU", "0")],
        "[fuse] add+rmsnorm pairs: 5 (3 forward, 2 backward)",
        "[fuse] dgrad accumulations: 6",
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
