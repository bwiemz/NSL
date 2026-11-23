//! SwiGLU fusion gates.
//!
//! * P5 item 20 slice B — the gate-backward peephole (bit-exact: Mul(dy, up)
//!   + silu_backward → swiglu_gate_backward) on the UNFUSED graph, i.e. with
//!   `NSL_FUSE_SWIGLU=0`: since the forward fusion, the fused `swiglu` op's
//!   own adjoint emits the gate op directly and the peephole has nothing left
//!   to match.
//! * The forward fusion (`silu(g) * u` → one `swiglu` op, its adjoint lowered
//!   as one launch): with f32 storage it must be bit-identical to the unfused
//!   lowering, on the host (decomposed runtime ops) and on the GPU (the fused
//!   KIR kernels), and its kernels must really run.
//!
//! Trains a real SwiGLU MLP under source-AD and tape-AD (the independent
//! reference) and asserts the trained gate AND up weights agree. GPU variants
//! additionally assert bit-determinism.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_fixture(tag: &str, gpu: bool, extra: &[&str]) -> String {
    run_fixture_env(tag, gpu, extra, &[]).0
}

/// [`run_fixture`] with extra environment; returns (stdout, stderr).
fn run_fixture_env(tag: &str, gpu: bool, extra: &[&str], env: &[(&str, &str)]) -> (String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_swiglu_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/swiglu_parity.nsl"),
    )
    .unwrap();
    if gpu {
        src = src.replace(
            "# GPU_PLACEMENT",
            "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)",
        );
        src = src.replace("m.forward(x)", "m.forward(xg)");
        src = src.replace("l1_loss(pred, y)", "l1_loss(pred, yg)");
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--deterministic"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env_remove("NSL_FUSE_SWIGLU")
        .env_remove("NSL_SWIGLU_ROUND_SILU");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    assert!(
        out.status.success(),
        "run failed (tag={tag}):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// The GPU fixture under `--source-ad` with the kernel profiler on, returning
/// (stderr, kernel_profile.json). The child writes the profile into its working
/// directory at exit; its trace names every kernel `kernel_launch` issued, so a
/// name there means the kernel really ran on the device.
fn run_gpu_fixture_profiled(tag: &str, env: &[(&str, &str)]) -> (String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_swiglu_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/swiglu_parity.nsl"))
        .unwrap()
        .replace("# GPU_PLACEMENT", "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)")
        .replace("m.forward(x)", "m.forward(xg)")
        .replace("l1_loss(pred, y)", "l1_loss(pred, yg)");
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let profile = tmp.join("kernel_profile.json");
    let _ = std::fs::remove_file(&profile);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--deterministic", "--source-ad"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_PROFILE_KERNELS", "1")
        .env_remove("NSL_FUSE_SWIGLU")
        .env_remove("NSL_SWIGLU_ROUND_SILU");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed (tag={tag}):\n{stderr}");
    let json = std::fs::read_to_string(&profile).unwrap_or_else(|e| {
        panic!("no {} after a profiled run (tag={tag}): {e}\n{stderr}", profile.display())
    });
    (stderr, json)
}

/// A NaN or infinite value is a FAILURE, not a token to skip. It prints as
/// `NaN` / `inf` / `-inf`, which the digit split below drops, so an all-NaN
/// arm used to parse as an empty list and compare nothing.
fn parse_between(stdout: &str, begin: &str, end: &str) -> Vec<f64> {
    let after = stdout.split_once(begin).map(|(_, r)| r).unwrap_or("");
    let inner = after.split_once(end).map(|(l, _)| l).unwrap_or("");
    if let Some(t) = inner
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '+'))
        .find(|t| {
            let word = t.trim_start_matches(['-', '+']).to_ascii_lowercase();
            matches!(word.as_str(), "nan" | "inf" | "infinity")
        })
    {
        panic!("non-finite value {t:?} printed between {begin} and {end}");
    }
    inner
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|t| t.parse::<f64>().ok())
        .inspect(|v| assert!(v.is_finite(), "non-finite value {v} printed between {begin} and {end}"))
        .collect()
}

#[test]
fn swiglu_fused_backward_matches_tape_ad_cpu() {
    let fused = run_fixture("cpu_sa", false, &["--source-ad"]);
    let tape = run_fixture("cpu_tape", false, &[]);
    // The fixture's inits, to measure movement against: a weight that never
    // left its init would "match" trivially.
    let inits: [(&str, &str, &str, fn(usize) -> f64); 2] = [
        ("w_gate", "WG_BEGIN", "WG_END", |i| i as f64 * 0.02),
        ("w_up", "WU_BEGIN", "WU_END", |i| i as f64 * 0.01 + 0.05),
    ];
    for (name, b, e, init) in inits {
        let f = parse_between(&fused, b, e);
        let t = parse_between(&tape, b, e);
        assert_eq!(f.len(), 32, "{name}: fused produced {} values", f.len());
        assert_eq!(t.len(), 32, "{name}: tape produced {} values", t.len());
        let moved = t.iter().enumerate().map(|(i, v)| (v - init(i)).abs()).fold(0.0, f64::max);
        assert!(moved > 1e-2, "{name} barely moved off its init ({moved:.2e}): {t:?}");
        // Both paths run the same f64 formulas on the CPU; they differ by f32
        // rounding (~1e-8). Under SGD a wrong silu' moves a weight by a
        // fraction of its displacement, far past 1e-5.
        for (i, (a, b)) in f.iter().zip(&t).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "{name}[{i}] fused={a} vs tape={b} (|Δ|={})",
                (a - b).abs()
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn swiglu_fused_backward_gpu_deterministic_and_matches_reference() {
    let a = run_fixture("gpu_a", true, &["--source-ad"]);
    let b = run_fixture("gpu_b", true, &["--source-ad"]);
    assert_eq!(a, b, "GPU fused SwiGLU backward not deterministic");

    let tape = run_fixture("cpu_ref", false, &[]);
    for (name, bg, en) in [("w_gate", "WG_BEGIN", "WG_END"), ("w_up", "WU_BEGIN", "WU_END")] {
        let f = parse_between(&a, bg, en);
        let t = parse_between(&tape, bg, en);
        assert_eq!(f.len(), 32, "{name}: GPU produced {} values", f.len());
        assert_eq!(t.len(), 32, "{name}: CPU tape reference produced {} values", t.len());
        for (i, (x, y)) in f.iter().zip(&t).enumerate() {
            assert!(
                (x - y).abs() < 2e-3,
                "{name}[{i}] gpu-fused={x} vs cpu-tape={y} (|Δ|={})",
                (x - y).abs()
            );
        }
    }

    // ANTI-VACUITY — the fused backward must actually RUN on the device.
    //
    // This gate once compared the UNFUSED path for as long as the peephole
    // failed to fire (see the CPU gate below), and nothing in it noticed: the
    // values above are equally right either way. The firing proof was added to
    // the CPU gate only. Here it is on silicon, at both layers:
    //  * the codegen marker says the fused op was EMITTED;
    //  * the kernel profile says `nsl_swiglu_gate_backward_f32` LAUNCHED.
    // The kill-switch run is the control: without it, a name in the profile
    // could come from some other op and the check would not discriminate.
    //
    // The peephole matches the UNFUSED graph, so these runs keep it
    // (`NSL_FUSE_SWIGLU=0`); the fused graph's kernels are checked by
    // `swiglu_forward_fusion_gpu_is_bit_exact_and_its_kernels_run`.
    let kernel = "\"name\":\"nsl_swiglu_gate_backward_f32\"";
    let (err_on, prof_on) = run_gpu_fixture_profiled("gpu_prof_on", &[("NSL_FUSE_SWIGLU", "0")]);
    assert!(
        err_on.contains("[fuse] swiglu gate-backward pairs: "),
        "the GPU source-AD run did not report fusing any SwiGLU gate-backward pair:\n{err_on}"
    );
    assert!(
        prof_on.contains(kernel),
        "codegen fused the pair, but the fused kernel never launched on the device \
         (no {kernel} in the kernel profile) — the values above certified the unfused path"
    );
    let (_err_off, prof_off) =
        run_gpu_fixture_profiled("gpu_prof_off", &[("NSL_FUSE_SWIGLU", "0"), ("NSL_FUSE_SWIGLU_GATE", "0")]);
    assert!(
        !prof_off.contains(kernel),
        "control: with NSL_FUSE_SWIGLU_GATE=0 the fused kernel still launched, so the \
         profile check above cannot tell the fused path from the unfused one"
    );
}

/// The peephole must actually fire on the fixture, and must be bit-exact:
/// the fused source-AD run prints the same weights, to the last digit, as a
/// source-AD run with `NSL_FUSE_SWIGLU_GATE=0`. (It used to never fire —
/// `MulElementwise` wraps the product in `reduce_to_shape`, which the
/// peephole did not see through — so the gates above compared the unfused
/// path with itself.)
#[test]
fn swiglu_peephole_fires_and_is_bit_exact_cpu() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_swiglu_fires_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::copy(root.join("crates/nsl-cli/tests/fixtures/swiglu_parity.nsl"), tmp.join("prog.nsl")).unwrap();
    let run = |fuse: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
        // The peephole matches the unfused graph (see the module header).
        cmd.args(["run", "--deterministic", "--source-ad", "prog.nsl"])
            .current_dir(&tmp)
            .env("NSL_STDLIB_PATH", root.join("stdlib"))
            .env("NSL_FUSE_SWIGLU", "0");
        if !fuse {
            cmd.env("NSL_FUSE_SWIGLU_GATE", "0");
        }
        let out = cmd.output().expect("spawn nsl run");
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "run failed (fuse={fuse}):\n{stderr}");
        (String::from_utf8_lossy(&out.stdout).into_owned(), stderr)
    };
    let (fused_out, fused_err) = run(true);
    let (plain_out, plain_err) = run(false);
    assert!(
        fused_err.contains("[fuse] swiglu gate-backward pairs: 1"),
        "the SwiGLU gate peephole did not fire on the fixture:\n{fused_err}"
    );
    assert!(
        !plain_err.contains("[fuse] swiglu gate-backward"),
        "NSL_FUSE_SWIGLU_GATE=0 did not disable the peephole:\n{plain_err}"
    );
    for (b, e) in [("WG_BEGIN", "WG_END"), ("WU_BEGIN", "WU_END")] {
        let section = |s: &str| s.split_once(b).and_then(|(_, r)| r.split_once(e)).map(|(v, _)| v.to_string());
        let (f, p) = (section(&fused_out), section(&plain_out));
        assert!(f.is_some(), "{b} missing from fused output:\n{fused_out}");
        // A text compare passes two identical NaN sections; `parse_between`
        // refuses a non-finite value.
        for out in [&fused_out, &plain_out] {
            parse_between(out, b, e);
        }
        assert_eq!(f, p, "{b}: fused and unfused weights differ");
    }
}

/// The `[WG|WU]_BEGIN .. _END` sections of a run's stdout, refusing a
/// non-finite value (a text compare would pass two identical NaN sections).
fn weight_sections(out: &str) -> Vec<String> {
    let mut v = Vec::new();
    for (b, e) in [("WG_BEGIN", "WG_END"), ("WU_BEGIN", "WU_END")] {
        let sec = out
            .split_once(b)
            .and_then(|(_, r)| r.split_once(e))
            .map(|(v, _)| v.to_string())
            .unwrap_or_else(|| panic!("{b} missing:\n{out}"));
        assert_eq!(parse_between(out, b, e).len(), 32, "{b}: 32 values");
        v.push(sec);
    }
    v
}

/// The forward fusion fires on the fixture, and on the host (whose runtime
/// ops decompose) it is bit-exact against the unfused lowering, with the
/// backward paired and unpaired.
#[test]
fn swiglu_forward_fusion_fires_and_is_bit_exact_cpu() {
    let (fused_out, fused_err) = run_fixture_env("fwd_cpu_on", false, &["--source-ad"], &[]);
    let (fwd_out, fwd_err) = run_fixture_env("fwd_cpu_fwd", false, &["--source-ad"], &[("NSL_FUSE_SWIGLU", "fwd")]);
    let (plain_out, plain_err) = run_fixture_env("fwd_cpu_off", false, &["--source-ad"], &[("NSL_FUSE_SWIGLU", "0")]);
    assert!(
        fused_err.contains("[fuse] swiglu forward: 1"),
        "the forward SwiGLU fusion did not fire on the fixture:\n{fused_err}"
    );
    assert!(fwd_err.contains("[fuse] swiglu forward: 1"), "{fwd_err}");
    assert!(
        !plain_err.contains("[fuse] swiglu forward"),
        "NSL_FUSE_SWIGLU=0 did not disable the forward fusion:\n{plain_err}"
    );
    // The fused op's adjoint emits the gate op itself: nothing for the
    // peephole. Unfused, the peephole fires as before.
    assert!(!fused_err.contains("[fuse] swiglu gate-backward pairs:"), "{fused_err}");
    assert!(plain_err.contains("[fuse] swiglu gate-backward pairs: 1"), "{plain_err}");
    let plain = weight_sections(&plain_out);
    assert_eq!(weight_sections(&fused_out), plain, "fused and unfused weights differ");
    assert_eq!(weight_sections(&fwd_out), plain, "forward-only fusion and unfused weights differ");
}

/// The fused forward under the tape is the independent reference again: the
/// trained weights agree with tape AD, as the unfused ones do.
#[test]
fn swiglu_forward_fusion_matches_tape_ad_cpu() {
    let (fused, err) = run_fixture_env("fwd_cpu_sa", false, &["--source-ad"], &[]);
    assert!(err.contains("[fuse] swiglu forward: 1"), "{err}");
    let tape = run_fixture("fwd_cpu_tape", false, &[]);
    for (name, b, e) in [("w_gate", "WG_BEGIN", "WG_END"), ("w_up", "WU_BEGIN", "WU_END")] {
        let (f, t) = (parse_between(&fused, b, e), parse_between(&tape, b, e));
        assert_eq!(f.len(), 32);
        assert_eq!(t.len(), 32);
        for (i, (x, y)) in f.iter().zip(&t).enumerate() {
            assert!((x - y).abs() < 1e-5, "{name}[{i}] fused={x} vs tape={y}");
        }
    }
}

/// On the GPU with f32 storage the fused kernels are bit-identical to the
/// unfused launches: the fused run (paired backward), the forward-only run
/// and the unfused run train the same weights to the last digit. The kernel
/// profile proves each fused kernel launched where it should and never
/// under the kill switch.
#[test]
#[ignore = "requires CUDA GPU"]
fn swiglu_forward_fusion_gpu_is_bit_exact_and_its_kernels_run() {
    let (on_err, on_prof) = run_gpu_fixture_profiled("fwd_gpu_on", &[]);
    let (fwd_err, fwd_prof) = run_gpu_fixture_profiled("fwd_gpu_fwd", &[("NSL_FUSE_SWIGLU", "fwd")]);
    let (off_err, off_prof) = run_gpu_fixture_profiled("fwd_gpu_off", &[("NSL_FUSE_SWIGLU", "0")]);
    assert!(on_err.contains("[fuse] swiglu forward: 1"), "{on_err}");
    assert!(fwd_err.contains("[fuse] swiglu forward: 1"), "{fwd_err}");
    assert!(!off_err.contains("[fuse] swiglu forward"), "{off_err}");
    let has = |prof: &str, stem: &str| prof.contains(&format!("\"name\":\"{stem}"));
    // The forward kernel (any width) and the paired backward.
    assert!(has(&on_prof, "nsl_swiglu_f32"), "the fused forward kernel never launched:\n{on_prof}");
    assert!(has(&on_prof, "nsl_swiglu_backward_f32"), "the paired backward never launched:\n{on_prof}");
    assert!(!has(&on_prof, "nsl_swiglu_up_backward_f32"), "an unpaired up adjoint launched:\n{on_prof}");
    // Forward-only: the two adjoints as two launches.
    assert!(has(&fwd_prof, "nsl_swiglu_f32"), "{fwd_prof}");
    assert!(has(&fwd_prof, "nsl_swiglu_up_backward_f32"), "{fwd_prof}");
    assert!(has(&fwd_prof, "nsl_swiglu_gate_backward_f32"), "{fwd_prof}");
    assert!(!has(&fwd_prof, "nsl_swiglu_backward_f32"), "{fwd_prof}");
    // The kill switch: none of the family.
    for stem in ["nsl_swiglu_f32", "nsl_swiglu_backward_f32", "nsl_swiglu_up_backward_f32"] {
        assert!(!has(&off_prof, stem), "NSL_FUSE_SWIGLU=0 still launched {stem}:\n{off_prof}");
    }

    let gpu = |tag: &str, env: &[(&str, &str)]| run_fixture_env(tag, true, &["--source-ad"], env).0;
    let on = weight_sections(&gpu("fwd_gpu_w_on", &[]));
    assert_eq!(on, weight_sections(&gpu("fwd_gpu_w_on2", &[])), "the fused GPU run is not deterministic");
    assert_eq!(on, weight_sections(&gpu("fwd_gpu_w_fwd", &[("NSL_FUSE_SWIGLU", "fwd")])), "fwd-only differs");
    assert_eq!(on, weight_sections(&gpu("fwd_gpu_w_off", &[("NSL_FUSE_SWIGLU", "0")])), "fused vs unfused differ");
}

/// Two pre-norm SwiGLU blocks with residual adds (the bf16 storage gate's
/// fixture, its blocks in a `blocks` array so that `--checkpoint-blocks`
/// plans them), trained four micro-batches per optimizer step on the GPU;
/// `precision` is the text after `grad_accumulation=4`.
fn mlp_blocks_fixture(precision: &str) -> String {
    r#"from nsl.nn.norms import RMSNorm
from nsl.nn.losses import mse_loss

model Block:
    norm: RMSNorm = RMSNorm(32)
    w_gate: Tensor = ((arange(2048).reshape([32, 64])) * 0.00007) - 0.06
    w_up: Tensor = ((arange(2048).reshape([32, 64])) * 0.00005) - 0.045
    w_down: Tensor = ((arange(2048).reshape([64, 32])) * 0.00004) - 0.04

    fn forward(self, x: Tensor) -> Tensor:
        let n = self.norm.forward(x)
        let gate = silu(n @ self.w_gate)
        let up = n @ self.w_up
        return x + (gate * up) @ self.w_down

model Net:
    blocks: [Block; 2] = Block()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
m.to(cuda)
let x = ((arange(1024).reshape([2, 16, 32])) * 0.003) - 1.3
let y = ((arange(1024).reshape([2, 16, 32])) * 0.002) - 0.9
let xd = x.to(cuda)
let yd = y.to(cuda)

train(model=m, epochs=8, grad_accumulation=4PRECISION):
    optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-2, weight_decay=0.0)
    step(batch):
        let pred = m.forward(xd)
        let loss = mse_loss(pred, yd)
    callbacks:
        on_step(step, loss):
            print(loss)
"#
    .replace("PRECISION", precision)
}

/// `nsl run` of [`mlp_blocks_fixture`] (`precision="bf16"`) with the
/// production flag line and `--deterministic`: (stdout, stderr,
/// kernel_profile.json).
fn run_bf16_mlp(tag: &str, env: &[(&str, &str)]) -> (String, String, String) {
    run_mlp_blocks(tag, r#", precision="bf16""#, env)
}

fn run_mlp_blocks(tag: &str, precision: &str, env: &[(&str, &str)]) -> (String, String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_swiglu_bf16_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, mlp_blocks_fixture(precision)).unwrap();
    let profile = tmp.join("kernel_profile.json");
    let _ = std::fs::remove_file(&profile);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args([
        "run",
        "--deterministic",
        "--source-ad",
        "--checkpoint-blocks",
        "--checkpoint-selective",
        "--fuse-rmsnorm-backward",
        "--fuse-wgrad-accum",
    ])
    .arg(&prog)
    .current_dir(&tmp)
    .env("NSL_STDLIB_PATH", root.join("stdlib"))
    .env("NSL_PROFILE_KERNELS", "1")
    .env_remove("NSL_MATMUL_BF16")
    .env_remove("NSL_FUSE_SWIGLU")
    .env_remove("NSL_SWIGLU_ROUND_SILU");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed (tag={tag}):\n{stderr}");
    let prof = std::fs::read_to_string(&profile).unwrap_or_default();
    (String::from_utf8_lossy(&out.stdout).into_owned(), stderr, prof)
}

fn bf16_losses(out: &str) -> Vec<f64> {
    out.lines()
        .filter_map(|l| l.trim().strip_prefix("tensor([").and_then(|r| r.strip_suffix("])")))
        .map(|v| {
            let x: f64 = v.parse().unwrap_or_else(|_| panic!("unparsable loss {v:?}"));
            assert!(x.is_finite(), "non-finite loss {v}");
            x
        })
        .collect()
}

/// bf16 activation storage, production flags. With the diagnostic that
/// reinstates the unfused chain's rounding of `silu(g)`, the fused program
/// prints the unfused program's losses bit for bit: CCR's recompute folded
/// into the backward, the grouped launch, the storage plan of the fused ops
/// and the shape stub are all exact. Without it the fused program carries
/// one rounding fewer and tracks the unfused one to that rounding; it is
/// deterministic run to run. The profile shows the grouped launch (adjoints
/// and the recomputed value) and no forward recompute of `p`.
#[test]
#[ignore = "requires CUDA GPU"]
fn swiglu_fusion_under_bf16_storage_is_exact_up_to_the_silu_rounding() {
    let (fused, fused_err, fused_prof) = run_bf16_mlp("fused", &[]);
    let (fused2, _, _) = run_bf16_mlp("fused2", &[]);
    let (rounded, _, _) = run_bf16_mlp("rounded", &[("NSL_SWIGLU_ROUND_SILU", "1")]);
    let (plain, plain_err, plain_prof) = run_bf16_mlp("plain", &[("NSL_FUSE_SWIGLU", "0")]);
    for err in [&fused_err, &plain_err] {
        assert!(err.contains("[bf16-storage] 2 SwiGLU MLP block(s) stored bf16"), "{err}");
        assert!(!err.contains("widens to f32, computes and narrows"), "a dtype-explicit op fell back:\n{err}");
    }
    assert!(fused_err.contains("[fuse] swiglu forward: 2"), "{fused_err}");
    assert!(fused_err.contains("[fuse] swiglu recompute folded into the backward: 2"), "{fused_err}");
    assert!(!plain_err.contains("[fuse] swiglu recompute folded"), "{plain_err}");
    // CCR serves `p`'s remaining shape read with a stub instead of a clone.
    assert!(fused_err.contains("2 read for shape only"), "{fused_err}");
    for err in [&fused_err, &plain_err] {
        assert!(err.contains("[wgrad-fusion] 6 chain(s) fused"), "a weight-gradient chain stopped fusing:
{err}");
    }
    let has = |prof: &str, stem: &str| prof.contains(&format!("\"name\":\"{stem}"));
    assert!(has(&fused_prof, "nsl_swiglu_bb_b"), "no fused forward:\n{fused_prof}");
    assert!(has(&fused_prof, "nsl_swiglu_backward_h_bbb_b"), "no grouped backward:\n{fused_prof}");
    for stem in ["nsl_silu_b_b", "nsl_swiglu_gate_backward_bbb_b", "nsl_swiglu_up_backward"] {
        assert!(!has(&fused_prof, stem), "the fused program still launched {stem}:\n{fused_prof}");
    }
    assert!(has(&plain_prof, "nsl_silu_b_b") && !has(&plain_prof, "nsl_swiglu_bb_b"), "{plain_prof}");

    let (f, f2, r, p) = (bf16_losses(&fused), bf16_losses(&fused2), bf16_losses(&rounded), bf16_losses(&plain));
    eprintln!("fused   {f:?}\nrounded {r:?}\nunfused {p:?}");
    assert_eq!(p.len(), 8, "one loss per micro-batch:\n{plain}");
    assert!(p[7] < p[0], "the unfused program did not train: {p:?}");
    assert_eq!(f, f2, "the fused bf16 program is not deterministic");
    assert_eq!(r, p, "with the silu rounding reinstated the fused program must equal the unfused one");
    assert_ne!(f, p, "the fused kernels should skip one rounding (vacuous comparison otherwise)");
    for (i, (x, y)) in f.iter().zip(&p).enumerate() {
        assert!((x - y).abs() <= 1e-2 * y.abs(), "step {i}: fused {x} vs unfused {y}");
    }
}

/// f32 storage, production flags with checkpointing: the fused program (the
/// grouped backward launch producing the recomputed forward value, CCR's
/// shape stub in place of its clone) prints the unfused program's losses bit
/// for bit, since the f32 kernels compute the unfused launches' values.
#[test]
#[ignore = "requires CUDA GPU"]
fn swiglu_fusion_with_checkpointing_is_bit_exact_under_f32_storage() {
    let (fused, fused_err, fused_prof) = run_mlp_blocks("f32_fused", "", &[]);
    let (plain, plain_err, _) = run_mlp_blocks("f32_plain", "", &[("NSL_FUSE_SWIGLU", "0")]);
    assert!(fused_err.contains("[fuse] swiglu recompute folded into the backward: 2"), "{fused_err}");
    assert!(fused_err.contains("[wgrad-fusion] 6 chain(s) fused"), "the moved weight-gradient chains must still fuse:\n{fused_err}");
    assert!(plain_err.contains("[wgrad-fusion] 6 chain(s) fused"), "{plain_err}");
    assert!(
        fused_prof.contains("\"name\":\"nsl_swiglu_backward_h_f32"),
        "no grouped backward launch:\n{fused_prof}"
    );
    let (f, p) = (bf16_losses(&fused), bf16_losses(&plain));
    assert_eq!(p.len(), 8, "{plain}");
    assert!(p[7] < p[0], "the unfused program did not train: {p:?}");
    assert_eq!(f, p, "fused and unfused f32 programs differ");
}
