//! `train(..., precision="bf16", param_dtype=...)`: bf16 parameter storage on
//! the normal schedule (design `srbf16-design.md`, steps 2 and F).
//!
//! * **Planning** (CPU build): the weight matrices of a two-block
//!   transformer (attention and SwiGLU halves) are selected, the norm gains
//!   are not, and the compile reports it.
//! * **Refusals** (CPU build): every composition the design refuses (6.2) is
//!   a compile error naming what to drop, and the test-only master modes need
//!   their environment switch.
//! * **GPU** (`#[ignore]`, `--features cuda`), all `--deterministic` with
//!   `NSL_FLASH_BWD_CPU=0` on the production flag line, 50 optimizer updates
//!   with clipping, weight decay and a warmup-cosine schedule:
//!   - `bf16-master` trains byte-identically to f32 parameters: every loss,
//!     the checkpoint after update 50 (`.nslm` whole; `.optim` header and
//!     moments, the fingerprint aside), the model saved after the block; the
//!     cast cache registers no matrix in master mode; no dtype-explicit op
//!     takes the fallback.
//!   - `bf16-split` trains byte-identically to the master mode whose image
//!     rounds as the split does (`bf16-master-tz`): the split's storage and
//!     reassembly are exact over a run, not only in the kernel gates.
//!   - `bf16-split` resumed from a checkpoint continues the uninterrupted
//!     run bit for bit; a resume across parameter storages is refused.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const PROD: &[&str] = &[
    "--source-ad",
    "--checkpoint-blocks",
    "--checkpoint-selective",
    "--fuse-rmsnorm-backward",
    "--fuse-wgrad-accum",
];

/// Two pre-norm blocks, the stdlib's grouped-query attention (RoPE, GQA,
/// head_dim 64) and a SwiGLU MLP, with random weights and data (the GPU runs
/// pass one `--seed`, so every arm starts from the same values). `header` is spliced after
/// `grad_accumulation=4` in the train header, `epochs` micro-batches run.
fn fixture(header: &str, epochs: usize, cuda: bool, tail: &str) -> String {
    let (to_device, xd, yd) = if cuda {
        ("m.to(cuda)\n", "x.to(cuda)", "y.to(cuda)")
    } else {
        ("", "x", "y")
    };
    format!(
        r#"from nsl.nn.norms import RMSNorm
from nsl.nn.gqa import GroupedQueryAttention
from nsl.nn.losses import mse_loss

model Block:
    attn_norm: RMSNorm = RMSNorm(128)
    attn: GroupedQueryAttention = GroupedQueryAttention(128, 2, 1, 64, 0.0, 64, 10000.0, 2)
    ffn_norm: RMSNorm = RMSNorm(128)
    w_gate: Tensor = randn([128, 256]) * 0.06
    w_up: Tensor = randn([128, 256]) * 0.06
    w_down: Tensor = randn([256, 128]) * 0.04

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + self.attn.forward(self.attn_norm.forward(x), true)
        let n = self.ffn_norm.forward(h)
        return h + (silu(n @ self.w_gate) * (n @ self.w_up)) @ self.w_down

model Net:
    b1: Block = Block()
    b2: Block = Block()

    fn forward(self, x: Tensor) -> Tensor:
        return self.b2.forward(self.b1.forward(x))

let m = Net()
{to_device}let x = randn([2, 32, 128])
let y = randn([2, 32, 128]) * 0.5
let xd = {xd}
let yd = {yd}

print("LOSS_STREAM_BEGIN")
train(model=m, epochs={epochs}, grad_accumulation=4, grad_clip=1.0, schedule_unit="optimizer_step", precision="bf16"{header}):
    optimizer: AdamW(lr=0.002, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)
    scheduler: warmup_cosine(warmup_steps=5, total_steps=60, min_lr=0.0002)
    step(batch):
        let pred = m.forward(xd)
        let loss = mse_loss(pred, yd)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")
{tail}"#
    )
}

struct Run {
    out: String,
    ok: bool,
}

fn nsl(args: &[&str], src: &str, dir: &Path, stem: &str, env: &[(&str, &str)]) -> Run {
    let root = repo_root();
    std::fs::create_dir_all(dir).expect("mkdir fixture dir");
    let fixture = dir.join(format!("{stem}.nsl"));
    std::fs::write(&fixture, src).expect("write fixture");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(args).arg(&fixture);
    if args.first() == Some(&"build") {
        cmd.args(["--emit-obj", "-o"])
            .arg(dir.join(format!("{stem}.o")));
    }
    cmd.current_dir(dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env_remove("NSL_MATMUL_BF16")
        .env_remove("NSL_PARAM_DTYPE_TEST_MODES")
        .env_remove("NSL_FASE_MULTI_STEP")
        .env_remove("NSL_FASE_FUSED_STEP");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let o = cmd.output().expect("spawn nsl");
    Run {
        out: format!(
            "{}\n{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
        ok: o.status.success(),
    }
}

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "nsl_param_storage_gate_{tag}_{}",
        std::process::id()
    ))
}

const SPLIT: &str = r#", param_dtype="bf16-split""#;
const MASTER: &str = r#", param_dtype="bf16-master""#;
const MASTER_TZ: &str = r#", param_dtype="bf16-master-tz""#;
const TEST_MODES: (&str, &str) = ("NSL_PARAM_DTYPE_TEST_MODES", "1");

fn build(header: &str, extra: &[&str], env: &[(&str, &str)], tag: &str) -> Run {
    let mut args = vec!["build"];
    // `--optim-state-offload` cannot be combined with the fused weight
    // gradient (a clap conflict that would speak first).
    let offload = extra.contains(&"--optim-state-offload");
    args.extend(
        PROD.iter()
            .copied()
            .filter(|f| !(offload && *f == "--fuse-wgrad-accum")),
    );
    args.extend_from_slice(extra);
    nsl(&args, &fixture(header, 8, false, ""), &tmp(tag), tag, env)
}

/// The 14 matrices (q, k, v, o and the three MLP projections, two blocks)
/// are stored bf16; the four norm gains are not.
#[test]
fn the_weight_matrices_are_selected_and_the_norm_gains_are_not() {
    let r = build(SPLIT, &[], &[], "plan_split");
    assert!(r.ok, "bf16-split build failed:\n{}", r.out);
    assert!(
        r.out
            .contains("[param-storage] 14 of 18 parameters stored bf16-split"),
        "{}",
        r.out
    );
    // The q and k dgrads read the f32 RoPE adjoints: the only GEMMs whose
    // sole bf16 operand is a stored parameter (2 per block).
    assert!(
        r.out
            .contains("(4 of the GEMMs reading them keep the eager bf16 mode's"),
        "{}",
        r.out
    );
    let f32 = build("", &[], &[], "plan_f32");
    assert!(
        f32.ok && !f32.out.contains("[param-storage]"),
        "{}",
        f32.out
    );
}

#[test]
fn param_dtype_refusals_are_compile_errors() {
    let expect = |r: Run, needle: &str, what: &str| {
        assert!(!r.ok, "{what} built:\n{}", r.out);
        assert!(
            r.out.contains(needle),
            "{what}: expected `{needle}` in:\n{}",
            r.out
        );
    };
    // No precision="bf16".
    let mut args = vec!["build"];
    args.extend_from_slice(PROD);
    let src = fixture(SPLIT, 8, false, "").replace(r#", precision="bf16""#, "");
    expect(
        nsl(&args, &src, &tmp("no_precision"), "no_precision", &[]),
        "requires precision=\"bf16\"",
        "no precision",
    );
    // The reserved SR spelling and an unknown value.
    expect(
        build(r#", param_dtype="bf16-sr""#, &[], &[], "sr"),
        "reserved and not implemented yet",
        "bf16-sr",
    );
    expect(
        build(r#", param_dtype="bf16""#, &[], &[], "bad"),
        "must be \"f32\" or \"bf16-split\"",
        "bf16",
    );
    // The test-only master modes without their switch, and with it.
    expect(
        build(MASTER, &[], &[], "master_no_env"),
        "NSL_PARAM_DTYPE_TEST_MODES=1",
        "bf16-master",
    );
    expect(
        build(MASTER_TZ, &[], &[], "tz_no_env"),
        "NSL_PARAM_DTYPE_TEST_MODES=1",
        "bf16-master-tz",
    );
    let ok = build(MASTER, &[], &[TEST_MODES], "master_env");
    assert!(
        ok.ok && ok.out.contains("parameters stored bf16-master"),
        "{}",
        ok.out
    );
    // Each composition that takes another optimizer path or another device
    // layout.
    for (flags, env, needle) in [
        (
            &["--optim-state-offload"][..],
            &[][..],
            "--optim-state-offload",
        ),
        (&["--zero-stage", "1"][..], &[][..], "--zero-stage"),
        (
            &["--training-reference"][..],
            &[][..],
            "--training-reference",
        ),
        (
            &[][..],
            &[("NSL_FASE_MULTI_STEP", "0")][..],
            "NSL_FASE_MULTI_STEP=0",
        ),
        (
            &[][..],
            &[("NSL_FASE_FUSED_STEP", "0")][..],
            "NSL_FASE_FUSED_STEP=0",
        ),
        (
            &[][..],
            &[("NSL_FASE_FUSED_OVERRIDE", "1,0")][..],
            "NSL_FASE_FUSED_OVERRIDE",
        ),
    ] {
        let tag = format!(
            "refuse_{}",
            needle.trim_start_matches('-').replace(['=', ' ', '-'], "_")
        );
        expect(
            build(SPLIT, flags, env, &tag),
            &format!("does not compose with {needle}"),
            needle,
        );
    }
    // The CSLA mode's flag (refused by precision="bf16" too).
    expect(
        build(SPLIT, &["--param-dtype", "bf16-sr"], &[], "csla"),
        "bf16-sr",
        "--param-dtype bf16-sr",
    );
    // An optimizer other than the fused AdamW step, and a step with no
    // accumulation (no FASE Deferred plan). Without the wgrad fusion flag,
    // whose own admission would speak first.
    let sgd = fixture(SPLIT, 8, false, "").replace(
        "AdamW(lr=0.002, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)",
        "SGD(lr=0.01)",
    );
    assert!(sgd.contains("SGD("), "fixture edit did not apply");
    let args = ["build", "--source-ad"];
    expect(
        nsl(&args, &sgd, &tmp("sgd"), "sgd", &[]),
        "needs the fused multi-tensor AdamW",
        "SGD",
    );
    let no_accum = fixture(SPLIT, 8, false, "").replace("grad_accumulation=4, ", "");
    assert_ne!(
        no_accum,
        fixture(SPLIT, 8, false, ""),
        "fixture edit did not apply"
    );
    expect(
        nsl(&args, &no_accum, &tmp("no_accum"), "no_accum", &[]),
        "needs the fused multi-tensor AdamW",
        "no accumulation",
    );
}

// ---------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------

/// `--deterministic` routes the attention backward to the host unless
/// `NSL_FLASH_BWD_CPU=0`, which would take bf16 attention away.
const DET_ENV: [(&str, &str); 2] = [
    ("NSL_FLASH_BWD_CPU", "0"),
    ("NSL_BF16_CAST_CACHE_COUNTER", "1"),
];

fn run_gpu(
    header: &str,
    epochs: usize,
    dir: &Path,
    stem: &str,
    ck: &str,
    env: &[(&str, &str)],
) -> Run {
    let mut args = vec!["run", "--deterministic", "--seed", "7"];
    args.extend_from_slice(PROD);
    let mut envs: Vec<(&str, &str)> = DET_ENV.to_vec();
    envs.extend_from_slice(env);
    // After the block: the model saved, and an eager forward of it (the
    // validation a recipe runs), which reads f32 parameters only once the
    // teardown has widened them back (mutation M9 crashes it).
    let tail = format!(
        "model_save(m, \"{stem}_final.nslm\")\nlet ev = mse_loss(m.forward(xd), yd)\nprint(\"EVAL_LOSS\")\nprint(ev)\n"
    );
    let src = fixture(&format!("{header}{ck}"), epochs, true, &tail);
    nsl(&args, &src, dir, stem, &envs)
}

fn losses(out: &str) -> String {
    out.split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default()
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{}: {e}", dir.join(name).display()))
}

/// A `.optim` sidecar split into its header JSON with the execution
/// fingerprint blanked (it records the parameter storage, which is the one
/// thing that differs on purpose) and its data section.
fn optim_parts(bytes: &[u8]) -> (String, Vec<u8>) {
    assert_eq!(&bytes[0..4], b"NSLO");
    let hsize = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let header = String::from_utf8_lossy(&bytes[16..16 + hsize]).into_owned();
    let start = 16 + hsize + (64 - (16 + hsize) % 64) % 64;
    let exec = header
        .split_once("\"exec\":\"")
        .map(|(_, r)| r.split_once('"').unwrap().0.to_string())
        .expect("exec field");
    (header.replace(&exec, "<exec>"), bytes[start..].to_vec())
}

fn cache_recasts(out: &str) -> u64 {
    let line = out
        .lines()
        .find(|l| l.contains("[bf16-cast-cache] hits="))
        .unwrap_or_else(|| panic!("no cast-cache line:\n{out}"));
    line.split("recasts=")
        .nth(1)
        .and_then(|r| r.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .expect("recasts")
}

/// 50 updates: 200 micro-batches at accumulation 4; a checkpoint every 25.
const EPOCHS: usize = 200;
const CK: &str = r#", checkpoint_save="ck.nslm", checkpoint_every=25"#;

/// Two arms of the 50-update run must agree byte for byte.
fn assert_identical_runs(a: &Run, b: &Run, da: &Path, db: &Path, sa: &str, sb: &str, what: &str) {
    assert!(a.ok, "{sa} failed:\n{}", a.out);
    assert!(b.ok, "{sb} failed:\n{}", b.out);
    for r in [a, b] {
        assert!(
            !r.out.contains("widens to f32, computes and narrows"),
            "a dtype-explicit op took the fallback:\n{}",
            r.out
        );
    }
    let (la, lb) = (losses(&a.out), losses(&b.out));
    assert_eq!(
        la.lines().count(),
        EPOCHS,
        "one loss per micro-batch:\n{}",
        a.out
    );
    assert_eq!(la, lb, "{what}: the loss streams differ");
    assert_eq!(
        read(da, "ck.nslm"),
        read(db, "ck.nslm"),
        "{what}: the checkpoint .nslm after update 50 differs"
    );
    let (ha, xa) = optim_parts(&read(da, "ck.nslm.optim"));
    let (hb, xb) = optim_parts(&read(db, "ck.nslm.optim"));
    assert_eq!(
        ha, hb,
        "{what}: the .optim headers differ (fingerprint aside)"
    );
    assert_eq!(xa, xb, "{what}: the AdamW moments differ");
    assert_eq!(
        read(da, &format!("{sa}_final.nslm")),
        read(db, &format!("{sb}_final.nslm")),
        "{what}: the model after the block differs"
    );
    let eval = |o: &str| {
        o.split_once("EVAL_LOSS")
            .map(|(_, r)| r.lines().nth(1).unwrap_or("").to_string())
    };
    assert!(
        eval(&a.out).is_some(),
        "no evaluation after the block:\n{}",
        a.out
    );
    assert_eq!(
        eval(&a.out),
        eval(&b.out),
        "{what}: the evaluation after the block differs"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_master_trains_byte_identically_to_f32_parameters_gpu() {
    let (df, dm) = (tmp("det_f32"), tmp("det_master"));
    let f32 = run_gpu("", EPOCHS, &df, "f32", CK, &[]);
    let master = run_gpu(MASTER, EPOCHS, &dm, "master", CK, &[TEST_MODES]);
    assert!(
        master.out.contains("parameter storage: bf16-master"),
        "the banner is missing:\n{}",
        master.out
    );
    assert!(
        master
            .out
            .contains("[param-storage] 14 parameters stored bf16-master"),
        "{}",
        master.out
    );
    assert!(
        master
            .out
            .contains("[param-storage] teardown: 14 parameters widened back"),
        "{}",
        master.out
    );
    assert_identical_runs(
        &f32,
        &master,
        &df,
        &dm,
        "f32",
        "master",
        "bf16-master vs f32",
    );
    // The f32 arm re-casts each matrix's image after every update; the
    // master arm stores the image and the cache never sees a matrix.
    assert!(cache_recasts(&f32.out) >= 14 * 49, "{}", f32.out);
    assert_eq!(cache_recasts(&master.out), 0, "{}", master.out);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_split_trains_byte_identically_to_the_tz_master_gpu() {
    let (dt, ds) = (tmp("det_tz"), tmp("det_split"));
    let tz = run_gpu(MASTER_TZ, EPOCHS, &dt, "tz", CK, &[TEST_MODES]);
    let split = run_gpu(SPLIT, EPOCHS, &ds, "split", CK, &[]);
    assert!(
        split.out.contains("parameter storage: bf16-split"),
        "the banner is missing:\n{}",
        split.out
    );
    assert!(
        split
            .out
            .contains("[param-storage] 14 parameters stored bf16-split"),
        "{}",
        split.out
    );
    assert_identical_runs(
        &tz,
        &split,
        &dt,
        &ds,
        "tz",
        "split",
        "bf16-split vs bf16-master-tz",
    );
    assert_eq!(cache_recasts(&split.out), 0, "{}", split.out);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_split_resumes_like_an_uninterrupted_run_gpu() {
    let dir = tmp("resume_split");
    // 20 updates, saving at the end.
    let first = run_gpu(
        SPLIT,
        80,
        &dir,
        "first",
        r#", checkpoint_save="half.nslm", checkpoint_every=20"#,
        &[],
    );
    assert!(first.ok, "{}", first.out);
    // The resumed run continues to 40 updates (`epochs` is the run's total).
    let resumed = run_gpu(
        SPLIT,
        160,
        &dir,
        "resumed",
        r#", checkpoint_load="half.nslm""#,
        &[],
    );
    assert!(resumed.ok, "{}", resumed.out);
    let straight = run_gpu(SPLIT, 160, &dir, "straight", "", &[]);
    assert!(straight.ok, "{}", straight.out);
    let (ls, lr) = (losses(&straight.out), losses(&resumed.out));
    let tail: Vec<&str> = ls.lines().skip(80).collect();
    assert_eq!(
        lr.lines().collect::<Vec<_>>(),
        tail,
        "the resumed losses are not the uninterrupted run's"
    );
    assert_eq!(
        read(&dir, "resumed_final.nslm"),
        read(&dir, "straight_final.nslm"),
        "resumed θ differs from the uninterrupted run"
    );
    // Across parameter storages, a resume is refused by the fingerprint.
    let f32_dir = tmp("resume_f32");
    let f32 = run_gpu(
        "",
        80,
        &f32_dir,
        "first",
        r#", checkpoint_save="half.nslm", checkpoint_every=20"#,
        &[],
    );
    assert!(f32.ok, "{}", f32.out);
    let across = run_gpu(
        SPLIT,
        160,
        &f32_dir,
        "across",
        r#", checkpoint_load="half.nslm""#,
        &[],
    );
    assert!(
        !across.ok,
        "a split run resumed an f32 checkpoint:\n{}",
        across.out
    );
    assert!(
        across.out.contains("ARITHMETIC differs") && across.out.contains("pdt:"),
        "{}",
        across.out
    );
    let back = run_gpu(
        "",
        160,
        &dir,
        "back",
        r#", checkpoint_load="half.nslm""#,
        &[],
    );
    assert!(
        !back.ok && back.out.contains("pdt:"),
        "an f32 run resumed a split checkpoint:\n{}",
        back.out
    );
}
