//! `first_moment_dtype="bf16"`: Adam/AdamW keep the first moment `m` in bf16
//! (rounded to nearest even on store, f32 arithmetic inside the update); `v`,
//! the master weights and the gradient accumulator stay f32.
//!
//! End to end on the CPU path (the fused step's CPU twin): the argument takes
//! effect, a checkpoint carries the bf16 moments and resumes bit-exactly, a
//! resume across a change of the argument is refused in both directions, and
//! every composition that would hand the bf16 buffer to f32 arithmetic is a
//! compile error. The kernels themselves are gated against an f64 reference in
//! `crates/nsl-codegen/tests/fase_adamw_mbf16_reference.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// A small AdamW program with gradient accumulation (the FASE-Deferred plan
/// the fused step needs). `train_cfg` is spliced into `train(...)`, `optim`
/// is the optimizer's argument list.
fn fixture(train_cfg: &str, optim: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([4, 4]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()

let x = full([3, 4], 0.7)
let y = full([3, 4], 0.1)
train(model = m, grad_accumulation = 2{train_cfg}):
    optimizer: AdamW({optim})
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(step)
            print(loss)

print("FIXTURE_DONE")
"#
    )
}

const BF16: &str = r#"lr = 0.01, weight_decay = 0.1, first_moment_dtype = "bf16""#;
const F32: &str = "lr = 0.01, weight_decay = 0.1";

struct RunOut {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run_in(dir: &Path, name: &str, src: &str, extra: &[&str]) -> RunOut {
    let prog = dir.join(name);
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad"])
        .args(extra)
        .arg(&prog)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    RunOut {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

fn fresh_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nsl_m1bf16_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// (step, loss) line pairs, raw: compared as bytes.
fn loss_lines(stdout: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = stdout.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < lines.len() {
        if lines[i].trim().parse::<i64>().is_ok() {
            out.push((lines[i].trim().to_string(), lines[i + 1].trim().to_string()));
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

/// The argument takes effect: the run trains, and its trajectory leaves the
/// f32-moment run's once the rounded moments start to matter (the first
/// update reads the f32 m', so the first window agrees).
#[test]
fn a_bf16_first_moment_trains_and_changes_the_trajectory() {
    let d = fresh_dir("effect");
    let b = run_in(&d, "b.nsl", &fixture(", epochs = 12", BF16), &[]);
    let f = run_in(&d, "f.nsl", &fixture(", epochs = 12", F32), &[]);
    assert!(b.ok && b.stdout.contains("FIXTURE_DONE"), "bf16 run failed:\n{}", b.stderr);
    assert!(f.ok, "f32 run failed:\n{}", f.stderr);
    let (lb, lf) = (loss_lines(&b.stdout), loss_lines(&f.stdout));
    assert_eq!(lb.len(), 12, "{}", b.stdout);
    assert_eq!(lb[..2], lf[..2], "the first window's forward is the same model");
    assert_ne!(lb, lf, "bf16 moments left the trajectory byte-identical: the argument did nothing");
    let loss = |l: &[(String, String)], k: usize| -> f64 {
        l[k].1.trim_start_matches("tensor([").trim_end_matches("])").parse().unwrap()
    };
    assert!(loss(&lb, 11) < loss(&lb, 0), "the bf16-moment run did not train: {lb:?}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_checkpoint_carries_the_bf16_moments_and_resumes_bit_exactly() {
    let d = fresh_dir("resume");
    let a = run_in(&d, "a.nsl", &fixture(r#", epochs = 6, checkpoint_save = "ck.nslm", checkpoint_every = 1"#, BF16), &[]);
    assert!(a.ok, "phase A failed:\n{}", a.stderr);
    // The sidecar names the first moments bf16 and the second moments f32.
    let header = std::fs::read(d.join("ck.nslm.optim")).unwrap();
    let header = String::from_utf8_lossy(&header[..header.len().min(4096)]).to_string();
    assert!(header.contains(r#""name":"m:m.w","shape":[4, 4],"dtype":"bf16""#), "{header}");
    assert!(header.contains(r#""name":"v:m.w","shape":[4, 4],"dtype":"f32""#), "{header}");

    let b = run_in(&d, "b.nsl", &fixture(r#", epochs = 10, checkpoint_load = "ck.nslm""#, BF16), &[]);
    assert!(b.ok && b.stderr.contains("[checkpoint] resumed:"), "phase B failed:\n{}", b.stderr);
    let ctl = fresh_dir("resume_ctl");
    let c = run_in(&ctl, "c.nsl", &fixture(", epochs = 10", BF16), &[]);
    assert!(c.ok, "control failed:\n{}", c.stderr);
    let (lb, lc) = (loss_lines(&b.stdout), loss_lines(&c.stdout));
    assert_eq!(lb.len(), 4, "{}", b.stdout);
    assert_eq!(lb, lc[6..].to_vec(), "the resumed run left the uninterrupted one");
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_dir_all(&ctl);
}

/// A checkpoint written with one first-moment dtype does not resume under the
/// other: converting would either round the saved moments or continue a
/// different optimizer. Refused in both directions.
#[test]
fn a_resume_across_a_first_moment_dtype_change_is_refused() {
    for (saved, live) in [(BF16, F32), (F32, BF16)] {
        let d = fresh_dir("cross");
        let a = run_in(&d, "a.nsl", &fixture(r#", epochs = 2, checkpoint_save = "ck.nslm", checkpoint_every = 1"#, saved), &[]);
        assert!(a.ok, "save failed:\n{}", a.stderr);
        let b = run_in(&d, "b.nsl", &fixture(r#", epochs = 4, checkpoint_load = "ck.nslm""#, live), &[]);
        assert!(!b.ok, "{saved} -> {live} resumed:\n{}", b.stderr);
        assert!(!b.stderr.contains("[checkpoint] resumed:"), "{}", b.stderr);
        assert!(
            b.stderr.contains("m1") || b.stderr.contains("first_moment_dtype"),
            "{saved} -> {live}: the refusal does not name the cause:\n{}",
            b.stderr
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}

/// The same program with its train block inside a function, which compiles
/// before `main` emits the execution fingerprint.
fn nested_fixture(optim: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([4, 4]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

fn run_training():
    let m = Tiny()
    let x = full([3, 4], 0.7)
    let y = full([3, 4], 0.1)
    train(model = m, grad_accumulation = 2, epochs = 2, checkpoint_save = "ck.nslm", checkpoint_every = 1):
        optimizer: AdamW({optim})
        step(batch):
            let pred = m.forward(x)
            let loss = mse_loss(pred, y)

run_training()
print("FIXTURE_DONE")
"#
    )
}

/// The checkpoint's execution fingerprint says `m1=bf16` exactly when the
/// program asks for a bf16 first moment — wherever its train block is: a
/// block inside a function lowers before `main` writes the fingerprint, so the
/// program is scanned before either.
#[test]
fn the_fingerprint_records_a_bf16_first_moment_wherever_the_block_is() {
    // The resume record (with its `"exec":"..."` fingerprint) is in the
    // optimizer sidecar.
    let has_m1 = |dir: &Path| -> bool {
        let bytes = std::fs::read(dir.join("ck.nslm.optim")).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        let exec = text.split("\"exec\":\"").nth(1).expect("no exec fingerprint in the sidecar");
        let exec = &exec[..exec.find('"').unwrap()];
        assert!(exec.contains("ad=source"), "not the fingerprint: {exec}");
        exec.split(',').any(|k| k == "m1=bf16")
    };
    let ckpt = r#", epochs = 2, checkpoint_save = "ck.nslm", checkpoint_every = 1"#;
    for (name, src, bf16) in [
        ("top level, bf16", fixture(ckpt, BF16), true),
        ("top level, f32", fixture(ckpt, F32), false),
        ("in a function, bf16", nested_fixture(BF16), true),
        ("in a function, f32", nested_fixture(F32), false),
    ] {
        let d = fresh_dir("fp");
        let r = run_in(&d, "p.nsl", &src, &[]);
        assert!(r.ok && r.stdout.contains("FIXTURE_DONE"), "{name}: failed:\n{}", r.stderr);
        assert_eq!(has_m1(&d), bf16, "{name}: the checkpoint's fingerprint");
        let _ = std::fs::remove_dir_all(&d);
    }
}

#[test]
fn compositions_that_would_read_m_as_f32_are_compile_errors() {
    let d = fresh_dir("refuse");
    let cases: [(&str, String, &[&str], &str); 4] = [
        (
            "no accumulation: the stdlib step, not the fused one",
            fixture(", epochs = 2", BF16).replace("grad_accumulation = 2", "grad_accumulation = 1"),
            &[],
            "needs the FASE-Deferred plan",
        ),
        ("offload", fixture(", epochs = 2", BF16), &["--optim-state-offload"], "--optim-state-offload"),
        ("training reference", fixture(", epochs = 2", BF16), &["--training-reference"], "--training-reference"),
        (
            "an optimizer without a bf16 first moment",
            fixture(", epochs = 2", r#"lr = 0.01, first_moment_dtype = "bf16""#).replace("AdamW(", "SGD("),
            &[],
            "does not accept 'first_moment_dtype'",
        ),
    ];
    for (name, src, flags, needle) in cases {
        let r = run_in(&d, "r.nsl", &src, flags);
        assert!(!r.ok, "{name}: compiled and ran:\n{}", r.stdout);
        assert!(r.stderr.contains(needle), "{name}: expected `{needle}` in:\n{}", r.stderr);
    }
    let _ = std::fs::remove_dir_all(&d);
}
