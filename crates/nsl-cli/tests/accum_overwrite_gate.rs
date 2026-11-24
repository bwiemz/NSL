//! Gradient accumulators left to the next window's first writes
//! (`nsl_runtime::accum_fresh`, kill switch `NSL_ACCUM_OVERWRITE=0`).
//!
//! The multi-tensor AdamW step used to store zeros into every accumulation
//! buffer once it had read it. Under the protocol it leaves them and marks them
//! fresh; every writer then overwrites a fresh buffer (a GEMM with `beta = 0`) or
//! zeroes it before adding, and whatever no gradient reached is zeroed before
//! the clip or the step reads it. Nothing a reader sees changes, so a run with
//! the protocol must be the same bits as a run without it.
//!
//! Fixture: the native-GQA probe (`fixtures/sdpa_native_gqa.nsl`) under
//! `@fused_lm_ce`, its embedding tied to the LM head, plus a parameter the
//! forward never reads, with `grad_accumulation=3` over 8 micro-batches: two full
//! windows and a trailing partial window of 2 (rescaled before its step). Every
//! accumulator writer of the production line runs: the fused weight-gradient
//! GEMMs, the norm gains' scaled add, the embedding's scatter and the LM head's
//! weight gradient computed in its forward. Three arms, each against the kill
//! switch under `--deterministic`:
//!
//! - f32 storage with the global-norm clip: the f32 kernel's keep variant
//!   through the clip's launch, and the head's vocabulary-chunked route (it
//!   zeroes a fresh buffer before adding);
//! - bf16 storage without the clip: the plain launch, and the head's
//!   row-chunked route from the second window on (its first GEMM overwrites);
//! - `param_dtype="bf16-split"` with the clip: the split master's keep kernel.
//!
//! In each, every loss and every trained weight (the untouched parameter too)
//! must be the same bits, and the counters must show the protocol ran: buffers
//! marked, overwritten and zeroed with it, none marked without it.
//!
//! The protocol's run fills every buffer with NaN as it is marked
//! (`NSL_ACCUM_TEST_POISON=1`). Without that, the stale bytes a missed zeroing
//! would leave are the previous window's gradient, and for the untouched
//! parameter they are zero, which no comparison could tell from the right
//! answer; with it, any add onto a fresh buffer and any read before the zeroing
//! turns the numbers into NaN. It also holds the `beta = 0` GEMMs to not
//! reading the buffer they overwrite.

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

const PROD_FLAGS: [&str; 5] = [
    "--source-ad",
    "--checkpoint-blocks",
    "--checkpoint-selective",
    "--fuse-rmsnorm-backward",
    "--fuse-wgrad-accum",
];

#[derive(Clone, Copy, Debug)]
struct Arm {
    bf16: bool,
    split: bool,
    clip: bool,
}

fn program(arm: Arm) -> String {
    let root = repo_root();
    let src =
        std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/sdpa_native_gqa.nsl"))
            .expect("sdpa_native_gqa fixture missing");
    let mut header = String::from("grad_accumulation=3");
    if arm.clip {
        header.push_str(", grad_clip=1.0");
    }
    if arm.bf16 {
        header.push_str(", precision=\"bf16\"");
    }
    if arm.split {
        assert!(arm.bf16, "bf16-split needs bf16 storage");
        header.push_str(", param_dtype=\"bf16-split\"");
    }
    let rewrites: [(&str, String); 7] = [
        (
            "attn: ATTN = ATTN(",
            "attn: GroupedQueryAttention = GroupedQueryAttention(".into(),
        ),
        // A parameter the forward never reads: no gradient reaches its buffer.
        (
            "    norm: RMSNorm = RMSNorm(256)\n",
            "    norm: RMSNorm = RMSNorm(256)\n    unused: Tensor = randn([64, 64]) * full([1], 0.02)\n".into(),
        ),
        ("# GPU_PLACEMENT", "m.to(cuda)".into()),
        (
            "train(model=m, epochs=1,",
            "@fused_lm_ce(enabled=true, vocab_size=128, hidden_size=256, batch_size=2, seq_len=128, vocab_tile=128)\ntrain(model=m, epochs=1,".into(),
        ),
        (
            "        let ls = logits.shape\n        let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])\n        let flat_labels = batch.labels.reshape([ls[0] * ls[1]])",
            "        let flat_logits = logits.reshape([256, 128])\n        let flat_labels = batch.labels.reshape([256])".into(),
        ),
        ("grad_accumulation=2):", format!("{header}):")),
        (
            "print(m.embed.sum())\n",
            "print(m.embed.sum())\nprint(m.norm.weight.sum())\nprint(m.unused.sum())\n".into(),
        ),
    ];
    let mut src = src;
    for (from, to) in rewrites {
        assert!(src.contains(from), "the fixture changed: no {from:?}");
        src = src.replacen(from, &to, 1);
    }
    src
}

struct Out {
    stdout: String,
    stderr: String,
}

fn run(tag: &str, arm: Arm, env: &[(&str, &str)]) -> Out {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_accum_fresh_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, program(arm)).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--seed", "5", "--deterministic"])
        .args(PROD_FLAGS)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_ACCUM_COUNTER", "1")
        .env_remove("NSL_ACCUM_OVERWRITE")
        .envs(env.iter().copied());
    if arm.bf16 {
        // `--deterministic` sends the flash-attention backward to the host,
        // which reads no bf16; keep it on the device's tensor-core kernels
        // (deterministic: no atomics), as `lce_grad_in_forward_gate` does.
        cmd.env("NSL_FLASH_BWD_CPU", "0");
    }
    let out = cmd.output().expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&tmp);
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(
        out.status.success(),
        "{tag} failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    Out { stdout, stderr }
}

/// The printed values between `begin` and `end`, `n` of them, all finite.
fn section(stdout: &str, begin: &str, end: &str, n: usize) -> Vec<String> {
    let lines: Vec<String> = stdout
        .split(begin)
        .nth(1)
        .and_then(|s| s.split(end).next())
        .unwrap_or("")
        .lines()
        .map(str::trim)
        // A device scalar prints as `tensor([..])`, a host one as the bare number.
        .filter(|l| l.starts_with("tensor([") || l.parse::<f64>().is_ok())
        .map(str::to_string)
        .collect();
    assert_eq!(lines.len(), n, "{begin}: {n} values expected:\n{stdout}");
    for l in &lines {
        let lower = l.to_ascii_lowercase();
        assert!(
            !lower.contains("nan") && !lower.contains("inf"),
            "non-finite value: {l:?}"
        );
    }
    lines
}

/// `[accum] marked=M overwritten=O zeroed=Z`.
fn counters(stderr: &str) -> (u64, u64, u64) {
    let line = stderr
        .lines()
        .find(|l| l.starts_with("[accum] "))
        .unwrap_or_else(|| panic!("no [accum] counter line:\n{stderr}"));
    let field = |k: &str| -> u64 {
        line.split_whitespace()
            .find_map(|w| w.strip_prefix(&format!("{k}=")))
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("no {k}= in {line:?}"))
    };
    (field("marked"), field("overwritten"), field("zeroed"))
}

fn case(tag: &str, arm: Arm) {
    let on = run(&format!("{tag}_on"), arm, &[("NSL_ACCUM_TEST_POISON", "1")]);
    let off = run(&format!("{tag}_off"), arm, &[("NSL_ACCUM_OVERWRITE", "0")]);
    for (name, o) in [("on", &on), ("off", &off)] {
        assert!(
            o.stderr.contains("[fase-multi] batched"),
            "{tag} {name}: the multi-tensor step did not run:\n{}",
            o.stderr
        );
    }

    let losses = |o: &Out| section(&o.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END", 8);
    let weights = |o: &Out| section(&o.stdout, "WEIGHTS_BEGIN", "WEIGHTS_END", 7);
    let (l_on, l_off) = (losses(&on), losses(&off));
    assert_eq!(
        l_on, l_off,
        "{tag}: the losses differ from the zeroing step's"
    );
    assert_ne!(
        l_off[0], l_off[7],
        "{tag}: the program did not train: {l_off:?}"
    );
    assert_eq!(
        weights(&on),
        weights(&off),
        "{tag}: the trained weights differ from the zeroing step's"
    );

    // Two full windows and the trailing flush: three steps. Every step but the
    // last marks the accumulators; the window after it overwrites the ones its
    // GEMMs write and zeroes the rest (the norm gain before its add, the
    // embedding before its scatter when the head did not overwrite it, the
    // untouched parameter before the read).
    let (marked, overwritten, zeroed) = counters(&on.stderr);
    assert!(marked > 0, "{tag}: nothing was marked:\n{}", on.stderr);
    assert!(
        overwritten > 0,
        "{tag}: no first write overwrote:\n{}",
        on.stderr
    );
    assert!(zeroed > 0, "{tag}: nothing was zeroed:\n{}", on.stderr);
    assert!(
        overwritten + zeroed <= marked,
        "{tag}: a mark was taken twice: marked={marked} overwritten={overwritten} zeroed={zeroed}"
    );
    assert_eq!(
        counters(&off.stderr),
        (0, 0, 0),
        "{tag}: the kill switch still marked:\n{}",
        off.stderr
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn f32_storage_with_the_clip_is_the_zeroing_step_bit_for_bit() {
    case(
        "f32_clip",
        Arm {
            bf16: false,
            split: false,
            clip: true,
        },
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_storage_is_the_zeroing_step_bit_for_bit() {
    case(
        "bf16",
        Arm {
            bf16: true,
            split: false,
            clip: false,
        },
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_split_parameters_with_the_clip_are_the_zeroing_step_bit_for_bit() {
    case(
        "split_clip",
        Arm {
            bf16: true,
            split: true,
            clip: true,
        },
    );
}
