//! The fused LM head's gradient computed in its forward (`nsl-codegen`
//! `lce_grad_in_forward`): when the head's upstream gradient is the loss seed,
//! the forward runs the head's whole backward (`nsl_fused_linear_ce_forward_
//! gemm_grad`): the weight gradient added, times `1/N`, into the parameter's
//! FASE accumulation buffer, the x gradient saved; the backward hands the
//! saved gradient on after checking the upstream gradient.
//!
//! The fixture is the native-GQA one (`fixtures/sdpa_native_gqa.nsl`) under
//! `@fused_lm_ce` with `grad_accumulation=2`, 8 micro-batches: its embedding
//! tied to the LM head (the production shape, through
//! `tied_embed_fusion`), and a variant with an untied head behind a final
//! norm (the weight gradient is the parameter's whole adjoint, and the head's
//! input is dead after the head). Each is trained with the route and with
//! `NSL_LCE_GRAD_IN_FORWARD=0` (today's backward), under the production
//! flags. The gate checks that:
//!
//! - the plan fired (`[lce-grad-in-forward] 1 of 1`) and the kill switch keeps
//!   it off; the launch counters say which runtime route ran;
//! - in f32 storage (no bf16 GEMMs, so the runtime takes the
//!   vocabulary-chunked route: today's two halves, both in the forward) under
//!   `--deterministic`, every loss and trained weight is the SAME BITS as the
//!   backward route: the scale, the accumulation alpha, the x gradient's hand-
//!   off and the order of the adds into the accumulation buffer are exact, also
//!   with a trailing partial window and the global-norm clip;
//! - at `precision="bf16"` the row-chunked route (bf16 logits) runs from the
//!   second window on (the head weight's bf16 image is cached from the first
//!   optimizer step); the first loss is the same bits and the rest agree to
//!   rounding, and under `--deterministic` it is bit-reproducible run to run;
//! - a head whose forward assumed another upstream gradient stops the run in
//!   its backward (`NSL_LCE_GIF_TEST_ASSUMED_GRAD`), and a scaled loss keeps
//!   the head in its backward.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOutput {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn spawn_drain<R: Read + Send + 'static>(
    mut pipe: R,
    buf: Arc<Mutex<Vec<u8>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.lock().unwrap().extend_from_slice(&chunk[..n]),
            }
        }
    })
}

const PROD_FLAGS: [&str; 5] = [
    "--source-ad",
    "--checkpoint-blocks",
    "--checkpoint-selective",
    "--fuse-rmsnorm-backward",
    "--fuse-wgrad-accum",
];

#[derive(Clone, Copy)]
struct Arm {
    bf16: bool,
    deterministic: bool,
    tied: bool,
    /// `0.5 * cross_entropy(..)`: the head's upstream gradient is not the seed.
    scaled_loss: bool,
    /// AdamW with `beta1 = 0` and an `eps` far above `sqrt(v)`: every update
    /// is `lr / eps` times the micro-batch mean gradient, LINEAR in it (plain
    /// AdamW divides a gradient's scale away), and the program prints each
    /// weight's update (`UPDATES`): its squared norm and its projection on a
    /// fixed direction.
    linear: bool,
    /// `grad_accumulation=3, grad_clip=1.0`: windows of 3, 3 and a trailing
    /// partial window of 2 micro-batches (rescaled before its step), and the
    /// global-norm clip over each window's accumulated gradient.
    tail_clip: bool,
}

const COUNTS: [i64; 4] = [4, 5, 6, 7];

fn program(arm: Arm) -> String {
    let root = repo_root();
    let mut src = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/sdpa_native_gqa.nsl"))
        .expect("sdpa_native_gqa fixture missing")
        .replace("attn: ATTN = ATTN(", "attn: GroupedQueryAttention = GroupedQueryAttention(")
        .replace("# GPU_PLACEMENT", "m.to(cuda)")
        .replace(
            "train(model=m, epochs=1,",
            "@fused_lm_ce(enabled=true, vocab_size=128, hidden_size=256, batch_size=2, seq_len=128, vocab_tile=128)\ntrain(model=m, epochs=1,",
        )
        .replace(
            "        let ls = logits.shape\n        let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])\n        let flat_labels = batch.labels.reshape([ls[0] * ls[1]])",
            "        let flat_logits = logits.reshape([256, 128])\n        let flat_labels = batch.labels.reshape([256])",
        );
    assert!(
        src.contains("logits.reshape([256, 128])"),
        "the fixture's step changed: update the rewrite above"
    );
    if !arm.tied {
        let before = src.clone();
        src = src
            .replace(
                "    attn: GroupedQueryAttention = GroupedQueryAttention(256, 4, 2, 64, 0.0, 128, 10000.0, 1)\n",
                "    attn: GroupedQueryAttention = GroupedQueryAttention(256, 4, 2, 64, 0.0, 128, 10000.0, 1)\n    final_norm: RMSNorm = RMSNorm(256)\n    head: Tensor = randn([128, 256]) * full([1], 0.02)\n",
            )
            .replace(
                "        return h @ self.embed.transpose(0, 1)",
                "        return self.final_norm.forward(h) @ self.head.transpose(0, 1)",
            )
            .replace("print(m.embed.sum())\n", "print(m.embed.sum())\nprint(m.head.sum())\n");
        assert!(
            src != before
                && src.contains("final_norm: RMSNorm")
                && src.contains("self.head.transpose"),
            "the fixture's model changed: update the untied rewrite"
        );
    }
    if arm.scaled_loss {
        src = src.replace(
            "        let loss = cross_entropy(flat_logits, flat_labels)",
            "        let loss = cross_entropy(flat_logits, flat_labels) * 0.5",
        );
    }
    if arm.tail_clip {
        assert!(!arm.bf16, "the tail/clip arm is an f32 arm");
        src = src.replace(
            "grad_accumulation=2):",
            "grad_accumulation=3, grad_clip=1.0):",
        );
    }
    if arm.bf16 {
        src = src.replace(
            "grad_accumulation=2):",
            "grad_accumulation=2, precision=\"bf16\"):",
        );
    }
    if arm.linear {
        let adamw = "AdamW(lr=0.01, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)";
        assert!(
            src.contains(adamw),
            "the fixture's optimizer changed: update the linear rewrite"
        );
        src = src.replace(
            adamw,
            "AdamW(lr=100.0, weight_decay=0.0, beta1=0.0, beta2=0.95, eps=10000.0)",
        );
        let mut names = vec![
            "m.attn.wq",
            "m.attn.wk",
            "m.attn.wv",
            "m.attn.wo",
            "m.embed",
        ];
        if !arm.tied {
            names.push("m.head");
        }
        let mut before = String::new();
        let mut after = String::from("print(\"UPDATES_BEGIN\")\n");
        for (i, n) in names.iter().enumerate() {
            before.push_str(&format!(
                "let w0_{i} = {n}.clone()\nlet dir_{i} = randn({n}.shape)\n"
            ));
            after.push_str(&format!(
                "let u_{i} = {n} - w0_{i}\nprint((u_{i} * u_{i}).sum())\nprint((u_{i} * dir_{i}).sum())\n"
            ));
        }
        after.push_str("print(\"UPDATES_END\")\n");
        src = src.replace(
            "print(\"LOSS_STREAM_BEGIN\")\n",
            &format!("{before}print(\"LOSS_STREAM_BEGIN\")\n"),
        );
        src.push_str(&after);
    }
    let counts: String = COUNTS
        .iter()
        .map(|k| format!("print(fused_lce_launch_count({k}))\n"))
        .collect();
    src.push_str(&format!(
        "print(\"COUNTS_BEGIN\")\n{counts}print(\"COUNTS_END\")\n"
    ));
    src
}

fn run(tag: &str, arm: Arm, envs: &[(&str, &str)]) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_lce_gif_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("lce_gif.nsl");
    std::fs::write(&prog, program(arm)).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--seed", "5"]).args(PROD_FLAGS);
    if arm.deterministic {
        cmd.arg("--deterministic");
    }
    cmd.arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .envs(envs.iter().copied())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn nsl run");
    let out_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let err_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let out_reader = spawn_drain(child.stdout.take().unwrap(), out_buf.clone());
    let err_reader = spawn_drain(child.stderr.take().unwrap(), err_buf.clone());
    let status = child.wait().expect("wait nsl run");
    let _ = out_reader.join();
    let _ = err_reader.join();
    let stdout = String::from_utf8_lossy(&out_buf.lock().unwrap()).to_string();
    let stderr = String::from_utf8_lossy(&err_buf.lock().unwrap()).to_string();
    let _ = std::fs::remove_dir_all(&tmp);
    RunOutput {
        ok: status.success(),
        stdout,
        stderr,
    }
}

fn run_ok(tag: &str, arm: Arm, envs: &[(&str, &str)]) -> RunOutput {
    let out = run(tag, arm, envs);
    assert!(
        out.ok,
        "{tag} failed\nstdout:\n{}\nstderr:\n{}",
        out.stdout, out.stderr
    );
    out
}

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

fn value(line: &str) -> f64 {
    line.trim_start_matches("tensor([")
        .split(|c: char| c == ']' || c == ',')
        .next()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no value in {line:?}"))
}

/// Per weight: the update's squared norm, then its projection.
fn updates(o: &RunOutput, arm: Arm) -> Vec<String> {
    section(
        &o.stdout,
        "UPDATES_BEGIN",
        "UPDATES_END",
        if arm.tied { 10 } else { 12 },
    )
}

fn losses(o: &RunOutput) -> Vec<String> {
    section(&o.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END", 8)
}

fn weights(o: &RunOutput, arm: Arm) -> Vec<String> {
    section(
        &o.stdout,
        "WEIGHTS_BEGIN",
        "WEIGHTS_END",
        if arm.tied { 5 } else { 6 },
    )
}

/// The launch counters, by kind: 4 = the backward GEMM route, 5 = the
/// forward with the gradient, 6 / 7 = its row- / vocabulary-chunked runs.
fn counts(o: &RunOutput) -> std::collections::HashMap<i64, i64> {
    let v = section(&o.stdout, "COUNTS_BEGIN", "COUNTS_END", COUNTS.len());
    COUNTS
        .iter()
        .copied()
        .zip(v.iter().map(|l| value(l) as i64))
        .collect()
}

const MARKER: &str = "[lce-grad-in-forward]";
const OFF: (&str, &str) = ("NSL_LCE_GRAD_IN_FORWARD", "0");

fn assert_planned(tag: &str, on: &RunOutput, off: &RunOutput) {
    assert!(
        on.stderr.contains(&format!(
            "{MARKER} 1 of 1 fused LM head(s) compute their gradient in the forward"
        )),
        "{tag}: the head was not planned:\n{}",
        on.stderr
    );
    assert!(
        !off.stderr.contains(MARKER),
        "{tag}: the kill switch did not keep it off:\n{}",
        off.stderr
    );
    let (c_on, c_off) = (counts(on), counts(off));
    assert_eq!(
        (c_on[&4], c_on[&5]),
        (0, 8),
        "{tag}: the planned run's head launches {c_on:?}"
    );
    assert_eq!(
        (c_off[&4], c_off[&5]),
        (8, 0),
        "{tag}: the kill switch's head launches {c_off:?}"
    );
}

/// `|a - b| <= rel · max(|b|, floor)` element-wise.
fn assert_close(tag: &str, what: &str, a: &[String], b: &[String], rel: f64, floor: f64) {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let (x, y) = (value(x), value(y));
        assert!(
            (x - y).abs() <= rel * y.abs().max(floor),
            "{tag} {what}[{i}]: {x} vs {y}\n{a:?}\n{b:?}"
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn in_f32_storage_the_gradient_in_the_forward_is_the_backwards_bit_for_bit() {
    for (tied, tail_clip) in [(true, false), (false, false), (true, true)] {
        let arm = Arm {
            bf16: false,
            deterministic: true,
            tied,
            scaled_loss: false,
            linear: false,
            tail_clip,
        };
        let tag = format!("f32_tied_{tied}_tail_clip_{tail_clip}");
        let on = run_ok(&format!("{tag}_on"), arm, &[]);
        let off = run_ok(&format!("{tag}_off"), arm, &[OFF]);
        assert_planned(&tag, &on, &off);
        let c = counts(&on);
        assert_eq!(
            (c[&6], c[&7]),
            (0, 8),
            "{tag}: f32 storage runs the vocabulary-chunked route: {c:?}"
        );
        assert_eq!(losses(&on), losses(&off), "{tag}: the loss streams differ");
        assert_eq!(
            weights(&on, arm),
            weights(&off, arm),
            "{tag}: the trained weights differ"
        );
        if !tied {
            assert!(
                on.stderr
                    .contains(&format!("{MARKER} 1 head input(s) freed by the forward")),
                "{tag}: the untied head's input was not freed after the head:\n{}",
                on.stderr
            );
        }
    }
}

/// The row-chunked route rounds the logits to bf16 before the softmax (as
/// PyTorch autocast does); the backward route keeps them f32. Measured on
/// this fixture: the first loss the row-chunked route evaluates (micro-batch
/// 2, the same weights on both arms) differs by 2e-5 (tied) and 1.2e-4
/// (untied); AdamW at lr 0.01 carries the gradients' difference through the
/// updates to 4.2e-3 in the last loss (falling to 0.16) and 1.8e-2 in the
/// smallest weight sum.
const FIRST_ROW_LOSS_REL: f64 = 1e-3;
const LOSS_REL: f64 = 2e-2;
const WEIGHT_SUM_REL: f64 = 5e-2;

#[test]
#[ignore = "requires CUDA GPU"]
fn at_bf16_the_row_chunked_route_agrees_with_the_backward_route() {
    for tied in [true, false] {
        let arm = Arm {
            bf16: true,
            deterministic: false,
            tied,
            scaled_loss: false,
            linear: false,
            tail_clip: false,
        };
        let tag = format!("bf16_tied_{tied}");
        let on = run_ok(&format!("{tag}_on"), arm, &[]);
        let off = run_ok(&format!("{tag}_off"), arm, &[OFF]);
        assert_planned(&tag, &on, &off);
        // Window 1 (2 micro-batches) has no bf16 image of the head weight yet.
        let c = counts(&on);
        assert_eq!((c[&6], c[&7]), (6, 2), "{tag}: routes {c:?}");
        let (lon, loff) = (losses(&on), losses(&off));
        assert_eq!(
            lon[..2],
            loff[..2],
            "{tag}: window 1 (the vocabulary-chunked route) differs"
        );
        assert_close(
            &tag,
            "first row-chunked loss",
            &lon[2..3],
            &loff[2..3],
            FIRST_ROW_LOSS_REL,
            1.0,
        );
        assert_close(&tag, "loss", &lon, &loff, LOSS_REL, 1.0);
        assert_close(
            &tag,
            "weight sum",
            &weights(&on, arm),
            &weights(&off, arm),
            WEIGHT_SUM_REL,
            1.0,
        );
        // Row chunks of 96 (three, the last 64 rows): the same rows' gradient
        // in another split. The GEMMs' and the accumulation's order differ,
        // nothing else.
        let ch = run_ok(
            &format!("{tag}_chunk96"),
            arm,
            &[("NSL_LCE_ROW_CHUNK", "96")],
        );
        assert_close(&tag, "chunk-96 loss", &losses(&ch), &lon, LOSS_REL, 1.0);
        assert_close(
            &tag,
            "chunk-96 weight sum",
            &weights(&ch, arm),
            &weights(&on, arm),
            WEIGHT_SUM_REL,
            1.0,
        );
        assert_close(
            &tag,
            "chunk-96 first row-chunked loss",
            &losses(&ch)[2..3],
            &lon[2..3],
            1e-6,
            1.0,
        );
    }
}

/// AdamW made linear in the gradient (see [`Arm::linear`]): the row-chunked
/// route's updates are the backward route's to the bf16 logits' rounding, so
/// a gradient scaled wrong (a lost `1/num_valid` or `1/N`, an `x` gradient
/// off by a factor) cannot hide behind AdamW's normalisation.
#[test]
#[ignore = "requires CUDA GPU"]
fn at_bf16_the_row_chunked_route_scales_the_gradient_as_the_backward_route_does() {
    for tied in [true, false] {
        let arm = Arm {
            bf16: true,
            deterministic: false,
            tied,
            scaled_loss: false,
            linear: true,
            tail_clip: false,
        };
        let tag = format!("linear_tied_{tied}");
        let on = run_ok(&format!("{tag}_on"), arm, &[]);
        let off = run_ok(&format!("{tag}_off"), arm, &[OFF]);
        assert_planned(&tag, &on, &off);
        assert_eq!(
            counts(&on)[&6],
            6,
            "{tag}: the row-chunked route did not run"
        );
        let (uon, uoff) = (updates(&on, arm), updates(&off, arm));
        for (w, (a, b)) in uon.chunks(2).zip(uoff.chunks(2)).enumerate() {
            let (sq_on, sq_off, p_on, p_off) =
                (value(&a[0]), value(&b[0]), value(&a[1]), value(&b[1]));
            assert!(sq_off > 0.0, "{tag}: weight {w} did not move");
            assert!(
                (sq_on - sq_off).abs() <= LINEAR_UPDATE_REL * sq_off,
                "{tag}: weight {w}'s update has squared norm {sq_on} against {sq_off}\n{uon:?}\n{uoff:?}"
            );
            // A projection on a random direction has the update's norm as its
            // natural scale.
            assert!(
                (p_on - p_off).abs() <= LINEAR_UPDATE_REL * sq_off.sqrt(),
                "{tag}: weight {w}'s update projects to {p_on} against {p_off} (norm {})\n{uon:?}\n{uoff:?}",
                sq_off.sqrt()
            );
        }
    }
}

/// Measured: the squared norms agree to 6e-4 and the projections to 2e-3 of
/// the update's norm (bf16 logits). A gradient 1% off moves a squared norm by
/// 2%; a factor of 2 by 300%.
const LINEAR_UPDATE_REL: f64 = 1e-2;

#[test]
#[ignore = "requires CUDA GPU"]
fn deterministic_runs_of_the_row_chunked_route_are_reproducible() {
    let arm = Arm {
        bf16: true,
        deterministic: true,
        tied: true,
        scaled_loss: false,
        linear: false,
        tail_clip: false,
    };
    let a = run_ok("det_a", arm, &[("NSL_FLASH_BWD_CPU", "0")]);
    let b = run_ok("det_b", arm, &[("NSL_FLASH_BWD_CPU", "0")]);
    let c = counts(&a);
    assert_eq!(
        (c[&5], c[&6]),
        (8, 6),
        "the row-chunked route did not run: {c:?}"
    );
    assert_eq!(
        losses(&a),
        losses(&b),
        "--deterministic loss streams differ run to run"
    );
    assert_eq!(
        weights(&a, arm),
        weights(&b, arm),
        "--deterministic trained weights differ run to run"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_backward_handed_another_upstream_gradient_stops_the_run() {
    let arm = Arm {
        bf16: false,
        deterministic: false,
        tied: true,
        scaled_loss: false,
        linear: false,
        tail_clip: false,
    };
    let out = run("assumed", arm, &[("NSL_LCE_GIF_TEST_ASSUMED_GRAD", "0.5")]);
    assert!(
        !out.ok,
        "the run went on with a gradient scaled for another upstream gradient:\n{}",
        out.stdout
    );
    assert!(
        out.stderr.contains("nsl_fused_lce_check_upstream_grad")
            && out.stderr.contains("assuming an upstream gradient"),
        "the run did not stop at the upstream-gradient check:\n{}",
        out.stderr
    );
    assert!(
        !out.stdout.contains("LOSS_STREAM_END"),
        "the run trained past the check:\n{}",
        out.stdout
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_scaled_loss_keeps_the_head_in_its_backward() {
    let arm = Arm {
        bf16: false,
        deterministic: false,
        tied: true,
        scaled_loss: true,
        linear: false,
        tail_clip: false,
    };
    let out = run_ok("scaled", arm, &[]);
    assert!(
        out.stderr
            .contains(&format!("{MARKER} 0 of 1 fused LM head(s)"))
            && out
                .stderr
                .contains("its upstream gradient is not the loss seed"),
        "a scaled loss was not kept in the backward:\n{}",
        out.stderr
    );
    let c = counts(&out);
    assert_eq!((c[&4], c[&5]), (8, 0), "head launches {c:?}");
}
