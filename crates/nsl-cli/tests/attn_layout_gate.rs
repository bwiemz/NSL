//! The attention head layout: `nsl.nn.gqa` hands `scaled_dot_product_attention`
//! the head views `x.reshape([b, s, h, d]).transpose(1, 2)` of its projections
//! (V without a contiguous copy), RoPE keeps that `[b, s, h, d]` layout, and the
//! tensor-core attention reads and writes it in place, forward and backward
//! (nsl-runtime `attn_layout`). Under `precision="bf16"` the RoPE adjoint also
//! writes its q/k gradients bf16 when only the projections' GEMMs read them
//! (the storage plan's rule 4b).
//!
//! The claim pinned here: with rule 4b off, every path trains BIT FOR BIT like
//! main. The fixture (`fixtures/attn_layout.nsl`, two pre-norm blocks with
//! grouped-query attention under the production flag line) is compiled with
//! `ATTN = GroupedQueryAttention` (the stdlib) and with `ATTN = PrevGqa` (the
//! stdlib's forward before this change, which copied V's head view), the
//! latter run with the layout off (`NSL_ATTN_BSHD=0`), the previous CTA order
//! (`NSL_TC_ATTN_LPT=0`) and rule 4b off (`NSL_ROPE_BWD_BF16=0`): main's
//! computation. The loss streams and trained weights are compared as text:
//!
//! - `--deterministic`, which takes the tensor-core backward (single writer),
//!   f32, and bf16 storage with the attention's output dropout; the
//!   tensor-core forward with the CPU reference backward (`NSL_FLASH_BWD_CPU=1`),
//!   f32;
//! - the tensor-core backward (`NSL_FLASH_BWD_CPU=0`), f32 and bf16;
//! - the previous attention kernels (`NSL_TC_ATTENTION=0`), which copy the views;
//! - `NSL_SDPA_GQA_EXPAND=1`;
//! - tape AD on the GPU;
//! - tape AD and source AD on the host;
//! - packed (`forward_packed`) and masked (`forward_masked`) attention, with
//!   the layout on and off (`fixtures/stage_c_packed_gqa.nsl`).
//!
//! Rule 4b changes numerics (the q/k projections' GEMMs read bf16 operands
//! where the bf16 matmul mode's shape rule gave this small fixture f32 ones),
//! so its arm checks that it is taken and that the losses stay close.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const PROD: &[&str] =
    &["--source-ad", "--checkpoint-blocks", "--checkpoint-selective", "--fuse-rmsnorm-backward", "--fuse-wgrad-accum"];

/// main's computation: the previous stdlib forward with the layout, the CTA
/// order and rule 4b off.
const MAIN_ENV: &[(&str, &str)] = &[("NSL_ATTN_BSHD", "0"), ("NSL_TC_ATTN_LPT", "0"), ("NSL_ROPE_BWD_BF16", "0")];

#[derive(Clone, Copy)]
struct Arm<'a> {
    gpu: bool,
    source_ad: bool,
    deterministic: bool,
    bf16: bool,
    heads: (u32, u32),
    dropout: bool,
    envs: &'a [(&'a str, &'a str)],
}

struct RunOutput {
    stdout: String,
    stderr: String,
}

fn spawn_drain<R: Read + Send + 'static>(mut pipe: R, buf: Arc<Mutex<Vec<u8>>>) -> std::thread::JoinHandle<()> {
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

/// Train the fixture with `attn` (`GroupedQueryAttention` or `PrevGqa`) in
/// configuration `arm`, plus `extra` environment variables.
fn run(tag: &str, attn: &str, arm: Arm<'_>, extra: &[(&str, &str)]) -> RunOutput {
    let root = repo_root();
    let mut src = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/attn_layout.nsl"))
        .expect("attn_layout fixture missing")
        .replace("attn: ATTN = ATTN(", &format!("attn: {attn} = {attn}("))
        .replace("NH, NKV, 64, DROP,", &format!("{}, {}, 64, {},", arm.heads.0, arm.heads.1, if arm.dropout { "0.1" } else { "0.0" }));
    if arm.gpu {
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    if arm.bf16 {
        src = src.replace("grad_accumulation=2):", "grad_accumulation=2, precision=\"bf16\"):");
    }
    let tmp = std::env::temp_dir().join(format!("nsl_attn_layout_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("attn_layout.nsl");
    std::fs::write(&prog, &src).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--seed", "7"]);
    if arm.source_ad {
        cmd.args(PROD);
    }
    if arm.deterministic {
        cmd.arg("--deterministic");
    }
    cmd.arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_FLASH_DEBUG", "1")
        .envs(arm.envs.iter().copied())
        .envs(extra.iter().copied())
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
    assert!(status.success(), "{tag} ({attn}) failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    RunOutput { stdout, stderr }
}

/// The printed lines between two markers, as text (bit-exact comparison).
/// Non-finite values fail the gate.
fn section(stdout: &str, begin: &str, end: &str) -> Vec<String> {
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
    for l in &lines {
        let lower = l.to_ascii_lowercase();
        assert!(!lower.contains("nan") && !lower.contains("inf"), "non-finite value: {l:?}");
    }
    lines
}

fn losses(out: &RunOutput) -> Vec<String> {
    let l = section(&out.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END");
    assert_eq!(l.len(), 8, "8 micro-batch losses expected:\n{}", out.stdout);
    l
}

fn weights(out: &RunOutput) -> Vec<String> {
    let w = section(&out.stdout, "WEIGHTS_BEGIN", "WEIGHTS_END");
    assert_eq!(w.len(), 5, "5 weight sums expected:\n{}", out.stdout);
    w
}

fn value(line: &str) -> f64 {
    line.trim_start_matches("tensor([")
        .split(|c: char| c == ']' || c == ',')
        .next()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no value in {line:?}"))
}

/// Train the stdlib program (default layout, CTA order and rule 4b unless
/// `new_env` says otherwise) and main's computation in one configuration and
/// require the same loss stream and trained weights, bit for bit.
fn assert_main(tag: &str, arm: Arm<'_>, new_env: &[(&str, &str)]) -> (RunOutput, RunOutput) {
    let new = run(&format!("{tag}_new"), "GroupedQueryAttention", arm, new_env);
    let main = run(&format!("{tag}_main"), "PrevGqa", arm, MAIN_ENV);
    assert_eq!(losses(&new), losses(&main), "{tag}: loss streams differ from main's computation");
    assert_eq!(weights(&new), weights(&main), "{tag}: trained weights differ from main's computation");
    assert!(!main.stderr.contains("[attn-layout]"), "{tag}: main's computation kept the head layout:\n{}", main.stderr);
    (new, main)
}

fn assert_layout_kept(out: &RunOutput, backward: bool) {
    for what in ["[attn-layout] rope keeps the [b, s, h, d] head layout", "[attn-layout] tensor-core attention forward"] {
        assert!(out.stderr.contains(what), "missing `{what}`:\n{}", out.stderr);
    }
    assert_eq!(
        out.stderr.contains("[attn-layout] tensor-core attention backward"),
        backward,
        "the backward's layout line:\n{}",
        out.stderr
    );
}

const HOST: Arm<'static> = Arm {
    gpu: false,
    source_ad: false,
    deterministic: false,
    bf16: false,
    heads: (4, 2),
    dropout: false,
    envs: &[],
};

#[test]
fn on_the_host_the_stdlib_trains_like_main() {
    for (tag, source_ad) in [("host_tape", false), ("host_src", true)] {
        assert_main(tag, Arm { source_ad, ..HOST }, &[]);
    }
}

const GPU: Arm<'static> = Arm {
    gpu: true,
    source_ad: true,
    deterministic: true,
    bf16: false,
    heads: (4, 2),
    dropout: false,
    envs: &[],
};

const RULE_4B_OFF: &[(&str, &str)] = &[("NSL_ROPE_BWD_BF16", "0")];

#[test]
#[ignore = "requires CUDA GPU"]
fn the_head_layout_trains_like_main_under_deterministic() {
    // `--deterministic` takes the tensor-core backward, which is deterministic
    // itself (one writer per output element), in the head layout.
    let (new, _) = assert_main("det_f32", GPU, &[]);
    assert_layout_kept(&new, true);
    assert!(
        new.stderr.contains("[flash-bwd] tensor-core backward dispatched"),
        "--deterministic did not take the tensor-core backward:\n{}",
        new.stderr
    );
    // Tensor-core forward in the head layout, CPU reference backward.
    let cpu = [("NSL_FLASH_BWD_CPU", "1")];
    let (new, _) = assert_main("det_f32_cpu", Arm { envs: &cpu, ..GPU }, &[]);
    assert_layout_kept(&new, false);
    assert!(
        new.stderr.contains("deterministic CPU reference backward forced"),
        "the CPU reference backward did not run:\n{}",
        new.stderr
    );
    // bf16 storage (which the CPU reference cannot read) with the attention's
    // output dropout, under `--deterministic` alone.
    let (new, _) = assert_main("det_bf16_drop", Arm { bf16: true, dropout: true, ..GPU }, RULE_4B_OFF);
    assert_layout_kept(&new, true);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_tensor_core_backward_in_the_head_layout_is_main_bit_for_bit() {
    let envs = [("NSL_FLASH_BWD_CPU", "0")];
    for bf16 in [false, true] {
        let (new, _) = assert_main(&format!("tcbwd_{bf16}"), Arm { bf16, envs: &envs, ..GPU }, RULE_4B_OFF);
        assert_layout_kept(&new, true);
        assert!(
            new.stderr.contains("[flash-bwd] tensor-core backward dispatched"),
            "the tensor-core backward did not run:\n{}",
            new.stderr
        );
    }
}

/// Rule 4b: the RoPE adjoint writes q/k's gradients bf16 for the
/// projections' GEMMs. It is taken (the streams differ from rule 4b off: on
/// this small fixture the bf16 matmul mode gave those GEMMs f32 operands),
/// the forward is untouched (the first loss is the same), and the losses stay
/// at bf16 distance.
#[test]
#[ignore = "requires CUDA GPU"]
fn rule_4b_stores_the_rope_gradients_bf16_and_stays_close() {
    let arm = Arm { bf16: true, envs: &[("NSL_FLASH_BWD_CPU", "0")], ..GPU };
    let on = run("r4b_on", "GroupedQueryAttention", arm, &[]);
    let off = run("r4b_off", "GroupedQueryAttention", arm, RULE_4B_OFF);
    let (lon, loff) = (losses(&on), losses(&off));
    assert_eq!(lon[0], loff[0], "the first loss differs: rule 4b changed the forward");
    assert_ne!(lon, loff, "rule 4b changed nothing: was it applied?");
    for (a, b) in lon.iter().zip(&loff) {
        let (a, b) = (value(a), value(b));
        assert!((a - b).abs() <= 2e-2 * b.abs(), "losses {a} vs {b}");
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_previous_kernels_and_the_gqa_kill_switch_take_head_views() {
    let envs = [("NSL_TC_ATTENTION", "0")];
    let (new, _) = assert_main("prev_kernels", Arm { envs: &envs, ..GPU }, &[]);
    assert!(
        new.stderr.contains("sdpa fused forward: launched (segmask=0"),
        "the previous fused forward did not run:\n{}",
        new.stderr
    );
    let envs = [("NSL_SDPA_GQA_EXPAND", "1"), ("NSL_FLASH_BWD_CPU", "0")];
    let (new, _) = assert_main("gqa_expand", Arm { envs: &envs, ..GPU }, &[]);
    assert!(new.stderr.contains("gqa_expanded=true"), "{}", new.stderr);
}

/// Tape AD on the GPU, grouped K/V and one key/value head per query head.
/// (An `@flash_attention` context applies to attention in its own module,
/// not to the stdlib's GroupedQueryAttention; the decorated launch copies
/// its operands contiguous, `expr/calls.rs`.)
#[test]
#[ignore = "requires CUDA GPU"]
fn tape_ad_on_the_gpu_takes_head_views() {
    // The embedding backward's atomics are the fixture's other
    // nondeterministic op under tape AD; keep it on the host.
    let envs = [("NSL_EMBEDDING_BWD_CPU", "1")];
    let tape = Arm { source_ad: false, deterministic: false, envs: &envs, ..GPU };
    for heads in [(4, 2), (4, 4)] {
        assert_main(&format!("tape_gpu_{}_{}", heads.0, heads.1), Arm { heads, ..tape }, &[]);
    }
}

/// Packed and masked attention (`fixtures/stage_c_packed_gqa.nsl`, head_dim
/// 32, packed batches): `forward_packed` (its own RoPE, the fused
/// segment-masked kernels) and `forward_masked` (the `rope` builtin, whose
/// result keeps the head layout, into the decomposed masked attention, which
/// reads through strides) train bit for bit with the layout on and off.
#[test]
#[ignore = "requires CUDA GPU"]
fn packed_and_masked_attention_train_alike_with_the_layout_on_and_off() {
    let root = repo_root();
    let base = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/stage_c_packed_gqa.nsl"))
        .expect("stage_c_packed_gqa fixture missing")
        .replace("# GPU_PLACEMENT", "m.to(cuda)");
    let masked = base.replace(
        "self.attn.forward_packed(rmsnorm(h, self.norm_a, 0.00001), seg, pos, training)",
        "self.attn.forward_masked(rmsnorm(h, self.norm_a, 0.00001), mask, training)",
    );
    assert_ne!(masked, base, "forward_packed marker not found — resync with the stage C fixture");
    for (form, src) in [("packed", &base), ("masked", &masked)] {
        let mut outs = Vec::new();
        for (arm, env) in [("on", &[][..]), ("off", &[("NSL_ATTN_BSHD", "0")][..])] {
            let tmp = std::env::temp_dir().join(format!("nsl_attn_layout_{form}_{arm}_{}", std::process::id()));
            std::fs::create_dir_all(&tmp).unwrap();
            let save = tmp.join("final.nslm");
            std::fs::write(tmp.join("p.nsl"), src.replace("STAGE_C_SAVE_PATH", &save.display().to_string())).unwrap();
            let o = Command::new(env!("CARGO_BIN_EXE_nsl"))
                .args(["run", "--source-ad", "--deterministic", "--seed", "3"])
                .arg(tmp.join("p.nsl"))
                .current_dir(&tmp)
                .env("NSL_STDLIB_PATH", root.join("stdlib"))
                .envs(env.iter().copied())
                .output()
                .expect("spawn nsl run");
            let (stdout, stderr) = (String::from_utf8_lossy(&o.stdout).to_string(), String::from_utf8_lossy(&o.stderr).to_string());
            assert!(o.status.success(), "{form} {arm} failed:\n{stdout}\n{stderr}");
            let l = section(&stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END");
            assert!(l.len() >= 8, "{form} {arm}: {} losses:\n{stdout}", l.len());
            let bytes = std::fs::read(&save).expect("saved model");
            let _ = std::fs::remove_dir_all(&tmp);
            outs.push((l, bytes, stderr));
        }
        assert_eq!(outs[0].0, outs[1].0, "{form}: loss streams differ with the layout on and off");
        assert!(outs[0].1 == outs[1].1, "{form}: trained models differ with the layout on and off");
        if form == "masked" {
            assert!(outs[0].2.contains("[attn-layout] rope keeps"), "masked: RoPE did not keep the layout:\n{}", outs[0].2);
        }
    }
}
