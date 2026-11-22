//! Source-AD dropout keeps a 32-byte key and redraws its mask in the backward
//! (`NSL_DROPOUT_REGEN`, the default), end to end.
//!
//! The kernels are held bit-exact to the stored-mask path on the PTX
//! interpreter (`nsl-codegen/tests/dropout_regen_kir_equivalence.rs`) and on
//! silicon (`nsl-runtime/tests/hwcert_kir_numerics_gpu.rs`); the planning is
//! unit tested in `ccr.rs`. This gate holds whole programs to the claim that
//! the switch changes memory and nothing else:
//!
//! * **Witness** (CPU build): the build log says how many dropouts keep a key,
//!   how many outputs CCR frees at their block's end and how many it redraws
//!   from the key in the backward. A dropout after the block's last
//!   projection (the recipes' pattern: the residual add reads its output for
//!   shape only) is freed (under `--layerwise-accum` too, whose window buffers
//!   the shape stub); one whose output a matrix product reads stays saved
//!   under `--checkpoint-selective` and is redrawn under full block recompute
//!   and under `--layerwise-accum`. `NSL_DROPOUT_REGEN=0` says it stores the
//!   mask.
//! * **Equivalence** (CPU run, and GPU under `--deterministic`): the same
//!   program built with and without the switch prints the same losses and
//!   saves the same model bytes, for both patterns, without checkpointing,
//!   with the production checkpoint flags, with full block recompute, under
//!   `--layerwise-accum`, and (GPU) under `precision="bf16"`. Each GPU comparison first checks that
//!   two runs of the default agree, so an inequality is the switch's.
//! * **Memory** (GPU): with 2 MiB dropout tensors the allocator peak drops by
//!   the masks and outputs the default no longer keeps.
//! * **Resume** (GPU): a run checkpointed with the key path and resumed with the
//!   stored-mask path (and the other way round) continues the uninterrupted
//!   run's losses bit for bit: the counter a checkpoint saves is the same in
//!   both modes, so the switch is not part of the execution fingerprint.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const PROD: &[&str] =
    &["--source-ad", "--checkpoint-blocks", "--checkpoint-selective", "--fuse-rmsnorm-backward", "--fuse-wgrad-accum"];
const CSLA: &[&str] = &["--source-ad", "--checkpoint-blocks", "--fuse-rmsnorm-backward", "--layerwise-accum"];
/// Full block recompute (the Block policy): a dropout output is a recompute
/// victim, redrawn from its key when the backward reads its value.
const BLOCKS: &[&str] = &["--source-ad", "--checkpoint-blocks", "--fuse-rmsnorm-backward"];
const PLAIN: &[&str] = &["--source-ad"];

/// Where the dropout sits in each block.
#[derive(Clone, Copy, Debug)]
enum Site {
    /// `h + dropout(act @ w_down)`: the recipes' pattern; the backward reads
    /// the dropout output for its shape only.
    Tail,
    /// `h + dropout(act) @ w_down`: the projection's backward reads the
    /// dropout output's value.
    Interior,
}

/// A two-block residual MLP (`blocks: [Blk; 2]`, so CCR segments it) with
/// one dropout per block at `site` and an output projection after the blocks
/// (as the recipes' final norm and LM head: without one, CCR's last segment
/// runs to the loss and recomputes the last residual add, whose clone reads
/// the dropout output's value), eight micro-batches in two optimizer steps,
/// deterministic weights. `d` is the hidden width (the dropout tensor
/// is `rows x d` f32); `train_extra` is spliced into the train header.
fn fixture(site: Site, rows: usize, d: usize, cuda: bool, train_extra: &str) -> String {
    let body = match site {
        Site::Tail => "        let act = relu(h @ self.w_up)\n        return h + dropout(act @ self.w_down, 0.25, true)",
        Site::Interior => "        let act = dropout(relu(h @ self.w_up), 0.25, true)\n        return h + act @ self.w_down",
    };
    let dev = if cuda { "m.to(cuda)\n" } else { "" };
    let xd = if cuda { ".to(cuda)" } else { "" };
    let (n, f) = (d, 2 * d);
    format!(
        r#"from nsl.nn.losses import mse_loss

model Blk:
    w_up: Tensor = ((arange({nf}).reshape([{n}, {f}])) * {su}) - 0.05
    w_down: Tensor = ((arange({nf}).reshape([{f}, {n}])) * {sd}) - 0.04

    fn forward(self, h: Tensor) -> Tensor:
{body}

model Net:
    blocks: [Blk; 2] = Blk()
    w_out: Tensor = ((arange({nn}).reshape([{n}, {n}])) * {so}) - 0.02

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h @ self.w_out

let m = Net()
{dev}let x = (((arange({rn}).reshape([{rows}, {n}])) * {sx}) - 0.7){xd}
let y = (((arange({rn}).reshape([{rows}, {n}])) * {sy}) - 0.4){xd}

print("LOSS_STREAM_BEGIN")
train(model=m, epochs=8, grad_accumulation=4{train_extra}):
    optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-3, weight_decay=0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")
model_save(m, "after.nslm")
print("PEAK_BYTES")
print(gpu_peak_bytes())
print("REGEN_GATE_DONE")
"#,
        nf = n * f,
        nn = n * n,
        so = 0.6 / (n * n) as f64,
        su = 0.1 / (n * f) as f64,
        sd = 0.08 / (n * f) as f64,
        rn = rows * n,
        sx = 1.4 / (rows * n) as f64,
        sy = 0.8 / (rows * n) as f64,
    )
}

/// Two pre-norm SwiGLU blocks with the 500M recipe's dropout tail, for
/// `precision="bf16"` (the storage plan stores the MLP bf16 and the
/// dropout's operand f32).
fn swiglu_fixture(cuda: bool, precision: &str) -> String {
    let (dev, xd) = if cuda { ("m.to(cuda)\n", ".to(cuda)") } else { ("", "") };
    format!(
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
        return x + dropout((gate * up) @ self.w_down, 0.1, true)

model Net:
    blocks: [Block; 2] = Block()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
{dev}let x = (((arange(1024).reshape([2, 16, 32])) * 0.003) - 1.3){xd}
let y = (((arange(1024).reshape([2, 16, 32])) * 0.002) - 0.9){xd}

print("LOSS_STREAM_BEGIN")
train(model=m, epochs=8, grad_accumulation=4{precision}):
    optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-2, weight_decay=0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")
model_save(m, "after.nslm")
print("REGEN_GATE_DONE")
"#
    )
}

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
    model: Vec<u8>,
}

impl Run {
    fn losses(&self) -> &str {
        let s = self.stdout.split("LOSS_STREAM_BEGIN").nth(1).unwrap_or("");
        s.split("LOSS_STREAM_END").next().unwrap_or("")
    }

    fn peak_bytes(&self) -> u64 {
        let after = self.stdout.split("PEAK_BYTES").nth(1).expect("no PEAK_BYTES");
        after.split_whitespace().next().and_then(|v| v.parse().ok()).expect("PEAK_BYTES value")
    }
}

/// `nsl run` of `src` in a fresh directory, with `NSL_DROPOUT_REGEN=0` when
/// `stored_mask`. Reads back the saved model.
fn run(tag: &str, src: &str, args: &[&str], stored_mask: bool, env: &[(&str, &str)]) -> Run {
    let dir = std::env::temp_dir().join(format!("nsl_dropout_regen_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let r = run_in(&dir, src, args, stored_mask, env);
    let _ = std::fs::remove_dir_all(&dir);
    r
}

fn run_in(dir: &Path, src: &str, args: &[&str], stored_mask: bool, env: &[(&str, &str)]) -> Run {
    let root = repo_root();
    let prog = dir.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run").args(args).arg(&prog).current_dir(dir).env("NSL_STDLIB_PATH", root.join("stdlib"));
    cmd.env_remove("NSL_DROPOUT_REGEN").env_remove("NSL_MATMUL_BF16");
    if stored_mask {
        cmd.env("NSL_DROPOUT_REGEN", "0");
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    let model = std::fs::read(dir.join("after.nslm")).unwrap_or_default();
    let _ = std::fs::remove_file(dir.join("after.nslm"));
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        model,
    }
}

fn witness(r: &Run) -> &str {
    r.stderr.lines().find(|l| l.contains("[dropout]")).unwrap_or("")
}

/// Default and stored-mask runs of one program: both succeed, train, and
/// agree on every printed loss and every saved model byte.
fn assert_switch_is_exact(tag: &str, src: &str, args: &[&str], env: &[(&str, &str)]) -> (Run, Run) {
    let key = run(&format!("{tag}_key"), src, args, false, env);
    assert!(key.ok, "{tag}: default run failed:\n{}\n{}", key.stdout, key.stderr);
    assert!(key.stdout.contains("REGEN_GATE_DONE"), "{tag}: default run did not finish:\n{}", key.stdout);
    assert!(witness(&key).contains("keep a 32-byte key"), "{tag}: no key witness:\n{}", key.stderr);
    let mask = run(&format!("{tag}_mask"), src, args, true, env);
    assert!(mask.ok, "{tag}: NSL_DROPOUT_REGEN=0 run failed:\n{}\n{}", mask.stdout, mask.stderr);
    assert!(witness(&mask).contains("NSL_DROPOUT_REGEN=0"), "{tag}: the kill switch did not take:\n{}", mask.stderr);
    let losses: Vec<&str> = key.losses().split_whitespace().collect();
    assert!(losses.len() >= 8, "{tag}: expected 8 losses:\n{}", key.stdout);
    assert_ne!(losses.first(), losses.last(), "{tag}: the fixture did not train");
    assert_eq!(key.losses(), mask.losses(), "{tag}: losses differ between the key and the stored mask");
    assert!(!key.model.is_empty(), "{tag}: no model saved");
    assert!(key.model == mask.model, "{tag}: model bytes differ between the key and the stored mask");
    (key, mask)
}

#[test]
fn the_build_log_says_which_dropout_outputs_are_freed() {
    const KEY: &str = "[dropout] 2 dropout(s) keep a 32-byte key and redraw their mask in the backward; ";
    for (site, name, args, want) in [
        // Selective: a kept save, freed when read for shape only.
        (Site::Tail, "prod", PROD, "2 of 2 output(s) freed after their block (read for shape only), 0 recomputed"),
        (Site::Interior, "prod", PROD, "0 of 2 output(s) freed after their block (read for shape only), 0 recomputed"),
        // Block: a recompute victim, freed when read for shape only and
        // redrawn from the key when read for value.
        (Site::Tail, "blocks", BLOCKS, "2 of 2 output(s) freed after their block (read for shape only), 0 recomputed"),
        (Site::Interior, "blocks", BLOCKS, "0 of 2 output(s) freed after their block (read for shape only), 2 recomputed"),
        // CSLA frees each block's interiors at the block's end too (#876), and
        // its window buffers the shape stubs: the shape-only output is freed,
        // the value-read one redrawn in the window backward.
        (Site::Tail, "csla", CSLA, "2 of 2 output(s) freed after their block (read for shape only), 0 recomputed"),
        (Site::Interior, "csla", CSLA, "0 of 2 output(s) freed after their block (read for shape only), 2 recomputed"),
        // No checkpointing: nothing frees a primal value before the backward.
        (Site::Tail, "plain", PLAIN, "0 of 2 output(s) freed after their block (read for shape only), 0 recomputed"),
    ] {
        let r = run(&format!("witness_{site:?}_{name}"), &fixture(site, 4, 8, false, ""), args, false, &[]);
        assert!(r.ok, "{site:?} {name}: {}", r.stderr);
        let want = format!("{KEY}{want}");
        assert!(r.stderr.contains(&want), "{site:?} {name}: want `{want}`:\n{}", r.stderr);
    }
    let r = run("witness_off", &fixture(Site::Tail, 4, 8, false, ""), PROD, true, &[]);
    assert!(r.ok, "{}", r.stderr);
    assert!(
        r.stderr.contains("[dropout] NSL_DROPOUT_REGEN=0: 2 dropout(s) store an f32 mask"),
        "{}",
        r.stderr
    );
}

/// On the CPU the "key" is the mask (the CPU stream is not counter-based):
/// the switch must not change a CPU program either, including the freed
/// dropout output's shape stub.
#[test]
fn the_switch_does_not_change_a_cpu_program() {
    for site in [Site::Tail, Site::Interior] {
        for (name, args) in [("plain", PLAIN), ("prod", PROD), ("blocks", BLOCKS), ("csla", CSLA)] {
            assert_switch_is_exact(&format!("cpu_{site:?}_{name}"), &fixture(site, 4, 8, false, ""), args, &[]);
        }
    }
}

/// Two runs of the default must agree before a difference can be blamed on
/// the switch (production mode is not run-to-run reproducible; see bugs.md).
fn assert_deterministic(tag: &str, src: &str, args: &[&str], env: &[(&str, &str)]) {
    let a = run(&format!("{tag}_det_a"), src, args, false, env);
    let b = run(&format!("{tag}_det_b"), src, args, false, env);
    assert!(a.ok && b.ok, "{tag}: control runs failed:\n{}\n{}", a.stderr, b.stderr);
    assert_eq!(a.losses(), b.losses(), "{tag}: two default runs disagree; the comparison would be noise");
    assert!(a.model == b.model, "{tag}: two default runs saved different models");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_switch_does_not_change_a_gpu_program_under_deterministic() {
    for site in [Site::Tail, Site::Interior] {
        for (name, flags) in [("plain", PLAIN), ("prod", PROD), ("blocks", BLOCKS), ("csla", CSLA)] {
            let mut args = flags.to_vec();
            args.push("--deterministic");
            let src = fixture(site, 64, 128, true, "");
            let tag = format!("gpu_{site:?}_{name}");
            assert_deterministic(&tag, &src, &args, &[]);
            assert_switch_is_exact(&tag, &src, &args, &[]);
        }
    }
}

/// `precision="bf16"` stores the MLP bf16 and keeps every dropout tail f32;
/// the regenerating backward reads only f32. The switch changes nothing.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_switch_does_not_change_a_bf16_storage_program() {
    let src = swiglu_fixture(true, r#", precision="bf16""#);
    let mut args = PROD.to_vec();
    args.push("--deterministic");
    assert_deterministic("gpu_bf16", &src, &args, &[]);
    let (key, _) = assert_switch_is_exact("gpu_bf16", &src, &args, &[]);
    assert!(key.stderr.contains("[bf16-storage] 2 SwiGLU MLP block(s) stored bf16"), "{}", key.stderr);
    assert!(key.stderr.contains("activation storage: bf16"), "{}", key.stderr);
}

/// The memory the default no longer keeps. Each dropout tensor is
/// `8192 x 256` f32 = 8 MiB against 2 MiB of weights, so the activations
/// make the peak; the tail pattern's backward needs neither the mask nor
/// (under CCR) the output, so the stored-mask run holds 2 x 2 x 8 MiB more
/// at its peak, which sits in the backward.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_allocator_peak_drops_by_the_masks_and_outputs_not_kept() {
    let src = fixture(Site::Tail, 8192, 256, true, "");
    let key = run("mem_key", &src, PROD, false, &[]);
    let mask = run("mem_mask", &src, PROD, true, &[]);
    assert!(key.ok && mask.ok, "{}\n{}", key.stderr, mask.stderr);
    assert!(witness(&key).contains("2 of 2 output(s) freed"), "{}", key.stderr);
    let (k, m) = (key.peak_bytes(), mask.peak_bytes());
    eprintln!("allocator peak: key {k} B, stored mask {m} B, saved {} MiB", (m as i64 - k as i64) / (1 << 20));
    let tensor = 8192 * 256 * 4;
    assert!(m >= k + 3 * tensor, "the default saved {} B at the peak, want at least {} (3 of the 4 dropped tensors)", m as i64 - k as i64, 3 * tensor);
}

/// A checkpoint written by one mode resumes in the other: the GPU dropout
/// counter is the only RNG state a dropout leaves in a checkpoint, and both
/// modes claim it identically.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_resume_across_the_switch_continues_the_run_bit_for_bit() {
    let src = |cfg: &str| fixture(Site::Tail, 64, 128, true, cfg);
    let mut args = PROD.to_vec();
    args.push("--deterministic");
    let control = run("resume_control", &src(""), &args, false, &[]);
    assert!(control.ok, "{}", control.stderr);
    let want: Vec<&str> = control.losses().split_whitespace().collect();
    assert_eq!(want.len(), 8, "{}", control.stdout);
    for (writer_stored, reader_stored) in [(false, true), (true, false)] {
        let dir = std::env::temp_dir().join(format!("nsl_dropout_regen_{}_resume_{writer_stored}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Phase A: the first optimizer step (4 micro-batches), checkpoint.
        let a = run_in(
            &dir,
            &src(r#", checkpoint_save="ck.nslm", checkpoint_every=1"#).replace("epochs=8", "epochs=4"),
            &args,
            writer_stored,
            &[],
        );
        assert!(a.ok, "phase A failed:\n{}", a.stderr);
        // Phase B: resume in the other mode, out to the same 8 micro-batches.
        let b = run_in(&dir, &src(r#", checkpoint_load="ck.nslm""#), &args, reader_stored, &[]);
        assert!(b.ok, "phase B failed:\n{}", b.stderr);
        let got: Vec<&str> = a.losses().split_whitespace().chain(b.losses().split_whitespace()).collect();
        assert_eq!(got, want, "written with stored mask={writer_stored}, resumed with stored mask={reader_stored}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The counters of the `[cuda-graph] regions=...` exit banner.
fn graph_counters(stderr: &str) -> std::collections::HashMap<String, u64> {
    let line = stderr
        .lines()
        .filter(|l| l.starts_with("[cuda-graph] regions="))
        .next_back()
        .unwrap_or_else(|| panic!("no [cuda-graph] banner in stderr:\n{stderr}"));
    line.trim_start_matches("[cuda-graph] ")
        .split_whitespace()
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            Some((k.to_string(), v.parse().ok()?))
        })
        .collect()
}

/// `--cuda-graphs`: the regenerating backward reads its counter base from
/// the key in device memory, so its launch arguments are pointers and
/// constants and an adjoint region holding it digests the same every step,
/// as the stored-mask backward's does. A forward region holding a dropout
/// never captured (its counter base is an argument) and still does not. The
/// key path must capture and replay at least as much as the stored mask.
#[test]
#[ignore = "requires CUDA GPU"]
fn cuda_graph_regions_capture_no_less_with_the_key() {
    // 40 micro-batches in 10 windows: room for the record streak and replays.
    let src = fixture(Site::Tail, 64, 128, true, "").replace("epochs=8", "epochs=40");
    let args = ["--source-ad", "--checkpoint-blocks", "--checkpoint-selective", "--cuda-graphs"];
    let key = run("graphs_key", &src, &args, false, &[]);
    let mask = run("graphs_mask", &src, &args, true, &[]);
    assert!(key.ok && mask.ok, "{}\n{}", key.stderr, mask.stderr);
    let (k, m) = (graph_counters(&key.stderr), graph_counters(&mask.stderr));
    eprintln!("cuda graphs, key: {k:?}\ncuda graphs, stored mask: {m:?}");
    for c in ["captured", "replays"] {
        assert!(k.get(c) >= m.get(c), "{c}: key {:?} < stored mask {:?}", k.get(c), m.get(c));
    }
    assert!(k.get("replays").copied().unwrap_or(0) > 0, "nothing replayed: the comparison is vacuous");
}
