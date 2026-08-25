//! Item 11: per-segment forward early-free — the three guards.
//!
//! `--checkpoint-blocks` classified each block's interior activations as
//! recompute victims and freed them... in ONE list, lowered AFTER the whole
//! forward. Every segment's interiors were therefore still live at
//! end-of-forward — which is where the 1B global peak sits — so checkpointing
//! reduced the backward's activation wall and never the forward's. The
//! per-segment split lowers each segment's `FreeTensor` list at that
//! segment's end instead. Measured at 1B (32-micro-step probe, byte-identical
//! to the full-epoch peak): activations 16.54 GiB -> 3.66 GiB, and the
//! fully-resident configuration (no `--optim-state-offload`, f32 AdamW,
//! two-phase grad clip) goes from OOM-in-forward to completing.
//!
//! Two guards because each alone passes with the feature broken:
//!
//! 1. the WITNESS — the `[ccr] per-segment early-free` line with a non-zero
//!    count. Without it, guard 2 passes trivially when the feature silently
//!    stops engaging (both arms run the old path and of course agree).
//! 2. KILL-SWITCH BIT-PARITY — `NSL_CCR_SEGMENT_FREE=0` vs default must
//!    produce byte-identical loss streams under `--deterministic`. The
//!    feature moves WHEN storage returns to the allocator; a value change
//!    means it freed something that was still readable.
//!
//! There is deliberately NO GPU memory guard — see the note above the tests
//! for why, and models/benchmarks/SEGMENT_EARLY_FREE_2026_08_24.md for where
//! the memory numbers are pinned instead.

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

/// One fixture shape. The parity guard runs it on CPU under
/// `--deterministic`, where byte-parity is byte-parity and FLOPs buy
/// nothing: 4 blocks at d=128/ff=512, 2 epochs (14 optimizer steps of
/// free/realloc cycling). Follows the det_train_check/production pattern —
/// DataLoader batches + embedding + cross_entropy — because hand-rolled
/// variants each tripped a different pre-existing runtime quirk (see the
/// no-GPU-guard note below).
#[derive(Clone, Copy)]
enum Shape {
    Small,
    /// Sized for the GPU memory guards (item 2, 2026-08-25): per-segment
    /// interiors must be individually >= the caching allocator's 2 MB
    /// large-class quantum and collectively >> its 20 MB segment growth, or
    /// moving WHEN they free cannot move the measured peak (the
    /// tiny-fixture trap: an earlier KB-interior fixture measured IDENTICAL
    /// peaks with the witness firing). d=512/ff=2048 at 2048 rows makes
    /// each block's interiors ~16-32 MB — dozens of rounding quanta.
    GpuSized,
}

fn fixture(gpu: bool, shape: Shape) -> String {
    let (blocks, rows, d, ff, epochs) = match shape {
        Shape::Small => (4, 8, 128, 512, 2),
        Shape::GpuSized => (4, 8, 512, 2048, 1),
    };
    // seq_len 64 x batch `rows/64`... keep it simpler: batch 4, seq = rows.
    // The program follows the det_train_check/production pattern — DataLoader
    // batches + embedding + cross_entropy — because two hand-rolled variants
    // of this fixture each tripped a DIFFERENT pre-existing runtime quirk
    // before the feature was ever exercised: a host-resident `full()` input
    // makes an upstream adjoint op produce CPU gradients (single-threaded
    // host backward, ~forever under the debug binary), and a bare
    // `zeros().to(cuda)` target hits a `data_f64() on non-f64 tensor` abort
    // in the loss path. The production input path avoids both, and gating
    // THIS feature on the shapes production actually runs is the point.
    let _ = rows;
    format!(
        r#"from nsl.nn.losses import cross_entropy

model Blk:
    w1: Tensor = randn([{d}, {ff}]) * 0.02
    w2: Tensor = randn([{ff}, {d}]) * 0.02

    fn forward(self, x: Tensor) -> Tensor:
        let h = relu(x @ self.w1)
        return x + (h @ self.w2)

model Seg:
    embed: Tensor = randn([128, {d}]) * 0.1
    blocks: [Blk; {blocks}] = Blk()

    fn forward(self, ids: Tensor) -> Tensor:
        let sh = ids.shape
        let e = embedding_lookup(self.embed, ids.reshape([sh[0] * sh[1]]))
        let h = e.reshape([sh[0], sh[1], {d}]).reshape([sh[0] * sh[1], {d}])
        for block in self.blocks:
            h = block.forward(h)
        return h @ self.embed.transpose(0, 1)

let m = Seg()
{place}
let unit = arange(128)
let tokens = unit.reshape([1, 128]).expand([{tiles}, 128]).contiguous().reshape([{n_tokens}])
let loader = DataLoader(tokens, batch_size=4, seq_len={seq}, shuffle=false, drop_last=true)

print("LOSS_STREAM_BEGIN")
train(model = m, epochs = {epochs}):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let logits = m.forward(batch.input_ids)
        let ls = logits.shape
        let flat_labels = batch.labels.reshape([ls[0]])
        let loss = cross_entropy(logits, flat_labels)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")
{peak}
print("SEG_FREE_FIXTURE_DONE")
"#,
        place = if gpu { "m.to(cuda)" } else { "" },
        tiles = match shape {
            Shape::Small => 16,
            Shape::GpuSized => 128, // 16,384 tokens = 8 micro-steps of 4x512
        },
        n_tokens = match shape {
            Shape::Small => 2048,
            Shape::GpuSized => 16384,
        },
        seq = match shape {
            Shape::Small => 64,
            Shape::GpuSized => 512,
        },
        peak = if gpu {
            "print(\"PEAK_BYTES\")\nprint(gpu_peak_bytes())\nprint(\"PEAK_ACTIVATIONS\")\nprint(gpu_surface_at_peak_bytes(6))"
        } else {
            ""
        },
    )
}

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
}

/// `det` gates `--deterministic`. The parity guard NEEDS it (byte equality);
/// the memory guard must NOT pass it: deterministic mode routes reductions to
/// a single-threaded host path that is fine in release and takes tens of
/// minutes under the DEBUG binary `CARGO_BIN_EXE_nsl` resolves to — this gate
/// originally timed out at 50 minutes on exactly that, and the allocator
/// peak it measures is run-to-run stable without the flag (the 1B record's
/// EC1: byte-identical allocator peaks across non-deterministic arms).
fn run(tag: &str, gpu: bool, shape: Shape, det: bool, seg_free: bool) -> Run {
    run_with_events(tag, gpu, shape, det, seg_free, None)
}

fn run_with_events(
    tag: &str,
    gpu: bool,
    shape: Shape,
    det: bool,
    seg_free: bool,
    events: Option<&std::path::Path>,
) -> Run {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_segfree_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(&prog, fixture(gpu, shape)).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--seed", "11", "--checkpoint-blocks"])
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if det {
        cmd.arg("--deterministic");
    }
    if !seg_free {
        cmd.env("NSL_CCR_SEGMENT_FREE", "0");
    }
    if let Some(ev) = events {
        let _ = std::fs::remove_file(ev);
        cmd.env("NSL_EVENTS", ev);
    }
    let out = cmd.output().expect("spawn nsl run");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        ok: out.status.success(),
    }
}

fn losses(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .skip_while(|l| !l.contains("LOSS_STREAM_BEGIN"))
        .take_while(|l| !l.contains("LOSS_STREAM_END"))
        .map(|l| l.trim().to_string())
        // CPU losses print as bare floats, GPU ones as tensor([..]) — accept
        // either; byte comparison downstream needs the raw strings.
        .filter(|l| l.contains("tensor(") || l.parse::<f64>().is_ok())
        .collect()
}

fn witness_count(stderr: &str) -> Option<usize> {
    let line = stderr
        .lines()
        .find(|l| l.contains("per-segment early-free"))?;
    line.split(':')
        .nth(1)?
        .trim()
        .split(' ')
        .next()?
        .parse()
        .ok()
}

#[test]
fn per_segment_free_engages_and_is_bit_exact_with_the_kill_switch() {
    let on = run("on", false, Shape::Small, true, true);
    assert!(on.ok, "feature-on run failed:\n{}", on.stderr);

    // Guard 1: the witness. Zero interiors freed means the pass ran but did
    // nothing — on THIS fixture that is a regression, not a configuration.
    let n = witness_count(&on.stderr).unwrap_or_else(|| {
        panic!(
            "no per-segment early-free witness line — the feature did not \
             engage under --checkpoint-blocks on a blocks.N model:\n{}",
            on.stderr
        )
    });
    assert!(n > 0, "witness reports zero interiors freed:\n{}", on.stderr);

    let off = run("off", false, Shape::Small, true, false);
    assert!(off.ok, "kill-switch run failed:\n{}", off.stderr);
    assert!(
        witness_count(&off.stderr).is_none(),
        "NSL_CCR_SEGMENT_FREE=0 must run the post-forward path, but the \
         witness line still appeared:\n{}",
        off.stderr
    );

    // Guard 2: byte parity. Deterministic CPU, same binary, same seed — the
    // ONLY difference is when the frees are emitted.
    let (l_on, l_off) = (losses(&on.stdout), losses(&off.stdout));
    assert!(!l_on.is_empty(), "no losses captured:\n{}", on.stdout);
    assert_eq!(
        l_on, l_off,
        "per-segment freeing changed the numbers — it freed something that \
         was still readable"
    );
}

// The GPU memory guards below exist as of item 2 (2026-08-25). Their
// original absence was blocked by what looked like THREE runtime quirks and
// root-caused to TWO: (a) mixed-dtype `nsl_tensor_compare`/`nsl_tensor_where`
// aborted every GPU relu-backward Condition (misattributed at the time to
// mse targets and 2D-logits cross_entropy — both were this), fixed with
// per-operand dtype dispatch; (b) a host-resident dense-float input dragged
// the whole graph to the host via left-operand device reconciliation, now a
// train-entry REFUSAL (`nsl_train_input_device_guard`). This fixture uses
// relu deliberately: every GPU run of these guards also regression-tests (a).
//
// The 1B-scale numbers stay pinned by models/benchmarks/
// SEGMENT_EARLY_FREE_2026_08_24.md and the committed mem_probe_32step.nsl;
// the guards below pin the MECHANISM at fixture scale, sized above the
// allocator's rounding quanta (see Shape::GpuSized).

fn peak_value(stdout: &str, marker: &str) -> u64 {
    let mut lines = stdout.lines().skip_while(|l| l.trim() != marker);
    lines.next();
    lines
        .next()
        .and_then(|l| l.trim().parse::<f64>().ok())
        .map(|v| v as u64)
        .unwrap_or_else(|| panic!("no {marker} value in stdout:\n{stdout}"))
}

/// Peak guard: per-segment freeing must move the measured FORWARD activation
/// peak by a margin far above allocator granularity — a kill-switch A/B on
/// the same binary, non-deterministic (deterministic mode routes reductions
/// to a single-threaded host path that crawls under the debug test binary,
/// and allocator peaks are run-to-run byte-stable without it).
#[test]
#[ignore = "requires CUDA GPU (two training runs, kill-switch A/B)"]
fn gpu_forward_activation_peak_drops_with_per_segment_free() {
    let on = run("gpu_on", true, Shape::GpuSized, false, true);
    assert!(on.ok, "feature-on GPU run failed:\n{}", on.stderr);
    let n = witness_count(&on.stderr)
        .unwrap_or_else(|| panic!("no witness on GPU run:\n{}", on.stderr));
    assert!(n > 0, "witness reports zero interiors freed");

    let off = run("gpu_off", true, Shape::GpuSized, false, false);
    assert!(off.ok, "kill-switch GPU run failed:\n{}", off.stderr);

    let (on_act, off_act) = (
        peak_value(&on.stdout, "PEAK_ACTIVATIONS"),
        peak_value(&off.stdout, "PEAK_ACTIVATIONS"),
    );
    // 32 MiB margin. Measured on hardware (2026-08-25): the drop is 44 MiB
    // (on=127,142,400 off=174,098,432) — allocator reuse eats part of the
    // theoretical ~96 MiB, so the cushion is 1.4x, not 3x. Still an order
    // of magnitude above the 2 MiB block / 20 MiB segment quanta, and a
    // broken feature drops the delta to ~0.
    const MARGIN: u64 = 32 << 20;
    // Print the measurement so a passing run leaves auditable evidence
    // (review: the actual headroom was previously inferred, not recorded).
    println!(
        "PEAK_ACTIVATIONS on={on_act} off={off_act} drop={} MiB",
        (off_act.saturating_sub(on_act)) >> 20
    );
    assert!(
        on_act + MARGIN <= off_act,
        "per-segment freeing did not move the forward activation peak: \
         on={on_act} off={off_act} (need >= {MARGIN} drop). The witness \
         fired (n={n}), so the pass ran but freed nothing the peak felt — \
         the item-11 regression shape."
    );
}

/// Allocation-slope guard: the post-cleanup allocated-bytes series from
/// NSL_EVENTS must be FLAT from steady state on — byte equality, the
/// assertion class that catches what a peak assertion hides (the +8 MB/step
/// fused-SDPA LSE leak shipped INSIDE item 11's first draft and was found
/// by ramp, not by any peak). Exact bytes, every step, from one snapshot
/// per emission — no stderr parsing, no MB rounding.
#[test]
#[ignore = "requires CUDA GPU (one training run)"]
fn gpu_allocation_is_flat_across_steps() {
    let ev = std::env::temp_dir().join(format!("nsl_segfree_events_{}.jsonl", std::process::id()));
    let r = run_with_events("gpu_flat", true, Shape::GpuSized, false, true, Some(&ev));
    assert!(r.ok, "GPU run failed:\n{}", r.stderr);

    // Per step, the LAST gpu_mem_step event (highest seq) is the
    // post-cleanup snapshot — the step-boundary state a leak inflates.
    let text = std::fs::read_to_string(&ev).expect("events file");
    let mut by_step: std::collections::BTreeMap<i64, (i64, u64)> = Default::default();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v["kind"] != "gpu_mem_step" {
            continue;
        }
        let (Some(step), Some(seq), Some(alloc)) = (
            v["step"].as_i64(),
            v["seq"].as_i64(),
            v["fields"]["allocated_bytes"].as_u64(),
        ) else {
            continue;
        };
        let e = by_step.entry(step).or_insert((seq, alloc));
        if seq >= e.0 {
            *e = (seq, alloc);
        }
    }
    let series: Vec<(i64, u64)> = by_step.into_iter().map(|(s, (_, a))| (s, a)).collect();
    assert!(
        series.len() >= 6,
        "expected >= 6 stepped gpu_mem_step events, got {} — the fixture \
         shrank below the point where a slope is measurable:\n{text}",
        series.len()
    );
    // Steady state after the first two steps (allocator warm-up: pools grow,
    // caches fill). From there: byte equality.
    let steady = &series[2..];
    let first = steady[0].1;
    for &(step, alloc) in steady {
        assert_eq!(
            alloc, first,
            "allocated bytes moved at step {step}: {alloc} != {first} — a \
             per-step allocation slope (the LSE-leak class). Full series: \
             {series:?}"
        );
    }
    let _ = std::fs::remove_file(&ev);
}



/// Defect-1 refusal (item 2): a host-resident dense-float step input on a
/// GPU-parameter model must REFUSE at the first step with the fix named —
/// not silently reconcile every weight down to a single-threaded f64 host
/// graph (which under this debug test binary presents as a hang).
#[test]
#[ignore = "requires CUDA GPU (one refused run)"]
fn gpu_train_refuses_a_host_resident_float_input() {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_hostinput_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(
        &prog,
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = randn([64, 64]) * 0.02

    fn forward(self, x: Tensor) -> Tensor:
        return relu(x @ self.w)

let m = Tiny()
m.to(cuda)
let x = full([8, 64], 0.5)
let y = full([8, 64], 1.0).to(cuda)

train(model = m, epochs = 1):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
print("SHOULD_NOT_COMPLETE")
"#,
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--seed", "11"])
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "a host-resident float input on a GPU train must refuse; it completed:\n{stdout}"
    );
    assert!(
        stderr.contains("host-resident while the model's parameters are on the GPU"),
        "refusal must name the hazard and the fix; stderr:\n{stderr}"
    );
    assert!(
        !stdout.contains("SHOULD_NOT_COMPLETE"),
        "program ran to completion despite the refusal claim"
    );
}
