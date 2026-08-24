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
}

fn fixture(gpu: bool, shape: Shape) -> String {
    let (blocks, rows, d, ff, epochs) = match shape {
        Shape::Small => (4, 8, 128, 512, 2),
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
let tokens = unit.reshape([1, 128]).expand([16, 128]).contiguous().reshape([2048])
let loader = DataLoader(tokens, batch_size=4, seq_len=64, shuffle=false, drop_last=true)

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
        peak = if gpu {
            "print(\"PEAK_ACTIVATIONS\")\nprint(gpu_surface_at_peak_bytes(6))"
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

// NO GPU MEMORY GUARD HERE, stated so nobody mistakes the omission for
// coverage. Three fixture attempts each tripped a DIFFERENT pre-existing
// runtime quirk before the feature was exercised: (1) a host-resident
// `full()` input makes an upstream adjoint op emit CPU gradients — a
// single-threaded host backward that runs ~forever under the debug binary
// CARGO_BIN_EXE_nsl resolves to; (2) a bare `zeros().to(cuda)` target hits a
// `data_f64() on non-f64 tensor` abort in the loss path; (3) a 2D-logits
// cross_entropy variant aborts in `nsl_tensor_compare`. Each is logged in
// the memory bug ledger; none involves this feature (all reproduce with the
// kill switch set).
//
// The memory effect is pinned instead by models/benchmarks/
// SEGMENT_EARLY_FREE_2026_08_24.md: the 32-micro-step 1B probe (committed
// alongside) reproduces the full-epoch allocator peak to the byte and
// measures activations 16.54 GiB -> 3.66 GiB, with the fully-resident
// configuration going from OOM-in-forward to completing at 19.71 GiB. Item
// 12's re-profile re-measures it on every change to this machinery.

