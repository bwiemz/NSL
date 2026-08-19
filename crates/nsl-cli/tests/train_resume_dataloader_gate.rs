//! Item 8 gates: a resumed run continues the DATA STREAM, not just θ.
//!
//! Milestone B made `checkpoint_load` restore θ, the AdamW moments and the
//! step counter — enough for `train_checkpoint_gate`'s loader-free fixture to
//! be bit-exact. It was not enough for a real run: the epoch loop restarted at
//! 0, the DataLoader restarted at batch 0, and the dropout RNG restarted from
//! wherever the process happened to be. A run interrupted 80% through epoch 3
//! resumed by re-reading epoch 0 from the top, under a late-training LR
//! schedule, with a fresh mask stream.
//!
//! The fixture below is built so all three are observable in the loss bytes:
//!
//! * `shuffle = true` over a corpus of DISTINCT token ids, embedded through a
//!   `randn` table — so the loss depends on WHICH batch arrived, and a loader
//!   that restarted at slot 0 prints different numbers;
//! * `dropout(..., 0.5, true)` — so the loss depends on the RNG position, and
//!   a resumed run drawing from a fresh stream prints different numbers.
//!
//! Bit-exactness against an uninterrupted control is therefore the whole
//! contract in one assertion. CPU + fixed corpus is deterministic, so this is
//! strict where strictness is cheap (same doctrine as train_checkpoint_gate).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// `train_cfg` is spliced into the train(...) arg list verbatim. 64 tokens /
/// (batch 2 x seq 4) = 8 batches per epoch.
fn fixture(train_cfg: &str) -> String {
    fixture_with_corpus(train_cfg, "arange(64)")
}

fn fixture_with_corpus(train_cfg: &str, corpus: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    emb: Tensor = randn([64, 4])

    fn forward(self, ids: Tensor) -> Tensor:
        let e = embedding_lookup(self.emb, ids.reshape([8]))
        return dropout(e, 0.5, true)

let m = Tiny()
let corpus = {corpus}
let loader = DataLoader(corpus, batch_size = 2, seq_len = 4, shuffle = true, drop_last = true, num_workers = 1)
let target = zeros([8, 4])

train(model = m{train_cfg}):
    optimizer: AdamW(lr = 0.01)
    step(batch):
        let pred = m.forward(batch.input_ids)
        let loss = mse_loss(pred, target)
    callbacks:
        on_step(step, loss):
            print(step)
            print(loss)

print("FIXTURE_DONE")
"#
    )
}

/// Same shape of program, but with no DataLoader at all — for the
/// loader-present/absent mismatch refusal.
fn loaderless_fixture(train_cfg: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    emb: Tensor = randn([64, 4])

    fn forward(self, ids: Tensor) -> Tensor:
        let e = embedding_lookup(self.emb, ids.reshape([8]))
        return dropout(e, 0.5, true)

let m = Tiny()
let ids = full([2, 4], 3.0)
let target = zeros([8, 4])

train(model = m{train_cfg}):
    optimizer: AdamW(lr = 0.01)
    step(batch):
        let pred = m.forward(ids)
        let loss = mse_loss(pred, target)

print("FIXTURE_DONE")
"#
    )
}

struct RunOut {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run_in(dir: &std::path::Path, name: &str, src: &str) -> RunOut {
    let root = repo_root();
    let prog = dir.join(name);
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad"])
        .arg(&prog)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    RunOut {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

fn fresh_dir(tag: &str) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("nsl_resumegate_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    tmp
}

/// (step, loss) pairs as RAW strings — the comparison is byte equality, not
/// parsed-float closeness.
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

/// Phase A: 2 epochs (16 micro-batches), saving every 3 optimizer steps. The
/// final save lands at step 15 — epoch 1, seven batches in — so the resume
/// point is MID-EPOCH, which is the case the old checkpoint could not express.
fn run_phase_a(tmp: &std::path::Path) -> RunOut {
    run_in(
        tmp,
        "phase_a.nsl",
        &fixture(r#", epochs = 2, checkpoint_save = "ck.nslm", checkpoint_every = 3"#),
    )
}

#[test]
fn resume_continues_the_data_stream_bit_exactly() {
    let tmp = fresh_dir("stream");

    let a = run_phase_a(&tmp);
    assert!(a.ok, "phase A failed:\n{}", a.stderr);
    assert_eq!(
        a.stdout.matches("FIXTURE_DONE").count(),
        1,
        "phase A must complete:\n{}",
        a.stdout
    );
    assert_eq!(
        loss_lines(&a.stdout).len(),
        16,
        "2 epochs x 8 batches:\n{}",
        a.stdout
    );
    // The witness that the saved position is mid-epoch, not an epoch boundary.
    assert!(
        a.stderr.contains("epoch 1 loader slot 7"),
        "the last save must record a MID-EPOCH position; without it this gate \
         would only prove boundary resume works:\n{}",
        a.stderr
    );

    // Phase B: new process, resume, run out to the total of 3 epochs.
    let b = run_in(
        &tmp,
        "phase_b.nsl",
        &fixture(r#", epochs = 3, checkpoint_load = "ck.nslm""#),
    );
    assert!(b.ok, "phase B failed:\n{}", b.stderr);
    assert!(
        b.stderr.contains("[checkpoint] resumed:")
            && b.stderr.contains("epoch 1 loader slot 7"),
        "resume witness must name the restored position:\n{}",
        b.stderr
    );

    // Control: 3 epochs uninterrupted, in its own dir so it cannot see the
    // checkpoint files.
    let ctl_dir = fresh_dir("stream_ctl");
    let c = run_in(&ctl_dir, "control.nsl", &fixture(", epochs = 3"));
    assert!(c.ok, "control failed:\n{}", c.stderr);

    let b_stream = loss_lines(&b.stdout);
    let c_stream = loss_lines(&c.stdout);
    assert_eq!(c_stream.len(), 24, "control must print 3 x 8 steps:\n{}", c.stdout);
    assert_eq!(
        b_stream.len(),
        9,
        "resumed run must finish epoch 1 (1 batch) and run epoch 2 (8):\n{}",
        b.stdout
    );
    // The contract. Every byte: the batch that arrives (loader slot +
    // permutation), the mask that is drawn (RNG position), the update that is
    // applied (θ, moments, step counter, LR schedule).
    assert_eq!(
        b_stream,
        c_stream[15..].to_vec(),
        "resumed trajectory diverged from the uninterrupted control — the \
         resumed run is not a continuation"
    );

    let _ = std::fs::remove_dir_all(&tmp);
    let _ = std::fs::remove_dir_all(&ctl_dir);
}

/// Grad accumulation moves the save cadence (interval = every x accum) and
/// therefore the slot the checkpoint lands on. The optimizer-step boundary is
/// also the only point at which the accumulation buffers are zeroed, so a
/// resume that mis-records the slot here would restart mid-window: the first
/// resumed micro-batches would be accumulated into a window whose earlier
/// members were already applied to θ.
#[test]
fn resume_is_bit_exact_under_grad_accumulation() {
    let tmp = fresh_dir("accum");
    // accum=2, every=3 → interval 6 micro-batches: saves at 6 and 12. Step 12
    // is epoch 1 (steps 9..16), four batches in.
    let a = run_in(
        &tmp,
        "phase_a.nsl",
        &fixture(
            r#", epochs = 2, grad_accumulation = 2, checkpoint_save = "ck.nslm", checkpoint_every = 3"#,
        ),
    );
    assert!(a.ok, "phase A failed:\n{}", a.stderr);
    assert!(
        a.stderr.contains("epoch 1 loader slot 4"),
        "the last save must land mid-epoch at an optimizer boundary:\n{}",
        a.stderr
    );

    let b = run_in(
        &tmp,
        "phase_b.nsl",
        &fixture(r#", epochs = 3, grad_accumulation = 2, checkpoint_load = "ck.nslm""#),
    );
    assert!(b.ok, "phase B failed:\n{}", b.stderr);

    let ctl_dir = fresh_dir("accum_ctl");
    let c = run_in(
        &ctl_dir,
        "control.nsl",
        &fixture(", epochs = 3, grad_accumulation = 2"),
    );
    assert!(c.ok, "control failed:\n{}", c.stderr);

    let b_stream = loss_lines(&b.stdout);
    let c_stream = loss_lines(&c.stdout);
    assert_eq!(c_stream.len(), 24, "control must print 3 x 8 steps:\n{}", c.stdout);
    assert_eq!(
        b_stream.len(),
        12,
        "resumed run must finish epoch 1 (4 batches) and run epoch 2 (8):\n{}",
        b.stdout
    );
    assert_eq!(
        b_stream,
        c_stream[12..].to_vec(),
        "resumed trajectory diverged under grad accumulation"
    );

    let _ = std::fs::remove_dir_all(&tmp);
    let _ = std::fs::remove_dir_all(&ctl_dir);
}

/// The sidecar's own format. The bit-exactness gates above would still pass if
/// a resume field were written under one key and read under the same WRONG
/// key, so pin the wire format directly — this is the artifact other tools and
/// future runtimes parse.
#[test]
fn sidecar_v2_header_carries_the_whole_resume_block() {
    let tmp = fresh_dir("format");
    let a = run_phase_a(&tmp);
    assert!(a.ok, "phase A failed:\n{}", a.stderr);

    let bytes = std::fs::read(tmp.join("ck.nslm.optim")).unwrap();
    assert_eq!(&bytes[0..4], b"NSLO");
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
    let header_size = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[16..16 + header_size]).unwrap();

    for key in [
        "\"step_count\":",
        "\"model_sig\":",
        "\"resume\":{",
        "\"train_epoch\":",
        "\"has_loader\":1",
        "\"loader_epoch\":",
        "\"loader_slot\":",
        "\"loader_id\":",
        "\"rng_seed\":\"",
        "\"rng_pos_hi\":",
        "\"rng_pos_lo\":",
        "\"gpu_dropout_ctr\":",
        // The --seed SCALAR, separate from the sampling stream's ChaCha key:
        // it keys the SR-BF16 and ZeRO dither directly.
        "\"global_seed\":",
        "\"global_seed_set\":",
    ] {
        assert!(header.contains(key), "sidecar header lacks {key}:\n{header}");
    }
    // No field may be negative: every header parser is a needle scan over
    // ASCII digits, so a '-' silently truncates the value being read.
    assert!(
        !header.contains(":-"),
        "a negative header field would be misparsed, not rejected:\n{header}"
    );
    // The RNG seed is 64 hex digits — a short or malformed one is refused at
    // load rather than half-parsed.
    let seed_start = header.find("\"rng_seed\":\"").unwrap() + "\"rng_seed\":\"".len();
    let seed_hex = &header[seed_start..seed_start + 64];
    assert!(
        seed_hex.chars().all(|c| c.is_ascii_hexdigit()),
        "rng_seed must be 64 hex digits, got {seed_hex}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// The data position only means something against the corpus it indexes.
/// Resuming "epoch 1, slot 7" over different data is a different seven
/// batches — and the permutation itself differs once the length changes.
#[test]
fn resume_refuses_a_foreign_corpus() {
    let tmp = fresh_dir("corpus");
    let a = run_phase_a(&tmp);
    assert!(a.ok, "phase A failed:\n{}", a.stderr);

    let b = run_in(
        &tmp,
        "phase_b.nsl",
        &fixture_with_corpus(
            r#", epochs = 3, checkpoint_load = "ck.nslm""#,
            "arange(64) + 1.0",
        ),
    );
    assert!(!b.ok, "a resume onto a different corpus must refuse:\n{}", b.stdout);
    assert!(
        b.stderr.contains("does not match") && b.stderr.contains("loader_id"),
        "refusal must name the identity mismatch:\n{}",
        b.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// `epochs` is the run TOTAL under resume. A value at or below the restored
/// epoch leaves the loop with nothing to do — zero steps, exit 0 — which is
/// the silent-no-op the codebase refuses elsewhere.
#[test]
fn resume_refuses_an_epoch_budget_already_spent() {
    let tmp = fresh_dir("epochs");
    let a = run_phase_a(&tmp);
    assert!(a.ok, "phase A failed:\n{}", a.stderr);

    let b = run_in(
        &tmp,
        "phase_b.nsl",
        &fixture(r#", epochs = 1, checkpoint_load = "ck.nslm""#),
    );
    assert!(
        !b.ok,
        "resuming into an exhausted epoch budget must refuse, not run zero \
         steps and exit 0:\n{}",
        b.stdout
    );
    assert!(
        b.stderr.contains("TOTAL for the run"),
        "refusal must explain the epochs semantics:\n{}",
        b.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Review finding (high): the epochs-budget refusal could not fire for a
/// DataLoader run.
///
/// With a loader, codegen records the IN-PROGRESS epoch, so a run's own final
/// checkpoint always has `train_epoch <= epochs - 1` and `train_epoch >=
/// epochs` was unreachable. A checkpoint landing on an epoch's LAST delivery
/// slot then passed every guard, drew an empty epoch, and exited 0 having run
/// no optimizer step — precisely the silent no-op the refusal exists for, in
/// the exact shape (`epochs = 1` + a DataLoader) that the 1B recipe uses.
///
/// The fix normalizes an exhausted epoch to the start of the next one at SAVE
/// time, so "epoch E is finished" is recorded as `E+1` and the budget check
/// sees it.
#[test]
fn resume_of_a_finished_run_refuses_instead_of_training_nothing() {
    let tmp = fresh_dir("finished");
    // 8 batches/epoch, every=4 → the last save is at micro-batch 16, which is
    // the final delivery slot of the final epoch: the run is complete.
    let a = run_in(
        &tmp,
        "phase_a.nsl",
        &fixture(r#", epochs = 2, checkpoint_save = "ck.nslm", checkpoint_every = 4"#),
    );
    assert!(a.ok, "phase A failed:\n{}", a.stderr);
    assert!(
        a.stderr.contains("at micro-batch step 16"),
        "the last save must be the run's final step:\n{}",
        a.stderr
    );
    // Normalized: the exhausted epoch 1 is recorded as epoch 2 slot 0, not
    // epoch 1 slot 8. Without this the refusal below cannot fire.
    assert!(
        a.stderr.contains("epoch 2 loader slot 0"),
        "an exhausted epoch must be normalized to the start of the next one, \
         or 'this run is finished' is unrepresentable:\n{}",
        a.stderr
    );

    let b = run_in(
        &tmp,
        "phase_b.nsl",
        &fixture(r#", epochs = 2, checkpoint_load = "ck.nslm""#),
    );
    assert!(
        !b.ok,
        "re-running a FINISHED recipe unchanged must refuse, not train zero \
         steps and exit 0:\nstdout:{}\nstderr:{}",
        b.stdout, b.stderr
    );
    assert!(
        b.stderr.contains("TOTAL for the run"),
        "refusal must explain the epochs semantics:\n{}",
        b.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Review finding: the `--seed` SCALAR is a live training-RNG input in its own
/// right — SR-BF16's per-element dither is `mix64(seed ^ step*SALT, ...)` and
/// the composed ZeRO-3 slice update reads the same global. Restoring θ, the
/// moments and the sampling stream while that silently changed would switch
/// every parameter's stochastic-rounding stream mid-run. The seed lives on the
/// command line, not in the recipe, so "re-run the recipe unchanged" makes
/// dropping it a routine slip.
#[test]
fn resume_refuses_a_different_global_seed() {
    let tmp = fresh_dir("seed");
    let root = repo_root();
    let src = fixture(r#", epochs = 2, checkpoint_save = "ck.nslm", checkpoint_every = 3"#);
    let prog = tmp.join("phase_a.nsl");
    std::fs::write(&prog, &src).unwrap();
    let a = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--seed", "1234"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(
        a.status.success(),
        "seeded phase A failed:\n{}",
        String::from_utf8_lossy(&a.stderr)
    );

    // Resume without the flag — everything else identical.
    let b = run_in(
        &tmp,
        "phase_b.nsl",
        &fixture(r#", epochs = 3, checkpoint_load = "ck.nslm""#),
    );
    assert!(
        !b.ok,
        "a resume under a different --seed must refuse:\n{}",
        b.stdout
    );
    assert!(
        b.stderr.contains("saved with --seed 1234"),
        "refusal must name the saved seed:\n{}",
        b.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// A loader checkpoint carries a data position that a loader-less train block
/// cannot apply; resuming anyway would silently train on a different stream.
#[test]
fn resume_refuses_a_train_block_without_the_saved_loader() {
    let tmp = fresh_dir("noloader");
    let a = run_phase_a(&tmp);
    assert!(a.ok, "phase A failed:\n{}", a.stderr);

    let b = run_in(
        &tmp,
        "phase_b.nsl",
        &loaderless_fixture(r#", epochs = 3, checkpoint_load = "ck.nslm""#),
    );
    assert!(!b.ok, "loader/loader-less mismatch must refuse:\n{}", b.stdout);
    assert!(
        b.stderr.contains("no DataLoader") && b.stderr.contains("model_load"),
        "refusal must name the mismatch and the weights-only escape hatch:\n{}",
        b.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// The pipelined train path (`@pipeline`) emits no checkpoint call at all, so
/// accepting the args would mean `checkpoint_save` silently writes nothing and
/// `checkpoint_load` silently trains from scratch. Milestone B added that
/// refusal; nothing pinned it, and item 8 gives it much more to protect (a
/// data position and RNG state that also would not be saved or restored). The
/// gate exists so a future edit to the pipelined dispatch cannot drop it
/// unnoticed.
#[test]
fn pipelined_train_path_refuses_checkpoint_config() {
    let tmp = fresh_dir("pipeline");
    let base = r#"from nsl.nn.losses import mse_loss

model Block:
    w: Tensor = (arange(16).reshape([4, 4])) * 0.02

    fn forward(self, h: Tensor) -> Tensor:
        return gelu(h @ self.w)

model Net:
    @pipeline(stages=2)
    blocks: [Block; 2] = Block()
    head: Tensor = (arange(12).reshape([4, 3])) * 0.03

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for b in self.blocks:
            h = b.forward(h)
        return h @ self.head

let m = Net()
let x = (arange(40).reshape([2, 5, 4])) * 0.01
let y = (arange(30).reshape([2, 5, 3])) * 0.01

train(model=m, epochs=2CKPT_ARGS):
    optimizer: AdamW(lr=0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print("FIXTURE_DONE")
"#;
    assert!(
        base.contains("@pipeline(stages=2)"),
        "the fixture must actually reach the pipelined path, or both arms below \
         would silently test the ordinary train path"
    );

    for (tag, cfg) in [
        ("save", r#", checkpoint_save = "ck.nslm", checkpoint_every = 1"#),
        ("load", r#", checkpoint_load = "ck.nslm""#),
    ] {
        let r = run_in(
            &tmp,
            &format!("pipe_{tag}.nsl"),
            &base.replace("CKPT_ARGS", cfg),
        );
        assert!(
            !r.ok,
            "checkpoint_{tag} under @pipeline must refuse, not silently no-op:\n{}",
            r.stdout
        );
        assert!(
            r.stderr.contains("pipelined train path")
                && r.stderr.contains(&format!("train 'checkpoint_{tag}'")),
            "the refusal must name BOTH the offending arg and the path — a \
             generic @pipeline error would pass a weaker check while the \
             checkpoint guard was gone ({tag}):\n{}",
            r.stderr
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// v1 sidecars (written before item 8) still load — θ, moments and the step
/// counter are honestly in them — but must say plainly that the data position
/// and RNG state are NOT restored. Constructed by rewriting the version word
/// of a real v2 file: the rest of the layout is unchanged between versions,
/// so this is exactly what an older runtime wrote.
#[test]
fn v1_sidecar_loads_but_warns_about_the_data_order() {
    let tmp = fresh_dir("v1");
    let a = run_phase_a(&tmp);
    assert!(a.ok, "phase A failed:\n{}", a.stderr);

    let sidecar = tmp.join("ck.nslm.optim");
    let mut bytes = std::fs::read(&sidecar).unwrap();
    assert_eq!(&bytes[0..4], b"NSLO", "sidecar magic changed");
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        2,
        "this runtime must be writing v2 sidecars"
    );
    bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&sidecar, &bytes).unwrap();

    let b = run_in(
        &tmp,
        "phase_b.nsl",
        &fixture(r#", epochs = 3, checkpoint_load = "ck.nslm""#),
    );
    assert!(b.ok, "a v1 sidecar must still load:\n{}", b.stderr);
    assert!(
        b.stderr.contains("v1 sidecar")
            && b.stderr.contains("NOT the data position")
            && b.stderr.contains("is not a continuation")
            // A v1 file also has no epoch, so `epochs` silently reverts to
            // its pre-item-8 "how many more" meaning and the budget refusal
            // cannot fire — the warning must say so, since the spec now
            // documents the total semantics unconditionally.
            && b.stderr.contains("PRE-item-8 meaning"),
        "a v1 resume must state exactly what it does not restore:\n{}",
        b.stderr
    );
    // And it behaves as advertised: epoch loop from 0, so all 3 epochs run.
    assert_eq!(
        loss_lines(&b.stdout).len(),
        24,
        "a v1 resume restarts the epoch loop — 3 full epochs:\n{}",
        b.stdout
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
