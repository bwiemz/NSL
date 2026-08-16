//! Source-AD vs tape-AD dropout backward parity (the mask fix).
//!
//! The defect this gates: the source-AD adjoint for dropout used to multiply
//! the upstream gradient by the p ARGUMENT var (dx = dy·p/(1-p)) because the
//! RNG mask was never an SSA value — gradients decorrelated from the forward
//! that produced them. The two-op split (wengert.rs `DropoutMask` launch +
//! `Dropout` apply) makes the exact forward mask a first-class value the
//! adjoint consumes: dx = dy·mask·1/(1-p).
//!
//! Oracle: tape AD, whose `TapeOp::Dropout { saved_mask, scale }` backward
//! has always been mask-correct. Both modes draw the dropout mask from the
//! SAME deterministic CPU RNG (`sampling::rng_f64`, thread-local StdRng,
//! seeded via `--seed`), and the fixture is careful to consume NO other RNG
//! (deterministic init, no DataLoader shuffle, no sampling) — so with equal
//! seeds the tape run's masks ARE the source run's masks, per element, per
//! iteration. The forward kernels are line-identical (the source-AD FFI
//! `nsl_tensor_dropout_fwd_mask` reuses the tape kernel's rounding), so the
//! FIRST printed loss must match bit-for-bit; trained weights get a small
//! f32 tolerance (the source backward's 1/(1-p) constant is an f32 scalar
//! tensor, the tape's an f64 immediate).
//!
//! Discrimination check (why the tolerance still catches the old defect):
//! at p=0.5 the old backward scaled every gradient element by p/(1-p) = 1.0
//! — the EXPECTED value of the correct backward — but the correct backward
//! zeroes exactly the dropped elements and doubles the kept ones. Element-
//! wise weight comparison against the tape oracle diverges immediately.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// w1's gradient flows back THROUGH the dropout backward; w2's uses the
/// mask-applied forward output. 6 epochs advance the RNG stream across
/// iterations. Deterministic init, no DataLoader — no RNG draws besides
/// dropout itself.
fn fixture(p_expr: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model M:
    w1: Tensor = (arange(16).reshape([4, 4])) * 0.05
    w2: Tensor = (arange(16).reshape([4, 4])) * 0.03 + full([4, 4], 0.1)

    fn forward(self, x: Tensor, training: bool) -> Tensor:
        let h = relu(x @ self.w1)
        let d = dropout(h, {p_expr}, training)
        return d @ self.w2

let m = M()
let x = (arange(8).reshape([2, 4])) * 0.1 + full([2, 4], 0.1)
let y = zeros([2, 4])

print("LOSS_STREAM_BEGIN")
train(model=m, epochs=6, grad_accumulation=1):
    optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    step(batch):
        let pred = m.forward(x, true)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")

print("W1_BEGIN")
print(m.w1)
print("W1_END")
print("W2_BEGIN")
print(m.w2)
print("W2_END")
print("DROPOUT_PARITY_DONE")
"#
    )
}

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run(tag: &str, src: &str, source_ad: bool, extra_args: &[&str]) -> Run {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!(
        "nsl_dropparity_{}_{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut args: Vec<&str> = vec!["run", "--seed", "1234"];
    if source_ad {
        args.push("--source-ad");
    }
    args.extend_from_slice(extra_args);
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(&args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let res = Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    };
    let _ = std::fs::remove_dir_all(&tmp);
    res
}

/// Lines strictly between the two marker lines.
fn between<'a>(stdout: &'a str, begin: &str, end: &str) -> &'a str {
    let s = stdout.find(begin).unwrap_or_else(|| {
        panic!("marker {begin} missing in:\n{stdout}")
    }) + begin.len();
    let e = stdout[s..]
        .find(end)
        .unwrap_or_else(|| panic!("marker {end} missing in:\n{stdout}"));
    &stdout[s..s + e]
}

fn parse_floats(block: &str) -> Vec<f64> {
    block
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E'))
        .filter(|s| !s.is_empty() && s.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|s| s.parse::<f64>().ok())
        .collect()
}

fn assert_source_ad_actually_ran(r: &Run) {
    assert!(
        r.stderr.contains("Using source-to-source AD"),
        "source-AD run did not take the source-AD path:\n{}",
        r.stderr
    );
    assert!(
        !r.stderr.contains("falling back to tape-based AD"),
        "source-AD run silently fell back to tape — the parity assert would \
         be vacuous:\n{}",
        r.stderr
    );
    assert!(
        !r.stderr.contains("[source-ad] WARNING: dropout"),
        "the pre-fix wrong-backward warning is back:\n{}",
        r.stderr
    );
}

/// Weight-block parity: elementwise |a-b| within tol; anti-vacuity: the
/// weights must have MOVED off their deterministic init (a no-op train
/// would "match" trivially). The init closures carry the FULL init
/// expression — comparing W2 against a bare i*0.03 would count its +0.1
/// offset as "movement" and never fail (review finding on 6e3993ab).
fn assert_weight_parity(sa: &Run, tape: &Run, tol: f64) {
    let inits: [(&str, fn(usize) -> f64); 2] = [
        ("W1", |i| (i as f64) * 0.05),
        ("W2", |i| (i as f64) * 0.03 + 0.1),
    ];
    for (block, init) in inits {
        let begin = format!("{block}_BEGIN");
        let end = format!("{block}_END");
        let a = parse_floats(between(&sa.stdout, &begin, &end));
        let b = parse_floats(between(&tape.stdout, &begin, &end));
        assert_eq!(a.len(), 16, "{block}: expected 16 values, got {a:?}");
        assert_eq!(b.len(), 16, "{block}: expected 16 values, got {b:?}");
        let moved = b
            .iter()
            .enumerate()
            .any(|(i, v)| (v - init(i)).abs() > 1e-3);
        assert!(
            moved,
            "{block} never moved off its init — the fixture trained nothing"
        );
        for i in 0..16 {
            assert!(
                (a[i] - b[i]).abs() <= tol,
                "{block}[{i}]: source {} vs tape {} (|diff| {} > {tol})",
                a[i],
                b[i],
                (a[i] - b[i]).abs()
            );
        }
    }
}

/// The first printed loss precedes any weight update: it is a pure-forward
/// witness. With aligned RNG streams and the shared kernel rounding it must
/// match bit-for-bit — a mismatch means the masks (or the forward math)
/// diverged between modes.
fn assert_first_loss_bit_equal(sa: &Run, tape: &Run) {
    let sa_first = between(&sa.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END")
        .lines()
        .find(|l| !l.trim().is_empty())
        .expect("source loss stream empty")
        .trim()
        .to_string();
    let tape_first = between(&tape.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END")
        .lines()
        .find(|l| !l.trim().is_empty())
        .expect("tape loss stream empty")
        .trim()
        .to_string();
    assert_eq!(
        sa_first, tape_first,
        "first (pure-forward) loss differs — dropout masks or forward \
         rounding diverged between source-AD and tape-AD"
    );
}

fn parity_case(tag: &str, p_expr: &str, extra_source_args: &[&str], tol: f64) {
    let src = fixture(p_expr);

    // Determinism probe FIRST (run the control): the CPU path is a seeded
    // thread-local StdRng with no threading, so the tape reference must be
    // bit-reproducible. If this ever fails, every comparison below is
    // meaningless — fail loudly rather than compare noise.
    let tape_a = run(&format!("{tag}_tape_a"), &src, false, &[]);
    let tape_b = run(&format!("{tag}_tape_b"), &src, false, &[]);
    assert!(tape_a.ok, "tape run failed:\n{}", tape_a.stderr);
    assert!(tape_b.ok, "tape probe failed:\n{}", tape_b.stderr);
    assert_eq!(
        tape_a.stdout, tape_b.stdout,
        "tape-AD reference is not run-to-run deterministic on CPU — \
         cannot gate parity against a noisy oracle"
    );

    let sa = run(&format!("{tag}_sa"), &src, true, extra_source_args);
    assert!(sa.ok, "source-AD run failed:\n{}", sa.stderr);
    assert!(
        sa.stdout.contains("DROPOUT_PARITY_DONE"),
        "source-AD fixture did not finish:\n{}",
        sa.stdout
    );
    assert_source_ad_actually_ran(&sa);
    assert_first_loss_bit_equal(&sa, &tape_a);
    assert_weight_parity(&sa, &tape_a, tol);
}

#[test]
fn p_zero_elision_matches_tape() {
    // p=0: source-AD elides the op entirely; tape clones through the
    // runtime short-circuit. Neither draws RNG. Same tolerance as the
    // nonzero cases — source-vs-tape backward kernel orderings differ
    // slightly for the surrounding ops either way.
    //
    // The anti-vacuity check in assert_weight_parity doubles as the gate
    // for a second defect this fixture found: the tape short-circuit used
    // to return a BARE clone (no tape node), silently disconnecting the
    // graph at p=0 — w1 never received a gradient. The runtime now records
    // a same-shape Reshape relabel there, so the tape oracle trains w1.
    parity_case("p00", "0.0", &[], 2e-3);
}

#[test]
fn p01_source_backward_matches_tape_mask_backward() {
    parity_case("p01", "0.1", &[], 2e-3);
}

#[test]
fn p05_source_backward_matches_tape_mask_backward() {
    // p=0.5 is the discriminating case: the old broken backward scaled
    // gradients by exactly 1.0 (the expected value of the correct one) —
    // only the per-element mask pattern tells them apart.
    parity_case("p05", "0.5", &[], 2e-3);
}

#[test]
fn config_scalar_p_through_item_matches_tape() {
    // The house pattern: ctor-folded config field read via .item(). The
    // resolver must land the same p=0.5 as the literal case — and the
    // backward must still be mask-exact.
    let src = fixture("self._p.item()").replace(
        "model M:",
        "model M:\n    _p: Tensor = full([1], 0.5)",
    );
    let tape = run("cfg_tape", &src, false, &[]);
    assert!(tape.ok, "tape run failed:\n{}", tape.stderr);
    let sa = run("cfg_sa", &src, true, &[]);
    assert!(sa.ok, "source-AD run failed:\n{}", sa.stderr);
    assert_source_ad_actually_ran(&sa);
    assert_first_loss_bit_equal(&sa, &tape);
    assert_weight_parity(&sa, &tape, 2e-3);
}

/// A 2-block residual MLP with dropout inside each block. `--checkpoint-blocks`
/// segments on `blocks.N`-style param names ONLY (ccr.rs plan_impl via
/// wggo_graph::layer_prefix) — the flat w1/w2 fixture above makes CCR decline
/// and the composition test vacuous (review finding on 6e3993ab). Two dropout
/// sites per step also exercise multi-call RNG stream consumption.
fn fixture_blocks(p_expr: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Blk:
    w_up: Tensor = (arange(32).reshape([4, 8])) * 0.02
    w_down: Tensor = (arange(32).reshape([8, 4])) * 0.01

    fn forward(self, h: Tensor, training: bool) -> Tensor:
        let pre = h @ self.w_up
        let act = relu(pre)
        let d = dropout(act, {p_expr}, training)
        let out = d @ self.w_down
        return h + out

model Net:
    blocks: [Blk; 2] = Blk()

    fn forward(self, x: Tensor, training: bool) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h, training)
        return h

let m = Net()
let x = (arange(16).reshape([4, 4])) * 0.1
let y = zeros([4, 4])

print("LOSS_STREAM_BEGIN")
train(model=m, epochs=6, grad_accumulation=1):
    optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    step(batch):
        let pred = m.forward(x, true)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")

model_save(m, "ccr_dropout_parity.nslm")
print("DROPOUT_PARITY_DONE")
"#
    )
}

/// Like `run`, but keeps the temp dir alive long enough to read back the
/// saved model bytes (empty vec if the fixture never saved).
fn run_with_model_bytes(
    tag: &str,
    src: &str,
    source_ad: bool,
    extra_args: &[&str],
) -> (Run, Vec<u8>) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!(
        "nsl_dropparity_{}_{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut args: Vec<&str> = vec!["run", "--seed", "1234"];
    if source_ad {
        args.push("--source-ad");
    }
    args.extend_from_slice(extra_args);
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(&args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let run = Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    };
    let bytes = std::fs::read(tmp.join("ccr_dropout_parity.nslm")).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&tmp);
    (run, bytes)
}

fn loss_stream(r: &Run) -> Vec<f64> {
    parse_floats(between(&r.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END"))
}

#[test]
fn ccr_checkpoint_blocks_preserve_dropout_parity() {
    // CCR must force-save the mask (and the dropout output) rather than
    // replay the RNG: a replayed mask decorrelates the backward exactly
    // like the pre-fix defect. Two witnesses on a fixture CCR actually
    // segments:
    //   1. source+CCR is BIT-IDENTICAL to plain source (recompute replays
    //      are exact; RNG never redrawn) — the ccr_checkpoint_parity
    //      invariant, applied to a dropout-bearing model.
    //   2. plain source matches the tape oracle (mask-correct backward).
    let src = fixture_blocks("0.5");

    // Determinism probe (run the control).
    let (plain_a, bytes_a) = run_with_model_bytes("ccrb_plain_a", &src, true, &[]);
    let (plain_b, bytes_b) = run_with_model_bytes("ccrb_plain_b", &src, true, &[]);
    assert!(plain_a.ok, "plain source run failed:\n{}", plain_a.stderr);
    assert!(plain_b.ok, "plain source probe failed:\n{}", plain_b.stderr);
    assert_source_ad_actually_ran(&plain_a);
    assert_eq!(
        plain_a.stdout, plain_b.stdout,
        "plain source-AD baseline is not run-to-run deterministic on CPU"
    );
    assert_eq!(bytes_a, bytes_b, "baseline model bytes not deterministic");
    assert!(!bytes_a.is_empty(), "fixture never saved the model");

    // The CCR arm — must actually ENGAGE (anti-vacuity: every plan-decline
    // path prints "running without checkpointing").
    let (ccr, bytes_ccr) =
        run_with_model_bytes("ccrb_ccr", &src, true, &["--checkpoint-blocks"]);
    assert!(ccr.ok, "CCR run failed:\n{}", ccr.stderr);
    assert_source_ad_actually_ran(&ccr);
    assert!(
        !ccr.stderr.contains("running without checkpointing"),
        "CCR declined on the blocks fixture — the composition test is \
         vacuous:\n{}",
        ccr.stderr
    );
    assert_eq!(
        loss_stream(&ccr),
        loss_stream(&plain_a),
        "loss stream diverged under --checkpoint-blocks — a saved mask was \
         replayed or freed"
    );
    assert_eq!(
        bytes_ccr, bytes_a,
        "model bytes diverged under --checkpoint-blocks — the checkpointed \
         backward did not reproduce the baseline gradients"
    );

    // Mask-correctness on this fixture: plain source vs the tape oracle.
    let (tape, _) = run_with_model_bytes("ccrb_tape", &src, false, &[]);
    assert!(tape.ok, "tape run failed:\n{}", tape.stderr);
    assert_first_loss_bit_equal(&plain_a, &tape);
    let sl = loss_stream(&plain_a);
    let tl = loss_stream(&tape);
    assert_eq!(sl.len(), tl.len(), "loss stream lengths differ");
    assert!(sl.len() >= 6, "expected >=6 loss lines, got {}", sl.len());
    assert!(
        (sl.first().unwrap() - sl.last().unwrap()).abs() > 1e-6,
        "loss never changed — the fixture trained nothing"
    );
    for (i, (a, b)) in sl.iter().zip(tl.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 2e-3,
            "loss[{i}]: source {a} vs tape {b} (|diff| > 2e-3)"
        );
    }
}

/// GPU arm: the same parity claim on CUDA. GPU dropout masks come from a
/// process-global counter seeded 42 (`DROPOUT_SEED`, ignores --seed), so two
/// separate processes with the SAME dropout call order draw the SAME masks —
/// source vs tape alignment holds per-process. Looser tolerance (TF32/atomics
/// noise) and no first-loss bit-equality on GPU.
///
/// BLOCKED (2026-08-16): this fixture aborts on GPU under --source-ad in a
/// PRE-EXISTING defect UNRELATED to dropout — relu's backward reaches
/// `nsl_tensor_compare` (ad_ops.rs), which calls `data_f64()` on an f32
/// tensor. The control reproduces it with `dropout(h, 0.0, ...)`, i.e. with
/// the dropout op fully elided. Un-ignore once that compare dtype bug is
/// fixed in its own change.
#[test]
#[ignore = "requires CUDA GPU; blocked on a pre-existing source-AD GPU relu-backward dtype abort (nsl_tensor_compare data_f64 on f32) — reproduces with dropout elided"]
fn gpu_source_backward_matches_tape_mask_backward() {
    let src = fixture("0.5")
        .replace("let m = M()", "let m = M()\nm.to(cuda)")
        .replace(
            "let x = (arange(8).reshape([2, 4])) * 0.1 + full([2, 4], 0.1)",
            "let x = ((arange(8).reshape([2, 4])) * 0.1 + full([2, 4], 0.1)).to(cuda)",
        )
        .replace("let y = zeros([2, 4])", "let y = zeros([2, 4]).to(cuda)");
    let sa = run("gpu_sa", &src, true, &[]);
    assert!(sa.ok, "source-AD GPU run failed:\n{}", sa.stderr);
    assert_source_ad_actually_ran(&sa);
    let tape = run("gpu_tape", &src, false, &[]);
    assert!(tape.ok, "tape GPU run failed:\n{}", tape.stderr);
    assert_weight_parity(&sa, &tape, 2e-2);
}
