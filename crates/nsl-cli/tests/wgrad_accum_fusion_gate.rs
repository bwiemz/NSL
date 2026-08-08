//! Item 7 (`--fuse-wgrad-accum`) end-to-end gate.
//!
//! The flag collapses source-AD's weight-gradient chain
//! `Transpose -> Matmul -> reduce_to_shape` AND the FASE-Deferred accumulate
//! into one cuBLAS call writing straight into `m_partial`. That deletes two
//! tape ops and leaves their results UNMAPPED, which is the risky part: the
//! P0.2 grad-integrity guard treats an unresolved input on a live adjoint op
//! as a compile error precisely because the pre-#396 behaviour (skip it) would
//! zero a real gradient. This gate exists to prove the elision produces the
//! same trained weights rather than a quietly missing one.
//!
//! Three layers, because each catches something the others cannot:
//!
//! * **CPU parity** (default features) — the runtime's precondition check
//!   rejects CPU tensors and falls back to the exact decomposed chain, so the
//!   two arms must come out BIT-IDENTICAL. This isolates the *codegen*: if the
//!   plan mis-identifies a chain or rewires the operands wrongly, the numbers
//!   move even though the arithmetic is unchanged.
//! * **GPU parity** (`#[ignore]`, `--features cuda`) — exercises the actual
//!   fused GEMM, which is NOT bit-exact (cuBLAS sums the `B*T` products in its
//!   own order instead of rounding each per-batch partial first). Held to the
//!   repo's f32 training tolerance.
//! * **Refusals** — the compositions that would be silently wrong.
//!
//! Both the batched (`[B, T, d]`, real reduce) and 2-D (`[T, d]`, identity
//! reduce) shapes are covered at each layer; they take different ownership
//! paths in the runtime fallback.
//!
//! Every parity assertion is paired with a non-vacuity check on the
//! `[wgrad-fusion] N chain(s) fused` line. Without it, a tape change that
//! stops matching the pattern turns this gate green while testing nothing:
//! flag-on would simply BE flag-off.

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

/// `batched = true` feeds a 3-D `[B, T, d]` activation, so the weight adjoint
/// builds a `[B, d, o]` raw gradient — B times the parameter — that
/// `reduce_to_shape` then sums away. That temporary is what the fused GEMM
/// removes, and it is the case the flag exists for.
///
/// `batched = false` feeds a 2-D `[T, d]` activation, where the reduce is an
/// IDENTITY. The fusion still applies (N = T rather than B*T), and the
/// identity reduce is the branch with the delicate refcounting in the runtime
/// fallback — `nsl_tensor_reduce_to_shape` returns its input with a refcount
/// bump rather than a fresh tensor, so `raw` reaches rc=3 and is released by
/// three separate frees. Both shapes are exercised; an earlier version of this
/// gate claimed to cover the identity path but fed `w2` the 3-D result, so it
/// ran the batched path twice.
///
/// `grad_accumulation=4` + AdamW + `--source-ad` is what selects the
/// FASE-Deferred path, which owns the accumulate the fusion folds into.
/// Deterministic `arange` init (not `randn`) so the two arms are comparable
/// rather than merely similar.
/// The ACTIVATIONS have to be device-resident, not just the model. `m.to(cuda)`
/// alone leaves the inputs on the host, the whole forward runs there, and the
/// fused GEMM's `x.device == m.device` precondition fails on every call — the
/// runtime then falls back to the exact decomposed chain and the GPU arm
/// silently becomes a second CPU arm. (That is not hypothetical: this gate was
/// green that way until the `[wgrad-accum]` counter exposed 16 of 16 calls
/// falling back.)
fn fixture_src(cuda: bool, batched: bool) -> String {
    let (to_device, xs, ys) = if cuda {
        ("m.to(cuda)\n", "x.to(cuda)", "y.to(cuda)")
    } else {
        ("", "x", "y")
    };
    let (x_shape, y_shape) = if batched {
        ("(arange(40).reshape([2, 5, 4])) * 0.01", "(arange(30).reshape([2, 5, 3])) * 0.01")
    } else {
        ("(arange(20).reshape([5, 4])) * 0.01", "(arange(15).reshape([5, 3])) * 0.01")
    };
    format!(
        r#"from nsl.nn.losses import mse_loss

model Net:
    w1: Tensor = (arange(24).reshape([4, 6])) * 0.02
    w2: Tensor = (arange(18).reshape([6, 3])) * 0.03

    fn forward(self, x: Tensor) -> Tensor:
        let h = gelu(x @ self.w1)
        return h @ self.w2

let m = Net()
{to_device}let x = {x_shape}
let y = {y_shape}
let xd = {xs}
let yd = {ys}

train(model=m, epochs=8, grad_accumulation=4):
    optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    step(batch):
        let pred = m.forward(xd)
        let loss = mse_loss(pred, yd)

print("W1_BEGIN")
print(m.w1)
print("W1_END")
print("W2_BEGIN")
print(m.w2)
print("W2_END")
"#
    )
}

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
}

/// Compile-only arm: `nsl build --emit-obj` on an inline source, with the
/// object landing in a temp dir.
///
/// Used by every arm whose subject is a compile-time ADMISSION rather than
/// numerics. Two reasons not to reuse `run` for those: executing an 8-epoch
/// train loop to observe a stderr line is pure cost, and several of these
/// fixtures (`@pipeline`) cannot be executed at all. `-o` into the temp dir is
/// load-bearing — `nsl build` with no `-o` derives its output path from the
/// SOURCE stem, which is how a 128 MB object has twice reached a commit.
fn build_src(src: &str, stem: &str, extra: &[&str]) -> Run {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_wgrad_gate_{stem}"));
    std::fs::create_dir_all(&tmp).expect("mkdir fixture dir");
    let fixture = tmp.join(format!("{stem}.nsl"));
    std::fs::write(&fixture, src).expect("write fixture");
    let out = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["--", "build"])
        .arg(&fixture)
        .args(extra)
        .args(["--emit-obj", "-o"])
        .arg(tmp.join(format!("{stem}.o")))
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl build");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        ok: out.status.success(),
    }
}

fn run(fixture: &Path, cuda: bool, extra: &[&str]) -> Run {
    let root = repo_root();
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli"]);
    if cuda {
        cmd.args(["--features", "cuda"]);
    }
    cmd.args(["--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["--", "run", "--source-ad"]);
    if cuda {
        // Bit-reproducible GPU kernels, so the only difference between the
        // arms is the fusion itself and not nondeterministic reduction order.
        cmd.args(["--deterministic", "--target", "sm_89"]);
    }
    cmd.args(extra)
        .arg(fixture)
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // Which path the RUNTIME took. The codegen line below only proves the
        // call was emitted; this proves the GEMM ran instead of the silent
        // decomposed fallback. Harmless in both arms.
        .env("NSL_WGRAD_COUNTER", "1")
        // Lets the CCR arm prove checkpointing actually engaged. Both env vars
        // only add stderr lines — no arm's arithmetic depends on either.
        .env("NSL_CCR_DEBUG", "1");
    let out = cmd.output().expect("spawn nsl run");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        ok: out.status.success(),
    }
}

/// The loud decline. Emitted whenever `--fuse-wgrad-accum` is on and the
/// FASE-Deferred `on_param_grad` hook — the fusion's ONLY lowering path — is
/// absent. Before it existed such a build printed nothing at all: no count (the
/// `N chain(s) fused` line is gated on the same hook), no warning, no error.
///
/// A distinct suffix on the same token as the count, deliberately: the two are
/// mutually exclusive outcomes of one flag, and `fused_chains` below must keep
/// returning `None` for a decline rather than parsing it as a zero count.
const DECLINED: &str = "[wgrad-fusion] declined:";

/// Total chains fused, summed over every adjoint lowering in the compile.
/// `None` when the flag was off (the line is only emitted when it is on).
fn fused_chains(stderr: &str) -> Option<u64> {
    let mut total = None;
    for line in stderr.lines() {
        if let Some(rest) = line.trim().strip_prefix("[wgrad-fusion] ") {
            if let Some(n) = rest
                .split_once(" chain(s)")
                .and_then(|(n, _)| n.trim().parse::<u64>().ok())
            {
                total = Some(total.unwrap_or(0) + n);
            }
        }
    }
    total
}

/// `(fused GEMM calls, decomposed fallback calls)` from the
/// `NSL_WGRAD_COUNTER=1` atexit report.
fn wgrad_runtime_counts(stderr: &str) -> Option<(u64, u64)> {
    let rest = stderr
        .lines()
        .find_map(|l| l.trim().strip_prefix("[wgrad-accum] fused GEMM: "))?;
    let (fused, tail) = rest.split_once(", decomposed fallback: ")?;
    Some((fused.trim().parse().ok()?, tail.trim().parse().ok()?))
}

fn parse_between(stdout: &str, begin: &str, end: &str) -> Vec<f64> {
    let after = stdout.split_once(begin).map(|(_, r)| r).unwrap_or("");
    let inner = after.split_once(end).map(|(l, _)| l).unwrap_or("");
    inner
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

fn weights(stdout: &str) -> (Vec<f64>, Vec<f64>) {
    (
        parse_between(stdout, "W1_BEGIN", "W1_END"),
        parse_between(stdout, "W2_BEGIN", "W2_END"),
    )
}

/// The trained weights must have MOVED off their `arange` init. Without this
/// a gate comparing two all-zero-gradient runs would pass while proving
/// nothing about gradients at all.
fn assert_weights_trained(w1: &[f64], w2: &[f64]) {
    assert_eq!(w1.len(), 24, "expected 24 w1 values, got {}", w1.len());
    assert_eq!(w2.len(), 18, "expected 18 w2 values, got {}", w2.len());
    // w1[0] and w2[0] both init to exactly 0.0.
    assert!(
        w1[0].abs() > 1e-4,
        "w1[0] never moved off its 0.0 init ({}) — no gradient flowed, so a \
         parity comparison here would be vacuous",
        w1[0]
    );
    assert!(
        w2[0].abs() > 1e-4,
        "w2[0] never moved off its 0.0 init ({})",
        w2[0]
    );
}

fn write_fixture(name: &str, cuda: bool, batched: bool) -> PathBuf {
    let p = std::env::temp_dir().join(name);
    std::fs::write(&p, fixture_src(cuda, batched)).expect("write fixture");
    p
}

/// A `blocks.N`-shaped model, which `fixture_src`'s flat `Net` is NOT.
///
/// CCR refuses to checkpoint a tape with no `blocks.N`-style parameters
/// ("running without checkpointing"), so passing `--checkpoint-blocks` to the
/// flat fixture is a NO-OP — a CCR-interaction test written against it would
/// pass identically on the broken code, which is exactly the vacuous-gate
/// shape this file's header warns about. Three fused chains here: the two
/// block weights and the head.
fn ccr_fixture_src() -> String {
    r#"from nsl.nn.losses import mse_loss

model Block:
    w: Tensor = (arange(16).reshape([4, 4])) * 0.02

    fn forward(self, h: Tensor) -> Tensor:
        return gelu(h @ self.w)

model Net:
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

train(model=m, epochs=8, grad_accumulation=4):
    optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print("W1_BEGIN")
print(m.head)
print("W1_END")
"#
    .to_string()
}

/// Number of adjoint last-use `FreeTensor` markers CCR inserted, from the
/// `NSL_CCR_DEBUG=1` line. `None` when CCR never ran.
fn ccr_frees(stderr: &str) -> Option<u64> {
    stderr.lines().find_map(|l| {
        l.trim()
            .strip_prefix("[ccr] adjoint last-use frees inserted: ")
            .and_then(|n| n.trim().parse::<u64>().ok())
    })
}

/// A build that will NOT fuse must keep every CCR free marker it had.
///
/// The fix above lets CCR delete the `FreeTensor` markers for intermediates
/// a fused chain elides. That is only sound when the LOWERER goes on to
/// elide them — and the two sides plan from different gates: the lowerer
/// requires the FASE hook, while `--fuse-wgrad-accum` alone does not imply
/// it. `grad_accumulation == 1` makes FASE `Passthrough` for EVERY
/// optimizer, and `--pretrain-optimized` sets the flag with no accumulation
/// precondition — so on such a build CCR deleted markers for chains nobody
/// fused. Not a leak (the transpose result is a view the end-of-backward
/// bulk free sweeps) but it pins the base activation's buffer until then,
/// in exactly the pass whose purpose is cutting the adjoint peak.
/// Adversarial review caught it; measured at 3 lost markers.
///
/// Pinned by COUNT EQUALITY rather than an absolute number, so the arm
/// survives any change in how many frees the fixture warrants.
///
/// The flag-on arm goes through `--pretrain-optimized`, NOT an explicit
/// `--fuse-wgrad-accum`: a typed flag on a build that cannot fuse ANYWHERE is
/// a hard refusal (see `an_explicitly_typed_flag_refuses_where_the_bundle_only_warns`),
/// and this fixture's single train block is exactly that, so the bundle is the
/// only remaining way to reach flag-on-without-hook — and the bundle is where
/// the defect was found. The `[wgrad-fusion] declined:` assertion below is what
/// keeps that honest: if a future edit stops the bundle from setting the flag,
/// this arm would silently become a comparison of two identical builds.
///
/// ATTRIBUTION. Because the `on` arm is the whole bundle, the `off` arm has to
/// be the whole bundle MINUS `--fuse-wgrad-accum` — otherwise the two builds
/// differ by five flags and the free-count equality is not attributable to the
/// one this file is about. Three of the other members rewrite the adjoint tape,
/// and CCR's free count is a function of that tape: measured on a fixture with
/// an `RMSNorm` in the block, `--fuse-rmsnorm-backward` ALONE moves the count
/// from 43 to 16. This fixture has no norm and no fusable LM head, so today all
/// three contribute zero and both spellings measure the same number — which is
/// precisely why the confound would be invisible until the fixture grew a norm,
/// at which point the failure message would blame `--fuse-wgrad-accum` for 27
/// markers it did not touch.
///
/// `BUNDLE_MEMBERS_EXCEPT_WGRAD` is that hand-written mirror, and mirrors go
/// stale — so the arm also drift-checks `PretrainBundle`'s field count. A new
/// bundle member re-opens the confound silently otherwise.
#[test]
fn a_non_fusing_build_keeps_every_ccr_free_marker() {
    /// Every member of `--pretrain-optimized` EXCEPT `--fuse-wgrad-accum`.
    /// `--source-ad` is omitted because `run` already passes it.
    const BUNDLE_MEMBERS_EXCEPT_WGRAD: &[&str] = &[
        "--fuse-rmsnorm-backward",
        "--fuse-lm-head",
        "auto",
        "--wggo",
        "greedy",
        "--csha",
        "auto",
    ];
    // DRIFT CHECK. The list above mirrors `expand_pretrain_optimized`, which
    // this test cannot call (`PretrainBundle` is crate-private). Field count is
    // the cheapest thing that changes when a member is added: 5 flag fields
    // (wggo, csha, source_ad, fuse_rmsnorm_backward, fuse_lm_head) + 1 for
    // fuse_wgrad_accum + 1 for its provenance out-parameter = 7.
    let meta = std::fs::read_to_string(
        repo_root().join("crates").join("nsl-cli").join("src").join("meta_flags.rs"),
    )
    .expect("read meta_flags.rs");
    let body = meta
        .split_once("pub(crate) struct PretrainBundle {")
        .expect("PretrainBundle struct not found — did it move or get renamed?")
        .1
        .split_once("\n}")
        .expect("unterminated PretrainBundle struct")
        .0;
    let fields = body
        .lines()
        .filter(|l| l.trim_start().starts_with("pub(crate) ") && l.contains(':'))
        .count();
    assert_eq!(
        fields, 7,
        "PretrainBundle now has {fields} fields, not 7 — --pretrain-optimized \
         gained or lost a member. Update BUNDLE_MEMBERS_EXCEPT_WGRAD above to \
         match, or this arm's two builds stop differing only in \
         --fuse-wgrad-accum and its free-marker equality becomes a comparison \
         of two unrelated tapes."
    );

    let src = ccr_fixture_src()
        // SGD + no `grad_accumulation` ⇒ FASE Passthrough ⇒ no hook ⇒ the
        // lowerer never plans, so nothing may be elided.
        .replace(
            "train(model=m, epochs=8, grad_accumulation=4):",
            "train(model=m, epochs=8):",
        )
        .replace(
            "optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)",
            "optimizer: SGD(lr=0.01)",
        );
    assert!(
        src.contains("optimizer: SGD") && !src.contains("grad_accumulation"),
        "the fixture rewrite matched nothing — this arm would silently test \
         the fusing configuration instead"
    );
    let p = std::env::temp_dir().join("nsl_wgrad_accum_gate_ccr_nofuse.nsl");
    std::fs::write(&p, src).expect("write fixture");

    // The bundle minus the flag under test, spelled out — see the header.
    let mut off_args: Vec<&str> = BUNDLE_MEMBERS_EXCEPT_WGRAD.to_vec();
    off_args.push("--checkpoint-blocks");
    let off = run(&p, false, &off_args);
    assert!(off.ok, "flag-off run failed:\n{}", off.stderr);
    let on = run(&p, false, &["--pretrain-optimized", "--checkpoint-blocks"]);
    assert!(on.ok, "flag-on run failed:\n{}", on.stderr);
    // The `off` arm must really be flag-OFF. `--pretrain-optimized` is the only
    // thing that sets `--fuse-wgrad-accum` here, but spelling out its members
    // one by one is exactly the edit that could accidentally include it.
    assert!(
        !off.stderr.contains("[wgrad-fusion]"),
        "the `off` arm reports wgrad activity, so it is not flag-off and the \
         equality below compares two flag-on builds:\n{}",
        off.stderr
    );

    // ANTI-VACUITY: the bundle really did turn the fusion on, and it really
    // could not fire. Both halves matter — a bundle that stopped setting the
    // flag, or a train block that started reaching the hook, would each make
    // the free-count equality below a tautology.
    assert!(
        on.stderr.contains(DECLINED),
        "the bundle arm shows no `{DECLINED}` line, so either \
         --pretrain-optimized no longer sets --fuse-wgrad-accum or this \
         fixture now reaches the FASE hook; either way the comparison below \
         no longer tests anything:\n{}",
        on.stderr
    );
    // Premise: this configuration really does not fuse. If it ever starts
    // to, the equality below stops being the property under test.
    assert_eq!(
        fused_chains(&on.stderr),
        None,
        "this arm requires a NON-fusing build; the lowerer planned:\n{}",
        on.stderr
    );

    let frees_off = ccr_frees(&off.stderr).expect("CCR must run (flag off)");
    let frees_on = ccr_frees(&on.stderr).expect("CCR must run (flag on)");
    assert!(frees_off > 0, "CCR inserted no frees — nothing to compare");
    assert_eq!(
        frees_on, frees_off,
        "--fuse-wgrad-accum removed {} CCR free marker(s) on a build that \
         fuses NOTHING — those activations now live to the end-of-backward \
         bulk free",
        frees_off.saturating_sub(frees_on)
    );
}

/// `--checkpoint-blocks` must not silently switch the fusion off.
///
/// CCR's adjoint last-use freeing computes `last_use[a_t] = matmul_idx` and
/// so lands a `FreeTensor` at `matmul_idx + 1` — exactly between the matmul
/// and the reduce. That breaks BOTH of `wgrad_fusion::plan`'s preconditions
/// at once (the contiguity walk, and `a_t`'s single-reader count, since the
/// marker is itself a reader). The failure was silent and total: on every
/// checkpointed build the flag fused ZERO chains and emitted not one
/// `nsl_tensor_wgrad_accum` call, while still reporting success — i.e. the
/// shipped feature was inert on exactly the checkpointed pretraining
/// configuration it was built for. Nothing in this file passed both flags,
/// so no gate could see it.
///
/// Measured on this fixture: main = 0 chains / 0 runtime calls;
/// fixed = 3 chains / 24 runtime calls.
#[test]
fn checkpoint_blocks_does_not_disable_the_fusion() {
    let p = std::env::temp_dir().join("nsl_wgrad_accum_gate_ccr.nsl");
    std::fs::write(&p, ccr_fixture_src()).expect("write fixture");

    let off = run(&p, false, &["--checkpoint-blocks"]);
    assert!(off.ok, "flag-off run failed:\n{}", off.stderr);
    let on = run(&p, false, &["--fuse-wgrad-accum", "--checkpoint-blocks"]);
    assert!(on.ok, "flag-on run failed:\n{}", on.stderr);

    // ANTI-VACUITY 1 — CCR actually ran. If the fixture ever loses its
    // `blocks.N` shape, CCR declines, `--checkpoint-blocks` becomes a no-op,
    // and everything below would pass on the BROKEN code too.
    let frees = ccr_frees(&on.stderr).unwrap_or_else(|| {
        panic!(
            "no `[ccr] adjoint last-use frees inserted` line — CCR did not run, \
             so this arm is not testing the interaction it exists for.\n\
             stderr:\n{}",
            on.stderr
        )
    });
    assert!(
        frees > 0,
        "CCR inserted no last-use frees, so nothing could have broken the \
         chain's contiguity — the regression this arm pins is unreachable here"
    );
    assert!(
        !on.stderr.contains("running without checkpointing"),
        "CCR declined to checkpoint this fixture, so --checkpoint-blocks was a \
         no-op:\n{}",
        on.stderr
    );

    // ANTI-VACUITY 2 — the plan admitted the chains (0 on main).
    let fired = fused_chains(&on.stderr).unwrap_or(0);
    assert!(
        fired >= 3,
        "expected >= 3 fused chains under --checkpoint-blocks (two block \
         weights + the head), got {fired}. This is the regression itself: \
         CCR's free placement takes the count to 0.\nstderr:\n{}",
        on.stderr
    );

    // ANTI-VACUITY 3 — and the FFI was actually CALLED. The codegen line
    // above only proves the plan admitted a chain; this proves the emitted
    // call ran. On main both read zero, which is what made the inertness
    // invisible.
    let (gemm, fallback) = wgrad_runtime_counts(&on.stderr).unwrap_or_else(|| {
        panic!("no [wgrad-accum] counter line\nstderr:\n{}", on.stderr)
    });
    assert_eq!(gemm, 0, "the cuBLAS path fired in a non-cuda build");
    assert!(
        fallback >= 3,
        "expected at least one decomposed-fallback call per fused chain, got \
         {fallback} — the chains were planned but never executed"
    );

    // The protection must not merely move the breakage: the compiler's own
    // invariant check reports any chain the free placement still kills.
    assert!(
        !on.stderr.contains("last-use freeing broke"),
        "the compiler reported losing fusion chains to free placement:\n{}",
        on.stderr
    );

    // Parity: on CPU every call takes the exact decomposed fallback, so the
    // arithmetic is unchanged and any drift is a CODEGEN defect — a
    // mis-planned chain, a mis-wired operand, or a gradient freed before the
    // GEMM that consumes it reads it.
    let (on_w1, on_w2) = weights(&on.stdout);
    let (off_w1, off_w2) = weights(&off.stdout);
    assert_eq!(on_w1.len(), 12, "expected the 12 head values, got {:?}", on_w1.len());
    assert!(
        on_w1[0].abs() > 1e-6,
        "head[0] never moved off its 0.0 init — no gradient flowed, so this \
         parity comparison would be vacuous"
    );
    assert_eq!(
        on_w1, off_w1,
        "the head weight differs under --checkpoint-blocks, where the fused \
         path falls back to the exact decomposed chain — a codegen defect"
    );
    assert_eq!(on_w2, off_w2, "trailing parse block differs (see above)");
}

// ─────────── The silent-inertness pair: accum >= 2 vs accum == 1 ───────────

/// `grad_accumulation` is the whole difference between a fusion that fires and
/// one that cannot, and until now the second case said NOTHING.
///
/// `--fuse-wgrad-accum` has exactly one lowering path: the FASE-Deferred
/// `on_param_grad` hook. `fase::plan` returns `Passthrough` unconditionally at
/// `accumulation == 1`, and a train block that omits `grad_accumulation`
/// defaults to 1 — so the hook never exists, `wgrad_fusion::plan` is never
/// called, the sole `nsl_tensor_wgrad_accum` emission site is unreachable, and
/// the `[wgrad-fusion] N chain(s) fused` counter (the anti-vacuity instrument
/// for every other arm in this file) is ITSELF gated on the hook. Measured on
/// the unmodified branch: `nsl run --source-ad --fuse-wgrad-accum` on the
/// accum-1 form of this fixture exited 0 with not one `wgrad` line on stderr.
///
/// The two arms are the SAME fixture differing ONLY in `grad_accumulation`, so
/// neither can pass for the wrong reason: whatever makes one green is exactly
/// what makes the other red.
#[test]
fn removing_grad_accumulation_turns_the_fusion_from_firing_into_a_refusal() {
    let fusing = ccr_fixture_src();
    let inert = fusing.replace(
        "train(model=m, epochs=8, grad_accumulation=4):",
        "train(model=m, epochs=8):",
    );
    assert!(
        fusing.contains("grad_accumulation=4") && !inert.contains("grad_accumulation"),
        "the fixture rewrite matched nothing — the two arms would be identical"
    );

    // ARM 1 — the POSITIVE control. Three trainable weights (two block
    // weights + the head) ⇒ three chains. If the decline path ever swallowed
    // the working configuration this goes red first.
    let p_fusing = std::env::temp_dir().join("nsl_wgrad_accum_gate_accum4.nsl");
    std::fs::write(&p_fusing, &fusing).expect("write fixture");
    let ok = run(&p_fusing, false, &["--fuse-wgrad-accum"]);
    assert!(ok.ok, "the accum=4 arm must still build:\n{}", ok.stderr);
    let fired = fused_chains(&ok.stderr).unwrap_or(0);
    assert!(
        fired >= 3,
        "expected >= 3 fused chains at grad_accumulation=4, got {fired} — the \
         decline path broke the working case\nstderr:\n{}",
        ok.stderr
    );
    assert!(
        !ok.stderr.contains(DECLINED),
        "a build that FUSED must not also report a decline:\n{}",
        ok.stderr
    );

    // ARM 2 — RED ON REVERT. Same fixture, `grad_accumulation` gone, flag
    // typed explicitly. On the unmodified branch this exits 0 in silence.
    let p_inert = std::env::temp_dir().join("nsl_wgrad_accum_gate_accum1.nsl");
    std::fs::write(&p_inert, &inert).expect("write fixture");
    let refused = run(&p_inert, false, &["--fuse-wgrad-accum"]);
    assert!(
        !refused.ok,
        "an explicit --fuse-wgrad-accum on a train block with no \
         grad_accumulation must REFUSE — it can fuse nothing, and a flag that \
         silently does nothing is the exact defect this gate exists for.\n\
         stderr:\n{}",
        refused.stderr
    );
    assert!(
        refused.stderr.contains("grad_accumulation"),
        "the refusal must name grad_accumulation — that is the one thing the \
         user has to change:\n{}",
        refused.stderr
    );
    assert!(
        refused.stderr.contains(DECLINED),
        "the refusal must be preceded by the `{DECLINED}` note, which is the \
         line the bundle path relies on:\n{}",
        refused.stderr
    );
    assert_eq!(
        fused_chains(&refused.stderr),
        None,
        "the decline note must not parse as a chain count — `fused_chains` is \
         what every other arm's non-vacuity check reads:\n{}",
        refused.stderr
    );
}

/// The shipped pretraining script, in the exact state it ships in.
///
/// `models/coder50m/pretrain_cert.nsl` declares `train(model=m, epochs=1,
/// grad_clip=1.0)` — no `grad_accumulation` — and `--pretrain-optimized` turns
/// `--fuse-wgrad-accum` on with no accumulation precondition. So the flagship
/// "optimized pretraining" configuration fused ZERO chains and said nothing
/// about it, which is how a +11% tok/s measurement came to be credited to
/// "both backward fusions" when only one of them ran.
///
/// Two properties, and the pairing is the point:
///
/// * the decline NOTE is present — the inertness is now on the record; and
/// * the exit code is ZERO — the bundle warns rather than refusing, because
///   erroring would break both shipped scripts and the benchmark matrix.
///
/// The day someone adds `grad_accumulation` to this script the first assertion
/// goes red, which is the correct outcome: the pin exists to be re-examined,
/// not to freeze the script.
#[test]
fn the_shipped_pretraining_script_declines_the_wgrad_fusion_without_failing() {
    let dir = repo_root().join("models").join("coder50m");
    let tmp = tempfile::TempDir::new().unwrap();
    // `-o` into a temp dir: `nsl build` with no `-o` derives its output path
    // from the SOURCE stem, which drops a multi-MB object into the repo.
    let out = tmp.path().join("pretrain_cert.o");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--manifest-path"])
        .arg(repo_root().join("Cargo.toml"))
        .args(["--", "build", "pretrain_cert.nsl", "--pretrain-optimized", "--emit-obj", "-o"])
        .arg(&out)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl build");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    assert!(
        output.status.success(),
        "the BUNDLE must not turn a shipped script into a build failure:\n{stderr}"
    );
    assert!(
        stderr.contains(DECLINED),
        "models/coder50m/pretrain_cert.nsl under --pretrain-optimized reports \
         no `{DECLINED}` line. Either the fusion now fires here (add \
         grad_accumulation to this assertion's reasoning and pin the chain \
         count instead) or the decline went silent again:\n{stderr}"
    );
    assert!(
        stderr.contains("bundle partially disabled"),
        "the decline must be phrased as a bundle partial-disable, matching the \
         bundle's other notes:\n{stderr}"
    );
    assert_eq!(
        fused_chains(&stderr),
        None,
        "this pin asserts the shipped script is INERT; it fused something:\n{stderr}"
    );
}

/// EXPLICIT vs BUNDLE, on ONE program. This is what stops a later edit from
/// collapsing the two paths into whichever is easier.
///
/// Same source file, same precondition failure, opposite outcomes: typed →
/// non-zero exit, bundle-filled → zero. Adding `--fuse-wgrad-accum` to a
/// `--pretrain-optimized` invocation must flip it, because
/// `expand_pretrain_optimized` reads the clap value before overwriting it.
#[test]
fn an_explicitly_typed_flag_refuses_where_the_bundle_only_warns() {
    let dir = repo_root().join("models").join("coder50m");
    let tmp = tempfile::TempDir::new().unwrap();

    let build = |extra: &[&str], stem: &str| {
        let out = tmp.path().join(format!("{stem}.o"));
        let output = Command::new(env!("CARGO"))
            .args(["run", "-q", "-p", "nsl-cli", "--manifest-path"])
            .arg(repo_root().join("Cargo.toml"))
            .args(["--", "build", "pretrain_cert.nsl", "--pretrain-optimized"])
            .args(extra)
            .args(["--emit-obj", "-o"])
            .arg(&out)
            .current_dir(&dir)
            .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
            .output()
            .expect("spawn nsl build");
        (
            output.status.success(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        )
    };

    let (bundle_ok, bundle_err) = build(&[], "bundle");
    let (typed_ok, typed_err) = build(&["--fuse-wgrad-accum"], "typed");

    assert!(
        bundle_ok,
        "bundle-only must succeed (it is what the shipped scripts run):\n{bundle_err}"
    );
    assert!(
        !typed_ok,
        "`--pretrain-optimized --fuse-wgrad-accum` must REFUSE: the user typed \
         the flag, and it cannot fuse anything on this train block. If this \
         passes, the bundle laundered the provenance and every explicit \
         --fuse-wgrad-accum is now a silent no-op again:\n{typed_err}"
    );
    // Both arms must have reached the same decline — otherwise the exit-code
    // difference could come from some unrelated failure.
    for (label, err) in [("bundle", &bundle_err), ("typed", &typed_err)] {
        assert!(
            err.contains(DECLINED),
            "{label} arm never reached the decline, so the exit-code contrast \
             above proves nothing:\n{err}"
        );
    }
}

// ───────────── The SCOPE of the refusal: compile, not train block ─────────────

/// `--fuse-wgrad-accum` is per COMPILE; its precondition is per TRAIN BLOCK.
/// Reconciling those the wrong way kills builds the flag is helping.
///
/// The admission first shipped inside `compile_train_block_inner`, so the FIRST
/// block that could not fuse aborted the whole compile — even when an earlier
/// block had already fused real chains and said so two stderr lines above the
/// fatal error. The refusal's own justification ("a flag the user typed that
/// cannot do anything") is a claim about the PROGRAM, and it was false there.
///
/// This pins the reconciliation: NOTE per block, REFUSAL per compile.
///
/// Deliberately asserts the positive count as well as the exit code. Asserting
/// only `ok` would pass if the flag were removed from clap, if the second block
/// stopped compiling, or if the admission were deleted outright — none of which
/// is the property. The pairing (fused >= 3 AND a decline AND exit 0) is only
/// satisfiable by a build in which both halves happened.
#[test]
fn a_declining_block_does_not_kill_a_compile_where_another_block_fused() {
    // Block 1 accumulates (fuses); block 2 deliberately does not (a warm-up,
    // fine-tune or eval pass). Same model, same optimizer family — so the ONLY
    // thing that differs between them is `grad_accumulation`.
    let src = ccr_fixture_src().replace(
        "print(\"W1_BEGIN\")",
        "train(model=m, epochs=1):\n\
         \x20   optimizer: AdamW(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)\n\
         \x20   step(batch):\n\
         \x20       let pred = m.forward(x)\n\
         \x20       let loss = mse_loss(pred, y)\n\n\
         print(\"W1_BEGIN\")",
    );
    assert!(
        src.matches("train(model=m").count() == 2,
        "the fixture rewrite did not produce two train blocks — this arm would \
         then be a duplicate of the single-block case"
    );

    let typed = build_src(&src, "two_blocks_typed", &["--source-ad", "--fuse-wgrad-accum"]);
    assert!(
        typed.ok,
        "a typed --fuse-wgrad-accum must NOT fail a compile in which it fused: \
         block 1 accumulates, only block 2 cannot. Before the refusal was moved \
         to compile scope this exited 1 with `3 chain(s) fused` printed two \
         lines above the fatal error.\nstderr:\n{}",
        typed.stderr
    );
    let fired = fused_chains(&typed.stderr).unwrap_or(0);
    assert!(
        fired >= 3,
        "expected >= 3 fused chains from block 1, got {fired} — if this is 0 \
         the arm is passing because nothing fuses anywhere, not because the \
         scope is right\nstderr:\n{}",
        typed.stderr
    );
    assert!(
        typed.stderr.contains(DECLINED),
        "block 2 fuses nothing and must still be named — dropping the note is \
         how the flag went silent in the first place:\n{}",
        typed.stderr
    );
    // Which block. A compile-scoped refusal deliberately does not fire here,
    // so this line is the ONLY thing that tells the user where the inert
    // block is.
    assert!(
        typed.stderr.contains("train block #2"),
        "the decline must identify the block; a bare `{DECLINED}` note in a \
         multi-block program is not actionable:\n{}",
        typed.stderr
    );

    // The bundle must reach the same outcome, or the exit-code contrast the
    // sibling arms draw would be coming from somewhere else.
    let bundle = build_src(&src, "two_blocks_bundle", &["--pretrain-optimized"]);
    assert!(
        bundle.ok,
        "the bundle arm on the same program must build:\n{}",
        bundle.stderr
    );
}

/// A typed flag that fused nothing ANYWHERE still refuses — including the two
/// shapes the per-block check could not see.
///
/// Both were live holes in the first version of this admission:
///
/// * **no train block at all** — `compile_train_block_inner` never runs, so
///   nothing evaluated the precondition and nothing printed; and
/// * **`@pipeline`** — `compile_train_block` returns via
///   `compile_train_block_pipelined` BEFORE the admission, and that lowering
///   passes `None` for `on_param_grad`, which also suppresses the
///   `N chain(s) fused` counter. The flag was exactly as silently inert there
///   as it had been everywhere before this gate existed.
///
/// The `@pipeline` arm additionally pins that the refusal arrives as a clean
/// diagnostic rather than the cranelift panic that path dies with a moment
/// later — an assertion on the exit code alone could not tell those apart,
/// since both are non-zero.
#[test]
fn a_typed_flag_that_can_fuse_nowhere_refuses_even_off_the_ordinary_path() {
    // ── no train block ──────────────────────────────────────────────
    let no_train = {
        let src = ccr_fixture_src();
        let (head, tail) = src.split_once("train(model=m").expect("fixture has a train block");
        let after = tail.split_once("print(\"W1_BEGIN\")").expect("fixture prints").1;
        format!("{head}print(\"W1_BEGIN\"){after}")
    };
    assert!(
        !no_train.contains("train("),
        "the rewrite left a train block behind — this arm would test nothing"
    );
    let r = build_src(&no_train, "no_train", &["--source-ad", "--fuse-wgrad-accum"]);
    assert!(
        !r.ok,
        "--fuse-wgrad-accum on a program with no train block must refuse: \
         there is no FASE hook to fold into anywhere, so the flag is inert by \
         construction\nstderr:\n{}",
        r.stderr
    );
    assert!(
        r.stderr.contains(DECLINED),
        "the refusal must still print the instrument line — nothing else in \
         this compile had an opportunity to:\n{}",
        r.stderr
    );

    // ── @pipeline ───────────────────────────────────────────────────
    let pipelined = ccr_fixture_src().replace(
        "model Net:\n    blocks:",
        "model Net:\n    @pipeline(stages=2)\n    blocks:",
    );
    assert!(
        pipelined.contains("@pipeline(stages=2)"),
        "the @pipeline rewrite matched nothing — this arm would silently test \
         the ordinary train path"
    );
    let p = build_src(&pipelined, "pipelined", &["--source-ad", "--fuse-wgrad-accum"]);
    assert!(
        !p.ok,
        "--fuse-wgrad-accum on the pipelined train path must refuse, like the \
         seven sibling flags that path already refuses:\n{}",
        p.stderr
    );
    assert!(
        p.stderr.contains(DECLINED) && p.stderr.contains("@pipeline"),
        "the refusal must name @pipeline as the cause and print the decline \
         note:\n{}",
        p.stderr
    );
    // THE discriminator. Without the refusal this fixture reaches
    // `compile_train_block_pipelined_inner` and dies on a cranelift
    // `!self.is_sealed(block)` assertion — also a non-zero exit, but with no
    // wgrad diagnostic at all. Asserting `!ok` alone would pass either way.
    assert!(
        !p.stderr.contains("panicked"),
        "the pipelined refusal must arrive BEFORE the pipelined lowering \
         panics; a panic here means the admission was skipped and the \
         non-zero exit above proves nothing about this flag:\n{}",
        p.stderr
    );
}

/// The decline has to describe the program the user actually wrote.
///
/// `train(..., grad_accumulation=GA)` with `const GA = 4` parses, type-checks,
/// and then lowers a window of **1** — the config reader honours an
/// `IntLiteral` only and clamps everything else silently. That clamp is
/// specified (`spec/05-training-loop.nsl.md` documents `distill` refusing a
/// non-literal as the DIFFERENCE from `train`) and is not changed here. What
/// is changed is the message: it used to assert "the default when the train
/// block omits it" unconditionally, which is false for this block and sends
/// the reader hunting for a `grad_accumulation=` that is right there.
#[test]
fn the_decline_does_not_blame_an_omission_for_a_declared_non_literal_window() {
    let src = ccr_fixture_src()
        .replace("let m = Net()", "const GA = 4\nlet m = Net()")
        .replace("grad_accumulation=4", "grad_accumulation=GA");
    assert!(
        src.contains("grad_accumulation=GA") && src.contains("const GA = 4"),
        "the fixture rewrite matched nothing — this arm would test the literal \
         path instead"
    );
    let r = build_src(&src, "ga_non_literal", &["--source-ad", "--fuse-wgrad-accum"]);
    assert!(
        !r.ok,
        "the window really is 1 here (the non-literal is clamped), so the \
         typed flag must still refuse:\n{}",
        r.stderr
    );
    assert!(
        r.stderr.contains("non-literal"),
        "the decline must name the real cause — the block DOES declare \
         grad_accumulation, it was clamped:\n{}",
        r.stderr
    );
    assert!(
        !r.stderr.contains("omits grad_accumulation"),
        "the decline claims the block omits grad_accumulation, but it declares \
         `grad_accumulation=GA`. That is the wrong-reason diagnostic this arm \
         exists to prevent:\n{}",
        r.stderr
    );
    // The optimizer remedy is dead in this branch for EVERY input: `fase::plan`
    // returns Passthrough at accumulation 1 before it inspects the optimizer at
    // all, and this fixture is AdamW already. Offering it here sent the reader
    // to change something that could not matter.
    assert!(
        !r.stderr.contains("Adam-family"),
        "the accumulation branch must not offer the optimizer remedy — \
         fase::plan short-circuits at accumulation 1 before reading the \
         optimizer, so switching optimizers cannot help:\n{}",
        r.stderr
    );
}

#[test]
fn cpu_fused_wgrad_is_bit_identical_to_the_unfused_chain() {
    let f = write_fixture("nsl_wgrad_accum_gate_cpu.nsl", false, true);

    let off = run(&f, false, &[]);
    assert!(off.ok, "flag-off run failed:\n{}", off.stderr);
    let on = run(&f, false, &["--fuse-wgrad-accum"]);
    assert!(on.ok, "flag-on run failed:\n{}", on.stderr);

    // NON-VACUITY: the fusion must actually have fired. Two trainable weights
    // ⇒ two chains in the adjoint tape that carries them.
    let fired = fused_chains(&on.stderr).unwrap_or(0);
    assert!(
        fired >= 2,
        "expected >= 2 fused chains (one per trainable weight), got {fired}. \
         Without this the parity below is vacuous — flag-on would simply be \
         flag-off.\nstderr:\n{}",
        on.stderr
    );
    assert!(
        fused_chains(&off.stderr).is_none(),
        "the fusion reported chains with the flag OFF — it must be opt-in"
    );

    // Mirror image of the GPU check: on CPU every call must take the
    // FALLBACK. This is what licenses the bit-identical assertion below —
    // and if a future change ever made the GEMM path CPU-reachable, that
    // assertion would start failing for a reason nobody could explain
    // without this line.
    let (gemm, fallback) = wgrad_runtime_counts(&on.stderr).unwrap_or_else(|| {
        panic!("no [wgrad-accum] counter line\nstderr:\n{}", on.stderr)
    });
    assert_eq!(gemm, 0, "the cuBLAS path fired in a non-cuda build");
    assert!(
        fallback >= 2,
        "expected the decomposed fallback to run at least once per weight, \
         got {fallback}"
    );

    let (on_w1, on_w2) = weights(&on.stdout);
    let (off_w1, off_w2) = weights(&off.stdout);
    assert_weights_trained(&on_w1, &on_w2);

    // On CPU the runtime rejects the GEMM preconditions and falls back to the
    // exact decomposed chain, so the arithmetic is unchanged and any drift is
    // a CODEGEN defect (mis-planned chain, mis-wired operand, dropped grad).
    assert_eq!(
        on_w1, off_w1,
        "w1 differs on CPU, where the fused path falls back to the exact \
         decomposed chain — this is a codegen defect, not a numerics one"
    );
    assert_eq!(on_w2, off_w2, "w2 differs on CPU (see above)");
}

#[test]
fn cpu_identity_reduce_chain_also_fuses_and_matches() {
    // The 2-D case: `x` is `[T, d]`, so `reduce_to_shape` is an IDENTITY and
    // returns its input with a refcount bump instead of a fresh tensor. That
    // makes the fallback's ownership the tricky part (`raw` reaches rc=3 and
    // is released by three separate frees), and it is a shape `plan` admits
    // just as readily as the batched one. Covered separately because the
    // batched fixture cannot reach it.
    let f = write_fixture("nsl_wgrad_accum_gate_cpu_2d.nsl", false, false);

    let off = run(&f, false, &[]);
    assert!(off.ok, "flag-off run failed:\n{}", off.stderr);
    let on = run(&f, false, &["--fuse-wgrad-accum"]);
    assert!(on.ok, "flag-on run failed:\n{}", on.stderr);

    let fired = fused_chains(&on.stderr).unwrap_or(0);
    assert!(
        fired >= 2,
        "expected >= 2 fused chains on the 2-D shape, got {fired}\nstderr:\n{}",
        on.stderr
    );

    let (on_w1, on_w2) = weights(&on.stdout);
    let (off_w1, off_w2) = weights(&off.stdout);
    assert_weights_trained(&on_w1, &on_w2);
    assert_eq!(on_w1, off_w1, "w1 differs on the identity-reduce path");
    assert_eq!(on_w2, off_w2, "w2 differs on the identity-reduce path");
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gpu_identity_reduce_chain_uses_the_fused_gemm() {
    // The identity reduce on GPU: proves the flattened contraction handles
    // N = T (no batch dim) through the real cuBLAS path, not just the
    // fallback. Same tolerance contract as the batched GPU arm.
    let f = write_fixture("nsl_wgrad_accum_gate_gpu_2d.nsl", true, false);

    let off = run(&f, true, &[]);
    assert!(off.ok, "flag-off GPU run failed:\n{}", off.stderr);
    let on = run(&f, true, &["--fuse-wgrad-accum"]);
    assert!(on.ok, "flag-on GPU run failed:\n{}", on.stderr);

    let (gemm, fallback) = wgrad_runtime_counts(&on.stderr)
        .unwrap_or_else(|| panic!("no counter line\nstderr:\n{}", on.stderr));
    assert!(
        gemm > 0 && fallback == 0,
        "identity-reduce shape did not reach the fused GEMM \
         (fused={gemm}, fallback={fallback})"
    );

    let (on_w1, on_w2) = weights(&on.stdout);
    let (off_w1, off_w2) = weights(&off.stdout);
    assert_weights_trained(&on_w1, &on_w2);
    const TOL: f64 = 2e-3;
    for (name, (a, b)) in [("w1", (&on_w1, &off_w1)), ("w2", (&on_w2, &off_w2))] {
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < TOL,
                "{name}[{i}] fused={x} unfused={y} (|Δ|={})",
                (x - y).abs()
            );
        }
    }
}

#[test]
fn fuse_wgrad_accum_requires_source_ad() {
    // ANTI-VACUITY for the refusals below: prove the CLI rejects things at
    // all in this position, and specifically that the flag cannot be used on
    // the tape-AD path (which has no FASE accumulate hook to fold into).
    let f = write_fixture("nsl_wgrad_accum_gate_nosad.nsl", false, true);
    let root = repo_root();
    let out = Command::new(env!("CARGO"))
        .args([
            "run",
            "-q",
            "-p",
            "nsl-cli",
            "--manifest-path",
        ])
        .arg(root.join("Cargo.toml"))
        .args(["--", "run", "--fuse-wgrad-accum"])
        .arg(&f)
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(
        !out.status.success(),
        "--fuse-wgrad-accum without --source-ad must be refused"
    );
}

/// The three compositions that would be silently wrong rather than merely
/// slower. Each is refused by clap at parse time, so the check is cheap and
/// needs no device.
#[test]
fn incompatible_compositions_are_refused() {
    let f = write_fixture("nsl_wgrad_accum_gate_conflict.nsl", false, true);
    let root = repo_root();
    // (flag, why it cannot compose)
    let cases: [(&[&str], &str); 3] = [
        (
            &["--grad-integrity"],
            "must read the raw gradient the fusion never materializes",
        ),
        (
            &["--optim-state-offload"],
            "m_partial is host-resident; the device GEMM cannot write it",
        ),
        (
            &["--layerwise-accum"],
            "CSLA lowers pre-sliced tapes, defeating the single-reader proof",
        ),
    ];
    for (extra, why) in cases {
        let mut cmd = Command::new(env!("CARGO"));
        cmd.args(["run", "-q", "-p", "nsl-cli", "--manifest-path"])
            .arg(root.join("Cargo.toml"))
            .args(["--", "run", "--source-ad", "--fuse-wgrad-accum"])
            .args(extra)
            .arg(&f)
            .current_dir(&root)
            .env("NSL_STDLIB_PATH", root.join("stdlib"));
        let out = cmd.output().expect("spawn nsl run");
        assert!(
            !out.status.success(),
            "--fuse-wgrad-accum {extra:?} must be refused: {why}"
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU (2 FASE training runs)"]
fn gpu_fused_wgrad_matches_the_unfused_chain_within_f32_tolerance() {
    let f = write_fixture("nsl_wgrad_accum_gate_gpu.nsl", true, true);

    let off = run(&f, true, &[]);
    assert!(off.ok, "flag-off GPU run failed:\n{}", off.stderr);
    let on = run(&f, true, &["--fuse-wgrad-accum"]);
    assert!(on.ok, "flag-on GPU run failed:\n{}", on.stderr);

    let fired = fused_chains(&on.stderr).unwrap_or(0);
    assert!(
        fired >= 2,
        "expected >= 2 fused chains, got {fired}\nstderr:\n{}",
        on.stderr
    );
    // NON-VACUITY, the part that actually matters on GPU: the codegen line
    // above only proves the CALL was emitted. `nsl_tensor_wgrad_accum` falls
    // back to the exact decomposed chain — silently — whenever its
    // preconditions do not hold, and every such fallback would make this
    // tolerance comparison a test of the chain against itself.
    let (gemm, fallback) = wgrad_runtime_counts(&on.stderr).unwrap_or_else(|| {
        panic!(
            "no [wgrad-accum] counter line; cannot tell the fused GEMM from \
             the fallback\nstderr:\n{}",
            on.stderr
        )
    });
    assert!(
        gemm > 0,
        "every wgrad call took the DECOMPOSED FALLBACK (fused={gemm}, \
         fallback={fallback}) — this comparison would be vacuous. The GEMM \
         preconditions (contiguous, GPU-resident f32, matching leading dims) \
         are not being met on this shape."
    );
    assert_eq!(
        fallback, 0,
        "{fallback} of {} wgrad calls silently fell back; the arms are then \
         only partly comparable",
        gemm + fallback
    );

    let (on_w1, on_w2) = weights(&on.stdout);
    let (off_w1, off_w2) = weights(&off.stdout);
    assert_weights_trained(&on_w1, &on_w2);
    assert_weights_trained(&off_w1, &off_w2);

    // NOT bit-exact by construction: the fused GEMM sums the B*T products in
    // cuBLAS's order rather than rounding each per-batch partial before the
    // reduce. Measured against an f64 reference the fused form is the more
    // accurate of the two (nsl-runtime/tests/wgrad_accum_fused.rs), but the
    // contract here is only agreement to the repo's f32 training tolerance.
    const TOL: f64 = 2e-3;
    for (name, (a, b)) in [("w1", (&on_w1, &off_w1)), ("w2", (&on_w2, &off_w2))] {
        assert_eq!(a.len(), b.len(), "{name}: arm length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < TOL,
                "{name}[{i}] fused={x} unfused={y} (|Δ|={} >= {TOL})",
                (x - y).abs()
            );
        }
    }
}
