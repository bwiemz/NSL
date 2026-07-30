//! AdamW parameter groups — `no_decay=[...]`.
//!
//! `AdamW(weight_decay=λ)` used to apply λ to EVERY trainable parameter,
//! RMSNorm gains and the tied embedding included, because the train-block
//! optimizer DSL had no parameter groups. `no_decay=[...]` names parameter
//! ROLES to exempt, reusing the vocabulary `@param_role` and the mixed
//! Muon/AdamW router already share.
//!
//! The two directions that actually matter are exact equivalences, not
//! "the numbers moved":
//!
//! * **All-rank-1 model** — every parameter is exempt under
//!   `no_decay=["vector"]`, so the run must be BIT-IDENTICAL to the same
//!   configuration with `weight_decay=0.0`. This is what proves exemption
//!   truly removes the decay term rather than scaling it by something small.
//! * **All-rank-2 model** — nothing is exempt, so the run must be
//!   BIT-IDENTICAL to plain `weight_decay=λ` with no `no_decay` at all. This
//!   is what proves the feature does not over-exempt, and that the default
//!   path's numerics are untouched.
//!
//! Both are run through the real `nsl run` pipeline, because the property is
//! about emitted code, not about a Rust function.
//!
//! Anti-vacuity is explicit throughout: a `no_decay` that matches nothing is
//! a compile ERROR (not a silent no-op), an unknown role name is a compile
//! error, and the `[wd-groups]` table must name the parameters it exempted.
//! The whole failure mode this feature exists to remove is silence.

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

fn nsl_binary() -> PathBuf {
    let root = repo_root();
    for profile in ["release", "debug"] {
        let p = root.join("target").join(profile).join("nsl");
        if p.exists() {
            return p;
        }
    }
    panic!("no nsl binary in target/{{release,debug}} — build it first");
}

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
}

fn run_nsl(tag: &str, src: &str, source_ad: bool) -> Run {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_wd_groups_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join(format!("{tag}.nsl"));
    std::fs::write(&path, src).expect("write fixture");

    let mut cmd = Command::new(nsl_binary());
    cmd.arg("run");
    if source_ad {
        cmd.arg("--source-ad");
    }
    let out = cmd
        .arg(&path)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("run nsl");
    let _ = std::fs::remove_dir_all(&dir);
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        ok: out.status.success(),
    }
}

/// Run with arbitrary driver flags — used by the composition refusals, whose
/// flags each have their OWN prerequisites (`--layerwise-accum` needs
/// `--source-ad --checkpoint-blocks` plus `grad_accumulation >= 2`, and
/// `--muon-batch-ns` needs the Muon optimizer). Those stricter checks fire
/// FIRST, so a refusal test that passes only the flag under test would pass
/// on the wrong error entirely.
fn run_nsl_flags(tag: &str, src: &str, flags: &[&str]) -> Run {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_wd_groups_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join(format!("{tag}.nsl"));
    std::fs::write(&path, src).expect("write fixture");
    let out = Command::new(nsl_binary())
        .arg("run")
        .args(flags)
        .arg(&path)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("run nsl");
    let _ = std::fs::remove_dir_all(&dir);
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        ok: out.status.success(),
    }
}

/// Collect the bare floats a fixture prints — its loss stream plus the final
/// parameter checksums. Bit-exactness is compared on the full-precision
/// decimal text, so a 1-ulp divergence shows up as a different string.
fn floats(run: &Run) -> Vec<String> {
    run.stdout
        .lines()
        .map(str::trim)
        .filter(|l| l.parse::<f64>().is_ok())
        .map(str::to_string)
        .collect()
}

/// A model whose every trainable parameter is RANK 1 — so
/// `no_decay=["vector"]` exempts all of them.
///
/// `{OPT}` is substituted with the optimizer call.
const ALL_RANK1: &str = r#"
# `ones([dim])` — dim is a CONSTRUCTOR PARAMETER, not a literal, exactly as
# stdlib RMSNorm declares its gain. So no static rank is derivable and these
# params classify as role `hidden`, which is what makes the runtime rank check
# load-bearing. Writing `ones([4])` here would classify them `vector`
# statically and quietly test a case real models never hit.
model AllVec(dim: int):
    a: Tensor = ones([dim])
    b: Tensor = ones([dim])

    fn forward(self, x: Tensor) -> Tensor:
        return (x * self.a) * self.b

let m = AllVec(4)
let x = randn([4, 4]) * 0.1

train(model=m, epochs=6):
    optimizer: {OPT}
    step(batch):
        let out = m.forward(x)
        let loss = sum(out * out)
    callbacks:
        on_step(step, loss):
            print(loss)

print(m.a.sum().item())
print(m.b.sum().item())
"#;

/// A model whose every trainable parameter is RANK 2 — so
/// `no_decay=["vector"]` exempts nothing.
const ALL_RANK2: &str = r#"
model AllMat:
    w1: Tensor = randn([4, 4]) * 0.1
    w2: Tensor = randn([4, 4]) * 0.1

    fn forward(self, x: Tensor) -> Tensor:
        return (x @ self.w1) @ self.w2

let m = AllMat()
let x = randn([4, 4]) * 0.1

train(model=m, epochs=6):
    optimizer: {OPT}
    step(batch):
        let out = m.forward(x)
        let loss = sum(out * out)
    callbacks:
        on_step(step, loss):
            print(loss)

print(m.w1.sum().item())
print(m.w2.sum().item())
"#;

/// `--layerwise-accum` requires a `blocks.N`-structured model for its
/// checkpoint plan to attach to, so the CSLA tests below cannot reuse the flat
/// fixtures. Mirrors `crates/nsl-cli/tests/fixtures/grad_integrity_fullbuffer.nsl`.
const BLOCKS_MODEL: &str = r#"
model Blk:
    w_up: Tensor = randn([4, 8]) * 0.1
    w_down: Tensor = randn([8, 4]) * 0.1

    fn forward(self, h: Tensor) -> Tensor:
        let pre = h @ self.w_up
        let act = gelu(pre)
        let out = act @ self.w_down
        return h + out

model Net:
    blocks: [Blk; 2] = Blk()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
let x = randn([4, 4]) * 0.1

train(model=m, epochs=2, grad_accumulation=2):
    optimizer: {OPT}
    step(batch):
        let out = m.forward(x)
        let loss = sum(out * out)
"#;

const WD: &str = "AdamW(lr=0.05, weight_decay=0.5, beta1=0.9, beta2=0.95, eps=1e-8)";
const WD_NO_VEC: &str = "AdamW(lr=0.05, weight_decay=0.5, beta1=0.9, beta2=0.95, eps=1e-8, no_decay=[\"vector\"])";
const WD_ZERO: &str = "AdamW(lr=0.05, weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)";

/// Direction 1: everything exempt == weight_decay=0.0, BIT-EXACTLY.
#[test]
fn all_rank1_exempt_is_bit_identical_to_zero_weight_decay() {
    let exempt = run_nsl("r1_exempt", &ALL_RANK1.replace("{OPT}", WD_NO_VEC), false);
    let zero = run_nsl("r1_zero", &ALL_RANK1.replace("{OPT}", WD_ZERO), false);
    let decayed = run_nsl("r1_decayed", &ALL_RANK1.replace("{OPT}", WD), false);

    assert!(exempt.ok, "exempt run failed:\n{}\n{}", exempt.stdout, exempt.stderr);
    assert!(zero.ok, "zero-wd run failed:\n{}\n{}", zero.stdout, zero.stderr);
    assert!(decayed.ok, "decayed run failed:\n{}\n{}", decayed.stdout, decayed.stderr);

    let (fe, fz, fd) = (floats(&exempt), floats(&zero), floats(&decayed));
    assert!(!fe.is_empty(), "fixture printed no numbers:\n{}", exempt.stdout);
    assert_eq!(
        fe, fz,
        "no_decay=[\"vector\"] on an all-rank-1 model must be bit-identical to \
         weight_decay=0.0.\nexempt: {fe:?}\nzero:   {fz:?}"
    );

    // Anti-vacuity: the comparison above is only meaningful if λ actually
    // does something to this fixture. If decay-all matched too, the fixture
    // is insensitive to weight decay and proves nothing.
    assert_ne!(
        fd, fz,
        "weight_decay={} changed nothing on this fixture, so the equivalence \
         above is vacuous",
        0.5
    );
}

/// Direction 2: nothing exempt == no `no_decay` at all, BIT-EXACTLY. This is
/// the guard against over-exempting, and it pins that the default numerics
/// are untouched by the feature.
#[test]
fn all_rank2_is_bit_identical_with_and_without_no_decay() {
    let with = run_nsl("r2_nodecay", &ALL_RANK2.replace("{OPT}", WD_NO_VEC), false);
    let without = run_nsl("r2_plain", &ALL_RANK2.replace("{OPT}", WD), false);
    let zero = run_nsl("r2_zero", &ALL_RANK2.replace("{OPT}", WD_ZERO), false);

    assert!(with.ok, "no_decay run failed:\n{}\n{}", with.stdout, with.stderr);
    assert!(without.ok, "plain run failed:\n{}\n{}", without.stdout, without.stderr);

    let (fw, fp, fz) = (floats(&with), floats(&without), floats(&zero));
    assert!(!fw.is_empty(), "fixture printed no numbers:\n{}", with.stdout);
    assert_eq!(
        fw, fp,
        "no_decay=[\"vector\"] must exempt NOTHING on an all-rank-2 model, so \
         the run must be bit-identical to plain weight_decay.\n\
         with no_decay: {fw:?}\nwithout:       {fp:?}"
    );
    assert_ne!(
        fp, fz,
        "weight_decay had no effect on the all-rank-2 fixture — the \
         equivalence above is vacuous"
    );
}

/// The runtime rank check is load-bearing. An RMSNorm-style gain declared
/// `ones([dim])` over a CONSTRUCTOR PARAMETER yields no derivable rank, so it
/// classifies as role `hidden` — not `vector`. If exemption were decided from
/// the static role alone it would silently miss every norm in every real
/// model. Verified against models/coder50m: 74 params, 73 `hidden`, zero
/// `vector`, both RMSNorm gains among the `hidden`.
///
/// (A gain written `ones([4])` DOES get a static rank and classifies
/// `vector`; that is the case real models do not have, which is why this
/// fixture takes `dim` as a parameter.)
#[test]
fn rank1_gain_is_exempt_even_though_its_role_is_hidden() {
    let src = ALL_RANK1.replace("{OPT}", WD_NO_VEC);
    let run = run_nsl("role_is_hidden", &src, false);
    assert!(run.ok, "run failed:\n{}\n{}", run.stdout, run.stderr);
    let combined = format!("{}{}", run.stdout, run.stderr);

    // Zero params are exempt by ROLE, and the table has to say so — the
    // exemption comes entirely from the runtime rank.
    assert!(
        combined.contains("[wd-groups]"),
        "no [wd-groups] table was printed; a parameter group that reports \
         nothing is the silence this feature removes:\n{combined}"
    );
    assert!(
        combined.contains("0 exempt by role at compile time"),
        "expected zero COMPILE-TIME role exemptions on this fixture (both \
         params are role `hidden`), so the runtime rank check is what fires:\n{combined}"
    );
    assert!(
        combined.contains("not rank-2 at step time"),
        "the table must say the runtime rank check is active:\n{combined}"
    );
}

/// An unknown role name is a compile error, never a dropped entry. A typo'd
/// `"vectors"` that parsed as "exclusion configured" would decay every norm.
#[test]
fn unknown_role_name_refuses() {
    let opt = "AdamW(lr=0.05, weight_decay=0.5, no_decay=[\"vectors\"])";
    let run = run_nsl("bad_role", &ALL_RANK1.replace("{OPT}", opt), false);
    assert!(!run.ok, "a typo'd role name compiled:\n{}", run.stdout);
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("unknown parameter role"),
        "expected an unknown-role refusal naming the valid set:\n{combined}"
    );
}

/// A non-list `no_decay` refuses rather than being ignored.
#[test]
fn non_list_no_decay_refuses() {
    let opt = "AdamW(lr=0.05, weight_decay=0.5, no_decay=\"vector\")";
    let run = run_nsl("bad_shape", &ALL_RANK1.replace("{OPT}", opt), false);
    assert!(!run.ok, "a string no_decay compiled:\n{}", run.stdout);
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("no_decay must be a list"),
        "expected a shape refusal:\n{combined}"
    );
}

/// A scope that matches no parameter is a configuration error. Silently
/// exempting nothing is exactly the outcome this feature exists to prevent.
#[test]
fn scope_matching_nothing_refuses() {
    // `head` never fires on a weight-tied / head-less model, and this fixture
    // has no embedding_lookup at all.
    let opt = "AdamW(lr=0.05, weight_decay=0.5, no_decay=[\"head\"])";
    let run = run_nsl("vacuous_scope", &ALL_RANK1.replace("{OPT}", opt), false);
    assert!(!run.ok, "a scope matching nothing compiled:\n{}", run.stdout);
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("matches no parameter"),
        "expected a vacuous-scope refusal that suggests \"vector\":\n{combined}"
    );
    assert!(
        combined.contains("no_decay=[\"vector\"]"),
        "the refusal should point at the fix (runtime-rank exemption):\n{combined}"
    );
}

/// Exempting every role a parameter can have is `weight_decay=0.0` spelled
/// obscurely; say so instead of silently disabling decay.
#[test]
fn exempting_every_role_refuses() {
    let opt = "AdamW(lr=0.05, weight_decay=0.5, \
               no_decay=[\"embedding\", \"head\", \"hidden\"])";
    let run = run_nsl("all_roles", &ALL_RANK1.replace("{OPT}", opt), false);
    assert!(!run.ok, "exempting every role compiled:\n{}", run.stdout);
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("disables weight decay entirely"),
        "expected an all-roles refusal pointing at weight_decay=0.0:\n{combined}"
    );
}

/// The embedding role comes from `embedding_lookup` usage, and exempting it
/// must reach the tied LM head too (same tensor).
#[test]
fn embedding_role_exemption_fires_and_is_reported() {
    let src = r#"
from nsl.nn.losses import cross_entropy

model Tied:
    embed: Tensor = randn([32, 8]) * full([1], 0.02)
    w: Tensor = randn([8, 8]) * full([1], 0.02)

    fn forward_train(self, input_ids: Tensor, training: bool) -> Tensor:
        let s = input_ids.shape
        let b = s[0]
        let n = s[1]
        let flat = input_ids.reshape([b * n])
        let emb = embedding_lookup(self.embed, flat)
        let x = emb.reshape([b, n, 8])
        let h = x @ self.w
        return h @ self.embed.transpose(0, 1)

let m = Tied()
let tokens = full([256], 1.0)
let loader = DataLoader(tokens, batch_size=1, seq_len=8, shuffle=false, drop_last=true)

train(model=m, epochs=1):
    optimizer: AdamW(lr=0.05, weight_decay=0.5, beta1=0.9, beta2=0.95, eps=1e-8, no_decay=["embedding"])
    step(batch):
        let logits = m.forward_train(batch.input_ids, true)
        let ls = logits.shape
        let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])
        let flat_labels = batch.labels.reshape([ls[0] * ls[1]])
        let loss = cross_entropy(flat_logits, flat_labels)
"#;
    let run = run_nsl("embed_role", src, true);
    assert!(run.ok, "run failed:\n{}\n{}", run.stdout, run.stderr);
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("1 exempt by role at compile time"),
        "the embedding should be exempt by ROLE at compile time:\n{combined}"
    );
    assert!(
        combined.contains("m.embed role=embedding"),
        "the table must name the exempted embedding:\n{combined}"
    );
}

// ─── Composition refusals ──────────────────────────────────────────────────
//
// Each of these arms hoists ONE weight_decay scalar out of the per-parameter
// loop, so `no_decay` cannot be expressed there. They must refuse, because
// the alternative is training with decay applied to the parameters the user
// asked to exempt — silently.

/// `--muon-batch-ns` batches the Newton-Schulz pre-loop over many params with
/// one λ. Needs the Muon optimizer to get past the flag's own check.
#[test]
fn muon_batch_ns_composition_refuses() {
    let src = ALL_RANK1.replace(
        "{OPT}",
        "Muon(lr=0.02, weight_decay=0.5, beta1=0.9, beta2=0.95, eps=1e-8, \
         no_decay=[\"vector\"])",
    );
    let run = run_nsl_flags("muon_batch", &src, &["--muon-batch-ns"]);
    assert!(!run.ok, "the composition compiled:\n{}", run.stdout);
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("not supported with --muon-batch-ns"),
        "expected the no_decay/--muon-batch-ns refusal, got:\n{combined}"
    );
}

/// `--layerwise-accum`'s window-buffered group update hoists λ out of the
/// loop. Its own prerequisites are --source-ad, --checkpoint-blocks and
/// grad_accumulation >= 2.
#[test]
fn layerwise_accum_composition_refuses() {
    let src = BLOCKS_MODEL.replace("{OPT}", WD_NO_VEC);
    let run = run_nsl_flags("csla", &src, &["--source-ad", "--checkpoint-blocks", "--layerwise-accum"]);
    assert!(!run.ok, "the composition compiled:\n{}", run.stdout);
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("not supported with --layerwise-accum"),
        "expected the no_decay/--layerwise-accum refusal, got:\n{combined}"
    );
}

/// `--optim-state-offload` stages m/v across the host boundary; the envelope
/// is untested against per-parameter decay.
#[test]
fn optim_state_offload_composition_refuses() {
    let src = ALL_RANK1.replace("{OPT}", WD_NO_VEC);
    let run = run_nsl_flags("offload", &src, &["--optim-state-offload"]);
    assert!(!run.ok, "the composition compiled:\n{}", run.stdout);
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.contains("not supported with --optim-state-offload"),
        "expected the no_decay/--optim-state-offload refusal, got:\n{combined}"
    );
}

/// Anti-vacuity for the three refusals above: each flag combination must
/// COMPILE without `no_decay`. Otherwise the tests would be passing on the
/// flag's own unrelated prerequisite error rather than on the refusal.
#[test]
fn refused_compositions_are_otherwise_valid() {
    let muon_src = ALL_RANK1.replace(
        "{OPT}",
        "Muon(lr=0.02, weight_decay=0.5, beta1=0.9, beta2=0.95, eps=1e-8)",
    );
    // --muon-batch-ns also requires a CUDA build, which this gate does not,
    // so "it compiles" is not assertable here. The property that matters is
    // narrower and build-independent: without no_decay it must not fail with
    // the no_decay refusal.
    let r = run_nsl_flags("muon_batch_ok", &muon_src, &["--muon-batch-ns"]);
    let combined = format!("{}{}", r.stdout, r.stderr);
    assert!(
        !combined.contains("not supported with --muon-batch-ns"),
        "the no_decay refusal fired WITHOUT no_decay being set, so \
         muon_batch_ns_composition_refuses proves nothing:\n{combined}"
    );

    let r = run_nsl_flags("csla_ok", &BLOCKS_MODEL.replace("{OPT}", WD), &["--source-ad", "--checkpoint-blocks", "--layerwise-accum"]);
    assert!(
        r.ok,
        "--layerwise-accum does not compile even WITHOUT no_decay, so \
         layerwise_accum_composition_refuses proves nothing:\n{}\n{}",
        r.stdout, r.stderr
    );

    let r = run_nsl_flags("offload_ok", &ALL_RANK1.replace("{OPT}", WD), &["--optim-state-offload"]);
    assert!(
        r.ok,
        "--optim-state-offload does not compile even WITHOUT no_decay, so \
         optim_state_offload_composition_refuses proves nothing:\n{}\n{}",
        r.stdout, r.stderr
    );
}

// ─── GPU: the fused flat-grid launch partitions on λ ───────────────────────

/// The fused AdamW kernel takes `neg_lr_wd`/`has_wd` as LAUNCH parameters, so
/// two λ values cannot share a grid. λ is therefore part of `UpdateKey`
/// alongside device and dtype, and this is the end-to-end witness: the SAME
/// model must batch into ONE bucket without `no_decay` and TWO with it.
///
/// The CPU fixtures above cannot cover this — the fused multi-param path is
/// GPU-only (`uniform_gpu_f32` eligibility), so without this test the
/// `UpdateKey` change would rest on unit tests of `bucket_updates` alone.
///
/// Measured on an RTX 5070 Ti: 1 bucket vs 2 buckets, with the trained output
/// differing (-0.41462621092796326 vs -0.4853600263595581).
#[test]
#[ignore = "requires a CUDA GPU"]
fn fused_multi_launch_partitions_on_weight_decay_gpu() {
    const SRC: &str = r#"
model Blk:
    gain: Tensor = ones([8])
    w_up: Tensor = randn([8, 16]) * 0.1
    w_down: Tensor = randn([16, 8]) * 0.1

    fn forward(self, h: Tensor) -> Tensor:
        let f = (rmsnorm(h, self.gain, 0.00001) @ self.w_up) @ self.w_down
        return h + f

model Net:
    blocks: [Blk; 2] = Blk()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
m.to(cuda)
let x = (randn([8, 8]) * 0.1).to(cuda)

train(model=m, epochs=4, grad_accumulation=2):
    optimizer: AdamW(lr=0.05, weight_decay=0.5, beta1=0.9, beta2=0.95, eps=1e-8{EXTRA})
    step(batch):
        let out = m.forward(x)
        let loss = sum(out * out)

print(m.forward(x).sum().item())
"#;

    let all = run_nsl_flags(
        "gpu_wd_all",
        &SRC.replace("{EXTRA}", ""),
        &["--source-ad"],
    );
    let exempt = run_nsl_flags(
        "gpu_wd_exempt",
        &SRC.replace("{EXTRA}", ", no_decay=[\"vector\"]"),
        &["--source-ad"],
    );
    assert!(all.ok, "decay-all run failed:\n{}\n{}", all.stdout, all.stderr);
    assert!(exempt.ok, "exempt run failed:\n{}\n{}", exempt.stdout, exempt.stderr);

    let marker_of = |r: &Run| -> String {
        format!("{}{}", r.stdout, r.stderr)
            .lines()
            .find(|l| l.contains("[fase-multi]"))
            .unwrap_or("<no [fase-multi] marker>")
            .to_string()
    };
    let (m_all, m_ex) = (marker_of(&all), marker_of(&exempt));

    // Anti-vacuity: the fused multi path must actually have run. If it
    // declined, both arms would trivially agree and prove nothing about
    // bucketing.
    assert!(
        m_all.contains("[fase-multi]") && m_ex.contains("[fase-multi]"),
        "the fused multi-param launch did not fire, so this gate says nothing \
         about λ bucketing:\nall:    {m_all}\nexempt: {m_ex}"
    );
    assert!(
        m_all.contains("(1 bucket(s))"),
        "without no_decay every param shares one λ and must batch into ONE \
         bucket: {m_all}"
    );
    assert!(
        m_ex.contains("(2 bucket(s))"),
        "with no_decay the rank-1 gains take λ=0 and the matrices λ=0.5, so \
         the launch must split into TWO buckets: {m_ex}"
    );

    // And the exemption must change training, not just the launch shape.
    assert_ne!(
        floats(&all),
        floats(&exempt),
        "exempting the norm gains changed nothing about the trained output"
    );
}
