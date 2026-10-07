//! GPU checkpoint-resume equivalence (roadmap item 4 audit, hole #4).
//!
//! `train_checkpoint_gate.rs` proves a resumed run byte-identical to an
//! uninterrupted one, but on the CPU with one-parameter fixtures, so the
//! branch of `nsl_train_checkpoint_load` that restores AdamW moments into
//! DEVICE tensors (`memcpy_htod`) never ran in a gate. The 1B runs resume
//! from checkpoints on the GPU, under AdamW and gradient accumulation.
//!
//! This trains `fixtures/posture_lm.nsl` (NSL-Coder-1B's module structure at
//! toy scale: GQA with RoPE, SwiGLU, RMSNorm, a tied head) on the GPU under
//! the production postures, with the 1B recipe's AdamW (betas 0.9/0.95,
//! weight decay 0.1, eps 1e-8; lr 0.01) and four micro-batches of distinct
//! data per optimizer step, from a DataLoader with `epochs = 1` (the 1B
//! recipe's shape). Three processes per posture:
//!
//! * CONTROL: S = 5 optimizer steps, no checkpoint arguments.
//! * SAVE: the same run with `checkpoint_save` every k = 3 steps. 2k > S, so
//!   step k is its only save. It trains on past k, like a run that crashes
//!   some time after its last checkpoint (models/benchmarks/endurance_1b.py
//!   resumes the same way). It is also a second uninterrupted run, so its gap
//!   to the control is a fresh noise sample.
//! * RESUME: a new process, `checkpoint_load` of the step-k pair, trained
//!   out to S. It must PROVE it resumed: the `[checkpoint] resumed:` witness
//!   at micro-batch 12 / loader slot 12, and step labels 13..20. A run that
//!   silently starts over prints labels 1..20.
//!
//! The resumed losses (micro-batches 13..20) and final parameters are
//! compared with the control's. The first post-resume window's losses see
//! only the restored θ and data position; the two later updates see the
//! moments and the step counter (AdamW's bias correction), so S - k = 2 is
//! the least that covers the optimizer state.
//!
//! Two tiers, from the measured noise floor (RTX PRO 4500, 2026-10-07; the
//! numbers and logs are in .claude/campaign-evidence/gpu-resume/):
//!
//! * BIT-EXACT, under `--deterministic` (the no-atomics embedding backward,
//!   the CPU flash backward): two uninterrupted runs were bit-identical in
//!   every pair measured (15 to 35 control pairs per posture, and the save
//!   run of every gate run), so the resumed run must match every loss byte
//!   and every parameter bit. Covers the canonical posture, the chain
//!   posture, and the canonical 1B recipe exactly (`--param-dtype bf16-sr`,
//!   whose stochastic rounding is keyed by the seed and the step).
//! * PRODUCTION KERNELS, no `--deterministic`: two uninterrupted runs differ.
//!   The gaps fall into a few values that repeat exactly, as from alternative
//!   orderings of an atomic reduction. Over more than 40 pairs per posture
//!   (the diagnostic below twice, plus the save run of every gate run) the
//!   worst was a loss gap of 1.04e-4 and a parameter gap of 2.9e-4 (L2 of
//!   the final-parameter difference over the L2 of the whole run's update;
//!   the per-tensor max reached 2.4e-3, which is why it is reported and not
//!   bounded). The bounds sit ten times above that.
//!
//! Mutation-checked on the GPU (2026-10-07). Each mutant fails all five
//! gates; gaps are the resumed run's, loss / parameter:
//!
//! * the device moments not restored (the `memcpy_htod` skipped, so m and v
//!   stay zero): 3.0e-2..4.0e-2 / 0.355, against bounds of 1e-3 / 3e-3;
//! * the step counter one optimizer step late (only AdamW's bias correction
//!   and the step labels change): 2.9e-3..4.2e-3 / 2.2e-2..3.0e-2, and the
//!   labels read 17..24;
//! * the step counter one micro-batch late: the layerwise postures abort
//!   (the window backward refuses a partial window), the chain posture is off
//!   by 2.3e-2..2.4e-2 / 8.3e-2;
//! * the step counter re-zeroed: 2.3e-2..2.5e-2 / 0.21, labels 1..8;
//! * `checkpoint_load` not arming the resume: no witness, labels 1..20 and a
//!   stream of 20 micro-batches where 8 were due. Its final parameters match
//!   the control's (bit for bit under `--deterministic`), since a run that
//!   starts over IS the control, so only the witness and the stream catch it.
//!
//! Run: cargo test -p nsl-cli --features cuda --test train_checkpoint_resume_gpu \
//!        -- --ignored --test-threads=1

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Optimizer steps in the whole run (S).
const STEPS: usize = 5;
/// The checkpoint step (k). `checkpoint_every = k` with 2k > S makes step k
/// the save run's ONLY save, so the file on disk is the step-k state.
const RESUME_AT: usize = 3;
const ACCUM: usize = 4;
const BATCH: usize = 2;
const SEQ: usize = 64;
const SEED: &str = "7";
/// The 1B recipe's AdamW (models/coder1b/pretrain_1b2048.nsl) at a toy-scale
/// learning rate.
const OPTIMIZER: &str = "AdamW(lr=0.01, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8)";
/// The micro-batch counter at the checkpoint.
const RESUME_MICRO: usize = RESUME_AT * ACCUM;
const TOTAL_MICRO: usize = STEPS * ACCUM;

/// Production-kernel tier bounds (see the module doc): ten times the worst
/// control-vs-control gap measured.
const PROD_LOSS_TOL: f64 = 1e-3;
const PROD_PARAM_TOL: f64 = 3e-3;

#[derive(Clone, Copy)]
struct Posture {
    name: &'static str,
    flags: &'static [&'static str],
    clip: Option<f64>,
}

/// The canonical 1B recipe (pretrain_1b2048.nsl) minus its precision mode.
const CANONICAL: Posture = Posture {
    name: "canonical",
    flags: &[
        "--source-ad",
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
        "--fuse-lm-head",
        "require",
        "--fuse-rmsnorm-backward",
    ],
    clip: None,
};

/// The canonical 1B recipe exactly, precision mode included.
const CANONICAL_BF16SR: Posture = Posture {
    name: "canonical_bf16sr",
    flags: &[
        "--source-ad",
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
        "--param-dtype",
        "bf16-sr",
        "--fuse-lm-head",
        "require",
        "--fuse-rmsnorm-backward",
    ],
    clip: None,
};

/// The posture the 1B chain switched to: selective recompute and fused wgrad
/// accumulation, clipped (`--layerwise-accum` refuses `grad_clip`). The
/// posture certificate's threshold; under this AdamW run it fires from the
/// first update on (the clipped trajectory departs from an unclipped one at
/// micro-batch 5, by 3.2e-3 in loss over the run).
const CHAIN: Posture = Posture {
    name: "chain",
    flags: &[
        "--source-ad",
        "--checkpoint-blocks",
        "--checkpoint-selective",
        "--fuse-lm-head",
        "require",
        "--fuse-rmsnorm-backward",
        "--fuse-wgrad-accum",
    ],
    clip: Some(1.69),
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// `sequences` rows of aperiodic tokens in [1, 127] (the posture
/// certificate's LCG), so no two micro-batches carry the same data.
fn write_tokens(path: &Path, sequences: usize) {
    let n = sequences * SEQ + 1;
    let mut state: u32 = 0x9E37_79B9;
    let bytes: Vec<u8> = (0..n)
        .flat_map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let tok = 1 + (state >> 8) % 127;
            (tok as u16).to_le_bytes()
        })
        .collect();
    std::fs::write(path, bytes).unwrap();
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    Control,
    Save,
    Resume,
}

struct Run {
    /// The `step` the train block's on_step callback saw, per micro-batch.
    steps: Vec<u64>,
    loss_text: Vec<String>,
    losses: Vec<f64>,
    init: HashMap<String, Vec<f32>>,
    end: HashMap<String, Vec<f32>>,
    stderr: String,
}

/// A scratch directory under TMPDIR, removed when dropped: on the failure
/// paths too, where a failed `nsl run` panics before any cleanup line could
/// run (and /tmp is a tmpfs on the reference machine).
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let d = std::env::temp_dir().join(format!("nsl_gpuresume_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// posture_lm.nsl with the posture certificate's marker rewrites, plus three
/// rewrites of its train block: AdamW for SGD, the checkpoint arguments
/// spliced in after the clip argument, and the step counter printed before
/// each loss.
fn program(dir: &Path, phase: Phase, clip: Option<f64>) -> String {
    let p = |f: &str| dir.join(f).display().to_string().replace('\\', "/");
    let ckpt = match phase {
        Phase::Control => String::new(),
        Phase::Save => format!(", checkpoint_save=\"{}\", checkpoint_every={RESUME_AT}", p("ck.nslm")),
        Phase::Resume => format!(", checkpoint_load=\"{}\"", p("ck.nslm")),
    };
    let train_args = format!("{}{ckpt}", clip.map_or_else(String::new, |c| format!(", grad_clip={c:?}")));
    let fixture = std::fs::read_to_string(repo_root().join("crates/nsl-cli/tests/fixtures/posture_lm.nsl")).unwrap();
    let markers = [
        ("POSTURE_TOKENS_PATH", p("tokens.bin")),
        ("POSTURE_INIT_PATH", p("init.nslm")),
        ("POSTURE_SAVE_PATH", p("end.nslm")),
        ("POSTURE_BATCH", BATCH.to_string()),
        ("POSTURE_ACCUM", ACCUM.to_string()),
        ("POSTURE_ROWS", (BATCH * SEQ).to_string()),
        ("POSTURE_CLIP_ARG", train_args),
    ];
    let mut seen = [0usize; 7];
    let (mut placed, mut optimizer, mut printed) = (0, 0, 0);
    // Code lines only: the header comment documents the markers by name.
    let src: Vec<String> = fixture
        .lines()
        .map(|line| {
            let t = line.trim();
            if t == "# GPU_PLACEMENT" {
                placed += 1;
                return "m.to(cuda)".to_string();
            }
            if t.starts_with('#') {
                return line.to_string();
            }
            if t == "optimizer: SGD(lr=0.5)" {
                optimizer += 1;
                return line.replace("SGD(lr=0.5)", OPTIMIZER);
            }
            if t == "print(loss)" {
                printed += 1;
                let indent = &line[..line.len() - line.trim_start().len()];
                return format!("{indent}print(step)\n{line}");
            }
            let mut l = line.to_string();
            for (i, (marker, value)) in markers.iter().enumerate() {
                seen[i] += l.matches(marker).count();
                l = l.replace(marker, value);
            }
            l
        })
        .collect();
    for (i, (marker, _)) in markers.iter().enumerate() {
        assert_eq!(seen[i], 1, "fixture marker {marker} must occur once in code");
    }
    assert_eq!(
        (placed, optimizer, printed),
        (1, 1, 1),
        "posture_lm.nsl no longer has exactly one `# GPU_PLACEMENT`, `optimizer: SGD(lr=0.5)` and `print(loss)` line"
    );
    src.join("\n")
}

fn run(dir: &Path, flags: &[&str], phase: Phase, clip: Option<f64>, tag: &str) -> Run {
    let root = repo_root();
    let tokens = dir.join("tokens.bin");
    if !tokens.exists() {
        write_tokens(&tokens, STEPS * ACCUM * BATCH);
    }
    let prog = dir.join(format!("{tag}.nsl"));
    std::fs::write(&prog, program(dir, phase, clip)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .args(flags)
        .args(["--seed", SEED])
        .arg(&prog)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned());
    assert!(out.status.success(), "{tag} failed:\nstdout:\n{stdout}\nstderr:\n{stderr}");
    let (steps, loss_text) = step_stream(&stdout, tag);
    let losses = loss_text
        .iter()
        .map(|l| {
            let v: f64 = l.trim_start_matches("tensor([").trim_end_matches("])").parse().unwrap_or_else(|e| panic!("{tag}: loss line {l:?}: {e}"));
            assert!(v.is_finite(), "{tag}: non-finite loss {l:?}");
            v
        })
        .collect();
    Run { steps, loss_text, losses, init: nslm::read(&dir.join("init.nslm")), end: nslm::read(&dir.join("end.nslm")), stderr }
}

/// The (step, loss) line pairs between the fixture's stream markers.
fn step_stream(stdout: &str, tag: &str) -> (Vec<u64>, Vec<String>) {
    let start = stdout.find("LOSS_STREAM_BEGIN").expect("LOSS_STREAM_BEGIN") + "LOSS_STREAM_BEGIN".len();
    let end = stdout.find("LOSS_STREAM_END").expect("LOSS_STREAM_END");
    let lines: Vec<&str> = stdout[start..end].lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    assert!(lines.len().is_multiple_of(2), "{tag}: the stream is (step, loss) pairs:\n{}", lines.join("\n"));
    let steps = lines.iter().step_by(2).map(|s| s.parse::<u64>().unwrap_or_else(|e| panic!("{tag}: step line {s:?}: {e}"))).collect();
    (steps, lines.iter().skip(1).step_by(2).map(|s| s.to_string()).collect())
}

/// Fields the model stores but does not train (posture_certificate_gpu.rs).
fn trainable(name: &str) -> bool {
    let leaf = name.rsplit('.').next().unwrap_or(name);
    !leaf.starts_with('_') && leaf != "inv_freq"
}

/// How far a run is from the uninterrupted control.
struct Gap {
    /// max |loss difference| over the micro-batches both printed.
    loss: f64,
    /// ||end - end_ctl|| / ||end_ctl - init|| over every trainable element:
    /// the final-parameter difference relative to what the whole run moved.
    param: f64,
    /// The largest per-tensor max|end - end_ctl| / max|end_ctl - init|
    /// (reported, not bounded: a single raced element dominates it).
    param_max: f64,
    /// Every loss line byte-identical and every final parameter bit-identical.
    bit_exact: bool,
}

/// `run` against `ctl`'s stream from micro-batch `from` on.
fn gap(ctl: &Run, run: &Run, from: usize) -> Gap {
    let want = &ctl.losses[from..];
    let same_len = want.len() == run.losses.len();
    let loss = if same_len { want.iter().zip(&run.losses).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max) } else { f64::INFINITY };
    let mut bit_exact = same_len && ctl.loss_text[from..] == run.loss_text[..];
    let (mut num, mut den, mut param_max) = (0.0f64, 0.0f64, 0.0f64);
    assert_eq!(ctl.end.len(), run.end.len(), "the runs saved different parameter sets");
    for (k, v0) in &ctl.init {
        let (ec, er) = (&ctl.end[k], &run.end[k]);
        bit_exact &= ec.len() == er.len() && ec.iter().zip(er).all(|(x, y)| x.to_bits() == y.to_bits());
        if !trainable(k) {
            continue;
        }
        let scale = v0.iter().zip(ec).fold(0.0f64, |m, (x, y)| m.max((*y - *x).abs() as f64));
        assert!(scale > 0.0, "{k}: the control never updated it");
        let d = ec.iter().zip(er).fold(0.0f64, |m, (x, y)| m.max((*y - *x).abs() as f64));
        param_max = param_max.max(d / scale);
        num += ec.iter().zip(er).map(|(x, y)| ((*y - *x) as f64).powi(2)).sum::<f64>();
        den += v0.iter().zip(ec).map(|(x, y)| ((*y - *x) as f64).powi(2)).sum::<f64>();
    }
    Gap { loss, param: (num / den).sqrt(), param_max, bit_exact }
}

impl std::fmt::Display for Gap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "loss {:.3e} param {:.3e} (max {:.3e}) bit-exact {}", self.loss, self.param, self.param_max, self.bit_exact)
    }
}

/// What "a continuation" means for one tier.
#[derive(Clone, Copy)]
enum Bound {
    /// Every loss byte and every parameter bit.
    BitExact,
    /// The production-kernel bounds.
    Within { loss: f64, param: f64 },
}

/// Control, save and resume runs of one posture, checked. Every check runs
/// and the failure lists all that failed, so a broken resume reports each
/// symptom at once.
fn certify(posture: Posture, extra: &[&str], bound: Bound, tag: &str) {
    let flags: Vec<&str> = posture.flags.iter().chain(extra).copied().collect();
    let ctl_dir = Scratch::new(&format!("{tag}_ctl"));
    let ctl = run(&ctl_dir, &flags, Phase::Control, posture.clip, &format!("{tag}_control"));
    let dir = Scratch::new(&format!("{tag}_ckpt"));
    let saved = run(&dir, &flags, Phase::Save, posture.clip, &format!("{tag}_save"));
    let resumed = run(&dir, &flags, Phase::Resume, posture.clip, &format!("{tag}_resume"));

    let mut failures = Vec::new();
    let mut check = |ok: bool, what: String| {
        if !ok {
            failures.push(what);
        }
    };
    // The arms ran the production line, not a fallback of it.
    for (name, r) in [("control", &ctl), ("save", &saved), ("resume", &resumed)] {
        check(r.stderr.contains("Using source-to-source AD for backward pass"), format!("{name}: source AD did not engage"));
        check(!r.stderr.contains("falling back to tape-based AD"), format!("{name}: fell back to the tape"));
        check(r.stderr.contains("[lm-head-fusion] inferred"), format!("{name}: the fused LM head did not engage"));
    }
    // The save landed at step k and nowhere else; the new process RESUMED
    // from it and continued the step counter and the data stream.
    let at = format!("at micro-batch step {RESUME_MICRO} ");
    let slot = format!("loader slot {RESUME_MICRO})");
    let saves: Vec<&str> = saved.stderr.lines().filter(|l| l.contains("[checkpoint] saved:")).collect();
    check(
        saves.len() == 1 && saves[0].contains(&at) && saves[0].contains(&slot),
        format!("save: want exactly one save, at micro-batch {RESUME_MICRO} / loader slot {RESUME_MICRO}: {saves:?}"),
    );
    let resumes: Vec<&str> = resumed.stderr.lines().filter(|l| l.contains("[checkpoint] resumed:")).collect();
    check(
        resumes.len() == 1 && resumes[0].contains(&at) && resumes[0].contains(&slot),
        format!("resume: no `[checkpoint] resumed:` witness at micro-batch {RESUME_MICRO} / loader slot {RESUME_MICRO}: {resumes:?}"),
    );
    let all: Vec<u64> = (1..=TOTAL_MICRO as u64).collect();
    check(ctl.steps == all, format!("control: step labels {:?}", ctl.steps));
    check(saved.steps == all, format!("save: step labels {:?}", saved.steps));
    check(
        resumed.steps == all[RESUME_MICRO..],
        format!("resume: step labels {:?}, want {:?}", resumed.steps, &all[RESUME_MICRO..]),
    );
    for (k, v) in &ctl.init {
        check(saved.init.get(k) == Some(v) && resumed.init.get(k) == Some(v), format!("{k}: the runs did not initialize alike"));
    }

    let noise = gap(&ctl, &saved, 0);
    let res = gap(&ctl, &resumed, RESUME_MICRO);
    for (name, g) in [("save run (uninterrupted)", &noise), ("resumed run", &res)] {
        match bound {
            Bound::BitExact => check(g.bit_exact, format!("{name} is not bit-identical to the control: {g}")),
            Bound::Within { loss, param } => {
                check(g.loss <= loss, format!("{name}: loss gap {:.3e} > {loss:e}", g.loss));
                check(g.param <= param, format!("{name}: parameter gap {:.3e} > {param:e}", g.param));
            }
        }
    }
    let report = format!("[{tag}] save vs control: {noise} | resumed vs control: {res}");
    eprintln!("{report}");
    assert!(failures.is_empty(), "GPU resume is not a continuation:\n  {}\n{report}", failures.join("\n  "));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_resume_bit_exact_canonical() {
    certify(CANONICAL, &["--deterministic"], Bound::BitExact, "bitexact_canonical");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_resume_bit_exact_canonical_bf16sr() {
    certify(CANONICAL_BF16SR, &["--deterministic"], Bound::BitExact, "bitexact_canonical_bf16sr");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_resume_bit_exact_chain() {
    certify(CHAIN, &["--deterministic"], Bound::BitExact, "bitexact_chain");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_resume_canonical() {
    certify(CANONICAL, &[], Bound::Within { loss: PROD_LOSS_TOL, param: PROD_PARAM_TOL }, "canonical");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_resume_chain() {
    certify(CHAIN, &[], Bound::Within { loss: PROD_LOSS_TOL, param: PROD_PARAM_TOL }, "chain");
}

/// Calibration, not a gate: control-vs-control gaps for every tier above,
/// fifteen pairs each — the measurement the tiers' bounds come from.
#[test]
#[ignore = "diagnostic: GPU resume noise floor (control vs control)"]
fn gpu_resume_noise_floor() {
    const PAIRS: usize = 15;
    let tiers: [(Posture, bool); 5] = [(CANONICAL, true), (CANONICAL_BF16SR, true), (CHAIN, true), (CANONICAL, false), (CHAIN, false)];
    for (posture, deterministic) in tiers {
        let mut flags = posture.flags.to_vec();
        if deterministic {
            flags.push("--deterministic");
        }
        let tag = format!("{}_{}", posture.name, if deterministic { "det" } else { "prod" });
        let base_dir = Scratch::new(&format!("{tag}_n0"));
        let base = run(&base_dir, &flags, Phase::Control, posture.clip, &format!("{tag}_n0"));
        for i in 1..=PAIRS {
            let d = Scratch::new(&format!("{tag}_n{i}"));
            let r = run(&d, &flags, Phase::Control, posture.clip, &format!("{tag}_n{i}"));
            eprintln!("NOISE {tag} #{i}: {}", gap(&base, &r, 0));
        }
    }
}

mod nslm {
    //! Minimal .nslm reader (the parser of stage_c_packed_parity.rs: test
    //! modules do not cross crate boundaries).
    use std::collections::HashMap;
    use std::path::Path;

    pub fn read(path: &Path) -> HashMap<String, Vec<f32>> {
        let buf = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        assert!(buf.len() >= 16 && &buf[0..4] == b"NSLM", "bad magic in {path:?}");
        let header_size = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        let header_end = 16 + header_size;
        let json = std::str::from_utf8(&buf[16..header_end]).expect("header utf8");
        let data_start = header_end + (64 - header_end % 64) % 64;
        let mut out = HashMap::new();
        let mut cursor = 0;
        while let Some(i) = json[cursor..].find("\"name\":\"") {
            let abs = cursor + i + 8;
            let name_end = json[abs..].find('"').expect("name");
            let name = json[abs..abs + name_end].to_string();
            cursor = abs + name_end;
            let dtype = field(json, cursor, "dtype", true);
            let offset: usize = field(json, cursor, "offset", false).parse().unwrap();
            let nbytes: usize = field(json, cursor, "nbytes", false).parse().unwrap();
            assert_eq!(dtype, "f32", "{name} in {path:?}");
            let slab = &buf[data_start + offset..data_start + offset + nbytes];
            out.insert(name, slab.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
        }
        out
    }

    fn field(json: &str, from: usize, key: &str, string: bool) -> String {
        let pat = if string { format!("\"{key}\":\"") } else { format!("\"{key}\":") };
        let abs = from + json[from..].find(&pat).expect(key) + pat.len();
        let end = json[abs..].find(|c: char| if string { c == '"' } else { c == ',' || c == '}' }).expect(key);
        json[abs..abs + end].trim().to_string()
    }
}
