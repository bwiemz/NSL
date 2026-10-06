//! The 1B-posture certificate (roadmap item 4, mutation audit).
//!
//! Every GPU training gate before this one compared two GPU arms with each
//! other, mostly under AdamW. None held a GPU run of the 1B flag line to an
//! independent reference, so the accumulation window, the clip norm,
//! selective recompute, the fused LM head and the fused backward kernels were
//! protected only by peer comparisons in which a shared or scale-only bug
//! cancels (campaign-evidence/item4-mutation-audit/REPORT.md).
//!
//! This trains `fixtures/posture_lm.nsl` -- NSL-Coder-1B's module structure
//! at toy scale -- from the same seed on the GPU under each production
//! posture (`CANONICAL_FLAGS`, `CHAIN_FLAGS`) and on the CPU under
//! `--training-reference` with the TAPE (host kernels and an AD independent
//! of source AD), with SGD (update = lr * grad), four micro-batches of
//! distinct data per step, and (chain) a clip threshold that fires on some
//! steps and not others. It checks:
//!
//! * EXACT-ATTENTION tier: with the f16-operand flash kernels swapped for
//!   exact ones, every other production transformation on the GPU matches
//!   the reference to ~1e-5 (bound 3e-4). A flag bisection showed the fused
//!   LM head, fused RMSNorm backward, fused wgrad accumulation, block
//!   checkpointing and selective recompute each add nothing measurable.
//! * FUSED-ATTENTION tier: the production kernels, within the f16-operand
//!   bound (measured ~2e-3, bound 5e-3).
//! * the ACCUMULATION IDENTITY (4 x 2 sequences == 2 x 4 sequences per step):
//!   the reference shares the FASE accumulation recipe with the GPU arm, so
//!   only an identity can see a wrong 1/N. A GLOBAL scale error is caught by
//!   the SGD-vs-f64 certificates in fused_loss_gradient_cert_gpu.rs.
//! * every trainable field updates in both arms (the #806 class), and no
//!   configuration field moves.
//!
//! Mutation-checked (2026-10-07): the GPU clip norm summing one partial per
//! tensor fails the exact chain tier (loss gap 3.8e-3 vs 1e-5); dropping the
//! 1/N accumulation scale fails the identity (1.0 vs 1e-4).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const STEPS: usize = 4;
const ACCUM: usize = 4;
const BATCH: usize = 2;
const SEQ: usize = 64;
const SEED: &str = "7";
const LR: f64 = 0.5;

/// The two production postures, each certified against the same CPU
/// reference. They cannot be one flag line: `--fuse-wgrad-accum` refuses
/// `--layerwise-accum`. `--matmul-mode f32` keeps the GEMMs at f32 in both,
/// so the comparison measures the training pipeline, not TF32/bf16 rounding.
///
/// The canonical 1B recipe (models/coder1b/pretrain_1b2048.nsl), minus its
/// precision mode (`--param-dtype bf16-sr`).
const CANONICAL_FLAGS: &[&str] = &[
    "--source-ad",
    "--checkpoint-blocks",
    "--layerwise-accum",
    "--weight-stream",
    "--fuse-lm-head",
    "require",
    "--fuse-rmsnorm-backward",
    "--matmul-mode",
    "f32",
];

/// The pre-chain posture the 1B chain switched to (selective recompute +
/// fused wgrad accumulation; precision mode again dropped).
const CHAIN_FLAGS: &[&str] = &[
    "--source-ad",
    "--checkpoint-blocks",
    "--checkpoint-selective",
    "--fuse-lm-head",
    "require",
    "--fuse-rmsnorm-backward",
    "--fuse-wgrad-accum",
    "--matmul-mode",
    "f32",
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// `sequences` rows of aperiodic tokens in [1, 127] (a 32-bit LCG), so no
/// two micro-batches carry the same data.
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

/// Extra environment for one run.
type Env = &'static [(&'static str, &'static str)];

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Gpu(&'static [&'static str]),
    /// The CPU with other flags (the noise-floor diagnostic).
    Cpu(&'static [&'static str]),
    CpuReference,
}

struct Run {
    losses: Vec<f64>,
    init: HashMap<String, Vec<f32>>,
    end: HashMap<String, Vec<f32>>,
    stderr: String,
}

fn run(arm: Arm, steps: usize, clip: Option<f64>, tag: &str) -> Run {
    run_env(arm, steps, clip, tag, &[])
}

fn run_env(arm: Arm, steps: usize, clip: Option<f64>, tag: &str, env: &[(&str, &str)]) -> Run {
    run_shaped(arm, steps, (BATCH, ACCUM), clip, tag, env)
}

/// `shape` = (micro-batch size, grad_accumulation).
fn run_shaped(
    arm: Arm,
    steps: usize,
    shape: (usize, usize),
    clip: Option<f64>,
    tag: &str,
    env: &[(&str, &str)],
) -> Run {
    let (batch, accum) = shape;
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_posture_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let tokens = tmp.join("tokens.bin");
    write_tokens(&tokens, steps * accum * batch);
    let p = |f: &str| tmp.join(f).display().to_string().replace('\\', "/");
    let fixture = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/posture_lm.nsl")).unwrap();
    let markers = [
        ("POSTURE_TOKENS_PATH", p("tokens.bin")),
        ("POSTURE_INIT_PATH", p("init.nslm")),
        ("POSTURE_SAVE_PATH", p("end.nslm")),
        ("POSTURE_BATCH", batch.to_string()),
        ("POSTURE_ACCUM", accum.to_string()),
        ("POSTURE_ROWS", (batch * SEQ).to_string()),
        ("POSTURE_CLIP_ARG", clip.map_or_else(String::new, |c| format!(", grad_clip={c:?}"))),
    ];
    let mut seen = [0usize; 7];
    // Code lines only: the header comment documents the markers by name.
    let src: String = fixture
        .lines()
        .map(|line| {
            if line.trim() == "# GPU_PLACEMENT" {
                return if matches!(arm, Arm::Gpu(_)) { "m.to(cuda)".to_string() } else { String::new() };
            }
            if line.trim_start().starts_with('#') {
                return line.to_string();
            }
            let mut l = line.to_string();
            for (i, (marker, value)) in markers.iter().enumerate() {
                seen[i] += l.matches(marker).count();
                l = l.replace(marker, value);
            }
            l
        })
        .collect::<Vec<_>>()
        .join("\n");
    for (i, (marker, _)) in markers.iter().enumerate() {
        assert_eq!(seen[i], 1, "fixture marker {marker} must occur once in code");
    }
    assert_eq!(fixture.lines().filter(|l| l.trim() == "# GPU_PLACEMENT").count(), 1);
    let prog = tmp.join("posture.nsl");
    std::fs::write(&prog, src).unwrap();
    let flags: &[&str] = match arm {
        Arm::Gpu(flags) | Arm::Cpu(flags) => flags,
        Arm::CpuReference => &["--training-reference"],
    };
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .args(flags)
        .args(["--seed", SEED])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .envs(env.iter().copied())
        .output()
        .expect("spawn nsl run");
    let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned());
    assert!(out.status.success(), "{tag} failed:\nstdout:\n{stdout}\nstderr:\n{stderr}");
    let losses = loss_stream(&stdout);
    assert_eq!(losses.len(), steps * accum, "{tag}: one loss per micro-batch:\n{stdout}");
    let run = Run { losses, init: nslm::read(&tmp.join("init.nslm")), end: nslm::read(&tmp.join("end.nslm")), stderr };
    let _ = std::fs::remove_dir_all(&tmp);
    run
}

fn loss_stream(stdout: &str) -> Vec<f64> {
    let start = stdout.find("LOSS_STREAM_BEGIN").expect("LOSS_STREAM_BEGIN") + "LOSS_STREAM_BEGIN".len();
    let end = stdout.find("LOSS_STREAM_END").expect("LOSS_STREAM_END");
    stdout[start..end]
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let v: f64 = l.trim_start_matches("tensor([").trim_end_matches("])").parse().unwrap_or_else(|e| panic!("loss line {l:?}: {e}"));
            assert!(v.is_finite(), "non-finite loss {l:?}");
            v
        })
        .collect()
}

/// Fields the model stores but does not train: the stdlib GQA / RoPE
/// configuration scalars and tables (`_`-prefixed by convention) and RoPE's
/// `inv_freq`.
fn trainable(name: &str) -> bool {
    let leaf = name.rsplit('.').next().unwrap_or(name);
    !leaf.starts_with('_') && leaf != "inv_freq"
}

/// ||end - init|| over every parameter.
fn update_norm(r: &Run) -> f64 {
    r.init
        .iter()
        .map(|(k, v0)| v0.iter().zip(&r.end[k]).map(|(a, b)| ((*b - *a) as f64).powi(2)).sum::<f64>())
        .sum::<f64>()
        .sqrt()
}

/// The clip threshold: between the reference's per-step gradient norms, so
/// clipping fires on some steps and not others (`posture_gradient_norms`
/// prints them).
const CLIP: f64 = 1.69;
/// Two tiers per posture.
///
/// EXACT ATTENTION: the fused flash-attention kernels are swapped for the
/// unfused forward and the CPU flash backward (`EXACT_ATTENTION`); every
/// other production transformation still runs on the GPU. Measured
/// (2026-10-07, RTX PRO 4500): ~1.5e-5 relative after one step against the
/// CPU tape reference, whose own floor against CPU source AD is ~1e-6..1e-5.
/// The bound sits a decade above the measurement -- a 1/accum scale, a clip
/// norm, a stale micro-batch buffer or a recompute that does not reproduce
/// the forward all move updates by 1e-2 or more.
const EXACT_ATTENTION: &[(&str, &str)] = &[("NSL_SDPA_FUSED_DISABLE", "1"), ("NSL_FLASH_BWD_CPU", "1")];
const EXACT_LOSS_TOL: f64 = 1e-5;
const EXACT_UPDATE_TOL: f64 = 3e-4;
/// FUSED ATTENTION (the production kernels): their MMA operands are f16, so
/// the bound is the f16-operand one -- measured ~1.4e-3 relative after one
/// step, ~2e-3 after four; the packed-SDPA parity gate's composed
/// forward-then-backward bound is 1e-2. Bisection showed every other flag
/// adds nothing measurable on top of the attention error.
const FUSED_LOSS_TOL: f64 = 5e-4;
const FUSED_UPDATE_TOL: f64 = 5e-3;

#[test]
#[ignore = "requires CUDA GPU"]
fn posture_certificate_canonical_exact_attention() {
    // Unclipped: --layerwise-accum refuses grad_clip (the layerwise schedule
    // never materialises the global norm), and the canonical recipe has none.
    certify(CANONICAL_FLAGS, None, "canonical_exact", EXACT_ATTENTION, EXACT_LOSS_TOL, EXACT_UPDATE_TOL);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn posture_certificate_chain_exact_attention() {
    certify(CHAIN_FLAGS, Some(CLIP), "chain_exact", EXACT_ATTENTION, EXACT_LOSS_TOL, EXACT_UPDATE_TOL);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn posture_certificate_canonical() {
    certify(CANONICAL_FLAGS, None, "canonical", &[], FUSED_LOSS_TOL, FUSED_UPDATE_TOL);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn posture_certificate_chain() {
    certify(CHAIN_FLAGS, Some(CLIP), "chain", &[], FUSED_LOSS_TOL, FUSED_UPDATE_TOL);
}

fn certify(
    flags: &'static [&'static str],
    clip: Option<f64>,
    tag: &str,
    env: &[(&str, &str)],
    loss_tol: f64,
    update_tol: f64,
) {
    let (loss, update, report) = compare_arms(Arm::Gpu(flags), STEPS, clip, tag, env);
    assert!(loss <= loss_tol, "loss streams disagree (bound {loss_tol:e}):\n{report}");
    assert!(update <= update_tol, "parameter updates disagree (bound {update_tol:e}):\n{report}");
}

/// (max loss difference, max relative update difference, report).
fn compare_arms(arm: Arm, steps: usize, clip: Option<f64>, tag: &str, env: &[(&str, &str)]) -> (f64, f64, String) {
    let cpu = run(Arm::CpuReference, steps, clip, &format!("{tag}_cpu"));
    let gpu = run_env(arm, steps, clip, &format!("{tag}_gpu"), env);
    // The GPU arm is the production line, not a fallback of it.
    if let Arm::Gpu(flags) = arm {
        assert!(gpu.stderr.contains("Using source-to-source AD for backward pass"), "{}", gpu.stderr);
        if flags.contains(&"--fuse-lm-head") {
            assert!(gpu.stderr.contains("[lm-head-fusion] inferred"), "GPU arm lacks the fused LM head:\n{}", gpu.stderr);
        }
    }
    assert!(!gpu.stderr.contains("falling back to tape-based AD"), "{}", gpu.stderr);
    assert!(!cpu.stderr.contains("Using source-to-source AD"), "the reference must be the tape:\n{}", cpu.stderr);

    // Same seed, same initial parameters (initialisation runs on the host).
    assert_eq!(cpu.init.len(), gpu.init.len());
    for (k, v) in &cpu.init {
        assert_eq!(v, &gpu.init[k], "{k}: the arms must start from the same parameters");
    }

    let mut report = Vec::new();
    let worst_loss = cpu.losses.iter().zip(&gpu.losses).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    report.push(format!("loss: max |gpu - cpu| = {worst_loss:.3e} over {} micro-batches", cpu.losses.len()));
    let mut names: Vec<&String> = cpu.init.keys().collect();
    names.sort();
    let mut worst_update = 0.0f64;
    for k in names {
        let d_cpu: Vec<f64> = cpu.init[k].iter().zip(&cpu.end[k]).map(|(a, b)| (*b - *a) as f64).collect();
        let d_gpu: Vec<f64> = gpu.init[k].iter().zip(&gpu.end[k]).map(|(a, b)| (*b - *a) as f64).collect();
        let scale = d_cpu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if !trainable(k) {
            // Configuration scalars and RoPE tables: neither arm may move them.
            assert!(scale == 0.0 && d_gpu.iter().all(|v| *v == 0.0), "{k} is not trainable but moved");
            continue;
        }
        // Every trainable field must train in the reference (the #806 class:
        // a parameter the step never reaches) -- and so must the GPU arm,
        // which the update comparison below checks element by element.
        assert!(scale > 0.0, "{k}: the reference never updated it -- a parameter the step does not reach");
        let diff = d_cpu.iter().zip(&d_gpu).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        worst_update = worst_update.max(diff / scale);
        report.push(format!("{k}: max |dgpu - dcpu| / max|dcpu| = {:.3e} (scale {scale:.3e})", diff / scale));
    }
    let report = report.join("\n");
    eprintln!("[{tag}]\n{report}");
    (worst_loss, worst_update, report)
}

/// Calibration, not a gate: the CPU reference's per-step gradient norm,
/// unclipped (the update of step k divided by lr), which `CLIP` is chosen
/// between.
#[test]
#[ignore = "diagnostic: prints the per-step gradient norms CLIP is chosen from"]
fn posture_gradient_norms() {
    for clip in [None, Some(CLIP)] {
        eprintln!("clip {clip:?}:");
        step_norms(clip);
    }
}

/// Per-step update norm / lr: the gradient norm where clipping did not
/// fire, exactly `clip` where it did.
fn step_norms(clip: Option<f64>) {
    let mut prev = None;
    for k in 1..=STEPS {
        let r = run(Arm::CpuReference, k, clip, &format!("norms{k}"));
        let total = update_norm(&r) / LR;
        let step = match &prev {
            None => total,
            Some(p) => {
                let p: &Run = p;
                r.end.iter().map(|(n, v)| v.iter().zip(&p.end[n]).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>()).sum::<f64>().sqrt() / LR
            }
        };
        eprintln!("  step {k}: |update| / lr = {step:.6}");
        prev = Some(r);
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

/// Diagnostic: the noise floor. Two independent AD implementations on the
/// SAME hardware (CPU tape under --training-reference vs CPU source AD), and
/// the GPU after ONE step instead of four, to separate per-step gradient
/// error from SGD amplifying small differences over steps.
#[test]
#[ignore = "diagnostic: posture certificate noise floor"]
fn posture_noise_floor() {
    compare_arms(Arm::Cpu(&["--source-ad"]), STEPS, Some(CLIP), "cpu_source_ad_4", &[]);
    compare_arms(Arm::Cpu(&["--source-ad"]), 1, Some(CLIP), "cpu_source_ad_1", &[]);
    compare_arms(Arm::Gpu(CHAIN_FLAGS), 1, Some(CLIP), "chain_1", &[]);
    compare_arms(Arm::Gpu(CHAIN_FLAGS), 1, Some(CLIP), "chain_unfused_1", &[("NSL_SDPA_FUSED_DISABLE", "1")]);
}

/// Diagnostic: attribute the GPU-vs-reference gap to a flag. One step per
/// configuration, adding the production flags one at a time.
#[test]
#[ignore = "diagnostic: posture certificate flag bisection"]
fn posture_flag_bisection() {
    const STAGES: &[(&str, &[&str], u8)] = &[
        ("bare_cpu_attn_bwd", &["--source-ad", "--matmul-mode", "f32"], 2),
        ("bare_unfused_attn", &["--source-ad", "--matmul-mode", "f32"], 1),
        ("bare", &["--source-ad", "--matmul-mode", "f32"], 0),
        ("lmhead", &["--source-ad", "--matmul-mode", "f32", "--fuse-lm-head", "require"], 0),
        ("rmsnorm", &["--source-ad", "--matmul-mode", "f32", "--fuse-lm-head", "require", "--fuse-rmsnorm-backward"], 0),
        ("wgrad", &["--source-ad", "--matmul-mode", "f32", "--fuse-lm-head", "require", "--fuse-rmsnorm-backward", "--fuse-wgrad-accum"], 0),
        ("blocks", &["--source-ad", "--matmul-mode", "f32", "--fuse-lm-head", "require", "--fuse-rmsnorm-backward", "--fuse-wgrad-accum", "--checkpoint-blocks"], 0),
        ("selective", CHAIN_FLAGS, 0),
    ];
    let mut summary = Vec::new();
    for (tag, flags, unfused) in STAGES {
        let env: &[(&str, &str)] = match unfused {
            2 => &[("NSL_SDPA_FUSED_DISABLE", "1"), ("NSL_FLASH_BWD_CPU", "1")],
            1 => &[("NSL_SDPA_FUSED_DISABLE", "1")],
            _ => &[],
        };
        let (loss, update, _) = compare_arms(Arm::Gpu(flags), 1, Some(CLIP), tag, env);
        summary.push(format!("{tag:>20}: loss {loss:.3e}  update {update:.3e}"));
    }
    eprintln!("BISECTION\n{}", summary.join("\n"));
}

/// The accumulation identity, independent of the accumulation code: one
/// optimizer step over four micro-batches of two sequences must equal one
/// step over the same eight sequences as two micro-batches of four (the loss
/// is a mean over tokens, so every split of the step averages to the same
/// gradient). Both sides accumulate -- `--fuse-wgrad-accum` refuses
/// grad_accumulation 1 -- but with different N, so a missing or wrong 1/N
/// puts the two updates in the ratio 4:2. The CPU reference shares the FASE accumulation recipe with the
/// GPU arm, so comparing the two cannot see a missing 1/N; this can, on
/// every arm. (A GLOBAL scale error is caught by the SGD-vs-f64 certificates
/// in fused_loss_gradient_cert_gpu.rs instead.) Unclipped, so the update is
/// the raw gradient.
#[test]
#[ignore = "requires CUDA GPU"]
fn posture_accumulation_identity() {
    let arms: [(Arm, &str, Env); 3] = [
        (Arm::CpuReference, "ref", &[]),
        (Arm::Gpu(CHAIN_FLAGS), "chain", EXACT_ATTENTION),
        (Arm::Gpu(CANONICAL_FLAGS), "canonical", EXACT_ATTENTION),
    ];
    let mut report = Vec::new();
    for (arm, tag, env) in arms {
        let split = run_shaped(arm, 1, (2, 4), None, &format!("accum_split_{tag}"), env);
        let whole = run_shaped(arm, 1, (4, 2), None, &format!("accum_whole_{tag}"), env);
        let mut worst = 0.0f64;
        for (k, v0) in &whole.init {
            assert_eq!(v0, &split.init[k], "{tag}: {k} must start equal");
            if !trainable(k) {
                continue;
            }
            let dw: Vec<f64> = v0.iter().zip(&whole.end[k]).map(|(a, b)| (*b - *a) as f64).collect();
            let ds: Vec<f64> = split.init[k].iter().zip(&split.end[k]).map(|(a, b)| (*b - *a) as f64).collect();
            let scale = dw.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(scale > 0.0, "{tag}: {k} never updated");
            worst = worst.max(dw.iter().zip(&ds).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / scale);
        }
        report.push(format!("{tag}: max |d(4 x 2) - d(2 x 4)| / max|d(2 x 4)| = {worst:.3e}"));
        assert!(worst <= 1e-4, "{tag}: the step depends on how its sequences are split:\n{}", report.join("\n"));
    }
    eprintln!("{}", report.join("\n"));
}
