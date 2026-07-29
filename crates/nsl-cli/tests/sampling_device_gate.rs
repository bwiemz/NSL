//! Regression gate: the sampling FFIs must accept GPU-resident inputs.
//!
//! `nsl_tensor_topk` / `nsl_tensor_multinomial` / `nsl_tensor_argmax` /
//! `nsl_tensor_cumsum` / `nsl_tensor_lt_scalar` (sampling.rs) are host-side
//! implementations that read `tensor.data` raw. A GPU tensor's `data` is a
//! device pointer (`cuda::inner::alloc_managed` — "CPU code must NOT
//! dereference it"), so every one of them segfaulted the moment inference
//! logits lived on the GPU — which is exactly what
//! `stdlib/nsl/inference/sampling.nsl` feeds them. Reproduced 2026-07-28 on
//! an RTX 5070 Ti / CUDA 13.3: all five ops = SIGSEGV on a `.to(cuda)`
//! input.
//!
//! The fix stages GPU inputs through the host and returns results on the
//! input's device (the established CPU-redirect idiom of
//! `nsl_tensor_gather`'s non-dim-0 arm). These gates pin:
//!   1. each primitive runs on GPU inputs and matches its CPU result
//!      exactly (the f32→f64→f32 round trip is value-exact);
//!   2. the user-facing stdlib chain (`sample_greedy` / `sample_top_k` /
//!      `sample_top_p`) completes on GPU logits with the right token;
//!   3. the staging copies do not strand — exit `live_blocks` are identical
//!      across call counts (same design as `view_chain_leak_gate.rs`).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Run a whole program and return (exit live_blocks, stdout, stderr).
fn run_fixture(src: &str, tag: &str, n: usize) -> (i64, String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_sampdev_{}_{tag}_{n}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_GPU_MEM_REPORT", "1")
        .output()
        .expect("spawn nsl run");
    assert!(
        out.status.success(),
        "[{tag}/{n}] run failed (the pre-fix failure mode here is a SIGSEGV \
         with no program output):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stdout.contains("DONE"),
        "[{tag}/{n}] fixture did not complete:\n{stdout}"
    );
    let live_blocks = stderr
        .lines()
        .filter_map(|l| {
            l.split("live_blocks=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<i64>().ok())
        })
        .next_back()
        .unwrap_or_else(|| panic!("[{tag}/{n}] no [gpu-mem] live_blocks report:\n{stderr}"));
    // `nsl run` links ~140 MB of objects per invocation into the system temp
    // dir (a 31 GB tmpfs here) — always remove the scratch dir.
    std::fs::remove_dir_all(&tmp).ok();
    (live_blocks, stdout, stderr)
}

/// Every primitive, GPU input vs CPU reference, exact parity.
///
/// The one-hot probability rows make `multinomial` deterministic without
/// touching RNG state: all the mass sits on one category, so the sampled
/// index is forced regardless of the draw.
const PRIMITIVES_SRC: &str = r#"
let x = arange(0.0, 16.0).reshape([2, 8])
let g = x.to(cuda)

let a_cpu = argmax(x, -1)
let a_gpu = argmax(g, -1)
let d1 = a_cpu - a_gpu
let e1 = sum(d1 * d1).item()

let c_cpu = cumsum(x, -1)
let c_gpu = cumsum(g, -1)
let d2 = c_cpu - c_gpu
let e2 = sum(d2 * d2).item()

let m_cpu = lt_scalar(x, 5.0)
let m_gpu = lt_scalar(g, 5.0)
let d3 = m_cpu - m_gpu
let e3 = sum(d3 * d3).item()

let t_cpu = topk(x, 3)
let t_gpu = topk(g, 3)
let dv = t_cpu["values"] - t_gpu["values"]
let e4 = sum(dv * dv).item()
let di = t_cpu["indices"] - t_gpu["indices"]
let e5 = sum(di * di).item()

let ramp = arange(0.0, 4.0)
let onehot2 = lt_scalar(ramp, 3.0) - lt_scalar(ramp, 2.0)
let s_cpu = multinomial(onehot2, 4)
let s_gpu = multinomial(onehot2.to(cuda), 4)
let d6 = s_cpu - s_gpu
let e6 = sum(d6 * d6).item()

let err = e1 + e2 + e3 + e4 + e5 + e6
print(err)
if err == 0.0:
    print("PARITY_OK")
print("DONE")
"#;

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_inputs_match_cpu_for_each_primitive() {
    let (_lb, stdout, _stderr) = run_fixture(PRIMITIVES_SRC, "prims", 1);
    assert!(
        stdout.contains("PARITY_OK"),
        "GPU-input results diverged from CPU (err printed above DONE):\n{stdout}"
    );
}

/// The stdlib chain that motivated the fix, on GPU logits. Logits are a
/// one-hot at index 5 scaled by 100, so after softmax the winning
/// probability is exactly 1.0 in f32 (exp(-100) underflows) and every
/// strategy must return token 5 deterministically.
fn chain_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
from nsl.inference.sampling import sample_greedy, sample_top_k, sample_top_p
let ramp = arange(0.0, 16.0)
let logits = (lt_scalar(ramp, 6.0) - lt_scalar(ramp, 5.0)) * 100.0
let gl = logits.to(cuda)
let err = 0.0
"#,
    );
    for _ in 0..calls {
        s.push_str(
            r#"let tg = sample_greedy(gl).item() - 5.0
let tk = sample_top_k(gl, 3, 1.0).item() - 5.0
let tp = sample_top_p(gl, 0.9, 1.0).item() - 5.0
err = err + tg * tg + tk * tk + tp * tp
"#,
        );
    }
    s.push_str(
        r#"print(err)
if err == 0.0:
    print("PARITY_OK")
print("DONE")
"#,
    );
    s
}

#[test]
#[ignore = "requires CUDA GPU"]
fn stdlib_sampling_chain_runs_on_gpu_logits() {
    let (_lb, stdout, _stderr) = run_fixture(&chain_src(1), "chain", 1);
    assert!(
        stdout.contains("PARITY_OK"),
        "stdlib sampling on GPU logits returned the wrong token:\n{stdout}"
    );
}

/// The redirect allocates two host stagings + a device upload per op; every
/// one must be freed. Anything stranded scales with the call count.
///
/// `topk` is deliberately absent here — see the gate below. The per-op
/// results are BOUND to locals rather than nested (`let t = cumsum(gl, -1)`,
/// not `sum(cumsum(gl, -1))`): most builtin dispatch arms compile their
/// arguments with `compile_expr`, which never registers a nested call's
/// fresh result as a statement temporary, so the nested spelling strands
/// 1 block per call — measured 2026-07-28, device-independent, and
/// PRE-EXISTING (the redirect only made it visible in `live_blocks`). That
/// class belongs to its own item; bound locals are swept correctly today
/// and isolate what THIS fix owns.
fn redirect_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
from nsl.inference.sampling import sample_greedy
let ramp = arange(0.0, 16.0)
let logits = (lt_scalar(ramp, 6.0) - lt_scalar(ramp, 5.0)) * 100.0
let gl = logits.to(cuda)
let onehot2g = (lt_scalar(ramp, 3.0) - lt_scalar(ramp, 2.0)).to(cuda)
"#,
    );
    for i in 0..calls {
        s.push_str(&format!(
            "let g{i} = sample_greedy(gl).item()\n\
             let ct{i} = cumsum(gl, -1)\n\
             let c{i} = sum(ct{i}).item()\n\
             let mt{i} = lt_scalar(gl, 0.5)\n\
             let m{i} = sum(mt{i}).item()\n\
             let st{i} = multinomial(onehot2g, 4)\n\
             let s{i} = sum(st{i}).item()\n"
        ));
    }
    s.push_str("print(\"DONE\")\n");
    s
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_redirect_does_not_strand_per_call() {
    let (lb1, ..) = run_fixture(&redirect_src(1), "leak", 1);
    let (lb3, ..) = run_fixture(&redirect_src(3), "leak", 3);
    assert_eq!(
        lb1, lb3,
        "sampling CPU-redirect strands {} block(s) per call round",
        (lb3 - lb1) / 2
    );
}

/// `return multinomial(x, 1)` / `return lt_scalar(x, p)` from a user fn:
/// both were missing from the owning-ref Ident allowlist, so the callee's
/// Return arm retained an already-owning transfer and one block stranded
/// per call (the `return rmsnorm(...)` class of PR #424, measured red at
/// 2/round on 2026-07-28 before the allowlist entries landed).
fn return_position_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
fn pick(x: Tensor) -> Tensor:
    return multinomial(x, 4)

fn mask_of(x: Tensor) -> Tensor:
    return lt_scalar(x, 0.5)

let ramp = arange(0.0, 16.0)
let oh = (lt_scalar(ramp, 3.0) - lt_scalar(ramp, 2.0)).to(cuda)
let gl = ramp.to(cuda)
"#,
    );
    for i in 0..calls {
        s.push_str(&format!(
            "let p{i} = pick(oh)\n\
             let a{i} = sum(p{i}).item()\n\
             let w{i} = mask_of(gl)\n\
             let b{i} = sum(w{i}).item()\n"
        ));
    }
    s.push_str("print(\"DONE\")\n");
    s
}

#[test]
#[ignore = "requires CUDA GPU"]
fn returned_sampling_results_do_not_double_own() {
    let (lb1, ..) = run_fixture(&return_position_src(1), "retown", 1);
    let (lb3, ..) = run_fixture(&return_position_src(3), "retown", 3);
    assert_eq!(
        lb1, lb3,
        "return-position sampling results strand {} block(s) per round",
        (lb3 - lb1) / 2
    );
}

/// PRE-EXISTING dict-content strand, pinned — not caused by the redirect.
///
/// `nsl_tensor_topk` returns a dict owning its `values`/`indices` tensors
/// (rc=1 each, `nsl_dict_get_str` hands out borrows). Codegen emits
/// `nsl_dict_free_tensor_values` only for DataLoader batch dicts, so an
/// ordinary dict local is never freed — the aggregate-lifetime gap. On CPU
/// inputs that has always stranded heap (measured 2026-07-28: a
/// `topk(x, 1M)` loop grows ~79 MB/call of resident heap, fix present or
/// not); on GPU inputs the same two stranded tensors now live in VRAM,
/// where `live_blocks` can finally see them. This gate pins the strand at
/// EXACTLY 2 blocks per topk call so a regression fails loudly and the
/// aggregate-lifetime fix must ratchet it down to zero when it lands.
#[test]
#[ignore = "requires CUDA GPU"]
fn topk_dict_strand_is_exactly_the_known_aggregate_gap() {
    let (lb1, ..) = run_fixture(&chain_src(1), "dictpin", 1);
    let (lb3, ..) = run_fixture(&chain_src(3), "dictpin", 3);
    // chain_src runs sample_top_k + sample_top_p per round = 2 topk dicts,
    // 2 tensors each.
    assert_eq!(
        lb3 - lb1,
        2 * 4,
        "per-round GPU strand changed (was exactly the 2 topk dicts' \
         values+indices): lb1={lb1} lb3={lb3}"
    );
}
