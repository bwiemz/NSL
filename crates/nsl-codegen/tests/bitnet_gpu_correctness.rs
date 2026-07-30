//! BitNet b1.58 forward kernel: assembly gate + GPU numerical correctness.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §4.
//!
//! ## Why this file exists
//!
//! M35.1 shipped `synthesize_kernel` as a PTX *text* emitter whose output was
//! never assembled or launched. The module docstring admitted it was "NOT
//! runtime-loadable"; in practice `ptxas` rejected essentially every
//! instruction, because no `.reg` was declared and no index was derived from
//! `%ctaid`/`%tid`. The structural snapshot tests passed anyway — they only
//! string-matched the emitted text.
//!
//! Two gates close that hole:
//!
//! - [`synthesized_kernel_assembles`] — runs by default (no GPU needed, only
//!   `ptxas`). This is the regression gate: it fails loudly if the emitter ever
//!   goes back to producing text that is not a valid kernel.
//! - [`gpu_matches_reference_on_committed_fixtures`] and
//!   [`gpu_matches_reference_random`] — `#[ignore]`'d, `cuda`-feature-gated.
//!   They launch the kernel and require it to reproduce
//!   `bitnet_reference_impl.rs` **bit-exactly**.
//!
//! ## Oracle choice
//!
//! The committed fixtures store `f32` activations, but the kernel consumes
//! F16/BF16. Rounding to F16 changes the absmax scale and therefore the int8
//! quantization, so comparing the kernel against the raw-`f32` fixture outputs
//! would conflate storage precision with kernel correctness. The oracle here is
//! therefore `forward_reference` evaluated on *exactly the values the kernel
//! sees* (the F16-rounded activations), using the fixtures' canonical edge-case
//! inputs. `bitnet_reference::reference_matches_committed_fixtures` remains the
//! test that pins the reference itself against the committed JSON.
//!
//! Run the GPU tests with:
//! `cargo test -p nsl-codegen --features cuda --test bitnet_gpu_correctness -- --ignored --nocapture`

// The fixture loader and host-side packing helpers are consumed only by the
// `cuda`-gated GPU tests; without that feature they are legitimately unused.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

#[path = "bitnet_reference_impl.rs"]
mod reference;

use nsl_codegen::bitnet::pack::pack_trit_slice;
use nsl_codegen::bitnet::{synthesize_kernel, BitNetKernelConfig};
use nsl_codegen::kernel_ir::KirType;
use serde::Deserialize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct FixtureInput {
    activations: Vec<Vec<f32>>,
    weights: Vec<Vec<i8>>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    input: FixtureInput,
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    fixtures: Vec<Fixture>,
}

fn load_fixtures() -> Vec<Fixture> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bitnet_reference_outputs.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let parsed: FixtureFile =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse fixture JSON: {e}"));
    parsed.fixtures
}

/// Config for a given (hidden_dim, out_dim). `block_n` is the CTA width and
/// must be a multiple of 32; the fixtures are tiny, so 32 is the natural pick
/// and it deliberately exercises the out-of-range column guard (out_dim is 4
/// or 8, so most of the warp exits early).
fn config_for(hidden_dim: u32, out_dim: u32, block_n: u32) -> BitNetKernelConfig {
    BitNetKernelConfig {
        block_m: 64,
        block_n,
        block_k: 128,
        activation_dtype: KirType::F16,
        output_dtype: KirType::F16,
        hidden_dim,
        out_dim,
        fused_rmsnorm: false,
        fused_bias_add: false,
        fused_residual_add: false,
        block_m_backward: 64,
        block_n_backward: 128,
        block_k_backward: 128,
    }
}

// ---------------------------------------------------------------------------
// Assembly gate (no GPU required)
// ---------------------------------------------------------------------------

fn find_ptxas() -> Option<PathBuf> {
    // Deliberately local rather than reusing tests/sass_baseline_helpers.rs:
    // that module has no `#![allow(dead_code)]`, so importing it here would
    // emit dead-code warnings for the SASS-baseline helpers this file does not
    // use.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let p = PathBuf::from(cuda_path).join("bin").join("ptxas");
        if p.exists() {
            return Some(p);
        }
    }
    for cand in ["/opt/cuda/bin/ptxas", "/usr/local/cuda/bin/ptxas"] {
        let p = PathBuf::from(cand);
        if p.exists() {
            return Some(p);
        }
    }
    // Fall back to PATH.
    let probe = std::process::Command::new("ptxas").arg("--version").output();
    match probe {
        Ok(o) if o.status.success() => Some(PathBuf::from("ptxas")),
        _ => None,
    }
}

/// Assemble `ptx` for `sm`, returning ptxas's stderr on failure.
fn assemble(ptxas: &PathBuf, ptx: &str, sm: u32) -> Result<(), String> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Unique per call: `cargo test` runs these in parallel threads, and a dir
    // keyed only on PID would be deleted out from under a concurrent case.
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "nsl_bitnet_ptxas_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    let src = dir.join(format!("bitnet_sm{sm}.ptx"));
    let mut f = std::fs::File::create(&src).map_err(|e| format!("create: {e}"))?;
    f.write_all(ptx.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    drop(f);

    let out = std::process::Command::new(ptxas)
        .arg(format!("--gpu-name=sm_{sm}"))
        .arg(&src)
        .arg("-o")
        .arg(dir.join(format!("bitnet_sm{sm}.cubin")))
        .output()
        .map_err(|e| format!("spawn ptxas: {e}"))?;
    let _ = std::fs::remove_dir_all(&dir);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// THE regression gate. Before this landed, `ptxas` rejected the emitted module
/// at essentially every instruction while all the structural tests stayed green.
#[test]
fn synthesized_kernel_assembles() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("SKIP: ptxas not found (set CUDA_PATH or put ptxas on PATH)");
        return;
    };

    // A spread of shapes: the default, a fixture-sized tiny one, a non-multiple
    // out_dim that forces the column guard, and a 3B-model-sized hidden dim.
    let cases = [
        (1024u32, 1024u32, 128u32),
        (4, 4, 32),
        (128, 100, 64),
        (3200, 8640, 256),
    ];

    for (hidden, out_dim, block_n) in cases {
        let config = config_for(hidden, out_dim, block_n);
        let ptx_bytes = synthesize_kernel(&config);
        let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");
        for sm in [80u32, 90, 120] {
            match assemble(&ptxas, &ptx, sm) {
                Ok(()) => {}
                Err(log) => panic!(
                    "ptxas REJECTED the synthesized BitNet kernel \
                     (hidden={hidden}, out_dim={out_dim}, block_n={block_n}, sm_{sm}).\n\
                     The emitter must produce an assemblable module.\n\
                     ptxas said:\n{log}\n\nPTX:\n{ptx}"
                ),
            }
        }
        eprintln!(
            "ptxas OK: hidden={hidden} out_dim={out_dim} block_n={block_n} (sm_80/90/120)"
        );
    }
}

/// Bias/residual epilogues must also assemble — they add params and registers.
#[test]
fn synthesized_kernel_assembles_with_epilogues() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("SKIP: ptxas not found");
        return;
    };
    for (bias, residual) in [(true, false), (false, true), (true, true)] {
        let mut config = config_for(256, 256, 64);
        config.fused_bias_add = bias;
        config.fused_residual_add = residual;
        let ptx = String::from_utf8(synthesize_kernel(&config)).expect("utf8");
        if let Err(log) = assemble(&ptxas, &ptx, 80) {
            panic!("ptxas rejected bias={bias} residual={residual}:\n{log}\n\nPTX:\n{ptx}");
        }
    }
}

/// BF16 activations select `cvt.f32.bf16` / `cvt.rn.bf16.f32`; make sure that
/// variant assembles too (it is a different opcode pair, not just a rename).
#[test]
fn synthesized_kernel_assembles_bf16() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("SKIP: ptxas not found");
        return;
    };
    let mut config = config_for(256, 256, 64);
    config.activation_dtype = KirType::Bf16;
    config.output_dtype = KirType::Bf16;
    let ptx = String::from_utf8(synthesize_kernel(&config)).expect("utf8");
    if let Err(log) = assemble(&ptxas, &ptx, 80) {
        panic!("ptxas rejected the BF16 variant:\n{log}\n\nPTX:\n{ptx}");
    }
}

// ---------------------------------------------------------------------------
// Shared host-side helpers
// ---------------------------------------------------------------------------

/// Pack a `[out_dim][hidden_dim]` ternary matrix into the kernel's layout.
///
/// The fixtures store weights as `[hidden_dim][out_dim]` (`weights[k][c]`,
/// the CPU reference's convention). The kernel wants `[out_dim, hidden_dim]`
/// — HF `nn.Linear`'s native `[out_features, in_features]` order — so this
/// transposes before packing. See `bitnet/phases/packed_load.rs`.
fn pack_transposed(weights_k_by_c: &[Vec<i8>], hidden_dim: usize, out_dim: usize) -> Vec<u8> {
    let mut flat = Vec::with_capacity(hidden_dim * out_dim);
    for c in 0..out_dim {
        for k in 0..hidden_dim {
            flat.push(weights_k_by_c[k][c]);
        }
    }
    pack_trit_slice(&flat)
}

/// Round activations to the kernel's storage dtype and back, so the CPU
/// reference sees exactly what the GPU will read.
fn round_trip_f16(activations: &[Vec<f32>]) -> (Vec<u16>, Vec<Vec<f32>>) {
    let mut bits = Vec::new();
    let mut rounded = Vec::with_capacity(activations.len());
    for row in activations {
        let mut r = Vec::with_capacity(row.len());
        for &v in row {
            let h = half::f16::from_f32(v);
            bits.push(h.to_bits());
            r.push(h.to_f32());
        }
        rounded.push(r);
    }
    (bits, rounded)
}

// ---------------------------------------------------------------------------
// GPU correctness
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use cudarc::driver::sys;
    use std::ffi::CString;
    use std::os::raw::c_void;

    /// Establish a CUDA context on this thread.
    ///
    /// CUDA 13 removed `cuCtxCreate`; the primary-context retain path is the
    /// documented replacement (NSL MEMORY.md "GPU invariants"). Returns false
    /// when no device is usable, so the tests can skip rather than fail.
    pub unsafe fn ensure_context() -> bool {
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        if sys::cuCtxGetCurrent(&mut ctx) == sys::CUresult::CUDA_SUCCESS && !ctx.is_null() {
            return true;
        }
        if sys::cuInit(0) != sys::CUresult::CUDA_SUCCESS {
            return false;
        }
        let mut dev: sys::CUdevice = 0;
        if sys::cuDeviceGet(&mut dev, 0) != sys::CUresult::CUDA_SUCCESS {
            return false;
        }
        if sys::cuDevicePrimaryCtxRetain(&mut ctx, dev) != sys::CUresult::CUDA_SUCCESS {
            return false;
        }
        sys::cuCtxSetCurrent(ctx) == sys::CUresult::CUDA_SUCCESS
    }

    unsafe fn load_module(ptx: &str) -> Result<sys::CUmodule, String> {
        let ptx_c = CString::new(ptx).map_err(|_| "PTX contains embedded NUL".to_string())?;
        let mut module: sys::CUmodule = std::ptr::null_mut();
        let mut err_log = vec![0u8; 8192];
        let mut options = [
            sys::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER,
            sys::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut values: [*mut c_void; 2] = [
            err_log.as_mut_ptr() as *mut c_void,
            err_log.len() as *mut c_void,
        ];
        let rc = sys::cuModuleLoadDataEx(
            &mut module,
            ptx_c.as_ptr() as *const c_void,
            options.len() as u32,
            options.as_mut_ptr(),
            values.as_mut_ptr(),
        );
        if rc != sys::CUresult::CUDA_SUCCESS {
            let nul = err_log.iter().position(|&b| b == 0).unwrap_or(err_log.len());
            return Err(format!(
                "cuModuleLoadDataEx rc={rc:?}; JIT log: {}",
                String::from_utf8_lossy(&err_log[..nul]).trim()
            ));
        }
        Ok(module)
    }

    /// Launch the BitNet kernel and return the `[rows, out_dim]` F16 output as f32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        ptx: &str,
        kernel_name: &str,
        acts_f16_bits: &[u16],
        packed: &[u8],
        rows: u32,
        hidden_dim: u32,
        out_dim: u32,
        block_n: u32,
        weight_scale: f32,
    ) -> Result<Vec<f32>, String> {
        let module = load_module(ptx)?;
        let name_c = CString::new(kernel_name).map_err(|_| "bad kernel name".to_string())?;
        let mut func: sys::CUfunction = std::ptr::null_mut();
        if sys::cuModuleGetFunction(&mut func, module, name_c.as_ptr())
            != sys::CUresult::CUDA_SUCCESS
        {
            return Err(format!("cuModuleGetFunction({kernel_name}) failed"));
        }

        let act_bytes = (acts_f16_bits.len() * 2) as usize;
        let w_bytes = packed.len();
        let y_elems = (rows * out_dim) as usize;
        let y_bytes = y_elems * 2;

        let mut d_act: sys::CUdeviceptr = 0;
        let mut d_w: sys::CUdeviceptr = 0;
        let mut d_y: sys::CUdeviceptr = 0;
        if sys::cuMemAlloc_v2(&mut d_act, act_bytes) != sys::CUresult::CUDA_SUCCESS
            || sys::cuMemAlloc_v2(&mut d_w, w_bytes) != sys::CUresult::CUDA_SUCCESS
            || sys::cuMemAlloc_v2(&mut d_y, y_bytes) != sys::CUresult::CUDA_SUCCESS
        {
            return Err("cuMemAlloc failed".to_string());
        }
        sys::cuMemcpyHtoD_v2(d_act, acts_f16_bits.as_ptr() as *const c_void, act_bytes);
        sys::cuMemcpyHtoD_v2(d_w, packed.as_ptr() as *const c_void, w_bytes);
        sys::cuMemsetD8_v2(d_y, 0, y_bytes);

        let mut p_act = d_act;
        let mut p_w = d_w;
        let mut p_y = d_y;
        let mut p_ws = weight_scale;
        let mut params: [*mut c_void; 4] = [
            &mut p_act as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut p_ws as *mut _ as *mut c_void,
        ];

        let grid_y = out_dim.div_ceil(block_n);
        let rc = sys::cuLaunchKernel(
            func, rows, grid_y, 1, block_n, 1, 1, 0, std::ptr::null_mut(),
            params.as_mut_ptr(), std::ptr::null_mut(),
        );
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuLaunchKernel rc={rc:?}"));
        }
        if sys::cuCtxSynchronize() != sys::CUresult::CUDA_SUCCESS {
            return Err("cuCtxSynchronize failed (kernel faulted)".to_string());
        }

        let mut out_bits = vec![0u16; y_elems];
        sys::cuMemcpyDtoH_v2(out_bits.as_mut_ptr() as *mut c_void, d_y, y_bytes);

        sys::cuMemFree_v2(d_act);
        sys::cuMemFree_v2(d_w);
        sys::cuMemFree_v2(d_y);
        sys::cuModuleUnload(module);

        Ok(out_bits
            .into_iter()
            .map(|b| half::f16::from_bits(b).to_f32())
            .collect())
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_matches_reference_on_committed_fixtures() {
    use reference::forward_reference;

    if !unsafe { gpu::ensure_context() } {
        eprintln!("SKIP: no usable CUDA device");
        return;
    }

    // weight_scale is applied by the kernel's finalize phase but is NOT part of
    // `forward_reference` (which stops at the dequantized GEMM output), so the
    // oracle comparison uses 1.0. `weight_scale_is_applied_multiplicatively`
    // covers the non-unit case.
    const WEIGHT_SCALE: f32 = 1.0;
    const BLOCK_N: u32 = 32;

    let fixtures = load_fixtures();
    assert_eq!(fixtures.len(), 10, "expected the 10 committed fixtures");

    let mut checked = 0usize;
    for fx in &fixtures {
        let hidden_dim = fx.input.activations[0].len();
        let out_dim = fx.input.weights[0].len();
        let rows = fx.input.activations.len();

        let (acts_bits, acts_rounded) = round_trip_f16(&fx.input.activations);
        let (_scales, _q, expected) = forward_reference(&acts_rounded, &fx.input.weights);

        let config = config_for(hidden_dim as u32, out_dim as u32, BLOCK_N);
        let ptx = String::from_utf8(synthesize_kernel(&config)).expect("utf8");
        let packed = pack_transposed(&fx.input.weights, hidden_dim, out_dim);

        let got = unsafe {
            gpu::launch(
                &ptx, &config.kernel_name(), &acts_bits, &packed,
                rows as u32, hidden_dim as u32, out_dim as u32, BLOCK_N, WEIGHT_SCALE,
            )
        }
        .unwrap_or_else(|e| panic!("fixture {}: launch failed: {e}", fx.name));

        for r in 0..rows {
            for c in 0..out_dim {
                let want = half::f16::from_f32(expected[r][c]).to_f32();
                let have = got[r * out_dim + c];
                assert_eq!(
                    have.to_bits(),
                    want.to_bits(),
                    "fixture {} [{r},{c}]: GPU={have} reference={want} \
                     (hidden={hidden_dim}, out_dim={out_dim})",
                    fx.name
                );
            }
        }
        checked += rows * out_dim;
        eprintln!(
            "fixture {:24} OK  rows={rows} hidden={hidden_dim} out_dim={out_dim}",
            fx.name
        );
    }
    eprintln!("All 10 fixtures bit-exact across {checked} output elements.");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_matches_reference_random() {
    use reference::forward_reference;

    if !unsafe { gpu::ensure_context() } {
        eprintln!("SKIP: no usable CUDA device");
        return;
    }

    // Shapes chosen to stress the geometry: out_dim not a multiple of block_n
    // (column guard), a single row, a multi-warp CTA, and a hidden dim that is
    // large enough to need many GEMM iterations.
    let cases: [(usize, usize, usize, u32); 4] = [
        (1, 256, 100, 64),
        (5, 128, 128, 32),
        (3, 512, 300, 256),
        (2, 1024, 64, 128),
    ];

    for (rows, hidden_dim, out_dim, block_n) in cases {
        // Deterministic per-shape seed for the LCG below; no rand dependency.
        const SEED_BASE: u64 = 0xB175_E700_D1CE_F00D;
        let mut state: u64 = SEED_BASE ^ ((hidden_dim as u64) << 17) ^ (out_dim as u64);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };

        let activations: Vec<Vec<f32>> = (0..rows)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| ((next() >> 40) as f32 / 8192.0) - 1.0)
                    .collect()
            })
            .collect();
        let weights: Vec<Vec<i8>> = (0..hidden_dim)
            .map(|_| {
                (0..out_dim)
                    .map(|_| (((next() >> 32) % 3) as i8) - 1)
                    .collect()
            })
            .collect();

        let (acts_bits, acts_rounded) = round_trip_f16(&activations);
        let (_s, _q, expected) = forward_reference(&acts_rounded, &weights);

        let config = config_for(hidden_dim as u32, out_dim as u32, block_n);
        let ptx = String::from_utf8(synthesize_kernel(&config)).expect("utf8");
        let packed = pack_transposed(&weights, hidden_dim, out_dim);

        let got = unsafe {
            gpu::launch(
                &ptx, &config.kernel_name(), &acts_bits, &packed,
                rows as u32, hidden_dim as u32, out_dim as u32, block_n, 1.0,
            )
        }
        .unwrap_or_else(|e| panic!("launch failed for {rows}x{hidden_dim}x{out_dim}: {e}"));

        let mut mismatches = 0usize;
        for r in 0..rows {
            for c in 0..out_dim {
                let want = half::f16::from_f32(expected[r][c]).to_f32();
                let have = got[r * out_dim + c];
                if have.to_bits() != want.to_bits() {
                    if mismatches < 5 {
                        eprintln!("  MISMATCH [{r},{c}]: GPU={have} reference={want}");
                    }
                    mismatches += 1;
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "{mismatches} mismatches for rows={rows} hidden={hidden_dim} \
             out_dim={out_dim} block_n={block_n}"
        );
        eprintln!("random case OK: rows={rows} hidden={hidden_dim} out_dim={out_dim} block_n={block_n}");
    }
}

/// `weight_scale` must scale the output linearly. This is the M35.1 PR #172
/// wiring; a regression here silently produces outputs off by a per-layer
/// constant, which no structural test would catch.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA GPU"]
fn weight_scale_is_applied_multiplicatively() {
    if !unsafe { gpu::ensure_context() } {
        eprintln!("SKIP: no usable CUDA device");
        return;
    }

    let (rows, hidden_dim, out_dim, block_n) = (2usize, 64usize, 32usize, 32u32);
    let activations: Vec<Vec<f32>> = (0..rows)
        .map(|r| {
            (0..hidden_dim)
                .map(|k| ((k as f32) * 0.03125) - 1.0 + (r as f32) * 0.5)
                .collect()
        })
        .collect();
    let weights: Vec<Vec<i8>> = (0..hidden_dim)
        .map(|k| (0..out_dim).map(|c| (((k + c) % 3) as i8) - 1).collect())
        .collect();

    let (acts_bits, _) = round_trip_f16(&activations);
    let config = config_for(hidden_dim as u32, out_dim as u32, block_n);
    let ptx = String::from_utf8(synthesize_kernel(&config)).expect("utf8");
    let packed = pack_transposed(&weights, hidden_dim, out_dim);

    let run = |ws: f32| unsafe {
        gpu::launch(
            &ptx, &config.kernel_name(), &acts_bits, &packed,
            rows as u32, hidden_dim as u32, out_dim as u32, block_n, ws,
        )
        .expect("launch")
    };

    let base = run(1.0);
    let scaled = run(4.0);

    let mut compared = 0usize;
    for (b, s) in base.iter().zip(scaled.iter()) {
        if *b == 0.0 {
            continue;
        }
        // Exact in F16: 4.0 is a power of two, so scaling cannot change the
        // mantissa, only the exponent.
        assert_eq!(
            *s, *b * 4.0,
            "weight_scale=4.0 must scale the output exactly (base={b}, scaled={s})"
        );
        compared += 1;
    }
    assert!(compared > 0, "fixture produced an all-zero output; test is vacuous");
    eprintln!("weight_scale linearity verified across {compared} non-zero outputs");
}
