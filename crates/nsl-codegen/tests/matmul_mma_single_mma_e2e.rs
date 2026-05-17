//! End-to-end numerical validation of the `matmul_mma` helper stack:
//! `emit_load_a_fragment_smem` + `emit_load_b_fragment_smem` +
//! `emit_mma_instruction`.
//!
//! ## Why this test exists
//!
//! The N4 disambiguation probes (`tier_b1_n4_helper_probe.rs`,
//! `tier_b1_n4_b_helper_probe.rs`) classified per-lane footprints of the
//! load helpers in isolation and confirmed both now match the PTX
//! m16n8k16 spec (32/32 lanes each). This test takes the next step: it
//! actually runs ONE `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
//! against known A and B operands in SMEM and compares the per-lane D
//! accumulator output to a CPU `A @ B` reference. It does not exercise
//! Tier B.1's projection / softmax / attention chain — it isolates the
//! fragment-load + mma path so a regression in any of those three
//! helpers shows up here directly.
//!
//! ## Inputs
//!
//! * `A` (m × k = 16 × 16) f16 row-major. `A[r, c] = (r * 100 + c) / 1024`
//!   — encodes (r, c) at small magnitudes; exact in f16 for r,c < 16.
//! * `B` (k × n = 16 × 8) f16 col-major. `B[k, n] = (k * 100 + n) / 4096`
//!   — small magnitudes again; chosen so `A @ B` stays within f16 range.
//! * `C = 0` (no bias).
//!
//! ## CPU reference
//!
//! `D[r, n] = sum over k of (A[r, k] * B[k, n])` for r ∈ [0, 16), n ∈ [0, 8).
//!
//! ## GPU output reconstruction
//!
//! Each lane t holds 4 f32 D-fragment values per PTX m16n8k16 spec:
//!   reg 0: D[t/4,     (t%4)*2    ]
//!   reg 1: D[t/4,     (t%4)*2 + 1]
//!   reg 2: D[t/4 + 8, (t%4)*2    ]
//!   reg 3: D[t/4 + 8, (t%4)*2 + 1]
//!
//! Each lane writes its 4 f32 to HBM at `lane * 16 bytes`; host then
//! gathers them into the full 16x8 D matrix and compares.
//!
//! ## Running
//!
//! ```bash
//! cargo test --package nsl-codegen --features cuda \
//!     --test matmul_mma_single_mma_e2e -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_codegen::matmul_mma::{
    emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
};
use std::ffi::{c_void, CString};

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};

extern "C" {
    fn nsl_kernel_launch(
        ptx_ptr: i64,
        name_ptr: i64,
        grid_x: i64,
        grid_y: i64,
        grid_z: i64,
        block_x: i64,
        block_y: i64,
        block_z: i64,
        args_ptr: i64,
        num_args: i64,
        shared_mem_bytes: i64,
    ) -> i64;
}

const M: u32 = 16;
const N: u32 = 8;
const K: u32 = 16;
const A_BYTES: usize = (M * K * 2) as usize; // 512
const B_BYTES: usize = (K * N * 2) as usize; // 256
const D_F32_BYTES: usize = (M * N * 4) as usize; // 512
/// Per-lane D output: 4 f32 = 16 bytes.
const PER_LANE_D_F32: usize = 4;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e: i32 = -1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let e = (127 + e - 14) as u32;
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let e = exp + (127 - 15);
        (sign << 31) | (e << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_f16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7E00;
    }
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 255 {
        return ((sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 }) as u16;
    }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 <= 0 {
        let shift = (1 - exp_f16).min(24) as u32;
        let shifted = (mant | 0x800000) >> shift;
        let rounded = (shifted + 0x1000) >> 13;
        return ((sign << 15) | rounded) as u16;
    }
    if exp_f16 >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    let mant16 = (mant + 0x1000) >> 13;
    let overflow = (mant16 >> 10) & 1;
    let exp16 = (exp_f16 as u32 + overflow) & 0x1F;
    ((sign << 15) | (exp16 << 10) | (mant16 & 0x3FF)) as u16
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

/// Build the probe PTX kernel:
///  - 32 threads (single warp)
///  - SMEM holds A (row-major) + B (col-major)
///  - Each lane: copies its slice of A/B from HBM into SMEM, calls the
///    fragment-load helpers + mma instruction, writes 4 f32 to HBM
fn build_probe_ptx() -> Vec<u8> {
    let mut ptx = String::new();
    ptx.push_str(".version 8.7\n");
    ptx.push_str(".target sm_80\n");
    ptx.push_str(".address_size 64\n\n");
    ptx.push_str(".visible .entry matmul_mma_single_mma_probe (\n");
    ptx.push_str("    .param .u64 a_ptr,\n");
    ptx.push_str("    .param .u64 b_ptr,\n");
    ptx.push_str("    .param .u64 d_ptr\n");
    ptx.push_str(")\n{\n");
    // SMEM: A_tile + B_tile contiguous. A = 512 bytes, B = 256 bytes;
    // total = 768 bytes.
    ptx.push_str(&format!(
        "    .shared .align 16 .b8 ab_smem[{}];\n",
        A_BYTES + B_BYTES
    ));
    ptx.push_str("    .reg .u32 %tid_x, %lane, %ofs;\n");
    ptx.push_str("    .reg .u64 %rd_ptr, %rd_off;\n");
    ptx.push_str("    .reg .b64 %tmp;\n");
    ptx.push_str("    .reg .u64 %ab_u64;\n");
    ptx.push_str("    .reg .u32 %a_base_u32, %b_base_u32, %smem_addr_u32;\n");
    // Scratch regs used by the helpers (per their internal convention).
    ptx.push_str("    .reg .u32 %mma_addr, %mma_a_row, %mma_b_row;\n");
    // A-frag (4 b32), B-frag (2 b32), D accumulator (4 f32), C = 0 (4 f32).
    ptx.push_str("    .reg .b32 %ra0, %ra1, %ra2, %ra3;\n");
    ptx.push_str("    .reg .b32 %rb0, %rb1;\n");
    ptx.push_str("    .reg .f32 %rd0, %rd1, %rd2, %rd3;\n");
    ptx.push_str("    .reg .f32 %rc0, %rc1, %rc2, %rc3;\n");

    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    and.b32 %lane, %tid_x, 31;\n");

    // -- Copy A (512 bytes) and B (256 bytes) into SMEM. 32 lanes × 24
    //    bytes = 768 bytes; each lane handles 24 bytes (= 12 f16).
    //    Simpler: stage in two passes — A (16 bytes/lane × 32 = 512) then B (8 bytes/lane × 32 = 256).
    ptx.push_str("    cvta.shared.u64 %ab_u64, ab_smem;\n");
    ptx.push_str("    cvt.u32.u64 %a_base_u32, %ab_u64;\n");
    ptx.push_str(&format!(
        "    add.u32 %b_base_u32, %a_base_u32, {};\n",
        A_BYTES
    ));

    // A copy: lane t handles bytes [t*16, t*16+16) of A.
    ptx.push_str("    ld.param.u64 %rd_ptr, [a_ptr];\n");
    ptx.push_str("    cvt.u64.u32 %rd_off, %lane;\n");
    ptx.push_str("    shl.b64 %rd_off, %rd_off, 4;\n"); // lane * 16
    ptx.push_str("    add.u64 %rd_off, %rd_ptr, %rd_off;\n");
    ptx.push_str("    ld.global.b64 %tmp, [%rd_off];\n");
    ptx.push_str("    shl.b32 %ofs, %lane, 4;\n");
    ptx.push_str("    add.u32 %smem_addr_u32, %a_base_u32, %ofs;\n");
    ptx.push_str("    st.shared.b64 [%smem_addr_u32], %tmp;\n");
    ptx.push_str("    ld.global.b64 %tmp, [%rd_off + 8];\n");
    ptx.push_str("    st.shared.b64 [%smem_addr_u32 + 8], %tmp;\n");

    // B copy: lane t handles bytes [t*8, t*8+8) of B.
    ptx.push_str("    ld.param.u64 %rd_ptr, [b_ptr];\n");
    ptx.push_str("    cvt.u64.u32 %rd_off, %lane;\n");
    ptx.push_str("    shl.b64 %rd_off, %rd_off, 3;\n"); // lane * 8
    ptx.push_str("    add.u64 %rd_off, %rd_ptr, %rd_off;\n");
    ptx.push_str("    ld.global.b64 %tmp, [%rd_off];\n");
    ptx.push_str("    shl.b32 %ofs, %lane, 3;\n");
    ptx.push_str("    add.u32 %smem_addr_u32, %b_base_u32, %ofs;\n");
    ptx.push_str("    st.shared.b64 [%smem_addr_u32], %tmp;\n");

    ptx.push_str("    bar.sync 0;\n");

    // -- Load A-fragment + B-fragment via the helpers.
    // A is row-major [M=16, K=16] f16 → row stride = K * 2 = 32 bytes.
    let a_regs = [
        "ra0".to_string(),
        "ra1".to_string(),
        "ra2".to_string(),
        "ra3".to_string(),
    ];
    emit_load_a_fragment_smem(&mut ptx, &a_regs, "%a_base_u32", 32);
    // B is col-major [K=16, N=8] f16 → col stride = K * 2 = 32 bytes.
    let b_regs = ["rb0".to_string(), "rb1".to_string()];
    emit_load_b_fragment_smem(&mut ptx, &b_regs, "%b_base_u32", 32);

    // -- C = 0.
    ptx.push_str("    mov.f32 %rc0, 0f00000000;\n");
    ptx.push_str("    mov.f32 %rc1, 0f00000000;\n");
    ptx.push_str("    mov.f32 %rc2, 0f00000000;\n");
    ptx.push_str("    mov.f32 %rc3, 0f00000000;\n");

    // -- mma.sync: D = A * B + C.
    let d_regs = [
        "%rd0".to_string(),
        "%rd1".to_string(),
        "%rd2".to_string(),
        "%rd3".to_string(),
    ];
    let a_regs_pct: [String; 4] = std::array::from_fn(|i| format!("%{}", a_regs[i]));
    let b_regs_pct: [String; 2] = std::array::from_fn(|i| format!("%{}", b_regs[i]));
    let c_regs = [
        "%rc0".to_string(),
        "%rc1".to_string(),
        "%rc2".to_string(),
        "%rc3".to_string(),
    ];
    emit_mma_instruction(&mut ptx, &d_regs, &a_regs_pct, &b_regs_pct, &c_regs);

    // -- Write per-lane D (4 f32 = 16 bytes) to HBM at lane * 16.
    ptx.push_str("    ld.param.u64 %rd_ptr, [d_ptr];\n");
    ptx.push_str("    cvt.u64.u32 %rd_off, %lane;\n");
    ptx.push_str("    shl.b64 %rd_off, %rd_off, 4;\n");
    ptx.push_str("    add.u64 %rd_ptr, %rd_ptr, %rd_off;\n");
    ptx.push_str("    st.global.f32 [%rd_ptr], %rd0;\n");
    ptx.push_str("    st.global.f32 [%rd_ptr + 4], %rd1;\n");
    ptx.push_str("    st.global.f32 [%rd_ptr + 8], %rd2;\n");
    ptx.push_str("    st.global.f32 [%rd_ptr + 12], %rd3;\n");

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    let mut bytes = ptx.into_bytes();
    if bytes.last() != Some(&b'\n') {
        bytes.push(b'\n');
    }
    bytes.push(0);
    bytes
}

#[test]
#[ignore = "requires CUDA GPU"]
fn single_mma_matches_cpu_reference() {
    if !cuda_available() {
        return;
    }

    // -- Build A row-major [16, 16] f16: A[r, c] = (r * 100 + c) / 1024.
    let mut a_host = vec![0u16; (M * K) as usize];
    let mut a_f32 = vec![0f32; (M * K) as usize];
    for r in 0..M {
        for c in 0..K {
            let v = (r as f32 * 100.0 + c as f32) / 1024.0;
            a_f32[(r * K + c) as usize] = v;
            a_host[(r * K + c) as usize] = f32_to_f16_bits(v);
        }
    }
    // -- Build B col-major [16, 8] f16: B[k, n] = (k * 100 + n) / 4096.
    //    Col-major byte: (n * K + k) * 2.
    let mut b_host = vec![0u16; (K * N) as usize];
    let mut b_f32 = vec![0f32; (K * N) as usize]; // logical [k, n]
    for k in 0..K {
        for n in 0..N {
            let v = (k as f32 * 100.0 + n as f32) / 4096.0;
            b_f32[(k * N + n) as usize] = v;
            // col-major idx = n * K + k
            b_host[(n * K + k) as usize] = f32_to_f16_bits(v);
        }
    }

    // -- CPU reference D = A @ B (rounded f16 inputs read back).
    // Read back f16-rounded A, B values to match what the GPU sees.
    let a_rounded: Vec<f32> = a_host.iter().map(|&b| f16_to_f32(b)).collect();
    let mut b_rounded = vec![0f32; (K * N) as usize];
    for k in 0..K {
        for n in 0..N {
            let col_major_idx = (n * K + k) as usize;
            b_rounded[(k * N + n) as usize] = f16_to_f32(b_host[col_major_idx]);
        }
    }
    let _ = (a_f32, b_f32); // mark unused (original f32 unrounded values not needed)

    let mut cpu_d = vec![0f32; (M * N) as usize];
    for r in 0..M {
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += a_rounded[(r * K + k) as usize] * b_rounded[(k * N + n) as usize];
            }
            cpu_d[(r * N + n) as usize] = acc;
        }
    }

    // -- Device allocations + H2D.
    let a_dev = unsafe { nsl_test_cuda_alloc(A_BYTES as i64) };
    let b_dev = unsafe { nsl_test_cuda_alloc(B_BYTES as i64) };
    let d_dev = unsafe { nsl_test_cuda_alloc(D_F32_BYTES as i64) };
    assert!(a_dev != 0 && b_dev != 0 && d_dev != 0, "alloc failed");

    unsafe {
        nsl_test_cuda_h2d(a_dev, a_host.as_ptr() as i64, A_BYTES as i64);
        nsl_test_cuda_h2d(b_dev, b_host.as_ptr() as i64, B_BYTES as i64);
    }

    // -- Build + launch PTX kernel.
    let ptx = build_probe_ptx();
    let dump = std::env::temp_dir().join("matmul_mma_single_mma_probe.ptx");
    std::fs::write(&dump, &ptx[..ptx.len() - 1]).ok();
    eprintln!("[mma-probe] PTX dumped to: {}", dump.display());

    let kernel_name = CString::new("matmul_mma_single_mma_probe").unwrap();
    let mut a_p = a_dev as u64;
    let mut b_p = b_dev as u64;
    let mut d_p = d_dev as u64;
    let args: [*mut c_void; 3] = [
        &mut a_p as *mut _ as *mut c_void,
        &mut b_p as *mut _ as *mut c_void,
        &mut d_p as *mut _ as *mut c_void,
    ];

    let rc = unsafe {
        nsl_kernel_launch(
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            1, 1, 1,
            32, 1, 1,
            args.as_ptr() as i64,
            args.len() as i64,
            0,
        )
    };

    if rc != 0 {
        let log_ptr = unsafe { nsl_test_cuda_jit_log(ptx.as_ptr() as i64) };
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no log>".into()
        };
        unsafe {
            nsl_test_cuda_free(a_dev);
            nsl_test_cuda_free(b_dev);
            nsl_test_cuda_free(d_dev);
        }
        panic!("mma probe launch failed rc={}\nJIT log:\n{}", rc, log);
    }

    // -- Readback per-lane D (32 × 4 f32) and gather into [M, N] matrix.
    let mut lane_d = vec![0f32; (32 * PER_LANE_D_F32) as usize];
    unsafe {
        nsl_test_cuda_d2h(lane_d.as_mut_ptr() as i64, d_dev, D_F32_BYTES as i64);
        nsl_test_cuda_free(a_dev);
        nsl_test_cuda_free(b_dev);
        nsl_test_cuda_free(d_dev);
    }
    // PTX m16n8k16 D-fragment lane mapping:
    //   reg 0: D[t/4,     (t%4)*2    ]
    //   reg 1: D[t/4,     (t%4)*2 + 1]
    //   reg 2: D[t/4 + 8, (t%4)*2    ]
    //   reg 3: D[t/4 + 8, (t%4)*2 + 1]
    let mut gpu_d = vec![f32::NAN; (M * N) as usize];
    for t in 0..32u32 {
        let base = (t * 4) as usize;
        let row_lo = t / 4;
        let row_hi = row_lo + 8;
        let col_lo = (t % 4) * 2;
        let col_hi = col_lo + 1;
        gpu_d[(row_lo * N + col_lo) as usize] = lane_d[base];
        gpu_d[(row_lo * N + col_hi) as usize] = lane_d[base + 1];
        gpu_d[(row_hi * N + col_lo) as usize] = lane_d[base + 2];
        gpu_d[(row_hi * N + col_hi) as usize] = lane_d[base + 3];
    }

    // -- Compare.
    let mut max_abs = 0f32;
    let mut max_row = 0u32;
    let mut max_col = 0u32;
    for r in 0..M {
        for n in 0..N {
            let g = gpu_d[(r * N + n) as usize];
            let c = cpu_d[(r * N + n) as usize];
            let diff = (g - c).abs();
            if diff > max_abs {
                max_abs = diff;
                max_row = r;
                max_col = n;
            }
        }
    }
    eprintln!("\n[mma-probe] ── per-row max-abs diff ──");
    for r in 0..M {
        let row_max = (0..N)
            .map(|n| {
                let g = gpu_d[(r * N + n) as usize];
                let c = cpu_d[(r * N + n) as usize];
                (g - c).abs()
            })
            .fold(0f32, f32::max);
        eprintln!("[mma-probe]   row {:2}: max_abs_diff = {:.4e}", r, row_max);
    }
    eprintln!(
        "\n[mma-probe] overall max_abs_diff = {:.4e} at (row={}, col={})",
        max_abs, max_row, max_col
    );
    eprintln!(
        "[mma-probe]   gpu = {:.6}   cpu = {:.6}",
        gpu_d[(max_row * N + max_col) as usize],
        cpu_d[(max_row * N + max_col) as usize]
    );

    // Tolerance: f16 inputs (~10^-3 mantissa precision) × K=16 fma chain
    // → expect <~ 1e-3 cumulative. Allow 5e-3 for slack.
    assert!(
        max_abs < 5e-3,
        "[mma-probe] max_abs_diff = {:.4e} exceeds 5e-3 tolerance — \
         the matmul_mma helper stack is producing wrong results",
        max_abs
    );
    eprintln!("[mma-probe] ✓ PASS: helper stack matches CPU reference within f16 tolerance");
}
