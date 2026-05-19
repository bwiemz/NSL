//! N4 focused probe: isolate `matmul_mma::emit_load_b_fragment_smem`'s
//! per-lane coverage from any integration chain. Companion to
//! `tier_b1_n4_helper_probe.rs` (which already classified the A-frag
//! helper). Together they prove the helpers' lane→(row, col) coverage
//! against the NVIDIA PTX ISA section 9.7.13.4 spec.
//!
//! ## What the B-fragment is supposed to load
//!
//! For `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, the B
//! operand is conceptually `[k=16, n=8]` f16 col-major. Each lane t
//! holds 2 b32 = 4 f16 values:
//!
//! ```text
//!   reg 0: B[(t%4)*2,     t/4]   (2 f16 packed: rows (t%4)*2, (t%4)*2+1)
//!   reg 1: B[(t%4)*2 + 8, t/4]   (2 f16 packed: rows (t%4)*2+8, (t%4)*2+9)
//! ```
//!
//! In col-major SMEM, `B[k, n]` lives at byte offset `(n*16 + k) * 2`.
//! With col_stride_bytes = `k_dim * 2 = 32`, each lane reads from:
//!
//! ```text
//!   reg 0 byte: smem_base + (t/4) * 32 + (t%4) * 4
//!   reg 1 byte: smem_base + (t/4) * 32 + (t%4) * 4 + 16
//! ```
//!
//! ## What this probe does
//!
//! Stages a known col-major 16×8 f16 tile into SMEM where each cell
//! encodes `(k_row, n_col) → k*100 + n` (k 0..15, n 0..7, so values
//! 0..1507 — all exactly representable in f16). Single warp (32 lanes)
//! calls `emit_load_b_fragment_smem` once with `row_stride_bytes = 32`
//! (col stride for k=16). Each lane writes its 2 b32 (4 f16) to HBM
//! at `lane * 8 bytes`. Host decodes back to (k_row, n_col) coords and
//! classifies vs:
//!
//!   (a) **PTX m16n8k16 spec** — per-lane footprint as above.
//!   (b) **Helper's emitted code** — depends on caller's `%mma_b_row`
//!       setup, but the helper itself reads 2 b32 from
//!       `[smem + mma_b_row*stride + {0, 16}]`. With Tier B.1's
//!       `%mma_b_row = %mma_a_row = (lane%4)*2 + lane/16`, every
//!       lane reads from a single row (the per-lane `mma_b_row`) at
//!       fixed col byte offsets {0, 16} (k positions {0, 8}).
//!
//! ## Running
//!
//! ```bash
//! cargo test --package nsl-codegen --features cuda \
//!     --test tier_b1_n4_b_helper_probe -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_codegen::matmul_mma::emit_load_b_fragment_smem;
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

const K_ROWS: u32 = 16;
const N_COLS: u32 = 8;
const TILE_BYTES: usize = (K_ROWS * N_COLS * 2) as usize; // 256
const PER_LANE_OUT_F16: usize = 4; // 2 b32 = 4 f16

fn encode_pos(k: u32, n: u32) -> u16 {
    f32_to_f16_bits((k * 100 + n) as f32)
}

fn decode_pos(bits: u16) -> Option<(u32, u32)> {
    let v = f16_to_f32(bits);
    if !v.is_finite() {
        return None;
    }
    let val = v.round() as i64;
    if !(0..=1600).contains(&val) {
        return None;
    }
    Some(((val / 100) as u32, (val % 100) as u32))
}

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
        return false;
    }
    true
}

/// Build the probe kernel PTX. Single warp; col-major 16×8 SMEM tile.
fn build_probe_ptx() -> Vec<u8> {
    let mut ptx = String::new();
    ptx.push_str(".version 8.7\n");
    ptx.push_str(".target sm_80\n");
    ptx.push_str(".address_size 64\n\n");
    ptx.push_str(".visible .entry n4_b_helper_probe (\n");
    ptx.push_str("    .param .u64 input_ptr,\n");
    ptx.push_str("    .param .u64 output_ptr\n");
    ptx.push_str(")\n{\n");
    ptx.push_str(&format!(
        "    .shared .align 16 .b8 tile_smem[{}];\n",
        TILE_BYTES
    ));
    ptx.push_str("    .reg .u32 %tid_x, %lane, %ofs;\n");
    ptx.push_str("    .reg .u64 %rd_in, %rd_out, %rd_addr;\n");
    ptx.push_str("    .reg .b64 %tmp0;\n");
    ptx.push_str("    .reg .u64 %tile_base_u64;\n");
    ptx.push_str("    .reg .u32 %tile_base_u32, %smem_addr_u32;\n");
    // Registers the B-frag helper uses as scratch + lane-row reg.
    ptx.push_str("    .reg .u32 %mma_addr, %mma_b_row;\n");
    ptx.push_str("    .reg .b32 %rb0, %rb1;\n");

    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    and.b32 %lane, %tid_x, 31;\n");

    // Copy 256 bytes from input → SMEM. 32 threads × 8 bytes each.
    ptx.push_str("    ld.param.u64 %rd_in, [input_ptr];\n");
    ptx.push_str("    cvt.u64.u32 %rd_addr, %lane;\n");
    ptx.push_str("    shl.b64 %rd_addr, %rd_addr, 3;\n"); // lane * 8 bytes
    ptx.push_str("    add.u64 %rd_addr, %rd_in, %rd_addr;\n");
    ptx.push_str("    ld.global.b64 %tmp0, [%rd_addr];\n");

    ptx.push_str("    cvta.shared.u64 %tile_base_u64, tile_smem;\n");
    ptx.push_str("    cvt.u32.u64 %tile_base_u32, %tile_base_u64;\n");
    ptx.push_str("    shl.b32 %ofs, %lane, 3;\n");
    ptx.push_str("    add.u32 %smem_addr_u32, %tile_base_u32, %ofs;\n");
    ptx.push_str("    st.shared.b64 [%smem_addr_u32], %tmp0;\n");

    ptx.push_str("    bar.sync 0;\n");

    // Caller is expected to pre-set %mma_b_row before calling the
    // helper. We mirror Tier B.1's setup convention (the buggy one)
    // since that's the actual production call path we want to probe.
    // %mma_b_row = (lane % 4) * 2 + lane / 16
    ptx.push_str("    and.b32 %mma_b_row, %lane, 3;\n");
    ptx.push_str("    shl.b32 %mma_b_row, %mma_b_row, 1;\n");
    ptx.push_str("    shr.u32 %mma_addr, %lane, 4;\n");
    ptx.push_str("    add.u32 %mma_b_row, %mma_b_row, %mma_addr;\n");

    // Call the helper under test. col_stride_bytes = 32 = k_dim * 2
    // (col stride for col-major [k=16, n=8] f16).
    let regs = ["rb0".to_string(), "rb1".to_string()];
    emit_load_b_fragment_smem(&mut ptx, &regs, "%tile_base_u32", 32);

    // Write 2 b32 per lane to HBM at lane * 8 bytes.
    ptx.push_str("    ld.param.u64 %rd_out, [output_ptr];\n");
    ptx.push_str("    cvt.u64.u32 %rd_addr, %lane;\n");
    ptx.push_str("    shl.b64 %rd_addr, %rd_addr, 3;\n"); // lane * 8 (2 b32 = 8 bytes)
    ptx.push_str("    add.u64 %rd_out, %rd_out, %rd_addr;\n");
    ptx.push_str("    st.global.b32 [%rd_out], %rb0;\n");
    ptx.push_str("    st.global.b32 [%rd_out + 4], %rb1;\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    let mut bytes = ptx.into_bytes();
    if bytes.last() != Some(&b'\n') {
        bytes.push(b'\n');
    }
    bytes.push(0);
    bytes
}

#[derive(Debug, Clone)]
struct LaneFootprint {
    /// 4 entries: (reg_idx, half_idx, k_row, n_col)
    cells: Vec<(u32, u32, u32, u32)>,
}

/// PTX m16n8k16 B-frag spec for lane t.
/// reg 0: cols=t/4, rows=(t%4)*2 and (t%4)*2+1 (2 f16 packed)
/// reg 1: cols=t/4, rows=(t%4)*2+8 and (t%4)*2+9
fn ptx_spec_footprint(t: u32) -> LaneFootprint {
    let k0 = (t % 4) * 2;
    let k1 = (t % 4) * 2 + 8;
    let n = t / 4;
    LaneFootprint {
        cells: vec![
            (0, 0, k0, n),
            (0, 1, k0 + 1, n),
            (1, 0, k1, n),
            (1, 1, k1 + 1, n),
        ],
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn probe_emit_load_b_fragment_smem_layout() {
    if !cuda_available() {
        return;
    }

    // Build input tile: col-major B[k, n] at byte (n*16 + k)*2.
    let mut tile_host = vec![0u16; (K_ROWS * N_COLS) as usize];
    for n in 0..N_COLS {
        for k in 0..K_ROWS {
            let idx = (n * K_ROWS + k) as usize; // col-major
            tile_host[idx] = encode_pos(k, n);
        }
    }

    let input_bytes = TILE_BYTES as i64;
    let out_bytes = (32 * PER_LANE_OUT_F16 * 2) as i64;

    let input_dev = unsafe { nsl_test_cuda_alloc(input_bytes) };
    let output_dev = unsafe { nsl_test_cuda_alloc(out_bytes) };
    assert!(input_dev != 0 && output_dev != 0);

    unsafe {
        nsl_test_cuda_h2d(input_dev, tile_host.as_ptr() as i64, input_bytes);
    }

    let ptx = build_probe_ptx();
    let dump = std::env::temp_dir().join("n4_b_helper_probe.ptx");
    std::fs::write(&dump, &ptx[..ptx.len() - 1]).ok();
    eprintln!("[N4-b-probe] PTX dumped to: {}", dump.display());

    let kernel_name = CString::new("n4_b_helper_probe").unwrap();
    let mut in_ptr = input_dev as u64;
    let mut out_ptr = output_dev as u64;
    let args: [*mut c_void; 2] = [
        &mut in_ptr as *mut _ as *mut c_void,
        &mut out_ptr as *mut _ as *mut c_void,
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
            nsl_test_cuda_free(input_dev);
            nsl_test_cuda_free(output_dev);
        }
        panic!("B-probe launch failed rc={}\nJIT log:\n{}", rc, log);
    }

    let mut out_host = vec![0u16; (32 * PER_LANE_OUT_F16) as usize];
    unsafe {
        nsl_test_cuda_d2h(out_host.as_mut_ptr() as i64, output_dev, out_bytes);
        nsl_test_cuda_free(input_dev);
        nsl_test_cuda_free(output_dev);
    }

    let mut observed: Vec<LaneFootprint> = Vec::with_capacity(32);
    for lane in 0u32..32 {
        let base = (lane as usize) * PER_LANE_OUT_F16;
        let mut cells = Vec::with_capacity(4);
        for reg in 0u32..2 {
            let lo = out_host[base + (reg as usize) * 2];
            let hi = out_host[base + (reg as usize) * 2 + 1];
            for (half, &bits) in [lo, hi].iter().enumerate() {
                match decode_pos(bits) {
                    Some((k, n)) => cells.push((reg, half as u32, k, n)),
                    None => cells.push((reg, half as u32, u32::MAX, u32::MAX)),
                }
            }
        }
        observed.push(LaneFootprint { cells });
    }

    eprintln!("\n[N4-b-probe] ── per-lane B-frag footprint ──");
    eprintln!("[N4-b-probe] lane | reg.half → (k_row, n_col)");
    let mut matches_spec = 0usize;
    let mut undecodable = 0usize;
    for lane in 0u32..32 {
        let obs = &observed[lane as usize];
        let spec = ptx_spec_footprint(lane);

        let undec = obs
            .cells
            .iter()
            .filter(|&&(_, _, r, _)| r == u32::MAX)
            .count();
        undecodable += undec;

        let obs_set: std::collections::BTreeSet<(u32, u32)> = obs
            .cells
            .iter()
            .filter(|&&(_, _, r, _)| r != u32::MAX)
            .map(|&(_, _, r, c)| (r, c))
            .collect();
        let spec_set: std::collections::BTreeSet<(u32, u32)> =
            spec.cells.iter().map(|&(_, _, r, c)| (r, c)).collect();

        let spec_match = obs_set == spec_set;
        if spec_match {
            matches_spec += 1;
        }

        let cells_str: Vec<String> = obs
            .cells
            .iter()
            .map(|&(reg, half, k, n)| {
                if k == u32::MAX {
                    format!("{}.{}:?", reg, half)
                } else {
                    format!("{}.{}:({},{})", reg, half, k, n)
                }
            })
            .collect();
        eprintln!(
            "[N4-b-probe] {:2} | {}{}",
            lane,
            cells_str.join("  "),
            if spec_match { "  [✓SPEC]" } else { "" }
        );
    }

    eprintln!("\n[N4-b-probe] ── summary ──");
    eprintln!("[N4-b-probe] lanes matching PTX m16n8k16 spec : {}/32", matches_spec);
    eprintln!("[N4-b-probe] undecodable cells               : {}", undecodable);

    if matches_spec == 32 {
        eprintln!("[N4-b-probe] DIAGNOSIS: B-frag helper is PTX-spec CORRECT.");
    } else if matches_spec == 0 {
        eprintln!(
            "[N4-b-probe] DIAGNOSIS: B-frag helper does NOT match PTX spec. \
             Inspect per-lane table to determine the actual layout."
        );
    } else {
        eprintln!(
            "[N4-b-probe] DIAGNOSIS: PARTIAL — {}/32 lanes match spec. \
             Inspect per-lane table to determine the actual layout.",
            matches_spec
        );
    }
}
