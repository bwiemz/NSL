//! N4 focused probe: isolate `matmul_mma::emit_load_a_fragment_smem`'s
//! per-lane coverage from the full Tier B.1 integration chain.
//!
//! ## Why a standalone probe
//!
//! The original N4 test (`tier_b1_n4_disambiguation.rs`) wires the helper
//! into Tier B.1's full RMSNorm → projection → softmax → PV pipeline and
//! compares numerically against the CPU reference. Three latent codegen
//! bugs (SMEM under-decl + HBM ptr 64→32 truncation + f32-HBM/f16-SMEM
//! byte-stride mismatch on x_q) had to be peeled before that path could
//! produce non-garbage output — and the third bug is non-trivial enough
//! that it would gate N4 indefinitely.
//!
//! This probe bypasses every stage between the SMEM tile and the helper.
//! It stages a known 16×16 f16 pattern into SMEM (where each cell
//! encodes its own (row, col) as `row*100 + col`), runs
//! `emit_load_a_fragment_smem` once per warp, and writes each lane's 4
//! b32 registers (8 packed f16 values) back to HBM. The host decodes
//! the packed f16 values, recovers the (row, col) coordinates each lane
//! "loaded", and classifies the layout against three candidate
//! conventions:
//!
//!   (a) **PTX m16n8k16 A-frag spec** — each lane t holds (row=t/4,
//!       row=t/4+8) × (col_pair=(t%4)*2, col_pair=(t%4)*2+8).
//!   (b) **Helper's emitted code** — each lane reads 4 b32 from a
//!       SINGLE row `%mma_a_row` at byte offsets {0, 8, 16, 24}
//!       (cols {0,1, 4,5, 8,9, 12,13}). With Tier B.1's `%mma_a_row`
//!       formula = `(laneid%4)*2 + laneid/16`, lanes cover rows 0..7
//!       only — the upper 8 rows of a 16×16 tile are never touched.
//!   (c) **None of the above** — surprise.
//!
//! ## Running
//!
//! ```bash
//! cargo test --package nsl-codegen --features cuda \
//!     --test tier_b1_n4_helper_probe -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_codegen::matmul_mma::emit_load_a_fragment_smem;
use std::ffi::{c_void, CString};

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
    nsl_test_cuda_jit_log,
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

const TILE_ROWS: u32 = 16;
const TILE_COLS: u32 = 16;
const TILE_BYTES: usize = (TILE_ROWS * TILE_COLS * 2) as usize; // f16
const PER_LANE_OUT_F16: usize = 8; // 4 b32 = 8 f16

/// Encode `(row, col)` as f16. We pick small integer values so the f16
/// rounding is exact and decoding is bit-trivial: value = row*100 + col,
/// then cast to f16. For row,col < 16, max value = 15*100+15 = 1515,
/// representable exactly in f16 (mantissa 11 bits).
fn encode_pos(row: u32, col: u32) -> u16 {
    f32_to_f16_bits((row * 100 + col) as f32)
}

fn decode_pos(bits: u16) -> Option<(u32, u32)> {
    let v = f16_to_f32(bits);
    if !v.is_finite() {
        return None;
    }
    let n = v.round() as i64;
    if !(0..=1600).contains(&n) {
        return None;
    }
    Some(((n / 100) as u32, (n % 100) as u32))
}

/// IEEE 754 f16 → f32.
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
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

/// Build the probe kernel PTX. 32 threads (single warp).
///
/// Kernel layout:
///   1. Each lane copies 16 bytes (8 f16 values) of input from HBM into
///      SMEM at byte offset `lane*16`. The 32 lanes cover exactly the
///      16×16 f16 tile (512 bytes total).
///   2. `bar.sync 0`.
///   3. Each lane sets up `%mma_a_row` per Tier B.1's convention:
///         `%mma_a_row = (laneid%4)*2 + laneid/16`
///      (matches `tier_b1::register_budget::declare_registers`).
///   4. Call `emit_load_a_fragment_smem` with `row_stride_bytes = 32`
///      (16 cols × 2 bytes) and `smem_base_expr = %tile_base_u32`.
///   5. Each lane writes its 4 b32 registers to HBM at `lane*16 bytes`.
fn build_probe_ptx() -> Vec<u8> {
    let mut ptx = String::new();
    ptx.push_str(".version 8.7\n");
    ptx.push_str(".target sm_80\n");
    ptx.push_str(".address_size 64\n\n");
    ptx.push_str(".visible .entry n4_helper_probe (\n");
    ptx.push_str("    .param .u64 input_ptr,\n");
    ptx.push_str("    .param .u64 output_ptr\n");
    ptx.push_str(")\n{\n");
    ptx.push_str(&format!(
        "    .shared .align 16 .b8 tile_smem[{}];\n",
        TILE_BYTES
    ));
    ptx.push_str("    .reg .u32 %tid_x, %lane, %ofs;\n");
    ptx.push_str("    .reg .u64 %rd_in, %rd_out, %rd_addr;\n");
    ptx.push_str("    .reg .b64 %tmp0, %tmp1;\n");
    ptx.push_str("    .reg .u64 %tile_base_u64;\n");
    ptx.push_str("    .reg .u32 %tile_base_u32, %smem_addr_u32;\n");
    // Registers consumed by the helper.
    ptx.push_str("    .reg .u32 %mma_addr, %mma_laneid, %mma_a_row;\n");
    // Per-lane A-fragment registers.
    ptx.push_str("    .reg .b32 %ra0, %ra1, %ra2, %ra3;\n");

    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    and.b32 %lane, %tid_x, 31;\n");

    // -- Step 1+2: copy input → SMEM ----------------------------------
    ptx.push_str("    ld.param.u64 %rd_in, [input_ptr];\n");
    // %rd_addr = input_ptr + lane * 16
    ptx.push_str("    cvt.u64.u32 %rd_addr, %lane;\n");
    ptx.push_str("    shl.b64 %rd_addr, %rd_addr, 4;\n");
    ptx.push_str("    add.u64 %rd_addr, %rd_in, %rd_addr;\n");
    ptx.push_str("    ld.global.b64 %tmp0, [%rd_addr];\n");
    ptx.push_str("    ld.global.b64 %tmp1, [%rd_addr + 8];\n");

    ptx.push_str("    cvta.shared.u64 %tile_base_u64, tile_smem;\n");
    ptx.push_str("    cvt.u32.u64 %tile_base_u32, %tile_base_u64;\n");
    ptx.push_str("    shl.b32 %ofs, %lane, 4;\n");
    ptx.push_str("    add.u32 %smem_addr_u32, %tile_base_u32, %ofs;\n");
    ptx.push_str("    st.shared.b64 [%smem_addr_u32], %tmp0;\n");
    ptx.push_str("    st.shared.b64 [%smem_addr_u32 + 8], %tmp1;\n");

    ptx.push_str("    bar.sync 0;\n");

    // -- Step 3: set up %mma_a_row (matches Tier B.1's formula) -------
    ptx.push_str("    mov.u32 %mma_laneid, %lane;\n");
    ptx.push_str("    and.b32 %mma_a_row, %mma_laneid, 3;\n");
    ptx.push_str("    shl.b32 %mma_a_row, %mma_a_row, 1;\n");
    ptx.push_str("    shr.u32 %mma_addr, %mma_laneid, 4;\n");
    ptx.push_str("    add.u32 %mma_a_row, %mma_a_row, %mma_addr;\n");

    // -- Step 4: call the helper under test ----------------------------
    // row_stride_bytes = TILE_COLS * 2 = 32 (16 cols × 2 bytes).
    let frag = [
        "ra0".to_string(),
        "ra1".to_string(),
        "ra2".to_string(),
        "ra3".to_string(),
    ];
    emit_load_a_fragment_smem(&mut ptx, &frag, "%tile_base_u32", 32);

    // -- Step 5: write 4 b32 regs per lane to HBM ---------------------
    ptx.push_str("    ld.param.u64 %rd_out, [output_ptr];\n");
    ptx.push_str("    cvt.u64.u32 %rd_addr, %lane;\n");
    ptx.push_str("    shl.b64 %rd_addr, %rd_addr, 4;\n"); // lane * 16 bytes
    ptx.push_str("    add.u64 %rd_out, %rd_out, %rd_addr;\n");
    ptx.push_str("    st.global.b32 [%rd_out], %ra0;\n");
    ptx.push_str("    st.global.b32 [%rd_out + 4], %ra1;\n");
    ptx.push_str("    st.global.b32 [%rd_out + 8], %ra2;\n");
    ptx.push_str("    st.global.b32 [%rd_out + 12], %ra3;\n");

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    let mut bytes = ptx.into_bytes();
    if bytes.last() != Some(&b'\n') {
        bytes.push(b'\n');
    }
    bytes.push(0); // null-terminate for cuModuleLoadData
    bytes
}

/// Per-lane "footprint": the (row, col) coordinates of the 8 f16 values
/// the lane held in its 4 b32 registers, plus the reg index they came
/// from (0..3) and the half (0=lo, 1=hi).
#[derive(Debug, Clone)]
struct LaneFootprint {
    /// 4 entries: (reg, half) → (row, col)
    cells: Vec<(u32, u32, u32, u32)>, // (reg_idx, half_idx, row, col)
}

/// Reference: PTX m16n8k16 A-frag layout per lane t.
///   reg 0 lo: row=t/4,     col=(t%4)*2
///   reg 0 hi: row=t/4,     col=(t%4)*2 + 1
///   reg 1 lo: row=t/4 + 8, col=(t%4)*2
///   reg 1 hi: row=t/4 + 8, col=(t%4)*2 + 1
///   reg 2 lo: row=t/4,     col=(t%4)*2 + 8
///   reg 2 hi: row=t/4,     col=(t%4)*2 + 9
///   reg 3 lo: row=t/4 + 8, col=(t%4)*2 + 8
///   reg 3 hi: row=t/4 + 8, col=(t%4)*2 + 9
fn ptx_spec_footprint(t: u32) -> LaneFootprint {
    let r0 = t / 4;
    let r1 = t / 4 + 8;
    let c0 = (t % 4) * 2;
    let c1 = c0 + 8;
    LaneFootprint {
        cells: vec![
            (0, 0, r0, c0),
            (0, 1, r0, c0 + 1),
            (1, 0, r1, c0),
            (1, 1, r1, c0 + 1),
            (2, 0, r0, c1),
            (2, 1, r0, c1 + 1),
            (3, 0, r1, c1),
            (3, 1, r1, c1 + 1),
        ],
    }
}

/// Reference: what the helper SHOULD load given:
///   row = (lane%4)*2 + lane/16
///   col_byte_offsets = {0, 8, 16, 24} → col_pair {0, 4, 8, 12}
///   each pair → lo=col, hi=col+1
fn helper_emitted_footprint(t: u32) -> LaneFootprint {
    let row = (t % 4) * 2 + t / 16;
    LaneFootprint {
        cells: vec![
            (0, 0, row, 0),
            (0, 1, row, 1),
            (1, 0, row, 4),
            (1, 1, row, 5),
            (2, 0, row, 8),
            (2, 1, row, 9),
            (3, 0, row, 12),
            (3, 1, row, 13),
        ],
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn probe_emit_load_a_fragment_smem_layout() {
    if !cuda_available() {
        return;
    }

    // -- Build the input tile: tile[row][col] = row*100 + col ----------
    let mut tile_host_u16 = vec![0u16; (TILE_ROWS * TILE_COLS) as usize];
    for row in 0..TILE_ROWS {
        for col in 0..TILE_COLS {
            tile_host_u16[(row * TILE_COLS + col) as usize] = encode_pos(row, col);
        }
    }

    // -- Device allocations + H2D --------------------------------------
    let input_bytes = TILE_BYTES as i64;
    let out_bytes = (32 * PER_LANE_OUT_F16 * 2) as i64; // 32 lanes × 8 f16 × 2 bytes

    let input_dev = unsafe { nsl_test_cuda_alloc(input_bytes) };
    let output_dev = unsafe { nsl_test_cuda_alloc(out_bytes) };
    assert!(input_dev != 0 && output_dev != 0, "device alloc failed");

    unsafe {
        nsl_test_cuda_h2d(input_dev, tile_host_u16.as_ptr() as i64, input_bytes);
    }

    // -- PTX + kernel name --------------------------------------------
    let ptx = build_probe_ptx();
    let dump = std::env::temp_dir().join("n4_helper_probe.ptx");
    std::fs::write(&dump, &ptx[..ptx.len() - 1]).ok();
    eprintln!("[N4-probe] PTX dumped to: {}", dump.display());

    let kernel_name = CString::new("n4_helper_probe").unwrap();

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
            /* grid */ 1,
            1,
            1,
            /* block */ 32,
            1,
            1, // single warp
            args.as_ptr() as i64,
            args.len() as i64,
            /* smem_dynamic */ 0,
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
        panic!("probe kernel launch failed rc={}\nJIT log:\n{}", rc, log);
    }

    // -- Readback ------------------------------------------------------
    let mut out_host = vec![0u16; (32 * PER_LANE_OUT_F16) as usize];
    unsafe {
        nsl_test_cuda_d2h(out_host.as_mut_ptr() as i64, output_dev, out_bytes);
        nsl_test_cuda_free(input_dev);
        nsl_test_cuda_free(output_dev);
    }

    // -- Decode each lane's footprint ----------------------------------
    let mut observed: Vec<LaneFootprint> = Vec::with_capacity(32);
    for lane in 0u32..32 {
        let base = (lane as usize) * PER_LANE_OUT_F16;
        let mut cells = Vec::with_capacity(8);
        for reg in 0u32..4 {
            // Each b32 = 2 f16 (little-endian: lo half = first f16).
            let lo = out_host[base + (reg as usize) * 2];
            let hi = out_host[base + (reg as usize) * 2 + 1];
            for (half, &bits) in [lo, hi].iter().enumerate() {
                match decode_pos(bits) {
                    Some((r, c)) => cells.push((reg, half as u32, r, c)),
                    None => cells.push((reg, half as u32, u32::MAX, u32::MAX)),
                }
            }
        }
        observed.push(LaneFootprint { cells });
    }

    // -- Print full table + classify -----------------------------------
    eprintln!("\n[N4-probe] ── per-lane fragment footprint ──");
    eprintln!("[N4-probe] lane | reg.half → (row, col) for each of 8 cells");
    let mut matches_spec = 0usize;
    let mut matches_helper = 0usize;
    let mut undecodable = 0usize;
    for lane in 0u32..32 {
        let obs = &observed[lane as usize];
        let spec = ptx_spec_footprint(lane);
        let emit = helper_emitted_footprint(lane);

        let undec_n = obs
            .cells
            .iter()
            .filter(|&&(_, _, r, _)| r == u32::MAX)
            .count();
        undecodable += undec_n;

        let obs_set: std::collections::BTreeSet<(u32, u32)> = obs
            .cells
            .iter()
            .filter(|&&(_, _, r, _)| r != u32::MAX)
            .map(|&(_, _, r, c)| (r, c))
            .collect();
        let spec_set: std::collections::BTreeSet<(u32, u32)> =
            spec.cells.iter().map(|&(_, _, r, c)| (r, c)).collect();
        let emit_set: std::collections::BTreeSet<(u32, u32)> =
            emit.cells.iter().map(|&(_, _, r, c)| (r, c)).collect();

        let spec_match = obs_set == spec_set;
        let emit_match = obs_set == emit_set;
        if spec_match {
            matches_spec += 1;
        }
        if emit_match {
            matches_helper += 1;
        }

        let cells_str: Vec<String> = obs
            .cells
            .iter()
            .map(|&(reg, half, r, c)| {
                if r == u32::MAX {
                    format!("{}.{}:?", reg, half)
                } else {
                    format!("{}.{}:({},{})", reg, half, r, c)
                }
            })
            .collect();
        eprintln!(
            "[N4-probe] {:2} | {}{}{}{}",
            lane,
            cells_str.join("  "),
            if spec_match { "  [✓SPEC]" } else { "" },
            if emit_match { "  [✓HELPER-EMIT]" } else { "" },
            if !spec_match && !emit_match {
                "  [⚠OTHER]"
            } else {
                ""
            }
        );
    }

    eprintln!("\n[N4-probe] ── summary ──");
    eprintln!(
        "[N4-probe] lanes matching PTX m16n8k16 spec     : {}/32",
        matches_spec
    );
    eprintln!(
        "[N4-probe] lanes matching helper-emitted layout : {}/32",
        matches_helper
    );
    eprintln!(
        "[N4-probe] undecodable cells (across all lanes) : {}",
        undecodable
    );

    // -- Diagnosis ----------------------------------------------------
    eprintln!("\n[N4-probe] ── diagnosis ──");
    if matches_spec == 32 {
        eprintln!(
            "[N4-probe] (a) HELPER IS PTX-SPEC CORRECT — every lane covers \
             the m16n8k16 A-frag layout. Output drift in the integration \
             test (`tier_b1_n4_disambiguation.rs`) is therefore due to a \
             different bug upstream (the known f32-HBM/f16-SMEM mismatch \
             on x_q, or a different MMA step)."
        );
    } else if matches_helper == 32 {
        eprintln!(
            "[N4-probe] (b) HELPER MATCHES ITS EMITTED CODE BUT NOT PTX SPEC \
             — the helper reads 4 b32 from a single row at col offsets \
             {{0,8,16,24}} (cols {{0,1, 4,5, 8,9, 12,13}}). Per PTX spec, \
             each lane should hold (row=t/4 + {{0,8}}) × (col=(t%4)*2 + \
             {{0,1, 8,9}}). The kernel's mma.sync receives operands that \
             do NOT form a valid A-fragment; the entire fragment-load \
             helper needs a rewrite to the PTX-spec convention. With \
             Tier B.1's `%mma_a_row = (laneid%4)*2 + laneid/16` formula, \
             lanes cover only rows 0..7 — the upper 8 rows of a 16×16 \
             A-frag are never touched."
        );
    } else if matches_spec + matches_helper > 0 {
        eprintln!(
            "[N4-probe] PARTIAL — {} lanes match spec, {} lanes match \
             emitted layout, {} undecodable cells. Inspect the per-lane \
             table above; the helper may be reading mixed conventions \
             across the warp.",
            matches_spec, matches_helper, undecodable
        );
    } else {
        eprintln!(
            "[N4-probe] (c) NEITHER convention matches. {} undecodable \
             cells. The helper is reading from an unexpected SMEM region \
             — likely a stride bug or %mma_a_row miscalc upstream of the \
             helper itself.",
            undecodable
        );
    }
}
