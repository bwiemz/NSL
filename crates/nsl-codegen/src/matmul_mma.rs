//! WRGA B.3 Task 1: reusable MMA primitives for arbitrary matmul.
//!
//! Copied-then-parameterized from flash_attention.rs:1323-1422.  FA itself
//! stays untouched; a later migration could unify the two paths.

/// MMA shape — B.3 ships only m16n8k16 (Ampere f16 tensor cores).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmaShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub const MMA_M16N8K16: MmaShape = MmaShape { m: 16, n: 8, k: 16 };

/// Fragment layout.  m16n8k16 always uses row-A × col-B in PTX terms,
/// but the caller can remap via SMEM stride if a different source
/// layout is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentLayout { Row, Col }

/// Accumulator dtype for MMA.  B.3 uses only F32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccDtype { F32 }

/// Emit PTX for a single m16n8k16 mma.sync.
///
/// Produces exactly one line of the form
/// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {d...}, {a...}, {b...}, {c...};`.
///
/// Register name conventions match the FA helpers (register names are
/// passed by the caller as strings — this fn doesn't own register
/// allocation).
pub fn emit_mma_instruction(
    ptx: &mut String,
    d_regs: &[String; 4],
    a_regs: &[String; 4],
    b_regs: &[String; 2],
    c_regs: &[String; 4],
) {
    emit_mma_instruction_impl(ptx, d_regs, a_regs, b_regs, c_regs, None);
}

/// Predicated variant — emits `@%<pred> mma.sync...` prefix so callers
/// can gate the instruction on a runtime predicate (e.g., warp-ownership
/// for CSHA Tier B.1). `pred` is the predicate-register name WITHOUT the
/// leading `%` (the emitter adds it).
pub fn emit_mma_instruction_predicated(
    ptx: &mut String,
    d_regs: &[String; 4],
    a_regs: &[String; 4],
    b_regs: &[String; 2],
    c_regs: &[String; 4],
    pred: &str,
) {
    emit_mma_instruction_impl(ptx, d_regs, a_regs, b_regs, c_regs, Some(pred));
}

fn emit_mma_instruction_impl(
    ptx: &mut String,
    d_regs: &[String; 4],
    a_regs: &[String; 4],
    b_regs: &[String; 2],
    c_regs: &[String; 4],
    pred: Option<&str>,
) {
    match pred {
        Some(p) => ptx.push_str(&format!(
            "    @%{} mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n",
            p
        )),
        None => ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"),
    }
    ptx.push_str(&format!(
        "        {{{}, {}, {}, {}}},\n",
        d_regs[0], d_regs[1], d_regs[2], d_regs[3]
    ));
    ptx.push_str(&format!(
        "        {{{}, {}, {}, {}}},\n",
        a_regs[0], a_regs[1], a_regs[2], a_regs[3]
    ));
    ptx.push_str(&format!("        {{{}, {}}},\n", b_regs[0], b_regs[1]));
    ptx.push_str(&format!(
        "        {{{}, {}, {}, {}}};\n",
        c_regs[0], c_regs[1], c_regs[2], c_regs[3]
    ));
}

/// Emit PTX to load an m16n8k16 A-fragment (row-major f16 16×16) from SMEM,
/// matching the NVIDIA PTX ISA §9.7.13.4 lane mapping for `mma.sync.aligned.
/// m16n8k16.row.col.f32.f16.f16.f32`.
///
/// Each warp lane (32 lanes per warp) holds 4 `.b32` registers = 8 packed f16
/// values laid out across 2 rows × 2 column-pair positions:
///
/// ```text
///   reg 0: SMEM[row_lo, col_lo .. col_lo+1]   (2 f16 packed, lo half of b32)
///   reg 1: SMEM[row_hi, col_lo .. col_lo+1]
///   reg 2: SMEM[row_lo, col_hi .. col_hi+1]
///   reg 3: SMEM[row_hi, col_hi .. col_hi+1]
///
///   where  row_lo = lane / 4
///          row_hi = row_lo + 8
///          col_lo = (lane % 4) * 2
///          col_hi = col_lo + 8
/// ```
///
/// **Important**: the helper computes the per-lane row/col INTERNALLY from
/// `%lane` (the standard skeleton's `tid.x & 31`). Callers do **not** need to
/// pre-set `%mma_a_row`; it is used here as a u32 scratch register that the
/// helper overwrites. `%mma_addr` is also used as scratch. Both registers
/// must be declared in scope (the standard `register_budget` /
/// `wrga_kernel_helpers` register pools already do this).
///
/// `smem_base_expr` is a u32 PTX expression evaluating to the byte address
/// of the top-left of the A-fragment tile in SMEM.  `row_stride_bytes` is
/// the byte distance between adjacent rows in that tile (= `tile_cols × 2`
/// for tightly-packed f16).
///
/// ### Lane-mapping bug history
///
/// Prior to commit (this commit), the helper iterated 4 col positions
/// `{0, 4, 8, 12}` (in f16 element units) from a single externally-supplied
/// `%mma_a_row`. That layout covers only 1 row × 4 col-pair positions per
/// lane (cols `{0,1, 4,5, 8,9, 12,13}` — fixed across all lanes) and never
/// touches rows 8..15 of any A-fragment. The standalone N4 probe
/// (`tier_b1_n4_helper_probe.rs`, commit e3354e26) confirmed 0/32 lanes
/// matched PTX spec, 32/32 matched the broken layout. Every Tier B.1 MMA
/// (Q/K/V proj, QK^T, PV) silently computed wrong dot products as a result.
pub fn emit_load_a_fragment_smem(
    ptx: &mut String,
    frag_regs: &[String; 4],
    smem_base_expr: &str,
    row_stride_bytes: usize,
) {
    ptx.push_str("    // Load A-fragment (m16xk16 f16 row-major) per PTX m16n8k16 spec\n");
    ptx.push_str(
        "    // Lane t holds: reg0=SMEM[t/4,(t%4)*2], reg1=SMEM[t/4+8,(t%4)*2],\n",
    );
    ptx.push_str("    //                reg2=SMEM[t/4,(t%4)*2+8], reg3=SMEM[t/4+8,(t%4)*2+8]\n");
    // Self-contained lane derivation from %tid.x (a PTX built-in special
    // register, always available). We can't rely on a caller-declared
    // `%lane`/`%mma_laneid` since v1 callers (`flash_attention.rs`),
    // Tier-A inline callers, and WRGA each use different conventions for
    // the pre-computed lane register name.
    ptx.push_str("    mov.u32 %mma_addr, %tid.x;              // load tid\n");
    ptx.push_str("    and.b32 %mma_addr, %mma_addr, 31;       // lane = tid & 31\n");
    // Step 1: %mma_a_row = (lane / 4) * row_stride_bytes (byte offset of row_lo).
    ptx.push_str("    shr.u32 %mma_a_row, %mma_addr, 2;       // row_lo = lane / 4\n");
    ptx.push_str(&format!(
        "    mul.lo.u32 %mma_a_row, %mma_a_row, {};  // row_lo * stride\n",
        row_stride_bytes
    ));
    // Step 2: add col_lo bytes = (lane % 4) * 4 bytes (2 f16 packed = 4 bytes).
    ptx.push_str("    and.b32 %mma_addr, %mma_addr, 3;        // lane % 4\n");
    ptx.push_str("    shl.b32 %mma_addr, %mma_addr, 2;        // * 4 bytes (col_lo_bytes)\n");
    ptx.push_str(
        "    add.u32 %mma_a_row, %mma_a_row, %mma_addr;  // + col_lo_bytes\n",
    );
    // Step 3: add smem base.
    ptx.push_str(&format!(
        "    add.u32 %mma_a_row, %mma_a_row, {};  // + smem base\n",
        smem_base_expr
    ));
    // Step 4: reg 0 = SMEM[row_lo, col_lo].
    ptx.push_str(&format!(
        "    ld.shared.b32 %{}, [%mma_a_row];           // reg0 @ [row_lo, col_lo]\n",
        frag_regs[0]
    ));
    // Step 5: reg 2 = SMEM[row_lo, col_lo + 8] (col offset +16 bytes).
    ptx.push_str(&format!(
        "    ld.shared.b32 %{}, [%mma_a_row + 16];      // reg2 @ [row_lo, col_lo+8]\n",
        frag_regs[2]
    ));
    // Step 6: shift to row_hi = row_lo + 8 rows.
    let row_hi_shift = 8 * row_stride_bytes;
    ptx.push_str(&format!(
        "    add.u32 %mma_a_row, %mma_a_row, {};  // shift to row_hi (+8 rows)\n",
        row_hi_shift
    ));
    // Step 7: reg 1 = SMEM[row_hi, col_lo].
    ptx.push_str(&format!(
        "    ld.shared.b32 %{}, [%mma_a_row];           // reg1 @ [row_hi, col_lo]\n",
        frag_regs[1]
    ));
    // Step 8: reg 3 = SMEM[row_hi, col_lo + 8].
    ptx.push_str(&format!(
        "    ld.shared.b32 %{}, [%mma_a_row + 16];      // reg3 @ [row_hi, col_lo+8]\n",
        frag_regs[3]
    ));
}

/// Emit PTX to load an m16n8k16 B-fragment (col-major f16 x8) from SMEM.
/// Each thread holds 2 .b32 registers covering 4 pairs of f16 values.
pub fn emit_load_b_fragment_smem(
    ptx: &mut String,
    frag_regs: &[String; 2],
    smem_base_expr: &str,
    row_stride_bytes: usize,
) {
    ptx.push_str("    // Load B-fragment (k16xn8 col-major) from shared memory\n");
    for (reg_idx, k_base_pair) in [(0, 0usize), (1, 8)].iter() {
        let byte_col_offset = k_base_pair * 2;
        // IMPORTANT: Use mul.lo.u32 + add.u32, NOT mad.lo.u32.
        // mad.lo.u32 causes CUDA_ERROR_INVALID_PTX at runtime on ISA 7.0
        // even though ptxas offline accepts it.  See M17 notes in MEMORY.md.
        ptx.push_str(&format!(
            "    mul.lo.u32 %mma_addr, %mma_b_row, {};  // row * stride\n",
            row_stride_bytes
        ));
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // + smem base\n",
            smem_base_expr
        ));
        if byte_col_offset > 0 {
            ptx.push_str(&format!(
                "    add.u32 %mma_addr, %mma_addr, {};  // + k byte offset\n",
                byte_col_offset
            ));
        }
        ptx.push_str(&format!(
            "    ld.shared.b32 %{}, [%mma_addr];\n",
            frag_regs[*reg_idx]
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(xs: &[&str; 4]) -> [String; 4] {
        [xs[0].into(), xs[1].into(), xs[2].into(), xs[3].into()]
    }
    fn s2(xs: &[&str; 2]) -> [String; 2] { [xs[0].into(), xs[1].into()] }

    #[test]
    fn emit_mma_instruction_produces_expected_shape() {
        let mut ptx = String::new();
        let d = s(&["%d0", "%d1", "%d2", "%d3"]);
        let a = s(&["%a0", "%a1", "%a2", "%a3"]);
        let b = s2(&["%b0", "%b1"]);
        let c = s(&["%c0", "%c1", "%c2", "%c3"]);
        emit_mma_instruction(&mut ptx, &d, &a, &b, &c);
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "expected MMA shape header, got: {ptx}",
        );
        assert!(ptx.contains("{%d0, %d1, %d2, %d3}"));
        assert!(ptx.contains("{%a0, %a1, %a2, %a3}"));
        assert!(ptx.contains("{%b0, %b1}"));
        assert!(ptx.contains("{%c0, %c1, %c2, %c3}"));
    }

    #[test]
    fn emit_load_a_fragment_emits_four_ld_shared() {
        let mut ptx = String::new();
        let regs = s(&["ra0", "ra1", "ra2", "ra3"]);
        emit_load_a_fragment_smem(&mut ptx, &regs, "%smem_base_x", 16);
        assert_eq!(ptx.matches("ld.shared.b32").count(), 4,
            "should emit 4 ld.shared for A-fragment; got: {ptx}");
        assert!(ptx.contains("%ra0"));
        assert!(ptx.contains("%ra3"));
    }

    #[test]
    fn emit_load_b_fragment_emits_two_ld_shared() {
        let mut ptx = String::new();
        let regs = s2(&["rb0", "rb1"]);
        emit_load_b_fragment_smem(&mut ptx, &regs, "%smem_base_w", 8);
        assert_eq!(ptx.matches("ld.shared.b32").count(), 2);
        assert!(ptx.contains("%rb0"));
        assert!(ptx.contains("%rb1"));
    }
}
