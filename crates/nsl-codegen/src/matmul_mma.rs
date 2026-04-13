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
    ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
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

/// Emit PTX to load an m16n8k16 A-fragment (row-major f16 x16) from SMEM.
/// Each thread holds 4 .b32 registers covering 8 pairs of f16 values.
pub fn emit_load_a_fragment_smem(
    ptx: &mut String,
    frag_regs: &[String; 4],
    smem_base_expr: &str,
    row_stride_bytes: usize,
) {
    ptx.push_str("    // Load A-fragment (m16xk16 row-major) from shared memory\n");
    for (reg_idx, k_base_pair) in [(0, 0usize), (1, 4), (2, 8), (3, 12)].iter() {
        let byte_col_offset = k_base_pair * 2;
        ptx.push_str(&format!(
            "    mad.lo.u32 %mma_addr, %mma_a_row, {}, {};  // row * stride + base\n",
            row_stride_bytes, smem_base_expr
        ));
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // + k byte offset\n",
            byte_col_offset
        ));
        ptx.push_str(&format!(
            "    ld.shared.b32 %{}, [%mma_addr];\n",
            frag_regs[*reg_idx]
        ));
    }
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
        ptx.push_str(&format!(
            "    mad.lo.u32 %mma_addr, %mma_b_row, {}, {};  // row * stride + base\n",
            row_stride_bytes, smem_base_expr
        ));
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // + k byte offset\n",
            byte_col_offset
        ));
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
