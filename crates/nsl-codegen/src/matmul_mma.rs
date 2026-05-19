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

/// Emit PTX to load an m16n8k16 B-fragment (col-major f16 16×8) from SMEM,
/// matching the NVIDIA PTX ISA §9.7.13.4 lane mapping for
/// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
///
/// Each warp lane (32 lanes per warp) holds 2 `.b32` registers = 4 packed
/// f16 values laid out at a single col × 2 k-row pair positions:
///
/// ```text
///   reg 0: B[k_lo,     n] .. B[k_lo + 1, n]   (2 f16 packed, contiguous in col-major)
///   reg 1: B[k_lo + 8, n] .. B[k_lo + 9, n]
///
///   where  k_lo = (lane % 4) * 2
///          n    = lane / 4
/// ```
///
/// The two f16 values per `.b32` are at adjacent K rows of the SAME
/// column — which IS contiguous in col-major storage (one b32 read =
/// 4 bytes = 2 consecutive f16 in the K direction).
///
/// **SMEM layout assumption**: B must be stored col-major in SMEM —
/// i.e., consecutive bytes step through the K dimension first, then N.
/// In row-major storage of an `[K, N]` tile, two K-adjacent values
/// live `N * 2` bytes apart (NOT contiguous), so the b32 read would
/// fetch B[k, n] and B[k, n+1] (wrong fragment). Callers staging
/// row-major-from-HBM weight tiles into SMEM must transpose during
/// the cp.async path, or refactor the SMEM layout. WRGA already
/// stores its B tiles col-major; Tier B.1's `Wq`/`Wk`/`Wv` tiles
/// currently stage row-major from HBM and need a separate fix to
/// realize correct B-frag loads.
///
/// `row_stride_bytes` is the BYTE stride between adjacent COLUMNS in
/// col-major storage (= `k_dim * 2` for tightly packed f16; typically
/// 32 bytes for k=16). The parameter name is preserved from the old
/// signature for source compatibility, but semantically it's a column
/// stride.
///
/// `smem_base_expr` is a u32 PTX expression evaluating to the byte
/// address of B[0, 0] in SMEM.
///
/// `load_transposed` controls the SMEM-to-fragment mapping:
/// - `false`: col-major SMEM → col-major B-frag presentation (B.1 forward S=QK^T pattern).
///   Second .b32 reads from the same column at byte offset +16 (+8 k-rows in col-major).
/// - `true`: col-major SMEM → row-major B-frag presentation (Phase 2 dQ-update pattern).
///   Second .b32 reads from a DIFFERENT row (row + 8), presenting col-major SMEM
///   transposed to the MMA. Used when the same SMEM tile needs to feed both row-col
///   MMAs (S, dP) and row-row-effective MMAs (dQ_acc += dS @ K).
///
/// As with `emit_load_a_fragment_smem`, the helper computes the
/// per-lane mapping internally from `%tid.x` and uses `%mma_addr` +
/// `%mma_b_row` as scratch — no caller setup of `%mma_b_row` is
/// required (whatever the caller stored in it before the call is
/// overwritten).
///
/// ### Lane-mapping bug history
///
/// Prior to this commit, the helper read 2 b32 from
/// `[smem_base + %mma_b_row * stride + {0, 16}]`. Whether this matched
/// PTX spec depended ENTIRELY on the caller's `%mma_b_row` setup AND
/// whether the caller had pre-baked the per-lane column offset into
/// `smem_base_expr`. WRGA happened to do both correctly (so the
/// effective addressing matched spec); Tier B.1 did neither, so its
/// B-frag loads put 31/32 lanes at wrong (k, n) positions. The probe
/// `tier_b1_n4_b_helper_probe.rs` confirmed 1/32 lanes matched spec
/// before this rewrite. Making the helper self-contained eliminates
/// the dependency on caller convention.
pub fn emit_load_b_fragment_smem(
    ptx: &mut String,
    frag_regs: &[String; 2],
    smem_base_expr: &str,
    row_stride_bytes: usize,
    load_transposed: bool,
) {
    if load_transposed {
        ptx.push_str("    // Load B-fragment (transposed: col-major SMEM -> row-major B-frag)\n");
        ptx.push_str("    // Lane t: reg0=B[row, 0], reg1=B[row+8, 0] where row=(t%4)*2, same-col-0\n");
        // Self-contained lane derivation from %tid.x.
        ptx.push_str("    mov.u32 %mma_addr, %tid.x;              // load tid\n");
        ptx.push_str("    and.b32 %mma_addr, %mma_addr, 31;       // lane = tid & 31\n");
        // k_row_lo = (lane % 4) * 2 — same k-row formula as the non-transposed variant.
        ptx.push_str("    and.b32 %mma_b_row, %mma_addr, 3;       // lane % 4\n");
        ptx.push_str("    shl.b32 %mma_b_row, %mma_b_row, 1;      // k_row_lo = (lane%4)*2\n");
        // reg 0: B[k_row_lo, col=0] — byte address = k_row_lo * row_stride_bytes + smem base.
        // IMPORTANT: Use mul.lo.u32 + add.u32, NOT mad.lo.u32 (ISA 7.0 runtime rejection).
        ptx.push_str(&format!(
            "    mul.lo.u32 %mma_addr, %mma_b_row, {};  // k_row_lo * stride\n",
            row_stride_bytes
        ));
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // + smem base\n",
            smem_base_expr
        ));
        ptx.push_str(&format!(
            "    ld.shared.b32 %{}, [%mma_addr];           // reg0 @ B[k_row_lo, col=0]\n",
            frag_regs[0]
        ));
        // reg 1: B[k_row_lo + 8, col=0] — +8 rows = +8 * row_stride_bytes bytes.
        let row8_bytes = 8 * row_stride_bytes;
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // + 8 rows\n",
            row8_bytes
        ));
        ptx.push_str(&format!(
            "    ld.shared.b32 %{}, [%mma_addr];           // reg1 @ B[k_row_lo+8, col=0]\n",
            frag_regs[1]
        ));
    } else {
        ptx.push_str("    // Load B-fragment (k16xn8 col-major f16) per PTX m16n8k16 spec\n");
        ptx.push_str("    // Lane t holds: reg0=B[(t%4)*2 .. +1, t/4], reg1=B[(t%4)*2+8 .. +9, t/4]\n");
        // Self-contained lane derivation from %tid.x.
        ptx.push_str("    mov.u32 %mma_addr, %tid.x;              // load tid\n");
        ptx.push_str("    and.b32 %mma_addr, %mma_addr, 31;       // lane = tid & 31\n");
        // %mma_b_row = (lane / 4) * row_stride_bytes  (byte offset of column n_col).
        ptx.push_str("    shr.u32 %mma_b_row, %mma_addr, 2;       // n_col = lane / 4\n");
        ptx.push_str(&format!(
            "    mul.lo.u32 %mma_b_row, %mma_b_row, {};  // n_col * col_stride\n",
            row_stride_bytes
        ));
        // Add k_lo bytes = (lane % 4) * 4 bytes (2 f16 packed = 4 bytes wide).
        ptx.push_str("    and.b32 %mma_addr, %mma_addr, 3;        // lane % 4\n");
        ptx.push_str("    shl.b32 %mma_addr, %mma_addr, 2;        // * 4 bytes (k_lo bytes)\n");
        ptx.push_str(
            "    add.u32 %mma_b_row, %mma_b_row, %mma_addr;  // + k_lo bytes\n",
        );
        // Add smem base.
        ptx.push_str(&format!(
            "    add.u32 %mma_b_row, %mma_b_row, {};  // + smem base\n",
            smem_base_expr
        ));
        // reg 0 = B[k_lo .. k_lo+1, n_col].
        ptx.push_str(&format!(
            "    ld.shared.b32 %{}, [%mma_b_row];           // reg0 @ B[k_lo, n_col]\n",
            frag_regs[0]
        ));
        // reg 1 = B[k_lo+8 .. k_lo+9, n_col] (+16 bytes = +8 k-rows in col-major).
        ptx.push_str(&format!(
            "    ld.shared.b32 %{}, [%mma_b_row + 16];      // reg1 @ B[k_lo+8, n_col]\n",
            frag_regs[1]
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
        emit_load_b_fragment_smem(&mut ptx, &regs, "%smem_base_w", 8, false);  // ADDED: false
        assert_eq!(ptx.matches("ld.shared.b32").count(), 2);
        assert!(ptx.contains("%rb0"));
        assert!(ptx.contains("%rb1"));
    }

    #[test]
    fn emit_load_b_fragment_default_unchanged() {
        let mut ptx_default = String::new();
        let regs = s2(&["rb0", "rb1"]);
        emit_load_b_fragment_smem(&mut ptx_default, &regs, "%smem_base_w", 8, false);
        assert_eq!(ptx_default.matches("ld.shared.b32").count(), 2);
        assert!(ptx_default.contains("%rb0"));
        assert!(ptx_default.contains("%rb1"));
        assert!(ptx_default.contains("16"), "default variant uses 16-byte offset: {ptx_default}");
    }

    #[test]
    fn emit_load_b_fragment_transposed_offsets() {
        let mut ptx_trans = String::new();
        let regs = s2(&["rb0", "rb1"]);
        emit_load_b_fragment_smem(&mut ptx_trans, &regs, "%smem_base_w", 8, true);
        assert_eq!(ptx_trans.matches("ld.shared.b32").count(), 2);
        assert!(
            ptx_trans.contains("mul.lo.u32"),
            "transposed variant emits a separate row*stride for the second load: {ptx_trans}",
        );
    }

    #[test]
    fn emit_load_b_fragment_variants_differ() {
        let mut p_false = String::new();
        let mut p_true = String::new();
        let regs = s2(&["rb0", "rb1"]);
        emit_load_b_fragment_smem(&mut p_false, &regs, "%smem", 8, false);
        emit_load_b_fragment_smem(&mut p_true, &regs, "%smem", 8, true);
        assert_ne!(p_false, p_true, "load_transposed must change emitted PTX");
    }
}
