//! Which matrix element each warp lane's fragment register holds
//! (roadmap A2, step 6).
//!
//! A `mma` is warp-collective: the 16x8xK tile is spread across 32 lanes,
//! and the PTX ISA fixes exactly which `(row, col)` element lane `t`'s
//! register `r` carries. Get it wrong and the instruction still assembles —
//! it just multiplies the wrong numbers.
//!
//! That is not hypothetical. `matmul_mma.rs` carries the scars in its own
//! doc comments: a B-fragment helper whose correctness "depended ENTIRELY
//! on the caller's `%mma_b_row` setup", where a probe "confirmed 1/32 lanes
//! matched spec"; and a `load_transposed` extension reverted after
//! verification found "32 lanes all read col=0". Both were emitters
//! disagreeing with the ISA, and in both cases the disagreement was
//! invisible until someone wrote a probe.
//!
//! So the mapping lives here, once, as functions with a table of expected
//! values and — more usefully — a *coverage* property: every element of the
//! tile is held by exactly one (lane, register) pair. Neither historical
//! bug survives that check. A lane that reads the wrong column collides
//! with another lane and leaves an element unheld, and the test says which.
//!
//! Reference: NVIDIA PTX ISA §9.7.13.4.

use crate::kernel_ir::MmaShape;

/// Lanes in a warp.
pub const WARP_LANES: u32 = 32;

/// The `(row, col)` of the FIRST of the two packed f16 values that lane
/// `lane`'s A-fragment register `reg` holds. The second is `(row, col + 1)`.
///
/// ```text
///   reg 0: A[row_lo, col_lo]     row_lo = lane / 4     col_lo = (lane % 4) * 2
///   reg 1: A[row_hi, col_lo]     row_hi = row_lo + 8   col_hi = col_lo + 8
///   reg 2: A[row_lo, col_hi]
///   reg 3: A[row_hi, col_hi]
/// ```
///
/// m16n8k8 has half the K extent and so only the two `col_lo` registers.
pub fn a_fragment_element(shape: MmaShape, lane: u32, reg: usize) -> (u32, u32) {
    let row_lo = lane / 4;
    let col_lo = (lane % 4) * 2;
    match (shape, reg) {
        (_, 0) => (row_lo, col_lo),
        (_, 1) => (row_lo + 8, col_lo),
        (MmaShape::M16N8K16, 2) => (row_lo, col_lo + 8),
        (MmaShape::M16N8K16, 3) => (row_lo + 8, col_lo + 8),
        _ => panic!(
            "mma.{} has no A register {reg} (it takes {})",
            shape.ptx_shape(),
            shape.fragment_counts().0
        ),
    }
}

/// The `(k, n)` of the FIRST of the two packed f16 values that lane `lane`'s
/// B-fragment register `reg` holds. The second is `(k + 1, n)` — B is
/// col-major, so the pair is contiguous down the k axis, not across n.
///
/// ```text
///   reg 0: B[k_lo,     n]        k_lo = (lane % 4) * 2    n = lane / 4
///   reg 1: B[k_lo + 8, n]
/// ```
pub fn b_fragment_element(shape: MmaShape, lane: u32, reg: usize) -> (u32, u32) {
    let k_lo = (lane % 4) * 2;
    let n = lane / 4;
    match (shape, reg) {
        (_, 0) => (k_lo, n),
        (MmaShape::M16N8K16, 1) => (k_lo + 8, n),
        _ => panic!(
            "mma.{} has no B register {reg} (it takes {})",
            shape.ptx_shape(),
            shape.fragment_counts().1
        ),
    }
}

/// The `(row, col)` of the f32 accumulator element lane `lane`'s C/D
/// register `reg` holds. Unpacked — one f32 per register, not a pair.
///
/// ```text
///   reg 0: C[row_lo,     col]        row_lo = lane / 4
///   reg 1: C[row_lo,     col + 1]    col    = (lane % 4) * 2
///   reg 2: C[row_lo + 8, col]
///   reg 3: C[row_lo + 8, col + 1]
/// ```
///
/// The accumulator shape is m16n8 for both K extents, so this does not vary
/// with `shape`.
pub fn acc_fragment_element(lane: u32, reg: usize) -> (u32, u32) {
    let row_lo = lane / 4;
    let col = (lane % 4) * 2;
    match reg {
        0 => (row_lo, col),
        1 => (row_lo, col + 1),
        2 => (row_lo + 8, col),
        3 => (row_lo + 8, col + 1),
        _ => panic!("an m16n8 accumulator has no register {reg} (it takes 4)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The mapping stated as a table, for the lanes whose values are quoted
    /// in `matmul_mma.rs`'s doc comments. If this and those comments ever
    /// disagree, one of them is the bug.
    #[test]
    fn the_quoted_lane_values_hold() {
        // Lane 0 sits at the origin of every fragment.
        assert_eq!(a_fragment_element(MmaShape::M16N8K16, 0, 0), (0, 0));
        assert_eq!(a_fragment_element(MmaShape::M16N8K16, 0, 1), (8, 0));
        assert_eq!(a_fragment_element(MmaShape::M16N8K16, 0, 2), (0, 8));
        assert_eq!(a_fragment_element(MmaShape::M16N8K16, 0, 3), (8, 8));
        assert_eq!(b_fragment_element(MmaShape::M16N8K16, 0, 0), (0, 0));
        assert_eq!(b_fragment_element(MmaShape::M16N8K16, 0, 1), (8, 0));

        // Lane 5: row_lo = 1, col_lo = 2 / k_lo = 2, n = 1.
        assert_eq!(a_fragment_element(MmaShape::M16N8K16, 5, 0), (1, 2));
        assert_eq!(a_fragment_element(MmaShape::M16N8K16, 5, 3), (9, 10));
        assert_eq!(b_fragment_element(MmaShape::M16N8K16, 5, 0), (2, 1));
        assert_eq!(b_fragment_element(MmaShape::M16N8K16, 5, 1), (10, 1));

        // Lane 31: the far corner.
        assert_eq!(a_fragment_element(MmaShape::M16N8K16, 31, 3), (15, 14));
        assert_eq!(b_fragment_element(MmaShape::M16N8K16, 31, 1), (14, 7));
        assert_eq!(acc_fragment_element(31, 3), (15, 7));
    }

    /// The property that would have caught both historical bugs: across the
    /// warp, every element of the tile is held by EXACTLY ONE (lane,
    /// register) pair. A lane reading the wrong column collides with
    /// another and leaves an element unheld.
    fn assert_tiles_exactly<F>(rows: u32, cols: u32, regs: usize, pair_axis: PairAxis, f: F)
    where
        F: Fn(u32, usize) -> (u32, u32),
    {
        let mut holder: HashMap<(u32, u32), (u32, usize)> = HashMap::new();
        for lane in 0..WARP_LANES {
            for reg in 0..regs {
                let (r, c) = f(lane, reg);
                // Each register holds a packed PAIR along one axis.
                let cells = match pair_axis {
                    PairAxis::Col => vec![(r, c), (r, c + 1)],
                    PairAxis::Row => vec![(r, c), (r + 1, c)],
                    PairAxis::None => vec![(r, c)],
                };
                for cell in cells {
                    assert!(
                        cell.0 < rows && cell.1 < cols,
                        "lane {lane} reg {reg} addresses {cell:?}, outside the {rows}x{cols} tile"
                    );
                    if let Some(prev) = holder.insert(cell, (lane, reg)) {
                        panic!(
                            "element {cell:?} is held twice: by lane {} reg {} and lane {lane} reg {reg}",
                            prev.0, prev.1
                        );
                    }
                }
            }
        }
        let want = (rows * cols) as usize;
        assert_eq!(
            holder.len(),
            want,
            "the warp holds {} of the {want} elements; {} are unheld",
            holder.len(),
            want - holder.len()
        );
    }

    enum PairAxis {
        Row,
        Col,
        None,
    }

    #[test]
    fn the_a_fragment_tiles_m16k16_exactly() {
        // A is row-major: the packed pair runs across columns.
        assert_tiles_exactly(16, 16, 4, PairAxis::Col, |lane, reg| {
            a_fragment_element(MmaShape::M16N8K16, lane, reg)
        });
    }

    #[test]
    fn the_a_fragment_tiles_m16k8_exactly() {
        assert_tiles_exactly(16, 8, 2, PairAxis::Col, |lane, reg| {
            a_fragment_element(MmaShape::M16N8K8, lane, reg)
        });
    }

    #[test]
    fn the_b_fragment_tiles_k16n8_exactly() {
        // B is col-major: the packed pair runs down k, so the "row" axis.
        assert_tiles_exactly(16, 8, 2, PairAxis::Row, |lane, reg| {
            b_fragment_element(MmaShape::M16N8K16, lane, reg)
        });
    }

    #[test]
    fn the_b_fragment_tiles_k8n8_exactly() {
        assert_tiles_exactly(8, 8, 1, PairAxis::Row, |lane, reg| {
            b_fragment_element(MmaShape::M16N8K8, lane, reg)
        });
    }

    #[test]
    fn the_accumulator_tiles_m16n8_exactly() {
        // The accumulator is unpacked f32 — one element per register.
        assert_tiles_exactly(16, 8, 4, PairAxis::None, |lane, reg| {
            acc_fragment_element(lane, reg)
        });
    }

    /// Anti-vacuity for the coverage property itself: the historical
    /// "32 lanes all read col=0" bug must fail it. Without this, a
    /// `assert_tiles_exactly` that silently accepted anything would look
    /// exactly like the passing tests above.
    #[test]
    #[should_panic(expected = "is held twice")]
    fn the_coverage_property_rejects_the_all_lanes_read_col_zero_bug() {
        assert_tiles_exactly(16, 8, 2, PairAxis::Row, |lane, reg| {
            let (k, _n) = b_fragment_element(MmaShape::M16N8K16, lane, reg);
            (k, 0) // every lane reads column 0 — the reverted `load_transposed` bug
        });
    }

    /// And the other failure direction: a fragment that covers only part of
    /// the tile. This is the shape of the FIRST historical bug — a helper
    /// that "covers only 1 row x 4 col-pair positions per lane" instead of
    /// the full 2x2. Half the registers, half the tile, no collisions.
    #[test]
    #[should_panic(expected = "unheld")]
    fn the_coverage_property_rejects_a_partial_fragment() {
        // m16k16 needs all four A registers; two cover only the col_lo half.
        assert_tiles_exactly(16, 16, 2, PairAxis::Col, |lane, reg| {
            a_fragment_element(MmaShape::M16N8K16, lane, reg)
        });
    }
}
