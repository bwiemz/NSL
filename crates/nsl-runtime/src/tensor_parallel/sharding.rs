//! Weight sharding utilities for tensor parallelism.
//!
//! Provides functions to compute shard boundaries and copy shard data
//! from a full tensor buffer into a per-rank destination buffer.

/// Compute the shard slice for a given rank along `shard_dim`.
///
/// Returns `(offset_elements, shard_elements, shard_shape)`:
/// - `offset_elements`: element offset into the flat buffer where this rank's data begins
///   (for dim 0, this is the contiguous start; for dim > 0, this is the offset within
///   each inner block where strided reads begin).
/// - `shard_elements`: total number of elements in this rank's shard.
/// - `shard_shape`: the shape of the shard tensor.
///
/// # Panics
/// - If `shard_dim >= shape.len()`
/// - If `shape[shard_dim]` is not evenly divisible by `world_size`
pub fn compute_shard_slice(
    shape: &[usize],
    shard_dim: usize,
    rank: usize,
    world_size: usize,
) -> (usize, usize, Vec<usize>) {
    assert!(
        shard_dim < shape.len(),
        "shard_dim ({shard_dim}) must be < shape.len() ({ndim})",
        ndim = shape.len(),
    );
    assert!(
        shape[shard_dim].is_multiple_of(world_size),
        "shape[{shard_dim}] ({dim_size}) must be divisible by world_size ({world_size})",
        dim_size = shape[shard_dim],
    );

    let shard_size_along_dim = shape[shard_dim] / world_size;

    // Build shard shape: same as original but with shard_dim divided.
    let mut shard_shape = shape.to_vec();
    shard_shape[shard_dim] = shard_size_along_dim;

    // Total elements in the shard.
    let shard_elements: usize = shard_shape.iter().product();

    // Compute element offset.
    if shard_dim == 0 {
        // Contiguous blocks: each rank owns a contiguous chunk of `shard_elements`.
        let offset_elements = rank * shard_elements;
        (offset_elements, shard_elements, shard_shape)
    } else {
        // Strided access: offset is within each "row" of the inner dimensions.
        // The offset is the starting column index within the innermost shard_dim stride.
        // For shape [a, b, c, ...] sharded on dim d, the offset within each
        // outer-dimension slice is rank * shard_size_along_dim * (product of dims after shard_dim).
        let inner_stride: usize = shape[shard_dim + 1..].iter().product::<usize>().max(1);
        let offset_elements = rank * shard_size_along_dim * inner_stride;
        (offset_elements, shard_elements, shard_shape)
    }
}

/// Copy a shard from `src` (full tensor) into `dst` (shard buffer).
///
/// - For `shard_dim == 0`: a single contiguous `memcpy` (fast path).
/// - For `shard_dim > 0`: strided copy — iterates over outer dimension slices and
///   copies each rank's chunk within the shard dimension.
///
/// # Parameters
/// - `src`: pointer to the full tensor's data.
/// - `dst`: pointer to the destination shard buffer (must be large enough).
/// - `shape`: shape of the full tensor.
/// - `shard_dim`: dimension along which to shard.
/// - `rank`: this rank's index (0-based).
/// - `world_size`: total number of ranks.
/// - `elem_bytes`: byte width of each element (e.g., 4 for f32, 8 for f64).
///
/// # Safety
/// The caller must ensure:
/// - `src` points to at least `product(shape) * elem_bytes` readable bytes.
/// - `dst` points to at least `shard_elements * elem_bytes` writable bytes.
/// - Pointers are properly aligned for the element type.
///
/// # Panics
/// Same precondition panics as `compute_shard_slice`.
pub unsafe fn copy_shard(
    src: *const u8,
    dst: *mut u8,
    shape: &[usize],
    shard_dim: usize,
    rank: usize,
    world_size: usize,
    elem_bytes: usize,
) {
    let (offset_elements, shard_elements, _shard_shape) =
        compute_shard_slice(shape, shard_dim, rank, world_size);

    if shard_dim == 0 {
        // Fast path: contiguous memcpy.
        let byte_offset = offset_elements * elem_bytes;
        let nbytes = shard_elements * elem_bytes;
        unsafe {
            std::ptr::copy_nonoverlapping(src.add(byte_offset), dst, nbytes);
        }
    } else {
        // Strided copy: iterate over all outer-dimension combinations and copy
        // each chunk within the shard dimension.
        //
        // For a tensor of shape [d0, d1, ..., d_{k-1}, d_k, d_{k+1}, ..., d_{n-1}]
        // sharded on dim k:
        //   outer_count = d0 * d1 * ... * d_{k-1}
        //   shard_chunk = (d_k / world_size) * d_{k+1} * ... * d_{n-1}  (elements per outer slice)
        //   full_row    = d_k * d_{k+1} * ... * d_{n-1}  (elements per outer slice in full tensor)
        //   inner_stride = d_{k+1} * ... * d_{n-1}
        //   chunk_start_in_row = rank * shard_chunk_elements_per_outer

        let outer_count: usize = shape[..shard_dim].iter().product::<usize>().max(1);
        let shard_size_along_dim = shape[shard_dim] / world_size;
        let inner_stride: usize = shape[shard_dim + 1..].iter().product::<usize>().max(1);

        // Number of contiguous elements to copy per outer-dimension iteration.
        let chunk_elements = shard_size_along_dim * inner_stride;
        let chunk_bytes = chunk_elements * elem_bytes;

        // Stride in the full tensor for one step along the outer dimension
        // (i.e., total elements in dims [shard_dim..]).
        let full_row_elements = shape[shard_dim] * inner_stride;

        // Offset within each full row where this rank's chunk starts.
        let rank_offset_in_row = rank * chunk_elements;

        let mut dst_offset: usize = 0;
        for outer_idx in 0..outer_count {
            let src_byte_offset =
                (outer_idx * full_row_elements + rank_offset_in_row) * elem_bytes;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.add(src_byte_offset),
                    dst.add(dst_offset),
                    chunk_bytes,
                );
            }
            dst_offset += chunk_bytes;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_shard_slice_dim0() {
        // [8, 4] sharded on dim 0 across 2 ranks
        let shape = [8, 4];
        let (off0, elems0, sh0) = compute_shard_slice(&shape, 0, 0, 2);
        assert_eq!(off0, 0);
        assert_eq!(elems0, 16);
        assert_eq!(sh0, vec![4, 4]);

        let (off1, elems1, sh1) = compute_shard_slice(&shape, 0, 1, 2);
        assert_eq!(off1, 16);
        assert_eq!(elems1, 16);
        assert_eq!(sh1, vec![4, 4]);
    }

    #[test]
    fn compute_shard_slice_dim1() {
        // [4, 8] sharded on dim 1 across 2 ranks
        let shape = [4, 8];
        let (off0, elems0, sh0) = compute_shard_slice(&shape, 1, 0, 2);
        assert_eq!(off0, 0);
        assert_eq!(elems0, 16); // 4 * 4
        assert_eq!(sh0, vec![4, 4]);

        let (off1, elems1, sh1) = compute_shard_slice(&shape, 1, 1, 2);
        assert_eq!(off1, 4); // starts at column 4 within each row
        assert_eq!(elems1, 16);
        assert_eq!(sh1, vec![4, 4]);
    }

    #[test]
    fn copy_shard_dim0_contiguous() {
        // Source: [4, 2] = [1,2, 3,4, 5,6, 7,8]  (f32)
        let src: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = [4usize, 2];
        let elem_bytes = std::mem::size_of::<f32>();

        // rank 0 gets first half: [1,2, 3,4]
        let mut dst0 = [0.0f32; 4];
        unsafe {
            copy_shard(
                src.as_ptr() as *const u8,
                dst0.as_mut_ptr() as *mut u8,
                &shape,
                0,
                0,
                2,
                elem_bytes,
            );
        }
        assert_eq!(dst0, [1.0, 2.0, 3.0, 4.0]);

        // rank 1 gets second half: [5,6, 7,8]
        let mut dst1 = [0.0f32; 4];
        unsafe {
            copy_shard(
                src.as_ptr() as *const u8,
                dst1.as_mut_ptr() as *mut u8,
                &shape,
                0,
                1,
                2,
                elem_bytes,
            );
        }
        assert_eq!(dst1, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn copy_shard_dim1_strided() {
        // Source: [2, 4] = [1,2,3,4, 5,6,7,8]  (f32)
        let src: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = [2usize, 4];
        let elem_bytes = std::mem::size_of::<f32>();

        // rank 0 gets cols 0-1: [1,2, 5,6]
        let mut dst0 = [0.0f32; 4];
        unsafe {
            copy_shard(
                src.as_ptr() as *const u8,
                dst0.as_mut_ptr() as *mut u8,
                &shape,
                1,
                0,
                2,
                elem_bytes,
            );
        }
        assert_eq!(dst0, [1.0, 2.0, 5.0, 6.0]);

        // rank 1 gets cols 2-3: [3,4, 7,8]
        let mut dst1 = [0.0f32; 4];
        unsafe {
            copy_shard(
                src.as_ptr() as *const u8,
                dst1.as_mut_ptr() as *mut u8,
                &shape,
                1,
                1,
                2,
                elem_bytes,
            );
        }
        assert_eq!(dst1, [3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn compute_shard_slice_dim0_4_ranks() {
        // [12, 3] sharded on dim 0 across 4 ranks (but 12 % 4 == 0 only if 3 rows each)
        let shape = [12, 3];
        for rank in 0..4 {
            let (off, elems, sh) = compute_shard_slice(&shape, 0, rank, 4);
            assert_eq!(elems, 9); // 3 * 3
            assert_eq!(off, rank * 9);
            assert_eq!(sh, vec![3, 3]);
        }
    }

    #[test]
    fn copy_shard_dim1_3d_tensor() {
        // 3D tensor: [2, 4, 2] sharded on dim 1 across 2 ranks
        // Flat: [1,2, 3,4, 5,6, 7,8,  9,10, 11,12, 13,14, 15,16]
        //        ^-- outer=0, d1=0,1,2,3  ^-- outer=1, d1=0,1,2,3
        // rank 0 (d1=0..2): [1,2, 3,4, 9,10, 11,12]
        // rank 1 (d1=2..4): [5,6, 7,8, 13,14, 15,16]
        let src: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let shape = [2usize, 4, 2];
        let elem_bytes = std::mem::size_of::<f32>();

        let mut dst0 = [0.0f32; 8];
        unsafe {
            copy_shard(
                src.as_ptr() as *const u8,
                dst0.as_mut_ptr() as *mut u8,
                &shape,
                1,
                0,
                2,
                elem_bytes,
            );
        }
        assert_eq!(dst0, [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);

        let mut dst1 = [0.0f32; 8];
        unsafe {
            copy_shard(
                src.as_ptr() as *const u8,
                dst1.as_mut_ptr() as *mut u8,
                &shape,
                1,
                1,
                2,
                elem_bytes,
            );
        }
        assert_eq!(dst1, [5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    #[should_panic(expected = "shard_dim (2) must be < shape.len() (2)")]
    fn compute_shard_slice_dim_out_of_bounds() {
        compute_shard_slice(&[4, 4], 2, 0, 2);
    }

    #[test]
    #[should_panic(expected = "must be divisible by world_size")]
    fn compute_shard_slice_not_divisible() {
        compute_shard_slice(&[5, 4], 0, 0, 2);
    }
}
