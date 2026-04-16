//! Predicated SMEM zero-padding for sub-MMA tile regions.
//!
//! Used by WRGA to zero the [real_extent, padded_extent) slice of an
//! SMEM tile when the real rank/m/k is below the MMA tile size.
//! Designed to be adoptable by FA for head_dim edge-case padding later.
