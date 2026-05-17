//! Three-site identity test for `pca_tile_config::num_tiles`.
//!
//! For every (seq_len, block_size) tuple in the supported config matrix,
//! the Rust formula and the value used at the `range_table_addrs` consumer
//! site must agree. (The third site — emitted PTX loop bound — is verified
//! once `emit_range_table_preamble` lands in Task 5.)

use nsl_codegen::pca_tile_config::num_tiles;

const SUPPORTED_CONFIGS: &[(u32, u32)] = &[
    (2048,  32), (2048,  64),
    (4096,  32), (4096,  64), (4096, 128),
    (8192,  64), (8192, 128),
    (16_384,  64), (16_384, 128),
];

#[test]
fn num_tiles_matches_ceiling_division() {
    for &(seq_len, block_size) in SUPPORTED_CONFIGS {
        let expected = seq_len.div_ceil(block_size);
        assert_eq!(num_tiles(seq_len, block_size), expected);
    }
}

#[test]
fn num_tiles_at_exact_multiple() {
    assert_eq!(num_tiles(4096, 64), 64);
    assert_eq!(num_tiles(16_384, 128), 128);
}

#[test]
fn num_tiles_at_non_multiple_rounds_up() {
    assert_eq!(num_tiles(4097, 64), 65);
    assert_eq!(num_tiles(2049, 64), 33);
    assert_eq!(num_tiles(1, 64), 1);
}

#[test]
#[should_panic]
fn num_tiles_rejects_zero_block_size() {
    let _ = num_tiles(4096, 0);
}
