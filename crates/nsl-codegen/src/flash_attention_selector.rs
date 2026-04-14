//! Selects between v1 and v2 FlashAttention-2 emitters.

use crate::flash_attention::{
    FlashAttentionConfig, flash_attention_kernel_name as v1_kernel_name,
    shared_mem_bytes as v1_shared_mem, synthesize_flash_attention_ptx as v1_synth,
    use_mma_path,
};
use crate::flash_attention_v2::{
    flash_attention_kernel_name_v2, shared_mem_bytes_v2, synthesize_flash_attention_ptx_v2,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emitter { V1, V2 }

pub fn select_emitter(config: &FlashAttentionConfig) -> Emitter {
    // MMA path is not this spec's concern — stays on v1 until MMA spec lands.
    if use_mma_path(config.gpu_sm) { return Emitter::V1; }
    match std::env::var("NSL_FA_EMITTER").as_deref() {
        Ok("v2") => Emitter::V2,
        Ok("v1") => Emitter::V1,
        _        => Emitter::V1, // default = v1 until Task 15 flips it
    }
}

pub fn synthesize_flash_attention_ptx_selected(config: &FlashAttentionConfig) -> Vec<u8> {
    match select_emitter(config) {
        Emitter::V1 => v1_synth(config),
        Emitter::V2 => synthesize_flash_attention_ptx_v2(config),
    }
}

pub fn flash_attention_kernel_name_selected(config: &FlashAttentionConfig) -> String {
    match select_emitter(config) {
        Emitter::V1 => v1_kernel_name(config),
        Emitter::V2 => flash_attention_kernel_name_v2(config),
    }
}

pub fn shared_mem_bytes_selected(config: &FlashAttentionConfig) -> u32 {
    match select_emitter(config) {
        Emitter::V1 => v1_shared_mem(config),
        Emitter::V2 => shared_mem_bytes_v2(config),
    }
}
