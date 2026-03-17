//! Paged KV-cache for autoregressive generation.
pub mod block_alloc;
pub mod page_table;
pub mod manager;
pub mod cow;

pub type BlockId = u32;
pub type SeqId = u64;
