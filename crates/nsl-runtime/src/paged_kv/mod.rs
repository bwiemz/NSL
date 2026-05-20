//! Paged KV-cache for autoregressive generation.
pub mod block_alloc;
pub mod cow;
pub mod manager;
pub mod page_table;

pub type BlockId = u32;
pub type SeqId = u64;
