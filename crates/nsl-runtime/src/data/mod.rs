//! M60: Exabyte-scale distributed data streaming.
//!
//! Provides:
//! - Binary shard format with deterministic per-rank assignment
//! - GPUDirect Storage (GDS) integration with mmap fallback
//! - N-stage prefetch pipeline overlapping I/O, CPU processing, and GPU compute
//! - Multi-modal streaming (text, image, audio, video)
//! - Checkpoint-aware data position for exact resumption

pub mod shards;
pub mod gds;
pub mod pipeline;
pub mod multimodal;
