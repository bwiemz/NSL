//! M58: Cluster-scale elastic fault tolerance.
//!
//! Three components:
//! 1. **Heartbeat monitor** — out-of-band UDP heartbeat for failure detection
//! 2. **Micro-checkpointing** — three-tier GPU→CPU→NVMe→remote checkpointing
//! 3. **Elastic resize** — shrink/grow DP dimension without stopping training

pub mod heartbeat;
pub mod checkpoint;
pub mod resize;
