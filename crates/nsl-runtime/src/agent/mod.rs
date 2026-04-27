//! M56 agent runtime: mailboxes, scheduler, pool, FFI.
//!
//! Spec: docs/superpowers/specs/2026-04-23-m56-multi-agent-v1-design.md §3.

pub mod mailbox;
pub mod pool;
pub mod scheduler;

pub use mailbox::{PortMailbox, PortMessage, StructPayload};
pub use pool::{AcquireError, Lease, PipelineContext, PipelineContextPool};
pub use scheduler::{AgentId, AgentPorts, ReactorScheduler, StepOutcome};
