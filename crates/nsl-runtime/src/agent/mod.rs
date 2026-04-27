//! M56 agent runtime: mailboxes, scheduler, pool, FFI.
//!
//! Spec: docs/superpowers/specs/2026-04-23-m56-multi-agent-v1-design.md §3.

pub mod mailbox;

pub use mailbox::{PortMailbox, PortMessage, StructPayload};
