//! `nsl` subcommand handlers.
//!
//! This module groups the per-command implementations that were historically
//! inlined in `main.rs`. Each submodule owns one command (or a closely related
//! family); `main.rs` parses arguments and dispatches into these handlers.
//!
//! Extractions are behavior-preserving: code is moved verbatim, with only the
//! visibility (`pub(crate)`) and `use` paths adjusted.

pub(crate) mod convert;
pub(crate) mod fmt;
pub(crate) mod init;
pub(crate) mod profile_merge;
pub(crate) mod tokenize;
