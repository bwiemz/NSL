// Shared helpers for FP8 integration tests. Declared as a module by every
// test file via `mod common;`. Cargo treats `tests/common/mod.rs` as a
// shared submodule, not its own test binary.
#![allow(dead_code)]

pub mod fp8_reference;
