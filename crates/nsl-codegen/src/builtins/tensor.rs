//! The tensor surface a program writes directly: creation, shape,
//! indexing, elementwise arithmetic, reductions and the common layers.
//!
//! The lowering half of this group of the runtime-function registry; the
//! declarations are rows of the ABI table in `crates/nsl-abi/src/table.rs`
//! (see `builtins/mod.rs`).

