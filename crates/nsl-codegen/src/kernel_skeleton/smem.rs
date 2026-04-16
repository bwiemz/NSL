//! Shared memory declarations.
//!
//! Three helpers matching FA's existing three-way split:
//! - emit_static_smem_decl     — inside function body, fixed size
//! - emit_dynamic_smem_extern  — module scope, extern declaration
//! - emit_shmem_base_cvta      — inside function body, casts shmem to %shmem_base
