//! Shared memory declarations.
//!
//! Three helpers matching FA's existing three-way split:
//! - emit_static_smem_decl     — inside function body, fixed size
//! - emit_dynamic_smem_extern  — module scope, extern declaration
//! - emit_shmem_base_cvta      — inside function body, casts shmem to %shmem_base

/// Emit static SMEM declaration inside a function body:
///     .shared .align 16 .b8 shmem[{bytes}];
///
/// Caller must emit this AFTER the `.visible .entry ... { ` opener but
/// BEFORE any register decl that uses shmem (via %shmem_base).
pub fn emit_static_smem_decl(ptx: &mut String, bytes: usize) {
    ptx.push_str(&format!(
        "    .shared .align 16 .b8 shmem[{}];\n",
        bytes
    ));
}

/// Emit dynamic SMEM extern declaration at MODULE scope:
///     .extern .shared .align 16 .b8 shmem[];
///
/// Caller must emit this BEFORE the `.visible .entry` directive.  PTX
/// disallows `.extern .shared` inside a function body.
pub fn emit_dynamic_smem_extern(ptx: &mut String) {
    ptx.push_str(".extern .shared .align 16 .b8 shmem[];\n\n");
}

/// Emit the `cvta.shared.u64 %shmem_base, shmem;` line that casts the
/// SMEM array symbol to a register-holdable shared-address value.
///
/// Caller must have already declared `.reg .u64 %shmem_base;` before
/// calling this.
pub fn emit_shmem_base_cvta(ptx: &mut String) {
    ptx.push_str("    cvta.shared.u64 %shmem_base, shmem;\n");
}
