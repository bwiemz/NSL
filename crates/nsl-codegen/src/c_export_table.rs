//! Codegen for the runtime export table that `nsl_model_create` reads
//! to enumerate `@export`'d functions in the compiled .so/.dll.
//!
//! Emits two FFI accessors **inside the compiled artifact itself** so
//! the runtime can probe every shared lib uniformly via dlsym without
//! having to parse the host platform's symbol table:
//!
//!   - `nsl_get_num_exports() -> i64` — returns the export count.
//!   - `nsl_get_export_name(idx: i64) -> *const c_char` — returns the
//!     null-terminated name at index `idx ∈ [0, num_exports)`, or NULL
//!     if `idx` is out of bounds.
//!
//! Both symbols are declared with `Linkage::Export`. On MSVC the CLI
//! must additionally pass them through `link.exe /EXPORT:...` (see
//! `crates/nsl-cli/src/main.rs` near the `runtime_exports` list).
//!
//! ## Data layout (`__nsl_export_names` data symbol)
//!
//! A single read-only blob of two contiguous sections:
//!
//! 1. **Offset table** — `num_exports × i64` little-endian, where each
//!    `i64` is the byte offset (from the start of the blob) at which
//!    the corresponding null-terminated name begins. This means the
//!    accessor only needs the blob's base address + the index to find
//!    each name, with no extra fix-up pass at load time.
//! 2. **String pool** — the names back-to-back as
//!    `bytes-with-nul`, in declaration order.

use crate::c_header::ExportInfo;
use crate::compiler::Compiler;
use crate::error::CodegenError;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    types as cl_types, AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName,
};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, Linkage, Module};
use std::ffi::CString;

/// Emit the export-table data section plus the two accessor FFIs into
/// `compiler.module`. Fresh `UserFuncName` indices are pulled from
/// `compiler.next_func_index()` so the new functions don't collide with
/// the rest of the compilation unit.
pub(crate) fn emit_export_table(
    compiler: &mut Compiler<'_>,
    exports: &[ExportInfo],
) -> Result<(), CodegenError> {
    // ── 1. Build the offset table + names blob ────────────────────────────
    let n = exports.len() as i64;
    let header_bytes: i64 = (exports.len() as i64) * 8;
    let mut offsets: Vec<i64> = Vec::with_capacity(exports.len());
    let mut names_pool: Vec<u8> = Vec::new();
    let mut cursor: i64 = header_bytes;
    for export in exports {
        offsets.push(cursor);
        let cstr = CString::new(export.symbol_name.as_str()).map_err(|_| {
            CodegenError::new(format!(
                "@export symbol name '{}' contains an interior NUL byte",
                export.symbol_name
            ))
        })?;
        let bytes = cstr.as_bytes_with_nul();
        names_pool.extend_from_slice(bytes);
        cursor += bytes.len() as i64;
    }
    let mut full_blob: Vec<u8> = Vec::with_capacity(header_bytes as usize + names_pool.len());
    for off in &offsets {
        full_blob.extend_from_slice(&off.to_le_bytes());
    }
    full_blob.extend_from_slice(&names_pool);

    // ── 2. Declare + define the data symbol ───────────────────────────────
    // Linkage::Local — only the accessors below need it; the runtime
    // reaches the bytes through the accessors, never via dlsym on the
    // data symbol itself.
    let names_data = compiler
        .module
        .declare_data("__nsl_export_names", Linkage::Local, false, false)
        .map_err(|e| CodegenError::new(format!("declare __nsl_export_names: {e:?}")))?;
    let mut desc = DataDescription::new();
    desc.define(full_blob.into_boxed_slice());
    compiler
        .module
        .define_data(names_data, &desc)
        .map_err(|e| CodegenError::new(format!("define __nsl_export_names: {e:?}")))?;

    let call_conv = compiler.module.target_config().default_call_conv;

    // ── 3. Emit `nsl_get_num_exports() -> i64` ────────────────────────────
    {
        let mut num_sig = Signature::new(call_conv);
        num_sig.returns.push(AbiParam::new(cl_types::I64));
        let num_fn = compiler
            .module
            .declare_function("nsl_get_num_exports", Linkage::Export, &num_sig)
            .map_err(|e| CodegenError::new(format!("declare nsl_get_num_exports: {e:?}")))?;
        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, compiler.next_func_index()),
            num_sig,
        ));
        let mut fb_ctx = FunctionBuilderContext::new();
        {
            let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
            let block = fb.create_block();
            fb.switch_to_block(block);
            fb.seal_block(block);
            let count = fb.ins().iconst(cl_types::I64, n);
            fb.ins().return_(&[count]);
            fb.finalize();
        }
        compiler
            .module
            .define_function(num_fn, &mut ctx)
            .map_err(|e| CodegenError::new(format!("define nsl_get_num_exports: {e:?}")))?;
    }

    // ── 4. Emit `nsl_get_export_name(idx: i64) -> *const c_char` ──────────
    // Returned as i64 (pointer-sized integer) — matches the calling
    // convention used by the other runtime FFIs in c_wrapper.rs.
    {
        let mut name_sig = Signature::new(call_conv);
        name_sig.params.push(AbiParam::new(cl_types::I64));
        name_sig.returns.push(AbiParam::new(cl_types::I64));
        let name_fn = compiler
            .module
            .declare_function("nsl_get_export_name", Linkage::Export, &name_sig)
            .map_err(|e| CodegenError::new(format!("declare nsl_get_export_name: {e:?}")))?;
        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, compiler.next_func_index()),
            name_sig,
        ));
        let mut fb_ctx = FunctionBuilderContext::new();
        {
            let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

            let entry = fb.create_block();
            let oob = fb.create_block();
            let ok = fb.create_block();
            fb.append_block_params_for_function_params(entry);
            let idx = fb.block_params(entry)[0];

            fb.switch_to_block(entry);
            let n_val = fb.ins().iconst(cl_types::I64, n);
            // Unsigned-less-than handles negative idx (e.g. -1) by
            // routing to the OOB path automatically.
            let in_range = fb.ins().icmp(IntCC::UnsignedLessThan, idx, n_val);
            fb.ins().brif(in_range, ok, &[], oob, &[]);
            fb.seal_block(entry);

            // ── OOB path: return NULL ──────────────────────────────
            fb.switch_to_block(oob);
            fb.seal_block(oob);
            let null = fb.ins().iconst(cl_types::I64, 0);
            fb.ins().return_(&[null]);

            // ── In-range path: base + offset_table[idx] ───────────
            fb.switch_to_block(ok);
            fb.seal_block(ok);
            let names_gv = compiler.module.declare_data_in_func(names_data, fb.func);
            let base = fb.ins().symbol_value(cl_types::I64, names_gv);
            // idx * 8 — header is i64 (8-byte) offsets.
            let eight = fb.ins().iconst(cl_types::I64, 8);
            let off_bytes = fb.ins().imul(idx, eight);
            let off_addr = fb.ins().iadd(base, off_bytes);
            let off = fb.ins().load(cl_types::I64, MemFlags::trusted(), off_addr, 0);
            let name_ptr = fb.ins().iadd(base, off);
            fb.ins().return_(&[name_ptr]);

            fb.finalize();
        }
        compiler
            .module
            .define_function(name_fn, &mut ctx)
            .map_err(|e| CodegenError::new(format!("define nsl_get_export_name: {e:?}")))?;
    }

    Ok(())
}

