//! M29: Serve block codegen.

use cranelift_codegen::ir::{types as cl_types, InstBuilder};
use cranelift_frontend::FunctionBuilder;

use nsl_ast::block::ServeBlock;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

impl Compiler<'_> {
    pub fn compile_serve_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        serve: &ServeBlock,
    ) -> Result<(), CodegenError> {
        // Extract config values with defaults
        let mut max_batch: i64 = 32;
        let mut max_seq_len: i64 = 4096;
        let mut kv_blocks: i64 = 2048;
        let mut prefill_chunk: i64 = 512;

        for entry in &serve.config {
            let key_name = self.resolve_sym(entry.key).to_string();
            match key_name.as_str() {
                "max_batch" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        max_batch = *v;
                    }
                }
                "max_seq_len" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        max_seq_len = *v;
                    }
                }
                "kv_blocks" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        kv_blocks = *v;
                    }
                }
                "prefill_chunk" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        prefill_chunk = *v;
                    }
                }
                _ => {
                    self.compile_expr(builder, state, &entry.value)?;
                }
            }
        }

        // Emit: nsl_serve_init(max_batch, max_seq_len, kv_blocks, prefill_chunk)
        let v_max_batch = builder.ins().iconst(cl_types::I64, max_batch);
        let v_max_seq_len = builder.ins().iconst(cl_types::I64, max_seq_len);
        let v_kv_blocks = builder.ins().iconst(cl_types::I64, kv_blocks);
        let v_prefill_chunk = builder.ins().iconst(cl_types::I64, prefill_chunk);

        self.compile_call_by_name(
            builder,
            "nsl_serve_init",
            &[v_max_batch, v_max_seq_len, v_kv_blocks, v_prefill_chunk],
        )?;

        // Compile endpoint bodies as statements
        for endpoint in &serve.endpoints {
            for stmt in &endpoint.body.stmts {
                self.compile_stmt(builder, state, stmt)?;
            }
        }

        // Emit: nsl_serve_destroy()
        self.compile_call_by_name(builder, "nsl_serve_destroy", &[])?;

        Ok(())
    }
}
