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
        // M30: Initialize tensor parallelism if multi-device
        if self.features.world_size > 1 {
            self.compile_call_by_name(builder, "nsl_tp_init", &[])?;
        }

        // Extract config values with defaults
        let mut max_batch: i64 = 32;
        let mut max_seq_len: i64 = 4096;
        let mut kv_blocks: i64 = 2048;
        let mut prefill_chunk: i64 = 512;
        let mut prefill_workers: i64 = 1;
        let mut decode_workers: i64 = 1;
        let mut kv_transfer_backend: String = "auto".to_string();

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
                "prefill_workers" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        prefill_workers = *v;
                    }
                }
                "decode_workers" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        decode_workers = *v;
                    }
                }
                "kv_transfer" => {
                    if let nsl_ast::expr::ExprKind::StringLiteral(s) = &entry.value.kind {
                        kv_transfer_backend = s.clone();
                    }
                }
                _ => {
                    self.compile_expr(builder, state, &entry.value)?;
                }
            }
        }

        // Override from CLI flags if set
        if self.features.prefill_workers > 1 {
            prefill_workers = self.features.prefill_workers as i64;
        }
        if self.features.decode_workers > 1 {
            decode_workers = self.features.decode_workers as i64;
        }

        let is_disaggregated = prefill_workers > 1 || decode_workers > 1;

        if is_disaggregated {
            self.compile_disaggregated_serve(
                builder, state, serve,
                max_batch, max_seq_len, kv_blocks, prefill_chunk,
                prefill_workers, decode_workers, &kv_transfer_backend,
            )?;
        } else {
            // Monolithic M29 path (unchanged)
            let v_max_batch = builder.ins().iconst(cl_types::I64, max_batch);
            let v_max_seq_len = builder.ins().iconst(cl_types::I64, max_seq_len);
            let v_kv_blocks = builder.ins().iconst(cl_types::I64, kv_blocks);
            let v_prefill_chunk = builder.ins().iconst(cl_types::I64, prefill_chunk);

            self.compile_call_by_name(
                builder,
                "nsl_serve_init",
                &[v_max_batch, v_max_seq_len, v_kv_blocks, v_prefill_chunk],
            )?;

            for endpoint in &serve.endpoints {
                for stmt in &endpoint.body.stmts {
                    self.compile_stmt(builder, state, stmt)?;
                }
            }

            self.compile_call_by_name(builder, "nsl_serve_destroy", &[])?;
        }

        // M30: Tear down tensor parallelism if multi-device
        if self.features.world_size > 1 {
            self.compile_call_by_name(builder, "nsl_tp_destroy", &[])?;
        }

        Ok(())
    }

    /// M41: Compile a disaggregated serve block.
    ///
    /// Generates role-dispatch code by reading NSL_ROLE env var at runtime:
    ///   - role == 0 (router):  nsl_disagg_init() + nsl_disagg_router_loop()
    ///   - role == 1 (prefill): nsl_disagg_worker_init(1, rank, 0) + nsl_disagg_prefill_loop(0)
    ///   - role == 2 (decode):  nsl_disagg_worker_init(2, rank, 0) + nsl_disagg_decode_loop(0)
    ///
    /// The runtime FFI `nsl_disagg_get_role()` reads NSL_ROLE env var and returns
    /// 0/1/2. This avoids string comparison in Cranelift IR.
    #[allow(clippy::too_many_arguments)]
    fn compile_disaggregated_serve(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        serve: &ServeBlock,
        max_batch: i64,
        _max_seq_len: i64,
        kv_blocks: i64,
        _prefill_chunk: i64,
        prefill_workers: i64,
        decode_workers: i64,
        kv_transfer_backend: &str,
    ) -> Result<(), CodegenError> {
        // M41b: Map kv_transfer backend name to ID for runtime init
        let kv_backend_id: i64 = match kv_transfer_backend {
            "shared_mem" => 0,
            "nvlink" => 1,
            "rdma" => 2,
            "tcp" => 3,
            _ => -1, // auto-detect at runtime (includes "auto")
        };

        // Step 1: Get role from env var via FFI (returns 0=router, 1=prefill, 2=decode)
        let role = self.compile_call_by_name(builder, "nsl_disagg_get_role", &[])?;

        // Step 2: Branch on role
        let router_block = builder.create_block();
        let check_prefill_block = builder.create_block();
        let prefill_block = builder.create_block();
        let decode_block = builder.create_block();
        let merge_block = builder.create_block();

        // if role == 0 → router_block, else → check_prefill_block
        let zero = builder.ins().iconst(cl_types::I64, 0);
        let is_router = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, role, zero);
        builder.ins().brif(is_router, router_block, &[], check_prefill_block, &[]);

        // check_prefill_block: if role == 1 → prefill_block, else → decode_block
        builder.switch_to_block(check_prefill_block);
        builder.seal_block(check_prefill_block);
        let one = builder.ins().iconst(cl_types::I64, 1);
        let is_prefill = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, role, one);
        builder.ins().brif(is_prefill, prefill_block, &[], decode_block, &[]);

        // --- Router block ---
        builder.switch_to_block(router_block);
        builder.seal_block(router_block);
        let v_prefill = builder.ins().iconst(cl_types::I64, prefill_workers);
        let v_decode = builder.ins().iconst(cl_types::I64, decode_workers);
        let v_batch = builder.ins().iconst(cl_types::I64, max_batch);
        let v_kv = builder.ins().iconst(cl_types::I64, kv_blocks);
        self.compile_call_by_name(
            builder, "nsl_disagg_init", &[v_prefill, v_decode, v_batch, v_kv],
        )?;
        // Router runs endpoint bodies (sets up the event loop)
        for endpoint in &serve.endpoints {
            for stmt in &endpoint.body.stmts {
                self.compile_stmt(builder, state, stmt)?;
            }
        }
        self.compile_call_by_name(builder, "nsl_disagg_destroy", &[])?;
        builder.ins().jump(merge_block, &[]);

        // --- Prefill worker block ---
        builder.switch_to_block(prefill_block);
        builder.seal_block(prefill_block);
        let role_prefill = builder.ins().iconst(cl_types::I64, 1);
        let rank = self.compile_call_by_name(builder, "nsl_disagg_get_rank", &[])?;
        let model_zero = builder.ins().iconst(cl_types::I64, 0); // model ptr placeholder
        self.compile_call_by_name(
            builder, "nsl_disagg_worker_init", &[role_prefill, rank, model_zero],
        )?;
        // M41b: Initialize KV transfer backend for this worker
        let v_kv_backend = builder.ins().iconst(cl_types::I64, kv_backend_id);
        self.compile_call_by_name(builder, "nsl_kv_transfer_init", &[v_kv_backend, rank])?;
        let config_zero = builder.ins().iconst(cl_types::I64, 0);
        self.compile_call_by_name(builder, "nsl_disagg_prefill_loop", &[config_zero])?;
        self.compile_call_by_name(builder, "nsl_kv_transfer_destroy", &[])?;
        self.compile_call_by_name(builder, "nsl_disagg_worker_destroy", &[])?;
        builder.ins().jump(merge_block, &[]);

        // --- Decode worker block ---
        builder.switch_to_block(decode_block);
        builder.seal_block(decode_block);
        let role_decode = builder.ins().iconst(cl_types::I64, 2);
        let rank2 = self.compile_call_by_name(builder, "nsl_disagg_get_rank", &[])?;
        let model_zero2 = builder.ins().iconst(cl_types::I64, 0);
        self.compile_call_by_name(
            builder, "nsl_disagg_worker_init", &[role_decode, rank2, model_zero2],
        )?;
        // M41b: Initialize KV transfer backend for this worker
        let v_kv_backend2 = builder.ins().iconst(cl_types::I64, kv_backend_id);
        self.compile_call_by_name(builder, "nsl_kv_transfer_init", &[v_kv_backend2, rank2])?;
        // M33: Check for speculative decoding config — if any @speculative decorator was
        // collected during the compilation pass, log that speculative mode is active.
        // The actual draft→verify loop replacement is deferred (needs draft model forward
        // function wired); for now we pass a flag to the decode loop.
        let speculative_flag: i64 = if self.features.speculative_configs.values().next().is_some() {
            eprintln!("[nsl] Speculative decoding enabled for serve block");
            1
        } else {
            0
        };
        let config_zero2 = builder.ins().iconst(cl_types::I64, speculative_flag);
        self.compile_call_by_name(builder, "nsl_disagg_decode_loop", &[config_zero2])?;
        self.compile_call_by_name(builder, "nsl_kv_transfer_destroy", &[])?;
        self.compile_call_by_name(builder, "nsl_disagg_worker_destroy", &[])?;
        builder.ins().jump(merge_block, &[]);

        // --- Merge block ---
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        Ok(())
    }
}
