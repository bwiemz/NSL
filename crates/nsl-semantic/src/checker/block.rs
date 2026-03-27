use super::*;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_train_block(&mut self, train: &TrainBlock) {
        // Check config expressions (model=, epochs=, etc.)
        for arg in &train.config {
            self.check_expr(&arg.value);
        }

        let mut has_step = false;

        for section in &train.sections {
            match section {
                TrainSection::Data(stmts) => {
                    for stmt in stmts {
                        self.check_stmt(stmt);
                    }
                }
                TrainSection::Optimizer(expr) => {
                    self.check_expr(expr);
                }
                TrainSection::Scheduler(_expr) => {
                    // Skip type-checking the scheduler call: codegen auto-injects
                    // base_lr and step as the first two arguments, so the user-facing
                    // arg count is intentionally less than the stdlib signature.
                }
                TrainSection::Step { param, body } => {
                    has_step = true;
                    // DataLoader yields Dict<Str, Tensor> batches. Type the step
                    // parameter concretely so field access (batch.input_ids) is validated.
                    use crate::types::{Shape, DType, Device};
                    let batch_type = Type::Dict(
                        Box::new(Type::Str),
                        Box::new(Type::Tensor {
                            shape: Shape::unknown(),
                            dtype: DType::Unknown,
                            device: Device::Unknown,
                        }),
                    );
                    let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
                    let prev = self.current_scope;
                    self.current_scope = scope;
                    self.declare_symbol(*param, batch_type, body.span, false, true);
                    for s in &body.stmts {
                        self.check_stmt(s);
                    }
                    self.current_scope = prev;
                }
                TrainSection::Eval { param, body } => {
                    let eval_type = Type::Dict(
                        Box::new(Type::Str),
                        Box::new(Type::Tensor {
                            shape: Shape::unknown(),
                            dtype: DType::Unknown,
                            device: Device::Unknown,
                        }),
                    );
                    let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
                    let prev = self.current_scope;
                    self.current_scope = scope;
                    self.declare_symbol(*param, eval_type, body.span, false, true);
                    for s in &body.stmts {
                        self.check_stmt(s);
                    }
                    self.current_scope = prev;
                }
                TrainSection::Callbacks(callbacks) => {
                    for cb in callbacks {
                        // Create a new scope and declare callback params
                        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
                        let prev = self.current_scope;
                        self.current_scope = scope;

                        for param in &cb.params {
                            let param_ty = if let Some(type_ann) = &param.type_ann {
                                self.resolve_type(type_ann)
                            } else {
                                // Infer types for well-known callback params
                                let pname = self.resolve_name(param.name);
                                match pname.as_str() {
                                    "step" | "epoch" => Type::Int,
                                    "loss" => Type::Tensor {
                                        shape: Shape::unknown(),
                                        dtype: DType::F64,
                                        device: Device::Cpu,
                                    },
                                    _ => Type::Unknown,
                                }
                            };
                            self.declare_symbol(param.name, param_ty, param.span, false, true);
                        }

                        for s in &cb.body.stmts {
                            self.check_stmt(s);
                        }

                        self.current_scope = prev;
                    }
                }
                TrainSection::Distribute(expr) => {
                    self.check_expr(expr);
                }
                TrainSection::Stmt(stmt) => {
                    self.check_stmt(stmt);
                }
            }
        }

        if !has_step {
            self.diagnostics.push(
                Diagnostic::warning("train block missing 'step' section")
                    .with_label(train.span, "expected a step(batch): section"),
            );
        }
    }

    pub(crate) fn check_serve_block(&mut self, serve: &nsl_ast::block::ServeBlock) {
        for entry in &serve.config {
            if let Some(ref type_ann) = entry.type_ann {
                self.resolve_type(type_ann);
            }
            self.check_expr(&entry.value);
        }
        for endpoint in &serve.endpoints {
            for param in &endpoint.params {
                if let Some(ref type_ann) = param.type_ann {
                    self.resolve_type(type_ann);
                }
            }
            if let Some(ref ret_type) = endpoint.return_type {
                self.resolve_type(ret_type);
            }
            self.check_block(&endpoint.body, ScopeKind::Block);
        }
        if serve.endpoints.is_empty() {
            self.diagnostics.push(
                Diagnostic::error("serve block must define at least one @endpoint function")
                    .with_label(serve.span, "no endpoints defined"),
            );
        }

        // M41: Validate disaggregated inference config if present
        self.check_disaggregated_serve(serve);
    }

    /// M41: Validate disaggregated inference configuration in a serve block.
    ///
    /// Checks that `prefill_workers >= 1`, `decode_workers >= 1`,
    /// `kv_transfer` is a recognized backend, and `drain_timeout_ms >= 0`.
    pub(crate) fn check_disaggregated_serve(&mut self, serve: &nsl_ast::block::ServeBlock) {
        let mut _prefill_workers = 1i64;
        let mut _decode_workers = 1i64;
        let mut _kv_transfer: Option<String> = None;

        for entry in &serve.config {
            let key = self.resolve_name(entry.key);
            match key.as_str() {
                "prefill_workers" => {
                    if let ExprKind::IntLiteral(v) = &entry.value.kind {
                        _prefill_workers = *v;
                        if *v < 1 {
                            self.diagnostics.push(
                                Diagnostic::error("prefill_workers must be >= 1")
                                    .with_label(entry.value.span, "invalid worker count"),
                            );
                        }
                    }
                }
                "decode_workers" => {
                    if let ExprKind::IntLiteral(v) = &entry.value.kind {
                        _decode_workers = *v;
                        if *v < 1 {
                            self.diagnostics.push(
                                Diagnostic::error("decode_workers must be >= 1")
                                    .with_label(entry.value.span, "invalid worker count"),
                            );
                        }
                    }
                }
                "kv_transfer" => {
                    if let ExprKind::StringLiteral(s) = &entry.value.kind {
                        let valid = ["rdma", "nvlink", "tcp", "shared_mem", "auto"];
                        if !valid.contains(&s.as_str()) {
                            self.diagnostics.push(
                                Diagnostic::error(format!(
                                    "unknown kv_transfer backend '{}', expected one of: {}",
                                    s,
                                    valid.join(", ")
                                ))
                                .with_label(entry.value.span, "invalid backend"),
                            );
                        }
                        _kv_transfer = Some(s.clone());
                    }
                }
                "drain_timeout_ms" => {
                    if let ExprKind::IntLiteral(v) = &entry.value.kind {
                        if *v < 0 {
                            self.diagnostics.push(
                                Diagnostic::error("drain_timeout_ms must be >= 0")
                                    .with_label(entry.value.span, "negative timeout"),
                            );
                        }
                    }
                }
                _ => {} // other config entries validated elsewhere
            }
        }
    }
}
