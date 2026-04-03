use std::collections::HashSet;

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

    pub(crate) fn check_tokenizer_def(&mut self, tok: &TokenizerDef) {
        self.declare_symbol(tok.name, Type::Unknown, tok.span, true, false);

        let mut seen_config = HashSet::new();
        for arg in &tok.config {
            let Some(name_sym) = arg.name else {
                self.diagnostics.push(
                    Diagnostic::error("tokenizer config entries must be named")
                        .with_label(arg.span, "expected key = value"),
                );
                continue;
            };

            let name = self.resolve_name(name_sym);
            if !seen_config.insert(name_sym) {
                self.diagnostics.push(
                    Diagnostic::error(format!("duplicate tokenizer config entry '{name}'"))
                        .with_label(arg.span, "duplicate config entry"),
                );
            }

            match name.as_str() {
                "algorithm" => {
                    self.check_enum_ident(
                        &arg.value,
                        &[
                            "bpe",
                            "wordpiece",
                            "sentencepiece",
                            "unigram",
                            "char",
                            "byte",
                        ],
                        "tokenizer algorithm",
                    );
                }
                "vocab_size" => {
                    self.check_assignable_expr(&arg.value, &Type::Int, "tokenizer vocab_size");
                }
                "model_file" => {
                    self.check_assignable_expr(&arg.value, &Type::Str, "tokenizer model_file");
                }
                _ => {
                    self.diagnostics.push(
                        Diagnostic::error(format!("unknown tokenizer config entry '{name}'"))
                            .with_label(arg.span, "unknown tokenizer config entry"),
                    );
                }
            }
        }

        let mut seen_sections = HashSet::new();
        for item in &tok.body {
            let (section_name, section_span) = match item {
                TokenizerStmt::SpecialTokens { span, .. } => ("special_tokens", *span),
                TokenizerStmt::Normalize { span, .. } => ("normalize", *span),
                TokenizerStmt::PreTokenize { span, .. } => ("pre_tokenize", *span),
                TokenizerStmt::Padding { span, .. } => ("padding", *span),
                TokenizerStmt::Truncation { span, .. } => ("truncation", *span),
            };
            if !seen_sections.insert(section_name) {
                self.diagnostics.push(
                    Diagnostic::error(format!("duplicate tokenizer section '{section_name}'"))
                        .with_label(section_span, "duplicate tokenizer section"),
                );
            }

            match item {
                TokenizerStmt::SpecialTokens { entries, .. } => {
                    let mut seen_keys = HashSet::new();
                    for entry in entries {
                        self.check_duplicate_key(&mut seen_keys, entry, "special token");
                        match entry.value.kind {
                            ExprKind::StringLiteral(_) => {}
                            _ => {
                                self.diagnostics.push(
                                    Diagnostic::error("special token values must be string literals")
                                        .with_label(entry.value.span, "expected string literal"),
                                );
                            }
                        }
                    }
                }
                TokenizerStmt::Normalize { rules, span } => {
                    self.check_symbol_list_duplicates(rules, *span, "normalize rule");
                }
                TokenizerStmt::PreTokenize { rules, span } => {
                    self.check_symbol_list_duplicates(rules, *span, "pre-tokenize rule");
                }
                TokenizerStmt::Padding { entries, .. } => {
                    let mut seen_keys = HashSet::new();
                    for entry in entries {
                        self.check_duplicate_key(&mut seen_keys, entry, "padding entry");
                        let key = self.resolve_name(entry.key);
                        match key.as_str() {
                            "side" => {
                                self.check_enum_ident(
                                    &entry.value,
                                    &["left", "right"],
                                    "padding.side",
                                );
                            }
                            "pad_to" => match &entry.value.kind {
                                ExprKind::Ident(sym)
                                    if self.resolve_name(*sym) == "longest" => {}
                                ExprKind::Ident(_) => {
                                    self.diagnostics.push(
                                        Diagnostic::error(
                                            "padding.pad_to must be an integer or 'longest'",
                                        )
                                        .with_label(entry.value.span, "invalid pad_to value"),
                                    );
                                }
                                _ => {
                                    self.check_assignable_expr(
                                        &entry.value,
                                        &Type::Int,
                                        "padding.pad_to",
                                    );
                                }
                            },
                            _ => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "unknown padding entry '{key}'"
                                    ))
                                    .with_label(entry.span, "unknown padding entry"),
                                );
                            }
                        }
                    }
                }
                TokenizerStmt::Truncation { entries, .. } => {
                    let mut seen_keys = HashSet::new();
                    for entry in entries {
                        self.check_duplicate_key(&mut seen_keys, entry, "truncation entry");
                        let key = self.resolve_name(entry.key);
                        match key.as_str() {
                            "max_length" => {
                                self.check_assignable_expr(
                                    &entry.value,
                                    &Type::Int,
                                    "truncation.max_length",
                                );
                            }
                            "strategy" => {
                                self.check_enum_ident(
                                    &entry.value,
                                    &["longest_first", "only_first", "only_second"],
                                    "truncation.strategy",
                                );
                            }
                            _ => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!(
                                        "unknown truncation entry '{key}'"
                                    ))
                                    .with_label(entry.span, "unknown truncation entry"),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn check_dataset_def(&mut self, ds: &DatasetDef) {
        self.declare_symbol(ds.name, Type::Unknown, ds.span, true, false);

        self.check_assignable_expr(&ds.source, &Type::Str, "dataset name");

        let mut seen_fields = HashSet::new();
        let mut has_source_field = false;
        for entry in &ds.body {
            self.check_duplicate_key(&mut seen_fields, entry, "dataset field");
            let key = self.resolve_name(entry.key);
            match key.as_str() {
                "source" => {
                    has_source_field = true;
                    self.check_expr(&entry.value);
                }
                "format" => match &entry.value.kind {
                    ExprKind::Ident(_) | ExprKind::StringLiteral(_) => {}
                    _ => {
                        self.check_expr(&entry.value);
                    }
                },
                "transform" | "filter" => {
                    self.check_expr(&entry.value);
                }
                "shuffle" | "packing" => {
                    self.check_assignable_expr(&entry.value, &Type::Bool, &format!("dataset {key}"));
                }
                "sequence_length" | "max_samples" | "shuffle_buffer" | "pack_separator" => {
                    self.check_assignable_expr(&entry.value, &Type::Int, &format!("dataset {key}"));
                }
                "resume_from" => {
                    self.check_assignable_expr(&entry.value, &Type::Str, "dataset resume_from");
                }
                _ => {
                    self.diagnostics.push(
                        Diagnostic::error(format!("unknown dataset field '{key}'"))
                            .with_label(entry.span, "unknown dataset field"),
                    );
                }
            }
        }

        if !has_source_field {
            self.diagnostics.push(
                Diagnostic::error("dataset body must declare 'source'")
                    .with_label(ds.span, "missing source field"),
            );
        }
    }

    fn check_duplicate_key(
        &mut self,
        seen: &mut HashSet<Symbol>,
        entry: &KeyValueEntry,
        context: &str,
    ) {
        if !seen.insert(entry.key) {
            let key = self.resolve_name(entry.key);
            self.diagnostics.push(
                Diagnostic::error(format!("duplicate {context} '{key}'"))
                    .with_label(entry.span, "duplicate entry"),
            );
        }
    }

    fn check_symbol_list_duplicates(&mut self, items: &[Symbol], span: Span, context: &str) {
        let mut seen = HashSet::new();
        for item in items {
            if !seen.insert(*item) {
                let name = self.resolve_name(*item);
                self.diagnostics.push(
                    Diagnostic::error(format!("duplicate {context} '{name}'"))
                        .with_label(span, "duplicate entry"),
                );
            }
        }
    }

    fn check_assignable_expr(&mut self, expr: &Expr, expected: &Type, context: &str) {
        let ty = self.check_expr(expr);
        if !is_assignable(&ty, expected) {
            self.diagnostics.push(
                Diagnostic::error(format!(
                    "{context} must be {}, got {}",
                    display_type(expected),
                    display_type(&ty)
                ))
                .with_label(expr.span, "wrong type"),
            );
        }
    }

    fn check_enum_ident(&mut self, expr: &Expr, allowed: &[&str], context: &str) {
        match &expr.kind {
            ExprKind::Ident(sym) => {
                let value = self.resolve_name(*sym);
                if !allowed.iter().any(|candidate| *candidate == value) {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "{context} must be one of: {}",
                            allowed.join(", ")
                        ))
                        .with_label(expr.span, format!("found '{value}'")),
                    );
                }
            }
            _ => {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "{context} must be one of: {}",
                        allowed.join(", ")
                    ))
                    .with_label(expr.span, "expected identifier value"),
                );
            }
        }
    }
}
