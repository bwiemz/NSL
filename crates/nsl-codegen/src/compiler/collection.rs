use std::collections::HashMap;

use cranelift_codegen::ir::types as cl_types;

use nsl_ast::decl::ModelMember;
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_ast::types::TypeExprKind;
use nsl_semantic::types::Type;

use super::Compiler;
use crate::context::StructField;
use crate::context::StructLayout;
use crate::error::CodegenError;
use crate::types::nsl_type_to_cl;

impl Compiler<'_> {
    // ── Pass 0: Collect string literals ─────────────────────────────

    pub fn collect_strings(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            self.collect_strings_in_stmt(stmt)?;
        }
        Ok(())
    }

    fn collect_strings_in_stmt(&mut self, stmt: &Stmt) -> Result<(), CodegenError> {
        match &stmt.kind {
            StmtKind::VarDecl { value: Some(expr), .. } => self.collect_strings_in_expr(expr),
            StmtKind::FnDef(fn_def) => {
                for s in &fn_def.body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            StmtKind::If { condition, then_block, elif_clauses, else_block } => {
                self.collect_strings_in_expr(condition)?;
                for s in &then_block.stmts { self.collect_strings_in_stmt(s)?; }
                for (cond, block) in elif_clauses {
                    self.collect_strings_in_expr(cond)?;
                    for s in &block.stmts { self.collect_strings_in_stmt(s)?; }
                }
                if let Some(block) = else_block {
                    for s in &block.stmts { self.collect_strings_in_stmt(s)?; }
                }
                Ok(())
            }
            StmtKind::For { iterable, body, .. } => {
                self.collect_strings_in_expr(iterable)?;
                for s in &body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            StmtKind::While { condition, body } => {
                self.collect_strings_in_expr(condition)?;
                for s in &body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            StmtKind::WhileLet { expr, body, .. } => {
                self.collect_strings_in_expr(expr)?;
                for s in &body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            StmtKind::Match { subject, arms } => {
                self.collect_strings_in_expr(subject)?;
                for arm in arms {
                    if let nsl_ast::pattern::PatternKind::Literal(lit) = &arm.pattern.kind {
                        self.collect_strings_in_expr(lit)?;
                    }
                    for s in &arm.body.stmts { self.collect_strings_in_stmt(s)?; }
                }
                Ok(())
            }
            StmtKind::ModelDef(md) => {
                for member in &md.members {
                    match member {
                        ModelMember::LayerDecl { init: Some(expr), .. } => {
                            self.collect_strings_in_expr(expr)?;
                        }
                        ModelMember::Method(fn_def, _decos) => {
                            for s in &fn_def.body.stmts { self.collect_strings_in_stmt(s)?; }
                        }
                        _ => {}
                    }
                }
                Ok(())
            }
            StmtKind::Decorated { stmt, .. } => self.collect_strings_in_stmt(stmt),
            StmtKind::Return(Some(expr)) => self.collect_strings_in_expr(expr),
            StmtKind::Assign { value, .. } => self.collect_strings_in_expr(value),
            StmtKind::Expr(expr) => self.collect_strings_in_expr(expr),
            StmtKind::GradBlock(grad) => {
                self.collect_strings_in_expr(&grad.targets)?;
                for s in &grad.body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn collect_strings_in_expr(&mut self, expr: &Expr) -> Result<(), CodegenError> {
        match &expr.kind {
            ExprKind::StringLiteral(s) => { self.intern_string(s)?; Ok(()) }
            ExprKind::FString(parts) => {
                for part in parts {
                    match part {
                        nsl_ast::expr::FStringPart::Text(s) => { self.intern_string(s)?; }
                        nsl_ast::expr::FStringPart::Expr(e) => { self.collect_strings_in_expr(e)?; }
                    }
                }
                Ok(())
            }
            ExprKind::BinaryOp { left, right, .. } | ExprKind::Pipe { left, right } => {
                self.collect_strings_in_expr(left)?;
                self.collect_strings_in_expr(right)
            }
            ExprKind::UnaryOp { operand, .. } => self.collect_strings_in_expr(operand),
            ExprKind::Call { callee, args } => {
                self.collect_strings_in_expr(callee)?;
                for arg in args { self.collect_strings_in_expr(&arg.value)?; }
                Ok(())
            }
            ExprKind::ListLiteral(elems) => {
                for e in elems { self.collect_strings_in_expr(e)?; }
                Ok(())
            }
            ExprKind::MemberAccess { object, .. } | ExprKind::Paren(object) => {
                self.collect_strings_in_expr(object)
            }
            ExprKind::IfExpr { condition, then_expr, else_expr } => {
                self.collect_strings_in_expr(condition)?;
                self.collect_strings_in_expr(then_expr)?;
                self.collect_strings_in_expr(else_expr)
            }
            ExprKind::Subscript { object, index } => {
                self.collect_strings_in_expr(object)?;
                if let nsl_ast::expr::SubscriptKind::Index(idx_expr) = index.as_ref() {
                    self.collect_strings_in_expr(idx_expr)?;
                }
                Ok(())
            }
            ExprKind::ListComp { element, generators } => {
                self.collect_strings_in_expr(element)?;
                for gen in generators {
                    self.collect_strings_in_expr(&gen.iterable)?;
                    for cond in &gen.conditions {
                        self.collect_strings_in_expr(cond)?;
                    }
                }
                Ok(())
            }
            ExprKind::DictLiteral(pairs) => {
                for (k, v) in pairs {
                    match &k.kind {
                        ExprKind::Ident(sym) => {
                            let name = self.resolve_sym(*sym).to_string();
                            self.intern_string(&name)?;
                        }
                        _ => { self.collect_strings_in_expr(k)?; }
                    }
                    self.collect_strings_in_expr(v)?;
                }
                Ok(())
            }
            ExprKind::TupleLiteral(elems) => {
                for e in elems { self.collect_strings_in_expr(e)?; }
                Ok(())
            }
            ExprKind::Lambda { body, .. } => {
                self.collect_strings_in_expr(body)
            }
            ExprKind::MatchExpr { subject, arms } => {
                self.collect_strings_in_expr(subject)?;
                for arm in arms {
                    // Collect strings in pattern literals
                    if let nsl_ast::pattern::PatternKind::Literal(lit) = &arm.pattern.kind {
                        self.collect_strings_in_expr(lit)?;
                    }
                    for s in &arm.body.stmts { self.collect_strings_in_stmt(s)?; }
                }
                Ok(())
            }
            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start { self.collect_strings_in_expr(s)?; }
                if let Some(e) = end { self.collect_strings_in_expr(e)?; }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    // ── Pass 0.5a: Collect enum definitions ─────────────────────────

    pub fn collect_enums(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            if let StmtKind::EnumDef(ed) = &stmt.kind {
                let enum_name = self.resolve_sym(ed.name).to_string();
                let mut variants = Vec::new();
                for (i, variant) in ed.variants.iter().enumerate() {
                    let variant_name = self.resolve_sym(variant.name).to_string();
                    let tag = i as i64;
                    // Store qualified name (always unique)
                    let qualified = format!("{}.{}", enum_name, variant_name);
                    self.enum_variants.insert(qualified, tag);
                    // Store bare name (may collide, last-write-wins for unqualified use)
                    self.enum_variants.insert(variant_name.clone(), tag);
                    variants.push((variant_name, tag));
                }
                self.enum_defs.insert(enum_name, variants);
            }
        }
        Ok(())
    }

    /// Look up an enum variant by name and return its integer tag.
    /// Accepts both bare names ("Red") and qualified names ("Color.Red").
    pub fn lookup_enum_variant_tag(&self, name: &str) -> Option<i64> {
        self.enum_variants.get(name).copied()
    }

    /// Look up with qualified enum name (EnumName.VariantName).
    pub fn lookup_qualified_variant(&self, enum_name: &str, variant_name: &str) -> Option<i64> {
        let qualified = format!("{}.{}", enum_name, variant_name);
        self.enum_variants.get(&qualified).copied()
    }

    // ── Pass 0.5b: Collect struct definitions ─────────────────────────

    pub fn collect_structs(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            if let StmtKind::StructDef(sd) = &stmt.kind {
                let name = self.resolve_sym(sd.name).to_string();
                let mut fields = Vec::new();
                let mut offset = 0usize;

                let struct_type = self.node_type(stmt.id).clone();
                if let Type::Struct { fields: type_fields, .. } = &struct_type {
                    for (field_sym, field_type) in type_fields {
                        let field_name = self.resolve_sym(*field_sym).to_string();
                        let cl_type = nsl_type_to_cl(field_type);
                        let size = cl_type.bytes() as usize;
                        let align = size.max(1);
                        offset = (offset + align - 1) & !(align - 1);
                        fields.push(StructField { name: field_name, cl_type, offset });
                        offset += size;
                    }
                } else {
                    for field in &sd.fields {
                        let field_name = self.resolve_sym(field.name).to_string();
                        let cl_type = match &field.type_ann.kind {
                            TypeExprKind::Named(sym) => self.resolve_type_name_to_cl(*sym),
                            _ => cl_types::I64,
                        };
                        let size = cl_type.bytes() as usize;
                        let align = size.max(1);
                        offset = (offset + align - 1) & !(align - 1);
                        fields.push(StructField { name: field_name, cl_type, offset });
                        offset += size;
                    }
                }

                self.struct_layouts.insert(name.clone(), StructLayout { name, fields, total_size: offset });
            }
        }
        Ok(())
    }

    // ── Pass 0.5c: Collect model definitions ─────────────────────────

    pub fn collect_models(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            if let StmtKind::ModelDef(md) = &stmt.kind {
                let name = self.resolve_sym(md.name).to_string();
                let mut fields = Vec::new();
                let mut offset = 0usize;
                let mut field_type_map: HashMap<String, String> = HashMap::new();

                // Extract fields from AST LayerDecl members (type_map may not have stmt types)
                for member in &md.members {
                    if let nsl_ast::decl::ModelMember::LayerDecl { name: field_sym, type_ann, .. } = member {
                        let field_name = self.resolve_sym(*field_sym).to_string();

                        // Check if this is a FixedArray type
                        if let nsl_ast::types::TypeExprKind::FixedArray { element_type, size } = &type_ann.kind {
                            let align = 8usize;
                            offset = (offset + align - 1) & !(align - 1);
                            // Store a single StructField pointing to the base of the array
                            fields.push(StructField { name: field_name.clone(), cl_type: cl_types::I64, offset });
                            offset += (*size as usize) * 8;

                            // Record array marker in field_type_map
                            if let nsl_ast::types::TypeExprKind::Named(elem_sym) = &element_type.kind {
                                let elem_name = self.resolve_sym(*elem_sym).to_string();
                                field_type_map.insert(field_name, format!("[{};{}]", elem_name, size));
                            }
                            continue;
                        }

                        let cl_type = match &type_ann.kind {
                            nsl_ast::types::TypeExprKind::Named(sym) => self.resolve_type_name_to_cl(*sym),
                            _ => cl_types::I64,
                        };
                        let size = cl_type.bytes() as usize;
                        let align = size.max(1);
                        offset = (offset + align - 1) & !(align - 1);
                        fields.push(StructField { name: field_name.clone(), cl_type, offset });
                        offset += size;

                        // Check if this is a nested model field
                        if let nsl_ast::types::TypeExprKind::Named(type_sym) = &type_ann.kind {
                            let type_name = self.resolve_sym(*type_sym).to_string();
                            if self.struct_layouts.contains_key(&type_name) {
                                field_type_map.insert(field_name, type_name);
                            }
                        }
                    }
                }

                // Check for @paged_kv and @shard decorators on members
                for member in &md.members {
                    if let nsl_ast::decl::ModelMember::LayerDecl { name: field_sym, decorators, .. } = member {
                        for deco in decorators {
                            // M30: @shard decorator extraction
                            if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "shard" {
                                if let Some(info) = crate::tensor_parallel::extract_shard_decorator(
                                    std::slice::from_ref(deco),
                                    &|sym| self.resolve_sym(sym),
                                ) {
                                    let model_name = self.resolve_sym(md.name).to_string();
                                    let layer_name_str = self.resolve_sym(*field_sym).to_string();
                                    let layer_key = format!("{}.{}", model_name, layer_name_str);
                                    self.features.shard_configs.insert(layer_key, info);
                                }
                            }
                            // M32: @moe decorator extraction
                            if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "moe" {
                                if let Some(info) = crate::moe::extract_moe_decorator(
                                    std::slice::from_ref(deco),
                                    &|sym| self.resolve_sym(sym),
                                ) {
                                    let model_name = self.resolve_sym(md.name).to_string();
                                    let layer_name_str = self.resolve_sym(*field_sym).to_string();
                                    let layer_key = format!("{}.{}", model_name, layer_name_str);
                                    self.features.moe_configs.insert(layer_key, info);
                                }
                            }
                            // M34: @context_parallel decorator extraction
                            if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "context_parallel" {
                                if let Some(info) = crate::context_parallel::extract_context_parallel_decorator(
                                    std::slice::from_ref(deco),
                                    &|sym| self.resolve_sym(sym),
                                ) {
                                    let model_name = self.resolve_sym(md.name).to_string();
                                    let layer_name_str = self.resolve_sym(*field_sym).to_string();
                                    let layer_key = format!("{}.{}", model_name, layer_name_str);
                                    self.features.cp_ring_size = info.ring_size;
                                    self.features.context_parallel_configs.insert(layer_key, info);
                                }
                            }
                            if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "paged_kv" {
                                let mut block_size: i64 = 16;
                                let mut num_blocks: i64 = 1024;
                                let mut num_heads: i64 = 1;
                                let mut head_dim: i64 = 64;
                                let mut num_layers: i64 = 1;
                                if let Some(ref args) = deco.args {
                                    for arg in args {
                                        if let Some(ref name_sym) = arg.name {
                                            let aname = self.resolve_sym(*name_sym).to_string();
                                            if let nsl_ast::expr::ExprKind::IntLiteral(n) = &arg.value.kind {
                                                match aname.as_str() {
                                                    "block_size" => block_size = *n,
                                                    "num_blocks" => num_blocks = *n,
                                                    "num_heads" => num_heads = *n,
                                                    "head_dim" => head_dim = *n,
                                                    "num_layers" => num_layers = *n,
                                                    _ => {}
                                                }
                                            }
                                        }
                                    }
                                }
                                let model_name = self.resolve_sym(md.name).to_string();
                                self.paged_kv_configs.insert(model_name, (num_blocks, block_size, num_heads, head_dim, num_layers));
                            }
                            // M33: @speculative decorator extraction
                            if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "speculative" {
                                if let Some(info) = crate::speculative::extract_speculative_decorator(
                                    std::slice::from_ref(deco),
                                    &|sym| self.resolve_sym(sym),
                                ) {
                                    let model_name = self.resolve_sym(md.name).to_string();
                                    let layer_name_str = self.resolve_sym(*field_sym).to_string();
                                    let layer_key = format!("{}.{}", model_name, layer_name_str);
                                    self.features.speculative_configs.insert(layer_key, info);
                                }
                            }
                            // M42: @kv_compress decorator extraction
                            if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "kv_compress" {
                                let mut scheme = 0i64; // 0=none, 1=int8_head, 2=int8_token, 3=int4_group, 4=fp8
                                let mut window = 0i64;
                                let mut sinks = 0i64;
                                if let Some(ref args) = deco.args {
                                    for arg in args {
                                        if let Some(ref name_sym) = arg.name {
                                            let key = self.resolve_sym(*name_sym).to_string();
                                            match key.as_str() {
                                                "scheme" => {
                                                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &arg.value.kind {
                                                        scheme = *v;
                                                    }
                                                }
                                                "window" => {
                                                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &arg.value.kind {
                                                        window = *v;
                                                    }
                                                }
                                                "sinks" => {
                                                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &arg.value.kind {
                                                        sinks = *v;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                let model_name = self.resolve_sym(md.name).to_string();
                                let layer_name_str = self.resolve_sym(*field_sym).to_string();
                                let layer_key = format!("{}.{}", model_name, layer_name_str);
                                let policy = super::KvCompressPolicy {
                                    method: "quantize".to_string(),
                                    scheme: scheme as u8,
                                    window: window as usize,
                                    sinks: sinks as usize,
                                    budget: 0,
                                };
                                self.features.kv_compress_policies.entry(layer_key).or_default().push(policy);
                            }
                            // M43b: @pipeline decorator extraction
                            if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "pipeline" {
                                let mut stages = 4usize; // default
                                let mut schedule_type = crate::pipeline::ScheduleType::OneF1B;
                                let mut checkpoint_stages = true;
                                if let Some(ref args) = deco.args {
                                    for arg in args {
                                        if let Some(ref name_sym) = arg.name {
                                            let key = self.resolve_sym(*name_sym).to_string();
                                            match key.as_str() {
                                                "stages" => {
                                                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &arg.value.kind {
                                                        stages = *v as usize;
                                                    }
                                                }
                                                "schedule" => {
                                                    if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                                                        schedule_type = match s.as_str() {
                                                            "gpipe" => crate::pipeline::ScheduleType::GPipe,
                                                            _ => crate::pipeline::ScheduleType::OneF1B,
                                                        };
                                                    }
                                                }
                                                "checkpoint_stages" => {
                                                    if let nsl_ast::expr::ExprKind::BoolLiteral(b) = &arg.value.kind {
                                                        checkpoint_stages = *b;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                self.features.pipeline_config = Some(crate::pipeline::PipelineConfig {
                                    num_stages: stages,
                                    schedule_type,
                                    checkpoint_stages,
                                });
                            }
                        }
                    }
                }

                if !field_type_map.is_empty() {
                    self.model_field_types.insert(name.clone(), field_type_map);
                }
                self.struct_layouts.insert(name.clone(), StructLayout { name, fields, total_size: offset });
            }
        }
        Ok(())
    }
}
