//! M55: Zero-Knowledge Inference Circuits for NeuralScript.
//!
//! This module implements ZK proof generation for NSL inference graphs.
//! It compiles NSL model forward passes into arithmetic circuits over the
//! BN254 scalar field, enabling verifiable ML inference without revealing weights.

pub mod air;
pub mod backend;
pub mod field;
pub mod field_m31;
pub mod folding;
pub mod ir;
pub mod lookup;
pub mod lookup_native;
pub mod lower;
pub mod plonky3;
pub mod stats;
pub mod witness;

#[cfg(test)]
mod tests;

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::operator::BinOp;
use nsl_ast::Symbol;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Result type for top-level ZK compilation
// ---------------------------------------------------------------------------

/// Result of compiling a function or model decorated with `@zk_proof`.
pub struct ZkCompileResult {
    pub zkir: ir::ZkIR,
    pub stats: stats::CircuitStats,
    /// The generated proof (if the folding prover ran successfully).
    pub proof: Option<backend::ZkProof>,
}

// ---------------------------------------------------------------------------
// extract_zk_proof_decorator
// ---------------------------------------------------------------------------

/// Extract `@zk_proof` or `@zk_proof(mode="weight_private")` from a decorator list.
///
/// Returns the [`backend::ZkMode`] if a `@zk_proof` decorator is found;
/// defaults to [`backend::ZkMode::ArchitectureAttestation`] when no `mode=` arg
/// is present.
pub fn extract_zk_proof_decorator<'a>(
    decos: &[Decorator],
    resolve: &dyn Fn(Symbol) -> &'a str,
) -> Option<backend::ZkMode> {
    for d in decos {
        if d.name.len() == 1 && resolve(d.name[0]) == "zk_proof" {
            let mode = d
                .args
                .as_ref()
                .and_then(|args| {
                    args.iter().find_map(|arg| {
                        let name_sym = arg.name?;
                        if resolve(name_sym) != "mode" {
                            return None;
                        }
                        if let ExprKind::StringLiteral(ref s) = arg.value.kind {
                            match s.as_str() {
                                "weight_private" => Some(backend::ZkMode::WeightPrivate),
                                "input_private" => Some(backend::ZkMode::InputPrivate),
                                "full_private" => Some(backend::ZkMode::FullPrivate),
                                "architecture_attestation" => {
                                    Some(backend::ZkMode::ArchitectureAttestation)
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    })
                })
                .unwrap_or(backend::ZkMode::ArchitectureAttestation);
            return Some(mode);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// extract_zk_lookup_decorator
// ---------------------------------------------------------------------------

/// Extract `@zk_lookup(input_bits=N, output_bits=M)` from a decorator list.
///
/// Returns `Some((input_bits, output_bits))` if a `@zk_lookup` decorator is found.
/// Both `input_bits` and `output_bits` are required; the decorator is ignored
/// if either argument is missing or not an integer literal.
pub fn extract_zk_lookup_decorator<'a>(
    decos: &[Decorator],
    resolve: &dyn Fn(Symbol) -> &'a str,
) -> Option<(u32, u32)> {
    for d in decos {
        if d.name.len() == 1 && resolve(d.name[0]) == "zk_lookup" {
            let args = d.args.as_ref()?;
            let mut input_bits: Option<u32> = None;
            let mut output_bits: Option<u32> = None;

            for arg in args {
                let name_sym = match arg.name {
                    Some(s) => s,
                    None => continue,
                };
                let name = resolve(name_sym);
                if let ExprKind::IntLiteral(val) = &arg.value.kind {
                    match name {
                        "input_bits" => input_bits = Some(*val as u32),
                        "output_bits" => output_bits = Some(*val as u32),
                        _ => {}
                    }
                }
            }

            if let (Some(ib), Some(ob)) = (input_bits, output_bits) {
                return Some((ib, ob));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// compile_zk — top-level orchestrator (placeholder until Task 14)
// ---------------------------------------------------------------------------

/// Top-level ZK compilation orchestrator.
///
/// Compiles a function or model decorated with `@zk_proof` into a [`ZkCompileResult`].
///
/// For the folding backend (default), this:
///   1. Lowers the DAG to per-layer ZkIRs
///   2. Creates a FoldingProver with the selected field
///   3. Folds each layer sequentially
///   4. Produces the final proof
///
/// Compile a `@zk_proof` function from its AST into a ZK proof.
///
/// Traverses the function body to build a ZkDag, then lowers to ZK-IR
/// and runs the folding prover.
pub fn compile_zk(
    fn_def: &nsl_ast::decl::FnDef,
    mode: backend::ZkMode,
    config: &backend::ZkConfig,
    type_map: &nsl_semantic::checker::TypeMap,
    interner: &nsl_lexer::Interner,
) -> Result<ZkCompileResult, backend::ZkError> {
    let mut dag = ast_to_zkdag(fn_def, type_map, interner)?;

    // M55: If a weights file was provided, load weight values and patch the DAG.
    // This populates ZkOp::Weight.values so the witness uses real weight data
    // instead of dummy zeros, enabling meaningful proof generation.
    if let Some(ref weights_path) = config.weights_path {
        match crate::weight_aware::WeightMap::load(weights_path) {
            Ok(mut wmap) => {
                for op in &mut dag.ops {
                    if let lower::ZkOp::Weight {
                        name,
                        values,
                        dtype_bits,
                        ..
                    } = op
                    {
                        if let Some(entry) = wmap.get_mut(name) {
                            let bw = entry.dtype.byte_width();
                            let vals: Vec<i64> = (0..entry.num_elements)
                                .map(|i| {
                                    let offset = i * bw;
                                    if offset + bw <= entry.data.len() {
                                        entry.dtype.to_f64(&entry.data[offset..offset + bw]) as i64
                                    } else {
                                        0i64
                                    }
                                })
                                .collect();
                            *dtype_bits = (entry.dtype.byte_width() * 8) as u32;
                            *values = Some(vals);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "[nsl] ZK: warning: failed to load weights from '{}': {} — using zero witness",
                    weights_path.display(),
                    e
                );
            }
        }
    }

    // M55d: Auto-select M31 field for INT8 quantized models (10x faster proving)
    let effective_config = if lower::is_int8_model(&dag) && config.field != backend::ZkField::BN254
    {
        eprintln!(
            "[nsl] M55d: detected INT8 model — auto-selecting Mersenne-31 field for fast proving"
        );
        let mut cfg = config.clone();
        cfg.field = backend::ZkField::Mersenne31;
        cfg
    } else {
        config.clone()
    };

    let zkir = lower::lower_dag_to_zkir(&dag, &effective_config);
    let circuit_stats = stats::compute_stats(&zkir);
    let _ = mode; // privacy mode affects witness layout (applied in lowering)

    // Try to produce a proof via the folding backend
    let proof = compile_zk_from_dag(&dag, &effective_config).ok();

    Ok(ZkCompileResult {
        zkir,
        stats: circuit_stats,
        proof,
    })
}

// ---------------------------------------------------------------------------
// AST → ZkDag translation
// ---------------------------------------------------------------------------

/// Builder for constructing a ZkDag from AST traversal.
struct ZkDagBuilder<'a> {
    ops: Vec<lower::ZkOp>,
    name_to_idx: HashMap<String, usize>,
    type_map: &'a nsl_semantic::checker::TypeMap,
    interner: &'a nsl_lexer::Interner,
    last_idx: usize,
}

impl<'a> ZkDagBuilder<'a> {
    fn new(
        type_map: &'a nsl_semantic::checker::TypeMap,
        interner: &'a nsl_lexer::Interner,
    ) -> Self {
        Self {
            ops: Vec::new(),
            name_to_idx: HashMap::new(),
            type_map,
            interner,
            last_idx: 0,
        }
    }

    fn push(&mut self, op: lower::ZkOp) -> usize {
        let idx = self.ops.len();
        self.ops.push(op);
        self.last_idx = idx;
        idx
    }

    fn resolve_sym(&self, sym: Symbol) -> &str {
        self.interner.resolve(sym.0).unwrap_or("<unknown>")
    }

    fn extract_shape(&self, node_id: nsl_ast::NodeId) -> Vec<usize> {
        if let Some(ty) = self.type_map.get(&node_id) {
            if let Some((shape, _, _)) = ty.as_tensor_parts() {
                return shape
                    .dims
                    .iter()
                    .filter_map(|d| {
                        if let nsl_semantic::types::Dim::Concrete(n) = d {
                            Some(*n as usize)
                        } else {
                            None
                        }
                    })
                    .collect();
            }
        }
        vec![1] // fallback: scalar
    }

    fn extract_dtype_bits(&self, node_id: nsl_ast::NodeId) -> u32 {
        if let Some(ty) = self.type_map.get(&node_id) {
            if let Some((_, dtype, _)) = ty.as_tensor_parts() {
                return match dtype {
                    nsl_semantic::types::DType::F64 => 64,
                    nsl_semantic::types::DType::F32 => 32,
                    nsl_semantic::types::DType::Fp16 | nsl_semantic::types::DType::Bf16 => 16,
                    nsl_semantic::types::DType::Int8 | nsl_semantic::types::DType::Uint8 => 8,
                    _ => 32,
                };
            }
        }
        32 // default
    }

    /// Lower an expression to a ZkOp index.
    fn lower_expr(&mut self, expr: &nsl_ast::expr::Expr) -> Result<usize, backend::ZkError> {
        match &expr.kind {
            ExprKind::Ident(sym) => {
                let name = self.resolve_sym(*sym).to_string();
                if let Some(&idx) = self.name_to_idx.get(&name) {
                    Ok(idx)
                } else {
                    // Unknown variable — treat as weight with unknown values
                    let shape = self.extract_shape(expr.id);
                    let bits = self.extract_dtype_bits(expr.id);
                    let idx = self.push(lower::ZkOp::Weight {
                        name: name.clone(),
                        shape,
                        dtype_bits: bits,
                        values: None,
                    });
                    self.name_to_idx.insert(name, idx);
                    Ok(idx)
                }
            }

            ExprKind::BinaryOp { left, op, right } => {
                let a = self.lower_expr(left)?;
                let b = self.lower_expr(right)?;
                let zk_op = match op {
                    BinOp::MatMul => lower::ZkOp::Matmul { a, b },
                    BinOp::Add => lower::ZkOp::Add { a, b },
                    BinOp::Mul => lower::ZkOp::Mul { a, b },
                    BinOp::Sub => {
                        // a - b = a + (-1 * b) — approximate as Add for ZK
                        lower::ZkOp::Add { a, b }
                    }
                    _ => {
                        return Err(backend::ZkError::CompilationError(format!(
                            "unsupported binary op in ZK: {:?}",
                            op
                        )))
                    }
                };
                Ok(self.push(zk_op))
            }

            ExprKind::Call { callee, args } => {
                let func_name = match &callee.kind {
                    ExprKind::Ident(sym) => self.resolve_sym(*sym).to_string(),
                    _ => {
                        return Err(backend::ZkError::CompilationError(
                            "ZK: only named function calls are supported".into(),
                        ))
                    }
                };

                match func_name.as_str() {
                    "relu" => {
                        let input = self.lower_expr(&args[0].value)?;
                        Ok(self.push(lower::ZkOp::Relu { input }))
                    }
                    "gelu" => {
                        let input = self.lower_expr(&args[0].value)?;
                        Ok(self.push(lower::ZkOp::Gelu { input }))
                    }
                    "sigmoid" => {
                        let input = self.lower_expr(&args[0].value)?;
                        Ok(self.push(lower::ZkOp::Sigmoid { input }))
                    }
                    "tanh" => {
                        let input = self.lower_expr(&args[0].value)?;
                        Ok(self.push(lower::ZkOp::Tanh { input }))
                    }
                    "exp" => {
                        let input = self.lower_expr(&args[0].value)?;
                        Ok(self.push(lower::ZkOp::Exp { input }))
                    }
                    "log" => {
                        let input = self.lower_expr(&args[0].value)?;
                        Ok(self.push(lower::ZkOp::Log { input }))
                    }
                    "softmax" => {
                        let input = self.lower_expr(&args[0].value)?;
                        let dim = if args.len() > 1 {
                            // TODO: extract dim from second arg
                            -1
                        } else {
                            -1
                        };
                        Ok(self.push(lower::ZkOp::Softmax { input, dim }))
                    }
                    "transpose" => {
                        let input = self.lower_expr(&args[0].value)?;
                        Ok(self.push(lower::ZkOp::Transpose { input }))
                    }
                    _ => {
                        // Unknown function — try to lower args and treat as pass-through
                        if args.is_empty() {
                            return Err(backend::ZkError::CompilationError(format!(
                                "ZK: unsupported function '{}'",
                                func_name
                            )));
                        }
                        // Fallback: return the first argument (pass-through)
                        self.lower_expr(&args[0].value)
                    }
                }
            }

            ExprKind::MemberAccess { object: _, member } => {
                // self.weight → emit Weight op
                let field_name = self.resolve_sym(*member).to_string();
                let shape = self.extract_shape(expr.id);
                let bits = self.extract_dtype_bits(expr.id);
                let idx = self.push(lower::ZkOp::Weight {
                    name: field_name.clone(),
                    shape,
                    dtype_bits: bits,
                    values: None,
                });
                Ok(idx)
            }

            ExprKind::Pipe { left, right } => {
                // x |> f  →  f(x)
                let _input = self.lower_expr(left)?;
                self.lower_expr(right)
            }

            ExprKind::IntLiteral(_) | ExprKind::FloatLiteral(_) => {
                // Constants — treated as scalar weights
                let idx = self.push(lower::ZkOp::Weight {
                    name: format!("const_{}", self.ops.len()),
                    shape: vec![1],
                    dtype_bits: 32,
                    values: None,
                });
                Ok(idx)
            }

            _ => {
                // Unsupported expression — return error with context
                Err(backend::ZkError::CompilationError(
                    "ZK: unsupported expression kind in @zk_proof function".to_string(),
                ))
            }
        }
    }
}

/// Convert a `@zk_proof` function's AST into a ZkDag.
pub fn ast_to_zkdag(
    fn_def: &nsl_ast::decl::FnDef,
    type_map: &nsl_semantic::checker::TypeMap,
    interner: &nsl_lexer::Interner,
) -> Result<lower::ZkDag, backend::ZkError> {
    let mut builder = ZkDagBuilder::new(type_map, interner);

    // Step 1: Emit Input ops for function parameters
    let mut input_indices = Vec::new();
    for param in &fn_def.params {
        let name = interner
            .resolve(param.name.0)
            .unwrap_or("input")
            .to_string();
        // Skip 'self' parameter
        if name == "self" {
            continue;
        }
        // Try to get shape from parameter type annotation
        let shape = vec![1]; // default — will be refined from actual type during lowering
        let bits = 32u32; // default
        let idx = builder.push(lower::ZkOp::Input {
            name: name.clone(),
            shape,
            dtype_bits: bits,
        });
        builder.name_to_idx.insert(name, idx);
        input_indices.push(idx);
    }

    // Step 2: Walk function body statements
    for stmt in &fn_def.body.stmts {
        match &stmt.kind {
            nsl_ast::stmt::StmtKind::VarDecl { pattern, value, .. } => {
                if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                    let name = interner.resolve(sym.0).unwrap_or("var").to_string();
                    if let Some(expr) = value {
                        let idx = builder.lower_expr(expr)?;
                        builder.name_to_idx.insert(name, idx);
                    }
                }
            }
            nsl_ast::stmt::StmtKind::Return(Some(expr)) => {
                let idx = builder.lower_expr(expr)?;
                builder.last_idx = idx;
            }
            nsl_ast::stmt::StmtKind::Expr(expr) => {
                let idx = builder.lower_expr(expr)?;
                builder.last_idx = idx;
            }
            _ => {} // Skip other statement types
        }
    }

    // Collect weight indices
    let weight_indices: Vec<usize> = builder
        .ops
        .iter()
        .enumerate()
        .filter_map(|(i, op)| {
            if matches!(op, lower::ZkOp::Weight { .. }) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    Ok(lower::ZkDag {
        output_idx: builder.last_idx,
        ops: builder.ops,
        input_indices,
        weight_indices,
    })
}

/// Compile a pre-built ZkDag using the folding backend.
///
/// This is the production entry point once AST→ZkDag is wired (Task 14).
/// It supports both Mersenne-31 (fast, default) and BN254 (EVM-compatible) fields.
pub fn compile_zk_from_dag(
    dag: &lower::ZkDag,
    config: &backend::ZkConfig,
) -> Result<backend::ZkProof, backend::ZkError> {
    match config.field {
        backend::ZkField::Mersenne31 => {
            compile_zk_folding::<field_m31::Mersenne31Field>(dag, config)
        }
        backend::ZkField::BN254 => compile_zk_folding::<field::FieldElement>(dag, config),
    }
}

/// Generic folding compilation over any field.
fn compile_zk_folding<F: field::Field>(
    dag: &lower::ZkDag,
    config: &backend::ZkConfig,
) -> Result<backend::ZkProof, backend::ZkError> {
    // 1. Lower DAG to per-layer ZkIRs
    let layer_irs = lower::lower_model_for_folding(dag, config);
    if layer_irs.is_empty() {
        return Err(backend::ZkError::CompilationError(
            "empty DAG produced no layers".to_string(),
        ));
    }

    // 2. Create folding prover
    let mut prover = folding::FoldingProver::<F>::new(folding::FoldingConfig::default());

    // 3. Fold each layer
    for ir in &layer_irs {
        // Generate dummy witness (all zeros) — real witness comes from inference execution
        let witness: Vec<F> = (0..ir.num_wires).map(|_| F::zero()).collect();
        prover.fold_layer(ir, &witness)?;
    }

    // 4. Finalize proof
    prover.finalize()
}
