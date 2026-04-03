use std::collections::HashMap;

use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::types::{DeviceExpr, DimExpr as AstDimExpr, DimValue, TypeExpr, TypeExprKind};
use nsl_ast::Symbol;
use nsl_errors::{Diagnostic, Span};
use nsl_lexer::Interner;

use crate::scope::{ScopeId, ScopeMap};
use crate::types::*;

pub struct TypeResolver<'a> {
    pub interner: &'a Interner,
    pub scopes: &'a ScopeMap,
    pub diagnostics: &'a mut Vec<Diagnostic>,
}

impl<'a> TypeResolver<'a> {
    /// Resolve a parsed TypeExpr into a semantic Type.
    pub fn resolve(&mut self, type_expr: &TypeExpr, scope: ScopeId) -> Type {
        match &type_expr.kind {
            TypeExprKind::Named(sym) => self.resolve_named(*sym, type_expr.span, scope),
            TypeExprKind::Generic { name, args } => {
                self.resolve_generic(*name, args, type_expr.span, scope)
            }
            TypeExprKind::Tensor {
                shape,
                dtype,
                device,
            } => Type::Tensor {
                shape: self.resolve_shape(shape),
                dtype: self.resolve_dtype(*dtype),
                device: self.resolve_device(device),
            },
            TypeExprKind::Param { shape, dtype } => Type::Param {
                shape: self.resolve_shape(shape),
                dtype: self.resolve_dtype(*dtype),
            },
            TypeExprKind::Buffer { shape, dtype } => Type::Buffer {
                shape: self.resolve_shape(shape),
                dtype: self.resolve_dtype(*dtype),
            },
            TypeExprKind::Sparse {
                shape,
                dtype,
                format,
            } => Type::Sparse {
                shape: self.resolve_shape(shape),
                dtype: self.resolve_dtype(*dtype),
                format: self.resolve_sparse_format(*format),
            },
            TypeExprKind::Function { params, ret, effect } => {
                let resolved_effect = if let Some(eff_expr) = effect {
                    self.resolve_effect_expr(eff_expr, scope)
                } else {
                    Effect::Inferred
                };
                Type::Function {
                    params: params.iter().map(|p| self.resolve(p, scope)).collect(),
                    ret: Box::new(self.resolve(ret, scope)),
                    effect: resolved_effect,
                }
            },
            TypeExprKind::Union(types) => {
                Type::Union(types.iter().map(|t| self.resolve(t, scope)).collect())
            }
            TypeExprKind::Tuple(types) => {
                Type::Tuple(types.iter().map(|t| self.resolve(t, scope)).collect())
            }
            TypeExprKind::Wildcard => Type::Unknown,
            TypeExprKind::FixedArray { element_type, size } => {
                let elem = self.resolve(element_type, scope);
                if let Type::Model { name, .. } = &elem {
                    Type::FixedModelArray { element_model: *name, size: *size }
                } else {
                    // For now, only model arrays supported
                    Type::Unknown
                }
            }
            TypeExprKind::Borrow(inner) => {
                let inner_ty = self.resolve(inner, scope);
                Type::Borrow(Box::new(inner_ty))
            }
        }
    }

    fn resolve_named(&mut self, sym: Symbol, span: Span, scope: ScopeId) -> Type {
        let name = self.interner.resolve(sym.0).unwrap_or("");
        match name {
            "int" => Type::Int,
            "float" => Type::Float,
            "bool" => Type::Bool,
            "str" => Type::Str,
            "void" => Type::Void,
            "f32" => Type::F32,
            "f64" => Type::F64,
            "fp16" => Type::Fp16,
            "bf16" => Type::Bf16,
            "fp8_e4m3" => Type::Fp8E4m3,
            "fp8_e5m2" => Type::Fp8E5m2,
            "int8" => Type::Int8,
            "int16" => Type::Int16,
            "int32" => Type::Int32,
            "int64" => Type::Int64,
            "int4" => Type::Int4,
            "uint8" => Type::Uint8,
            // Unparameterized tensor family: bare `Tensor`, `Param`, `Buffer`
            "Tensor" => Type::Tensor {
                shape: Shape::unknown(),
                dtype: DType::Unknown,
                device: Device::Unknown,
            },
            "Param" => Type::Param {
                shape: Shape::unknown(),
                dtype: DType::Unknown,
            },
            "Buffer" => Type::Buffer {
                shape: Shape::unknown(),
                dtype: DType::Unknown,
            },
            _ => {
                // Look up in scope for user-defined types
                if let Some((_sid, info)) = self.scopes.lookup(scope, sym) {
                    info.ty.clone()
                } else {
                    self.diagnostics.push(
                        Diagnostic::error(format!("undefined type `{name}`"))
                            .with_label(span, "not found in scope"),
                    );
                    Type::Error
                }
            }
        }
    }

    fn resolve_generic(
        &mut self,
        name: Symbol,
        args: &[TypeExpr],
        span: Span,
        scope: ScopeId,
    ) -> Type {
        let name_str = self.interner.resolve(name.0).unwrap_or("").to_string();
        match name_str.as_str() {
            "list" | "List" => {
                if args.len() == 1 {
                    Type::List(Box::new(self.resolve(&args[0], scope)))
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("list<T> expects exactly 1 type argument")
                            .with_label(span, "wrong number of type arguments"),
                    );
                    Type::Error
                }
            }
            "dict" | "Dict" => {
                if args.len() == 2 {
                    Type::Dict(
                        Box::new(self.resolve(&args[0], scope)),
                        Box::new(self.resolve(&args[1], scope)),
                    )
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("dict<K, V> expects exactly 2 type arguments")
                            .with_label(span, "wrong number of type arguments"),
                    );
                    Type::Error
                }
            }
            "Optional" => {
                if args.len() == 1 {
                    Type::Optional(Box::new(self.resolve(&args[0], scope)))
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("Optional<T> expects exactly 1 type argument")
                            .with_label(span, "wrong number of type arguments"),
                    );
                    Type::Error
                }
            }
            _ => {
                let base = self.resolve_named(name, span, scope);
                match base {
                    Type::Struct {
                        name,
                        type_params,
                        fields,
                        ..
                    } => {
                        let Some((type_args, bindings)) =
                            self.resolve_user_bindings(&name_str, &type_params, args, span, scope)
                        else {
                            return Type::Error;
                        };
                        Type::Struct {
                            name,
                            type_params,
                            type_args,
                            fields: fields
                                .iter()
                                .map(|(field_name, field_ty)| {
                                    (*field_name, substitute_type(field_ty, &bindings))
                                })
                                .collect(),
                        }
                    }
                    Type::Enum {
                        name,
                        type_params,
                        variants,
                        ..
                    } => {
                        let Some((type_args, bindings)) =
                            self.resolve_user_bindings(&name_str, &type_params, args, span, scope)
                        else {
                            return Type::Error;
                        };
                        Type::Enum {
                            name,
                            type_params,
                            type_args,
                            variants: variants
                                .iter()
                                .map(|(variant_name, field_tys)| {
                                    (
                                        *variant_name,
                                        field_tys
                                            .iter()
                                            .map(|field_ty| substitute_type(field_ty, &bindings))
                                            .collect(),
                                    )
                                })
                                .collect(),
                        }
                    }
                    Type::Trait {
                        name,
                        type_params,
                        methods,
                        ..
                    } => {
                        let Some((type_args, bindings)) =
                            self.resolve_user_bindings(&name_str, &type_params, args, span, scope)
                        else {
                            return Type::Error;
                        };
                        Type::Trait {
                            name,
                            type_params,
                            type_args,
                            methods: methods
                                .iter()
                                .map(|(method_name, method_ty)| {
                                    (*method_name, substitute_type(method_ty, &bindings))
                                })
                                .collect(),
                        }
                    }
                    Type::Model {
                        name,
                        type_params,
                        fields,
                        methods,
                        ..
                    } => {
                        let Some((type_args, bindings)) =
                            self.resolve_user_bindings(&name_str, &type_params, args, span, scope)
                        else {
                            return Type::Error;
                        };
                        Type::Model {
                            name,
                            type_params,
                            type_args,
                            fields: fields
                                .iter()
                                .map(|(field_name, field_ty)| {
                                    (*field_name, substitute_type(field_ty, &bindings))
                                })
                                .collect(),
                            methods: methods
                                .iter()
                                .map(|(method_name, method_ty)| {
                                    (*method_name, substitute_type(method_ty, &bindings))
                                })
                                .collect(),
                        }
                    }
                    Type::Unknown | Type::Error => base,
                    _ => {
                        self.diagnostics.push(
                            Diagnostic::error(format!("type `{name_str}` is not generic"))
                                .with_label(span, "unexpected type arguments"),
                        );
                        Type::Error
                    }
                }
            }
        }
    }

    fn resolve_user_bindings(
        &mut self,
        name: &str,
        type_params: &[Symbol],
        args: &[TypeExpr],
        span: Span,
        scope: ScopeId,
    ) -> Option<(Vec<Type>, HashMap<Symbol, Type>)> {
        if type_params.is_empty() {
            self.diagnostics.push(
                Diagnostic::error(format!("type `{name}` is not generic"))
                    .with_label(span, "unexpected type arguments"),
            );
            return None;
        }

        if type_params.len() != args.len() {
            self.diagnostics.push(
                Diagnostic::error(format!(
                    "{name} expects {} type argument(s), got {}",
                    type_params.len(),
                    args.len()
                ))
                .with_label(span, "wrong number of type arguments"),
            );
            return None;
        }

        let type_args: Vec<Type> = args.iter().map(|arg| self.resolve(arg, scope)).collect();
        let bindings = type_params
            .iter()
            .copied()
            .zip(type_args.iter().cloned())
            .collect();
        Some((type_args, bindings))
    }

    fn resolve_shape(&self, dims: &[AstDimExpr]) -> Shape {
        Shape {
            dims: dims.iter().map(|d| self.resolve_dim(d)).collect(),
        }
    }

    fn resolve_dim(&self, dim: &AstDimExpr) -> Dim {
        match dim {
            AstDimExpr::Concrete(n) => Dim::Concrete(*n),
            AstDimExpr::Symbolic(sym) => Dim::Symbolic(*sym),
            AstDimExpr::Named { name, value } => {
                let size = match value {
                    DimValue::Int(n) => Dim::Concrete(*n),
                    DimValue::String(_) => Dim::Wildcard,
                };
                Dim::Named {
                    name: *name,
                    size: Box::new(size),
                }
            }
            AstDimExpr::Bounded { name, upper_bound } => {
                Dim::Bounded { name: *name, upper_bound: *upper_bound }
            }
            AstDimExpr::Wildcard => Dim::Wildcard,
        }
    }

    pub fn resolve_dtype(&self, sym: Symbol) -> DType {
        let name = self.interner.resolve(sym.0).unwrap_or("");
        match name {
            "f64" | "fp64" => DType::F64,
            "f32" | "fp32" => DType::F32,
            "fp16" => DType::Fp16,
            "bf16" => DType::Bf16,
            "fp8_e4m3" => DType::Fp8E4m3,
            "fp8_e5m2" => DType::Fp8E5m2,
            "int64" => DType::Int64,
            "int32" => DType::Int32,
            "int16" => DType::Int16,
            "int8" => DType::Int8,
            "int4" => DType::Int4,
            "uint8" => DType::Uint8,
            "bool" => DType::Bool,
            _ => DType::Unknown,
        }
    }

    fn resolve_device(&self, device: &Option<DeviceExpr>) -> Device {
        match device {
            None => Device::Unknown,
            Some(DeviceExpr::Cpu) => Device::Cpu,
            Some(DeviceExpr::Cuda(None)) => Device::Cuda(None),
            Some(DeviceExpr::Cuda(Some(idx_expr))) => {
                Device::Cuda(self.extract_device_index(idx_expr))
            }
            Some(DeviceExpr::Metal) => Device::Metal,
            Some(DeviceExpr::Rocm(None)) => Device::Rocm(None),
            Some(DeviceExpr::Rocm(Some(idx_expr))) => {
                Device::Rocm(self.extract_device_index(idx_expr))
            }
            Some(DeviceExpr::Npu(sym)) => Device::Npu(*sym),
        }
    }

    fn extract_device_index(&self, expr: &Expr) -> Option<u32> {
        if let ExprKind::IntLiteral(n) = &expr.kind {
            Some(*n as u32)
        } else {
            None
        }
    }

    fn resolve_sparse_format(&self, sym: Symbol) -> SparseFormat {
        let name = self.interner.resolve(sym.0).unwrap_or("");
        match name {
            "coo" => SparseFormat::Coo,
            "csr" => SparseFormat::Csr,
            "csc" => SparseFormat::Csc,
            "bsr" => SparseFormat::Bsr,
            _ => SparseFormat::Unknown,
        }
    }

    /// Resolve an effect expression into a semantic Effect.
    pub fn resolve_effect_expr(&self, eff: &nsl_ast::decl::EffectExpr, _scope: ScopeId) -> Effect {
        use nsl_ast::decl::EffectExpr;
        use crate::effects::EffectSet;

        match eff {
            EffectExpr::Var(sym) => {
                let name = self.interner.resolve(sym.0).unwrap_or("");
                // Check if it's a known concrete effect name
                match name {
                    "IO" => Effect::Concrete(EffectSet::IO),
                    "RANDOM" | "Random" => Effect::Concrete(EffectSet::RANDOM),
                    "MUTATION" | "Mutation" => Effect::Concrete(EffectSet::MUTATION),
                    "COMMUNICATION" | "Communication" => Effect::Concrete(EffectSet::COMMUNICATION),
                    "PURE" | "Pure" => Effect::Concrete(EffectSet::PURE),
                    _ => Effect::Var(*sym), // treat as effect variable
                }
            }
            EffectExpr::Named(sym) => {
                let name = self.interner.resolve(sym.0).unwrap_or("");
                match name {
                    "IO" => Effect::Concrete(EffectSet::IO),
                    "RANDOM" | "Random" => Effect::Concrete(EffectSet::RANDOM),
                    "MUTATION" | "Mutation" => Effect::Concrete(EffectSet::MUTATION),
                    "COMMUNICATION" | "Communication" => Effect::Concrete(EffectSet::COMMUNICATION),
                    _ => Effect::Concrete(EffectSet::PURE),
                }
            }
            EffectExpr::Union(parts) => {
                let resolved: Vec<Effect> = parts.iter()
                    .map(|p| self.resolve_effect_expr(p, _scope))
                    .collect();
                // Fold into nested unions
                resolved.into_iter().reduce(|a, b| {
                    Effect::Union(Box::new(a), Box::new(b))
                }).unwrap_or(Effect::pure())
            }
        }
    }
}

fn nested_generic_bindings(
    bindings: &HashMap<Symbol, Type>,
    type_params: &[Symbol],
    type_args: &[Type],
) -> HashMap<Symbol, Type> {
    // An uninstantiated nested nominal type still owns its own type parameters.
    // Drop same-named outer bindings so Wrapper<T> containing Box<T> keeps Box
    // generic until Box itself is instantiated.
    if type_args.is_empty() && !type_params.is_empty() {
        let mut shadowed = bindings.clone();
        for param in type_params {
            shadowed.remove(param);
        }
        shadowed
    } else {
        bindings.clone()
    }
}

fn substitute_type(ty: &Type, bindings: &HashMap<Symbol, Type>) -> Type {
    match ty {
        Type::TypeVar(sym) => bindings.get(sym).cloned().unwrap_or(Type::TypeVar(*sym)),
        Type::List(inner) => Type::List(Box::new(substitute_type(inner, bindings))),
        Type::Dict(key, value) => Type::Dict(
            Box::new(substitute_type(key, bindings)),
            Box::new(substitute_type(value, bindings)),
        ),
        Type::Tuple(items) => {
            Type::Tuple(items.iter().map(|item| substitute_type(item, bindings)).collect())
        }
        Type::Optional(inner) => Type::Optional(Box::new(substitute_type(inner, bindings))),
        Type::Function { params, ret, effect } => Type::Function {
            params: params.iter().map(|param| substitute_type(param, bindings)).collect(),
            ret: Box::new(substitute_type(ret, bindings)),
            effect: effect.clone(),
        },
        Type::Borrow(inner) => Type::Borrow(Box::new(substitute_type(inner, bindings))),
        Type::Struct {
            name,
            type_params,
            type_args,
            fields,
        } => {
            let effective_bindings = nested_generic_bindings(bindings, type_params, type_args);
            Type::Struct {
                name: *name,
                type_params: type_params.clone(),
                type_args: type_args
                    .iter()
                    .map(|arg| substitute_type(arg, &effective_bindings))
                    .collect(),
                fields: fields
                    .iter()
                    .map(|(field_name, field_ty)| {
                        (*field_name, substitute_type(field_ty, &effective_bindings))
                    })
                    .collect(),
            }
        }
        Type::Enum {
            name,
            type_params,
            type_args,
            variants,
        } => {
            let effective_bindings = nested_generic_bindings(bindings, type_params, type_args);
            Type::Enum {
                name: *name,
                type_params: type_params.clone(),
                type_args: type_args
                    .iter()
                    .map(|arg| substitute_type(arg, &effective_bindings))
                    .collect(),
                variants: variants
                    .iter()
                    .map(|(variant_name, field_tys)| {
                        (
                            *variant_name,
                            field_tys
                                .iter()
                                .map(|field_ty| substitute_type(field_ty, &effective_bindings))
                                .collect(),
                        )
                    })
                    .collect(),
            }
        }
        Type::Trait {
            name,
            type_params,
            type_args,
            methods,
        } => {
            let effective_bindings = nested_generic_bindings(bindings, type_params, type_args);
            Type::Trait {
                name: *name,
                type_params: type_params.clone(),
                type_args: type_args
                    .iter()
                    .map(|arg| substitute_type(arg, &effective_bindings))
                    .collect(),
                methods: methods
                    .iter()
                    .map(|(method_name, method_ty)| {
                        (*method_name, substitute_type(method_ty, &effective_bindings))
                    })
                    .collect(),
            }
        }
        Type::Model {
            name,
            type_params,
            type_args,
            fields,
            methods,
        } => {
            let effective_bindings = nested_generic_bindings(bindings, type_params, type_args);
            Type::Model {
                name: *name,
                type_params: type_params.clone(),
                type_args: type_args
                    .iter()
                    .map(|arg| substitute_type(arg, &effective_bindings))
                    .collect(),
                fields: fields
                    .iter()
                    .map(|(field_name, field_ty)| {
                        (*field_name, substitute_type(field_ty, &effective_bindings))
                    })
                    .collect(),
                methods: methods
                    .iter()
                    .map(|(method_name, method_ty)| {
                        (*method_name, substitute_type(method_ty, &effective_bindings))
                    })
                    .collect(),
            }
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use string_interner::backend::BucketBackend;
    use string_interner::{DefaultSymbol, StringInterner};

    type TestInterner = StringInterner<BucketBackend<DefaultSymbol>>;

    fn sym(interner: &mut TestInterner, name: &str) -> Symbol {
        Symbol(interner.get_or_intern(name))
    }

    #[test]
    fn substitute_type_instantiates_generic_enum_variants() {
        let mut interner = TestInterner::new();
        let t = sym(&mut interner, "T");
        let e = sym(&mut interner, "E");
        let result = sym(&mut interner, "Result");
        let ok = sym(&mut interner, "Ok");
        let err = sym(&mut interner, "Err");

        let ty = Type::Enum {
            name: result,
            type_params: vec![t, e],
            type_args: vec![Type::TypeVar(t), Type::TypeVar(e)],
            variants: vec![(ok, vec![Type::TypeVar(t)]), (err, vec![Type::TypeVar(e)])],
        };

        let bindings = HashMap::from([(t, Type::Int), (e, Type::Str)]);

        assert_eq!(
            substitute_type(&ty, &bindings),
            Type::Enum {
                name: result,
                type_params: vec![t, e],
                type_args: vec![Type::Int, Type::Str],
                variants: vec![(ok, vec![Type::Int]), (err, vec![Type::Str])],
            }
        );
    }

    #[test]
    fn substitute_type_preserves_uninstantiated_inner_generic_type_params() {
        let mut interner = TestInterner::new();
        let t = sym(&mut interner, "T");
        let box_name = sym(&mut interner, "Box");
        let value = sym(&mut interner, "value");

        let ty = Type::Struct {
            name: box_name,
            type_params: vec![t],
            type_args: Vec::new(),
            fields: vec![(value, Type::TypeVar(t))],
        };

        let bindings = HashMap::from([(t, Type::Int)]);

        assert_eq!(substitute_type(&ty, &bindings), ty);
    }
}
