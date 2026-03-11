use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::types::{DeviceExpr, DimExpr, DimValue, TypeExpr, TypeExprKind};
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
            TypeExprKind::Function { params, ret } => Type::Function {
                params: params.iter().map(|p| self.resolve(p, scope)).collect(),
                ret: Box::new(self.resolve(ret, scope)),
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
                // For user-defined generics, just resolve the base name for now.
                // Full generic instantiation deferred to M3.
                self.resolve_named(name, span, scope)
            }
        }
    }

    fn resolve_shape(&self, dims: &[DimExpr]) -> Shape {
        Shape {
            dims: dims.iter().map(|d| self.resolve_dim(d)).collect(),
        }
    }

    fn resolve_dim(&self, dim: &DimExpr) -> Dim {
        match dim {
            DimExpr::Concrete(n) => Dim::Concrete(*n),
            DimExpr::Symbolic(sym) => Dim::Symbolic(*sym),
            DimExpr::Named { name, value } => {
                let size = match value {
                    DimValue::Int(n) => Dim::Concrete(*n),
                    DimValue::String(_) => Dim::Wildcard,
                };
                Dim::Named {
                    name: *name,
                    size: Box::new(size),
                }
            }
            DimExpr::Wildcard => Dim::Wildcard,
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
}
