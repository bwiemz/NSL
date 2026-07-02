//! M34: Context parallelism codegen — @context_parallel extraction.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

#[derive(Debug, Clone)]
pub struct ContextParallelInfo {
    pub ring_size: usize,
}

pub fn extract_context_parallel_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<ContextParallelInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "context_parallel" {
            let mut ring_size: usize = 0;
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        if resolve_sym(name_sym) == "ring_size" {
                            if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                // M34 v1: accept ring_size >= 1 (semantic
                                // identity for r=1). The semantic layer at
                                // crates/nsl-semantic/src/context_parallel.rs
                                // enforces >= 1 with a user-facing diagnostic;
                                // r=0 never reaches here.
                                if *v >= 1 {
                                    ring_size = *v as usize;
                                }
                            }
                        }
                    }
                }
            }
            if ring_size > 0 {
                return Some(ContextParallelInfo { ring_size });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_empty() {
        assert!(extract_context_parallel_decorator(&[], &|_| "").is_none());
    }
}
