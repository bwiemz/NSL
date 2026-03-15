//! M30: Tensor parallelism codegen — @shard extraction, DistState tracking.

/// Compile-time info about a sharded layer.
#[derive(Debug, Clone)]
pub struct ShardInfo {
    pub dim: usize,
}

/// Distributed state of an intermediate tensor activation.
#[derive(Clone, Debug, PartialEq)]
pub enum DistState {
    Replicated,
    Sharded { dim: usize },
}

/// Extract @shard decorator from a list of decorators.
pub fn extract_shard_decorator<'a>(
    decorators: &[nsl_ast::decl::Decorator],
    resolve_sym: &dyn Fn(nsl_ast::Symbol) -> &'a str,
) -> Option<ShardInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "shard" {
            let mut dim: usize = 0;
            if let Some(args) = &deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        let name = resolve_sym(name_sym);
                        if name == "dim" {
                            if let nsl_ast::expr::Expr {
                                kind: nsl_ast::expr::ExprKind::IntLiteral(v),
                                ..
                            } = &arg.value
                            {
                                dim = *v as usize;
                            }
                        }
                    }
                }
            }
            return Some(ShardInfo { dim });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist_state_default() {
        let state = DistState::Replicated;
        assert_eq!(state, DistState::Replicated);
    }

    #[test]
    fn test_dist_state_sharded() {
        let state = DistState::Sharded { dim: 1 };
        assert_eq!(state, DistState::Sharded { dim: 1 });
        assert_ne!(state, DistState::Replicated);
    }

    #[test]
    fn test_shard_info_clone() {
        let info = ShardInfo { dim: 2 };
        let cloned = info.clone();
        assert_eq!(cloned.dim, 2);
    }

    #[test]
    fn test_extract_shard_decorator_empty() {
        let decorators: Vec<nsl_ast::decl::Decorator> = vec![];
        let result = extract_shard_decorator(&decorators, &|_| "");
        assert!(result.is_none());
    }
}
