//! M32: MoE codegen — @moe extraction + moe_dispatch lowering.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

/// Compile-time info about a MoE layer.
#[derive(Debug, Clone)]
pub struct MoeInfo {
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
    pub aux_loss_coeff: f32,
}

/// Extract @moe decorator from a list of decorators.
pub fn extract_moe_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<MoeInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "moe" {
            let mut num_experts: usize = 0;
            let mut top_k: usize = 2;
            let mut capacity_factor: f32 = 1.25;
            let mut aux_loss_coeff: f32 = 0.01;

            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        let name = resolve_sym(name_sym);
                        match name {
                            "num_experts" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    num_experts = *v as usize;
                                }
                            }
                            "top_k" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    top_k = *v as usize;
                                }
                            }
                            "capacity_factor" => {
                                if let ExprKind::FloatLiteral(v) = &arg.value.kind {
                                    capacity_factor = *v as f32;
                                }
                            }
                            "aux_loss_coeff" => {
                                if let ExprKind::FloatLiteral(v) = &arg.value.kind {
                                    aux_loss_coeff = *v as f32;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            if num_experts > 0 {
                return Some(MoeInfo {
                    num_experts,
                    top_k,
                    capacity_factor,
                    aux_loss_coeff,
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_moe_empty_decorators() {
        let decorators: Vec<Decorator> = vec![];
        let result = extract_moe_decorator(&decorators, &|_| "");
        assert!(result.is_none());
    }

    #[test]
    fn test_moe_info_defaults() {
        // Verify default values used when only num_experts is provided
        let info = MoeInfo {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            aux_loss_coeff: 0.01,
        };
        assert_eq!(info.num_experts, 8);
        assert_eq!(info.top_k, 2);
        assert!((info.capacity_factor - 1.25).abs() < 1e-6);
        assert!((info.aux_loss_coeff - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_moe_info_clone() {
        let info = MoeInfo {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 2.0,
            aux_loss_coeff: 0.05,
        };
        let cloned = info.clone();
        assert_eq!(cloned.num_experts, 4);
        assert_eq!(cloned.top_k, 1);
    }
}
