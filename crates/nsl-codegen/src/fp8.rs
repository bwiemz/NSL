//! M35: FP8 compute codegen — @fp8_compute extraction.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

#[derive(Debug, Clone)]
pub struct Fp8ComputeInfo {
    pub calibrate: bool,
}

pub fn extract_fp8_compute_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<Fp8ComputeInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "fp8_compute" {
            let mut calibrate = false;
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        if resolve_sym(name_sym) == "calibrate" {
                            if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                                calibrate = *b;
                            }
                        }
                    }
                }
            }
            return Some(Fp8ComputeInfo { calibrate });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_empty() {
        assert!(extract_fp8_compute_decorator(&[], &|_| "").is_none());
    }
}
