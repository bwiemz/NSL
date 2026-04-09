//! M33: Speculative decoding codegen — @speculative extraction.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

/// Speculative decoding method.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeculativeMethod {
    /// Standard draft model (autoregressive K-token generation)
    Draft,
    /// Medusa (multi-head parallel draft)
    Medusa,
    /// EAGLE-2 (dynamic confidence-scored tree expansion)
    Eagle2,
    /// Lookahead (n-gram based, no draft model needed)
    Lookahead,
}

#[derive(Debug, Clone)]
pub struct SpeculativeInfo {
    pub method: SpeculativeMethod,
    pub draft_model: Option<String>,
    pub num_tokens: usize,
    pub temperature: f32,
    pub tree_width: usize,
    /// EAGLE-2: token budget for dynamic tree expansion
    pub token_budget: usize,
    /// EAGLE-2: top-k expansion factor per node
    pub expansion_k: usize,
    /// Lookahead: n-gram size
    pub ngram_size: usize,
    /// Lookahead: lookahead window
    pub lookahead_window: usize,
    /// Backward compat — true if method == Medusa
    pub medusa: bool,
}

pub fn extract_speculative_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<SpeculativeInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "speculative" {
            let mut draft_model: Option<String> = None;
            let mut num_tokens: usize = 5;
            let mut temperature: f32 = 0.0;
            let mut tree_width: usize = 1;
            let mut medusa = false;
            let mut method_str = String::new();
            let mut token_budget: usize = 60;
            let mut expansion_k: usize = 10;
            let mut ngram_size: usize = 3;
            let mut lookahead_window: usize = 5;

            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        let name = resolve_sym(name_sym);
                        match name {
                            "draft_model" => {
                                if let ExprKind::StringLiteral(ref s) = arg.value.kind {
                                    draft_model = Some(s.clone());
                                }
                            }
                            "method" => {
                                if let ExprKind::StringLiteral(ref s) = arg.value.kind {
                                    method_str = s.clone();
                                }
                            }
                            "num_tokens" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    num_tokens = *v as usize;
                                }
                            }
                            "temperature" => {
                                if let ExprKind::FloatLiteral(v) = &arg.value.kind {
                                    temperature = *v as f32;
                                }
                            }
                            "tree_width" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    tree_width = *v as usize;
                                }
                            }
                            "token_budget" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    token_budget = *v as usize;
                                }
                            }
                            "expansion_k" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    expansion_k = *v as usize;
                                }
                            }
                            "ngram" | "ngram_size" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    ngram_size = *v as usize;
                                }
                            }
                            "window" | "lookahead_window" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    lookahead_window = *v as usize;
                                }
                            }
                            "medusa" => {
                                medusa = true;
                            }
                            _ => {}
                        }
                    }
                }
            }

            let method = match method_str.as_str() {
                "eagle2" => SpeculativeMethod::Eagle2,
                "lookahead" => SpeculativeMethod::Lookahead,
                "medusa" => {
                    medusa = true;
                    SpeculativeMethod::Medusa
                }
                _ if medusa => SpeculativeMethod::Medusa,
                _ => SpeculativeMethod::Draft,
            };

            return Some(SpeculativeInfo {
                method,
                draft_model,
                num_tokens,
                temperature,
                tree_width,
                token_budget,
                expansion_k,
                ngram_size,
                lookahead_window,
                medusa,
            });
        }
    }
    None
}

/// Compile-time info about Medusa multi-head speculation.
#[derive(Debug, Clone)]
pub struct MedusaInfo {
    pub num_heads: usize,
    pub tree_width: usize,
}

/// Extract @medusa decorator from a list of decorators.
pub fn extract_medusa_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<MedusaInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "medusa" {
            let mut num_heads: usize = 0;
            let mut tree_width: usize = 1;

            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        let name = resolve_sym(name_sym);
                        match name {
                            "num_heads" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    num_heads = *v as usize;
                                }
                            }
                            "tree_width" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    tree_width = *v as usize;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            if num_heads > 0 {
                return Some(MedusaInfo {
                    num_heads,
                    tree_width,
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
    fn test_extract_empty() {
        let result = extract_speculative_decorator(&[], &|_| "");
        assert!(result.is_none());
    }

    #[test]
    fn test_speculative_info_defaults() {
        let info = SpeculativeInfo {
            method: SpeculativeMethod::Draft,
            draft_model: Some("draft".to_string()),
            num_tokens: 5,
            temperature: 0.0,
            tree_width: 1,
            token_budget: 60,
            expansion_k: 10,
            ngram_size: 3,
            lookahead_window: 5,
            medusa: false,
        };
        assert_eq!(info.num_tokens, 5);
        assert_eq!(info.method, SpeculativeMethod::Draft);
        assert!(!info.medusa);
    }
}
