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

#[derive(Debug, Clone, PartialEq)]
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

/// The single `@speculative` configuration a disaggregated worker slot can
/// carry, or `Err` with the conflicting keys when the program declares more
/// than one distinct configuration.
///
/// `compile_disaggregated_serve` builds ONE `WorkerConfigSpec` -- four scalars
/// baked into the emitted binary -- so it can represent exactly one
/// configuration. It used to choose with `configs.values().next()`, which picks
/// an arbitrary entry of a `HashMap`: with two differing `@speculative` layers
/// the compiler silently baked one of them, and which one changed from build to
/// build with the hash seed. Two compilations of identical source could emit
/// different decode behaviour.
///
/// Identical duplicates are NOT a conflict -- several layers may carry the same
/// decorator, and the worker can represent that faithfully. Only a genuine
/// disagreement is refused, because there is no answer the slot can hold.
///
/// The call-site path uses `moe::resolve_decorator_config_for_call_site`, which
/// scopes by model name; a serve block has no such context, so this is the
/// stricter rule rather than a different one.
pub(crate) fn unique_worker_config(
    configs: &std::collections::HashMap<String, SpeculativeInfo>,
) -> Result<Option<&SpeculativeInfo>, Vec<String>> {
    let mut it = configs.iter();
    let Some((_, first)) = it.next() else {
        return Ok(None);
    };
    if it.any(|(_, info)| info != first) {
        let mut keys: Vec<String> = configs.keys().cloned().collect();
        keys.sort();
        return Err(keys);
    }
    Ok(Some(first))
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

    fn info(num_tokens: usize, tree_width: usize) -> SpeculativeInfo {
        SpeculativeInfo {
            method: SpeculativeMethod::Draft,
            draft_model: None,
            num_tokens,
            temperature: 0.0,
            tree_width,
            token_budget: 0,
            expansion_k: 0,
            ngram_size: 0,
            lookahead_window: 0,
            medusa: false,
        }
    }

    fn map(entries: &[(&str, SpeculativeInfo)]) -> std::collections::HashMap<String, SpeculativeInfo> {
        entries.iter().map(|(k, v)| ((*k).to_string(), v.clone())).collect()
    }

    #[test]
    fn no_speculative_layers_means_no_worker_config() {
        assert_eq!(unique_worker_config(&map(&[])), Ok(None));
    }

    #[test]
    fn a_single_layer_is_the_worker_config() {
        let m = map(&[("M.draft", info(5, 4))]);
        assert_eq!(unique_worker_config(&m), Ok(Some(&info(5, 4))));
    }

    /// Several layers carrying the SAME decorator are not a conflict: the slot
    /// can represent that faithfully. Ten identical entries also exercise the
    /// determinism claim -- with `values().next()` the answer was whichever
    /// entry the hash seed surfaced, so a test like this could only ever assert
    /// "some entry".
    #[test]
    fn identical_duplicates_are_not_a_conflict() {
        let m = map(&[
            ("M.a", info(3, 2)),
            ("M.b", info(3, 2)),
            ("M.c", info(3, 2)),
        ]);
        assert_eq!(unique_worker_config(&m), Ok(Some(&info(3, 2))));
    }

    /// THE BUG. Two differing decorators used to bake an arbitrary one of them,
    /// chosen by HashMap iteration order, so two builds of identical source
    /// could emit different decode behaviour. The keys come back SORTED so the
    /// diagnostic is stable too.
    #[test]
    fn differing_configs_are_refused_with_sorted_keys() {
        let m = map(&[("M.b", info(5, 4)), ("M.a", info(3, 2))]);
        assert_eq!(
            unique_worker_config(&m),
            Err(vec!["M.a".to_string(), "M.b".to_string()])
        );
    }

    /// A difference in ANY field is a difference -- not just the two the old
    /// code happened to read first.
    #[test]
    fn a_difference_in_one_field_is_enough() {
        let mut other = info(5, 4);
        other.method = SpeculativeMethod::Eagle2;
        assert!(unique_worker_config(&map(&[("M.a", info(5, 4)), ("M.b", other)])).is_err());

        let mut temp = info(5, 4);
        temp.temperature = 0.7;
        assert!(unique_worker_config(&map(&[("M.a", info(5, 4)), ("M.b", temp)])).is_err());
    }
}
