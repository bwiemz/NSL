//! M45: Compile-time NaN risk analysis.

use std::collections::HashMap;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Known value constraint for a tensor binding.
#[derive(Clone, Debug, PartialEq)]
pub enum ValueConstraint {
    Unconstrained,
    NonNegative,        // >= 0 (from relu, abs, x*x)
    StrictlyPositive,   // > 0  (from relu(x) + eps, softplus, exp)
}

/// Analyzes function bodies for NaN/Inf risk patterns.
pub struct NanAnalyzer {
    constraints: HashMap<Symbol, ValueConstraint>,
    pub diagnostics: Vec<Diagnostic>,
}

impl Default for NanAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl NanAnalyzer {
    pub fn new() -> Self {
        NanAnalyzer { constraints: HashMap::new(), diagnostics: Vec::new() }
    }

    /// Mark a binding as having a known constraint.
    pub fn set_constraint(&mut self, sym: Symbol, constraint: ValueConstraint) {
        self.constraints.insert(sym, constraint);
    }

    /// Get the constraint for a binding.
    pub fn get_constraint(&self, sym: &Symbol) -> ValueConstraint {
        self.constraints.get(sym).cloned().unwrap_or(ValueConstraint::Unconstrained)
    }

    /// Check if a function call is a NaN risk given its argument constraints.
    ///
    /// Returns a warning diagnostic if a risk is detected, None otherwise.
    pub fn check_call_risk(
        &self,
        func_name: &str,
        arg_syms: &[Symbol],
        span: nsl_errors::Span,
    ) -> Option<Diagnostic> {
        match func_name {
            "log" => {
                if let Some(arg) = arg_syms.first() {
                    let c = self.get_constraint(arg);
                    if c != ValueConstraint::StrictlyPositive {
                        return Some(
                            Diagnostic::warning("log() argument may be zero or negative — NaN risk")
                                .with_label(span, "here")
                        );
                    }
                }
                None
            }
            "sqrt" => {
                if let Some(arg) = arg_syms.first() {
                    let c = self.get_constraint(arg);
                    if c == ValueConstraint::Unconstrained {
                        return Some(
                            Diagnostic::warning("sqrt() argument may be negative — NaN risk")
                                .with_label(span, "here")
                        );
                    }
                }
                None
            }
            // Division: second argument could be zero -> Inf
            "div" => {
                if arg_syms.len() >= 2 {
                    let c = self.get_constraint(&arg_syms[1]);
                    if c != ValueConstraint::StrictlyPositive {
                        return Some(
                            Diagnostic::warning("division by tensor that may contain zeros — Inf risk")
                                .with_label(span, "here")
                        );
                    }
                }
                None
            }
            // pow and log(softmax(...)) detection deferred to M45b
            _ => None,
        }
    }

    /// Infer constraints from known function results.
    pub fn infer_constraint(&self, func_name: &str) -> ValueConstraint {
        match func_name {
            "relu" | "abs" => ValueConstraint::NonNegative,
            "exp" | "softplus" => ValueConstraint::StrictlyPositive,
            "sigmoid" => ValueConstraint::StrictlyPositive, // (0, 1)
            _ => ValueConstraint::Unconstrained,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::Span;
    use nsl_lexer::Interner;

    /// Helper to create a Symbol from a string via a string interner.
    fn make_sym(interner: &mut Interner, name: &str) -> Symbol {
        Symbol(interner.get_or_intern(name))
    }

    #[test]
    fn log_unconstrained_warns() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let analyzer = NanAnalyzer::new();
        let result = analyzer.check_call_risk("log", &[x], Span::dummy());
        assert!(result.is_some(), "log() on unconstrained arg should warn");
    }

    #[test]
    fn log_strictly_positive_no_warn() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut analyzer = NanAnalyzer::new();
        analyzer.set_constraint(x, ValueConstraint::StrictlyPositive);
        let result = analyzer.check_call_risk("log", &[x], Span::dummy());
        assert!(result.is_none(), "log() on StrictlyPositive arg should not warn");
    }

    #[test]
    fn sqrt_unconstrained_warns() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let analyzer = NanAnalyzer::new();
        let result = analyzer.check_call_risk("sqrt", &[x], Span::dummy());
        assert!(result.is_some(), "sqrt() on unconstrained arg should warn");
    }

    #[test]
    fn sqrt_non_negative_no_warn() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut analyzer = NanAnalyzer::new();
        analyzer.set_constraint(x, ValueConstraint::NonNegative);
        let result = analyzer.check_call_risk("sqrt", &[x], Span::dummy());
        assert!(result.is_none(), "sqrt() on NonNegative arg should not warn");
    }

    #[test]
    fn div_unconstrained_warns() {
        let mut interner = Interner::new();
        let a = make_sym(&mut interner, "a");
        let b = make_sym(&mut interner, "b");
        let analyzer = NanAnalyzer::new();
        let result = analyzer.check_call_risk("div", &[a, b], Span::dummy());
        assert!(result.is_some(), "div() with unconstrained divisor should warn");
    }

    #[test]
    fn relu_produces_non_negative() {
        let analyzer = NanAnalyzer::new();
        assert_eq!(analyzer.infer_constraint("relu"), ValueConstraint::NonNegative);
        assert_eq!(analyzer.infer_constraint("abs"), ValueConstraint::NonNegative);
        assert_eq!(analyzer.infer_constraint("exp"), ValueConstraint::StrictlyPositive);
        assert_eq!(analyzer.infer_constraint("sigmoid"), ValueConstraint::StrictlyPositive);
        assert_eq!(analyzer.infer_constraint("softplus"), ValueConstraint::StrictlyPositive);
        assert_eq!(analyzer.infer_constraint("unknown_fn"), ValueConstraint::Unconstrained);
    }
}
