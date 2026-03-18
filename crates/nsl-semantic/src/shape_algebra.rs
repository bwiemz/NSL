//! M49: Compile-time shape algebra — symbolic dimension solver.
//!
//! Proves reshape/view/split correctness at compile time by maintaining
//! a constraint database of equalities, divisibility facts, and range bounds.
//!
//! Uses the EXISTING `DimExpr` from `crate::types`.

use std::collections::HashMap;
use nsl_ast::Symbol;
use crate::types::DimExpr;

// ---------------------------------------------------------------------------
// ProofFailure — explains why a proof could not be completed
// ---------------------------------------------------------------------------

/// Explanation of why a shape proof failed.
#[derive(Debug, Clone)]
pub struct ProofFailure {
    /// What was being proved.
    pub goal: String,
    /// What the solver tried.
    pub attempts: Vec<String>,
    /// Why it failed.
    pub reason: String,
}

impl std::fmt::Display for ProofFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot prove: {}\n  reason: {}", self.goal, self.reason)?;
        for attempt in &self.attempts {
            write!(f, "\n  tried: {}", attempt)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ShapeAlgebraSolver
// ---------------------------------------------------------------------------

/// Compile-time symbolic dimension solver.
///
/// Maintains a constraint database and proves shape relationships:
/// - Equality: S*D == S*D (for reshape validation)
/// - Divisibility: D % H == 0 (for multi-head attention split)
/// - Range: S > 0, B <= 256
pub struct ShapeAlgebraSolver {
    /// Concrete bindings: symbol = value.
    bindings: HashMap<Symbol, i64>,
    /// Known divisibility facts: symbol is divisible by these divisors.
    divisibility: HashMap<Symbol, Vec<i64>>,
    /// Known range bounds: symbol in [lower, upper].
    /// NOTE: Stored but not yet queried by prove_eq/prove_divisible.
    /// prove_bound() and range reasoning deferred to M49b.
    #[allow(dead_code)]
    bounds: HashMap<Symbol, (Option<i64>, Option<i64>)>,
    /// Known equalities between expressions.
    equalities: Vec<(DimExpr, DimExpr)>,
}

impl ShapeAlgebraSolver {
    pub fn new() -> Self {
        ShapeAlgebraSolver {
            bindings: HashMap::new(),
            divisibility: HashMap::new(),
            bounds: HashMap::new(),
            equalities: Vec::new(),
        }
    }

    /// Bind a symbol to a concrete value: D = 768.
    pub fn bind(&mut self, sym: Symbol, value: i64) {
        self.bindings.insert(sym, value);
    }

    /// Assert divisibility: D % divisor == 0.
    pub fn assert_divisible(&mut self, sym: Symbol, divisor: i64) {
        self.divisibility.entry(sym).or_default().push(divisor);
    }

    /// Assert a range bound: sym in [lower, upper].
    pub fn assert_bound(&mut self, sym: Symbol, lower: Option<i64>, upper: Option<i64>) {
        self.bounds.insert(sym, (lower, upper));
    }

    /// Assert equality between two expressions.
    pub fn assert_eq(&mut self, lhs: DimExpr, rhs: DimExpr) {
        self.equalities.push((lhs, rhs));
    }

    /// Evaluate a DimExpr to a concrete value if all symbols are bound.
    pub fn evaluate(&self, expr: &DimExpr) -> Option<i64> {
        match expr {
            DimExpr::Lit(v) => Some(*v),
            DimExpr::Sym(s) => self.bindings.get(s).copied(),
            DimExpr::Add(a, b) => {
                let va = self.evaluate(a)?;
                let vb = self.evaluate(b)?;
                Some(va + vb)
            }
            DimExpr::Mul(a, b) => {
                let va = self.evaluate(a)?;
                let vb = self.evaluate(b)?;
                Some(va * vb)
            }
            DimExpr::Div(a, b) => {
                let va = self.evaluate(a)?;
                let vb = self.evaluate(b)?;
                if vb == 0 { None } else { Some(va / vb) }
            }
            DimExpr::Mod(a, b) => {
                let va = self.evaluate(a)?;
                let vb = self.evaluate(b)?;
                if vb == 0 { None } else { Some(va % vb) }
            }
        }
    }

    /// Simplify a DimExpr by substituting known bindings.
    pub fn simplify(&self, expr: &DimExpr) -> DimExpr {
        match expr {
            DimExpr::Sym(s) => {
                if let Some(&v) = self.bindings.get(s) {
                    DimExpr::Lit(v)
                } else {
                    expr.clone()
                }
            }
            DimExpr::Lit(_) => expr.clone(),
            DimExpr::Add(a, b) => {
                let sa = self.simplify(a);
                let sb = self.simplify(b);
                if let (DimExpr::Lit(va), DimExpr::Lit(vb)) = (&sa, &sb) {
                    DimExpr::Lit(va + vb)
                } else {
                    DimExpr::Add(Box::new(sa), Box::new(sb))
                }
            }
            DimExpr::Mul(a, b) => {
                let sa = self.simplify(a);
                let sb = self.simplify(b);
                if let (DimExpr::Lit(va), DimExpr::Lit(vb)) = (&sa, &sb) {
                    DimExpr::Lit(va * vb)
                } else {
                    DimExpr::Mul(Box::new(sa), Box::new(sb))
                }
            }
            DimExpr::Div(a, b) => {
                let sa = self.simplify(a);
                let sb = self.simplify(b);
                if let (DimExpr::Lit(va), DimExpr::Lit(vb)) = (&sa, &sb) {
                    if *vb != 0 { DimExpr::Lit(va / vb) } else { DimExpr::Div(Box::new(sa), Box::new(sb)) }
                } else {
                    DimExpr::Div(Box::new(sa), Box::new(sb))
                }
            }
            DimExpr::Mod(a, b) => {
                let sa = self.simplify(a);
                let sb = self.simplify(b);
                if let (DimExpr::Lit(va), DimExpr::Lit(vb)) = (&sa, &sb) {
                    if *vb != 0 { DimExpr::Lit(va % vb) } else { DimExpr::Mod(Box::new(sa), Box::new(sb)) }
                } else {
                    DimExpr::Mod(Box::new(sa), Box::new(sb))
                }
            }
        }
    }

    /// Try to prove lhs == rhs.
    pub fn prove_eq(&self, lhs: &DimExpr, rhs: &DimExpr) -> Result<(), ProofFailure> {
        // Strategy 1: Structural equality
        if lhs == rhs {
            return Ok(());
        }

        // Strategy 2: Simplify both sides and compare
        let sl = self.simplify(lhs);
        let sr = self.simplify(rhs);
        if sl == sr {
            return Ok(());
        }

        // Strategy 3: Evaluate both to concrete values
        if let (Some(vl), Some(vr)) = (self.evaluate(lhs), self.evaluate(rhs)) {
            if vl == vr {
                return Ok(());
            } else {
                return Err(ProofFailure {
                    goal: format!("{:?} == {:?}", lhs, rhs),
                    attempts: vec!["concrete evaluation".into()],
                    reason: format!("lhs={}, rhs={}", vl, vr),
                });
            }
        }

        // Strategy 4: Check known equalities
        for (eq_l, eq_r) in &self.equalities {
            if (eq_l == lhs && eq_r == rhs) || (eq_l == rhs && eq_r == lhs) {
                return Ok(());
            }
        }

        Err(ProofFailure {
            goal: format!("{:?} == {:?}", lhs, rhs),
            attempts: vec![
                "structural equality".into(),
                "simplification".into(),
                "concrete evaluation".into(),
                "known equalities".into(),
            ],
            reason: "insufficient information to prove equality".into(),
        })
    }

    /// Try to prove expr % divisor == 0.
    pub fn prove_divisible(&self, expr: &DimExpr, divisor: i64) -> Result<(), ProofFailure> {
        if divisor == 0 {
            return Err(ProofFailure {
                goal: format!("{:?} % 0 == 0", expr),
                attempts: vec![],
                reason: "division by zero".into(),
            });
        }
        if divisor == 1 {
            return Ok(()); // everything is divisible by 1
        }

        // Strategy 1: Concrete evaluation
        if let Some(v) = self.evaluate(expr) {
            if v % divisor == 0 {
                return Ok(());
            } else {
                return Err(ProofFailure {
                    goal: format!("{:?} % {} == 0", expr, divisor),
                    attempts: vec!["concrete evaluation".into()],
                    reason: format!("{} % {} = {} (not zero)", v, divisor, v % divisor),
                });
            }
        }

        // Strategy 2: Known divisibility facts
        if let DimExpr::Sym(s) = expr {
            if let Some(divs) = self.divisibility.get(s) {
                if divs.contains(&divisor) {
                    return Ok(());
                }
                // Check if any known divisor is a multiple of the required divisor
                if divs.iter().any(|&d| d % divisor == 0) {
                    return Ok(());
                }
            }
        }

        // Strategy 3: Product of terms — if a*b and b % divisor == 0, then a*b % divisor == 0
        if let DimExpr::Mul(a, b) = expr {
            if self.prove_divisible(a, divisor).is_ok() || self.prove_divisible(b, divisor).is_ok() {
                return Ok(());
            }
        }

        Err(ProofFailure {
            goal: format!("{:?} % {} == 0", expr, divisor),
            attempts: vec![
                "concrete evaluation".into(),
                "known divisibility facts".into(),
                "product factor analysis".into(),
            ],
            reason: format!("cannot prove {:?} is divisible by {}", expr, divisor),
        })
    }
}

impl Default for ShapeAlgebraSolver {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Interner = string_interner::StringInterner<
        string_interner::backend::BucketBackend<string_interner::DefaultSymbol>,
    >;

    fn make_sym(interner: &mut Interner, name: &str) -> Symbol {
        Symbol(interner.get_or_intern(name))
    }

    #[test]
    fn literal_equality() {
        let solver = ShapeAlgebraSolver::new();
        assert!(solver.prove_eq(&DimExpr::Lit(42), &DimExpr::Lit(42)).is_ok());
        assert!(solver.prove_eq(&DimExpr::Lit(42), &DimExpr::Lit(43)).is_err());
    }

    #[test]
    fn symbolic_structural_equality() {
        let mut interner = Interner::new();
        let s = make_sym(&mut interner, "S");
        let solver = ShapeAlgebraSolver::new();

        let expr = DimExpr::Mul(
            Box::new(DimExpr::Sym(s)),
            Box::new(DimExpr::Lit(768)),
        );
        assert!(solver.prove_eq(&expr, &expr).is_ok());
    }

    #[test]
    fn concrete_binding_proves_equality() {
        let mut interner = Interner::new();
        let d = make_sym(&mut interner, "D");
        let mut solver = ShapeAlgebraSolver::new();
        solver.bind(d, 768);

        // D == 768
        assert!(solver.prove_eq(&DimExpr::Sym(d), &DimExpr::Lit(768)).is_ok());
        // D != 512
        assert!(solver.prove_eq(&DimExpr::Sym(d), &DimExpr::Lit(512)).is_err());
    }

    #[test]
    fn product_evaluation() {
        let mut interner = Interner::new();
        let s = make_sym(&mut interner, "S");
        let d = make_sym(&mut interner, "D");
        let mut solver = ShapeAlgebraSolver::new();
        solver.bind(s, 128);
        solver.bind(d, 768);

        let prod = DimExpr::Mul(Box::new(DimExpr::Sym(s)), Box::new(DimExpr::Sym(d)));
        assert_eq!(solver.evaluate(&prod), Some(128 * 768));
    }

    #[test]
    fn simplify_with_bindings() {
        let mut interner = Interner::new();
        let d = make_sym(&mut interner, "D");
        let mut solver = ShapeAlgebraSolver::new();
        solver.bind(d, 768);

        let expr = DimExpr::Div(Box::new(DimExpr::Sym(d)), Box::new(DimExpr::Lit(12)));
        let simplified = solver.simplify(&expr);
        assert_eq!(simplified, DimExpr::Lit(64)); // 768 / 12
    }

    #[test]
    fn divisibility_concrete() {
        let mut interner = Interner::new();
        let d = make_sym(&mut interner, "D");
        let mut solver = ShapeAlgebraSolver::new();
        solver.bind(d, 768);

        assert!(solver.prove_divisible(&DimExpr::Sym(d), 12).is_ok());  // 768 % 12 == 0
        assert!(solver.prove_divisible(&DimExpr::Sym(d), 11).is_err()); // 768 % 11 != 0
    }

    #[test]
    fn divisibility_from_assert() {
        let mut interner = Interner::new();
        let d = make_sym(&mut interner, "D");
        let mut solver = ShapeAlgebraSolver::new();
        solver.assert_divisible(d, 12);

        // D % 12 == 0 from assertion
        assert!(solver.prove_divisible(&DimExpr::Sym(d), 12).is_ok());
        // D % 6 == 0 because 12 % 6 == 0
        assert!(solver.prove_divisible(&DimExpr::Sym(d), 6).is_ok());
        // D % 7 unknown
        assert!(solver.prove_divisible(&DimExpr::Sym(d), 7).is_err());
    }

    #[test]
    fn divisibility_by_one_always_true() {
        let solver = ShapeAlgebraSolver::new();
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "X");
        assert!(solver.prove_divisible(&DimExpr::Sym(x), 1).is_ok());
    }

    #[test]
    fn divisibility_product_factor() {
        let mut interner = Interner::new();
        let h = make_sym(&mut interner, "H");
        let dh = make_sym(&mut interner, "Dh");
        let mut solver = ShapeAlgebraSolver::new();
        solver.bind(h, 12);

        // H * Dh — since H = 12, and 12 % 12 == 0, H*Dh % 12 == 0
        let expr = DimExpr::Mul(Box::new(DimExpr::Sym(h)), Box::new(DimExpr::Sym(dh)));
        assert!(solver.prove_divisible(&expr, 12).is_ok());
    }

    #[test]
    fn reshape_s_d_to_sd() {
        let mut interner = Interner::new();
        let s = make_sym(&mut interner, "S");
        let d = make_sym(&mut interner, "D");
        let solver = ShapeAlgebraSolver::new();

        // prove S*D == S*D (structural equality)
        let sd = DimExpr::Mul(Box::new(DimExpr::Sym(s)), Box::new(DimExpr::Sym(d)));
        assert!(solver.prove_eq(&sd, &sd).is_ok());
    }

    #[test]
    fn known_equality_proves() {
        let mut interner = Interner::new();
        let a = make_sym(&mut interner, "A");
        let b = make_sym(&mut interner, "B");
        let mut solver = ShapeAlgebraSolver::new();
        solver.assert_eq(DimExpr::Sym(a), DimExpr::Sym(b));

        assert!(solver.prove_eq(&DimExpr::Sym(a), &DimExpr::Sym(b)).is_ok());
    }

    #[test]
    fn proof_failure_has_explanation() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "X");
        let y = make_sym(&mut interner, "Y");
        let solver = ShapeAlgebraSolver::new();

        let err = solver.prove_eq(&DimExpr::Sym(x), &DimExpr::Sym(y)).unwrap_err();
        assert!(!err.attempts.is_empty());
        assert!(err.reason.contains("insufficient"));
    }

    #[test]
    fn evaluate_nested() {
        let mut interner = Interner::new();
        let b = make_sym(&mut interner, "B");
        let s = make_sym(&mut interner, "S");
        let mut solver = ShapeAlgebraSolver::new();
        solver.bind(b, 32);
        solver.bind(s, 128);

        // (B + S) * 2 = (32 + 128) * 2 = 320
        let expr = DimExpr::Mul(
            Box::new(DimExpr::Add(Box::new(DimExpr::Sym(b)), Box::new(DimExpr::Sym(s)))),
            Box::new(DimExpr::Lit(2)),
        );
        assert_eq!(solver.evaluate(&expr), Some(320));
    }

    #[test]
    fn division_by_zero_returns_none() {
        let solver = ShapeAlgebraSolver::new();
        let expr = DimExpr::Div(Box::new(DimExpr::Lit(10)), Box::new(DimExpr::Lit(0)));
        assert_eq!(solver.evaluate(&expr), None);
    }
}
