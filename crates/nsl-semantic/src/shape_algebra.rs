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

        // Strategy 5: Bound-derived equality (if both sides have singleton bounds [v, v])
        if let (DimExpr::Sym(a), DimExpr::Sym(b)) = (&sl, &sr) {
            if let (Some((Some(lo_a), Some(hi_a))), Some((Some(lo_b), Some(hi_b))))
                = (self.bounds.get(a), self.bounds.get(b))
            {
                if lo_a == hi_a && lo_b == hi_b && lo_a == lo_b {
                    return Ok(());
                }
            }
        }
        // Also: symbol vs literal via bounds
        if let DimExpr::Sym(s) = &sl {
            if let DimExpr::Lit(v) = &sr {
                if let Some((Some(lo), Some(hi))) = self.bounds.get(s) {
                    if lo == hi && lo == v {
                        return Ok(());
                    }
                }
            }
        }
        if let DimExpr::Lit(v) = &sl {
            if let DimExpr::Sym(s) = &sr {
                if let Some((Some(lo), Some(hi))) = self.bounds.get(s) {
                    if lo == hi && lo == v {
                        return Ok(());
                    }
                }
            }
        }

        Err(ProofFailure {
            goal: format!("{:?} == {:?}", lhs, rhs),
            attempts: vec![
                "structural equality".into(),
                "simplification".into(),
                "concrete evaluation".into(),
                "known equalities".into(),
                "bound-derived equality".into(),
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

    // ---------------------------------------------------------------------------
    // Bound-aware reasoning
    // ---------------------------------------------------------------------------

    /// Get the bounds for a symbol, if any.
    pub fn get_bound(&self, sym: Symbol) -> Option<(Option<i64>, Option<i64>)> {
        self.bounds.get(&sym).copied()
    }

    /// Compute upper bound of an expression from stored bounds and bindings.
    pub fn upper_bound(&self, expr: &DimExpr) -> Option<i64> {
        match expr {
            DimExpr::Lit(v) => Some(*v),
            DimExpr::Sym(s) => {
                if let Some(&v) = self.bindings.get(s) {
                    Some(v)
                } else if let Some((_, hi)) = self.bounds.get(s) {
                    *hi
                } else {
                    None
                }
            }
            DimExpr::Add(a, b) => {
                let ua = self.upper_bound(a)?;
                let ub = self.upper_bound(b)?;
                Some(ua + ub)
            }
            DimExpr::Mul(a, b) => {
                let ua = self.upper_bound(a)?;
                let ub = self.upper_bound(b)?;
                // Only correct for non-negative bounds
                if ua >= 0 && ub >= 0 { Some(ua * ub) } else { None }
            }
            _ => self.evaluate(expr),
        }
    }

    /// Compute lower bound of an expression from stored bounds and bindings.
    pub fn lower_bound(&self, expr: &DimExpr) -> Option<i64> {
        match expr {
            DimExpr::Lit(v) => Some(*v),
            DimExpr::Sym(s) => {
                if let Some(&v) = self.bindings.get(s) {
                    Some(v)
                } else if let Some((lo, _)) = self.bounds.get(s) {
                    *lo
                } else {
                    None
                }
            }
            DimExpr::Add(a, b) => {
                let la = self.lower_bound(a)?;
                let lb = self.lower_bound(b)?;
                Some(la + lb)
            }
            DimExpr::Mul(a, b) => {
                let la = self.lower_bound(a)?;
                let lb = self.lower_bound(b)?;
                if la >= 0 && lb >= 0 { Some(la * lb) } else { None }
            }
            _ => self.evaluate(expr),
        }
    }

    // ---------------------------------------------------------------------------
    // Inequality proofs
    // ---------------------------------------------------------------------------

    /// Comparison operators for inequality proofs.
    /// Try to prove `expr op value` (e.g., S <= 4096).
    pub fn prove_le(&self, expr: &DimExpr, value: i64) -> Result<(), ProofFailure> {
        // Strategy 1: concrete evaluation
        if let Some(v) = self.evaluate(expr) {
            return if v <= value { Ok(()) } else {
                Err(ProofFailure {
                    goal: format!("{:?} <= {}", expr, value),
                    attempts: vec!["concrete evaluation".into()],
                    reason: format!("{} > {}", v, value),
                })
            };
        }

        // Strategy 2: upper bound from bounds database
        if let Some(ub) = self.upper_bound(expr) {
            if ub <= value { return Ok(()); }
        }

        // Strategy 3: Fourier-Motzkin for transitive bounds
        if self.fm_prove_le(expr, value) {
            return Ok(());
        }

        Err(ProofFailure {
            goal: format!("{:?} <= {}", expr, value),
            attempts: vec!["concrete".into(), "upper bound".into(), "Fourier-Motzkin".into()],
            reason: "insufficient information".into(),
        })
    }

    /// Try to prove `expr >= value`.
    pub fn prove_ge(&self, expr: &DimExpr, value: i64) -> Result<(), ProofFailure> {
        if let Some(v) = self.evaluate(expr) {
            return if v >= value { Ok(()) } else {
                Err(ProofFailure {
                    goal: format!("{:?} >= {}", expr, value),
                    attempts: vec!["concrete evaluation".into()],
                    reason: format!("{} < {}", v, value),
                })
            };
        }

        if let Some(lb) = self.lower_bound(expr) {
            if lb >= value { return Ok(()); }
        }

        Err(ProofFailure {
            goal: format!("{:?} >= {}", expr, value),
            attempts: vec!["concrete".into(), "lower bound".into()],
            reason: "insufficient information".into(),
        })
    }

    // ---------------------------------------------------------------------------
    // Fourier-Motzkin variable elimination (limited)
    // ---------------------------------------------------------------------------

    /// Try to prove `expr <= value` using transitive bound reasoning.
    ///
    /// Converts stored bounds into linear constraints and eliminates variables
    /// to derive a bound on the target expression. Limited to simple symbol
    /// expressions with ≤3 elimination steps to avoid exponential blowup.
    fn fm_prove_le(&self, expr: &DimExpr, value: i64) -> bool {
        // Only handle simple symbol case for now
        let target_sym = match expr {
            DimExpr::Sym(s) => *s,
            _ => return false,
        };

        // Collect transitive upper bounds: if A <= B and B <= C and C <= value,
        // then A <= value. Walk the equality chain.
        // Simple implementation: check if any bound chain leads to <= value
        self.fm_trace_upper(target_sym, value, &mut Vec::new(), 3)
    }

    /// Recursively trace upper bounds. `depth` limits recursion.
    fn fm_trace_upper(&self, sym: Symbol, target: i64, visited: &mut Vec<Symbol>, depth: u32) -> bool {
        if depth == 0 { return false; }
        if visited.contains(&sym) { return false; }
        visited.push(sym);

        // Direct bound check
        if let Some((_, Some(hi))) = self.bounds.get(&sym) {
            if *hi <= target { return true; }
        }

        // Check if bound on sym is via another symbol (from equalities)
        // e.g., assert_eq(A, B) and B has bound <= target
        for (eq_l, eq_r) in &self.equalities {
            if let DimExpr::Sym(s) = eq_l {
                if *s == sym {
                    if let DimExpr::Sym(other) = eq_r {
                        if self.fm_trace_upper(*other, target, visited, depth - 1) {
                            return true;
                        }
                    }
                }
            }
            if let DimExpr::Sym(s) = eq_r {
                if *s == sym {
                    if let DimExpr::Sym(other) = eq_l {
                        if self.fm_trace_upper(*other, target, visited, depth - 1) {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    // ---------------------------------------------------------------------------
    // Commutative normalization
    // ---------------------------------------------------------------------------

    /// Normalize a DimExpr by sorting commutative operands lexicographically.
    /// This makes `A*B` and `B*A` compare as equal after normalization.
    pub fn normalize(&self, expr: &DimExpr) -> DimExpr {
        let simplified = self.simplify(expr);
        self.normalize_inner(&simplified)
    }

    fn normalize_inner(&self, expr: &DimExpr) -> DimExpr {
        match expr {
            DimExpr::Lit(_) | DimExpr::Sym(_) => expr.clone(),
            DimExpr::Add(a, b) => {
                let na = self.normalize_inner(a);
                let nb = self.normalize_inner(b);
                // Sort: smaller debug representation first
                if format!("{:?}", na) <= format!("{:?}", nb) {
                    DimExpr::Add(Box::new(na), Box::new(nb))
                } else {
                    DimExpr::Add(Box::new(nb), Box::new(na))
                }
            }
            DimExpr::Mul(a, b) => {
                let na = self.normalize_inner(a);
                let nb = self.normalize_inner(b);
                if format!("{:?}", na) <= format!("{:?}", nb) {
                    DimExpr::Mul(Box::new(na), Box::new(nb))
                } else {
                    DimExpr::Mul(Box::new(nb), Box::new(na))
                }
            }
            DimExpr::Div(a, b) => {
                DimExpr::Div(Box::new(self.normalize_inner(a)), Box::new(self.normalize_inner(b)))
            }
            DimExpr::Mod(a, b) => {
                DimExpr::Mod(Box::new(self.normalize_inner(a)), Box::new(self.normalize_inner(b)))
            }
        }
    }

    /// Try to prove equality using normalization (handles commutativity).
    pub fn prove_eq_normalized(&self, lhs: &DimExpr, rhs: &DimExpr) -> Result<(), ProofFailure> {
        // Try standard proof first
        if self.prove_eq(lhs, rhs).is_ok() {
            return Ok(());
        }

        // Try normalized comparison
        let nl = self.normalize(lhs);
        let nr = self.normalize(rhs);
        if nl == nr {
            return Ok(());
        }

        Err(ProofFailure {
            goal: format!("{:?} == {:?}", lhs, rhs),
            attempts: vec!["standard proof".into(), "commutative normalization".into()],
            reason: "insufficient information after normalization".into(),
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

    // ── Bound reasoning tests (M49b) ──────────────────────────────────

    #[test]
    fn bound_derived_equality() {
        let mut interner = Interner::new();
        let s = make_sym(&mut interner, "S");
        let mut solver = ShapeAlgebraSolver::new();
        // S is exactly 32 (singleton bound)
        solver.assert_bound(s, Some(32), Some(32));

        assert!(solver.prove_eq(&DimExpr::Sym(s), &DimExpr::Lit(32)).is_ok(),
            "S with bounds [32, 32] should equal 32");
    }

    #[test]
    fn bound_derived_equality_two_symbols() {
        let mut interner = Interner::new();
        let a = make_sym(&mut interner, "A");
        let b = make_sym(&mut interner, "B");
        let mut solver = ShapeAlgebraSolver::new();
        solver.assert_bound(a, Some(64), Some(64));
        solver.assert_bound(b, Some(64), Some(64));

        assert!(solver.prove_eq(&DimExpr::Sym(a), &DimExpr::Sym(b)).is_ok(),
            "A=[64,64] and B=[64,64] should be equal");
    }

    #[test]
    fn prove_le_from_bound() {
        let mut interner = Interner::new();
        let s = make_sym(&mut interner, "S");
        let mut solver = ShapeAlgebraSolver::new();
        solver.assert_bound(s, Some(1), Some(4096));

        assert!(solver.prove_le(&DimExpr::Sym(s), 4096).is_ok(),
            "S <= 4096 should hold when S in [1, 4096]");
        assert!(solver.prove_le(&DimExpr::Sym(s), 8192).is_ok(),
            "S <= 8192 should hold when S in [1, 4096]");
        assert!(solver.prove_le(&DimExpr::Sym(s), 100).is_err(),
            "S <= 100 cannot be proven when S upper bound is 4096");
    }

    #[test]
    fn prove_ge_from_bound() {
        let mut interner = Interner::new();
        let s = make_sym(&mut interner, "S");
        let mut solver = ShapeAlgebraSolver::new();
        solver.assert_bound(s, Some(1), Some(4096));

        assert!(solver.prove_ge(&DimExpr::Sym(s), 1).is_ok(),
            "S >= 1 should hold when S in [1, 4096]");
        assert!(solver.prove_ge(&DimExpr::Sym(s), 0).is_ok(),
            "S >= 0 should hold when S >= 1");
        assert!(solver.prove_ge(&DimExpr::Sym(s), 2).is_err(),
            "S >= 2 cannot be proven when lower bound is 1");
    }

    #[test]
    fn prove_le_concrete() {
        let mut interner = Interner::new();
        let d = make_sym(&mut interner, "D");
        let mut solver = ShapeAlgebraSolver::new();
        solver.bind(d, 768);

        assert!(solver.prove_le(&DimExpr::Sym(d), 1000).is_ok());
        assert!(solver.prove_le(&DimExpr::Sym(d), 768).is_ok());
        assert!(solver.prove_le(&DimExpr::Sym(d), 500).is_err());
    }

    #[test]
    fn upper_bound_of_product() {
        let mut interner = Interner::new();
        let a = make_sym(&mut interner, "A");
        let b = make_sym(&mut interner, "B");
        let mut solver = ShapeAlgebraSolver::new();
        solver.assert_bound(a, Some(1), Some(32));
        solver.assert_bound(b, Some(1), Some(64));

        let prod = DimExpr::Mul(Box::new(DimExpr::Sym(a)), Box::new(DimExpr::Sym(b)));
        assert_eq!(solver.upper_bound(&prod), Some(32 * 64));
        assert!(solver.prove_le(&prod, 2048).is_ok());
    }

    #[test]
    fn fm_transitive_bound() {
        let mut interner = Interner::new();
        let a = make_sym(&mut interner, "A");
        let b = make_sym(&mut interner, "B");
        let mut solver = ShapeAlgebraSolver::new();
        // A == B (equality), B <= 1024
        solver.assert_eq(DimExpr::Sym(a), DimExpr::Sym(b));
        solver.assert_bound(b, Some(1), Some(1024));

        // A <= 1024 via transitive: A == B, B <= 1024
        assert!(solver.prove_le(&DimExpr::Sym(a), 1024).is_ok(),
            "A <= 1024 via A == B, B <= 1024");
    }

    // ── Commutative normalization tests ──────────────────────────────

    #[test]
    fn normalize_commutative_mul() {
        let mut interner = Interner::new();
        let a = make_sym(&mut interner, "A");
        let b = make_sym(&mut interner, "B");
        let solver = ShapeAlgebraSolver::new();

        let ab = DimExpr::Mul(Box::new(DimExpr::Sym(a)), Box::new(DimExpr::Sym(b)));
        let ba = DimExpr::Mul(Box::new(DimExpr::Sym(b)), Box::new(DimExpr::Sym(a)));

        let n_ab = solver.normalize(&ab);
        let n_ba = solver.normalize(&ba);
        assert_eq!(n_ab, n_ba, "A*B and B*A should normalize to same form");
    }

    #[test]
    fn prove_eq_normalized_commutative() {
        let mut interner = Interner::new();
        let a = make_sym(&mut interner, "A");
        let b = make_sym(&mut interner, "B");
        let solver = ShapeAlgebraSolver::new();

        let ab = DimExpr::Mul(Box::new(DimExpr::Sym(a)), Box::new(DimExpr::Sym(b)));
        let ba = DimExpr::Mul(Box::new(DimExpr::Sym(b)), Box::new(DimExpr::Sym(a)));

        // Standard prove_eq fails (different structure)
        assert!(solver.prove_eq(&ab, &ba).is_err());
        // Normalized prove_eq succeeds
        assert!(solver.prove_eq_normalized(&ab, &ba).is_ok(),
            "A*B == B*A should be provable via normalization");
    }
}
