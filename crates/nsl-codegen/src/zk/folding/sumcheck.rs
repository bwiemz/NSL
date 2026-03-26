//! Univariate sumcheck argument for fast fold verification.
//!
//! The sumcheck protocol reduces a multivariate polynomial sum to a sequence
//! of univariate polynomial evaluations. This is used in the folding backend
//! to efficiently verify that the accumulator constraints are satisfied.
//!
//! The protocol runs in `num_variables` rounds:
//!   1. Prover sends a univariate polynomial g_i(X) = sum_{rest} f(x_1,...,X,...,x_n)
//!   2. Verifier checks g_i(0) + g_i(1) == claim
//!   3. Verifier sends random challenge r_i
//!   4. Claim is updated to g_i(r_i)
//!
//! After all rounds, the verifier makes a single evaluation query to f.

use super::super::field::Field;

// ---------------------------------------------------------------------------
// Sumcheck types
// ---------------------------------------------------------------------------

/// A univariate polynomial represented by its evaluations at 0, 1, 2, ...
///
/// For degree-d polynomials, we need d+1 evaluation points.
#[derive(Debug, Clone)]
pub struct UnivariatePoly<F: Field> {
    /// Evaluations at points 0, 1, 2, ..., degree.
    pub evals: Vec<F>,
}

impl<F: Field> UnivariatePoly<F> {
    /// Evaluate the polynomial at a given point using Lagrange interpolation.
    pub fn evaluate(&self, point: &F) -> F {
        let n = self.evals.len();
        if n == 0 {
            return F::zero();
        }

        // Lagrange interpolation at integer nodes 0, 1, ..., n-1
        let mut result = F::zero();
        for (i, eval_i) in self.evals.iter().enumerate() {
            let mut basis = F::one();
            for j in 0..n {
                if i == j {
                    continue;
                }
                // basis *= (point - j) / (i - j)
                let j_field = F::from_u64(j as u64);
                let i_field = F::from_u64(i as u64);
                let numer = point.field_sub(&j_field);
                let denom = i_field.field_sub(&j_field);
                basis = basis.field_mul(&numer).field_mul(&denom.field_inv());
            }
            result = result.field_add(&eval_i.field_mul(&basis));
        }
        result
    }

    /// Check that g(0) + g(1) equals the claimed sum.
    pub fn check_sum(&self, claim: &F) -> bool {
        if self.evals.len() < 2 {
            return false;
        }
        let sum = self.evals[0].field_add(&self.evals[1]);
        sum == *claim
    }
}

// ---------------------------------------------------------------------------
// Sumcheck prover
// ---------------------------------------------------------------------------

/// Run a single-round sumcheck for a multilinear polynomial.
///
/// Given evaluations of a multilinear polynomial `f` over the boolean hypercube
/// {0,1}^n, and the claimed sum, produce the round polynomial and final evaluation.
///
/// # Arguments
/// * `polynomial` — evaluations of `f` at all 2^n points (in lexicographic order)
/// * `claim` — claimed sum of `f` over the boolean hypercube
///
/// # Returns
/// * `round_polys` — one univariate polynomial per round
/// * `final_eval` — evaluation of `f` at the challenge point
pub fn sumcheck_prove<F: Field>(
    polynomial: &[F],
    claim: &F,
) -> (Vec<UnivariatePoly<F>>, F) {
    let n = polynomial.len();
    if n == 0 {
        return (vec![], F::zero());
    }

    // Number of variables = log2(n), or 0 if n=1
    let num_vars = if n <= 1 { 0 } else { (n as f64).log2().ceil() as usize };

    if num_vars == 0 {
        return (vec![], polynomial[0].clone());
    }

    let mut round_polys = Vec::with_capacity(num_vars);
    let mut current_evals = polynomial.to_vec();
    let mut _current_claim = claim.clone();

    for round in 0..num_vars {
        let half = current_evals.len() / 2;
        if half == 0 {
            break;
        }

        // Compute round polynomial g(X) such that:
        //   g(0) = sum of f where x_round = 0
        //   g(1) = sum of f where x_round = 1
        let mut sum_0 = F::zero();
        let mut sum_1 = F::zero();

        for i in 0..half {
            sum_0 = sum_0.field_add(&current_evals[2 * i]);
            sum_1 = sum_1.field_add(&current_evals[2 * i + 1]);
        }

        let round_poly = UnivariatePoly {
            evals: vec![sum_0.clone(), sum_1.clone()],
        };

        round_polys.push(round_poly);

        // For subsequent rounds, bind the variable to a "challenge" (0 for simplicity in prover)
        // In a real interactive protocol, the verifier would send a random challenge here.
        // We use a deterministic binding for the prover's output.
        let mut next_evals = Vec::with_capacity(half);
        for i in 0..half {
            // Bind x_round = 0 for deterministic prover path
            next_evals.push(current_evals[2 * i].clone());
        }
        current_evals = next_evals;
        let _ = round; // suppress unused warning
    }

    let final_eval = if current_evals.is_empty() {
        F::zero()
    } else {
        current_evals[0].clone()
    };

    (round_polys, final_eval)
}

// ---------------------------------------------------------------------------
// Sumcheck verifier
// ---------------------------------------------------------------------------

/// A sumcheck proof transcript: the round polynomials and the final evaluation.
#[derive(Debug, Clone)]
pub struct SumcheckProof<F: Field> {
    /// Round polynomials g_1, g_2, ..., g_n
    pub round_polys: Vec<UnivariatePoly<F>>,
    /// Final evaluation of f at the challenge point
    pub final_eval: F,
    /// Challenges used (Fiat-Shamir derived)
    pub challenges: Vec<F>,
}

/// Verify a sumcheck proof transcript.
///
/// The verifier replays the protocol:
///   1. For each round i, check g_i(0) + g_i(1) == current_claim
///   2. Derive the challenge r_i (Fiat-Shamir from round poly)
///   3. Update claim = g_i(r_i)
///   4. After all rounds, check final_eval == last claim
///
/// Returns Ok(true) if the proof is valid, Ok(false) if invalid.
pub fn sumcheck_verify<F: Field>(
    proof: &SumcheckProof<F>,
    initial_claim: &F,
) -> bool {
    let mut current_claim = initial_claim.clone();

    for (i, round_poly) in proof.round_polys.iter().enumerate() {
        // Step 1: Check g_i(0) + g_i(1) == current_claim
        if !round_poly.check_sum(&current_claim) {
            return false;
        }

        // Step 2: Get the challenge for this round
        let challenge = if i < proof.challenges.len() {
            &proof.challenges[i]
        } else {
            // If no challenge provided, use 0 (degenerate)
            return false;
        };

        // Step 3: Update claim = g_i(r_i)
        current_claim = round_poly.evaluate(challenge);
    }

    // Step 4: Final check — the last claim should equal final_eval
    current_claim == proof.final_eval
}

/// Run sumcheck prove with Fiat-Shamir challenges (non-interactive).
///
/// Like `sumcheck_prove` but generates random challenges from the round polynomials
/// using a simple hash-based Fiat-Shamir transform.
pub fn sumcheck_prove_interactive<F: Field>(
    polynomial: &[F],
    claim: &F,
) -> SumcheckProof<F> {
    let n = polynomial.len();
    if n == 0 {
        return SumcheckProof {
            round_polys: vec![],
            final_eval: F::zero(),
            challenges: vec![],
        };
    }

    let num_vars = if n <= 1 { 0 } else { (n as f64).log2().ceil() as usize };
    if num_vars == 0 {
        return SumcheckProof {
            round_polys: vec![],
            final_eval: polynomial[0].clone(),
            challenges: vec![],
        };
    }

    let mut round_polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    let mut current_evals = polynomial.to_vec();

    for round in 0..num_vars {
        let half = current_evals.len() / 2;
        if half == 0 { break; }

        // Compute round polynomial
        let mut sum_0 = F::zero();
        let mut sum_1 = F::zero();
        for i in 0..half {
            sum_0 = sum_0.field_add(&current_evals[2 * i]);
            sum_1 = sum_1.field_add(&current_evals[2 * i + 1]);
        }

        let round_poly = UnivariatePoly {
            evals: vec![sum_0.clone(), sum_1.clone()],
        };

        // Fiat-Shamir challenge: derive from round polynomial evaluations
        // Simple hash: challenge = sum_0 * 7 + sum_1 * 13 + round + 1
        let challenge = sum_0.field_mul(&F::from_u64(7))
            .field_add(&sum_1.field_mul(&F::from_u64(13)))
            .field_add(&F::from_u64(round as u64 + 1));
        let challenge = if challenge == F::zero() { F::one() } else { challenge };

        // Bind variable to challenge for next round
        let mut next_evals = Vec::with_capacity(half);
        for i in 0..half {
            // Linear interpolation: f(r) = f(0) + r * (f(1) - f(0))
            let diff = current_evals[2 * i + 1].field_sub(&current_evals[2 * i]);
            next_evals.push(current_evals[2 * i].field_add(&challenge.field_mul(&diff)));
        }

        round_polys.push(round_poly);
        challenges.push(challenge);
        current_evals = next_evals;
    }

    let final_eval = if current_evals.is_empty() {
        F::zero()
    } else {
        current_evals[0].clone()
    };

    SumcheckProof {
        round_polys,
        final_eval,
        challenges,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::field_m31::Mersenne31Field;

    type M31 = Mersenne31Field;

    #[test]
    fn test_univariate_poly_evaluate() {
        // Polynomial with evals at 0,1: g(0)=3, g(1)=7
        // This is the line g(x) = 3 + 4x
        let poly = UnivariatePoly {
            evals: vec![M31::from_u64(3), M31::from_u64(7)],
        };

        assert_eq!(poly.evaluate(&M31::from_u64(0)), M31::from_u64(3));
        assert_eq!(poly.evaluate(&M31::from_u64(1)), M31::from_u64(7));
        // g(2) = 3 + 4*2 = 11
        assert_eq!(poly.evaluate(&M31::from_u64(2)), M31::from_u64(11));
    }

    #[test]
    fn test_check_sum() {
        let poly = UnivariatePoly {
            evals: vec![M31::from_u64(3), M31::from_u64(7)],
        };
        assert!(poly.check_sum(&M31::from_u64(10))); // 3 + 7 = 10
        assert!(!poly.check_sum(&M31::from_u64(9)));
    }

    #[test]
    fn test_sumcheck_prove_basic() {
        // f(x) for x in {0, 1}: f(0) = 3, f(1) = 5
        // Sum = 3 + 5 = 8
        let polynomial = vec![M31::from_u64(3), M31::from_u64(5)];
        let claim = M31::from_u64(8);

        let (round_polys, _final_eval) = sumcheck_prove(&polynomial, &claim);
        assert_eq!(round_polys.len(), 1);
        // g(0) + g(1) should equal the claim
        assert!(round_polys[0].check_sum(&claim));
    }

    #[test]
    fn test_sumcheck_prove_two_vars() {
        // f(x0, x1) for (x0, x1) in {0,1}^2:
        //   f(0,0) = 1, f(0,1) = 2, f(1,0) = 3, f(1,1) = 4
        // Sum = 1 + 2 + 3 + 4 = 10
        let polynomial = vec![
            M31::from_u64(1),
            M31::from_u64(2),
            M31::from_u64(3),
            M31::from_u64(4),
        ];
        let claim = M31::from_u64(10);

        let (round_polys, _final_eval) = sumcheck_prove(&polynomial, &claim);
        assert_eq!(round_polys.len(), 2);

        // First round: g1(0) = f(0,0) + f(0,1) = 3, g1(1) = f(1,0) + f(1,1) = 7
        assert!(round_polys[0].check_sum(&claim));
    }

    #[test]
    fn test_sumcheck_empty() {
        let (polys, eval) = sumcheck_prove::<M31>(&[], &M31::zero());
        assert!(polys.is_empty());
        assert_eq!(eval, M31::zero());
    }
}
