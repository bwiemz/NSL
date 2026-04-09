//! Accumulator state for folding proofs (Nova-style).
//!
//! The accumulator tracks a "relaxed" R1CS instance: the instance vector,
//! witness vector, and an error term. Folding combines two instances into
//! one by taking a random linear combination, absorbing the cross-term error.
//!
//! After N folds, the final accumulator can be verified with a single proof
//! (the "decider"), making total proof size O(1) in the number of layers.

use super::super::backend::ZkError;
use super::super::field::Field;

// ---------------------------------------------------------------------------
// Accumulator
// ---------------------------------------------------------------------------

/// A folding accumulator: relaxed R1CS instance + witness + error term.
///
/// Represents the running state of the folding prover. Each fold combines
/// this accumulator with a new instance using a random challenge.
#[derive(Debug, Clone)]
pub struct Accumulator<F: Field> {
    /// Committed instance values (public inputs/outputs for this fold).
    pub instance: Vec<F>,
    /// Witness values (private, used by prover only).
    pub witness: Vec<F>,
    /// Relaxation error term (Nova-style). Starts at zero, grows with cross-terms.
    /// The final decider verifies that E satisfies the relaxed relation.
    pub error_term: F,
    /// Number of folds applied to produce this accumulator.
    pub num_folds: u32,
}

impl<F: Field> Accumulator<F> {
    /// Create the initial accumulator from a single layer's instance/witness.
    pub fn initial(instance: Vec<F>, witness: Vec<F>) -> Self {
        Self {
            instance,
            witness,
            error_term: F::zero(),
            num_folds: 0,
        }
    }

    /// Fold a new instance into this accumulator using the given challenge.
    ///
    /// Nova-style folding: for challenge `r`:
    ///   - instance' = instance + r * new_instance
    ///   - witness' = witness + r * new_witness
    ///   - error' = error + r^2 * cross_term
    ///
    /// The cross-term captures the interaction between old and new constraints.
    /// A real implementation would compute it from the R1CS matrices; here we
    /// approximate it as the product of instance norms.
    pub fn fold(&self, new: &Self, challenge: &F) -> Result<Self, ZkError> {
        // Linear combination of instances
        let instance = self
            .instance
            .iter()
            .zip(new.instance.iter())
            .map(|(a, b)| a.field_add(&challenge.field_mul(b)))
            .collect();

        // Linear combination of witnesses (pad shorter with zeros)
        let max_len = self.witness.len().max(new.witness.len());
        let mut witness = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = if i < self.witness.len() {
                &self.witness[i]
            } else {
                &F::zero()
            };
            let b = if i < new.witness.len() {
                &new.witness[i]
            } else {
                &F::zero()
            };
            witness.push(a.field_add(&challenge.field_mul(b)));
        }

        // Error term update: E' = E_old + r^2 * cross_term
        // Cross-term approximation: sum of pairwise products of instance elements
        let mut cross_term = F::zero();
        for (a, b) in self.instance.iter().zip(new.instance.iter()) {
            cross_term = cross_term.field_add(&a.field_mul(b));
        }
        let r_squared = challenge.field_mul(challenge);
        let error_term = self.error_term.field_add(&r_squared.field_mul(&cross_term));

        Ok(Self {
            instance,
            witness,
            error_term,
            num_folds: self.num_folds + 1,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::field_m31::Mersenne31Field;

    #[test]
    fn test_initial_accumulator() {
        let acc = Accumulator::<Mersenne31Field>::initial(
            vec![Mersenne31Field::from_u64(1), Mersenne31Field::from_u64(2)],
            vec![Mersenne31Field::from_u64(3)],
        );
        assert_eq!(acc.instance.len(), 2);
        assert_eq!(acc.witness.len(), 1);
        assert_eq!(acc.error_term, Mersenne31Field::zero());
        assert_eq!(acc.num_folds, 0);
    }

    #[test]
    fn test_fold_updates_instance() {
        let acc = Accumulator::<Mersenne31Field>::initial(
            vec![Mersenne31Field::from_u64(10)],
            vec![Mersenne31Field::from_u64(100)],
        );
        let new = Accumulator::<Mersenne31Field>::initial(
            vec![Mersenne31Field::from_u64(20)],
            vec![Mersenne31Field::from_u64(200)],
        );
        let challenge = Mersenne31Field::from_u64(3);

        let folded = acc.fold(&new, &challenge).unwrap();
        // instance' = 10 + 3 * 20 = 70
        assert_eq!(folded.instance[0], Mersenne31Field::from_u64(70));
        // witness' = 100 + 3 * 200 = 700
        assert_eq!(folded.witness[0], Mersenne31Field::from_u64(700));
        assert_eq!(folded.num_folds, 1);
    }

    #[test]
    fn test_fold_error_term_grows() {
        let acc =
            Accumulator::<Mersenne31Field>::initial(vec![Mersenne31Field::from_u64(5)], vec![]);
        let new =
            Accumulator::<Mersenne31Field>::initial(vec![Mersenne31Field::from_u64(7)], vec![]);
        let challenge = Mersenne31Field::from_u64(2);

        let folded = acc.fold(&new, &challenge).unwrap();
        // error' = 0 + 2^2 * (5*7) = 4 * 35 = 140
        assert_eq!(folded.error_term, Mersenne31Field::from_u64(140));
    }

    #[test]
    fn test_fold_mismatched_witness_lengths() {
        let acc = Accumulator::<Mersenne31Field>::initial(
            vec![Mersenne31Field::from_u64(1)],
            vec![Mersenne31Field::from_u64(10), Mersenne31Field::from_u64(20)],
        );
        let new = Accumulator::<Mersenne31Field>::initial(
            vec![Mersenne31Field::from_u64(2)],
            vec![Mersenne31Field::from_u64(30)],
        );
        let challenge = Mersenne31Field::from_u64(1);

        let folded = acc.fold(&new, &challenge).unwrap();
        // Padded: witness' = [10+1*30, 20+1*0] = [40, 20]
        assert_eq!(folded.witness.len(), 2);
        assert_eq!(folded.witness[0], Mersenne31Field::from_u64(40));
        assert_eq!(folded.witness[1], Mersenne31Field::from_u64(20));
    }
}
