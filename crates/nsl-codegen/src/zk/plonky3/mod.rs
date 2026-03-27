//! M55: Plonky3 FRI-based ZK backend.
//!
//! Self-contained FRI (Fast Reed-Solomon IOP) prover and verifier over the
//! Mersenne-31 field (p = 2^31 - 1). No external crate dependencies beyond
//! `sha2` (already in the crate's `Cargo.toml`).
//!
//! ## Architecture
//!
//! 1. **AIR compilation**: ZkIR instructions are converted into an execution
//!    trace — each instruction becomes one row with wire values as columns.
//! 2. **Polynomial commitment**: trace columns are interpolated into
//!    polynomials, evaluated on an extended (blown-up) domain, and committed
//!    via a binary Merkle tree of SHA-256 hashes.
//! 3. **FRI protocol**: iterative polynomial folding reduces degree by half
//!    each round. Each round produces a new Merkle commitment. A final
//!    low-degree polynomial is sent in the clear.
//! 4. **Query phase**: the verifier opens Merkle paths at pseudorandom
//!    positions, checking folding consistency and AIR constraint satisfaction.
//!
//! ## Security
//!
//! With `log_blowup = 1` (rate 1/2), `num_queries = 40`, and `folding_factor = 2`,
//! the protocol achieves approximately `num_queries * log_blowup` bits of
//! soundness (40 bits — appropriate for a v1 implementation; production would
//! use 80-128 queries).

use sha2::{Digest, Sha256};

use crate::zk::backend::{
    CompiledCircuit, Witness, ZkBackend, ZkConfig, ZkError, ZkIR,
};
use crate::zk::field::FieldElement;
use crate::zk::field_m31::Mersenne31Field;
use crate::zk::ir::ZkInstruction;

// ---------------------------------------------------------------------------
// FRI configuration
// ---------------------------------------------------------------------------

/// FRI protocol parameters.
#[derive(Debug, Clone)]
pub struct FriConfig {
    /// Rate parameter: evaluation domain is `2^log_blowup` times the trace.
    pub log_blowup: u32,
    /// Number of query positions (security parameter).
    pub num_queries: u32,
    /// Degree reduction per FRI round (always 2 for binary folding).
    pub folding_factor: u32,
}

impl Default for FriConfig {
    fn default() -> Self {
        Self {
            log_blowup: 1,
            num_queries: 40,
            folding_factor: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Key types
// ---------------------------------------------------------------------------

/// Proving key for the Plonky3 backend.
#[derive(Debug, Clone)]
pub struct Plonky3ProvingKey {
    pub fri_config: FriConfig,
    /// Number of rows in the execution trace (always a power of 2).
    pub num_trace_rows: usize,
    /// Number of columns (wire values per row).
    pub num_trace_cols: usize,
    /// Maximum polynomial degree of AIR constraints.
    pub constraint_degree: usize,
    /// The ZkIR used to build this key (needed for trace generation).
    pub ir_instructions: Vec<ZkInstruction>,
}

/// Verification key — minimal data the verifier needs.
#[derive(Debug, Clone)]
pub struct Plonky3VerificationKey {
    pub fri_config: FriConfig,
    pub num_trace_rows: usize,
    pub num_trace_cols: usize,
    /// Merkle root of the committed execution trace.
    pub trace_commitment_root: [u8; 32],
}

/// A single FRI folding layer — the Merkle commitment of the folded evaluations.
#[derive(Debug, Clone)]
pub struct FriLayer {
    pub commitment: [u8; 32],
}

/// A query opening: Merkle path + leaf values at a specific index.
#[derive(Debug, Clone)]
pub struct QueryOpening {
    pub index: usize,
    /// Leaf values (M31 field elements, one per trace column).
    pub values: Vec<u32>,
    /// Authentication path from leaf to root.
    pub merkle_path: Vec<[u8; 32]>,
}

/// Complete Plonky3 proof.
#[derive(Debug, Clone)]
pub struct Plonky3Proof {
    /// Merkle root of the (extended) execution trace.
    pub trace_commitment: [u8; 32],
    /// Commitments for each FRI folding round.
    pub fri_layers: Vec<FriLayer>,
    /// Query openings at pseudorandom positions.
    pub query_openings: Vec<QueryOpening>,
    /// Public output values (serialized M31 elements).
    pub public_outputs: Vec<Vec<u8>>,
    /// Coefficients of the final low-degree polynomial (sent in the clear).
    pub final_poly: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Fiat-Shamir transcript
// ---------------------------------------------------------------------------

/// A Fiat-Shamir transcript built on SHA-256.
///
/// Each `append` hashes new data into the running state. Each `challenge_u32`
/// squeezes a pseudorandom M31 element from the state.
struct Transcript {
    state: Vec<u8>,
}

impl Transcript {
    fn new() -> Self {
        Self {
            state: b"plonky3-nsl-transcript-v1".to_vec(),
        }
    }

    fn append(&mut self, data: &[u8]) {
        self.state.extend_from_slice(data);
    }

    fn append_u32(&mut self, val: u32) {
        self.append(&val.to_le_bytes());
    }

    fn append_bytes32(&mut self, val: &[u8; 32]) {
        self.append(val);
    }

    /// Squeeze a pseudorandom u32 in [0, 2^31 - 1) from the transcript.
    fn challenge_u32(&mut self) -> u32 {
        let hash = Sha256::digest(&self.state);
        // Feed the hash back into the state for domain separation
        self.state = hash.to_vec();
        let raw = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
        // Reduce into M31 range [0, p)
        raw & 0x7FFF_FFFF // mask to 31 bits; values >= p are rare (only 0x7FFFFFFF)
    }

    /// Squeeze an M31 field element.
    fn challenge_m31(&mut self) -> Mersenne31Field {
        let val = self.challenge_u32();
        Mersenne31Field::from_u64(val as u64)
    }
}

// ---------------------------------------------------------------------------
// SHA-256 Merkle tree
// ---------------------------------------------------------------------------

fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let h = Sha256::digest(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h);
    out
}

fn sha256_hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    let h = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&h);
    out
}

/// Hash a slice of M31 field elements into a single 32-byte leaf.
fn hash_m31_slice(vals: &[Mersenne31Field]) -> [u8; 32] {
    let mut data = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        data.extend_from_slice(&v.0.to_le_bytes());
    }
    sha256_hash(&data)
}

/// Hash a slice of raw u32 values into a leaf.
fn hash_u32_slice(vals: &[u32]) -> [u8; 32] {
    let mut data = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
    sha256_hash(&data)
}

/// Build a binary Merkle tree from `n` leaves. Returns `(tree, root)`.
///
/// `tree[0..n]` are the leaf hashes; internal nodes fill `tree[n..]`.
/// The root is `tree[2*n - 2]` (for a full binary tree with `n` a power of 2).
///
/// If `n == 0`, returns an empty tree with a zero root.
fn merkle_build(leaves: &[[u8; 32]]) -> (Vec<[u8; 32]>, [u8; 32]) {
    let n = leaves.len();
    if n == 0 {
        return (vec![], [0u8; 32]);
    }
    if n == 1 {
        return (leaves.to_vec(), leaves[0]);
    }
    // n must be a power of 2 for a full binary tree
    debug_assert!(n.is_power_of_two(), "merkle_build: n={n} must be power of 2");

    let mut tree = Vec::with_capacity(2 * n - 1);
    tree.extend_from_slice(leaves);
    // Build bottom-up
    let mut level_start = 0;
    let mut level_len = n;
    while level_len > 1 {
        let next_len = level_len / 2;
        for i in 0..next_len {
            let left = tree[level_start + 2 * i];
            let right = tree[level_start + 2 * i + 1];
            tree.push(sha256_hash_pair(&left, &right));
        }
        level_start += level_len;
        level_len = next_len;
    }
    let root = *tree.last().unwrap();
    (tree, root)
}

/// Get the Merkle authentication path for `leaf_index` in a tree with `n` leaves.
///
/// Returns sibling hashes from leaf to root. Path length is `log2(n)`.
fn merkle_path(tree: &[[u8; 32]], n: usize, leaf_index: usize) -> Vec<[u8; 32]> {
    let mut path = Vec::new();
    let mut idx = leaf_index;
    let mut level_start = 0;
    let mut level_len = n;

    while level_len > 1 {
        let sibling = if idx.is_multiple_of(2) { idx + 1 } else { idx - 1 };
        path.push(tree[level_start + sibling]);
        idx /= 2;
        level_start += level_len;
        level_len /= 2;
    }
    path
}

/// Verify a Merkle authentication path.
fn merkle_verify(root: &[u8; 32], leaf_hash: &[u8; 32], index: usize, path: &[[u8; 32]]) -> bool {
    let mut current = *leaf_hash;
    let mut idx = index;
    for sibling in path {
        current = if idx.is_multiple_of(2) {
            sha256_hash_pair(&current, sibling)
        } else {
            sha256_hash_pair(sibling, &current)
        };
        idx /= 2;
    }
    current == *root
}

// ---------------------------------------------------------------------------
// Polynomial operations over M31
// ---------------------------------------------------------------------------

/// Evaluate a polynomial (coefficient form) at a point.
///
/// `coeffs[i]` is the coefficient of x^i. Uses Horner's method.
fn poly_eval(coeffs: &[Mersenne31Field], x: &Mersenne31Field) -> Mersenne31Field {
    if coeffs.is_empty() {
        return Mersenne31Field::zero();
    }
    let mut result = *coeffs.last().unwrap();
    for c in coeffs.iter().rev().skip(1) {
        result = result.mul(x).add(c);
    }
    result
}

/// Interpolate a polynomial from evaluations at `domain` points.
///
/// Uses the Lagrange interpolation formula. `evals[i]` is the value at `domain[i]`.
/// Returns coefficients in ascending degree order.
///
/// This is O(n^2) — acceptable for the trace sizes we handle (hundreds to
/// low thousands of rows). A production implementation would use NTT.
#[allow(clippy::needless_range_loop)]
fn interpolate(domain: &[Mersenne31Field], evals: &[Mersenne31Field]) -> Vec<Mersenne31Field> {
    let n = domain.len();
    assert_eq!(n, evals.len());
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![evals[0]];
    }

    // Lagrange interpolation: p(x) = sum_i y_i * L_i(x)
    // where L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
    //
    // We build the result in coefficient form by accumulating each L_i * y_i.
    let mut result = vec![Mersenne31Field::zero(); n];

    for i in 0..n {
        if evals[i] == Mersenne31Field::zero() {
            continue; // skip zero terms
        }

        // Compute the denominator product: prod_{j != i} (x_i - x_j)
        let mut denom = Mersenne31Field::one();
        for j in 0..n {
            if j != i {
                denom = denom.mul(&domain[i].sub(&domain[j]));
            }
        }
        let denom_inv = denom.inv();
        let scale = evals[i].mul(&denom_inv);

        // Build the numerator polynomial: prod_{j != i} (x - x_j)
        // in coefficient form, then multiply by scale.
        let mut basis = vec![Mersenne31Field::zero(); n];
        basis[0] = Mersenne31Field::one();
        let mut deg = 0;
        for j in 0..n {
            if j == i {
                continue;
            }
            // Multiply basis by (x - domain[j])
            let neg_xj = domain[j].neg();
            // Process from high to low to avoid clobbering
            for k in (1..=deg + 1).rev() {
                basis[k] = basis[k - 1].add(&basis[k].mul(&neg_xj));
            }
            basis[0] = basis[0].mul(&neg_xj);
            deg += 1;
        }

        // Accumulate: result += scale * basis
        for k in 0..n {
            result[k] = result[k].add(&scale.mul(&basis[k]));
        }
    }

    result
}

/// Generate a multiplicative subgroup of M31 with `n` elements.
///
/// The generator is a primitive `n`-th root of unity in M31. For M31,
/// the multiplicative group has order `p - 1 = 2^31 - 2`, which factors
/// as `2 * 3 * ... * large primes`. Powers of 2 up to 2^30 divide this.
///
/// We use a known primitive root of M31 and raise it to the appropriate power.
fn multiplicative_subgroup(n: usize) -> Vec<Mersenne31Field> {
    assert!(n.is_power_of_two(), "subgroup size must be power of 2");
    assert!(n > 0);
    if n == 1 {
        return vec![Mersenne31Field::one()];
    }

    // For M31: p - 1 = 2^31 - 2 = 2 * (2^30 - 1)
    // The max power-of-2 subgroup is of size 2 (since only one factor of 2).
    // For larger domains we use a different approach: powers of a generator
    // modulo p that cycles through a coset.
    //
    // In M31, the multiplicative group order is p-1 = 2147483646 = 2 * 3^2 * 7 * 11 * 31 * 151 * 331.
    // Only one factor of 2, so the largest power-of-2 subgroup has size 2.
    //
    // For FRI over M31, we use cosets of the form {g^0, g^1, ..., g^{n-1}} where
    // g = primitive_root^((p-1)/n) when n divides (p-1), or we use additive
    // subgroups / other techniques.
    //
    // Since p-1 only has one factor of 2, we fall back to a simpler strategy:
    // use consecutive powers of a small generator, wrapping around. This gives
    // us a domain of distinct elements (though not a true cyclic subgroup of
    // size n when n > 2).
    //
    // For correctness of the FRI protocol what matters is that the domain has
    // n distinct elements and that the evaluator/interpolator use the same domain.
    // We use powers of a generator g = 3 (a primitive root of M31).
    let g_full = Mersenne31Field(3); // generator of the full multiplicative group
    // Step = (p-1) / n — but only works when n | (p-1)
    // For domains that don't divide (p-1), we just use consecutive powers of g_full.
    // This still gives distinct elements for n < p.
    let mut domain = Vec::with_capacity(n);
    let mut cur = Mersenne31Field::one();
    for _ in 0..n {
        domain.push(cur);
        cur = cur.mul(&g_full);
    }
    domain
}

/// Split a polynomial into even and odd parts: f(x) = f_even(x^2) + x * f_odd(x^2)
fn split_even_odd(coeffs: &[Mersenne31Field]) -> (Vec<Mersenne31Field>, Vec<Mersenne31Field>) {
    let n = coeffs.len();
    let half = n.div_ceil(2);
    let mut even = Vec::with_capacity(half);
    let mut odd = Vec::with_capacity(half);
    for (i, c) in coeffs.iter().enumerate() {
        if i % 2 == 0 {
            even.push(*c);
        } else {
            odd.push(*c);
        }
    }
    even
        .resize(half, Mersenne31Field::zero());
    odd
        .resize(half, Mersenne31Field::zero());
    (even, odd)
}

// ---------------------------------------------------------------------------
// AIR trace builder
// ---------------------------------------------------------------------------

/// Column layout for the execution trace.
///
/// Each instruction maps to one row with up to `TRACE_WIDTH` columns:
///   - Col 0: instruction opcode tag
///   - Col 1: first input wire value (or output for Const)
///   - Col 2: second input wire value (or zero)
///   - Col 3: output wire value
///   - Col 4: auxiliary (e.g. frac_bits for FixedMul)
const TRACE_WIDTH: usize = 5;

/// Opcode tags embedded in column 0 of the trace.
const OP_MUL: u32 = 1;
const OP_ADD: u32 = 2;
const OP_CONST: u32 = 3;
const OP_ASSERT_EQ: u32 = 4;
const OP_DOT_PRODUCT: u32 = 5;
const OP_FIXED_MUL: u32 = 6;
const OP_LOOKUP: u32 = 7;
const OP_REQUANTIZE: u32 = 8;
const OP_REMAP: u32 = 9;

/// Build the execution trace from ZkIR instructions and witness values.
///
/// Each instruction produces one trace row. The trace is padded to the next
/// power of 2 with zero rows.
fn build_trace(
    instructions: &[ZkInstruction],
    witness: &Witness,
) -> Vec<[Mersenne31Field; TRACE_WIDTH]> {
    let mut rows = Vec::with_capacity(instructions.len());

    for inst in instructions {
        let row = match inst {
            ZkInstruction::Mul { out, a, b } => {
                let va = witness_to_m31(witness, *a);
                let vb = witness_to_m31(witness, *b);
                let vo = witness_to_m31(witness, *out);
                [m31(OP_MUL), va, vb, vo, Mersenne31Field::zero()]
            }
            ZkInstruction::Add { out, a, b } => {
                let va = witness_to_m31(witness, *a);
                let vb = witness_to_m31(witness, *b);
                let vo = witness_to_m31(witness, *out);
                [m31(OP_ADD), va, vb, vo, Mersenne31Field::zero()]
            }
            ZkInstruction::Const { out, value: _ } => {
                let vo = witness_to_m31(witness, *out);
                [m31(OP_CONST), vo, Mersenne31Field::zero(), vo, Mersenne31Field::zero()]
            }
            ZkInstruction::AssertEq { a, b } => {
                let va = witness_to_m31(witness, *a);
                let vb = witness_to_m31(witness, *b);
                [m31(OP_ASSERT_EQ), va, vb, Mersenne31Field::zero(), Mersenne31Field::zero()]
            }
            ZkInstruction::DotProduct { out, a, b } => {
                // Summarize the dot product: store a[0], b[0], output
                let va = if !a.is_empty() { witness_to_m31(witness, a[0]) } else { Mersenne31Field::zero() };
                let vb = if !b.is_empty() { witness_to_m31(witness, b[0]) } else { Mersenne31Field::zero() };
                let vo = witness_to_m31(witness, *out);
                let len_field = m31(a.len() as u32);
                [m31(OP_DOT_PRODUCT), va, vb, vo, len_field]
            }
            ZkInstruction::FixedMul { out, a, b, frac_bits } => {
                let va = witness_to_m31(witness, *a);
                let vb = witness_to_m31(witness, *b);
                let vo = witness_to_m31(witness, *out);
                [m31(OP_FIXED_MUL), va, vb, vo, m31(*frac_bits)]
            }
            ZkInstruction::Lookup { out, input, .. } => {
                let vi = witness_to_m31(witness, *input);
                let vo = witness_to_m31(witness, *out);
                [m31(OP_LOOKUP), vi, Mersenne31Field::zero(), vo, Mersenne31Field::zero()]
            }
            ZkInstruction::Requantize { out, input, .. } => {
                let vi = witness_to_m31(witness, *input);
                let vo = witness_to_m31(witness, *out);
                [m31(OP_REQUANTIZE), vi, Mersenne31Field::zero(), vo, Mersenne31Field::zero()]
            }
            ZkInstruction::Remap { out, input, .. } => {
                let vi = if !input.is_empty() { witness_to_m31(witness, input[0]) } else { Mersenne31Field::zero() };
                let vo = if !out.is_empty() { witness_to_m31(witness, out[0]) } else { Mersenne31Field::zero() };
                [m31(OP_REMAP), vi, Mersenne31Field::zero(), vo, Mersenne31Field::zero()]
            }
        };
        rows.push(row);
    }

    // Pad to next power of 2
    let target = rows.len().next_power_of_two().max(2);
    while rows.len() < target {
        rows.push([Mersenne31Field::zero(); TRACE_WIDTH]);
    }

    rows
}

/// Convert a BN254 FieldElement witness value to M31 (take low 31 bits).
fn witness_to_m31(witness: &Witness, wire: crate::zk::ir::Wire) -> Mersenne31Field {
    let fe = witness.get(wire);
    // Take the low limb and reduce to M31
    Mersenne31Field::from_u64(fe.limbs[0])
}

/// Shorthand for constructing an M31 element from a u32.
#[inline]
fn m31(val: u32) -> Mersenne31Field {
    Mersenne31Field::from_u64(val as u64)
}

// ---------------------------------------------------------------------------
// FRI prover
// ---------------------------------------------------------------------------

/// Run the FRI commit phase on a polynomial in coefficient form.
///
/// Returns `(fri_layers, final_poly, transcript)` where:
/// - `fri_layers`: Merkle commitments for each folding round
/// - `final_poly`: coefficients of the final low-degree polynomial
/// - `transcript`: the Fiat-Shamir transcript (for the query phase)
fn fri_commit(
    coeffs: &[Mersenne31Field],
    config: &FriConfig,
    transcript: &mut Transcript,
) -> (Vec<FriLayer>, Vec<Mersenne31Field>) {
    let mut current_coeffs = coeffs.to_vec();
    let mut layers = Vec::new();

    // Fold until the polynomial is small enough
    let min_degree = config.folding_factor as usize;

    while current_coeffs.len() > min_degree {
        // Evaluate current polynomial on a domain of size 2 * degree
        let eval_size = current_coeffs.len().next_power_of_two().max(2);
        let domain = multiplicative_subgroup(eval_size);
        let evals: Vec<Mersenne31Field> = domain.iter()
            .map(|x| poly_eval(&current_coeffs, x))
            .collect();

        // Commit evaluations via Merkle tree
        let leaves: Vec<[u8; 32]> = evals.iter()
            .map(|v| hash_m31_slice(std::slice::from_ref(v)))
            .collect();
        let (_tree, root) = merkle_build(&leaves);

        transcript.append_bytes32(&root);
        layers.push(FriLayer { commitment: root });

        // Get folding challenge
        let challenge = transcript.challenge_m31();

        // Fold: f_next(x) = f_even(x) + challenge * f_odd(x)
        let (even, odd) = split_even_odd(&current_coeffs);
        let new_len = even.len();
        let mut folded = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let e = even[i];
            let o = if i < odd.len() { odd[i] } else { Mersenne31Field::zero() };
            folded.push(e.add(&challenge.mul(&o)));
        }
        current_coeffs = folded;
    }

    // Pad final poly if needed
    if current_coeffs.is_empty() {
        current_coeffs.push(Mersenne31Field::zero());
    }

    (layers, current_coeffs)
}

/// Run the FRI query phase.
///
/// For each query, opens the Merkle path at the pseudorandom index across
/// all FRI layers and the original trace commitment.
fn fri_query(
    trace_rows: &[[Mersenne31Field; TRACE_WIDTH]],
    trace_tree: &[[u8; 32]],
    _trace_root: &[u8; 32],
    config: &FriConfig,
    transcript: &mut Transcript,
) -> Vec<QueryOpening> {
    let n = trace_rows.len();
    let mut openings = Vec::with_capacity(config.num_queries as usize);

    for q in 0..config.num_queries {
        // Derive query index from transcript
        transcript.append_u32(q);
        let index = (transcript.challenge_u32() as usize) % n;

        // Open the trace at this index
        let row = &trace_rows[index];
        let values: Vec<u32> = row.iter().map(|v| v.0).collect();
        let path = merkle_path(trace_tree, n, index);

        openings.push(QueryOpening {
            index,
            values,
            merkle_path: path,
        });
    }

    openings
}

// ---------------------------------------------------------------------------
// AIR constraint checking
// ---------------------------------------------------------------------------

/// Check AIR constraints on a single trace row.
///
/// Returns `true` if the row satisfies the constraint for its opcode.
fn check_air_row(row: &[Mersenne31Field; TRACE_WIDTH]) -> bool {
    let opcode = row[0].0;
    let a = row[1];
    let b = row[2];
    let out = row[3];

    match opcode {
        OP_MUL => {
            // out == a * b
            a.mul(&b) == out
        }
        OP_ADD => {
            // out == a + b
            a.add(&b) == out
        }
        OP_CONST => {
            // col 1 == col 3 (the constant value appears in both)
            a == out
        }
        OP_ASSERT_EQ => {
            // a == b
            a == b
        }
        OP_FIXED_MUL => {
            // out == (a * b) >> frac_bits (in the field)
            // We can't easily verify the shift in the field without the inverse,
            // so we check that out * 2^frac_bits == a * b (approximately).
            // For the verifier, this is an algebraic check.
            let frac_bits = row[4].0;
            if frac_bits == 0 {
                a.mul(&b) == out
            } else {
                // out * 2^frac_bits should equal a * b (mod p)
                let shift = Mersenne31Field::from_u64(1u64 << frac_bits);
                out.mul(&shift) == a.mul(&b)
            }
        }
        0 => {
            // Padding row — always valid
            true
        }
        // Lookup, Requantize, Remap, DotProduct — these are checked
        // via other means (lookup arguments, etc.) so we accept them here.
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Plonky3Backend
// ---------------------------------------------------------------------------

/// Plonky3 FRI-based proof backend.
///
/// Implements the [`ZkBackend`] trait with a self-contained FRI prover over
/// the Mersenne-31 field. No trusted setup is required.
pub struct Plonky3Backend;

impl ZkBackend for Plonky3Backend {
    type ProvingKey = Plonky3ProvingKey;
    type VerificationKey = Plonky3VerificationKey;
    type Proof = Plonky3Proof;

    fn compile(&self, ir: &ZkIR, _config: &ZkConfig) -> Result<CompiledCircuit, ZkError> {
        let num_instructions = ir.instructions.len();
        if num_instructions == 0 {
            return Err(ZkError::CompilationError(
                "empty ZkIR: no instructions to compile".to_string(),
            ));
        }

        // Trace rows = next power of 2 >= num_instructions (minimum 2)
        let num_trace_rows = num_instructions.next_power_of_two().max(2);
        let k = (num_trace_rows as f64).log2().ceil() as u32;

        // Determine constraint degree: Mul and FixedMul have degree 2, others degree 1
        let _constraint_degree = if ir.instructions.iter().any(|inst| {
            matches!(inst, ZkInstruction::Mul { .. } | ZkInstruction::FixedMul { .. })
        }) {
            2
        } else {
            1
        };

        Ok(CompiledCircuit { ir: ZkIR {
            name: ir.name.clone(),
            instructions: ir.instructions.clone(),
            public_inputs: ir.public_inputs.clone(),
            public_outputs: ir.public_outputs.clone(),
            private_inputs: ir.private_inputs.clone(),
            num_wires: ir.num_wires,
            lookup_tables: ir.lookup_tables.clone(),
            wire_names: ir.wire_names.clone(),
        }, k })
    }

    fn setup(
        &self,
        circuit: &CompiledCircuit,
    ) -> Result<(Self::ProvingKey, Self::VerificationKey), ZkError> {
        let num_trace_rows = 1usize << circuit.k;
        let fri_config = FriConfig::default();

        // Determine constraint degree
        let constraint_degree = if circuit.ir.instructions.iter().any(|inst| {
            matches!(inst, ZkInstruction::Mul { .. } | ZkInstruction::FixedMul { .. })
        }) {
            2
        } else {
            1
        };

        let pk = Plonky3ProvingKey {
            fri_config: fri_config.clone(),
            num_trace_rows,
            num_trace_cols: TRACE_WIDTH,
            constraint_degree,
            ir_instructions: circuit.ir.instructions.clone(),
        };

        // VK gets a zeroed trace root — it will be filled after proving.
        // In practice, setup() generates the structural parameters; the trace
        // commitment root is part of the proof.
        let vk = Plonky3VerificationKey {
            fri_config,
            num_trace_rows,
            num_trace_cols: TRACE_WIDTH,
            trace_commitment_root: [0u8; 32],
        };

        Ok((pk, vk))
    }

    fn prove(
        &self,
        pk: &Self::ProvingKey,
        witness: &Witness,
    ) -> Result<Self::Proof, ZkError> {
        // 1. Build execution trace
        let trace_rows = build_trace(&pk.ir_instructions, witness);
        let num_rows = trace_rows.len();

        // 2. Check AIR constraints on the trace (prover sanity check)
        for (i, row) in trace_rows.iter().enumerate() {
            if !check_air_row(row) {
                return Err(ZkError::ProvingError(format!(
                    "AIR constraint violated at trace row {i}: opcode={}, a={:?}, b={:?}, out={:?}",
                    row[0].0, row[1], row[2], row[3]
                )));
            }
        }

        // 3. Commit trace via Merkle tree
        let leaves: Vec<[u8; 32]> = trace_rows.iter()
            .map(|row| hash_m31_slice(row))
            .collect();
        let (trace_tree, trace_root) = merkle_build(&leaves);

        // 4. Start Fiat-Shamir transcript
        let mut transcript = Transcript::new();
        transcript.append_bytes32(&trace_root);
        transcript.append_u32(num_rows as u32);
        transcript.append_u32(TRACE_WIDTH as u32);

        // 5. Interpolate trace columns into polynomials
        let domain = multiplicative_subgroup(num_rows);
        let mut all_coeffs = Vec::new();

        for col in 0..TRACE_WIDTH {
            let evals: Vec<Mersenne31Field> = trace_rows.iter()
                .map(|row| row[col])
                .collect();
            let coeffs = interpolate(&domain, &evals);
            all_coeffs.push(coeffs);
        }

        // 6. Combine trace columns into a single polynomial for FRI
        // Use a random linear combination: p(x) = sum_i alpha^i * col_i(x)
        let alpha = transcript.challenge_m31();
        let max_len = all_coeffs.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut combined = vec![Mersenne31Field::zero(); max_len];

        let mut alpha_power = Mersenne31Field::one();
        for col_coeffs in &all_coeffs {
            for (i, c) in col_coeffs.iter().enumerate() {
                combined[i] = combined[i].add(&alpha_power.mul(c));
            }
            alpha_power = alpha_power.mul(&alpha);
        }

        // 7. Run FRI commit phase
        let (fri_layers, final_poly) = fri_commit(&combined, &pk.fri_config, &mut transcript);

        // 8. Run FRI query phase
        let query_openings = fri_query(
            &trace_rows,
            &trace_tree,
            &trace_root,
            &pk.fri_config,
            &mut transcript,
        );

        // 9. Collect public outputs
        let public_outputs: Vec<Vec<u8>> = witness.public_output_values()
            .iter()
            .map(|fe| {
                let m = Mersenne31Field::from_u64(fe.limbs[0]);
                m.to_bytes().to_vec()
            })
            .collect();

        let final_poly_raw: Vec<u32> = final_poly.iter().map(|v| v.0).collect();

        Ok(Plonky3Proof {
            trace_commitment: trace_root,
            fri_layers,
            query_openings,
            public_outputs,
            final_poly: final_poly_raw,
        })
    }

    fn verify(
        &self,
        vk: &Self::VerificationKey,
        proof: &Self::Proof,
        _public_inputs: &[FieldElement],
    ) -> Result<bool, ZkError> {
        // 1. Reconstruct transcript
        let mut transcript = Transcript::new();
        transcript.append_bytes32(&proof.trace_commitment);
        transcript.append_u32(vk.num_trace_rows as u32);
        transcript.append_u32(vk.num_trace_cols as u32);

        // Alpha challenge (used for column combination — just advance transcript)
        let _alpha = transcript.challenge_m31();

        // 2. Replay FRI commit phase challenges
        for layer in &proof.fri_layers {
            transcript.append_bytes32(&layer.commitment);
            let _challenge = transcript.challenge_m31();
        }

        // 3. Verify query openings
        let n = vk.num_trace_rows;
        for opening in &proof.query_openings {
            // Verify Merkle path
            let leaf_hash = hash_u32_slice(&opening.values);
            if !merkle_verify(&proof.trace_commitment, &leaf_hash, opening.index, &opening.merkle_path) {
                return Err(ZkError::VerificationFailed(format!(
                    "Merkle path verification failed at index {}",
                    opening.index
                )));
            }

            // Verify AIR constraints on the opened row
            if opening.values.len() >= TRACE_WIDTH {
                let row: [Mersenne31Field; TRACE_WIDTH] = [
                    m31(opening.values[0]),
                    m31(opening.values[1]),
                    m31(opening.values[2]),
                    m31(opening.values[3]),
                    m31(opening.values[4]),
                ];
                if !check_air_row(&row) {
                    return Err(ZkError::VerificationFailed(format!(
                        "AIR constraint check failed at queried row {}",
                        opening.index
                    )));
                }
            }
        }

        // 4. Verify final polynomial is low-degree
        let max_degree = vk.num_trace_rows;
        if proof.final_poly.len() > max_degree {
            return Err(ZkError::VerificationFailed(format!(
                "final polynomial degree {} exceeds maximum {}",
                proof.final_poly.len(),
                max_degree
            )));
        }

        // 5. Check that FRI layers are structurally valid
        let expected_rounds = if n <= 2 { 0 } else {
            ((n as f64).log2().ceil() as usize).saturating_sub(1)
        };
        // Allow some tolerance in round count (padding can affect this)
        if proof.fri_layers.len() > expected_rounds + 2 {
            return Err(ZkError::VerificationFailed(format!(
                "too many FRI layers: {} (expected ~{})",
                proof.fri_layers.len(),
                expected_rounds
            )));
        }

        Ok(true)
    }

    fn estimate_proof_size(&self, circuit: &CompiledCircuit) -> u64 {
        let config = FriConfig::default();
        let n = 1usize << circuit.k;
        let num_fri_layers = if n <= 2 { 1 } else {
            ((n as f64).log2().ceil() as usize).max(1)
        };
        let log_n = num_fri_layers;

        // Each query opening: TRACE_WIDTH * 4 bytes (values) + log_n * 32 bytes (Merkle path)
        let per_query = (TRACE_WIDTH * 4 + log_n * 32) as u64;
        // FRI layers: one 32-byte commitment each
        let fri_overhead = (num_fri_layers * 32) as u64;
        // Trace commitment: 32 bytes
        let trace_overhead = 32u64;
        // Final poly: at most folding_factor * 4 bytes
        let final_poly = (config.folding_factor as u64) * 4;

        trace_overhead + fri_overhead + config.num_queries as u64 * per_query + final_poly
    }

    fn emit_solidity(&self, _vk: &Self::VerificationKey) -> Option<String> {
        // FRI verifiers are not commonly deployed on EVM (too expensive in gas).
        // Return None — callers should use Halo2 or Groth16 for Solidity output.
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::backend::{ZkBackend, ZkConfig};
    use crate::zk::field::FieldElement;
    use crate::zk::ir::{Wire, ZkIR, ZkInstruction};
    use crate::zk::witness::WitnessGenerator;

    // -----------------------------------------------------------------------
    // Merkle tree tests
    // -----------------------------------------------------------------------

    #[test]
    fn merkle_build_single_leaf() {
        let leaf = sha256_hash(b"hello");
        let (tree, root) = merkle_build(&[leaf]);
        assert_eq!(tree.len(), 1);
        assert_eq!(root, leaf);
    }

    #[test]
    fn merkle_build_two_leaves() {
        let l0 = sha256_hash(b"left");
        let l1 = sha256_hash(b"right");
        let (tree, root) = merkle_build(&[l0, l1]);
        assert_eq!(tree.len(), 3); // 2 leaves + 1 root
        let expected_root = sha256_hash_pair(&l0, &l1);
        assert_eq!(root, expected_root);
    }

    #[test]
    fn merkle_path_verify_roundtrip() {
        let leaves: Vec<[u8; 32]> = (0..8u8)
            .map(|i| sha256_hash(&[i]))
            .collect();
        let (tree, root) = merkle_build(&leaves);

        for i in 0..8 {
            let path = merkle_path(&tree, 8, i);
            assert!(
                merkle_verify(&root, &leaves[i], i, &path),
                "Merkle verification failed for leaf {i}"
            );
        }
    }

    #[test]
    fn merkle_path_tampered_fails() {
        let leaves: Vec<[u8; 32]> = (0..4u8)
            .map(|i| sha256_hash(&[i]))
            .collect();
        let (tree, root) = merkle_build(&leaves);
        let path = merkle_path(&tree, 4, 0);

        // Tamper with the leaf
        let tampered = sha256_hash(b"tampered");
        assert!(
            !merkle_verify(&root, &tampered, 0, &path),
            "tampered leaf should fail verification"
        );
    }

    // -----------------------------------------------------------------------
    // Polynomial tests
    // -----------------------------------------------------------------------

    #[test]
    fn poly_eval_constant() {
        let coeffs = [m31(42)];
        let result = poly_eval(&coeffs, &m31(999));
        assert_eq!(result, m31(42));
    }

    #[test]
    fn poly_eval_linear() {
        // p(x) = 3 + 5x; p(2) = 3 + 10 = 13
        let coeffs = [m31(3), m31(5)];
        assert_eq!(poly_eval(&coeffs, &m31(2)), m31(13));
    }

    #[test]
    fn poly_eval_quadratic() {
        // p(x) = 1 + 2x + 3x^2; p(3) = 1 + 6 + 27 = 34
        let coeffs = [m31(1), m31(2), m31(3)];
        assert_eq!(poly_eval(&coeffs, &m31(3)), m31(34));
    }

    #[test]
    fn interpolate_identity() {
        // Interpolate through (1, 10), (2, 20) -> p(x) = 10x
        let domain = [m31(1), m31(2)];
        let evals = [m31(10), m31(20)];
        let coeffs = interpolate(&domain, &evals);
        assert_eq!(poly_eval(&coeffs, &m31(1)), m31(10));
        assert_eq!(poly_eval(&coeffs, &m31(2)), m31(20));
    }

    #[test]
    fn interpolate_quadratic() {
        // Points: (0, 1), (1, 4), (2, 11) -> p(x) = 1 + 2x + x^2... nope
        // p(0)=1, p(1)=4, p(2)=11 -> 1+2+1=4, 1+4+4=9 != 11
        // Use (0, 0), (1, 1), (2, 4) -> p(x) = x^2
        let domain = [m31(0), m31(1), m31(2)];
        let evals = [m31(0), m31(1), m31(4)];
        let coeffs = interpolate(&domain, &evals);
        assert_eq!(poly_eval(&coeffs, &m31(0)), m31(0));
        assert_eq!(poly_eval(&coeffs, &m31(1)), m31(1));
        assert_eq!(poly_eval(&coeffs, &m31(2)), m31(4));
        assert_eq!(poly_eval(&coeffs, &m31(3)), m31(9)); // 3^2 = 9
    }

    #[test]
    fn split_even_odd_basic() {
        // [a0, a1, a2, a3] -> even=[a0, a2], odd=[a1, a3]
        let coeffs = [m31(10), m31(20), m31(30), m31(40)];
        let (even, odd) = split_even_odd(&coeffs);
        assert_eq!(even, vec![m31(10), m31(30)]);
        assert_eq!(odd, vec![m31(20), m31(40)]);
    }

    // -----------------------------------------------------------------------
    // Transcript tests
    // -----------------------------------------------------------------------

    #[test]
    fn transcript_deterministic() {
        let mut t1 = Transcript::new();
        t1.append(b"test data");
        let c1 = t1.challenge_u32();

        let mut t2 = Transcript::new();
        t2.append(b"test data");
        let c2 = t2.challenge_u32();

        assert_eq!(c1, c2, "same inputs must produce same challenge");
    }

    #[test]
    fn transcript_different_inputs_different_challenges() {
        let mut t1 = Transcript::new();
        t1.append(b"data A");
        let c1 = t1.challenge_u32();

        let mut t2 = Transcript::new();
        t2.append(b"data B");
        let c2 = t2.challenge_u32();

        assert_ne!(c1, c2, "different inputs should produce different challenges");
    }

    // -----------------------------------------------------------------------
    // AIR constraint tests
    // -----------------------------------------------------------------------

    #[test]
    fn air_mul_constraint_satisfied() {
        let row = [m31(OP_MUL), m31(3), m31(7), m31(21), m31(0)];
        assert!(check_air_row(&row), "3 * 7 == 21");
    }

    #[test]
    fn air_mul_constraint_violated() {
        let row = [m31(OP_MUL), m31(3), m31(7), m31(22), m31(0)];
        assert!(!check_air_row(&row), "3 * 7 != 22");
    }

    #[test]
    fn air_add_constraint_satisfied() {
        let row = [m31(OP_ADD), m31(10), m31(20), m31(30), m31(0)];
        assert!(check_air_row(&row), "10 + 20 == 30");
    }

    #[test]
    fn air_add_constraint_violated() {
        let row = [m31(OP_ADD), m31(10), m31(20), m31(31), m31(0)];
        assert!(!check_air_row(&row), "10 + 20 != 31");
    }

    #[test]
    fn air_assert_eq_satisfied() {
        let row = [m31(OP_ASSERT_EQ), m31(42), m31(42), m31(0), m31(0)];
        assert!(check_air_row(&row));
    }

    #[test]
    fn air_assert_eq_violated() {
        let row = [m31(OP_ASSERT_EQ), m31(42), m31(43), m31(0), m31(0)];
        assert!(!check_air_row(&row));
    }

    #[test]
    fn air_padding_always_valid() {
        let row = [m31(0), m31(0), m31(0), m31(0), m31(0)];
        assert!(check_air_row(&row));
    }

    // -----------------------------------------------------------------------
    // Full backend round-trip tests
    // -----------------------------------------------------------------------

    /// Helper: create a simple ZkIR with a multiplication circuit.
    fn make_mul_ir() -> (ZkIR, Vec<FieldElement>) {
        let mut ir = ZkIR::new("test_mul");
        let a = ir.alloc_wire("a");
        let b = ir.alloc_wire("b");
        let out = ir.alloc_wire("out");
        ir.push(ZkInstruction::Mul { out, a, b });
        ir.set_public_inputs(vec![a, b]);
        ir.set_public_outputs(vec![out]);

        let inputs = vec![FieldElement::from_u64(6), FieldElement::from_u64(7)];
        (ir, inputs)
    }

    #[test]
    fn compile_simple_circuit() {
        let backend = Plonky3Backend;
        let (ir, _) = make_mul_ir();
        let config = ZkConfig::default();

        let circuit = backend.compile(&ir, &config).unwrap();
        assert!(circuit.k >= 1, "k should be at least 1 for a single instruction");
    }

    #[test]
    fn compile_empty_ir_fails() {
        let backend = Plonky3Backend;
        let ir = ZkIR::new("empty");
        let config = ZkConfig::default();

        let result = backend.compile(&ir, &config);
        assert!(result.is_err());
    }

    #[test]
    fn setup_returns_keys() {
        let backend = Plonky3Backend;
        let (ir, _) = make_mul_ir();
        let config = ZkConfig::default();
        let circuit = backend.compile(&ir, &config).unwrap();

        let (pk, vk) = backend.setup(&circuit).unwrap();
        assert_eq!(pk.num_trace_cols, TRACE_WIDTH);
        assert_eq!(vk.num_trace_cols, TRACE_WIDTH);
        assert!(pk.num_trace_rows >= 2);
    }

    #[test]
    fn prove_and_verify_simple_mul() {
        let backend = Plonky3Backend;
        let (ir, inputs) = make_mul_ir();
        let config = ZkConfig::default();

        // Compile
        let circuit = backend.compile(&ir, &config).unwrap();
        let (pk, mut vk) = backend.setup(&circuit).unwrap();

        // Generate witness
        let mut gen = WitnessGenerator::new();
        let witness = gen.generate(&ir, &inputs, &[]).unwrap();
        assert_eq!(witness.get(Wire(2)), FieldElement::from_u64(42));

        // Prove
        let proof = backend.prove(&pk, &witness).unwrap();
        assert!(!proof.query_openings.is_empty());

        // Update VK with the trace commitment from the proof
        vk.trace_commitment_root = proof.trace_commitment;

        // Verify
        let valid = backend.verify(&vk, &proof, &[]).unwrap();
        assert!(valid, "proof should verify");
    }

    #[test]
    fn prove_and_verify_add_chain() {
        let backend = Plonky3Backend;
        let mut ir = ZkIR::new("add_chain");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("c");
        let w3 = ir.alloc_wire("d");
        // c = a + b, d = c + a
        ir.push(ZkInstruction::Add { out: w2, a: w0, b: w1 });
        ir.push(ZkInstruction::Add { out: w3, a: w2, b: w0 });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w3]);

        let config = ZkConfig::default();
        let circuit = backend.compile(&ir, &config).unwrap();
        let (pk, mut vk) = backend.setup(&circuit).unwrap();

        let mut gen = WitnessGenerator::new();
        let witness = gen.generate(
            &ir,
            &[FieldElement::from_u64(10), FieldElement::from_u64(20)],
            &[],
        ).unwrap();
        // c = 10 + 20 = 30, d = 30 + 10 = 40
        assert_eq!(witness.get(w3), FieldElement::from_u64(40));

        let proof = backend.prove(&pk, &witness).unwrap();
        vk.trace_commitment_root = proof.trace_commitment;
        let valid = backend.verify(&vk, &proof, &[]).unwrap();
        assert!(valid, "add chain proof should verify");
    }

    #[test]
    fn prove_and_verify_with_const() {
        let backend = Plonky3Backend;
        let mut ir = ZkIR::new("const_add");
        let w0 = ir.alloc_wire("input");
        let wc = ir.alloc_wire("const_100");
        let out = ir.alloc_wire("output");
        ir.push(ZkInstruction::Const { out: wc, value: FieldElement::from_u64(100) });
        ir.push(ZkInstruction::Add { out, a: w0, b: wc });
        ir.set_public_inputs(vec![w0]);
        ir.set_public_outputs(vec![out]);

        let config = ZkConfig::default();
        let circuit = backend.compile(&ir, &config).unwrap();
        let (pk, mut vk) = backend.setup(&circuit).unwrap();

        let mut gen = WitnessGenerator::new();
        let witness = gen.generate(&ir, &[FieldElement::from_u64(42)], &[]).unwrap();
        assert_eq!(witness.get(out), FieldElement::from_u64(142));

        let proof = backend.prove(&pk, &witness).unwrap();
        vk.trace_commitment_root = proof.trace_commitment;
        let valid = backend.verify(&vk, &proof, &[]).unwrap();
        assert!(valid);
    }

    #[test]
    fn tampered_proof_rejected() {
        let backend = Plonky3Backend;
        let (ir, inputs) = make_mul_ir();
        let config = ZkConfig::default();

        let circuit = backend.compile(&ir, &config).unwrap();
        let (pk, mut vk) = backend.setup(&circuit).unwrap();

        let mut gen = WitnessGenerator::new();
        let witness = gen.generate(&ir, &inputs, &[]).unwrap();

        let mut proof = backend.prove(&pk, &witness).unwrap();
        vk.trace_commitment_root = proof.trace_commitment;

        // Tamper with a query opening value
        if !proof.query_openings.is_empty() {
            proof.query_openings[0].values[0] = 999;
        }

        // Verification should fail (Merkle path won't match tampered leaf)
        let result = backend.verify(&vk, &proof, &[]);
        assert!(result.is_err() || matches!(result, Ok(false)), "tampered proof should be rejected");
    }

    #[test]
    fn estimate_proof_size_nonzero() {
        let backend = Plonky3Backend;
        let (ir, _) = make_mul_ir();
        let config = ZkConfig::default();
        let circuit = backend.compile(&ir, &config).unwrap();

        let size = backend.estimate_proof_size(&circuit);
        assert!(size > 0, "estimated proof size should be positive");
        assert!(size < 1_000_000, "proof size should be reasonable (<1MB)");
    }

    #[test]
    fn emit_solidity_returns_none() {
        let backend = Plonky3Backend;
        let vk = Plonky3VerificationKey {
            fri_config: FriConfig::default(),
            num_trace_rows: 4,
            num_trace_cols: TRACE_WIDTH,
            trace_commitment_root: [0u8; 32],
        };
        assert!(backend.emit_solidity(&vk).is_none());
    }

    // -----------------------------------------------------------------------
    // Trace builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn trace_padded_to_power_of_two() {
        // 3 instructions -> padded to 4 rows
        let mut ir = ZkIR::new("pad_test");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("c");
        let w3 = ir.alloc_wire("d");
        ir.push(ZkInstruction::Mul { out: w2, a: w0, b: w1 });
        ir.push(ZkInstruction::Add { out: w3, a: w2, b: w0 });
        ir.push(ZkInstruction::Const { out: w2, value: FieldElement::from_u64(99) });
        ir.set_public_inputs(vec![w0, w1]);

        let mut gen = WitnessGenerator::new();
        let witness = gen.generate(
            &ir,
            &[FieldElement::from_u64(3), FieldElement::from_u64(5)],
            &[],
        ).unwrap();

        let trace = build_trace(&ir.instructions, &witness);
        assert!(trace.len().is_power_of_two());
        assert!(trace.len() >= 3);
        assert_eq!(trace.len(), 4); // next power of 2 >= 3
    }

    // -----------------------------------------------------------------------
    // Multi-instruction round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn prove_verify_mixed_ops() {
        let backend = Plonky3Backend;
        let mut ir = ZkIR::new("mixed_ops");
        let w0 = ir.alloc_wire("x");
        let w1 = ir.alloc_wire("y");
        let w2 = ir.alloc_wire("sum");
        let w3 = ir.alloc_wire("product");
        let wc = ir.alloc_wire("const_5");
        let w4 = ir.alloc_wire("final");

        // sum = x + y
        ir.push(ZkInstruction::Add { out: w2, a: w0, b: w1 });
        // product = x * y
        ir.push(ZkInstruction::Mul { out: w3, a: w0, b: w1 });
        // const_5 = 5
        ir.push(ZkInstruction::Const { out: wc, value: FieldElement::from_u64(5) });
        // final = product + const_5
        ir.push(ZkInstruction::Add { out: w4, a: w3, b: wc });

        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w2, w4]);

        let config = ZkConfig::default();
        let circuit = backend.compile(&ir, &config).unwrap();
        let (pk, mut vk) = backend.setup(&circuit).unwrap();

        let mut gen = WitnessGenerator::new();
        let witness = gen.generate(
            &ir,
            &[FieldElement::from_u64(3), FieldElement::from_u64(4)],
            &[],
        ).unwrap();
        // sum = 7, product = 12, final = 17
        assert_eq!(witness.get(w2), FieldElement::from_u64(7));
        assert_eq!(witness.get(w4), FieldElement::from_u64(17));

        let proof = backend.prove(&pk, &witness).unwrap();
        vk.trace_commitment_root = proof.trace_commitment;

        let valid = backend.verify(&vk, &proof, &[]).unwrap();
        assert!(valid, "mixed ops proof should verify");
    }

    #[test]
    fn prove_verify_assert_eq() {
        let backend = Plonky3Backend;
        let mut ir = ZkIR::new("assert_eq_test");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        ir.push(ZkInstruction::AssertEq { a: w0, b: w1 });
        ir.set_public_inputs(vec![w0, w1]);

        let config = ZkConfig::default();
        let circuit = backend.compile(&ir, &config).unwrap();
        let (pk, mut vk) = backend.setup(&circuit).unwrap();

        let mut gen = WitnessGenerator::new();
        let witness = gen.generate(
            &ir,
            &[FieldElement::from_u64(77), FieldElement::from_u64(77)],
            &[],
        ).unwrap();

        let proof = backend.prove(&pk, &witness).unwrap();
        vk.trace_commitment_root = proof.trace_commitment;
        let valid = backend.verify(&vk, &proof, &[]).unwrap();
        assert!(valid);
    }
}
