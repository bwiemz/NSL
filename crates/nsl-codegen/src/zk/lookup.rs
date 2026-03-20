//! M55: Lookup table pre-computation for ZK inference circuits.
//!
//! Non-linear activations (ReLU, GELU, sigmoid, tanh, exp, log, inv, rsqrt)
//! cannot be expressed directly as arithmetic constraints. Instead, each
//! function is pre-computed over the full input domain at circuit-build time
//! and encoded as a table of (input, output) field-element pairs.
//!
//! ## Quantization semantics
//!
//! For an N-bit table:
//! - The input index `i` ranges over `[0, 2^N)`.
//! - The index is interpreted as a signed two's-complement integer (int8 for
//!   8-bit tables): `i_signed = if i < 128 { i as i32 } else { i as i32 - 256 }`.
//! - The signed value is treated as a fixed-point number: `x_f64 = i_signed as f64`.
//! - The mathematical function is evaluated in f64.
//! - The result is quantized back to the output integer range using
//!   round-half-to-even (banker's rounding), then clamped to the output range.
//! - The clamped integer is stored as a [`FieldElement`] via [`FieldElement::from_fixed_point`].

use super::field::FieldElement;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A pre-computed lookup table for a single non-linear function.
///
/// The table maps every possible `input_bits`-wide input to the corresponding
/// quantised `output_bits`-wide output, both encoded as field elements.
#[derive(Debug, Clone)]
pub struct LookupTable {
    /// Human-readable name of the function (e.g. `"relu"`, `"gelu"`).
    pub name: String,
    /// Bit-width of the input domain. The table has `2^input_bits` entries.
    pub input_bits: u32,
    /// Bit-width of the output range used for quantisation.
    pub output_bits: u32,
    /// `(input_field_element, output_field_element)` pairs, in order of
    /// ascending raw index (i.e. index 0 first, 2^input_bits - 1 last).
    pub entries: Vec<(FieldElement, FieldElement)>,
}

// ---------------------------------------------------------------------------
// Quantisation helpers
// ---------------------------------------------------------------------------

/// Convert an unsigned table index to its signed interpretation.
///
/// For an N-bit table the top half of indices (`>= 2^(N-1)`) represent
/// negative integers: `index - 2^N`.
#[inline]
fn index_to_signed(index: u64, bits: u32) -> i64 {
    let half = 1u64 << (bits - 1); // 2^(N-1)
    if index < half {
        index as i64
    } else {
        index as i64 - (1i64 << bits)
    }
}

/// Quantise an f64 value to an integer in the range `[-(2^(bits-1)), 2^(bits-1) - 1]`
/// using round-half-to-even (banker's rounding), then clamp.
fn quantise(val: f64, output_bits: u32) -> i64 {
    let max_val = ((1i64 << (output_bits - 1)) - 1) as f64; //  127 for 8-bit
    let min_val = -(1i64 << (output_bits - 1)) as f64; // -128 for 8-bit

    if val.is_nan() {
        return 0;
    }

    let clamped = val.clamp(min_val, max_val);
    // Round half to even (banker's rounding).
    let rounded = round_half_to_even(clamped);
    rounded.clamp(min_val as i64, max_val as i64)
}

/// Round an f64 to the nearest integer using half-to-even (banker's rounding).
fn round_half_to_even(x: f64) -> i64 {
    let floor = x.floor();
    let frac = x - floor;
    let floor_i = floor as i64;
    if frac < 0.5 {
        floor_i
    } else if frac > 0.5 {
        floor_i + 1
    } else {
        // Exactly 0.5 — round to even.
        if floor_i % 2 == 0 {
            floor_i
        } else {
            floor_i + 1
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in mathematical functions
// ---------------------------------------------------------------------------

/// Evaluate the named built-in function for a single f64 input.
///
/// Returns `None` for unrecognised names (the caller falls through to
/// `precompute_table_from_fn`).
fn eval_builtin(name: &str, x: f64) -> Option<f64> {
    let result = match name {
        "relu" => x.max(0.0),
        "gelu" => {
            // Standard GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
            let sqrt2 = std::f64::consts::SQRT_2;
            x * 0.5 * (1.0 + erf(x / sqrt2))
        }
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        "tanh" => x.tanh(),
        "exp" => x.exp(),
        "log" => {
            if x <= 0.0 {
                0.0
            } else {
                x.ln()
            }
        }
        "inv" => {
            if x == 0.0 {
                0.0
            } else {
                1.0 / x
            }
        }
        "rsqrt" => {
            if x <= 0.0 {
                0.0
            } else {
                1.0 / x.sqrt()
            }
        }
        _ => return None,
    };
    Some(result)
}

/// Abramowitz & Stegun approximation of the error function (erf).
///
/// Maximum error ≈ 1.5 × 10⁻⁷, sufficient for 8-bit quantisation.
fn erf(x: f64) -> f64 {
    // Handle sign symmetry.
    if x < 0.0 {
        return -erf(-x);
    }
    // Constants from A&S 7.1.26.
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    1.0 - poly * (-x * x).exp()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Pre-compute a lookup table for a named built-in activation function.
///
/// # Arguments
/// - `name`: one of `"relu"`, `"gelu"`, `"sigmoid"`, `"tanh"`, `"exp"`,
///   `"log"`, `"inv"`, `"rsqrt"`.
/// - `input_bits`: bit-width of the input domain (e.g. `8` → 256 entries).
/// - `output_bits`: bit-width of the output quantisation range.
///
/// # Panics
/// Panics if `name` is not a recognised built-in.
pub fn precompute_table(name: &str, input_bits: u32, output_bits: u32) -> LookupTable {
    let f = move |x: f64| {
        eval_builtin(name, x).unwrap_or_else(|| panic!("unknown built-in function: '{name}'"))
    };
    let mut table = precompute_table_from_fn(f, input_bits, output_bits);
    table.name = name.to_string();
    table
}

/// Pre-compute a lookup table from an arbitrary closure.
///
/// The closure receives the dequantised `f64` input (signed interpretation of
/// the raw index) and must return the corresponding `f64` output.  The output
/// is quantised to `output_bits` bits before being stored.
///
/// The generated [`LookupTable`] will have `name` set to `"<custom>"`.
pub fn precompute_table_from_fn(
    f: impl Fn(f64) -> f64,
    input_bits: u32,
    output_bits: u32,
) -> LookupTable {
    let n = 1u64 << input_bits; // 2^input_bits entries
    let mut entries = Vec::with_capacity(n as usize);

    for i in 0..n {
        // Dequantise: interpret index as signed integer.
        let x: f64 = index_to_signed(i, input_bits) as f64;
        // Evaluate the function.
        let y: f64 = f(x);
        // Quantise result.
        let q: i64 = quantise(y, output_bits);
        // Encode as field elements.
        let input_fe = FieldElement::from_u64(i);
        let output_fe = FieldElement::from_fixed_point(q, 0);
        entries.push((input_fe, output_fe));
    }

    LookupTable {
        name: "<custom>".to_string(),
        input_bits,
        output_bits,
        entries,
    }
}

// ---------------------------------------------------------------------------
// Lookup registry
// ---------------------------------------------------------------------------

/// A registry that deduplicates lookup tables by `(name, input_bits)`.
///
/// Tables are expensive to build for large bit-widths; the registry ensures
/// each `(function, input_bits)` combination is only computed once per
/// circuit-build session.
pub struct LookupRegistry {
    pub tables: HashMap<(String, u32), LookupTable>,
}

impl LookupRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
        }
    }

    /// Return a reference to the table for `(name, input_bits)`, creating it
    /// on first access.
    ///
    /// If the table already exists, `output_bits` is ignored and the cached
    /// table is returned unchanged.
    pub fn get_or_create(
        &mut self,
        name: &str,
        input_bits: u32,
        output_bits: u32,
    ) -> &LookupTable {
        self.tables
            .entry((name.to_string(), input_bits))
            .or_insert_with(|| precompute_table(name, input_bits, output_bits))
    }
}

impl Default for LookupRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Table size and basic structure
    // ------------------------------------------------------------------

    #[test]
    fn relu_table_8bit() {
        let table = precompute_table("relu", 8, 8);
        assert_eq!(table.entries.len(), 256, "8-bit table must have 256 entries");

        // Index 128 corresponds to signed integer 128 - 256 = -128, but we
        // actually want to test a few specific signed values.
        // Signed int8 encoding: index = (value + 256) % 256
        //   value  0  -> index 0
        //   value  1  -> index 1
        //   value -1  -> index 255 (i.e. 255 - 256 = -1)

        let zero_index = 0u64; // signed 0
        let one_index = 1u64; // signed 1
        let neg_one_index = 255u64; // signed -1

        let (_, out_zero) = table.entries[zero_index as usize];
        let (_, out_one) = table.entries[one_index as usize];
        let (_, out_neg_one) = table.entries[neg_one_index as usize];

        assert_eq!(out_zero, FieldElement::from_u64(0), "relu(0) == 0");
        assert_eq!(out_one, FieldElement::from_u64(1), "relu(1) == 1");
        assert_eq!(out_neg_one, FieldElement::from_u64(0), "relu(-1) == 0");
    }

    #[test]
    fn gelu_table_8bit_zero_is_zero() {
        let table = precompute_table("gelu", 8, 8);
        assert_eq!(table.entries.len(), 256, "8-bit table must have 256 entries");

        // GELU(0) = 0 * 0.5 * (1 + erf(0)) = 0
        let (_, out_zero) = table.entries[0]; // index 0 = signed 0
        assert_eq!(out_zero, FieldElement::from_u64(0), "gelu(0) == 0");
    }

    #[test]
    fn custom_table_from_closure() {
        let table = precompute_table_from_fn(|x: f64| x * x, 8, 8);
        assert_eq!(table.entries.len(), 256, "8-bit table must have 256 entries");
        assert_eq!(table.name, "<custom>");

        // x=0 -> x^2 = 0; quantised -> 0
        let (_, out_zero) = table.entries[0];
        assert_eq!(out_zero, FieldElement::from_u64(0), "x^2 at x=0 is 0");
    }

    #[test]
    fn registry_deduplicates_same_table() {
        let mut reg = LookupRegistry::new();
        reg.get_or_create("gelu", 8, 8);
        reg.get_or_create("gelu", 8, 8);
        assert_eq!(reg.tables.len(), 1, "same (name, bits) must not create a duplicate");
    }

    #[test]
    fn registry_stores_different_bitwidths_separately() {
        let mut reg = LookupRegistry::new();
        reg.get_or_create("gelu", 8, 8);
        reg.get_or_create("gelu", 16, 16);
        assert_eq!(reg.tables.len(), 2, "different input_bits must be stored separately");
    }

    // ------------------------------------------------------------------
    // Activation correctness spot-checks
    // ------------------------------------------------------------------

    #[test]
    fn sigmoid_at_zero_is_half() {
        let table = precompute_table("sigmoid", 8, 8);
        // sigmoid(0) = 0.5, quantised to 8-bit -> 0 (rounds to 0 not 1 with half-to-even)
        // max = 127, so 0.5 rounds to 0 (banker's rounding: floor=0, frac=0.5, 0 is even)
        let (_, out) = table.entries[0];
        // 0.5 rounds to 0 under banker's rounding (0 is even).
        assert_eq!(out, FieldElement::from_u64(0), "sigmoid(0) quantised to 0");
    }

    #[test]
    fn tanh_at_zero_is_zero() {
        let table = precompute_table("tanh", 8, 8);
        let (_, out) = table.entries[0];
        assert_eq!(out, FieldElement::from_u64(0), "tanh(0) == 0");
    }

    #[test]
    fn log_of_nonpositive_is_zero() {
        let table = precompute_table("log", 8, 8);
        // Signed index 0 -> x=0, log(0) should return 0.
        let (_, out) = table.entries[0];
        assert_eq!(out, FieldElement::from_u64(0), "log(0) returns 0 sentinel");
        // Signed index 255 -> x=-1, log(-1) should also return 0.
        let (_, out_neg) = table.entries[255];
        assert_eq!(out_neg, FieldElement::from_u64(0), "log(-1) returns 0 sentinel");
    }

    #[test]
    fn inv_of_zero_is_zero() {
        let table = precompute_table("inv", 8, 8);
        let (_, out) = table.entries[0];
        assert_eq!(out, FieldElement::from_u64(0), "inv(0) returns 0 sentinel");
    }

    #[test]
    fn rsqrt_of_nonpositive_is_zero() {
        let table = precompute_table("rsqrt", 8, 8);
        // x=0 -> rsqrt(0) = 0
        let (_, out) = table.entries[0];
        assert_eq!(out, FieldElement::from_u64(0), "rsqrt(0) returns 0 sentinel");
        // x=-1 (index 255) -> rsqrt(-1) = 0
        let (_, out_neg) = table.entries[255];
        assert_eq!(out_neg, FieldElement::from_u64(0), "rsqrt(-1) returns 0 sentinel");
    }

    #[test]
    fn exp_at_zero_is_one() {
        let table = precompute_table("exp", 8, 8);
        // exp(0) = 1.0, quantised to 1
        let (_, out) = table.entries[0];
        assert_eq!(out, FieldElement::from_u64(1), "exp(0) == 1");
    }

    // ------------------------------------------------------------------
    // Input field element encoding
    // ------------------------------------------------------------------

    #[test]
    fn input_field_elements_are_sequential_indices() {
        let table = precompute_table("relu", 8, 8);
        for (i, (input_fe, _)) in table.entries.iter().enumerate() {
            assert_eq!(
                *input_fe,
                FieldElement::from_u64(i as u64),
                "input field element at position {i} must equal from_u64({i})"
            );
        }
    }

    // ------------------------------------------------------------------
    // Banker's rounding edge case
    // ------------------------------------------------------------------

    #[test]
    fn round_half_to_even_rounds_down_for_even_floor() {
        // 0.5 -> floor=0 (even), round to 0
        assert_eq!(round_half_to_even(0.5), 0);
    }

    #[test]
    fn round_half_to_even_rounds_up_for_odd_floor() {
        // 1.5 -> floor=1 (odd), round to 2
        assert_eq!(round_half_to_even(1.5), 2);
    }

    #[test]
    fn round_half_to_even_normal_round_down() {
        assert_eq!(round_half_to_even(2.3), 2);
    }

    #[test]
    fn round_half_to_even_normal_round_up() {
        assert_eq!(round_half_to_even(2.7), 3);
    }

    // ------------------------------------------------------------------
    // User-defined custom lookup tables (Task 4b: @zk_lookup)
    // ------------------------------------------------------------------

    #[test]
    fn custom_table_from_closure_verifies_entries() {
        // Create a custom lookup table for abs(x) over 8-bit signed inputs.
        let table = precompute_table_from_fn(|x: f64| x.abs(), 8, 8);
        assert_eq!(table.entries.len(), 256);
        assert_eq!(table.name, "<custom>");

        // Verify specific entries:
        // x = 0 (index 0) -> abs(0) = 0
        let (_, out_zero) = table.entries[0];
        assert_eq!(out_zero, FieldElement::from_u64(0), "abs(0) == 0");

        // x = 5 (index 5) -> abs(5) = 5
        let (_, out_5) = table.entries[5];
        assert_eq!(out_5, FieldElement::from_u64(5), "abs(5) == 5");

        // x = -5 (index 251 in 8-bit two's complement) -> abs(-5) = 5
        let (_, out_neg5) = table.entries[251];
        assert_eq!(out_neg5, FieldElement::from_u64(5), "abs(-5) == 5");

        // x = -128 (index 128) -> abs(-128) = 128, clamped to 127 for 8-bit signed output
        let (_, out_neg128) = table.entries[128];
        assert_eq!(out_neg128, FieldElement::from_u64(127), "abs(-128) clamped to 127");

        // x = 127 (index 127) -> abs(127) = 127
        let (_, out_127) = table.entries[127];
        assert_eq!(out_127, FieldElement::from_u64(127), "abs(127) == 127");
    }

    #[test]
    fn custom_table_clamp_fn_verifies_entries() {
        // Create a lookup table for clamp(x, -3, 3) via a closure.
        let table = precompute_table_from_fn(|x: f64| x.clamp(-3.0, 3.0), 8, 8);
        assert_eq!(table.entries.len(), 256);

        // x = 0 -> clamp(0) = 0
        let (_, out_0) = table.entries[0];
        assert_eq!(out_0, FieldElement::from_u64(0));

        // x = 10 (index 10) -> clamp(10, -3, 3) = 3
        let (_, out_10) = table.entries[10];
        assert_eq!(out_10, FieldElement::from_u64(3));

        // x = -10 (index 246) -> clamp(-10, -3, 3) = -3
        // -3 in the field is p - 3, represented as from_fixed_point(-3, 0)
        let (_, out_neg10) = table.entries[246];
        assert_eq!(out_neg10, FieldElement::from_fixed_point(-3, 0));
    }
}
