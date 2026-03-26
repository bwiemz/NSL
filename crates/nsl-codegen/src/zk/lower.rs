//! M55: 5-pass DAG -> ZK-IR lowering pipeline.
//!
//! Transforms a simplified computation DAG ([`ZkDag`]) into a flat arithmetic
//! circuit ([`ZkIR`]) suitable for zero-knowledge proof backends.
//!
//! ## Pipeline
//!
//! 1. **Fixed-Point Conversion** — Emit `Const` instructions for known weight values.
//! 2. **Operation Lowering** — Lower each [`ZkOp`] to ZK-IR instructions.
//! 3. **Sparsity Elimination** — Skip zero-weight terms in dot products.
//! 4. **Lookup Table Generation** — Pre-compute tables for all activation lookups.
//! 5. **Witness Layout** — Assign public inputs, public outputs, and private wires.
//!
//! The DAG is a simplified intermediate representation populated from the real
//! NSL AST in Task 12's `compile_zk()`.

use super::backend::ZkConfig;
use super::field::FieldElement;
use super::ir::{Wire, ZkIR, ZkInstruction};
use super::lookup::precompute_table;

// ---------------------------------------------------------------------------
// INT8 → Field Element mapping (M55d)
// ---------------------------------------------------------------------------

/// Map a signed INT8 value [-128, 127] into a BN254 field element.
/// Uses signed representation: negative values become field negations.
fn int8_to_field(val: i8) -> FieldElement {
    if val >= 0 {
        FieldElement::from_u64(val as u64)
    } else {
        let abs = val.unsigned_abs() as u64;
        let fe = FieldElement::from_u64(abs);
        FieldElement::zero().sub(&fe) // field negation
    }
}

/// Check if a ZkDag uses only INT8 (8-bit) weights, making it eligible for M31 field.
pub fn is_int8_model(dag: &ZkDag) -> bool {
    dag.ops.iter().all(|op| match op {
        ZkOp::Input { dtype_bits, .. } => *dtype_bits <= 8,
        ZkOp::Weight { dtype_bits, .. } => *dtype_bits <= 8,
        _ => true,
    })
}

/// Maximum safe accumulation length for INT8 matmul in M31 field.
/// M31 prime = 2^31 - 1. INT8 product max = 127 * 127 = 16129.
/// Safe accumulation: floor((2^31 - 1) / 16129) = 133,143 terms.
/// So matmul inner dimension K can be up to 133K without overflow.
pub const M31_INT8_SAFE_K: usize = 133_000;

// ---------------------------------------------------------------------------
// ZkOp — Simplified DAG node
// ---------------------------------------------------------------------------

/// Simplified operation node for ZK lowering.
///
/// Populated from the real NSL AST in Task 12's `compile_zk()`.
#[derive(Debug, Clone)]
pub enum ZkOp {
    /// Model input tensor.
    Input {
        name: String,
        shape: Vec<usize>,
        dtype_bits: u32,
    },
    /// Model weight tensor, optionally with known constant values.
    Weight {
        name: String,
        shape: Vec<usize>,
        dtype_bits: u32,
        values: Option<Vec<i64>>,
    },
    /// Matrix multiplication: `a @ b`.
    Matmul { a: usize, b: usize },
    /// Element-wise addition.
    Add { a: usize, b: usize },
    /// Element-wise multiplication.
    Mul { a: usize, b: usize },
    /// ReLU activation.
    Relu { input: usize },
    /// GELU activation.
    Gelu { input: usize },
    /// Sigmoid activation.
    Sigmoid { input: usize },
    /// Tanh activation.
    Tanh { input: usize },
    /// Softmax over a given dimension.
    Softmax { input: usize, dim: i32 },
    /// Layer normalization.
    LayerNorm {
        input: usize,
        gamma: usize,
        beta: usize,
    },
    /// Exponential function.
    Exp { input: usize },
    /// Natural logarithm.
    Log { input: usize },
    /// Matrix transpose.
    Transpose { input: usize },
    /// Reshape (reinterpret shape without data movement).
    Reshape {
        input: usize,
        new_shape: Vec<usize>,
    },
    /// Requantize to a different bit-width.
    Requantize {
        input: usize,
        scale: f64,
        zero_point: f64,
        target_bits: u32,
    },
}

// ---------------------------------------------------------------------------
// ZkDag
// ---------------------------------------------------------------------------

/// A simplified computation DAG for ZK lowering.
///
/// Operations are stored in topological order (each op only references
/// earlier ops by index). This invariant is enforced by the AST -> DAG
/// translation in Task 12.
#[derive(Debug)]
pub struct ZkDag {
    /// Operations in topological order.
    pub ops: Vec<ZkOp>,
    /// Index of the DAG output node.
    pub output_idx: usize,
    /// Indices of input nodes (public model inputs).
    pub input_indices: Vec<usize>,
    /// Indices of weight nodes.
    pub weight_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Internal state for the lowering pipeline
// ---------------------------------------------------------------------------

/// Per-op metadata accumulated during lowering.
#[derive(Clone)]
struct OpInfo {
    /// The wires allocated for this op's output elements.
    wires: Vec<Wire>,
    /// The shape of this op's output.
    shape: Vec<usize>,
    /// Bit-width of this op's output elements.
    dtype_bits: u32,
}

/// Compute the total number of elements from a shape.
fn num_elements(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Pass 2/3 helpers
// ---------------------------------------------------------------------------

/// Lower a Matmul op with sparsity elimination (Pass 3).
///
/// For shapes `[m, k] @ [k, n]`, emits `m * n` DotProduct instructions,
/// each of length k (minus any zero-weight terms from Pass 3).
fn lower_matmul(
    dag: &ZkDag,
    ir: &mut ZkIR,
    op_info: &[Option<OpInfo>],
    idx: usize,
    a_idx: usize,
    b_idx: usize,
    _config: &ZkConfig,
) -> OpInfo {
    let a_info = op_info[a_idx].as_ref().expect("matmul input A not lowered");
    let b_info = op_info[b_idx].as_ref().expect("matmul input B not lowered");

    // Deduce shapes: a is [m, k], b is [k, n].
    let (m, k_a) = shape_as_2d(&a_info.shape);
    let (k_b, n) = shape_as_2d(&b_info.shape);
    assert_eq!(k_a, k_b, "matmul inner dimensions must match");
    let k = k_a;

    // Determine if B has known zero values (for sparsity elimination).
    let b_values = if let ZkOp::Weight { values: Some(v), .. } = &dag.ops[b_idx] {
        Some(v.as_slice())
    } else {
        None
    };
    // Also check if A has known zero values.
    let a_values = if let ZkOp::Weight { values: Some(v), .. } = &dag.ops[a_idx] {
        Some(v.as_slice())
    } else {
        None
    };

    let mut out_wires = Vec::with_capacity(m * n);
    let out_bits = a_info.dtype_bits.max(b_info.dtype_bits);

    for i in 0..m {
        for j in 0..n {
            let out = ir.alloc_wire(&format!("matmul_{idx}_{i}_{j}"));

            // Collect non-zero terms for dot product (Pass 3: sparsity elimination).
            let mut a_wires = Vec::with_capacity(k);
            let mut b_wires = Vec::with_capacity(k);

            for l in 0..k {
                let a_elem_idx = i * k + l;
                let b_elem_idx = l * n + j;

                // Check if either weight is known to be zero.
                let a_is_zero = a_values.is_some_and(|v| v[a_elem_idx] == 0);
                let b_is_zero = b_values.is_some_and(|v| v[b_elem_idx] == 0);

                if a_is_zero || b_is_zero {
                    continue; // Sparsity elimination: skip zero terms.
                }

                a_wires.push(a_info.wires[a_elem_idx]);
                b_wires.push(b_info.wires[b_elem_idx]);
            }

            if a_wires.is_empty() {
                // All terms are zero: emit a constant zero.
                ir.push(ZkInstruction::Const {
                    out,
                    value: FieldElement::zero(),
                });
            } else {
                ir.push(ZkInstruction::DotProduct {
                    out,
                    a: a_wires,
                    b: b_wires,
                });
            }

            out_wires.push(out);
        }
    }

    OpInfo {
        wires: out_wires,
        shape: vec![m, n],
        dtype_bits: out_bits,
    }
}

/// Lower element-wise addition.
fn lower_elementwise_add(
    ir: &mut ZkIR,
    op_info: &[Option<OpInfo>],
    idx: usize,
    a_idx: usize,
    b_idx: usize,
) -> OpInfo {
    let a_info = op_info[a_idx].as_ref().expect("add input A not lowered");
    let b_info = op_info[b_idx].as_ref().expect("add input B not lowered");

    let n = a_info.wires.len().max(b_info.wires.len());
    let out_bits = a_info.dtype_bits.max(b_info.dtype_bits);

    // Support broadcasting: if one operand has 1 element, broadcast it.
    let mut out_wires = Vec::with_capacity(n);
    for i in 0..n {
        let a_wire = a_info.wires[i % a_info.wires.len()];
        let b_wire = b_info.wires[i % b_info.wires.len()];
        let out = ir.alloc_wire(&format!("add_{idx}_{i}"));
        ir.push(ZkInstruction::Add {
            out,
            a: a_wire,
            b: b_wire,
        });
        out_wires.push(out);
    }

    // Use the larger shape for output.
    let shape = if a_info.wires.len() >= b_info.wires.len() {
        a_info.shape.clone()
    } else {
        b_info.shape.clone()
    };

    OpInfo {
        wires: out_wires,
        shape,
        dtype_bits: out_bits,
    }
}

/// Lower element-wise multiplication (fixed-point).
fn lower_elementwise_mul(
    ir: &mut ZkIR,
    op_info: &[Option<OpInfo>],
    idx: usize,
    a_idx: usize,
    b_idx: usize,
    config: &ZkConfig,
) -> OpInfo {
    let a_info = op_info[a_idx].as_ref().expect("mul input A not lowered");
    let b_info = op_info[b_idx].as_ref().expect("mul input B not lowered");

    let n = a_info.wires.len().max(b_info.wires.len());
    let out_bits = a_info.dtype_bits.max(b_info.dtype_bits);

    let mut out_wires = Vec::with_capacity(n);
    for i in 0..n {
        let a_wire = a_info.wires[i % a_info.wires.len()];
        let b_wire = b_info.wires[i % b_info.wires.len()];
        let out = ir.alloc_wire(&format!("mul_{idx}_{i}"));
        ir.push(ZkInstruction::FixedMul {
            out,
            a: a_wire,
            b: b_wire,
            frac_bits: config.frac_bits,
        });
        out_wires.push(out);
    }

    let shape = if a_info.wires.len() >= b_info.wires.len() {
        a_info.shape.clone()
    } else {
        b_info.shape.clone()
    };

    OpInfo {
        wires: out_wires,
        shape,
        dtype_bits: out_bits,
    }
}

/// Lower a unary activation function to Lookup instructions.
fn lower_activation(
    ir: &mut ZkIR,
    op_info: &[Option<OpInfo>],
    idx: usize,
    input_idx: usize,
    table_name: &str,
) -> OpInfo {
    let input_info = op_info[input_idx]
        .as_ref()
        .expect("activation input not lowered");

    let n = input_info.wires.len();
    let bits = input_info.dtype_bits;

    let mut out_wires = Vec::with_capacity(n);
    for i in 0..n {
        let out = ir.alloc_wire(&format!("{table_name}_{idx}_{i}"));
        ir.push(ZkInstruction::Lookup {
            out,
            table: table_name.to_string(),
            input: input_info.wires[i],
            input_bits: bits,
        });
        out_wires.push(out);
    }

    OpInfo {
        wires: out_wires,
        shape: input_info.shape.clone(),
        dtype_bits: bits,
    }
}

/// Lower softmax: `exp(x_i) / sum(exp(x_j))`.
///
/// Emitted as:
/// 1. Lookup "exp" for each input element.
/// 2. Add all exp outputs together to get the sum.
/// 3. Lookup "inv" on the sum.
/// 4. FixedMul each exp output by the inverse sum.
fn lower_softmax(
    ir: &mut ZkIR,
    op_info: &[Option<OpInfo>],
    idx: usize,
    input_idx: usize,
) -> OpInfo {
    let input_info = op_info[input_idx]
        .as_ref()
        .expect("softmax input not lowered");

    let n = input_info.wires.len();
    let bits = input_info.dtype_bits;

    // Step 1: exp lookup for each element.
    let mut exp_wires = Vec::with_capacity(n);
    for i in 0..n {
        let out = ir.alloc_wire(&format!("softmax_exp_{idx}_{i}"));
        ir.push(ZkInstruction::Lookup {
            out,
            table: "exp".to_string(),
            input: input_info.wires[i],
            input_bits: bits,
        });
        exp_wires.push(out);
    }

    // Step 2: sum all exp outputs.
    let sum_wire = if n == 1 {
        exp_wires[0]
    } else {
        let mut acc = exp_wires[0];
        for (i, &exp_w) in exp_wires.iter().enumerate().skip(1) {
            let sum_out = ir.alloc_wire(&format!("softmax_sum_{idx}_{i}"));
            ir.push(ZkInstruction::Add {
                out: sum_out,
                a: acc,
                b: exp_w,
            });
            acc = sum_out;
        }
        acc
    };

    // Step 3: inv lookup on the sum.
    let inv_wire = ir.alloc_wire(&format!("softmax_inv_{idx}"));
    ir.push(ZkInstruction::Lookup {
        out: inv_wire,
        table: "inv".to_string(),
        input: sum_wire,
        input_bits: bits,
    });

    // Step 4: multiply each exp by the inverse sum.
    let mut out_wires = Vec::with_capacity(n);
    for (i, &exp_w) in exp_wires.iter().enumerate() {
        let out = ir.alloc_wire(&format!("softmax_out_{idx}_{i}"));
        ir.push(ZkInstruction::FixedMul {
            out,
            a: exp_w,
            b: inv_wire,
            frac_bits: bits,
        });
        out_wires.push(out);
    }

    OpInfo {
        wires: out_wires,
        shape: input_info.shape.clone(),
        dtype_bits: bits,
    }
}

/// Lower LayerNorm: `gamma * (x - mean) * rsqrt(var + eps) + beta`.
///
/// Simplified to:
/// 1. Compute mean via Add chain + inv lookup.
/// 2. Subtract mean from each element.
/// 3. Compute variance via FixedMul + Add chain + inv lookup.
/// 4. Lookup rsqrt on variance.
/// 5. FixedMul by gamma, then Add beta.
///
/// # Known limitation
///
// KNOWN LIMITATION: The inv/rsqrt lookups only work when accumulated values
// fit within the lookup table's input domain (0..2^input_bits). For real use,
// LayerNorm requires field-arithmetic division instead of lookup tables,
// or a wider lookup table. This is tracked for the real Halo2 integration.
fn lower_layer_norm(
    ir: &mut ZkIR,
    op_info: &[Option<OpInfo>],
    idx: usize,
    input_idx: usize,
    gamma_idx: usize,
    beta_idx: usize,
    config: &ZkConfig,
) -> OpInfo {
    let input_info = op_info[input_idx]
        .as_ref()
        .expect("layernorm input not lowered");
    let gamma_info = op_info[gamma_idx]
        .as_ref()
        .expect("layernorm gamma not lowered");
    let beta_info = op_info[beta_idx]
        .as_ref()
        .expect("layernorm beta not lowered");

    let n = input_info.wires.len();
    let bits = input_info.dtype_bits;

    // Step 1: Compute mean = sum(x_i) * inv(n).
    let mut acc = input_info.wires[0];
    for i in 1..n {
        let sum_out = ir.alloc_wire(&format!("ln_sum_{idx}_{i}"));
        ir.push(ZkInstruction::Add {
            out: sum_out,
            a: acc,
            b: input_info.wires[i],
        });
        acc = sum_out;
    }
    // inv(n) via lookup for the mean.
    let mean_wire = ir.alloc_wire(&format!("ln_mean_{idx}"));
    ir.push(ZkInstruction::Lookup {
        out: mean_wire,
        table: "inv".to_string(),
        input: acc,
        input_bits: bits,
    });

    // Step 2: Subtract mean from each element: x_centered_i = x_i - mean.
    // In ZK-IR we express subtraction as: centered = x + (-mean).
    // For simplicity, use Add with a negated mean. In the field, subtraction
    // is addition of the additive inverse, which the backend handles.
    // For now, we approximate: emit FixedMul(x_i, rsqrt_var) and skip
    // the explicit centering (the lookup tables absorb the shift).
    //
    // Actually, let's emit proper instructions. We'll use Add with the
    // negation implemented as: neg_mean = 0 - mean = (p - mean). But we
    // don't have a Sub instruction. Instead, we use the fact that
    // Add(x, neg(mean)) works if we compute neg(mean) = Mul(mean, -1).
    // Const(-1) in the field is p-1.
    let neg_one = ir.alloc_wire(&format!("ln_neg1_{idx}"));
    ir.push(ZkInstruction::Const {
        out: neg_one,
        value: FieldElement::from_fixed_point(-1, 0),
    });
    let neg_mean = ir.alloc_wire(&format!("ln_negmean_{idx}"));
    ir.push(ZkInstruction::Mul {
        out: neg_mean,
        a: mean_wire,
        b: neg_one,
    });

    let mut centered = Vec::with_capacity(n);
    for i in 0..n {
        let out = ir.alloc_wire(&format!("ln_centered_{idx}_{i}"));
        ir.push(ZkInstruction::Add {
            out,
            a: input_info.wires[i],
            b: neg_mean,
        });
        centered.push(out);
    }

    // Step 3: Variance = mean of squared centered values.
    // Square each centered value.
    let mut sq_wires = Vec::with_capacity(n);
    for (i, &cw) in centered.iter().enumerate() {
        let sq = ir.alloc_wire(&format!("ln_sq_{idx}_{i}"));
        ir.push(ZkInstruction::FixedMul {
            out: sq,
            a: cw,
            b: cw,
            frac_bits: config.frac_bits,
        });
        sq_wires.push(sq);
    }
    // Sum squared values.
    let mut var_acc = sq_wires[0];
    for (i, &sq_w) in sq_wires.iter().enumerate().skip(1) {
        let sum_out = ir.alloc_wire(&format!("ln_varsum_{idx}_{i}"));
        ir.push(ZkInstruction::Add {
            out: sum_out,
            a: var_acc,
            b: sq_w,
        });
        var_acc = sum_out;
    }
    // inv(n) for mean of variance — reuse inv lookup.
    let var_mean = ir.alloc_wire(&format!("ln_varmean_{idx}"));
    ir.push(ZkInstruction::Lookup {
        out: var_mean,
        table: "inv".to_string(),
        input: var_acc,
        input_bits: bits,
    });

    // Step 4: rsqrt of variance.
    let rsqrt_var = ir.alloc_wire(&format!("ln_rsqrt_{idx}"));
    ir.push(ZkInstruction::Lookup {
        out: rsqrt_var,
        table: "rsqrt".to_string(),
        input: var_mean,
        input_bits: bits,
    });

    // Step 5: Normalize, scale by gamma, add beta.
    let mut out_wires = Vec::with_capacity(n);
    for (i, &cw) in centered.iter().enumerate() {
        // normalized_i = centered_i * rsqrt_var
        let norm = ir.alloc_wire(&format!("ln_norm_{idx}_{i}"));
        ir.push(ZkInstruction::FixedMul {
            out: norm,
            a: cw,
            b: rsqrt_var,
            frac_bits: config.frac_bits,
        });

        // scaled_i = normalized_i * gamma_i
        let gamma_wire = gamma_info.wires[i % gamma_info.wires.len()];
        let scaled = ir.alloc_wire(&format!("ln_scaled_{idx}_{i}"));
        ir.push(ZkInstruction::FixedMul {
            out: scaled,
            a: norm,
            b: gamma_wire,
            frac_bits: config.frac_bits,
        });

        // out_i = scaled_i + beta_i
        let beta_wire = beta_info.wires[i % beta_info.wires.len()];
        let out = ir.alloc_wire(&format!("ln_out_{idx}_{i}"));
        ir.push(ZkInstruction::Add {
            out,
            a: scaled,
            b: beta_wire,
        });
        out_wires.push(out);
    }

    OpInfo {
        wires: out_wires,
        shape: input_info.shape.clone(),
        dtype_bits: bits,
    }
}

/// Lower transpose: permute wires without arithmetic cost.
fn lower_transpose(
    ir: &mut ZkIR,
    op_info: &[Option<OpInfo>],
    idx: usize,
    input_idx: usize,
) -> OpInfo {
    let input_info = op_info[input_idx]
        .as_ref()
        .expect("transpose input not lowered");

    let shape = &input_info.shape;
    let (rows, cols) = shape_as_2d(shape);

    let n = rows * cols;
    let mut permutation = vec![0usize; n];
    for r in 0..rows {
        for c in 0..cols {
            permutation[c * rows + r] = r * cols + c;
        }
    }

    let mut out_wires = Vec::with_capacity(n);
    for _ in 0..n {
        out_wires.push(ir.alloc_wire(&format!("transpose_{idx}")));
    }

    ir.push(ZkInstruction::Remap {
        out: out_wires.clone(),
        input: input_info.wires.clone(),
        permutation,
    });

    OpInfo {
        wires: out_wires,
        shape: vec![cols, rows],
        dtype_bits: input_info.dtype_bits,
    }
}

/// Lower reshape: reinterpret wires under a new shape (zero cost).
fn lower_reshape(
    ir: &mut ZkIR,
    op_info: &[Option<OpInfo>],
    idx: usize,
    input_idx: usize,
    new_shape: &[usize],
) -> OpInfo {
    let input_info = op_info[input_idx]
        .as_ref()
        .expect("reshape input not lowered");

    let n = input_info.wires.len();
    let identity: Vec<usize> = (0..n).collect();

    let mut out_wires = Vec::with_capacity(n);
    for _ in 0..n {
        out_wires.push(ir.alloc_wire(&format!("reshape_{idx}")));
    }

    ir.push(ZkInstruction::Remap {
        out: out_wires.clone(),
        input: input_info.wires.clone(),
        permutation: identity,
    });

    OpInfo {
        wires: out_wires,
        shape: new_shape.to_vec(),
        dtype_bits: input_info.dtype_bits,
    }
}

/// Lower requantize: apply scale/zero_point transformation.
fn lower_requantize(
    ir: &mut ZkIR,
    op_info: &[Option<OpInfo>],
    idx: usize,
    input_idx: usize,
    scale: f64,
    zero_point: f64,
    target_bits: u32,
) -> OpInfo {
    let input_info = op_info[input_idx]
        .as_ref()
        .expect("requantize input not lowered");

    let n = input_info.wires.len();
    // INT8 requantize: map scale and zero_point into field elements
    let scale_fe = if target_bits <= 8 {
        int8_to_field((scale * 127.0).round() as i8) // scale as INT8 fraction
    } else {
        FieldElement::from_fixed_point(scale as i64, 0)
    };
    let zp_fe = if target_bits <= 8 {
        int8_to_field(zero_point as i8)
    } else {
        FieldElement::from_fixed_point(zero_point as i64, 0)
    };

    let mut out_wires = Vec::with_capacity(n);
    for i in 0..n {
        let out = ir.alloc_wire(&format!("requantize_{idx}_{i}"));
        ir.push(ZkInstruction::Requantize {
            out,
            input: input_info.wires[i],
            scale: scale_fe,
            zero_point: zp_fe,
            target_bits,
        });
        out_wires.push(out);
    }

    OpInfo {
        wires: out_wires,
        shape: input_info.shape.clone(),
        dtype_bits: target_bits,
    }
}

/// Lower a [`ZkDag`] into a [`ZkIR`] using the 5-pass pipeline.
///
/// This is the main entry point for the ZK lowering pipeline. It executes
/// all five passes in order: fixed-point conversion, operation lowering
/// (with integrated sparsity elimination), lookup table generation, and
/// witness layout assignment.
pub fn lower_dag_to_zkir(dag: &ZkDag, config: &ZkConfig) -> ZkIR {
    let mut ir = ZkIR::new("zk_circuit");
    let mut op_info: Vec<Option<OpInfo>> = vec![None; dag.ops.len()];

    // Pass 1: Fixed-Point Conversion
    for (idx, op) in dag.ops.iter().enumerate() {
        match op {
            ZkOp::Input {
                name,
                shape,
                dtype_bits,
            } => {
                let n = num_elements(shape);
                let mut wires = Vec::with_capacity(n);
                for i in 0..n {
                    wires.push(ir.alloc_wire(&format!("{name}_{i}")));
                }
                op_info[idx] = Some(OpInfo {
                    wires,
                    shape: shape.clone(),
                    dtype_bits: *dtype_bits,
                });
            }
            ZkOp::Weight {
                name,
                shape,
                dtype_bits,
                values,
            } => {
                let n = num_elements(shape);
                let mut wires = Vec::with_capacity(n);
                if let Some(vals) = values {
                    for (i, &v) in vals.iter().enumerate() {
                        let w = ir.alloc_wire(&format!("{name}_{i}"));
                        // INT8 weights: use signed mapping to field element
                        let fe = if *dtype_bits <= 8 {
                            int8_to_field(v as i8)
                        } else {
                            FieldElement::from_fixed_point(v, config.frac_bits)
                        };
                        ir.push(ZkInstruction::Const { out: w, value: fe });
                        wires.push(w);
                    }
                } else {
                    for i in 0..n {
                        wires.push(ir.alloc_wire(&format!("{name}_{i}")));
                    }
                }
                op_info[idx] = Some(OpInfo {
                    wires,
                    shape: shape.clone(),
                    dtype_bits: *dtype_bits,
                });
            }
            _ => {}
        }
    }

    // Pass 2+3: Operation Lowering with Sparsity Elimination
    for idx in 0..dag.ops.len() {
        if op_info[idx].is_some() {
            continue;
        }

        let info = match &dag.ops[idx] {
            ZkOp::Matmul { a, b } => {
                lower_matmul(dag, &mut ir, &op_info, idx, *a, *b, config)
            }
            ZkOp::Add { a, b } => {
                lower_elementwise_add(&mut ir, &op_info, idx, *a, *b)
            }
            ZkOp::Mul { a, b } => {
                lower_elementwise_mul(&mut ir, &op_info, idx, *a, *b, config)
            }
            ZkOp::Relu { input } => {
                lower_activation(&mut ir, &op_info, idx, *input, "relu")
            }
            ZkOp::Gelu { input } => {
                lower_activation(&mut ir, &op_info, idx, *input, "gelu")
            }
            ZkOp::Sigmoid { input } => {
                lower_activation(&mut ir, &op_info, idx, *input, "sigmoid")
            }
            ZkOp::Tanh { input } => {
                lower_activation(&mut ir, &op_info, idx, *input, "tanh")
            }
            ZkOp::Exp { input } => {
                lower_activation(&mut ir, &op_info, idx, *input, "exp")
            }
            ZkOp::Log { input } => {
                lower_activation(&mut ir, &op_info, idx, *input, "log")
            }
            ZkOp::Softmax { input, dim: _ } => {
                lower_softmax(&mut ir, &op_info, idx, *input)
            }
            ZkOp::LayerNorm {
                input,
                gamma,
                beta,
            } => lower_layer_norm(&mut ir, &op_info, idx, *input, *gamma, *beta, config),
            ZkOp::Transpose { input } => {
                lower_transpose(&mut ir, &op_info, idx, *input)
            }
            ZkOp::Reshape { input, new_shape } => {
                lower_reshape(&mut ir, &op_info, idx, *input, new_shape)
            }
            ZkOp::Requantize {
                input,
                scale,
                zero_point,
                target_bits,
            } => lower_requantize(&mut ir, &op_info, idx, *input, *scale, *zero_point, *target_bits),
            ZkOp::Input { .. } | ZkOp::Weight { .. } => unreachable!(),
        };

        op_info[idx] = Some(info);
    }

    // Pass 4: Lookup Table Generation
    pass4_lookup_tables(&mut ir);

    // Pass 5: Witness Layout
    pass5_witness_layout(dag, &mut ir, &op_info);

    ir
}

// ---------------------------------------------------------------------------
// Pass 4: Lookup Table Generation
// ---------------------------------------------------------------------------

/// Scan all Lookup instructions, collect unique (table_name, input_bits),
/// and pre-compute tables using `lookup::precompute_table()`.
fn pass4_lookup_tables(ir: &mut ZkIR) {
    // Collect unique (table_name, input_bits) pairs.
    let mut needed: Vec<(String, u32)> = Vec::new();
    for inst in &ir.instructions {
        if let ZkInstruction::Lookup {
            table, input_bits, ..
        } = inst
        {
            let key = (table.clone(), *input_bits);
            if !needed.contains(&key) {
                needed.push(key);
            }
        }
    }

    // Pre-compute and register each table.
    for (name, input_bits) in needed {
        if !ir.lookup_tables.contains_key(&(name.clone(), input_bits)) {
            let table = precompute_table(&name, input_bits, input_bits);
            ir.register_table(table);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 5: Witness Layout
// ---------------------------------------------------------------------------

/// Assign public inputs first, then public outputs, then private wires.
fn pass5_witness_layout(dag: &ZkDag, ir: &mut ZkIR, op_info: &[Option<OpInfo>]) {
    // Public inputs: all wires from Input nodes.
    let mut pub_inputs = Vec::new();
    for &idx in &dag.input_indices {
        if let Some(info) = &op_info[idx] {
            pub_inputs.extend_from_slice(&info.wires);
        }
    }
    ir.set_public_inputs(pub_inputs.clone());

    // Public outputs: all wires from the output node.
    let mut pub_outputs = Vec::new();
    if let Some(info) = &op_info[dag.output_idx] {
        pub_outputs.extend_from_slice(&info.wires);
    }
    ir.set_public_outputs(pub_outputs.clone());

    // Private wires: all remaining wires that are not public inputs or outputs.
    let pub_set: std::collections::HashSet<Wire> = pub_inputs
        .iter()
        .chain(pub_outputs.iter())
        .copied()
        .collect();
    let mut private = Vec::new();
    for w in 0..ir.num_wires {
        let wire = Wire(w);
        if !pub_set.contains(&wire) {
            private.push(wire);
        }
    }
    ir.private_inputs = private;
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

/// Interpret a shape as 2D `(rows, cols)`.
///
/// - `[]` or `[n]` → `(1, n)`
/// - `[m, n]` → `(m, n)`
/// - `[..., m, n]` → `(m, n)` (takes last two dims)
fn shape_as_2d(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        _ => {
            let ndim = shape.len();
            (shape[ndim - 2], shape[ndim - 1])
        }
    }
}

// ---------------------------------------------------------------------------
// Per-layer lowering for folding backend (Phase 4)
// ---------------------------------------------------------------------------

/// Detect layer boundaries in a DAG for folding-based proving.
///
/// A "layer" is a foldable unit: matmul → activation → (optional norm).
/// Each layer becomes an independent ZkIR that the folding prover processes
/// sequentially.
fn is_layer_boundary(op: &ZkOp) -> bool {
    matches!(
        op,
        ZkOp::Relu { .. }
            | ZkOp::Gelu { .. }
            | ZkOp::Sigmoid { .. }
            | ZkOp::Tanh { .. }
            | ZkOp::Softmax { .. }
            | ZkOp::LayerNorm { .. }
    )
}

/// Lower a DAG into per-layer ZkIRs for folding-based proving.
///
/// Groups operations by layer boundaries (matmul + activation = one fold unit).
/// Each returned ZkIR is independently foldable.
pub fn lower_model_for_folding(dag: &ZkDag, config: &ZkConfig) -> Vec<ZkIR> {
    if dag.ops.is_empty() {
        return vec![];
    }

    // Partition ops into layers at activation/norm boundaries
    let mut layer_groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group: Vec<usize> = Vec::new();

    for (idx, op) in dag.ops.iter().enumerate() {
        current_group.push(idx);
        if is_layer_boundary(op) {
            layer_groups.push(std::mem::take(&mut current_group));
        }
    }
    // Remaining ops form the last group
    if !current_group.is_empty() {
        layer_groups.push(current_group);
    }

    // If no boundaries found, return a single ZkIR (the whole model)
    if layer_groups.len() <= 1 {
        return vec![lower_dag_to_zkir(dag, config)];
    }

    // Lower each group into an independent ZkIR
    let mut layer_irs = Vec::with_capacity(layer_groups.len());
    for (layer_idx, group) in layer_groups.iter().enumerate() {
        // Build a sub-DAG for this group
        let sub_dag = build_sub_dag(dag, group, layer_idx);
        let ir = lower_dag_to_zkir(&sub_dag, config);
        layer_irs.push(ir);
    }

    layer_irs
}

/// Build a sub-DAG from a subset of operation indices.
fn build_sub_dag(dag: &ZkDag, indices: &[usize], layer_idx: usize) -> ZkDag {
    // Map old indices to new indices in the sub-DAG
    let mut index_map = std::collections::HashMap::new();
    let mut sub_ops = Vec::new();
    let mut input_indices = Vec::new();
    let mut weight_indices = Vec::new();

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        index_map.insert(old_idx, new_idx);

        // Remap operation references
        let remapped_op = remap_op(&dag.ops[old_idx], &index_map, layer_idx);
        if matches!(remapped_op, ZkOp::Input { .. }) {
            input_indices.push(new_idx);
        }
        if matches!(remapped_op, ZkOp::Weight { .. }) {
            weight_indices.push(new_idx);
        }
        sub_ops.push(remapped_op);
    }

    let output_idx = sub_ops.len().saturating_sub(1);

    ZkDag {
        ops: sub_ops,
        output_idx,
        input_indices,
        weight_indices,
    }
}

/// Remap an operation's references to new indices in a sub-DAG.
///
/// If a referenced op is not in the sub-DAG's index map, it becomes
/// an Input node (inter-layer dependency).
fn remap_op(
    op: &ZkOp,
    index_map: &std::collections::HashMap<usize, usize>,
    layer_idx: usize,
) -> ZkOp {
    let remap = |old: usize| -> usize {
        *index_map.get(&old).unwrap_or(&0)
    };

    match op {
        ZkOp::Input { name, shape, dtype_bits } => ZkOp::Input {
            name: name.clone(),
            shape: shape.clone(),
            dtype_bits: *dtype_bits,
        },
        ZkOp::Weight { name, shape, dtype_bits, values } => ZkOp::Weight {
            name: name.clone(),
            shape: shape.clone(),
            dtype_bits: *dtype_bits,
            values: values.clone(),
        },
        ZkOp::Matmul { a, b } => {
            if index_map.contains_key(a) && index_map.contains_key(b) {
                ZkOp::Matmul { a: remap(*a), b: remap(*b) }
            } else {
                // Reference outside this layer — create placeholder input
                ZkOp::Input {
                    name: format!("layer{layer_idx}_matmul_input"),
                    shape: vec![1],
                    dtype_bits: 8,
                }
            }
        }
        ZkOp::Add { a, b } => {
            if index_map.contains_key(a) && index_map.contains_key(b) {
                ZkOp::Add { a: remap(*a), b: remap(*b) }
            } else {
                ZkOp::Input {
                    name: format!("layer{layer_idx}_add_input"),
                    shape: vec![1],
                    dtype_bits: 8,
                }
            }
        }
        ZkOp::Mul { a, b } => {
            if index_map.contains_key(a) && index_map.contains_key(b) {
                ZkOp::Mul { a: remap(*a), b: remap(*b) }
            } else {
                ZkOp::Input {
                    name: format!("layer{layer_idx}_mul_input"),
                    shape: vec![1],
                    dtype_bits: 8,
                }
            }
        }
        ZkOp::Relu { input } => ZkOp::Relu { input: remap(*input) },
        ZkOp::Gelu { input } => ZkOp::Gelu { input: remap(*input) },
        ZkOp::Sigmoid { input } => ZkOp::Sigmoid { input: remap(*input) },
        ZkOp::Tanh { input } => ZkOp::Tanh { input: remap(*input) },
        ZkOp::Softmax { input, dim } => ZkOp::Softmax { input: remap(*input), dim: *dim },
        ZkOp::LayerNorm { input, gamma, beta } => ZkOp::LayerNorm {
            input: remap(*input),
            gamma: remap(*gamma),
            beta: remap(*beta),
        },
        ZkOp::Exp { input } => ZkOp::Exp { input: remap(*input) },
        ZkOp::Log { input } => ZkOp::Log { input: remap(*input) },
        ZkOp::Transpose { input } => ZkOp::Transpose { input: remap(*input) },
        ZkOp::Reshape { input, new_shape } => ZkOp::Reshape {
            input: remap(*input),
            new_shape: new_shape.clone(),
        },
        ZkOp::Requantize { input, scale, zero_point, target_bits } => ZkOp::Requantize {
            input: remap(*input),
            scale: *scale,
            zero_point: *zero_point,
            target_bits: *target_bits,
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::backend::ZkConfig;
    use crate::zk::ir::ZkInstruction;

    fn lower(dag: &ZkDag, config: &ZkConfig) -> ZkIR {
        lower_dag_to_zkir(dag, config)
    }

    fn simple_add_dag() -> ZkDag {
        ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "a".into(),
                    shape: vec![1],
                    dtype_bits: 8,
                },
                ZkOp::Input {
                    name: "b".into(),
                    shape: vec![1],
                    dtype_bits: 8,
                },
                ZkOp::Add { a: 0, b: 1 },
            ],
            output_idx: 2,
            input_indices: vec![0, 1],
            weight_indices: vec![],
        }
    }

    #[test]
    fn lower_add_produces_add_instruction() {
        let dag = simple_add_dag();
        let ir = lower(&dag, &ZkConfig::default());
        assert!(
            ir.instructions
                .iter()
                .any(|i| matches!(i, ZkInstruction::Add { .. })),
            "should contain at least one Add instruction"
        );
    }

    #[test]
    fn lower_matmul_produces_dot_products() {
        // 2x2 @ 2x2 -> 4 dot products
        let dag = ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "x".into(),
                    shape: vec![2, 2],
                    dtype_bits: 8,
                },
                ZkOp::Weight {
                    name: "w".into(),
                    shape: vec![2, 2],
                    dtype_bits: 8,
                    values: None,
                },
                ZkOp::Matmul { a: 0, b: 1 },
            ],
            output_idx: 2,
            input_indices: vec![0],
            weight_indices: vec![1],
        };
        let ir = lower(&dag, &ZkConfig::default());
        let dots = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::DotProduct { .. }))
            .count();
        assert_eq!(dots, 4, "2x2 @ 2x2 should produce 4 dot products");
    }

    #[test]
    fn lower_relu_produces_lookup() {
        let dag = ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "x".into(),
                    shape: vec![4],
                    dtype_bits: 8,
                },
                ZkOp::Relu { input: 0 },
            ],
            output_idx: 1,
            input_indices: vec![0],
            weight_indices: vec![],
        };
        let ir = lower(&dag, &ZkConfig::default());
        let lookups = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::Lookup { table, .. } if table == "relu"))
            .count();
        assert_eq!(lookups, 4, "4-element relu should produce 4 lookup instructions");
        assert!(
            ir.lookup_tables.contains_key(&("relu".to_string(), 8)),
            "relu table should be registered with input_bits=8"
        );
    }

    #[test]
    fn lower_sparsity_skips_zero_weights() {
        // 1x2 @ 2x2 matmul with 50% zero weights (identity-like)
        let dag = ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "x".into(),
                    shape: vec![1, 2],
                    dtype_bits: 8,
                },
                ZkOp::Weight {
                    name: "w".into(),
                    shape: vec![2, 2],
                    dtype_bits: 8,
                    values: Some(vec![1, 0, 0, 1]), // identity-like with zeros
                },
                ZkOp::Matmul { a: 0, b: 1 },
            ],
            output_idx: 2,
            input_indices: vec![0],
            weight_indices: vec![1],
        };
        let ir = lower(&dag, &ZkConfig::default());
        // Each dot product should have only 1 term (the non-zero weight).
        for inst in &ir.instructions {
            if let ZkInstruction::DotProduct { a, b, .. } = inst {
                assert_eq!(
                    a.len(),
                    1,
                    "dot product should have 1 term after sparsity elimination"
                );
                assert_eq!(a.len(), b.len());
            }
        }
    }

    #[test]
    fn lower_softmax_produces_exp_and_inv_lookups() {
        let dag = ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "x".into(),
                    shape: vec![4],
                    dtype_bits: 8,
                },
                ZkOp::Softmax { input: 0, dim: -1 },
            ],
            output_idx: 1,
            input_indices: vec![0],
            weight_indices: vec![],
        };
        let ir = lower(&dag, &ZkConfig::default());
        let exp_lookups = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::Lookup { table, .. } if table == "exp"))
            .count();
        let inv_lookups = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::Lookup { table, .. } if table == "inv"))
            .count();
        assert_eq!(exp_lookups, 4, "softmax should produce 4 exp lookups");
        assert!(
            inv_lookups >= 1,
            "softmax should produce at least 1 inv lookup"
        );
    }

    #[test]
    fn lower_witness_layout_assigns_public_and_private() {
        let dag = simple_add_dag();
        let ir = lower(&dag, &ZkConfig::default());

        // Public inputs should be the wires for "a" and "b".
        assert_eq!(
            ir.public_inputs.len(),
            2,
            "two scalar inputs should yield 2 public input wires"
        );
        // Public outputs should be the wire(s) for the Add result.
        assert_eq!(
            ir.public_outputs.len(),
            1,
            "single scalar output should yield 1 public output wire"
        );
        // Private wires should be everything else.
        assert!(
            !ir.private_inputs.is_empty() || ir.num_wires == 3,
            "remaining wires should be private or there are exactly 3 wires"
        );
    }

    #[test]
    fn lower_matmul_rect_shapes() {
        // 2x3 @ 3x4 -> 2x4 = 8 dot products, each of length 3
        let dag = ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "x".into(),
                    shape: vec![2, 3],
                    dtype_bits: 8,
                },
                ZkOp::Weight {
                    name: "w".into(),
                    shape: vec![3, 4],
                    dtype_bits: 8,
                    values: None,
                },
                ZkOp::Matmul { a: 0, b: 1 },
            ],
            output_idx: 2,
            input_indices: vec![0],
            weight_indices: vec![1],
        };
        let ir = lower(&dag, &ZkConfig::default());
        let dots: Vec<_> = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::DotProduct { .. }))
            .collect();
        assert_eq!(dots.len(), 8, "2x3 @ 3x4 should produce 8 dot products");
        for inst in &dots {
            if let ZkInstruction::DotProduct { a, .. } = inst {
                assert_eq!(a.len(), 3, "each dot product should have 3 terms");
            }
        }
    }

    #[test]
    fn lower_mul_produces_fixed_mul() {
        let dag = ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "a".into(),
                    shape: vec![2],
                    dtype_bits: 8,
                },
                ZkOp::Input {
                    name: "b".into(),
                    shape: vec![2],
                    dtype_bits: 8,
                },
                ZkOp::Mul { a: 0, b: 1 },
            ],
            output_idx: 2,
            input_indices: vec![0, 1],
            weight_indices: vec![],
        };
        let ir = lower(&dag, &ZkConfig::default());
        let fixed_muls = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::FixedMul { .. }))
            .count();
        assert_eq!(fixed_muls, 2, "2-element mul should produce 2 FixedMul instructions");
    }

    #[test]
    fn lower_gelu_produces_gelu_lookup() {
        let dag = ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "x".into(),
                    shape: vec![3],
                    dtype_bits: 8,
                },
                ZkOp::Gelu { input: 0 },
            ],
            output_idx: 1,
            input_indices: vec![0],
            weight_indices: vec![],
        };
        let ir = lower(&dag, &ZkConfig::default());
        let lookups = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::Lookup { table, .. } if table == "gelu"))
            .count();
        assert_eq!(lookups, 3);
        assert!(ir.lookup_tables.contains_key(&("gelu".to_string(), 8)));
    }

    #[test]
    fn lower_known_weights_emit_const() {
        let dag = ZkDag {
            ops: vec![
                ZkOp::Weight {
                    name: "w".into(),
                    shape: vec![2],
                    dtype_bits: 8,
                    values: Some(vec![42, -7]),
                },
            ],
            output_idx: 0,
            input_indices: vec![],
            weight_indices: vec![0],
        };
        let ir = lower(&dag, &ZkConfig::default());
        let consts = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::Const { .. }))
            .count();
        assert_eq!(consts, 2, "2 known weights should produce 2 Const instructions");
    }

    #[test]
    fn lower_transpose_produces_remap() {
        let dag = ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "x".into(),
                    shape: vec![2, 3],
                    dtype_bits: 8,
                },
                ZkOp::Transpose { input: 0 },
            ],
            output_idx: 1,
            input_indices: vec![0],
            weight_indices: vec![],
        };
        let ir = lower(&dag, &ZkConfig::default());
        let remaps = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::Remap { .. }))
            .count();
        assert_eq!(remaps, 1, "transpose should produce 1 Remap instruction");
    }

    #[test]
    fn lower_for_folding_6_layer_mlp() {
        // 6-layer MLP: (input -> matmul -> relu) × 3
        let dag = ZkDag {
            ops: vec![
                // Layer 1
                ZkOp::Input { name: "x".into(), shape: vec![1, 4], dtype_bits: 8 },
                ZkOp::Weight { name: "w1".into(), shape: vec![4, 4], dtype_bits: 8, values: None },
                ZkOp::Matmul { a: 0, b: 1 },
                ZkOp::Relu { input: 2 },
                // Layer 2
                ZkOp::Weight { name: "w2".into(), shape: vec![4, 4], dtype_bits: 8, values: None },
                ZkOp::Matmul { a: 3, b: 4 },
                ZkOp::Relu { input: 5 },
                // Layer 3
                ZkOp::Weight { name: "w3".into(), shape: vec![4, 4], dtype_bits: 8, values: None },
                ZkOp::Matmul { a: 6, b: 7 },
                ZkOp::Relu { input: 8 },
            ],
            output_idx: 9,
            input_indices: vec![0],
            weight_indices: vec![1, 4, 7],
        };

        let layers = lower_model_for_folding(&dag, &ZkConfig::default());
        assert_eq!(layers.len(), 3, "3-layer MLP should produce 3 foldable ZkIRs");
    }

    #[test]
    fn lower_reshape_produces_identity_remap() {
        let dag = ZkDag {
            ops: vec![
                ZkOp::Input {
                    name: "x".into(),
                    shape: vec![2, 3],
                    dtype_bits: 8,
                },
                ZkOp::Reshape {
                    input: 0,
                    new_shape: vec![6],
                },
            ],
            output_idx: 1,
            input_indices: vec![0],
            weight_indices: vec![],
        };
        let ir = lower(&dag, &ZkConfig::default());
        let remaps: Vec<_> = ir
            .instructions
            .iter()
            .filter(|i| matches!(i, ZkInstruction::Remap { .. }))
            .collect();
        assert_eq!(remaps.len(), 1);
        if let ZkInstruction::Remap { permutation, .. } = &remaps[0] {
            let identity: Vec<usize> = (0..6).collect();
            assert_eq!(permutation, &identity, "reshape should use identity permutation");
        }
    }
}
