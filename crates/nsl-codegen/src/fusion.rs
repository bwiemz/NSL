//! Elementwise operator fusion: explicit @fuse and lexical auto-fusion.
//! Synthesizes single PTX kernels from chains of elementwise ops.
//! let-binding = hard fusion barrier (no DAG infrastructure).

/// A chain of elementwise operations that can be fused into a single PTX kernel.
pub struct FusedKernel {
    /// Names of ops in the chain (e.g., ["add", "relu"])
    pub op_chain: Vec<String>,
    /// Input tensor count
    pub num_inputs: usize,
    /// Generated PTX bytes (null-terminated)
    pub ptx: Vec<u8>,
    /// Human-readable name for profiler traces
    pub name: String,
}

/// Classification of operations for fusion eligibility.
pub fn is_fusible_op(name: &str) -> bool {
    matches!(
        name,
        "add" | "sub"
            | "mul"
            | "div"
            | "pow"
            | "neg"
            | "abs"
            | "relu"
            | "sigmoid"
            | "tanh"
            | "exp"
            | "log"
            | "sqrt"
            | "sign"
            | "clamp"
    )
}

/// Binary ops that are fusible (from BinOp AST nodes).
pub fn is_fusible_binop(op: &str) -> bool {
    matches!(op, "Add" | "Sub" | "Mul" | "Div" | "Pow")
}

/// Unary ops that are fusible.
pub fn is_fusible_unaryop(op: &str) -> bool {
    matches!(op, "Neg")
}

/// Check if an op is NOT fusible (matmul, reductions, etc.).
pub fn is_fusion_barrier(name: &str) -> bool {
    matches!(
        name,
        "matmul"
            | "sum"
            | "mean"
            | "reduce_max"
            | "reduce_min"
            | "reshape"
            | "transpose"
            | "conv"
            | "softmax"
            | "layernorm"
            | "gather"
            | "scatter"
    )
}

/// Attempt to synthesize a fused PTX kernel for a chain of elementwise ops.
/// Returns None if chain has fewer than 2 ops (no fusion benefit).
pub fn try_synthesize_fused(op_chain: &[&str], num_inputs: usize) -> Option<FusedKernel> {
    if op_chain.len() < 2 {
        return None;
    }
    let name = format!("fused_{}", op_chain.join("_"));
    let ptx = synthesize_fused_ptx(&name, op_chain, num_inputs);
    Some(FusedKernel {
        op_chain: op_chain.iter().map(|s| s.to_string()).collect(),
        num_inputs,
        ptx,
        name,
    })
}

/// Generate PTX for a fused elementwise kernel.
/// Each input is loaded once from global memory, all ops are register-to-register,
/// output is stored once to global memory.
///
/// **ISA correctness:** `ex2.approx.f32` computes base-2 exp, so natural exp requires
/// multiplying by log2(e) ~ 1.4427 first. Similarly, `lg2.approx.f32` computes base-2
/// log, so natural log requires multiplying by ln(2) ~ 0.6931 after.
pub fn synthesize_fused_ptx(name: &str, ops: &[&str], num_inputs: usize) -> Vec<u8> {
    // log2(e) = 1.4426950408889634 -> IEEE 754 f32: 0x3FB8AA3B
    const LOG2_E_HEX: &str = "0f3FB8AA3B";
    // ln(2) = 0.6931471805599453 -> IEEE 754 f32: 0x3F317218
    const LN_2_HEX: &str = "0f3F317218";

    let mut ptx = String::new();

    // Header
    ptx.push_str(".version 7.0\n");
    ptx.push_str(".target sm_52\n");
    ptx.push_str(".address_size 64\n\n");

    // Function signature: (output_ptr, input0_ptr, ..., inputN_ptr, num_elements)
    ptx.push_str(&format!(".visible .entry {}(\n", name));
    ptx.push_str("    .param .u64 param_out,\n");
    for i in 0..num_inputs {
        ptx.push_str(&format!("    .param .u64 param_in{},\n", i));
    }
    ptx.push_str("    .param .u64 param_n\n");
    ptx.push_str(") {\n");

    // Register declarations
    let max_regs = num_inputs + ops.len() * 2 + 8;
    ptx.push_str(&format!("    .reg .u32 %r<{}>;\n", 8_usize.max(max_regs)));
    ptx.push_str(&format!(
        "    .reg .u64 %rd<{}>;\n",
        (4 + num_inputs + 2).max(16)
    ));
    ptx.push_str(&format!("    .reg .f32 %f<{}>;\n", max_regs.max(16)));
    ptx.push_str("    .reg .pred %p<4>;\n\n");

    // Thread ID = blockIdx.x * blockDim.x + threadIdx.x
    ptx.push_str("    mov.u32 %r0, %tid.x;\n");
    ptx.push_str("    mov.u32 %r1, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %r2, %ntid.x;\n");
    ptx.push_str("    mul.lo.u32 %r3, %r1, %r2;\n");
    ptx.push_str("    add.u32 %r0, %r0, %r3;\n");
    ptx.push_str("    cvt.u64.u32 %rd0, %r0;\n\n");

    // Bounds check
    ptx.push_str("    ld.param.u64 %rd1, [param_n];\n");
    ptx.push_str("    setp.ge.u64 %p0, %rd0, %rd1;\n");
    ptx.push_str("    @%p0 bra $L_end;\n\n");

    // Byte offset: tid * 4 (f32)
    ptx.push_str("    shl.b64 %rd2, %rd0, 2;\n\n");

    // Load inputs from global memory
    for i in 0..num_inputs {
        ptx.push_str(&format!(
            "    ld.param.u64 %rd{}, [param_in{}];\n",
            3 + i,
            i
        ));
        ptx.push_str(&format!("    add.u64 %rd{}, %rd{}, %rd2;\n", 3 + i, 3 + i));
        ptx.push_str(&format!("    ld.global.f32 %f{}, [%rd{}];\n\n", i, 3 + i));
    }

    // Apply fused ops -- register-to-register
    let mut result_reg = 0;
    for (op_idx, op) in ops.iter().enumerate() {
        let out_reg = num_inputs + op_idx;
        match *op {
            "add" => {
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                let rhs = if op_idx == 0 { 1 } else { 1 };
                ptx.push_str(&format!(
                    "    add.f32 %f{}, %f{}, %f{};\n",
                    out_reg, lhs, rhs
                ));
            }
            "sub" => {
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!(
                    "    sub.f32 %f{}, %f{}, %f{};\n",
                    out_reg, lhs, 1
                ));
            }
            "mul" => {
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!(
                    "    mul.f32 %f{}, %f{}, %f{};\n",
                    out_reg, lhs, 1
                ));
            }
            "div" => {
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!(
                    "    div.rn.f32 %f{}, %f{}, %f{};\n",
                    out_reg, lhs, 1
                ));
            }
            "pow" => {
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!(
                    "    lg2.approx.f32 %f{}, %f{};\n",
                    out_reg, lhs
                ));
                ptx.push_str(&format!(
                    "    mul.f32 %f{}, %f{}, %f{};\n",
                    out_reg, out_reg, 1
                ));
                ptx.push_str(&format!(
                    "    ex2.approx.f32 %f{}, %f{};\n",
                    out_reg, out_reg
                ));
            }
            "relu" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", out_reg));
                ptx.push_str(&format!(
                    "    max.f32 %f{}, %f{}, %f{};\n",
                    out_reg, src, out_reg
                ));
            }
            "neg" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, src));
            }
            "abs" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    abs.f32 %f{}, %f{};\n", out_reg, src));
            }
            "exp" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    mov.f32 %f{}, {};\n", out_reg, LOG2_E_HEX));
                ptx.push_str(&format!(
                    "    mul.f32 %f{}, %f{}, %f{};\n",
                    out_reg, src, out_reg
                ));
                ptx.push_str(&format!(
                    "    ex2.approx.f32 %f{}, %f{};\n",
                    out_reg, out_reg
                ));
            }
            "log" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!(
                    "    lg2.approx.f32 %f{}, %f{};\n",
                    out_reg, src
                ));
                ptx.push_str(&format!(
                    "    mov.f32 %f{}, {};\n",
                    out_reg + 1,
                    LN_2_HEX
                ));
                ptx.push_str(&format!(
                    "    mul.f32 %f{}, %f{}, %f{};\n",
                    out_reg,
                    out_reg,
                    out_reg + 1
                ));
            }
            "sqrt" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!(
                    "    sqrt.approx.f32 %f{}, %f{};\n",
                    out_reg, src
                ));
            }
            "clamp" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", out_reg));
                ptx.push_str(&format!(
                    "    max.f32 %f{}, %f{}, %f{};\n",
                    out_reg, src, out_reg
                ));
                ptx.push_str(&format!("    mov.f32 %f{}, 0f3F800000;\n", out_reg + 1));
                ptx.push_str(&format!(
                    "    min.f32 %f{}, %f{}, %f{};\n",
                    out_reg,
                    out_reg,
                    out_reg + 1
                ));
            }
            "sigmoid" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, src));
                ptx.push_str(&format!(
                    "    mov.f32 %f{}, {};\n",
                    out_reg + 1,
                    LOG2_E_HEX
                ));
                ptx.push_str(&format!(
                    "    mul.f32 %f{}, %f{}, %f{};\n",
                    out_reg,
                    out_reg,
                    out_reg + 1
                ));
                ptx.push_str(&format!(
                    "    ex2.approx.f32 %f{}, %f{};\n",
                    out_reg, out_reg
                ));
                ptx.push_str(&format!(
                    "    add.f32 %f{}, %f{}, 0f3F800000;\n",
                    out_reg, out_reg
                ));
                ptx.push_str(&format!(
                    "    rcp.approx.f32 %f{}, %f{};\n",
                    out_reg, out_reg
                ));
            }
            "tanh" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!(
                    "    add.f32 %f{}, %f{}, %f{};\n",
                    out_reg, src, src
                ));
                ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, out_reg));
                ptx.push_str(&format!(
                    "    mov.f32 %f{}, {};\n",
                    out_reg + 1,
                    LOG2_E_HEX
                ));
                ptx.push_str(&format!(
                    "    mul.f32 %f{}, %f{}, %f{};\n",
                    out_reg,
                    out_reg,
                    out_reg + 1
                ));
                ptx.push_str(&format!(
                    "    ex2.approx.f32 %f{}, %f{};\n",
                    out_reg, out_reg
                ));
                ptx.push_str(&format!(
                    "    add.f32 %f{}, %f{}, 0f3F800000;\n",
                    out_reg, out_reg
                ));
                ptx.push_str(&format!(
                    "    rcp.approx.f32 %f{}, %f{};\n",
                    out_reg, out_reg
                ));
                ptx.push_str(&format!(
                    "    add.f32 %f{}, %f{}, %f{};\n",
                    out_reg, out_reg, out_reg
                ));
                ptx.push_str(&format!(
                    "    sub.f32 %f{}, %f{}, 0f3F800000;\n",
                    out_reg, out_reg
                ));
            }
            "sign" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!(
                    "    setp.gt.f32 %p1, %f{}, 0f00000000;\n",
                    src
                ));
                ptx.push_str(&format!(
                    "    setp.lt.f32 %p2, %f{}, 0f00000000;\n",
                    src
                ));
                ptx.push_str(&format!(
                    "    selp.f32 %f{}, 0f3F800000, 0f00000000, %p1;\n",
                    out_reg
                ));
                ptx.push_str(&format!(
                    "    selp.f32 %f{}, 0fBF800000, %f{}, %p2;\n",
                    out_reg, out_reg
                ));
            }
            _ => {
                // Unknown op: pass through
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    mov.f32 %f{}, %f{};\n", out_reg, src));
            }
        }
        result_reg = out_reg;
    }

    // Store result to output
    ptx.push_str(&format!(
        "\n    ld.param.u64 %rd{}, [param_out];\n",
        3 + num_inputs
    ));
    ptx.push_str(&format!(
        "    add.u64 %rd{}, %rd{}, %rd2;\n",
        3 + num_inputs,
        3 + num_inputs
    ));
    ptx.push_str(&format!(
        "    st.global.f32 [%rd{}], %f{};\n",
        3 + num_inputs,
        result_reg
    ));

    // End
    ptx.push_str("\n$L_end:\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    let mut bytes = ptx.into_bytes();
    bytes.push(0); // null terminator
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesize_fused_add_relu() {
        let ptx = synthesize_fused_ptx("fused_add_relu", &["add", "relu"], 2);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
        assert!(ptx_str.contains(".version 7.0"));
        assert!(ptx_str.contains(".entry fused_add_relu"));
        assert!(ptx_str.contains("ld.global.f32"));
        assert!(ptx_str.contains("add.f32"));
        assert!(ptx_str.contains("max.f32")); // relu = max(x, 0)
        assert!(ptx_str.contains("st.global.f32"));
    }

    #[test]
    fn test_synthesize_single_op_returns_none() {
        let result = try_synthesize_fused(&["relu"], 1);
        assert!(result.is_none());
    }

    #[test]
    fn test_synthesize_two_ops_returns_some() {
        let result = try_synthesize_fused(&["add", "relu"], 2);
        assert!(result.is_some());
        let kernel = result.unwrap();
        assert_eq!(kernel.name, "fused_add_relu");
        assert_eq!(kernel.op_chain, vec!["add", "relu"]);
        assert_eq!(kernel.num_inputs, 2);
    }

    #[test]
    fn test_synthesize_exp_uses_log2e_conversion() {
        let ptx = synthesize_fused_ptx("fused_add_exp", &["add", "exp"], 2);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
        // exp uses log2(e) multiply before ex2.approx
        assert!(ptx_str.contains("0f3FB8AA3B")); // log2(e) hex
        assert!(ptx_str.contains("ex2.approx.f32"));
    }

    #[test]
    fn test_synthesize_log_uses_ln2_conversion() {
        let ptx = synthesize_fused_ptx("fused_add_log", &["add", "log"], 2);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
        // log uses lg2.approx then multiplies by ln(2)
        assert!(ptx_str.contains("lg2.approx.f32"));
        assert!(ptx_str.contains("0f3F317218")); // ln(2) hex
    }

    #[test]
    fn test_synthesize_sigmoid() {
        let ptx = synthesize_fused_ptx("fused_add_sigmoid", &["add", "sigmoid"], 2);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
        assert!(ptx_str.contains("neg.f32"));
        assert!(ptx_str.contains("ex2.approx.f32"));
        assert!(ptx_str.contains("rcp.approx.f32"));
        assert!(ptx_str.contains("0f3F800000")); // 1.0
    }

    #[test]
    fn test_ptx_null_terminated() {
        let ptx = synthesize_fused_ptx("test_kernel", &["add", "relu"], 2);
        assert_eq!(*ptx.last().unwrap(), 0u8);
    }

    #[test]
    fn test_ptx_thread_id_uses_mul_add_not_mad() {
        let ptx = synthesize_fused_ptx("test_kernel", &["add", "relu"], 2);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
        // Must use mul.lo.u32 + add.u32, NOT mad.lo.u32 (INVALID_PTX on some ISA)
        assert!(ptx_str.contains("mul.lo.u32"));
        assert!(ptx_str.contains("add.u32"));
        assert!(!ptx_str.contains("mad.lo.u32"));
    }

    #[test]
    fn test_op_classification() {
        assert!(is_fusible_op("add"));
        assert!(is_fusible_op("relu"));
        assert!(is_fusible_op("sigmoid"));
        assert!(!is_fusible_op("matmul"));
        assert!(!is_fusible_op("softmax"));

        assert!(is_fusible_binop("Add"));
        assert!(is_fusible_binop("Mul"));
        assert!(!is_fusible_binop("add")); // lowercase not matched

        assert!(is_fusible_unaryop("Neg"));
        assert!(!is_fusible_unaryop("neg")); // lowercase not matched

        assert!(is_fusion_barrier("matmul"));
        assert!(is_fusion_barrier("softmax"));
        assert!(!is_fusion_barrier("add"));
    }

    #[test]
    fn test_three_op_chain() {
        let result = try_synthesize_fused(&["mul", "add", "relu"], 2);
        assert!(result.is_some());
        let kernel = result.unwrap();
        assert_eq!(kernel.name, "fused_mul_add_relu");
        assert_eq!(kernel.op_chain.len(), 3);
    }
}
