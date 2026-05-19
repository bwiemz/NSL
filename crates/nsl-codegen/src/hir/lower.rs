//! KIR → HIR pass per M57 v1 spec §2.3 (load-bearing pass).
//! Three concerns: op recognition + dispatch; HIR construction; dtype propagation.

use crate::fpga_error::{FpgaLoweringError, V1_SUPPORTED_OPS};
use crate::hir::module::HirModule;
use crate::kernel_ir::{KirOp, KirType, KernelIR};

pub struct KirToHirPass {
    pub test_taps: bool,
}

impl KirToHirPass {
    pub fn new(test_taps: bool) -> Self {
        Self { test_taps }
    }

    /// Lower a KernelIR (representing one NSL `model` block) to a single HirModule.
    pub fn lower(
        &self,
        kir: &KernelIR,
        module_name: &str,
    ) -> Result<HirModule, FpgaLoweringError> {
        let mut module = HirModule::new(module_name);
        module.test_taps = self.test_taps;

        // Iterate KIR ops in order (single-block KIR at v1 — multi-block deferred).
        for op in kir.ops() {
            match op {
                KirOp::Matmul { .. } => self.lower_matmul(op, &mut module)?,
                KirOp::ElementwiseAdd { .. } => self.lower_elementwise_add(op, &mut module)?,
                KirOp::Relu { .. } => self.lower_relu(op, &mut module)?,

                // Unsupported in v1: error at first unrecognized op with localized
                // op-kind name + the supported list (spec §2.5).
                other => {
                    return Err(FpgaLoweringError::UnsupportedKirOp {
                        op_kind: kir_op_kind_name(other),
                        supported: V1_SUPPORTED_OPS,
                    })
                }
            }
        }

        Ok(module)
    }

    fn lower_matmul(
        &self,
        op: &KirOp,
        module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        use crate::hir::control::GenerateFor;
        use crate::hir::module::HirNode;
        use crate::hir::nodes::{Add, Mul, SignExtend};
        use crate::hir::signals::SignalRef;

        let (a_dtype, b_dtype, out_dtype, a_shape, b_shape) = match op {
            KirOp::Matmul {
                a_dtype,
                b_dtype,
                out_dtype,
                a_shape,
                b_shape,
                ..
            } => (a_dtype.clone(), b_dtype.clone(), out_dtype.clone(), *a_shape, *b_shape),
            _ => unreachable!("lower_matmul called with non-Matmul op"),
        };

        let n_inputs = a_shape[1];
        let n_outputs = b_shape[1];
        assert_eq!(a_shape[1], b_shape[0], "Matmul inner dimensions must match");

        let a_width = kir_dtype_width(a_dtype);
        let b_width = kir_dtype_width(b_dtype);
        let acc_width = kir_dtype_width(out_dtype);
        let prod_width = a_width + b_width;

        // Outer GenerateFor: iterate output dimension
        let mut outer_body: Vec<HirNode> = Vec::new();

        // Inner GenerateFor: MAC chain over input dimension
        let mut inner_body: Vec<HirNode> = Vec::new();

        // Mul: x[i] * W[i,o] → prod_i  (width: a_width × b_width → prod_width)
        let mul = Mul::new(
            SignalRef::port("__matmul_a"),
            SignalRef::port("__matmul_b"),
            a_width,
            b_width,
            prod_width,
        );
        let prod_id = mul.out;
        inner_body.push(HirNode::Mul(mul));

        // SignExtend prod_i → acc_width if needed (3-node MAC chain when prod < acc)
        let add_input = if prod_width < acc_width {
            let se = SignExtend::new(SignalRef::wire(prod_id), prod_width, acc_width);
            let se_id = se.dst;
            inner_body.push(HirNode::SignExtend(se));
            SignalRef::wire(se_id)
        } else {
            // When prod_width == acc_width, MAC chain is 2-node: Mul → Add directly.
            SignalRef::wire(prod_id)
        };

        // Add: acc_prev + extended_product → acc_next  (width: acc_width)
        let add = Add::new(
            SignalRef::port("__matmul_acc_prev"),
            add_input,
            acc_width,
        );
        inner_body.push(HirNode::Add(add));

        outer_body.push(HirNode::GenerateFor(GenerateFor::new(
            0,
            n_inputs as i64,
            inner_body,
        )));

        module.add_node(HirNode::GenerateFor(GenerateFor::new(
            0,
            n_outputs as i64,
            outer_body,
        )))?;

        Ok(())
    }

    fn lower_elementwise_add(
        &self,
        op: &KirOp,
        module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        use crate::hir::control::GenerateFor;
        use crate::hir::module::HirNode;
        use crate::hir::nodes::Add;
        use crate::hir::signals::SignalRef;

        let (dtype, shape) = match op {
            KirOp::ElementwiseAdd { dtype, shape, .. } => (dtype.clone(), *shape),
            _ => unreachable!(),
        };

        let width = kir_dtype_width(dtype);
        let n = shape[0];

        // GenerateFor over output dimension; body is one Add node.
        let body = vec![HirNode::Add(Add::new(
            SignalRef::port("__bias_pre"),
            SignalRef::port("__bias_b"),
            width,
        ))];

        module.add_node(HirNode::GenerateFor(GenerateFor::new(0, n as i64, body)))?;

        Ok(())
    }

    fn lower_relu(
        &self,
        op: &KirOp,
        module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        use crate::hir::control::GenerateFor;
        use crate::hir::module::HirNode;
        use crate::hir::nodes::Max0;
        use crate::hir::signals::SignalRef;

        let (dtype, shape) = match op {
            KirOp::Relu { dtype, shape, .. } => (dtype.clone(), *shape),
            _ => unreachable!(),
        };

        let width = kir_dtype_width(dtype);
        let n = shape[0];

        let body = vec![HirNode::Max0(Max0::new(SignalRef::port("__relu_in"), width))];

        module.add_node(HirNode::GenerateFor(GenerateFor::new(0, n as i64, body)))?;

        Ok(())
    }
}

/// Map KirType to bit-width for FPGA HIR construction.
/// Panics on types not supported in v1 FPGA target.
pub(crate) fn kir_dtype_width(dtype: KirType) -> usize {
    match dtype {
        KirType::I8 => 8,
        KirType::I16 => 16,
        KirType::I32 => 32,
        KirType::I64 => 64,
        other => panic!("KirType {:?} unsupported in v1 FPGA target", other),
    }
}

/// Maps every KirOp variant to a stable string for the error message.
/// Tedious but necessary; KirOp doesn't derive a stable name function.
pub(crate) fn kir_op_kind_name(op: &KirOp) -> &'static str {
    match op {
        KirOp::Add(..) => "Add",
        KirOp::Sub(..) => "Sub",
        KirOp::Mul(..) => "Mul",
        KirOp::Div(..) => "Div",
        KirOp::Fma(..) => "Fma",
        KirOp::Neg(..) => "Neg",
        KirOp::Abs(..) => "Abs",
        KirOp::Sqrt(..) => "Sqrt",
        KirOp::Exp(..) => "Exp",
        KirOp::Log(..) => "Log",
        KirOp::Sin(..) => "Sin",
        KirOp::Cos(..) => "Cos",
        KirOp::Tanh(..) => "Tanh",
        KirOp::Pow(..) => "Pow",
        KirOp::Cast(..) => "Cast",
        KirOp::Load(..) => "Load",
        KirOp::Store(..) => "Store",
        KirOp::AtomicAdd(..) => "AtomicAdd",
        KirOp::ThreadId(..) => "ThreadId",
        KirOp::BlockIdx(..) => "BlockIdx",
        KirOp::BlockDim(..) => "BlockDim",
        KirOp::GridDim(..) => "GridDim",
        KirOp::GlobalId(..) => "GlobalId",
        KirOp::Barrier => "Barrier",
        KirOp::WarpShuffle(..) => "WarpShuffle",
        KirOp::Cmp(..) => "Cmp",
        KirOp::Select(..) => "Select",
        KirOp::Const(..) => "Const",
        KirOp::PtrOffset(..) => "PtrOffset",
        KirOp::SharedMemFence => "SharedMemFence",
        KirOp::Matmul { .. } => "Matmul",
        KirOp::ElementwiseAdd { .. } => "ElementwiseAdd",
        KirOp::Relu { .. } => "Relu",
    }
}
