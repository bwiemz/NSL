//! KIR → HIR pass per M57 v1 spec §2.3 (load-bearing pass).
//! Three concerns: op recognition + dispatch; HIR construction; dtype propagation.

use crate::fpga_error::{FpgaLoweringError, V1_SUPPORTED_OPS};
use crate::hir::module::HirModule;
use crate::kernel_ir::{KirOp, KernelIR};

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
        _op: &KirOp,
        _module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        // Implemented in Task 2.5.
        Ok(())
    }

    fn lower_elementwise_add(
        &self,
        _op: &KirOp,
        _module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        // Implemented in Task 2.6.
        Ok(())
    }

    fn lower_relu(
        &self,
        _op: &KirOp,
        _module: &mut HirModule,
    ) -> Result<(), FpgaLoweringError> {
        // Implemented in Task 2.7.
        Ok(())
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
