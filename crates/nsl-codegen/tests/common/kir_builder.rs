/// Build a KernelIR from a list of ops using the real KirBuilder API.
/// Shared by hir_pass_snapshots, verilog_emission_snapshots, and yosys_gate.
pub fn kir_with_ops(name: &str, ops: Vec<nsl_codegen::kernel_ir::KirOp>) -> nsl_codegen::kernel_ir::KernelIR {
    use nsl_codegen::kernel_ir::{KirBuilder, KirTerminator};
    let mut b = KirBuilder::new(name);
    let blk = b.new_block();
    b.set_block(blk);
    for op in ops {
        b.emit(op);
    }
    b.terminate(KirTerminator::Return);
    b.finalize()
}
