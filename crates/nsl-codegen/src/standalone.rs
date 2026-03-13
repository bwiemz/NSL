/// Standalone utilities for M24 standalone export.
///
/// This module provides helpers used by the build pipeline (nsl-cli) to create
/// separate object files that carry weight data independently of the compiled
/// NSL program object.

use cranelift_module::Module;

/// Create a Cranelift object file containing .nslweights data in .rodata.
/// Exports two symbols: __nsl_weights_data (the bytes) and __nsl_weights_size (u64 length).
pub fn create_weight_object(nslweights_data: &[u8]) -> Result<Vec<u8>, String> {
    // 1. Create ISA from native target
    let flag_builder = cranelift_codegen::settings::builder();
    let isa = cranelift_native::builder()
        .map_err(|e| format!("cranelift native builder: {}", e))?
        .finish(cranelift_codegen::settings::Flags::new(flag_builder))
        .map_err(|e| format!("cranelift ISA: {}", e))?;

    // 2. Create ObjectModule
    let builder = cranelift_object::ObjectBuilder::new(
        isa,
        "nsl_weights",
        cranelift_module::default_libcall_names(),
    )
    .map_err(|e| format!("object builder: {}", e))?;
    let mut module = cranelift_object::ObjectModule::new(builder);

    // 3. Embed weight data
    let data_id = module
        .declare_data(
            "__nsl_weights_data",
            cranelift_module::Linkage::Export,
            false,
            false,
        )
        .map_err(|e| format!("declare weights data: {}", e))?;
    let mut data_desc = cranelift_module::DataDescription::new();
    data_desc.define(nslweights_data.to_vec().into_boxed_slice());
    module
        .define_data(data_id, &data_desc)
        .map_err(|e| format!("define weights data: {}", e))?;

    // 4. Embed size as u64
    let size_id = module
        .declare_data(
            "__nsl_weights_size",
            cranelift_module::Linkage::Export,
            false,
            false,
        )
        .map_err(|e| format!("declare weights size: {}", e))?;
    let mut size_desc = cranelift_module::DataDescription::new();
    size_desc.define(Box::new((nslweights_data.len() as u64).to_le_bytes()));
    module
        .define_data(size_id, &size_desc)
        .map_err(|e| format!("define weights size: {}", e))?;

    // 5. Emit object
    let product = module.finish();
    product
        .emit()
        .map_err(|e| format!("emit weight object: {}", e))
}
