pub mod autotune;
pub mod builtins;
pub mod compiler;
pub mod context;
pub mod error;
pub mod expr;
pub mod func;
pub mod dynamic_shapes;
pub mod flash_attention;
pub mod fusion;
pub mod kernel;
pub mod linker;
pub mod standalone;
pub mod stmt;
pub mod types;

pub use compiler::{compile, compile_entry, compile_module, compile_module_with_imports, compile_test, compile_standalone, StandaloneConfig};
pub use error::CodegenError;
pub use standalone::create_weight_object;

/// Compiler configuration flags passed from CLI.
#[derive(Default, Clone)]
pub struct CompileOptions {
    pub no_autotune: bool,
    pub autotune_fresh: bool,
}
