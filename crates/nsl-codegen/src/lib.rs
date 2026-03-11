pub mod builtins;
pub mod compiler;
pub mod context;
pub mod error;
pub mod expr;
pub mod func;
pub mod kernel;
pub mod linker;
pub mod stmt;
pub mod types;

pub use compiler::{compile, compile_entry, compile_module, compile_test};
pub use error::CodegenError;
