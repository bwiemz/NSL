use std::fmt;

#[derive(Debug)]
pub struct CodegenError {
    pub message: String,
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "codegen error: {}", self.message)
    }
}

impl std::error::Error for CodegenError {}

impl CodegenError {
    pub fn new(msg: impl Into<String>) -> Self {
        CodegenError {
            message: msg.into(),
        }
    }
}
