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

    /// Construct a `MissingScales` error for the given projection path.
    ///
    /// Emitted during final-compile AWQ lowering when a calibration sidecar
    /// is present but does not contain scales for the named projection.
    /// Silent fallback to uncalibrated is a correctness trap, so this is
    /// always a hard error.
    pub fn missing_scales(projection_path: impl Into<String>) -> Self {
        let path = projection_path.into();
        CodegenError {
            message: format!(
                "MissingScales: AWQ calibration sidecar is present but missing \
                 scales for projection '{path}'"
            ),
        }
    }
}

impl From<crate::calibration::DiscoveryError> for CodegenError {
    fn from(e: crate::calibration::DiscoveryError) -> Self {
        CodegenError::new(format!("calibration discovery: {e}"))
    }
}
