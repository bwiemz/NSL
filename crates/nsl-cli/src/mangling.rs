use std::path::Path;

/// Compute module prefix from a file path relative to a base directory.
/// Strips `.nsl` extension, replaces `/` and `\` with `_`.
/// If the path is not under base_dir (e.g., stdlib), tries each stdlib root.
/// Falls back to the file stem if no prefix can be computed.
pub fn module_prefix(path: &Path, base_dir: &Path) -> String {
    // Try stripping base_dir first
    if let Ok(rel) = path.strip_prefix(base_dir) {
        return rel
            .with_extension("")
            .to_string_lossy()
            .replace(['/', '\\'], "_");
    }

    // Try stdlib roots
    for root in crate::resolver::stdlib_roots() {
        if let Ok(rel) = path.strip_prefix(&root) {
            return rel
                .with_extension("")
                .to_string_lossy()
                .replace(['/', '\\'], "_");
        }
    }

    // Fallback: just use the file stem
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

/// Mangle a function name with its module prefix.
/// If prefix is empty, return the name unchanged.
/// Otherwise return `{prefix}__{name}`.
pub fn mangle(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}__{name}")
    }
}

/// Demangle a mangled symbol for error messages.
/// `nsl_math__clamp` → `nsl.math.clamp`.
/// If no `__` separator, return as-is.
#[allow(dead_code)] // Utility for debugging and future tooling
pub fn demangle(mangled: &str) -> String {
    match mangled.split_once("__") {
        Some((prefix, name)) => {
            let module_path = prefix.replace('_', ".");
            format!("{module_path}.{name}")
        }
        None => mangled.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_module_prefix() {
        let base = PathBuf::from("/project");
        assert_eq!(
            module_prefix(Path::new("/project/nsl/math.nsl"), &base),
            "nsl_math"
        );
        assert_eq!(
            module_prefix(Path::new("/project/utils.nsl"), &base),
            "utils"
        );
    }

    #[test]
    fn test_mangle() {
        assert_eq!(mangle("nsl_math", "clamp"), "nsl_math__clamp");
        assert_eq!(mangle("", "main_helper"), "main_helper");
    }

    #[test]
    fn test_demangle() {
        assert_eq!(demangle("nsl_math__clamp"), "nsl.math.clamp");
        assert_eq!(demangle("main_helper"), "main_helper");
    }
}
