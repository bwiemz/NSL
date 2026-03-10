use std::path::{Path, PathBuf};

use nsl_ast::Symbol;
use nsl_lexer::Interner;

/// Resolve an import path (e.g., `[utils]` or `[math, utils]`) to a filesystem path.
///
/// The path is resolved relative to the directory of the importing file.
/// `math.utils` becomes `math/utils.nsl`.
pub fn resolve_import(
    import_path: &[Symbol],
    importing_file: &Path,
    interner: &Interner,
) -> Result<PathBuf, String> {
    if import_path.is_empty() {
        return Err("empty import path".to_string());
    }

    let segments: Vec<&str> = import_path
        .iter()
        .map(|sym| {
            interner
                .resolve(sym.0)
                .unwrap_or("<unknown>")
        })
        .collect();

    let module_name = segments.join(".");

    // Build relative path: math.utils → math/utils.nsl
    let mut rel = PathBuf::new();
    for (i, seg) in segments.iter().enumerate() {
        if i < segments.len() - 1 {
            rel.push(seg);
        } else {
            rel.push(format!("{seg}.nsl"));
        }
    }

    // Resolve relative to the importing file's parent directory
    let base_dir = importing_file
        .parent()
        .ok_or_else(|| format!("cannot determine parent directory of '{}'", importing_file.display()))?;

    let candidate = base_dir.join(&rel);

    if candidate.is_file() {
        // Canonicalize to get a stable key for the module graph
        candidate
            .canonicalize()
            .map_err(|e| format!("module '{module_name}' found at '{}' but cannot canonicalize: {e}", candidate.display()))
    } else {
        Err(format!(
            "module '{}' not found (searched: {})",
            module_name,
            candidate.display()
        ))
    }
}
