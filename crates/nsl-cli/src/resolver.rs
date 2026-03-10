use std::path::{Path, PathBuf};

use nsl_ast::Symbol;
use nsl_lexer::Interner;

/// Return all valid stdlib root directories, ordered by priority.
///
/// Resolution order:
/// 1. `NSL_STDLIB_PATH` environment variable (if set and directory exists)
/// 2. `<exe_dir>/stdlib/` (relative to the running executable)
pub(crate) fn stdlib_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    // 1. Check NSL_STDLIB_PATH env var
    if let Ok(env_path) = std::env::var("NSL_STDLIB_PATH") {
        let p = PathBuf::from(&env_path);
        if p.is_dir() {
            roots.push(p);
        }
    }

    // 2. exe-relative stdlib/
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let stdlib = exe_dir.join("stdlib");
            if stdlib.is_dir() {
                roots.push(stdlib);
            }
        }
    }

    roots
}

/// Resolve an import path (e.g., `[utils]` or `[nsl, math]`) to a filesystem path.
///
/// Resolution order:
/// 1. Relative to the importing file's directory (e.g., `math.utils` → `./math/utils.nsl`)
/// 2. Stdlib via `NSL_STDLIB_PATH` env var (e.g., `nsl.math` → `$NSL_STDLIB_PATH/nsl/math.nsl`)
/// 3. Stdlib relative to executable (e.g., `nsl.math` → `<exe_dir>/stdlib/nsl/math.nsl`)
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

    // 1. Resolve relative to the importing file's parent directory
    let base_dir = importing_file
        .parent()
        .ok_or_else(|| format!("cannot determine parent directory of '{}'", importing_file.display()))?;

    let relative_candidate = base_dir.join(&rel);

    if relative_candidate.is_file() {
        // Canonicalize to get a stable key for the module graph
        return relative_candidate
            .canonicalize()
            .map_err(|e| format!("module '{module_name}' found at '{}' but cannot canonicalize: {e}", relative_candidate.display()));
    }

    // 2 & 3. Try all stdlib roots (env var first, then exe-relative)
    let mut searched = vec![relative_candidate.display().to_string()];

    for stdlib in stdlib_roots() {
        let stdlib_candidate = stdlib.join(&rel);
        searched.push(stdlib_candidate.display().to_string());

        if stdlib_candidate.is_file() {
            return stdlib_candidate
                .canonicalize()
                .map_err(|e| format!("module '{module_name}' found at '{}' but cannot canonicalize: {e}", stdlib_candidate.display()));
        }
    }

    Err(format!(
        "module '{}' not found (searched: {})",
        module_name,
        searched.join(", ")
    ))
}
