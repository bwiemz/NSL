use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::CodegenError;

/// Path to the pre-built nsl-runtime static library, set at compile time by build.rs.
const RUNTIME_LIB_PATH: &str = env!("NSL_RUNTIME_LIB_PATH");

/// Find the runtime library, searching multiple locations for toolchain distribution.
fn find_runtime_lib() -> Result<PathBuf, CodegenError> {
    // 1. Check env var override (for custom installations)
    if let Ok(path) = std::env::var("NSL_RUNTIME_LIB_PATH_OVERRIDE") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Ok(p);
        }
    }

    // 2. Check relative to executable: <exe>/../lib/ (toolchain distribution)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(bin_dir) = exe.parent() {
            if let Some(toolchain_dir) = bin_dir.parent() {
                let lib_dir = toolchain_dir.join("lib");
                let lib_name = if cfg!(windows) {
                    "nsl_runtime.lib"
                } else {
                    "libnsl_runtime.a"
                };
                let lib_path = lib_dir.join(lib_name);
                if lib_path.exists() {
                    return Ok(lib_path);
                }
            }
        }
    }

    // 3. Fallback to compile-time path (cargo build scenario)
    let compile_time = PathBuf::from(RUNTIME_LIB_PATH);
    if compile_time.exists() {
        return Ok(compile_time);
    }

    Err(CodegenError::new(
        "nsl-runtime static library not found. Ensure the lib/ directory is next to bin/.".to_string(),
    ))
}

/// Link a single object file into an executable (backward compatible).
pub fn link(obj_path: &Path, output_path: &Path) -> Result<(), CodegenError> {
    link_multi(&[obj_path.to_path_buf()], output_path)
}

/// Link multiple object files into an executable.
pub fn link_multi(obj_paths: &[PathBuf], output_path: &Path) -> Result<(), CodegenError> {
    let runtime_lib = find_runtime_lib()?;

    if cfg!(target_os = "windows") {
        link_msvc_multi(obj_paths, output_path, &runtime_lib)
            .or_else(|_| link_gcc_multi(obj_paths, output_path, &runtime_lib))
    } else {
        link_gcc_multi(obj_paths, output_path, &runtime_lib)
    }
}

/// Link using gcc/cc/clang (MinGW on Windows, standard on Unix).
fn link_gcc_multi(
    obj_paths: &[PathBuf],
    output_path: &Path,
    runtime_lib: &Path,
) -> Result<(), CodegenError> {
    let cc = find_c_compiler()?;

    let mut cmd = Command::new(&cc);
    cmd.arg("-o").arg(output_path);
    for obj_path in obj_paths {
        cmd.arg(obj_path);
    }
    cmd.arg(runtime_lib);

    if !cfg!(target_os = "windows") {
        cmd.arg("-lm");
        cmd.arg("-lpthread");
        cmd.arg("-ldl");
    }

    // Link CUDA driver library if available
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(&cuda_path).join("lib64");
        if cuda_lib.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib.display()));
            cmd.arg("-lcuda");
        }
    }

    let status = cmd
        .status()
        .map_err(|e| CodegenError::new(format!("failed to run linker '{cc}': {e}")))?;

    if !status.success() {
        return Err(CodegenError::new(format!(
            "linker '{cc}' failed (exit: {status})"
        )));
    }

    Ok(())
}

/// Link using MSVC toolchain (link.exe).
fn link_msvc_multi(
    obj_paths: &[PathBuf],
    output_path: &Path,
    runtime_lib: &Path,
) -> Result<(), CodegenError> {
    let msvc = find_msvc()?;

    let mut cmd = Command::new(&msvc.link);
    cmd.arg("/nologo")
        .arg(format!("/OUT:{}", output_path.display()));
    for obj_path in obj_paths {
        cmd.arg(obj_path);
    }
    cmd.arg(runtime_lib);

    // Add library paths
    for lib_path in &msvc.lib_paths {
        cmd.arg(format!("/LIBPATH:{}", lib_path.display()));
    }

    // Required system libraries for the Rust static lib
    cmd.args([
        "msvcrt.lib",
        "legacy_stdio_definitions.lib",
        "kernel32.lib",
        "advapi32.lib",
        "bcrypt.lib",
        "ntdll.lib",
        "userenv.lib",
        "ws2_32.lib",
        "synchronization.lib",
        "shell32.lib",
        "ole32.lib",
        "/NODEFAULTLIB:LIBCMT",
    ]);

    // Link CUDA driver library if available (needed when runtime has cuda feature)
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(&cuda_path).join("lib").join("x64");
        if cuda_lib.is_dir() {
            cmd.arg(format!("/LIBPATH:{}", cuda_lib.display()));
            cmd.arg("cuda.lib");
        }
    }

    let status = cmd
        .status()
        .map_err(|e| CodegenError::new(format!("failed to run link.exe: {e}")))?;

    if !status.success() {
        return Err(CodegenError::new(format!(
            "link.exe failed (exit: {status})"
        )));
    }

    Ok(())
}

struct MsvcPaths {
    link: PathBuf,
    lib_paths: Vec<PathBuf>,
}

/// Find MSVC tools (link.exe) and library paths.
fn find_msvc() -> Result<MsvcPaths, CodegenError> {
    // Search common VS installation paths
    let vs_roots = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
    ];

    let mut msvc_bin = None;
    let mut msvc_lib = None;

    for root in &vs_roots {
        let vc_tools = PathBuf::from(root).join(r"VC\Tools\MSVC");
        if vc_tools.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&vc_tools) {
                let mut versions: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect();
                versions.sort();
                if let Some(latest) = versions.last() {
                    let bin = latest.join(r"bin\Hostx64\x64");
                    let lib = latest.join(r"lib\x64");
                    if bin.join("link.exe").exists() {
                        msvc_bin = Some(bin);
                        msvc_lib = Some(lib);
                        break;
                    }
                }
            }
        }
    }

    let msvc_bin = msvc_bin.ok_or_else(|| {
        CodegenError::new("MSVC tools not found. Install Visual Studio Build Tools or gcc")
    })?;
    let msvc_lib = msvc_lib.unwrap();

    // Find Windows SDK
    let sdk_base = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10");
    let sdk_lib_root = sdk_base.join("Lib");
    let mut sdk_lib_paths = Vec::new();

    // Find latest SDK version for libs
    if sdk_lib_root.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&sdk_lib_root) {
            let mut versions: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            versions.sort();
            if let Some(latest) = versions.last() {
                let ucrt = latest.join(r"ucrt\x64");
                let um = latest.join(r"um\x64");
                if ucrt.is_dir() { sdk_lib_paths.push(ucrt); }
                if um.is_dir() { sdk_lib_paths.push(um); }
            }
        }
    }

    let mut lib_paths = vec![msvc_lib];
    lib_paths.extend(sdk_lib_paths);

    Ok(MsvcPaths {
        link: msvc_bin.join("link.exe"),
        lib_paths,
    })
}

/// Find a C compiler (gcc/cc/clang) on the PATH.
fn find_c_compiler() -> Result<String, CodegenError> {
    for candidate in &["gcc", "cc", "clang"] {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Ok(candidate.to_string());
        }
    }

    Err(CodegenError::new(
        "no C compiler found. Install gcc, clang, or Visual Studio Build Tools",
    ))
}

/// Determine default output path from input .nsl path.
pub fn default_output_path(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default();
    let mut out = input.with_file_name(stem);
    if cfg!(target_os = "windows") {
        out.set_extension("exe");
    }
    out
}
