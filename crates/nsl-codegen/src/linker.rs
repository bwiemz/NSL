use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

use serde::Deserialize;

use crate::error::CodegenError;

/// Path to the pre-built nsl-runtime static library, set at compile time by build.rs.
const RUNTIME_LIB_PATH: &str = env!("NSL_RUNTIME_LIB_PATH");
const RUNTIME_RUSTFLAGS: &str = env!("NSL_RUNTIME_RUSTFLAGS");

#[derive(Deserialize)]
struct RuntimeFingerprintMetadata {
    features: String,
    #[serde(default)]
    rustflags: Vec<String>,
}

fn wants_cuda_runtime() -> bool {
    cfg!(feature = "cuda")
}

fn current_target_os() -> &'static str {
    if cfg!(windows) {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "other"
    }
}

fn current_target_env() -> &'static str {
    if cfg!(target_env = "msvc") {
        "msvc"
    } else if cfg!(target_env = "gnu") {
        "gnu"
    } else {
        "other"
    }
}

fn runtime_dep_filenames(target_os: &str, target_env: &str, hash: &str) -> Vec<String> {
    if target_os == "windows" && target_env == "msvc" {
        vec![format!("nsl_runtime-{hash}.lib")]
    } else if target_os == "windows" {
        vec![format!("libnsl_runtime-{hash}.a")]
    } else {
        vec![
            format!("libnsl_runtime-{hash}.a"),
            format!("libnsl_runtime-{hash}.rlib"),
        ]
    }
}

fn runtime_candidate_paths(deps_dir: &Path, hash: &str) -> Vec<PathBuf> {
    runtime_dep_filenames(current_target_os(), current_target_env(), hash)
        .into_iter()
        .map(|name| deps_dir.join(name))
        .collect()
}

fn runtime_candidate_mtime(path: &Path) -> Duration {
    path.metadata()
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(SystemTime::UNIX_EPOCH).ok())
        .unwrap_or_default()
}

fn runtime_top_level_filename(target_os: &str, target_env: &str) -> &'static str {
    if target_os == "windows" && target_env == "msvc" {
        "nsl_runtime.lib"
    } else {
        "libnsl_runtime.a"
    }
}

fn find_runtime_lib_in_toolchain_dir(
    toolchain_dir: &Path,
    target_os: &str,
    target_env: &str,
) -> Option<PathBuf> {
    let lib_path = toolchain_dir
        .join("lib")
        .join(runtime_top_level_filename(target_os, target_env));
    lib_path.exists().then_some(lib_path)
}

fn read_fingerprint_metadata(path: &Path) -> Option<RuntimeFingerprintMetadata> {
    let raw = fs::read(path).ok()?;
    serde_json::from_slice(&raw).ok()
}

fn fingerprint_has_cuda(path: &Path) -> Option<bool> {
    let metadata = read_fingerprint_metadata(path)?;
    let features: Vec<String> = serde_json::from_str(&metadata.features).ok()?;
    Some(features.iter().any(|feature| feature == "cuda"))
}

fn fingerprint_matches_current_rustflags(path: &Path) -> Option<bool> {
    let metadata = read_fingerprint_metadata(path)?;
    Some(metadata.rustflags.join("\u{1f}") == RUNTIME_RUSTFLAGS)
}

fn find_feature_matched_runtime_lib(compile_time_path: &Path) -> Option<PathBuf> {
    let profile_dir = compile_time_path.parent()?;
    let fingerprints_dir = profile_dir.join(".fingerprint");
    let deps_dir = profile_dir.join("deps");
    if !fingerprints_dir.is_dir() || !deps_dir.is_dir() {
        return None;
    }

    let wants_cuda = wants_cuda_runtime();
    let mut exact_matches = Vec::new();
    let mut fallback_matches = Vec::new();

    let entries = fs::read_dir(&fingerprints_dir).ok()?;
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        let Some(hash) = file_name.strip_prefix("nsl-runtime-") else {
            continue;
        };

        let fingerprint_path = entry.path().join("lib-nsl_runtime.json");
        let Some(has_cuda) = fingerprint_has_cuda(&fingerprint_path) else {
            continue;
        };
        if has_cuda != wants_cuda {
            continue;
        }
        let rustflags_match =
            fingerprint_matches_current_rustflags(&fingerprint_path).unwrap_or(false);

        for candidate in runtime_candidate_paths(&deps_dir, hash) {
            if candidate.exists() {
                let entry = (runtime_candidate_mtime(&candidate), candidate);
                if rustflags_match {
                    exact_matches.push(entry);
                } else {
                    fallback_matches.push(entry);
                }
                break;
            }
        }
    }

    exact_matches.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));
    if let Some((_, path)) = exact_matches.into_iter().next() {
        return Some(path);
    }

    fallback_matches.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));
    fallback_matches.into_iter().next().map(|(_, path)| path)
}

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
                if let Some(lib_path) = find_runtime_lib_in_toolchain_dir(
                    toolchain_dir,
                    current_target_os(),
                    current_target_env(),
                ) {
                    return Ok(lib_path);
                }
            }
        }
    }

    // 3. Fallback to compile-time path (cargo build scenario)
    let compile_time = PathBuf::from(RUNTIME_LIB_PATH);
    if let Some(feature_matched) = find_feature_matched_runtime_lib(&compile_time) {
        return Ok(feature_matched);
    }
    if compile_time.exists() {
        return Ok(compile_time);
    }

    Err(CodegenError::new(
        "nsl-runtime static library not found. Ensure the lib/ directory is next to bin/."
            .to_string(),
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
        // Prefer GCC (MinGW/MSYS2) over MSVC on Windows when available.
        // MSVC's /GS security cookies cause false STATUS_STACK_BUFFER_OVERRUN
        // when Cranelift-generated code calls into the Rust runtime.
        // GCC doesn't have this issue.
        link_gcc_multi(obj_paths, output_path, &runtime_lib)
            .or_else(|_| link_msvc_multi(obj_paths, output_path, &runtime_lib))
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
    // 64MB stack for deep model training (tape AD on 8+ transformer blocks)
    if cfg!(target_os = "windows") {
        cmd.arg("-Wl,--stack,67108864");
    }
    for obj_path in obj_paths {
        cmd.arg(obj_path);
    }
    cmd.arg(runtime_lib);

    // Windows system libraries AFTER runtime lib (GCC link order: libs after objects)
    if cfg!(target_os = "windows") {
        cmd.args([
            "-lws2_32",
            "-lntdll",
            "-lbcrypt",
            "-ladvapi32",
            "-luserenv",
            "-lkernel32",
            "-lsynchronization",
            "-lshell32",
            "-lole32",
        ]);
    }

    if cfg!(target_os = "linux") {
        cmd.arg("-lm");
        cmd.arg("-lpthread");
        cmd.arg("-ldl");
    }

    // macOS: inject platform version since Cranelift object files lack LC_BUILD_VERSION
    if cfg!(target_os = "macos") {
        cmd.args(["-Wl,-platform_version,macos,11.0,11.0"]);
    }

    // Link CUDA driver library if available (Linux: lib64/, Windows: lib/x64/)
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(&cuda_path).join("lib64");
        let cuda_lib_win = PathBuf::from(&cuda_path).join("lib").join("x64");
        if cuda_lib.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib.display()));
            cmd.arg("-lcuda");
        } else if cuda_lib_win.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib_win.display()));
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
        .arg(format!("/OUT:{}", output_path.display()))
        // 64MB stack — tape-based AD on deep models (8+ transformer blocks)
        // creates deep call chains that overflow the default 1MB stack.
        .arg("/STACK:67108864");
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
        // Disable Control Flow Guard — Cranelift object files don't have CFG
        // metadata, so the linker's guard checks trigger false positives.
        "/GUARD:NO",
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
                if ucrt.is_dir() {
                    sdk_lib_paths.push(ucrt);
                }
                if um.is_dir() {
                    sdk_lib_paths.push(um);
                }
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
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Ok(candidate.to_string());
        }
    }

    Err(CodegenError::new(
        "no C compiler found. Install gcc, clang, or Visual Studio Build Tools",
    ))
}

/// M62a: Link multiple object files into a shared library (.so/.dylib/.dll).
pub fn link_shared(obj_paths: &[PathBuf], output_path: &Path) -> Result<(), CodegenError> {
    let runtime_lib = find_runtime_lib()?;

    if cfg!(target_os = "windows") {
        link_shared_msvc(obj_paths, output_path, &runtime_lib)
            .or_else(|_| link_shared_gcc(obj_paths, output_path, &runtime_lib))
    } else {
        link_shared_gcc(obj_paths, output_path, &runtime_lib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::tempdir;

    fn top_level_runtime_name_for_target(target_os: &str, target_env: &str) -> &'static str {
        if target_os == "windows" && target_env == "msvc" {
            "nsl_runtime.lib"
        } else {
            "libnsl_runtime.a"
        }
    }

    fn deps_runtime_name(hash: &str) -> String {
        runtime_dep_filenames(current_target_os(), current_target_env(), hash)
            .into_iter()
            .next()
            .unwrap()
    }

    fn write_fingerprint(profile_dir: &Path, hash: &str, features: &str, rustflags: &str) {
        let fingerprint_dir = profile_dir
            .join(".fingerprint")
            .join(format!("nsl-runtime-{hash}"));
        fs::create_dir_all(&fingerprint_dir).unwrap();
        let escaped_features = features.replace('"', "\\\"");
        let rustflags_json = if rustflags.is_empty() {
            "[]".to_string()
        } else {
            let values: Vec<String> = rustflags
                .split('\u{1f}')
                .filter(|value| !value.is_empty())
                .map(|value| format!(r#""{value}""#))
                .collect();
            format!("[{}]", values.join(","))
        };
        fs::write(
            fingerprint_dir.join("lib-nsl_runtime.json"),
            format!(r#"{{"features":"{escaped_features}","rustflags":{rustflags_json}}}"#),
        )
        .unwrap();
    }

    #[test]
    fn selects_feature_matched_runtime_from_deps() {
        let temp_dir = tempdir().unwrap();
        let profile_dir = temp_dir.path();
        fs::create_dir_all(profile_dir.join("deps")).unwrap();

        let cpu_hash = "cpu123";
        let cuda_hash = "cuda123";
        write_fingerprint(profile_dir, cpu_hash, r#"["default"]"#, RUNTIME_RUSTFLAGS);
        write_fingerprint(
            profile_dir,
            cuda_hash,
            r#"["cuda", "default"]"#,
            RUNTIME_RUSTFLAGS,
        );

        let cpu_lib = profile_dir.join("deps").join(deps_runtime_name(cpu_hash));
        let cuda_lib = profile_dir.join("deps").join(deps_runtime_name(cuda_hash));
        fs::write(&cpu_lib, b"cpu").unwrap();
        fs::write(&cuda_lib, b"cuda").unwrap();

        let compile_time = profile_dir.join(top_level_runtime_name_for_target(
            current_target_os(),
            current_target_env(),
        ));
        fs::write(&compile_time, b"stale-top-level").unwrap();

        let selected = find_feature_matched_runtime_lib(&compile_time).unwrap();
        let expected = if cfg!(feature = "cuda") {
            cuda_lib
        } else {
            cpu_lib
        };

        assert_eq!(selected, expected);
    }

    #[test]
    fn returns_none_when_no_feature_match_exists() {
        let temp_dir = tempdir().unwrap();
        let profile_dir = temp_dir.path();
        fs::create_dir_all(profile_dir.join("deps")).unwrap();

        let mismatched_hash = "mismatch123";
        let mismatched_features = if cfg!(feature = "cuda") {
            r#"["default"]"#
        } else {
            r#"["cuda", "default"]"#
        };

        write_fingerprint(
            profile_dir,
            mismatched_hash,
            mismatched_features,
            RUNTIME_RUSTFLAGS,
        );
        fs::write(
            profile_dir
                .join("deps")
                .join(deps_runtime_name(mismatched_hash)),
            b"mismatch",
        )
        .unwrap();

        let compile_time = profile_dir.join(top_level_runtime_name_for_target(
            current_target_os(),
            current_target_env(),
        ));
        fs::write(&compile_time, b"fallback").unwrap();

        assert_eq!(find_feature_matched_runtime_lib(&compile_time), None);
    }

    #[test]
    fn uses_windows_gnu_archive_names() {
        assert_eq!(
            runtime_dep_filenames("windows", "gnu", "abc123"),
            vec!["libnsl_runtime-abc123.a".to_string()]
        );
        assert_eq!(
            top_level_runtime_name_for_target("windows", "gnu"),
            "libnsl_runtime.a"
        );
    }

    #[test]
    fn matches_cuda_feature_by_exact_name() {
        let temp_dir = tempdir().unwrap();
        let fingerprint_path = temp_dir.path().join("lib-nsl_runtime.json");
        fs::write(
            &fingerprint_path,
            r#"{"features":"[\"cuda_extra\", \"default\"]","rustflags":[]}"#,
        )
        .unwrap();

        assert_eq!(fingerprint_has_cuda(&fingerprint_path), Some(false));
    }

    #[test]
    fn finds_windows_gnu_runtime_in_toolchain_lib_dir() {
        let temp_dir = tempdir().unwrap();
        let toolchain_dir = temp_dir.path();
        let lib_dir = toolchain_dir.join("lib");
        fs::create_dir_all(&lib_dir).unwrap();

        let expected = lib_dir.join("libnsl_runtime.a");
        fs::write(&expected, b"gnu-runtime").unwrap();

        let selected = find_runtime_lib_in_toolchain_dir(toolchain_dir, "windows", "gnu").unwrap();
        assert_eq!(selected, expected);
    }

    #[test]
    fn prefers_exact_rustflags_match_over_newer_fallback() {
        let temp_dir = tempdir().unwrap();
        let profile_dir = temp_dir.path();
        fs::create_dir_all(profile_dir.join("deps")).unwrap();

        let exact_hash = "exact123";
        let fallback_hash = "fallback123";
        let features = if cfg!(feature = "cuda") {
            r#"["cuda", "default"]"#
        } else {
            r#"["default"]"#
        };

        write_fingerprint(profile_dir, fallback_hash, features, "-Ctarget-cpu=native");
        write_fingerprint(profile_dir, exact_hash, features, RUNTIME_RUSTFLAGS);

        let fallback_lib = profile_dir
            .join("deps")
            .join(deps_runtime_name(fallback_hash));
        let exact_lib = profile_dir.join("deps").join(deps_runtime_name(exact_hash));
        fs::write(&fallback_lib, b"fallback").unwrap();
        fs::write(&exact_lib, b"exact").unwrap();

        let compile_time = profile_dir.join(top_level_runtime_name_for_target(
            current_target_os(),
            current_target_env(),
        ));
        fs::write(&compile_time, b"top-level").unwrap();

        let selected = find_feature_matched_runtime_lib(&compile_time).unwrap();
        assert_eq!(selected, exact_lib);
    }
}

/// M62a: Link a shared library using gcc/cc/clang (-shared flag).
fn link_shared_gcc(
    obj_paths: &[PathBuf],
    output_path: &Path,
    runtime_lib: &Path,
) -> Result<(), CodegenError> {
    let cc = find_c_compiler()?;

    let mut cmd = Command::new(&cc);
    cmd.arg("-shared");
    cmd.arg("-o").arg(output_path);
    for obj_path in obj_paths {
        cmd.arg(obj_path);
    }
    cmd.arg(runtime_lib);

    if cfg!(target_os = "linux") {
        cmd.arg("-lm");
        cmd.arg("-lpthread");
        cmd.arg("-ldl");
        // Set SONAME so the dynamic linker can identify the library
        if let Some(filename) = output_path.file_name() {
            cmd.arg(format!("-Wl,-soname,{}", filename.to_string_lossy()));
        }
    }

    // macOS: inject platform version since Cranelift object files lack LC_BUILD_VERSION
    if cfg!(target_os = "macos") {
        cmd.args(["-Wl,-platform_version,macos,11.0,11.0"]);
    }

    // Link CUDA driver library if available (Linux: lib64/, Windows: lib/x64/)
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(&cuda_path).join("lib64");
        let cuda_lib_win = PathBuf::from(&cuda_path).join("lib").join("x64");
        if cuda_lib.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib.display()));
            cmd.arg("-lcuda");
        } else if cuda_lib_win.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib_win.display()));
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

/// M62a: Link a shared library using MSVC toolchain (link.exe /DLL).
fn link_shared_msvc(
    obj_paths: &[PathBuf],
    output_path: &Path,
    runtime_lib: &Path,
) -> Result<(), CodegenError> {
    let msvc = find_msvc()?;

    let mut cmd = Command::new(&msvc.link);
    cmd.arg("/nologo")
        .arg("/DLL")
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

    // Link CUDA driver library if available
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

/// M62a: Determine default shared library output path from input .nsl path.
pub fn default_shared_lib_path(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    if cfg!(target_os = "windows") {
        parent.join(format!("{stem}.dll"))
    } else if cfg!(target_os = "macos") {
        parent.join(format!("lib{stem}.dylib"))
    } else {
        parent.join(format!("lib{stem}.so"))
    }
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
