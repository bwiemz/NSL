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

/// A symbol that exists ONLY in cuda-featured runtime archives. Used to
/// verify a candidate archive actually matches this binary's features —
/// the fingerprint scan can transiently fail mid-rebuild (concurrent
/// workspace builds rewrite .fingerprint entries), and the mtime-newest
/// fallback then happily links a CPU-only runtime into a CUDA program,
/// which surfaces later as a runtime "CUDA support not compiled" abort in
/// the trained program rather than a link-time error.
const CUDA_MARKER_SYMBOL: &[u8] = b"gpu_fase_fused_adamw_step";

/// Cheap containment probe: the GNU ar symbol index sits at the head of the
/// archive and names every exported symbol, so scanning the first slice of
/// the file is sufficient (and pinned wrong by neither strip level nor
/// member ordering). Reads at most 64 MiB.
fn archive_has_cuda_marker(path: &Path) -> bool {
    use std::io::Read;
    const HEAD_LIMIT: usize = 64 * 1024 * 1024;
    let Ok(file) = fs::File::open(path) else {
        return false;
    };
    let mut head = Vec::with_capacity(HEAD_LIMIT.min(8 * 1024 * 1024));
    if file.take(HEAD_LIMIT as u64).read_to_end(&mut head).is_err() {
        return false;
    }
    let m = CUDA_MARKER_SYMBOL;
    if head.len() < m.len() {
        return false;
    }
    // Naive first-byte-skip search — runs once per link over the index head.
    let first = m[0];
    let mut i = 0;
    while i + m.len() <= head.len() {
        match head[i..].iter().position(|&b| b == first) {
            Some(off) => {
                let start = i + off;
                if start + m.len() > head.len() {
                    return false;
                }
                if &head[start..start + m.len()] == m {
                    return true;
                }
                i = start + 1;
            }
            None => return false,
        }
    }
    false
}

/// Feature-verify a candidate archive against this binary's build. Only the
/// cuda direction is checked: linking a cuda archive into a non-cuda build
/// already fails loudly at link time on the missing driver symbols, but the
/// reverse silently produces a binary that aborts mid-training.
fn archive_matches_features(path: &Path) -> bool {
    !wants_cuda_runtime() || archive_has_cuda_marker(path)
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
    // Feature-verify each candidate newest-first: the fingerprint said cuda,
    // but the archive itself is the ground truth (a fingerprint written by a
    // concurrent build can point at an archive that is mid-write or from a
    // differently-featured unification).
    for (_, path) in exact_matches {
        if archive_matches_features(&path) {
            return Some(path);
        }
    }

    fallback_matches.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));
    fallback_matches
        .into_iter()
        .map(|(_, path)| path)
        .find(|path| archive_matches_features(path))
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
    if compile_time.exists() && archive_matches_features(&compile_time) {
        return Ok(compile_time);
    }

    if compile_time.exists() {
        // The archive exists but fails the feature probe — refuse rather
        // than link a CPU-only runtime into a CUDA binary (the failure
        // would otherwise surface as "CUDA support not compiled" aborts
        // inside the trained program, far from the cause).
        return Err(CodegenError::new(
            "nsl-runtime static library found, but no candidate contains the \
             CUDA runtime this binary was built with (concurrent workspace \
             builds can leave differently-featured archives newest in \
             target/). Rebuild with `cargo build -p nsl-cli --features cuda` \
             or set NSL_RUNTIME_LIB_PATH_OVERRIDE."
                .to_string(),
        ));
    }

    Err(CodegenError::new(
        "nsl-runtime static library not found. Ensure the lib/ directory is next to bin/."
            .to_string(),
    ))
}

/// Strip the host-ABI symbol-name prefix so logical NSL symbol names compare
/// uniformly across platforms.
///
/// Mach-O (macOS) decorates every exported and undefined symbol with a
/// leading underscore: a symbol declared in Cranelift as
/// `nsl_calib_model_forward` appears in the object file as
/// `_nsl_calib_model_forward`. ELF (Linux) and COFF (Windows MSVC) do not.
/// Every consumer that reads symbol names back out of an emitted object
/// (production post-checks and object-inspecting tests alike) must compare
/// against the un-prefixed logical name, so strip a single leading
/// underscore on macOS only.
#[inline]
pub fn strip_host_symbol_prefix(name: &str) -> &str {
    if cfg!(target_os = "macos") {
        name.strip_prefix('_').unwrap_or(name)
    } else {
        name
    }
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
        // nsl-runtime (with `interop`) pulls in `hf-hub` → `native-tls` →
        // `openssl-sys`. Cargo threads the resulting `-lssl -lcrypto` link
        // args into its own test binaries, but `nsl build`'s standalone
        // linker invocation does not see them — so NSL-built binaries fail
        // to resolve `SSL_CTX_ctrl` etc. unless we add them here.
        cmd.arg("-lssl");
        cmd.arg("-lcrypto");
    }

    // macOS: inject platform version since Cranelift object files lack
    // LC_BUILD_VERSION. Without this flag, ld on arm64 treats unversioned
    // objects as old-style (potentially bearing text-relocations) and rejects
    // them under PIE ("Found illegal text-relocations"; -read_only_relocs,suppress
    // has no effect on modern macOS). The platform_version must be >= the SDK
    // version that ring/native-tls objects were compiled against (the runner's
    // native Xcode SDK), which has changed across runner image refreshes
    // (11.0 → 14.0 → 15.0 in prior fixes). We now detect the SDK version
    // dynamically via `xcrun --show-sdk-version` so no manual bump is needed
    // when Apple ships a new SDK on the CI runners.
    if cfg!(target_os = "macos") {
        let sdk_ver = std::process::Command::new("xcrun")
            .args(["--show-sdk-version"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "15.5".to_string());
        cmd.args([format!("-Wl,-platform_version,macos,{sdk_ver},{sdk_ver}")]);
        // `native-tls` uses the Security framework on macOS (via the
        // `security-framework` crate); the symbols end up in
        // `libnsl_runtime.a` but Cargo's link directives don't reach this
        // standalone link step, so we add them explicitly.
        cmd.args([
            "-framework", "Security",
            "-framework", "CoreFoundation",
        ]);
    }

    // Link CUDA driver library if available (Linux: lib64/, Windows: lib/x64/)
    // Also link cuBLAS (added 2026-04-21 for the nsl_matmul_f32 cublasSgemm swap).
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(&cuda_path).join("lib64");
        let cuda_lib_win = PathBuf::from(&cuda_path).join("lib").join("x64");
        if cuda_lib.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib.display()));
            cmd.arg("-lcuda");
            cmd.arg("-lcublas");
        } else if cuda_lib_win.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib_win.display()));
            cmd.arg("-lcuda");
            cmd.arg("-lcublas");
        }
    }
    add_nccl_link_args(&mut cmd);

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

/// Is a failed link.exe invocation a TRANSIENT output-file lock (retryable)
/// rather than a real link error? On Windows, `link.exe` reports LNK1104
/// "cannot open file '<path>'" both for a missing INPUT library (a real,
/// persistent error) and for an OUTPUT file it cannot write (a transient
/// lock — Defender's real-time scan grabs a handle on the freshly-created
/// .exe, or the failed GCC attempt on the SAME output path left a lingering
/// handle). We only retry when the LNK1104 names the OUTPUT file, so a
/// genuinely-missing input lib still fails fast.
fn is_transient_link_failure(link_output: &str, out_file_name: &str) -> bool {
    if !link_output.contains("LNK1104") {
        return false;
    }
    // A missing input lib is "cannot open file 'foo.lib'"; the transient case
    // is the output binary itself. Match on the output file name (present in
    // the LNK1104 line) — if we somehow can't identify it, don't retry.
    !out_file_name.is_empty() && link_output.contains(out_file_name)
}

/// Link using MSVC toolchain (link.exe).
fn link_msvc_multi(
    obj_paths: &[PathBuf],
    output_path: &Path,
    runtime_lib: &Path,
) -> Result<(), CodegenError> {
    let msvc = find_msvc()?;

    // CUDA libs are resolved once (env + dir probe) and reused across retries.
    let cuda_lib_dir = std::env::var("CUDA_PATH")
        .ok()
        .map(|p| PathBuf::from(p).join("lib").join("x64"))
        .filter(|d| d.is_dir());

    // Build a fresh Command each attempt (Command is not reusable after run).
    let build_cmd = || -> Command {
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
        // Link CUDA driver + cuBLAS libraries if available (needed when runtime
        // has cuda feature).  cublas.lib added 2026-04-21 for the nsl_matmul_f32
        // cublasSgemm swap.
        if let Some(ref cuda_lib) = cuda_lib_dir {
            cmd.arg(format!("/LIBPATH:{}", cuda_lib.display()));
            cmd.arg("cuda.lib");
            cmd.arg("cublas.lib");
        }
        cmd
    };

    // Retry on a transient LNK1104 output-file lock. This is the chronic AWQ
    // calibration-binary flake on Windows CI: link_multi tries GCC first, which
    // fails for this binary and leaves a partial calibration.exe, then this
    // MSVC fallback hits LNK1104 because Defender (or the lingering GCC handle)
    // still holds the output. It clears within a moment — so capture output,
    // drop the locked file, back off, and retry rather than fail the build.
    let out_file_name = output_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    const MAX_ATTEMPTS: u32 = 4;
    for attempt in 1..=MAX_ATTEMPTS {
        let out = build_cmd()
            .output()
            .map_err(|e| CodegenError::new(format!("failed to run link.exe: {e}")))?;
        if out.status.success() {
            // .status() used to STREAM link.exe's output; .output() captures it,
            // so re-emit on the happy path to preserve linker warnings
            // (LNK4098 defaultlib conflict, LNK4099 missing PDB, …) that would
            // otherwise be silently swallowed. Route stdout→stdout, stderr→stderr.
            if !out.stdout.is_empty() {
                print!("{}", String::from_utf8_lossy(&out.stdout));
            }
            if !out.stderr.is_empty() {
                eprint!("{}", String::from_utf8_lossy(&out.stderr));
            }
            return Ok(());
        }
        let diag = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let transient = is_transient_link_failure(&diag, out_file_name);
        if transient && attempt < MAX_ATTEMPTS {
            // Drop the locked/partial output so the retry writes cleanly, then
            // back off (exponential: 200/400/800ms) to let the AV scan or the
            // lingering handle release.
            let _ = std::fs::remove_file(output_path);
            std::thread::sleep(std::time::Duration::from_millis(200 * (1u64 << (attempt - 1))));
            continue;
        }
        // Terminal: either a real (non-transient) failure, or a transient lock
        // that never cleared across all attempts — name which so the operator
        // isn't left with the generic message in the exact case the retry
        // exists to explain.
        return Err(CodegenError::new(if transient {
            format!(
                "link.exe failed after {attempt} attempts — a transient LNK1104 \
                 output-file lock on '{out_file_name}' did not clear (Defender \
                 real-time scan or a lingering handle):\n{diag}"
            )
        } else {
            format!("link.exe failed (exit: {}):\n{diag}", out.status)
        }));
    }
    // The final iteration (attempt == MAX_ATTEMPTS) never `continue`s, so the
    // loop always returns from its body — this is unreachable.
    unreachable!("MSVC link retry loop returns on every path")
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
    let mut checked_roots: Vec<String> = Vec::new();

    for root in &vs_roots {
        let vc_tools = PathBuf::from(root).join(r"VC\Tools\MSVC");
        checked_roots.push(vc_tools.display().to_string());
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

    // Fallback: ask `vswhere.exe` for the latest VS installation. This handles
    // editions / paths that the hard-coded list misses (e.g. side-by-side
    // installs, custom layouts, or runner images that move VS to a different
    // directory). vswhere.exe ships with Visual Studio Installer at a fixed
    // path on every Windows host that has any VS edition installed.
    if msvc_bin.is_none() {
        let vswhere =
            PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
        checked_roots.push(format!("vswhere.exe at {}", vswhere.display()));
        if vswhere.exists() {
            let output = Command::new(&vswhere)
                .args([
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ])
                .output();
            if let Ok(out) = output {
                let install_path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !install_path.is_empty() {
                    let vc_tools = PathBuf::from(&install_path).join(r"VC\Tools\MSVC");
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
                            }
                        }
                    }
                }
            }
        }
    }

    let msvc_bin = msvc_bin.ok_or_else(|| {
        CodegenError::new(format!(
            "MSVC tools not found. Install Visual Studio Build Tools or gcc. \
             Checked: {}",
            checked_roots.join("; ")
        ))
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
    link_shared_with_exports(obj_paths, output_path, &[])
}

/// M62 Task 6: like [`link_shared`] but also forces the named symbols into
/// the export table.  On MSVC, Cranelift's `Linkage::Export` is not enough —
/// `link.exe /DLL` still needs an explicit `/EXPORT:<sym>` to expose the
/// function via `GetProcAddress` / `ctypes.CDLL`.  On Unix `Linkage::Export`
/// is sufficient; the names are accepted but unused.
pub fn link_shared_with_exports(
    obj_paths: &[PathBuf],
    output_path: &Path,
    extra_exports: &[&str],
) -> Result<(), CodegenError> {
    let runtime_lib = find_runtime_lib()?;

    if cfg!(target_os = "windows") {
        link_shared_msvc(obj_paths, output_path, &runtime_lib, extra_exports)
            .or_else(|_| link_shared_gcc(obj_paths, output_path, &runtime_lib))
    } else {
        link_shared_gcc(obj_paths, output_path, &runtime_lib)
    }
}

// These tests stay adjacent to the runtime-library discovery helpers they exercise.
// Keeping the module here preserves that context without changing linker behavior.
#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    use tempfile::tempdir;

    #[test]
    fn transient_lnk1104_on_output_is_retryable() {
        // The chronic AWQ flake: LNK1104 naming the OUTPUT binary (Defender /
        // lingering GCC handle) — retry.
        let diag = "LINK : fatal error LNK1104: cannot open file \
             'C:\\Users\\RUNNER~1\\AppData\\Local\\Temp\\nsl-calibration-1224-178\\calibration.exe'";
        assert!(is_transient_link_failure(diag, "calibration.exe"));
    }

    #[test]
    fn lnk1104_on_input_lib_is_not_retryable() {
        // A genuinely-missing input library must fail fast, not spin retries.
        let diag = "LINK : fatal error LNK1104: cannot open file 'cublas.lib'";
        assert!(!is_transient_link_failure(diag, "calibration.exe"));
    }

    #[test]
    fn non_lnk1104_failure_is_not_retryable() {
        // Unresolved-symbol errors etc. are real and persistent.
        let diag = "model.obj : error LNK2019: unresolved external symbol nsl_calib_model_forward";
        assert!(!is_transient_link_failure(diag, "calibration.exe"));
    }

    #[test]
    fn unidentifiable_output_name_does_not_retry() {
        // Defensive: with no output name we cannot tell output-lock from
        // missing-input, so we do NOT spin retries.
        let diag = "LINK : fatal error LNK1104: cannot open file 'whatever.exe'";
        assert!(!is_transient_link_failure(diag, ""));
    }

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
        // The archive-content probe requires the marker symbol in a
        // cuda-featured archive — fake it in the fixture body.
        fs::write(&cuda_lib, [b"cuda ".as_slice(), CUDA_MARKER_SYMBOL].concat()).unwrap();

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
        // Under cfg(cuda) the content probe must accept both fixtures so the
        // rustflags preference (not the probe) decides.
        fs::write(
            &fallback_lib,
            [b"fallback ".as_slice(), CUDA_MARKER_SYMBOL].concat(),
        )
        .unwrap();
        fs::write(&exact_lib, [b"exact ".as_slice(), CUDA_MARKER_SYMBOL].concat()).unwrap();

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
        // native-tls → openssl-sys (see `link_gcc_multi` comment).
        cmd.arg("-lssl");
        cmd.arg("-lcrypto");
        // Set SONAME so the dynamic linker can identify the library
        if let Some(filename) = output_path.file_name() {
            cmd.arg(format!("-Wl,-soname,{}", filename.to_string_lossy()));
        }
    }

    // macOS: inject platform version — see `link_gcc_multi` above for why.
    // SDK version is detected dynamically so runner image refreshes don't
    // require manual bumps to this file.
    if cfg!(target_os = "macos") {
        let sdk_ver = std::process::Command::new("xcrun")
            .args(["--show-sdk-version"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "15.5".to_string());
        cmd.args([format!("-Wl,-platform_version,macos,{sdk_ver},{sdk_ver}")]);
        // native-tls → security-framework (see `link_gcc_multi` comment).
        cmd.args([
            "-framework", "Security",
            "-framework", "CoreFoundation",
        ]);
    }

    // Link CUDA driver + cuBLAS libraries if available (Linux: lib64/, Windows: lib/x64/).
    // cublas added 2026-04-21 for the nsl_matmul_f32 cublasSgemm swap.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(&cuda_path).join("lib64");
        let cuda_lib_win = PathBuf::from(&cuda_path).join("lib").join("x64");
        if cuda_lib.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib.display()));
            cmd.arg("-lcuda");
            cmd.arg("-lcublas");
        } else if cuda_lib_win.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib_win.display()));
            cmd.arg("-lcuda");
            cmd.arg("-lcublas");
        }
    }
    add_nccl_link_args(&mut cmd);

    // Item #5: re-export test-hooks peak FFIs so dlopen can resolve them.
    // -u forces the symbol to be pulled in from the staticlib;
    // --export-dynamic-symbol adds it to the .so/.dylib's export table.
    #[cfg(feature = "test-hooks")]
    {
        if cfg!(target_os = "linux") {
            cmd.arg("-Wl,-u,nsl_cpu_peak_reset");
            cmd.arg("-Wl,-u,nsl_cpu_peak_bytes");
            cmd.arg("-Wl,--export-dynamic-symbol=nsl_cpu_peak_reset");
            cmd.arg("-Wl,--export-dynamic-symbol=nsl_cpu_peak_bytes");
        } else if cfg!(target_os = "macos") {
            // On macOS, `-u` (or `-Wl,-u,`) forces the symbol in; staticlib
            // symbols are exported by default from a dylib unless
            // `-exported_symbols_list` is used, so no extra flag needed.
            cmd.arg("-Wl,-u,_nsl_cpu_peak_reset");
            cmd.arg("-Wl,-u,_nsl_cpu_peak_bytes");
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

/// M62 Task 6: cheap probe — does any of the given object files contain
/// the given symbol name?  Used to decide whether `/EXPORT:main` is safe
/// to pass to `link.exe` (the flag requires the symbol to resolve).
///
/// Scans raw bytes for the null-terminated ASCII symbol name.  Good enough
/// for the `main` / short-name case; we don't need to parse COFF here.
fn objs_contain_symbol(obj_paths: &[PathBuf], name: &str) -> bool {
    let mut needle = Vec::with_capacity(name.len() + 1);
    needle.extend_from_slice(name.as_bytes());
    needle.push(0);
    for p in obj_paths {
        let Ok(bytes) = std::fs::read(p) else { continue };
        if bytes.windows(needle.len()).any(|w| w == needle) {
            return true;
        }
    }
    false
}

/// M62a: Link a shared library using MSVC toolchain (link.exe /DLL).
fn link_shared_msvc(
    obj_paths: &[PathBuf],
    output_path: &Path,
    runtime_lib: &Path,
    extra_exports: &[&str],
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

    // Link CUDA driver + cuBLAS libraries if available.  cublas.lib added
    // 2026-04-21 for the nsl_matmul_f32 cublasSgemm swap.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_lib = PathBuf::from(&cuda_path).join("lib").join("x64");
        if cuda_lib.is_dir() {
            cmd.arg(format!("/LIBPATH:{}", cuda_lib.display()));
            cmd.arg("cuda.lib");
            cmd.arg("cublas.lib");
        }
    }

    // Export the program entry point so dlopen callers can invoke it.
    // On MSVC, Linkage::Export in the .obj is not enough — we must add
    // /EXPORT explicitly.  argc/argv params are passed as i32/i64 so the
    // MSVC decorated name remains plain `main`.
    //
    // M62 Task 6: shared libraries built from pure-`@export` sources have
    // no top-level `main`.  `/EXPORT:<sym>` requires the symbol to resolve,
    // so probe the objects for `main` via dumpbin /SYMBOLS before adding
    // the export.  If unavailable, we fall back to emitting it and let link
    // fail loudly (matches prior behavior for train-block builds).
    if objs_contain_symbol(obj_paths, "main") {
        cmd.arg("/EXPORT:main");
    }

    // M62 Task 6: force @export functions into the DLL export table.
    for sym in extra_exports {
        cmd.arg(format!("/EXPORT:{sym}"));
    }

    // Item #5: re-export test-hooks peak FFIs.
    // /INCLUDE forces the staticlib symbol to be linked in;
    // /EXPORT adds it to the DLL's export table.
    #[cfg(feature = "test-hooks")]
    {
        cmd.arg("/INCLUDE:nsl_cpu_peak_reset");
        cmd.arg("/INCLUDE:nsl_cpu_peak_bytes");
        cmd.arg("/EXPORT:nsl_cpu_peak_reset");
        cmd.arg("/EXPORT:nsl_cpu_peak_bytes");
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

/// P4 item 14 (nccl feature): compiled programs embed libnsl_runtime.a whose
/// NCCL symbols must resolve at the final link. NSL_NCCL_LIB_DIR points at a
/// non-system libnccl (e.g. a local package extraction); the rpath entry lets
/// the produced binary load it at run time without LD_LIBRARY_PATH.
#[cfg(feature = "nccl")]
fn add_nccl_link_args(cmd: &mut std::process::Command) {
    if let Ok(dir) = std::env::var("NSL_NCCL_LIB_DIR") {
        cmd.arg(format!("-L{dir}"));
        cmd.arg(format!("-Wl,-rpath,{dir}"));
    }
    cmd.arg("-lnccl");
}
#[cfg(not(feature = "nccl"))]
fn add_nccl_link_args(_cmd: &mut std::process::Command) {}
