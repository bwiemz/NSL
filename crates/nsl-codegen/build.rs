use std::env;
use std::path::PathBuf;

fn main() {
    // Find the nsl-runtime static library in the target directory.
    // Since nsl-codegen depends on nsl-runtime (via Cargo), it's already built.
    let out_dir = env::var("OUT_DIR").unwrap();
    // OUT_DIR is something like target/debug/build/nsl-codegen-<hash>/out
    // The static lib is at target/<profile>/nsl_runtime.lib (Windows) or libnsl_runtime.a (Unix)
    let out_path = PathBuf::from(&out_dir);
    // Walk up from OUT_DIR to find the target/<profile> directory
    // OUT_DIR = target/debug/build/nsl-codegen-HASH/out → go up 3 levels to target/debug
    let target_profile = out_path
        .parent() // nsl-codegen-HASH
        .and_then(|p| p.parent()) // build
        .and_then(|p| p.parent()) // debug (or release)
        .expect("could not find target profile directory from OUT_DIR");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let lib_name = if target_os == "windows" && target_env == "msvc" {
        "nsl_runtime.lib"
    } else {
        "libnsl_runtime.a"
    };

    let lib_path = target_profile.join(lib_name);

    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();

    // Pass the library path as an environment variable for linker.rs
    println!(
        "cargo:rustc-env=NSL_RUNTIME_LIB_PATH={}",
        lib_path.display()
    );
    println!("cargo:rustc-env=NSL_RUNTIME_RUSTFLAGS={rustflags}");
    // Rerun if the static lib changes
    println!("cargo:rerun-if-changed={}", lib_path.display());
}
