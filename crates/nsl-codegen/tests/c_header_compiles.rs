//! ABI hardening: compile the generated C header with a real C compiler and
//! assert the `NslTensorDesc` layout from the C side via `_Static_assert`.
//!
//! This guards two things the Rust-side golden test in `nsl-runtime` cannot:
//!   1. `c_header::emit` produces *valid, compilable C* (catches syntax/type
//!      regressions in the header generator).
//!   2. The header's `NslTensorDesc` layout (size + field offsets) matches the
//!      runtime's `#[repr(C)]` struct — i.e. the generated header and the
//!      runtime agree on the ABI.
//!
//! Best-effort: if no C compiler is found (e.g. a Windows runner where `cc`
//! is absent), the test prints a skip note and returns. On Linux/macOS CI the
//! workspace already requires a C toolchain (the runtime links via `cc`), so
//! this runs there.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

/// Locate a C compiler: `$CC` first, then `cc`/`gcc`/`clang` on `PATH`.
fn find_cc() -> Option<String> {
    if let Ok(cc) = std::env::var("CC") {
        if !cc.trim().is_empty() {
            return Some(cc);
        }
    }
    for candidate in ["cc", "gcc", "clang"] {
        // `<cc> --version` succeeds iff the compiler is present and runnable.
        if Command::new(candidate)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Some(candidate.to_string());
        }
    }
    None
}

#[test]
fn generated_c_header_compiles_and_pins_abi_layout() {
    let Some(cc) = find_cc() else {
        eprintln!("skip: no C compiler (cc/gcc/clang) found — ABI header compile check not run");
        return;
    };

    // 1. Generate the header for a module with no @export functions: this still
    //    emits the core ABI surface (NslTensorDesc, NSL_ABI_VERSION_*, the
    //    nsl_model_* / nsl_abi_version prototypes).
    let header = nsl_codegen::c_header::emit(&[], "abi_check");

    // 2. Lay the header + a C translation unit that exercises it down on disk.
    let dir: PathBuf = std::env::temp_dir().join(format!("nsl_abi_hdr_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    std::fs::write(dir.join("abi_check.h"), &header).expect("write header");

    // The C side re-derives the layout independently of Rust and asserts it.
    // If `c_header::emit` ever drifts from the runtime's NslTensorDesc, or emits
    // invalid C, this translation unit fails to compile.
    let check_c = r#"
#include "abi_check.h"
#include <stddef.h>

/* Layout must match nsl-runtime's #[repr(C)] NslTensorDesc (48 bytes). */
_Static_assert(sizeof(NslTensorDesc) == 48, "NslTensorDesc must be 48 bytes");
_Static_assert(offsetof(NslTensorDesc, data)        == 0,  "data offset");
_Static_assert(offsetof(NslTensorDesc, shape)       == 8,  "shape offset");
_Static_assert(offsetof(NslTensorDesc, strides)     == 16, "strides offset");
_Static_assert(offsetof(NslTensorDesc, ndim)        == 24, "ndim offset");
_Static_assert(offsetof(NslTensorDesc, dtype)       == 28, "dtype offset");
_Static_assert(offsetof(NslTensorDesc, device_type) == 32, "device_type offset");
_Static_assert(offsetof(NslTensorDesc, device_id)   == 36, "device_id offset");
_Static_assert(offsetof(NslTensorDesc, tape_id)     == 40, "tape_id offset");

/* The ABI version macros must be present and well-formed. */
_Static_assert(NSL_ABI_VERSION_MAJOR >= 1, "ABI major version present");
_Static_assert(NSL_ABI_VERSION == (((int64_t)NSL_ABI_VERSION_MAJOR << 16) | NSL_ABI_VERSION_MINOR),
               "packed ABI version macro is consistent");

/* Reference a prototype so the declarations are type-checked, not just parsed. */
int64_t (*nsl_abi_version_ptr)(void) = &nsl_abi_version;
"#;
    let c_path = dir.join("abi_check.c");
    let mut f = std::fs::File::create(&c_path).expect("write C check");
    f.write_all(check_c.as_bytes()).expect("write C check body");
    drop(f);

    // 3. Syntax/semantics-only compile (no link needed) in C11 for _Static_assert.
    let out = Command::new(&cc)
        .arg("-std=c11")
        .arg("-fsyntax-only")
        .arg("-Wall")
        .arg("-Werror")
        .arg(format!("-I{}", dir.display()))
        .arg(&c_path)
        .output()
        .expect("invoke C compiler");

    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        out.status.success(),
        "generated C header failed to compile / ABI layout mismatch:\n--- cc stderr ---\n{}\n--- header ---\n{}",
        String::from_utf8_lossy(&out.stderr),
        header,
    );
}

/// The generated C header pins the ABI version it was built against via the
/// `NSL_ABI_VERSION_MAJOR` / `_MINOR` macros. This is a **golden** check: it
/// compares the emitted macros against hard-coded expected values, not against
/// the runtime constants. Comparing the parsed macro to the runtime constant
/// would be tautological — `c_header::emit` writes the macro *from* that same
/// constant, so the two can never disagree.
///
/// Anchoring to a literal instead means an ABI bump is a deliberate two-step:
/// change the runtime constants **and** these goldens in the same PR. A bump of
/// one without the other turns this test red. Pure text comparison (no C
/// compiler), so it runs everywhere including Windows runners without a
/// toolchain.
#[test]
fn c_header_abi_version_matches_golden() {
    // Bump these together with `nsl_runtime::c_api::NSL_ABI_VERSION_{MAJOR,MINOR}`
    // whenever the C ABI changes.
    const EXPECTED_ABI_MAJOR: u32 = 1;
    const EXPECTED_ABI_MINOR: u32 = 0;

    // Anchor the runtime constants to the goldens (catches an undocumented bump
    // of the constants without updating this test / the ABI policy).
    assert_eq!(
        nsl_runtime::c_api::NSL_ABI_VERSION_MAJOR, EXPECTED_ABI_MAJOR,
        "runtime NSL_ABI_VERSION_MAJOR changed; update this golden and confirm the bump is intentional",
    );
    assert_eq!(
        nsl_runtime::c_api::NSL_ABI_VERSION_MINOR, EXPECTED_ABI_MINOR,
        "runtime NSL_ABI_VERSION_MINOR changed; update this golden and confirm the bump is intentional",
    );

    let header = nsl_codegen::c_header::emit(&[], "version_check");
    let macro_u32 = |name: &str| -> u32 {
        let needle = format!("#define {name} ");
        let start = header
            .find(&needle)
            .unwrap_or_else(|| panic!("generated header is missing `{name}`:\n{header}"));
        let rest = &header[start + needle.len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits
            .parse()
            .unwrap_or_else(|_| panic!("`{name}` macro value is not a u32: {rest:?}"))
    };

    // And anchor the *emitted header* to the goldens (catches a header generator
    // that hard-codes a value diverging from the runtime, independent of the
    // constants above).
    assert_eq!(
        macro_u32("NSL_ABI_VERSION_MAJOR"), EXPECTED_ABI_MAJOR,
        "generated header NSL_ABI_VERSION_MAJOR diverges from the expected ABI version",
    );
    assert_eq!(
        macro_u32("NSL_ABI_VERSION_MINOR"), EXPECTED_ABI_MINOR,
        "generated header NSL_ABI_VERSION_MINOR diverges from the expected ABI version",
    );
}
