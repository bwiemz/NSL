//! Public-API smoke for `nsl_codegen::ptx_metadata`.
//!
//! The parser + report formatter must be usable from *outside* the crate —
//! the `nsl ptx-metadata` CLI and any host tooling depend on this surface
//! staying `pub`. Pure text in / structs out, so it runs on every `cargo
//! test` with no GPU or CUDA toolkit.

use nsl_codegen::ptx_metadata::{extract_ptx_metadata, format_ptx_metadata_report};

const PTX: &str = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry demo_kernel(
    .param .u64 p
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<48>;
    .reg .f32 %f<16>;
    .shared .align 4 .b8 smem[1024];
    ld.param.u64 %rd1, [p];
    ret;
}
"#;

#[test]
fn public_api_parses_and_reports() {
    let kernels = extract_ptx_metadata(PTX.as_bytes());
    assert_eq!(kernels.len(), 1, "should find exactly one .entry kernel");

    let k = &kernels[0];
    assert_eq!(k.name, "demo_kernel");
    assert_eq!(k.target_sm, 80);
    assert_eq!(k.registers.u32_regs, 48);
    assert_eq!(k.registers.f32_regs, 16);
    assert_eq!(k.registers.pred_regs, 2);
    assert_eq!(k.shared_mem_bytes, 1024);
    assert!(!k.has_dynamic_shared);

    let report = format_ptx_metadata_report(&kernels);
    assert!(report.contains("demo_kernel"), "report names the kernel");
    assert!(report.contains("sm_80"), "report shows the target SM");
    assert!(report.contains("1024 bytes"), "report shows shared memory");
}
