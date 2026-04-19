//! Diagnostic: dump fused GatedLoRA PTX at the prescribed B.3.2-trigger shape
//! so we can feed it to `ptxas -v` and diagnose CUDA_ERROR_INVALID_PTX.
//!
//! Invoke with:
//!   cargo test --test dump_gatedlora_ptx dump_prescribed_shape_ptx -- --nocapture
//!
//! Writes the PTX to `target/gatedlora_prescribed.ptx`.

use std::fs;
use std::path::PathBuf;

#[test]
fn dump_prescribed_shape_ptx() {
    let cfg = nsl_codegen::wrga_fused_ptx::FusedGatedLoraConfig {
        site_id: "diag".into(),
        m: 1,
        n: 4096,
        k: 4096,
        rank: 16,
        target_sm: 80,
    };
    let ptx = nsl_codegen::wrga_fused_ptx::synthesize_fused_gatedlora_ptx(&cfg);
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target")
        .join("gatedlora_prescribed.ptx");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&out, &ptx).expect("write ptx");
    eprintln!("PTX length: {} bytes", ptx.len());
    eprintln!("PTX line count: {}", ptx.lines().count());
    eprintln!("Wrote: {}", out.display());
}
