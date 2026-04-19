//! Diagnostic: measure PTX size as a function of k and n to confirm the
//! hypothesis that the fused emitter unrolls the K-loop at generation time,
//! producing unbounded PTX at realistic scales.

#[test]
fn gatedlora_ptx_size_sweep() {
    let rank = 16;
    let target_sm = 80;

    eprintln!("shape         | ptx_bytes  | ptx_lines");
    eprintln!("--------------+------------+----------");
    for &(n, k) in &[
        (64u64, 64u64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ] {
        let cfg = nsl_codegen::wrga_fused_ptx::FusedGatedLoraConfig {
            site_id: "diag".into(),
            m: 1,
            n: n as u32,
            k: k as u32,
            rank,
            target_sm,
        };
        let ptx = nsl_codegen::wrga_fused_ptx::synthesize_fused_gatedlora_ptx(&cfg);
        eprintln!(
            "n={:5} k={:5} | {:10} | {:9}",
            n,
            k,
            ptx.len(),
            ptx.lines().count()
        );
    }
}
