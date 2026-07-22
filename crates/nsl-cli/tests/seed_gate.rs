//! P0 certification: `--seed` must produce reproducible AND distinct
//! model inits. Historically `nsl_rng_seed` stored a seed the sampling
//! RNG never read, so every "seeded" run drew the SAME randn stream.

use std::process::Command;

fn run_seed(seed: u64) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let tmp = std::env::temp_dir().join(format!("nsl_seed_gate_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, "let w = randn([8])\nprint(sum(w * w).item())\n").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic", "--seed", &seed.to_string()])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(out.status.success(), "run failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .last()
        .unwrap_or_default()
        .trim()
        .to_string()
}

#[test]
fn seed_is_reproducible_and_distinct() {
    let a1 = run_seed(1);
    let a2 = run_seed(1);
    let b = run_seed(2);
    assert_eq!(a1, a2, "same seed must reproduce bit-identically");
    assert_ne!(a1, b, "different seeds must produce different inits");
    assert!(a1.parse::<f64>().is_ok(), "not a number: {a1}");
}
