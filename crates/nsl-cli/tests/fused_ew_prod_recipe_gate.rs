//! MFU campaign C3 anti-vacuity gate — the elementwise chain fuser and the
//! RoPE backward fold must FIRE ON THE REAL PRODUCTION RECIPE, not just on
//! gate fixtures. Compile-only (`nsl build`), so it costs zero GPU and runs
//! in CI's no-cuda build.
//!
//! Floors are 0.8x the counts observed on models/coder500m/pretrain_prod.nsl
//! at implementation time (2026-08-25, prod posture `--source-ad
//! --checkpoint-blocks --fuse-rmsnorm-backward`): 72 chains (48x
//! mul+scale-imm, 24x add+add residual joins — reduce_to_shape is a chain
//! BARRIER after measurement showed every absorbed rts was a real GQA
//! reduce that replayed) and 48 rope folds. Drift below the floor means a matcher regression or a
//! recipe/stdlib change that silently starved the fuser — either way the
//! perf the campaign measured is gone and this gate is the tell.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn marker_count(stderr: &str, marker: &str) -> Option<u64> {
    stderr.lines().find_map(|l| {
        l.strip_prefix(marker)
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|n| n.parse().ok())
    })
}

#[test]
fn prod_500m_recipe_fuses_above_the_observed_floor() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_ewprod_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["build", "--source-ad", "--checkpoint-blocks", "--fuse-rmsnorm-backward"])
        .arg(root.join("models/coder500m/pretrain_prod.nsl"))
        .args(["--emit-obj", "-o"])
        .arg(tmp.join("prod500m.o"))
        .current_dir(root.join("models/coder500m"))
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl build");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    // Remove the temp dir BEFORE asserting — a panic must not leak objects
    // onto the 31G tmpfs.
    let _ = std::fs::remove_dir_all(&tmp);

    assert!(
        out.status.success(),
        "prod recipe must build in the prod posture:\n{stderr}"
    );

    let chains = marker_count(&stderr, "[fuse] elementwise backward chains:")
        .unwrap_or_else(|| {
            panic!(
                "no `[fuse] elementwise backward chains:` marker — the fuser \
                 did not run on the prod recipe (vacuous):\n{stderr}"
            )
        });
    assert!(
        chains >= 57,
        "chain count {chains} fell below the floor (57 = 0.8 x 72 observed \
         2026-08-25, post rts-barrier) — matcher regression or recipe drift \
         starved the fuser:\n{stderr}"
    );

    let rope = marker_count(&stderr, "[fuse] rope backward folds:").unwrap_or_else(|| {
        panic!(
            "no `[fuse] rope backward folds:` marker — the RoPE fold did not \
             fire on the prod recipe (vacuous):\n{stderr}"
        )
    });
    assert!(
        rope >= 40,
        "rope fold count {rope} fell below the floor (40 = 0.8 x 48 observed \
         2026-08-25):\n{stderr}"
    );
}
