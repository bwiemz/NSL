//! Roadmap item 4's acceptance criterion, machine-checked: "no machine
//! paths / synthetic data; config and scheduler agree".
//!
//! `models/coder50m/config.nsl` is pure documentation — nothing imports it
//! (the train-config contract requires literals in the header, so the
//! script cannot read the consts) — which is exactly how its pretrain
//! block drifted into fiction (batch 32, total_steps 305000, a data path
//! this repo never contained) while pretrain.nsl hardcoded a Windows
//! machine path. This gate parses BOTH committed files and refuses drift,
//! and re-derives the scheduler length from the corpus arithmetic so the
//! numbers cannot merely agree on a shared fiction.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// `const NAME = <value>` from config.nsl.
fn config_const(config: &str, name: &str) -> f64 {
    let needle = format!("const {name} = ");
    let line = config
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("config.nsl is missing `const {name}`"));
    line.trim_start()
        .strip_prefix(&needle)
        .unwrap()
        .trim()
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("config.nsl `{name}` is not numeric: {e}"))
}

/// `key=<numeric>` from a pretrain_prod.nsl line (train header, optimizer
/// or scheduler call — kwarg spellings are unique across the file).
fn prod_kwarg(prod: &str, key: &str) -> f64 {
    let needle = format!("{key}=");
    let mut vals = Vec::new();
    for line in prod.lines() {
        // Skip comment lines — the derivation block quotes the numbers.
        if line.trim_start().starts_with('#') {
            continue;
        }
        let mut rest = line;
        while let Some(pos) = rest.find(&needle) {
            // Guard against key-suffix collisions (e.g. `lr=` inside
            // `min_lr=` or `adamw_lr=`): require a non-identifier char
            // before the match.
            let boundary_ok = pos == 0
                || !rest[..pos]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_');
            let after = &rest[pos + needle.len()..];
            if boundary_ok {
                let num: String = after
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
                    .collect();
                if let Ok(v) = num.parse::<f64>() {
                    vals.push(v);
                }
            }
            rest = after;
        }
    }
    // Some keys legitimately appear at more than one site (batch_size and
    // seq_len are both DataLoader args and @fused_lm_ce hints) — every
    // occurrence must agree, which is itself part of the contract this
    // gate pins.
    assert!(
        !vals.is_empty(),
        "no non-comment `{key}=<number>` found in pretrain_prod.nsl"
    );
    assert!(
        vals.iter().all(|v| (v - vals[0]).abs() < 1e-12),
        "`{key}=` appears with DISAGREEING values in pretrain_prod.nsl: {vals:?}"
    );
    vals[0]
}

#[test]
fn pretrain_prod_agrees_with_config_and_the_corpus_arithmetic() {
    let root = repo_root();
    let config =
        std::fs::read_to_string(root.join("models/coder50m/config.nsl")).expect("config.nsl");
    let prod = std::fs::read_to_string(root.join("models/coder50m/pretrain_prod.nsl"))
        .expect("pretrain_prod.nsl");

    // ── config.nsl ↔ pretrain_prod.nsl, key by key ─────────────────────
    for (const_name, prod_key) in [
        ("PRETRAIN_LR", "lr"),
        ("PRETRAIN_WARMUP", "warmup_steps"),
        ("PRETRAIN_TOTAL_STEPS", "total_steps"),
        ("PRETRAIN_MIN_LR", "min_lr"),
        ("PRETRAIN_BATCH_SIZE", "batch_size"),
        ("PRETRAIN_GRAD_ACCUMULATION", "grad_accumulation"),
        ("PRETRAIN_WEIGHT_DECAY", "weight_decay"),
        ("PRETRAIN_BETA1", "beta1"),
        ("PRETRAIN_BETA2", "beta2"),
        ("PRETRAIN_EPOCHS", "epochs"),
        ("PRETRAIN_GRAD_CLIP", "grad_clip"),
        ("PRETRAIN_CHECKPOINT_EVERY", "checkpoint_every"),
    ] {
        let c = config_const(&config, const_name);
        let p = prod_kwarg(&prod, prod_key);
        assert!(
            (c - p).abs() < 1e-12,
            "config.nsl {const_name}={c} disagrees with pretrain_prod.nsl {prod_key}={p}"
        );
    }

    // ── the scheduler length is DERIVED, not asserted ──────────────────
    // total_steps = epochs * floor(CORPUS_TOKENS / seq_len), in
    // micro-batches (batch_size=1; the scheduler counts micro-batches).
    let corpus_tokens = config_const(&config, "CORPUS_TOKENS");
    let seq_len = config_const(&config, "MAX_SEQ_LEN");
    let epochs = config_const(&config, "PRETRAIN_EPOCHS");
    let batch = config_const(&config, "PRETRAIN_BATCH_SIZE");
    let derived = epochs * (corpus_tokens / (seq_len * batch)).floor();
    let total = config_const(&config, "PRETRAIN_TOTAL_STEPS");
    assert!(
        (derived - total).abs() < 1e-9,
        "PRETRAIN_TOTAL_STEPS={total} but epochs*floor(CORPUS_TOKENS/(seq*batch))={derived} — \
         the scheduler no longer matches the corpus it schedules over"
    );

    // Warmup must fit inside the schedule with room to decay.
    let warmup = config_const(&config, "PRETRAIN_WARMUP");
    assert!(
        warmup < total / 2.0,
        "PRETRAIN_WARMUP={warmup} is not a warmup for a {total}-step schedule"
    );

    // ── no machine paths ───────────────────────────────────────────────
    // Positive check first (review finding: a denylist alone lets a
    // second load site under /mnt or D:/ slip through): EVERY load_mmap
    // in the file must use the one repo-anchored corpus path.
    let load_sites: Vec<&str> = prod
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter(|l| l.contains("load_mmap("))
        .collect();
    assert!(
        !load_sites.is_empty(),
        "pretrain_prod.nsl has no load_mmap site — the corpus is not loaded"
    );
    for site in &load_sites {
        assert!(
            site.contains("\"../../data/tokens/train_new.bin\""),
            "load_mmap site does not use the repo-anchored corpus path: {site}"
        );
    }
    // config.nsl's documented path must be the same one (review finding:
    // PRETRAIN_DATA was previously unchecked and had drifted to a
    // directory that never existed).
    assert!(
        config.contains("const PRETRAIN_DATA = \"../../data/tokens/train_new.bin\""),
        "config.nsl PRETRAIN_DATA disagrees with the corpus pretrain_prod.nsl loads"
    );
    // Denylist as belt-and-braces, widened past the original five roots.
    for (file, text) in [("pretrain_prod.nsl", &prod), ("config.nsl", &config)] {
        for bad in [
            "C:/", "C:\\", "D:/", "D:\\", "/home/", "/Users/", "/mnt/", "/media/", "/tmp/",
            "bwiem",
        ] {
            assert!(
                !text.contains(bad),
                "{file} contains a machine-specific path fragment {bad:?}"
            );
        }
    }

    // ── the fused-CE hints match the architecture consts ───────────────
    let vocab = config_const(&config, "VOCAB_SIZE");
    let d_model = config_const(&config, "D_MODEL");
    assert!((prod_kwarg(&prod, "vocab_size") - vocab).abs() < 1e-12);
    assert!((prod_kwarg(&prod, "hidden_size") - d_model).abs() < 1e-12);
    assert!((prod_kwarg(&prod, "seq_len") - seq_len).abs() < 1e-12);

    // ── the file COMPILES under the production AD mode, and the tape
    //    limitation is pinned, not assumed ─────────────────────────────
    // Source AD is the production path (validated end-to-end on GPU).
    // Tape AD does not yet support DataLoader-driven train blocks: it
    // refuses with "must assign to a variable named 'loss'" (the tape
    // lowering never finds the binding when a loader drives the step).
    // Item 5 (tape-vs-source differential) needs that fixed; this pin
    // flips CONSCIOUSLY when it is. Both are codegen-time outcomes, so a
    // no-cuda build exercises them.
    let tmp = std::env::temp_dir().join(format!("nsl_prodgate_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let run_build = |extra: &[&str]| {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
            .arg("build")
            .args(extra)
            .arg(root.join("models/coder50m/pretrain_prod.nsl"))
            .arg("-o")
            .arg(tmp.join("prod_gate.o"))
            .current_dir(root.join("models/coder50m"))
            .env("NSL_STDLIB_PATH", root.join("stdlib"))
            .output()
            .expect("spawn nsl build");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stderr).to_string(),
        )
    };
    // Collect BOTH outcomes, remove the temp dir, THEN assert — a panic
    // must not leak a 125 MB prod_gate.o onto the 31G tmpfs (review
    // finding: the tape pin below is DESIGNED to fire when item 5's fix
    // lands, which is exactly when the leak would start).
    let (srcad_ok, srcad_stderr) = run_build(&["--source-ad"]);
    let (tape_ok, tape_stderr) = run_build(&[]);
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
        srcad_ok,
        "pretrain_prod.nsl must build under --source-ad (the production \
         mode):\n{srcad_stderr}"
    );
    assert!(
        !tape_ok && tape_stderr.contains("must assign to a variable named 'loss'"),
        "the tape x DataLoader limitation moved — if tape AD now supports \
         loader-driven train blocks, item 5 is unblocked: update this pin \
         and the pretrain_prod.nsl header together.\n{tape_stderr}"
    );

    // ── CORPUS_TOKENS is real, when the local corpus is present ────────
    // The corpus is local data (not committed); when this machine has it,
    // the recorded token count must be the file's actual u16 count so the
    // derivation above cannot agree on a stale number.
    let corpus = root.join("data/tokens/train_new.bin");
    if corpus.exists() {
        let bytes = std::fs::metadata(&corpus).expect("corpus metadata").len();
        assert_eq!(
            bytes / 2,
            corpus_tokens as u64,
            "CORPUS_TOKENS disagrees with the on-disk corpus ({} u16 tokens)",
            bytes / 2
        );
    } else {
        eprintln!("[gate] data/tokens/train_new.bin absent — corpus-count arm skipped");
    }
}
