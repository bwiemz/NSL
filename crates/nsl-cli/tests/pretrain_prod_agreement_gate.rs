//! Roadmap items 4 and 9's acceptance criterion, machine-checked: "no machine
//! paths / synthetic data; config and scheduler agree" — at BOTH model sizes.
//!
//! `models/coder*/config.nsl` is pure documentation — nothing imports it (the
//! train-config contract requires literals in the header, so the script cannot
//! read the consts) — which is exactly how both pretrain blocks drifted into
//! fiction. At 50M: batch 32, total_steps 305000, a data path this repo never
//! contained, plus a Windows machine path in pretrain.nsl. At 500M the SAME
//! 305000/3000 pair survived item 4 untouched, because that campaign only
//! rewrote the 50M pair. This gate parses BOTH committed pairs and refuses
//! drift, and re-derives each scheduler length from the corpus arithmetic so
//! the numbers cannot merely agree on a shared fiction.
//!
//! The two recipes differ in ONE deliberate way, encoded in `train_tokens_const`
//! below: 50M (item 4) schedules over the whole corpus, while 500M (item 9)
//! trains on a prefix slice and reports cross-entropy on a held-out tail. At
//! 0.017 tokens/param a training loss alone cannot distinguish an under-trained
//! model from a memorizing one, which is the question a 500M run on this corpus
//! actually raises.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// One model size's production pair.
struct ProdPair {
    /// Directory under `models/`, also the label in assertion messages.
    dir: &'static str,
    /// Which const carries the token count the SCHEDULE is derived from.
    /// 50M schedules over the whole corpus; 500M over its train slice.
    train_tokens_const: &'static str,
    /// Every `load_mmap` path the recipe is allowed to use, and the file each
    /// one must exist as (relative to the repo root) when the local data is
    /// present. A positive allowlist, not a denylist: the original gate's
    /// denylist alone would let a second load site under `/mnt` slip through.
    load_sites: &'static [(&'static str, &'static str)],
    /// `(config const, expected u16 token count)` for each local data file
    /// above, so a recorded count cannot drift from the bytes on disk.
    token_counts: &'static [(&'static str, &'static str)],
    /// Whether the training loader shuffles. `None` for a model whose config
    /// does not record the choice (50M predates the const). Load-bearing at
    /// 500M: read sequentially, this concatenated corpus makes the training
    /// loss track corpus POSITION (r = 0.915 against per-region unigram
    /// entropy) rather than learning, and a silent flip back to sequential
    /// would restore that without any other symptom.
    shuffle: Option<bool>,
}

const PAIRS: &[ProdPair] = &[
    ProdPair {
        dir: "coder50m",
        train_tokens_const: "CORPUS_TOKENS",
        load_sites: &[(
            "../../data/tokens/train_new.bin",
            "data/tokens/train_new.bin",
        )],
        token_counts: &[("CORPUS_TOKENS", "data/tokens/train_new.bin")],
        shuffle: None,
    },
    ProdPair {
        dir: "coder500m",
        train_tokens_const: "TRAIN_TOKENS",
        load_sites: &[
            (
                "../../data/tokens/prod_train_slice.bin",
                "data/tokens/prod_train_slice.bin",
            ),
            (
                "../../data/tokens/prod_val_slice.bin",
                "data/tokens/prod_val_slice.bin",
            ),
        ],
        token_counts: &[
            ("TRAIN_TOKENS", "data/tokens/prod_train_slice.bin"),
            ("VAL_TOKENS", "data/tokens/prod_val_slice.bin"),
        ],
        shuffle: Some(true),
    },
];

/// `const NAME = <value>` from config.nsl.
fn config_const(config: &str, name: &str, dir: &str) -> f64 {
    let needle = format!("const {name} = ");
    let line = config
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("{dir}/config.nsl is missing `const {name}`"));
    line.trim_start()
        .strip_prefix(&needle)
        .unwrap()
        .trim()
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("{dir}/config.nsl `{name}` is not numeric: {e}"))
}

/// `key=<numeric>` from a pretrain_prod.nsl line (train header, optimizer or
/// scheduler call — kwarg spellings are unique across the file).
fn prod_kwarg(prod: &str, key: &str, dir: &str) -> f64 {
    let needle = format!("{key}=");
    let mut vals = Vec::new();
    for line in prod.lines() {
        // Skip comment lines — the derivation block quotes the numbers.
        if line.trim_start().starts_with('#') {
            continue;
        }
        let mut rest = line;
        while let Some(pos) = rest.find(&needle) {
            // Guard against key-suffix collisions (e.g. `lr=` inside `min_lr=`
            // or `adamw_lr=`): require a non-identifier char before the match.
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
    // occurrence must agree, which is itself part of the contract this gate
    // pins.
    assert!(
        !vals.is_empty(),
        "{dir}: no non-comment `{key}=<number>` found in pretrain_prod.nsl"
    );
    assert!(
        vals.iter().all(|v| (v - vals[0]).abs() < 1e-12),
        "{dir}: `{key}=` appears with DISAGREEING values in pretrain_prod.nsl: {vals:?}"
    );
    vals[0]
}

#[test]
fn pretrain_prod_agrees_with_config_and_the_corpus_arithmetic() {
    let root = repo_root();

    for pair in PAIRS {
        let dir = pair.dir;
        let config = std::fs::read_to_string(root.join(format!("models/{dir}/config.nsl")))
            .unwrap_or_else(|e| panic!("{dir}/config.nsl: {e}"));
        let prod = std::fs::read_to_string(root.join(format!("models/{dir}/pretrain_prod.nsl")))
            .unwrap_or_else(|e| panic!("{dir}/pretrain_prod.nsl: {e}"));

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
            let c = config_const(&config, const_name, dir);
            let p = prod_kwarg(&prod, prod_key, dir);
            assert!(
                (c - p).abs() < 1e-12,
                "{dir}: config.nsl {const_name}={c} disagrees with \
                 pretrain_prod.nsl {prod_key}={p}"
            );
        }

        // ── the scheduler length is DERIVED, not asserted ──────────────────
        // total_steps = epochs * floor(train_tokens / (seq_len * batch_size)),
        // in micro-batches (the scheduler counts micro-batches).
        let train_tokens = config_const(&config, pair.train_tokens_const, dir);
        let seq_len = config_const(&config, "MAX_SEQ_LEN", dir);
        let epochs = config_const(&config, "PRETRAIN_EPOCHS", dir);
        let batch = config_const(&config, "PRETRAIN_BATCH_SIZE", dir);
        let derived = epochs * (train_tokens / (seq_len * batch)).floor();
        let total = config_const(&config, "PRETRAIN_TOTAL_STEPS", dir);
        assert!(
            (derived - total).abs() < 1e-9,
            "{dir}: PRETRAIN_TOTAL_STEPS={total} but epochs*floor({}/(seq*batch))\
             ={derived} — the scheduler no longer matches the corpus it \
             schedules over",
            pair.train_tokens_const
        );

        // Warmup must fit inside the schedule with room to decay.
        let warmup = config_const(&config, "PRETRAIN_WARMUP", dir);
        assert!(
            warmup < total / 2.0,
            "{dir}: PRETRAIN_WARMUP={warmup} is not a warmup for a \
             {total}-step schedule"
        );

        // ── no machine paths ───────────────────────────────────────────────
        // Positive check first: EVERY load_mmap in the file must use one of
        // this model's declared, repo-anchored corpus paths.
        let load_lines: Vec<&str> = prod
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .filter(|l| l.contains("load_mmap("))
            .collect();
        assert_eq!(
            load_lines.len(),
            pair.load_sites.len(),
            "{dir}: expected {} load_mmap site(s), found {}: {load_lines:?}",
            pair.load_sites.len(),
            load_lines.len()
        );
        for line in &load_lines {
            assert!(
                pair.load_sites
                    .iter()
                    .any(|(p, _)| line.contains(&format!("\"{p}\""))),
                "{dir}: load_mmap site is not one of the declared corpus \
                 paths: {line}"
            );
        }
        // config.nsl's documented path must be the one the recipe loads (this
        // const was previously unchecked at 50M and had drifted to a
        // directory that never existed).
        let documented = pair.load_sites[0].0;
        assert!(
            config.contains(&format!("const PRETRAIN_DATA = \"{documented}\"")),
            "{dir}: config.nsl PRETRAIN_DATA disagrees with the corpus \
             pretrain_prod.nsl loads ({documented})"
        );
        // Denylist as belt-and-braces.
        for (file, text) in [("pretrain_prod.nsl", &prod), ("config.nsl", &config)] {
            for bad in [
                "C:/", "C:\\", "D:/", "D:\\", "/home/", "/Users/", "/mnt/", "/media/", "/tmp/",
                "bwiem",
            ] {
                assert!(
                    !text.contains(bad),
                    "{dir}/{file} contains a machine-specific path fragment {bad:?}"
                );
            }
        }

        // ── the shuffle choice, where the config records it ────────────────
        if let Some(want) = pair.shuffle {
            let want_str = if want { "shuffle=true" } else { "shuffle=false" };
            let train_loader = prod
                .lines()
                .filter(|l| !l.trim_start().starts_with('#'))
                .find(|l| l.contains("DataLoader(") && l.contains("tokens,"))
                .unwrap_or_else(|| panic!("{dir}: no training DataLoader line"));
            assert!(
                train_loader.contains(want_str),
                "{dir}: the training DataLoader must use {want_str} — see \
                 config.nsl PRETRAIN_SHUFFLE. Line: {train_loader}"
            );
            assert!(
                (config_const(&config, "PRETRAIN_SHUFFLE", dir) != 0.0) == want,
                "{dir}: config.nsl PRETRAIN_SHUFFLE disagrees with the recipe"
            );
        }

        // ── the fused-CE hints match the architecture consts ───────────────
        let vocab = config_const(&config, "VOCAB_SIZE", dir);
        let d_model = config_const(&config, "D_MODEL", dir);
        assert!((prod_kwarg(&prod, "vocab_size", dir) - vocab).abs() < 1e-12);
        assert!((prod_kwarg(&prod, "hidden_size", dir) - d_model).abs() < 1e-12);
        assert!((prod_kwarg(&prod, "seq_len", dir) - seq_len).abs() < 1e-12);

        // ── recorded token counts are real, when the local data is present ─
        // Not committed; when this machine has it, each recorded count must be
        // the file's actual u16 count so the derivation above cannot agree on
        // a stale number.
        for (const_name, rel) in pair.token_counts {
            let path = root.join(rel);
            if !path.exists() {
                eprintln!("[gate] {rel} absent — {const_name} count arm skipped");
                continue;
            }
            let bytes = std::fs::metadata(&path).expect("token file metadata").len();
            assert_eq!(
                bytes / 2,
                config_const(&config, const_name, dir) as u64,
                "{dir}: {const_name} disagrees with the on-disk {rel} ({} u16 \
                 tokens)",
                bytes / 2
            );
        }
    }
}

/// Item 9: the held-out split must actually be held out.
///
/// The train slice is a PREFIX and the validation slice is the TAIL of one
/// corpus, so "held out" is arithmetic, not a naming convention — and the
/// arithmetic lives in two places (config.nsl and make_prod_split.py) that can
/// drift apart. A split whose halves overlap would report a validation loss
/// measured on training data, which is worse than reporting none.
#[test]
fn the_held_out_split_does_not_overlap_the_training_slice() {
    let root = repo_root();
    let config = std::fs::read_to_string(root.join("models/coder500m/config.nsl")).unwrap();
    let corpus = config_const(&config, "CORPUS_TOKENS", "coder500m");
    let train = config_const(&config, "TRAIN_TOKENS", "coder500m");
    let val = config_const(&config, "VAL_TOKENS", "coder500m");
    let seq = config_const(&config, "MAX_SEQ_LEN", "coder500m");

    assert!(
        train + val < corpus,
        "TRAIN_TOKENS({train}) + VAL_TOKENS({val}) >= CORPUS_TOKENS({corpus}) — \
         the validation tail would overlap the training prefix"
    );
    // A strictly positive gap is not enough: DataLoader(drop_last=true)
    // consumes whole seq_len windows, so a sub-window gap still lets the last
    // training window read into the first validation window.
    let gap = corpus - train - val;
    assert!(
        gap >= seq,
        "the unused gap between the training prefix and the held-out tail is \
         {gap} tokens, under one {seq}-token window — a training window could \
         straddle the boundary and leak a suffix the val loss then scores on"
    );
    // Both slices must be whole windows, or drop_last silently discards a
    // partial batch and the derived step count is wrong.
    assert_eq!(train % seq, 0.0, "TRAIN_TOKENS is not a whole number of windows");
    assert_eq!(val % seq, 0.0, "VAL_TOKENS is not a whole number of windows");

    // The materializer's constants must match config.nsl — it is what writes
    // the files the recipe reads.
    let script = std::fs::read_to_string(root.join("models/benchmarks/make_prod_split.py"))
        .expect("make_prod_split.py");
    for (name, want) in [("TRAIN_TOKENS", train), ("VAL_TOKENS", val)] {
        let line = script
            .lines()
            .find(|l| l.starts_with(&format!("{name} = ")))
            .unwrap_or_else(|| panic!("make_prod_split.py is missing `{name} = `"));
        let got: f64 = line
            .split('=')
            .nth(1)
            .unwrap()
            .split('#')
            .next()
            .unwrap()
            .trim()
            .replace('_', "")
            .parse()
            .expect("numeric");
        assert_eq!(
            got, want,
            "make_prod_split.py {name}={got} disagrees with coder500m/config.nsl {want} — \
             the script writes the files the recipe reads"
        );
    }
}

/// Both recipes must COMPILE under BOTH AD modes.
///
/// Source AD is the production path (validated end-to-end on GPU). Tape AD
/// gained DataLoader-driven train blocks in the item-5 campaign (the loader
/// path's packing-registry stash used to leave `state.current_block` on a
/// terminated block, so `compile_stmt` silently skipped the whole step body
/// and the tape lowering refused with "must assign to a variable named
/// 'loss'"). Both arms are codegen-time outcomes, so a no-cuda build exercises
/// them; the runtime leg lives in tape_dataloader_train_gate.rs.
#[test]
fn pretrain_prod_builds_under_both_ad_modes() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_prodgate_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let mut outcomes: Vec<(&str, &str, bool, String)> = Vec::new();
    for pair in PAIRS {
        for (label, extra) in [("--source-ad", vec!["--source-ad"]), ("tape", vec![])] {
            let out = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
                .arg("build")
                .args(&extra)
                .arg(root.join(format!("models/{}/pretrain_prod.nsl", pair.dir)))
                .arg("-o")
                .arg(tmp.join(format!("{}_{}.o", pair.dir, label.trim_start_matches('-'))))
                .current_dir(root.join(format!("models/{}", pair.dir)))
                .env("NSL_STDLIB_PATH", root.join("stdlib"))
                .output()
                .expect("spawn nsl build");
            outcomes.push((
                pair.dir,
                label,
                out.status.success(),
                String::from_utf8_lossy(&out.stderr).to_string(),
            ));
        }
    }
    // Remove the temp dir BEFORE asserting — a panic must not leak several
    // hundred MB of objects onto the 31G tmpfs.
    let _ = std::fs::remove_dir_all(&tmp);

    for (dir, label, ok, stderr) in &outcomes {
        assert!(
            *ok,
            "models/{dir}/pretrain_prod.nsl must build under {label}. If the \
             tape arm regressed, the misleading historical symptom was \"train \
             step body must assign to a variable named 'loss'\" from a \
             silently skipped step body.\n{stderr}"
        );
    }
}
