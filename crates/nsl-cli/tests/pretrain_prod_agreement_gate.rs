//! Roadmap items 4, 9 and 10's acceptance criterion, machine-checked: "no
//! machine paths / synthetic data; config and scheduler agree" — at ALL THREE
//! model sizes.
//!
//! `models/coder*/config.nsl` is pure documentation — nothing imports it (the
//! train-config contract requires literals in the header, so the script cannot
//! read the consts) — which is exactly how all three pretrain blocks drifted
//! into fiction. At 50M: batch 32, total_steps 305000, a data path this repo
//! never contained, plus a Windows machine path in pretrain.nsl. At 500M the
//! SAME 305000/3000 pair survived item 4 untouched, because that campaign only
//! rewrote the 50M pair; at 1B it was 100000/2000, a ~3.28B-token budget for a
//! corpus of 8.93M. Each campaign fixed only its own size, which is why this
//! gate is a TABLE: adding a size means adding a row, and a size that is
//! missing from the table is the state every one of those fictions lived in.
//! It parses each committed pair, refuses drift, and re-derives each scheduler
//! length from the corpus arithmetic so the numbers cannot merely agree on a
//! shared fiction.
//!
//! The recipes differ in ONE deliberate way, encoded in `train_tokens_const`
//! below: 50M (item 4) schedules over the whole corpus, while 500M (item 9)
//! and 1B (item 10) train on a prefix slice and report cross-entropy on a
//! held-out tail. At 0.017 (500M) and 0.008 (1B) tokens/param a training loss
//! alone cannot distinguish an under-trained model from a memorizing one,
//! which is the question a run on this corpus actually raises.
//!
//! Item 10 adds two things a size table alone would not catch. `run_line_flags`
//! pins the flag set each header RECOMMENDS, in both directions — the gate
//! builds with exactly that set (a documented flag that stopped being honored
//! is a build error under Milestone A's inert-request enforcement) and requires
//! the header's RUN block to name exactly those flags. It deliberately does NOT
//! claim those flags are each necessary: item 10 measured that one of the 1B
//! line's three has no memory effect the measurement can resolve. And
//! `two_phase_clip` asserts, through `nsl check --training-report`, that a
//! recipe declaring `grad_clip` still PLANS the clip under its own flag set:
//! `--layerwise-accum`, which Milestone B's endurance benchmark uses, refuses
//! grad_clip outright, and the failure mode a production recipe at this scale
//! actually hits is a loss excursion.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// The line constructing the loader the TRAIN BLOCK consumes, comment-stripped.
///
/// Codegen binds `state.cleanup.dataloader_vars.last()` — the last DataLoader
/// constructed before the train block — so the variable NAME carries no
/// meaning. Anchoring on a literal `let loader = DataLoader(` instead lets a
/// recipe rename its loaders and have these assertions land on the validation
/// loader while the training loader goes unchecked.
fn training_loader_line<'a>(prod: &'a str, dir: &str) -> &'a str {
    let train_line = prod
        .lines()
        .position(|l| l.trim_start().starts_with("train("))
        .unwrap_or_else(|| panic!("{dir}: no `train(` block"));
    let loaders: Vec<&str> = prod
        .lines()
        .take(train_line)
        .map(|l| l.split('#').next().unwrap_or(l))
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("let ") && t.contains("= DataLoader(")
        })
        .collect();
    // Every `DataLoader(` before the train block must be one of these simple
    // `let` bindings. A construction the parse below cannot see — inside an
    // `if`, say, where codegen INTERSECTS `dataloader_vars` across branches —
    // would make this line-order rule disagree with codegen's `.last()`.
    let constructions = prod
        .lines()
        .take(train_line)
        .map(|l| l.split('#').next().unwrap_or(l))
        .map(|l| l.matches("DataLoader(").count())
        .sum::<usize>();
    assert_eq!(
        constructions,
        loaders.len(),
        "{dir}: {constructions} `DataLoader(` constructions before the train \
         block but only {} are plain `let NAME = DataLoader(` bindings. This \
         gate resolves the training loader by source order, which only tracks \
         codegen's `dataloader_vars.last()` while every construction is \
         unconditional and top-level",
        loaders.len()
    );
    loaders
        .last()
        .copied()
        .unwrap_or_else(|| panic!("{dir}: no DataLoader is constructed before the train block"))
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
    ///
    /// ORDER IS MEANINGFUL: `[0]` is the TRAINING corpus and `[1]`, where a
    /// size has one, is the HELD-OUT slice. The surjectivity check below only
    /// pins the SET of paths; `loader_bindings_name_the_right_slice` is what
    /// binds each path to its ROLE. Review finding: with surjectivity alone,
    /// swapping the two `load_mmap` paths kept both "used exactly once" and
    /// passed the whole gate while training on the held-out set and scoring on
    /// the training set — the same defect the surjectivity check was added to
    /// close, one level up.
    load_sites: &'static [(&'static str, &'static str)],
    /// `(config const, expected u16 token count)` for each local data file
    /// above, so a recorded count cannot drift from the bytes on disk.
    token_counts: &'static [(&'static str, &'static str)],
    /// Whether the training loader shuffles. `None` for a model whose config
    /// does not record the choice (50M predates the const). Pinned at 500M/1B
    /// because it is a deliberate choice with a recorded rationale — a
    /// concatenated corpus fed in file order — that carries no other symptom
    /// if it silently flips back.
    shuffle: Option<bool>,
    /// The flags BEYOND `--source-ad` that this recipe's header's RUN block
    /// names. Checked in both directions: the recipe must build with exactly
    /// this set applied, and the RUN block must name exactly these flags. A
    /// header that names a flag the recipe no longer needs sends the next
    /// person to a slower or larger configuration for no reason; a header that
    /// omits one sends them to an OOM.
    ///
    /// NOT "required" — this field used to be called `required_flags`, and item
    /// 10 MEASURED that the name was wrong. Dropping `--fuse-rmsnorm-backward`
    /// at 1B leaves the run surviving 40 micro-steps, with a driver-peak
    /// difference (19 MiB raw, 3 MiB net, opposite signs) far below the
    /// ~1548 MiB run-to-run spread on that quantity — i.e. no resolvable
    /// effect; dropping `--checkpoint-blocks` OOMs at step 0. Both
    /// are in the RUN line, only one is load-bearing. The recommended line is
    /// the thing worth pinning against drift — necessity is a separate,
    /// measured question, and its per-flag table lives in
    /// `models/benchmarks/PROD1B_VALIDATION_2026_08_19.md` (EC6).
    run_line_flags: &'static [&'static str],
    /// Whether this size's schedule was MEASURED and must therefore point at a
    /// validation record. Checked per size rather than repo-wide: a global
    /// count is satisfied by whichever size already cites one.
    cites_validation_record: bool,
    /// Whether train and val are a prefix/tail cut of ONE stream (the
    /// make_prod_split model). The overlap-arithmetic check only applies
    /// then; the 1B pair scores on SEPARATE source bins whose disjointness
    /// is owned by the corpus machinery (repo_id intersection in verify.py,
    /// file-list disjointness in corpus_manifest_gate).
    same_stream_split: bool,
    /// Whether the FASE planner must report `two_phase_clip: true`.
    ///
    /// SCOPE, stated because it is narrower than it looks: `nsl check
    /// --training-report` accepts none of the codegen flags (`--help` lists
    /// neither `--layerwise-accum` nor `--optim-state-offload`), so this is a
    /// FLAG-INDEPENDENT property of the recipe's own optimizer and
    /// accumulation shape — it catches an optimizer swap whose plan cannot
    /// clip, not a flag set that drops the clip. The flag-set half of the
    /// question is answered by `run_line_flags` building and by the pinned
    /// `--layerwise-accum` refusal, on the contract that this codebase refuses
    /// incompatible compositions rather than silently degrading them.
    two_phase_clip: bool,
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
        run_line_flags: &[],
        cites_validation_record: false,
        same_stream_split: false,
        two_phase_clip: true,
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
        run_line_flags: &["--checkpoint-blocks"],
        cites_validation_record: true,
        same_stream_split: true,
        two_phase_clip: true,
    },
    // Item 6: the 1B recipe trains on the v2 corpus and scores on its TWO
    // held-out sets (stack_val repository-disjoint, web_val file-disjoint) —
    // separate SOURCE bins, not a prefix/tail cut of one stream, so the
    // same-stream overlap check below does not apply to this pair (see
    // `same_stream_split`). Resident posture per the item-1 measurement:
    // --optim-state-offload left the run line (+30% throughput; offload
    // stays the smaller-card escape hatch, documented in the header).
    ProdPair {
        dir: "coder1b",
        train_tokens_const: "TRAIN_TOKENS",
        load_sites: &[
            (
                "../../data/tokens/mix/pretrain_train.bin",
                "data/tokens/mix/pretrain_train.bin",
            ),
            (
                "../../data/tokens/mix/stack_val.bin",
                "data/tokens/mix/stack_val.bin",
            ),
            (
                "../../data/tokens/mix/web_val.bin",
                "data/tokens/mix/web_val.bin",
            ),
        ],
        token_counts: &[
            ("TRAIN_TOKENS", "data/tokens/mix/pretrain_train.bin"),
            ("STACK_VAL_TOKENS", "data/tokens/mix/stack_val.bin"),
            ("WEB_VAL_TOKENS", "data/tokens/mix/web_val.bin"),
        ],
        shuffle: Some(true),
        run_line_flags: &["--checkpoint-blocks", "--fuse-rmsnorm-backward"],
        cites_validation_record: true,
        same_stream_split: false,
        two_phase_clip: true,
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
        // SURJECTIVE, not just "each site is allowed". Review finding: with an
        // any() check alone, a recipe whose VAL loader pointed at the TRAIN
        // slice satisfied every assertion here — two allowed sites, both in
        // the list — while scoring validation on training data. Each declared
        // path must be used EXACTLY once.
        for (path, _) in pair.load_sites {
            let uses = load_lines
                .iter()
                .filter(|l| l.contains(&format!("\"{path}\"")))
                .count();
            assert_eq!(
                uses, 1,
                "{dir}: declared corpus path {path} is loaded {uses} times, \
                 expected exactly 1 — two loaders naming the SAME slice is how \
                 a validation set silently becomes the training set"
            );
        }
        // config.nsl's documented paths must be the ones the recipe loads
        // (PRETRAIN_DATA was previously unchecked at 50M and had drifted to a
        // directory that never existed). PRETRAIN_VAL_DATA likewise, where the
        // model declares one.
        let documented = pair.load_sites[0].0;
        assert!(
            config.contains(&format!("const PRETRAIN_DATA = \"{documented}\"")),
            "{dir}: config.nsl PRETRAIN_DATA disagrees with the corpus \
             pretrain_prod.nsl loads ({documented})"
        );
        for (val_path, _) in &pair.load_sites[1..] {
            // One PRETRAIN_VAL_DATA* const per held-out site (a size may
            // have several — item 6's 1B documents _STACK and _WEB).
            let documented_val = config.lines().any(|l| {
                l.trim_start().starts_with("const PRETRAIN_VAL_DATA")
                    && l.contains(&format!("\"{val_path}\""))
            });
            assert!(
                documented_val,
                "{dir}: no PRETRAIN_VAL_DATA* const in config.nsl names the \
                 held-out slice pretrain_prod.nsl loads ({val_path})"
            );
        }
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
            // Resolve the training loader the way CODEGEN does — the last
            // DataLoader constructed before the train block — not by its
            // variable name. Review finding: anchoring on the literal string
            // `let loader = DataLoader(` let a recipe rename its training
            // loader to `trainer`, name the VALIDATION loader `loader`, and
            // train unshuffled with this assertion passing against the val
            // loader. That is exactly the silent flip-back this field exists
            // to catch, and it is the same name-independence
            // `loader_bindings_name_the_right_slice` already argues for.
            let train_loader = training_loader_line(&prod, dir);
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

        // ── the step body's reshape literals ───────────────────────────────
        // `logits.reshape([N, V])` / `labels.reshape([N])` carry BARE literals,
        // so `prod_kwarg` (which parses `key=value`) cannot see them and every
        // other check in this gate is blind to them.
        //
        // Review finding, confirmed by building the mutants: a wrong N is NOT a
        // build error. Under `--source-ad` the fused-LCE substitution DISCARDS
        // the reshape, so the build links clean and still reports
        // `[fused-lce] forward route=gemm` — while under tape AD, the gate's
        // other arm, the reshape actually executes. A stale literal therefore
        // makes the two AD modes compute different things with the whole gate
        // green, which is the exact class of silent cross-mode divergence that
        // cost this repo the SDPA causal-flag bug.
        let n = (prod_kwarg(&prod, "batch_size", dir) * prod_kwarg(&prod, "seq_len", dir)) as u64;
        let v = config_const(&config, "VOCAB_SIZE", dir) as u64;
        for (want, what) in [
            (format!("reshape([{n}, {v}])"), "logits"),
            (format!("reshape([{n}])"), "labels"),
        ] {
            assert!(
                prod.contains(&want),
                "{dir}: the step body must flatten {what} with `{want}` — \
                 batch_size * seq_len = {n}, vocab = {v}. A wrong literal here \
                 is invisible: source-AD's fused-LCE substitution discards the \
                 reshape and links clean, while tape AD executes it, so the two \
                 AD modes silently diverge"
            );
        }

        // ── drop_last is LOAD-BEARING for the derivation above ─────────────
        // The scheduler length is derived with `floor(train_tokens / slot)`.
        // That is only the number of batches the loader actually yields when
        // `drop_last` is TRUE: the runtime defaults it to FALSE
        // (`dataloader.rs`, `as_bool().unwrap_or(false)`) and then uses
        // `data_len.div_ceil(tokens_per_batch)` instead.
        //
        // Review finding: nothing pinned this, and the premise lived only in a
        // comment. At 50M it has a live consequence — CORPUS_TOKENS 8,925,916
        // over a 1024-token slot leaves a remainder of 732, so floor gives
        // 8716 batches/epoch (26,148 over 3 epochs, the committed
        // PRETRAIN_TOTAL_STEPS) while div_ceil gives 8717 (26,151). Dropping
        // `drop_last=true` would run 3 micro-steps past the end of the cosine
        // with this whole gate green. 500M and 1B divide exactly, so the
        // remainder never bites there — which is precisely why the assertion
        // has to be unconditional rather than "where it matters today".
        let train_loader = training_loader_line(&prod, dir);
        assert!(
            train_loader.contains("drop_last=true"),
            "{dir}: the training DataLoader must set `drop_last=true` — the \
             scheduler length is derived with floor(train_tokens / slot), and \
             the runtime DEFAULTS drop_last to false and switches to div_ceil, \
             which yields more batches than the schedule covers. Line: \
             {train_loader}"
        );

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

/// Items 9 and 10: the held-out split must actually be held out — at EVERY
/// size that uses it.
///
/// The train slice is a PREFIX and the validation slice is the TAIL of one
/// corpus, so "held out" is arithmetic, not a naming convention — and the
/// arithmetic lives in two places (config.nsl and make_prod_split.py) that can
/// drift apart. A split whose halves overlap would report a validation loss
/// measured on training data, which is worse than reporting none.
///
/// Item 10 made this a loop rather than a coder500m constant. 500M and 1B read
/// the SAME two files but slice them differently: a delivery slot is
/// `batch_size * seq_len` contiguous tokens, which is 2048 at 500M and 4096 at
/// 1B. The alignment assertion is therefore per-size and a single hardcoded
/// check would have gone on passing for a size it was no longer describing.
#[test]
fn the_held_out_split_does_not_overlap_the_training_slice() {
    let root = repo_root();
    let script = std::fs::read_to_string(root.join("models/benchmarks/make_prod_split.py"))
        .expect("make_prod_split.py");

    let mut checked = 0usize;
    for pair in PAIRS {
        // Only the sizes that train on a prefix and score on a tail of ONE
        // stream. The 1B pair (item 6) scores on separate source bins; its
        // disjointness is owned by verify.py's repo_id intersection and the
        // corpus gate's file-list check.
        if !pair.same_stream_split || pair.load_sites.len() < 2 {
            continue;
        }
        checked += 1;
        let dir = pair.dir;
        let config =
            std::fs::read_to_string(root.join(format!("models/{dir}/config.nsl"))).unwrap();
        let corpus = config_const(&config, "CORPUS_TOKENS", dir);
        let train = config_const(&config, "TRAIN_TOKENS", dir);
        let val = config_const(&config, "VAL_TOKENS", dir);
        let seq = config_const(&config, "MAX_SEQ_LEN", dir);

        assert!(
            train + val < corpus,
            "{dir}: TRAIN_TOKENS({train}) + VAL_TOKENS({val}) >= \
             CORPUS_TOKENS({corpus}) — the validation tail would overlap the \
             training prefix"
        );
        // A strictly positive gap is not enough — but NOT because a window
        // could straddle it: the slices are separate files, so they are
        // index-disjoint even at zero gap. The gap buys SEPARATION. This
        // corpus is a concatenation of source trees, so the tokens immediately
        // after the training prefix are usually the continuation of the same
        // file, and a held-out set starting there is scored on text whose
        // preceding context was trained on. One window is the minimum that
        // means anything — and the window is wider at 1B, which is why this
        // reads each size's own MAX_SEQ_LEN.
        let gap = corpus - train - val;
        assert!(
            gap >= seq,
            "{dir}: the unused gap between the training prefix and the \
             held-out tail is {gap} tokens, under one {seq}-token window — too \
             close for the held-out set to be scoring anything but the \
             continuation of the last file trained on"
        );
        // Both slices must be a whole number of DELIVERY SLOTS, and a slot is
        // batch_size * seq_len contiguous tokens — `build_simple_batch` reads
        // one flat span per slot and reshapes it to [batch, seq]. Checking
        // modulo seq_len alone (the first version of this gate) passes token
        // counts that leave a partial batch for `drop_last=true` to discard,
        // which silently shortens the epoch below the derived step count.
        let batch = config_const(&config, "PRETRAIN_BATCH_SIZE", dir);
        let slot = seq * batch;
        assert_eq!(
            train % slot,
            0.0,
            "{dir}: TRAIN_TOKENS({train}) is not a whole number of \
             {slot}-token delivery slots — drop_last would discard the \
             remainder and the derived step count would overstate the epoch"
        );
        assert_eq!(
            val % slot,
            0.0,
            "{dir}: VAL_TOKENS({val}) is not a whole number of {slot}-token \
             delivery slots"
        );

        // CORPUS_TOKENS is the sole input to the gap arithmetic above, so pin
        // it to the bytes on disk when the local corpus is present —
        // otherwise the "held out" guarantee rests on a number nothing checks.
        let corpus_path = root.join("data/tokens/train_new.bin");
        if corpus_path.exists() {
            let bytes = std::fs::metadata(&corpus_path).unwrap().len();
            assert_eq!(
                bytes / 2,
                corpus as u64,
                "{dir}: CORPUS_TOKENS={corpus} disagrees with the on-disk \
                 corpus ({} u16 tokens) — the gap assertion above is derived \
                 from it",
                bytes / 2
            );
            // ── and the slices must actually BE that prefix and that tail ──
            // Everything above is arithmetic over CONSTANTS. Review finding:
            // the disjointness this test is named for lives in
            // make_prod_split.py's slice expressions, which nothing here read.
            // Rewriting `data[val_start * 2 :]` as
            // `data[(TRAIN_TOKENS - VAL_TOKENS) * 2 : TRAIN_TOKENS * 2]` makes
            // the "held-out" set a SUBSET of the training prefix while every
            // constant, every byte COUNT, and the whole gap computation stay
            // identical — the recipe would then report a held-out loss
            // measured on trained text. Compare content, not lengths.
            let corpus = std::fs::read(&corpus_path).unwrap();
            for (rel, want) in [
                (
                    "data/tokens/prod_train_slice.bin",
                    &corpus[..(train as usize) * 2],
                ),
                (
                    "data/tokens/prod_val_slice.bin",
                    &corpus[corpus.len() - (val as usize) * 2..],
                ),
            ] {
                let p = root.join(rel);
                if !p.exists() {
                    continue;
                }
                let got = std::fs::read(&p).unwrap();
                assert!(
                    got == want,
                    "{dir}: {rel} is not the cut make_prod_split.py documents. \
                     The train slice must be the first TRAIN_TOKENS of \
                     train_new.bin and the val slice its last VAL_TOKENS; a \
                     slice of the right LENGTH taken from the wrong offset \
                     passes every count-based check in this test while \
                     scoring held-out loss on trained text"
                );
            }
        } else {
            eprintln!("[gate] data/tokens/train_new.bin absent — CORPUS_TOKENS arm skipped");
        }

        // The materializer's constants must match config.nsl — it is what
        // writes the files the recipe reads. Every size that reads the split
        // must agree with it, so two sizes cannot silently want different cuts
        // of the same two files.
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
                "make_prod_split.py {name}={got} disagrees with \
                 {dir}/config.nsl {want} — the script writes the files the \
                 recipe reads"
            );
        }
    }
    // A loop that silently iterated over nothing would pass this test while
    // checking no split at all.
    // Item 6 moved the 1B to SEPARATE held-out source bins, so only 500M
    // remains on the same-stream split this check models.
    assert!(
        checked >= 1,
        "expected at least one same-stream held-out split, checked {checked}"
    );
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

/// Every validation record a recipe cites must exist.
///
/// These files carry their evidence by REFERENCE: "LR and warmup are measured
/// — see models/benchmarks/PRODxxx_VALIDATION_*.md". A citation is the entire
/// warrant for the two constants a reader is most likely to question, and a
/// dangling one is worse than no citation at all, because it reads as though
/// the measurement happened.
///
/// This gate exists because it caught exactly that in its own campaign: item
/// 10 staged `PROD1B_VALIDATION_2026_08_19.md` in four places — config.nsl,
/// the recipe header, the README and this file's own doc comments — while the
/// arms it was supposed to record were still running and the document did not
/// exist. Nothing else in this gate could have failed: it reads numeric consts
/// and flag strings, never a cited path.
#[test]
fn every_cited_validation_record_exists() {
    let root = repo_root();
    // This FILE cites the 1B record too, in the `run_line_flags` doc comment —
    // a review pointed out that the first version scanned the recipes but not
    // the gate that polices them, so the one citation site guaranteed to be
    // read by whoever next edits these rules was the one site unchecked.
    // Scanned outside the per-size loop because it belongs to no single size.
    let mut extra: Vec<String> = vec!["crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs".into()];
    for pair in PAIRS {
        let mut cited_here = 0usize;
        for rel in [
            format!("models/{}/pretrain_prod.nsl", pair.dir),
            format!("models/{}/config.nsl", pair.dir),
            format!("models/{}/README.md", pair.dir),
        ]
        .into_iter()
        .chain(extra.drain(..))
        {
            let path = root.join(&rel);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            // Citations found in the shared file above are checked for
            // existence but must not count toward THIS size's floor.
            let counts_for_size = rel.starts_with(&format!("models/{}/", pair.dir));
            for raw in text.split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == '`') {
                // Only the validation-record family: a general markdown-link
                // checker would drag in every doc reference in the tree and
                // fail for reasons that have nothing to do with this contract.
                if !raw.contains("_VALIDATION_") {
                    continue;
                }
                // A glob is prose naming the FAMILY of records, not a citation
                // of one — this test's own doc comment says "PRODxxx_VALIDATION_*.md".
                // Skipping it is not a bypass: a real citation rewritten as a
                // glob stops counting toward its size's floor below, so the
                // per-size assertion catches the dodge.
                if raw.contains('*') {
                    continue;
                }
                // Recover the path from whatever punctuation surrounds it. A
                // review found that matching on a bare `ends_with(".md")` let a
                // `#EC1` anchor or a trailing colon slip past — and because a
                // skipped token was also never counted, the same edit hid the
                // dangling citation AND lowered the vacuity floor guarding it.
                // Cut at the first `#` (section anchor), then take everything
                // up to and including `.md`.
                let head = raw.split('#').next().unwrap_or(raw);
                let Some(end) = head.find(".md") else {
                    continue;
                };
                let tok = head[..end + 3].trim_start_matches(['(', '[', '"', '\'']);
                if counts_for_size {
                    cited_here += 1;
                }
                let found = if tok.contains('/') {
                    root.join(tok).exists()
                } else {
                    // Cited by basename (the recipe header does this) — accept
                    // it anywhere under models/.
                    std::fs::read_dir(root.join("models"))
                        .into_iter()
                        .flatten()
                        .flatten()
                        .any(|e| e.path().join(tok).exists())
                };
                assert!(
                    found,
                    "{rel} cites `{tok}`, which does not exist. A recipe that \
                     points at a validation record for its hyperparameters is \
                     asserting the measurement happened; if the file is not \
                     there, the assertion is unearned."
                );
            }
        }
        // PER SIZE, not a total. A repo-wide count is satisfied by one file:
        // coder500m/config.nsl alone carries two citations, so deleting every
        // 1B citation — the obvious way to make this test green before the 1B
        // record exists — would leave a global floor of 2 intact while the size
        // whose LR provenance is the whole point cites nothing.
        assert_eq!(
            cited_here > 0,
            pair.cites_validation_record,
            "{}: found {cited_here} validation-record citation(s), but this \
             gate declares cites_validation_record={}. A size whose schedule was \
             measured must point at the record; a size that measured nothing \
             must not pretend it did.",
            pair.dir,
            pair.cites_validation_record
        );
    }
}

/// Each `DataLoader` must be fed by the slice its ROLE requires — where "role"
/// is decided the way CODEGEN decides it, not by variable name or file order.
///
/// The surjectivity check in the main test pins the SET of corpus paths a
/// recipe may load, and that is weaker than it looks: swap the two `load_mmap`
/// paths and each is still loaded exactly once, so the recipe trains on the
/// held-out tail and reports "validation" on the training prefix with the
/// whole gate green.
///
/// THE RULE THIS ENCODES. A `train(...)` block does not name its loader. It
/// takes `state.cleanup.dataloader_vars.last()`
/// (crates/nsl-codegen/src/stmt.rs, `compile_train_block_inner`), and that vec
/// is pushed in CONSTRUCTION order at crates/nsl-codegen/src/expr/calls.rs
/// (`nsl_dataloader_create`). So the training loader is *the last DataLoader
/// constructed before the train block*, and the variable it was bound to is
/// irrelevant. The first version of this test assumed "the first DataLoader in
/// the file is the training one" — true of every recipe as written, but a
/// PROXY, and a review found the hole: hoisting the validation loader above
/// the train block (a plausible "group the data section" refactor, no path or
/// name changed) makes `.last()` the val loader, so the run trains on the
/// 524,288-token held-out tail and then scores on the tail it just trained on,
/// with every static check still green.
///
/// The validation loader is resolved through its own consumer, `for batch in
/// <name>:`, for the same reason.
#[test]
fn loader_bindings_name_the_right_slice() {
    let root = repo_root();
    let mut checked_val = 0usize;
    for pair in PAIRS {
        let dir = pair.dir;
        let prod = std::fs::read_to_string(root.join(format!("models/{dir}/pretrain_prod.nsl")))
            .unwrap();
        let code: Vec<&str> = prod
            .lines()
            .map(|l| l.split('#').next().unwrap_or(l))
            .collect();

        // `let NAME = load_mmap("PATH", ...)` -> NAME: PATH.
        let mut bound: Vec<(String, String)> = Vec::new();
        // `let NAME = DataLoader(ARG, ...)` -> (line, NAME, ARG).
        let mut loaders: Vec<(usize, String, String)> = Vec::new();
        let mut train_line: Option<usize> = None;
        // `for <x> in <NAME>:` -> NAME, the loaders actually iterated.
        let mut iterated: Vec<(usize, String)> = Vec::new();

        for (i, line) in code.iter().enumerate() {
            let t = line.trim_start();
            if t.starts_with("train(") && train_line.is_none() {
                train_line = Some(i);
            }
            if let Some(rest) = t.strip_prefix("for ") {
                if let Some((_, tail)) = rest.split_once(" in ") {
                    iterated.push((i, tail.trim().trim_end_matches(':').trim().to_string()));
                }
            }
            let Some(rest) = t.strip_prefix("let ") else {
                continue;
            };
            let Some((name, expr)) = rest.split_once('=') else {
                continue;
            };
            let (name, expr) = (name.trim().to_string(), expr.trim());
            if let Some(after) = expr.strip_prefix("load_mmap(") {
                if let Some(path) = after.split('"').nth(1) {
                    bound.push((name, path.to_string()));
                }
            } else if let Some(after) = expr.strip_prefix("DataLoader(") {
                let arg = after.split(',').next().unwrap_or("").trim().to_string();
                loaders.push((i, name, arg));
            }
        }

        assert_eq!(
            loaders.len(),
            pair.load_sites.len(),
            "{dir}: expected {} DataLoader binding(s), found {loaders:?}",
            pair.load_sites.len()
        );
        let train_line = train_line
            .unwrap_or_else(|| panic!("{dir}: pretrain_prod.nsl has no `train(` block"));

        let resolve = |arg: &str| -> String {
            bound
                .iter()
                .find(|(n, _)| n == arg)
                .map(|(_, p)| p.clone())
                .unwrap_or_else(|| {
                    panic!(
                        "{dir}: DataLoader is fed `{arg}`, which is not bound by any \
                         load_mmap in this file — the gate cannot tell which slice it reads"
                    )
                })
        };

        // THE TRAINING LOADER = the last one constructed before the train block.
        let (_, train_name, train_arg) = loaders
            .iter()
            .rfind(|(i, _, _)| *i < train_line)
            .unwrap_or_else(|| {
                panic!(
                    "{dir}: no DataLoader is constructed before the train block, so \
                     `dataloader_vars.last()` has nothing to bind"
                )
            });
        let train_path = resolve(train_arg);
        assert_eq!(
            train_path, pair.load_sites[0].0,
            "{dir}: the train block consumes `{train_name}` (the LAST DataLoader \
             built before it, which is how codegen picks it), and that loader is \
             fed {train_path} — but this size's TRAINING corpus is {}. Training \
             on the held-out slice passes every set-based check in this gate",
            pair.load_sites[0].0
        );

        // Every held-out site (item 6: a size may have SEVERAL — the 1B
        // scores stack_val and web_val separately). Each must be iterated by
        // exactly one post-train loop, in load_sites ORDER (the printed
        // VAL_LOSS_* labels follow that order), and no post-train loop may
        // read anything else.
        let val_sites = &pair.load_sites[1..];
        if !val_sites.is_empty() {
            checked_val += 1;
            let val_iters: Vec<&(usize, String)> =
                iterated.iter().filter(|(i, _)| *i > train_line).collect();
            assert_eq!(
                val_iters.len(),
                val_sites.len(),
                "{dir}: expected {} held-out `for ... in <loader>:` loop(s) \
                 after the train block, found {val_iters:?}",
                val_sites.len()
            );
            for (k, (val_path_want, _)) in val_sites.iter().enumerate() {
                let val_name = &val_iters[k].1;
                let (_, _, val_arg) = loaders
                    .iter()
                    .find(|(_, n, _)| n == val_name)
                    .unwrap_or_else(|| {
                        panic!("{dir}: the held-out loop iterates `{val_name}`, which is not a DataLoader")
                    });
                let val_path = resolve(val_arg);
                assert_eq!(
                    val_path, *val_path_want,
                    "{dir}: held-out loop #{k} iterates `{val_name}`, fed \
                     {val_path}, but this size's held-out site #{k} is \
                     {val_path_want}. A validation loop reading the training \
                     corpus reports a loss measured on data the model trained on"
                );
                assert_ne!(
                    val_name, train_name,
                    "{dir}: a held-out loop and the train block consume the \
                     SAME loader `{val_name}`"
                );
            }
        }
    }
    assert!(
        checked_val >= 2,
        "expected at least two sizes with a validation loader, checked {checked_val}"
    );
}

/// Every `pretrain_prod.nsl` in the tree must have a row in `PAIRS`.
///
/// This gate's whole value is the table, and a table is only as good as its
/// coverage. Item 4 rewrote the 50M pair and left 500M's `305000/3000` fiction
/// standing; item 9 rewrote 500M and left 1B's `100000/2000` standing. In both
/// cases the gate was green the entire time, because the size that was wrong
/// simply was not in it. A new production recipe that nothing checks is
/// exactly that state, so make adding one a test failure rather than an
/// omission nobody notices.
#[test]
fn every_production_recipe_has_a_row_in_the_table() {
    let root = repo_root();
    let mut on_disk: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(root.join("models")).expect("models/").flatten() {
        let path = entry.path();
        if path.join("pretrain_prod.nsl").exists() {
            on_disk.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    on_disk.sort();
    let mut in_table: Vec<String> = PAIRS.iter().map(|p| p.dir.to_string()).collect();
    in_table.sort();
    assert_eq!(
        on_disk, in_table,
        "models/*/pretrain_prod.nsl on disk is {on_disk:?} but this gate's \
         PAIRS table covers {in_table:?}. A production recipe outside the table \
         is unchecked — which is the exact state 500M's total_steps=305000 and \
         1B's total_steps=100000 both sat in while this file was green."
    );
    assert!(
        !on_disk.is_empty(),
        "found no models/*/pretrain_prod.nsl at all — this gate would pass \
         vacuously"
    );
}

/// The `--flag` tokens named on the header's `nsl run ...` line, following
/// backslash continuations onto later comment lines.
fn documented_flags(prod: &str, dir: &str) -> Vec<String> {
    let lines: Vec<&str> = prod.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.trim_start().starts_with('#') && l.contains("nsl run"))
        .unwrap_or_else(|| {
            panic!("{dir}/pretrain_prod.nsl: header has no `nsl run ...` line to document flags")
        });
    let mut flags = Vec::new();
    for line in lines.iter().skip(start) {
        assert!(
            line.trim_start().starts_with('#'),
            "{dir}: the RUN block's backslash continuation left the header comment: {line}"
        );
        for tok in line.split_whitespace() {
            if tok.starts_with("--") && tok.len() > 2 {
                flags.push(tok.to_string());
            }
        }
        if !line.trim_end().ends_with('\\') {
            break;
        }
    }
    flags
}

/// Item 10: the flag set a header documents as required must be the flag set
/// the recipe actually builds under — in both directions.
///
/// The 1B recipe needs three flags beyond `--source-ad` and OOMs on a 32 GB
/// card without `--optim-state-offload`, so its header is the only place that
/// knowledge lives. A header that omits a flag sends the next person to an
/// OOM; a header that names one the recipe no longer needs sends them to a
/// slower or larger configuration for nothing. Neither has any other symptom.
///
/// What the build arm establishes: Milestone A's inert-request enforcement
/// makes a *requested* flag that records no disposition a hard error, and this
/// codebase refuses incompatible compositions rather than degrading them
/// silently. It does NOT establish that a flag is load-bearing at RUNTIME —
/// dropping `--optim-state-offload` compiles fine and dies on the card. That
/// half is a measurement, banked in
/// models/benchmarks/PROD1B_VALIDATION_2026_08_19.md, and it is the recipe's
/// own printed `PEAK_OPTIM_M` / `PEAK_OPTIM_V` / `PEAK_M_PARTIAL` witness that
/// keeps it honest run to run.
#[test]
fn the_flag_set_each_header_documents_is_the_one_that_builds() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_prodflags_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let mut outcomes: Vec<(&str, Vec<String>, bool, String)> = Vec::new();
    for pair in PAIRS {
        let prod = std::fs::read_to_string(root.join(format!(
            "models/{}/pretrain_prod.nsl",
            pair.dir
        )))
        .unwrap();

        // Direction 1: the header names exactly `--source-ad` + run_line_flags.
        let mut documented = documented_flags(&prod, pair.dir);
        documented.sort();
        documented.dedup();
        let mut want: Vec<String> = std::iter::once("--source-ad".to_string())
            .chain(pair.run_line_flags.iter().map(|f| f.to_string()))
            .collect();
        want.sort();
        assert_eq!(
            documented, want,
            "{}: the header's RUN block documents {documented:?} but this gate \
             declares {want:?}. One of the two is wrong, and the header is what \
             a human follows.",
            pair.dir
        );

        // Direction 2: it builds with exactly that set.
        let mut args: Vec<&str> = vec!["--source-ad"];
        args.extend(pair.run_line_flags.iter().copied());
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
            .arg("build")
            .args(&args)
            .arg(root.join(format!("models/{}/pretrain_prod.nsl", pair.dir)))
            .arg("-o")
            .arg(tmp.join(format!("{}_flags.o", pair.dir)))
            .current_dir(root.join(format!("models/{}", pair.dir)))
            .env("NSL_STDLIB_PATH", root.join("stdlib"))
            .output()
            .expect("spawn nsl build");
        outcomes.push((
            pair.dir,
            args.iter().map(|s| s.to_string()).collect(),
            out.status.success(),
            String::from_utf8_lossy(&out.stderr).to_string(),
        ));
    }
    let _ = std::fs::remove_dir_all(&tmp);

    for (dir, args, ok, stderr) in &outcomes {
        assert!(
            *ok,
            "models/{dir}/pretrain_prod.nsl must build under the flag set its \
             own header documents: {args:?}\n{stderr}"
        );
    }
}

/// Item 10: every production recipe declares `grad_clip`, and the flag choice
/// that would silently cost it is pinned as a refusal.
///
/// `--layerwise-accum` is the memory lever Milestone B's endurance benchmark
/// uses, and reaching for it here is the obvious "make 1B fit" move. It
/// refuses grad_clip — two-phase clipping needs the global L2 norm over every
/// parameter's completed `m_partial` before any update, which the layerwise
/// schedule never materializes. That refusal is the reason the 1B production
/// flag set buys its memory from `--optim-state-offload` instead, and pinning
/// it here means a future change that turns the refusal into a silent
/// downgrade fails this test rather than shipping unclipped 1B pretraining.
#[test]
fn grad_clip_is_planned_and_the_incompatible_flag_still_refuses() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_prodclip_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let mut reports: Vec<(&str, bool, String)> = Vec::new();
    let mut refusals: Vec<(&str, bool, String)> = Vec::new();
    for pair in PAIRS {
        if !pair.two_phase_clip {
            continue;
        }
        let src = root.join(format!("models/{}/pretrain_prod.nsl", pair.dir));
        let cwd = root.join(format!("models/{}", pair.dir));

        let out = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
            .args(["check", "--training-report"])
            .arg(&src)
            .current_dir(&cwd)
            .env("NSL_STDLIB_PATH", root.join("stdlib"))
            .output()
            .expect("spawn nsl check");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        reports.push((
            pair.dir,
            out.status.success() && text.contains("two_phase_clip:    true"),
            text,
        ));

        // The negative arm. Without it the assertion above is satisfied by a
        // planner that reports a clip nothing can take away.
        //
        // `--layerwise-accum` requires `--checkpoint-blocks` at the CLI layer,
        // so a size that does not otherwise need it (50M) still has to pass it
        // here — otherwise clap rejects the invocation and the "refusal" this
        // arm observes is an argument-parsing error, not the codegen refusal
        // it is supposed to be pinning. That is exactly how this arm failed
        // the first time it ran.
        let mut args: Vec<&str> = vec!["--source-ad"];
        args.extend(pair.run_line_flags.iter().copied());
        if !args.contains(&"--checkpoint-blocks") {
            args.push("--checkpoint-blocks");
        }
        args.push("--layerwise-accum");
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
            .arg("build")
            .args(&args)
            .arg(&src)
            .arg("-o")
            .arg(tmp.join(format!("{}_lw.o", pair.dir)))
            .current_dir(&cwd)
            .env("NSL_STDLIB_PATH", root.join("stdlib"))
            .output()
            .expect("spawn nsl build");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        refusals.push((
            pair.dir,
            !out.status.success() && text.contains("--layerwise-accum is incompatible with grad_clip"),
            text,
        ));
    }
    let _ = std::fs::remove_dir_all(&tmp);

    assert!(
        !reports.is_empty(),
        "no size declared two_phase_clip — this test checked nothing"
    );
    for (dir, ok, text) in &reports {
        assert!(
            *ok,
            "models/{dir}/pretrain_prod.nsl must plan a two-phase gradient \
             clip (`nsl check --training-report` -> `two_phase_clip:    true`). \
             An optimizer or grad_accumulation change can take that away while \
             `grad_clip=` still reads fine in the source.\n{text}"
        );
    }
    for (dir, ok, text) in &refusals {
        assert!(
            *ok,
            "models/{dir}/pretrain_prod.nsl + --layerwise-accum must REFUSE, \
             naming grad_clip. If this now builds, the layerwise schedule is \
             either materializing the global norm (in which case the 1B \
             recipe should reconsider its flag set) or it is silently dropping \
             the clip.\n{text}"
        );
    }
}

/// Item 10's literal acceptance criterion: "separate real recipe from
/// endurance benchmark".
///
/// `models/coder1b/pretrain_1b2048.nsl` is Milestone B's certification
/// workload. It is NOT runnable by hand: `load_mmap("B2048_TOKENS_PATH", 3)`
/// and `B2048_CKPT_ARGS` are marker strings that only
/// models/benchmarks/endurance_1b.py rewrites, and it prints WITNESS_* blocks
/// for that harness to assert on. Item 10 did not convert it — a benchmark
/// that certifies the memory stack and a recipe that trains a model want
/// different things, and collapsing them is how `pretrain_prod.nsl` would
/// grow marker strings or the benchmark would lose its witnesses. This test
/// holds both halves of that separation, in both directions, so "separate"
/// cannot quietly become "duplicated" or "merged".
#[test]
fn the_1b_production_recipe_is_separate_from_the_endurance_benchmark() {
    let root = repo_root();
    let bench_path = root.join("models/coder1b/pretrain_1b2048.nsl");
    let bench = std::fs::read_to_string(&bench_path).expect("pretrain_1b2048.nsl");
    let prod = std::fs::read_to_string(root.join("models/coder1b/pretrain_prod.nsl"))
        .expect("coder1b/pretrain_prod.nsl");

    // The production recipe is allowed — encouraged — to NAME the markers in
    // its header while explaining why the two files are separate. What it must
    // not do is contain one as code. Compare against the CODE only.
    let prod_code: String = prod
        .lines()
        .map(|l| l.split('#').next().unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n");
    // The benchmark keeps its harness contract.
    for marker in ["B2048_TOKENS_PATH", "B2048_CKPT_ARGS"] {
        assert!(
            bench.contains(marker),
            "pretrain_1b2048.nsl lost the {marker} marker — endurance_1b.py \
             rewrites it by exact string and a missing rewrite is a hard error \
             in the harness"
        );
        // ...and the production recipe never grows one. A marker string in a
        // recipe a human is told to run by hand is a file that cannot run.
        assert!(
            !prod_code.contains(marker),
            "coder1b/pretrain_prod.nsl uses the harness marker {marker} in \
             CODE (a header mention is fine). The production recipe must be \
             runnable by hand exactly as the header documents"
        );
    }
    // The harness must still point at the benchmark, not at the new recipe:
    // "separate" is only true if the certification workload kept its consumer.
    let harness = std::fs::read_to_string(root.join("models/benchmarks/endurance_1b.py"))
        .expect("endurance_1b.py");
    assert!(
        harness.contains("pretrain_1b2048.nsl"),
        "endurance_1b.py no longer references pretrain_1b2048.nsl — Milestone \
         B's certification workload lost its harness"
    );
    assert!(
        !harness.contains("coder1b/pretrain_prod.nsl"),
        "endurance_1b.py now drives the production recipe. The benchmark \
         needs marker rewriting and prints WITNESS_* blocks; the recipe has a \
         scheduler, a held-out pass and real corpus paths. Driving one with \
         the other's harness silently changes what Milestone B certifies"
    );
    // The benchmark is a benchmark: it must not have grown a held-out pass,
    // which is how it would drift back into being a second, unmaintained
    // recipe.
    let bench_code: String = bench
        .lines()
        .map(|l| l.split('#').next().unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !bench_code.contains("VAL_LOSS"),
        "pretrain_1b2048.nsl grew a held-out pass. Validation belongs in \
         pretrain_prod.nsl; the benchmark's job is the memory/throughput \
         witness"
    );
    // And the recipe is a recipe: it reports held-out loss and schedules.
    // Against the CODE, so a header that merely discusses a scheduler cannot
    // satisfy this.
    for needed in ["VAL_LOSS", "scheduler:", "checkpoint_save="] {
        assert!(
            prod_code.contains(needed),
            "coder1b/pretrain_prod.nsl is missing `{needed}` — without it this \
             is the benchmark again, not a production recipe"
        );
    }
}
