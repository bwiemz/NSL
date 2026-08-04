//! Drift gate: a model's `const` block, its `config.nsl`, and the LITERALS it
//! actually builds itself from must all agree.
//!
//! Why this exists: NSL `const`s are not in scope inside model field
//! initializers, and `[TransformerBlock; N_LAYERS]` does not parse — the
//! array size and every constructor argument must be a literal. So each
//! `models/<name>/model.nsl` restates its architecture as `const`s for
//! readability and then *ignores them*, passing hand-written literals to the
//! block constructor. Nothing tied the two together.
//!
//! That is not hypothetical. `coder1b/config.nsl` and `coder7b/config.nsl`
//! both declared `ROPE_THETA = 500000.0` while the attention layer hardcoded
//! `RotaryEmbedding(head_dim, 1024, 10000.0)`: the constant was dead, and both
//! models would have pretrained at theta=10000 with a 1024-row rotary table
//! despite `MAX_SEQ_LEN = 2048`. The fix threads `max_seq_len`/`rope_theta`
//! through the constructor; this gate is what stops the literals from
//! drifting away from the constants again.
//!
//! The gate is deliberately dumb (regex over source, no compiler): it must
//! keep working when the parser changes, and a model file that stops matching
//! the expected shape FAILS rather than silently passing — see
//! `every_model_is_actually_covered`.

use std::collections::BTreeMap;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Every model whose `model.nsl` is expected to have the
/// `blocks: [TransformerBlock; N] = TransformerBlock(...)` shape this gate
/// checks. Adding a model here without the right shape fails the gate.
const COVERED_MODELS: &[&str] = &[
    "coder50m",
    "coder500m",
    "coder1b",
    "coder7b",
    "coder-rl",
    "benchmarks/muon50m",
    "benchmarks/muon500m",
];

/// Architecture constants that must agree between `config.nsl` and
/// `model.nsl` when both declare them. Training hyperparameters are
/// intentionally excluded — those live only in `config.nsl`.
const ARCH_KEYS: &[&str] = &[
    "VOCAB_SIZE",
    "D_MODEL",
    "N_LAYERS",
    "N_HEADS",
    "N_KV_HEADS",
    "HEAD_DIM",
    "D_FF",
    "MAX_SEQ_LEN",
    "ROPE_THETA",
    "DROPOUT",
];

/// Parse `const NAME = VALUE` lines into a name -> normalized-number map.
fn parse_consts(src: &str) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for line in src.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("const ") else {
            continue;
        };
        let Some((name, value)) = rest.split_once('=') else {
            continue;
        };
        let name = name.trim();
        // Strip trailing comments before parsing the literal.
        let value = value.split('#').next().unwrap_or("").trim();
        if let Ok(v) = value.parse::<f64>() {
            out.insert(name.to_string(), v);
        }
    }
    out
}

/// Extract `blocks: [<Type>; N] = <Type>(a, b, c, ...)` -> (N, args).
///
/// Returns `None` when the file has no such declaration, which the coverage
/// test turns into a failure rather than a silent skip.
fn parse_block_decl(src: &str) -> Option<(i64, Vec<f64>)> {
    let line = src
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("blocks:") && l.contains('=') && l.contains(';'))?;

    let (decl, ctor) = line.split_once('=')?;

    // "blocks: [TransformerBlock; 16]" -> 16
    let count: i64 = decl
        .rsplit_once(';')?
        .1
        .trim()
        .trim_end_matches(']')
        .trim()
        .parse()
        .ok()?;

    // "TransformerBlock(2048, 32, 8, 64, 8192, 0.0, 2048, 500000.0, 16)"
    let args_str = ctor.trim().split_once('(')?.1.rsplit_once(')')?.0;
    if args_str.trim().is_empty() {
        return Some((count, Vec::new()));
    }
    let args = args_str
        .split(',')
        .map(|a| a.trim().parse::<f64>().ok())
        .collect::<Option<Vec<_>>>()?;

    Some((count, args))
}

fn read(path: &PathBuf) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// `config.nsl` and `model.nsl` must not disagree about the architecture.
#[test]
fn config_and_model_constants_agree() {
    let root = repo_root();
    let mut checked = 0usize;

    for model in COVERED_MODELS {
        let dir = root.join("models").join(model);
        let config_path = dir.join("config.nsl");
        if !config_path.exists() {
            // Several models (coder-rl, the muon benchmarks) carry their
            // constants only in model.nsl. Nothing to cross-check.
            continue;
        }
        let config = parse_consts(&read(&config_path));
        let model_consts = parse_consts(&read(&dir.join("model.nsl")));

        for key in ARCH_KEYS {
            if let (Some(c), Some(m)) = (config.get(*key), model_consts.get(*key)) {
                assert_eq!(
                    c,
                    m,
                    "{model}: {key} drifted — config.nsl says {c}, model.nsl says {m}"
                );
                checked += 1;
            }
        }
    }

    assert!(
        checked >= 30,
        "expected to cross-check at least 30 constants, only saw {checked} — \
         did the const blocks or the parser shape change?"
    );
}

/// The literals passed to the block constructor must match the `const` block
/// in the same file. This is the check that would have caught the dead
/// `ROPE_THETA`.
#[test]
fn block_constructor_literals_match_constants() {
    let root = repo_root();

    // Positional contract of `TransformerBlock(...)` across every model here.
    let expected_order = [
        "D_MODEL",
        "N_HEADS",
        "N_KV_HEADS",
        "HEAD_DIM",
        "D_FF",
        "DROPOUT",
        "MAX_SEQ_LEN",
        "ROPE_THETA",
        "N_LAYERS",
    ];

    for model in COVERED_MODELS {
        let path = root.join("models").join(model).join("model.nsl");
        let src = read(&path);
        let consts = parse_consts(&src);
        let (count, args) = parse_block_decl(&src)
            .unwrap_or_else(|| panic!("{model}: no parsable `blocks: [T; N] = T(...)` declaration"));

        let n_layers = *consts
            .get("N_LAYERS")
            .unwrap_or_else(|| panic!("{model}: model.nsl declares no N_LAYERS"));
        assert_eq!(
            count as f64, n_layers,
            "{model}: block array size {count} != N_LAYERS {n_layers}"
        );

        assert_eq!(
            args.len(),
            expected_order.len(),
            "{model}: TransformerBlock takes {} literals, gate expects {} \
             ({expected_order:?}) — update the gate together with the signature",
            args.len(),
            expected_order.len()
        );

        for (i, key) in expected_order.iter().enumerate() {
            let want = *consts
                .get(*key)
                .unwrap_or_else(|| panic!("{model}: model.nsl declares no {key}"));
            assert_eq!(
                args[i], want,
                "{model}: TransformerBlock arg {i} ({key}) is {} but {key} = {want}",
                args[i]
            );
        }
    }
}

/// The embedding literal must match `VOCAB_SIZE` x `D_MODEL`.
#[test]
fn embedding_literal_matches_constants() {
    let root = repo_root();
    for model in COVERED_MODELS {
        let path = root.join("models").join(model).join("model.nsl");
        let src = read(&path);
        let consts = parse_consts(&src);

        let line = src
            .lines()
            .map(str::trim)
            .find(|l| l.starts_with("embed:") && l.contains("randn(["))
            .unwrap_or_else(|| panic!("{model}: no `embed: ... randn([V, D])` declaration"));

        let dims = line
            .split_once("randn([")
            .and_then(|(_, r)| r.split_once("])"))
            .map(|(d, _)| d)
            .unwrap_or_else(|| panic!("{model}: cannot parse embed shape from `{line}`"));
        let dims: Vec<f64> = dims
            .split(',')
            .map(|d| d.trim().parse::<f64>().expect("embed dim must be a literal"))
            .collect();

        assert_eq!(dims.len(), 2, "{model}: embed must be rank 2");
        assert_eq!(
            dims[0], consts["VOCAB_SIZE"],
            "{model}: embed rows {} != VOCAB_SIZE {}",
            dims[0], consts["VOCAB_SIZE"]
        );
        assert_eq!(
            dims[1], consts["D_MODEL"],
            "{model}: embed cols {} != D_MODEL {}",
            dims[1], consts["D_MODEL"]
        );
    }
}

/// Anti-vacuity: every `models/*/model.nsl` that imports the stdlib GQA must
/// appear in `COVERED_MODELS`, so a new model cannot quietly escape the gate.
#[test]
fn every_model_is_actually_covered() {
    let root = repo_root();
    let mut found = Vec::new();

    fn walk(dir: &std::path::Path, root: &std::path::Path, found: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, root, found);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("model.nsl") {
                let src = std::fs::read_to_string(&path).unwrap_or_default();
                if src.contains("from nsl.nn.gqa import GroupedQueryAttention") {
                    let rel = path
                        .parent()
                        .unwrap()
                        .strip_prefix(root.join("models"))
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    found.push(rel);
                }
            }
        }
    }
    walk(&root.join("models"), &root, &mut found);
    found.sort();

    let mut covered: Vec<String> = COVERED_MODELS.iter().map(|s| s.to_string()).collect();
    covered.sort();

    assert_eq!(
        found, covered,
        "models/ contains stdlib-GQA models that this drift gate does not check \
         (or COVERED_MODELS lists one that no longer exists)"
    );
}

// ─── @fused_lm_ce shape hints ───────────────────────────────────────────────
//
// Same defect class this file exists for, one level out. The decorator's four
// hints are hand-written INT LITERALS — the compiler derives none of them from
// the model — so a training script restates `VOCAB_SIZE` and `D_MODEL` a
// fourth time, and its `batch_size`/`seq_len` a second time after the
// DataLoader. There IS a runtime guard (`nsl_fused_lce_pin_hint_extents`), but
// it needs a GPU and a full run to fire; a change to `D_MODEL` would otherwise
// leave every decorated script stale with nothing catching it in CI.

/// Parse `key=value` pairs out of a `@fused_lm_ce(...)` decorator line.
fn parse_decorator_hints(line: &str) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    let Some(args) = line
        .split_once("@fused_lm_ce(")
        .and_then(|(_, r)| r.rsplit_once(')'))
        .map(|(a, _)| a)
    else {
        return out;
    };
    for pair in args.split(',') {
        if let Some((k, v)) = pair.split_once('=') {
            if let Ok(n) = v.trim().parse::<f64>() {
                out.insert(k.trim().to_string(), n);
            }
        }
    }
    out
}

/// `DataLoader(..., batch_size=N, seq_len=M, ...)` -> (N, M).
fn parse_loader_shape(src: &str) -> Option<(f64, f64)> {
    let line = src.lines().find(|l| l.contains("DataLoader("))?;
    let hints = parse_decorator_hints(&line.replace("DataLoader(", "@fused_lm_ce("));
    Some((*hints.get("batch_size")?, *hints.get("seq_len")?))
}

/// Every `@fused_lm_ce` in a covered model's scripts must agree with that
/// model's constants, with the DataLoader beside it, and with the flatten it
/// feeds.
#[test]
fn fused_lm_ce_hints_match_the_model_and_the_loader() {
    let root = repo_root();
    let mut checked = 0usize;

    for model in COVERED_MODELS {
        let dir = root.join("models").join(model);
        let model_path = dir.join("model.nsl");
        if !model_path.exists() {
            continue;
        }
        let consts = parse_consts(&read(&model_path));
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("nsl") {
                continue;
            }
            let src = read(&path);
            let Some(line) = src.lines().find(|l| l.trim_start().starts_with("@fused_lm_ce(")) else {
                continue;
            };
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let hints = parse_decorator_hints(line);
            let where_ = format!("{model}/{name}");

            for (hint, konst) in [("vocab_size", "VOCAB_SIZE"), ("hidden_size", "D_MODEL")] {
                if let (Some(h), Some(c)) = (hints.get(hint), consts.get(konst)) {
                    assert_eq!(
                        h, c,
                        "{where_}: @fused_lm_ce {hint} = {h} but model.nsl says \
                         {konst} = {c}. The fused kernel strides the head from \
                         the decorator literal, so this reads out of bounds at \
                         runtime."
                    );
                    checked += 1;
                }
            }

            if let Some((lb, ls)) = parse_loader_shape(&src) {
                for (hint, want, which) in [
                    ("batch_size", lb, "DataLoader batch_size"),
                    ("seq_len", ls, "DataLoader seq_len"),
                ] {
                    if let Some(h) = hints.get(hint) {
                        assert_eq!(
                            *h, want,
                            "{where_}: @fused_lm_ce {hint} = {h} but the {which} \
                             is {want}. The backward allocates dx as \
                             [batch_size, seq_len, hidden_size] from the hints."
                        );
                        checked += 1;
                    }
                }

                // The loss flatten must use the same dims. A stale row count
                // here is what silently defeats the fusion.
                if let (Some(v), Some(h)) = (hints.get("vocab_size"), hints.get("hidden_size")) {
                    let _ = h;
                    let needle = format!(", {}])", *v as i64);
                    for l in src.lines().filter(|l| l.contains(".reshape([") && l.contains(&needle)) {
                        let rows: f64 = l
                            .split_once(".reshape([")
                            .and_then(|(_, r)| r.split_once(','))
                            .and_then(|(a, _)| a.trim().parse().ok())
                            .unwrap_or_else(|| panic!("{where_}: cannot parse rows from `{}`", l.trim()));
                        assert_eq!(
                            rows,
                            lb * ls,
                            "{where_}: logits flatten is [{rows}, {v}] but \
                             batch_size * seq_len = {}. Line: `{}`",
                            lb * ls,
                            l.trim()
                        );
                        checked += 1;
                    }
                }
            }
        }
    }

    // Anti-vacuity: this whole test is `if let`s over a directory scan, so a
    // parser that stopped matching -- or a decorator that was removed -- would
    // check nothing and pass. models/coder50m/pretrain.nsl carries one.
    assert!(
        checked >= 5,
        "expected to cross-check at least 5 @fused_lm_ce hints, saw {checked} \
         — did the decorator move, or did the parser stop matching?"
    );
}
