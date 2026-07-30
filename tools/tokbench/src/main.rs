//! Tokenizer compression benchmark for NSL.
//!
//! Measures the metric that actually matters for pretraining cost: how many
//! bytes of raw corpus each token carries. Fewer tokens for the same corpus
//! means fewer sequence positions to train on, which is a direct multiplier on
//! both wall-clock and the effective context a model sees.
//!
//! `train` builds a tokenizer under a named configuration, `eval` scores one
//! against a held-out corpus, and `roundtrip` checks that decode(encode(x)) == x
//! — a property the current NSL tokenizer does not have.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::{
    DecoderWrapper, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer,
    PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer, Trainer,
};

// ---------------------------------------------------------------------------
// Pre-tokenizer regexes
// ---------------------------------------------------------------------------

/// The GPT-2 split pattern, which is what `ByteLevel::default()` applies.
/// Note `\s+(?!\S)`: a run of whitespace is split so that all but the final
/// character joins the *preceding* chunk, and digits are unbounded.
const RE_GPT2: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// The cl100k_base pattern (GPT-3.5/4). Caps number runs at 3 digits, allows a
/// leading non-space before a letter run, and handles trailing newlines better.
const RE_CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The o200k_base pattern (GPT-4o). Splits number runs at 3 digits and gives
/// contractions and punctuation runs more room than cl100k.
const RE_O200K: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"\p{N}{1,3}|",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*|",
    r"\s*[\r\n]+|",
    r"\s+(?!\S)|",
    r"\s+"
);

/// Code-oriented pattern. Two deliberate departures from cl100k:
///
/// 1. A run of spaces is kept whole (` +`) instead of being severed by
///    `\s+(?!\S)`, so BPE can learn indentation units (4, 8, 12, 16 spaces) as
///    single tokens. Leading indentation is a large fraction of every code
///    corpus and the GPT-2 rule spends a token per indent level on it.
/// 2. Identifier runs may include `_` and digits, so `snake_case_name` and
///    `buf2` are single pre-token chunks that BPE can merge freely, rather than
///    being pre-split into fragments it is forbidden to join.
const RE_CODE: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
    r" ?[\p{L}_][\p{L}\p{N}_]*|",
    r"\p{N}{1,3}|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|",
    r"\s*[\r\n]+|",
    r" +|",
    r"\s+"
);

/// As `code`, but a newline carries the indentation of the line it opens
/// (`[\r\n]+ *`). Under every GPT-style pattern a line break and the following
/// indent are forced into separate chunks, so each line of an indented file
/// costs at least two tokens before any content. This is only learnable when
/// the trainer sees whole documents rather than single lines.
const RE_CODE_NL: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
    r" ?[\p{L}_][\p{L}\p{N}_]*|",
    r"\p{N}{1,3}|",
    r" ?[^\s\p{L}\p{N}]+|",
    r"[\r\n]+ *|",
    r" +|",
    r"\s+"
);

/// As `code_nl`, but a run of punctuation also absorbs the line break and
/// indent that follow it, so a statement terminator plus the next line's
/// indentation (`);\n        `) can become a single token.
const RE_CODE_NL2: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
    r" ?[\p{L}_][\p{L}\p{N}_]*|",
    r"\p{N}{1,3}|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]* *|",
    r"[\r\n]+ *|",
    r" +|",
    r"\s+"
);

/// Whole physical lines. The loosest regex-expressible chunking that still
/// bounds encode cost; every merge within a line is permitted.
const RE_LINE: &str = r"[^\r\n]*[\r\n]*";

fn regex_for(name: &str) -> Option<&'static str> {
    match name {
        "gpt2" => Some(RE_GPT2),
        "cl100k" => Some(RE_CL100K),
        "o200k" => Some(RE_O200K),
        "code" => Some(RE_CODE),
        "code_nl" => Some(RE_CODE_NL),
        "code_nl2" => Some(RE_CODE_NL2),
        "line" => Some(RE_LINE),
        "none" => None,
        other => panic!("unknown pretokenizer regex '{other}'"),
    }
}

// ---------------------------------------------------------------------------
// Tokenizer construction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildConfig {
    /// One of: gpt2, cl100k, o200k, code, none.
    pretokenizer: String,
    vocab_size: usize,
    min_freq: u64,
    /// Whether to seed the alphabet with all 256 byte-level characters.
    /// Without this, any byte absent from the training corpus is unencodable.
    full_byte_alphabet: bool,
    /// Whether to attach a ByteLevel decoder. Without it, decode() returns the
    /// byte-level surrogate characters instead of the original text.
    with_decoder: bool,
    add_prefix_space: bool,
}

/// Assemble the pre-tokenizer. When a custom regex is used we split on it
/// first, then apply ByteLevel with its own regex disabled so it only performs
/// the bytes-to-visible-chars mapping.
fn build_pretokenizer(cfg: &BuildConfig) -> PreTokenizerWrapper {
    let byte_level = ByteLevel::new(cfg.add_prefix_space, /* trim_offsets */ true, /* use_regex */ false);

    match cfg.pretokenizer.as_str() {
        "gpt2" => PreTokenizerWrapper::ByteLevel(ByteLevel::new(cfg.add_prefix_space, true, true)),
        "none" => PreTokenizerWrapper::ByteLevel(byte_level),
        name => {
            let pattern = regex_for(name).expect("named regex");
            let split = Split::new(
                SplitPattern::Regex(pattern.to_string()),
                SplitDelimiterBehavior::Isolated,
                /* invert */ false,
            )
            .expect("valid split regex");
            PreTokenizerWrapper::Sequence(PreSequence::new(vec![
                PreTokenizerWrapper::Split(split),
                PreTokenizerWrapper::ByteLevel(byte_level),
            ]))
        }
    }
}

fn build_tokenizer(cfg: &BuildConfig) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(tokenizers::ModelWrapper::BPE(BPE::default()));
    tokenizer.with_pre_tokenizer(Some(build_pretokenizer(cfg)));
    if cfg.with_decoder {
        tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevel::default())));
    }
    tokenizer
}

fn build_trainer(cfg: &BuildConfig) -> TrainerWrapper {
    let mut builder = BpeTrainer::builder()
        .vocab_size(cfg.vocab_size)
        .min_frequency(cfg.min_freq)
        .show_progress(false);
    if cfg.full_byte_alphabet {
        builder = builder.initial_alphabet(ByteLevel::alphabet().into_iter().collect());
    }
    TrainerWrapper::BpeTrainer(builder.build())
}

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct HeldoutDoc {
    path: String,
    bytes: usize,
    text: String,
}

fn load_heldout(path: &Path) -> Vec<HeldoutDoc> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    BufReader::new(file)
        .lines()
        .map(|line| serde_json::from_str(&line.expect("read line")).expect("parse jsonl"))
        .collect()
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct EvalReport {
    name: String,
    vocab_size: usize,
    total_bytes: usize,
    total_tokens: usize,
    bytes_per_token: f64,
    /// Sequence positions needed per megabyte of corpus — the number that
    /// directly scales pretraining cost.
    tokens_per_mib: f64,
    by_ext: HashMap<String, ExtStat>,
    roundtrip_exact: usize,
    roundtrip_total: usize,
    /// Encode throughput. Tokenizing a pretraining corpus is a real cost, so a
    /// design that compresses better but encodes far slower is a tradeoff, not
    /// a free win.
    encode_mib_per_s: f64,
}

#[derive(Debug, Serialize, Default, Clone)]
struct ExtStat {
    docs: usize,
    bytes: usize,
    tokens: usize,
    bytes_per_token: f64,
}

fn evaluate(name: &str, tokenizer: &Tokenizer, docs: &[HeldoutDoc], check_roundtrip: bool) -> EvalReport {
    let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();
    let started = std::time::Instant::now();
    let encodings = tokenizer
        .encode_batch_fast(texts, false)
        .expect("encode batch");
    let encode_secs = started.elapsed().as_secs_f64();

    let mut by_ext: HashMap<String, ExtStat> = HashMap::new();
    let mut total_bytes = 0usize;
    let mut total_tokens = 0usize;
    let mut roundtrip_exact = 0usize;
    let mut roundtrip_total = 0usize;

    for (doc, enc) in docs.iter().zip(encodings.iter()) {
        let n_tokens = enc.get_ids().len();
        total_bytes += doc.bytes;
        total_tokens += n_tokens;

        let ext = Path::new(&doc.path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("none")
            .to_string();
        let slot = by_ext.entry(ext).or_default();
        slot.docs += 1;
        slot.bytes += doc.bytes;
        slot.tokens += n_tokens;

        if check_roundtrip {
            roundtrip_total += 1;
            if let Ok(decoded) = tokenizer.decode(enc.get_ids(), false) {
                if decoded == doc.text {
                    roundtrip_exact += 1;
                }
            }
        }
    }

    for slot in by_ext.values_mut() {
        slot.bytes_per_token = slot.bytes as f64 / slot.tokens.max(1) as f64;
    }

    let bytes_per_token = total_bytes as f64 / total_tokens.max(1) as f64;
    EvalReport {
        name: name.to_string(),
        vocab_size: tokenizer.get_vocab_size(true),
        total_bytes,
        total_tokens,
        bytes_per_token,
        tokens_per_mib: total_tokens as f64 / (total_bytes as f64 / (1024.0 * 1024.0)),
        by_ext,
        roundtrip_exact,
        roundtrip_total,
        encode_mib_per_s: (total_bytes as f64 / (1024.0 * 1024.0)) / encode_secs.max(1e-9),
    }
}

fn print_report(report: &EvalReport, baseline: Option<f64>) {
    println!("\n=== {} ===", report.name);
    println!("  vocab            {}", report.vocab_size);
    println!("  held-out bytes   {}", report.total_bytes);
    println!("  tokens           {}", report.total_tokens);
    println!("  BYTES/TOKEN      {:.4}", report.bytes_per_token);
    println!("  tokens per MiB   {:.0}", report.tokens_per_mib);
    println!("  encode           {:.1} MiB/s", report.encode_mib_per_s);
    if let Some(base) = baseline {
        let reduction = 100.0 * (1.0 - base / report.bytes_per_token);
        println!("  vs baseline      {:+.2}% tokens", -reduction);
    }
    if report.roundtrip_total > 0 {
        println!(
            "  roundtrip exact  {}/{}",
            report.roundtrip_exact, report.roundtrip_total
        );
    }
    let mut exts: Vec<_> = report.by_ext.iter().collect();
    exts.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes));
    for (ext, stat) in exts {
        println!(
            "    .{:<5} {:>4} docs  {:>9} B  {:>8} tok  {:.3} B/tok",
            ext, stat.docs, stat.bytes, stat.tokens, stat.bytes_per_token
        );
    }
}


// ---------------------------------------------------------------------------
// Two-stage training
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Tokenizer compression benchmark for NSL")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train one tokenizer configuration and save it.
    Train {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "gpt2")]
        pretokenizer: String,
        #[arg(long, default_value_t = 32768)]
        vocab_size: usize,
        #[arg(long, default_value_t = 2)]
        min_freq: u64,
        #[arg(long, default_value_t = false)]
        no_byte_alphabet: bool,
        #[arg(long, default_value_t = false)]
        no_decoder: bool,
        #[arg(long, default_value_t = false)]
        add_prefix_space: bool,
        /// Feed whole documents to the trainer instead of single lines.
        #[arg(long, default_value_t = false)]
        whole_docs: bool,
    },
    /// Score a saved tokenizer against the held-out corpus.
    Eval {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        heldout: PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long, default_value_t = false)]
        roundtrip: bool,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Two-stage (word boundary, then relaxed) BPE training.
    Train2 {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Pre-tokenizer used for stage one.
        #[arg(long, default_value = "cl100k")]
        stage1: String,
        /// Pre-tokenizer used for stage two and shipped with the tokenizer.
        #[arg(long, default_value = "line")]
        relaxed: String,
        #[arg(long, default_value_t = 49152)]
        vocab_size: usize,
        /// Vocabulary size at which the boundary is relaxed.
        #[arg(long, default_value_t = 24576)]
        transition: usize,
        #[arg(long, default_value_t = 2)]
        min_freq: u64,
        /// Longest token surface allowed, in bytes. 0 disables the cap.
        #[arg(long, default_value_t = 0)]
        max_token_bytes: usize,
    },
    /// Encode a corpus into a flat u16 token stream for pretraining.
    Pretokenize {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Report the token-length profile of a trained vocabulary.
    VocabStats {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        heldout: Option<PathBuf>,
    },
    /// Reproduce exactly what NSL's runtime builds today, and score it.
    NslBaseline {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        heldout: PathBuf,
        #[arg(long, default_value_t = 32768)]
        vocab_size: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

/// Split the packed corpus back into documents.
fn read_documents(corpus: &Path) -> Vec<String> {
    let raw = std::fs::read_to_string(corpus).unwrap_or_else(|e| panic!("read corpus: {e}"));
    raw.split(DOC_SEP_CHAR).map(|s| s.to_string()).collect()
}

const DOC_SEP_CHAR: char = '\u{0}';

/// Apply a pre-tokenizer to a string, yielding the chunk strings the BPE
/// trainer will count. This mirrors what `Tokenizer::train_from_files` does
/// internally, but lets us choose the unit we feed.
fn pre_tokenize_to_words(pre: &PreTokenizerWrapper, text: &str) -> tokenizers::Result<Vec<String>> {
    let mut pts = PreTokenizedString::from(text);
    pre.pre_tokenize(&mut pts)?;
    Ok(pts
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(s, _, _)| s.to_owned())
        .collect())
}

fn train_and_save(corpus: &Path, out: &Path, cfg: &BuildConfig, whole_docs: bool) -> Tokenizer {
    let mut tokenizer = build_tokenizer(cfg);
    let mut trainer = build_trainer(cfg);

    if whole_docs {
        // Feed entire documents so that merges may span line boundaries.
        // `train_from_files` reads one line at a time, which makes a newline
        // plus the following indentation structurally unlearnable.
        let docs = read_documents(corpus);
        let pre = build_pretokenizer(cfg);
        trainer
            .feed(docs.iter(), |text| pre_tokenize_to_words(&pre, text))
            .unwrap_or_else(|e| panic!("feed: {e}"));
        let mut model = tokenizers::ModelWrapper::BPE(BPE::default());
        let special = trainer
            .train(&mut model)
            .unwrap_or_else(|e| panic!("train: {e}"));
        tokenizer.with_model(model);
        tokenizer.add_special_tokens(&special);
    } else {
        tokenizer
            .train_from_files(&mut trainer, vec![corpus.display().to_string()])
            .unwrap_or_else(|e| panic!("train: {e}"));
    }

    tokenizer.save(out, false).unwrap_or_else(|e| panic!("save: {e}"));
    tokenizer
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Train {
            corpus,
            out,
            pretokenizer,
            vocab_size,
            min_freq,
            no_byte_alphabet,
            no_decoder,
            add_prefix_space,
            whole_docs,
        } => {
            let cfg = BuildConfig {
                pretokenizer,
                vocab_size,
                min_freq,
                full_byte_alphabet: !no_byte_alphabet,
                with_decoder: !no_decoder,
                add_prefix_space,
            };
            let started = std::time::Instant::now();
            let tok = train_and_save(&corpus, &out, &cfg, whole_docs);
            println!(
                "trained {} ({} merges reachable, vocab {}) in {:.1}s -> {}",
                cfg.pretokenizer,
                cfg.vocab_size,
                tok.get_vocab_size(true),
                started.elapsed().as_secs_f64(),
                out.display()
            );
        }
        Command::Eval {
            tokenizer,
            heldout,
            name,
            roundtrip,
            json_out,
        } => {
            let tok = Tokenizer::from_file(&tokenizer).expect("load tokenizer");
            let docs = load_heldout(&heldout);
            let label = name.unwrap_or_else(|| tokenizer.display().to_string());
            let report = evaluate(&label, &tok, &docs, roundtrip);
            print_report(&report, None);
            if let Some(path) = json_out {
                let mut f = File::create(&path).expect("create json out");
                f.write_all(serde_json::to_string_pretty(&report).unwrap().as_bytes())
                    .expect("write json");
            }
        }
        Command::Train2 {
            corpus,
            out,
            stage1,
            relaxed,
            vocab_size,
            transition,
            min_freq,
            max_token_bytes,
        } => {
            // Delegates to nsl-runtime's trainer so this benchmark measures the
            // shipped code path. The runtime relaxes to whole lines, so the
            // `relaxed` flag only selects between that and no relaxation.
            use nsl_runtime::tokenizer_bpe as bpe;

            let stage1_kind = bpe::PreTokenizerKind::parse(&stage1)
                .unwrap_or_else(|| panic!("unsupported stage-1 pre-tokenizer '{stage1}';                                            the shipped trainer offers gpt2, cl100k, line"));
            assert_eq!(
                relaxed, "line",
                "the shipped trainer always relaxes to whole lines; pass --relaxed line"
            );

            let spec = bpe::TrainSpec {
                vocab_size,
                min_frequency: min_freq,
                transition,
                max_token_bytes,
                stage1: stage1_kind,
                special_tokens: Vec::new(),
            };

            let started = std::time::Instant::now();
            let docs = read_documents(&corpus);
            let trained = bpe::train_two_stage(docs, &spec);
            eprintln!(
                "trained: vocab {}, {} merges ({} before relaxation) in {:.1}s",
                trained.vocab.len(),
                trained.merges.len(),
                trained.stage1_merges,
                started.elapsed().as_secs_f64()
            );
            let tokenizer = bpe::assemble(&trained, &spec).expect("assemble");
            tokenizer.save(&out, false).expect("save tokenizer");
            println!("saved {}", out.display());
        }
        Command::Pretokenize { tokenizer, corpus, out } => {
            let tok = Tokenizer::from_file(&tokenizer).expect("load tokenizer");
            let vocab = tok.get_vocab_size(true);
            assert!(
                vocab <= u16::MAX as usize + 1,
                "vocab {vocab} does not fit the u16 token stream format"
            );
            let docs = read_documents(&corpus);
            let refs: Vec<&str> = docs.iter().map(|d| d.as_str()).collect();
            let started = std::time::Instant::now();
            let encs = tok.encode_batch_fast(refs, false).expect("encode");
            let mut bytes: Vec<u8> = Vec::new();
            let mut n_tokens = 0usize;
            for enc in &encs {
                for id in enc.get_ids() {
                    bytes.extend_from_slice(&(*id as u16).to_le_bytes());
                    n_tokens += 1;
                }
            }
            std::fs::write(&out, &bytes).expect("write token stream");
            let corpus_bytes: usize = docs.iter().map(|d| d.len()).sum();
            println!(
                "{} tokens from {:.2} MB ({:.3} B/tok) in {:.1}s -> {}",
                n_tokens,
                corpus_bytes as f64 / 1e6,
                corpus_bytes as f64 / n_tokens as f64,
                started.elapsed().as_secs_f64(),
                out.display()
            );
        }
        Command::VocabStats { tokenizer, heldout } => {
            let tok = Tokenizer::from_file(&tokenizer).expect("load tokenizer");
            let vocab = tok.get_vocab(true);
            let mut by_len: HashMap<usize, usize> = HashMap::new();
            let mut with_newline = 0usize;
            let mut longest: Vec<(usize, String)> = Vec::new();
            for surface in vocab.keys() {
                // Byte-level surfaces spell a newline as a visible surrogate.
                let n = surface.chars().count();
                *by_len.entry(n).or_default() += 1;
                if surface.contains('\u{010A}') || surface.contains('\n') {
                    with_newline += 1;
                }
                longest.push((n, surface.clone()));
            }
            longest.sort_by(|a, b| b.0.cmp(&a.0));

            let total = vocab.len();
            println!("vocab {total}, tokens containing a newline: {with_newline} ({:.1}%)",
                     100.0 * with_newline as f64 / total as f64);
            let mut lens: Vec<_> = by_len.into_iter().collect();
            lens.sort();
            println!("  length histogram (chars -> count):");
            for (len, count) in &lens {
                if *count > 0 {
                    println!("    {len:>3}  {count:>6}  {:>5.1}%", 100.0 * *count as f64 / total as f64);
                }
            }
            println!("  10 longest tokens:");
            for (len, surface) in longest.iter().take(10) {
                println!("    {len:>3}  {:?}", surface);
            }

            if let Some(path) = heldout {
                // What share of held-out *text* is carried by long tokens?
                let docs = load_heldout(&path);
                let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();
                let encs = tok.encode_batch_fast(texts, false).expect("encode");
                let id_to_len: HashMap<u32, usize> =
                    vocab.iter().map(|(s, id)| (*id, s.chars().count())).collect();
                let mut used: HashMap<usize, usize> = HashMap::new();
                let mut n_tokens = 0usize;
                for enc in &encs {
                    for id in enc.get_ids() {
                        let len = id_to_len.get(id).copied().unwrap_or(0);
                        *used.entry(len.min(32)).or_default() += 1;
                        n_tokens += 1;
                    }
                }
                let mut rows: Vec<_> = used.into_iter().collect();
                rows.sort();
                println!("  held-out token usage by length (capped at 32):");
                let mut cum = 0usize;
                for (len, count) in rows {
                    cum += count;
                    println!("    {len:>3}  {count:>8}  {:>5.1}%  cum {:>5.1}%",
                             100.0 * count as f64 / n_tokens as f64,
                             100.0 * cum as f64 / n_tokens as f64);
                }
            }
        }
        Command::NslBaseline {
            corpus,
            heldout,
            vocab_size,
            out,
        } => {
            // Exactly what crates/nsl-runtime/src/tokenizer.rs nsl_bpe_train does:
            // BPE::default() + ByteLevel::default() pre-tokenizer, no decoder,
            // no post-processor, no initial alphabet.
            let cfg = BuildConfig {
                pretokenizer: "gpt2".into(),
                vocab_size,
                min_freq: 2,
                full_byte_alphabet: false,
                with_decoder: false,
                add_prefix_space: true,
            };
            let path = out.unwrap_or_else(|| PathBuf::from("nsl_baseline.json"));
            let tok = train_and_save(&corpus, &path, &cfg, false);
            let docs = load_heldout(&heldout);
            let report = evaluate("NSL baseline (as shipped)", &tok, &docs, true);
            print_report(&report, None);
        }
    }
}
