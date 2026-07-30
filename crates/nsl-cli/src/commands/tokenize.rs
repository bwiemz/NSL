//! `nsl tokenize` — train a BPE tokenizer over NSL source directories.
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use std::path::PathBuf;
use std::process;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_tokenize(
    dirs: &[String],
    output: &std::path::Path,
    vocab_size: usize,
    min_freq: u64,
    ext: &str,
    transition: usize,
    max_token_bytes: usize,
) {
    use std::io::Write;

    // Default directories if none specified
    let search_dirs: Vec<String> = if dirs.is_empty() {
        vec!["stdlib".into(), "examples".into(), "tests".into(), "models".into()]
    } else {
        dirs.to_vec()
    };

    // Collect all source files
    let mut source_files: Vec<PathBuf> = Vec::new();
    for dir in &search_dirs {
        let dir_path = PathBuf::from(dir);
        if !dir_path.exists() {
            eprintln!("warning: directory '{}' not found, skipping", dir);
            continue;
        }
        collect_files_recursive(&dir_path, ext, &mut source_files);
    }
    source_files.sort();

    if source_files.is_empty() {
        eprintln!("error: no .{ext} files found in {:?}", search_dirs);
        process::exit(1);
    }

    eprintln!("[tokenize] Found {} .{} files across {} directories", source_files.len(), ext, search_dirs.len());

    // Concatenate all source text into a temporary corpus file
    let corpus_path = std::env::temp_dir().join("nsl_tokenizer_corpus.txt");
    {
        let mut corpus = std::fs::File::create(&corpus_path).unwrap_or_else(|e| {
            eprintln!("error: could not create corpus file: {e}");
            process::exit(1);
        });
        let mut total_bytes: usize = 0;
        for file in &source_files {
            match std::fs::read_to_string(file) {
                Ok(content) => {
                    total_bytes += content.len();
                    let _ = corpus.write_all(content.as_bytes());
                    let _ = corpus.write_all(b"\n");
                }
                Err(e) => {
                    eprintln!("warning: could not read '{}': {e}", file.display());
                }
            }
        }
        eprintln!("[tokenize] Corpus: {} bytes from {} files", total_bytes, source_files.len());
    }

    // Train through the runtime's two-stage trainer, which is the tokenizer the
    // project actually ships. This command previously built its own
    // `BpeTrainer` inline with a ByteLevel pre-tokenizer and no decoder, so the
    // trainer's design work — the relaxed second stage, the seeded byte
    // alphabet, the attached decoder — was unreachable from the CLI.
    let spec = nsl_runtime::tokenizer_bpe::TrainSpec {
        vocab_size,
        min_frequency: min_freq,
        transition,
        max_token_bytes,
        stage1: nsl_runtime::tokenizer_bpe::PreTokenizerKind::Cl100k,
        special_tokens: vec![
            "<|endoftext|>".to_string(),
            "<|padding|>".to_string(),
            "<|fim_prefix|>".to_string(),
            "<|fim_middle|>".to_string(),
            "<|fim_suffix|>".to_string(),
        ],
    };
    if transition >= vocab_size {
        eprintln!(
            "[tokenize] Training BPE tokenizer (vocab_size={vocab_size}, min_freq={min_freq}, \
             word-bounded)..."
        );
    } else {
        eprintln!(
            "[tokenize] Training two-stage BPE tokenizer (vocab_size={vocab_size}, \
             min_freq={min_freq}, transition={transition})..."
        );
    }

    let tokenizer = match nsl_runtime::tokenizer_bpe::train_from_file(
        corpus_path.to_string_lossy().as_ref(),
        &spec,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: BPE training failed: {e}");
            process::exit(1);
        }
    };

    // Ensure output directory exists
    if let Some(parent) = output.parent() {
        if !parent.exists() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    // Save tokenizer
    match tokenizer.save(output.to_string_lossy().as_ref(), true) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: could not save tokenizer to '{}': {e}", output.display());
            process::exit(1);
        }
    }

    let final_vocab = tokenizer.get_vocab_size(true);
    eprintln!("[tokenize] Saved tokenizer to '{}' (vocab_size={})", output.display(), final_vocab);

    // Clean up corpus
    let _ = std::fs::remove_file(&corpus_path);

    // Test: encode a sample string
    let sample = "fn forward(self, x: Tensor) -> Tensor:";
    if let Ok(encoding) = tokenizer.encode(sample, false) {
        let tokens = encoding.get_tokens();
        eprintln!("[tokenize] Sample: \"{}\" -> {} tokens: {:?}", sample, tokens.len(), &tokens[..tokens.len().min(10)]);
    }
}

fn collect_files_recursive(dir: &PathBuf, ext: &str, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(&path, ext, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}
