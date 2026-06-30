//! `nsl tokenize` — train a BPE tokenizer over NSL source directories.
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use std::path::PathBuf;
use std::process;

pub(crate) fn run_tokenize(dirs: &[String], output: &std::path::Path, vocab_size: usize, min_freq: u64, ext: &str) {
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

    // Train BPE tokenizer using the runtime's nsl_bpe_train
    eprintln!("[tokenize] Training BPE tokenizer (vocab_size={}, min_freq={})...", vocab_size, min_freq);

    // We call the runtime's BPE trainer directly via Rust (not through FFI)
    use tokenizers::models::bpe::{BPE, BpeTrainer};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::Tokenizer;

    let trainer = BpeTrainer::builder()
        .vocab_size(vocab_size)
        .min_frequency(min_freq)
        .special_tokens(vec![
            tokenizers::AddedToken::from("<|endoftext|>".to_string(), true),
            tokenizers::AddedToken::from("<|padding|>".to_string(), true),
            tokenizers::AddedToken::from("<|fim_prefix|>".to_string(), true),
            tokenizers::AddedToken::from("<|fim_middle|>".to_string(), true),
            tokenizers::AddedToken::from("<|fim_suffix|>".to_string(), true),
        ])
        .build();

    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::PreTokenizerWrapper::ByteLevel(ByteLevel::default()),
    ));

    let mut trainer_wrapper = tokenizers::models::TrainerWrapper::BpeTrainer(trainer);
    match tokenizer.train_from_files(&mut trainer_wrapper, vec![corpus_path.to_string_lossy().to_string()]) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: BPE training failed: {e}");
            process::exit(1);
        }
    }

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
