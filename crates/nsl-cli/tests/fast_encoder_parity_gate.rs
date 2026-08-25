//! Same-token parity: the fast byte-domain encoder vs the `tokenizers` crate,
//! on the committed tokenizer of record (item 16).
//!
//! WHY. The fast encoder (`nsl_runtime::tokenizer_fast`) exists to replace
//! the HF encode path for corpus generation, where every corpus fingerprint
//! in `models/datasets/CORPUS_MANIFEST_v2.json` is downstream of the exact
//! token stream. "Roughly the same tokenization" is not a property — either
//! every document encodes to identical ids or the backend cannot be used.
//! These gates run the two encoders side by side in every CI build:
//!
//!  * a fixed edge-case battery aimed at the seams — line-split boundaries
//!    (`\r`, `\r\n`, blank-line runs, missing trailing newline), added-token
//!    surfaces whole/partial/adjacent/repeated, non-ASCII (CJK, RTL digits,
//!    combining marks, NBSP), and cache-interaction shapes (repeated lines,
//!    lines longer than the memo cap);
//!  * a seeded generated corpus (~2 MB) mixing those ingredients at random,
//!    so the battery cannot overfit to hand-picked cases. The generator is
//!    a plain xorshift — deterministic, no external dep, same stream on
//!    every run and platform.
//!
//! The cache is deliberately SHARED across all documents in each test: a
//! cache-poisoning bug (hit returning another line's ids) is exactly the
//! kind of failure the differential must catch, so the encoder runs in its
//! production configuration, not a cache-free one.

use std::path::PathBuf;

use nsl_runtime::tokenizer_fast::{EncodeCache, FastBpeEncoder};
use tokenizers::Tokenizer;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn tokenizer_of_record() -> PathBuf {
    repo_root().join("models/tokenizers/nsl_mix_v2_t40960_v49152.json")
}

fn load_both() -> (FastBpeEncoder, Tokenizer) {
    let path = tokenizer_of_record();
    let fast = FastBpeEncoder::from_tokenizer_file(&path)
        .unwrap_or_else(|e| panic!("fast encoder refused the tokenizer of record: {e}"));
    let hf = Tokenizer::from_file(&path)
        .unwrap_or_else(|e| panic!("cannot load {}: {e}", path.display()));
    (fast, hf)
}

fn assert_parity(fast: &FastBpeEncoder, hf: &Tokenizer, cache: &mut EncodeCache, doc: &str) {
    let mut got = Vec::new();
    fast.encode_doc_into(doc, cache, &mut got);
    let want: Vec<u16> = hf
        .encode_fast(doc, false)
        .expect("hf encode failed")
        .get_ids()
        .iter()
        .map(|&id| id as u16)
        .collect();
    if got != want {
        // Locate the first divergence for a debuggable failure.
        let k = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        panic!(
            "fast != hf on doc {:?}\n first divergence at token {k}: fast {:?} vs hf {:?}\n \
             fast len {} vs hf len {}",
            &doc[..doc.len().min(120)],
            &got[k..(k + 8).min(got.len())],
            &want[k..(k + 8).min(want.len())],
            got.len(),
            want.len()
        );
    }
}

const EDGE_CASES: &[&str] = &[
    "",
    "\n",
    "\r",
    "\r\n",
    "\n\n\n",
    " ",
    "   \n  \n",
    "no trailing newline",
    "hello world\n",
    "let x = 1\n    let y = 2\n        return x + y\n",
    "\tindented with tabs\n\t\tdeeper\n",
    "trailing spaces   \nand more  ",
    "a\r\nb\rc\nd",
    // Added-token surfaces: whole, adjacent, repeated, at both ends.
    "<|endoftext|>",
    "<|endoftext|><|endoftext|>",
    "x<|endoftext|>y",
    "<|file_sep|>src/main.rs\nfn main() {}\n",
    "a <|im_start|>user\nhi<|im_end|> b",
    "<|tool_call|>{\"name\":\"f\"}<|tool_result|>ok<|pad|>",
    "ends with surface<|endoftext|>",
    // Partial / lookalike surfaces must NOT extract.
    "<|endo",
    "<|endoftext|",
    "<|EndOfText|>",
    "<|file_sep|x",
    "< |endoftext|>",
    "<<|endoftext|>>",
    // The invisible-separator neutralization the extractor applies.
    "<|\u{2063}endoftext|>",
    // Non-ASCII: CJK (no spaces), RTL digits, combining marks, NBSP,
    // unicode whitespace, astral plane.
    "日本語のテキストです。改行\nもある。",
    "١٢٣٤٥ عربى\n",
    "e\u{301}f a\u{300}\n",
    "nbsp\u{a0}word\n",
    "a\u{2028}b\u{2029}c\n",
    "emoji 🚀🔥 and beyond 𝄞\n",
    "café voilà ¡hola!\n",
    // Contractions and cl100k-ish shapes (stage-1 heritage).
    "don't they'll we've it's\n",
    "1234567890 123 12 1\n",
    "price: $5.99! (50% off)\n",
    // Repeated lines (cache hits) and a line beyond the memo cap.
    "}\n}\n}\n}\n",
    "import numpy as np\nimport numpy as np\n",
    // 200 bytes, above MAX_CACHED_LINE_BYTES.
    "this line is deliberately much longer than the one hundred and twenty eight byte cache \
     admission cap so it exercises the uncached path every single time it appears in a document\n",
];

#[test]
fn fast_encoder_matches_hf_on_edge_cases() {
    let (fast, hf) = load_both();
    let mut cache = EncodeCache::with_budget(16 << 20);
    for doc in EDGE_CASES {
        assert_parity(&fast, &hf, &mut cache, doc);
    }
    // Second pass over the same battery: every cacheable line is now a hit,
    // so divergence here isolates cache poisoning specifically.
    for doc in EDGE_CASES {
        assert_parity(&fast, &hf, &mut cache, doc);
    }
}

/// Deterministic xorshift64* — no external dep, identical stream everywhere.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const WORDS: &[&str] = &[
    "the", "of", "tensor", "gradient", "return", "let", "fn", "self", "import",
    "während", "数据", "モデル", "vector", "Ġ-free", "naïve", "don't", "12",
    "1234", "0x7f", "===", "-->", "();", "```", "<div>", "&amp;", "über",
];
const SEPARATORS: &[&str] = &[" ", "  ", "\t", ", ", " = ", "::", ".", " |> "];
const LINE_ENDS: &[&str] = &["\n", "\n", "\n", "\r\n", "\r", "\n\n", "\n\n\n", ""];
const SURFACES: &[&str] = &[
    "<|endoftext|>", "<|file_sep|>", "<|im_start|>", "<|im_end|>",
    "<|pad|>", "<|tool_call|>", "<|tool_result|>",
    // Near-misses, generated at the same rate as the real thing.
    "<|endoftext|", "<|file_sep", "<|im_star", "<|", "|>", "<",
];
const INDENTS: &[&str] = &["", "    ", "        ", "            ", "\t", "  "];

fn generate_doc(rng: &mut Rng) -> String {
    let mut doc = String::new();
    for _ in 0..1 + rng.below(40) {
        doc.push_str(INDENTS[rng.below(INDENTS.len())]);
        for _ in 0..rng.below(12) {
            doc.push_str(WORDS[rng.below(WORDS.len())]);
            doc.push_str(SEPARATORS[rng.below(SEPARATORS.len())]);
        }
        if rng.below(6) == 0 {
            doc.push_str(SURFACES[rng.below(SURFACES.len())]);
        }
        doc.push_str(LINE_ENDS[rng.below(LINE_ENDS.len())]);
    }
    doc
}

#[test]
fn fast_encoder_matches_hf_on_a_generated_corpus() {
    let (fast, hf) = load_both();
    let mut cache = EncodeCache::with_budget(64 << 20);
    let mut rng = Rng(0x5EED_2026_0824);
    let mut total_bytes = 0usize;
    let mut docs = 0usize;
    while total_bytes < 2 << 20 {
        let doc = generate_doc(&mut rng);
        total_bytes += doc.len();
        docs += 1;
        assert_parity(&fast, &hf, &mut cache, &doc);
    }
    assert!(docs > 200, "generator degenerated: only {docs} docs");
}
