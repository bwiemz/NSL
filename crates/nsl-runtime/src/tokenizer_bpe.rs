//! Tokenizer construction and two-stage BPE training.
//!
//! # Why this exists
//!
//! Standard BPE never learns a token that spans a pre-tokenizer boundary. In
//! source code that means a line break and the indentation opening the next
//! line are always at least two tokens, however often they co-occur — and
//! indentation is both a large fraction of a code corpus and almost perfectly
//! predictable. Under the GPT-2 split pattern the compiler's own sources cost
//! about 3.6 bytes per token; simply allowing merges to cross that boundary
//! raises it to about 4.4 on unseen projects and about 6.4 in domain.
//!
//! Relaxing the boundary all the way is not free: merges become able to
//! swallow whole idiomatic lines, which compresses the training distribution
//! far better than anything else and transfers markedly worse. So training
//! runs in two stages. Stage one uses a word-level pattern and learns ordinary
//! subword units; at [`TrainSpec::transition`] the boundary widens to whole
//! lines and the rest of the vocabulary is spent on cross-boundary units built
//! on top of a compositional inventory rather than instead of one.
//!
//! # A known approximation in stage two
//!
//! Stage two seeds each relaxed chunk by concatenating the stage-one encodings
//! of its words. That is cheap — it reuses work stage one already did instead of
//! re-merging from bytes — but it is not exactly the state the encoder reaches,
//! because BPE is not compositional across a boundary once merges may cross it.
//!
//! Concretely, the line `"    let"` splits into cl100k words `"ĠĠĠ"` and
//! `"Ġlet"`. Stage one may have learned `("ĠĠ", "Ġ") -> "ĠĠĠ"`, so the seed is
//! `["ĠĠĠ", ...]`. At encode time the merge list runs over the whole line, where
//! the lower-ranked `("Ġ", "Ġ") -> "ĠĠ"` fires first and yields `ĠĠ ĠĠ ...`
//! instead. Stage two therefore collects its pair statistics over a segmentation
//! the encoder does not always occupy.
//!
//! This costs compression rather than correctness: encoding stays deterministic
//! and round-trips, and the measured figures above come from scoring the saved
//! tokenizer through the real encoder, so they hold as reported. But it means
//! stage two is somewhat mis-optimised, and re-encoding each relaxed chunk from
//! bytes at the transition is the most likely source of further gain.
//!
//! # Invariants this module is responsible for
//!
//! * A [`Decoder`] is always attached. Without one `Tokenizer::decode` joins
//!   the raw byte-level surfaces with spaces, so decode(encode(x)) never
//!   returns x for any text at all.
//! * The alphabet is always seeded with all 256 byte-level characters, so no
//!   byte is unrepresentable merely because the training corpus lacked it.
//!   Un-seeded, a byte absent from training is silently dropped at encode time.

use std::collections::HashMap;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::{FxHashMap, FxHashSet};
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{
    AddedToken, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer,
    PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer,
};

// ---------------------------------------------------------------------------
// Pre-tokenizer designs
// ---------------------------------------------------------------------------

/// The cl100k_base pattern. Caps digit runs at three and keeps punctuation
/// runs with any trailing newlines.
const RE_CL100K: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
    r"[^\r\n\p{L}\p{N}]?\p{L}+|",
    r"\p{N}{1,3}|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|",
    r"\s*[\r\n]+|",
    r"\s+(?!\S)|",
    r"\s+"
);

/// Whole physical lines: the loosest chunking that still bounds encode cost,
/// since every merge stays within one line.
const RE_LINE: &str = r"[^\r\n]*[\r\n]*";

/// How the corpus is cut up before merges are considered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizerKind {
    /// The GPT-2 word pattern. It has no regex constant here because
    /// `ByteLevel` applies it internally and the two are not separable.
    Gpt2,
    /// cl100k_base word pattern.
    Cl100k,
    /// One chunk per physical line.
    Line,
}

impl PreTokenizerKind {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "gpt2" => Some(Self::Gpt2),
            "cl100k" => Some(Self::Cl100k),
            "line" => Some(Self::Line),
            _ => None,
        }
    }

    fn regex(self) -> Option<&'static str> {
        match self {
            Self::Gpt2 => None,
            Self::Cl100k => Some(RE_CL100K),
            Self::Line => Some(RE_LINE),
        }
    }

    /// The full pre-tokenizer, including the byte-level mapping.
    pub fn pre_tokenizer(self) -> PreTokenizerWrapper {
        match self.regex() {
            // ByteLevel's own regex is not separable from its mapping.
            None => PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, true)),
            Some(pattern) => {
                let split = Split::new(
                    SplitPattern::Regex(pattern.to_string()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .expect("built-in split pattern is valid");
                PreTokenizerWrapper::Sequence(PreSequence::new(vec![
                    PreTokenizerWrapper::Split(split),
                    // use_regex=false: the split above already chose the chunks,
                    // so ByteLevel is only performing the bytes-to-surrogates
                    // mapping here.
                    PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
                ]))
            }
        }
    }
}

/// Split `text` into the chunks this design produces, with the byte-level
/// mapping applied — i.e. exactly the strings BPE will merge over.
fn chunk_to_surfaces(pre: &PreTokenizerWrapper, text: &str) -> Vec<String> {
    let mut pts = PreTokenizedString::from(text);
    if pre.pre_tokenize(&mut pts).is_err() {
        return Vec::new();
    }
    pts.get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(s, _, _)| s.to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// Everything that determines the shape of the learned vocabulary.
#[derive(Debug, Clone)]
pub struct TrainSpec {
    pub vocab_size: usize,
    pub min_frequency: u64,
    /// Vocabulary size at which the boundary widens from `stage1` to whole
    /// lines. At or above `vocab_size` the boundary never widens, which
    /// reproduces ordinary BPE.
    pub transition: usize,
    /// Longest token surface permitted, in bytes; 0 disables the cap. Bounding
    /// token length is what keeps cross-boundary units useful rather than
    /// letting them become whole-line lookups.
    pub max_token_bytes: usize,
    pub stage1: PreTokenizerKind,
    pub special_tokens: Vec<String>,
}

impl Default for TrainSpec {
    fn default() -> Self {
        Self {
            vocab_size: 49152,
            min_frequency: 2,
            // 5/6 of the vocabulary is spent on ordinary subwords, the last
            // sixth on cross-boundary units. Measured on the compiler sources
            // plus twenty unrelated local projects, this keeps essentially all
            // of the compression that full relaxation buys while cutting the
            // in-domain/out-of-domain gap from 1.44x to 1.24x.
            transition: 40960,
            max_token_bytes: 0,
            stage1: PreTokenizerKind::Cl100k,
            special_tokens: Vec::new(),
        }
    }
}

pub type Pair = (u32, u32);

/// Incremental pair statistics with a lazily-invalidated priority queue.
///
/// `counts` is exact. The heap may hold superseded entries, which are dropped
/// on pop by re-checking against `counts`; `words_with` may name words that no
/// longer contain a pair, which costs a wasted scan but never a wrong answer.
struct Stats {
    counts: FxHashMap<Pair, i64>,
    words_with: FxHashMap<Pair, FxHashSet<u32>>,
    heap: BinaryHeap<(i64, Reverse<Pair>)>,
}

impl Stats {
    fn new() -> Self {
        Self {
            counts: FxHashMap::default(),
            words_with: FxHashMap::default(),
            heap: BinaryHeap::new(),
        }
    }

    fn bump(&mut self, pair: Pair, delta: i64, word: u32) {
        let entry = self.counts.entry(pair).or_insert(0);
        *entry += delta;
        let now = *entry;
        if now <= 0 {
            self.counts.remove(&pair);
        } else {
            self.heap.push((now, Reverse(pair)));
        }
        if delta > 0 {
            self.words_with.entry(pair).or_default().insert(word);
        }
    }

    fn pop_best(&mut self) -> Option<(Pair, i64)> {
        while let Some((count, Reverse(pair))) = self.heap.pop() {
            if self.counts.get(&pair) == Some(&count) {
                return Some((pair, count));
            }
        }
        None
    }

    fn forget(&mut self, pair: Pair) {
        self.counts.remove(&pair);
        self.words_with.remove(&pair);
    }
}

struct Learner {
    vocab: Vec<String>,
    lookup: FxHashMap<String, u32>,
    merges: Vec<Pair>,
    min_frequency: u64,
    max_token_bytes: usize,
}

impl Learner {
    fn new(alphabet: &[String], min_frequency: u64, max_token_bytes: usize) -> Self {
        let mut vocab = Vec::with_capacity(alphabet.len());
        let mut lookup = FxHashMap::default();
        for symbol in alphabet {
            if !lookup.contains_key(symbol) {
                lookup.insert(symbol.clone(), vocab.len() as u32);
                vocab.push(symbol.clone());
            }
        }
        Self {
            vocab,
            lookup,
            merges: Vec::new(),
            min_frequency,
            max_token_bytes,
        }
    }

    /// Rewrite one word, merging every non-overlapping occurrence of `pair`,
    /// patching pair counts only at the merge sites.
    fn merge_in_word(
        symbols: &[u32],
        pair: Pair,
        new_id: u32,
        count: u64,
        index: u32,
        stats: &mut Stats,
    ) -> Option<Vec<u32>> {
        if symbols.len() < 2 {
            return None;
        }
        let mut out: Vec<u32> = Vec::with_capacity(symbols.len());
        let mut merged_any = false;
        let mut i = 0;
        let delta = count as i64;

        while i < symbols.len() {
            if i + 1 < symbols.len() && symbols[i] == pair.0 && symbols[i + 1] == pair.1 {
                merged_any = true;
                if let Some(&prev) = out.last() {
                    stats.bump((prev, pair.0), -delta, index);
                    stats.bump((prev, new_id), delta, index);
                }
                if i + 2 < symbols.len() {
                    // Read from the original sequence. If the next position
                    // opens another occurrence, the pair added here is undone
                    // when that occurrence is handled on the following turn.
                    let next = symbols[i + 2];
                    stats.bump((pair.1, next), -delta, index);
                    stats.bump((new_id, next), delta, index);
                }
                out.push(new_id);
                i += 2;
            } else {
                out.push(symbols[i]);
                i += 1;
            }
        }

        merged_any.then_some(out)
    }

    fn run(&mut self, words: &mut [(Vec<u32>, u64)], target_vocab: usize) {
        let mut stats = Stats::new();
        for (index, (symbols, count)) in words.iter().enumerate() {
            for window in symbols.windows(2) {
                stats.bump((window[0], window[1]), *count as i64, index as u32);
            }
        }

        while self.vocab.len() < target_vocab {
            let Some((pair, count)) = stats.pop_best() else {
                break;
            };
            if count < self.min_frequency as i64 {
                break;
            }

            let left = &self.vocab[pair.0 as usize];
            let right = &self.vocab[pair.1 as usize];
            // One char of a byte-level surface is one byte of original text.
            // `len()` would count the mapping's UTF-8 width instead — a space is
            // U+0120, two bytes — so a cap of 8 would admit only 4 spaces of
            // indentation, which is exactly the case the cap exists to bound.
            let surface_bytes = left.chars().count() + right.chars().count();
            if self.max_token_bytes > 0 && surface_bytes > self.max_token_bytes {
                stats.forget(pair);
                continue;
            }
            let surface = format!("{left}{right}");
            if self.lookup.contains_key(&surface) {
                stats.forget(pair);
                continue;
            }

            let new_id = self.vocab.len() as u32;
            self.lookup.insert(surface.clone(), new_id);
            self.vocab.push(surface);
            self.merges.push(pair);

            let affected: Vec<u32> = stats
                .words_with
                .get(&pair)
                .map(|set| set.iter().copied().collect())
                .unwrap_or_default();
            for index in affected {
                let slot = &mut words[index as usize];
                if let Some(rewritten) =
                    Self::merge_in_word(&slot.0, pair, new_id, slot.1, index, &mut stats)
                {
                    slot.0 = rewritten;
                }
            }
            stats.forget(pair);
        }
    }
}

/// Result of training: a vocabulary and an ordered merge list.
pub struct TrainedBpe {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
    /// How many merges were learned before the boundary widened.
    pub stage1_merges: usize,
}

/// Train over an iterator of relaxed chunks — in practice, corpus lines with
/// their line endings retained, which is exactly what [`PreTokenizerKind::Line`]
/// produces at encode time.
pub fn train_two_stage<I, S>(chunks: I, spec: &TrainSpec) -> TrainedBpe
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let stage1_pre = spec.stage1.pre_tokenizer();

    // Decompose each relaxed chunk into stage-one words. A relaxed chunk is by
    // construction a concatenation of its stage-one words, so stage two can
    // reuse their learned encodings instead of re-merging from bytes.
    let mut word_ids: FxHashMap<String, u32> = FxHashMap::default();
    let mut word_surfaces: Vec<String> = Vec::new();
    let mut word_counts: Vec<u64> = Vec::new();
    let mut chunk_counts: FxHashMap<Vec<u32>, u64> = FxHashMap::default();

    for chunk in chunks {
        // Split on original text, not on `relaxed_pre` output: that already
        // carries the byte-level mapping, and stage one must not encode twice.
        for piece in split_relaxed(chunk.as_ref()) {
            let words = chunk_to_surfaces(&stage1_pre, &piece);
            if words.is_empty() {
                continue;
            }
            let mut ids = Vec::with_capacity(words.len());
            for word in words {
                let id = *word_ids.entry(word.clone()).or_insert_with(|| {
                    let id = word_surfaces.len() as u32;
                    word_surfaces.push(word);
                    word_counts.push(0);
                    id
                });
                word_counts[id as usize] += 1;
                ids.push(id);
            }
            *chunk_counts.entry(ids).or_insert(0) += 1;
        }
    }

    let alphabet: Vec<String> = {
        let mut chars: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        chars.sort_unstable();
        chars.into_iter().map(|c| c.to_string()).collect()
    };

    let mut learner = Learner::new(&alphabet, spec.min_frequency, spec.max_token_bytes);
    let mut words: Vec<(Vec<u32>, u64)> = word_surfaces
        .iter()
        .enumerate()
        .map(|(id, surface)| {
            let symbols = surface
                .chars()
                .map(|c| {
                    *learner
                        .lookup
                        .get(&c.to_string())
                        .expect("byte-level alphabet covers every surface character")
                })
                .collect();
            (symbols, word_counts[id])
        })
        .collect();

    // Special tokens are assigned ids starting at the model's vocabulary size,
    // so learning a full `vocab_size` merges and then adding them pushes ids to
    // vocab_size + n — out of range for a [vocab_size, d_model] embedding.
    // Reserve their slots up front instead.
    let learned_target = spec.vocab_size.saturating_sub(spec.special_tokens.len());
    let transition = spec.transition.min(learned_target);
    if transition > learner.vocab.len() {
        learner.run(&mut words, transition);
    }
    let stage1_merges = learner.merges.len();

    // Gate on the transition, not on how full the vocabulary happens to be:
    // stage one can stop early on the frequency floor, and widening the
    // boundary to use up the remainder would silently turn a request for
    // ordinary BPE into two-stage training.
    if transition < learned_target && learner.vocab.len() < learned_target {
        let mut relaxed: Vec<(Vec<u32>, u64)> = chunk_counts
            .into_iter()
            .map(|(ids, count)| {
                let mut symbols = Vec::new();
                for id in ids {
                    symbols.extend_from_slice(&words[id as usize].0);
                }
                (symbols, count)
            })
            .collect();
        learner.run(&mut relaxed, learned_target);
    }

    let merges = learner
        .merges
        .iter()
        .map(|(a, b)| {
            (
                learner.vocab[*a as usize].clone(),
                learner.vocab[*b as usize].clone(),
            )
        })
        .collect();

    TrainedBpe {
        vocab: learner.vocab,
        merges,
        stage1_merges,
    }
}

/// Cut text into relaxed chunks without applying any byte-level mapping.
///
/// This must reproduce [`RE_LINE`] exactly — a run of non-break characters
/// followed by a *maximal* run of breaks, so consecutive blank lines form a
/// single chunk. Splitting at every break instead would hand stage two chunks
/// the encoder never produces, and it would optimise for a segmentation that
/// cannot occur.
fn split_relaxed(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let is_break = |c: char| c == '\n' || c == '\r';
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        while i < chars.len() && !is_break(chars[i]) {
            i += 1;
        }
        while i < chars.len() && is_break(chars[i]) {
            i += 1;
        }
        out.push(chars[start..i].iter().collect());
    }
    out
}

/// Wrap a trained vocabulary in a `Tokenizer` that round-trips.
pub fn assemble(trained: &TrainedBpe, spec: &TrainSpec) -> Result<Tokenizer, String> {
    // Ship whichever boundary training actually optimised for. Hardcoding `Line`
    // here would mean a spec that never relaxed still encoded without the word
    // pattern, so word-learned merges could fire across word boundaries at
    // inference and `transition >= vocab_size` would not be ordinary BPE.
    let encode_boundary = if spec.transition >= spec.vocab_size {
        spec.stage1
    } else {
        PreTokenizerKind::Line
    };
    let vocab: ahash::AHashMap<String, u32> = trained
        .vocab
        .iter()
        .enumerate()
        .map(|(id, surface)| (surface.clone(), id as u32))
        .collect();
    let model = BPE::builder()
        .vocab_and_merges(vocab, trained.merges.clone())
        .build()
        .map_err(|e| format!("assembling BPE model: {e}"))?;

    let mut tokenizer = Tokenizer::new(tokenizers::ModelWrapper::BPE(model));
    tokenizer.with_pre_tokenizer(Some(encode_boundary.pre_tokenizer()));
    // Without a decoder, `decode` joins the byte-level surfaces with spaces and
    // never reproduces the input.
    tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevel::default())));

    let added: Vec<AddedToken> = spec
        .special_tokens
        .iter()
        .map(|t| AddedToken::from(t.clone(), true))
        .collect();
    if !added.is_empty() {
        tokenizer.add_special_tokens(&added);
    }
    Ok(tokenizer)
}

/// Train from a corpus file.
///
/// The whole file is passed as a single chunk so that [`split_relaxed`] performs
/// the chunking. Reading it line by line instead would hand training one chunk
/// per `\n`, and a run of blank lines is ONE relaxed chunk — so stage two would
/// collect statistics over a segmentation the encoder never produces. Corpus
/// `"a\n\nb\n"` splits as `["a\n\n", "b\n"]`, not `["a\n", "\n", "b\n"]`.
///
/// This holds the corpus in memory. That is the same order of cost as the word
/// map training builds regardless, but it does bound the corpus size.
pub fn train_from_file(path: &str, spec: &TrainSpec) -> Result<Tokenizer, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("reading '{path}': {e}"))?;
    let trained = train_two_stage([text], spec);
    assemble(&trained, spec)
}

/// Build a tokenizer whose ids are byte values, with a decoder attached.
pub fn byte_level_identity() -> Tokenizer {
    let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::new();
    let mut chars: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    chars.sort_unstable();
    for (id, c) in chars.into_iter().enumerate() {
        vocab.insert(c.to_string(), id as u32);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .expect("byte alphabet is a valid BPE model");
    let mut tokenizer = Tokenizer::new(tokenizers::ModelWrapper::BPE(model));
    tokenizer.with_pre_tokenizer(Some(PreTokenizerKind::Line.pre_tokenizer()));
    tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevel::default())));
    tokenizer
}

/// Number of distinct byte-level characters the vocabulary covers. A tokenizer
/// missing any of the 256 silently drops those bytes at encode time.
pub fn byte_coverage(tokenizer: &Tokenizer) -> usize {
    let vocab: HashMap<String, u32> = tokenizer.get_vocab(true).into_iter().collect();
    ByteLevel::alphabet()
        .into_iter()
        .filter(|c| vocab.contains_key(&c.to_string()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
fn forward(self, x: Tensor) -> Tensor:
    let h = self.norm(x)
    let y = self.attn(h)
    return x + y

fn backward(self, g: Tensor) -> Tensor:
    let d = self.attn_bwd(g)
    return d
";

    /// True when a token could not have been learned under the stage-one
    /// pattern, because that pattern would have cut it into several chunks.
    ///
    /// This is the precise definition of a cross-boundary token. Note that
    /// `Line` chunking puts the break at the *end* of a chunk, so a token
    /// reaching past a break into the next line is impossible by construction;
    /// what relaxation actually buys is joining a line's leading indentation to
    /// its first word, and its last word to the break.
    fn crosses_stage1_boundary(surface: &str, stage1: PreTokenizerKind) -> bool {
        use tokenizers::Decoder;
        let decoder = ByteLevel::default();
        let Ok(text) = decoder.decode(vec![surface.to_string()]) else {
            return false;
        };
        chunk_to_surfaces(&stage1.pre_tokenizer(), &text).len() > 1
    }

    fn count_crossing(tok: &Tokenizer, stage1: PreTokenizerKind) -> usize {
        tok.get_vocab(false)
            .into_keys()
            .filter(|s| crosses_stage1_boundary(s, stage1))
            .count()
    }

    fn train_sample(spec: TrainSpec) -> Tokenizer {
        let lines: Vec<String> = SAMPLE.lines().map(|l| format!("{l}\n")).collect();
        // Repeat so merges clear the frequency floor.
        let corpus: Vec<String> = std::iter::repeat(lines).take(40).flatten().collect();
        let trained = train_two_stage(corpus, &spec);
        assemble(&trained, &spec).expect("assemble")
    }

    #[test]
    fn roundtrips_exactly() {
        let tok = train_sample(TrainSpec {
            vocab_size: 600,
            transition: 400,
            ..TrainSpec::default()
        });
        for text in [SAMPLE, "fn f(x): return x", "\t\tmixed\ttabs\n", "héllo wörld"] {
            let enc = tok.encode(text, false).expect("encode");
            let dec = tok.decode(enc.get_ids(), false).expect("decode");
            assert_eq!(dec, text, "roundtrip failed for {text:?}");
        }
    }

    /// Every byte must be representable even if the corpus never contained it,
    /// otherwise encoding silently discards input.
    #[test]
    fn covers_all_byte_level_characters() {
        let tok = train_sample(TrainSpec {
            vocab_size: 400,
            transition: 400,
            ..TrainSpec::default()
        });
        assert_eq!(byte_coverage(&tok), 256);

        let exotic = "\u{1F600}\u{0}\u{7f}\u{80}";
        let enc = tok.encode(exotic, false).expect("encode");
        assert_eq!(tok.decode(enc.get_ids(), false).expect("decode"), exotic);
    }

    /// A transition at or above the vocabulary size must never widen the
    /// boundary, so no token may contain a line break.
    #[test]
    fn transition_at_vocab_size_keeps_the_word_boundary() {
        let tok = train_sample(TrainSpec {
            vocab_size: 800,
            transition: 800,
            ..TrainSpec::default()
        });
        let offenders: Vec<String> = tok
            .get_vocab(false)
            .into_keys()
            .filter(|s| crosses_stage1_boundary(s, PreTokenizerKind::Cl100k))
            .take(15)
            .collect();
        assert_eq!(
            count_crossing(&tok, PreTokenizerKind::Cl100k),
            0,
            "word-bounded training must not learn a token crossing its own pattern; e.g. {offenders:?}"
        );
    }

    /// Relaxing the boundary is the whole point: it must actually produce
    /// cross-boundary tokens, and they must compress better.
    #[test]
    fn relaxed_boundary_learns_cross_line_tokens_and_compresses_better() {
        let bounded = train_sample(TrainSpec {
            vocab_size: 800,
            transition: 800,
            ..TrainSpec::default()
        });
        let relaxed = train_sample(TrainSpec {
            vocab_size: 800,
            transition: 300,
            ..TrainSpec::default()
        });

        assert!(
            count_crossing(&relaxed, PreTokenizerKind::Cl100k) > 0,
            "relaxed training learned no cross-boundary token"
        );

        let n_bounded = bounded.encode(SAMPLE, false).expect("encode").len();
        let n_relaxed = relaxed.encode(SAMPLE, false).expect("encode").len();
        assert!(
            n_relaxed < n_bounded,
            "relaxed boundary should need fewer tokens, got {n_relaxed} vs {n_bounded}"
        );
    }

    #[test]
    fn max_token_bytes_is_respected() {
        let spec = TrainSpec {
            vocab_size: 600,
            transition: 200,
            max_token_bytes: 6,
            ..TrainSpec::default()
        };
        let tok = train_sample(spec);
        for surface in tok.get_vocab(false).into_keys() {
            // One char of a byte-level surface is one byte of original text;
            // `len()` would measure the mapping's UTF-8 width instead.
            assert!(
                surface.chars().count() <= 6,
                "token {surface:?} covers more than 6 original bytes"
            );
        }
    }

    /// A cap of N must admit N *original bytes*. Under the old `len()`
    /// accounting a space cost 2 (U+0120 is two UTF-8 bytes), so a cap of 8
    /// admitted only 4 spaces — for exactly the whitespace runs the cap exists
    /// to bound.
    #[test]
    fn the_cap_counts_original_bytes_not_the_mapping_width() {
        let indent_heavy: Vec<String> =
            std::iter::repeat_n("        deeply(indented, call)\n".to_string(), 300).collect();
        let spec = TrainSpec {
            vocab_size: 500,
            transition: 300,
            max_token_bytes: 8,
            ..TrainSpec::default()
        };
        let trained = train_two_stage(indent_heavy, &spec);
        let tok = assemble(&trained, &spec).expect("assemble");

        let longest_run = tok
            .get_vocab(false)
            .into_keys()
            .filter(|s| s.chars().all(|c| c == '\u{0120}'))
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(0);
        assert!(
            longest_run > 4,
            "longest all-space token is {longest_run} chars; a cap of 8 original \
             bytes must allow more than the 4 that UTF-8-width accounting gave"
        );
        assert!(
            longest_run <= 8,
            "longest all-space token is {longest_run} chars, over the cap of 8"
        );
    }

    #[test]
    fn line_splitting_matches_the_shipped_pre_tokenizer() {
        // split_relaxed drives training; the Line pre-tokenizer drives encoding.
        // If they disagree, stage two optimises for chunks the encoder never
        // produces.
        let pre = PreTokenizerKind::Line.pre_tokenizer();
        for text in ["a\nb\n", "a\nb", "\n\n\n", "a\r\nb\r\n", "no newline", ""] {
            let ours = split_relaxed(text);
            let theirs = chunk_to_surfaces(&pre, text);
            assert_eq!(
                ours.len(),
                theirs.len(),
                "chunk count differs for {text:?}: {ours:?} vs {theirs:?}"
            );
        }
    }
}
