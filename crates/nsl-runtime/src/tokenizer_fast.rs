//! Fast byte-domain encoder for the shipped NSL tokenizer format (item 16).
//!
//! # What this is
//!
//! A same-token replacement for the `tokenizers` crate's encode path,
//! specialized to the ONE configuration NSL's tokenizers of record use at
//! encode time (see `tokenizer_bpe::assemble` and
//! `models/tokenizers/*.json`):
//!
//!   * no normalizer;
//!   * pre-tokenizer = `Split("[^\r\n]*[\r\n]*", Isolated)` — whole physical
//!     lines — followed by `ByteLevel(add_prefix_space=false, use_regex=false)`;
//!   * plain BPE (no dropout, no unk, no byte_fallback, no ignore_merges);
//!   * added tokens matched by raw surface, unnormalized, no strip flags.
//!
//! Anything else REFUSES at load. This module must never silently accept a
//! tokenizer whose encode semantics it does not reproduce: the corpus
//! fingerprints (`models/datasets/CORPUS_MANIFEST_v2.json`) are downstream
//! of same-token parity, and a close-enough encoder is a corpus that lies
//! about its own provenance.
//!
//! # Why it is fast (design adapted from marcelroed/gigatoken, MIT)
//!
//! The HF path pays for generality three times per document: an onig regex
//! for the line split, a bytes-to-surrogate-chars expansion so BPE can run
//! over `String`s, and a merge loop whose word cache misses on almost every
//! line (lines are long and mostly unique on web text). This encoder:
//!
//!   * splits lines with the byte scanner training already uses
//!     (`tokenizer_bpe::split_relaxed` — tested against the shipped
//!     pre-tokenizer);
//!   * inverts the ByteLevel mapping over the vocabulary ONCE at load, so
//!     encoding works directly on the input bytes — no per-token unicode
//!     round trip (gigatoken's `ByteRemapping` idea);
//!   * runs the merge loop as a doubly-linked list over symbol indices with
//!     a lazily-invalidated min-heap of `(rank, position)` packed into one
//!     `u64` (gigatoken's `bpe_merge_symbols`) — the same merge ORDER the
//!     HF `Word::merge_all` heap produces, so the output ids are identical;
//!   * memoizes whole-line encodings in a byte-budgeted cache: code corpora
//!     (60% of the pretraining mix) repeat lines constantly — blank lines,
//!     `}`, `import numpy as np` — and a hit skips the merge loop entirely.
//!
//! Parity is enforced, not assumed: `fast_encoder_parity_gate.rs` diffs this
//! encoder against the real `tokenizers` crate on edge-case fixtures and a
//! generated corpus in every CI build, and the full-corpus sha256 evidence
//! lives in models/benchmarks/.

use rustc_hash::FxHashMap;

use crate::tokenizer_bpe::split_relaxed;

// ---------------------------------------------------------------------------
// ByteLevel mapping
// ---------------------------------------------------------------------------

/// The GPT-2 byte-to-unicode table: printable/latin bytes map to themselves,
/// the rest map to U+0100.. in byte order. This is the mapping `ByteLevel`
/// applies and the one the vocabulary's surfaces are written in.
fn byte_to_char_table() -> [char; 256] {
    let self_mapped =
        |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut table = ['\0'; 256];
    let mut next = 0u32;
    for (b, slot) in table.iter_mut().enumerate() {
        let b = b as u32;
        *slot = if self_mapped(b) {
            char::from_u32(b).expect("latin-1 range")
        } else {
            let c = char::from_u32(256 + next).expect("BMP range");
            next += 1;
            c
        };
    }
    table
}

/// Invert a vocabulary surface (ByteLevel-mapped string) back to the raw
/// bytes it encodes. `None` when the surface contains a char outside the
/// mapping — such a token could never be produced from input bytes.
fn surface_to_bytes(surface: &str, inverse: &FxHashMap<char, u8>) -> Option<Vec<u8>> {
    surface.chars().map(|c| inverse.get(&c).copied()).collect()
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// One added token: raw surface bytes and the id it extracts to.
struct Special {
    surface: Vec<u8>,
    id: u16,
}

pub struct FastBpeEncoder {
    /// Base symbol id for each input byte. Every byte is representable — the
    /// loader refuses a vocabulary missing any of the 256 byte tokens.
    byte_id: [u32; 256],
    /// `(left_id << 32 | right_id)` → `(merge rank, merged id)`. Rank is the
    /// index in the merges list — the exact priority HF uses — not the
    /// merged id, so no id/rank-bijection assumption is needed.
    pair_merged: FxHashMap<u64, (u32, u32)>,
    /// Added tokens, matched longest-first at each position.
    specials: Vec<Special>,
    /// First-byte dispatch for the added-token scan.
    special_lead: [bool; 256],
}

/// Read a required boolean-ish JSON field, treating `null` as absent.
fn field<'a>(v: &'a serde_json::Value, key: &str) -> &'a serde_json::Value {
    v.get(key).unwrap_or(&serde_json::Value::Null)
}

impl std::fmt::Debug for FastBpeEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastBpeEncoder")
            .field("merges", &self.pair_merged.len())
            .field("specials", &self.specials.len())
            .finish_non_exhaustive()
    }
}

impl FastBpeEncoder {
    /// Load from a tokenizer JSON file, refusing any configuration this
    /// encoder does not reproduce token-for-token.
    pub fn from_tokenizer_file(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading tokenizer '{}': {e}", path.display()))?;
        let spec: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("tokenizer '{}' is not valid JSON: {e}", path.display()))?;
        Self::from_tokenizer_json(&spec)
    }

    pub fn from_tokenizer_json(spec: &serde_json::Value) -> Result<Self, String> {
        // --- Refuse everything outside the supported encode semantics. ----
        if !field(spec, "normalizer").is_null() {
            return Err("fast encoder: refusing a tokenizer with a normalizer".into());
        }
        let pre = field(spec, "pre_tokenizer");
        let seq = field(pre, "pretokenizers");
        let ok_pre = field(pre, "type") == "Sequence"
            && seq.as_array().map(|a| a.len()) == Some(2)
            && field(&seq[0], "type") == "Split"
            && field(field(&seq[0], "pattern"), "Regex") == r"[^\r\n]*[\r\n]*"
            && field(&seq[0], "behavior") == "Isolated"
            && field(&seq[0], "invert").as_bool() == Some(false)
            && field(&seq[1], "type") == "ByteLevel"
            && field(&seq[1], "add_prefix_space").as_bool() == Some(false)
            && field(&seq[1], "use_regex").as_bool() == Some(false);
        if !ok_pre {
            return Err(format!(
                "fast encoder: refusing pre_tokenizer {pre} — only the shipped \
                 line-split + ByteLevel(no prefix space, no regex) sequence is \
                 reproduced token-for-token"
            ));
        }
        let model = field(spec, "model");
        if field(model, "type") != "BPE"
            || !field(model, "dropout").is_null()
            || !field(model, "unk_token").is_null()
            || field(model, "byte_fallback").as_bool() == Some(true)
            || field(model, "ignore_merges").as_bool() == Some(true)
            || field(model, "fuse_unk").as_bool() == Some(true)
        {
            return Err(format!(
                "fast encoder: refusing model config — need plain BPE \
                 (no dropout/unk/byte_fallback/ignore_merges/fuse_unk), got: type={} \
                 dropout={} unk={} byte_fallback={} ignore_merges={} fuse_unk={}",
                field(model, "type"),
                field(model, "dropout"),
                field(model, "unk_token"),
                field(model, "byte_fallback"),
                field(model, "ignore_merges"),
                field(model, "fuse_unk"),
            ));
        }
        for k in ["continuing_subword_prefix", "end_of_word_suffix"] {
            let v = field(model, k);
            if !(v.is_null() || v == "") {
                return Err(format!("fast encoder: refusing model.{k} = {v}"));
            }
        }

        // --- Vocabulary, inverted to the byte domain. ---------------------
        let inverse: FxHashMap<char, u8> = byte_to_char_table()
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        let vocab = field(model, "vocab")
            .as_object()
            .ok_or("fast encoder: model.vocab must be an object")?;
        // surface bytes -> id, for merge resolution below.
        let mut by_bytes: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        let mut max_id = 0u32;
        for (surface, id) in vocab {
            let id = id
                .as_u64()
                .ok_or_else(|| format!("fast encoder: vocab id for '{surface}' is not an integer"))?
                as u32;
            max_id = max_id.max(id);
            let bytes = surface_to_bytes(surface, &inverse).ok_or_else(|| {
                format!("fast encoder: vocab surface {surface:?} is not byte-level mapped")
            })?;
            if by_bytes.insert(bytes, id).is_some() {
                return Err(format!(
                    "fast encoder: two vocab surfaces decode to the same bytes ({surface:?})"
                ));
            }
        }
        // Refuse an over-wide id space FIRST: it is the clearest diagnosis,
        // and letting the added-token checks below run first would blame an
        // innocent added token for sitting "inside" a learned id space that
        // is itself the problem.
        if max_id > u16::MAX as u32 {
            return Err(format!(
                "fast encoder: max vocab id {max_id} does not fit the u16 token stream"
            ));
        }
        let mut byte_id = [u32::MAX; 256];
        for (b, slot) in byte_id.iter_mut().enumerate() {
            *slot = *by_bytes.get(&[b as u8][..]).ok_or_else(|| {
                format!(
                    "fast encoder: vocabulary has no token for byte 0x{b:02x} — \
                     such a byte would be silently dropped; refusing"
                )
            })?;
        }

        // --- Merges, ranked by list order (HF's priority). ----------------
        let merges = field(model, "merges")
            .as_array()
            .ok_or("fast encoder: model.merges must be an array")?;
        let mut pair_merged: FxHashMap<u64, (u32, u32)> = FxHashMap::default();
        pair_merged.reserve(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let (a, b) = match m.as_array().map(|p| p.as_slice()) {
                Some([a, b]) => (
                    a.as_str().ok_or("fast encoder: merge side is not a string")?,
                    b.as_str().ok_or("fast encoder: merge side is not a string")?,
                ),
                // The legacy "left right" space-joined form is refused rather
                // than parsed: a byte-level surface can itself be `Ġ`-free
                // but the split would still be ambiguous in general.
                _ => return Err(format!("fast encoder: merge #{rank} is not a [left, right] pair")),
            };
            let ab = format!("{a}{b}");
            let left = *by_bytes
                .get(&surface_to_bytes(a, &inverse).ok_or("unmappable merge side")?)
                .ok_or_else(|| format!("fast encoder: merge left {a:?} not in vocab"))?;
            let right = *by_bytes
                .get(&surface_to_bytes(b, &inverse).ok_or("unmappable merge side")?)
                .ok_or_else(|| format!("fast encoder: merge right {b:?} not in vocab"))?;
            let merged = *by_bytes
                .get(&surface_to_bytes(&ab, &inverse).ok_or("unmappable merge result")?)
                .ok_or_else(|| format!("fast encoder: merge result {ab:?} not in vocab"))?;
            let key = ((left as u64) << 32) | right as u64;
            if pair_merged.insert(key, (rank as u32, merged)).is_some() {
                return Err(format!(
                    "fast encoder: duplicate merge pair ({a:?}, {b:?}) — priority undefined"
                ));
            }
        }

        // --- Added tokens. ------------------------------------------------
        // The `tokenizers` crate silently RE-ASSIGNS a declared added-token
        // id that is neither a reuse of that surface's learned id nor the
        // next free slot (verified: id 70000 loads as 49150). Reproducing
        // that would mean reproducing its internal assignment order, so any
        // added token outside the two well-defined cases is refused.
        let mut specials = Vec::new();
        let mut special_lead = [false; 256];
        if let Some(added) = field(spec, "added_tokens").as_array() {
            for t in added {
                let content = field(t, "content")
                    .as_str()
                    .ok_or("fast encoder: added token without content")?;
                let id = field(t, "id")
                    .as_u64()
                    .ok_or("fast encoder: added token without id")? as u32;
                if field(t, "normalized").as_bool() == Some(true)
                    || field(t, "single_word").as_bool() == Some(true)
                    || field(t, "lstrip").as_bool() == Some(true)
                    || field(t, "rstrip").as_bool() == Some(true)
                {
                    return Err(format!(
                        "fast encoder: refusing added token {content:?} — normalized/\
                         single_word/lstrip/rstrip matching is not reproduced"
                    ));
                }
                let learned = vocab.get(content).and_then(|v| v.as_u64()).map(|v| v as u32);
                let reuse = learned == Some(id);
                if !reuse && id <= max_id {
                    return Err(format!(
                        "fast encoder: added token {content:?} declares id {id} inside the \
                         learned id space without matching its learned id — the tokenizers \
                         crate would re-assign it, and parity cannot be guaranteed"
                    ));
                }
                let id = u16::try_from(id).map_err(|_| {
                    format!("fast encoder: added token {content:?} id {id} exceeds u16")
                })?;
                let surface = content.as_bytes().to_vec();
                if surface.is_empty() {
                    return Err("fast encoder: empty added-token surface".into());
                }
                special_lead[surface[0] as usize] = true;
                specials.push(Special { surface, id });
            }
        }
        // Longest-first so the per-position scan is leftmost-longest, the
        // aho-corasick match kind the crate uses.
        specials.sort_by_key(|sp| std::cmp::Reverse(sp.surface.len()));

        Ok(Self { byte_id, pair_merged, specials, special_lead })
    }

    #[inline]
    fn rank_of(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        self.pair_merged.get(&(((a as u64) << 32) | b as u64)).copied()
    }

    /// Id of an added token by exact surface. Deliberately restricted to
    /// added tokens: a structural document separator that is NOT an added
    /// token would also be reachable by ordinary merges from document text,
    /// making forged boundaries indistinguishable from real ones.
    pub fn special_id(&self, surface: &str) -> Option<u16> {
        self.specials
            .iter()
            .find(|sp| sp.surface == surface.as_bytes())
            .map(|sp| sp.id)
    }

    /// Encode one document into `out` as u16 ids. Semantics mirror the HF
    /// pipeline exactly: added-token extraction over the raw text first,
    /// then per-segment line splitting, then BPE per line.
    pub fn encode_doc_into(&self, text: &str, cache: &mut EncodeCache, out: &mut Vec<u16>) {
        let bytes = text.as_bytes();
        let mut seg_start = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            if self.special_lead[bytes[i] as usize] {
                if let Some(sp) = self
                    .specials
                    .iter()
                    .find(|sp| bytes[i..].starts_with(&sp.surface))
                {
                    // Text before the surface, then the surface's own id.
                    self.encode_span(&text[seg_start..i], cache, out);
                    out.push(sp.id);
                    i += sp.surface.len();
                    seg_start = i;
                    continue;
                }
            }
            i += 1;
        }
        self.encode_span(&text[seg_start..], cache, out);
    }

    /// Encode a special-free span: line split, then BPE per line, memoized.
    fn encode_span(&self, span: &str, cache: &mut EncodeCache, out: &mut Vec<u16>) {
        if span.is_empty() {
            return;
        }
        for line in split_relaxed(span) {
            let lb = line.as_bytes();
            if let Some(ids) = cache.get(lb) {
                out.extend_from_slice(ids);
                continue;
            }
            let start = out.len();
            self.merge_line(lb, &mut cache.scratch, out);
            cache.insert(lb, &out[start..]);
        }
    }

    /// The BPE merge loop over one line's bytes, appending u16 ids.
    ///
    /// Both paths realize the same order — repeatedly merge the valid pair
    /// with the smallest `(rank, position)` — which is HF `Word::merge_all`'s
    /// order, so the output ids are identical. Typical lines (≤
    /// `SMALL_MERGE_MAX` symbols; the overwhelming majority of corpus lines)
    /// take a tiktoken-style linear scan over a per-position rank array:
    /// O(n²) in cheap branch-free comparisons beats the heap's allocation
    /// and sift traffic at these sizes. Pathological lines (minified JS,
    /// data blobs) fall through to the heap + linked-list path, whose
    /// n·log n keeps them from going quadratic.
    fn merge_line(&self, line: &[u8], scratch: &mut Scratch, out: &mut Vec<u16>) {
        if line.len() <= SMALL_MERGE_MAX {
            self.merge_line_scan(line, scratch, out);
        } else {
            self.merge_line_heap(line, out);
        }
    }

    /// Linear-scan merge for short lines: `ranks[i]` caches the rank of the
    /// pair starting at active symbol `i`; each round finds the minimum by
    /// scan, merges in place, and recomputes only the two affected
    /// neighbors' ranks.
    fn merge_line_scan(&self, line: &[u8], scratch: &mut Scratch, out: &mut Vec<u16>) {
        let n = line.len();
        match n {
            0 => return,
            1 => {
                out.push(self.byte_id[line[0] as usize] as u16);
                return;
            }
            _ => {}
        }
        const NONE: u32 = u32::MAX;
        let symbols = &mut scratch.symbols;
        symbols.clear();
        symbols.extend(line.iter().map(|&b| self.byte_id[b as usize]));
        let ranks = &mut scratch.ranks;
        ranks.clear();
        for i in 0..n - 1 {
            let r = self.rank_of(symbols[i], symbols[i + 1]).map_or(NONE, |(rk, _)| rk);
            ranks.push(r);
        }

        loop {
            // Leftmost minimum: `<` keeps the earliest position on ties,
            // matching the heap's (rank, pos) ordering.
            let mut best = NONE;
            let mut pos = 0usize;
            for (i, &r) in ranks.iter().enumerate() {
                if r < best {
                    best = r;
                    pos = i;
                }
            }
            if best == NONE {
                break;
            }
            let (_, merged) = self
                .rank_of(symbols[pos], symbols[pos + 1])
                .expect("rank array desynchronized from symbols");
            symbols[pos] = merged;
            symbols.remove(pos + 1);
            ranks.remove(pos);
            if pos > 0 {
                ranks[pos - 1] =
                    self.rank_of(symbols[pos - 1], symbols[pos]).map_or(NONE, |(r, _)| r);
            }
            if pos < ranks.len() {
                ranks[pos] =
                    self.rank_of(symbols[pos], symbols[pos + 1]).map_or(NONE, |(r, _)| r);
            }
        }
        out.extend(symbols.iter().map(|&s| s as u16));
    }

    /// Heap + linked-list merge for long lines (lazily-invalidated min-heap
    /// of `(rank << 32 | position)`, re-validated on pop).
    fn merge_line_heap(&self, line: &[u8], out: &mut Vec<u16>) {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let n = line.len();
        match n {
            0 => return,
            1 => {
                out.push(self.byte_id[line[0] as usize] as u16);
                return;
            }
            _ => {}
        }

        let mut symbols: Vec<u32> = line.iter().map(|&b| self.byte_id[b as usize]).collect();
        const NONE: u32 = u32::MAX;
        let mut next: Vec<u32> = (1..n as u32).chain([NONE]).collect();
        let mut prev: Vec<u32> = [NONE].into_iter().chain(0..n as u32 - 1).collect();

        let mut heap: BinaryHeap<Reverse<u64>> = symbols
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                self.rank_of(w[0], w[1])
                    .map(|(rank, _)| Reverse(((rank as u64) << 32) | i as u64))
            })
            .collect();

        while let Some(Reverse(entry)) = heap.pop() {
            let pos = (entry & u32::MAX as u64) as usize;
            let expected_rank = (entry >> 32) as u32;
            let right = next[pos];
            if right == NONE {
                continue; // pos itself was merged away
            }
            let right = right as usize;
            let Some((rank, merged)) = self.rank_of(symbols[pos], symbols[right]) else {
                continue; // stale: the pair changed under this entry
            };
            if rank != expected_rank {
                continue; // stale: a different (later-priority) pair now sits here
            }
            symbols[pos] = merged;
            let right_right = next[right];
            next[pos] = right_right;
            if right_right != NONE {
                prev[right_right as usize] = pos as u32;
            }
            next[right] = NONE;

            let left = prev[pos];
            if left != NONE {
                if let Some((r, _)) = self.rank_of(symbols[left as usize], symbols[pos]) {
                    heap.push(Reverse(((r as u64) << 32) | left as u64));
                }
            }
            if next[pos] != NONE {
                if let Some((r, _)) = self.rank_of(symbols[pos], symbols[next[pos] as usize]) {
                    heap.push(Reverse(((r as u64) << 32) | pos as u64));
                }
            }
        }

        let mut i = 0usize;
        loop {
            out.push(symbols[i] as u16);
            match next[i] {
                NONE => break,
                j => i = j as usize,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Line cache
// ---------------------------------------------------------------------------

/// Longest line worth memoizing. Longer lines are overwhelmingly unique
/// (web sentences), so caching them buys hit-rate rounding error and costs
/// real memory.
const MAX_CACHED_LINE_BYTES: usize = 128;

/// Longest line (in bytes = initial symbols) the linear-scan merge handles;
/// beyond this the heap path's n·log n takes over so a minified-JS line
/// cannot go quadratic. 192 covers the overwhelming majority of real corpus
/// lines (web sentences and code lines sit at 40–120 bytes).
const SMALL_MERGE_MAX: usize = 192;

/// Whole-line encode memo with a byte budget. Insertion stops when the
/// budget is reached (no eviction: the head of a corpus's line distribution
/// is stable, and the tail would only thrash an LRU). Correctness does not
/// depend on the cache: a hit returns exactly what the merge loop returned
/// when the entry was created, keyed by full line bytes.
pub struct EncodeCache {
    map: FxHashMap<Box<[u8]>, Box<[u16]>>,
    bytes: usize,
    budget: usize,
    /// Reused merge-loop buffers — two allocations per line otherwise.
    scratch: Scratch,
}

#[derive(Default)]
struct Scratch {
    symbols: Vec<u32>,
    ranks: Vec<u32>,
}

impl EncodeCache {
    /// `budget` bounds the sum of key + value bytes (default 256 MiB in
    /// tokbench, one cache per worker thread).
    pub fn with_budget(budget: usize) -> Self {
        Self { map: FxHashMap::default(), bytes: 0, budget, scratch: Scratch::default() }
    }

    fn get(&self, line: &[u8]) -> Option<&[u16]> {
        if line.len() > MAX_CACHED_LINE_BYTES {
            return None;
        }
        self.map.get(line).map(|v| &**v)
    }

    fn insert(&mut self, line: &[u8], ids: &[u16]) {
        if line.len() > MAX_CACHED_LINE_BYTES || self.bytes >= self.budget {
            return;
        }
        let cost = line.len() + ids.len() * 2 + 48;
        self.bytes += cost;
        self.map.insert(line.into(), ids.into());
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte↔char mapping must be a bijection whose image is exactly the
    /// crate's ByteLevel alphabet.
    #[test]
    fn byte_mapping_matches_the_crate_alphabet() {
        let table = byte_to_char_table();
        let ours: std::collections::HashSet<char> = table.iter().copied().collect();
        assert_eq!(ours.len(), 256, "mapping must be injective");
        let theirs: std::collections::HashSet<char> =
            tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
                .into_iter()
                .collect();
        assert_eq!(ours, theirs, "byte-level alphabet must match the crate");
    }

    /// A minimal hand-built tokenizer: bytes 'a','b','c' + merges
    /// (a,b)->"ab" then (ab,c)->"abc". Checks merge order, linked-list
    /// compaction, and the cache round trip.
    #[test]
    fn tiny_merge_sequence_and_cache_agree() {
        // Build a spec through the real trainer so the JSON shape is honest.
        let spec = crate::tokenizer_bpe::TrainSpec {
            vocab_size: 300,
            min_frequency: 1,
            transition: 300, // single-stage
            max_token_bytes: 0,
            stage1: crate::tokenizer_bpe::PreTokenizerKind::Line,
            special_tokens: vec!["<|eot|>".into()],
        };
        let trained = crate::tokenizer_bpe::train_two_stage(["abab abc\nabab\n"], &spec);
        let tok = crate::tokenizer_bpe::assemble(&trained, &spec).expect("assemble");
        let json: serde_json::Value =
            serde_json::from_str(&tok.to_string(false).expect("serialize")).expect("json");
        let fast = FastBpeEncoder::from_tokenizer_json(&json).expect("load");

        let mut cache = EncodeCache::with_budget(1 << 20);
        for text in ["abab abc\nabab\n", "abc<|eot|>abc", "", "x", "\n\n\nabab"] {
            let mut got = Vec::new();
            fast.encode_doc_into(text, &mut cache, &mut got);
            // Second pass: everything cacheable now comes from the cache.
            let mut again = Vec::new();
            fast.encode_doc_into(text, &mut cache, &mut again);
            assert_eq!(got, again, "cache changed the encoding of {text:?}");

            let want: Vec<u16> = tok
                .encode_fast(text, false)
                .expect("hf encode")
                .get_ids()
                .iter()
                .map(|&id| id as u16)
                .collect();
            assert_eq!(got, want, "fast != hf on {text:?}");
        }
    }

    #[test]
    fn refuses_configs_it_does_not_reproduce() {
        let base: serde_json::Value = serde_json::json!({
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": "[^\\r\\n]*[\\r\\n]*"},
                     "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": false,
                     "trim_offsets": true, "use_regex": false}
                ]
            },
            "model": {"type": "BPE", "dropout": null, "unk_token": null,
                       "continuing_subword_prefix": null, "end_of_word_suffix": null,
                       "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                       "vocab": {}, "merges": []},
            "added_tokens": []
        });

        // The base itself fails later (no byte coverage), but must get PAST
        // the config checks — pin where it fails.
        let e = FastBpeEncoder::from_tokenizer_json(&base).unwrap_err();
        assert!(e.contains("no token for byte"), "unexpected refusal: {e}");

        let mut with_norm = base.clone();
        with_norm["normalizer"] = serde_json::json!({"type": "NFC"});
        assert!(FastBpeEncoder::from_tokenizer_json(&with_norm)
            .unwrap_err()
            .contains("normalizer"));

        let mut wrong_split = base.clone();
        wrong_split["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] =
            serde_json::json!(".*");
        assert!(FastBpeEncoder::from_tokenizer_json(&wrong_split)
            .unwrap_err()
            .contains("pre_tokenizer"));

        let mut with_dropout = base.clone();
        with_dropout["model"]["dropout"] = serde_json::json!(0.1);
        assert!(FastBpeEncoder::from_tokenizer_json(&with_dropout)
            .unwrap_err()
            .contains("model config"));
    }
}
