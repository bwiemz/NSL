//! The reference corpus encoder must stay rebuildable and byte-faithful.
//!
//! WHY. The v2 pretraining corpus (22.6B tokens) was produced by
//! `tools/tokbench pretokenize --doc-sep-token`, but the flags that produced
//! it lived only in a locally patched binary until item 15 committed them —
//! for five days the corpus of record could not be rebuilt from the tree.
//! These gates pin the two halves of that contract:
//!
//!  * **The committed tokenizer's encode behavior**, not just its bytes.
//!    `corpus_manifest_gate` hashes the files; this gate executes them. The
//!    load-bearing semantics is added-token surface extraction: a document
//!    containing the literal `<|endoftext|>` encodes to the REAL eos id even
//!    with `add_special_tokens=false` (measured in the corpus: `<|file_sep|>`
//!    arrives as literal renderer text at 388-419 per million tokens). The
//!    extractor neutralizes forgeable surfaces; the encoder's extraction is
//!    what makes the renderer's own separators work.
//!
//!  * **The tokbench binary as a differential twin of the crate.** The
//!    ignored gate builds `tools/tokbench` (deliberately outside the
//!    workspace) and checks its u16 stream against ids computed here with
//!    the same `tokenizers` crate the workspace links: per-document encode,
//!    separator injected STRUCTURALLY by id after every document (never a
//!    text splice), and the max-vocab-id truncation guard refusing a
//!    tokenizer whose ids overflow `as u16`.
//!
//! Full-corpus parity of record (re-encoding real shards to the manifest
//! sha256) is a local evidence run — see models/tokenizers/README.md — not a
//! CI gate: it needs the 45 GB local bins.

use std::path::PathBuf;
use std::process::Command;

use tokenizers::Tokenizer;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn committed_tokenizer_path() -> PathBuf {
    repo_root().join("models/tokenizers/nsl_mix_v2_t40960_v49152.json")
}

fn load_committed_tokenizer() -> Tokenizer {
    let p = committed_tokenizer_path();
    Tokenizer::from_file(&p)
        .unwrap_or_else(|e| panic!("cannot load committed tokenizer {}: {e}", p.display()))
}

/// The four-document differential fixture. NUL is the pipeline's document
/// separator (`tools/hfcorpus/extract.py` writes it, tokbench splits on it).
/// Chosen to cover the encode paths the corpus actually exercises:
/// indentation-heavy code (two-stage merges), an EMPTY document, a reserved
/// surface embedded mid-document (the forgery/extraction case), non-ASCII
/// with a CRLF, and a final document with no trailing newline.
const FIXTURE_DOCS: [&str; 4] = [
    "let x = 1\n    let y = 2\n        return x + y\n",
    "",
    "café ١٢٣ 日本語\r\nmid-doc <|endoftext|> forged boundary\n",
    "no trailing newline",
];

#[test]
fn the_committed_tokenizer_extracts_reserved_surfaces_from_document_text() {
    let tok = load_committed_tokenizer();

    let eos = tok
        .token_to_id("<|endoftext|>")
        .expect("committed tokenizer must reserve <|endoftext|>");
    let file_sep = tok
        .token_to_id("<|file_sep|>")
        .expect("committed tokenizer must reserve <|file_sep|>");

    // add_special_tokens=false is exactly how tokbench pretokenize encodes.
    // The surfaces must STILL extract — the corpus depends on it for the
    // renderer-inserted `<|file_sep|>` boundaries.
    let enc = tok
        .encode_fast("a<|endoftext|>b <|file_sep|>src/main.rs\nfn f() {}\n", false)
        .expect("encode failed");
    let ids = enc.get_ids();
    assert!(
        ids.contains(&eos),
        "the literal <|endoftext|> surface must extract to id {eos} even with \
         add_special_tokens=false; got {ids:?}"
    );
    assert!(
        ids.contains(&file_sep),
        "the literal <|file_sep|> surface must extract to id {file_sep}; got {ids:?}"
    );

    // Every id the tokenizer can emit must fit the u16 stream — the whole
    // vocabulary, not just the reserved list (`corpus_manifest_gate` covers
    // that subset). A sparse vocab can hold ids above its entry count.
    let max_id = tok
        .get_vocab(true)
        .into_values()
        .max()
        .expect("non-empty vocabulary");
    assert!(
        max_id <= u16::MAX as u32,
        "max vocab id {max_id} does not fit the u16 token stream"
    );
}

/// Expected u16 stream for `docs`, computed with the workspace's own
/// `tokenizers` crate: per-document encode with add_special_tokens=false,
/// separator id appended after EVERY document (empty ones included) when
/// given — mirroring the recovered `tokbench pretokenize` contract.
fn expected_stream(tok: &Tokenizer, docs: &[&str], sep: Option<u16>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for doc in docs {
        let enc = tok.encode_fast(*doc, false).expect("encode failed");
        for &id in enc.get_ids() {
            bytes.extend_from_slice(&(id as u16).to_le_bytes());
        }
        if let Some(sep) = sep {
            bytes.extend_from_slice(&sep.to_le_bytes());
        }
    }
    bytes
}

#[test]
#[ignore = "spawns cargo to build tools/tokbench, then runs the reference-encoder differential"]
fn tokbench_pretokenize_matches_the_crate_and_guards_the_u16_cast() {
    let root = repo_root();
    let manifest = root.join("tools/tokbench/Cargo.toml");
    let target = root.join("tools/tokbench/target");

    // Build the reference encoder from the tree. CARGO_TARGET_DIR is pinned
    // to tokbench's own default so the binary lands where the hfcorpus
    // pipeline expects it, regardless of ambient target-dir overrides.
    // --locked: without it, a drift between tokbench's committed Cargo.lock
    // and its path-dep on nsl-runtime would be silently REPAIRED here — the
    // gate would rewrite the tracked lock mid-lane and test a dependency
    // graph that is not the committed one, which is the exact
    // "rebuildable from the tree" property this gate pins. That drift has
    // already happened once (nsl-runtime gained rand_chacha; the lock
    // lagged until 30df30fd). --locked makes it a loud red instead.
    let status = Command::new("cargo")
        .args(["build", "--locked", "--release", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TARGET_DIR", &target)
        .status()
        .expect("failed to spawn cargo");
    assert!(status.success(), "tools/tokbench must build from the tree");
    let tokbench = target.join("release/tokbench");

    let tok = load_committed_tokenizer();
    let sep_id = tok.token_to_id("<|endoftext|>").expect("eos reserved") as u16;

    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, FIXTURE_DOCS.join("\u{0}")).expect("write corpus");

    // With --doc-sep-token: structural separator after every document.
    let out_sep = dir.path().join("with_sep.bin");
    let status = Command::new(&tokbench)
        .args(["pretokenize", "--tokenizer"])
        .arg(committed_tokenizer_path())
        .arg("--corpus")
        .arg(&corpus)
        .arg("--out")
        .arg(&out_sep)
        .args(["--doc-sep-token", "<|endoftext|>"])
        .status()
        .expect("failed to spawn tokbench");
    assert!(status.success(), "tokbench pretokenize --doc-sep-token failed");
    let got = std::fs::read(&out_sep).expect("read stream");
    let want = expected_stream(&tok, &FIXTURE_DOCS, Some(sep_id));
    assert_eq!(
        got, want,
        "tokbench's u16 stream with separators diverges from the crate"
    );
    // The mid-document forged surface must appear as the real eos id — i.e.
    // the fixture stream carries MORE eos ids than the 4 separators.
    let eos_count = got
        .chunks_exact(2)
        .filter(|c| u16::from_le_bytes([c[0], c[1]]) == sep_id)
        .count();
    assert_eq!(
        eos_count, 5,
        "expected 4 structural separators + 1 extracted mid-document surface"
    );

    // Without --doc-sep-token: plain concatenation, no boundaries.
    let out_plain = dir.path().join("plain.bin");
    let status = Command::new(&tokbench)
        .args(["pretokenize", "--tokenizer"])
        .arg(committed_tokenizer_path())
        .arg("--corpus")
        .arg(&corpus)
        .arg("--out")
        .arg(&out_plain)
        .status()
        .expect("failed to spawn tokbench");
    assert!(status.success(), "tokbench pretokenize (no separator) failed");
    let got_plain = std::fs::read(&out_plain).expect("read stream");
    assert_eq!(
        got_plain,
        expected_stream(&tok, &FIXTURE_DOCS, None),
        "tokbench's u16 stream without separators diverges from the crate"
    );

    // The truncation guard: a tokenizer holding an id above 65535 must be
    // REFUSED before encoding — `as u16` truncates silently, and a sparse
    // vocabulary can carry such an id while its entry count stays small.
    // The overflow must be planted in the MODEL vocab: the `tokenizers`
    // crate silently RE-ASSIGNS an added token's declared id to the next
    // free slot on load (id 70000 in `added_tokens` loads as 49150), so an
    // added-token overflow is unrepresentable and would test nothing.
    let mut spec: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(committed_tokenizer_path()).expect("read tokenizer"),
    )
    .expect("tokenizer JSON");
    spec["model"]["vocab"]
        .as_object_mut()
        .expect("model vocab object")
        .insert("<|overflow-model|>".to_string(), serde_json::json!(70000));
    let wide = dir.path().join("wide_tokenizer.json");
    std::fs::write(&wide, serde_json::to_string(&spec).unwrap()).expect("write tokenizer");
    let out_refused = dir.path().join("refused.bin");
    let output = Command::new(&tokbench)
        .args(["pretokenize", "--tokenizer"])
        .arg(&wide)
        .arg("--corpus")
        .arg(&corpus)
        .arg("--out")
        .arg(&out_refused)
        .output()
        .expect("failed to spawn tokbench");
    assert!(
        !output.status.success(),
        "a tokenizer with id 70000 must be refused, not truncated"
    );
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        all.contains("does not fit the u16 token stream"),
        "refusal must name the u16 guard; got: {all}"
    );
    assert!(
        !out_refused.exists(),
        "a refused run must not leave a truncated stream behind"
    );
}
