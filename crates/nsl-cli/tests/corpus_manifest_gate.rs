//! The committed corpus manifest must be internally consistent, and the
//! reserved-surface list must have exactly ONE copy.
//!
//! WHY. The v2 pretraining corpus is 45 GB of local-only bins; everything
//! that identifies it — bin hashes, token counts, the tokenizer that encoded
//! it, the HF revisions it came from — lives in
//! `models/datasets/CORPUS_MANIFEST_v2.json`. Two failure shapes this gate
//! exists to refuse:
//!
//!  * **A silent tokenizer swap.** The per-bin manifests name the tokenizer
//!    by PATH; retraining to the same filename would invalidate every
//!    recorded hash while every path-based check kept passing. The committed
//!    tokenizer files must hash to exactly what the manifest records.
//!    (Hashing them here is cheap — they are ~1.7 MB each.)
//!  * **A forked reserved-token list.** A surface reserved by the tokenizer
//!    but not neutralized by the extractor is a boundary any document can
//!    forge — a real risk, not a theoretical one: 35 forgeable surfaces were
//!    measured in 48 MB of code text. The list previously existed as two
//!    inline Python copies "kept equal" by a comment; they now must load from
//!    `models/tokenizers/special_tokens.json`, and this gate refuses an
//!    inline list if one ever grows back.
//!
//! This gate reads committed files only — no GPU, no network, no 45 GB
//! hashing (that is `manifest.py check --hash-bins`, run where the bins are).

use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn manifest() -> serde_json::Value {
    let p = repo_root().join("models/datasets/CORPUS_MANIFEST_v2.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("corpus manifest missing at {}: {e}", p.display())
    }))
    .expect("corpus manifest is not valid JSON")
}

fn sha256_hex(path: &std::path::Path) -> String {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    format!("{:x}", Sha256::digest(&bytes))
}

#[test]
fn committed_tokenizers_hash_to_what_the_manifest_records() {
    let root = repo_root();
    let man = manifest();
    let hashes = man["tokenizers_sha256"]
        .as_object()
        .expect("tokenizers_sha256 must be an object");
    assert!(
        hashes.len() >= 3,
        "expected at least v1, v2 and special_tokens.json to be pinned, got {}",
        hashes.len()
    );
    for (rel, want) in hashes {
        let got = sha256_hex(&root.join(rel));
        assert_eq!(
            &got,
            want.as_str().unwrap(),
            "{rel} does not hash to the manifest's record — the committed \
             tokenizer changed without `manifest.py build`, which would let \
             a retrained tokenizer silently invalidate every bin hash"
        );
    }
}

#[test]
fn reserved_ids_agree_between_manifest_record_and_tokenizer() {
    let root = repo_root();
    let man = manifest();

    // manifest ↔ special_tokens.json
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("models/tokenizers/special_tokens.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        man["reserved"]["ids"], record["ids"],
        "manifest reserved ids diverge from special_tokens.json"
    );
    assert_eq!(
        man["reserved"]["surfaces"], record["surfaces"],
        "manifest surfaces diverge from special_tokens.json"
    );

    // record ↔ the actual tokenizer build. Both halves must describe the SAME
    // artifact, which is why the ids are re-derived from the tokenizer JSON
    // rather than trusted.
    let tok_rel = man["tokenizer_of_record"].as_str().unwrap();
    let tok: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join(tok_rel)).unwrap()).unwrap();
    let added: std::collections::HashMap<&str, i64> = tok["added_tokens"]
        .as_array()
        .expect("tokenizer has added_tokens")
        .iter()
        .map(|a| (a["content"].as_str().unwrap(), a["id"].as_i64().unwrap()))
        .collect();
    for (surface, id) in man["reserved"]["ids"].as_object().unwrap() {
        assert_eq!(
            added.get(surface.as_str()).copied(),
            id.as_i64(),
            "{surface}: the tokenizer of record assigns a different id than \
             the manifest — the two describe different builds"
        );
    }
    // Every reserved surface fits the u16 stream with headroom the loader
    // assumes (vocab 49152 + specials < 65536).
    for (surface, id) in man["reserved"]["ids"].as_object().unwrap() {
        assert!(
            id.as_i64().unwrap() < 65536,
            "{surface} id {} does not fit the u16 stream",
            id
        );
    }
}

#[test]
fn the_reserved_surface_list_has_exactly_one_copy() {
    let root = repo_root();
    // The scripts that must agree on the list. report.py only *mentions* one
    // surface in prose; the two that ACT on the list are extract (neutralizes)
    // and train_tokenizer (reserves).
    for script in ["tools/hfcorpus/extract.py", "tools/hfcorpus/train_tokenizer.py"] {
        let src = std::fs::read_to_string(root.join(script)).unwrap();
        assert!(
            src.contains("special_tokens.json"),
            "{script} no longer loads the shared reserved-surface record"
        );
        // An inline list literal is the failure mode this gate exists for:
        // two copies "kept equal" by a comment. `SPECIALS = [` with a
        // following string literal is the shape both dead copies had.
        let forked = src
            .lines()
            .zip(src.lines().skip(1))
            .any(|(a, b)| a.trim_start().starts_with("SPECIALS = [")
                && b.trim().starts_with("\"<|"));
        assert!(
            !forked,
            "{script} has grown back an inline SPECIALS list — the single \
             source of truth is models/tokenizers/special_tokens.json"
        );
    }
}

#[test]
fn bin_records_are_complete_and_composition_is_coherent() {
    let man = manifest();
    let bins = man["bins"].as_object().expect("bins object");
    assert_eq!(
        bins.len(),
        7,
        "the v2 corpus is exactly 7 bins; a new bin needs a deliberate \
         manifest rebuild, and a missing one means the record is stale"
    );
    for (name, entry) in bins {
        let sha = entry["sha256"].as_str().unwrap_or("");
        assert_eq!(sha.len(), 64, "{name}: sha256 malformed: {sha:?}");
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "{name}: sha256 is not hex"
        );
        assert!(
            entry["tokens"].as_i64().unwrap_or(0) > 0,
            "{name}: token count missing or zero"
        );
    }
    // The mixture's composition must account for its own token count: the
    // three component entries sum to the mixture total exactly (the mix is
    // concatenation + injection, not sampling with loss).
    let mix = &bins["pretrain_train.bin"];
    let total: i64 = mix["composition"]
        .as_object()
        .expect("pretrain_train.bin carries its composition")
        .values()
        .map(|c| c["tokens"].as_i64().unwrap())
        .sum();
    assert_eq!(
        total,
        mix["tokens"].as_i64().unwrap(),
        "composition token counts do not sum to the mixture's own count"
    );
    // Shares are fractions of the whole and must cover it (±0.5%).
    let share: f64 = mix["composition"]
        .as_object()
        .unwrap()
        .values()
        .map(|c| c["share"].as_f64().unwrap())
        .sum();
    assert!(
        (share - 1.0).abs() < 0.005,
        "composition shares sum to {share}, not 1.0"
    );
}

#[test]
fn every_hf_source_records_the_downloaded_revision() {
    let man = manifest();
    for (name, src) in man["sources"].as_object().unwrap() {
        let repo = src["repo"].as_str().unwrap();
        if repo.starts_with("local:") {
            continue;
        }
        let rev = src["revision"].as_str().unwrap_or("");
        assert_eq!(
            rev.len(),
            40,
            "{name}: HF revision missing or not a commit hash ({rev:?}) — \
             without it the corpus is not re-fetchable, only describable"
        );
        assert!(
            src["files"].as_array().is_some_and(|f| !f.is_empty()),
            "{name}: no on-disk file list recorded"
        );
        assert!(
            src["dedup"].as_str().is_some_and(|d| !d.is_empty()),
            "{name}: dedup status must be stated, not implied"
        );
    }
}
