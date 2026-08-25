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
    let record: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("models/tokenizers/special_tokens.json")).unwrap(),
    )
    .unwrap();
    let surfaces: Vec<String> = record["surfaces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect();

    // The two scripts that ACT on the list (extract neutralizes,
    // train_tokenizer reserves) must actually LOAD it — the check is for the
    // load expression, not for the filename appearing somewhere: a comment
    // mentioning special_tokens.json while the code carries its own list is
    // precisely the regression this test exists to refuse. (An earlier
    // version checked `contains("special_tokens.json")`, which a comment
    // satisfies — review finding, 2026-08-24.)
    for script in ["tools/hfcorpus/extract.py", "tools/hfcorpus/train_tokenizer.py"] {
        let src = std::fs::read_to_string(root.join(script)).unwrap();
        let loads = src.lines().any(|l| {
            !l.trim_start().starts_with('#')
                && l.contains("special_tokens.json")
                && (src.contains("[\"surfaces\"]") || src.contains("['surfaces']"))
        });
        assert!(
            loads,
            "{script} no longer loads `surfaces` from the shared record"
        );
    }

    // No pipeline script may carry an inline copy, in ANY spelling: one-line
    // lists, renamed variables and single quotes all evaded the old two-line
    // pattern match. The detector counts reserved surfaces appearing as
    // EXACTLY-QUOTED standalone string literals ("<|x|>" or '<|x|>') on one
    // non-comment line — which is what a list's elements are, whatever the
    // variable is named. It deliberately does NOT match a surface embedded
    // inside a longer template string: extract.py's chat renderer
    // legitimately EMITS `f"<|im_start|>{role}…"` when building documents,
    // and the first version of this detector flagged exactly that line.
    for entry in std::fs::read_dir(root.join("tools/hfcorpus")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("py") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        for (ln, line) in src.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with('#') {
                continue;
            }
            let hits = surfaces
                .iter()
                .filter(|s| {
                    t.contains(&format!("\"{s}\"")) || t.contains(&format!("'{s}'"))
                })
                .count();
            assert!(
                hits < 2,
                "{}:{}: {hits} reserved surfaces as literals on one code line \
                 — an inline copy of the list. The single source of truth is \
                 models/tokenizers/special_tokens.json:\n  {line}",
                path.display(),
                ln + 1
            );
        }
    }
}

#[test]
fn every_committed_tokenizer_file_is_pinned_by_the_manifest() {
    // The hash check walks the MANIFEST's list; a fourth .json quietly added
    // to models/tokenizers/ would be pinned by nothing (review finding,
    // 2026-08-24). The direction is inverted here: every committed file must
    // be listed.
    let root = repo_root();
    let man = manifest();
    let pinned = man["tokenizers_sha256"].as_object().unwrap();
    for entry in std::fs::read_dir(root.join("models/tokenizers")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let rel = format!("models/tokenizers/{}", path.file_name().unwrap().to_str().unwrap());
        assert!(
            pinned.contains_key(&rel),
            "{rel} is committed but not pinned by the manifest — run \
             manifest.py build (and mean it: an unpinned tokenizer is \
             invisible to every hash check)"
        );
    }
}

#[test]
fn bin_records_are_complete_and_composition_is_coherent() {
    let man = manifest();
    let bins = man["bins"].as_object().expect("bins object");
    assert_eq!(
        bins.len(),
        8,
        "the v2 corpus is exactly 8 bins (item 5 added stack_val); a new \
         bin needs a deliberate manifest rebuild, and a missing one means \
         the record is stale"
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

/// Item 5: the corpus is manifested for EVALUATION, so the manifest must
/// carry the decontamination record — method stated (not implied, the
/// `dedup` doctrine), benchmarks pinned as revisioned sources like every
/// other input, and every validation set scanned CLEAN. The gate reads
/// committed JSON only; the scan itself is local (decontaminate.py), which
/// is exactly why its RESULTS must live here — a claim with no committed
/// record is a claim the tree cannot check.
#[test]
fn the_decontamination_record_is_complete_and_val_sets_are_clean() {
    let man = manifest();
    let decon = man["decontamination"]
        .as_object()
        .expect("manifest carries no decontamination record — run \
                 tools/hfcorpus/decontaminate.py then manifest.py build");
    let method = decon["method"].as_str().unwrap_or("");
    assert!(
        method.len() > 40 && method.contains("normalized-line"),
        "the decontamination method must be STATED precisely, not implied: {method:?}"
    );

    // The benchmarks it scanned against must be pinned sources with
    // revisions (they ride the every_hf_source test's loop too; this
    // asserts the linkage in the other direction).
    let sources = man["sources"].as_object().expect("sources");
    for bench in decon["benchmarks"].as_object().expect("benchmarks").keys() {
        assert!(
            sources.contains_key(bench),
            "decontamination names benchmark '{bench}' but sources does not pin it"
        );
    }

    // Every validation set: present in the record and CLEAN. A val set that
    // contains benchmark text scores memorization, not generalization.
    let sets = decon["sets"].as_object().expect("sets");
    for val in ["val-stack", "val-web", "val-sft"] {
        let e = sets
            .get(val)
            .unwrap_or_else(|| panic!("decontamination record does not cover {val}"));
        let docs = e["documents"].as_u64().unwrap_or(0);
        assert!(docs > 100, "{val}: implausibly few documents scanned ({docs})");
        assert_eq!(
            e["contaminated_documents"].as_u64(),
            Some(0),
            "{val} is recorded as CONTAMINATED — re-cut it with \
             extract.py --drop-contaminated"
        );
    }

    // Train sets are REPORTED, not required clean (the corpus is shipped;
    // the record is what makes its evaluation honest) — but they must be
    // present, or the record is a val-only claim wearing a corpus name.
    for train in ["stack-train", "web-train", "sft-train"] {
        assert!(
            sets.contains_key(train),
            "decontamination record does not report {train}"
        );
    }
}
