//! Static gate: every Rust-side allocation-pool bracket goes through
//! `PoolGuard`, never the manual `set_alloc_pool(P); ...; set_alloc_pool(prev)`
//! spelling.
//!
//! The manual spelling is panic-UNSAFE: a panic inside the bracket leaves the
//! thread pinned to `Persistent`, and every later Transient allocation on
//! that thread lands in Persistent-tagged segments that `drain_all` never
//! returns to the driver — VRAM held at high-water forever — while the
//! per-pool accounting the two-pool design exists for silently lies. Four such brackets existed (cfie engine x2, the meta
//! upload cache, the strided-copy plan cache) before `PoolGuard`; this scan
//! keeps the count at zero the way `SurfaceGuard`'s discipline is kept — a
//! static gate over enumerable sites, not a runtime assert
//! (`debug_assert!` is a no-op in the release builds CI ships).
//!
//! `set_alloc_pool` legitimately remains in exactly TWO places outside the
//! allocator module: the two `extern "C"` entries in `tensor/mod.rs`
//! (`nsl_gpu_set_persistent_pool` / `nsl_gpu_set_transient_pool`). Those are
//! the CODEGEN-emitted brackets — emitted IR cannot hold a Rust guard, so
//! open/close calls are the only shape possible there, and their pairing is
//! codegen's contract, not this crate's.
//!
//! Known scan limits, accepted: an aliased import (`use ... as sap`) would
//! evade it, and the literal inside a string would false-POSITIVE — loud,
//! the safe direction. Neither exists in the tree today.

use std::path::{Path, PathBuf};

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Drop `//` comment lines and trailing comments, then strip all whitespace —
/// the multi-line-call-chain lesson: a line-based scan misses
/// `set_alloc_pool(\n    pool\n)` and certifies what it cannot see.
fn code_only(text: &str) -> String {
    text.lines()
        .map(|l| {
            let b = l.as_bytes();
            for i in 0..b.len().saturating_sub(1) {
                if b[i] == b'/' && b[i + 1] == b'/' && (i == 0 || b[i - 1] != b':') {
                    return &l[..i];
                }
            }
            l
        })
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
        .split_whitespace()
        .collect()
}

#[test]
fn every_pool_bracket_is_a_guard() {
    let allowed: &[&str] = &[
        // The definition, the guard, and their unit tests.
        "cuda/caching_allocator.rs",
        // The two codegen-emitted bracket entries (see the header).
        "tensor/mod.rs",
    ];

    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    let mut dirs = vec![src_root()];
    while let Some(dir) = dirs.pop() {
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                dirs.push(p);
                continue;
            }
            if p.extension().is_none_or(|x| x != "rs") {
                continue;
            }
            scanned += 1;
            let rel = p
                .strip_prefix(src_root())
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if allowed.contains(&rel.as_str()) {
                continue;
            }
            let src = code_only(&std::fs::read_to_string(&p).unwrap());
            if src.contains("set_alloc_pool(") {
                offenders.push(rel);
            }
        }
    }
    assert!(
        scanned > 50,
        "only {scanned} files scanned — the walk failed, so an empty result \
         would prove nothing"
    );
    assert!(
        offenders.is_empty(),
        "manual set_alloc_pool bracket(s) outside the allowed sites: \
         {offenders:?} — use PoolGuard (panic-safe restore), or if this is a \
         genuinely new codegen-emitted bracket entry, add the file here with \
         the reasoning written out"
    );
}

/// The allowlist must keep describing reality: the two extern entries must
/// still exist in `tensor/mod.rs` and still call `set_alloc_pool` — a stale
/// allowlist entry is how a real gap gets waved through later.
#[test]
fn the_allowed_extern_entries_still_exist() {
    let t = std::fs::read_to_string(src_root().join("tensor/mod.rs")).unwrap();
    for f in ["nsl_gpu_set_persistent_pool", "nsl_gpu_set_transient_pool"] {
        assert!(
            t.contains(&format!("pub extern \"C\" fn {f}()")),
            "{f} moved or was renamed — update the allowlist and this test"
        );
    }
    assert_eq!(
        code_only(&t).matches("set_alloc_pool(").count(),
        2,
        "tensor/mod.rs should contain exactly the two extern bracket calls"
    );
}
