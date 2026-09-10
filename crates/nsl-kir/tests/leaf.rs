//! `nsl-kir` is a leaf crate (roadmap A2 step 1): it depends on nothing, so
//! both `nsl-codegen` and `nsl-runtime` can build kernels on it without
//! either depending on the other. This pins the `[dependencies]` table of
//! the crate's own manifest empty — a `path =` or `workspace = true` entry
//! appearing there is the edge this crate exists to avoid.

#[test]
fn the_manifest_declares_no_dependencies() {
    let manifest = include_str!("../Cargo.toml");
    let deps = manifest
        .split("[dependencies]")
        .nth(1)
        .expect("Cargo.toml has a [dependencies] table");
    // Up to the next table header, every line is blank or a comment.
    let body = deps.split("\n[").next().unwrap();
    let offending: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    assert!(
        offending.is_empty(),
        "nsl-kir must stay dependency-free; found: {offending:?}"
    );
}
