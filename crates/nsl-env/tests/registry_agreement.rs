//! Workspace-level gate: every `NSL_*` variable the sources read is in the
//! registry, every registry entry is still read somewhere, and the wiki page
//! is the registry's rendering. A read the scanner cannot see through is
//! pinned by (file, expression) so a new one has to be acknowledged here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use nsl_env::scan::{scan_workspace, Read};
use nsl_env::REGISTRY;

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/nsl-env
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// `env::var(<expr>)` calls whose name the scanner cannot resolve, as
/// `(file, expr)`. Each is a read of something that is NOT an `NSL_*`
/// variable, or is registered under the names it can take — say which.
const KNOWN_DYNAMIC_READS: &[(&str, &str)] = &[
    // `find_ptxas` iterates over CUDA_PATH / CUDA_HOME: not NSL_ names.
    ("crates/nsl-runtime/src/cuda/mod.rs", "root"),
];

#[test]
fn every_env_read_is_registered_and_every_entry_is_read() {
    let sites = scan_workspace(&workspace_root()).expect("scan workspace sources");

    let mut read_at: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    let mut dynamic: BTreeSet<(String, String)> = BTreeSet::new();
    for site in &sites {
        match &site.read {
            Read::Named { name, line } => read_at
                .entry(name.as_str())
                .or_default()
                .push(format!("{}:{}", site.path, line)),
            Read::Dynamic { expr, .. } => {
                dynamic.insert((site.path.clone(), expr.clone()));
            }
        }
    }

    // Anti-vacuity floor: the scan found ~150 distinct names when this gate
    // was written. A parser regression that finds a handful must fail here,
    // not pass by finding nothing to complain about.
    assert!(
        read_at.len() >= 120,
        "scanner found only {} distinct NSL_* reads — the scan is broken, not the code",
        read_at.len()
    );

    let registered: BTreeSet<&str> = REGISTRY.iter().map(|v| v.name).collect();

    let unregistered: Vec<String> = read_at
        .iter()
        .filter(|(name, _)| !registered.contains(*name))
        .map(|(name, sites)| format!("  {name}\n      read at {}", sites.join(", ")))
        .collect();
    let stale: Vec<&str> = registered
        .iter()
        .copied()
        .filter(|name| !read_at.contains_key(name))
        .collect();
    let expected_dynamic: BTreeSet<(String, String)> = KNOWN_DYNAMIC_READS
        .iter()
        .map(|(p, e)| (p.to_string(), e.to_string()))
        .collect();
    let new_dynamic: Vec<String> = dynamic
        .difference(&expected_dynamic)
        .map(|(p, e)| format!("  {p}: env::var({e})"))
        .collect();
    let gone_dynamic: Vec<String> = expected_dynamic
        .difference(&dynamic)
        .map(|(p, e)| format!("  {p}: env::var({e})"))
        .collect();

    let mut problems = String::new();
    if !unregistered.is_empty() {
        problems.push_str(&format!(
            "\n{} NSL_* variable(s) are read but not in crates/nsl-env/src/registry.rs — add an entry \
             (name, kind, accepted values, default, tier, read_at, one-line doc):\n{}\n",
            unregistered.len(),
            unregistered.join("\n")
        ));
    }
    if !stale.is_empty() {
        problems.push_str(&format!(
            "\n{} registry entr{} no longer read anywhere — remove {}:\n  {}\n",
            stale.len(),
            if stale.len() == 1 { "y is" } else { "ies are" },
            if stale.len() == 1 { "it" } else { "them" },
            stale.join("\n  ")
        ));
    }
    if !new_dynamic.is_empty() {
        problems.push_str(&format!(
            "\n{} env::var read(s) the scanner cannot resolve to a name — if it reads NSL_* names, \
             register them and pass the name through a `const`; either way add it to \
             KNOWN_DYNAMIC_READS with a comment:\n{}\n",
            new_dynamic.len(),
            new_dynamic.join("\n")
        ));
    }
    if !gone_dynamic.is_empty() {
        problems.push_str(&format!(
            "\nKNOWN_DYNAMIC_READS lists read(s) that no longer exist — drop them:\n{}\n",
            gone_dynamic.join("\n")
        ));
    }
    assert!(problems.is_empty(), "{problems}");

    eprintln!(
        "nsl-env: {} registered variables, {} read sites, {} dynamic reads pinned",
        REGISTRY.len(),
        sites.len(),
        dynamic.len()
    );
}

#[test]
fn wiki_page_is_the_registry_rendering() {
    let page = workspace_root().join("docs/wiki/Environment-Variables.md");
    // `.gitattributes` pins the page to LF, but a checkout that predates the
    // pin (or a hand-edited copy) can carry CRLF; line endings are not what
    // this gate is about.
    let on_disk = std::fs::read_to_string(&page)
        .unwrap_or_else(|e| panic!("read {}: {e}", page.display()))
        .replace("\r\n", "\n");
    let rendered = nsl_env::render_markdown();
    assert!(
        on_disk == rendered,
        "{} differs from `nsl env list --markdown` — regenerate it:\n  \
         cargo run -p nsl-cli -- env list --markdown > {}",
        page.display(),
        page.display()
    );
}
