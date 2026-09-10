//! `python/nslpy/_abi.py` is generated from `nsl_abi::capi` (`nsl abi
//! python`, roadmap A3 step 2) and pinned here, the arrangement the
//! generated wiki pages use: the file cannot drift from the table because
//! this test compares the two byte for byte.

fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn python_mirror_is_the_table_rendering() {
    let path = workspace_root().join("python/nslpy/_abi.py");
    let on_disk = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .replace("\r\n", "\n");
    let rendered = nsl_abi::capi::render_python();
    if on_disk != rendered {
        let first_diff = on_disk
            .lines()
            .zip(rendered.lines())
            .position(|(a, b)| a != b)
            .map(|i| i + 1)
            .unwrap_or_else(|| on_disk.lines().count().min(rendered.lines().count()) + 1);
        panic!(
            "{} differs from `nsl abi python` (first difference at line {first_diff}; on disk {} \
             lines, rendered {}) — regenerate it:\n  cargo run -p nsl-cli -- abi python > {}",
            path.display(),
            on_disk.lines().count(),
            rendered.lines().count(),
            path.display()
        );
    }
}
