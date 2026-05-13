//! SASS baseline file format and helpers for PCA Tier B CI gate.
//!
//! Baseline file format (plain text, one per variant):
//! ```text
//! variant_name=<name>
//! sm=<arch>
//! instruction_count=<N>
//! spill_bytes=<N>
//! tolerance=<N>
//! recorded_date=<YYYY-MM-DD>
//! ```
//!
//! The `sm` field records the ptxas architecture flag used to compile the
//! PTX (e.g. 75 for `-arch=sm_75`).  This must match the `.target sm_NN`
//! directive inside the synthesized PTX.
//!
//! Per spec §5.2 (per-variant SASS baselines) and §6.3.1 (institutional
//! baseline pattern).

use std::path::PathBuf;
use std::process::Command;

// ---------------------------------------------------------------------------
// Baseline struct
// ---------------------------------------------------------------------------

pub struct Baseline {
    pub variant_name: String,
    pub sm: u32,
    pub instruction_count: usize,
    pub spill_bytes: usize,
    pub tolerance: usize,
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

pub fn parse_baseline_file(path: &PathBuf) -> Baseline {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read baseline {}: {}", path.display(), e));
    let mut b = Baseline {
        variant_name: String::new(),
        sm: 0,
        instruction_count: 0,
        spill_bytes: 0,
        tolerance: 2,
    };
    for line in content.lines() {
        if let Some(v) = line.strip_prefix("variant_name=") {
            b.variant_name = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("sm=") {
            b.sm = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("instruction_count=") {
            b.instruction_count = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("spill_bytes=") {
            b.spill_bytes = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("tolerance=") {
            b.tolerance = v.trim().parse().unwrap_or(2);
        }
    }
    b
}

pub fn write_baseline_file(path: &PathBuf, b: &Baseline, recorded_date: &str) {
    let content = format!(
        "variant_name={}\nsm={}\ninstruction_count={}\nspill_bytes={}\ntolerance={}\nrecorded_date={}\n",
        b.variant_name, b.sm, b.instruction_count, b.spill_bytes, b.tolerance, recorded_date,
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create baseline dir");
    }
    std::fs::write(path, content)
        .unwrap_or_else(|e| panic!("write baseline {}: {}", path.display(), e));
}

// ---------------------------------------------------------------------------
// Tool location (mirrors pca_sass_byte_identity.rs helpers — kept separate
// per Task 10 constraints, no cross-file refactor)
// ---------------------------------------------------------------------------

pub fn find_ptxas() -> Option<String> {
    for name in ["ptxas", "ptxas.exe"] {
        if Command::new(name)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
        {
            return Some(name.to_string());
        }
    }
    let win =
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe";
    if std::path::Path::new(win).exists() {
        return Some(win.to_string());
    }
    None
}

pub fn find_cuobjdump() -> Option<String> {
    for name in ["cuobjdump", "cuobjdump.exe"] {
        if Command::new(name)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
        {
            return Some(name.to_string());
        }
    }
    let win =
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\cuobjdump.exe";
    if std::path::Path::new(win).exists() {
        return Some(win.to_string());
    }
    None
}

// ---------------------------------------------------------------------------
// PTX → cubin → SASS pipeline
// ---------------------------------------------------------------------------

/// Compile PTX to a cubin, then dump SASS via cuobjdump.
///
/// Returns `Some((sass_text, ptxas_stderr))` on success, `None` when
/// ptxas or cuobjdump is unavailable or the PTX fails to assemble.
///
/// `ptxas_stderr` contains the verbose stats output (spill counts,
/// register usage) when ptxas is invoked with the `-v` flag.
pub fn ptx_to_sass_and_ptxas_log(ptx: &str, sm: u32) -> Option<(String, String)> {
    let ptxas_path = find_ptxas()?;
    let cuobjdump_path = find_cuobjdump()?;

    let tmpdir = tempfile::tempdir().ok()?;
    let ptx_path = tmpdir.path().join(format!("tier_b_baseline_sm{sm}.ptx"));
    let cubin_path = tmpdir.path().join(format!("tier_b_baseline_sm{sm}.cubin"));

    // Strip trailing NUL bytes: synthesize_flash_attention_ptx_v2_with_tier_b
    // appends a NUL terminator for cuModuleLoadData, but ptxas rejects NUL bytes.
    let ptx_bytes: Vec<u8> = ptx.as_bytes().iter().copied().filter(|&b| b != 0).collect();
    std::fs::write(&ptx_path, &ptx_bytes).ok()?;

    let ptxas_out = Command::new(&ptxas_path)
        .args([&format!("-arch=sm_{sm}"), "-v", "-O2", "-o"])
        .arg(&cubin_path)
        .arg(&ptx_path)
        .output()
        .ok()?;

    if !ptxas_out.status.success() {
        eprintln!(
            "[sass_baseline_helpers] ptxas rejected sm_{sm}: {}",
            String::from_utf8_lossy(&ptxas_out.stderr)
        );
        return None;
    }
    let ptxas_stderr = String::from_utf8_lossy(&ptxas_out.stderr).to_string();

    let dump = Command::new(&cuobjdump_path)
        .args(["--dump-sass"])
        .arg(&cubin_path)
        .output()
        .ok()?;

    if !dump.status.success() {
        eprintln!(
            "[sass_baseline_helpers] cuobjdump failed: {}",
            String::from_utf8_lossy(&dump.stderr)
        );
        return None;
    }

    let sass = String::from_utf8(dump.stdout).ok()?;
    Some((sass, ptxas_stderr))
}

// ---------------------------------------------------------------------------
// SASS instruction counting
// ---------------------------------------------------------------------------

/// Count SASS instruction lines.
///
/// cuobjdump SASS lines containing actual instructions look like:
/// ```text
///         /*0ed0*/                   MOV R1, c[0x0][0x28] ;
/// ```
/// i.e., the trimmed line starts with `/*` followed by hex digits.
///
/// Comment/label lines and blank lines do not match this pattern.
pub fn count_sass_instructions(sass: &str) -> usize {
    sass.lines()
        .filter(|line| {
            let t = line.trim_start();
            if !t.starts_with("/*") {
                return false;
            }
            // Must be `/*XXXX*/` — confirm the chars after `/*` are hex
            // until the closing `*/`.
            let rest = &t[2..];
            let hex_end = rest.find("*/").unwrap_or(0);
            hex_end > 0 && rest[..hex_end].chars().all(|c| c.is_ascii_hexdigit())
        })
        .count()
}

// ---------------------------------------------------------------------------
// Spill byte extraction
// ---------------------------------------------------------------------------

/// Parse the spill-store bytes from ptxas `-v` stderr.
///
/// ptxas emits a line like:
/// ```text
/// ptxas info    : 0 bytes spill stores, 0 bytes spill loads
/// ```
/// This function returns the first "N bytes spill stores" value, or 0 if
/// the line is absent (no spills, or old ptxas format).
pub fn parse_spill_bytes(ptxas_log: &str) -> usize {
    for line in ptxas_log.lines() {
        if let Some(idx) = line.find(" bytes spill stores") {
            let prefix = &line[..idx];
            if let Some(num_str) = prefix.split_whitespace().last() {
                return num_str.parse().unwrap_or(0);
            }
        }
    }
    0
}
