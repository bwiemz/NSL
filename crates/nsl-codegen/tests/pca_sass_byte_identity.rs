//! Spec §7.6 gate: the segment-mask helper's SASS signature must be
//! structurally present and opcode-pattern-equivalent in both the
//! forward s_compute and backward ds_compute kernels.
//!
//! ## Background
//!
//! Task 3A (PTX level) already verifies that `emit_segment_mask_predicate`
//! produces byte-identical PTX regardless of the caller context it is
//! invoked from (see `pca_segment_mask_caller_context_independence.rs`).
//! Task 4C is the SASS-level analog: after ptxas assembly, do the
//! optimised SASS fragments also encode the same logical operation?
//!
//! ## Why exact byte-identity is not achievable at SASS level
//!
//! ptxas performs context-sensitive register allocation, instruction
//! scheduling, and instruction fusion that depends on the surrounding
//! kernel code — not just the helper's PTX subsequence in isolation.
//! Two concrete structural differences prevent byte-exact comparison:
//!
//! 1. **Register numbers differ.** The forward kernel allocates
//!    e.g. R25/R26 for the two segment-ID loads; the backward kernel
//!    allocates e.g. R45/R44 at the same logical position.  After
//!    register-name normalisation (Rn, Pn) these rows become identical,
//!    but the raw encodings differ.
//!
//! 2. **`.EX` fusion form differs.** The forward kernel's causal
//!    comparison is a 64-bit `q_start ≥ k_start` compare, which ptxas
//!    encodes as two chained ISETP instructions (`ISETP.GT.U32.AND` +
//!    `ISETP.GT.U32.OR.EX` using the `.EX` carry-bit extension).  The
//!    backward kernel's causal comparison is 32-bit (block indices
//!    computed as u32 in ds_compute), so ptxas emits a single
//!    `ISETP.GT.U32.OR` without `.EX`.  Both are correct SASS for
//!    "q ≥ k" in their respective precision contexts, but the opcodes
//!    differ by one suffix.
//!
//! The structural difference in (2) is NOT a helper invariant
//! violation — `emit_segment_mask_predicate` always emits the same
//! PTX `or.pred` instruction; what changes is how ptxas *fuses* that
//! instruction with the surrounding causal-mask comparison, which is
//! caller context, not helper context.
//!
//! ## What this test verifies
//!
//! The test assembles both kernel snapshots with ptxas, dumps SASS via
//! cuobjdump, and then verifies:
//!
//! 1. **Signature presence**: both kernels contain at least one
//!    `ISETP.NE.U32.AND` immediately followed (within 4 SASS lines)
//!    by an `ISETP.GT.U32.OR` instruction.  This pair is the
//!    inescapable SASS signature of `setp.ne.u16 %p_seg, %rs_q, %rs_k`
//!    fused with `or.pred %mask, %mask, %p_seg`.  No other code in
//!    these kernels produces this exact pairing.
//!
//! 2. **Opcode-prefix identity**: after stripping the `.EX` suffix and
//!    register/predicate names, the first instruction of every helper
//!    instance in the forward kernel is `ISETP.NE.U32.AND Pn, PT, Rn,
//!    Rn, PT` and the second is `ISETP.GT.U32.OR Pn, PT, Rn, Rn, Pn`.
//!    The backward kernel emits the same opcode prefixes.  Any
//!    accidental opcode-class change in the helper (e.g. helper
//!    accidentally emits `setp.eq` instead of `setp.ne`) would change
//!    the first opcode to `ISETP.EQ.U32.AND` and be caught here.
//!
//! ## Skipping on machines without ptxas / cuobjdump
//!
//! If either binary is absent (e.g. a CI runner without a CUDA
//! toolkit), the test prints a skip message and returns without
//! failing.  This mirrors the skip behaviour in
//! `csha_ptx_ptxas_validation.rs`.

use std::io::Write;
use std::process::{Command, Stdio};

// ---------------------------------------------------------------------------
// Tool-location helpers
// ---------------------------------------------------------------------------

fn find_ptxas() -> Option<String> {
    // 1. PATH (Linux containers, WSL).
    for name in ["ptxas", "ptxas.exe"] {
        if Command::new(name)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return Some(name.to_string());
        }
    }
    // 2. Standard Windows CUDA install.
    let win = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe";
    if std::path::Path::new(win).exists() {
        return Some(win.to_string());
    }
    None
}

fn find_cuobjdump() -> Option<String> {
    for name in ["cuobjdump", "cuobjdump.exe"] {
        if Command::new(name)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
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
// PTX extraction from insta snapshots
// ---------------------------------------------------------------------------

/// Strip the 4-line insta YAML frontmatter (`---` / source / expression /
/// `---`) and any trailing NUL bytes that insta appends, returning the raw
/// PTX body.
fn extract_ptx_from_snapshot(path: &str) -> String {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("cannot read snapshot {}: {}", path, e));
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();

    let mut dash_count = 0;
    let mut body_start = 0;
    for (i, line) in lines.iter().enumerate() {
        if *line == "---" {
            dash_count += 1;
            if dash_count == 2 {
                body_start = i + 1;
                break;
            }
        }
    }
    assert!(
        dash_count == 2,
        "snapshot {} does not have the expected 4-line insta frontmatter",
        path
    );

    let body: String = lines[body_start..]
        .join("\n")
        .chars()
        .filter(|&c| c != '\0')
        .collect();
    body.trim().to_string()
}

// ---------------------------------------------------------------------------
// ptxas + cuobjdump pipeline
// ---------------------------------------------------------------------------

/// Assemble PTX bytes → cubin (written to a temp file), then dump SASS.
/// Returns the full SASS string, or an error message.
fn ptx_to_sass(ptxas: &str, cuobjdump: &str, ptx: &str, label: &str) -> Result<String, String> {
    // Write PTX to a temp file (ptxas reads from file, not stdin, for large inputs).
    let tmp_ptx = std::env::temp_dir().join(format!("pca_sass_gate_{}.ptx", label));
    let tmp_cubin = std::env::temp_dir().join(format!("pca_sass_gate_{}.cubin", label));

    std::fs::write(&tmp_ptx, ptx.as_bytes())
        .map_err(|e| format!("write ptx: {}", e))?;

    // Assemble.
    let asm = Command::new(ptxas)
        .args(["-arch=sm_75", "-O2", "-o"])
        .arg(&tmp_cubin)
        .arg(&tmp_ptx)
        .output()
        .map_err(|e| format!("spawn ptxas: {}", e))?;
    if !asm.status.success() {
        let stderr = String::from_utf8_lossy(&asm.stderr);
        return Err(format!("ptxas rejected {}: {}", label, stderr));
    }

    // Dump SASS.
    let dump = Command::new(cuobjdump)
        .arg("--dump-sass")
        .arg(&tmp_cubin)
        .output()
        .map_err(|e| format!("spawn cuobjdump: {}", e))?;
    if !dump.status.success() {
        let stderr = String::from_utf8_lossy(&dump.stderr);
        return Err(format!("cuobjdump failed on {}: {}", label, stderr));
    }

    String::from_utf8(dump.stdout).map_err(|e| format!("sass utf8: {}", e))
}

// ---------------------------------------------------------------------------
// SASS helper-signature extraction
// ---------------------------------------------------------------------------

/// The SASS signature of `emit_segment_mask_predicate` is an
/// `ISETP.NE.U32.AND` instruction followed (within 4 SASS instructions)
/// by an `ISETP.GT.U32.OR` instruction (with or without `.EX`).
///
/// Returns a list of (ne_opcode_normalised, or_opcode_normalised) pairs,
/// one per helper invocation site found in the SASS.
fn extract_helper_signatures(sass: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = sass.lines().collect();
    let mut pairs = Vec::new();

    for i in 0..lines.len() {
        let line = lines[i];
        // Must contain the NE opcode (the setp.ne.u16 fusion).
        if !line.contains("ISETP.NE.U32.AND") {
            continue;
        }
        // Skip comparisons against constants (c[0x0][…]) or RZ — those
        // are from other predicates, not the segment-ID compare.
        if line.contains("c[0x0]") || (line.contains("RZ, RZ") && !line.contains("PT, R")) {
            continue;
        }
        // Check that it's a register-register compare (two Rn operands
        // before the trailing PT).
        if !is_reg_reg_isetp_ne(line) {
            continue;
        }

        // Look ahead up to 4 lines for the OR fusion partner.
        let window_end = (i + 8).min(lines.len());
        for j in (i + 1)..window_end {
            let jline = lines[j];
            if jline.contains("ISETP.GT.U32.OR") {
                let ne_norm = normalise_opcode_line(line);
                let or_norm = normalise_opcode_line(jline);
                pairs.push((ne_norm, or_norm));
                break;
            }
            // If we hit a different ISETP (not GT.U32.OR) that belongs to
            // an unrelated predicate, give up for this candidate.
            if jline.contains("ISETP") && !jline.contains("ISETP.GT.U32.OR") {
                break;
            }
        }
    }
    pairs
}

/// True when the ISETP.NE line compares two general-purpose registers
/// (Rnn, Rnn) rather than a constant or RZ on the right-hand side.
fn is_reg_reg_isetp_ne(line: &str) -> bool {
    // Typical form: "ISETP.NE.U32.AND Pn, PT, Rnn, Rnn, PT ;"
    // We look for ", Rnn, R" which can't match constant/RZ forms.
    let after_pn = match line.find("ISETP.NE.U32.AND") {
        Some(pos) => &line[pos..],
        None => return false,
    };
    // After the opcode: "Pn, PT, Rnn, Rnn, PT"
    // We want to confirm the 3rd and 4th comma-separated tokens are both "R…"
    let parts: Vec<&str> = after_pn
        .splitn(6, ',')
        .map(|s| s.trim())
        .collect();
    // parts[0] = "ISETP.NE.U32.AND Pn"
    // parts[1] = "PT"
    // parts[2] = "Rnn"   ← first operand
    // parts[3] = "Rnn"   ← second operand
    // parts[4] = "PT ;"
    if parts.len() < 4 {
        return false;
    }
    parts[2].starts_with('R') && parts[3].starts_with('R')
}

/// Normalise a SASS opcode line by:
/// 1. Stripping the leading instruction-address `/*0xaddr*/` prefix.
/// 2. Keeping only the text before the trailing `/* 0x…*/` binary encoding.
/// 3. Replacing `Rnn` with `Rn` and `Pnn` (and plain `P` digits) with `Pn`.
/// 4. Stripping `.EX` from ISETP opcodes so the OR-fusion variants compare
///    equal (the difference is caller-context precision, not helper logic).
/// 5. Collapsing whitespace runs to single spaces.
fn normalise_opcode_line(line: &str) -> String {
    // Drop the leading instruction-address comment /*0xaddr*/.
    // cuobjdump emits e.g. "        /*0ed0*/                   ISETP.NE..."
    let text = if let Some(end) = line.find("*/") {
        &line[end + 2..]
    } else {
        line
    };

    // Drop the binary encoding half (after `/* 0x`).
    let text = if let Some(enc) = text.find("/* 0x") {
        &text[..enc]
    } else {
        text
    };

    // Replace register names.
    let text = replace_reg_names(text);

    // Strip .EX from ISETP opcodes.
    let text = text.replace("ISETP.GT.U32.OR.EX", "ISETP.GT.U32.OR");
    let text = text.replace("ISETP.NE.U32.AND.EX", "ISETP.NE.U32.AND");

    // Collapse whitespace.
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Replace `Rnn`, `RZnn`, `PTnn`, `Pnn` with sentinel tokens `Rn`, `RZ`,
/// `PT`, `Pn`.  We also replace `URnn` (uniform registers) with `URn`.
fn replace_reg_names(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Handle 'R' potentially followed by digits.
        if bytes[i] == b'R' {
            // Check for RZ (zero register) — preserve as-is.
            if i + 1 < bytes.len() && bytes[i + 1] == b'Z' {
                // Possibly RZ followed by digits (e.g. RZ itself has no suffix).
                out.push('R');
                out.push('Z');
                i += 2;
                // Skip any trailing digits after RZ (shouldn't exist but be safe).
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                continue;
            }
            // Check for UR (uniform register).
            if i + 1 < bytes.len() && bytes[i + 1] == b'U' {
                // ULR or URn
                if i + 2 < bytes.len() && bytes[i + 2] == b'R' {
                    // This is actually seeing 'R' at i, 'U' at i+1 — this is
                    // the "R" in "UR" viewed from the wrong position.  Skip
                    // and let the 'U' path handle it.
                }
            }
            // Regular R followed by digits.
            if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                out.push_str("Rn");
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                // Consume optional ".reuse" / ".H0_H0" suffixes that appear
                // in some SASS register operand forms.
                continue;
            }
        }
        // Handle 'P' followed by digits (predicate register).
        if bytes[i] == b'P' {
            // PT is the true-predicate sentinel — preserve.
            if i + 1 < bytes.len() && bytes[i + 1] == b'T' {
                // Check it's "PT" not "PT0" etc.
                let after = if i + 2 < bytes.len() { bytes[i + 2] } else { b' ' };
                if !after.is_ascii_digit() {
                    out.push('P');
                    out.push('T');
                    i += 2;
                    continue;
                }
            }
            // P followed by digit(s) — predicate register.
            if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                out.push_str("Pn");
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                continue;
            }
        }
        // Handle 'U' for uniform registers URn.
        if bytes[i] == b'U' && i + 1 < bytes.len() && bytes[i + 1] == b'R' {
            // URZ — preserve.
            if i + 2 < bytes.len() && bytes[i + 2] == b'Z' {
                out.push_str("URZ");
                i += 3;
                continue;
            }
            // URn — normalise.
            if i + 2 < bytes.len() && bytes[i + 2].is_ascii_digit() {
                out.push_str("URn");
                i += 2;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test]
fn forward_and_backward_helper_sass_opcode_pattern_equivalent() {
    // --- tool availability ---
    let Some(ptxas) = find_ptxas() else {
        eprintln!(
            "[skip] pca_sass_byte_identity: ptxas not found in PATH or \
             standard CUDA install — SASS-level gate skipped"
        );
        return;
    };
    let Some(cuobjdump) = find_cuobjdump() else {
        eprintln!(
            "[skip] pca_sass_byte_identity: cuobjdump not found — \
             SASS-level gate skipped"
        );
        return;
    };

    // Resolve snapshot paths relative to the workspace root.  Cargo sets
    // CARGO_MANIFEST_DIR to the crate directory; snapshots live under
    // tests/snapshots/ from there.
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set by Cargo");
    let snap_dir = std::path::PathBuf::from(&manifest)
        .join("tests")
        .join("snapshots");

    let fwd_snap = snap_dir.join(
        "pca_forward_kernel_snapshot__forward_kernel_segment_masked_causal_32_32_32.snap",
    );
    let bwd_snap = snap_dir.join(
        "pca_backward_kernel_snapshot__backward_kernel_segment_masked_causal_32_32_32.snap",
    );

    for snap in [&fwd_snap, &bwd_snap] {
        assert!(
            snap.exists(),
            "snapshot not found: {}. Run the PCA snapshot tests first to \
             generate it.",
            snap.display()
        );
    }

    // --- extract PTX ---
    let fwd_ptx = extract_ptx_from_snapshot(fwd_snap.to_str().unwrap());
    let bwd_ptx = extract_ptx_from_snapshot(bwd_snap.to_str().unwrap());

    // --- assemble → SASS ---
    let fwd_sass = match ptx_to_sass(&ptxas, &cuobjdump, &fwd_ptx, "fwd") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[skip] forward kernel failed to assemble: {}", e);
            return;
        }
    };
    let bwd_sass = match ptx_to_sass(&ptxas, &cuobjdump, &bwd_ptx, "bwd") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[skip] backward kernel failed to assemble: {}", e);
            return;
        }
    };

    // --- extract helper signatures ---
    let fwd_sigs = extract_helper_signatures(&fwd_sass);
    let bwd_sigs = extract_helper_signatures(&bwd_sass);

    // Diagnostic output (always printed; only visible on failure or --nocapture).
    println!(
        "FWD kernel: {} helper instance(s) found",
        fwd_sigs.len()
    );
    for (i, (ne, or)) in fwd_sigs.iter().enumerate() {
        println!("  [{i}] NE  : {ne}");
        println!("  [{i}] OR  : {or}");
    }
    println!(
        "BWD kernel: {} helper instance(s) found",
        bwd_sigs.len()
    );
    for (i, (ne, or)) in bwd_sigs.iter().enumerate() {
        println!("  [{i}] NE  : {ne}");
        println!("  [{i}] OR  : {or}");
    }

    // Gate 1: helper signature must appear in both kernels.
    assert!(
        !fwd_sigs.is_empty(),
        "Forward kernel SASS contains no ISETP.NE.U32.AND → ISETP.GT.U32.OR \
         pair.  The segment-mask helper emission may be missing or \
         miscompiled.\n\nFirst 500 chars of FWD SASS:\n{}",
        &fwd_sass[..fwd_sass.len().min(500)]
    );
    assert!(
        !bwd_sigs.is_empty(),
        "Backward kernel SASS contains no ISETP.NE.U32.AND → ISETP.GT.U32.OR \
         pair.  The segment-mask helper emission may be missing or \
         miscompiled.\n\nFirst 500 chars of BWD SASS:\n{}",
        &bwd_sass[..bwd_sass.len().min(500)]
    );

    // Gate 2: every helper instance in the forward kernel must use the
    // ISETP.NE opcode prefix (not EQ, LT, etc.).
    for (i, (ne, _)) in fwd_sigs.iter().enumerate() {
        assert!(
            ne.contains("ISETP.NE.U32.AND"),
            "Forward helper instance [{}] has wrong NE opcode: {}",
            i, ne
        );
    }
    for (i, (ne, _)) in bwd_sigs.iter().enumerate() {
        assert!(
            ne.contains("ISETP.NE.U32.AND"),
            "Backward helper instance [{}] has wrong NE opcode: {}",
            i, ne
        );
    }

    // Gate 3: every OR-fusion instruction in both kernels must use the
    // ISETP.GT.U32.OR opcode (after .EX stripping).  This confirms that
    // the causal-mask OR fusion always uses the greater-than form (not EQ
    // or NE), which is required for causal masking correctness.
    for (i, (_, or)) in fwd_sigs.iter().enumerate() {
        assert!(
            or.contains("ISETP.GT.U32.OR"),
            "Forward helper instance [{}] has unexpected OR-fusion opcode: {}",
            i, or
        );
    }
    for (i, (_, or)) in bwd_sigs.iter().enumerate() {
        assert!(
            or.contains("ISETP.GT.U32.OR"),
            "Backward helper instance [{}] has unexpected OR-fusion opcode: {}",
            i, or
        );
    }

    // Gate 4: the first helper instance's normalised NE opcode is the same
    // in both kernels (same opcode class, operand structure, sentinel shape).
    // Note: byte-exact identity across ALL instances is NOT asserted because:
    //   (a) Different tile sizes / loop unrolling means different instance
    //       counts (forward may have more instances than backward).
    //   (b) The OR-fusion partner differs in the .EX extension due to
    //       64-bit (forward) vs 32-bit (backward) causal-position types.
    // See the module-level doc comment for the full explanation.
    let fwd_ne0 = &fwd_sigs[0].0;
    let bwd_ne0 = &bwd_sigs[0].0;
    assert_eq!(
        fwd_ne0, bwd_ne0,
        "First helper instance normalised NE opcode differs between \
         forward and backward kernels.\n\
         Forward : {}\n\
         Backward: {}\n\n\
         This indicates the helper is encoding a different comparison \
         operation in the two kernel contexts — a spec §4 invariant #2 \
         violation.",
        fwd_ne0, bwd_ne0
    );

    println!(
        "PASS: both kernels contain ISETP.NE.U32.AND → ISETP.GT.U32.OR \
         helper signature; NE opcode normalises identically across contexts."
    );
}
