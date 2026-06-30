//! Static PTX/kernel metadata extraction.
//!
//! Architecture-hardening item: make GPU kernel resource usage *traceable*
//! without a GPU. The synthesized PTX text already carries everything we need
//! as plain directives, so we parse it instead of launching anything:
//!
//! - `.target sm_NN`          → target compute capability (module scope)
//! - `.reg .TYPE %p<N>;`      → per-class register *high-water mark* (the `<N>`
//!                              is the count ptxas will allocate for that class)
//! - `.shared ... name[N];`   → static shared-memory bytes
//! - `.visible .entry NAME(`  → kernel boundary / name
//!
//! This is an *estimate*, not ptxas ground truth: it counts declared registers
//! (the upper bound the emitter reserves) and statically-sized shared memory.
//! Dynamically-sized shared memory (`extern .shared`, sized at launch) and the
//! register *coalescing* ptxas performs are out of scope and noted as such.
//! For CSHA/FlashAttention kernels the emitter-side
//! [`crate::flash_attention_v2::register_budget`] /
//! [`crate::flash_attention_v2::smem_layout`] helpers carry exact figures; this
//! module is the general, backend-agnostic view derived purely from PTX text.
//!
//! Because it is pure text in / structs out, it runs on every `cargo test`
//! (no `cuda` feature, no device) and underpins the `--ptx-metadata` report.

/// Per-register-class declared counts for one kernel.
///
/// Mirrors the register classes the PTX backend emits (`backend_ptx.rs`):
/// `%r` (u32), `%rd` (u64), `%f` (f32), `%fd` (f64), `%h` (f16), `%p` (pred).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterCounts {
    /// 32-bit integer registers (`%r`).
    pub u32_regs: u32,
    /// 64-bit integer registers (`%rd`).
    pub u64_regs: u32,
    /// 32-bit float registers (`%f`).
    pub f32_regs: u32,
    /// 64-bit float registers (`%fd`).
    pub f64_regs: u32,
    /// 16-bit float registers (`%h`).
    pub f16_regs: u32,
    /// Predicate registers (`%p`).
    pub pred_regs: u32,
}

impl RegisterCounts {
    /// Total declared registers across all classes. This is the figure to
    /// compare against an architecture's per-thread register cap (255 on
    /// most NVIDIA SMs) for an occupancy sanity check.
    pub fn total(&self) -> u32 {
        self.u32_regs
            + self.u64_regs
            + self.f32_regs
            + self.f64_regs
            + self.f16_regs
            + self.pred_regs
    }
}

/// Static metadata for a single `.entry` kernel parsed from PTX text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelMetadata {
    /// Kernel entry name (the identifier after `.entry`).
    pub name: String,
    /// Target compute capability (e.g. `70` for `sm_70`). The most recent
    /// module-scope `.target` directive in effect at the kernel.
    pub target_sm: u32,
    /// Declared register counts per class.
    pub registers: RegisterCounts,
    /// Statically-declared shared-memory bytes attributed to this kernel
    /// (module-scope `.shared` declarations plus any inside the kernel body).
    /// Excludes dynamic `extern .shared` (sized at launch, not in the text).
    pub shared_mem_bytes: u32,
    /// Whether the kernel body declared/extern'd any dynamically-sized shared
    /// memory (`extern .shared`). When true, [`Self::shared_mem_bytes`] is a
    /// lower bound — the real figure depends on the launch config.
    pub has_dynamic_shared: bool,
}

/// Parse `.entry`-level metadata out of synthesized PTX text.
///
/// Robust to PTX that omits the `.visible`/`.weak` linkage prefix, and to
/// `.shared` declarations placed at either module or kernel scope. Never
/// panics on malformed input — unparseable directives are skipped.
pub fn extract_ptx_metadata(ptx: &[u8]) -> Vec<KernelMetadata> {
    let text = String::from_utf8_lossy(ptx);

    let mut kernels: Vec<KernelMetadata> = Vec::new();
    // Module-scope state carried into every kernel.
    let mut target_sm: u32 = 0;
    let mut module_shared: u32 = 0;

    // Current-kernel accumulators.
    let mut in_kernel = false;
    let mut depth: i32 = 0; // brace depth inside the current entry
    let mut seen_open = false; // has the body `{` opened yet?
    let mut name = String::new();
    let mut registers = RegisterCounts::default();
    let mut shared_mem_bytes: u32 = 0;
    let mut has_dynamic_shared = false;

    for raw in text.lines() {
        let line = raw.trim();

        // Module-scope target directive (applies to subsequent kernels).
        if !in_kernel {
            if let Some(sm) = parse_target_sm(line) {
                target_sm = sm;
                continue;
            }
        }

        // Shared-memory declaration (may be module- or kernel-scope).
        if line.starts_with(".shared") || line.starts_with(".extern .shared") {
            // `dynamic` is only load-bearing when `in_kernel` (used at the
            // `has_dynamic_shared` write below). For module scope a missing
            // `[...]` just means `parse_shared_bytes` returns `None`, so no
            // bytes are added — the flag is harmless there.
            let dynamic = line.contains("extern") || !line.contains('[');
            if let Some(bytes) = parse_shared_bytes(line) {
                if in_kernel {
                    shared_mem_bytes = shared_mem_bytes.saturating_add(bytes);
                } else {
                    module_shared = module_shared.saturating_add(bytes);
                }
            }
            if dynamic && in_kernel {
                has_dynamic_shared = true;
            }
            // Production PTX never puts a brace on a `.shared` line, but keep
            // depth consistent with the `.reg` branch so pathological input
            // can't desync the kernel boundary.
            depth += count_braces(line);
            continue;
        }

        // Kernel entry start: `.visible .entry NAME(` / `.entry NAME(` /
        // `.weak .entry NAME(`. The `(` may be on the same or next line, so we
        // key off `.entry ` and read the identifier token.
        if !in_kernel {
            if let Some(entry_name) = parse_entry_name(line) {
                in_kernel = true;
                depth = count_braces(line);
                seen_open = depth > 0;
                name = entry_name;
                registers = RegisterCounts::default();
                shared_mem_bytes = module_shared;
                has_dynamic_shared = false;
                continue;
            }
            continue;
        }

        // --- Inside a kernel body ---

        // Register declaration high-water marks.
        if line.starts_with(".reg ") {
            accumulate_reg(line, &mut registers);
            depth += count_braces(line);
            if depth > 0 {
                seen_open = true;
            }
            continue;
        }

        depth += count_braces(line);
        if depth > 0 {
            seen_open = true;
        }
        // Only close once the body brace has actually opened — an `.entry`
        // signature can span several lines (params before `{`), during which
        // depth is legitimately 0.
        if seen_open && depth <= 0 {
            kernels.push(KernelMetadata {
                name: std::mem::take(&mut name),
                target_sm,
                registers: std::mem::take(&mut registers),
                shared_mem_bytes,
                has_dynamic_shared,
            });
            in_kernel = false;
            seen_open = false;
            shared_mem_bytes = 0;
            has_dynamic_shared = false;
            depth = 0;
        }
    }

    // Tolerate a kernel whose closing brace we never saw (truncated PTX).
    if in_kernel && !name.is_empty() {
        kernels.push(KernelMetadata {
            name,
            target_sm,
            registers,
            shared_mem_bytes,
            has_dynamic_shared,
        });
    }

    kernels
}

/// Render a human-readable metadata report. One block per kernel, with a
/// register cap warning when a kernel's total exceeds the per-thread limit for
/// its target SM (255 on all current NVIDIA architectures).
pub fn format_ptx_metadata_report(kernels: &[KernelMetadata]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "PTX kernel metadata ({} kernel(s)):", kernels.len());
    for k in kernels {
        let _ = writeln!(
            s,
            "  {} [target sm_{}]",
            if k.name.is_empty() { "<anonymous>" } else { &k.name },
            k.target_sm
        );
        let r = &k.registers;
        let _ = writeln!(
            s,
            "    registers: {} total (u32={}, u64={}, f32={}, f64={}, f16={}, pred={})",
            r.total(),
            r.u32_regs,
            r.u64_regs,
            r.f32_regs,
            r.f64_regs,
            r.f16_regs,
            r.pred_regs,
        );
        let dyn_note = if k.has_dynamic_shared {
            " (+ dynamic extern .shared sized at launch)"
        } else {
            ""
        };
        let _ = writeln!(s, "    shared memory: {} bytes static{}", k.shared_mem_bytes, dyn_note);
        if r.total() > PER_THREAD_REGISTER_CAP {
            let _ = writeln!(
                s,
                "    WARNING: {} declared registers exceeds the {}-register per-thread cap; \
                 ptxas will spill to local memory",
                r.total(),
                PER_THREAD_REGISTER_CAP,
            );
        }
    }
    s
}

/// Per-thread register file cap on all current NVIDIA SMs (sm_50..sm_120).
const PER_THREAD_REGISTER_CAP: u32 = 255;

/// Parse `.target sm_NN` → `NN`. Returns `None` for non-`sm_` targets.
fn parse_target_sm(line: &str) -> Option<u32> {
    let rest = line.strip_prefix(".target ")?;
    // `.target sm_80` or `.target sm_80, debug` — take the first token.
    let first = rest.split([',', ' ']).next()?.trim();
    let digits = first.strip_prefix("sm_")?;
    digits.parse().ok()
}

/// Parse the byte count out of a `.shared` declaration like
/// `.shared .align 4 .b8 shared_mem[4096];` → `4096`.
fn parse_shared_bytes(line: &str) -> Option<u32> {
    // First `[` is the array dimension: PTX identifiers cannot contain `[`, so
    // the only `[` on a well-formed `.shared` line opens `[bytes]`. Using
    // `find` (not `rfind`) means a trailing comment containing brackets can't
    // hijack the count.
    let open = line.find('[')?;
    let close = line[open + 1..].find(']')? + open + 1;
    let inner = line[open + 1..close].trim();
    // Element count × element width. `.bN` element width is N/8 bytes; the
    // backend emits `.b8 name[bytes]`, so width is usually 1, but be general.
    let width = shared_element_width_bytes(line);
    inner.parse::<u32>().ok().map(|n| n.saturating_mul(width))
}

/// Element width in bytes for a `.shared .bN` / `.uN` / `.fN` declaration.
/// Defaults to 1 (`.b8`, the backend's convention) when no width token found.
///
/// Scans the whole line for the width token, which assumes `.shared` lines
/// carry no trailing comment and no identifier containing a `.bNN` substring —
/// both hold for the production emitter. (The backend emits `.b8 name[bytes]`,
/// so width is 1 in practice and `bytes` is already the byte count.)
fn shared_element_width_bytes(line: &str) -> u32 {
    for tok in [".b64", ".u64", ".s64", ".f64"] {
        if line.contains(tok) {
            return 8;
        }
    }
    for tok in [".b32", ".u32", ".s32", ".f32"] {
        if line.contains(tok) {
            return 4;
        }
    }
    for tok in [".b16", ".u16", ".s16", ".f16"] {
        if line.contains(tok) {
            return 2;
        }
    }
    1
}

/// Parse the kernel name from an `.entry` line, allowing optional linkage
/// (`.visible` / `.weak`) prefixes and a same-line `(`.
fn parse_entry_name(line: &str) -> Option<String> {
    let idx = line.find(".entry ")?;
    let after = line[idx + ".entry ".len()..].trim_start();
    // Name runs until `(`, whitespace, or end of line.
    let end = after
        .find(|c: char| c == '(' || c.is_whitespace())
        .unwrap_or(after.len());
    let name = after[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Add a `.reg` declaration's `<N>` count to the right class in `counts`.
fn accumulate_reg(line: &str, counts: &mut RegisterCounts) {
    let Some(n) = parse_angle_count(line) else {
        return;
    };
    // Order matters: check the longer prefixes (%rd, %fd) before %r / %f.
    let target = if line.contains("%rd") {
        &mut counts.u64_regs
    } else if line.contains("%fd") {
        &mut counts.f64_regs
    } else if line.contains("%f") {
        &mut counts.f32_regs
    } else if line.contains("%r") {
        &mut counts.u32_regs
    } else if line.contains("%h") {
        &mut counts.f16_regs
    } else if line.contains("%p") {
        &mut counts.pred_regs
    } else {
        return;
    };
    *target = (*target).max(n);
}

/// Parse the `<N>` register count from a `.reg` line.
fn parse_angle_count(line: &str) -> Option<u32> {
    let open = line.find('<')?;
    let close = line[open + 1..].find('>')? + open + 1;
    line[open + 1..close].trim().parse().ok()
}

/// Net brace delta for a line (`{` = +1, `}` = -1), ignoring braces in
/// comments is *not* attempted — PTX bodies don't put stray braces in `//`
/// comments in our emitters, and the entry/exit braces are always on their
/// own structural lines.
fn count_braces(line: &str) -> i32 {
    let mut d = 0;
    for c in line.chars() {
        match c {
            '{' => d += 1,
            '}' => d -= 1,
            _ => {}
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
//
// Generated by NSL
//
.version 7.0
.target sm_80
.address_size 64

.visible .entry add_kernel(
    .param .u64 a,
    .param .u64 b
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<32>;
    .shared .align 4 .b8 shared_mem[2048];
    ld.param.u64 %rd1, [a];
    ret;
}
"#;

    #[test]
    fn parses_single_kernel() {
        let k = extract_ptx_metadata(SAMPLE.as_bytes());
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].name, "add_kernel");
        assert_eq!(k[0].target_sm, 80);
        assert_eq!(k[0].registers.u32_regs, 16);
        assert_eq!(k[0].registers.u64_regs, 8);
        assert_eq!(k[0].registers.f32_regs, 32);
        assert_eq!(k[0].registers.pred_regs, 4);
        assert_eq!(k[0].registers.f64_regs, 0);
        assert_eq!(k[0].registers.total(), 16 + 8 + 32 + 4);
        assert_eq!(k[0].shared_mem_bytes, 2048);
        assert!(!k[0].has_dynamic_shared);
    }

    #[test]
    fn parses_multiple_kernels_and_carries_target() {
        let ptx = r#"
.target sm_90
.visible .entry fwd()
{
    .reg .b32 %r<100>;
    ret;
}
.entry bwd()
{
    .reg .b32 %r<200>;
    .reg .b64 %rd<50>;
    ret;
}
"#;
        let k = extract_ptx_metadata(ptx.as_bytes());
        assert_eq!(k.len(), 2);
        assert_eq!(k[0].name, "fwd");
        assert_eq!(k[0].target_sm, 90);
        assert_eq!(k[0].registers.u32_regs, 100);
        assert_eq!(k[1].name, "bwd"); // no `.visible` prefix still parses
        assert_eq!(k[1].target_sm, 90); // module target carried forward
        assert_eq!(k[1].registers.u64_regs, 50);
    }

    #[test]
    fn module_scope_shared_attributed_to_kernel() {
        let ptx = r#"
.target sm_75
.shared .align 4 .b8 g_smem[1024];
.visible .entry k()
{
    .reg .b32 %r<4>;
    ret;
}
"#;
        let k = extract_ptx_metadata(ptx.as_bytes());
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].shared_mem_bytes, 1024);
    }

    #[test]
    fn detects_dynamic_shared() {
        let ptx = r#"
.target sm_80
.visible .entry k()
{
    .reg .b32 %r<4>;
    .extern .shared .align 16 .b8 dyn_smem[];
    ret;
}
"#;
        let k = extract_ptx_metadata(ptx.as_bytes());
        assert_eq!(k.len(), 1);
        assert!(k[0].has_dynamic_shared);
    }

    #[test]
    fn no_panic_on_garbage() {
        let _ = extract_ptx_metadata(b"not ptx at all { } } { .entry");
        let _ = extract_ptx_metadata(&[]);
        let _ = extract_ptx_metadata(&[0xff, 0xfe, 0x00, 0x01]);
    }

    #[test]
    fn report_flags_register_overflow() {
        let over = vec![KernelMetadata {
            name: "huge".to_string(),
            target_sm: 80,
            registers: RegisterCounts {
                u32_regs: 300,
                ..Default::default()
            },
            shared_mem_bytes: 0,
            has_dynamic_shared: false,
        }];
        let report = format_ptx_metadata_report(&over);
        assert!(report.contains("WARNING"));
        assert!(report.contains("huge"));
        assert!(report.contains("sm_80"));
    }

    #[test]
    fn report_is_clean_for_modest_kernel() {
        let ok = vec![KernelMetadata {
            name: "ok".to_string(),
            target_sm: 70,
            registers: RegisterCounts {
                u32_regs: 32,
                f32_regs: 16,
                ..Default::default()
            },
            shared_mem_bytes: 512,
            has_dynamic_shared: false,
        }];
        let report = format_ptx_metadata_report(&ok);
        assert!(!report.contains("WARNING"));
        assert!(report.contains("48 total"));
        assert!(report.contains("512 bytes"));
    }
}
