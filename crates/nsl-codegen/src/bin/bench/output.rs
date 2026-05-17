//! Output-line emitter for the `nsl-codegen-bench` binary.
//!
//! Format is pinned by §5.2 of the 2026-05-13 design spec:
//!
//! ```text
//! tier_b_bench_result:fixture=<name>:tier_b=<on|off>:median_us=<float>:n=<int>:skip_ratio=<float>:seed=<u64>
//! ```
//!
//! Single line, key=value pairs separated by `:`, machine-parseable by shell
//! scripts that grep for the `tier_b_bench_result:` prefix.

#[derive(Debug, Clone)]
pub struct ResultLine {
    pub fixture: String,
    pub tier_b_on: bool,
    pub median_us: f64,
    pub n: u32,
    pub skip_ratio: f64,
    pub seed: u64,
}

/// Format a `ResultLine` per the §5.2 contract.
///
/// The output is a single line with no trailing newline; callers append `\n`
/// when writing to stdout.
pub fn emit_result(line: &ResultLine) -> String {
    format!(
        "tier_b_bench_result:fixture={}:tier_b={}:median_us={}:n={}:skip_ratio={}:seed={}",
        line.fixture,
        if line.tier_b_on { "on" } else { "off" },
        line.median_us,
        line.n,
        line.skip_ratio,
        line.seed,
    )
}
