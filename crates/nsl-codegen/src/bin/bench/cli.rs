//! CLI parser for the `nsl-codegen-bench` binary.
//!
//! Pinned by §5.2 of `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`.
//! Surface (v1):
//!
//! ```text
//! nsl-codegen-bench --fixture <name> --tier-b={on,off} \
//!     [--emit-time-only] [--seed <u64>] [--iterations <n>]
//! ```
//!
//! Errors propagate as `Result<_, String>`; the caller (`main`) maps them to
//! exit code 1 per the spec's exit-code table.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierB {
    On,
    Off,
}

#[derive(Debug, Clone)]
pub struct Args {
    pub fixture: String,
    pub tier_b: TierB,
    pub seed: u64,
    pub iterations: u32,
    pub emit_time_only: bool,
    /// When `Some(path)`, the bench binary writes the forward output `O`
    /// tensor to `path` after the timed loop. Used by the M3 parity tests
    /// (B1.5-3) to capture per-fixture Tier-B-on / Tier-B-off outputs and
    /// assert byte-equality at the test driver layer.
    pub dump_output: Option<std::path::PathBuf>,
}

impl Args {
    /// Parses from `std::env::args()`. Top-level `main` entry point.
    pub fn parse() -> Result<Self, String> {
        let argv: Vec<String> = std::env::args().collect();
        let argv_ref: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
        parse_from(&argv_ref)
    }
}

/// Parse a raw argv vector. argv[0] is the program name and is ignored.
pub fn parse_from(argv: &[&str]) -> Result<Args, String> {
    let mut fixture: Option<String> = None;
    let mut tier_b: Option<TierB> = None;
    let mut seed: u64 = 42;
    let mut iterations: u32 = 100;
    let mut emit_time_only = false;
    let mut dump_output: Option<std::path::PathBuf> = None;

    let mut i = 1;
    while i < argv.len() {
        match argv[i] {
            "--fixture" => {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| "missing value for --fixture".to_string())?;
                fixture = Some((*v).to_string());
                i += 2;
            }
            "--tier-b" => {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| "missing value for --tier-b".to_string())?;
                tier_b = Some(match *v {
                    "on" => TierB::On,
                    "off" => TierB::Off,
                    other => return Err(format!("invalid --tier-b value: {other} (expected 'on' or 'off')")),
                });
                i += 2;
            }
            "--seed" => {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| "missing value for --seed".to_string())?;
                seed = v
                    .parse::<u64>()
                    .map_err(|e| format!("invalid --seed value '{v}': {e}"))?;
                i += 2;
            }
            "--iterations" => {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| "missing value for --iterations".to_string())?;
                iterations = v
                    .parse::<u32>()
                    .map_err(|e| format!("invalid --iterations value '{v}': {e}"))?;
                i += 2;
            }
            "--emit-time-only" => {
                emit_time_only = true;
                i += 1;
            }
            "--dump-output" => {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| "missing value for --dump-output".to_string())?;
                dump_output = Some(std::path::PathBuf::from(*v));
                i += 2;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        fixture: fixture.ok_or_else(|| "--fixture is required".to_string())?,
        tier_b: tier_b.ok_or_else(|| "--tier-b is required".to_string())?,
        seed,
        iterations,
        emit_time_only,
        dump_output,
    })
}
