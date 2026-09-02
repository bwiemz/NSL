//! Library surface of the `nsl` CLI. Exposes modules that are also used by
//! the `nsl` binary and by integration tests.

pub mod analysis_bridges;
pub mod exec_markers;
pub mod feature_rules;
pub mod health_monitor;
pub mod monitor;
pub mod profile;
pub mod profile_render;
pub mod shape_debug;
pub mod wggo_explain;

// The multi-module frontend: `loader::load_all_modules` (lex, parse, resolve
// imports, analyze in dependency order with real imported types) and the two
// modules it leans on. Exposed so benches/frontend.rs can drive the path
// `nsl build` takes over a real recipe graph; the `nsl` binary keeps its own
// `mod` declarations in main.rs.
pub mod loader;
pub mod mangling;
pub mod resolver;

// `nsl fmt`'s text normalizer, exposed so tests/fmt_roundtrip.rs can run it
// over the whole corpus without spawning the binary.
pub mod formatter;
