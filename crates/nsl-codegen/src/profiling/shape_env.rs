//! `ShapeEnv` — resolves symbolic tensor dimensions (e.g. `B`, `S`, `D`) to
//! concrete `u64` sizes supplied via CLI flags like `--batch=4 --seq=2048`.
//!
//! The walker consults this map whenever it encounters a `Dim::Symbolic` or
//! `Dim::Named` dimension that doesn't have a concrete size attached.

use std::collections::HashMap;

/// Map from symbolic dimension name to concrete size.
#[derive(Debug, Clone, Default)]
pub struct ShapeEnv {
    dims: HashMap<String, u64>,
}

impl ShapeEnv {
    /// Empty env — every lookup returns `None`.
    pub fn new() -> Self {
        Self { dims: HashMap::new() }
    }

    /// Convenient defaults for a training/inference probe: `B=1`, `S=2048`.
    pub fn with_defaults() -> Self {
        let mut e = Self::new();
        e.set("B", 1);
        e.set("batch", 1);
        e.set("S", 2048);
        e.set("seq", 2048);
        e
    }

    /// Insert or overwrite a binding.
    pub fn set(&mut self, name: &str, value: u64) {
        self.dims.insert(name.to_string(), value);
    }

    /// Look up a concrete size by dim name.
    pub fn resolve(&self, name: &str) -> Option<u64> {
        self.dims.get(name).copied()
    }

    /// Parse a `name=N` flag (as produced by `--batch=4`, `--dim=D=768`).
    /// Returns `Err` with a human-readable message on malformed input.
    pub fn parse_dim_flag(&mut self, spec: &str) -> Result<(), String> {
        let (name, value) = spec
            .split_once('=')
            .ok_or_else(|| format!("expected NAME=VALUE, got `{spec}`"))?;
        let name = name.trim();
        let value = value.trim();
        if name.is_empty() {
            return Err(format!("empty dim name in `{spec}`"));
        }
        let parsed: u64 = value
            .parse()
            .map_err(|e| format!("invalid size `{value}` for dim `{name}`: {e}"))?;
        self.set(name, parsed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_include_batch_and_seq() {
        let env = ShapeEnv::with_defaults();
        assert_eq!(env.resolve("B"), Some(1));
        assert_eq!(env.resolve("S"), Some(2048));
        assert_eq!(env.resolve("Q"), None);
    }

    #[test]
    fn parse_dim_flag_ok() {
        let mut env = ShapeEnv::new();
        env.parse_dim_flag("D=768").unwrap();
        env.parse_dim_flag("batch=4").unwrap();
        assert_eq!(env.resolve("D"), Some(768));
        assert_eq!(env.resolve("batch"), Some(4));
    }

    #[test]
    fn parse_dim_flag_errors() {
        let mut env = ShapeEnv::new();
        assert!(env.parse_dim_flag("malformed").is_err());
        assert!(env.parse_dim_flag("=4").is_err());
        assert!(env.parse_dim_flag("D=notanumber").is_err());
    }
}
