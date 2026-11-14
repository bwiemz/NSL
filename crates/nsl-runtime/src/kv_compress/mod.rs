// crates/nsl-runtime/src/kv_compress/mod.rs
//! M42: KV-cache compression and eviction routines.
//!
//! Library code with FFI entry points and unit tests. The paged KV cache
//! does not call any of it: K and V are stored as written, and the
//! `@kv_compress` decorator is refused by the checker.
//!
//! - Quantizers for one K or V block (INT8 per head or per token, INT4 per
//!   group, FP8) and their inverses
//! - Sliding window with attention sinks — which blocks fall out of a window
//! - H2O (Heavy Hitter Oracle) — attention-score-based block eviction

pub mod quantize;
pub mod sliding_window;
pub mod h2o;
pub mod ffi;

/// Quantization scheme for KV-cache storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[derive(Default)]
pub enum KvQuantScheme {
    /// No compression — native dtype (FP16/FP32).
    #[default]
    None = 0,
    /// INT8 symmetric quantization, one scale per attention head.
    Int8PerHead = 1,
    /// INT8 symmetric quantization, one scale per token position.
    Int8PerToken = 2,
    /// INT4 asymmetric quantization, one scale+zero per group.
    Int4PerGroup = 3,
    /// FP8 E4M3 — direct cast, no scaling metadata.
    Fp8 = 4,
}

impl KvQuantScheme {
    /// Payload bytes per element for this scheme. The per-block metadata
    /// (`KvBlockQuantMeta`, 1,032 bytes for the INT8 and INT4 schemes) is
    /// not included; `quantize::block_total_bytes` counts it.
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            KvQuantScheme::None => 2.0,      // FP16
            KvQuantScheme::Int8PerHead => 1.0,
            KvQuantScheme::Int8PerToken => 1.0,
            KvQuantScheme::Int4PerGroup => 0.5,
            KvQuantScheme::Fp8 => 1.0,
        }
    }

    /// From integer discriminant (for FFI).
    pub fn from_i64(v: i64) -> Self {
        match v {
            1 => KvQuantScheme::Int8PerHead,
            2 => KvQuantScheme::Int8PerToken,
            3 => KvQuantScheme::Int4PerGroup,
            4 => KvQuantScheme::Fp8,
            _ => KvQuantScheme::None,
        }
    }
}


/// Sliding window configuration.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    pub window_size: usize,
    pub num_sinks: usize,
}

/// H2O (Heavy Hitter Oracle) configuration.
#[derive(Debug, Clone)]
pub struct H2OConfig {
    pub budget: usize,
    pub num_sinks: usize,
}

/// Per-layer compression policy, resolved from @kv_compress decorators.
#[derive(Debug, Clone, Default)]
pub struct LayerCompressionPolicy {
    pub quant_scheme: KvQuantScheme,
    pub window: Option<SlidingWindowConfig>,
    pub h2o: Option<H2OConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_scheme_bytes_per_element() {
        assert_eq!(KvQuantScheme::None.bytes_per_element(), 2.0);
        assert_eq!(KvQuantScheme::Int8PerHead.bytes_per_element(), 1.0);
        assert_eq!(KvQuantScheme::Int4PerGroup.bytes_per_element(), 0.5);
        assert_eq!(KvQuantScheme::Fp8.bytes_per_element(), 1.0);
    }

    #[test]
    fn quant_scheme_from_i64() {
        assert_eq!(KvQuantScheme::from_i64(0), KvQuantScheme::None);
        assert_eq!(KvQuantScheme::from_i64(1), KvQuantScheme::Int8PerHead);
        assert_eq!(KvQuantScheme::from_i64(2), KvQuantScheme::Int8PerToken);
        assert_eq!(KvQuantScheme::from_i64(3), KvQuantScheme::Int4PerGroup);
        assert_eq!(KvQuantScheme::from_i64(4), KvQuantScheme::Fp8);
        assert_eq!(KvQuantScheme::from_i64(99), KvQuantScheme::None);
    }

    #[test]
    fn layer_policy_default_is_none() {
        let p = LayerCompressionPolicy::default();
        assert_eq!(p.quant_scheme, KvQuantScheme::None);
        assert!(p.window.is_none());
        assert!(p.h2o.is_none());
    }
}
