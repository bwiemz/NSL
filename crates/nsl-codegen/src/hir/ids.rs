//! HIR ID types. Each fresh-allocated via `Id::fresh()` returning a unique value.

use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! define_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u64);

        impl $name {
            pub fn fresh() -> Self {
                static COUNTER: AtomicU64 = AtomicU64::new(1);
                Self(COUNTER.fetch_add(1, Ordering::Relaxed))
            }
        }
    };
}

define_id!(WireId);
define_id!(RegisterId);
define_id!(GenvarId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClockDomainId(pub u64);
impl ClockDomainId {
    pub const DEFAULT: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResetSignalId(pub u64);
impl ResetSignalId {
    pub const DEFAULT: Self = Self(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_ids_are_unique() {
        let a = WireId::fresh();
        let b = WireId::fresh();
        assert_ne!(a, b);
    }

    #[test]
    fn fresh_ids_are_strictly_increasing() {
        let a = WireId::fresh();
        let b = WireId::fresh();
        assert!(b.0 > a.0);
    }

    #[test]
    fn default_clock_domain_is_zero() {
        assert_eq!(ClockDomainId::DEFAULT.0, 0);
    }

    #[test]
    fn default_reset_signal_is_zero() {
        assert_eq!(ResetSignalId::DEFAULT.0, 0);
    }
}
