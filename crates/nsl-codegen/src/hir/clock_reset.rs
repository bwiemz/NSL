//! Clock and reset dialect types per M57 v1 spec §3.3 / Q6.
//!
//! HIR carries these fields generically (Pattern 1 / IR-009 discipline);
//! v1's public Register::new_v1_default constructor populates them with
//! the codegen default (sync active-high, single clock domain).

use crate::hir::ids::{ClockDomainId, ResetSignalId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClkRef {
    pub domain_id: ClockDomainId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResetRef {
    pub signal_id: ResetSignalId,
    pub polarity: ResetPolarity,
    pub sync: ResetSync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetPolarity { High, Low }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetSync { Sync, Async }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_default_dialect_is_sync_active_high() {
        let r = ResetRef {
            signal_id: ResetSignalId::DEFAULT,
            polarity: ResetPolarity::High,
            sync: ResetSync::Sync,
        };
        assert_eq!(r.polarity, ResetPolarity::High);
        assert_eq!(r.sync, ResetSync::Sync);
    }

    #[test]
    fn all_four_dialect_combinations_distinct() {
        let combos = [
            (ResetPolarity::High, ResetSync::Sync),
            (ResetPolarity::Low,  ResetSync::Sync),
            (ResetPolarity::High, ResetSync::Async),
            (ResetPolarity::Low,  ResetSync::Async),
        ];
        for i in 0..combos.len() {
            for j in i+1..combos.len() {
                assert_ne!(combos[i], combos[j]);
            }
        }
    }
}
