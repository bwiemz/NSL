//! Fusion report: collects optimization events and formats --fusion-report output.

use std::fmt;

/// Strategy used for a fusion event.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionStrategy {
    Epilogue,
    Reduction,
    Elementwise,
}

impl fmt::Display for FusionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Epilogue => write!(f, "epilogue"),
            Self::Reduction => write!(f, "reduction"),
            Self::Elementwise => write!(f, "elementwise"),
        }
    }
}

/// Reason a fusion opportunity was blocked.
#[derive(Debug, Clone, PartialEq)]
pub enum BarrierReason {
    MultiConsumer,
    LayoutChange,
    FlashAttention,
    NoFuseAnnotation,
    DimensionMismatch,
    UnsupportedOp,
}

impl fmt::Display for BarrierReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MultiConsumer => write!(f, "multi-consumer"),
            Self::LayoutChange => write!(f, "layout change"),
            Self::FlashAttention => write!(f, "FlashAttention barrier"),
            Self::NoFuseAnnotation => write!(f, "@no_fuse annotation"),
            Self::DimensionMismatch => write!(f, "dimension mismatch"),
            Self::UnsupportedOp => write!(f, "unsupported op"),
        }
    }
}

/// A successful fusion event.
#[derive(Debug, Clone)]
pub struct FusionEvent {
    pub function_name: String,
    pub strategy: FusionStrategy,
    pub matched_ops: Vec<String>,
    pub eliminated_launches: u32,
    pub estimated_bytes_saved: u64,
    pub location: String,
}

/// A blocked fusion opportunity.
#[derive(Debug, Clone)]
pub struct FusionBarrierEvent {
    pub function_name: String,
    pub reason: BarrierReason,
    pub node_description: String,
    pub location: String,
}

/// Format and print the fusion report to stderr.
pub fn print_fusion_report(events: &[FusionEvent], barriers: &[FusionBarrierEvent]) {
    if events.is_empty() && barriers.is_empty() {
        return;
    }

    eprintln!("\nFusion Report:");

    // Group by function
    let mut functions: Vec<String> = Vec::new();
    for e in events {
        if !functions.contains(&e.function_name) {
            functions.push(e.function_name.clone());
        }
    }
    for b in barriers {
        if !functions.contains(&b.function_name) {
            functions.push(b.function_name.clone());
        }
    }

    for func_name in &functions {
        let func_events: Vec<_> = events.iter().filter(|e| &e.function_name == func_name).collect();
        let func_barriers: Vec<_> = barriers.iter().filter(|b| &b.function_name == func_name).collect();

        let loc = func_events.first().map(|e| e.location.as_str())
            .or_else(|| func_barriers.first().map(|b| b.location.as_str()))
            .unwrap_or("unknown");
        eprintln!("  {} ({}):", func_name, loc);

        for e in &func_events {
            eprintln!(
                "    {} -> FUSED ({})",
                e.matched_ops.join(" + "),
                e.strategy
            );
            eprintln!(
                "      Savings: {} eliminated launch(es), ~{}MB eliminated traffic",
                e.eliminated_launches,
                e.estimated_bytes_saved / (1024 * 1024)
            );
        }

        for b in &func_barriers {
            eprintln!(
                "    {} -> not fused ({} barrier)",
                b.node_description,
                b.reason
            );
        }
    }

    // Summary
    let total_opportunities = events.len() + barriers.len();
    let applied = events.len();
    let blocked = barriers.len();
    eprintln!(
        "\n  Summary: {} opportunities found, {} applied, {} barriers",
        total_opportunities, applied, blocked
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_strategy_display() {
        assert_eq!(format!("{}", FusionStrategy::Epilogue), "epilogue");
        assert_eq!(format!("{}", FusionStrategy::Reduction), "reduction");
        assert_eq!(format!("{}", FusionStrategy::Elementwise), "elementwise");
    }

    #[test]
    fn test_barrier_reason_display() {
        assert_eq!(format!("{}", BarrierReason::MultiConsumer), "multi-consumer");
        assert_eq!(format!("{}", BarrierReason::NoFuseAnnotation), "@no_fuse annotation");
    }

    #[test]
    fn test_empty_report_no_output() {
        // Should not panic with empty input
        print_fusion_report(&[], &[]);
    }

    #[test]
    fn test_fusion_event_creation() {
        let event = FusionEvent {
            function_name: "forward".into(),
            strategy: FusionStrategy::Epilogue,
            matched_ops: vec!["matmul".into(), "bias_add".into(), "relu".into()],
            eliminated_launches: 2,
            estimated_bytes_saved: 32 * 1024 * 1024,
            location: "model.nsl:15".into(),
        };
        assert_eq!(event.eliminated_launches, 2);
    }
}
