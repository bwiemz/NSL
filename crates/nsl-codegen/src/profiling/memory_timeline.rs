//! HBM-usage timeline renderer. Converts SlotAssignment birth/death intervals
//! into a per-program-point live-bytes series and pretty-prints it.

use crate::profiling::types::MemoryTimelineEntry;
use crate::wrga_memory::MemoryPlan;

pub struct MemoryTimelineInput<'a> {
    pub plan: &'a MemoryPlan,
    pub phase_markers: Vec<(u32, String)>,
}

pub fn build(input: &MemoryTimelineInput) -> Vec<MemoryTimelineEntry> {
    let max_pp = input
        .plan
        .assignments
        .iter()
        .map(|s| s.death)
        .max()
        .unwrap_or(0);
    let mut out = Vec::with_capacity((max_pp + 1) as usize);
    for pp in 0..=max_pp {
        // Convention: [birth, death) — live WHILE birth <= pp < death.
        let live: u64 = input
            .plan
            .assignments
            .iter()
            .filter(|s| s.birth <= pp && pp < s.death)
            .map(|s| s.size_bytes)
            .sum();
        let phase = input
            .phase_markers
            .iter()
            .find(|(q, _)| *q == pp)
            .map(|(_, l)| l.clone());
        out.push(MemoryTimelineEntry {
            program_point: pp,
            live_bytes: live,
            phase,
        });
    }
    out
}

pub fn render(tl: &[MemoryTimelineEntry]) -> String {
    let peak = tl.iter().map(|e| e.live_bytes).max().unwrap_or(0);
    const BAR_WIDTH: usize = 20;
    let mut out = String::from("\n=== Memory Timeline ===\n\n");
    out.push_str("Time (pp)  HBM Usage\n");
    for e in tl {
        let filled = if peak == 0 {
            0
        } else {
            (e.live_bytes * BAR_WIDTH as u64 / peak) as usize
        };
        let bar = "█".repeat(filled) + &"░".repeat(BAR_WIDTH - filled);
        let mb = e.live_bytes as f64 / (1024.0 * 1024.0);
        let phase = e.phase.as_deref().unwrap_or("");
        out.push_str(&format!(
            "{:>4}       {} {:>7.1} MB  {}\n",
            e.program_point, bar, mb, phase
        ));
    }
    out.push_str(&format!(
        "\nPeak: {:.1} MB\n",
        peak as f64 / (1024.0 * 1024.0)
    ));
    out
}
