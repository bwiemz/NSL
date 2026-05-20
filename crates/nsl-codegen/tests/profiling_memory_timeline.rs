use nsl_codegen::profiling::memory_timeline::{build, render, MemoryTimelineInput};
use nsl_codegen::wrga_memory::{MemoryPlan, MemoryPlanStats, SlotAssignment};

fn plan_with_intervals(intervals: &[(u32, u32, u64)]) -> MemoryPlan {
    let assignments = intervals
        .iter()
        .enumerate()
        .map(|(i, &(birth, death, sz))| SlotAssignment {
            var: i as u32,
            slot: i as u32,
            size_bytes: sz,
            birth,
            death,
        })
        .collect();
    MemoryPlan {
        assignments,
        stats: MemoryPlanStats {
            live_activations: 0,
            slots_used: 0,
            naive_peak_bytes: 0,
            planned_peak_bytes: 0,
        },
    }
}

#[test]
fn peak_at_overlap() {
    // [0,5)=100, [2,8)=200, [4,6)=50. Peak at pp=4..5 = 350.
    let plan = plan_with_intervals(&[(0, 5, 100), (2, 8, 200), (4, 6, 50)]);
    let tl = build(&MemoryTimelineInput {
        plan: &plan,
        phase_markers: vec![],
    });
    let peak = tl.iter().map(|e| e.live_bytes).max().unwrap();
    assert_eq!(peak, 350);
}

#[test]
fn renders_ascii_bar_chart_with_peak_line() {
    let plan = plan_with_intervals(&[(0, 3, 1024 * 1024)]);
    let tl = build(&MemoryTimelineInput {
        plan: &plan,
        phase_markers: vec![],
    });
    let s = render(&tl);
    assert!(s.contains("MB"), "expected MB units in output");
    assert!(s.contains("Peak"), "expected 'Peak:' summary line");
    assert!(
        s.contains("█") || s.contains("#"),
        "expected filled bar glyphs"
    );
}

#[test]
fn phase_markers_annotate_rows() {
    let plan = plan_with_intervals(&[(0, 10, 512)]);
    let tl = build(&MemoryTimelineInput {
        plan: &plan,
        phase_markers: vec![(0, "forward_start".into()), (5, "loss".into())],
    });
    assert!(tl
        .iter()
        .any(|e| e.phase.as_deref() == Some("forward_start")));
    assert!(tl.iter().any(|e| e.phase.as_deref() == Some("loss")));
}
