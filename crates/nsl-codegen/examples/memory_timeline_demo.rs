//! Synthesizes a realistic transformer forward+backward activation
//! schedule and renders the HBM timeline staircase. Writes the output to
//! `c:/tmp/memory_timeline_demo.txt` for screenshotting.
//!
//! Run with: cargo run -p nsl-codegen --example memory_timeline_demo
//!
//! Config modeled: B=4, S=512, d_model=768, d_ff=3072, H=12, bf16.
//! Shape arithmetic:
//!   activation (B·S·d_model·2)    = 3  MB
//!   ffn_hidden (B·S·d_ff·2)       = 12 MB   (4× d_model)
//!   attn_mat   (B·H·S·S·2)        = 24 MB   (the quadratic term)
//!
//! Each forward activation is `save-for-backward`: it's born at its
//! producing program-point and dies one pp past its backward consumer
//! (the renderer uses [birth, death) lifetimes).

use nsl_codegen::profiling::memory_timeline::{build, render, MemoryTimelineInput};
use nsl_codegen::wrga_memory::{MemoryPlan, MemoryPlanStats, SlotAssignment};

fn mb(n: u64) -> u64 {
    n * 1024 * 1024
}

fn main() {
    let a = mb(3);
    let ff = mb(12);
    let sc = mb(24);

    // (birth, death_exclusive, size) — one slot per forward activation.
    // Backward pps 16..=22 consume activations in reverse but produce no
    // new long-lived slots (activation grads are immediately propagated).
    let intervals = [
        (0, 23, a),   // embed       consumed by dembed       @ pp22
        (1, 22, a),   // ln1         consumed by dln1         @ pp21
        (2, 21, a),   // q_proj      consumed by dq           @ pp20
        (3, 21, a),   // k_proj      consumed by dk           @ pp20
        (4, 21, a),   // v_proj      consumed by dv           @ pp20
        (5, 20, sc),  // attn_scores consumed by dscores      @ pp19
        (6, 19, sc),  // attn_probs  consumed by dprobs       @ pp18
        (7, 18, a),   // attn_out    consumed by dattn_out    @ pp17
        (8, 18, a),   // proj_out    consumed by dproj_out    @ pp17
        (9, 18, a),   // residual1   consumed by dres1        @ pp17
        (10, 17, a),  // ln2         consumed by dln2         @ pp16
        (11, 17, ff), // fc1_out     consumed by dfc1         @ pp16
        (12, 17, ff), // gelu_out    consumed by dgelu        @ pp16
        (13, 17, a),  // fc2_out     consumed by dfc2         @ pp16
        (14, 17, a),  // residual2   consumed by dres2        @ pp16
    ];

    let assignments: Vec<SlotAssignment> = intervals
        .iter()
        .enumerate()
        .map(|(i, &(birth, death, size))| SlotAssignment {
            var: i as u32,
            slot: i as u32,
            size_bytes: size,
            birth,
            death,
        })
        .collect();

    let plan = MemoryPlan {
        assignments,
        stats: MemoryPlanStats::default(),
    };

    let phase_markers = vec![
        (0, "embed".into()),
        (2, "attn: Q/K/V proj".into()),
        (5, "attn: scores (B·H·S·S)".into()),
        (7, "attn: out".into()),
        (10, "ffn: 4x up-proj".into()),
        (14, "forward peak".into()),
        (15, "=== LOSS ===".into()),
        (16, "backward: ffn".into()),
        (18, "backward: attn".into()),
        (22, "backward: embed".into()),
    ];

    let tl = build(&MemoryTimelineInput {
        plan: &plan,
        phase_markers,
    });
    let body = render(&tl);

    let header = "\
NSL — HBM memory timeline (transformer fwd+bwd, B=4 S=512 d=768 H=12, bf16)
-----------------------------------------------------------------------------
Each row is one program-point in the Wengert list. The bar shows live HBM
bytes at that instant; phase labels annotate the ops. Activations are saved
for backward and freed as their backward consumer runs — hence the
rise-plateau-fall mountain shape with peak at the loss.
";

    let out = format!("{header}{body}");
    let path = "c:/tmp/memory_timeline_demo.txt";
    std::fs::write(path, &out).expect("write demo output");
    print!("{out}");
    println!("\n(wrote {path})");
}
