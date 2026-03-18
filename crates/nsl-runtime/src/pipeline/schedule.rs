//! M43: Pipeline schedule generation — 1F1B and GPipe.

/// A single step in the pipeline schedule for one stage.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleStep {
    Forward(usize),         // micro-batch index
    Backward(usize),        // micro-batch index
    Idle,                   // pipeline bubble
    SendActivation(usize),  // send to next stage
    RecvActivation(usize),  // recv from prev stage
    SendGradient(usize),    // send grad to prev stage
    RecvGradient(usize),    // recv grad from next stage
}

/// Complete pipeline schedule: steps[stage_id][time_slot].
#[derive(Debug, Clone)]
pub struct PipelineSchedule {
    pub steps: Vec<Vec<ScheduleStep>>,
    pub num_stages: usize,
    pub num_micro_batches: usize,
}

impl PipelineSchedule {
    /// Generate 1F1B (one-forward-one-backward) schedule.
    ///
    #[allow(clippy::needless_range_loop)]
    /// Per-stage algorithm (correct staggering):
    ///   warmup_forwards = num_stages - 1 - stage  (stage 0: P-1 warmups, last stage: 0)
    ///   steady_pairs = num_micro_batches - warmup_forwards
    ///   cooldown_backwards = warmup_forwards
    ///
    /// Warmup: Forward(0..warmup_forwards)
    /// Steady: for i in 0..steady_pairs: Backward(i), Forward(warmup_forwards + i)
    /// Cooldown: Backward(steady_pairs..num_micro_batches)
    pub fn generate_1f1b(num_stages: usize, num_micro_batches: usize) -> Self {
        let mut steps = vec![Vec::new(); num_stages];

        for stage in 0..num_stages {
            let warmup_fwds = (num_stages - 1 - stage).min(num_micro_batches);
            let steady_pairs = num_micro_batches.saturating_sub(warmup_fwds);
            let cooldown_bwds = warmup_fwds;

            let mut fwd_idx = 0usize; // next forward micro-batch to schedule
            let mut bwd_idx = 0usize; // next backward micro-batch to schedule

            // Phase 1: Warmup — pure forward fills
            for _ in 0..warmup_fwds {
                if stage > 0 {
                    steps[stage].push(ScheduleStep::RecvActivation(fwd_idx));
                }
                steps[stage].push(ScheduleStep::Forward(fwd_idx));
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::SendActivation(fwd_idx));
                }
                fwd_idx += 1;
            }

            // Phase 2: Steady state — alternating 1B 1F
            for _ in 0..steady_pairs {
                // Backward
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::RecvGradient(bwd_idx));
                }
                steps[stage].push(ScheduleStep::Backward(bwd_idx));
                if stage > 0 {
                    steps[stage].push(ScheduleStep::SendGradient(bwd_idx));
                }
                bwd_idx += 1;

                // Forward (if micro-batches remain)
                if fwd_idx < num_micro_batches {
                    if stage > 0 {
                        steps[stage].push(ScheduleStep::RecvActivation(fwd_idx));
                    }
                    steps[stage].push(ScheduleStep::Forward(fwd_idx));
                    if stage < num_stages - 1 {
                        steps[stage].push(ScheduleStep::SendActivation(fwd_idx));
                    }
                    fwd_idx += 1;
                }
            }

            // Phase 3: Cooldown — remaining backwards
            for _ in 0..cooldown_bwds {
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::RecvGradient(bwd_idx));
                }
                steps[stage].push(ScheduleStep::Backward(bwd_idx));
                if stage > 0 {
                    steps[stage].push(ScheduleStep::SendGradient(bwd_idx));
                }
                bwd_idx += 1;
            }
        }

        PipelineSchedule { steps, num_stages, num_micro_batches }
    }

    /// Generate GPipe schedule (all forwards then all backwards).
    #[allow(clippy::needless_range_loop)]
    pub fn generate_gpipe(num_stages: usize, num_micro_batches: usize) -> Self {
        let mut steps = vec![Vec::new(); num_stages];

        // All forwards first
        for mb in 0..num_micro_batches {
            for stage in 0..num_stages {
                if stage > 0 {
                    steps[stage].push(ScheduleStep::RecvActivation(mb));
                }
                steps[stage].push(ScheduleStep::Forward(mb));
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::SendActivation(mb));
                }
            }
        }

        // Then all backwards (reverse micro-batch order)
        for mb in (0..num_micro_batches).rev() {
            for stage in (0..num_stages).rev() {
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::RecvGradient(mb));
                }
                steps[stage].push(ScheduleStep::Backward(mb));
                if stage > 0 {
                    steps[stage].push(ScheduleStep::SendGradient(mb));
                }
            }
        }

        PipelineSchedule { steps, num_stages, num_micro_batches }
    }

    /// Calculate the bubble ratio (fraction of idle time).
    pub fn bubble_ratio(&self) -> f64 {
        let total: usize = self.steps.iter().map(|s| s.len()).sum();
        let idle: usize = self.steps.iter()
            .map(|s| s.iter().filter(|step| matches!(step, ScheduleStep::Idle)).count())
            .sum();
        if total == 0 { return 0.0; }
        idle as f64 / total as f64
    }

    /// Count forward steps for a given stage.
    pub fn forward_count(&self, stage: usize) -> usize {
        self.steps[stage].iter()
            .filter(|s| matches!(s, ScheduleStep::Forward(_)))
            .count()
    }

    /// Count backward steps for a given stage.
    pub fn backward_count(&self, stage: usize) -> usize {
        self.steps[stage].iter()
            .filter(|s| matches!(s, ScheduleStep::Backward(_)))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1f1b_forward_backward_counts() {
        let sched = PipelineSchedule::generate_1f1b(4, 8);
        // Each stage should process all 8 micro-batches forward and backward
        for stage in 0..4 {
            assert_eq!(sched.forward_count(stage), 8,
                "stage {stage} forward count");
            assert_eq!(sched.backward_count(stage), 8,
                "stage {stage} backward count");
        }
    }

    #[test]
    fn test_1f1b_bubble_decreases_with_more_microbatches() {
        let sched_few = PipelineSchedule::generate_1f1b(4, 4);
        let sched_many = PipelineSchedule::generate_1f1b(4, 16);
        // More micro-batches -> smaller bubble fraction
        assert!(sched_many.bubble_ratio() <= sched_few.bubble_ratio(),
            "bubble ratio should decrease: few={}, many={}",
            sched_few.bubble_ratio(), sched_many.bubble_ratio());
    }

    #[test]
    fn test_gpipe_all_forwards_before_backwards() {
        let sched = PipelineSchedule::generate_gpipe(2, 4);
        // Stage 0: all forwards should come before all backwards
        let mut seen_backward = false;
        for step in &sched.steps[0] {
            if matches!(step, ScheduleStep::Backward(_)) {
                seen_backward = true;
            }
            if matches!(step, ScheduleStep::Forward(_)) && seen_backward {
                panic!("GPipe: forward after backward in stage 0");
            }
        }
    }

    #[test]
    fn test_gpipe_forward_backward_counts() {
        let sched = PipelineSchedule::generate_gpipe(3, 6);
        for stage in 0..3 {
            assert_eq!(sched.forward_count(stage), 6);
            assert_eq!(sched.backward_count(stage), 6);
        }
    }

    #[test]
    fn test_single_stage_no_communication() {
        let sched = PipelineSchedule::generate_1f1b(1, 4);
        // Single stage: no send/recv steps
        for step in &sched.steps[0] {
            assert!(!matches!(step,
                ScheduleStep::SendActivation(_) | ScheduleStep::RecvActivation(_) |
                ScheduleStep::SendGradient(_) | ScheduleStep::RecvGradient(_)
            ), "single stage should have no communication");
        }
    }

    #[test]
    fn test_bubble_ratio_single_stage_zero() {
        let sched = PipelineSchedule::generate_1f1b(1, 4);
        assert_eq!(sched.bubble_ratio(), 0.0);
    }
}
