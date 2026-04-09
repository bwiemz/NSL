//! M43: Pipeline parallelism codegen — 3D rank mapping and @pipeline config.

/// 3D parallelism configuration parsed from `distribute: "dp=2, tp=4, pp=4"`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelismConfig {
    pub data_parallel: usize,
    pub tensor_parallel: usize,
    pub pipeline_parallel: usize,
}

impl ParallelismConfig {
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut dp = 1;
        let mut tp = 1;
        let mut pp = 1;

        for part in s.split(',') {
            let part = part.trim();
            if let Some(val) = part.strip_prefix("dp=") {
                dp = val.parse().map_err(|_| format!("invalid dp: '{val}'"))?;
            } else if let Some(val) = part.strip_prefix("tp=") {
                tp = val.parse().map_err(|_| format!("invalid tp: '{val}'"))?;
            } else if let Some(val) = part.strip_prefix("pp=") {
                pp = val.parse().map_err(|_| format!("invalid pp: '{val}'"))?;
            } else {
                return Err(format!("unknown key: '{part}'"));
            }
        }
        if dp < 1 || tp < 1 || pp < 1 {
            return Err("all dimensions must be >= 1".into());
        }
        Ok(ParallelismConfig {
            data_parallel: dp,
            tensor_parallel: tp,
            pipeline_parallel: pp,
        })
    }

    pub fn total_gpus(&self) -> usize {
        self.data_parallel * self.tensor_parallel * self.pipeline_parallel
    }
}

/// Map flat rank to 3D (dp, pp, tp) coordinates.
/// Layout: dp outermost, pp middle, tp innermost (TP peers on adjacent GPUs).
pub fn rank_to_3d(rank: usize, config: &ParallelismConfig) -> (usize, usize, usize) {
    let tp_rank = rank % config.tensor_parallel;
    let pp_rank = (rank / config.tensor_parallel) % config.pipeline_parallel;
    let dp_rank = rank / (config.tensor_parallel * config.pipeline_parallel);
    (dp_rank, pp_rank, tp_rank)
}

/// Inverse: 3D coordinates to flat rank.
pub fn rank_from_3d(dp: usize, pp: usize, tp: usize, config: &ParallelismConfig) -> usize {
    dp * config.pipeline_parallel * config.tensor_parallel + pp * config.tensor_parallel + tp
}

/// Get tensor-parallel peers (same dp, same pp, all tp ranks).
pub fn get_tp_peers(rank: usize, config: &ParallelismConfig) -> Vec<usize> {
    let (dp, pp, _) = rank_to_3d(rank, config);
    (0..config.tensor_parallel)
        .map(|tp| rank_from_3d(dp, pp, tp, config))
        .collect()
}

/// Get pipeline-parallel neighbors (prev stage, next stage).
pub fn get_pp_neighbors(rank: usize, config: &ParallelismConfig) -> (Option<usize>, Option<usize>) {
    let (dp, pp, tp) = rank_to_3d(rank, config);
    let prev = if pp > 0 {
        Some(rank_from_3d(dp, pp - 1, tp, config))
    } else {
        None
    };
    let next = if pp < config.pipeline_parallel - 1 {
        Some(rank_from_3d(dp, pp + 1, tp, config))
    } else {
        None
    };
    (prev, next)
}

/// Get data-parallel peers (same pp, same tp, all dp ranks).
pub fn get_dp_peers(rank: usize, config: &ParallelismConfig) -> Vec<usize> {
    let (_, pp, tp) = rank_to_3d(rank, config);
    (0..config.data_parallel)
        .map(|dp| rank_from_3d(dp, pp, tp, config))
        .collect()
}

/// Pipeline configuration extracted from @pipeline decorator.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub num_stages: usize,
    pub schedule_type: ScheduleType,
    pub checkpoint_stages: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduleType {
    OneF1B,
    GPipe,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_distribute_basic() {
        let config = ParallelismConfig::parse("dp=2, tp=4, pp=4").unwrap();
        assert_eq!(config.data_parallel, 2);
        assert_eq!(config.tensor_parallel, 4);
        assert_eq!(config.pipeline_parallel, 4);
        assert_eq!(config.total_gpus(), 32);
    }

    #[test]
    fn parse_distribute_single() {
        let config = ParallelismConfig::parse("pp=4").unwrap();
        assert_eq!(config.data_parallel, 1);
        assert_eq!(config.pipeline_parallel, 4);
        assert_eq!(config.total_gpus(), 4);
    }

    #[test]
    fn parse_distribute_error() {
        assert!(ParallelismConfig::parse("dp=0").is_err());
        assert!(ParallelismConfig::parse("foo=2").is_err());
        assert!(ParallelismConfig::parse("dp=abc").is_err());
    }

    #[test]
    fn rank_3d_roundtrip() {
        let config = ParallelismConfig {
            data_parallel: 2,
            tensor_parallel: 4,
            pipeline_parallel: 4,
        };
        for rank in 0..32 {
            let (dp, pp, tp) = rank_to_3d(rank, &config);
            assert_eq!(
                rank_from_3d(dp, pp, tp, &config),
                rank,
                "roundtrip failed for rank {rank}"
            );
        }
    }

    #[test]
    fn rank_3d_known_values() {
        let config = ParallelismConfig {
            data_parallel: 2,
            tensor_parallel: 4,
            pipeline_parallel: 4,
        };
        // Rank 0: dp=0, pp=0, tp=0
        assert_eq!(rank_to_3d(0, &config), (0, 0, 0));
        // Rank 3: dp=0, pp=0, tp=3
        assert_eq!(rank_to_3d(3, &config), (0, 0, 3));
        // Rank 4: dp=0, pp=1, tp=0
        assert_eq!(rank_to_3d(4, &config), (0, 1, 0));
        // Rank 16: dp=1, pp=0, tp=0
        assert_eq!(rank_to_3d(16, &config), (1, 0, 0));
    }

    #[test]
    fn tp_peers() {
        let config = ParallelismConfig {
            data_parallel: 2,
            tensor_parallel: 4,
            pipeline_parallel: 2,
        };
        let peers = get_tp_peers(0, &config);
        assert_eq!(peers, vec![0, 1, 2, 3]);
    }

    #[test]
    fn pp_neighbors() {
        let config = ParallelismConfig {
            data_parallel: 1,
            tensor_parallel: 1,
            pipeline_parallel: 4,
        };
        assert_eq!(get_pp_neighbors(0, &config), (None, Some(1)));
        assert_eq!(get_pp_neighbors(1, &config), (Some(0), Some(2)));
        assert_eq!(get_pp_neighbors(3, &config), (Some(2), None));
    }

    #[test]
    fn dp_peers() {
        let config = ParallelismConfig {
            data_parallel: 4,
            tensor_parallel: 2,
            pipeline_parallel: 2,
        };
        // Rank 0: dp=0, pp=0, tp=0. DP peers: ranks 0, 4, 8, 12
        let peers = get_dp_peers(0, &config);
        assert_eq!(peers, vec![0, 4, 8, 12]);
    }
}
