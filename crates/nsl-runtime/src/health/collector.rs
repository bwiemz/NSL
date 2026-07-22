use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

const LOSS_WINDOW: usize = 100;
const EMA_ALPHA: f64 = 0.05;
const FLUSH_INTERVAL_DEFAULT: u64 = 100;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HealthSnapshot {
    pub step: u64,
    pub max_steps: Option<u64>,
    pub loss: Option<f64>,
    pub loss_ema: Option<f64>,
    pub loss_ema_slope: Option<f64>,
    pub grad_norm_total: Option<f64>,
    pub per_layer_grad_norm: HashMap<u32, f64>,
    /// P0 certification: global parameter L2 norm — sqrt of the sum of
    /// squared per-tensor weight norms recorded this window. The direct
    /// "parameter norm curve" signal (pct_delta below is relative-only).
    pub weight_norm_total: Option<f64>,
    pub per_tensor_weight_pct_delta: HashMap<String, f64>,
    pub nan_inf_count_window: u64,
    pub steps_in_window: u64,
}

pub struct HealthCollector {
    step: u64,
    last_flushed_step: u64,
    flush_interval: u64,
    loss_history: VecDeque<f64>,
    loss_ema: Option<f64>,
    grad_norm_per_layer: HashMap<u32, f64>,
    weight_init: HashMap<String, f64>,
    weight_current: HashMap<String, f64>,
    nan_inf_count: u64,
}

impl Default for HealthCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthCollector {
    pub fn new() -> Self {
        Self {
            step: 0,
            last_flushed_step: u64::MAX,
            flush_interval: FLUSH_INTERVAL_DEFAULT,
            loss_history: VecDeque::with_capacity(LOSS_WINDOW),
            loss_ema: None,
            grad_norm_per_layer: HashMap::new(),
            weight_init: HashMap::new(),
            weight_current: HashMap::new(),
            nan_inf_count: 0,
        }
    }

    pub fn set_flush_interval(&mut self, n: u64) {
        self.flush_interval = n.max(1);
    }

    pub fn record_loss(&mut self, step: u64, value: f64) {
        self.step = step;
        if value.is_nan() || value.is_infinite() {
            self.nan_inf_count += 1;
            return;
        }
        if self.loss_history.len() == LOSS_WINDOW {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(value);
        self.loss_ema = Some(match self.loss_ema {
            None => value,
            Some(prev) => prev * (1.0 - EMA_ALPHA) + value * EMA_ALPHA,
        });
    }

    /// Most recent finite loss recorded via `record_loss` (NaN/Inf are counted
    /// but never enter the history, so this is the last *usable* loss).
    pub fn last_loss(&self) -> Option<f64> {
        self.loss_history.back().copied()
    }

    pub fn record_grad_norm(&mut self, _path: &str, layer_idx: u32, norm: f64) {
        self.grad_norm_per_layer.insert(layer_idx, norm);
    }

    pub fn record_weight_norm(&mut self, path: &str, norm: f64, is_init: bool) {
        if is_init {
            self.weight_init.insert(path.to_string(), norm);
        }
        self.weight_current.insert(path.to_string(), norm);
    }

    pub fn snapshot(&mut self) -> HealthSnapshot {
        let last = self.loss_history.back().copied();
        let slope = if self.loss_history.len() >= 2 {
            let n = self.loss_history.len() as f64;
            let mean_x = (n - 1.0) / 2.0;
            let mean_y = self.loss_history.iter().sum::<f64>() / n;
            let (mut num, mut den) = (0.0_f64, 0.0_f64);
            for (i, &y) in self.loss_history.iter().enumerate() {
                let dx = i as f64 - mean_x;
                num += dx * (y - mean_y);
                den += dx * dx;
            }
            if den > 0.0 {
                Some(num / den)
            } else {
                None
            }
        } else {
            None
        };

        let grad_norm_total = if self.grad_norm_per_layer.is_empty() {
            None
        } else {
            let sumsq: f64 = self.grad_norm_per_layer.values().map(|n| n * n).sum();
            Some(sumsq.sqrt())
        };

        let per_tensor_weight_pct_delta: HashMap<String, f64> = self
            .weight_current
            .iter()
            .filter_map(|(path, cur)| {
                self.weight_init.get(path).map(|init| {
                    let pct = if *init > 0.0 {
                        (cur - init) / init * 100.0
                    } else {
                        0.0
                    };
                    (path.clone(), pct)
                })
            })
            .collect();

        let weight_norm_total = if self.weight_current.is_empty() {
            None
        } else {
            let sumsq: f64 = self.weight_current.values().map(|n| n * n).sum();
            Some(sumsq.sqrt())
        };

        HealthSnapshot {
            step: self.step,
            max_steps: None,
            loss: last,
            loss_ema: self.loss_ema,
            loss_ema_slope: slope,
            grad_norm_total,
            per_layer_grad_norm: self.grad_norm_per_layer.clone(),
            weight_norm_total,
            per_tensor_weight_pct_delta,
            nan_inf_count_window: self.nan_inf_count,
            steps_in_window: self.loss_history.len() as u64,
        }
    }

    pub fn should_flush(&mut self) -> bool {
        if self.step == 0
            || self.last_flushed_step == u64::MAX
            || self.step.saturating_sub(self.last_flushed_step) >= self.flush_interval
        {
            self.last_flushed_step = self.step;
            true
        } else {
            false
        }
    }
}
