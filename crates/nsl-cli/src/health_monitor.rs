//! Renders HealthSnapshot in PDF §4.3 layout. TTY-aware: in-place ANSI
//! cursor restore on terminals; plain append otherwise.

use std::io::{IsTerminal, Write};
use std::collections::{BTreeMap, HashMap};
use nsl_runtime::health::collector::HealthSnapshot;

const BAR_WIDTH: usize = 18;
const GRAD_NORM_HEALTHY_MAX: f64 = 16.0;
const WEIGHT_DELTA_WARN_PCT: f64 = 2.0;

pub struct HealthRenderer {
    last_render_lines: usize,
    is_tty: bool,
}

impl Default for HealthRenderer { fn default() -> Self { Self::new() } }

impl HealthRenderer {
    pub fn new() -> Self {
        Self { last_render_lines: 0, is_tty: std::io::stderr().is_terminal() }
    }

    pub fn render(&mut self, snap: &HealthSnapshot) {
        let body = self.format_block(snap);
        let mut stderr = std::io::stderr().lock();
        if self.is_tty && self.last_render_lines > 0 {
            for _ in 0..self.last_render_lines {
                let _ = write!(stderr, "\x1b[1A\x1b[2K");
            }
        }
        let _ = write!(stderr, "{}", body);
        let _ = stderr.flush();
        self.last_render_lines = body.lines().count();
    }

    pub fn format_block(&self, snap: &HealthSnapshot) -> String {
        let mut out = String::new();
        out.push_str("=== Training Health Monitor (live) ===\n");

        let step_label = match snap.max_steps {
            Some(m) => format!("Step {}/{}", snap.step, m),
            None => format!("Step {}", snap.step),
        };
        let loss_str = snap.loss.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "—".into());
        let grad_str = snap.grad_norm_total.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "—".into());
        out.push_str(&format!("{} | Loss: {} | Grad norm: {}\n\n", step_label, loss_str, grad_str));

        if !snap.per_layer_grad_norm.is_empty() {
            out.push_str("Per-layer gradient norms:\n");
            let mut layers: Vec<_> = snap.per_layer_grad_norm.iter().collect();
            layers.sort_by_key(|(idx, _)| **idx);
            let max_norm = snap.per_layer_grad_norm.values().cloned()
                .fold(0.0_f64, f64::max).max(1e-9);
            for (idx, norm) in &layers {
                let bar = render_bar(**norm, max_norm);
                let tag = if **norm > GRAD_NORM_HEALTHY_MAX { "⚠ elevated" } else { "(healthy)" };
                out.push_str(&format!(" L{}: {} {:>6.1} {}\n", idx, bar, norm, tag));
            }
            out.push('\n');
        }

        if !snap.per_tensor_weight_pct_delta.is_empty() {
            out.push_str("Per-layer weight norms (Δ from init):\n");
            for (label, entries) in group_by_layer_prefix(&snap.per_tensor_weight_pct_delta) {
                out.push_str(&format!(" {}: ", label));
                let parts: Vec<String> = entries.iter().map(|(leaf, pct)| {
                    let mark = if pct.abs() > WEIGHT_DELTA_WARN_PCT { " ⚠" } else { "" };
                    format!("{}: {:+.1}%{}", leaf, pct, mark)
                }).collect();
                out.push_str(&parts.join(" "));
                out.push('\n');
            }
            out.push('\n');
        }

        let nan_status = if snap.nan_inf_count_window == 0 {
            format!("clean (0 NaN/Inf detected in {} steps)", snap.steps_in_window)
        } else {
            format!("⚠ {} NaN/Inf detected in {} steps",
                snap.nan_inf_count_window, snap.steps_in_window)
        };
        out.push_str(&format!("NaN watch: {}\n", nan_status));

        match snap.loss_ema_slope {
            Some(slope) => {
                let trend = if slope < -0.001 { "decreasing" }
                            else if slope > 0.001 { "increasing ⚠" }
                            else { "flat" };
                out.push_str(&format!("Loss trend: {} (EMA slope: {:+.3}/step)\n", trend, slope));
            }
            None => out.push_str("Loss trend: (insufficient samples)\n"),
        }
        out
    }
}

fn render_bar(value: f64, max: f64) -> String {
    let frac = (value / max).clamp(0.0, 1.0);
    let filled = (frac * BAR_WIDTH as f64).round() as usize;
    "█".repeat(filled) + &"░".repeat(BAR_WIDTH.saturating_sub(filled))
}

pub fn group_by_layer_prefix(items: &HashMap<String, f64>)
    -> Vec<(String, Vec<(String, f64)>)>
{
    let mut groups: BTreeMap<(String, u32), Vec<(String, f64)>> = BTreeMap::new();
    for (path, value) in items {
        let segments: Vec<&str> = path.split('.').collect();
        let last_num_idx = segments.iter().enumerate().rev()
            .find_map(|(i, s)| s.parse::<u32>().ok().map(|_| i));
        match last_num_idx {
            Some(i) => {
                let prefix = segments[..i].join(".");
                let layer_idx: u32 = segments[i].parse().unwrap();
                let leaf = segments[i+1..].join(".");
                let label = make_layer_label(&prefix);
                groups.entry((label, layer_idx)).or_default().push((leaf, *value));
            }
            None => {
                groups.entry(("misc".to_string(), u32::MAX)).or_default()
                    .push((path.clone(), *value));
            }
        }
    }
    groups.into_iter().map(|((label, idx), v)| {
        let final_label = if idx == u32::MAX { "misc".to_string() }
            else if label == "L" { format!("L{}", idx) }
            else { format!("{}L{}", label, idx) };
        (final_label, v)
    }).collect()
}

fn make_layer_label(prefix: &str) -> String {
    if prefix.contains("encoder") { "Enc.".into() }
    else if prefix.contains("decoder") { "Dec.".into() }
    else { "L".into() }
}
