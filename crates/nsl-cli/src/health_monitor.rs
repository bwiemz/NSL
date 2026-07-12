//! Renders HealthSnapshot in PDF §4.3 layout. TTY-aware: in-place ANSI
//! cursor restore on terminals; plain append otherwise.

use std::io::{IsTerminal, Write};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::SystemTime;
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

/// One poll-loop step for the live monitor thread (`nsl run --monitor` on a
/// train program). Returns a parsed snapshot only when the file exists and
/// its modification time differs from `*last_mtime`, i.e. the training
/// runtime has flushed since the previous poll (or the file just appeared).
///
/// `last_mtime` advances only on a successful parse: a snapshot caught
/// mid-write (truncated or partial JSON) is skipped silently and retried on
/// the next poll instead of being lost. All I/O errors also return None.
pub fn poll_health_file(
    path: &Path,
    last_mtime: &mut Option<SystemTime>,
) -> Option<HealthSnapshot> {
    let mtime = std::fs::metadata(path).ok()?.modified().ok()?;
    if *last_mtime == Some(mtime) {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    let snap: HealthSnapshot = serde_json::from_str(&text).ok()?;
    *last_mtime = Some(mtime);
    Some(snap)
}

/// True when the snapshot file's current mtime differs from the one observed
/// before the training child was launched, i.e. THIS run's runtime flushed
/// at least once. A missing file reads as "not flushed"; a file that still
/// carries the pre-run mtime is a stale leftover from a previous run and
/// must not be rendered as this run's final state.
pub fn health_file_changed_since(
    path: &Path,
    initial_mtime: Option<SystemTime>,
) -> bool {
    match std::fs::metadata(path).ok().and_then(|m| m.modified().ok()) {
        None => false,
        Some(current) => Some(current) != initial_mtime,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn snapshot_json(step: u64) -> String {
        let snap = HealthSnapshot { step, ..Default::default() };
        serde_json::to_string(&snap).expect("serialize test snapshot")
    }

    /// Force the file's mtime to a known, strictly increasing value so the
    /// tests never depend on filesystem timestamp granularity or on real
    /// clock waits between writes.
    fn bump_mtime(path: &Path, offset: Duration) {
        let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open for mtime bump");
        f.set_modified(base + offset).expect("set_modified");
    }

    #[test]
    fn poll_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nsl.nsl-health.json");
        let mut last = None;
        assert!(poll_health_file(&path, &mut last).is_none());
        assert!(last.is_none());
    }

    #[test]
    fn poll_new_file_parses_then_unchanged_mtime_skips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nsl.nsl-health.json");
        std::fs::write(&path, snapshot_json(7)).unwrap();
        bump_mtime(&path, Duration::from_secs(1));

        let mut last = None;
        let snap = poll_health_file(&path, &mut last).expect("first poll parses");
        assert_eq!(snap.step, 7);
        assert!(last.is_some());

        // Same mtime -> treated as unchanged, no re-render.
        assert!(poll_health_file(&path, &mut last).is_none());
    }

    #[test]
    fn poll_rewrite_with_new_mtime_returns_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nsl.nsl-health.json");
        std::fs::write(&path, snapshot_json(100)).unwrap();
        bump_mtime(&path, Duration::from_secs(1));

        let mut last = None;
        assert_eq!(poll_health_file(&path, &mut last).unwrap().step, 100);

        // Runtime flushes again: new content, new mtime.
        std::fs::write(&path, snapshot_json(200)).unwrap();
        bump_mtime(&path, Duration::from_secs(2));
        assert_eq!(poll_health_file(&path, &mut last).unwrap().step, 200);

        // And once consumed it goes quiet again.
        assert!(poll_health_file(&path, &mut last).is_none());
    }

    #[test]
    fn changed_since_stale_missing_and_fresh_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nsl.nsl-health.json");

        // Missing file: never counts as flushed, whatever the seed was.
        assert!(!health_file_changed_since(&path, None));
        let seed = Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1));
        assert!(!health_file_changed_since(&path, seed));

        // Stale leftover from a previous run: mtime equals the pre-run
        // seed, so it must NOT be rendered as this run's final state.
        std::fs::write(&path, snapshot_json(1)).unwrap();
        bump_mtime(&path, Duration::from_secs(1));
        let stale_seed = std::fs::metadata(&path).unwrap().modified().ok();
        assert!(!health_file_changed_since(&path, stale_seed));

        // File appeared during the run (no file existed at seed time).
        assert!(health_file_changed_since(&path, None));

        // Runtime rewrote the file after the seed was taken.
        bump_mtime(&path, Duration::from_secs(2));
        assert!(health_file_changed_since(&path, stale_seed));
    }

    #[test]
    fn poll_truncated_json_is_skipped_and_retried() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nsl.nsl-health.json");
        std::fs::write(&path, snapshot_json(1)).unwrap();
        bump_mtime(&path, Duration::from_secs(1));

        let mut last = None;
        assert_eq!(poll_health_file(&path, &mut last).unwrap().step, 1);
        let consumed = last;

        // Mid-write torn read: parse fails, poll returns None without
        // panicking and does NOT advance last_mtime.
        std::fs::write(&path, "{\"step\": 9, \"los").unwrap();
        bump_mtime(&path, Duration::from_secs(2));
        assert!(poll_health_file(&path, &mut last).is_none());
        assert_eq!(last, consumed);

        // The write completes; even if the mtime were unchanged since the
        // torn read, the un-advanced last_mtime means we retry and pick
        // the completed snapshot up.
        std::fs::write(&path, snapshot_json(9)).unwrap();
        bump_mtime(&path, Duration::from_secs(2));
        assert_eq!(poll_health_file(&path, &mut last).unwrap().step, 9);
    }
}
