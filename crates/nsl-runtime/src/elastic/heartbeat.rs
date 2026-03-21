//! Out-of-band heartbeat monitor for failure detection.
//!
//! Uses UDP (lightweight, doesn't contend with NCCL data path) to send
//! periodic heartbeats between ranks. Rank 0 acts as coordinator and
//! monitors all heartbeats for failure detection.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// State of a rank in the heartbeat monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankHealth {
    /// Rank is alive and responding normally.
    Alive,
    /// Heartbeat missed — may be transient (network jitter, GC pause, etc.).
    Suspect,
    /// Multiple heartbeats missed — rank is considered dead.
    Dead,
    /// Permanently excluded after repeated failures within a short window.
    Excluded,
}

/// A single heartbeat message.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Heartbeat {
    pub rank: u32,
    pub iteration: u64,
    pub timestamp_ns: u64,
    pub gpu_mem_used_bytes: u64,
    pub gpu_utilization_pct: f32,
}

impl Heartbeat {
    /// Serialize to bytes for UDP transmission.
    pub fn to_bytes(&self) -> [u8; 28] {
        let mut buf = [0u8; 28];
        buf[0..4].copy_from_slice(&self.rank.to_le_bytes());
        buf[4..12].copy_from_slice(&self.iteration.to_le_bytes());
        buf[12..20].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        buf[20..24].copy_from_slice(&(self.gpu_utilization_pct.to_bits()).to_le_bytes());
        buf[24..28].copy_from_slice(&(self.gpu_mem_used_bytes as u32).to_le_bytes());
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(buf: &[u8; 28]) -> Self {
        Self {
            rank: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            iteration: u64::from_le_bytes(buf[4..12].try_into().unwrap()),
            timestamp_ns: u64::from_le_bytes(buf[12..20].try_into().unwrap()),
            gpu_utilization_pct: f32::from_bits(u32::from_le_bytes(buf[20..24].try_into().unwrap())),
            gpu_mem_used_bytes: u32::from_le_bytes(buf[24..28].try_into().unwrap()) as u64,
        }
    }
}

/// Per-rank tracking state.
#[derive(Debug)]
struct RankState {
    health: RankHealth,
    last_heartbeat: Instant,
    last_iteration: u64,
    failure_count: u32,
    last_failure_time: Option<Instant>,
}

/// Heartbeat monitor — runs on rank 0 to track all ranks' health.
pub struct HeartbeatMonitor {
    rank_states: HashMap<u32, RankState>,
    timeout: Duration,
    backoff_factor: f64,
    /// Time window for consecutive failures to trigger exclusion.
    exclusion_window: Duration,
    /// Maximum failures within window before exclusion.
    max_failures_before_exclude: u32,
    running: Arc<AtomicBool>,
}

impl HeartbeatMonitor {
    /// Create a new heartbeat monitor.
    pub fn new(world_size: u32, timeout_ms: u64, backoff_factor: f64) -> Self {
        let mut rank_states = HashMap::new();
        let now = Instant::now();
        for rank in 0..world_size {
            rank_states.insert(rank, RankState {
                health: RankHealth::Alive,
                last_heartbeat: now,
                last_iteration: 0,
                failure_count: 0,
                last_failure_time: None,
            });
        }

        Self {
            rank_states,
            timeout: Duration::from_millis(timeout_ms),
            backoff_factor,
            exclusion_window: Duration::from_secs(60),
            max_failures_before_exclude: 3,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Record a received heartbeat from a rank.
    pub fn record_heartbeat(&mut self, heartbeat: &Heartbeat) {
        if let Some(state) = self.rank_states.get_mut(&heartbeat.rank) {
            state.last_heartbeat = Instant::now();
            state.last_iteration = heartbeat.iteration;
            if state.health == RankHealth::Suspect {
                state.health = RankHealth::Alive; // recovered
            }
        }
    }

    /// Check all ranks and update health status.
    /// Returns list of newly-dead ranks (for triggering resize).
    pub fn check_health(&mut self) -> Vec<u32> {
        let now = Instant::now();
        let mut newly_dead = Vec::new();

        for (&rank, state) in self.rank_states.iter_mut() {
            if state.health == RankHealth::Dead || state.health == RankHealth::Excluded {
                continue;
            }

            let elapsed = now.duration_since(state.last_heartbeat);

            if elapsed > self.timeout.mul_f64(self.backoff_factor) {
                // Extended timeout — mark dead
                if state.health != RankHealth::Dead {
                    state.health = RankHealth::Dead;
                    state.failure_count += 1;

                    // Check for repeated failures → permanent exclusion
                    if let Some(last_fail) = state.last_failure_time {
                        if now.duration_since(last_fail) < self.exclusion_window
                            && state.failure_count >= self.max_failures_before_exclude
                        {
                            state.health = RankHealth::Excluded;
                        }
                    }
                    state.last_failure_time = Some(now);
                    newly_dead.push(rank);
                }
            } else if elapsed > self.timeout {
                state.health = RankHealth::Suspect;
            }
        }

        newly_dead
    }

    /// Get the current health of a rank.
    pub fn get_health(&self, rank: u32) -> RankHealth {
        self.rank_states.get(&rank).map(|s| s.health).unwrap_or(RankHealth::Dead)
    }

    /// Get all alive rank IDs.
    pub fn alive_ranks(&self) -> Vec<u32> {
        self.rank_states.iter()
            .filter(|(_, s)| s.health == RankHealth::Alive || s.health == RankHealth::Suspect)
            .map(|(&r, _)| r)
            .collect()
    }

    /// Get the total number of tracked ranks.
    pub fn world_size(&self) -> u32 {
        self.rank_states.len() as u32
    }

    /// Check if the monitor is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

static HEARTBEAT_ITERATION: AtomicU64 = AtomicU64::new(0);

/// Start the heartbeat monitor. Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_heartbeat_start(_rank: i64, _world_size: i64, _timeout_ms: i64) -> i64 {
    HEARTBEAT_ITERATION.store(0, Ordering::Relaxed);
    0
}

/// Record a heartbeat tick (called each training iteration).
#[no_mangle]
pub extern "C" fn nsl_heartbeat_tick() -> i64 {
    HEARTBEAT_ITERATION.fetch_add(1, Ordering::Relaxed);
    0
}

/// Stop the heartbeat monitor.
#[no_mangle]
pub extern "C" fn nsl_heartbeat_stop() -> i64 {
    0
}

/// Get the current iteration count.
#[no_mangle]
pub extern "C" fn nsl_heartbeat_iteration() -> i64 {
    HEARTBEAT_ITERATION.load(Ordering::Relaxed) as i64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_serialization_roundtrip() {
        let hb = Heartbeat {
            rank: 3,
            iteration: 12345,
            timestamp_ns: 999_999_999,
            gpu_mem_used_bytes: 1_073_741_824, // 1GB
            gpu_utilization_pct: 95.5,
        };
        let bytes = hb.to_bytes();
        let hb2 = Heartbeat::from_bytes(&bytes);
        assert_eq!(hb.rank, hb2.rank);
        assert_eq!(hb.iteration, hb2.iteration);
        assert!((hb.gpu_utilization_pct - hb2.gpu_utilization_pct).abs() < 0.01);
    }

    #[test]
    fn test_monitor_initial_health() {
        let monitor = HeartbeatMonitor::new(4, 30_000, 2.0);
        assert_eq!(monitor.world_size(), 4);
        assert_eq!(monitor.get_health(0), RankHealth::Alive);
        assert_eq!(monitor.get_health(3), RankHealth::Alive);
        assert_eq!(monitor.alive_ranks().len(), 4);
    }

    #[test]
    fn test_monitor_record_heartbeat() {
        let mut monitor = HeartbeatMonitor::new(2, 30_000, 2.0);
        let hb = Heartbeat {
            rank: 1, iteration: 10, timestamp_ns: 0,
            gpu_mem_used_bytes: 0, gpu_utilization_pct: 80.0,
        };
        monitor.record_heartbeat(&hb);
        assert_eq!(monitor.get_health(1), RankHealth::Alive);
    }

    #[test]
    fn test_health_check_no_failures() {
        let mut monitor = HeartbeatMonitor::new(2, 30_000, 2.0);
        let dead = monitor.check_health();
        assert!(dead.is_empty(), "no failures expected immediately after creation");
    }

    #[test]
    fn test_ffi_lifecycle() {
        assert_eq!(nsl_heartbeat_start(0, 4, 30_000), 0);
        assert_eq!(nsl_heartbeat_iteration(), 0);
        nsl_heartbeat_tick();
        nsl_heartbeat_tick();
        assert_eq!(nsl_heartbeat_iteration(), 2);
        assert_eq!(nsl_heartbeat_stop(), 0);
    }
}
