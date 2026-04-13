use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub trait ClockSource: Send {
    fn elapsed_us(&self, start: u64, end: u64) -> f64;
}

/// Default clock that treats (end - start) as nanoseconds and returns microseconds.
/// Used by the FFI layer where start/end are nanosecond timestamps.
pub struct NanoClock;
impl ClockSource for NanoClock {
    fn elapsed_us(&self, start: u64, end: u64) -> f64 {
        end.saturating_sub(start) as f64 / 1000.0
    }
}

#[cfg(feature = "cuda-real-events")]
pub struct CudaEventClock;
#[cfg(feature = "cuda-real-events")]
impl ClockSource for CudaEventClock {
    fn elapsed_us(&self, _start: u64, _end: u64) -> f64 {
        // TODO: wrap cudarc cuEventElapsedTime. Phase 2.
        unimplemented!("cuda-real-events backend not yet implemented")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aggregate {
    pub kernel_id: u32,
    pub count: u64,
    pub sum_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub sum_sq_us: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActualReport {
    pub aggregates: Vec<Aggregate>,
}

const DRAIN_CAP: usize = 64;

pub struct Collector {
    clock: Box<dyn ClockSource>,
    in_flight: HashMap<u32, u64>,
    drains: HashMap<u32, Vec<(u64, u64)>>,
    aggregates: HashMap<u32, Aggregate>,
}

impl Collector {
    pub fn new_with_clock(clock: Box<dyn ClockSource>) -> Self {
        Self {
            clock,
            in_flight: HashMap::new(),
            drains: HashMap::new(),
            aggregates: HashMap::new(),
        }
    }
    pub fn begin(&mut self, kernel_id: u32, start_handle: u64) {
        self.in_flight.insert(kernel_id, start_handle);
    }
    pub fn end(&mut self, kernel_id: u32, end_handle: u64) {
        if let Some(start) = self.in_flight.remove(&kernel_id) {
            let q = self.drains.entry(kernel_id).or_default();
            q.push((start, end_handle));
            if q.len() >= DRAIN_CAP {
                self.drain(kernel_id);
            }
        }
    }
    fn drain(&mut self, kernel_id: u32) {
        let Some(q) = self.drains.get_mut(&kernel_id) else { return };
        let agg = self.aggregates.entry(kernel_id).or_insert(Aggregate {
            kernel_id,
            count: 0,
            sum_us: 0.0,
            min_us: f64::INFINITY,
            max_us: 0.0,
            sum_sq_us: 0.0,
        });
        for (s, e) in q.drain(..) {
            let us = self.clock.elapsed_us(s, e);
            agg.count += 1;
            agg.sum_us += us;
            agg.sum_sq_us += us * us;
            if us < agg.min_us {
                agg.min_us = us;
            }
            if us > agg.max_us {
                agg.max_us = us;
            }
        }
    }
    pub fn drain_queue_len(&self, kernel_id: u32) -> usize {
        self.drains.get(&kernel_id).map_or(0, |v| v.len())
    }
    pub fn snapshot(&mut self) -> Vec<Aggregate> {
        let ids: Vec<_> = self.drains.keys().copied().collect();
        for id in ids {
            self.drain(id);
        }
        let mut v: Vec<Aggregate> = self.aggregates.values().cloned().collect();
        v.sort_by_key(|a| a.kernel_id);
        v
    }
    pub fn flush_to(&mut self, path: &Path) -> std::io::Result<()> {
        let report = ActualReport { aggregates: self.snapshot() };
        let s = serde_json::to_string_pretty(&report)?;
        std::fs::write(path, s)
    }
}
