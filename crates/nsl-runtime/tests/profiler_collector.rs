use nsl_runtime::profiler::collector::{ClockSource, Collector};

struct FakeClock(std::sync::atomic::AtomicU64);
impl FakeClock {
    fn new(us_hundredths: u64) -> Self {
        Self(std::sync::atomic::AtomicU64::new(us_hundredths))
    }
}
impl ClockSource for FakeClock {
    fn elapsed_us(&self, _start: u64, _end: u64) -> f64 {
        self.0.load(std::sync::atomic::Ordering::Relaxed) as f64 / 100.0
    }
}

#[test]
fn aggregates_across_many_pairs_without_growing_unbounded() {
    let clock = FakeClock::new(100); // 1.00 μs per pair
    let mut c = Collector::new_with_clock(Box::new(clock));
    for i in 0..10_000u64 {
        c.begin(0, i);
        c.end(0, i + 1);
    }
    let agg = c.snapshot().into_iter().find(|a| a.kernel_id == 0).unwrap();
    assert_eq!(agg.count, 10_000);
    assert!((agg.sum_us - 10_000.0).abs() < 1e-3);
    assert!(
        c.drain_queue_len(0) == 0,
        "drain queue should be flushed by snapshot()"
    );
}

#[test]
fn begin_without_end_is_idempotent() {
    let clock = FakeClock::new(100);
    let mut c = Collector::new_with_clock(Box::new(clock));
    c.begin(1, 0);
    c.begin(1, 1); // stale begin overwritten
    c.end(1, 2);
    let agg = c.snapshot().into_iter().find(|a| a.kernel_id == 1).unwrap();
    assert_eq!(agg.count, 1);
}

#[test]
fn flush_writes_actual_json_in_expected_shape() {
    let clock = FakeClock::new(250); // 2.5 μs
    let mut c = Collector::new_with_clock(Box::new(clock));
    c.begin(3, 0);
    c.end(3, 1);
    let tmp = tempfile::NamedTempFile::new().unwrap();
    c.flush_to(tmp.path()).unwrap();
    let s = std::fs::read_to_string(tmp.path()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_eq!(v["aggregates"][0]["kernel_id"], 3);
    assert_eq!(v["aggregates"][0]["count"], 1);
    assert!((v["aggregates"][0]["sum_us"].as_f64().unwrap() - 2.5).abs() < 1e-6);
}

#[test]
fn unknown_end_without_begin_is_noop() {
    let clock = FakeClock::new(100);
    let mut c = Collector::new_with_clock(Box::new(clock));
    c.end(42, 0);
    assert!(c.snapshot().is_empty());
}
