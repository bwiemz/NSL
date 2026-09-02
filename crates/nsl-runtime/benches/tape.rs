//! CPU runtime hot paths through the C ABI compiled programs call:
//! the autodiff tape (record a small MLP step, run backward), the tiled
//! f32 matmul, and tensor allocation churn. `cargo bench -p nsl-runtime`.
//!
//! Everything here is host-side (no `cuda` feature needed), and every
//! iteration returns the runtime to its starting state — tensors are
//! refcounted, so each transient the forward produces is freed after the
//! tape is stopped, and the parameters keep their tape ids across steps
//! exactly as a training loop's do.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nsl_runtime::autodiff::{nsl_tape_backward_train, nsl_tape_start, nsl_tape_stop};
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_len, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_randn, nsl_tensor_relu, nsl_tensor_sum,
    nsl_tensor_zeros,
};

/// An NslList holding `dims`, for the shape-list-taking constructors.
fn shape(dims: &[i64]) -> i64 {
    let list = nsl_list_new();
    for &d in dims {
        nsl_list_push(list, d);
    }
    list
}

fn randn(dims: &[i64]) -> i64 {
    let s = shape(dims);
    let t = nsl_tensor_randn(s);
    nsl_list_free(s);
    t
}

/// One optimizer-free training step of `sum(relu(x @ w1) @ w2)`: record on
/// the tape, backward to both weights, release everything the step made.
/// Returns the gradient count as a cheap liveness value. The backward is the
/// train-block entry, which aborts if every gradient comes back zero — so a
/// step that recorded nothing fails the bench instead of timing a no-op (the
/// permissive `nsl_tape_backward` zero-fills and always returns one gradient
/// per parameter).
fn tape_step(x: i64, w1: i64, w2: i64, params: i64) -> i64 {
    nsl_tape_start(params);
    let h_pre = nsl_tensor_matmul(x, w1, 0);
    let h = nsl_tensor_relu(h_pre);
    let y = nsl_tensor_matmul(h, w2, 0);
    let loss = nsl_tensor_sum(y);
    let grads = nsl_tape_backward_train(loss, params);
    let n = nsl_list_len(grads);
    for i in 0..n {
        nsl_tensor_free(nsl_list_get(grads, i));
    }
    nsl_list_free(grads);
    nsl_tape_stop();
    nsl_tensor_free(loss);
    nsl_tensor_free(y);
    nsl_tensor_free(h);
    nsl_tensor_free(h_pre);
    n
}

fn bench_tape(c: &mut Criterion) {
    let mut group = c.benchmark_group("tape");

    for &(batch, dim) in &[(16i64, 64i64), (64, 256)] {
        let x = randn(&[batch, dim]);
        let w1 = randn(&[dim, dim]);
        let w2 = randn(&[dim, dim]);
        let params = nsl_list_new();
        nsl_list_push(params, w1);
        nsl_list_push(params, w2);
        assert_eq!(tape_step(x, w1, w2, params), 2, "two parameter gradients");

        group.bench_with_input(
            BenchmarkId::new("record_backward", format!("b{batch}_d{dim}")),
            &(x, w1, w2, params),
            |b, &(x, w1, w2, params)| b.iter(|| tape_step(x, w1, w2, params)),
        );

        nsl_list_free(params);
        nsl_tensor_free(w2);
        nsl_tensor_free(w1);
        nsl_tensor_free(x);
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");
    // ~1 ms and ~9 ms per call: 100 samples do not fit the default window.
    group.sample_size(50);

    for &n in &[256i64, 512] {
        let a = randn(&[n, n]);
        let b_ = randn(&[n, n]);
        // "Elements" = multiply-adds x2, i.e. the reported rate is FLOP/s.
        group.throughput(Throughput::Elements((2 * n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("square", n), &(a, b_), |b, &(a, b_)| {
            b.iter(|| {
                let c = nsl_tensor_matmul(a, b_, 0);
                nsl_tensor_free(c);
            })
        });
        nsl_tensor_free(b_);
        nsl_tensor_free(a);
    }
    group.finish();
}

fn bench_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc");

    // Small (64 B payload: the struct and shape allocations dominate) and a
    // 256 KiB payload (the zeroed data allocation dominates). No bytes/s
    // figure: a zeroed allocation this size comes back from the allocator
    // as untouched pages, so a bandwidth number would be fiction — compare
    // the round-trip time.
    for &(rows, cols) in &[(4i64, 4i64), (256, 256)] {
        let s = shape(&[rows, cols]);
        group.bench_with_input(
            BenchmarkId::new("zeros_free", format!("{rows}x{cols}")),
            &s,
            |b, &s| {
                b.iter(|| {
                    let t = nsl_tensor_zeros(s);
                    nsl_tensor_free(t);
                })
            },
        );
        nsl_list_free(s);
    }
    group.finish();
}

criterion_group!(benches, bench_tape, bench_matmul, bench_alloc);
criterion_main!(benches);
