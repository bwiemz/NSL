//! Deterministic fuzz harness for memory lifecycle testing.
//! Generates random sequences of FFI tensor operations and asserts
//! that allocation counters balance after each sequence.
//!
//! Run: `cargo test -p nsl-runtime fuzz`

#[cfg(test)]
use crate::memory::stats;

/// Fuzzable tensor operations.
#[cfg(test)]
#[derive(Debug, Clone)]
enum FuzzOp {
    // Creation
    Zeros { dims: Vec<i64> },
    Ones { dims: Vec<i64> },
    Rand { dims: Vec<i64> },
    Randn { dims: Vec<i64> },
    Arange { start: f64, stop: f64, step: f64 },

    // Unary (index into live pool)
    Clone { idx: usize },
    Reshape { idx: usize, new_dims: Vec<i64> },
    Transpose { idx: usize, dim0: i64, dim1: i64 },
    Free { idx: usize },

    // Binary (first operand from pool, second created fresh with matching shape)
    Add { idx: usize },
    Mul { idx: usize },
    MatMul { idx: usize, n: i64 },
    ScalarMul { idx: usize, scalar: f64 },

    // Reductions
    Sum { idx: usize },
    Mean { idx: usize },

    // Activations
    ReLU { idx: usize },
    Sigmoid { idx: usize },
    Tanh { idx: usize },

    // Tape operations
    TapeStart,
    TapeBackwardAndStop,
    TapeStop,
}

/// Minimal xorshift64 PRNG (no external deps).
#[cfg(test)]
struct Xorshift64 {
    state: u64,
}

#[cfg(test)]
impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_bounded(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Fuzz session state.
#[cfg(test)]
struct FuzzState {
    rng: Xorshift64,
    live: Vec<i64>,
    shapes: Vec<Vec<i64>>,
    tape_active: bool,
    tape_params: Vec<i64>,
}

#[cfg(test)]
impl FuzzState {
    fn new(seed: u64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            live: Vec::new(),
            shapes: Vec::new(),
            tape_active: false,
            tape_params: Vec::new(),
        }
    }

    fn random_shape(&mut self) -> Vec<i64> {
        let ndim = self.rng.next_bounded(3) as usize + 1;
        (0..ndim)
            .map(|_| self.rng.next_bounded(32) as i64 + 1)
            .collect()
    }

    fn shape_len(shape: &[i64]) -> i64 {
        shape.iter().product()
    }

    fn pick_live(&mut self) -> Option<usize> {
        if self.live.is_empty() {
            None
        } else {
            Some(self.rng.next_bounded(self.live.len() as u64) as usize)
        }
    }

    fn generate_op(&mut self) -> FuzzOp {
        if self.live.is_empty() {
            return self.random_creation_op();
        }

        if self.tape_active {
            let r = self.rng.next_bounded(100);
            if r < 15 {
                return FuzzOp::TapeStop;
            } else if r < 25 {
                return FuzzOp::TapeBackwardAndStop;
            }
        } else if self.live.len() >= 2 && self.rng.next_bounded(100) < 10 {
            return FuzzOp::TapeStart;
        }

        let r = self.rng.next_bounded(100);
        match r {
            0..=19 => self.random_creation_op(),
            20..=24 => {
                let idx = self.pick_live().unwrap();
                if self.tape_active && self.tape_params.contains(&self.live[idx]) {
                    FuzzOp::Clone { idx }
                } else {
                    FuzzOp::Free { idx }
                }
            }
            25..=29 => FuzzOp::Clone { idx: self.pick_live().unwrap() },
            30..=34 => {
                let idx = self.pick_live().unwrap();
                let shape = &self.shapes[idx];
                let total = Self::shape_len(shape);
                let new_dims = self.random_reshape_target(total);
                FuzzOp::Reshape { idx, new_dims }
            }
            35..=39 => {
                let idx = self.pick_live().unwrap();
                let ndim = self.shapes[idx].len() as i64;
                if ndim >= 2 {
                    FuzzOp::Transpose { idx, dim0: 0, dim1: ndim - 1 }
                } else {
                    FuzzOp::Clone { idx }
                }
            }
            40..=49 => FuzzOp::Add { idx: self.pick_live().unwrap() },
            50..=59 => FuzzOp::Mul { idx: self.pick_live().unwrap() },
            60..=64 => {
                let idx = self.pick_live().unwrap();
                let shape = &self.shapes[idx];
                if shape.len() >= 2 {
                    let n = self.rng.next_bounded(16) as i64 + 1;
                    FuzzOp::MatMul { idx, n }
                } else {
                    FuzzOp::ScalarMul { idx, scalar: self.rng.next_f64() * 10.0 }
                }
            }
            65..=69 => FuzzOp::ScalarMul {
                idx: self.pick_live().unwrap(),
                scalar: self.rng.next_f64() * 10.0 - 5.0,
            },
            70..=74 => FuzzOp::Sum { idx: self.pick_live().unwrap() },
            75..=79 => FuzzOp::Mean { idx: self.pick_live().unwrap() },
            80..=86 => FuzzOp::ReLU { idx: self.pick_live().unwrap() },
            87..=93 => FuzzOp::Sigmoid { idx: self.pick_live().unwrap() },
            _ => FuzzOp::Tanh { idx: self.pick_live().unwrap() },
        }
    }

    fn random_creation_op(&mut self) -> FuzzOp {
        let dims = self.random_shape();
        match self.rng.next_bounded(5) {
            0 => FuzzOp::Zeros { dims },
            1 => FuzzOp::Ones { dims },
            2 => FuzzOp::Rand { dims },
            3 => FuzzOp::Randn { dims },
            _ => {
                let stop = self.rng.next_bounded(64) as f64 + 1.0;
                FuzzOp::Arange { start: 0.0, stop, step: 1.0 }
            }
        }
    }

    fn random_reshape_target(&mut self, total: i64) -> Vec<i64> {
        let ndim = self.rng.next_bounded(3) as usize + 1;
        if ndim == 1 {
            return vec![total];
        }
        let mut dims = Vec::new();
        let mut remaining = total;
        for _ in 0..ndim - 1 {
            let max_dim = remaining.min(32);
            if max_dim <= 1 {
                break;
            }
            let mut found = false;
            for _ in 0..10 {
                let d = self.rng.next_bounded(max_dim as u64) as i64 + 1;
                if remaining % d == 0 {
                    dims.push(d);
                    remaining /= d;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        dims.push(remaining);
        dims
    }

    fn make_shape_list(dims: &[i64]) -> i64 {
        let list = crate::list::nsl_list_new();
        for &d in dims {
            crate::list::nsl_list_push(list, d);
        }
        list
    }

    fn register(&mut self, ptr: i64, shape: Vec<i64>) {
        self.live.push(ptr);
        self.shapes.push(shape);
    }

    fn unregister(&mut self, idx: usize) -> i64 {
        self.shapes.swap_remove(idx);
        self.live.swap_remove(idx)
    }

    fn execute_op(&mut self, op: FuzzOp) {
        use crate::tensor::*;
        use crate::list::*;

        match op {
            FuzzOp::Zeros { dims } => {
                let shape = Self::make_shape_list(&dims);
                let t = nsl_tensor_zeros(shape);
                nsl_list_free(shape);
                self.register(t, dims);
            }
            FuzzOp::Ones { dims } => {
                let shape = Self::make_shape_list(&dims);
                let t = nsl_tensor_ones(shape);
                nsl_list_free(shape);
                self.register(t, dims);
            }
            FuzzOp::Rand { dims } => {
                let shape = Self::make_shape_list(&dims);
                let t = nsl_tensor_rand(shape);
                nsl_list_free(shape);
                self.register(t, dims);
            }
            FuzzOp::Randn { dims } => {
                let shape = Self::make_shape_list(&dims);
                let t = nsl_tensor_randn(shape);
                nsl_list_free(shape);
                self.register(t, dims);
            }
            FuzzOp::Arange { start, stop, step } => {
                let t = nsl_tensor_arange(start, stop, step);
                let len = ((stop - start) / step).ceil().max(0.0) as i64;
                self.register(t, vec![len]);
            }

            FuzzOp::Clone { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let t = nsl_tensor_clone(ptr);
                self.register(t, shape);
            }
            FuzzOp::Reshape { idx, new_dims } => {
                let ptr = self.live[idx];
                let shape = Self::make_shape_list(&new_dims);
                let t = nsl_tensor_reshape(ptr, shape);
                nsl_list_free(shape);
                self.register(t, new_dims);
            }
            FuzzOp::Transpose { idx, dim0, dim1 } => {
                let ptr = self.live[idx];
                let mut shape = self.shapes[idx].clone();
                let t = nsl_tensor_transpose(ptr, dim0, dim1);
                shape.swap(dim0 as usize, dim1 as usize);
                self.register(t, shape);
            }
            FuzzOp::Free { idx } => {
                let ptr = self.unregister(idx);
                nsl_tensor_free(ptr);
            }

            FuzzOp::Add { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let shape_list = Self::make_shape_list(&shape);
                let b = nsl_tensor_zeros(shape_list);
                nsl_list_free(shape_list);
                let result = nsl_tensor_add(ptr, b);
                nsl_tensor_free(b);
                self.register(result, shape);
            }
            FuzzOp::Mul { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let shape_list = Self::make_shape_list(&shape);
                let b = nsl_tensor_ones(shape_list);
                nsl_list_free(shape_list);
                let result = nsl_tensor_mul(ptr, b);
                nsl_tensor_free(b);
                self.register(result, shape);
            }
            FuzzOp::MatMul { idx, n } => {
                let ptr = self.live[idx];
                let shape = &self.shapes[idx];
                let k = *shape.last().unwrap();
                let rhs_shape = vec![k, n];
                let rhs_list = Self::make_shape_list(&rhs_shape);
                let rhs = nsl_tensor_ones(rhs_list);
                nsl_list_free(rhs_list);
                let result = nsl_tensor_matmul(ptr, rhs);
                nsl_tensor_free(rhs);
                let mut out_shape = shape[..shape.len() - 1].to_vec();
                out_shape.push(n);
                self.register(result, out_shape);
            }
            FuzzOp::ScalarMul { idx, scalar } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let result = nsl_tensor_mul_scalar(ptr, scalar);
                self.register(result, shape);
            }

            FuzzOp::Sum { idx } => {
                let ptr = self.live[idx];
                let result = nsl_tensor_sum(ptr);
                self.register(result, vec![1]);
            }
            FuzzOp::Mean { idx } => {
                let ptr = self.live[idx];
                let result = nsl_tensor_mean(ptr);
                self.register(result, vec![1]);
            }

            FuzzOp::ReLU { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let result = nsl_tensor_relu(ptr);
                self.register(result, shape);
            }
            FuzzOp::Sigmoid { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let result = nsl_tensor_sigmoid(ptr);
                self.register(result, shape);
            }
            FuzzOp::Tanh { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let result = nsl_tensor_tanh_act(ptr);
                self.register(result, shape);
            }

            FuzzOp::TapeStart => {
                let param_list = nsl_list_new();
                self.tape_params.clear();
                for &ptr in &self.live {
                    if self.rng.next_bounded(2) == 0 {
                        nsl_list_push(param_list, ptr);
                        self.tape_params.push(ptr);
                    }
                }
                if self.tape_params.is_empty() && !self.live.is_empty() {
                    let ptr = self.live[0];
                    nsl_list_push(param_list, ptr);
                    self.tape_params.push(ptr);
                }
                crate::autodiff::nsl_tape_start(param_list);
                nsl_list_free(param_list);
                self.tape_active = true;
            }
            FuzzOp::TapeBackwardAndStop => {
                if !self.tape_active || self.live.is_empty() || self.tape_params.is_empty() {
                    return;
                }
                let idx = self.rng.next_bounded(self.live.len() as u64) as usize;
                let loss_input = self.live[idx];
                let loss = nsl_tensor_sum(loss_input);

                let param_list = nsl_list_new();
                for &p in &self.tape_params {
                    nsl_list_push(param_list, p);
                }

                let grad_list = crate::autodiff::nsl_tape_backward(loss, param_list);

                let grad_list_ref = crate::list::NslList::from_ptr(grad_list);
                for i in 0..grad_list_ref.len as usize {
                    let grad_ptr = unsafe { *grad_list_ref.data.add(i) };
                    nsl_tensor_free(grad_ptr);
                }
                nsl_list_free(grad_list);
                nsl_list_free(param_list);
                nsl_tensor_free(loss);

                crate::autodiff::nsl_tape_stop();
                self.tape_active = false;
                self.tape_params.clear();
            }
            FuzzOp::TapeStop => {
                if !self.tape_active {
                    return;
                }
                crate::autodiff::nsl_tape_stop();
                self.tape_active = false;
                self.tape_params.clear();
            }
        }
    }

    fn teardown(&mut self) {
        use crate::tensor::nsl_tensor_free;

        if self.tape_active {
            crate::autodiff::nsl_tape_stop();
            self.tape_active = false;
            self.tape_params.clear();
        }

        let ptrs: Vec<i64> = self.live.drain(..).collect();
        self.shapes.clear();
        for ptr in ptrs {
            nsl_tensor_free(ptr);
        }
    }
}

/// Assert that allocation counters balance (allocs == frees).
/// Thread-local counters mean no interference from parallel tests.
#[cfg(test)]
fn assert_counter_balance(seed: u64) {
    let ac = stats::alloc_count();
    let fc = stats::free_count();
    let ab = stats::alloc_bytes();
    let fb = stats::free_bytes();

    if ac != fc || ab != fb {
        panic!(
            "FUZZ FAILURE at seed={}\n\
             CPU: allocated {} ({} bytes), freed {} ({} bytes)\n\
             Leaked: {} tensors, {} bytes\n\
             Replay: cargo test -p nsl-runtime fuzz_memory_lifecycle -- (with seed {})",
            seed,
            ac, ab, fc, fb,
            ac as isize - fc as isize,
            ab as isize - fb as isize,
            seed
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::stats;

    #[test]
    fn test_stats_counter_reset() {
        stats::reset();
        assert_eq!(stats::alloc_count(), 0);
        assert_eq!(stats::free_count(), 0);
        assert_eq!(stats::alloc_bytes(), 0);
        assert_eq!(stats::free_bytes(), 0);
        assert_eq!(stats::cuda_alloc_count(), 0);
        assert_eq!(stats::cuda_free_count(), 0);
        assert_eq!(stats::cuda_alloc_bytes(), 0);
        assert_eq!(stats::cuda_free_bytes(), 0);
    }

    #[test]
    fn test_stats_track_alloc_free() {
        stats::reset();

        let ptr = crate::memory::checked_alloc(256);
        assert_eq!(stats::alloc_count(), 1);
        assert_eq!(stats::alloc_bytes(), 256);
        assert_eq!(stats::free_count(), 0);

        unsafe { crate::memory::checked_free(ptr, 256); }
        assert_eq!(stats::free_count(), 1);
        assert_eq!(stats::free_bytes(), 256);
    }

    #[test]
    fn test_tensor_lifecycle_counter_balance() {
        use crate::tensor::{nsl_tensor_zeros, nsl_tensor_add, nsl_tensor_free};
        use crate::list::{nsl_list_new, nsl_list_push, nsl_list_free};

        stats::reset();

        let shape = nsl_list_new();
        nsl_list_push(shape, 4);
        nsl_list_push(shape, 4);
        let a = nsl_tensor_zeros(shape);

        let shape2 = nsl_list_new();
        nsl_list_push(shape2, 4);
        nsl_list_push(shape2, 4);
        let b = nsl_tensor_zeros(shape2);

        let c = nsl_tensor_add(a, b);

        nsl_tensor_free(a);
        nsl_tensor_free(b);
        nsl_tensor_free(c);
        nsl_list_free(shape);
        nsl_list_free(shape2);

        super::assert_counter_balance(0);
    }

    #[test]
    fn test_fuzz_generator_produces_ops() {
        let mut state = super::FuzzState::new(42);
        let mut ops = Vec::new();
        for _ in 0..100 {
            ops.push(state.generate_op());
        }
        let creation_count = ops.iter().filter(|op| matches!(op,
            super::FuzzOp::Zeros { .. } | super::FuzzOp::Ones { .. } |
            super::FuzzOp::Rand { .. } | super::FuzzOp::Randn { .. } |
            super::FuzzOp::Arange { .. }
        )).count();
        assert!(creation_count > 0, "should generate at least one creation op");
        assert!(ops.len() == 100, "should generate exactly 100 ops");
    }

    #[test]
    fn fuzz_memory_lifecycle() {
        let num_seeds: u64 = 100;
        let ops_per_seed = 150;

        for seed in 1..=num_seeds {
            stats::reset();
            let mut state = super::FuzzState::new(seed);

            for _ in 0..ops_per_seed {
                let op = state.generate_op();
                state.execute_op(op);
            }

            state.teardown();
            super::assert_counter_balance(seed);
        }
    }

    #[test]
    fn fuzz_tape_stress() {
        let num_seeds: u64 = 50;

        for seed in 1000..1000 + num_seeds {
            stats::reset();
            let mut state = super::FuzzState::new(seed);

            // Create a pool of tensors first
            for _ in 0..10 {
                let dims = state.random_shape();
                let op = super::FuzzOp::Rand { dims };
                state.execute_op(op);
            }

            // Alternate: tape_start → some math → backward+stop OR tape_stop
            for cycle in 0..20u64 {
                state.execute_op(super::FuzzOp::TapeStart);

                let n_ops = state.rng.next_bounded(11) as usize + 5;
                for _ in 0..n_ops {
                    let op = state.generate_op();
                    match &op {
                        // Skip tape control ops — we manage the lifecycle explicitly
                        super::FuzzOp::TapeStart |
                        super::FuzzOp::TapeStop |
                        super::FuzzOp::TapeBackwardAndStop => continue,
                        // Skip shape-altering ops — backward can't handle shape mismatches
                        super::FuzzOp::Reshape { .. } |
                        super::FuzzOp::Transpose { .. } |
                        super::FuzzOp::MatMul { .. } |
                        super::FuzzOp::Free { .. } => continue,
                        _ => state.execute_op(op),
                    }
                }

                if cycle % 10 < 7 {
                    state.execute_op(super::FuzzOp::TapeBackwardAndStop);
                } else {
                    state.execute_op(super::FuzzOp::TapeStop);
                }
            }

            state.teardown();
            super::assert_counter_balance(seed);
        }
    }
}
