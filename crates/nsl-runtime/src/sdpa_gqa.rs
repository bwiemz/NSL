//! Grouped-query attention for `scaled_dot_product_attention`: K and V with
//! `kv_heads` heads shared by `heads` query heads (`heads % kv_heads == 0`,
//! query head `h` reading key/value head `h / (heads / kv_heads)`).
//!
//! Two backward kernel families read grouped K/V in place and sum dK/dV over
//! each group inside the kernel: the tensor-core kernels
//! (`crate::attention_tc`) and the previous GPU backward's grouped phase-2
//! kernel (`NSL_TC_ATTENTION=0`, head dims other than 64 and 128). Every
//! other path that can serve the call computes on K/V expanded to every
//! query head, the way `nsl.nn.gqa` used to expand them in NSL source before
//! calling the attention:
//!
//! - the decomposed `matmul`/`softmax`/`matmul` chain (source AD's fallback
//!   when the fused forward declines, every CPU program, tape AD) expands
//!   through [`nsl_sdpa_expand_kv`];
//! - the previous fused forward kernel gets an expanded copy from the fused
//!   forward;
//! - the CPU reference backward (the `--deterministic` route) expands,
//!   computes per query head, and sums dK/dV over each group
//!   ([`group_sum_expanded_grad`]).
//!
//! Every expansion here is the op chain the stdlib ran
//! (`reshape([b, kv, 1, s, d]).expand([b, kv, g, s, d]).contiguous()
//! .reshape([b, h, s, d])`) and every group sum is the adjoint that chain
//! had (`reduce_to_shape` onto `[b, kv, 1, s, d]`), so on those paths a
//! program that passes grouped K/V computes exactly what it computed when the
//! stdlib expanded them, bit for bit. Every forward is bit for bit too (each
//! query head reads the same values); the in-kernel group sums round dK/dV in
//! a different order.
//!
//! `NSL_SDPA_GQA_EXPAND=1` ([`expand_mode`]) makes both in-place families
//! take the expanded route as well (expanded K/V forward, per-head dK/dV and
//! the group sum backward; the previous kernels through their expand-K/V
//! envelope): the pre-native-GQA computation in one binary, for A/B runs and
//! as an escape hatch.

use crate::tensor::NslTensor;

/// `NSL_SDPA_GQA_EXPAND=1`: the attention kernels see grouped K/V expanded to
/// every query head and dK/dV are summed over each group afterwards, as when
/// `nsl.nn.gqa` expanded them in NSL source. Read once per process. (Only
/// the tensor-core kernels read grouped K/V in place, so only CUDA builds
/// consult it.)
#[cfg(feature = "cuda")]
pub(crate) fn expand_mode() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NSL_SDPA_GQA_EXPAND").ok().as_deref() == Some("1"))
}

fn dims4(t: &NslTensor) -> [i64; 4] {
    // SAFETY: callers check `ndim == 4` first.
    unsafe {
        [
            *t.shape.add(0),
            *t.shape.add(1),
            *t.shape.add(2),
            *t.shape.add(3),
        ]
    }
}

/// The group size of a rank-4 `[batch, heads, seq, head_dim]` query and a
/// rank-4 key or value: `heads / kv_heads`, 1 without grouping. `Err` names
/// what does not line up (batch, sequence or head_dim differ, or `kv_heads`
/// does not divide `heads`).
pub(crate) fn kv_groups(q: &NslTensor, kv: &NslTensor) -> Result<usize, String> {
    if q.ndim != 4 || kv.ndim != 4 {
        return Err(format!(
            "expects rank-4 [batch, heads, seq, head_dim] operands, got Q of rank {} and K/V of rank {}",
            q.ndim, kv.ndim
        ));
    }
    let [b, h, s, d] = dims4(q);
    let [kb, kh, ks, kd] = dims4(kv);
    if kb != b || ks != s || kd != d {
        return Err(format!(
            "Q is [{b}, {h}, {s}, {d}] but K/V is [{kb}, {kh}, {ks}, {kd}]: batch, sequence and \
             head_dim must agree"
        ));
    }
    if kh <= 0 || h <= 0 || h % kh != 0 {
        return Err(format!(
            "Q has {h} heads and K/V {kh}: grouped-query attention needs the key/value head count \
             to divide the query head count"
        ));
    }
    Ok((h / kh) as usize)
}

fn shape_list(dims: &[i64]) -> i64 {
    let l = crate::list::nsl_list_new();
    for &d in dims {
        crate::list::nsl_list_push(l, d);
    }
    l
}

fn reshape(t: i64, dims: &[i64]) -> i64 {
    let l = shape_list(dims);
    let r = crate::tensor::shape_ops::nsl_tensor_reshape(t, l);
    crate::list::nsl_list_free(l);
    r
}

/// K or V `[b, kv, s, d]` expanded to `[b, kv * groups, s, d]` by the op chain
/// `nsl.nn.gqa` ran in NSL source — `reshape([b, kv, 1, s, d])`,
/// `expand([b, kv, groups, s, d])`, `contiguous()`, `reshape([b, kv * groups,
/// s, d])` — so it is recorded on a live tape exactly as that chain was, and
/// works for every device and dtype those ops serve. Returns an owned
/// contiguous tensor. The intermediates are released as transients (deferred
/// to the end of a recording tape).
pub(crate) fn expand_kv_like_stdlib(kv_ptr: i64, groups: usize) -> i64 {
    let [b, kv, s, d] = dims4(NslTensor::from_ptr_ref(kv_ptr));
    let g = groups as i64;
    let k5 = reshape(kv_ptr, &[b, kv, 1, s, d]);
    let l = shape_list(&[b, kv, g, s, d]);
    let ke = crate::tensor::shape_ops::nsl_tensor_expand(k5, l);
    crate::list::nsl_list_free(l);
    let kc = crate::tensor::shape_ops::nsl_tensor_contiguous(ke);
    let out = reshape(kc, &[b, kv * g, s, d]);
    for t in [k5, ke, kc] {
        crate::tensor::nsl_tensor_free_transient(t);
    }
    out
}

/// The adjoint of [`expand_kv_like_stdlib`]: a gradient `[b, kv * groups, s,
/// d]` for the expanded K/V summed over each group into `[b, kv, s, d]`, by
/// the op the expansion's adjoint was in source AD — `reduce_to_shape` of the
/// `[b, kv, groups, s, d]` view onto `[b, kv, 1, s, d]` (through f32 for a
/// bf16 gradient, rounded back once, as `reduce_to_shape_dt` does under bf16
/// activation storage). `kv_like` supplies the shape `[b, kv, s, d]` only.
/// Returns an owned tensor of the gradient's dtype and device.
pub(crate) fn group_sum_expanded_grad(grad_ptr: i64, kv_like: i64, groups: usize) -> i64 {
    let [b, kv, s, d] = dims4(NslTensor::from_ptr_ref(kv_like));
    let g = groups as i64;
    let grad_dtype = NslTensor::from_ptr_ref(grad_ptr).dtype;
    let view5 = reshape(grad_ptr, &[b, kv, g, s, d]);
    let target5 = reshape(kv_like, &[b, kv, 1, s, d]);
    let red = if grad_dtype == crate::tensor::DTYPE_BF16 {
        crate::storage_dt::nsl_tensor_reduce_to_shape_dt(
            view5,
            target5,
            i64::from(crate::tensor::DTYPE_BF16),
        )
    } else {
        crate::tensor::ad_ops::nsl_tensor_reduce_to_shape(view5, target5)
    };
    let out = reshape(red, &[b, kv, s, d]);
    for t in [view5, target5, red] {
        crate::tensor::nsl_tensor_free(t);
    }
    out
}

/// Refuse an attention whose operands cannot be grouped, naming the builtin.
fn refuse(which: &str, why: &str) -> ! {
    crate::fatal::die(
        crate::fatal::Fatal::ShapeMismatch,
        &format!("{which}(): {why}"),
    )
}

/// K or V of `scaled_dot_product_attention` expanded to Q's head count, for a
/// path that computes on expanded K/V (the decomposed chain, the
/// `@flash_attention` launch). Returns 0 when there is nothing to expand —
/// K/V already has Q's head count, or an operand is not rank 4 (attention
/// without a head axis) — and the caller uses `kv_ptr` as it is; otherwise an
/// owned tensor the caller releases ([`expand_kv_like_stdlib`]). Grouped K/V
/// whose shape cannot be served (batch, sequence or head_dim differ from Q's,
/// or `kv_heads` does not divide `heads`) stops the program.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sdpa_expand_kv(kv_ptr: i64, q_ptr: i64) -> i64 {
    if kv_ptr == 0 || q_ptr == 0 {
        return 0;
    }
    let q = NslTensor::from_ptr_ref(q_ptr);
    let kv = NslTensor::from_ptr_ref(kv_ptr);
    if q.ndim != 4 || kv.ndim != 4 {
        return 0;
    }
    if dims4(kv)[1] == dims4(q)[1] {
        return 0;
    }
    match kv_groups(q, kv) {
        Ok(groups) => expand_kv_like_stdlib(kv_ptr, groups),
        Err(why) => refuse("scaled_dot_product_attention", &why),
    }
}

/// The packed and masked attention builtins read K and V with Q's head
/// count; grouped K/V stops the program here with a message, rather than as
/// a batch-dimension mismatch inside a matmul. `which`: 0 =
/// `scaled_dot_product_attention_packed`, 1 =
/// `scaled_dot_product_attention_masked`. Rank-4 operands only; other ranks
/// pass through to the ops that check them.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sdpa_require_equal_kv_heads(q_ptr: i64, k_ptr: i64, v_ptr: i64, which: i64) {
    let name = if which == 0 {
        "scaled_dot_product_attention_packed"
    } else {
        "scaled_dot_product_attention_masked"
    };
    if q_ptr == 0 || k_ptr == 0 || v_ptr == 0 {
        return;
    }
    let q = NslTensor::from_ptr_ref(q_ptr);
    for kv_ptr in [k_ptr, v_ptr] {
        let kv = NslTensor::from_ptr_ref(kv_ptr);
        if q.ndim != 4 || kv.ndim != 4 {
            continue;
        }
        let (h, kh) = (dims4(q)[1], dims4(kv)[1]);
        if h != kh {
            refuse(
                name,
                &format!(
                    "Q has {h} heads and K/V {kh}. This builtin does not take grouped K/V: expand \
                     them to every query head first (`k.reshape([b, kv, 1, s, d]).expand([b, kv, \
                     g, s, d]).contiguous().reshape([b, h, s, d])`), or use \
                     scaled_dot_product_attention, which does"
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::NslTensor;

    /// A host f32 `[shape]` tensor holding `f(i)` at flat index `i`.
    fn host(shape: &[i64], f: impl Fn(usize) -> f32) -> i64 {
        let n: i64 = shape.iter().product();
        let l = shape_list(shape);
        let t = crate::tensor::creation::nsl_tensor_zeros(l);
        crate::list::nsl_list_free(l);
        let tt = NslTensor::from_ptr(t);
        assert_eq!(tt.len, n);
        for i in 0..n as usize {
            match tt.dtype {
                1 => unsafe { *tt.data_f32().add(i) = f(i) },
                _ => unsafe { *tt.data_f64().add(i) = f(i) as f64 },
            }
        }
        t
    }

    fn read(t: i64) -> Vec<f64> {
        let c = crate::tensor::shape_ops::nsl_tensor_contiguous(t);
        let tt = NslTensor::from_ptr(c);
        let v = (0..tt.len as usize)
            .map(|i| match tt.dtype {
                1 => unsafe { *tt.data_f32().add(i) as f64 },
                _ => unsafe { *tt.data_f64().add(i) },
            })
            .collect();
        crate::tensor::nsl_tensor_free(c);
        v
    }

    fn shape(t: i64) -> Vec<i64> {
        let tt = NslTensor::from_ptr_ref(t);
        (0..tt.ndim as usize)
            .map(|i| unsafe { *tt.shape.add(i) })
            .collect()
    }

    #[test]
    fn groups_follow_the_head_counts() {
        let q = host(&[2, 8, 3, 4], |_| 0.0);
        let k = host(&[2, 2, 3, 4], |_| 0.0);
        let kq = host(&[2, 8, 3, 4], |_| 0.0);
        assert_eq!(
            kv_groups(NslTensor::from_ptr_ref(q), NslTensor::from_ptr_ref(k)),
            Ok(4)
        );
        assert_eq!(
            kv_groups(NslTensor::from_ptr_ref(q), NslTensor::from_ptr_ref(kq)),
            Ok(1)
        );
        let k3 = host(&[2, 3, 3, 4], |_| 0.0);
        let e = kv_groups(NslTensor::from_ptr_ref(q), NslTensor::from_ptr_ref(k3)).unwrap_err();
        assert!(e.contains("divide"), "{e}");
        let ks = host(&[2, 2, 5, 4], |_| 0.0);
        let e = kv_groups(NslTensor::from_ptr_ref(q), NslTensor::from_ptr_ref(ks)).unwrap_err();
        assert!(e.contains("must agree"), "{e}");
        for t in [q, k, kq, k3, ks] {
            crate::tensor::nsl_tensor_free(t);
        }
    }

    #[test]
    fn expand_repeats_each_kv_head_for_its_group() {
        // q head h reads kv head h / g: destination head kv*g + gi is a copy
        // of source head kv (the stdlib's reshape/expand mapping).
        let (b, kvh, g, s, d) = (2usize, 3usize, 2usize, 3usize, 4usize);
        let k = host(&[b as i64, kvh as i64, s as i64, d as i64], |i| {
            i as f32 * 0.5 - 7.0
        });
        let q = host(&[b as i64, (kvh * g) as i64, s as i64, d as i64], |_| 0.0);
        let e = nsl_sdpa_expand_kv(k, q);
        assert_ne!(e, 0);
        assert_eq!(
            shape(e),
            vec![b as i64, (kvh * g) as i64, s as i64, d as i64]
        );
        assert!(NslTensor::from_ptr_ref(e).is_contiguous());
        let (kv_vals, e_vals) = (read(k), read(e));
        for bi in 0..b {
            for h in 0..kvh * g {
                for r in 0..s * d {
                    let src = (bi * kvh + h / g) * s * d + r;
                    let dst = (bi * kvh * g + h) * s * d + r;
                    assert_eq!(e_vals[dst], kv_vals[src], "b {bi} head {h} elem {r}");
                }
            }
        }
        // Equal head counts and non-rank-4 operands: nothing to expand.
        assert_eq!(nsl_sdpa_expand_kv(q, q), 0);
        let k3 = host(&[b as i64, s as i64, d as i64], |_| 0.0);
        assert_eq!(nsl_sdpa_expand_kv(k3, q), 0);
        for t in [k, q, e, k3] {
            crate::tensor::nsl_tensor_free(t);
        }
    }

    #[test]
    fn group_sum_adds_each_group_into_its_kv_head() {
        let (b, kvh, g, s, d) = (2usize, 2usize, 3usize, 2usize, 3usize);
        let h = kvh * g;
        let grad = host(&[b as i64, h as i64, s as i64, d as i64], |i| {
            ((i * 37) % 11) as f32 - 5.25
        });
        let kv = host(&[b as i64, kvh as i64, s as i64, d as i64], |_| 0.0);
        let out = group_sum_expanded_grad(grad, kv, g);
        assert_eq!(shape(out), vec![b as i64, kvh as i64, s as i64, d as i64]);
        let (gv, ov) = (read(grad), read(out));
        for bi in 0..b {
            for k in 0..kvh {
                for r in 0..s * d {
                    let want: f64 = (0..g).map(|gi| gv[(bi * h + k * g + gi) * s * d + r]).sum();
                    assert_eq!(ov[(bi * kvh + k) * s * d + r], want);
                }
            }
        }
        for t in [grad, kv, out] {
            crate::tensor::nsl_tensor_free(t);
        }
    }

    #[test]
    fn equal_heads_pass_the_packed_and_masked_check() {
        let q = host(&[1, 4, 2, 2], |_| 0.0);
        nsl_sdpa_require_equal_kv_heads(q, q, q, 0);
        nsl_sdpa_require_equal_kv_heads(q, q, q, 1);
        crate::tensor::nsl_tensor_free(q);
    }
}
