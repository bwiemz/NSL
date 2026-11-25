//! Silicon gate for the MXFP8 building blocks (`nsl_runtime::mxfp8`).
//!
//! 1. The quantizers write the host reference's bytes exactly: every data
//!    byte and every byte of the scale buffer (padding included, over a
//!    buffer pre-filled with garbage), f32 and bf16 sources, row-wise and
//!    transposed, at ragged and production shapes, with zero, subnormal,
//!    saturating-edge and non-finite blocks.
//! 2. The cublasLt MXFP8 GEMM is the f64 product of the dequantized
//!    operands (to f32 accumulation error), with per-block scales spread
//!    over 2^±20 so a scale applied to the wrong block cannot pass:
//!    bf16 and f32 outputs, `beta = 1` accumulation into f32, a K whose
//!    scale rows are padded, and the weight-gradient arrangement built from
//!    two transposed quantizations.
//! 3. Each block's scale lands on its own block, checked without the
//!    reference's layout function: one operand selects a single K block per
//!    output (one-hot), the other has a distinct power-of-two scale per
//!    block and values E4M3 holds exactly, so every output is an exact sum
//!    of the unquantized inputs and a misplaced scale is off by a power of
//!    two. Both operands, a padded scale row, and a NaN block, whose row
//!    turns NaN while the others stay finite.
//! 4. One production shape (the 1B gate+up weight gradient) against sampled
//!    reference elements.
//!
//!   cargo test -p nsl-runtime --features cuda --test mx_quant_gpu \
//!       -- --include-ignored --test-threads=1
#![cfg(feature = "cuda")]

use nsl_runtime::mxfp8::{MX_BLOCK, gemm, quant_cols_t, quant_rows, reference, scale_bytes, stats};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
};

/// The GEMM counters are process-global; one lock keeps this file's gates apart.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Largest `|err| / scale` seen, printed by each GEMM gate: the FP8 tensor
/// cores' accumulation precision, measured rather than assumed.
static WORST: std::sync::Mutex<f64> = std::sync::Mutex::new(0.0);

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    if nsl_cuda_init() != 0 {
        return false;
    }
    if !nsl_runtime::mxfp8::supported() {
        eprintln!("[mx_quant_gpu] skipped: MXFP8 needs compute capability 10.0 or newer");
        return false;
    }
    true
}

/// A device buffer freed on drop.
struct Dev(i64, usize);

impl Dev {
    fn new(bytes: usize) -> Dev {
        let p = nsl_test_cuda_alloc(bytes.max(16) as i64);
        assert_ne!(p, 0, "device alloc of {bytes} bytes");
        Dev(p, bytes)
    }
    fn from_bytes(b: &[u8]) -> Dev {
        let d = Dev::new(b.len());
        nsl_test_cuda_h2d(d.0, b.as_ptr() as i64, b.len() as i64);
        d
    }
    fn fill(&self, byte: u8) {
        let v = vec![byte; self.1];
        nsl_test_cuda_h2d(self.0, v.as_ptr() as i64, self.1 as i64);
    }
    fn read(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.1];
        nsl_test_cuda_d2h(v.as_mut_ptr() as i64, self.0, self.1 as i64);
        v
    }
    fn ptr(&self) -> u64 {
        self.0 as u64
    }
}

impl Drop for Dev {
    fn drop(&mut self) {
        nsl_test_cuda_free(self.0);
    }
}

/// Small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn to_bf16(x: f32) -> u16 {
    // Round to nearest even (no NaNs reach here except as written below).
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    (r >> 16) as u16
}

fn bf16_value(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// Values for a `[rows, cols]` tensor whose 32-element blocks along `along`
/// (`true`: along rows' contiguous dim; `false`: down columns) each get
/// their own magnitude `2^e`, `e` in [-20, 20], plus some special blocks.
fn spread(rows: usize, cols: usize, along_rows: bool, rng: &mut Rng, specials: bool) -> Vec<f32> {
    let mut x = vec![0f32; rows * cols];
    let (outer, inner) = if along_rows {
        (rows, cols)
    } else {
        (cols, rows)
    };
    for o in 0..outer {
        for blk in 0..inner / MX_BLOCK {
            let e = rng.below(41) as i32 - 20;
            let kind = if specials { rng.below(16) } else { 99 };
            for j in 0..MX_BLOCK {
                let i = blk * MX_BLOCK + j;
                let v = match kind {
                    0 => 0.0,
                    1 => rng.unit() * 1e-40, // f32 subnormals
                    2 => {
                        if j == 3 {
                            448.0
                        } else {
                            rng.unit() * 448.0
                        }
                    } // the saturation edge
                    3 => {
                        if j == 5 {
                            1.75 * 2f32.powi(e)
                        } else {
                            rng.unit() * 2f32.powi(e)
                        }
                    } // scale boundary
                    _ => rng.unit() * 2f32.powi(e),
                };
                let at = if along_rows {
                    o * cols + i
                } else {
                    i * cols + o
                };
                x[at] = v;
            }
        }
    }
    x
}

/// Upload `x` as f32, or rounded to bf16; returns the device buffer and the
/// values the kernel will see.
fn upload(x: &[f32], bf16: bool) -> (Dev, Vec<f32>) {
    if bf16 {
        let h: Vec<u16> = x.iter().map(|&v| to_bf16(v)).collect();
        let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
        (
            Dev::from_bytes(&bytes),
            h.iter().map(|&v| bf16_value(v)).collect(),
        )
    } else {
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        (Dev::from_bytes(&bytes), x.to_vec())
    }
}

/// Compare quantized bytes with the reference. Blocks whose scale is the NaN
/// byte compare only their scale (the data under a NaN scale is unread).
fn assert_same(tag: &str, got: (&[u8], &[u8]), want: (&[u8], &[u8]), outer: usize, k: usize) {
    assert_eq!(got.1, want.1, "{tag}: scale buffers differ");
    let sf = nsl_runtime::mxfp8::sf_inner(k);
    for o in 0..outer {
        for blk in 0..k / MX_BLOCK {
            if want.1[nsl_runtime::mxfp8::scale_offset(o, blk, sf)] == 0xff {
                continue;
            }
            let at = o * k + blk * MX_BLOCK;
            assert_eq!(
                &got.0[at..at + MX_BLOCK],
                &want.0[at..at + MX_BLOCK],
                "{tag}: data of row {o}, block {blk} differ"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_row_quantizer_writes_the_reference_bytes() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_available() {
        return;
    }
    let mut rng = Rng(0x5eed_0001);
    for (rows, k) in [(1, 32), (3, 64), (130, 96), (256, 2048), (64, 3520)] {
        for bf16 in [false, true] {
            let mut x = spread(rows, k, true, &mut rng, true);
            if rows >= 3 {
                // One NaN block and one Inf block.
                x[k + 7] = f32::NAN;
                x[2 * k + 33] = f32::NEG_INFINITY;
            }
            let (src, seen) = upload(&x, bf16);
            let data = Dev::new(rows * k);
            let scales = Dev::new(scale_bytes(rows, k));
            scales.fill(0xab);
            quant_rows(src.ptr(), bf16, rows, k, data.ptr(), scales.ptr());
            let (want_d, want_s) = reference::quant_rows(&seen, rows, k);
            let tag = format!("rows [{rows}, {k}] bf16={bf16}");
            assert_same(
                &tag,
                (&data.read(), &scales.read()),
                (&want_d, &want_s),
                rows,
                k,
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_transposing_quantizer_writes_the_reference_bytes() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_available() {
        return;
    }
    let mut rng = Rng(0x5eed_0002);
    for (r, c) in [(32, 32), (64, 96), (2048, 160), (4096, 2048)] {
        for bf16 in [false, true] {
            let mut x = spread(r, c, false, &mut rng, true);
            x[5 * c + 1] = f32::NAN;
            let (src, seen) = upload(&x, bf16);
            let data = Dev::new(r * c);
            let scales = Dev::new(scale_bytes(c, r));
            scales.fill(0xab);
            quant_cols_t(src.ptr(), bf16, r, c, data.ptr(), scales.ptr());
            let (want_d, want_s) = reference::quant_cols_t(&seen, r, c);
            let tag = format!("cols_t [{r}, {c}] bf16={bf16}");
            assert_same(
                &tag,
                (&data.read(), &scales.read()),
                (&want_d, &want_s),
                c,
                r,
            );
        }
    }
}

/// A quantized K-major operand: device data and scales, and their host copies.
struct Op {
    data: Dev,
    scales: Dev,
    hd: Vec<u8>,
    hs: Vec<u8>,
    k: usize,
}

impl Op {
    fn rows(x: &[f32], outer: usize, k: usize) -> Op {
        let (src, _) = upload(x, false);
        let data = Dev::new(outer * k);
        let scales = Dev::new(scale_bytes(outer, k));
        quant_rows(src.ptr(), false, outer, k, data.ptr(), scales.ptr());
        let (hd, hs) = (data.read(), scales.read());
        Op {
            data,
            scales,
            hd,
            hs,
            k,
        }
    }
    /// From row-major `[r, c]`, K = r: the operand has `c` rows.
    fn cols_t(x: &[f32], r: usize, c: usize, bf16: bool) -> Op {
        let (src, _) = upload(x, bf16);
        let data = Dev::new(r * c);
        let scales = Dev::new(scale_bytes(c, r));
        quant_cols_t(src.ptr(), bf16, r, c, data.ptr(), scales.ptr());
        let (hd, hs) = (data.read(), scales.read());
        Op {
            data,
            scales,
            hd,
            hs,
            k: r,
        }
    }
    fn deq(&self, o: usize, kk: usize) -> f64 {
        reference::dequant(&self.hd, &self.hs, self.k, o, kk)
    }
}

/// `sum_k a(i, k) · b(j, k)` in f64, and `sum_k |a · b|` (the error scale).
fn dot(a: &Op, b: &Op, i: usize, j: usize) -> (f64, f64) {
    let (mut s, mut mag) = (0f64, 0f64);
    for kk in 0..a.k {
        let p = a.deq(i, kk) * b.deq(j, kk);
        s += p;
        mag += p.abs();
    }
    (s, mag)
}

/// Tolerance of an f32-accumulated product, relative to `sum |a·b|`. A scale
/// applied to the wrong block is off by factors of up to 2^40 here.
const ACC_REL: f64 = 1e-3;

/// Print and reset the largest f32-accumulation error this gate saw.
fn report(tag: &str) {
    let mut w = WORST.lock().unwrap();
    eprintln!("[mx_quant_gpu] {tag}: worst |err| / sum|a*b| = {:e}", *w);
    *w = 0.0;
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn check(tag: &str, got: f64, want: f64, mag: f64, rel: f64) {
    let err = (got - want).abs();
    // Only the f32-accumulation checks: the bf16 output's rounding is its own.
    if mag > 0.0 && rel == ACC_REL {
        let mut w = WORST.lock().unwrap();
        *w = w.max(err / mag);
    }
    assert!(
        err <= rel * mag + 1e-30,
        "{tag}: got {got}, want {want} (|err| {err:e}, scale {mag:e})"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_gemm_is_the_product_of_the_dequantized_operands() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_available() {
        return;
    }
    let mut rng = Rng(0x5eed_0003);
    // K = 320: ten scale blocks, padded to twelve per row in the tile layout.
    let (m, n, k) = (256, 192, 320);
    let a = Op::rows(&spread(m, k, true, &mut rng, false), m, k);
    let b = Op::rows(&spread(n, k, true, &mut rng, false), n, k);
    let (issued0, _, declined0) = stats();

    // f32 output, beta = 0, then beta = 1 onto a known C.
    let c = Dev::new(m * n * 4);
    c.fill(0);
    assert!(unsafe {
        gemm(
            m,
            n,
            k,
            1.0,
            a.data.ptr(),
            a.scales.ptr(),
            b.data.ptr(),
            b.scales.ptr(),
            0.0,
            c.ptr(),
            m,
            false,
        )
    });
    let d0 = f32s(&c.read());
    let mut refs = vec![(0f64, 0f64); m * n];
    for j in 0..n {
        for i in 0..m {
            refs[j * m + i] = dot(&a, &b, i, j);
            let (want, mag) = refs[j * m + i];
            check(
                &format!("f32 ({i},{j})"),
                f64::from(d0[j * m + i]),
                want,
                mag,
                ACC_REL,
            );
        }
    }
    let c0: Vec<f32> = (0..m * n).map(|_| rng.unit() * 1e3).collect();
    let c0_bytes: Vec<u8> = c0.iter().flat_map(|v| v.to_le_bytes()).collect();
    nsl_test_cuda_h2d(c.0, c0_bytes.as_ptr() as i64, c0_bytes.len() as i64);
    assert!(unsafe {
        gemm(
            m,
            n,
            k,
            1.0,
            a.data.ptr(),
            a.scales.ptr(),
            b.data.ptr(),
            b.scales.ptr(),
            1.0,
            c.ptr(),
            m,
            false,
        )
    });
    let d1 = f32s(&c.read());
    for at in 0..m * n {
        let (want, mag) = refs[at];
        let want = want + f64::from(c0[at]);
        check(
            &format!("beta=1 [{at}]"),
            f64::from(d1[at]),
            want,
            mag + f64::from(c0[at]).abs(),
            ACC_REL,
        );
    }

    // bf16 output: the f32 accumulator rounded once.
    let cb = Dev::new(m * n * 2);
    assert!(unsafe {
        gemm(
            m,
            n,
            k,
            1.0,
            a.data.ptr(),
            a.scales.ptr(),
            b.data.ptr(),
            b.scales.ptr(),
            0.0,
            cb.ptr(),
            m,
            true,
        )
    });
    let hb = cb.read();
    for at in 0..m * n {
        let got = f64::from(bf16_value(u16::from_le_bytes([hb[2 * at], hb[2 * at + 1]])));
        let (want, mag) = refs[at];
        check(
            &format!("bf16 [{at}]"),
            got,
            want,
            mag + want.abs(),
            2f64.powi(-8),
        );
    }
    let (issued1, _, declined1) = stats();
    assert_eq!(issued1 - issued0, 3, "three MXFP8 GEMMs issued");
    assert_eq!(declined1, declined0, "no MXFP8 GEMM declined");
    report("small GEMMs");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_weight_gradient_arrangement_multiplies_two_transposed_quantizations() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_available() {
        return;
    }
    // dW[d_in, d_out] = X^T G with X [T, d_in], G [T, d_out] row-major, K = T:
    // A = cols_t(G) (d_out rows), B = cols_t(X) (d_in rows); the column-major
    // [d_out, d_in] result is row-major dW.
    let mut rng = Rng(0x5eed_0004);
    let (t, d_in, d_out) = (256, 192, 128);
    let x = spread(t, d_in, false, &mut rng, false);
    let g = spread(t, d_out, false, &mut rng, false);
    let a = Op::cols_t(&g, t, d_out, true);
    let b = Op::cols_t(&x, t, d_in, false);
    let dw = Dev::new(d_in * d_out * 4);
    dw.fill(0);
    assert!(unsafe {
        gemm(
            d_out,
            d_in,
            t,
            1.0,
            a.data.ptr(),
            a.scales.ptr(),
            b.data.ptr(),
            b.scales.ptr(),
            1.0,
            dw.ptr(),
            d_out,
            false,
        )
    });
    let got = f32s(&dw.read());
    for i in 0..d_in {
        for o in 0..d_out {
            let (want, mag) = dot(&a, &b, o, i);
            check(
                &format!("dW[{i},{o}]"),
                f64::from(got[i * d_out + o]),
                want,
                mag,
                ACC_REL,
            );
            // And the arrangement is X^T G itself, to MXFP8 accuracy.
            let exact: f64 = (0..t)
                .map(|tt| f64::from(x[tt * d_in + i]) * f64::from(g[tt * d_out + o]))
                .sum();
            let exact_mag: f64 = (0..t)
                .map(|tt| (f64::from(x[tt * d_in + i]) * f64::from(g[tt * d_out + o])).abs())
                .sum();
            check(
                &format!("dW[{i},{o}] vs X^T G"),
                f64::from(got[i * d_out + o]),
                exact,
                exact_mag,
                0.15,
            );
        }
    }
    report("weight-gradient arrangement");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_1b_gate_up_weight_gradient_shape_runs_on_mxfp8() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_available() {
        return;
    }
    // m = d_out 8192, n = d_in 2048, K = T 4096; f32 accumulation, beta = 1.
    let mut rng = Rng(0x5eed_0005);
    let (m, n, k) = (8192, 2048, 4096);
    let a = Op::rows(&spread(m, k, true, &mut rng, false), m, k);
    let b = Op::rows(&spread(n, k, true, &mut rng, false), n, k);
    let c = Dev::new(m * n * 4);
    c.fill(0);
    assert!(unsafe {
        gemm(
            m,
            n,
            k,
            1.0,
            a.data.ptr(),
            a.scales.ptr(),
            b.data.ptr(),
            b.scales.ptr(),
            1.0,
            c.ptr(),
            m,
            false,
        )
    });
    let got = f32s(&c.read());
    for _ in 0..64 {
        let (i, j) = (rng.below(m as u64) as usize, rng.below(n as u64) as usize);
        let (want, mag) = dot(&a, &b, i, j);
        check(
            &format!("({i},{j})"),
            f64::from(got[j * m + i]),
            want,
            mag,
            ACC_REL,
        );
    }
    report("1B gate+up wgrad shape");
}

/// `[outer, k]` with block `b` of row `o` holding `(1 + j/8) · 2^e(o, b)`,
/// `j` in 0..8: E4M3 holds these exactly under the block's scale, and
/// neighbouring blocks get different exponents.
fn exact_blocks(outer: usize, k: usize) -> Vec<f32> {
    let mut x = vec![0f32; outer * k];
    for o in 0..outer {
        for b in 0..k / MX_BLOCK {
            let e = ((o * 7 + b * 13) % 31) as i32 - 15;
            for t in 0..MX_BLOCK {
                let j = ((o + 3 * b + 5 * t) % 8) as f32;
                x[o * k + b * MX_BLOCK + t] = (1.0 + j / 8.0) * 2f32.powi(e);
            }
        }
    }
    x
}

/// `[outer, k]` whose row `o` is ones on K block `o % (k / 32)`, zeros elsewhere.
fn selectors(outer: usize, k: usize) -> Vec<f32> {
    let kb = k / MX_BLOCK;
    let mut x = vec![0f32; outer * k];
    for o in 0..outer {
        let b = o % kb;
        for t in 0..MX_BLOCK {
            x[o * k + b * MX_BLOCK + t] = 1.0;
        }
    }
    x
}

#[test]
#[ignore = "requires CUDA GPU"]
fn each_block_scale_lands_on_its_own_block() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_available() {
        return;
    }
    // K = 320: ten blocks, the scale rows padded to twelve.
    let (spread_rows, sel_rows, k) = (256, 40, 320);
    let kb = k / MX_BLOCK;
    let x = exact_blocks(spread_rows, k);
    let sel = selectors(sel_rows, k);
    let block_sum = |o: usize, b: usize| -> f64 {
        (0..MX_BLOCK)
            .map(|t| f64::from(x[o * k + b * MX_BLOCK + t]))
            .sum()
    };
    // The spread operand as A (m = 256) and as B (n = 256).
    for spread_is_a in [true, false] {
        let a_src = if spread_is_a { &x } else { &sel };
        let b_src = if spread_is_a { &sel } else { &x };
        let (m, n) = if spread_is_a {
            (spread_rows, sel_rows)
        } else {
            (sel_rows, spread_rows)
        };
        let a = Op::rows(a_src, m, k);
        let b = Op::rows(b_src, n, k);
        let c = Dev::new(m * n * 4);
        assert!(unsafe {
            gemm(
                m,
                n,
                k,
                1.0,
                a.data.ptr(),
                a.scales.ptr(),
                b.data.ptr(),
                b.scales.ptr(),
                0.0,
                c.ptr(),
                m,
                false,
            )
        });
        let got = f32s(&c.read());
        for j in 0..n {
            for i in 0..m {
                let (o, sel_row) = if spread_is_a { (i, j) } else { (j, i) };
                let want = block_sum(o, sel_row % kb);
                let g = f64::from(got[j * m + i]);
                assert!(
                    (g - want).abs() <= 1e-6 * want,
                    "spread_is_a={spread_is_a}: D({i},{j}) = {g}, the exact block sum is {want}"
                );
            }
        }
    }
    // A NaN in row 5, block 2 of A: that row of D is NaN (its scale is the
    // UE8M0 NaN), every other row stays finite.
    let mut xn = x.clone();
    xn[5 * k + 2 * MX_BLOCK + 4] = f32::NAN;
    let a = Op::rows(&xn, spread_rows, k);
    let b = Op::rows(&sel, sel_rows, k);
    let c = Dev::new(spread_rows * sel_rows * 4);
    assert!(unsafe {
        gemm(
            spread_rows,
            sel_rows,
            k,
            1.0,
            a.data.ptr(),
            a.scales.ptr(),
            b.data.ptr(),
            b.scales.ptr(),
            0.0,
            c.ptr(),
            spread_rows,
            false,
        )
    });
    let got = f32s(&c.read());
    for j in 0..sel_rows {
        for i in 0..spread_rows {
            let v = got[j * spread_rows + i];
            if i == 5 && j % kb == 2 {
                assert!(
                    v.is_nan(),
                    "D(5,{j}) = {v}: the NaN block did not reach the output"
                );
            } else if i != 5 {
                assert!(v.is_finite(), "D({i},{j}) = {v}: a NaN escaped its row");
            }
        }
    }
}
