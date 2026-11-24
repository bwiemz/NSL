//! Launchers for the MXFP8 quantizers (`nsl_kir::kernels::mx_quant`): f32 or
//! bf16 device data to E4M3 bytes plus cublasLt-layout UE8M0 block scales,
//! on the compute stream.

use std::ffi::{CStr, CString, c_void};
use std::sync::OnceLock;

use nsl_kir::kernels::mx_quant::{
    MX_BLOCK, MX_THREADS, MX_TILE, MxQuant, scale_bytes, scales_padded, sf_inner,
};
use nsl_kir::kernels::storage::Storage;

use super::inner;

/// `(ptx, name)` of one quantizer, both NUL-terminated, built once.
fn module(q: MxQuant, src: Storage) -> (&'static str, &'static CStr) {
    static MODULES: [OnceLock<(String, CString)>; 4] = [const { OnceLock::new() }; 4];
    let at = 2 * usize::from(q == MxQuant::ColsT) + usize::from(src == Storage::Bf16);
    let (p, n) = MODULES[at].get_or_init(|| {
        (
            String::from_utf8(nsl_kir::kernels::mx_quant::ptx(q, src)).expect("PTX must be ASCII"),
            CString::new(q.kernel_name(src)).expect("kernel names have no NUL"),
        )
    });
    (p.as_str(), n.as_c_str())
}

fn storage(src_bf16: bool) -> Storage {
    if src_bf16 {
        Storage::Bf16
    } else {
        Storage::F32
    }
}

/// Clear the scale buffer when it has entries outside the tensor (cublasLt
/// reads whole 128 × 4 tiles; the kernels write only in-range bytes).
fn clear_padding(scales: u64, outer: usize, k: usize) {
    if scales_padded(outer, k) {
        inner::memset_d8(scales as *mut c_void, scale_bytes(outer, k));
    }
}

/// Whether this device runs MXFP8: the quantizers need sm_89 (the E4M3
/// conversion), the block-scaled GEMM compute capability 10.0.
pub(crate) fn supported() -> bool {
    super::context::current().sm_version() >= 100
}

/// The kernels load 4 elements at once (16 bytes f32, 8 bf16), store 4
/// bytes, and cublasLt reads the scales 16-byte aligned.
fn check_buffers(what: &str, src: u64, src_bf16: bool, data: u64, scales: u64) {
    assert!(
        supported(),
        "{what}: MXFP8 needs compute capability 10.0 or newer"
    );
    let src_align = if src_bf16 { 8 } else { 16 };
    assert!(
        src.is_multiple_of(src_align) && data.is_multiple_of(4) && scales.is_multiple_of(16),
        "{what}: misaligned buffer (src {src:#x}, data {data:#x}, scales {scales:#x})"
    );
}

fn launch(q: MxQuant, src: Storage, grid: [i64; 3], args: &[*mut c_void]) {
    let (ptx, name) = module(q, src);
    let rc = inner::kernel_launch(
        ptx.as_ptr(),
        name.as_ptr().cast(),
        grid,
        [i64::from(MX_THREADS), 1, 1],
        args,
        0,
    );
    assert_eq!(rc as u32, 0, "{} launch failed: {rc:?}", q.kernel_name(src));
}

/// Quantize row-major `[rows, k]` with blocks along each row into `data`
/// (`rows · k` bytes, same layout) and `scales` (`scale_bytes(rows, k)`).
pub(crate) fn quant_rows(src: u64, src_bf16: bool, rows: usize, k: usize, data: u64, scales: u64) {
    assert!(
        k > 0 && k.is_multiple_of(MX_BLOCK),
        "mx_quant rows: k = {k} is not a multiple of {MX_BLOCK}"
    );
    let quads = rows.checked_mul(k).expect("mx_quant rows: size overflows") / 4;
    assert!(
        u32::try_from(quads).is_ok(),
        "mx_quant rows: {rows} x {k} is past the 32-bit index"
    );
    if rows == 0 {
        return;
    }
    check_buffers("mx_quant rows", src, src_bf16, data, scales);
    clear_padding(scales, rows, k);
    let (mut src, mut data, mut scales) = (src, data, scales);
    let mut nquads = quads as u32;
    let mut k_blocks = (k / MX_BLOCK) as u32;
    let mut sf = sf_inner(k) as u32;
    let p64 = |p: &mut u64| p as *mut u64 as *mut c_void;
    let p32 = |p: &mut u32| p as *mut u32 as *mut c_void;
    let args = [
        p64(&mut src),
        p64(&mut data),
        p64(&mut scales),
        p32(&mut nquads),
        p32(&mut k_blocks),
        p32(&mut sf),
    ];
    let grid = quads.div_ceil(MX_THREADS as usize) as i64;
    launch(MxQuant::Rows, storage(src_bf16), [grid, 1, 1], &args);
}

/// Quantize the transpose of row-major `[r, c]`, blocks along each column,
/// into `data` (`[c, r]`, `r · c` bytes) and `scales`
/// (`scale_bytes(c, r)`).
pub(crate) fn quant_cols_t(src: u64, src_bf16: bool, r: usize, c: usize, data: u64, scales: u64) {
    let tile = MX_TILE as usize;
    assert!(
        r.is_multiple_of(tile) && c.is_multiple_of(tile),
        "mx_quant cols_t: [{r}, {c}] is not a multiple of {tile}"
    );
    assert!(
        r / tile <= 65_535,
        "mx_quant cols_t: {r} rows exceed the grid's y extent"
    );
    assert!(
        u32::try_from(r).is_ok() && u32::try_from(c).is_ok(),
        "mx_quant cols_t: [{r}, {c}]"
    );
    if r == 0 || c == 0 {
        return;
    }
    check_buffers("mx_quant cols_t", src, src_bf16, data, scales);
    clear_padding(scales, c, r);
    let (mut src, mut data, mut scales) = (src, data, scales);
    let (mut rr, mut cc) = (r as u32, c as u32);
    let mut sf = sf_inner(r) as u32;
    let p64 = |p: &mut u64| p as *mut u64 as *mut c_void;
    let p32 = |p: &mut u32| p as *mut u32 as *mut c_void;
    let args = [
        p64(&mut src),
        p64(&mut data),
        p64(&mut scales),
        p32(&mut rr),
        p32(&mut cc),
        p32(&mut sf),
    ];
    launch(
        MxQuant::ColsT,
        storage(src_bf16),
        [(c / tile) as i64, (r / tile) as i64, 1],
        &args,
    );
}
