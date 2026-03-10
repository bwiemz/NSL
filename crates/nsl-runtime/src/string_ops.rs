use std::ffi::CStr;
use std::os::raw::c_char;

use crate::memory::checked_alloc;

unsafe fn as_cstr(ptr: i64) -> &'static CStr {
    CStr::from_ptr(ptr as *const c_char)
}

fn alloc_bytes(bytes: &[u8]) -> i64 {
    let ptr = checked_alloc(bytes.len() + 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        *ptr.add(bytes.len()) = 0;
    }
    ptr as i64
}

#[no_mangle]
pub extern "C" fn nsl_str_repeat(s: i64, n: i64) -> i64 {
    let bytes = unsafe { as_cstr(s) }.to_bytes();
    if n <= 0 || bytes.is_empty() {
        return alloc_bytes(b"");
    }
    let repeated: Vec<u8> = bytes.repeat(n as usize);
    alloc_bytes(&repeated)
}

#[no_mangle]
pub extern "C" fn nsl_str_eq(a: i64, b: i64) -> i64 {
    if a == b {
        return 1;
    }
    if a == 0 || b == 0 {
        return 0;
    }
    let sa = unsafe { as_cstr(a) }.to_bytes();
    let sb = unsafe { as_cstr(b) }.to_bytes();
    if sa == sb { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn nsl_str_slice(s: i64, lo: i64, hi: i64, step_val: i64) -> i64 {
    if s == 0 {
        return alloc_bytes(b"");
    }
    let bytes = unsafe { as_cstr(s) }.to_bytes();
    let len = bytes.len() as i64;

    let step = if step_val == i64::MIN { 1 } else { step_val };
    if step == 0 {
        eprintln!("nsl: slice step cannot be zero");
        std::process::abort();
    }

    let mut result = Vec::new();

    if step > 0 {
        let mut low = if lo == i64::MIN { 0 } else { lo };
        let mut high = if hi == i64::MIN { len } else { hi };
        if low < 0 { low += len; }
        if high < 0 { high += len; }
        if low < 0 { low = 0; }
        if high > len { high = len; }

        let mut i = low;
        while i < high {
            result.push(bytes[i as usize]);
            i += step;
        }
    } else {
        let mut low = if lo == i64::MIN { len - 1 } else { lo };
        let high = if hi == i64::MIN { -(len + 1) } else { hi };
        let mut adj_high = high;
        if low < 0 { low += len; }
        if adj_high < 0 { adj_high += len; }
        if low >= len { low = len - 1; }

        let mut i = low;
        while i > adj_high {
            if i >= 0 && i < len {
                result.push(bytes[i as usize]);
            }
            i += step;
        }
    }

    alloc_bytes(&result)
}
