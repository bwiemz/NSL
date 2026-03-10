use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use crate::list::{nsl_list_new, nsl_list_push};
use crate::memory::checked_alloc;

/// Helper: convert i64 pointer to &CStr
unsafe fn as_cstr(ptr: i64) -> &'static CStr {
    CStr::from_ptr(ptr as *const c_char)
}

/// Helper: allocate a C string copy
fn alloc_cstring(s: &str) -> i64 {
    let cstring = CString::new(s).unwrap_or_else(|_| CString::new("").unwrap());
    let bytes = cstring.as_bytes_with_nul();
    let ptr = checked_alloc(bytes.len());
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
    }
    ptr as i64
}

/// Helper: allocate a copy of a byte slice as C string
fn alloc_bytes(bytes: &[u8]) -> i64 {
    let ptr = checked_alloc(bytes.len() + 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        *ptr.add(bytes.len()) = 0; // null terminator
    }
    ptr as i64
}

#[no_mangle]
pub extern "C" fn nsl_str_concat(a: i64, b: i64) -> i64 {
    let sa = unsafe { as_cstr(a) }.to_bytes();
    let sb = unsafe { as_cstr(b) }.to_bytes();
    let mut result = Vec::with_capacity(sa.len() + sb.len());
    result.extend_from_slice(sa);
    result.extend_from_slice(sb);
    alloc_bytes(&result)
}

#[no_mangle]
pub extern "C" fn nsl_int_to_str(value: i64) -> i64 {
    alloc_cstring(&format!("{}", value))
}

#[no_mangle]
pub extern "C" fn nsl_float_to_str(value: f64) -> i64 {
    // Match C's %g: shortest representation
    if value == value.floor() && value.abs() < 1e15 && value != 0.0 {
        alloc_cstring(&format!("{}", value as i64))
    } else if value == 0.0 {
        alloc_cstring("0")
    } else {
        alloc_cstring(&format!("{}", value))
    }
}

#[no_mangle]
pub extern "C" fn nsl_bool_to_str(value: i8) -> i64 {
    if value != 0 {
        alloc_cstring("true")
    } else {
        alloc_cstring("false")
    }
}

#[no_mangle]
pub extern "C" fn nsl_str_to_int(s: i64) -> i64 {
    let cstr = unsafe { as_cstr(s) };
    let text = cstr.to_str().unwrap_or("");
    match text.trim().parse::<i64>() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("nsl: invalid int conversion from string '{}'", text);
            std::process::abort();
        }
    }
}

#[no_mangle]
pub extern "C" fn nsl_str_to_float(s: i64) -> f64 {
    let cstr = unsafe { as_cstr(s) };
    let text = cstr.to_str().unwrap_or("");
    match text.trim().parse::<f64>() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("nsl: invalid float conversion from string '{}'", text);
            std::process::abort();
        }
    }
}

#[no_mangle]
pub extern "C" fn nsl_str_len(s: i64) -> i64 {
    let cstr = unsafe { as_cstr(s) };
    cstr.to_bytes().len() as i64
}

#[no_mangle]
pub extern "C" fn nsl_str_upper(s: i64) -> i64 {
    let cstr = unsafe { as_cstr(s) };
    let text = cstr.to_str().unwrap_or("");
    alloc_cstring(&text.to_uppercase())
}

#[no_mangle]
pub extern "C" fn nsl_str_lower(s: i64) -> i64 {
    let cstr = unsafe { as_cstr(s) };
    let text = cstr.to_str().unwrap_or("");
    alloc_cstring(&text.to_lowercase())
}

#[no_mangle]
pub extern "C" fn nsl_str_strip(s: i64) -> i64 {
    let cstr = unsafe { as_cstr(s) };
    let text = cstr.to_str().unwrap_or("");
    alloc_cstring(text.trim())
}

#[no_mangle]
pub extern "C" fn nsl_str_split(s: i64, sep: i64) -> i64 {
    let text = unsafe { as_cstr(s) }.to_str().unwrap_or("");
    let separator = unsafe { as_cstr(sep) }.to_str().unwrap_or("");
    let result = nsl_list_new();

    if separator.is_empty() {
        // Split into individual characters
        for ch in text.chars() {
            let cs = alloc_cstring(&ch.to_string());
            nsl_list_push(result, cs);
        }
    } else {
        for part in text.split(separator) {
            let cs = alloc_cstring(part);
            nsl_list_push(result, cs);
        }
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_str_join(sep: i64, list_ptr: i64) -> i64 {
    let separator = unsafe { as_cstr(sep) }.to_str().unwrap_or("");
    let list = crate::list::NslList::from_ptr(list_ptr);

    let mut parts: Vec<String> = Vec::new();
    for i in 0..list.len {
        let elem_ptr = unsafe { *list.data.add(i as usize) };
        let elem = unsafe { as_cstr(elem_ptr) }.to_str().unwrap_or("");
        parts.push(elem.to_string());
    }
    alloc_cstring(&parts.join(separator))
}

#[no_mangle]
pub extern "C" fn nsl_str_replace(s: i64, old_s: i64, new_s: i64) -> i64 {
    let text = unsafe { as_cstr(s) }.to_str().unwrap_or("");
    let old = unsafe { as_cstr(old_s) }.to_str().unwrap_or("");
    let new = unsafe { as_cstr(new_s) }.to_str().unwrap_or("");

    if old.is_empty() {
        return alloc_cstring(text);
    }
    alloc_cstring(&text.replace(old, new))
}

#[no_mangle]
pub extern "C" fn nsl_str_find(s: i64, sub: i64) -> i64 {
    let text = unsafe { as_cstr(s) }.to_str().unwrap_or("");
    let substr = unsafe { as_cstr(sub) }.to_str().unwrap_or("");
    match text.find(substr) {
        Some(pos) => pos as i64,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn nsl_str_startswith(s: i64, prefix: i64) -> i8 {
    let text = unsafe { as_cstr(s) }.to_str().unwrap_or("");
    let pfx = unsafe { as_cstr(prefix) }.to_str().unwrap_or("");
    if text.starts_with(pfx) { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn nsl_str_endswith(s: i64, suffix: i64) -> i8 {
    let text = unsafe { as_cstr(s) }.to_str().unwrap_or("");
    let sfx = unsafe { as_cstr(suffix) }.to_str().unwrap_or("");
    if text.ends_with(sfx) { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn nsl_str_contains(s: i64, sub: i64) -> i8 {
    let text = unsafe { as_cstr(s) }.to_str().unwrap_or("");
    let substr = unsafe { as_cstr(sub) }.to_str().unwrap_or("");
    if text.contains(substr) { 1 } else { 0 }
}

/// Free a dynamically allocated string (allocated by checked_alloc).
/// Used for tokenizer decode return values and other dynamic strings.
#[no_mangle]
pub extern "C" fn nsl_string_free(ptr: i64) {
    if ptr == 0 { return; }
    // Strings are null-terminated C strings allocated via checked_alloc.
    // We need to find the length to free the right amount.
    let cstr = unsafe { CStr::from_ptr(ptr as *const c_char) };
    let len = cstr.to_bytes_with_nul().len();
    unsafe { crate::memory::checked_free(ptr as *mut u8, len); }
}
