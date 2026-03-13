use std::ffi::CStr;
use std::os::raw::c_char;

use crate::memory::checked_alloc;

unsafe fn as_cstr(ptr: i64) -> &'static CStr {
    CStr::from_ptr(ptr as *const c_char)
}

#[no_mangle]
pub extern "C" fn nsl_read_file(path: i64) -> i64 {
    let path_str = unsafe { as_cstr(path) }.to_str().unwrap_or("");
    match std::fs::read(path_str) {
        Ok(contents) => {
            let ptr = checked_alloc(contents.len() + 1);
            unsafe {
                std::ptr::copy_nonoverlapping(contents.as_ptr(), ptr, contents.len());
                *ptr.add(contents.len()) = 0;
            }
            ptr as i64
        }
        Err(_) => {
            eprintln!("nsl: could not open file '{}' for reading", path_str);
            std::process::abort();
        }
    }
}

#[no_mangle]
pub extern "C" fn nsl_write_file(path: i64, content: i64) {
    let path_str = unsafe { as_cstr(path) }.to_str().unwrap_or("");
    let content_str = unsafe { as_cstr(content) }.to_str().unwrap_or("");
    if std::fs::write(path_str, content_str).is_err() {
        eprintln!("nsl: could not open file '{}' for writing", path_str);
        std::process::abort();
    }
}

#[no_mangle]
pub extern "C" fn nsl_append_file(path: i64, content: i64) {
    use std::io::Write;
    let path_str = unsafe { as_cstr(path) }.to_str().unwrap_or("");
    let content_str = unsafe { as_cstr(content) }.to_str().unwrap_or("");
    let mut file = match std::fs::OpenOptions::new().append(true).create(true).open(path_str) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("nsl: could not open file '{}' for appending", path_str);
            std::process::abort();
        }
    };
    if file.write_all(content_str.as_bytes()).is_err() {
        eprintln!("nsl: could not write to file '{}'", path_str);
        std::process::abort();
    }
}

#[no_mangle]
pub extern "C" fn nsl_file_exists(path: i64) -> i8 {
    let path_str = unsafe { as_cstr(path) }.to_str().unwrap_or("");
    if std::path::Path::new(path_str).exists() { 1 } else { 0 }
}
