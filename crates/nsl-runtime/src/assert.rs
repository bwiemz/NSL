use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn nsl_assert(condition: i8, message: i64) {
    if condition == 0 {
        let msg = if message != 0 {
            unsafe { CStr::from_ptr(message as *const c_char) }
                .to_str()
                .unwrap_or("assertion failed")
        } else {
            "assertion failed"
        };
        eprintln!("nsl: assertion failed: {}", msg);
        std::process::abort();
    }
}

#[no_mangle]
pub extern "C" fn nsl_exit(code: i64) {
    std::process::exit(code as i32);
}
