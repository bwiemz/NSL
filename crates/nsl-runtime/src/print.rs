use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn nsl_print_int(value: i64) {
    println!("{}", value);
}

#[no_mangle]
pub extern "C" fn nsl_print_float(value: f64) {
    if value == value.floor() && value.abs() < 1e15 && value != 0.0 {
        // Integer-valued floats: print without decimal point (like C's %g)
        println!("{}", value as i64);
    } else if value == 0.0 {
        println!("0");
    } else {
        // Use Rust's default which is close to %g
        println!("{}", value);
    }
}

#[no_mangle]
pub extern "C" fn nsl_print_str(value: i64) {
    if value == 0 {
        println!("(null)");
        return;
    }
    let ptr = value as *const c_char;
    let cstr = unsafe { CStr::from_ptr(ptr) };
    match cstr.to_str() {
        Ok(s) => println!("{}", s),
        Err(_) => println!("(invalid utf-8)"),
    }
}

#[no_mangle]
pub extern "C" fn nsl_print_bool(value: i8) {
    if value != 0 {
        println!("true");
    } else {
        println!("false");
    }
}
