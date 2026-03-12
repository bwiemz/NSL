#[no_mangle]
pub extern "C" fn nsl_sqrt(x: f64) -> f64 { x.sqrt() }

#[no_mangle]
pub extern "C" fn nsl_log(x: f64) -> f64 { x.ln() }

#[no_mangle]
pub extern "C" fn nsl_exp(x: f64) -> f64 { x.exp() }

#[no_mangle]
pub extern "C" fn nsl_sin(x: f64) -> f64 { x.sin() }

#[no_mangle]
pub extern "C" fn nsl_cos(x: f64) -> f64 { x.cos() }

#[no_mangle]
pub extern "C" fn nsl_abs_float(x: f64) -> f64 { x.abs() }

#[no_mangle]
pub extern "C" fn nsl_abs_int(x: i64) -> i64 {
    x.checked_abs().unwrap_or_else(|| {
        eprintln!("nsl: integer overflow in abs({})", x);
        std::process::abort();
    })
}

#[no_mangle]
pub extern "C" fn nsl_min_int(a: i64, b: i64) -> i64 { a.min(b) }

#[no_mangle]
pub extern "C" fn nsl_max_int(a: i64, b: i64) -> i64 { a.max(b) }

#[no_mangle]
pub extern "C" fn nsl_min_float(a: f64, b: f64) -> f64 { a.min(b) }

#[no_mangle]
pub extern "C" fn nsl_max_float(a: f64, b: f64) -> f64 { a.max(b) }

#[no_mangle]
pub extern "C" fn nsl_floor(x: f64) -> f64 { x.floor() }
