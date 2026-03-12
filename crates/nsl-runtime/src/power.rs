#[no_mangle]
pub extern "C" fn nsl_pow_int(base: i64, exp: i64) -> i64 {
    if exp < 0 {
        // Integer base with negative exponent: result is always 0 in integer division
        // (except base=1 → 1, base=-1 → ±1)
        return match base {
            1 => 1,
            -1 => if exp & 1 != 0 { -1 } else { 1 },
            _ => 0,
        };
    }
    let mut result: i64 = 1;
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 != 0 {
            match result.checked_mul(b) {
                Some(v) => result = v,
                None => {
                    eprintln!("nsl: integer overflow in exponentiation ({} ** {})", base, exp);
                    std::process::abort();
                }
            }
        }
        if e > 1 {
            match b.checked_mul(b) {
                Some(v) => b = v,
                None if e & 1 == 0 => {
                    // Will overflow on next multiply with result
                    eprintln!("nsl: integer overflow in exponentiation ({} ** {})", base, exp);
                    std::process::abort();
                }
                None => b = 0, // won't be used since e <= 1 after this shift
            }
        }
        e >>= 1;
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_pow_float(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}
