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
                None => {
                    // b² overflowed — the final result cannot fit in i64.
                    eprintln!("nsl: integer overflow in exponentiation ({} ** {})", base, exp);
                    std::process::abort();
                }
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
