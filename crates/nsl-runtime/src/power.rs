#[no_mangle]
pub extern "C" fn nsl_pow_int(base: i64, exp: i64) -> i64 {
    if exp < 0 {
        return 0;
    }
    let mut result: i64 = 1;
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 != 0 {
            result = result.wrapping_mul(b);
        }
        b = b.wrapping_mul(b);
        e >>= 1;
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_pow_float(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}
