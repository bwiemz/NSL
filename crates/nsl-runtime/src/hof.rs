use crate::list::{nsl_list_new, nsl_list_push, NslList};

type NslFn1 = extern "C" fn(i64) -> i64;

/// `ret_is_bool`: the codegen passes 1 when the function's NSL return
/// type is `bool`, which lowers to an I8 return — only the LOW BYTE of
/// the return register is defined, the rest is leftover callee state.
/// Reading the full i64 in that case kept/produced garbage (a bool
/// predicate's `!= 0` check saw stale upper bits). Masking is gated on
/// the flag because an int-returning function's full 64 bits are valid.
#[no_mangle]
pub extern "C" fn nsl_map(fn_ptr: i64, list_ptr: i64, ret_is_bool: i64) -> i64 {
    let func: NslFn1 = unsafe { std::mem::transmute(fn_ptr) };
    let src = NslList::from_ptr(list_ptr);
    let result = nsl_list_new();
    for i in 0..src.len {
        let val = unsafe { *src.data.add(i as usize) };
        let mapped = func(val);
        let mapped = if ret_is_bool != 0 {
            ((mapped & 0xFF) != 0) as i64
        } else {
            mapped
        };
        nsl_list_push(result, mapped);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_filter(fn_ptr: i64, list_ptr: i64, ret_is_bool: i64) -> i64 {
    let func: NslFn1 = unsafe { std::mem::transmute(fn_ptr) };
    let src = NslList::from_ptr(list_ptr);
    let result = nsl_list_new();
    for i in 0..src.len {
        let val = unsafe { *src.data.add(i as usize) };
        let raw = func(val);
        let keep = if ret_is_bool != 0 {
            (raw & 0xFF) != 0
        } else {
            raw != 0
        };
        if keep {
            nsl_list_push(result, val);
        }
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_enumerate(list_ptr: i64) -> i64 {
    let src = NslList::from_ptr(list_ptr);
    let result = nsl_list_new();
    for i in 0..src.len {
        let pair = nsl_list_new();
        nsl_list_push(pair, i);
        nsl_list_push(pair, unsafe { *src.data.add(i as usize) });
        nsl_list_push(result, pair);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_zip(list_a: i64, list_b: i64) -> i64 {
    let a = NslList::from_ptr(list_a);
    let b = NslList::from_ptr(list_b);
    let min_len = a.len.min(b.len);
    let result = nsl_list_new();
    for i in 0..min_len {
        let pair = nsl_list_new();
        nsl_list_push(pair, unsafe { *a.data.add(i as usize) });
        nsl_list_push(pair, unsafe { *b.data.add(i as usize) });
        nsl_list_push(result, pair);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_sorted(list_ptr: i64) -> i64 {
    let src = NslList::from_ptr(list_ptr);
    let result = nsl_list_new();
    // Copy elements
    let mut data: Vec<i64> = Vec::with_capacity(src.len as usize);
    for i in 0..src.len {
        data.push(unsafe { *src.data.add(i as usize) });
    }
    data.sort();
    for val in data {
        nsl_list_push(result, val);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_reversed(list_ptr: i64) -> i64 {
    let src = NslList::from_ptr(list_ptr);
    let result = nsl_list_new();
    for i in (0..src.len).rev() {
        nsl_list_push(result, unsafe { *src.data.add(i as usize) });
    }
    result
}
