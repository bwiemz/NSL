use crate::memory::{checked_alloc, checked_realloc};

const INITIAL_CAP: i64 = 8;

#[repr(C)]
pub struct NslList {
    pub(crate) data: *mut i64,
    pub(crate) len: i64,
    cap: i64,
}

impl NslList {
    pub(crate) fn from_ptr(ptr: i64) -> &'static mut NslList {
        unsafe { &mut *(ptr as *mut NslList) }
    }
}

#[no_mangle]
pub extern "C" fn nsl_list_new() -> i64 {
    let list = Box::new(NslList {
        data: checked_alloc((INITIAL_CAP as usize) * std::mem::size_of::<i64>()) as *mut i64,
        len: 0,
        cap: INITIAL_CAP,
    });
    Box::into_raw(list) as i64
}

#[no_mangle]
pub extern "C" fn nsl_list_push(list_ptr: i64, value: i64) {
    let list = NslList::from_ptr(list_ptr);
    if list.len >= list.cap {
        let old_size = (list.cap as usize) * std::mem::size_of::<i64>();
        list.cap *= 2;
        let new_size = (list.cap as usize) * std::mem::size_of::<i64>();
        list.data = unsafe { checked_realloc(list.data as *mut u8, old_size, new_size) as *mut i64 };
    }
    unsafe {
        *list.data.add(list.len as usize) = value;
    }
    list.len += 1;
}

#[no_mangle]
pub extern "C" fn nsl_list_get(list_ptr: i64, index: i64) -> i64 {
    let list = NslList::from_ptr(list_ptr);
    if index < 0 || index >= list.len {
        eprintln!(
            "nsl: list index out of bounds (index {}, length {})",
            index, list.len
        );
        std::process::abort();
    }
    unsafe { *list.data.add(index as usize) }
}

#[no_mangle]
pub extern "C" fn nsl_list_len(list_ptr: i64) -> i64 {
    let list = NslList::from_ptr(list_ptr);
    list.len
}

#[no_mangle]
pub extern "C" fn nsl_list_set(list_ptr: i64, index: i64, value: i64) {
    let list = NslList::from_ptr(list_ptr);
    if index < 0 || index >= list.len {
        eprintln!(
            "nsl: list index out of bounds in assignment (index {}, length {})",
            index, list.len
        );
        std::process::abort();
    }
    unsafe {
        *list.data.add(index as usize) = value;
    }
}

#[no_mangle]
pub extern "C" fn nsl_list_contains(list_ptr: i64, value: i64) -> i8 {
    let list = NslList::from_ptr(list_ptr);
    for i in 0..list.len {
        if unsafe { *list.data.add(i as usize) } == value {
            return 1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nsl_list_slice(list_ptr: i64, lo: i64, hi: i64, step_val: i64) -> i64 {
    let list = NslList::from_ptr(list_ptr);
    let len = list.len;
    let step = if step_val == i64::MIN { 1 } else { step_val };

    if step == 0 {
        eprintln!("nsl: slice step cannot be zero");
        std::process::abort();
    }

    let result = nsl_list_new();

    if step > 0 {
        let mut low = if lo == i64::MIN { 0 } else { lo };
        let mut high = if hi == i64::MIN { len } else { hi };
        if low < 0 { low += len; }
        if high < 0 { high += len; }
        if low < 0 { low = 0; }
        if high > len { high = len; }

        let mut i = low;
        while i < high {
            nsl_list_push(result, unsafe { *list.data.add(i as usize) });
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
                nsl_list_push(result, unsafe { *list.data.add(i as usize) });
            }
            i += step;
        }
    }

    result
}
