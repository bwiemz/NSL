use std::ffi::CStr;
use std::os::raw::c_char;

use crate::list::{nsl_list_new, nsl_list_push};
use crate::memory::checked_alloc;

#[repr(C)]
struct NslDictEntry {
    key: *mut u8,
    value: i64,
    next: *mut NslDictEntry,
}

#[repr(C)]
struct NslDict {
    buckets: *mut *mut NslDictEntry,
    num_buckets: i64,
    len: i64,
}

fn fnv1a(s: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &b in s {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

unsafe fn as_cstr(ptr: i64) -> &'static CStr {
    CStr::from_ptr(ptr as *const c_char)
}

fn copy_cstr(s: &[u8]) -> *mut u8 {
    let ptr = checked_alloc(s.len() + 1);
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), ptr, s.len());
        *ptr.add(s.len()) = 0;
    }
    ptr
}

impl NslDict {
    fn from_ptr(ptr: i64) -> &'static mut NslDict {
        unsafe { &mut *(ptr as *mut NslDict) }
    }

    unsafe fn resize(&mut self, new_cap: i64) {
        let new_buckets = checked_alloc((new_cap as usize) * std::mem::size_of::<*mut NslDictEntry>())
            as *mut *mut NslDictEntry;
        for i in 0..new_cap as usize {
            *new_buckets.add(i) = std::ptr::null_mut();
        }
        for i in 0..self.num_buckets as usize {
            let mut e = *self.buckets.add(i);
            while !e.is_null() {
                let next = (*e).next;
                let key_bytes = CStr::from_ptr((*e).key as *const c_char).to_bytes();
                let idx = (fnv1a(key_bytes) % new_cap as u64) as usize;
                (*e).next = *new_buckets.add(idx);
                *new_buckets.add(idx) = e;
                e = next;
            }
        }
        // Free old buckets array (not entries — they've been re-linked into new_buckets)
        let old_size = (self.num_buckets as usize) * std::mem::size_of::<*mut NslDictEntry>();
        crate::memory::checked_free(self.buckets as *mut u8, old_size);
        self.buckets = new_buckets;
        self.num_buckets = new_cap;
    }
}

fn free_dict_impl(dict_ptr: i64, free_tensor_values: bool) {
    if dict_ptr == 0 {
        return;
    }

    let dict = unsafe { &mut *(dict_ptr as *mut NslDict) };
    for i in 0..dict.num_buckets as usize {
        let mut entry = unsafe { *dict.buckets.add(i) };
        while !entry.is_null() {
            let e = unsafe { &*entry };
            let next = e.next;
            let key_bytes = unsafe { CStr::from_ptr(e.key as *const c_char) }.to_bytes_with_nul();
            unsafe { crate::memory::checked_free(e.key, key_bytes.len()) };
            if free_tensor_values {
                crate::tensor::nsl_tensor_free(e.value);
            }
            unsafe { drop(Box::from_raw(entry)) };
            entry = next;
        }
    }

    let bucket_size = (dict.num_buckets as usize) * std::mem::size_of::<*mut NslDictEntry>();
    unsafe { crate::memory::checked_free(dict.buckets as *mut u8, bucket_size) };
    unsafe { drop(Box::from_raw(dict as *mut NslDict)) };
}

#[no_mangle]
pub extern "C" fn nsl_dict_new() -> i64 {
    let num_buckets: i64 = 16;
    let buckets = checked_alloc((num_buckets as usize) * std::mem::size_of::<*mut NslDictEntry>())
        as *mut *mut NslDictEntry;
    unsafe {
        for i in 0..num_buckets as usize {
            *buckets.add(i) = std::ptr::null_mut();
        }
    }
    let dict = Box::new(NslDict {
        buckets,
        num_buckets,
        len: 0,
    });
    Box::into_raw(dict) as i64
}

#[no_mangle]
pub extern "C" fn nsl_dict_set_str(dict_ptr: i64, key: i64, value: i64) {
    let d = NslDict::from_ptr(dict_ptr);
    let key_bytes = unsafe { as_cstr(key) }.to_bytes();
    let idx = (fnv1a(key_bytes) % d.num_buckets as u64) as usize;

    // Check for existing key
    unsafe {
        let mut e = *d.buckets.add(idx);
        while !e.is_null() {
            let e_key = CStr::from_ptr((*e).key as *const c_char).to_bytes();
            if e_key == key_bytes {
                (*e).value = value;
                return;
            }
            e = (*e).next;
        }
    }

    // Insert new entry
    let entry = Box::new(NslDictEntry {
        key: copy_cstr(key_bytes),
        value,
        next: unsafe { *d.buckets.add(idx) },
    });
    unsafe {
        *d.buckets.add(idx) = Box::into_raw(entry);
    }
    d.len += 1;

    // Resize at 0.75 load factor
    if d.len * 4 > d.num_buckets * 3 {
        unsafe { d.resize(d.num_buckets * 2) };
    }
}

#[no_mangle]
pub extern "C" fn nsl_dict_get_str(dict_ptr: i64, key: i64) -> i64 {
    let d = NslDict::from_ptr(dict_ptr);
    let key_bytes = unsafe { as_cstr(key) }.to_bytes();
    let idx = (fnv1a(key_bytes) % d.num_buckets as u64) as usize;

    unsafe {
        let mut e = *d.buckets.add(idx);
        while !e.is_null() {
            let e_key = CStr::from_ptr((*e).key as *const c_char).to_bytes();
            if e_key == key_bytes {
                return (*e).value;
            }
            e = (*e).next;
        }
    }
    let key_str = unsafe { as_cstr(key) }.to_str().unwrap_or("?");
    eprintln!("nsl: key not found in dict: '{}'", key_str);
    std::process::abort();
}

#[no_mangle]
pub extern "C" fn nsl_dict_len(dict_ptr: i64) -> i64 {
    NslDict::from_ptr(dict_ptr).len
}

#[no_mangle]
pub extern "C" fn nsl_dict_contains(dict_ptr: i64, key: i64) -> i8 {
    let d = NslDict::from_ptr(dict_ptr);
    let key_bytes = unsafe { as_cstr(key) }.to_bytes();
    let idx = (fnv1a(key_bytes) % d.num_buckets as u64) as usize;

    unsafe {
        let mut e = *d.buckets.add(idx);
        while !e.is_null() {
            let e_key = CStr::from_ptr((*e).key as *const c_char).to_bytes();
            if e_key == key_bytes {
                return 1;
            }
            e = (*e).next;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nsl_dict_keys(dict_ptr: i64) -> i64 {
    let d = NslDict::from_ptr(dict_ptr);
    let result = nsl_list_new();
    unsafe {
        for i in 0..d.num_buckets as usize {
            let mut e = *d.buckets.add(i);
            while !e.is_null() {
                nsl_list_push(result, (*e).key as i64);
                e = (*e).next;
            }
        }
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_dict_free(dict_ptr: i64) {
    // Values are owned by the caller — dict only frees its own structure.
    // This matches nsl_list_free which also does not free elements.
    free_dict_impl(dict_ptr, false);
}

#[no_mangle]
pub extern "C" fn nsl_dict_free_tensor_values(dict_ptr: i64) {
    free_dict_impl(dict_ptr, true);
}
