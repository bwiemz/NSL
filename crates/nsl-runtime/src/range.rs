use crate::list::{nsl_list_new, nsl_list_push};

#[no_mangle]
pub extern "C" fn nsl_range(start: i64, stop: i64, step: i64) -> i64 {
    let list = nsl_list_new();
    if step > 0 {
        let mut i = start;
        while i < stop {
            nsl_list_push(list, i);
            i += step;
        }
    } else if step < 0 {
        let mut i = start;
        while i > stop {
            nsl_list_push(list, i);
            i += step;
        }
    }
    list
}
