use std::io::{self, Write};

/// Read a line from stdin, return as NUL-terminated C string pointer (i64).
#[no_mangle]
pub extern "C" fn nsl_read_line() -> i64 {
    io::stdout().flush().ok();
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).unwrap_or(0);
    let trimmed = buf.trim_end_matches('\n').trim_end_matches('\r');
    crate::string::nsl_str_from_rust(trimmed)
}
