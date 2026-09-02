#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    nsl_fuzz::check::lex(&nsl_fuzz::source(data));
});
