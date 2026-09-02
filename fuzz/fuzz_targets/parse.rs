#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    nsl_fuzz::check::parse(&nsl_fuzz::source(data));
});
