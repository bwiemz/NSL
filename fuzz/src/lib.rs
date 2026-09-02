//! The frontend invariants the fuzz targets check, shared with the
//! stable-toolchain seed test by `#[path]` (one definition, two builds).

#[path = "../../crates/nsl-parser/tests/common/frontend_check.rs"]
pub mod check;

/// Fuzzer bytes as lexer input: lossily decoded rather than rejected, so
/// a mutation that breaks a multi-byte character still exercises the
/// frontend on the replacement character instead of being thrown away.
pub fn source(data: &[u8]) -> std::borrow::Cow<'_, str> {
    String::from_utf8_lossy(data)
}
