//! Quantization and calibration requests the toolchain does not honour are
//! refused here rather than accepted and dropped (roadmap: "turn ignored
//! calibration requests into either implemented behavior or compile/CLI
//! refusal").
//!
//! Each refusal names what was requested, why it cannot be honoured, and
//! what to write instead — the same contract as the closed optimizer kwargs
//! (`optim_config.rs`) and `UNIMPLEMENTED_DECORATORS`:
//!
//! - **`quant` block `calibration:`.** The parser stored `data` and
//!   `samples`, the checker looked `data` up, and codegen never read either:
//!   the block quantized from the weights' own min/max whatever the section
//!   said.
//! - **`quant` block `default: awq4 | gptq4 | gptq8`.** Codegen passed the
//!   dtype codes 2/3/4 to `nsl_qtensor_quantize`, which implements only int8
//!   and int4 and aborts the process on anything else, so a program that
//!   checked and built died when the block ran. GPTQ's Hessian API is never
//!   emitted, and AWQ's pre-scaled weight is never un-scaled.
//! - **`@quantize`.** Codegen reads `dtype` (only `"awq4"`, which marks the
//!   model for AWQ projection discovery) and logs `group_size` without using
//!   it; every other argument fell through a `_ => {}` arm. The decorator's
//!   arguments are now a closed set.
//!
//! `@fp8_compute(calibrate = true)` is refused in `fp8.rs`, next to the rest
//! of that decorator's validation.

use nsl_ast::block::{QuantBlock, QuantDtype};
use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Why a `quant` block's `calibration:` section is refused.
pub const QUANT_CALIBRATION_REFUSAL: &str = "a quant block's `calibration:` section is not implemented: its \
     `data` and `samples` were never read, and the block quantizes from the weights' own statistics \
     whatever the section says. Remove the section";

/// Why `default: awq4 | gptq4 | gptq8` is refused in a quant block.
pub const QUANT_DTYPE_REFUSAL: &str = "quant blocks implement only `int8` and `int4`: the runtime quantizer \
     has no AWQ or GPTQ path, and a program using this dtype aborted when the block ran. Use \
     `default: int4` (or `int8`)";

/// Refuse the `quant` block requests the toolchain never honoured.
pub fn check_quant_block_requests(quant: &QuantBlock, diagnostics: &mut Vec<Diagnostic>) {
    if quant.calibration.is_some() {
        diagnostics.push(
            Diagnostic::error("quant block `calibration:` is not implemented".to_string())
                .with_label(quant.span, QUANT_CALIBRATION_REFUSAL),
        );
    }
    let unsupported = match quant.default_dtype {
        Some(QuantDtype::Awq4) => Some("awq4"),
        Some(QuantDtype::Gptq4) => Some("gptq4"),
        Some(QuantDtype::Gptq8) => Some("gptq8"),
        Some(QuantDtype::Int8) | Some(QuantDtype::Int4) | None => None,
    };
    if let Some(name) = unsupported {
        diagnostics.push(
            Diagnostic::error(format!("quant block `default: {name}` is not implemented"))
                .with_label(quant.span, QUANT_DTYPE_REFUSAL),
        );
    }
}

/// The one `@quantize(dtype = ...)` value codegen acts on.
pub const QUANTIZE_DTYPES: &[&str] = &["awq4"];

/// Validate `@quantize(...)` on a model: at most `dtype = "awq4"`, named.
pub fn validate_quantize_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(ref args) = deco.args else { return };
    for arg in args {
        let Some(name_sym) = arg.name else {
            diagnostics.push(
                Diagnostic::error("@quantize takes named arguments only".to_string())
                    .with_label(arg.span, "write `dtype = \"awq4\"`"),
            );
            continue;
        };
        let name = resolve_sym(name_sym);
        match name.as_str() {
            "dtype" => match &arg.value.kind {
                ExprKind::StringLiteral(s) if QUANTIZE_DTYPES.contains(&s.as_str()) => {}
                ExprKind::StringLiteral(s) => diagnostics.push(
                    Diagnostic::error(format!("@quantize: dtype \"{s}\" is not implemented"))
                        .with_label(arg.span, "the only implemented @quantize dtype is \"awq4\""),
                ),
                _ => diagnostics.push(
                    Diagnostic::error("@quantize: dtype must be a string literal".to_string())
                        .with_label(arg.span, "write `dtype = \"awq4\"`"),
                ),
            },
            "group_size" => diagnostics.push(
                Diagnostic::error("@quantize: `group_size` is not implemented".to_string())
                    .with_label(arg.span, "the group size was logged and never used; remove the argument"),
            ),
            _ => diagnostics.push(
                Diagnostic::error(format!("@quantize: unknown argument '{name}'"))
                    .with_label(arg.span, "@quantize accepts only `dtype = \"awq4\"`"),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::Level;

    fn errors(src: &str) -> Vec<String> {
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parsed.diagnostics.iter().all(|d| !matches!(d.level, Level::Error)),
            "parse errors: {:?}",
            parsed.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        let analysis = crate::analyze(&parsed.module, &mut interner);
        analysis
            .diagnostics
            .iter()
            .filter(|d| matches!(d.level, Level::Error))
            .map(|d| format!("{} | {}", d.message, d.labels.iter().map(|l| l.message.as_str()).collect::<Vec<_>>().join(" / ")))
            .collect()
    }

    const MODEL: &str = "model Tiny:\n    w: Tensor = zeros([4, 4])\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n\n";

    fn quant(body: &str) -> String {
        format!("{MODEL}let m = Tiny()\nquant static q from m:\n{body}\n")
    }

    #[test]
    fn int_quant_blocks_still_check_clean() {
        for dtype in ["int4", "int8"] {
            let e = errors(&quant(&format!("    default: {dtype}")));
            assert!(e.is_empty(), "{dtype}: {e:?}");
        }
    }

    #[test]
    fn awq_and_gptq_quant_dtypes_are_refused() {
        for dtype in ["awq4", "gptq4", "gptq8"] {
            let e = errors(&quant(&format!("    default: {dtype}")));
            assert_eq!(e.len(), 1, "{dtype}: {e:?}");
            assert!(e[0].contains(&format!("`default: {dtype}` is not implemented")), "{e:?}");
            assert!(e[0].contains("aborted") && e[0].contains("default: int4"), "{e:?}");
        }
    }

    #[test]
    fn a_calibration_section_is_refused() {
        let src = format!(
            "{MODEL}let m = Tiny()\nlet calib = zeros([8, 4])\nquant static q from m:\n    default: int4\n    calibration:\n        data: calib\n        samples: 16\n"
        );
        let e = errors(&src);
        assert_eq!(e.len(), 1, "{e:?}");
        assert!(e[0].contains("`calibration:` is not implemented"), "{e:?}");
        assert!(e[0].contains("never read") && e[0].contains("Remove the section"), "{e:?}");
    }

    fn decorated(args: &str) -> String {
        format!("@quantize{args}\n{MODEL}")
    }

    #[test]
    fn quantize_accepts_awq4_and_no_arguments() {
        for args in ["", "(dtype=\"awq4\")"] {
            let e = errors(&decorated(args));
            assert!(e.is_empty(), "{args}: {e:?}");
        }
    }

    #[test]
    fn quantize_refuses_what_codegen_never_read() {
        for (args, want) in [
            ("(dtype=\"int8\")", "dtype \"int8\" is not implemented"),
            ("(dtype=\"gptq4\")", "dtype \"gptq4\" is not implemented"),
            ("(dtype=4)", "dtype must be a string literal"),
            ("(group_size=64)", "`group_size` is not implemented"),
            ("(bits=4)", "unknown argument 'bits'"),
            ("(\"awq4\")", "named arguments only"),
        ] {
            let e = errors(&decorated(args));
            assert_eq!(e.len(), 1, "{args}: {e:?}");
            assert!(e[0].contains(want), "{args}: {e:?}");
        }
    }
}
