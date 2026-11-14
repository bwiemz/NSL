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
//! - **`@quantize`.** Refused in every form. Codegen read `dtype` (only
//!   `"awq4"`, which marked the model for an AWQ activation-statistics
//!   pass whose output nothing consumed) and nothing else; that pass is
//!   deleted, and no dtype of the decorator has an implementation.
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

/// What to write instead of `@quantize`: the two things that do work.
const QUANTIZE_ALTERNATIVES: &str = "Remove the decorator. To quantize weights, use a `quant` block \
     (`default: int4` or `default: int8`); for FP8 matmuls, use `@fp8_compute`";

/// Why `@quantize(dtype = "awq4")` (and bare `@quantize`, which meant the
/// same) is refused.
///
/// The decorator read as "quantize this model with AWQ" and changed no
/// weight. It selected the model's linear projections for an
/// activation-statistics pass (and had retention copies compiled into the
/// model's forward for it) whose output, per-channel activation maxima in a
/// sidecar file, no quantizer read. A warning on a decorator that does
/// nothing is the same defect as a report that claims an effect the program
/// does not have, so the pass is deleted and the spelling refuses. `awq4`
/// stays reserved: it is refused here and as a `quant` block default.
pub const QUANTIZE_AWQ_REFUSAL: &str = "AWQ calibration is not implemented: this decorator selected the model \
     for an activation-statistics pass whose output no quantizer read, so it had no effect on the model";

/// Refuse `@quantize(...)` on a model. No dtype of it is implemented:
/// nothing in the compiler reads the decorator, so every form is an error
/// that says so and points at what does work.
pub fn validate_quantize_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    // The dtype the decorator names, if it names one as a string; bare
    // `@quantize` historically meant "awq4".
    let named_dtype: Option<String> = deco.args.as_ref().and_then(|args| {
        args.iter().find_map(|arg| {
            let name = resolve_sym(arg.name?);
            match (&arg.value.kind, name.as_str()) {
                (ExprKind::StringLiteral(s), "dtype") => Some(s.clone()),
                _ => None,
            }
        })
    });
    let diagnostic = match named_dtype.as_deref() {
        None | Some("awq4") => Diagnostic::error("@quantize(dtype=\"awq4\"): AWQ calibration is not implemented".to_string())
            .with_label(deco.span, QUANTIZE_AWQ_REFUSAL)
            .with_note(QUANTIZE_ALTERNATIVES),
        Some(other) => Diagnostic::error(format!("@quantize: dtype \"{other}\" is not implemented"))
            .with_label(deco.span, "@quantize has no implemented dtype: nothing in the compiler reads the decorator")
            .with_note(QUANTIZE_ALTERNATIVES),
    };
    diagnostics.push(diagnostic);
}

#[cfg(test)]
mod tests {
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

    /// Every form of the decorator is one error. `awq4` and the bare form
    /// (which meant awq4) say AWQ calibration is not implemented; any other
    /// dtype says that dtype is not implemented; both point at the `quant`
    /// block, which does quantize, and at `@fp8_compute`.
    ///
    /// This replaces the tests of the deleted AWQ calibration hook and its
    /// pipeline: the decorator used to be accepted (later with a warning)
    /// and to drive a calibration pass whose output nothing read.
    #[test]
    fn quantize_is_refused_in_every_form() {
        for (args, want) in [
            ("", "@quantize(dtype=\"awq4\"): AWQ calibration is not implemented"),
            ("(dtype=\"awq4\")", "@quantize(dtype=\"awq4\"): AWQ calibration is not implemented"),
            ("(dtype=\"awq4\", group_size=128)", "AWQ calibration is not implemented"),
            ("(group_size=64)", "AWQ calibration is not implemented"),
            ("(dtype=\"int8\")", "@quantize: dtype \"int8\" is not implemented"),
            ("(dtype=\"gptq4\")", "@quantize: dtype \"gptq4\" is not implemented"),
            ("(dtype=\"fp8\")", "@quantize: dtype \"fp8\" is not implemented"),
        ] {
            let e = errors(&decorated(args));
            assert_eq!(e.len(), 1, "{args}: {e:?}");
            assert!(e[0].contains(want), "{args}: {e:?}");
        }
        // The undecorated model is clean, so the error is the decorator's.
        assert!(errors(MODEL).is_empty());
    }

    /// The refusal names alternatives that exist: a `quant` block with
    /// int4 or int8, which the next assertion shows checks clean.
    #[test]
    fn the_refusal_points_at_a_quant_block_that_checks() {
        let mut interner = nsl_lexer::Interner::new();
        let src = decorated("(dtype=\"awq4\")");
        let (tokens, _) = nsl_lexer::tokenize(&src, nsl_errors::FileId(0), &mut interner);
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        let analysis = crate::analyze(&parsed.module, &mut interner);
        let refusal = analysis
            .diagnostics
            .iter()
            .find(|d| d.message.contains("AWQ calibration is not implemented"))
            .expect("the refusal");
        let notes = refusal.notes.join(" ");
        assert!(notes.contains("`quant` block") && notes.contains("default: int4"), "{notes}");
        assert!(notes.contains("@fp8_compute"), "{notes}");
        assert!(errors(&quant("    default: int4")).is_empty());
    }
}
