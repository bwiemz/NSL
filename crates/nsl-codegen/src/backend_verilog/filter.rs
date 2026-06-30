//! localparam-elision filter for Layer 1 structural-skeleton snapshots.
//! Drops `localparam signed [..] name = NN'sd<value>;` lines so the snapshot
//! captures structure (port list, generate-loop skeleton, registers, etc.)
//! without the 102K-line weight content (spec §2.6 Correction 1 / §7.1).
//!
//! M57.1 wire-array realization (Task W5):
//! - drops 2D/1D LocalParamArray lines like
//!   `localparam signed [7:0] W1 [0:783][0:127] = '{...};` — the
//!   `'{...}` literal is too wide to be human-auditable.
//! - collapses runs of per-element `assign acc_l<i>[<n>][0] = …` bias-seed
//!   assigns + `assign x_l<i>_a[<n>] = x_l<i>[<bit> +: <w>]` fan-out
//!   assigns into a single `… elided` marker line so the structural
//!   skeleton stays focused on the generate-for + concat shape.

use regex::Regex;

pub fn elide_localparams(verilog: &str) -> String {
    let scalar = Regex::new(
        r"^\s*localparam\s+signed\s+\[\d+:0\]\s+\w+\s*=\s*\d+'sd-?\d+;\s*$",
    ).unwrap();
    // Match `localparam signed [..] name [0:..]...[0:..] = '{...};` — the
    // SystemVerilog array form. The `'{}` literal can span the rest of the
    // line; we anchor only on the leading shape and terminating `};`.
    let array = Regex::new(
        r"^\s*localparam\s+signed\s+\[\d+:0\]\s+\w+(?:\s*\[0:\d+\])+\s*=\s*'\{.*\};\s*$",
    ).unwrap();
    // M57.1 wire-array mini §3.2: per-element fan-out / bias-seed assigns
    // (e.g. `assign x_l1_a[7] = x_l1[56 +: 8];` or
    // `assign acc_l1[7][0] = b1[7];`). Anchored by an integer-only second
    // bracket so the genvar ripple write `assign acc_l1[_gv1][(_gv2 + 1)]`
    // is NOT matched.
    let element_assign = Regex::new(
        r"^\s*assign\s+\w+\[\d+\](?:\[\d+\])*\s*=\s*\S.*;\s*$",
    ).unwrap();
    let mut elements_collapsed = 0usize;
    let mut prev_was_element = false;
    let mut result: Vec<String> = Vec::new();
    for line in verilog.lines() {
        if scalar.is_match(line) || array.is_match(line) {
            continue;
        }
        if element_assign.is_match(line) {
            elements_collapsed += 1;
            prev_was_element = true;
            continue;
        }
        if prev_was_element {
            // Emit a single marker for the just-closed run of per-element
            // assigns. We keep the count so reviewers can sanity-check
            // cardinality without reading every line.
            let indent = "    ";
            result.push(format!(
                "{indent}// ({} per-element assigns elided)",
                elements_collapsed,
            ));
            elements_collapsed = 0;
            prev_was_element = false;
        }
        result.push(line.to_string());
    }
    if prev_was_element {
        let indent = "    ";
        result.push(format!(
            "{indent}// ({} per-element assigns elided)",
            elements_collapsed,
        ));
    }
    result.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elides_signed_decimal_localparam() {
        let input = "    localparam signed [7:0] W1_0_0 = 8'sd-42;\nmodule foo();";
        let out = elide_localparams(input);
        assert!(!out.contains("W1_0_0"));
        assert!(out.contains("module foo"));
    }

    #[test]
    fn preserves_non_data_localparams() {
        // Hypothetical future non-numeric localparam (string concat, etc.)
        let input = "    localparam string FOO = \"bar\";";
        let out = elide_localparams(input);
        assert!(out.contains("FOO"));
    }

    #[test]
    fn elides_negative_values() {
        let input = "    localparam signed [31:0] b1_3 = 32'sd-12345;";
        let out = elide_localparams(input);
        assert_eq!(out.trim(), "");
    }

    #[test]
    fn elides_2d_localparam_array() {
        let input = "    localparam signed [7:0] W1 [0:783][0:127] = '{'{8'sd0, 8'sd1}, '{8'sd2, 8'sd3}};";
        let out = elide_localparams(input);
        assert!(!out.contains("W1"), "2D array localparam should be elided");
    }

    #[test]
    fn elides_1d_localparam_array() {
        let input = "    localparam signed [31:0] b1 [0:127] = '{32'sd0, 32'sd1};";
        let out = elide_localparams(input);
        assert!(!out.contains("b1"), "1D array localparam should be elided");
    }
}
