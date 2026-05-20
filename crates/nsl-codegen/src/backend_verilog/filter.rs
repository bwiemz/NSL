//! localparam-elision filter for Layer 1 structural-skeleton snapshots.
//! Drops `localparam signed [..] name = NN'sd<value>;` lines so the snapshot
//! captures structure (port list, generate-loop skeleton, registers, etc.)
//! without the 102K-line weight content (spec §2.6 Correction 1 / §7.1).

use regex::Regex;

pub fn elide_localparams(verilog: &str) -> String {
    let re = Regex::new(r"^\s*localparam\s+signed\s+\[\d+:0\]\s+\w+\s*=\s*\d+'sd-?\d+;\s*$").unwrap();
    verilog.lines()
        .filter(|line| !re.is_match(line))
        .collect::<Vec<_>>()
        .join("\n")
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
}
