//! Flatten packed multi-dim wire declarations + element references into 1-D
//! packed vectors with indexed part-selects.
//!
//! ## Why
//!
//! Yosys's built-in Verilog frontend rejects *unpacked* arrays
//! (`wire signed [W-1:0] name [0:N];`). The emitter therefore renders packed
//! multi-dim wires (`wire signed [d0-1:0]...[W-1:0] name;`). But the apt Yosys
//! pinned by CI (Ubuntu 24.04 → **Yosys 0.33**) *also* rejects multi-dim
//! *packed* arrays ("syntax error, unexpected '[', expecting TOK_ID"); only
//! newer Yosys (≥ ~0.36) accepts them. To parse on 0.33 we collapse every
//! multi-dim packed array to a flat 1-D vector and rewrite each `name[i][j]`
//! read/assign to an indexed part-select `name[(i*cols + j)*W +: W]`.
//!
//! This is a post-emit textual pass (not an emitter-internal change) so the
//! per-node templates stay simple; the forms it rewrites are fully regular.
//!
//! ## Layout
//!
//! For a packed `wire signed [rows-1:0][cols-1:0][W-1:0] name`, element
//! `[i][j][b]` lives at bit `i*cols*W + j*W + b`, so `name[i][j]` is
//! `name[(i*cols + j)*W +: W]` — bit-identical to the packed-array element
//! select. Declarations become `wire signed [rows*cols*W-1:0] name;`. The
//! flat weight-table `assign name = {…};` concat (already MSB-first, see
//! `emit_local_param_array`) is unchanged: same total width, same bit order.

use std::collections::HashMap;

use regex::Regex;

/// Rewrite all packed multi-dim `wire` declarations and their `name[i]...[k]`
/// element references into flat 1-D vectors + `[base +: W]` part-selects.
/// Scalar wires (single packed dim) and ports are left untouched.
pub fn flatten_packed_arrays(verilog: &str) -> String {
    // A module-scope wire declaration: `<indent>wire signed [A:0][B:0]... name;`.
    // Ports (`input`/`output` prefixed, comma-terminated) don't match — they are
    // single-dim and need no flattening.
    let decl_re = Regex::new(r"(?m)^(\s*)wire\s+signed\s+((?:\[\d+:0\])+)\s+(\w+);[ \t]*$").unwrap();
    let dim_re = Regex::new(r"\[(\d+):0\]").unwrap();

    let parse_dims = |dims_str: &str| -> Vec<usize> {
        dim_re
            .captures_iter(dims_str)
            .map(|c| c[1].parse::<usize>().unwrap() + 1)
            .collect()
    };

    // Pass 1: learn each array's (array_dims, elem_width). The LAST packed dim
    // is the element width; the rest are array dims. Only >= 2 dims = an array.
    let mut layouts: HashMap<String, (Vec<usize>, usize)> = HashMap::new();
    for cap in decl_re.captures_iter(verilog) {
        let dims = parse_dims(&cap[2]);
        if dims.len() >= 2 {
            let elem_width = *dims.last().unwrap();
            let array_dims = dims[..dims.len() - 1].to_vec();
            layouts.insert(cap[3].to_string(), (array_dims, elem_width));
        }
    }

    // Pass 2: flatten the declarations.
    let mut out = decl_re
        .replace_all(verilog, |cap: &regex::Captures| {
            let dims = parse_dims(&cap[2]);
            if dims.len() >= 2 {
                let total: usize = dims.iter().product();
                format!("{}wire signed [{}:0] {};", &cap[1], total - 1, &cap[3])
            } else {
                cap[0].to_string()
            }
        })
        .into_owned();

    // Pass 3: rewrite element references. Arrays with more dims first so a 2-D
    // `name[i][j]` is rewritten before any 1-D pass could match `name[i]`.
    let mut names: Vec<&String> = layouts.keys().collect();
    names.sort_by(|a, b| {
        layouts[*b]
            .0
            .len()
            .cmp(&layouts[*a].0.len())
            .then_with(|| a.cmp(b)) // deterministic tie-break
    });
    for name in names {
        let (array_dims, w) = &layouts[name];
        let esc = regex::escape(name);
        match array_dims.len() {
            2 => {
                let cols = array_dims[1];
                let re = Regex::new(&format!(r"\b{esc}\[([^\]]+)\]\[([^\]]+)\]")).unwrap();
                out = re
                    .replace_all(&out, |c: &regex::Captures| {
                        format!("{name}[(({}) * {cols} + ({})) * {w} +: {w}]", &c[1], &c[2])
                    })
                    .into_owned();
            }
            1 => {
                let re = Regex::new(&format!(r"\b{esc}\[([^\]]+)\]")).unwrap();
                out = re
                    .replace_all(&out, |c: &regex::Captures| {
                        format!("{name}[({}) * {w} +: {w}]", &c[1])
                    })
                    .into_owned();
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattens_2d_decl_and_refs() {
        // rows=2, cols=3, W=8 → total 48 bits; name[i][j] → [(i*3+j)*8 +: 8].
        let v = "    wire signed [1:0][2:0][7:0] W1;\n    assign _w0 = $signed(W1[1][2]);\n";
        let out = flatten_packed_arrays(v);
        assert!(out.contains("wire signed [47:0] W1;"), "decl flattened: {out}");
        assert!(
            out.contains("W1[((1) * 3 + (2)) * 8 +: 8]"),
            "2-D ref flattened: {out}"
        );
    }

    #[test]
    fn flattens_1d_decl_and_genvar_ref() {
        // n=4, W=8 → 32 bits; relu[_gv3] → relu[(_gv3)*8 +: 8].
        let v = "    wire signed [3:0][7:0] relu_l1;\n    assign relu_l1[_gv3] = _w4;\n";
        let out = flatten_packed_arrays(v);
        assert!(out.contains("wire signed [31:0] relu_l1;"), "1-D decl: {out}");
        assert!(out.contains("relu_l1[(_gv3) * 8 +: 8]"), "1-D genvar ref: {out}");
    }

    #[test]
    fn leaves_scalar_wires_and_ports_untouched() {
        let v = "    input  wire signed [6271:0] x_l1,\n    wire signed [15:0] _w1;\n    assign _w1 = x_l1[0 +: 8];\n";
        let out = flatten_packed_arrays(v);
        assert!(out.contains("input  wire signed [6271:0] x_l1,"), "port kept: {out}");
        assert!(out.contains("wire signed [15:0] _w1;"), "scalar kept: {out}");
        // x_l1 is a port (not in the array map) → its part-select is untouched.
        assert!(out.contains("_w1 = x_l1[0 +: 8];"), "port slice kept: {out}");
    }
}
