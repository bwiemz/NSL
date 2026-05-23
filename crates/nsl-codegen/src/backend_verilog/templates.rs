//! One Verilog emission function per HIR node kind. Mechanical — no decisions
//! beyond literal substitution into fixed templates.

use crate::hir::nodes::*;
use crate::hir::signals::SignalRef;

pub fn emit_signal_ref(s: &SignalRef) -> String {
    match s {
        SignalRef::Wire(w) => format!("_w{}", w.0),
        SignalRef::Register(r) => format!("_r{}", r.0),
        SignalRef::Port(name) => name.clone(),
        SignalRef::LocalParam(name) => name.clone(),
        // M57.1 wire-array realization (Task W2): indexed wire-array read.
        // Mechanical lowering — each IndexExpr variant has one fixed Verilog
        // form. Construction-time invariant (`indices.len() == dims.len()`)
        // is enforced by callers; emitter assumes well-formed HIR.
        SignalRef::WireArrayElement { array_name, indices } => {
            let idx_str = emit_index_exprs(indices);
            format!("{}{}", array_name, idx_str)
        }
        // M57.1 wire-array realization (Task W5): identical render shape to
        // WireArrayElement — both lower to `name[idx0][idx1]...`. The HIR
        // distinction (wire family vs const family) is preserved structurally
        // for declaration emission and snapshot fidelity.
        SignalRef::IndexedLocalParam { array_name, indices } => {
            let idx_str = emit_index_exprs(indices);
            format!("{}{}", array_name, idx_str)
        }
        // M57.1 wire-array realization (Task W5): Verilog indexed part-select.
        // Used to slice the `Port::Input("x_l<i>")` flat bus into per-element
        // WireArray cells via `assign x_l<i>_a[k] = x_l<i>[k*W +: W]`.
        SignalRef::PortBitSlice { name, base_bit, width } => {
            format!("{}[{} +: {}]", name, base_bit, width)
        }
        // M57.1 wire-array realization (Task W5): Verilog concat over an
        // array's elements (high-order first per packed-bus convention).
        // Used to drive tap ports and the final-layer `out` port.
        SignalRef::WireArrayConcat { array_name, n, fixed_index } => {
            let elems: Vec<String> = (0..*n).rev().map(|i| match fixed_index {
                Some(fix) => format!("{}[{}][{}]", array_name, i, fix),
                None => format!("{}[{}]", array_name, i),
            }).collect();
            format!("{{{}}}", elems.join(", "))
        }
        // M57.2: combinational read of a `RegArray` element. Renders
        // identically to `WireArrayElement` — `name[idx...]`.
        SignalRef::RegArrayElement { array_name, indices } => {
            format!("{}{}", array_name, emit_index_exprs(indices))
        }
    }
}

/// M57.1 wire-array realization helper: render an `IndexExpr` chain into
/// concatenated Verilog `[...]` subscripts. Shared between `WireArrayElement`,
/// `IndexedLocalParam`, and `AssignWireArrayElement`.
pub(crate) fn emit_index_exprs(indices: &[IndexExpr]) -> String {
    indices.iter()
        .map(|ix| match ix {
            IndexExpr::Literal(n) => format!("[{}]", n),
            IndexExpr::Genvar(name) => format!("[{}]", name),
            IndexExpr::GenvarPlus(name, k) => format!("[({} + {})]", name, k),
            IndexExpr::Reg(r) => format!("[_r{}]", r.0),
            IndexExpr::RegPlus(r, k) => format!("[(_r{} + {})]", r.0, k),
        })
        .collect()
}

pub fn emit_wire(w: &Wire) -> String {
    format!("wire signed [{}:0] _w{};", w.width - 1, w.id.0)
}

/// M57.1 wire-array realization (Task W2): module-scope multi-dim wire decl.
/// Mechanical lowering — emits `wire signed [W-1:0] name [0:dims[0]-1]...;`
/// (single space after name, no separator between dim brackets).
pub fn emit_wire_array(wa: &WireArray) -> String {
    let dims_str: String = wa.dims.iter()
        .map(|d| format!("[0:{}]", d - 1))
        .collect();
    format!("wire signed [{}:0] {} {};", wa.width - 1, wa.name, dims_str)
}

/// M57.1 wire-array realization (Task W4): drives one element of a WireArray.
/// Mechanical lowering — emits `assign {array_name}[idx0]...[idxN] = {src};`.
/// Each `IndexExpr` variant has one fixed Verilog form (same shape as the
/// read path in `emit_signal_ref` for `WireArrayElement`).
pub fn emit_assign_wire_array_element(a: &AssignWireArrayElement) -> String {
    let idx_str = emit_index_exprs(&a.indices);
    format!("assign {}{} = {};", a.array_name, idx_str, emit_signal_ref(&a.src))
}

pub fn emit_local_param(lp: &LocalParam) -> String {
    format!(
        "localparam signed [{}:0] {} = {}'sd{};",
        lp.width - 1, lp.name, lp.width, lp.value
    )
}

/// M57.1 wire-array realization (Task W5): emit a module-scope multi-dim
/// const array as `localparam signed [W-1:0] name [0:dim0-1]... = '{...};`.
/// For 2D arrays the literal renders as a nested SystemVerilog `'{ }`. For
/// 1D as a flat `'{ }`. Higher ranks panic (not exercised by v1 MLP).
///
/// Mechanical lowering — values flow verbatim through `{width}'sd{value}`,
/// matching `emit_local_param`'s signed-decimal encoding. Snapshot tests
/// elide localparam lines (`elide_localparams`) so the bulk of the W matrix
/// content doesn't dominate the structural-skeleton snapshot.
pub fn emit_local_param_array(lpa: &LocalParamArray) -> String {
    let dims_str: String = lpa.dims.iter()
        .map(|d| format!(" [0:{}]", d - 1))
        .collect();
    let values_str = match lpa.dims.len() {
        2 => {
            let (rows, cols) = (lpa.dims[0], lpa.dims[1]);
            assert_eq!(lpa.values.len(), rows * cols,
                "LocalParamArray.values len {} != rows({}) * cols({})",
                lpa.values.len(), rows, cols);
            let row_strs: Vec<String> = (0..rows).map(|r| {
                let cells: Vec<String> = (0..cols).map(|c| {
                    format!("{}'sd{}", lpa.width, lpa.values[r * cols + c])
                }).collect();
                format!("'{{{}}}", cells.join(", "))
            }).collect();
            format!("'{{{}}}", row_strs.join(", "))
        }
        1 => {
            assert_eq!(lpa.values.len(), lpa.dims[0],
                "LocalParamArray.values len {} != dim {}",
                lpa.values.len(), lpa.dims[0]);
            let cells: Vec<String> = lpa.values.iter()
                .map(|v| format!("{}'sd{}", lpa.width, v))
                .collect();
            format!("'{{{}}}", cells.join(", "))
        }
        n => panic!("emit_local_param_array: rank {} not supported in v1", n),
    };
    format!(
        "localparam signed [{}:0] {}{} = {};",
        lpa.width - 1, lpa.name, dims_str, values_str
    )
}

pub fn emit_mul(m: &Mul) -> String {
    format!(
        "assign _w{} = $signed({}) * $signed({});",
        m.out.0, emit_signal_ref(&m.a), emit_signal_ref(&m.b)
    )
}

pub fn emit_add(a: &Add) -> String {
    format!(
        "assign _w{} = $signed({}) + $signed({});",
        a.out.0, emit_signal_ref(&a.a), emit_signal_ref(&a.b)
    )
}

pub fn emit_max0(m: &Max0) -> String {
    format!(
        "assign _w{} = ($signed({}) > 0) ? {} : '0;",
        m.out.0, emit_signal_ref(&m.a), emit_signal_ref(&m.a)
    )
}

/// M57.2 (Task 3): combinational equality against a compile-time constant.
/// Emits `assign _w{out} = ($signed({lhs}) == {rhs});` (1-bit result).
pub fn emit_cmp_eq(c: &CmpEq) -> String {
    format!("assign _w{} = ($signed({}) == {});", c.out.0, emit_signal_ref(&c.lhs), c.rhs)
}

/// M57.2 (Task 4): combinational add of a compile-time constant.
/// Emits `assign _w{out} = $signed({src}) + {k};`.
pub fn emit_add_const(a: &AddConst) -> String {
    format!("assign _w{} = $signed({}) + {};", a.out.0, emit_signal_ref(&a.src), a.k)
}

/// M57.2 (Task 5): declaration-only scalar register.
/// Emits `reg signed [w-1:0] _r{id};` — no always_ff block.
pub fn emit_reg_decl(d: &RegDecl) -> String {
    format!("reg signed [{}:0] _r{};", d.width - 1, d.id.0)
}

/// M57.2 (Task 6): module-scope clocked register array — sequential sibling
/// of `WireArray`. Emits `reg signed [w-1:0] {name} [0:dims[0]-1]...;`.
pub fn emit_reg_array(ra: &RegArray) -> String {
    let dims_str: String = ra.dims.iter().map(|d| format!("[0:{}]", d - 1)).collect();
    format!("reg signed [{}:0] {} {};", ra.width - 1, ra.name, dims_str)
}

pub fn emit_sign_extend(s: &SignExtend) -> String {
    let pad = s.dst_width - s.src_width;
    let src = emit_signal_ref(&s.src);
    format!(
        "assign _w{} = {{{{{}{{{}[{}]}}}}, {}}};",
        s.dst.0, pad, src, s.src_width - 1, src
    )
}

// --- Sequential / clocked node emitters ---

/// M57.2 (Task 7): render a `SeqLValue` to a Verilog lvalue string.
/// Scalar register → `_r{id}`; array element → `name[idx...]`.
pub(crate) fn emit_seq_lvalue(lv: &crate::hir::nodes::SeqLValue) -> String {
    use crate::hir::nodes::SeqLValue;
    match lv {
        SeqLValue::Register(r) => format!("_r{}", r.0),
        SeqLValue::RegArrayElement { array_name, indices } => {
            format!("{}{}", array_name, emit_index_exprs(indices))
        }
    }
}

// --- Register lowering — all 4 dialect combinations ---

use crate::hir::clock_reset::{ResetPolarity, ResetSync};
use std::collections::BTreeMap;
use crate::hir::ids::{ClockDomainId, ResetSignalId};

pub fn emit_register(
    r: &Register,
    clock_domains: &BTreeMap<ClockDomainId, String>,
    reset_signals: &BTreeMap<ResetSignalId, String>,
) -> String {
    let clk_name = clock_domains.get(&r.clock.domain_id)
        .expect("clock domain name not registered");
    let rst_name = reset_signals.get(&r.reset.signal_id)
        .expect("reset signal name not registered");
    let d = emit_signal_ref(&r.d);
    let q = format!("_r{}", r.id.0);
    let w = r.width - 1;

    let rst_check = match r.reset.polarity {
        ResetPolarity::High => rst_name.clone(),
        ResetPolarity::Low => format!("!{}", rst_name),
    };

    let sensitivity = match r.reset.sync {
        ResetSync::Sync => format!("@(posedge {})", clk_name),
        ResetSync::Async => match r.reset.polarity {
            ResetPolarity::High => format!("@(posedge {} or posedge {})", clk_name, rst_name),
            ResetPolarity::Low  => format!("@(posedge {} or negedge {})", clk_name, rst_name),
        },
    };

    format!(
        "reg signed [{w}:0] {q};\n\
         always_ff {sensitivity} begin\n  \
             if ({rst_check}) {q} <= '0;\n  \
             else {q} <= {d};\n\
         end",
    )
}

// --- Port + GenerateFor + GenerateIf lowering ---

use crate::hir::control::*;

pub fn emit_port_input(p: &Port) -> String {
    match p {
        Port::Input { name, width } =>
            format!("input  wire signed [{}:0] {}", width - 1, name),
        _ => panic!("emit_port_input called with non-Input port"),
    }
}

pub fn emit_port_output_decl(p: &Port) -> String {
    match p {
        Port::Output { name, width, .. } =>
            format!("output wire signed [{}:0] {}", width - 1, name),
        _ => panic!("emit_port_output_decl called with non-Output port"),
    }
}

pub fn emit_port_output_assign(p: &Port) -> String {
    match p {
        Port::Output { name, driver, .. } =>
            format!("assign {} = {};", name, emit_signal_ref(driver)),
        _ => panic!("emit_port_output_assign called with non-Output port"),
    }
}

pub fn emit_generate_for(
    g: &GenerateFor,
    indent: usize,
    clock_domains: &BTreeMap<ClockDomainId, String>,
    reset_signals: &BTreeMap<ResetSignalId, String>,
) -> String {
    let pad = " ".repeat(indent);
    let inner = g.body.iter()
        .map(|n| emit_node(n, indent + 4, clock_domains, reset_signals))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{pad}generate for (genvar _gv{} = {}; _gv{} < {}; _gv{}++) begin : gen_{}\n\
         {inner}\n\
         {pad}end endgenerate",
        g.var.0, g.lo, g.var.0, g.hi, g.var.0, g.var.0,
    )
}

pub fn emit_generate_if(
    g: &GenerateIf,
    indent: usize,
    clock_domains: &BTreeMap<ClockDomainId, String>,
    reset_signals: &BTreeMap<ResetSignalId, String>,
) -> String {
    let pad = " ".repeat(indent);
    let cond_str = match &g.cond {
        ConstExpr::True => "1",
        ConstExpr::False => "0",
    };
    let then_str = g.then_body.iter()
        .map(|n| emit_node(n, indent + 4, clock_domains, reset_signals))
        .collect::<Vec<_>>().join("\n");
    let else_str = if g.else_body.is_empty() {
        String::new()
    } else {
        let inner = g.else_body.iter()
            .map(|n| emit_node(n, indent + 4, clock_domains, reset_signals))
            .collect::<Vec<_>>().join("\n");
        format!("\n{pad}else begin\n{inner}\n{pad}end")
    };
    format!("{pad}generate if ({cond_str}) begin\n{then_str}\n{pad}end{else_str} endgenerate")
}

/// Dispatch to the appropriate template for any HirNode.
pub fn emit_node(
    n: &crate::hir::module::HirNode,
    indent: usize,
    clock_domains: &BTreeMap<ClockDomainId, String>,
    reset_signals: &BTreeMap<ResetSignalId, String>,
) -> String {
    use crate::hir::module::HirNode;
    let pad = " ".repeat(indent);
    match n {
        HirNode::Wire(w) => format!("{pad}{}", emit_wire(w)),
        HirNode::Register(r) => {
            emit_register(r, clock_domains, reset_signals)
                .lines().map(|l| format!("{pad}{l}")).collect::<Vec<_>>().join("\n")
        }
        HirNode::Mul(m) => format!("{pad}{}", emit_mul(m)),
        HirNode::Add(a) => format!("{pad}{}", emit_add(a)),
        HirNode::Max0(m) => format!("{pad}{}", emit_max0(m)),
        HirNode::SignExtend(s) => format!("{pad}{}", emit_sign_extend(s)),
        HirNode::GenerateFor(g) => emit_generate_for(g, indent, clock_domains, reset_signals),
        HirNode::GenerateIf(g) => emit_generate_if(g, indent, clock_domains, reset_signals),
        // M57.1 wire-array realization (Task W2): module-scope multi-dim
        // wire-array declaration. Indented at the same level as plain Wire.
        HirNode::WireArray(wa) => format!("{pad}{}", emit_wire_array(wa)),
        // M57.1 wire-array realization (Task W4): drives one element of a
        // WireArray. Indented at the same level as plain `assign` ops (Mul,
        // Add, etc.).
        HirNode::AssignWireArrayElement(a) => {
            format!("{pad}{}", emit_assign_wire_array_element(a))
        }
        // M57.2 (Task 3): combinational equality against a compile-time constant.
        HirNode::CmpEq(c) => format!("{pad}{}", emit_cmp_eq(c)),
        // M57.2 (Task 4): combinational add of a compile-time constant.
        HirNode::AddConst(a) => format!("{pad}{}", emit_add_const(a)),
        // M57.2 (Task 5): declaration-only scalar register.
        HirNode::RegDecl(d) => format!("{pad}{}", emit_reg_decl(d)),
        // M57.2 (Task 6): module-scope clocked register array.
        HirNode::RegArray(ra) => format!("{pad}{}", emit_reg_array(ra)),
    }
}

// ---------------------------------------------------------------------------
// Unit tests — per-node templates
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::ids::WireId;

    #[test]
    fn emit_mul_i8() {
        let m = Mul {
            a: SignalRef::port("x"), b: SignalRef::port("w"),
            out: WireId(42), a_width: 8, b_width: 8, out_width: 16,
        };
        let v = emit_mul(&m);
        assert_eq!(v, "assign _w42 = $signed(x) * $signed(w);");
    }

    #[test]
    fn emit_add_i32() {
        let a = Add {
            a: SignalRef::port("acc_prev"), b: SignalRef::port("prod"),
            out: WireId(7), width: 32,
        };
        let v = emit_add(&a);
        assert_eq!(v, "assign _w7 = $signed(acc_prev) + $signed(prod);");
    }

    #[test]
    fn emit_max0_relu() {
        let m = Max0 {
            a: SignalRef::wire(WireId(5)), out: WireId(6), width: 32,
        };
        let v = emit_max0(&m);
        assert_eq!(v, "assign _w6 = ($signed(_w5) > 0) ? _w5 : '0;");
    }

    #[test]
    fn emit_sign_extend_i16_to_i32() {
        let s = SignExtend {
            src: SignalRef::wire(WireId(3)), dst: WireId(4),
            src_width: 16, dst_width: 32,
        };
        let v = emit_sign_extend(&s);
        // Pad 16 high bits with src[15]
        assert_eq!(v, "assign _w4 = {{16{_w3[15]}}, _w3};");
    }

    #[test]
    fn emit_local_param_i8() {
        let lp = LocalParam::new("W1_0_0", 8, -42);
        let v = emit_local_param(&lp);
        assert_eq!(v, "localparam signed [7:0] W1_0_0 = 8'sd-42;");
    }

    #[test]
    fn emit_wire_i32() {
        let w = Wire { id: WireId(9), width: 32 };
        let v = emit_wire(&w);
        assert_eq!(v, "wire signed [31:0] _w9;");
    }

    // --- M57.1 wire-array realization: emitter lowering (Task W2) ---

    #[test]
    fn emit_signal_ref_wire_array_element_literal_index() {
        let s = SignalRef::wire_array_element("acc_l1", vec![
            IndexExpr::Literal(5),
            IndexExpr::Literal(10),
        ]);
        assert_eq!(emit_signal_ref(&s), "acc_l1[5][10]");
    }

    #[test]
    fn emit_signal_ref_wire_array_element_genvar_index() {
        let s = SignalRef::wire_array_element("acc_l1", vec![
            IndexExpr::Genvar("_gv_o".to_string()),
            IndexExpr::Genvar("_gv_k".to_string()),
        ]);
        assert_eq!(emit_signal_ref(&s), "acc_l1[_gv_o][_gv_k]");
    }

    #[test]
    fn emit_signal_ref_wire_array_element_genvar_plus_index() {
        let s = SignalRef::wire_array_element("acc_l1", vec![
            IndexExpr::Genvar("_gv_o".to_string()),
            IndexExpr::GenvarPlus("_gv_k".to_string(), 1),
        ]);
        assert_eq!(emit_signal_ref(&s), "acc_l1[_gv_o][(_gv_k + 1)]");
    }

    #[test]
    fn emit_wire_array_2d() {
        let wa = WireArray {
            name: "acc_l1".to_string(),
            dims: vec![128, 785],
            width: 32,
        };
        assert_eq!(
            emit_wire_array(&wa),
            "wire signed [31:0] acc_l1 [0:127][0:784];"
        );
    }

    #[test]
    fn emit_wire_array_1d() {
        let wa = WireArray {
            name: "relu_l1".to_string(),
            dims: vec![128],
            width: 32,
        };
        assert_eq!(
            emit_wire_array(&wa),
            "wire signed [31:0] relu_l1 [0:127];"
        );
    }

    // --- M57.1 wire-array realization: AssignWireArrayElement emit (Task W4) ---

    #[test]
    fn emit_assign_wire_array_element_from_wire() {
        let a = AssignWireArrayElement {
            array_name: "acc_l1".to_string(),
            indices: vec![
                IndexExpr::Genvar("_gv_o".to_string()),
                IndexExpr::GenvarPlus("_gv_k".to_string(), 1),
            ],
            src: SignalRef::wire(WireId(42)),
        };
        assert_eq!(
            emit_assign_wire_array_element(&a),
            "assign acc_l1[_gv_o][(_gv_k + 1)] = _w42;"
        );
    }

    #[test]
    fn emit_assign_wire_array_element_from_local_param_literal_indices() {
        let a = AssignWireArrayElement {
            array_name: "acc_l1".to_string(),
            indices: vec![IndexExpr::Literal(5), IndexExpr::Literal(0)],
            src: SignalRef::local_param("b1_5"),
        };
        assert_eq!(
            emit_assign_wire_array_element(&a),
            "assign acc_l1[5][0] = b1_5;"
        );
    }

    #[test]
    fn index_expr_reg_and_regplus_emit() {
        use crate::hir::ids::RegisterId;
        let r = RegisterId(7);
        let s = emit_index_exprs(&[IndexExpr::Reg(r)]);
        assert_eq!(s, "[_r7]");
        let s2 = emit_index_exprs(&[IndexExpr::RegPlus(r, 1)]);
        assert_eq!(s2, "[(_r7 + 1)]");
    }

    #[test]
    fn signal_ref_reg_array_element_emits() {
        use crate::hir::ids::RegisterId;
        let s = SignalRef::reg_array_element("h_buf", vec![IndexExpr::Reg(RegisterId(2))]);
        assert_eq!(emit_signal_ref(&s), "h_buf[_r2]");
    }

    // --- M57.2 (Task 3): CmpEq combinational node ---

    #[test]
    fn cmp_eq_emits_equality() {
        use crate::hir::ids::{RegisterId, WireId};
        let c = CmpEq { lhs: SignalRef::Register(RegisterId(4)), rhs: 783, out: WireId(9) };
        assert_eq!(emit_cmp_eq(&c), "assign _w9 = ($signed(_r4) == 783);");
    }

    // --- M57.2 (Task 4): AddConst combinational node ---

    #[test]
    fn add_const_emits_increment() {
        use crate::hir::ids::{RegisterId, WireId};
        let a = AddConst { src: SignalRef::Register(RegisterId(4)), k: 1, out: WireId(12), width: 10 };
        assert_eq!(emit_add_const(&a), "assign _w12 = $signed(_r4) + 1;");
    }

    // --- M57.2 (Task 5): RegDecl declaration-only register ---

    #[test]
    fn reg_decl_emits_declaration_only() {
        use crate::hir::ids::RegisterId;
        let d = RegDecl { id: RegisterId(5), width: 64 };
        assert_eq!(emit_reg_decl(&d), "reg signed [63:0] _r5;");
    }

    // --- M57.2 (Task 6): RegArray clocked register array ---

    #[test]
    fn reg_array_emits_declaration() {
        let ra = RegArray { name: "x_buf".into(), dims: vec![784], width: 8 };
        assert_eq!(emit_reg_array(&ra), "reg signed [7:0] x_buf [0:783];");
    }

    // --- M57.2 (Task 7): SeqLValue ---

    #[test]
    fn seq_lvalue_emits() {
        use crate::hir::ids::RegisterId;
        use crate::hir::nodes::SeqLValue;
        assert_eq!(emit_seq_lvalue(&SeqLValue::Register(RegisterId(3))), "_r3");
        let e = SeqLValue::RegArrayElement {
            array_name: "h_buf".into(),
            indices: vec![IndexExpr::Reg(RegisterId(2))],
        };
        assert_eq!(emit_seq_lvalue(&e), "h_buf[_r2]");
    }
}

// ---------------------------------------------------------------------------
// Register dialect tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod register_tests {
    use super::*;

    fn default_name_tables() -> (BTreeMap<ClockDomainId, String>, BTreeMap<ResetSignalId, String>) {
        let mut cd = BTreeMap::new();
        cd.insert(ClockDomainId::DEFAULT, "clk".to_string());
        let mut rs = BTreeMap::new();
        rs.insert(ResetSignalId::DEFAULT, "rst".to_string());
        (cd, rs)
    }

    #[test]
    fn sync_active_high_v1_default() {
        let r = Register::new_v1_default(8, SignalRef::port("x"));
        let (cd, rs) = default_name_tables();
        let v = emit_register(&r, &cd, &rs);
        assert!(v.contains("@(posedge clk)"));
        assert!(v.contains("if (rst)"));
        assert!(!v.contains("posedge rst"));
    }

    #[test]
    fn sync_active_low_emits_negated_check() {
        let r = Register::new_active_low_for_test(8, SignalRef::port("x"));
        let (cd, rs) = default_name_tables();
        let v = emit_register(&r, &cd, &rs);
        assert!(v.contains("@(posedge clk)"));
        assert!(v.contains("if (!rst)"));
    }

    #[test]
    fn async_active_high_emits_or_posedge() {
        let r = Register::new_async_for_test(8, SignalRef::port("x"));
        let (cd, rs) = default_name_tables();
        let v = emit_register(&r, &cd, &rs);
        assert!(v.contains("@(posedge clk or posedge rst)"));
    }

    #[test]
    fn async_active_low_emits_or_negedge() {
        let r = Register::new_async_active_low_for_test(8, SignalRef::port("x"));
        let (cd, rs) = default_name_tables();
        let v = emit_register(&r, &cd, &rs);
        assert!(v.contains("@(posedge clk or negedge rst)"));
        assert!(v.contains("if (!rst)"));
    }

    #[test]
    fn alternate_clock_domain_uses_correct_name() {
        let alt_domain = ClockDomainId(42);
        let r = Register::new_with_clock_domain_for_test(
            8, SignalRef::port("x"), alt_domain,
        );
        let (mut cd, rs) = default_name_tables();
        cd.insert(alt_domain, "clk_io".to_string());
        let v = emit_register(&r, &cd, &rs);
        assert!(v.contains("@(posedge clk_io)"));
        assert!(!v.contains("posedge clk)"));   // not the default
    }
}
