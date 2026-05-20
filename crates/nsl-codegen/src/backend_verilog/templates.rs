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
        // M57.1 wire-array realization: variant added in Task W1 (HIR
        // primitive). Verilog emission for indexed wire-array reads lands
        // in a subsequent W-task (templates extension). Until then, no
        // codegen pass should produce this variant; reaching here means a
        // pass started emitting wire-array refs before the templates were
        // taught how.
        SignalRef::WireArrayElement { .. } => {
            unreachable!(
                "SignalRef::WireArrayElement emission not yet implemented \
                 (M57.1 wire-array realization Task W1 added the HIR \
                 primitive; Verilog template support arrives in a follow-up \
                 W-task)"
            )
        }
    }
}

pub fn emit_wire(w: &Wire) -> String {
    format!("wire signed [{}:0] _w{};", w.width - 1, w.id.0)
}

pub fn emit_local_param(lp: &LocalParam) -> String {
    format!(
        "localparam signed [{}:0] {} = {}'sd{};",
        lp.width - 1, lp.name, lp.width, lp.value
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

pub fn emit_sign_extend(s: &SignExtend) -> String {
    let pad = s.dst_width - s.src_width;
    let src = emit_signal_ref(&s.src);
    format!(
        "assign _w{} = {{{{{}{{{}[{}]}}}}, {}}};",
        s.dst.0, pad, src, s.src_width - 1, src
    )
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
        // M57.1 wire-array realization: variant added in Task W1 (HIR
        // primitive). Verilog emission for `wire signed [W-1:0] name [..][..];`
        // declarations lands in a subsequent W-task (templates extension).
        // Until then, no codegen pass should produce this node; reaching here
        // means a pass started emitting wire-array decls before the templates
        // were taught how.
        HirNode::WireArray(_) => {
            unreachable!(
                "HirNode::WireArray emission not yet implemented \
                 (M57.1 wire-array realization Task W1 added the HIR \
                 primitive; Verilog template support arrives in a follow-up \
                 W-task)"
            )
        }
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
