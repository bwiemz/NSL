//! HIR → Verilog module-level emitter per spec §2.4.
//! Produces one `.v` file per HirModule. Mechanical — no optimization.

use crate::hir::module::HirModule;
use crate::backend_verilog::templates::*;

pub struct VerilogEmitter;

impl VerilogEmitter {
    pub fn emit_module(module: &HirModule) -> String {
        let mut out = String::new();

        // M57.2 (Task 14): prepend a deterministic-latency comment for sequential modules.
        if let Some(n) = module.cycle_count {
            out.push_str(&format!("// M57.2 sequential FSM — deterministic latency: {n} cycles\n"));
        }

        // Module header with port list
        out.push_str(&format!("module {}(\n", module.name));
        let inputs: Vec<String> = module.ports.iter()
            .filter_map(|p| match p {
                crate::hir::nodes::Port::Input { .. } => Some(emit_port_input(p)),
                _ => None,
            })
            .collect();
        let outputs: Vec<String> = module.ports.iter()
            .filter_map(|p| match p {
                crate::hir::nodes::Port::Output { .. } => Some(emit_port_output_decl(p)),
                _ => None,
            })
            .collect();
        let port_lines: Vec<String> = inputs.into_iter().chain(outputs).collect();
        out.push_str(
            &port_lines.join(",\n").lines().map(|l| format!("    {l}")).collect::<Vec<_>>().join("\n")
        );
        out.push_str("\n);\n\n");

        // LocalParam block
        for lp in module.local_params() {
            out.push_str(&format!("    {}\n", emit_local_param(lp)));
        }
        // M57.1 wire-array realization (Task W5): multi-dim const arrays
        // (`W<i>`, `b<i>`) emitted in the same preamble. Rendered after the
        // scalar LocalParams so existing snapshots are extensionally additive.
        for lpa in module.local_param_arrays() {
            out.push_str(&format!("    {}\n", emit_local_param_array(lpa)));
        }
        out.push('\n');

        // Body — nodes in SSA order
        for node in module.nodes() {
            out.push_str(&emit_node(node, 4, &module.clock_domains, &module.reset_signals));
            out.push('\n');
        }

        // Output port assignments
        for p in &module.ports {
            if let crate::hir::nodes::Port::Output { .. } = p {
                out.push_str(&format!("    {}\n", emit_port_output_assign(p)));
            }
        }

        out.push_str("endmodule\n");
        // Collapse packed multi-dim arrays → flat 1-D vectors + part-selects so
        // the output parses on the apt Yosys 0.33 pinned by CI (which rejects
        // multi-dim packed arrays). See `flatten` module.
        crate::backend_verilog::flatten::flatten_packed_arrays(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::nodes::*;

    #[test]
    fn empty_module_emits_minimal_skeleton() {
        let m = HirModule::new("empty");
        let v = VerilogEmitter::emit_module(&m);
        assert!(v.starts_with("module empty(\n"));
        assert!(v.contains("endmodule"));
    }

    #[test]
    fn module_with_port_emits_input_decl() {
        let mut m = HirModule::new("with_input");
        m.add_port(Port::input("x", 8));
        let v = VerilogEmitter::emit_module(&m);
        assert!(v.contains("input  wire signed [7:0] x"));
    }

    /// M57.2 (Task 14): sequential modules prepend the cycle-count comment;
    /// combinational modules (cycle_count None) start directly with `module`.
    #[test]
    fn emit_module_prepends_cycle_count_comment_when_present() {
        let mut m = HirModule::new("seqmod");
        m.cycle_count = Some(101_634);
        let v = VerilogEmitter::emit_module(&m);
        assert!(
            v.starts_with("// M57.2 sequential FSM \u{2014} deterministic latency: 101634 cycles\n"),
            "sequential module must start with cycle-count comment; got: {v:?}",
        );
        // Combinational module (cycle_count None) starts with `module <name>(`
        let c = HirModule::new("combmod");
        let vc = VerilogEmitter::emit_module(&c);
        assert!(
            vc.starts_with("module combmod("),
            "combinational module must start with module header; got: {vc:?}",
        );
    }
}
