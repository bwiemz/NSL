//! P1 Muon item 6: parameter-ROLE classification for mixed Muon/AdamW routing.
//!
//! Replaces the name-substring exclusion list (embed/lm_head/unembed/wte/wpe/
//! vocab) that routed params to AdamW by accident of naming — a hidden weight
//! named `embed_proj` was silently excluded from Muon, and an embedding named
//! `tok_table` silently took the orthogonalized update. Roles are now
//! determined per parameter, in priority order:
//!
//! 1. **Explicit decorator** — `@param_role("embedding"|"head"|"hidden")` on
//!    the model tensor field (collected in `collection.rs`, invalid values
//!    are compile errors). Authoritative; wins over inference.
//! 2. **Structural inference** — a field that appears as the TABLE argument
//!    of an `embedding_lookup(self.field, ids)` call in any model method
//!    body is role `embedding`. Weight-tied LM heads (the coder models'
//!    `x @ self.embed.transpose(0, 1)`) are the same tensor as the
//!    embedding, so tying needs no separate head detection.
//! 3. **Declared rank** — a field whose declared tensor rank
//!    (`model_field_ranks`) is not 2 is role `vector` (biases, norms,
//!    scalars). The runtime rank-2 check in `muon_step` remains the
//!    backstop for undeclared ranks.
//! 4. **Default** — everything else is role `hidden` (the Muon route).
//!
//! Routing: `embedding`, `head`, and `vector` take the AdamW arm; `hidden`
//! takes Muon. An UNTIED lm_head cannot be structurally identified in v1 —
//! the routing table says so loudly and `@param_role("head")` covers it.

use std::collections::HashSet;

use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::visitor::{walk_expr, Visitor};

use crate::compiler::Compiler;

/// One classified parameter, parallel to the `param_paths` order.
pub(crate) struct MuonRouteEntry {
    pub path: String,
    /// "embedding" | "head" | "hidden" | "vector"
    pub role: &'static str,
    /// Human-readable provenance for the routing table print.
    pub source: &'static str,
    /// true → force the AdamW arm (flag 1 in the runtime route list).
    pub adamw: bool,
}

pub(crate) struct MuonRouteTable {
    pub entries: Vec<MuonRouteEntry>,
    pub warnings: Vec<String>,
}

/// AST scan for `embedding_lookup(self.<field>, ...)` inside one model
/// type's method bodies. Records attributable table fields; flags table
/// arguments the scan cannot see through (aliased locals, sub-expressions)
/// so the routing table can demand an explicit `@param_role`.
struct EmbedTableScan<'r> {
    resolve: &'r dyn Fn(nsl_ast::Symbol) -> String,
    /// Field names (within the scanned model type) used as lookup tables.
    tables: HashSet<String>,
    /// A lookup whose table arg was not a direct `self.<field>` reference.
    unattributed: bool,
}

impl Visitor for EmbedTableScan<'_> {
    fn visit_expr(&mut self, expr: &Expr) {
        if let ExprKind::Call { callee, args } = &expr.kind {
            let callee_is_lookup = match &callee.kind {
                ExprKind::Ident(sym) => (self.resolve)(*sym) == "embedding_lookup",
                // Method form: `table.embedding_lookup(ids)` — the object IS
                // the table argument.
                ExprKind::MemberAccess { member, .. } => {
                    (self.resolve)(*member) == "embedding_lookup"
                }
                _ => false,
            };
            if callee_is_lookup {
                let table_expr: Option<&Expr> = match &callee.kind {
                    ExprKind::MemberAccess { object, .. } => Some(object),
                    _ => args.first().map(|a| &a.value),
                };
                match table_expr.map(|e| &e.kind) {
                    Some(ExprKind::MemberAccess { object, member })
                        if matches!(object.kind, ExprKind::SelfRef)
                            || matches!(&object.kind, ExprKind::Ident(s)
                                if (self.resolve)(*s) == "self") =>
                    {
                        self.tables.insert((self.resolve)(*member));
                    }
                    _ => {
                        self.unattributed = true;
                    }
                }
            }
        }
        walk_expr(self, expr);
    }
}

impl Compiler<'_> {
    /// Resolve a runtime param path (`m.blocks.0.attn.wq`) to its OWNING
    /// model type and field name (`("GroupedQueryAttention", "wq")`) by
    /// walking `model_field_types` from the root model type. Numeric
    /// components consume one `[T; N]` fixed-array element hop. Returns
    /// None when the walk dead-ends (path format the type maps can't see).
    fn resolve_param_owner(&self, root_type: &str, path: &str) -> Option<(String, String)> {
        let comps: Vec<&str> = path.split('.').collect();
        if comps.len() < 2 {
            return None;
        }
        let mut current = root_type.to_string();
        let mut i = 1; // skip the model variable name
        while i < comps.len() {
            let comp = comps[i];
            if i == comps.len() - 1 {
                return Some((current, comp.to_string()));
            }
            let field_ty = self
                .models
                .model_field_types
                .get(&current)
                .and_then(|f| f.get(comp))?
                .clone();
            if field_ty.starts_with('[') && field_ty.contains(';') {
                // `[T; N]` — the next component must be the element index.
                let elem = field_ty
                    .trim_start_matches('[')
                    .split(';')
                    .next()?
                    .trim()
                    .to_string();
                current = elem;
                i += 2; // consume field + index
            } else {
                current = field_ty;
                i += 1;
            }
        }
        None
    }

    /// Classify every trainable param for mixed Muon/AdamW routing. Order
    /// (and length) of `entries` matches `param_paths` exactly — the caller
    /// builds the runtime flag list positionally from it.
    pub(crate) fn classify_muon_param_roles(
        &self,
        root_type: &str,
        param_paths: &[String],
    ) -> MuonRouteTable {
        // Pass 1: embedding-table usage over ALL model types' method bodies.
        // Scanning types not reachable from the root is harmless — their
        // (type, field) keys never match a resolved owner.
        let mut embed_tables: HashSet<(String, String)> = HashSet::new();
        let mut warnings: Vec<String> = Vec::new();
        let resolve = |sym: nsl_ast::Symbol| self.resolve_sym(sym).to_string();
        for (type_name, methods) in &self.models.model_method_bodies {
            for (method_name, fn_def) in methods {
                let mut scan = EmbedTableScan {
                    resolve: &resolve,
                    tables: HashSet::new(),
                    unattributed: false,
                };
                scan.visit_block(&fn_def.body);
                if scan.unattributed {
                    warnings.push(format!(
                        "{type_name}.{method_name} calls embedding_lookup with a \
                         table argument that is not a direct self.<field> \
                         reference — role inference cannot attribute it; if the \
                         table is a trainable param, annotate its field with \
                         @param_role(\"embedding\")"
                    ));
                }
                for field in scan.tables {
                    embed_tables.insert((type_name.clone(), field));
                }
            }
        }

        // Pass 2: per-param classification.
        let mut entries = Vec::with_capacity(param_paths.len());
        for path in param_paths {
            let owner = self.resolve_param_owner(root_type, path);
            let decorated = owner.as_ref().and_then(|(ty, field)| {
                self.models
                    .model_field_roles
                    .get(ty)
                    .and_then(|f| f.get(field))
                    .map(String::as_str)
            });
            let (role, source): (&'static str, &'static str) = if let Some(r) = decorated {
                match r {
                    "embedding" => ("embedding", "@param_role"),
                    "head" => ("head", "@param_role"),
                    _ => ("hidden", "@param_role"),
                }
            } else if owner
                .as_ref()
                .is_some_and(|key| embed_tables.contains(&(key.0.clone(), key.1.clone())))
            {
                ("embedding", "embedding_lookup usage")
            } else if owner.as_ref().is_some_and(|(ty, field)| {
                self.models
                    .model_field_ranks
                    .get(ty)
                    .and_then(|f| f.get(field))
                    .is_some_and(|&rank| rank != 2)
            }) {
                ("vector", "declared rank != 2")
            } else {
                if owner.is_none() {
                    warnings.push(format!(
                        "param {path}: owner type could not be resolved through \
                         model_field_types — role defaults to hidden (Muon); \
                         annotate with @param_role if this is an embedding or head"
                    ));
                }
                ("hidden", "default")
            };
            entries.push(MuonRouteEntry {
                path: path.clone(),
                role,
                source,
                adamw: matches!(role, "embedding" | "head" | "vector"),
            });
        }
        MuonRouteTable { entries, warnings }
    }
}

#[cfg(test)]
mod tests {
    // The classification passes are exercised end-to-end by
    // crates/nsl-cli/tests/muon_optimizer_gate.rs (routing-table asserts on
    // real models); resolve_param_owner's array hop is covered there via
    // the `[TransformerBlock; N]` paths of the mixed-routing fixtures.
}
