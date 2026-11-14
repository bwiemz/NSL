//! `-w`: the program runs with its weights file's values.
//!
//! See the section "Binding a program to its weights file" in
//! `weight_aware.rs` for why. This module is the code generation for it:
//!
//! - every model constructor copies the file's tensors into the parameters
//!   the file names ([`Compiler::emit_weight_file_loads`]);
//! - a value-dependent rewrite asks [`Compiler::may_specialize_on`] first;
//! - `main` opens the file before any user code and checks the content hash
//!   of every tensor the compile may have specialized on
//!   ([`Compiler::emit_weight_file_bind`]).
//!
//! # Which tensor a parameter is loaded from
//!
//! In the constructor of a model `M`, each parameter under the new instance
//! is looked up by:
//!
//! - for a field `f` of `M` itself: `M.f`, then `f`;
//! - for a parameter of a nested model: its path from the instance, as
//!   `model_save` spells it (`blocks[0].w`) or dotted (`blocks.0.w`).
//!
//! A nested model's own constructor has already run, so the enclosing
//! model's path-named tensor is the one that stays.
//!
//! # Which of them a rewrite may depend on
//!
//! The rewrites key on `M.f` or `f` at an access `self.f`, which stands for
//! every instance of `M`. That is the loaded value only when one tensor can
//! reach the field: no other tensor of the file ends in `.f` (a path-named
//! one would overwrite some instances), and for a bare `f`, no other model
//! has a tensor field `f` (the file would not say whose it is; such a load
//! is made only where the shapes agree, so it is not known here whether it
//! happens).

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use crate::compiler::Compiler;
use crate::context::FieldKind;
use crate::error::CodegenError;

/// The field a tensor name ends in: `w` for `w`, `Toy.w`, `blocks[0].w` and
/// `blocks.0.w`.
fn leaf(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

impl Compiler<'_> {
    /// Why the file's tensor `key` (`Model.field` or `field`) is not known
    /// to be what every instance of the field holds; `None` when it is.
    fn weight_key_ambiguity(&self, key: &str) -> Option<String> {
        let stored = &self.features.weight_binding.as_ref()?.stored;
        if let Some(other) = stored
            .keys()
            .find(|name| *name != key && leaf(name) == leaf(key))
        {
            return Some(format!(
                "the file also names '{other}', which may be loaded into the same field"
            ));
        }
        if key.contains('.') {
            return None;
        }
        let owners = self
            .types
            .struct_layouts
            .values()
            .filter(|layout| {
                layout.fields.iter().any(|f| {
                    f.name == key && matches!(f.kind, FieldKind::Tensor | FieldKind::Buffer)
                })
            })
            .count();
        (owners > 1).then(|| {
            format!(
                "{owners} models have a tensor field named '{key}' and the file does not say \
                 which one its '{key}' belongs to (name the tensor 'Model.{key}')"
            )
        })
    }

    /// Whether `rewrite` may assume the parameter `key` holds the file's
    /// values when the program runs. It may when the program is bound to the
    /// file, the file stores a tensor of that name, nothing in the program
    /// writes the parameter after it is loaded, no other tensor can be
    /// loaded into the field, and the compiler's copy of the tensor is the
    /// file's (unless the user accepted an approximation, see
    /// [`Self::weight_value_tolerance`]). Otherwise the caller compiles the
    /// ordinary code, and the reason is reported once.
    pub(crate) fn may_specialize_on(&mut self, key: &str, rewrite: &str) -> bool {
        let reason: Option<String> = match self.features.weight_binding.as_ref() {
            None => Some(
                "this build does not load the weights file into the model (it is an input \
                 to planning only)"
                    .to_string(),
            ),
            Some(binding) => {
                if !binding.stored.contains_key(key) {
                    Some("the file stores no tensor of that name".to_string())
                } else if let Some(written) = binding.writes.written(key) {
                    Some(written.to_string())
                } else if let Some(ambiguity) = self.weight_key_ambiguity(key) {
                    Some(ambiguity)
                } else if self.weight_value_tolerance().is_none()
                    && !self.compiler_copy_is_exact(key)
                {
                    Some(format!(
                        "dead-weight elimination zeroed its values below {:e} in the \
                         compiler's copy, which is then not the file's; a rewrite uses the \
                         file's exact values unless --dead-weight-threshold is given",
                        self.compile_options.weights.config.dead_weight_threshold
                    ))
                } else {
                    None
                }
            }
        };
        let Some(reason) = reason else {
            return true;
        };
        self.note_weight_refusal(key, rewrite, &reason);
        false
    }

    /// Report, once per rewrite and parameter, that `rewrite` was not made
    /// on `key` and why.
    pub(crate) fn note_weight_refusal(&mut self, key: &str, rewrite: &str, reason: &str) {
        if self
            .features
            .weight_notes
            .insert(format!("{rewrite}\u{0}{key}"))
        {
            nsl_log::nsl_log!(
                INFO,
                "nsl",
                "[nsl] weight-aware: not specialized on '{key}' ({rewrite} skipped): {reason}"
            );
        }
    }

    /// How far a value-dependent rewrite may let a weight differ from the
    /// file: the `--dead-weight-threshold` the user passed, or `None` when
    /// none was passed, and the rewrite must rest on the file's exact
    /// values. (The threshold's default, 1e-6, is a tolerance the user did
    /// not ask for: a weight 9e-7 off the identity was treated as the
    /// identity, and dead-weight elimination's zeroed copy was folded.)
    pub(crate) fn weight_value_tolerance(&self) -> Option<f64> {
        self.compile_options
            .weights
            .threshold_given
            .then_some(self.compile_options.weights.config.dead_weight_threshold)
    }

    /// The report's words for what a rewrite rested on.
    pub(crate) fn weight_value_basis(&self) -> String {
        match self.weight_value_tolerance() {
            None => "from the file's exact values".to_string(),
            Some(t) => format!("from values within the --dead-weight-threshold {t:e} given"),
        }
    }

    /// Whether the compiler's copy of the file's tensor `key` still holds
    /// the file's bytes: dead-weight elimination zeroes small values in it.
    fn compiler_copy_is_exact(&mut self, key: &str) -> bool {
        if self
            .features
            .weight_notes
            .contains(&format!("exact\u{0}{key}"))
        {
            return true;
        }
        if self
            .features
            .weight_notes
            .contains(&format!("inexact\u{0}{key}"))
        {
            return false;
        }
        let exact = match (
            self.features
                .weight_binding
                .as_ref()
                .and_then(|b| b.stored.get(key)),
            self.features.weight_map.as_ref().and_then(|w| w.get(key)),
        ) {
            (Some(stored), Some(entry)) => {
                entry.dtype == stored.dtype
                    && entry.shape == stored.shape
                    && crate::weight_aware::tensor_content_hash(
                        entry.dtype.safetensors_name(),
                        &entry.shape,
                        &entry.data,
                    ) == stored.hash
            }
            _ => false,
        };
        let tag = if exact { "exact" } else { "inexact" };
        self.features
            .weight_notes
            .insert(format!("{tag}\u{0}{key}"));
        exact
    }

    /// The tensors whose content the program checks at startup: every stored
    /// tensor a rewrite could be allowed to specialize on, in name order. It
    /// depends on the file and the whole program only, so it is the same
    /// list in every module of a build and the entry module can check for
    /// all of them.
    pub(crate) fn weight_expectations(&self) -> Vec<(String, [u8; 32])> {
        let Some(binding) = self.features.weight_binding.as_ref() else {
            return Vec::new();
        };
        binding
            .stored
            .iter()
            .filter(|(name, _)| binding.writes.written(name).is_none())
            .map(|(name, tensor)| (name.clone(), tensor.hash))
            .collect()
    }

    /// Every parameter under an instance of `model_name`, with the names
    /// the file may store it under, most specific first: its path as
    /// `model_save` spells it, the same path dotted, and the candidates.
    fn weight_names_under(&self, model_name: &str) -> Vec<(String, String, Vec<String>)> {
        self.generate_param_metadata(model_name, "", 0)
            .into_iter()
            .map(|(path, _offset, _transpose)| {
                let dotted = path.replace('[', ".").replace(']', "");
                let candidates = if dotted == path && !path.contains('.') {
                    vec![format!("{model_name}.{path}"), path.clone()]
                } else if dotted == path {
                    vec![path.clone()]
                } else {
                    vec![path.clone(), dotted.clone()]
                };
                (path, dotted, candidates)
            })
            .collect()
    }

    /// In a model constructor, after the fields are initialised: copy the
    /// file's tensor into each parameter under the new instance that the
    /// file names (see the module documentation for the names). The copy is
    /// in place (same tensor, new values) and converts between float widths.
    /// A shape that differs is an error, here when the field's shape is a
    /// literal and at run time otherwise; except for a bare name that more
    /// than one parameter could claim, which is loaded where the shapes
    /// agree.
    pub(crate) fn emit_weight_file_loads(
        &mut self,
        builder: &mut FunctionBuilder,
        model_ptr: Value,
        model_name: &str,
    ) -> Result<(), CodegenError> {
        let Some(stored) = self
            .features
            .weight_binding
            .as_ref()
            .map(|b| b.stored.clone())
        else {
            return Ok(());
        };
        let Some(layout) = self.types.struct_layouts.get(model_name).cloned() else {
            return Ok(());
        };
        for (path, dotted, candidates) in self.weight_names_under(model_name) {
            let own_field = dotted == path && !path.contains('.');
            let Some((key, stored_shape)) = candidates.iter().find_map(|key| {
                let shape: Vec<i64> = stored.get(key)?.shape.iter().map(|d| *d as i64).collect();
                Some((key.clone(), shape))
            }) else {
                continue;
            };
            // A bare name is a claim by field name alone; when something
            // else could claim it too, the shapes decide.
            let doubt = if own_field && key == path {
                self.weight_key_ambiguity(&key)
            } else {
                None
            };
            let declared = own_field
                .then(|| {
                    self.models
                        .model_field_dims
                        .get(model_name)?
                        .get(&path)
                        .cloned()
                })
                .flatten()
                .filter(|dims| dims.iter().all(|d| *d > 0));
            if let Some(declared) = declared
                && declared != stored_shape
            {
                if doubt.is_some() {
                    continue;
                }
                return Err(CodegenError::new(format!(
                    "-w: the weights file's '{key}' has shape {stored_shape:?} but \
                     '{model_name}.{path}' is declared {declared:?}"
                )));
            }
            let tensor = self
                .load_nested_field(
                    builder,
                    model_ptr,
                    &layout,
                    model_name,
                    &format!("$model.{dotted}"),
                )
                .ok_or_else(|| {
                    CodegenError::new(format!(
                        "-w: could not resolve the parameter '{model_name}.{path}' to load \
                         the weights file's '{key}' into"
                    ))
                })?;
            let name_id = self.intern_string(&key)?;
            let name_gv = self.module.declare_data_in_func(name_id, builder.func);
            let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);
            let name_len = builder.ins().iconst(cl_types::I64, key.len() as i64);
            let strict = builder
                .ins()
                .iconst(cl_types::I64, i64::from(doubt.is_none()));
            self.compile_call_by_name(
                builder,
                "nsl_weights_load_field",
                &[tensor, name_ptr, name_len, strict],
            )?;
            if self
                .features
                .weight_notes
                .insert(format!("load\u{0}{model_name}.{path}"))
            {
                match doubt {
                    None => nsl_log::nsl_log!(
                        INFO,
                        "nsl",
                        "[nsl] weight-aware: '{model_name}.{path}' is loaded from the file's '{key}'"
                    ),
                    Some(doubt) => nsl_log::nsl_log!(
                        INFO,
                        "nsl",
                        "[nsl] weight-aware: '{model_name}.{path}' is loaded from the file's \
                         '{key}' where their shapes agree: {doubt}"
                    ),
                }
            }
        }
        Ok(())
    }

    /// At the start of `main`: open the weights file and check the tensors
    /// the compile may have specialized on. Nothing is emitted for a program
    /// that is not bound to a file.
    pub(crate) fn emit_weight_file_bind(
        &mut self,
        builder: &mut FunctionBuilder,
    ) -> Result<(), CodegenError> {
        let Some(binding) = self.features.weight_binding.clone() else {
            return Ok(());
        };
        // A tensor no model of the program has a parameter for is worth a
        // line: the usual cause is a file whose names do not follow the
        // model. "Could be claimed" is asked of every model this module
        // defines or imports, by the names their constructors look up.
        let claimable: std::collections::HashSet<String> = self
            .types
            .struct_layouts
            .keys()
            .flat_map(|model| self.weight_names_under(model))
            .flat_map(|(_, _, candidates)| candidates)
            .collect();
        let unclaimed: Vec<&str> = binding
            .stored
            .keys()
            .filter(|name| !claimable.contains(*name))
            .map(String::as_str)
            .collect();
        if !unclaimed.is_empty() {
            let shown = unclaimed
                .iter()
                .take(8)
                .copied()
                .collect::<Vec<_>>()
                .join(", ");
            let more = match unclaimed.len().saturating_sub(8) {
                0 => String::new(),
                n => format!(" and {n} more"),
            };
            nsl_log::nsl_log!(
                INFO,
                "nsl",
                "[nsl] weight-aware: {} of the file's {} tensors match no parameter of a model \
                 this module defines or imports and are not loaded: {shown}{more}",
                unclaimed.len(),
                binding.stored.len()
            );
        }
        let path = binding.path.display().to_string();
        let path_id = self.intern_string(&path)?;
        let path_gv = self.module.declare_data_in_func(path_id, builder.func);
        let path_ptr = builder.ins().symbol_value(cl_types::I64, path_gv);
        let path_len = builder.ins().iconst(cl_types::I64, path.len() as i64);

        let table = crate::weight_aware::encode_weight_expectations(&self.weight_expectations());
        let table_len = builder.ins().iconst(cl_types::I64, table.len() as i64);
        let table_ptr = if table.is_empty() {
            builder.ins().iconst(cl_types::I64, 0)
        } else {
            let data_id = self
                .module
                .declare_data(
                    "__nsl_weight_expectations",
                    cranelift_module::Linkage::Local,
                    false,
                    false,
                )
                .map_err(|e| CodegenError::new(format!("weight expectations declare: {e}")))?;
            let mut desc = cranelift_module::DataDescription::new();
            desc.define(table.into_boxed_slice());
            self.module
                .define_data(data_id, &desc)
                .map_err(|e| CodegenError::new(format!("weight expectations define: {e}")))?;
            let gv = self.module.declare_data_in_func(data_id, builder.func);
            builder.ins().symbol_value(cl_types::I64, gv)
        };
        self.compile_call_by_name(
            builder,
            "nsl_weights_bind",
            &[path_ptr, path_len, table_ptr, table_len],
        )?;
        Ok(())
    }

    /// The execution-fingerprint key that records what the program was
    /// specialized on, or `None` when it was specialized on nothing.
    pub(crate) fn weight_fingerprint_key(&self) -> Option<String> {
        let expectations = self.weight_expectations();
        (!expectations.is_empty()).then(|| {
            format!(
                "wspec={}",
                crate::weight_aware::weight_expectations_digest(&expectations)
            )
        })
    }
}
