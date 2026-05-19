//! Eager-at-create, read-only-thereafter export registry.
//!
//! At nsl_model_create time, this module dlopens the model's own
//! .so/.dll, dlsyms the two codegen-emitted enumeration FFIs
//! (`nsl_get_num_exports`, `nsl_get_export_name`), then dlsyms each
//! export name to its raw C-ABI function pointer and stores everything
//! in a HashMap that is mutated nowhere else.
//!
//! Concurrency: After construction, the registry is read-only. No locks
//! needed at call sites. Cloning the registry is not supported (the
//! libloading::Library handle owns the dlopen lifetime).

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;

/// Raw export function pointer signature emitted by codegen for the
/// `NslTensorDesc*`-ABI path. Same signature for `forward`, `backward`,
/// and any user-defined `@export`.
///
/// (model_ptr, inputs_desc_ptr, n_inputs, outputs_desc_ptr, n_outputs) -> rc
pub type ExportFnPtr = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;

/// Per-model export dispatch table. Owns the dlopen handle and the
/// dlsym'd function pointers.
pub struct ExportRegistry {
    #[allow(dead_code)]
    library: libloading::Library,
    table: HashMap<CString, ExportFnPtr>,
}

impl std::fmt::Debug for ExportRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportRegistry")
            .field("library", &"<dlopen handle>")
            .field("exports", &self.available_names())
            .finish()
    }
}

#[derive(Debug)]
pub enum ExportRegistryError {
    LibraryOpen {
        path: String,
        source: libloading::Error,
    },
    ExportTableMissing {
        path: String,
        symbol: &'static str,
        source: libloading::Error,
    },
    ExportMissing {
        name: String,
        available: Vec<String>,
    },
}

impl std::fmt::Display for ExportRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryOpen { path, source } => {
                write!(f, "ExportRegistry: failed to dlopen '{}': {}", path, source)
            }
            Self::ExportTableMissing {
                path,
                symbol,
                source,
            } => write!(
                f,
                "ExportRegistry: '{}' missing required symbol '{}': {} \
                    (artifact built before export-table FFIs were emitted?)",
                path, symbol, source
            ),
            Self::ExportMissing { name, available } => write!(
                f,
                "ExportRegistry: export '{}' declared by enumeration but not \
                    found in symbol table. Available: {:?}",
                name, available
            ),
        }
    }
}

impl std::error::Error for ExportRegistryError {}

impl ExportRegistry {
    /// Open the library, enumerate exports via the codegen-emitted
    /// FFIs, dlsym each, and assemble the read-only HashMap.
    pub fn from_library_path(path: &Path) -> Result<Self, ExportRegistryError> {
        let path_str = path.to_string_lossy().into_owned();
        let library = unsafe { libloading::Library::new(path) }.map_err(|source| {
            ExportRegistryError::LibraryOpen {
                path: path_str.clone(),
                source,
            }
        })?;

        // Enumerate inside a block scope so the Symbol borrows expire
        // before we re-borrow `library` for individual export lookups.
        let names: Vec<String> = {
            let get_num: libloading::Symbol<unsafe extern "C" fn() -> i64> =
                unsafe { library.get(b"nsl_get_num_exports") }.map_err(|source| {
                    ExportRegistryError::ExportTableMissing {
                        path: path_str.clone(),
                        symbol: "nsl_get_num_exports",
                        source,
                    }
                })?;
            let get_name: libloading::Symbol<unsafe extern "C" fn(i64) -> *const c_char> =
                unsafe { library.get(b"nsl_get_export_name") }.map_err(|source| {
                    ExportRegistryError::ExportTableMissing {
                        path: path_str.clone(),
                        symbol: "nsl_get_export_name",
                        source,
                    }
                })?;

            let n = unsafe { get_num() };
            let mut names: Vec<String> = Vec::with_capacity(n.max(0) as usize);
            for i in 0..n {
                let name_ptr = unsafe { get_name(i) };
                if name_ptr.is_null() {
                    continue;
                }
                let s = unsafe { CStr::from_ptr(name_ptr) }
                    .to_string_lossy()
                    .into_owned();
                names.push(s);
            }
            names
        };

        // The HashMap is keyed by the user-facing export name (`alpha`,
        // `custom_beta`, ...) so `registry.lookup("alpha")` works. The
        // dlsym'd pointer, however, is the packed-array sibling
        // (`<name>__nsl_dispatch`) which honors the `ExportFnPtr` ABI used by
        // `nsl_model_call`. The typed `<name>` symbol still exists in the
        // shared library for direct ctypes callers; the registry just doesn't
        // route through it.
        let mut table: HashMap<CString, ExportFnPtr> = HashMap::with_capacity(names.len());
        for name in &names {
            let user_cname = CString::new(name.as_str())
                .expect("export name from codegen enumeration must not contain interior NUL");
            let dispatch_sym = format!("{}__nsl_dispatch", name);
            let dispatch_cname = CString::new(dispatch_sym.as_str()).expect(
                "export name + __nsl_dispatch suffix must not contain interior NUL",
            );
            let sym: libloading::Symbol<*mut c_void> =
                unsafe { library.get(dispatch_cname.as_bytes_with_nul()) }.map_err(|_| {
                    ExportRegistryError::ExportMissing {
                        name: dispatch_sym.clone(),
                        available: names.clone(),
                    }
                })?;
            let raw_ptr = *sym;
            let fn_ptr: ExportFnPtr =
                unsafe { std::mem::transmute::<*mut c_void, ExportFnPtr>(raw_ptr) };
            table.insert(user_cname, fn_ptr);
        }

        Ok(Self { library, table })
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn lookup(&self, name: &str) -> Option<ExportFnPtr> {
        let cname = CString::new(name).ok()?;
        self.table.get(&cname).copied()
    }

    pub fn available_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .table
            .keys()
            .map(|k| k.to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}
