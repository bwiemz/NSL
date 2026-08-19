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

/// The item-7 dispatch-ownership FFIs, dlsym'd from the SAME image as the
/// dispatch wrappers so the mode thread-local and every allocation/free stay
/// same-image (a `--shared-lib` artifact statically links its own runtime
/// copy — arming a host-linked runtime's thread-local would be invisible to
/// the wrapper's helpers).
#[derive(Clone, Copy)]
pub struct OwnershipFfis {
    /// `nsl_dispatch_ownership_arm(mode, caps_ptr, n_caps) -> rc`
    pub arm: unsafe extern "C" fn(i64, i64, i64) -> i64,
    /// `nsl_dispatch_ownership_release(model_ptr) -> rc`
    pub release: unsafe extern "C" fn(i64) -> i64,
    /// `nsl_dispatch_ownership_finish_alloc(model, name, out_dl, n) -> rc`
    pub finish_alloc: unsafe extern "C" fn(i64, i64, i64, i64) -> i64,
}

/// Per-model export dispatch table. Owns the dlopen handle and the
/// dlsym'd function pointers.
pub struct ExportRegistry {
    #[allow(dead_code)]
    library: libloading::Library,
    table: HashMap<CString, ExportFnPtr>,
    /// Per-export JSON signature pointers (item 7 introspection), keyed like
    /// `table`. The `i64` is a `*const c_char` into the dlopen'd image's
    /// static data — valid exactly as long as `library` (which this struct
    /// owns), so storing the raw pointer is sound. Empty when the artifact
    /// predates the `nsl_get_export_signature_json` accessor.
    signatures: HashMap<CString, i64>,
    /// `None` when the artifact's statically-linked runtime predates the
    /// ownership FFIs; `nsl_model_call_into`/`_alloc` refuse in that case.
    ownership: Option<OwnershipFfis>,
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

        // Item-7 introspection: bind the OPTIONAL per-export signature
        // accessor. Same index space as `nsl_get_export_name` — both are
        // emitted from the same exports slice in declaration order. Absent
        // in artifacts built before the signature table existed; that is
        // not an error here (lookup_signature reports it per-query).
        let mut signatures: HashMap<CString, i64> = HashMap::new();
        if let Ok(get_sig) = unsafe {
            library.get::<unsafe extern "C" fn(i64) -> *const c_char>(
                b"nsl_get_export_signature_json",
            )
        } {
            for (i, name) in names.iter().enumerate() {
                let sig_ptr = unsafe { get_sig(i as i64) };
                if sig_ptr.is_null() {
                    continue;
                }
                let cname = CString::new(name.as_str())
                    .expect("export name from codegen enumeration must not contain interior NUL");
                signatures.insert(cname, sig_ptr as i64);
            }
        }

        // Item-7 ownership FFIs — all-or-none, from the same image as the
        // dispatch wrappers. Absent in pre-item-7 artifacts (not an error;
        // the ownership entry points refuse per-call with a rebuild hint).
        let ownership = unsafe {
            let arm = library
                .get::<unsafe extern "C" fn(i64, i64, i64) -> i64>(b"nsl_dispatch_ownership_arm");
            let release = library
                .get::<unsafe extern "C" fn(i64) -> i64>(b"nsl_dispatch_ownership_release");
            let finish_alloc = library.get::<unsafe extern "C" fn(i64, i64, i64, i64) -> i64>(
                b"nsl_dispatch_ownership_finish_alloc",
            );
            match (arm, release, finish_alloc) {
                (Ok(a), Ok(r), Ok(f)) => Some(OwnershipFfis {
                    arm: *a,
                    release: *r,
                    finish_alloc: *f,
                }),
                _ => None,
            }
        };

        Ok(Self { library, table, signatures, ownership })
    }

    /// The item-7 ownership FFIs, or `None` for pre-item-7 artifacts.
    pub fn ownership_ffis(&self) -> Option<OwnershipFfis> {
        self.ownership
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

    /// Look up an export's JSON signature pointer (`*const c_char` as i64,
    /// valid for the registry's lifetime). `Err` carries a human-readable
    /// reason: signature table absent (old artifact) vs unknown name.
    pub fn lookup_signature(&self, name: &str) -> Result<i64, String> {
        if self.signatures.is_empty() {
            return Err(
                "this artifact has no export-signature table (built before \
                 nsl_get_export_signature_json existed) — rebuild it with the \
                 current compiler"
                    .to_string(),
            );
        }
        let cname = CString::new(name)
            .map_err(|_| format!("export name '{name}' contains an interior NUL"))?;
        match self.signatures.get(&cname) {
            Some(&ptr) => Ok(ptr),
            None => Err(format!(
                "export '{}' not in registry. Available: {:?}",
                name,
                self.available_names()
            )),
        }
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
