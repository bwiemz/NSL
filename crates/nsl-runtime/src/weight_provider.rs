//! Standalone weight provider: embedded or memory-mapped sidecar .nslweights file.
//!
//! Initialized once per process via `nsl_standalone_init_embedded` or
//! `nsl_standalone_init_sidecar`. After init, `nsl_model_load` checks here first
//! and skips file I/O entirely when a provider is active.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// NSLW file format constants
const NSLW_MAGIC: &[u8; 4] = b"NSLW";
const NSLW_VERSION: u32 = 1;

// ── Data structures ──────────────────────────────────────────────────────────

/// Metadata for a single tensor stored in a .nslweights blob.
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
    /// Byte offset into the raw data region (already adjusted to be absolute
    /// within the blob for mmap mode, relative to data_start for embedded mode).
    pub offset: usize,
    pub nbytes: usize,
}

/// The underlying byte source — either a static slice baked into the binary, or
/// a memory-mapped file.
enum WeightData {
    Embedded(&'static [u8]),
    Mmap(memmap2::Mmap),
}

impl WeightData {
    fn as_slice(&self) -> &[u8] {
        match self {
            WeightData::Embedded(s) => s,
            WeightData::Mmap(m) => m.as_ref(),
        }
    }
}

// SAFETY: the embedded slice is &'static, and Mmap is effectively read-only
// shared memory — both are safe to share across threads.
unsafe impl Send for WeightData {}
unsafe impl Sync for WeightData {}

/// Holds the parsed header index and the byte source.
pub struct WeightProvider {
    data: WeightData,
    /// Tensor name → metadata (offsets are absolute within `data`).
    pub index: HashMap<String, TensorMeta>,
    /// Ordered tensor names (same order as the JSON header).
    pub ordered_names: Vec<String>,
}

// ── Global slot ──────────────────────────────────────────────────────────────

static WEIGHT_PROVIDER: OnceLock<WeightProvider> = OnceLock::new();

// ── Header parsing ───────────────────────────────────────────────────────────

/// Parse a .nslweights blob header.
///
/// Layout:
///   [0..4]  magic "NSLW"
///   [4..8]  version u32 LE
///   [8..16] header_size u64 LE
///   [16..]  JSON header (header_size bytes)
///   <pad to 64-byte alignment>
///   <tensor data>
///
/// Returns `(metas, data_start_offset)` where `data_start_offset` is the byte
/// index at which the raw tensor data begins inside `raw`.
fn parse_nslweights_header(raw: &[u8]) -> (Vec<TensorMeta>, usize) {
    if raw.len() < 16 {
        eprintln!(
            "nsl: weight_provider: blob too small ({} bytes, need at least 16)",
            raw.len()
        );
        std::process::abort();
    }
    if &raw[0..4] != NSLW_MAGIC {
        eprintln!("nsl: weight_provider: invalid .nslweights blob (bad magic)");
        std::process::abort();
    }
    let version = u32::from_le_bytes(raw[4..8].try_into().unwrap_or_else(|_| std::process::abort()));
    if version != NSLW_VERSION {
        eprintln!(
            "nsl: weight_provider: unsupported version {} (expected {})",
            version, NSLW_VERSION
        );
        std::process::abort();
    }
    let header_size = u64::from_le_bytes(raw[8..16].try_into().unwrap_or_else(|_| std::process::abort())) as usize;

    let header_end = 16 + header_size;
    if raw.len() < header_end {
        eprintln!(
            "nsl: weight_provider: blob truncated (need {} bytes for header, have {})",
            header_end,
            raw.len()
        );
        std::process::abort();
    }

    // Parse JSON
    let json_bytes = &raw[16..header_end];
    let json_str = std::str::from_utf8(json_bytes).unwrap_or_else(|_| {
        eprintln!("nsl: weight_provider: header is not valid UTF-8");
        std::process::abort();
    });
    let root: serde_json::Value = serde_json::from_str(json_str).unwrap_or_else(|e| {
        eprintln!("nsl: weight_provider: header JSON parse error: {}", e);
        std::process::abort();
    });

    let params = root["params"].as_array().unwrap_or_else(|| {
        eprintln!("nsl: weight_provider: header missing 'params' array");
        std::process::abort();
    });

    let metas: Vec<TensorMeta> = params
        .iter()
        .map(|p| {
            let name = p["name"].as_str().unwrap_or("?").to_string();
            let dtype = p["dtype"].as_str().unwrap_or("f64").to_string();
            let offset = p["offset"].as_u64().unwrap_or_else(|| {
                eprintln!("nsl: weight_provider: tensor '{}' missing 'offset'", name);
                std::process::abort();
            }) as usize;
            let nbytes = p["nbytes"].as_u64().unwrap_or_else(|| {
                eprintln!("nsl: weight_provider: tensor '{}' missing 'nbytes'", name);
                std::process::abort();
            }) as usize;
            let shape: Vec<i64> = p["shape"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_i64().unwrap_or(0))
                .collect();
            TensorMeta { name, shape, dtype, offset, nbytes }
        })
        .collect();

    // data_start is right after the JSON header, padded to 64-byte alignment
    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;

    (metas, data_start)
}

// ── FFI init functions ───────────────────────────────────────────────────────

/// Initialize the weight provider from an embedded byte slice.
///
/// `data_ptr` points to `.rodata` (lives for the process lifetime).
/// `data_len` is the total byte length of the .nslweights blob.
#[no_mangle]
pub extern "C" fn nsl_standalone_init_embedded(data_ptr: i64, data_len: i64) {
    let raw: &'static [u8] =
        unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len as usize) };

    let (mut metas, data_start) = parse_nslweights_header(raw);

    // Adjust offsets: they are relative to the tensor-data region, so add
    // data_start so they become absolute within `raw`.
    for m in &mut metas {
        m.offset += data_start;
    }

    // The data slice is the full embedded blob (offsets are now absolute into it).
    let provider = build_provider(WeightData::Embedded(raw), metas);
    let _ = WEIGHT_PROVIDER.set(provider); // ignore if already set
}

/// Initialize the weight provider from a .nslweights sidecar file.
///
/// Search order:
///   1. Same directory as the compiled binary path given by the caller
///   2. Next to the current executable (`std::env::current_exe`)
///   3. `$NSL_WEIGHTS_PATH` environment variable
#[no_mangle]
pub extern "C" fn nsl_standalone_init_sidecar(compiled_path_ptr: i64, compiled_path_len: i64) {
    let compiled_path_str = unsafe {
        let slice = std::slice::from_raw_parts(
            compiled_path_ptr as *const u8,
            compiled_path_len as usize,
        );
        std::str::from_utf8(slice).unwrap_or("")
    };

    // Build candidate paths
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // 1. Same directory as the compiled binary
    if !compiled_path_str.is_empty() {
        let p = std::path::Path::new(compiled_path_str);
        let stem = p.file_stem().map(|s| {
            let mut pb = p.with_file_name(s);
            pb.set_extension("nslweights");
            pb
        });
        if let Some(path) = stem {
            candidates.push(path);
        }
    }

    // 2. Next to current executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // Try matching the compiled binary stem first, else the exe name
            let stem = if !compiled_path_str.is_empty() {
                std::path::Path::new(compiled_path_str)
                    .file_stem()
                    .map(|s| s.to_owned())
            } else {
                exe.file_stem().map(|s| s.to_owned())
            };
            if let Some(stem) = stem {
                let mut pb = dir.join(&stem);
                pb.set_extension("nslweights");
                candidates.push(pb);
            }
        }
    }

    // 3. $NSL_WEIGHTS_PATH env var
    if let Ok(env_path) = std::env::var("NSL_WEIGHTS_PATH") {
        candidates.push(std::path::PathBuf::from(env_path));
    }

    // Try each candidate
    let file = candidates.iter().find_map(|p| std::fs::File::open(p).ok());
    let file = match file {
        Some(f) => f,
        None => {
            eprintln!(
                "nsl: weight_provider: could not find .nslweights sidecar (searched: {:?})",
                candidates
            );
            std::process::abort();
        }
    };

    let mmap = unsafe {
        memmap2::Mmap::map(&file).unwrap_or_else(|e| {
            eprintln!("nsl: weight_provider: mmap failed: {}", e);
            std::process::abort();
        })
    };

    let (mut metas, data_start) = parse_nslweights_header(mmap.as_ref());

    // Adjust offsets to be absolute within the mmap'd buffer.
    for m in &mut metas {
        m.offset += data_start;
    }

    let provider = build_provider(WeightData::Mmap(mmap), metas);
    let _ = WEIGHT_PROVIDER.set(provider);
}

/// Returns 1 if the weight provider has been initialized, 0 otherwise.
#[no_mangle]
pub extern "C" fn nsl_standalone_has_weights() -> i64 {
    if WEIGHT_PROVIDER.get().is_some() { 1 } else { 0 }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn build_provider(data: WeightData, metas: Vec<TensorMeta>) -> WeightProvider {
    let mut index = HashMap::with_capacity(metas.len());
    let mut ordered_names = Vec::with_capacity(metas.len());
    for m in metas {
        ordered_names.push(m.name.clone());
        index.insert(m.name.clone(), m);
    }
    WeightProvider { data, index, ordered_names }
}

// ── Integration with nsl_model_load ─────────────────────────────────────────

/// If a weight provider is active, copy tensor data into the tensors in `list`
/// (by index order, matching `ordered_names`) and return `true`.
/// Returns `false` if no provider is set.
pub fn try_load_from_provider(tensors: &crate::list::NslList) -> bool {
    let provider = match WEIGHT_PROVIDER.get() {
        Some(p) => p,
        None => return false,
    };

    let raw = provider.data.as_slice();

    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = crate::tensor::NslTensor::from_ptr(tensor_ptr);
        let byte_count = (tensor.len as usize) * tensor.element_size();

        // Look up by index order
        let meta = provider.ordered_names.get(i).and_then(|n| provider.index.get(n));
        let meta = match meta {
            Some(m) => m,
            None => {
                eprintln!(
                    "nsl: weight_provider: no weight entry for tensor index {}",
                    i
                );
                std::process::abort();
            }
        };

        if meta.nbytes < byte_count {
            eprintln!(
                "nsl: weight_provider: tensor '{}' stored size {} < required {} bytes",
                meta.name, meta.nbytes, byte_count
            );
            std::process::abort();
        }

        let src_offset = meta.offset;
        if src_offset + byte_count > raw.len() {
            eprintln!(
                "nsl: weight_provider: tensor '{}' offset {} + {} exceeds blob size {}",
                meta.name, src_offset, byte_count, raw.len()
            );
            std::process::abort();
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                raw[src_offset..].as_ptr(),
                tensor.data as *mut u8,
                byte_count,
            );
        }
    }

    true
}

// ── Standalone CLI arg parser ────────────────────────────────────────────────

/// Runtime state for parsing CLI arguments in standalone executables.
struct ArgState {
    /// All argv strings (argv[0] is program name, rest are arguments).
    args: Vec<String>,
    /// Tracks which args have been consumed (by index into `args`).
    consumed: Vec<bool>,
    /// The program name (argv[0]).
    program_name: String,
    /// Registered parameter names (for --help generation).
    param_names: Vec<String>,
    /// Registered parameter types (for --help generation).
    param_types: Vec<String>,
    /// Registered parameter default values (empty string = required).
    param_defaults: Vec<String>,
}

static ARG_STATE: OnceLock<Mutex<ArgState>> = OnceLock::new();

fn arg_state_global() -> &'static Mutex<ArgState> {
    ARG_STATE.get_or_init(|| {
        Mutex::new(ArgState {
            args: Vec::new(),
            consumed: Vec::new(),
            program_name: String::new(),
            param_names: Vec::new(),
            param_types: Vec::new(),
            param_defaults: Vec::new(),
        })
    })
}

/// Print usage summary to stderr.
fn print_usage(state: &ArgState) {
    let prog = if state.program_name.is_empty() {
        "<program>"
    } else {
        &state.program_name
    };
    let mut usage = format!("Usage: {}", prog);
    for i in 0..state.param_names.len() {
        let name = &state.param_names[i];
        let default = &state.param_defaults[i];
        if default.is_empty() {
            usage.push_str(&format!(" --{} <{}>", name, &state.param_types[i]));
        } else {
            usage.push_str(&format!(" [--{} <{}>]", name, &state.param_types[i]));
        }
    }
    eprintln!("{}", usage);
    if !state.param_names.is_empty() {
        eprintln!();
        eprintln!("Options:");
        for i in 0..state.param_names.len() {
            let name = &state.param_names[i];
            let ty = &state.param_types[i];
            let default = &state.param_defaults[i];
            if default.is_empty() {
                eprintln!("  --{:<20} {} (required)", name, ty);
            } else {
                eprintln!("  --{:<20} {} (default: {})", name, ty, default);
            }
        }
    }
    eprintln!("  --{:<20} Show this help message", "help");
}

/// If --help or -h is present in the args, print usage and exit(0).
fn check_help_flag(state: &ArgState) {
    for arg in &state.args {
        if arg == "--help" || arg == "-h" {
            print_usage(state);
            std::process::exit(0);
        }
    }
}

/// Extract a name from (name_ptr, name_len) pair using raw pointer arithmetic.
///
/// # Safety
/// `name_ptr` must point to `name_len` valid UTF-8 bytes for the duration of
/// the call.
unsafe fn name_from_parts(name_ptr: i64, name_len: i64) -> String {
    let slice = std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize);
    // SAFETY: caller guarantees valid UTF-8
    std::str::from_utf8_unchecked(slice).to_string()
}

/// Find `--<name> <value>` in args.  Returns (value, index_of_flag) on success.
fn find_arg<'a>(state: &'a ArgState, name: &str) -> Option<(&'a str, usize)> {
    let flag = format!("--{}", name);
    for i in 0..state.args.len() {
        if state.args[i] == flag {
            if i + 1 < state.args.len() {
                return Some((&state.args[i + 1], i));
            }
            // Flag present but no value
            eprintln!("nsl: --{} requires a value", name);
            print_usage(state);
            std::process::exit(1);
        }
    }
    None
}

/// Mark `--<name> <value>` (two slots) as consumed.
fn consume_arg(state: &mut ArgState, flag_idx: usize) {
    if flag_idx < state.consumed.len() {
        state.consumed[flag_idx] = true;
    }
    if flag_idx + 1 < state.consumed.len() {
        state.consumed[flag_idx + 1] = true;
    }
}

/// Register a parameter in the state's help metadata.
fn register_param(state: &mut ArgState, name: &str, ty: &str, default: &str) {
    // Avoid duplicate registration (codegen may call helpers more than once in
    // some edge cases, though in practice each param is initialized once).
    if !state.param_names.contains(&name.to_string()) {
        state.param_names.push(name.to_string());
        state.param_types.push(ty.to_string());
        state.param_defaults.push(default.to_string());
    }
}

// ── FFI: init / finish ───────────────────────────────────────────────────────

/// Initialize the standalone arg parser from C `main`'s argc/argv.
///
/// `argc` is the argument count; `argv` is a `*const *const u8` (array of
/// NUL-terminated C strings) cast to i64.
#[no_mangle]
pub extern "C" fn nsl_standalone_args_init(argc: i64, argv: i64) {
    let argv_ptr = argv as *const *const u8;
    let mut args: Vec<String> = Vec::with_capacity(argc as usize);

    for i in 0..argc as usize {
        let ptr = unsafe { *argv_ptr.add(i) };
        let cstr = unsafe { std::ffi::CStr::from_ptr(ptr as *const std::os::raw::c_char) };
        let s = cstr.to_str().unwrap_or("").to_string();
        args.push(s);
    }

    let consumed = vec![false; args.len()];
    // argv[0] is always the program name
    let program_name = args.first().cloned().unwrap_or_default();

    let mutex = arg_state_global();
    let mut state = mutex.lock().unwrap();
    state.args = args;
    state.consumed = consumed;
    state.program_name = program_name;
    state.param_names.clear();
    state.param_types.clear();
    state.param_defaults.clear();
}

/// Check for --help/-h and warn about unrecognized arguments.
///
/// Call this after all `nsl_standalone_arg_*` helpers have run.
#[no_mangle]
pub extern "C" fn nsl_standalone_args_finish() {
    let mutex = arg_state_global();
    let state = mutex.lock().unwrap();

    // If --help/-h is present, print usage and exit.
    check_help_flag(&state);

    // Warn about any unconsumed arguments (skip argv[0] which is the program name).
    for i in 1..state.args.len() {
        if !state.consumed[i] {
            let arg = &state.args[i];
            // Only warn about flag-looking tokens; skip values that follow unknown flags
            if arg.starts_with('-') {
                eprintln!(
                    "nsl: warning: unrecognized argument '{}' (use --help for usage)",
                    arg
                );
            }
        }
    }
}

// ── FFI: string args ─────────────────────────────────────────────────────────

/// Parse a required `--<name> <value>` string argument.
///
/// Returns an NslString pointer (i64) via `nsl_str_from_rust`.
/// Exits with usage if the argument is missing.
#[no_mangle]
pub extern "C" fn nsl_standalone_arg_str(name_ptr: i64, name_len: i64) -> i64 {
    let name = unsafe { name_from_parts(name_ptr, name_len) };
    let mutex = arg_state_global();
    let mut state = mutex.lock().unwrap();
    register_param(&mut state, &name, "string", "");

    match find_arg(&state, &name) {
        Some((val, flag_idx)) => {
            let val = val.to_string();
            consume_arg(&mut state, flag_idx);
            drop(state);
            crate::string::nsl_str_from_rust(&val)
        }
        None => {
            check_help_flag(&state);
            eprintln!("nsl: missing required argument --{}", name);
            print_usage(&state);
            std::process::exit(1);
        }
    }
}

/// Parse an optional `--<name> <value>` string argument with a default value.
///
/// Returns an NslString pointer (i64).
#[no_mangle]
pub extern "C" fn nsl_standalone_arg_str_default(
    name_ptr: i64,
    name_len: i64,
    default_ptr: i64,
    default_len: i64,
) -> i64 {
    let name = unsafe { name_from_parts(name_ptr, name_len) };
    let default_val = unsafe {
        let slice = std::slice::from_raw_parts(default_ptr as *const u8, default_len as usize);
        std::str::from_utf8_unchecked(slice).to_string()
    };

    let mutex = arg_state_global();
    let mut state = mutex.lock().unwrap();
    register_param(&mut state, &name, "string", &default_val);

    match find_arg(&state, &name) {
        Some((val, flag_idx)) => {
            let val = val.to_string();
            consume_arg(&mut state, flag_idx);
            drop(state);
            crate::string::nsl_str_from_rust(&val)
        }
        None => {
            drop(state);
            crate::string::nsl_str_from_rust(&default_val)
        }
    }
}

// ── FFI: int args ─────────────────────────────────────────────────────────────

/// Parse a required `--<name> <value>` integer argument.
///
/// Exits with usage if the argument is missing or cannot be parsed as i64.
#[no_mangle]
pub extern "C" fn nsl_standalone_arg_int(name_ptr: i64, name_len: i64) -> i64 {
    let name = unsafe { name_from_parts(name_ptr, name_len) };
    let mutex = arg_state_global();
    let mut state = mutex.lock().unwrap();
    register_param(&mut state, &name, "int", "");

    match find_arg(&state, &name) {
        Some((val, flag_idx)) => {
            let parsed = val.parse::<i64>().unwrap_or_else(|_| {
                eprintln!("nsl: --{} requires an integer value, got '{}'", name, val);
                print_usage(&state);
                std::process::exit(1);
            });
            consume_arg(&mut state, flag_idx);
            parsed
        }
        None => {
            check_help_flag(&state);
            eprintln!("nsl: missing required argument --{}", name);
            print_usage(&state);
            std::process::exit(1);
        }
    }
}

/// Parse an optional `--<name> <value>` integer argument with a default value.
#[no_mangle]
pub extern "C" fn nsl_standalone_arg_int_default(
    name_ptr: i64,
    name_len: i64,
    default_val: i64,
) -> i64 {
    let name = unsafe { name_from_parts(name_ptr, name_len) };
    let mutex = arg_state_global();
    let mut state = mutex.lock().unwrap();
    register_param(&mut state, &name, "int", &default_val.to_string());

    match find_arg(&state, &name) {
        Some((val, flag_idx)) => {
            let parsed = val.parse::<i64>().unwrap_or_else(|_| {
                eprintln!("nsl: --{} requires an integer value, got '{}'", name, val);
                print_usage(&state);
                std::process::exit(1);
            });
            consume_arg(&mut state, flag_idx);
            parsed
        }
        None => default_val,
    }
}

// ── FFI: float args ───────────────────────────────────────────────────────────

/// Parse a required `--<name> <value>` float argument.
///
/// Returns the f64 value's raw bits as i64 (use `f64::from_bits(result as u64)`
/// on the receiving side).  Exits with usage if missing or unparseable.
#[no_mangle]
pub extern "C" fn nsl_standalone_arg_float(name_ptr: i64, name_len: i64) -> i64 {
    let name = unsafe { name_from_parts(name_ptr, name_len) };
    let mutex = arg_state_global();
    let mut state = mutex.lock().unwrap();
    register_param(&mut state, &name, "float", "");

    match find_arg(&state, &name) {
        Some((val, flag_idx)) => {
            let parsed = val.parse::<f64>().unwrap_or_else(|_| {
                eprintln!("nsl: --{} requires a float value, got '{}'", name, val);
                print_usage(&state);
                std::process::exit(1);
            });
            consume_arg(&mut state, flag_idx);
            parsed.to_bits() as i64
        }
        None => {
            check_help_flag(&state);
            eprintln!("nsl: missing required argument --{}", name);
            print_usage(&state);
            std::process::exit(1);
        }
    }
}

/// Parse an optional `--<name> <value>` float argument with a default value.
///
/// `default_bits` is the default f64 value encoded as its raw bits cast to i64
/// (i.e., `f64::to_bits(default) as i64`).  Returns f64 bits as i64.
#[no_mangle]
pub extern "C" fn nsl_standalone_arg_float_default(
    name_ptr: i64,
    name_len: i64,
    default_bits: i64,
) -> i64 {
    let name = unsafe { name_from_parts(name_ptr, name_len) };
    let default_f = f64::from_bits(default_bits as u64);

    let mutex = arg_state_global();
    let mut state = mutex.lock().unwrap();
    register_param(&mut state, &name, "float", &default_f.to_string());

    match find_arg(&state, &name) {
        Some((val, flag_idx)) => {
            let parsed = val.parse::<f64>().unwrap_or_else(|_| {
                eprintln!("nsl: --{} requires a float value, got '{}'", name, val);
                print_usage(&state);
                std::process::exit(1);
            });
            consume_arg(&mut state, flag_idx);
            parsed.to_bits() as i64
        }
        None => default_bits,
    }
}
