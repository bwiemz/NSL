//! Tokenizer runtime module for NSL (M15).
//!
//! Provides byte-level and BPE tokenizers via the HuggingFace `tokenizers` crate.
//! Tokenizer handles are stored as boxed `TokenizerKind` enums, converted to i64 pointers.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use tokenizers::models::bpe::BPE;
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::Tokenizer;

use std::ffi::c_void;

use crate::list::NslList;
use crate::memory::checked_alloc;
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// Tokenizer handle management
// ---------------------------------------------------------------------------

/// Distinguishes between a simple byte tokenizer (no HF overhead) and a
/// full HuggingFace tokenizer (BPE, trained, or loaded from file).
enum TokenizerKind {
    /// Trivial byte tokenizer: token ID == byte value, vocab size 256.
    Byte,
    /// Full HuggingFace tokenizer.
    HuggingFace(Box<Tokenizer>),
}

/// Box a `TokenizerKind` and return its raw pointer as i64.
fn store_tokenizer(kind: TokenizerKind) -> i64 {
    Box::into_raw(Box::new(kind)) as i64
}

/// Recover a `&mut TokenizerKind` from a handle previously returned by `store_tokenizer`.
///
/// # Safety
/// The caller must pass a valid handle that has not been freed.
fn get_tokenizer(handle: i64) -> &'static mut TokenizerKind {
    unsafe { &mut *(handle as *mut TokenizerKind) }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a C string pointer (i64) into a `&str`.
unsafe fn cstr_to_str(ptr: i64) -> &'static str {
    let cstr = CStr::from_ptr(ptr as *const c_char);
    cstr.to_str().unwrap_or("")
}

/// Allocate a new C string from `&str`, returning its pointer as i64.
fn alloc_cstring(s: &str) -> i64 {
    let cstring = CString::new(s).unwrap_or_else(|_| CString::new("").unwrap());
    let bytes = cstring.as_bytes_with_nul();
    let ptr = checked_alloc(bytes.len());
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
    }
    ptr as i64
}

/// Create a 1-D `NslTensor` from a slice of f64 values.
fn make_1d_tensor(values: &[f64]) -> i64 {
    let len = values.len() as i64;
    let ndim: i64 = 1;

    let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = len };

    let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides = 1 };

    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
    for (i, &v) in values.iter().enumerate() {
        unsafe { *data.add(i) = v };
    }

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        0,
        0,
        1,
        0,
    ));
    Box::into_raw(tensor) as i64
}

/// Create a 2-D `NslTensor` (rows x cols) from a flat Vec<f64>.
fn make_2d_tensor(rows: usize, cols: usize, flat: &[f64]) -> i64 {
    let len = (rows * cols) as i64;
    let ndim: i64 = 2;

    let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape = rows as i64;
        *shape.add(1) = cols as i64;
    }

    let strides = NslTensor::compute_strides(shape, ndim);

    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
    for (i, &v) in flat.iter().enumerate() {
        unsafe { *data.add(i) = v };
    }

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        0,
        0,
        1,
        0,
    ));
    Box::into_raw(tensor) as i64
}

// ---------------------------------------------------------------------------
// Public C ABI functions
// ---------------------------------------------------------------------------

/// Create a byte-level tokenizer (each byte 0-255 is its own token).
#[no_mangle]
pub extern "C" fn nsl_byte_tokenizer_new() -> i64 {
    store_tokenizer(TokenizerKind::Byte)
}

/// Train a BPE tokenizer from a corpus file.
///
/// Arguments (all i64):
///   - `corpus_path_ptr`: C string pointer to the corpus file path
///   - `vocab_size`: desired vocabulary size
///   - `min_freq`: minimum token frequency
///   - `special_tokens_list`: pointer to an `NslList` of C string pointers
///
/// Returns a tokenizer handle (i64).
#[no_mangle]
pub extern "C" fn nsl_bpe_train(
    corpus_path_ptr: i64,
    vocab_size: i64,
    min_freq: i64,
    special_tokens_list: i64,
) -> i64 {
    let path = unsafe { cstr_to_str(corpus_path_ptr) };

    // Collect special tokens from the NslList of C string pointers
    let mut special_tokens: Vec<tokenizers::AddedToken> = Vec::new();
    if special_tokens_list != 0 {
        let list = NslList::from_ptr(special_tokens_list);
        for i in 0..list.len as usize {
            let tok_ptr = unsafe { *list.data.add(i) };
            let tok_str = unsafe { cstr_to_str(tok_ptr) };
            special_tokens.push(tokenizers::AddedToken::from(tok_str.to_string(), true));
        }
    }

    let bpe_trainer = tokenizers::models::bpe::BpeTrainer::builder()
        .vocab_size(vocab_size as usize)
        .min_frequency(min_freq as u64)
        .special_tokens(special_tokens)
        .build();
    let mut trainer = TrainerWrapper::BpeTrainer(bpe_trainer);

    let mut tokenizer = Tokenizer::new(
        tokenizers::ModelWrapper::BPE(BPE::default()),
    );
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::PreTokenizerWrapper::ByteLevel(ByteLevel::default()),
    ));

    if let Err(e) = tokenizer.train_from_files(&mut trainer, vec![path.to_string()]) {
        eprintln!("nsl: BPE training failed: {e}");
        std::process::abort();
    }

    store_tokenizer(TokenizerKind::HuggingFace(Box::new(tokenizer)))
}

/// Load a tokenizer from a JSON file.
///
/// Arguments:
///   - `path_ptr`: C string pointer to the JSON file path
///
/// Returns a tokenizer handle (i64).
#[no_mangle]
pub extern "C" fn nsl_tokenizer_load(path_ptr: i64) -> i64 {
    let path = unsafe { cstr_to_str(path_ptr) };
    match Tokenizer::from_file(path) {
        Ok(tok) => store_tokenizer(TokenizerKind::HuggingFace(Box::new(tok))),
        Err(e) => {
            eprintln!("nsl: failed to load tokenizer from '{path}': {e}");
            std::process::abort();
        }
    }
}

/// Save a tokenizer to a JSON file.
///
/// Arguments:
///   - `handle`: tokenizer handle
///   - `path_ptr`: C string pointer to the output file path
#[no_mangle]
pub extern "C" fn nsl_tokenizer_save(handle: i64, path_ptr: i64) {
    let path = unsafe { cstr_to_str(path_ptr) };
    match get_tokenizer(handle) {
        TokenizerKind::Byte => {
            eprintln!("nsl: cannot save byte tokenizer (no serializable model)");
            std::process::abort();
        }
        TokenizerKind::HuggingFace(tok) => {
            if let Err(e) = tok.save(path, false) {
                eprintln!("nsl: failed to save tokenizer to '{path}': {e}");
                std::process::abort();
            }
        }
    }
}

/// Encode a string into a 1-D tensor of token IDs (f64).
///
/// Arguments:
///   - `handle`: tokenizer handle
///   - `text_ptr`: C string pointer to the text to encode
///
/// Returns a tensor pointer (i64) — shape [seq_len].
#[no_mangle]
pub extern "C" fn nsl_tokenizer_encode(handle: i64, text_ptr: i64) -> i64 {
    let text = unsafe { cstr_to_str(text_ptr) };
    match get_tokenizer(handle) {
        TokenizerKind::Byte => {
            let values: Vec<f64> = text.as_bytes().iter().map(|&b| b as f64).collect();
            make_1d_tensor(&values)
        }
        TokenizerKind::HuggingFace(tok) => {
            let encoding = match tok.encode(text, false) {
                Ok(enc) => enc,
                Err(e) => {
                    eprintln!("nsl: tokenizer encode failed: {e}");
                    std::process::abort();
                }
            };
            let values: Vec<f64> = encoding.get_ids().iter().map(|&id| id as f64).collect();
            make_1d_tensor(&values)
        }
    }
}

/// Decode a 1-D tensor of token IDs back into a string.
///
/// Arguments:
///   - `handle`: tokenizer handle
///   - `tensor_ptr`: pointer to a 1-D NslTensor of token IDs (f64)
///
/// Returns a C string pointer (i64). Caller should free with `nsl_string_free`.
#[no_mangle]
pub extern "C" fn nsl_tokenizer_decode(handle: i64, tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    match get_tokenizer(handle) {
        TokenizerKind::Byte => {
            let bytes: Vec<u8> = (0..tensor.len as usize)
                .map(|i| if tensor.dtype == 1 {
                    unsafe { (*tensor.data_f32().add(i)) as u8 }
                } else {
                    unsafe { (*tensor.data_f64().add(i)) as u8 }
                })
                .collect();
            let s = String::from_utf8_lossy(&bytes);
            alloc_cstring(&s)
        }
        TokenizerKind::HuggingFace(tok) => {
            let ids: Vec<u32> = (0..tensor.len as usize)
                .map(|i| if tensor.dtype == 1 {
                    unsafe { (*tensor.data_f32().add(i)) as u32 }
                } else {
                    unsafe { (*tensor.data_f64().add(i)) as u32 }
                })
                .collect();
            match tok.decode(&ids, true) {
                Ok(s) => alloc_cstring(&s),
                Err(e) => {
                    eprintln!("nsl: tokenizer decode failed: {e}");
                    std::process::abort();
                }
            }
        }
    }
}

/// Return the vocabulary size of the tokenizer.
#[no_mangle]
pub extern "C" fn nsl_tokenizer_vocab_size(handle: i64) -> i64 {
    match get_tokenizer(handle) {
        TokenizerKind::Byte => 256,
        TokenizerKind::HuggingFace(tok) => tok.get_vocab_size(true) as i64,
    }
}

/// Batch-encode a list of strings with optional padding and truncation.
///
/// Arguments:
///   - `handle`: tokenizer handle
///   - `texts_list`: pointer to an `NslList` of C string pointers
///   - `padding`: 1 to pad all sequences to `max_len`, 0 to skip
///   - `truncation`: 1 to truncate sequences longer than `max_len`, 0 to skip
///   - `max_len`: maximum sequence length (0 = use longest in batch)
///
/// Returns a pointer to an `NslList` containing two tensors:
///   [0] = input_ids  (2-D tensor [batch, seq_len])
///   [1] = attention_mask (2-D tensor [batch, seq_len])
#[no_mangle]
pub extern "C" fn nsl_tokenizer_encode_batch(
    handle: i64,
    texts_list: i64,
    padding: i8,
    truncation: i8,
    max_len: i64,
) -> i64 {
    let list = NslList::from_ptr(texts_list);
    let batch_size = list.len as usize;

    // Encode each text individually
    let mut all_ids: Vec<Vec<f64>> = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let text_ptr = unsafe { *list.data.add(i) };
        let text = unsafe { cstr_to_str(text_ptr) };

        let ids: Vec<f64> = match get_tokenizer(handle) {
            TokenizerKind::Byte => text.as_bytes().iter().map(|&b| b as f64).collect(),
            TokenizerKind::HuggingFace(tok) => {
                let encoding = match tok.encode(text, false) {
                    Ok(enc) => enc,
                    Err(e) => {
                        eprintln!("nsl: tokenizer batch encode failed: {e}");
                        std::process::abort();
                    }
                };
                encoding.get_ids().iter().map(|&id| id as f64).collect()
            }
        };
        all_ids.push(ids);
    }

    // Determine effective max_len
    let longest = all_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
    let effective_max = if max_len > 0 {
        max_len as usize
    } else {
        longest
    };

    // Apply truncation and padding, build flat arrays
    let seq_len = if padding != 0 { effective_max } else { longest };
    let mut flat_ids = Vec::with_capacity(batch_size * seq_len);
    let mut flat_mask = Vec::with_capacity(batch_size * seq_len);

    for ids in &all_ids {
        let mut row = ids.clone();

        // Truncation
        if truncation != 0 && row.len() > effective_max {
            row.truncate(effective_max);
        }

        let real_len = row.len();

        // Padding
        if padding != 0 && row.len() < seq_len {
            row.resize(seq_len, 0.0);
        }

        let row_len = row.len();
        flat_ids.extend_from_slice(&row);
        for j in 0..row_len {
            flat_mask.push(if j < real_len { 1.0 } else { 0.0 });
        }
    }

    let actual_seq_len = if batch_size > 0 {
        flat_ids.len() / batch_size
    } else {
        0
    };

    let ids_tensor = make_2d_tensor(batch_size, actual_seq_len, &flat_ids);
    let mask_tensor = make_2d_tensor(batch_size, actual_seq_len, &flat_mask);

    // Return as NslList [ids_tensor, mask_tensor]
    let result_list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(result_list, ids_tensor);
    crate::list::nsl_list_push(result_list, mask_tensor);
    result_list
}
