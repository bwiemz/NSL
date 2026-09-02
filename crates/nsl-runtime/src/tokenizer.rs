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
    let cstr = unsafe { CStr::from_ptr(ptr as *const c_char) };
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

    // Loud in tests, safe in release: a length mismatch means the caller built
    // ragged rows, which is a bug to fix rather than to silently truncate.
    debug_assert_eq!(
        flat.len(),
        len as usize,
        "make_2d_tensor got {} values for a {rows}x{cols} tensor",
        flat.len()
    );
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
    // `flat` is clamped rather than trusted: callers that build ragged rows
    // would otherwise write past an allocation sized rows*cols.
    for (i, &v) in flat.iter().take(len as usize).enumerate() {
        unsafe { *data.add(i) = v };
    }
    for i in flat.len()..(len as usize) {
        unsafe { *data.add(i) = 0.0 };
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
        // Seed the full 256-surrogate byte-level alphabet. Without it the
        // vocabulary only contains characters the corpus happened to exhibit, so
        // any byte absent from training has no token at all and encoding silently
        // DROPS it — the tokenizer is not total over its own input domain.
        // `ByteLevel::alphabet()` returns an `AHashSet`; the builder wants std's.
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build();
    let mut trainer = TrainerWrapper::BpeTrainer(bpe_trainer);

    let mut tokenizer = Tokenizer::new(
        tokenizers::ModelWrapper::BPE(BPE::default()),
    );
    // `add_prefix_space = false`, unlike `ByteLevel::default()`. With it on, the
    // pre-tokenizer PREPENDS a space to every input, so `decode(encode(x))`
    // returns " " + x — the tokenizer is not a faithful round trip even with a
    // decoder attached. `tokenizer_bpe::assemble` uses false for the same reason.
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, true)),
    ));
    // A ByteLevel pre-tokenizer REQUIRES the matching decoder. Without one,
    // `Tokenizer::decode` joins the raw surrogate surfaces with spaces, so
    // `decode(encode(x))` never returns `x` for any input — every space comes
    // back as "Ġ" and a space is inserted between tokens. This shipped broken:
    // 0 of 158 round-trip samples matched.
    tokenizer.with_decoder(Some(tokenizers::DecoderWrapper::ByteLevel(
        ByteLevel::default(),
    )));

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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
///   - `padding`: 1 to pad every row to at least `max_len`, 0 to pad only to the
///     longest row in the batch. A 2-D tensor cannot be ragged, so short rows
///     are always zero-filled and the attention mask marks the padding; this
///     argument controls the width, not whether padding happens.
///   - `truncation`: 1 to cap rows at `max_len`, 0 to widen the tensor instead so
///     no token is dropped
///   - `max_len`: target sequence length (0 = use the longest row in the batch)
///
/// Returns a pointer to an `NslList` containing two tensors:
///   [0] = input_ids  (2-D tensor [batch, seq_len])
///   [1] = attention_mask (2-D tensor [batch, seq_len])
#[unsafe(no_mangle)]
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

    // Every row is materialised at exactly `seq_len`. The previous version
    // emitted rows at their natural lengths and then inferred the width by
    // integer-dividing the total, so any ragged batch produced a flat buffer
    // longer than the rows*cols tensor it was written into.
    let cap = if truncation != 0 { effective_max } else { usize::MAX };
    let truncated: Vec<&[f64]> = all_ids
        .iter()
        .map(|ids| &ids[..ids.len().min(cap)])
        .collect();
    let widest = truncated.iter().map(|row| row.len()).max().unwrap_or(0);
    // `padding` and `truncation` are independent, as they are in the HF
    // tokenizers this mirrors: padding fills rows UP TO `max_len`, and only
    // truncation may cut one down. So a row longer than `max_len` widens the
    // batch rather than losing tokens the caller explicitly asked to keep.
    //
    // That is the right default — silently discarding tokens when
    // `truncation == 0` would be worse than a wide tensor — but it does mean the
    // returned width is not the requested one and can vary per batch, which
    // shows up downstream as a positional-table bounds abort or as a shape that
    // a cuda-graph capture cannot reuse. Those are confusing symptoms for a
    // cause that lives here, so say it once.
    let seq_len = if padding != 0 {
        if max_len > 0 && widest > effective_max {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[nsl] warning: encode_batch padded to {widest}, not the requested \
                     max_len={max_len}: a row is longer and truncation is off, so no tokens \
                     were dropped. The batch width will vary with the longest document. \
                     Pass truncation=1 for a fixed [batch, {max_len}] shape."
                );
            }
        }
        effective_max.max(widest)
    } else {
        widest
    };

    let mut flat_ids = Vec::with_capacity(batch_size * seq_len);
    let mut flat_mask = Vec::with_capacity(batch_size * seq_len);
    for row in &truncated {
        let real_len = row.len().min(seq_len);
        flat_ids.extend_from_slice(&row[..real_len]);
        flat_ids.resize(flat_ids.len() + (seq_len - real_len), 0.0);
        for j in 0..seq_len {
            flat_mask.push(if j < real_len { 1.0 } else { 0.0 });
        }
    }

    let actual_seq_len = seq_len;

    let ids_tensor = make_2d_tensor(batch_size, actual_seq_len, &flat_ids);
    let mask_tensor = make_2d_tensor(batch_size, actual_seq_len, &flat_mask);

    // Return as NslList [ids_tensor, mask_tensor]
    let result_list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(result_list, ids_tensor);
    crate::list::nsl_list_push(result_list, mask_tensor);
    result_list
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::list::{nsl_list_new, nsl_list_push};
    use crate::tensor::NslTensor;

    /// Build an `NslList` of C string pointers, as the codegen'd caller does.
    fn texts_list(texts: &[&str]) -> (i64, Vec<CString>) {
        let owned: Vec<CString> = texts.iter().map(|t| CString::new(*t).unwrap()).collect();
        let list = nsl_list_new();
        for s in &owned {
            nsl_list_push(list, s.as_ptr() as i64);
        }
        // Keep the CStrings alive for the duration of the call.
        (list, owned)
    }

    fn dims(tensor_ptr: i64) -> (i64, i64) {
        let t = NslTensor::from_ptr(tensor_ptr);
        assert_eq!(t.ndim, 2, "encode_batch must return 2-D tensors");
        unsafe { (*t.shape, *t.shape.add(1)) }
    }

    /// The batch encoder used to emit each row at its natural length and then
    /// infer the tensor width by integer-dividing the total token count by the
    /// batch size. For a ragged batch that width is too small, and every token
    /// past rows*cols was written past the end of the allocation.
    ///
    /// "aaaa", "b" and "cc" are 4, 1 and 2 bytes: 7 tokens over 3 rows floors to
    /// a width of 2, so the old code allocated 3*2 slots and wrote 7 values.
    #[test]
    fn ragged_unpadded_batch_stays_within_its_allocation() {
        let handle = nsl_byte_tokenizer_new();
        let (list, _owned) = texts_list(&["aaaa", "b", "cc"]);

        let result = nsl_tokenizer_encode_batch(handle, list, /*padding*/ 0, /*truncation*/ 0, 0);
        let ids = crate::list::nsl_list_get(result, 0);
        let mask = crate::list::nsl_list_get(result, 1);

        let (rows, cols) = dims(ids);
        assert_eq!(rows, 3);
        // Width must be the longest row, so no token is dropped and none is
        // written out of bounds.
        assert_eq!(cols, 4, "width must cover the longest row");
        assert_eq!(dims(mask), (3, 4));

        let t = NslTensor::from_ptr(ids);
        assert_eq!(t.len, rows * cols);
        let read = |r: i64, c: i64| unsafe { *t.data_f64().add((r * cols + c) as usize) };
        assert_eq!(
            [read(0, 0), read(0, 1), read(0, 2), read(0, 3)],
            [b'a' as f64; 4]
        );
        assert_eq!(read(1, 0), b'b' as f64);
        // Short rows are zero-filled, and the mask marks the padding.
        assert_eq!([read(1, 1), read(1, 2), read(1, 3)], [0.0; 3]);
        let m = NslTensor::from_ptr(mask);
        let mread = |r: i64, c: i64| unsafe { *m.data_f64().add((r * cols + c) as usize) };
        assert_eq!([mread(1, 0), mread(1, 1)], [1.0, 0.0]);
    }

    /// With truncation on, no row may exceed max_len even when padding is off.
    #[test]
    fn truncation_bounds_the_width() {
        let handle = nsl_byte_tokenizer_new();
        let (list, _owned) = texts_list(&["aaaaaaaa", "bb"]);
        let result = nsl_tokenizer_encode_batch(handle, list, 0, /*truncation*/ 1, /*max_len*/ 3);
        let ids = crate::list::nsl_list_get(result, 0);
        assert_eq!(dims(ids), (2, 3));
        assert_eq!(NslTensor::from_ptr(ids).len, 6);
    }

    /// Padding to max_len must not shrink a row that is longer than max_len
    /// when truncation is off — that would silently discard tokens.
    ///
    /// The width is therefore NOT guaranteed to be `max_len` in this
    /// combination; it is `max(max_len, longest)`, matching HF's independent
    /// padding/truncation knobs, and the call warns once so the wider shape is
    /// not a mystery downstream. Callers needing a fixed width pass
    /// `truncation=1` (see `truncation_bounds_the_width`).
    #[test]
    fn padding_without_truncation_keeps_long_rows() {
        let handle = nsl_byte_tokenizer_new();
        let (list, _owned) = texts_list(&["aaaaaa", "b"]);
        let result = nsl_tokenizer_encode_batch(handle, list, /*padding*/ 1, 0, /*max_len*/ 2);
        let ids = crate::list::nsl_list_get(result, 0);
        let (rows, cols) = dims(ids);
        assert_eq!((rows, cols), (2, 6));
        assert_eq!(NslTensor::from_ptr(ids).len, 12);
    }

    /// `nsl_bpe_train` is the tokenizer NSL programs get from `bpe_train(...)`.
    /// It attached a ByteLevel pre-tokenizer with no matching decoder, so
    /// `Tokenizer::decode` joined the raw surrogate surfaces with spaces and
    /// `decode(encode(x))` never returned `x` for ANY input — every space came
    /// back as "Ġ" plus an inserted separator. It also never seeded the
    /// byte-level alphabet, so a byte absent from the corpus had no token and
    /// encoding silently dropped it.
    #[test]
    fn bpe_train_roundtrips_and_covers_every_byte() {
        let dir = std::env::temp_dir().join(format!("nsl_bpe_train_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let corpus = dir.join("corpus.txt");
        // Deliberately narrow: no digits, no braces, no non-ASCII. An unseeded
        // alphabet would leave all of those unrepresentable.
        let body = "fn forward self x return x\n".repeat(200);
        std::fs::write(&corpus, &body).expect("write corpus");

        let path = CString::new(corpus.to_string_lossy().as_ref()).unwrap();
        let handle = nsl_bpe_train(path.as_ptr() as i64, 400, 2, 0);
        assert_ne!(handle, 0, "bpe_train returned no handle");

        for text in [
            "fn forward self x return x\n",
            "let y = 42",
            "  indented\twith\ttabs  ",
            "h\u{e9}llo \u{1F600}",
        ] {
            let c = CString::new(text).unwrap();
            let ids = nsl_tokenizer_encode(handle, c.as_ptr() as i64);
            let back = nsl_tokenizer_decode(handle, ids);
            let got = unsafe { cstr_to_str(back) };
            assert_eq!(got, text, "bpe_train tokenizer failed to round-trip {text:?}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_batch_does_not_panic() {
        let handle = nsl_byte_tokenizer_new();
        let (list, _owned) = texts_list(&[]);
        let result = nsl_tokenizer_encode_batch(handle, list, 1, 1, 8);
        let ids = crate::list::nsl_list_get(result, 0);
        assert_eq!(NslTensor::from_ptr(ids).len, 0);
    }
}
