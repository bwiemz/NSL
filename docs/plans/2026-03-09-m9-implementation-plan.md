# M9 Implementation Plan: Rust Runtime + Tensor Foundation

**Date:** 2026-03-09
**Depends on:** M1-M8 (complete)
**Deliverable:** `let x = zeros([3,4]); print(x @ ones([4,2]))` compiles and runs with `nsl run`

---

## Overview

M9 has three major workstreams:
1. **Migrate C runtime to Rust** -- replace `nsl_runtime.c` with a Rust static library
2. **Add tensor runtime** -- new Rust tensor primitives (alloc, ops, print)
3. **Add `nsl run` command** -- compile + execute in one step

---

## Step 1: Create `nsl-runtime` Rust Crate

**New crate:** `crates/nsl-runtime/`

This crate compiles to a **static library** (`.lib` / `.a`) that replaces `nsl_runtime.c`.
All functions are `#[no_mangle] pub extern "C"` so Cranelift-generated code can call them
with the same symbol names.

### 1.1 Crate Setup
- Add `crates/nsl-runtime/Cargo.toml` with `crate-type = ["staticlib"]`
- Add to workspace `Cargo.toml` members
- Directory structure:
  ```
  crates/nsl-runtime/
    Cargo.toml
    src/
      lib.rs          -- top-level exports
      print.rs        -- nsl_print_int, nsl_print_float, nsl_print_str, nsl_print_bool
      power.rs        -- nsl_pow_int, nsl_pow_float
      memory.rs       -- nsl_alloc, nsl_free
      list.rs         -- NslList, nsl_list_new/push/get/set/len/contains/slice
      string.rs       -- nsl_str_concat, conversions, methods (upper/lower/strip/split/join/replace/find/startswith/endswith/contains)
      string_ops.rs   -- nsl_str_repeat, nsl_str_eq, nsl_str_slice
      dict.rs         -- NslDict with FNV-1a, nsl_dict_new/set_str/get_str/len/contains/keys
      range.rs        -- nsl_range
      hof.rs          -- nsl_map, nsl_filter, nsl_enumerate, nsl_zip, nsl_sorted, nsl_reversed
      math.rs         -- nsl_sqrt, nsl_log, nsl_exp, nsl_sin, nsl_cos, abs, min, max
      assert.rs       -- nsl_assert, nsl_exit
      file_io.rs      -- nsl_read_file, nsl_write_file, nsl_append_file, nsl_file_exists
      args.rs         -- nsl_args_init, nsl_args
      tensor.rs       -- NEW: NslTensor, tensor operations
  ```

### 1.2 Migration Strategy
- Port each C function to Rust with identical symbol name and ABI
- Use `libc` crate for `malloc`/`free`/`printf` where needed, or use Rust's allocator
- For list/dict/string: use raw pointers and manual memory management to preserve the
  existing ABI (all values passed as i64 pointers through Cranelift)
- Run all existing examples after each module migration to verify no regressions
- Delete `nsl_runtime.c` once fully migrated

### 1.3 Validation
- Every existing example (`hello.nsl`, `features.nsl`, `m5_features.nsl`, `m6_features.nsl`,
  `m8_features.nsl`, `modules/main.nsl`) must produce identical output before and after migration

---

## Step 2: Update Linker to Use Rust Static Library

### 2.1 Changes to `nsl-codegen/src/linker.rs`
- Remove `RUNTIME_C_SOURCE` (the embedded C file)
- Remove C compiler detection (`find_c_compiler`, `find_msvc`)
- Instead: build the `nsl-runtime` crate as a static lib and link it
- **Approach:** At build time, `nsl-runtime` produces `nsl_runtime.lib` (Windows) or
  `libnsl_runtime.a` (Unix). The linker links the user's `.obj` file against this static lib.
- The `nsl-cli` crate depends on `nsl-runtime` and knows the path to the static lib
  via a build script or by locating it in the target directory.

### 2.2 Simplified Linking
- On Windows: use `link.exe` with the `.lib` file (still needs MSVC linker for PE format)
- On Unix: use `cc` with the `.a` file
- The key change: we no longer compile C at build time -- the Rust static lib is pre-built

### 2.3 Alternative: Embed Runtime as Object
- Build `nsl-runtime` in a `build.rs` script
- Embed the compiled object bytes with `include_bytes!`
- Write to temp at link time (similar to current approach but with Rust instead of C)
- This avoids requiring the user to have the Rust toolchain at runtime

---

## Step 3: Add `nsl run` Command

### 3.1 Changes to `nsl-cli/src/main.rs`
- Add `Run` variant to the CLI enum (clap)
- `nsl run <file.nsl>` = compile to temp dir + execute + clean up
- Passes through any args after `--` to the compiled program
- Implementation:
  ```
  1. Parse + type-check (existing)
  2. Codegen to .obj in temp dir (existing)
  3. Link to executable in temp dir (existing, updated for Rust runtime)
  4. Execute the binary via std::process::Command
  5. Forward exit code
  6. Clean up temp dir
  ```

### 3.2 CLI Definition
```
nsl run <FILE> [-- <PROGRAM_ARGS>...]    Compile and execute an NSL program
nsl build <FILE> [-o <OUTPUT>]           Compile to a native executable
nsl check <FILE>                         Type-check without compiling
```

---

## Step 4: Add Tensor Runtime (Rust)

### 4.1 Tensor Data Structure

```rust
// In crates/nsl-runtime/src/tensor.rs

#[repr(C)]
pub struct NslTensor {
    data: *mut f64,        // heap-allocated contiguous data (f64 for now)
    shape: *mut i64,       // heap-allocated shape array
    strides: *mut i64,     // heap-allocated strides array
    ndim: i64,             // number of dimensions
    len: i64,              // total number of elements
    refcount: i64,         // reference counting for shared ownership
}
```

### 4.2 Tensor Runtime Functions (Phase 1)

Creation:
- `nsl_tensor_zeros(shape_list: i64) -> i64` -- create tensor filled with 0.0
- `nsl_tensor_ones(shape_list: i64) -> i64` -- create tensor filled with 1.0
- `nsl_tensor_rand(shape_list: i64) -> i64` -- create tensor with uniform random [0,1)
- `nsl_tensor_full(shape_list: i64, value: f64) -> i64` -- create tensor filled with value
- `nsl_tensor_arange(start: f64, stop: f64, step: f64) -> i64` -- range tensor

Element access:
- `nsl_tensor_get(tensor: i64, indices_list: i64) -> f64` -- get single element
- `nsl_tensor_set(tensor: i64, indices_list: i64, value: f64)` -- set single element

Shape operations:
- `nsl_tensor_reshape(tensor: i64, new_shape: i64) -> i64` -- reshape (view if possible)
- `nsl_tensor_transpose(tensor: i64, dim0: i64, dim1: i64) -> i64` -- swap two dims
- `nsl_tensor_shape(tensor: i64) -> i64` -- return shape as list
- `nsl_tensor_ndim(tensor: i64) -> i64` -- number of dimensions

Arithmetic (elementwise, return new tensor):
- `nsl_tensor_add(a: i64, b: i64) -> i64`
- `nsl_tensor_sub(a: i64, b: i64) -> i64`
- `nsl_tensor_mul(a: i64, b: i64) -> i64`
- `nsl_tensor_div(a: i64, b: i64) -> i64`
- `nsl_tensor_neg(a: i64) -> i64`

Scalar-tensor ops:
- `nsl_tensor_add_scalar(a: i64, s: f64) -> i64`
- `nsl_tensor_mul_scalar(a: i64, s: f64) -> i64`

Matrix operations:
- `nsl_tensor_matmul(a: i64, b: i64) -> i64` -- matrix multiply (supports batched)

Reduction:
- `nsl_tensor_sum(tensor: i64) -> f64` -- sum all elements
- `nsl_tensor_mean(tensor: i64) -> f64` -- mean of all elements

Display:
- `nsl_tensor_print(tensor: i64)` -- pretty-print tensor to stdout

Memory:
- `nsl_tensor_clone(tensor: i64) -> i64` -- deep copy
- `nsl_tensor_free(tensor: i64)` -- decrement refcount, free if 0

### 4.3 Implementation Notes
- All tensors are f64 for now (M10 adds dtype system)
- All tensors are CPU-only (M17 adds GPU)
- Matmul uses naive triple-loop (M17 adds BLAS)
- Tensors are passed as i64 (pointer cast) through Cranelift, same pattern as lists/dicts
- Shape validation happens at runtime for now (M10 adds compile-time checking)

---

## Step 5: Tensor Support in Compiler Pipeline

### 5.1 Semantic Analysis (`nsl-semantic`)
- Add `Type::Tensor` variant to the type enum (basic -- just "this is a tensor")
- Register tensor builtin functions: `zeros`, `ones`, `rand`, `full`, `arange`
- Type-check `@` operator to work on tensors (matmul)
- Type-check `+`, `-`, `*`, `/` to work on tensors (elementwise)
- `print()` dispatch for tensor type

### 5.2 AST Changes (`nsl-ast`)
- No AST changes needed for basic tensors -- they use existing function call and operator syntax
- Tensor literals like `zeros([3, 4])` are just function calls with list arguments

### 5.3 Code Generation (`nsl-codegen`)
- Add tensor runtime functions to `RUNTIME_FUNCTIONS` table in `builtins.rs`
- Handle `@` (matmul) operator in binary expression codegen
- Dispatch `+`, `-`, `*`, `/` to tensor variants when operand types are tensors
- Handle `print(tensor)` dispatch
- Handle `zeros()`, `ones()`, `rand()` as builtin function calls

### 5.4 Type Tracking Through Codegen
- Codegen needs to know when a value is a tensor vs int/float/list
- Extend the codegen's value tracking to include a `Tensor` variant
- This determines which runtime function to call for `+`, `print`, etc.

---

## Step 6: Regression Testing

### 6.1 Capture Baseline
- Before any changes, run all examples and capture their output
- Store expected outputs in `tests/expected/` directory

### 6.2 Example Programs
Create `examples/m9_tensors.nsl`:
```nsl
# Basic tensor creation
let a = zeros([2, 3])
print(a)

let b = ones([2, 3])
print(b)

# Elementwise operations
let c = a + b
print(c)

# Scalar operations
let d = b * 2.0
print(d)

# Matrix multiply
let x = ones([3, 4])
let y = ones([4, 2])
let z = x @ y
print(z)

# Reshape
let flat = ones([6])
let matrix = flat.reshape([2, 3])
print(matrix)

# Reductions
let total = z.sum()
print(total)
```

### 6.3 Test Script
Create a shell script or cargo test that:
1. Builds all existing examples, verifies output matches baseline
2. Builds `m9_tensors.nsl`, verifies expected output
3. Tests `nsl run` command on all examples

---

## Step 7: Update Memory Files

After M9 is complete, update:
- `memory/nsl-architecture.md` -- add nsl-runtime crate, tensor runtime API
- `memory/MEMORY.md` -- note M9 completion, Rust runtime architecture

---

## File Changes Summary

### New Files
- `crates/nsl-runtime/Cargo.toml`
- `crates/nsl-runtime/src/lib.rs`
- `crates/nsl-runtime/src/print.rs`
- `crates/nsl-runtime/src/power.rs`
- `crates/nsl-runtime/src/memory.rs`
- `crates/nsl-runtime/src/list.rs`
- `crates/nsl-runtime/src/string.rs`
- `crates/nsl-runtime/src/string_ops.rs`
- `crates/nsl-runtime/src/dict.rs`
- `crates/nsl-runtime/src/range.rs`
- `crates/nsl-runtime/src/hof.rs`
- `crates/nsl-runtime/src/math.rs`
- `crates/nsl-runtime/src/assert.rs`
- `crates/nsl-runtime/src/file_io.rs`
- `crates/nsl-runtime/src/args.rs`
- `crates/nsl-runtime/src/tensor.rs`
- `examples/m9_tensors.nsl`
- `tests/expected/*.txt` (baseline outputs)

### Modified Files
- `Cargo.toml` -- add nsl-runtime to workspace members
- `crates/nsl-cli/Cargo.toml` -- depend on nsl-runtime
- `crates/nsl-cli/src/main.rs` -- add `run` subcommand
- `crates/nsl-codegen/Cargo.toml` -- depend on nsl-runtime
- `crates/nsl-codegen/src/linker.rs` -- link Rust static lib instead of C
- `crates/nsl-codegen/src/builtins.rs` -- add tensor runtime function declarations
- `crates/nsl-codegen/src/expr.rs` -- tensor operator codegen (`@`, `+` on tensors)
- `crates/nsl-codegen/src/stmt.rs` -- tensor print dispatch
- `crates/nsl-semantic/src/builtins.rs` -- register tensor builtins
- `crates/nsl-semantic/src/types.rs` -- add Type::Tensor
- `crates/nsl-semantic/src/checker.rs` -- type-check tensor ops

### Deleted Files
- `crates/nsl-codegen/runtime/nsl_runtime.c`

---

## Implementation Order

Execute in this sequence (each step should be verified before proceeding):

1. **Capture baselines** -- record output of all existing examples
2. **Create nsl-runtime crate** -- skeleton with Cargo.toml, empty lib.rs
3. **Port print functions** -- nsl_print_int/float/str/bool
4. **Port memory functions** -- nsl_alloc, nsl_free
5. **Port list functions** -- NslList and all operations
6. **Port string functions** -- all string operations
7. **Port dict functions** -- NslDict and all operations
8. **Port remaining functions** -- range, HOFs, math, assert, exit, file I/O, args
9. **Update linker** -- link Rust static lib instead of C
10. **Verify all baselines pass** -- every existing example produces identical output
11. **Delete nsl_runtime.c** -- remove C dependency
12. **Add `nsl run` command** -- compile + execute in one step
13. **Implement tensor struct** -- NslTensor allocation and memory management
14. **Implement tensor creation** -- zeros, ones, rand, full, arange
15. **Implement tensor arithmetic** -- add, sub, mul, div, neg, scalar ops
16. **Implement matmul** -- naive triple-loop matrix multiply
17. **Implement tensor utilities** -- reshape, transpose, shape, sum, mean, print
18. **Add tensor to semantic analysis** -- Type::Tensor, builtin registrations
19. **Add tensor to codegen** -- runtime function declarations, operator dispatch
20. **Create m9_tensors.nsl example** -- verify end-to-end
21. **Update memory files** -- record M9 architecture decisions
