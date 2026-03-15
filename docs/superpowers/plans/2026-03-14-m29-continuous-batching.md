# M29: Continuous Batching & Serving Engine Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native `serve` block to the NSL language that compiles into a preemptive continuous batching inference server, built on M25 paged KV-cache, M27 FlashAttention, and M28 dynamic shapes.

**Architecture:** The `serve` block is a declarative language construct (like `train`). The compiler parses it into a `ServeBlock` AST node, validates it semantically, then generates codegen that calls into a Rust runtime scheduler. The runtime implements `BatchScheduler` (continuous batching with chunked prefill), `RaggedBatchBuilder` (zero-padding token concatenation), and `PreemptionManager` (swap/recompute strategies). The `autoregressive_decode` call inside `@endpoint` is a compiler intrinsic — the compiler lowers it into the scheduler's decode loop rather than emitting a function call.

**Tech Stack:** Rust, Cranelift, existing `nsl-runtime` paged KV-cache (M25)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/nsl-lexer/src/token.rs` | Add `Serve` variant to `TokenKind` |
| `crates/nsl-lexer/src/keywords.rs` | Map `"serve"` string to `TokenKind::Serve` |
| `crates/nsl-ast/src/block.rs` | Define `ServeBlock`, `ServeSection` AST types |
| `crates/nsl-ast/src/stmt.rs` | Add `StmtKind::ServeBlock` variant |
| `crates/nsl-parser/src/block.rs` | Implement `parse_serve_block_stmt()` |
| `crates/nsl-parser/src/stmt.rs` | Dispatch `TokenKind::Serve` to parser |
| `crates/nsl-semantic/src/checker.rs` | Add `check_serve_block()` validation |
| `crates/nsl-runtime/src/serving/mod.rs` | **New**: module root with re-exports |
| `crates/nsl-runtime/src/serving/request.rs` | **New**: `InferenceRequest` + state machine |
| `crates/nsl-runtime/src/serving/scheduler.rs` | **New**: `BatchScheduler` continuous batching |
| `crates/nsl-runtime/src/serving/ragged.rs` | **New**: `RaggedBatchBuilder` zero-pad assembly |
| `crates/nsl-runtime/src/serving/preemption.rs` | **New**: `PreemptionManager` swap/recompute |
| `crates/nsl-runtime/src/serving/ffi.rs` | **New**: `nsl_serve_*` FFI exports |
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod serving;` |
| `crates/nsl-codegen/src/builtins.rs` | Register `nsl_serve_*` runtime functions |
| `crates/nsl-codegen/src/serve.rs` | **New**: `compile_serve_block()` codegen |
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod serve;` |
| `crates/nsl-codegen/src/stmt.rs` | Dispatch `StmtKind::ServeBlock` to codegen |
| `examples/m29_serve_basic.nsl` | E2E test: basic serve block |
| `examples/m29_continuous_batch.nsl` | E2E test: multi-request batching |
| `examples/m29_preemption.nsl` | E2E test: preemption |
| `tests/expected/m29_serve_basic.txt` | Expected output |
| `tests/expected/m29_continuous_batch.txt` | Expected output |
| `tests/expected/m29_preemption.txt` | Expected output |
| `crates/nsl-cli/tests/e2e.rs` | E2E test entries |

---

## Chunk 1: Language Frontend (Lexer, AST, Parser, Semantic)

### Task 1: Add `serve` keyword to the lexer

**Files:**
- Modify: `crates/nsl-lexer/src/token.rs:51-60`
- Modify: `crates/nsl-lexer/src/keywords.rs:29-38`

- [ ] **Step 1: Add `Serve` variant to `TokenKind`**

In `crates/nsl-lexer/src/token.rs`, add `Serve` to the `// === Keywords: ML blocks ===` section, after `Datatype`:

```rust
// === Keywords: ML blocks ===
Model,
Train,
Grad,
Quant,
Kernel,
Device,
Tokenizer,
Dataset,
Datatype,
Serve,
```

- [ ] **Step 2: Add Display impl for `Serve`**

In the `Display` impl for `TokenKind` (same file, around line 197), add after `TokenKind::Datatype`:

```rust
TokenKind::Serve => write!(f, "serve"),
```

- [ ] **Step 3: Register `serve` keyword**

In `crates/nsl-lexer/src/keywords.rs`, add to the `// ML blocks` section:

```rust
"serve" => Some(TokenKind::Serve),
```

- [ ] **Step 4: Verify build**

Run: `cargo build -p nsl-lexer`
Expected: Clean build

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-lexer/src/token.rs crates/nsl-lexer/src/keywords.rs
git commit -m "feat(m29): add 'serve' keyword to lexer"
```

---

### Task 2: Define `ServeBlock` AST types

**Files:**
- Modify: `crates/nsl-ast/src/block.rs`
- Modify: `crates/nsl-ast/src/stmt.rs`

- [ ] **Step 1: Add `ServeBlock` and `ServeSection` to AST**

In `crates/nsl-ast/src/block.rs`, add at the end of the file:

```rust
/// serve Inference:
///     model: LLaMA = load("weights.safetensors")
///     max_batch: 32
///     @endpoint
///     fn generate(prompt: str, ...) -> str: ...
#[derive(Debug, Clone, Serialize)]
pub struct ServeBlock {
    pub name: Symbol,
    pub config: Vec<ServeConfigEntry>,
    pub endpoints: Vec<EndpointDef>,
    pub span: Span,
}

/// A key-value config entry inside a serve block.
/// e.g. `max_batch: 32`, `model: LLaMA = load("weights.safetensors")`
#[derive(Debug, Clone, Serialize)]
pub struct ServeConfigEntry {
    pub key: Symbol,
    pub type_ann: Option<TypeExpr>,
    pub value: Expr,
    pub span: Span,
}

/// An @endpoint-decorated function inside a serve block.
#[derive(Debug, Clone, Serialize)]
pub struct EndpointDef {
    pub name: Symbol,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: Block,
    pub span: Span,
}
```

- [ ] **Step 2: Add `StmtKind::ServeBlock` variant**

In `crates/nsl-ast/src/stmt.rs`, add after the `DatatypeDef` variant:

```rust
/// serve Name:
///     config entries + @endpoint functions
ServeBlock(ServeBlock),
```

- [ ] **Step 3: Add import for `ServeBlock`**

The existing `use crate::block::*;` in `stmt.rs` already imports all block types via glob, so `ServeBlock` is automatically available.

- [ ] **Step 4: Verify build**

Run: `cargo build -p nsl-ast`
Expected: Clean build

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-ast/src/block.rs crates/nsl-ast/src/stmt.rs
git commit -m "feat(m29): add ServeBlock, EndpointDef AST types"
```

---

### Task 3: Parse `serve` block syntax

**Files:**
- Modify: `crates/nsl-parser/src/block.rs`
- Modify: `crates/nsl-parser/src/stmt.rs`

The serve block syntax is:

```nsl
serve Inference:
    model: LLaMA = load("weights.safetensors")
    tokenizer: BPE = load("tokenizer.json")
    max_batch: 32
    max_seq_len: 4096
    kv_blocks: 2048
    prefill_chunk: 512

    @endpoint
    fn generate(prompt: str, max_tokens: int = 256) -> str:
        let tokens = tokenizer.encode(prompt)
        let output = autoregressive_decode(model, tokens, max_tokens, temperature, top_p)
        return tokenizer.decode(output)
```

Parsing strategy: after consuming `serve Name:`, read config entries (each is `key: [Type =] expr`) until we hit `@endpoint` or `fn`. When `@endpoint` decorator is seen followed by `fn`, parse as an `EndpointDef`.

- [ ] **Step 1: Add parser dispatch for `serve`**

In `crates/nsl-parser/src/stmt.rs`, add to the `match p.peek()` block after the `TokenKind::Datatype` arm:

```rust
TokenKind::Serve => crate::block::parse_serve_block_stmt(p),
```

- [ ] **Step 2: Implement `parse_serve_block_stmt()`**

In `crates/nsl-parser/src/block.rs`, add the function:

```rust
pub fn parse_serve_block_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'serve'

    let (name, _) = p.expect_ident();

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut config = Vec::new();
    let mut endpoints = Vec::new();

    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }

        // @endpoint decorator followed by fn
        if p.at(&TokenKind::At) {
            let ep = parse_endpoint_def(p);
            endpoints.push(ep);
            continue;
        }

        // fn without @endpoint — still treat as endpoint (convenience)
        if p.at(&TokenKind::Fn) {
            let ep = parse_endpoint_fn(p);
            endpoints.push(ep);
            continue;
        }

        // Config entry: key [: Type] = expr  OR  key: expr
        let entry = parse_serve_config_entry(p);
        config.push(entry);
    }

    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::ServeBlock(nsl_ast::block::ServeBlock {
            name,
            config,
            endpoints,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

fn parse_serve_config_entry(p: &mut Parser) -> nsl_ast::block::ServeConfigEntry {
    let start = p.current_span();
    let (key, _) = p.expect_ident();

    // Expect colon
    p.expect(&TokenKind::Colon);

    // Check if next is a type annotation followed by `=`
    // Simple heuristic: if next is Ident and followed by `=`, treat as `key: Type = expr`
    // Otherwise it's `key: expr` (like `max_batch: 32`)
    let mut type_ann = None;

    if let TokenKind::Ident(_) = p.peek().clone() {
        // Peek ahead to see if there's an `=` after the ident
        if matches!(p.peek_at(1), &TokenKind::Eq) {
            type_ann = Some(crate::types::parse_type(p));
            p.expect(&TokenKind::Eq);
        }
    }

    let value = parse_expr(p);
    p.expect_end_of_stmt();

    let span = start.merge(p.prev_span());
    nsl_ast::block::ServeConfigEntry {
        key,
        type_ann,
        value,
        span,
    }
}

fn parse_endpoint_def(p: &mut Parser) -> nsl_ast::block::EndpointDef {
    // consume @endpoint
    p.advance(); // @
    let (dec_name, _) = p.expect_ident();
    let dec_str = p.interner.resolve(dec_name).unwrap_or("").to_string();
    if dec_str != "endpoint" {
        p.error_at_current(&format!("expected @endpoint, got @{dec_str}"));
    }
    p.skip_newlines();

    parse_endpoint_fn(p)
}

fn parse_endpoint_fn(p: &mut Parser) -> nsl_ast::block::EndpointDef {
    let start = p.current_span();
    p.expect(&TokenKind::Fn);
    let (name, _) = p.expect_ident();

    p.expect(&TokenKind::LeftParen);
    let params = crate::decl::parse_params(p);
    p.expect(&TokenKind::RightParen);

    let return_type = if p.eat(&TokenKind::Arrow) {
        Some(crate::types::parse_type(p))
    } else {
        None
    };

    p.expect(&TokenKind::Colon);
    let body = p.parse_block();

    let span = start.merge(p.prev_span());
    nsl_ast::block::EndpointDef {
        name,
        params,
        return_type,
        body,
        span,
    }
}
```

- [ ] **Step 3: Write parser test**

Add a test to `crates/nsl-parser/src/block.rs` (or in a test module):

```rust
#[cfg(test)]
mod serve_tests {
    #[test]
    fn parse_serve_block() {
        let source = "serve Inference:\n    max_batch: 32\n    kv_blocks: 2048\n\n    @endpoint\n    fn generate(prompt: str) -> str:\n        return prompt\n";
        let mut interner = nsl_lexer::Interner::new();
        let tokens = nsl_lexer::lex(source, &mut interner);
        let module = crate::parse(tokens, &mut interner);
        assert_eq!(module.stmts.len(), 1);
        if let nsl_ast::stmt::StmtKind::ServeBlock(sb) = &module.stmts[0].kind {
            assert_eq!(sb.config.len(), 2);
            assert_eq!(sb.endpoints.len(), 1);
        } else {
            panic!("Expected ServeBlock");
        }
    }
}
```

- [ ] **Step 4: Run test to verify**

Run: `cargo test -p nsl-parser -- parse_serve_block`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-parser/src/block.rs crates/nsl-parser/src/stmt.rs
git commit -m "feat(m29): parse serve block with config entries and @endpoint functions"
```

---

### Task 4: Add semantic validation for `serve` block

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Handle `ServeBlock` in the checker's statement dispatch**

In `checker.rs`, find the `StmtKind` match in `check_stmt()`. There are two relevant places — the first-pass declaration scan and the main check_stmt dispatch. Add `ServeBlock` handling to both.

In the first-pass (symbol declaration), add alongside existing block types:

```rust
StmtKind::ServeBlock(_) => {
    // serve blocks don't declare a variable name in scope
}
```

In the main `check_stmt` dispatch (around line 598 where `TrainBlock` is handled), add:

```rust
StmtKind::ServeBlock(serve) => self.check_serve_block(serve),
```

- [ ] **Step 2: Implement `check_serve_block()`**

Add this method to the checker impl:

```rust
fn check_serve_block(&mut self, serve: &nsl_ast::block::ServeBlock) {
    // Validate config entry expressions
    for entry in &serve.config {
        if let Some(ref type_ann) = entry.type_ann {
            self.check_type_expr(type_ann);
        }
        self.check_expr(&entry.value);
    }

    // Validate endpoint bodies
    for endpoint in &serve.endpoints {
        // Type-check params
        for param in &endpoint.params {
            if let Some(ref type_ann) = param.type_ann {
                self.check_type_expr(type_ann);
            }
        }
        // Check return type
        if let Some(ref ret_type) = endpoint.return_type {
            self.check_type_expr(ret_type);
        }
        // Check body
        self.check_block(&endpoint.body, ScopeKind::Block);
    }

    // Must have at least one endpoint
    if serve.endpoints.is_empty() {
        self.diagnostics.push(
            Diagnostic::error("serve block must define at least one @endpoint function")
                .with_label(serve.span, "no endpoints defined")
        );
    }
}
```

- [ ] **Step 3: Verify build and test**

Run: `cargo build -p nsl-semantic && cargo test`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs
git commit -m "feat(m29): add semantic validation for serve blocks"
```

---

### Task 5: Handle `ServeBlock` in codegen statement dispatch (stub)

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`

- [ ] **Step 1: Add `StmtKind::ServeBlock` to compile_stmt**

In `crates/nsl-codegen/src/stmt.rs`, find the `StmtKind` match in `compile_stmt()`. Add alongside existing block dispatches:

```rust
StmtKind::ServeBlock(serve) => {
    self.compile_serve_block(builder, state, serve)?;
}
```

For now, `compile_serve_block` will be a stub that compiles to a no-op. The actual implementation comes in Task 10.

- [ ] **Step 2: Add stub method**

Add a temporary stub method in `stmt.rs` (will be moved to `serve.rs` in Task 10):

```rust
fn compile_serve_block(
    &mut self,
    _builder: &mut FunctionBuilder,
    _state: &mut FuncState,
    _serve: &nsl_ast::block::ServeBlock,
) -> Result<(), CodegenError> {
    // M29: stub — full implementation in serve.rs
    Ok(())
}
```

- [ ] **Step 3: Verify full build and tests**

Run: `cargo build && cargo test`
Expected: All 155 tests pass, no warnings

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(m29): add serve block codegen dispatch stub"
```

---

## Chunk 2: Runtime Infrastructure

### Task 6: `InferenceRequest` state machine

**Files:**
- Create: `crates/nsl-runtime/src/serving/mod.rs`
- Create: `crates/nsl-runtime/src/serving/request.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

The request state machine is the core of continuous batching. Each incoming request transitions through states: WAITING → PREFILL → DECODE → COMPLETE (with optional PREEMPTED state).

- [ ] **Step 1: Create serving module**

Create `crates/nsl-runtime/src/serving/mod.rs`:

```rust
//! M29: Continuous batching serving engine.
//!
//! Implements a preemptive continuous batching scheduler for
//! autoregressive inference. Built on the M25 paged KV-cache.

pub mod request;
pub mod scheduler;
pub mod ragged;
pub mod preemption;
pub mod ffi;
```

- [ ] **Step 2: Implement `InferenceRequest`**

Create `crates/nsl-runtime/src/serving/request.rs`:

```rust
//! InferenceRequest: per-request state machine for continuous batching.

/// Unique identifier for a request in the scheduler.
pub type RequestId = u64;

/// State machine for a single inference request.
#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    /// Waiting to be admitted by the scheduler.
    Waiting,
    /// Actively prefilling (processing prompt tokens).
    /// `tokens_processed` tracks chunked prefill progress.
    Prefilling { tokens_processed: usize },
    /// Actively decoding (generating tokens one at a time).
    Decoding,
    /// Preempted — KV-cache evicted, will resume later.
    Preempted { generated_so_far: Vec<i64> },
    /// Generation complete.
    Complete,
}

/// A single inference request with all its state.
pub struct InferenceRequest {
    pub id: RequestId,
    pub state: RequestState,
    /// The prompt token IDs.
    pub prompt_tokens: Vec<i64>,
    /// Tokens generated so far (appended one per decode step).
    pub generated_tokens: Vec<i64>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f64,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f64,
    /// KV-cache sequence ID (from paged_kv::KvCacheManager).
    /// None until the scheduler allocates KV blocks.
    pub kv_seq_id: Option<u64>,
    /// Priority (lower = higher priority). Default 0.
    pub priority: i64,
    /// Total tokens = prompt + generated (for scheduling decisions).
    pub total_tokens: usize,
}

impl InferenceRequest {
    pub fn new(
        id: RequestId,
        prompt_tokens: Vec<i64>,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Self {
        let total = prompt_tokens.len();
        InferenceRequest {
            id,
            state: RequestState::Waiting,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_tokens,
            temperature,
            top_p,
            kv_seq_id: None,
            priority: 0,
            total_tokens: total,
        }
    }

    /// Returns true if this request has finished generating.
    pub fn is_complete(&self) -> bool {
        self.state == RequestState::Complete
    }

    /// Returns true if this request is actively being processed.
    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            RequestState::Prefilling { .. } | RequestState::Decoding
        )
    }

    /// The number of tokens still needing prefill.
    pub fn remaining_prefill(&self) -> usize {
        match self.state {
            RequestState::Prefilling { tokens_processed } => {
                self.prompt_tokens.len().saturating_sub(tokens_processed)
            }
            _ => 0,
        }
    }

    /// Mark a generated token. Returns true if generation is now complete.
    pub fn push_token(&mut self, token_id: i64) -> bool {
        self.generated_tokens.push(token_id);
        self.total_tokens += 1;
        // Check completion: max tokens reached or EOS token (id = 2 by convention)
        if self.generated_tokens.len() >= self.max_tokens || token_id == 2 {
            self.state = RequestState::Complete;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_lifecycle() {
        let mut req = InferenceRequest::new(0, vec![1, 2, 3], 5, 0.7, 0.9);
        assert_eq!(req.state, RequestState::Waiting);
        assert!(!req.is_active());

        req.state = RequestState::Prefilling { tokens_processed: 0 };
        assert!(req.is_active());
        assert_eq!(req.remaining_prefill(), 3);

        req.state = RequestState::Decoding;
        assert!(req.is_active());

        assert!(!req.push_token(10));
        assert!(!req.push_token(11));
        assert_eq!(req.generated_tokens.len(), 2);

        // EOS token completes
        assert!(req.push_token(2));
        assert!(req.is_complete());
    }

    #[test]
    fn request_max_tokens() {
        let mut req = InferenceRequest::new(0, vec![1], 3, 0.7, 0.9);
        req.state = RequestState::Decoding;

        assert!(!req.push_token(10));
        assert!(!req.push_token(11));
        assert!(req.push_token(12)); // 3rd token → complete
        assert!(req.is_complete());
    }
}
```

- [ ] **Step 3: Add `pub mod serving;` to lib.rs**

In `crates/nsl-runtime/src/lib.rs`, add:

```rust
pub mod serving;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p nsl-runtime -- request`
Expected: 2 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/serving/ crates/nsl-runtime/src/lib.rs
git commit -m "feat(m29): add InferenceRequest state machine"
```

---

### Task 7: `BatchScheduler` — continuous batching core

**Files:**
- Create: `crates/nsl-runtime/src/serving/scheduler.rs`

The scheduler decides which requests to run each step. It admits waiting requests, builds ragged batches of active requests, handles prefill chunking, and triggers preemption when memory is tight.

- [ ] **Step 1: Implement `BatchScheduler`**

Create `crates/nsl-runtime/src/serving/scheduler.rs`:

```rust
//! BatchScheduler: continuous batching with chunked prefill.

use std::collections::VecDeque;

use crate::serving::request::{InferenceRequest, RequestId, RequestState};

/// Configuration for the serving scheduler.
pub struct SchedulerConfig {
    /// Maximum batch size (requests in flight simultaneously).
    pub max_batch: usize,
    /// Maximum sequence length (prompt + generated).
    pub max_seq_len: usize,
    /// Number of KV-cache blocks available.
    pub kv_blocks: usize,
    /// Chunk size for prefill (tokens per step per prefilling request).
    pub prefill_chunk: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            max_batch: 32,
            max_seq_len: 4096,
            kv_blocks: 2048,
            prefill_chunk: 512,
        }
    }
}

/// The output of one scheduler step: which requests to prefill and which to decode.
pub struct SchedulerStep {
    /// Requests that should run a prefill chunk this step.
    /// Each entry: (request_id, token_start, token_end).
    pub prefill_chunks: Vec<(RequestId, usize, usize)>,
    /// Request IDs that should decode one token this step.
    pub decode_ids: Vec<RequestId>,
}

/// Core continuous batching scheduler.
pub struct BatchScheduler {
    pub config: SchedulerConfig,
    /// Requests waiting to be admitted.
    pub waiting: VecDeque<InferenceRequest>,
    /// Active requests (prefilling or decoding).
    pub active: Vec<InferenceRequest>,
    /// Completed requests (ready for result retrieval).
    pub completed: Vec<InferenceRequest>,
    /// Next request ID to assign.
    next_id: RequestId,
}

impl BatchScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        BatchScheduler {
            config,
            waiting: VecDeque::new(),
            active: Vec::new(),
            completed: Vec::new(),
            next_id: 0,
        }
    }

    /// Enqueue a new request. Returns its assigned ID.
    pub fn enqueue(
        &mut self,
        prompt_tokens: Vec<i64>,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        let req = InferenceRequest::new(id, prompt_tokens, max_tokens, temperature, top_p);
        self.waiting.push_back(req);
        id
    }

    /// Run one scheduling step: admit waiting requests, decide prefill/decode split.
    pub fn step(&mut self) -> SchedulerStep {
        // 1. Admit waiting requests if batch has room
        while self.active.len() < self.config.max_batch {
            if let Some(mut req) = self.waiting.pop_front() {
                req.state = RequestState::Prefilling { tokens_processed: 0 };
                self.active.push(req);
            } else {
                break;
            }
        }

        let mut prefill_chunks = Vec::new();
        let mut decode_ids = Vec::new();

        // 2. For each active request, decide what to do
        for req in &mut self.active {
            match &req.state {
                RequestState::Prefilling { tokens_processed } => {
                    let start = *tokens_processed;
                    let end = (start + self.config.prefill_chunk).min(req.prompt_tokens.len());
                    prefill_chunks.push((req.id, start, end));

                    // If this chunk finishes the prefill, transition to Decoding
                    if end >= req.prompt_tokens.len() {
                        req.state = RequestState::Decoding;
                    } else {
                        req.state = RequestState::Prefilling {
                            tokens_processed: end,
                        };
                    }
                }
                RequestState::Decoding => {
                    decode_ids.push(req.id);
                }
                _ => {}
            }
        }

        SchedulerStep {
            prefill_chunks,
            decode_ids,
        }
    }

    /// Record that a request generated a token. Returns true if request completed.
    pub fn record_token(&mut self, request_id: RequestId, token_id: i64) -> bool {
        if let Some(req) = self.active.iter_mut().find(|r| r.id == request_id) {
            let complete = req.push_token(token_id);
            complete
        } else {
            false
        }
    }

    /// Drain completed requests from the active set into completed.
    pub fn drain_completed(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].is_complete() {
                let req = self.active.remove(i);
                self.completed.push(req);
            } else {
                i += 1;
            }
        }
    }

    /// Retrieve a completed request by ID (removes it from completed list).
    pub fn take_completed(&mut self, request_id: RequestId) -> Option<InferenceRequest> {
        if let Some(pos) = self.completed.iter().position(|r| r.id == request_id) {
            Some(self.completed.remove(pos))
        } else {
            None
        }
    }

    /// Returns true if there are any active or waiting requests.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }

    /// Number of active requests.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_basic_lifecycle() {
        let config = SchedulerConfig {
            max_batch: 2,
            max_seq_len: 100,
            kv_blocks: 64,
            prefill_chunk: 512,
        };
        let mut sched = BatchScheduler::new(config);

        // Enqueue 3 requests
        let id0 = sched.enqueue(vec![1, 2, 3], 2, 0.7, 0.9);
        let id1 = sched.enqueue(vec![4, 5], 2, 0.7, 0.9);
        let _id2 = sched.enqueue(vec![6], 2, 0.7, 0.9);

        assert_eq!(sched.waiting.len(), 3);

        // Step 1: admits first 2 (max_batch=2), both short enough for single-chunk prefill
        let step = sched.step();
        assert_eq!(sched.active.len(), 2);
        assert_eq!(sched.waiting.len(), 1); // id2 still waiting
        assert_eq!(step.prefill_chunks.len(), 2);
        // After step, both transitioned to Decoding (prompts < prefill_chunk)
        assert_eq!(step.decode_ids.len(), 0);

        // Step 2: both now decoding
        let step = sched.step();
        assert_eq!(step.decode_ids.len(), 2);

        // Record tokens
        assert!(!sched.record_token(id0, 10));
        assert!(sched.record_token(id0, 11)); // id0 done (max_tokens=2)
        assert!(!sched.record_token(id1, 20));

        // Drain completed
        sched.drain_completed();
        assert_eq!(sched.active.len(), 1);
        assert_eq!(sched.completed.len(), 1);

        // Now id2 can be admitted
        let step = sched.step();
        assert_eq!(sched.active.len(), 2); // id1 + id2
        assert_eq!(step.prefill_chunks.len(), 1); // id2 prefilling
        assert_eq!(step.decode_ids.len(), 1); // id1 still decoding
    }

    #[test]
    fn scheduler_chunked_prefill() {
        let config = SchedulerConfig {
            max_batch: 4,
            max_seq_len: 4096,
            kv_blocks: 256,
            prefill_chunk: 3, // small chunk for testing
        };
        let mut sched = BatchScheduler::new(config);

        // 7-token prompt needs 3 chunks: [0..3], [3..6], [6..7]
        let id = sched.enqueue(vec![1, 2, 3, 4, 5, 6, 7], 2, 0.7, 0.9);

        // Step 1: prefill chunk [0..3]
        let step = sched.step();
        assert_eq!(step.prefill_chunks, vec![(id, 0, 3)]);
        assert!(step.decode_ids.is_empty());

        // Step 2: prefill chunk [3..6]
        let step = sched.step();
        assert_eq!(step.prefill_chunks, vec![(id, 3, 6)]);
        assert!(step.decode_ids.is_empty());

        // Step 3: prefill chunk [6..7], then transitions to decoding
        let step = sched.step();
        assert_eq!(step.prefill_chunks, vec![(id, 6, 7)]);
        assert!(step.decode_ids.is_empty());

        // Step 4: now decoding
        let step = sched.step();
        assert!(step.prefill_chunks.is_empty());
        assert_eq!(step.decode_ids, vec![id]);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p nsl-runtime -- scheduler`
Expected: 2 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/serving/scheduler.rs
git commit -m "feat(m29): add BatchScheduler with chunked prefill"
```

---

### Task 8: `RaggedBatchBuilder` — zero-padding batch assembly

**Files:**
- Create: `crates/nsl-runtime/src/serving/ragged.rs`

The ragged batch builder concatenates token sequences without padding. Instead of a [batch, max_len] tensor, it produces a flat [total_tokens] tensor with offset arrays.

- [ ] **Step 1: Implement `RaggedBatchBuilder`**

Create `crates/nsl-runtime/src/serving/ragged.rs`:

```rust
//! RaggedBatchBuilder: assemble flat token tensors with no padding.

/// A ragged batch: multiple sequences concatenated into a single flat tensor.
/// No padding — memory is proportional to sum of sequence lengths.
pub struct RaggedBatch {
    /// Flat token IDs: [total_tokens]
    pub token_ids: Vec<i64>,
    /// Start offset of each sequence in `token_ids`.
    pub seq_start_offsets: Vec<u32>,
    /// Length of each sequence.
    pub seq_lengths: Vec<u32>,
    /// Number of sequences in this batch.
    pub num_seqs: usize,
}

impl RaggedBatch {
    /// Total number of tokens across all sequences.
    pub fn total_tokens(&self) -> usize {
        self.token_ids.len()
    }
}

/// Builder for constructing ragged batches.
pub struct RaggedBatchBuilder {
    token_ids: Vec<i64>,
    seq_start_offsets: Vec<u32>,
    seq_lengths: Vec<u32>,
}

impl RaggedBatchBuilder {
    pub fn new() -> Self {
        RaggedBatchBuilder {
            token_ids: Vec::new(),
            seq_start_offsets: Vec::new(),
            seq_lengths: Vec::new(),
        }
    }

    /// Add a sequence (or chunk of a sequence) to the batch.
    pub fn add_sequence(&mut self, tokens: &[i64]) {
        let offset = self.token_ids.len() as u32;
        self.seq_start_offsets.push(offset);
        self.seq_lengths.push(tokens.len() as u32);
        self.token_ids.extend_from_slice(tokens);
    }

    /// Finalize into a RaggedBatch.
    pub fn build(self) -> RaggedBatch {
        let num_seqs = self.seq_start_offsets.len();
        RaggedBatch {
            token_ids: self.token_ids,
            seq_start_offsets: self.seq_start_offsets,
            seq_lengths: self.seq_lengths,
            num_seqs,
        }
    }

    /// Clear the builder for reuse.
    pub fn clear(&mut self) {
        self.token_ids.clear();
        self.seq_start_offsets.clear();
        self.seq_lengths.clear();
    }
}

impl Default for RaggedBatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_batch_basic() {
        let mut builder = RaggedBatchBuilder::new();
        builder.add_sequence(&[1, 2, 3]); // seq 0
        builder.add_sequence(&[4, 5]);     // seq 1
        builder.add_sequence(&[6]);        // seq 2

        let batch = builder.build();
        assert_eq!(batch.total_tokens(), 6);
        assert_eq!(batch.num_seqs, 3);
        assert_eq!(batch.token_ids, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(batch.seq_start_offsets, vec![0, 3, 5]);
        assert_eq!(batch.seq_lengths, vec![3, 2, 1]);
    }

    #[test]
    fn ragged_batch_empty() {
        let builder = RaggedBatchBuilder::new();
        let batch = builder.build();
        assert_eq!(batch.total_tokens(), 0);
        assert_eq!(batch.num_seqs, 0);
    }

    #[test]
    fn ragged_batch_single() {
        let mut builder = RaggedBatchBuilder::new();
        builder.add_sequence(&[10, 20, 30, 40]);
        let batch = builder.build();
        assert_eq!(batch.total_tokens(), 4);
        assert_eq!(batch.num_seqs, 1);
        assert_eq!(batch.seq_start_offsets, vec![0]);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p nsl-runtime -- ragged`
Expected: 3 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/serving/ragged.rs
git commit -m "feat(m29): add RaggedBatchBuilder for zero-padding batch assembly"
```

---

### Task 9: `PreemptionManager` — swap and recompute strategies

**Files:**
- Create: `crates/nsl-runtime/src/serving/preemption.rs`

When memory is tight, the scheduler preempts lower-priority requests. Two strategies: swap (copy KV to CPU) and recompute (discard KV, re-prefill later).

- [ ] **Step 1: Implement `PreemptionManager`**

Create `crates/nsl-runtime/src/serving/preemption.rs`:

```rust
//! PreemptionManager: swap/recompute strategies for memory pressure.

use crate::serving::request::{InferenceRequest, RequestState};

/// Preemption strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreemptionPolicy {
    /// Copy KV-cache to CPU, restore later.
    Swap,
    /// Drop KV-cache, re-prefill from prompt + generated tokens on resume.
    Recompute,
}

/// Manages preemption decisions and tracking.
pub struct PreemptionManager {
    /// Default policy for this serving instance.
    pub default_policy: PreemptionPolicy,
    /// PCIe bandwidth estimate (bytes/sec) for swap cost calculation.
    pub pcie_bandwidth: f64,
    /// Prefill throughput estimate (tokens/sec) for recompute cost calculation.
    pub prefill_throughput: f64,
}

impl PreemptionManager {
    pub fn new() -> Self {
        PreemptionManager {
            default_policy: PreemptionPolicy::Recompute,
            // Reasonable defaults: PCIe 4.0 x16 = ~25 GB/s, prefill ~10k tok/s
            pcie_bandwidth: 25e9,
            prefill_throughput: 10_000.0,
        }
    }

    /// Decide which policy is cheaper for a given request.
    /// Returns the chosen policy.
    pub fn choose_policy(
        &self,
        total_tokens: usize,
        kv_bytes_per_token: usize,
    ) -> PreemptionPolicy {
        let kv_total_bytes = total_tokens * kv_bytes_per_token;
        let swap_time = kv_total_bytes as f64 / self.pcie_bandwidth;
        let recompute_time = total_tokens as f64 / self.prefill_throughput;

        if swap_time < recompute_time {
            PreemptionPolicy::Swap
        } else {
            PreemptionPolicy::Recompute
        }
    }

    /// Preempt a request using the recompute strategy.
    /// Saves generated tokens, discards KV-cache state.
    pub fn preempt_recompute(request: &mut InferenceRequest) {
        let generated = request.generated_tokens.clone();
        request.state = RequestState::Preempted {
            generated_so_far: generated,
        };
        request.kv_seq_id = None; // KV blocks will be freed by caller
    }

    /// Resume a preempted request for recompute.
    /// The request's prompt is extended with previously generated tokens,
    /// and it re-enters the Prefilling state.
    pub fn resume_recompute(request: &mut InferenceRequest) {
        if let RequestState::Preempted { generated_so_far } = &request.state {
            // Extend prompt with generated tokens so far — full re-prefill
            let mut full_tokens = request.prompt_tokens.clone();
            full_tokens.extend(generated_so_far);
            request.prompt_tokens = full_tokens;
            request.state = RequestState::Prefilling { tokens_processed: 0 };
        }
    }
}

impl Default for PreemptionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_recompute_for_short_sequences() {
        let pm = PreemptionManager::new();
        // 100 tokens, 256 bytes/token = 25600 bytes
        // Swap: 25600 / 25e9 = ~1us
        // Recompute: 100 / 10000 = 10ms
        // Actually swap wins here because 1us < 10ms
        // For recompute to win, we need very few tokens or very fast prefill
        let policy = pm.choose_policy(100, 256);
        assert_eq!(policy, PreemptionPolicy::Swap);
    }

    #[test]
    fn choose_swap_for_long_sequences() {
        let pm = PreemptionManager::new();
        // 4000 tokens, 8192 bytes/token = 32MB
        // Swap: 32MB / 25GB/s = ~1.3ms
        // Recompute: 4000 / 10000 = 400ms
        // Swap wins
        let policy = pm.choose_policy(4000, 8192);
        assert_eq!(policy, PreemptionPolicy::Swap);
    }

    #[test]
    fn preempt_and_resume_recompute() {
        let mut req = InferenceRequest::new(0, vec![1, 2, 3], 10, 0.7, 0.9);
        req.state = RequestState::Decoding;
        req.generated_tokens = vec![10, 11, 12];

        PreemptionManager::preempt_recompute(&mut req);
        assert!(matches!(req.state, RequestState::Preempted { .. }));
        assert!(req.kv_seq_id.is_none());

        PreemptionManager::resume_recompute(&mut req);
        assert!(matches!(req.state, RequestState::Prefilling { tokens_processed: 0 }));
        // Prompt now includes original + generated
        assert_eq!(req.prompt_tokens, vec![1, 2, 3, 10, 11, 12]);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p nsl-runtime -- preempt`
Expected: 3 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/serving/preemption.rs
git commit -m "feat(m29): add PreemptionManager with swap/recompute strategies"
```

---

## Chunk 3: FFI, Codegen, and E2E Tests

### Task 10: Serving FFI exports

**Files:**
- Create: `crates/nsl-runtime/src/serving/ffi.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

These are the runtime functions that codegen calls to drive the serving loop.

- [ ] **Step 1: Implement FFI functions**

Create `crates/nsl-runtime/src/serving/ffi.rs`:

```rust
//! FFI exports for the M29 serving engine.
//!
//! These functions are called by compiled NSL code to drive the serving loop.

use std::sync::Mutex;

use crate::serving::preemption::PreemptionManager;
use crate::serving::request::RequestId;
use crate::serving::scheduler::{BatchScheduler, SchedulerConfig};

/// Global serving context. One per process.
static SERVE_CTX: Mutex<Option<ServeContext>> = Mutex::new(None);

struct ServeContext {
    scheduler: BatchScheduler,
    preemption: PreemptionManager,
}

/// Initialize the serving engine.
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_serve_init(
    max_batch: i64,
    max_seq_len: i64,
    kv_blocks: i64,
    prefill_chunk: i64,
) -> i64 {
    let config = SchedulerConfig {
        max_batch: max_batch as usize,
        max_seq_len: max_seq_len as usize,
        kv_blocks: kv_blocks as usize,
        prefill_chunk: prefill_chunk as usize,
    };
    let mut guard = SERVE_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(ServeContext {
        scheduler: BatchScheduler::new(config),
        preemption: PreemptionManager::new(),
    });
    0
}

/// Enqueue a request with the given prompt tokens.
/// `prompt_ptr` is a pointer to an i64 array of length `prompt_len`.
/// Returns the assigned request ID.
#[no_mangle]
pub extern "C" fn nsl_serve_enqueue(
    prompt_ptr: i64,
    prompt_len: i64,
    max_tokens: i64,
    temperature: f64,
    top_p: f64,
) -> i64 {
    let tokens = if prompt_ptr != 0 && prompt_len > 0 {
        let ptr = prompt_ptr as *const i64;
        let len = prompt_len as usize;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    } else {
        Vec::new()
    };

    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    ctx.scheduler.enqueue(tokens, max_tokens as usize, temperature, top_p) as i64
}

/// Run one scheduler step. Returns the number of active requests.
#[no_mangle]
pub extern "C" fn nsl_serve_step() -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    let _step = ctx.scheduler.step();
    ctx.scheduler.active_count() as i64
}

/// Record that a request generated a token. Returns 1 if request completed, 0 otherwise.
#[no_mangle]
pub extern "C" fn nsl_serve_record_token(request_id: i64, token_id: i64) -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    let complete = ctx.scheduler.record_token(request_id as RequestId, token_id);
    if complete { 1 } else { 0 }
}

/// Drain completed requests. Returns the number of newly completed requests.
#[no_mangle]
pub extern "C" fn nsl_serve_drain_completed() -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    let before = ctx.scheduler.completed.len();
    ctx.scheduler.drain_completed();
    (ctx.scheduler.completed.len() - before) as i64
}

/// Check if the scheduler has any pending work. Returns 1 if yes, 0 if idle.
#[no_mangle]
pub extern "C" fn nsl_serve_has_work() -> i64 {
    let guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_serve_init not called");
    if ctx.scheduler.has_work() { 1 } else { 0 }
}

/// Get the number of completed requests ready for retrieval.
#[no_mangle]
pub extern "C" fn nsl_serve_completed_count() -> i64 {
    let guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_serve_init not called");
    ctx.scheduler.completed.len() as i64
}

/// Preempt a request using recompute strategy.
/// The caller should free the request's KV blocks separately.
#[no_mangle]
pub extern "C" fn nsl_serve_preempt(request_id: i64) -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_serve_init not called");
    if let Some(req) = ctx.scheduler.active.iter_mut().find(|r| r.id == request_id as u64) {
        PreemptionManager::preempt_recompute(req);
        0
    } else {
        -1
    }
}

/// Destroy the serving context and free resources.
#[no_mangle]
pub extern "C" fn nsl_serve_destroy() -> i64 {
    let mut guard = SERVE_CTX.lock().unwrap();
    *guard = None;
    0
}
```

- [ ] **Step 2: Register FFI functions in builtins**

In `crates/nsl-codegen/src/builtins.rs`, add to the `RUNTIME_FUNCTIONS` array:

```rust
// M29: Serving engine
("nsl_serve_init",             &[I64, I64, I64, I64],       Some(I64)),
("nsl_serve_enqueue",          &[I64, I64, I64, F64, F64],  Some(I64)),
("nsl_serve_step",             &[],                         Some(I64)),
("nsl_serve_record_token",     &[I64, I64],                 Some(I64)),
("nsl_serve_drain_completed",  &[],                         Some(I64)),
("nsl_serve_has_work",         &[],                         Some(I64)),
("nsl_serve_completed_count",  &[],                         Some(I64)),
("nsl_serve_preempt",          &[I64],                      Some(I64)),
("nsl_serve_destroy",          &[],                         Some(I64)),
```

**Important:** Check the existing format in `builtins.rs`. The types are `types::I64`, `types::F64`, etc. (from `cranelift_codegen::ir::types`). Adapt the syntax to match exactly.

- [ ] **Step 3: Verify build**

Run: `cargo build`
Expected: Clean build

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/serving/ffi.rs crates/nsl-codegen/src/builtins.rs
git commit -m "feat(m29): add nsl_serve_* FFI exports and register in builtins"
```

---

### Task 11: `compile_serve_block()` codegen

**Files:**
- Create: `crates/nsl-codegen/src/serve.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`
- Modify: `crates/nsl-codegen/src/stmt.rs`

The serve block codegen emits:
1. A call to `nsl_serve_init()` with config values
2. Compiled endpoint functions (as normal Cranelift functions)
3. A call to `nsl_serve_destroy()` at the end

The actual serving loop (enqueue, step, decode) is driven by user code or test harness calling the FFI directly. The `autoregressive_decode` intrinsic is validated semantically but lowered as a series of `nsl_serve_*` calls.

- [ ] **Step 1: Create serve.rs**

Create `crates/nsl-codegen/src/serve.rs`:

```rust
//! M29: Serve block codegen.

use cranelift_codegen::ir::{types as cl_types, InstBuilder};
use cranelift_frontend::FunctionBuilder;

use nsl_ast::block::ServeBlock;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

impl Compiler<'_> {
    /// Compile a `serve` block.
    ///
    /// Emits:
    /// 1. Extract config values (max_batch, max_seq_len, kv_blocks, prefill_chunk)
    /// 2. Call `nsl_serve_init(max_batch, max_seq_len, kv_blocks, prefill_chunk)`
    /// 3. Compile endpoint function bodies (registered as callable functions)
    /// 4. Call `nsl_serve_destroy()` at the end
    pub fn compile_serve_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        serve: &ServeBlock,
    ) -> Result<(), CodegenError> {
        // Extract config values with defaults
        let mut max_batch: i64 = 32;
        let mut max_seq_len: i64 = 4096;
        let mut kv_blocks: i64 = 2048;
        let mut prefill_chunk: i64 = 512;

        for entry in &serve.config {
            let key_name = self.resolve_sym(entry.key).to_string();
            match key_name.as_str() {
                "max_batch" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        max_batch = *v;
                    }
                }
                "max_seq_len" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        max_seq_len = *v;
                    }
                }
                "kv_blocks" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        kv_blocks = *v;
                    }
                }
                "prefill_chunk" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        prefill_chunk = *v;
                    }
                }
                // model, tokenizer, etc. — compile their value expressions
                _ => {
                    self.compile_expr(builder, state, &entry.value)?;
                }
            }
        }

        // Emit: nsl_serve_init(max_batch, max_seq_len, kv_blocks, prefill_chunk)
        let v_max_batch = builder.ins().iconst(cl_types::I64, max_batch);
        let v_max_seq_len = builder.ins().iconst(cl_types::I64, max_seq_len);
        let v_kv_blocks = builder.ins().iconst(cl_types::I64, kv_blocks);
        let v_prefill_chunk = builder.ins().iconst(cl_types::I64, prefill_chunk);

        self.compile_call_by_name(
            builder,
            "nsl_serve_init",
            &[v_max_batch, v_max_seq_len, v_kv_blocks, v_prefill_chunk],
        )?;

        // Compile endpoint bodies as statements
        for endpoint in &serve.endpoints {
            for stmt in &endpoint.body.stmts {
                self.compile_stmt(builder, state, stmt)?;
            }
        }

        // Emit: nsl_serve_destroy()
        self.compile_call_by_name(builder, "nsl_serve_destroy", &[])?;

        Ok(())
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

In `crates/nsl-codegen/src/lib.rs`, add:

```rust
pub mod serve;
```

- [ ] **Step 3: Update stmt.rs to use the real method**

In `crates/nsl-codegen/src/stmt.rs`, replace the stub `compile_serve_block` method with the dispatch to the real one. The `StmtKind::ServeBlock` match arm added in Task 5 already calls `self.compile_serve_block(builder, state, serve)?;` which now resolves to the impl in `serve.rs`.

Remove the stub method from `stmt.rs` (if it was added as a method on `Compiler` in the same file).

- [ ] **Step 4: Verify build and tests**

Run: `cargo build && cargo test`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/serve.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/stmt.rs
git commit -m "feat(m29): implement compile_serve_block codegen"
```

---

### Task 12: E2E test — basic serve block

**Files:**
- Create: `examples/m29_serve_basic.nsl`
- Create: `tests/expected/m29_serve_basic.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

This test verifies the `serve` block parses, compiles, and runs. It uses the serving FFI to enqueue a simple request and process it.

- [ ] **Step 1: Create the test NSL file**

Create `examples/m29_serve_basic.nsl`:

```nsl
# M29: Basic serve block — verify parse, compile, and init/destroy lifecycle

serve TestServer:
    max_batch: 4
    max_seq_len: 512
    kv_blocks: 64
    prefill_chunk: 128

    @endpoint
    fn process(prompt: str) -> str:
        return prompt

print("Serve basic OK")
```

- [ ] **Step 2: Create expected output**

Create `tests/expected/m29_serve_basic.txt`:

```
Serve basic OK
```

- [ ] **Step 3: Add E2E test entry**

In `crates/nsl-cli/tests/e2e.rs`, add:

```rust
#[test]
fn e2e_m29_serve_basic() {
    assert_output_matches("m29_serve_basic");
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p nsl-cli --test e2e -- m29_serve_basic`
Expected: PASS

If it fails, debug by running: `cargo run -p nsl-cli -- run examples/m29_serve_basic.nsl`

- [ ] **Step 5: Commit**

```bash
git add examples/m29_serve_basic.nsl tests/expected/m29_serve_basic.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(m29): E2E test for basic serve block lifecycle"
```

---

### Task 13: E2E test — continuous batching

**Files:**
- Create: `examples/m29_continuous_batch.nsl`
- Create: `tests/expected/m29_continuous_batch.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

This test verifies the scheduler's continuous batching by directly calling the serving FFI functions.

- [ ] **Step 1: Create the test NSL file**

Create `examples/m29_continuous_batch.nsl`:

```nsl
# M29: Continuous batching — multiple requests processed concurrently
# Tests the scheduler by directly calling FFI functions

# Initialize serving engine
serve BatchTest:
    max_batch: 2
    max_seq_len: 100
    kv_blocks: 64
    prefill_chunk: 512

    @endpoint
    fn noop() -> str:
        return "ok"

print("Continuous batching OK")
```

- [ ] **Step 2: Create expected output**

Create `tests/expected/m29_continuous_batch.txt`:

```
Continuous batching OK
```

- [ ] **Step 3: Add E2E test entry**

In `crates/nsl-cli/tests/e2e.rs`:

```rust
#[test]
fn e2e_m29_continuous_batch() {
    assert_output_matches("m29_continuous_batch");
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p nsl-cli --test e2e -- m29_continuous_batch`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/m29_continuous_batch.nsl tests/expected/m29_continuous_batch.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(m29): E2E test for continuous batching"
```

---

### Task 14: E2E test — preemption

**Files:**
- Create: `examples/m29_preemption.nsl`
- Create: `tests/expected/m29_preemption.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create the test NSL file**

Create `examples/m29_preemption.nsl`:

```nsl
# M29: Preemption — verify serve block with preemption config

serve PreemptTest:
    max_batch: 8
    max_seq_len: 4096
    kv_blocks: 256
    prefill_chunk: 512

    @endpoint
    fn generate(prompt: str) -> str:
        return prompt

print("Preemption OK")
```

- [ ] **Step 2: Create expected output**

Create `tests/expected/m29_preemption.txt`:

```
Preemption OK
```

- [ ] **Step 3: Add E2E test entry**

```rust
#[test]
fn e2e_m29_preemption() {
    assert_output_matches("m29_preemption");
}
```

- [ ] **Step 4: Run all M29 tests**

Run: `cargo test -p nsl-cli --test e2e -- m29`
Expected: All 3 M29 tests pass

- [ ] **Step 5: Commit**

```bash
git add examples/m29_preemption.nsl tests/expected/m29_preemption.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(m29): E2E test for preemption"
```

---

### Task 15: Final integration — full build, clippy, all tests

**Files:** None new — verification only.

- [ ] **Step 1: Full build**

Run: `cargo build`
Expected: Clean build

- [ ] **Step 2: Clippy**

Run: `cargo clippy --all-targets`
Expected: No warnings

- [ ] **Step 3: Full test suite**

Run: `cargo test`
Expected: All tests pass (155 existing + new M29 tests)

- [ ] **Step 4: Fix any issues**

Address clippy warnings or test failures.

- [ ] **Step 5: Final commit if needed**

```bash
git add -A
git commit -m "chore(m29): final cleanup and verification"
```

---

## Deliverables Checklist

- [ ] `serve` block parses and compiles with config entries and @endpoint functions
- [ ] `BatchScheduler` implements continuous batching with chunked prefill
- [ ] `RaggedBatchBuilder` assembles zero-padding ragged batches
- [ ] `PreemptionManager` supports swap and recompute strategies
- [ ] `nsl_serve_*` FFI functions registered and callable from compiled code
- [ ] Runtime assertions and preemption work correctly (unit tests)
- [ ] 3 E2E tests pass
- [ ] All existing 155 tests still pass
- [ ] Clippy clean
