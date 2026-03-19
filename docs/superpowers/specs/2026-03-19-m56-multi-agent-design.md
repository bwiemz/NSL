# M56: Natively Compiled Multi-Agent Shared Memory — Design Specification

**Date:** 2026-03-19
**Status:** Planned
**Milestone:** M56 (Phase 11, v1.0.0)
**Prerequisites:** M38 (Linear Types — ownership proof for cross-agent memory safety), M51 (Effect System — Communication-only effect isolation across agent boundaries), M29 (Continuous Batching/Serve — KV-cache management), M25 (PagedAttention — paged KV-cache)
**Dependencies:** M38 provides compile-time ownership proofs ensuring no shared mutable state between agents; M51 proves agents are Communication-only across boundaries (no Mutation cross-boundary); M25/M29 provide the KV-cache infrastructure that agents share

## Overview

M56 compiles multiple AI agents into a single binary where they share KV-cache, embedding tables, and activations via zero-copy RAM transfers instead of JSON serialization over HTTP. This is the "LangChain killer" feature: instead of agents communicating through a Python orchestrator that serializes tensors to JSON, copies them across process boundaries, and deserializes on the other side, NSL agents share the same address space with compile-time guarantees that Agent A cannot corrupt Agent B's state.

The key design insight is that linear types (M38) and the effect system (M51) together provide the exact safety guarantees needed for multi-agent memory sharing. Linear types prove that when Agent A sends its KV-cache to Agent B, ownership transfers — Agent A cannot access it afterward. The effect system proves that agent functions are `Communication`-only across boundaries — no `Mutation` effect leaks between agents. These are compile-time proofs, not runtime checks, making NSL the only language that can guarantee multi-agent memory safety without garbage collection pauses or serialization overhead.

The `agent` keyword declares an agent (similar to `model` but with isolated state and a communication interface). Agents communicate via `agent.send()` and `agent.recv()` primitives that transfer tensor ownership without copying data. The `@shared` annotation (from M38) allows read-only shared embedding tables across all agents. The `@pipeline_agent` decorator composes multiple agents into a declarative multi-agent pipeline with compile-time–verified ownership flow.

**Why this is impossible in Python:** LangChain/AutoGPT agents communicate via JSON strings over HTTP, requiring full serialization/deserialization of every tensor. Even with shared memory (multiprocessing.shared_memory), Python cannot prove that concurrent agents won't corrupt each other's data — the GIL prevents true parallelism, and there's no ownership system to prevent data races. NSL's linear types + effect system provide compile-time safety with zero-copy performance.

---

## Section 1: Language Surface

### 1.1 `agent` Keyword

```
agent Drafter:
    model: SmallLLM
    kv_cache: KVCache<[MaxSeq, NumHeads, HeadDim], f16>
    temperature: float = 0.8

    fn init(self, model_path: str):
        self.model = load_model(model_path)
        self.kv_cache = KVCache.empty(max_seq=2048, num_heads=32, head_dim=128)

    fn draft(self, prompt: Tensor<[1, S], int32>, num_tokens: int) -> DraftResult:
        let tokens = self.model.generate(prompt, max_new=num_tokens, kv=self.kv_cache)
        return DraftResult {
            tokens: tokens,
            kv_snapshot: self.kv_cache.snapshot(),  # ownership of snapshot transfers out
        }
```

**`agent` semantics:**
- Like `model`, but with **isolated mutable state** (KV-cache, generation buffers)
- Each agent has its own memory region — the compiler enforces that no other agent can write to it
- Agent methods can mutate internal state (self.kv_cache) but cannot mutate other agents' state
- Communication between agents is explicit via `send()`/`recv()` primitives

### 1.2 Agent State Isolation

```
agent Reviewer:
    model: LargeLLM
    kv_cache: KVCache<[MaxSeq, NumHeads, HeadDim], f16>

    fn review(self, draft: DraftResult) -> ReviewResult:
        # draft.kv_snapshot is consumed here — ownership transferred from Drafter
        self.kv_cache.merge(draft.kv_snapshot)    # zero-copy merge

        let score = self.model.score(draft.tokens, kv=self.kv_cache)
        return ReviewResult { score: score, accepted: score > 0.9 }
```

The compiler verifies:
1. `draft.kv_snapshot` is consumed by `Reviewer.review()` — the Drafter can no longer access it
2. `self.kv_cache` is modified only by the Reviewer — no other agent can mutate it
3. The `merge()` operation is zero-copy: the snapshot's pages are remapped into the Reviewer's KV-cache

### 1.3 `agent.send()` and `agent.recv()`

```
# Explicit communication primitives
fn run_pipeline():
    let drafter = Drafter.init("small_model.safetensors")
    let reviewer = Reviewer.init("large_model.safetensors")

    let prompt = tokenize("Write a poem about ML compilers")

    # Drafter produces a draft
    let draft = drafter.draft(prompt, num_tokens=50)

    # Send draft to reviewer — ownership of draft.kv_snapshot transfers
    reviewer.send(draft)      # zero-copy: pointer transfer, no serialization

    # Reviewer processes the draft
    let review = reviewer.review()

    if review.accepted:
        print("Draft accepted!")
    else:
        # Re-draft with feedback
        let revised = drafter.draft(prompt, num_tokens=50)
```

**`send()` semantics:**
- Transfers ownership of the sent value (linear move — M38)
- Zero-copy: only the pointer (and metadata) is transferred, not the tensor data
- The sender can no longer access the sent value after `send()`
- If the value is `@shared`, a reference is sent (refcount bump) — the sender retains access

### 1.4 `@shared` Embedding Tables

```
# Shared read-only embeddings — all agents can read, none can write
@shared
let embeddings: Tensor<[VocabSize, EmbedDim], f16> = load("embeddings.safetensors")

agent Drafter:
    model: SmallLLM

    fn draft(self, prompt: Tensor<[1, S], int32>) -> DraftResult:
        # Read from shared embeddings — immutable borrow, no ownership transfer
        let h = embed(&embeddings, prompt)
        ...

agent Reviewer:
    model: LargeLLM

    fn review(self, draft: DraftResult) -> ReviewResult:
        # Same shared embeddings — both agents read, neither writes
        let h = embed(&embeddings, draft.tokens)
        ...
```

**`@shared` in multi-agent context:**
- Embedding tables marked `@shared` are readable by all agents (immutable borrow — M38)
- No agent can mutate a `@shared` tensor — enforced by the ownership checker
- No refcount overhead for reads — the shared tensor is pinned in memory for the pipeline's lifetime
- This is the primary use case for shared vocabulary/position embeddings across agent ensemble

### 1.5 `@pipeline_agent` Decorator

```
@pipeline_agent(agents=[Drafter, Reviewer, Editor])
fn speculative_review_pipeline(
    prompt: Tensor<[1, S], int32>,
    @shared embeddings: Tensor<[VocabSize, EmbedDim], f16>,
) -> Tensor<[1, OutS], int32>:
    # Stage 1: Drafter generates candidate tokens
    let draft = Drafter.draft(prompt, num_tokens=64)

    # Stage 2: Reviewer scores the draft (receives KV-cache ownership)
    let review = Reviewer.review(draft)

    # Stage 3: Editor refines accepted drafts
    if review.accepted:
        return Editor.edit(review)
    else:
        # Retry with new draft
        let new_draft = Drafter.draft(prompt, num_tokens=32)
        return Editor.edit(Reviewer.review(new_draft))
```

**`@pipeline_agent` guarantees (compile-time):**
1. Every tensor passed between agents is either moved (linear) or borrowed (`@shared`)
2. No mutable state is shared across agent boundaries
3. The ownership flow forms a DAG — no circular ownership transfers
4. All `@shared` tensors are immutable for the pipeline's duration
5. The pipeline function's effect is `Communication` only (no cross-boundary `Mutation`)

### 1.6 Agent Communication Types

```
# Structured communication types — ownership annotations on fields
struct DraftResult:
    tokens: Tensor<[1, N], int32>          # linear — ownership transfers with struct
    kv_snapshot: KVCacheSnapshot            # linear — ownership transfers
    metadata: dict<str, float>             # shared — small, copied by value

struct ReviewResult:
    score: float                           # scalar — copied
    accepted: bool                         # scalar — copied
    attention_map: Tensor<[N, N], f16>     # linear — ownership transfers

struct KVCacheSnapshot:
    keys: Tensor<[NumLayers, NumHeads, S, HeadDim], f16>    # linear
    values: Tensor<[NumLayers, NumHeads, S, HeadDim], f16>  # linear
    seq_len: int                                             # scalar — copied
```

**Field ownership rules:**
- Tensor fields in communication structs are linear by default — they move with the struct
- Scalar fields (int, float, bool) are Copy — no ownership concerns
- `@shared` fields are reference-counted — sending the struct bumps the refcount
- The struct itself is linear — sending it to another agent consumes it

### 1.7 Agent Lifecycle

```
agent Drafter:
    fn init(self, model_path: str):
        # Called once when the agent is created
        self.model = load_model(model_path)

    fn reset(self):
        # Called to reset agent state between pipeline invocations
        self.kv_cache.clear()

    fn shutdown(self):
        # Called when the agent is destroyed — resources freed
        # Compiler verifies all owned tensors are either consumed or freed here
        pass

# Agent lifecycle in a serve context
serve my_api on port=8080:
    # Agents are created once at server startup
    let drafter = Drafter.init("small.safetensors")
    let reviewer = Reviewer.init("large.safetensors")

    route "/generate":
        # Each request gets fresh agent state
        drafter.reset()
        reviewer.reset()

        let result = speculative_review_pipeline(request.tokens, embeddings)
        return result
```

---

## Section 2: Architecture

### 2.1 Agent Compilation Model

```
Source → Lexer → Parser → Type Checker → Shape Checker
                                              ↓
                                    Ownership Checker (M38)
                                              ↓
                                    Effect Checker (M51)
                                              ↓
                                    AgentChecker (NEW)
                                              ↓
                                    Agent Codegen (NEW)
                                              ↓
                                    Cranelift IR → Binary
```

Agents compile to normal Cranelift functions with an additional `AgentContext` pointer parameter. The agent's state is a heap-allocated struct. Agent methods are compiled as functions that receive `&mut AgentContext` as their first argument (similar to `self` in model methods, but with stricter ownership rules enforced by the `AgentChecker`).

### 2.2 Core Data Structures

New module: `crates/nsl-semantic/src/agent.rs`

```rust
/// Semantic representation of an agent declaration.
#[derive(Debug, Clone)]
pub struct AgentDef {
    /// Agent name (e.g., "Drafter", "Reviewer").
    pub name: Symbol,

    /// Agent state fields (KV-cache, model, buffers).
    pub fields: Vec<AgentField>,

    /// Agent methods (init, draft, review, reset, shutdown).
    pub methods: Vec<FnDef>,

    /// Source span for error reporting.
    pub span: Span,
}

/// A field in an agent's state.
#[derive(Debug, Clone)]
pub struct AgentField {
    pub name: Symbol,
    pub ty: Type,
    pub ownership: AgentFieldOwnership,
    pub default_value: Option<Expr>,
    pub span: Span,
}

/// Ownership classification for agent fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentFieldOwnership {
    /// Exclusively owned by this agent. No other agent can access.
    Exclusive,
    /// Shared read-only across all agents (@shared annotation).
    SharedReadOnly,
    /// Scalar/Copy type — no ownership concerns.
    Copy,
}

/// A communication channel between two agents.
#[derive(Debug, Clone)]
pub struct AgentChannel {
    /// Source agent.
    pub sender: Symbol,
    /// Destination agent.
    pub receiver: Symbol,
    /// Type of data transferred.
    pub payload_type: Type,
    /// Ownership transfer mode.
    pub transfer_mode: TransferMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Linear move — sender loses access.
    Move,
    /// Shared borrow — sender retains access (for @shared data).
    SharedBorrow,
}
```

### 2.3 AgentChecker Pass

New semantic analysis pass: `crates/nsl-semantic/src/agent.rs`

```rust
/// Verifies cross-agent ownership rules and communication safety.
pub struct AgentChecker<'a> {
    interner: &'a Interner,
    diagnostics: &'a mut Vec<Diagnostic>,

    /// All declared agents.
    agents: HashMap<Symbol, AgentDef>,

    /// Communication channels (sender -> receiver -> payload type).
    channels: Vec<AgentChannel>,

    /// Ownership state per agent per field.
    field_states: HashMap<(Symbol, Symbol), FieldState>,

    /// @shared tensors visible to all agents.
    shared_globals: HashSet<Symbol>,

    /// The agent currently being analyzed.
    current_agent: Option<Symbol>,

    /// Pipeline definitions (from @pipeline_agent).
    pipelines: Vec<PipelineDef>,
}

#[derive(Debug, Clone)]
enum FieldState {
    /// Field is live and owned by the agent.
    Owned,
    /// Field was sent to another agent — no longer accessible.
    Transferred { to: Symbol, at: Span },
    /// Field is @shared — always accessible.
    SharedReadOnly,
}

/// A declared multi-agent pipeline.
#[derive(Debug, Clone)]
pub struct PipelineDef {
    pub name: Symbol,
    pub agents: Vec<Symbol>,
    pub shared_params: Vec<Symbol>,
    pub body: Block,
    pub span: Span,
}

impl<'a> AgentChecker<'a> {
    pub fn new(
        interner: &'a Interner,
        diagnostics: &'a mut Vec<Diagnostic>,
    ) -> Self {
        Self {
            interner,
            diagnostics,
            agents: HashMap::new(),
            channels: Vec::new(),
            field_states: HashMap::new(),
            shared_globals: HashSet::new(),
            current_agent: None,
            pipelines: Vec::new(),
        }
    }

    /// Main entry point: check all agent declarations and pipelines.
    pub fn check_program(&mut self, program: &TypedProgram) {
        // Phase 1: Register all agents and their fields
        self.register_agents(program);

        // Phase 2: Check agent method bodies
        for (agent_sym, agent_def) in &self.agents.clone() {
            self.current_agent = Some(*agent_sym);
            for method in &agent_def.methods {
                self.check_agent_method(*agent_sym, method);
            }
        }

        // Phase 3: Check pipeline definitions
        for pipeline in &self.pipelines.clone() {
            self.check_pipeline(pipeline);
        }

        // Phase 4: Verify communication graph is acyclic
        self.check_acyclic_communication();
    }

    /// Verify that an agent method does not access other agents' exclusive state.
    fn check_agent_method(&mut self, agent: Symbol, method: &FnDef) {
        self.walk_stmts(&method.body.stmts, agent);
    }

    fn walk_stmts(&mut self, stmts: &[Stmt], current_agent: Symbol) {
        for stmt in stmts {
            match &stmt.kind {
                // Check for cross-agent field access
                StmtKind::Expr(expr) => self.check_expr(expr, current_agent),
                StmtKind::Let { value, .. } => self.check_expr(value, current_agent),
                StmtKind::Return(Some(expr)) => self.check_expr(expr, current_agent),
                StmtKind::If { then_branch, else_branch, .. } => {
                    self.walk_stmts(&then_branch.stmts, current_agent);
                    if let Some(else_b) = else_branch {
                        self.walk_stmts(&else_b.stmts, current_agent);
                    }
                }
                _ => {}
            }
        }
    }

    fn check_expr(&mut self, expr: &Expr, current_agent: Symbol) {
        match &expr.kind {
            // Check field access: agent.field
            ExprKind::FieldAccess { object, field } => {
                if let Some(target_agent) = self.resolve_agent(object) {
                    if target_agent != current_agent {
                        let field_ownership = self.get_field_ownership(target_agent, *field);
                        match field_ownership {
                            AgentFieldOwnership::Exclusive => {
                                self.diagnostics.push(Diagnostic::error(
                                    format!(
                                        "agent '{}' cannot access exclusive field '{}' of agent '{}'",
                                        self.interner.resolve(current_agent),
                                        self.interner.resolve(*field),
                                        self.interner.resolve(target_agent),
                                    ),
                                    expr.span,
                                ));
                            }
                            AgentFieldOwnership::SharedReadOnly => {
                                // OK — @shared fields are readable by all agents
                            }
                            AgentFieldOwnership::Copy => {
                                // OK — Copy types have no ownership concerns
                            }
                        }
                    }
                }
            }

            // Check send(): ownership transfer
            ExprKind::MethodCall { object, method, args }
                if self.interner.resolve(*method) == "send" =>
            {
                self.check_send(object, args, current_agent, expr.span);
            }

            _ => {}
        }
    }

    /// Validate a send() call: ownership must transfer correctly.
    fn check_send(
        &mut self,
        receiver_expr: &Expr,
        args: &[Expr],
        sender_agent: Symbol,
        span: Span,
    ) {
        let receiver_agent = match self.resolve_agent(receiver_expr) {
            Some(a) => a,
            None => {
                self.diagnostics.push(Diagnostic::error(
                    "send() target must be an agent".to_string(),
                    span,
                ));
                return;
            }
        };

        if sender_agent == receiver_agent {
            self.diagnostics.push(Diagnostic::error(
                "agent cannot send() to itself".to_string(),
                span,
            ));
            return;
        }

        // Record the communication channel
        for arg in args {
            let payload_type = self.infer_type(arg);
            let transfer_mode = if self.is_shared_type(&payload_type) {
                TransferMode::SharedBorrow
            } else {
                TransferMode::Move
            };

            self.channels.push(AgentChannel {
                sender: sender_agent,
                receiver: receiver_agent,
                payload_type,
                transfer_mode,
            });
        }
    }

    /// Verify communication graph has no cycles (prevents deadlock).
    fn check_acyclic_communication(&mut self) {
        // Build adjacency list from channels
        let mut graph: HashMap<Symbol, HashSet<Symbol>> = HashMap::new();
        for channel in &self.channels {
            graph.entry(channel.sender)
                .or_default()
                .insert(channel.receiver);
        }

        // DFS cycle detection
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        for &agent in self.agents.keys() {
            if self.has_cycle(&graph, agent, &mut visited, &mut in_stack) {
                self.diagnostics.push(Diagnostic::error(
                    "circular ownership transfer detected between agents — \
                     this would cause deadlock or use-after-move".to_string(),
                    Span::default(),
                ));
                break;
            }
        }
    }

    fn has_cycle(
        &self,
        graph: &HashMap<Symbol, HashSet<Symbol>>,
        node: Symbol,
        visited: &mut HashSet<Symbol>,
        in_stack: &mut HashSet<Symbol>,
    ) -> bool {
        if in_stack.contains(&node) { return true; }
        if visited.contains(&node) { return false; }

        visited.insert(node);
        in_stack.insert(node);

        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if self.has_cycle(graph, neighbor, visited, in_stack) {
                    return true;
                }
            }
        }

        in_stack.remove(&node);
        false
    }

    /// Check @pipeline_agent ownership flow.
    fn check_pipeline(&mut self, pipeline: &PipelineDef) {
        // Verify all agents in the pipeline are declared
        for &agent_sym in &pipeline.agents {
            if !self.agents.contains_key(&agent_sym) {
                self.diagnostics.push(Diagnostic::error(
                    format!(
                        "agent '{}' in @pipeline_agent not declared",
                        self.interner.resolve(agent_sym),
                    ),
                    pipeline.span,
                ));
            }
        }

        // Verify shared params are @shared
        for &param_sym in &pipeline.shared_params {
            if !self.shared_globals.contains(&param_sym) {
                self.diagnostics.push(Diagnostic::error(
                    format!(
                        "parameter '{}' passed to @pipeline_agent must be @shared",
                        self.interner.resolve(param_sym),
                    ),
                    pipeline.span,
                ));
            }
        }

        // Walk pipeline body and verify ownership flow
        self.check_pipeline_body(&pipeline.body, &pipeline.agents);
    }

    fn check_pipeline_body(&mut self, body: &Block, agents: &[Symbol]) {
        // Track which agent owns which tensor at each point in the pipeline.
        // When a tensor is passed from agent A to agent B, A loses ownership.
        let mut tensor_owners: HashMap<Symbol, Symbol> = HashMap::new();

        for stmt in &body.stmts {
            self.check_pipeline_stmt(stmt, agents, &mut tensor_owners);
        }
    }

    fn check_pipeline_stmt(
        &mut self,
        stmt: &Stmt,
        agents: &[Symbol],
        tensor_owners: &mut HashMap<Symbol, Symbol>,
    ) {
        // Track tensor ownership through the pipeline
        match &stmt.kind {
            StmtKind::Let { name, value, .. } => {
                // If value is an agent method call, record the output tensor's owner
                if let Some(agent) = self.extract_agent_from_call(value) {
                    tensor_owners.insert(*name, agent);
                }
            }
            _ => {}
        }
    }

    fn resolve_agent(&self, expr: &Expr) -> Option<Symbol> {
        match &expr.kind {
            ExprKind::Ident(sym) if self.agents.contains_key(sym) => Some(*sym),
            _ => None,
        }
    }

    fn get_field_ownership(&self, agent: Symbol, field: Symbol) -> AgentFieldOwnership {
        self.agents.get(&agent)
            .and_then(|def| def.fields.iter().find(|f| f.name == field))
            .map(|f| f.ownership)
            .unwrap_or(AgentFieldOwnership::Exclusive)
    }

    fn is_shared_type(&self, _ty: &Type) -> bool {
        // Check if the type has @shared annotation
        false // conservative default
    }

    fn infer_type(&self, _expr: &Expr) -> Type {
        Type::Unknown // placeholder — real implementation delegates to type checker
    }

    fn extract_agent_from_call(&self, _expr: &Expr) -> Option<Symbol> {
        None // placeholder
    }

    fn register_agents(&mut self, _program: &TypedProgram) {
        // Walk the program and register all agent declarations
    }
}
```

### 2.4 Effect System Integration (M51)

The `AgentChecker` composes with M51's `EffectChecker` to verify effect isolation:

```rust
/// Verify that cross-agent communication only uses Communication effect.
pub fn check_agent_effects(
    agent_checker: &AgentChecker,
    effect_checker: &EffectChecker,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for channel in &agent_checker.channels {
        // The send/recv operations should only have Communication effect
        // No Mutation effect should cross agent boundaries

        // Check that the sending agent's method effects at the send point
        // do not include Mutation on any of the receiver's state
        let sender_method_effects = effect_checker.get_effects_at_call(
            channel.sender,
            "send",
        );

        if sender_method_effects.contains(EffectSet::MUTATION) {
            // Need to verify the mutation is on sender's own state, not receiver's
            let mutation_targets = effect_checker.get_mutation_targets(
                channel.sender,
                "send",
            );

            for target in mutation_targets {
                if agent_checker.is_field_of(target, channel.receiver) {
                    diagnostics.push(Diagnostic::error(
                        format!(
                            "agent '{}' mutates field '{}' of agent '{}' — \
                             cross-agent mutation is not allowed",
                            agent_checker.interner.resolve(channel.sender),
                            agent_checker.interner.resolve(target),
                            agent_checker.interner.resolve(channel.receiver),
                        ),
                        Span::default(),
                    ));
                }
            }
        }
    }
}
```

### 2.5 Component Map

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `AgentChecker` | `crates/nsl-semantic/src/agent.rs` | Cross-agent ownership validation, communication graph analysis |
| `AgentParser` | `crates/nsl-parser/src/agent.rs` | Parse `agent` keyword, agent fields, agent methods |
| `AgentDef` AST | `crates/nsl-ast/src/agent.rs` | AST nodes for agent declarations |
| `AgentCodegen` | `crates/nsl-codegen/src/agent_codegen.rs` | Compile agents to Cranelift IR, AgentContext struct layout |
| `AgentRuntime` | `crates/nsl-runtime/src/agent.rs` | Runtime support: agent context allocation, send/recv primitives |
| Effect integration | `crates/nsl-semantic/src/agent.rs` | Compose with M51 EffectChecker for cross-boundary effect validation |
| Pipeline checker | `crates/nsl-semantic/src/agent.rs` | @pipeline_agent ownership flow verification |
| CLI integration | `crates/nsl-cli/src/main.rs` | No new flags — agents are a language-level feature |

---

## Section 3: Runtime Support

### 3.1 AgentContext

```rust
/// Runtime representation of an agent's isolated state.
/// Allocated on the heap. Each agent gets its own context.
#[repr(C)]
pub struct AgentContext {
    /// Unique agent ID (for debugging and send/recv routing).
    pub agent_id: u64,

    /// Agent name (interned string index).
    pub name_idx: u64,

    /// Pointer to the agent's state struct (fields laid out by codegen).
    pub state_ptr: *mut u8,

    /// Size of the state struct in bytes.
    pub state_size: u64,

    /// Mailbox for incoming messages (ring buffer of TensorTransfer).
    pub mailbox: *mut AgentMailbox,

    /// Whether the agent is currently active (processing a request).
    pub active: bool,
}

/// A pending tensor transfer between agents.
#[repr(C)]
pub struct TensorTransfer {
    /// Source agent ID.
    pub from_agent: u64,

    /// The tensor being transferred (pointer moves, data stays in place).
    pub tensor: NslTensor,

    /// Transfer mode: 0 = move (source loses access), 1 = shared borrow (refcount bump).
    pub mode: u8,

    /// Metadata for debugging.
    pub transfer_id: u64,
}

/// Lock-free single-producer single-consumer ring buffer for agent communication.
#[repr(C)]
pub struct AgentMailbox {
    /// Ring buffer of transfers.
    pub slots: *mut TensorTransfer,

    /// Number of slots (power of 2).
    pub capacity: u64,

    /// Write index (only written by sender).
    pub write_idx: AtomicU64,

    /// Read index (only written by receiver).
    pub read_idx: AtomicU64,
}
```

### 3.2 Runtime FFI Functions

```rust
/// Allocate and initialize an agent context.
#[no_mangle]
pub extern "C" fn nsl_agent_create(
    agent_id: u64,
    name_idx: u64,
    state_size: u64,
    mailbox_capacity: u64,
) -> *mut AgentContext {
    let state_ptr = unsafe { alloc::alloc_zeroed(Layout::from_size_align(state_size as usize, 64).unwrap()) };
    let mailbox = AgentMailbox::new(mailbox_capacity);

    let ctx = Box::new(AgentContext {
        agent_id,
        name_idx,
        state_ptr,
        state_size,
        mailbox: Box::into_raw(Box::new(mailbox)),
        active: false,
    });

    Box::into_raw(ctx)
}

/// Destroy an agent context and free all owned memory.
#[no_mangle]
pub extern "C" fn nsl_agent_destroy(ctx: *mut AgentContext) {
    unsafe {
        let ctx = Box::from_raw(ctx);
        alloc::dealloc(ctx.state_ptr, Layout::from_size_align(ctx.state_size as usize, 64).unwrap());
        let _ = Box::from_raw(ctx.mailbox);
    }
}

/// Send a tensor from one agent to another (zero-copy move).
/// After this call, the sender's reference to the tensor is invalidated.
#[no_mangle]
pub extern "C" fn nsl_agent_send(
    sender_ctx: *mut AgentContext,
    receiver_ctx: *mut AgentContext,
    tensor: NslTensor,
    mode: u8,
) -> i32 {
    let receiver = unsafe { &mut *receiver_ctx };
    let mailbox = unsafe { &mut *receiver.mailbox };

    let transfer = TensorTransfer {
        from_agent: unsafe { (*sender_ctx).agent_id },
        tensor,
        mode,
        transfer_id: 0, // assigned by mailbox
    };

    // Write to the receiver's mailbox (lock-free)
    if mailbox.try_push(transfer) {
        0 // success
    } else {
        -1 // mailbox full — caller should retry or increase capacity
    }
}

/// Receive a tensor from another agent (blocking until available or timeout).
#[no_mangle]
pub extern "C" fn nsl_agent_recv(
    ctx: *mut AgentContext,
    timeout_ms: u64,
) -> TensorTransfer {
    let agent = unsafe { &mut *ctx };
    let mailbox = unsafe { &mut *agent.mailbox };

    // Spin-wait with backoff until a transfer is available
    let start = std::time::Instant::now();
    loop {
        if let Some(transfer) = mailbox.try_pop() {
            return transfer;
        }

        if start.elapsed().as_millis() as u64 > timeout_ms {
            // Timeout — return empty transfer
            return TensorTransfer {
                from_agent: 0,
                tensor: NslTensor::null(),
                mode: 0,
                transfer_id: 0,
            };
        }

        std::hint::spin_loop();
    }
}

/// Reset an agent's state for a new request (clears KV-cache, resets buffers).
#[no_mangle]
pub extern "C" fn nsl_agent_reset(ctx: *mut AgentContext) {
    let agent = unsafe { &mut *ctx };
    // Zero the state struct (fields will be re-initialized by the agent's reset() method)
    unsafe {
        std::ptr::write_bytes(agent.state_ptr, 0, agent.state_size as usize);
    }
    // Drain the mailbox
    let mailbox = unsafe { &mut *agent.mailbox };
    while mailbox.try_pop().is_some() {}
}

/// Get a pointer to the agent's state struct (for field access in codegen).
#[no_mangle]
pub extern "C" fn nsl_agent_get_state(ctx: *mut AgentContext) -> *mut u8 {
    unsafe { (*ctx).state_ptr }
}
```

### 3.3 Zero-Copy KV-Cache Transfer

The critical performance feature: transferring KV-cache between agents without copying tensor data.

```rust
/// Transfer a KV-cache snapshot from one agent to another.
/// This remaps page table entries (if using PagedAttention M25) or
/// moves the raw pointer (if using contiguous KV-cache).
#[no_mangle]
pub extern "C" fn nsl_agent_transfer_kv_cache(
    src_ctx: *mut AgentContext,
    dst_ctx: *mut AgentContext,
    kv_tensor: NslTensor,
) -> NslTensor {
    // The tensor data stays in place — only the ownership metadata moves.
    // The source agent's pointer to this tensor is invalidated by the compiler
    // (enforced at compile-time by linear types, not checked at runtime).

    // For PagedAttention (M25): transfer page table entries
    // For contiguous KV-cache: just move the pointer

    // Return the same tensor with updated metadata
    NslTensor {
        data: kv_tensor.data,
        shape: kv_tensor.shape,
        ndim: kv_tensor.ndim,
        strides: kv_tensor.strides,
        dtype: kv_tensor.dtype,
        device: kv_tensor.device,
        refcount: kv_tensor.refcount,
        owns_data: kv_tensor.owns_data,
    }
}
```

### 3.4 Shared Embedding Table Registration

```rust
/// Register a tensor as shared across all agents (read-only).
/// The tensor is pinned in memory and its refcount is set to a sentinel value
/// indicating "shared-global" (never freed until pipeline shutdown).
#[no_mangle]
pub extern "C" fn nsl_agent_register_shared(
    tensor: *mut NslTensor,
) {
    unsafe {
        // Set refcount to max value — this tensor is never freed by individual agents
        (*tensor).refcount = i64::MAX;
        // Ensure owns_data is 0 — agents don't own the shared data
        // (the pipeline owns it)
        // Note: the tensor still points to valid data, agents just can't free it
    }
}
```

---

## Section 4: Codegen Changes

### 4.1 Agent Struct Layout

```rust
/// Codegen for agent declarations. Compiles agent fields into a C struct layout
/// and agent methods into Cranelift functions.
pub struct AgentCodegen<'a> {
    compiler: &'a mut Compiler,

    /// Layout of each agent's state struct (field offsets).
    agent_layouts: HashMap<Symbol, AgentLayout>,
}

/// Computed memory layout for an agent's state.
#[derive(Debug, Clone)]
pub struct AgentLayout {
    pub agent_name: Symbol,
    /// Field name -> byte offset in the state struct.
    pub field_offsets: HashMap<Symbol, usize>,
    /// Total size of the state struct in bytes.
    pub total_size: usize,
    /// Alignment requirement.
    pub alignment: usize,
}

impl<'a> AgentCodegen<'a> {
    /// Compile an agent declaration.
    pub fn compile_agent(&mut self, agent: &AgentDef) -> Result<(), CodegenError> {
        // Step 1: Compute field layout (same as model field layout)
        let layout = self.compute_layout(agent);
        self.agent_layouts.insert(agent.name, layout.clone());

        // Step 2: Emit init function
        // agent_init(ctx: *mut AgentContext, ...init_params...)
        self.compile_agent_init(agent, &layout)?;

        // Step 3: Emit each agent method
        for method in &agent.methods {
            self.compile_agent_method(agent, method, &layout)?;
        }

        // Step 4: Emit reset function
        self.compile_agent_reset(agent, &layout)?;

        // Step 5: Emit shutdown function
        self.compile_agent_shutdown(agent, &layout)?;

        Ok(())
    }

    /// Compile an agent method. The method receives AgentContext* as first argument.
    fn compile_agent_method(
        &mut self,
        agent: &AgentDef,
        method: &FnDef,
        layout: &AgentLayout,
    ) -> Result<(), CodegenError> {
        // Method signature: fn agent_Drafter_draft(ctx: *mut AgentContext, ...) -> ReturnType
        let mangled_name = format!(
            "agent_{}_{}",
            self.compiler.interner.resolve(agent.name),
            self.compiler.interner.resolve(method.name),
        );

        // The AgentContext pointer is the implicit first parameter
        // Field access (self.kv_cache) compiles to:
        //   load(agent_get_state(ctx) + field_offset)

        // For send() calls, compile to nsl_agent_send FFI call
        // For recv() calls, compile to nsl_agent_recv FFI call

        // The rest is standard function compilation (delegates to existing codegen)
        todo!("compile agent method body — delegates to existing fn compilation")
    }

    fn compute_layout(&self, agent: &AgentDef) -> AgentLayout {
        let mut offset = 0usize;
        let mut field_offsets = HashMap::new();
        let alignment = 64; // cache-line aligned

        for field in &agent.fields {
            let field_size = self.type_size(&field.ty);
            let field_align = self.type_alignment(&field.ty);

            // Align offset
            offset = (offset + field_align - 1) & !(field_align - 1);
            field_offsets.insert(field.name, offset);
            offset += field_size;
        }

        AgentLayout {
            agent_name: agent.name,
            field_offsets,
            total_size: (offset + alignment - 1) & !(alignment - 1),
            alignment,
        }
    }

    fn type_size(&self, ty: &Type) -> usize {
        match ty {
            Type::Tensor { .. } => std::mem::size_of::<NslTensor>(),
            Type::Int | Type::Int32 => 4,
            Type::Int64 => 8,
            Type::Float | Type::Float32 => 4,
            Type::Float64 => 8,
            Type::Bool => 1,
            _ => 8, // pointer-sized default
        }
    }

    fn type_alignment(&self, ty: &Type) -> usize {
        match ty {
            Type::Tensor { .. } => 8, // pointer alignment
            _ => std::cmp::min(self.type_size(ty), 8),
        }
    }
}
```

### 4.2 Pipeline Compilation

```rust
/// Compile a @pipeline_agent function into an orchestration function
/// that creates agents, runs the pipeline, and cleans up.
pub fn compile_pipeline(
    compiler: &mut Compiler,
    pipeline: &PipelineDef,
    agent_layouts: &HashMap<Symbol, AgentLayout>,
) -> Result<(), CodegenError> {
    // The pipeline function:
    // 1. Creates AgentContext for each agent (nsl_agent_create)
    // 2. Registers @shared tensors (nsl_agent_register_shared)
    // 3. Calls agent init methods
    // 4. Executes the pipeline body (agent method calls with send/recv)
    // 5. Cleans up (nsl_agent_destroy)

    // Pipeline body compilation:
    // - Agent method calls: compile as function calls with AgentContext* first arg
    // - Tensor transfers between agents: compile as nsl_agent_send + nsl_agent_recv
    // - @shared tensor access: compile as direct pointer read (no send/recv needed)

    todo!("pipeline body compilation")
}
```

### 4.3 Compiler Context Extension

```rust
// Added to Compiler struct in compiler.rs:
pub agent_defs: HashMap<Symbol, AgentDef>,
pub agent_layouts: HashMap<Symbol, AgentLayout>,
pub pipeline_defs: Vec<PipelineDef>,
```

### 4.4 Builtin Registration

New FFI functions registered in `register_builtins`:

```rust
fn register_agent_builtins(builtins: &mut BuiltinRegistry) {
    builtins.register("nsl_agent_create", &[I64, I64, I64, I64], Some(I64));
    builtins.register("nsl_agent_destroy", &[I64], None);
    builtins.register("nsl_agent_send", &[I64, I64, I64, I8], Some(I32));
    builtins.register("nsl_agent_recv", &[I64, I64], Some(I64));
    builtins.register("nsl_agent_reset", &[I64], None);
    builtins.register("nsl_agent_get_state", &[I64], Some(I64));
    builtins.register("nsl_agent_register_shared", &[I64], None);
    builtins.register("nsl_agent_transfer_kv_cache", &[I64, I64, I64], Some(I64));
}
```

---

## Section 5: Type System Changes

### 5.1 Agent Type

```rust
// In crates/nsl-semantic/src/types.rs
pub enum Type {
    // ... existing variants ...

    /// An agent type — like a model but with isolated state and communication.
    Agent {
        name: Symbol,
        fields: Vec<(Symbol, Type)>,
        methods: Vec<(Symbol, FunctionType)>,
    },
}
```

### 5.2 Communication Type Checking

```rust
/// Type-check agent.send() calls.
fn check_agent_send(
    &mut self,
    receiver_type: &Type,
    args: &[TypedExpr],
    span: Span,
) -> Type {
    // The argument type must match the receiver's expected input type
    // The argument is consumed (linear move) unless @shared

    for arg in args {
        let arg_type = &arg.ty;

        // Check that tensor arguments are either linear (will be moved)
        // or @shared (will be borrowed)
        if let Type::Tensor { ownership, .. } = arg_type {
            match ownership {
                Ownership::Owned => {
                    // Linear move — arg is consumed after send()
                }
                Ownership::Shared => {
                    // Shared borrow — arg remains accessible
                }
                Ownership::Borrowed | Ownership::MutBorrowed => {
                    self.error(
                        "cannot send a borrowed tensor — must be owned or @shared",
                        span,
                    );
                }
            }
        }
    }

    Type::Void
}
```

### 5.3 Agent Parser Extension

```rust
// In crates/nsl-parser/src/agent.rs
impl Parser {
    /// Parse an agent declaration.
    /// agent Name:
    ///     field: Type = default_value
    ///     fn method(self, ...) -> ReturnType:
    ///         body
    pub fn parse_agent(&mut self) -> Result<AgentDef, ParseError> {
        self.expect(Token::Agent)?;      // new keyword
        let name = self.parse_ident()?;
        self.expect(Token::Colon)?;
        self.expect_indent()?;

        let mut fields = Vec::new();
        let mut methods = Vec::new();

        while !self.at_dedent() {
            if self.peek() == Token::Fn {
                methods.push(self.parse_fn_def()?);
            } else {
                fields.push(self.parse_agent_field()?);
            }
        }

        self.expect_dedent()?;

        Ok(AgentDef {
            name,
            fields,
            methods,
            span: self.current_span(),
        })
    }
}
```

---

## Section 6: Error Messages

### 6.1 Cross-Agent State Access

```
error[E0601]: agent 'Drafter' cannot access exclusive field 'kv_cache' of agent 'Reviewer'
  --> pipeline.nsl:25:20
   |
25 |     let cache = reviewer.kv_cache
   |                 ^^^^^^^^^^^^^^^^^^ exclusive field — owned by Reviewer
   |
   = note: agents have isolated state — each agent's fields are private
   = help: use reviewer.send(kv_snapshot) to transfer ownership explicitly
   = help: or annotate the field as @shared for read-only access by all agents
```

### 6.2 Cross-Agent Mutation

```
error[E0602]: cross-agent mutation detected — agent 'Drafter' mutates 'Reviewer.score'
  --> pipeline.nsl:30:5
   |
30 |     reviewer.score = 0.0
   |     ^^^^^^^^^^^^^^^^^^^^ mutation across agent boundary
   |
   = note: cross-agent effect must be Communication only (M51)
   = note: Mutation effect is not allowed across agent boundaries
   = help: use send/recv to communicate — let Reviewer mutate its own state
```

### 6.3 Circular Ownership

```
error[E0603]: circular ownership transfer detected between agents
  --> pipeline.nsl:15
   |
15 |     drafter.send(reviewer, draft)
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Drafter -> Reviewer
   ...
22 |     reviewer.send(drafter, feedback)
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Reviewer -> Drafter (creates cycle)
   |
   = note: ownership transfers must form a DAG — cycles cause deadlock or use-after-move
   = help: use @shared for data that needs to flow in both directions
   = help: or restructure the pipeline to avoid bidirectional ownership transfer
```

### 6.4 Send of Borrowed Tensor

```
error[E0604]: cannot send a borrowed tensor to another agent
  --> pipeline.nsl:18:25
   |
17 |     let ref_cache = &drafter.kv_cache
   |                     -- immutable borrow here
18 |     reviewer.send(ref_cache)
   |                   ^^^^^^^^^ cannot send borrow — must be owned or @shared
   |
   = note: send() transfers ownership — borrows cannot be transferred
   = help: use drafter.kv_cache.snapshot() to create an owned copy
   = help: or use @shared annotation for read-only sharing
```

### 6.5 Self-Send

```
error[E0605]: agent cannot send() to itself
  --> pipeline.nsl:20:5
   |
20 |     drafter.send(drafter, data)
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ self-send is meaningless
   |
   = help: just assign the data directly: drafter.field = data
```

---

## Section 7: Testing Strategy

### 7.1 Unit Tests (AgentChecker)

| Test | What it verifies |
|------|-----------------|
| `test_agent_field_isolation` | Agent A cannot access Agent B's exclusive fields — compile error |
| `test_agent_shared_field_access` | Agent A can read Agent B's @shared fields — no error |
| `test_agent_send_ownership_transfer` | After send(), sender cannot access the transferred tensor |
| `test_agent_send_shared_retains` | @shared tensor sent to another agent: sender retains access |
| `test_agent_self_send_error` | Agent sending to itself produces compile error |
| `test_agent_cross_mutation_error` | Agent mutating another agent's field produces compile error |
| `test_agent_acyclic_communication` | Linear ownership flow (A->B->C) passes |
| `test_agent_cyclic_communication_error` | Circular flow (A->B->A) produces compile error |
| `test_pipeline_agent_validation` | @pipeline_agent with valid agents and ownership flow passes |
| `test_pipeline_undeclared_agent_error` | @pipeline_agent referencing non-existent agent produces error |
| `test_pipeline_non_shared_param_error` | Non-@shared param in @pipeline_agent produces error |
| `test_agent_effect_isolation` | Cross-agent Communication effect passes; cross-agent Mutation fails |
| `test_agent_init_method` | Agent init() method compiles correctly |
| `test_agent_reset_clears_state` | Agent reset() clears all owned state |
| `test_agent_lifecycle` | Create -> init -> process -> reset -> process -> shutdown lifecycle works |

### 7.2 Integration Tests

| Test | What it verifies |
|------|-----------------|
| `test_two_agent_kv_transfer` | Two agents: Drafter produces KV-cache snapshot, Reviewer consumes it. Zero-copy verified. |
| `test_shared_embedding_multi_agent` | @shared embedding table readable by both agents, mutation rejected |
| `test_pipeline_speculative_decoding` | Three-agent pipeline (Drafter/Verifier/Merger) for speculative decoding |
| `test_agent_with_serve_block` | Agents created in serve context, reset between requests |
| `test_agent_linear_types_integration` | Agent send/recv correctly composes with M38 ownership checker |
| `test_agent_effect_system_integration` | Agent boundaries correctly compose with M51 effect checker |

### 7.3 E2E Tests

| Test File | Description |
|-----------|-------------|
| `examples/m56_basic_two_agents.nsl` | Two agents (Encoder, Decoder) sharing KV-cache. Compile and run. |
| `examples/m56_shared_embeddings.nsl` | Three agents sharing a @shared embedding table. Verify no cross-mutation. |
| `examples/m56_pipeline_agent.nsl` | @pipeline_agent with Drafter/Reviewer/Editor. Verify ownership flow. |
| `examples/m56_cross_access_error.nsl` | Agent accessing another's exclusive field. Expect compile error. |
| `examples/m56_cross_mutation_error.nsl` | Agent mutating another's state. Expect compile error. |
| `examples/m56_circular_error.nsl` | Circular ownership transfer. Expect compile error. |
| `examples/m56_speculative_pipeline.nsl` | Full speculative decoding pipeline with three agents. End-to-end test. |

### 7.4 Performance Tests

| Test | What it verifies |
|------|-----------------|
| `bench_agent_send_latency` | send() latency < 1 microsecond (pointer transfer only) |
| `bench_agent_vs_json_serialization` | Agent send/recv vs JSON serialize/deserialize: expect 1000x+ speedup |
| `bench_kv_cache_transfer` | KV-cache transfer between agents: verify zero-copy (no memcpy) |
| `bench_shared_embedding_read` | @shared embedding read from multiple agents: verify no contention |

---

## Section 8: Deliverables

1. `agent` keyword — parser, AST node, semantic analysis, codegen for agent declarations
2. `AgentChecker` semantic pass — cross-agent ownership validation, communication graph acyclicity, effect isolation verification
3. `agent.send()` / `agent.recv()` — zero-copy tensor transfer primitives with linear ownership semantics
4. `@shared` embedding tables — read-only tensors accessible by all agents, mutation rejected at compile time
5. `@pipeline_agent` decorator — declarative multi-agent pipeline with compile-time ownership flow verification
6. `AgentContext` runtime struct — per-agent isolated state, lock-free mailbox for send/recv
7. `nsl_agent_create` / `nsl_agent_destroy` / `nsl_agent_send` / `nsl_agent_recv` / `nsl_agent_reset` FFI functions
8. Zero-copy KV-cache transfer: `nsl_agent_transfer_kv_cache` — pointer move, no data copy
9. Integration with M38 (linear types): send() consumes the tensor, preventing use-after-transfer
10. Integration with M51 (effect system): cross-agent effects restricted to Communication only

---

## Section 9: File Changes Summary

**New files:**

| File | Lines (est.) | Responsibility |
|------|-------------|----------------|
| `crates/nsl-ast/src/agent.rs` | 80 | AgentDef, AgentField AST nodes |
| `crates/nsl-parser/src/agent.rs` | 150 | Parse `agent` keyword, fields, methods |
| `crates/nsl-semantic/src/agent.rs` | 600 | AgentChecker pass: ownership validation, acyclicity, effect isolation |
| `crates/nsl-codegen/src/agent_codegen.rs` | 450 | Agent struct layout, method compilation, pipeline orchestration |
| `crates/nsl-runtime/src/agent.rs` | 300 | AgentContext, AgentMailbox, send/recv/reset FFI functions |
| `examples/m56_basic_two_agents.nsl` | 40 | E2E test: basic two-agent communication |
| `examples/m56_shared_embeddings.nsl` | 35 | E2E test: shared embedding table |
| `examples/m56_pipeline_agent.nsl` | 50 | E2E test: @pipeline_agent |
| `examples/m56_cross_access_error.nsl` | 20 | E2E test: cross-agent access error |
| `examples/m56_cross_mutation_error.nsl` | 20 | E2E test: cross-agent mutation error |
| `examples/m56_circular_error.nsl` | 25 | E2E test: circular ownership error |
| `examples/m56_speculative_pipeline.nsl` | 60 | E2E test: full speculative pipeline |

**Modified files:**

| File | Change |
|------|--------|
| `crates/nsl-ast/src/lib.rs` | `pub mod agent;` |
| `crates/nsl-parser/src/lib.rs` | `pub mod agent;` |
| `crates/nsl-parser/src/parser.rs` | Route `Token::Agent` to `parse_agent()` |
| `crates/nsl-lexer/src/lexer.rs` | Add `agent` keyword token |
| `crates/nsl-semantic/src/lib.rs` | `pub mod agent;` |
| `crates/nsl-semantic/src/checker.rs` | Invoke `AgentChecker` after type/shape/ownership/effect checking |
| `crates/nsl-codegen/src/lib.rs` | `pub mod agent_codegen;` |
| `crates/nsl-codegen/src/compiler.rs` | `agent_defs`, `agent_layouts` fields; compile agents in `compile_main()` |
| `crates/nsl-codegen/src/builtins.rs` | Register agent FFI functions |
| `crates/nsl-runtime/src/lib.rs` | `pub mod agent;` |
| `crates/nsl-cli/tests/e2e.rs` | 7 new E2E tests |

---

## Out of Scope

- Dynamic agent creation at runtime (agents are declared at compile time — no `spawn_agent()`)
- Agent migration between machines (agents live in a single process; distributed agents are future work)
- Agent-level fault tolerance (if an agent crashes, the pipeline crashes — M58's fault tolerance is at the training level)
- Priority-based agent scheduling (agents execute in pipeline order; priority queues are future work)
- Agent introspection/debugging tools (agent state visualization is deferred to future DX milestone)
- GPU-to-GPU agent communication (agents share CPU-side tensors; GPU-to-GPU requires NCCL, which is M30)
- Agent persistence (agent state is ephemeral; long-lived agent state requires serialization, which is future work)
- Multi-process agents (all agents in one process; multi-process requires IPC, deferred)
- Agent-level load balancing (all agents run on the same hardware; load-aware scheduling is future work)
- Formal verification of agent isolation soundness (empirically tested via M38/M51 composition)
- Dynamic agent composition (the agent graph is fixed at compile time; runtime composition deferred)

## Success Criteria

- Two-agent KV-cache transfer: zero-copy verified (no memcpy in trace, pointer value identical)
- send() latency: < 1 microsecond for pointer-sized transfer
- Cross-agent field access: compile error with clear message pointing to the exclusive field
- Cross-agent mutation: compile error with clear message about effect isolation
- Circular ownership: compile error with cycle path
- @shared embedding: readable by all agents, write attempt produces compile error
- @pipeline_agent: ownership flow verified at compile time, three-agent pipeline runs correctly
- Performance vs LangChain: 1000x+ speedup for equivalent multi-agent workload (tensor transfer, not Python string processing)
- All existing tests pass unchanged — agent keyword is additive functionality
- M38 composition: agent send/recv correctly integrates with linear type ownership checker
- M51 composition: cross-agent effects correctly restricted to Communication only
