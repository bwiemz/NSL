# M56: Multi-Agent Shared Memory — v1 Design Specification

**Date:** 2026-04-23
**Status:** Planned
**Milestone:** M56 (Phase 11, v1.0.0)
**Prerequisites:** M38 (linear types — ownership checker shipped, `--linear-types` gated), M51 (effect system — `EffectSet::COMMUNICATION` exists), M25 (paged KV-cache — `crates/nsl-runtime/src/paged_kv/` shipped)
**Supersedes:** `docs/superpowers/specs/2026-03-19-m56-multi-agent-design.md` and `docs/superpowers/plans/2026-03-21-m56-multi-agent.md`. Both predecessors are retained for history; this spec is authoritative.

## Overview

M56 compiles multiple AI agents into a single NSL binary where they share KV-cache, embedding tables, and activations via zero-copy pointer transfers instead of JSON serialization across process boundaries. It is the "LangChain killer": NSL agents share the same address space with compile-time guarantees that no agent can corrupt another's state.

The design combines three load-bearing ideas:

1. **Reactor-model semantics** from the Lingua Franca research (typed ports + Action-Port Graph + logical-time scheduling) — these are the compile-time-correctness primitives that give NSL's agent system the properties LangChain deployments spend engineering effort trying to approximate.
2. **Linear types and the effect system** (M38, M51) supply the cross-boundary ownership proofs: a port send is a linear move; a port receive is a linear acquire; cross-agent mutation is rejected at the effect-checker level.
3. **Separation of semantic model from scheduler implementation.** The semantic model (ports, APG, logical time, linear ownership) ships in v1. The v1 scheduler is a single-threaded deterministic event loop. A v2 reactor scheduler (thread-per-agent pinning, lock-free dispatch) will layer on top, validated by replaying v1-derived execution traces as its reference implementation.

This split is deliberate: overcommitting to thread-per-core pinning before the semantic model has iterated is the dominant failure mode for concurrency features. The v1 event-loop scheduler is deterministic, trivially testable, and replaceable. v2's reactor scheduler will be correct-by-definition iff it produces the same logical-time-ordered outputs as v1 for equivalent inputs — a regression invariant that requires v1 to exist first.

---

## Section 1: Language Surface

### 1.1 `agent` keyword

```nsl
agent Drafter:
    model: SmallLLM
    kv_cache: KvCache<[MaxSeq, NumHeads, HeadDim], f16, device=gpu>
    temperature: f32 = 0.8

    fn init(self, model_path: str):
        self.model = load_model(model_path)
        self.kv_cache = KvCache.empty(max_seq=2048, num_heads=32, head_dim=128)

    fn draft(self, prompt: Tensor<[1, S], int32, device=gpu>) -> DraftResult:
        let tokens = self.model.generate(prompt, max_new=64, kv=self.kv_cache)
        return DraftResult {
            tokens: tokens,
            kv_snapshot: self.kv_cache.snapshot(),
        }

    fn reset(self):
        self.kv_cache.clear()

    fn shutdown(self):
        pass
```

Semantics:

- Like `model`, but with isolated mutable state. The compiler enforces that no other agent can read or write an agent's fields unless they are declared `@shared`.
- Agent methods may mutate `self` but never another agent's state.
- Communication between agents is implicit through the method-call syntax in `@pipeline_agent` bodies (Section 1.4); no user-facing `send()`/`recv()` is exposed in v1.
- Lifecycle hooks: `init(self, ...)` (construction), `reset(self)` (called at pipeline-lease release, Section 3.3), `shutdown(self)` (called when the agent instance is destroyed).

### 1.2 Port declarations (method-signature-derived)

**The method signature IS the port declaration.** For a method `fn review(self, draft: DraftResult) -> ReviewResult`, the compiler derives two ports:

- `in_draft: DraftResult` — one input port per tensor/struct parameter.
- `out_review: ReviewResult` — one output port for the return value, named `out_<method_name>` (or `out_<return_binding_name>` if explicitly named; default is method name).

Port names come from the parameter name (`in_<param_name>`). Agents can also declare `port` fields explicitly only for channels that do not correspond 1:1 with a method (feedback, side channels); these are v2+ scope and not part of v1.

Port types are **invariant**: the argument passed at a call site must match the declared type exactly. Subtyping, generics over device, and struct-subset coercions are all deferred; extending to variance is a specifically-additive v2+ language feature.

### 1.3 Communication types (structs-as-messages)

Structs passed across port boundaries follow field-wise ownership rules:

```nsl
struct DraftResult:
    tokens: Tensor<[1, N], int32, device=gpu>         # linear — moves with struct
    kv_snapshot: KvCacheSnapshot                       # linear — moves with struct
    metadata: dict<str, f32>                           # scalar/copy — copied

struct ReviewResult:
    score: f32                                         # scalar — copied
    accepted: bool                                     # scalar — copied
    attention_map: Tensor<[N, N], f16, device=gpu>     # linear — moves with struct

struct KvCacheSnapshot:
    keys: Tensor<[NumLayers, NumHeads, S, HeadDim], f16, device=gpu>
    values: Tensor<[NumLayers, NumHeads, S, HeadDim], f16, device=gpu>
    seq_len: i32
```

**Port sends are struct-granularity moves.** `agent.port_out.send(my_struct)` — desugared from method-call syntax — is equivalent to "move `my_struct` into the port." Partial moves (extracting a field before the send) are allowed under existing M38 rules, but the sender must reconstitute the struct before sending. Once sent, the sender's binding is invalidated.

**@shared fields on struct send:** the @shared field's refcount is bumped at the send site; the downstream receiver holds a reference at the bumped refcount; drops decrement. When a struct with @shared fields is sent, the sender loses access to the entire struct (including its @shared fields). To retain access to a @shared field after sending the struct, the sender must extract/clone it before the send:

```nsl
let snapshot_for_me = draft.kv_snapshot.clone()    # refcount bump; new binding
reviewer.review(draft)                             # struct moves; draft unusable now
# snapshot_for_me is still valid — it was extracted before the send
```

This is the "(a) sender loses access on send; extract-before-send to retain" rule pinned during Q2.

### 1.4 `@pipeline_agent` decorator and APG extraction

```nsl
@pipeline_agent(agents=[Drafter, Reviewer, Editor])
fn speculative_review(
    prompt: Tensor<[1, S], int32, device=gpu>,
    @shared embeddings: Tensor<[VocabSize, EmbedDim], f16, device=gpu>,
) -> Tensor<[1, OutS], int32, device=gpu>:
    let draft = drafter.draft(prompt)        # edge: pipeline-input -> drafter.in_prompt;
                                             #       edge: drafter.out_draft -> binding `draft`
    let review = reviewer.review(draft)      # edge: binding `draft` -> reviewer.in_draft;
                                             #       edge: reviewer.out_review -> binding `review`
    if review.accepted:
        return editor.edit(review)
    else:
        let new_draft = drafter.draft(prompt)
        return editor.edit(reviewer.review(new_draft))
```

**APG extraction is data-flow-inferred, not explicitly wired.** The compiler walks the pipeline body and extracts one edge per (binding → consumer-method-parameter) pair. The rule:

- Each agent method call produces one output edge terminating at a pipeline-site binding.
- Each variable binding has zero or more downstream uses; each use is one edge into a consumer method's parameter port.
- Use counts are enforced by the existing M38 ownership walker: linear bindings must be used exactly once, @shared bindings may be used multiple times, copy/scalar bindings are unconstrained.
- The compiler refuses cycles in the APG at compile time (DFS cycle detection with a path-qualified error pointing at the offending edges).

**Destructure-and-forward is NOT port-level fan-out.**

```nsl
let (draft, metadata) = drafter.draft(prompt)    # struct destructured at binding site
reviewer.review(draft)                           # field moves to one consumer
logger.log(metadata)                             # different field moves to another consumer
```

Here `drafter.draft` has one outgoing edge (to the pipeline-site binding that performs destructuring). The fields are then independently consumed. This is *not* fan-out — each field has exactly one consumer; the struct itself moves once.

**True fan-out** (the same struct value delivered to multiple downstream agents) requires @shared on the struct's content — the struct's linear fields cannot be fanned out. The compiler refuses fan-out of linear content with a message suggesting @shared or an explicit clone:

```nsl
let draft = drafter.draft(prompt)    # linear
reviewer.review(draft)               # draft consumed here
logger.log(draft)                    # ERROR: use after move; draft was consumed by reviewer
```

Fan-out of @shared content works because the refcount bumps per downstream edge.

### 1.5 `@shared` read-only globals

```nsl
@shared
let embeddings: Tensor<[VocabSize, EmbedDim], f16, device=gpu> = load("embeddings.safetensors")

agent Drafter:
    fn draft(self, prompt: Tensor<[1, S], int32, device=gpu>) -> DraftResult:
        let h = embed(&embeddings, prompt)    # immutable borrow, no ownership transfer
        ...
```

All agents in the pipeline can read `@shared` globals. Mutation is rejected at compile time. The refcount for `@shared` globals is pinned to a sentinel value for the pipeline's lifetime; agents cannot free them.

### 1.6 Device placement on ports

Every `Tensor<...>` type carries a `device` parameter. Port device rules:

1. **Same device on both endpoints** — zero-copy pointer transfer, no diagnostic emitted.
2. **Cross-device mismatch at a port** — **hard refusal by default** with a clear error that cites the specific mismatch.
3. **`@auto_device_transfer` annotation on the target method** — when and only when the receiving method is annotated `@auto_device_transfer`, the compiler inserts an explicit `.to(dst_device)` at the call site, emits a diagnostic with the transfer size, and proceeds.

```nsl
agent Tokenizer:
    fn tokenize(self, text: str) -> Tensor<[1, S], int32, device=cpu>:
        ...

agent Model:
    @auto_device_transfer
    fn forward(self, tokens: Tensor<[1, S], int32, device=gpu>) -> Tensor<[1, S], f32, device=gpu>:
        ...

@pipeline_agent(agents=[tokenizer, model])
fn pipeline(text: str) -> Tensor<[1, S], f32, device=gpu>:
    let tokens = tokenizer.tokenize(text)    # device=cpu
    return model.forward(tokens)             # requires device=gpu; auto-transfer inserted
```

The compiler's diagnostic for the inserted transfer:

```text
note: inserted device transfer at call site `model.forward(tokens)`
      source device: cpu  destination device: gpu
      size: 4.0 KB (shape [1, S=512], dtype int32)
      per Model.forward's @auto_device_transfer annotation.
```

Three diagnostic cases:

- **Same device** — no transfer, no diagnostic, zero-copy.
- **Mismatched, target method opts in** — transfer inserted, diagnostic emitted with the size estimate.
- **Mismatched, target method does not opt in** — hard error with the suggested fix.

**`@auto_device_transfer` applies to input parameters only, not return values.** When a caller binds the return of an `@auto_device_transfer`-annotated method and then passes that value to a downstream method on a different device, the *downstream* method's annotation (or lack of one) governs whether that transfer is auto-inserted or refused. Each method independently declares its transfer policy at its own parameter boundary; there is no spooky action at a distance where one method's annotation permits transfers at unrelated call sites.

**@shared preserves device across ports.** A `@shared<Tensor, device=gpu>` flowing through a struct port is only accepted by downstream methods whose parameter declares `device=gpu`. Cross-device @shared is refused just as cross-device linear is refused; the refcount bookkeeping would be well-formed but the pointer is device-addressed and can't be dereferenced from the wrong device.

**Cross-GPU is hard-refused in v1** (different device IDs, e.g. `gpu:0` vs `gpu:1`). The error cites M30 explicitly as the planned resolution; see Section 6.4.

### 1.7 `serve` block and agent pooling

```nsl
serve my_api on port=8080 pool_size=4:
    route "/generate":
        let result = speculative_review(request.tokens, embeddings)
        return result
```

Agents referenced by a `@pipeline_agent` function invoked inside a `serve` block are managed by a pool of **pipeline-execution contexts**. Each context contains one instance of every agent in the pipeline; the pool holds `pool_size` such contexts. When a request arrives, it acquires one context (lease), runs the pipeline to completion, and releases the context.

- `pool_size` defaults to `1` (single-context, requests serialize). Concurrent request handling requires an explicit `pool_size=N` in the serve block.
- Memory cost at server startup is `pool_size × sum(per-agent memory)`. A request for a pool that can't fit in available memory is a **serve-block-construction error** (reported at server start, not at the Nth request). This includes GPU memory if any agent holds GPU state.
- Agent instances are never shared across leases. `pool_size=4` with a 3-agent pipeline means 12 distinct agent instances at startup. The ownership model disallows cross-lease sharing; each lease owns its entire pipeline context.

### 1.8 Pool lease as a linear resource

The lease itself is a linear resource exposed to the compiler (not to user code — leases are acquired/released by the serve-block machinery, not by user method calls). `acquire()` produces a linear value; `release()` or drop consumes it. This composes with M38 without adding new analysis surface: the pool is a typed resource that produces and consumes linear values at its API boundary.

### 1.9 Lifecycle: `reset()` and reset-failure handling

`reset(self)` is called at **lease-release**, not lease-acquire. The pool's invariant is:

> Every instance handed out is either in post-reset state or has been removed from the pool.

The release path:

1. Call `instance.reset()`.
2. If `reset()` succeeds, return the instance to the pool.
3. If `reset()` fails (panic, error return, deadlock-timeout), **remove the instance from the pool** and emit a reset-failure diagnostic for operations.

**On reset failure, `contexts[i]` becomes a tombstone** (`Option::None` in the backing `Vec`). The index is not re-added to the `available` queue, and the pool's effective `size` decrements by one. **Replacement construction is not attempted in v1**: pool capacity erodes with accumulated reset failures. Operations should monitor the reset-failure diagnostic; sustained reset failures indicate a bug in the pipeline or in an agent's `reset()` implementation. A replacement-construction path (either eager at release time or lazy in a background task) is v2+ scope.

**The failing request returns its original error to the client**, not the reset failure. The reset-failure is an internal pool-health event; the client sees only what went wrong with their request. This matches the "(a)" framing from Q5.

The implementation MUST NOT return a non-reset-verified instance to the pool under any failure mode. Silent state-corruption across requests is the exact class of bug this rule exists to prevent.

---

## Section 2: Semantic Model

### 2.1 Logical time

Execution proceeds in discrete **logical time steps** indexed by `T ∈ {0, 1, 2, ...}`. The scheduler maintains a global logical time counter. At each step:

1. Compute the set of agents that are **ready** at time `T` (all of their input ports have values written at some `T' < T`).
2. Fire each ready agent at most once during step `T`, in APG topological order.
3. Outputs written by an agent firing at time `T` become visible to downstream agents at time `T+1`.
4. Advance to time `T+1`.

This is the reactor-model default: an agent fires at most once per step; produced-at-T is visible-at-T+1. It is the invariant that both v1's event-loop scheduler and v2's multi-threaded reactor scheduler must respect. The identical logical-time trace produced by v1 for a given pipeline + inputs is the reference output v2 must match.

### 2.2 Action-Port Graph (APG)

An APG is a directed graph where:

- Nodes: `(agent_instance, port_name, direction)` triples. Directions are `Input` or `Output`.
- Edges: Output-port → Input-port. One edge per (binding → consumer-method-parameter) pair extracted from the `@pipeline_agent` body.
- The APG must be acyclic. Cycles are rejected at compile time with a path-qualified error.

The APG is extracted entirely at compile time; there is no runtime topology construction in v1.

### 2.3 Ownership model across ports

All cross-port transfers are M38-typed. The port boundary is a type-checked move point:

- **Linear types** — moved. Sender binding invalidated after the call.
- **@shared types** — refcount bumped on send; receiver gets a refcount-increased reference; both can access. Drop semantics proceed normally.
- **Copy/scalar types** — byte-copied.

The existing `OwnershipChecker` walks each `@pipeline_agent` body and enforces these rules. No new ownership analysis is introduced; the existing linear-types pass must be active (Section 7).

### 2.4 Effect isolation across agent boundaries

Cross-port calls carry the `Communication` effect only. The compiler rejects:

- An agent method whose effects propagated via a cross-port call would introduce `Mutation` on another agent's state.
- An agent method annotated as pure that has any effect at a port boundary.

`Communication` is a pre-existing effect in [crates/nsl-semantic/src/effects.rs:21](crates/nsl-semantic/src/effects.rs#L21) (`EffectSet::COMMUNICATION`); M56 uses it without extending the effect system.

---

## Section 3: Runtime

### 3.1 v1 Scheduler: single-threaded logical-time event loop

**File:** `crates/nsl-runtime/src/agent/scheduler.rs` (new).

The v1 scheduler runs on one thread. Each logical time step:

```rust
pub fn step(&mut self) -> StepOutcome {
    let ready = self.compute_ready_set();       // agents with all inputs ready at time T
    if ready.is_empty() { return StepOutcome::Idle; }

    for agent_id in self.apg.topo_order(&ready) {
        let outputs = self.fire(agent_id);      // synchronous call into agent's method
        for (port, value) in outputs {
            self.mailbox_at(port).write(value, self.logical_time + 1);
        }
    }

    self.logical_time += 1;
    StepOutcome::Advanced
}
```

Determinism: same inputs + same APG → same logical-time trace, every run. No thread scheduling, no race conditions, no test flakiness.

### 3.2 Mailbox layout

**File:** `crates/nsl-runtime/src/agent/mailbox.rs` (new).

One mailbox per port. Each mailbox holds at most one value per logical time step:

```rust
#[repr(C)]
pub struct PortMailbox {
    /// The message written at `stamped_time`, if any. Holds the entire port
    /// payload — for tensor-typed ports this is a single `NslTensor`; for
    /// struct-typed ports (Q2) this is a heap-allocated struct payload whose
    /// field ownership matches the declared struct type.
    pub slot: Option<PortMessage>,
    /// The logical time at which the slot was written.
    pub stamped_time: u64,
    /// The expected reader's logical time (for visibility-at-T+1 enforcement).
    pub expected_read_time: u64,
}

/// A port's wire payload. Struct ports carry a heap-allocated struct whose
/// field ownership is governed by M38's struct-move rules (Section 1.3);
/// tensor ports carry an `NslTensor` by value.
pub enum PortMessage {
    Tensor(NslTensor),
    Struct(Box<StructPayload>),
}
```

The scheduler enforces visibility-at-T+1 by requiring `expected_read_time > stamped_time`. If a downstream agent attempts to read before the visibility window, that's an internal scheduler bug (not user-observable; the APG topological order should prevent it).

For tensor-typed ports, `NslTensor` is the existing 13-field `#[repr(C)]` struct from `crates/nsl-runtime/src/tensor/mod.rs:154`. The `device: u8` tag travels with the tensor; the scheduler does not copy tensor data, only the struct. For struct-typed ports, `StructPayload` is a heap allocation whose layout matches the declared struct's codegen layout (Section 4); moving the struct across a port is a pointer move plus refcount bumps for any `@shared` fields per Section 1.3.

### 3.3 Pool

**File:** `crates/nsl-runtime/src/agent/pool.rs` (new).

```rust
pub struct PipelineContextPool {
    /// Fixed-length Vec sized at construction. Entries become `None` (tombstoned)
    /// on reset failure and are never repopulated in v1.
    contexts: Vec<Option<PipelineContext>>,
    /// Indices of currently-available (non-leased, non-tombstoned) contexts.
    available: VecDeque<usize>,
    /// Current effective pool size — decrements on reset failure (tombstoning).
    /// Starts equal to pool_size; monotonically non-increasing in v1.
    size: usize,
}

pub struct PipelineContext {
    agents: HashMap<AgentTypeId, AgentInstance>,    // one instance per agent type in the pipeline
    scheduler: ReactorScheduler,                    // the v1 single-threaded scheduler
}

impl PipelineContextPool {
    pub fn acquire(&mut self, timeout: Duration) -> Result<LeasedContext, AcquireError> { ... }
    pub fn release(&mut self, lease: LeasedContext) { /* calls reset_or_remove */ }
    fn reset_or_remove(&mut self, ctx: &mut PipelineContext) -> ResetOutcome { ... }
}
```

The `release` path calls `reset_or_remove` per Section 1.9.

### 3.4 FFI surface

| Function | Signature | Purpose |
|----------|-----------|---------|
| `nsl_agent_pool_new` | `(pool_size: u64, pipeline_fn_id: u64) -> *mut PipelineContextPool` | Construct pool at serve block entry |
| `nsl_agent_pool_destroy` | `(*mut PipelineContextPool)` | Destroy pool at serve block exit |
| `nsl_agent_pool_acquire` | `(pool: *mut PipelineContextPool, timeout_ms: u64) -> i64` | Acquire lease; returns lease id or -1 |
| `nsl_agent_pool_release` | `(pool: *mut PipelineContextPool, lease_id: i64)` | Release lease; runs reset_or_remove |
| `nsl_agent_scheduler_step` | `(sched: *mut ReactorScheduler) -> i32` | Run one logical time step; returns StepOutcome |
| `nsl_agent_mailbox_write` | `(mb: *mut PortMailbox, msg: PortMessage, time: u64) -> i32` | Write a port payload at time T+1 (tensor or struct per Section 3.2) |
| `nsl_agent_mailbox_read` | `(mb: *mut PortMailbox) -> PortMessage` | Read a port payload at current time |

These are the *only* new FFI symbols. The runtime pool and scheduler are not user-visible — users write `serve` blocks and `@pipeline_agent` functions; the compiler emits the appropriate FFI sequence.

---

## Section 4: Type System Changes

### 4.1 New types

```rust
// crates/nsl-semantic/src/types.rs
pub enum Type {
    // ... existing variants ...
    Agent {
        name: Symbol,
        fields: Vec<(Symbol, Type, OwnershipMarker)>,
        methods: Vec<(Symbol, FunctionType)>,
    },
    Port {
        direction: PortDirection,    // Input | Output
        payload_type: Box<Type>,
    },
}
```

Ports are *not* a user-declared type in v1 — they exist internally, materialized from method signatures. They appear in type-error messages where they help explain failures (e.g. "`in_draft` expects `DraftResult`, got `RawTokens`"), but they're not instantiable directly.

### 4.2 Annotation: `@auto_device_transfer`

A method-level annotation accepted by the parser; the type checker consults it when validating cross-device port calls per Section 1.6. No new type variants; the annotation is metadata on the function definition.

### 4.3 Pipeline validation

`crates/nsl-semantic/src/agent.rs` (new) runs after type/shape/ownership/effect checking. It:

1. Registers all `agent` declarations.
2. For each `@pipeline_agent` function, walks the body to extract the APG.
3. Runs cycle detection on the APG.
4. Runs the device-compatibility check at each edge.
5. Composes with the existing `OwnershipChecker` to verify linear-move semantics at each method call (the ownership checker already handles this; no duplicated logic).
6. Composes with `EffectChecker` to verify Communication-only cross-boundary effects.

---

## Section 5: Codegen

### 5.1 Agent compilation

Agents compile to C-layout structs. Each agent method compiles to a Cranelift function taking `*mut AgentInstance` as its first argument. Field access (`self.kv_cache`) compiles to `load(state_ptr + field_offset)`.

**File:** `crates/nsl-codegen/src/agent.rs` (new).

### 5.2 `@pipeline_agent` function compilation

A `@pipeline_agent` function compiles to:

1. Scheduler initialization (using precomputed APG from semantic analysis).
2. For each method call in the pipeline body, emit:
   - Write arguments to the target agent's input mailboxes.
   - **Exactly one** `nsl_agent_scheduler_step` call. v1 pipeline bodies are linear chains over an acyclic APG (Q3 data-flow extraction, Section 2.2), and the scheduler fires in topological order within a step — so the called agent's output is guaranteed available after exactly one step. The compiler statically derives this from the APG shape; emitting a step-loop is unnecessary and rejected as an implementation choice in v1.
   - Read outputs from the agent's output mailbox; bind to the pipeline-site variable.
3. Device-transfer inserts at `@auto_device_transfer` sites.
4. Return the final output.

**Forward note:** v2 topologies with feedback edges or internal multi-step reactor computation will require loop-based step emission. The single-step form is v1-specific and is validated by Section 2's acyclic-APG constraint; v2 will relax this.

### 5.3 `serve` integration

`serve` blocks compile to:

1. At entry: `nsl_agent_pool_new` with pool_size (from `pool_size=N` syntax or default 1).
2. Per route: the route handler acquires a lease, calls the pipeline function with the leased context, releases.
3. At exit: `nsl_agent_pool_destroy`.

---

## Section 6: Error Messages

Each error follows the three-part `requested / expected / found` template pinned by `feedback_transformation_precondition_refusal.md`.

### 6.1 Cross-agent state access

```text
error[E0601]: agent 'Drafter' cannot access exclusive field 'kv_cache' of agent 'Reviewer'
  --> pipeline.nsl:25:20
   |
25 |     let cache = reviewer.kv_cache
   |                 ^^^^^^^^^^^^^^^^^^ exclusive field — owned by Reviewer

  requested: cross-agent field read
  expected:  field marked @shared, or self-access inside the owning agent
  found:     Drafter accessing Reviewer.kv_cache (Exclusive)

  fix: use method-call syntax to move/borrow via a port, or annotate the field
       as @shared for read-only access.
```

### 6.2 Cross-agent mutation

```text
error[E0602]: cross-agent Mutation effect rejected
  --> pipeline.nsl:30:5
   |
30 |     reviewer.score = 0.0
   |     ^^^^^^^^^^^^^^^^^^^^ Mutation crosses agent boundary

  requested: cross-agent effect
  expected:  Communication only
  found:     Mutation on Reviewer.score from within Drafter
```

### 6.3 Circular ownership (cycle in APG)

```text
error[E0603]: circular port topology rejected — APG contains a cycle
  --> pipeline.nsl:15-22

  requested: acyclic APG
  expected:  no cycle in port-to-port edges
  found:     cycle Drafter.out_draft -> Reviewer.in_draft -> Drafter.in_feedback

  fix: restructure the pipeline so ownership flows in one direction, or use
       @shared for data that must flow bidirectionally.
```

### 6.4 Cross-GPU port connection (v1 constraint)

```text
error[E0607]: cross-GPU port connection not supported in v1
  --> pipeline.nsl:42:5

  requested: port connection from agent A (device=gpu:0) to agent B (device=gpu:1)
  expected:  both agents on the same device (v1 constraint)
  found:     A is on gpu:0, B is on gpu:1

  planned: cross-device communication via NCCL is scheduled for M30.
           Until then, either colocate the agents on one GPU or use a
           CPU intermediary agent (bearing the transfer cost).
```

### 6.5 Device mismatch without `@auto_device_transfer`

```text
error[E0608]: cross-device port call rejected — target method has no @auto_device_transfer
  --> pipeline.nsl:51:5

  requested: call Model.forward with a cpu tensor
  expected:  device=gpu (declared on Model.forward's parameter)
  found:     argument `tokens` is device=cpu

  fix: either (a) insert an explicit transfer: `let t = tokens.to(gpu); model.forward(t)`,
       or (b) annotate Model.forward as @auto_device_transfer to opt in to
       compiler-inserted transfers (transfer cost will be reported at compile time).
```

### 6.6 Fan-out of linear content

```text
error[E0609]: cannot fan out linear content — use after move
  --> pipeline.nsl:55:5

  requested: two downstream consumers of a linear binding
  expected:  exactly one consumer, OR @shared on the fanned-out content
  found:     `draft` consumed by both reviewer.review and logger.log

  fix: choose based on intent:
       (a) both consumers need the *same* data → make the struct's tensor
           fields @shared, or explicitly clone: `logger.log(draft.clone())`.
       (b) consumers need *different parts* of the struct → destructure at
           the binding: `let (tokens, meta) = drafter.draft(prompt);`
           then route each field to its one consumer (destructure-and-forward,
           per Section 1.4 — this is NOT fan-out).
```

### 6.7 `--linear-types` required

```text
error[E0610]: M56 agent declarations require --linear-types
  --> pipeline.nsl:3:1
   |
 3 | agent Drafter:
   | ^^^^^ found here

  requested: compile a program containing an agent declaration
  expected:  the linear ownership checker (--linear-types) active
  found:     --linear-types not passed to the compiler

  fix: add --linear-types to `nsl check` or `nsl build`. Note: `nsl run`
       does not currently expose --linear-types; for run, use `nsl build`
       followed by direct execution.
```

The error does **not** mention an `Nsl.toml` setting — that config mechanism doesn't exist in the repo as of this spec. Tracking the `nsl run` flag gap as a follow-up (see Section 10).

---

## Section 7: Flag Gating

M56 requires `--linear-types`. The error fires at the first `agent` declaration encountered during compilation if the flag is absent; the message is specified in Section 6.7.

**Rationale.** Implicit auto-enable (option A in Q6) would spread linear-type errors across non-agent code for users who added a single agent declaration — poor diagnostic locality. A separate `--agents` flag (option C) would commit the language to two-flag ceremony for a pairing that has no realistic decoupled use. Requiring `--linear-types` explicitly matches `feedback_unknown_ownership_strict.md`: M38 is still being shaken out and should be opt-in until it graduates.

**`nsl run` gap.** The `--linear-types` flag is accepted by `nsl check` and `nsl build` but not `nsl run` ([crates/nsl-cli/src/main.rs:1518](crates/nsl-cli/src/main.rs#L1518) hardcodes `linear_types_enabled: false`). Fixing this is a prerequisite for the primary use case (running pipeline `.nsl` files directly) and is a named M56 v1 task (Section 10).

**Future graduation.** When M56 v1 has been in production use long enough that `--linear-types` no longer produces surprise errors in previously-compiling code (empirical criterion, tracked as a post-v1 milestone), `--linear-types` becomes default-on and both M38 and M56 fold into the standard compile path. Until then, the flag is the opt-in gate. This is explicit future work so the flag does not become permanent scaffolding.

---

## Section 8: Testing Strategy

### 8.1 Unit tests (AgentChecker + APG extraction)

| Test | What it verifies |
|------|------------------|
| `test_agent_field_isolation` | Agent A cannot access Agent B's exclusive fields — compile error with E0601 |
| `test_agent_shared_field_access` | Agent A can read `@shared` fields — no error |
| `test_port_name_inference_from_params` | `fn review(self, draft: DraftResult)` derives `in_draft` |
| `test_port_invariance` | `DerivedDraftResult` sent through a `port: DraftResult` — compile error |
| `test_apg_extraction_linear_chain` | Three-agent linear pipeline extracts three edges |
| `test_apg_extraction_destructure_forward` | `let (a, b) = method(x)` extracts two outgoing edges from binding, one incoming to method |
| `test_apg_cycle_detection_direct` | A → B → A refused with E0603 |
| `test_apg_cycle_detection_transitive` | A → B → C → A refused with E0603 |
| `test_linear_fan_out_refused` | Two consumers of a linear binding — E0609 |
| `test_shared_fan_out_allowed` | Two consumers of a `@shared`-field binding — no error |
| `test_struct_move_invalidates_fields` | After `send(my_struct)`, field access on `my_struct` is use-after-move |
| `test_shared_retention_via_extract_before_send` | `let x = s.field.clone(); send(s); use(x)` — OK |
| `test_cross_agent_mutation_refused` | A mutating B.f — E0602 |

### 8.2 Device-handling tests

| Test | What it verifies |
|------|------------------|
| `test_same_device_zero_copy` | No transfer, no diagnostic |
| `test_auto_device_transfer_opted_in` | Transfer inserted, diagnostic includes size |
| `test_device_mismatch_without_annotation` | E0608 refusal |
| `test_cross_gpu_refused` | E0607 refusal with M30 citation |
| `test_shared_preserves_device` | `@shared<Tensor, gpu>` into a cpu-method parameter — E0608 |

### 8.3 Scheduler + pool tests

| Test | What it verifies |
|------|------------------|
| `test_scheduler_deterministic_replay` | Same APG + same inputs → same output trace across runs |
| `test_scheduler_logical_time_advance` | Outputs at T visible at T+1, not before |
| `test_pool_default_size_one` | Serve block without `pool_size=` creates a 1-context pool |
| `test_pool_concurrent_leases` | `pool_size=4` serves 4 concurrent requests in parallel |
| `test_pool_exhaustion_queues` | Pool size N, N+1 concurrent requests — the extra queues |
| `test_pool_reset_success_returns_instance` | After a clean pipeline, instance returns to pool post-reset |
| `test_pool_reset_failure_removes_instance` | Reset panics → instance removed, pool size decrements |
| `test_pool_reset_failure_request_sees_original_error` | Failing request's client receives its pipeline error, not the reset error |
| `test_pool_oversize_refused_at_construction` | `pool_size=1000` for agents totaling > available memory — serve-block-construction error |

### 8.4 Flag-gating tests

| Test | What it verifies |
|------|------------------|
| `test_agent_without_linear_types_flag` | Compiling an `agent` without `--linear-types` — E0610 |
| `test_agent_with_linear_types_flag` | Same source with `--linear-types` — compiles |
| `test_nsl_run_linear_types_gap_documented` | Invoking `nsl run` on a pipeline file currently fails (regression guard for the known gap) |

### 8.5 End-to-end examples

| Example | Purpose |
|---------|---------|
| `examples/m56_basic_two_agents.nsl` | Simplest working pipeline — Drafter → Editor on GPU |
| `examples/m56_shared_embeddings.nsl` | `@shared` embedding table read by three agents |
| `examples/m56_speculative_decoding.nsl` | Three-agent speculative pipeline (Drafter/Verifier/Editor) end-to-end |
| `examples/m56_serve_pool.nsl` | `serve` block with `pool_size=4` |
| `examples/m56_device_transfer_opt_in.nsl` | Tokenizer (CPU) → Model (GPU, `@auto_device_transfer`) |
| `examples/m56_cross_agent_access_error.nsl` | Negative: agent reading another's field |
| `examples/m56_cycle_error.nsl` | Negative: circular APG |
| `examples/m56_cross_gpu_error.nsl` | Negative: cross-GPU port; verifies M30 citation in error |

### 8.6 Performance tests

| Test | Target |
|------|--------|
| `bench_port_send_latency` | Port handoff < 1 μs (pointer/metadata move only) |
| `bench_kv_cache_pointer_transfer` | KV-cache transfer between agents on same GPU: zero-copy verified (no memcpy in trace) |
| `bench_pool_lease_overhead` | `acquire + release` round-trip < 10 μs at steady state |
| `bench_pipeline_vs_manual_orchestration` | A three-agent pipeline via M56 is within 10% of hand-written orchestration |

Note: the "1000× vs LangChain" claim from the predecessor spec is aspirational and removed from this one — it's a marketing number, not a test target. The concrete targets above are what the test suite asserts.

---

## Section 9: v2 Forward Path (Out of Scope for This Spec)

Named v2+ items, each with a specific trigger for revival:

- **Reactor scheduler** (thread-per-agent pinning, lock-free atomic dispatch). Trigger: v1 shipped, real users deploying pipelines, measurable concurrency bottleneck in the single-threaded scheduler. v1's deterministic trace becomes the reference implementation; v2 must produce identical logical-time-ordered outputs to be accepted.
- **Cross-GPU communication** (M30, NCCL). Trigger: M30 ships. M56 v1 error message (E0607) cites this explicitly.
- **Explicit `connect` syntax** for topologies the method-call syntax cannot express (feedback loops, side channels, fan-out to multiple agents). Trigger: a real user use case that v1's data-flow extraction cannot express without awkward workarounds. Until then, B's data-flow-only surface is the design; adding a second surface should be driven by specific needs.
- **Port-type variance** (subtyping, covariance/contravariance). Trigger: schema evolution in production pipelines. v1 is invariant by design.
- **Dynamic agent spawning** (agents created at runtime rather than declared). Out of scope indefinitely; agents are a compile-time concept.
- **Multi-process agents** and distributed agent pipelines. Out of scope indefinitely in the current milestone planning; requires IPC infrastructure beyond M56.
- **Per-agent pools** (separate pool for each agent type, instances leased independently). Trigger: pipelines with very asymmetric per-agent memory costs where full replication is wasteful. Not on the horizon.

---

## Section 10: Known v1 Tasks (Named, Not Plan-Scheduled)

These are load-bearing tasks the implementation plan will schedule; they are enumerated here so the spec is internally complete.

1. Add `agent` and `@pipeline_agent` tokens to the lexer keyword table.
2. Extend the AST (`crates/nsl-ast/src/decl.rs`) with `AgentDef` wrapped in `StmtKind::AgentDef`, following the `ModelDef` template.
3. Extend the parser (`crates/nsl-parser/src/decl.rs`) with `parse_agent_def_stmt`, dispatched from `crates/nsl-parser/src/stmt.rs` on `TokenKind::Agent`.
4. Create `crates/nsl-semantic/src/agent.rs`: APG extraction, cycle detection, device-compatibility checking, composition with existing ownership/effect checkers.
5. Create `crates/nsl-runtime/src/agent/` with `scheduler.rs`, `mailbox.rs`, `pool.rs`, `ffi.rs`.
6. Create `crates/nsl-codegen/src/agent.rs`: agent struct layout, method compilation, `@pipeline_agent` lowering, `serve`-block integration, `@auto_device_transfer` inserts with compile-time size reporting.
7. Register new FFI symbols in `crates/nsl-codegen/src/builtins.rs`.
8. Close the `nsl run` `--linear-types` gap ([crates/nsl-cli/src/main.rs:1518](crates/nsl-cli/src/main.rs#L1518)) so pipeline examples are runnable via `nsl run` directly.
9. Write the E2E examples enumerated in Section 8.5; wire them into the test suite.
10. Add performance benchmarks from Section 8.6.

---

## Success Criteria

- All positive unit/integration/E2E tests listed in Section 8 pass.
- All negative tests produce the exact error codes specified in Section 6.
- Deterministic-replay test passes across 100 consecutive runs with identical output traces.
- Pool with `pool_size=4` serves 4 concurrent requests in parallel; reset-failure injection causes the offending instance to be removed from the pool without corrupting others.
- Same-device zero-copy: KV-cache transfer between agents on the same GPU shows no `memcpy` in a CUDA trace.
- `--linear-types` flag requirement produces E0610 at the `agent` declaration site (localized diagnostic, not cascading errors in non-agent code).
- No regressions in the existing test suite.

---

## Out of Scope

- All items in Section 9.
- Formal verification of agent isolation soundness (empirically tested via M38/M51 composition, not formally proven).
- Agent persistence or checkpointing of live agent state.
- Agent-level fault tolerance (pipeline-level failures propagate; M58 covers fault tolerance at the training level).
- Priority-based request scheduling (FIFO lease acquisition in v1).
- Agent introspection/debugger UIs.
- The predecessor spec's `send()` / `recv()` user-visible API is explicitly *not* in v1 — port communication is exclusively via method-call syntax under `@pipeline_agent`. If users need lower-level send/recv, that's a v2 language feature discussion.
