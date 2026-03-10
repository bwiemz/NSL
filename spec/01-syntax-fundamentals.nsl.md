# NeuralScript Language Specification v0.1

# Section 1 — Syntax & Language Fundamentals

## Design Rationale

NeuralScript (NSL) syntax is designed for immediate familiarity to Python developers while
eliminating Python's pain points: ambiguous scoping, lack of static types, and the GIL.
The syntax is indentation-based with optional type annotations, where the compiler infers
types at compile time. Keywords read like natural English, making ML code self-documenting.

## Formal Grammar (EBNF Subset)

```ebnf
program        ::= (statement NEWLINE)*
statement      ::= declaration | assignment | expression | control_flow | block_stmt
declaration    ::= ('let' | 'const') IDENT (':' type)? '=' expression
assignment     ::= lvalue '=' expression
lvalue         ::= IDENT | IDENT '[' expression ']' | IDENT '.' IDENT
type           ::= primitive_type | tensor_type | function_type | generic_type | union_type
primitive_type ::= 'int' | 'float' | 'bool' | 'str' | 'void' | 'f32' | 'f64'
function_type  ::= '(' type_list? ')' '->' type
generic_type   ::= IDENT '<' type_list '>'
union_type     ::= type '|' type
tensor_type    ::= 'Tensor' '<' shape ',' dtype (',' device)? '>'
shape          ::= '[' dim_list ']'
dim_list       ::= dim (',' dim)*
dim            ::= INT | IDENT | IDENT '=' STRING
dtype          ::= 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'int4' | 'uint8' | 'int32' | 'int64'
device         ::= 'cpu' | 'cuda' | 'metal' | 'rocm' | 'npu' '<' IDENT '>'

control_flow   ::= if_stmt | for_stmt | while_stmt | match_stmt
if_stmt        ::= 'if' expression ':' INDENT block DEDENT ('elif' expression ':' INDENT block DEDENT)* ('else' ':' INDENT block DEDENT)?
for_stmt       ::= 'for' pattern 'in' expression ':' INDENT block DEDENT
while_stmt     ::= 'while' expression ':' INDENT block DEDENT
match_stmt     ::= 'match' expression ':' INDENT (pattern_arm NEWLINE)+ DEDENT
pattern_arm    ::= 'case' pattern (':' 'if' expression)? '=>' expression

block_stmt     ::= fn_def | model_def | train_def | kernel_def
fn_def         ::= 'fn' IDENT '(' param_list? ')' ('->' type)? ':' INDENT block DEDENT
param_list     ::= param (',' param)*
param          ::= IDENT (':' type)? ('=' expression)?

expression     ::= pipe_expr | lambda_expr | binary_expr | unary_expr | call_expr | literal
pipe_expr      ::= expression '|>' expression
lambda_expr    ::= '|' param_list? '|' expression
call_expr      ::= expression '(' arg_list? ')'
binary_expr    ::= expression BINOP expression
BINOP          ::= '+' | '-' | '*' | '/' | '//' | '%' | '**' | '@' | '==' | '!=' | '<' | '>' | '<=' | '>=' | 'and' | 'or'
```

## Lexical Rules

- **Indentation**: 4 spaces per level (tabs are a compile error)
- **Comments**: `#` for line comments, `##` for doc comments
- **String literals**: `"double"`, `'single'`, `"""triple"""` for multiline
- **f-strings**: `f"value is {x}"` with compile-time format checking
- **Newlines**: statement terminators (no semicolons)
- **Line continuation**: trailing `\` or implicit inside `()`, `[]`, `{}`

## Keywords (Reserved)

```
let const fn return if elif else for in while break continue
match case model train grad quant kernel device import from as
true false none and or not is typeof sizeof
pub priv mut ref yield async await
```

## 10 Annotated Syntax Examples

### Example 1: Variable Declaration & Type Inference

```nsl
# 'let' declares mutable variables, 'const' declares immutable ones
# Type inference: compiler determines types from right-hand side

let x = 42                      # inferred as int
let name = "NeuralScript"       # inferred as str
let ratio = 3.14                # inferred as f64
const MAX_SEQ = 2048            # inferred as int, immutable
const PI: f32 = 3.14159         # explicit type annotation

# Multiple assignment via destructuring
let (a, b, c) = (1, 2, 3)      # tuple destructuring
let {width, height} = config    # struct destructuring

# Mutable vs immutable semantics
let counter = 0
counter = counter + 1            # OK: let is mutable
# MAX_SEQ = 4096                 # COMPILE ERROR: const is immutable
```

### Example 2: Function Definition with Type Annotations

```nsl
# Functions use 'fn' keyword. Return type inferred or annotated with '->'
# Parameters can have defaults. All types are checked at compile time.

fn add(a: int, b: int) -> int:
    return a + b

# Type inference on return — compiler infers -> f64
fn sigmoid(x: f64):
    return 1.0 / (1.0 + (-x).exp())

# Generic function — the compiler monomorphizes at call sites
fn identity<T>(value: T) -> T:
    return value

# Default parameters and named arguments
fn create_layer(in_features: int, out_features: int, bias: bool = true) -> Linear:
    return Linear(in_features, out_features, bias=bias)

# Variadic arguments with typed spread
fn concat_all(*tensors: Tensor<[_, _], fp32>) -> Tensor<[_, _], fp32>:
    return nsl.cat(tensors, dim=0)
```

### Example 3: For Loops and Iteration

```nsl
# Range-based for loop (exclusive upper bound)
for i in 0..100:
    print(f"step {i}")

# Inclusive range
for i in 0..=99:
    print(f"step {i}")

# Iterating over collections with index
let names = ["alice", "bob", "charlie"]
for (idx, name) in names.enumerate():
    print(f"{idx}: {name}")

# Iterating over a dataset (lazy streaming)
for batch in dataloader:
    let (inputs, targets) = batch
    # process batch...

# Loop with step
for i in (0..100).step(2):
    print(f"even: {i}")

# Parallel iteration (syntactic sugar for zip)
for (x, y) in zip(inputs, labels):
    let loss = criterion(x, y)
```

### Example 4: While Loop and Break/Continue

```nsl
# Standard while loop
let mut loss = 1.0
let mut epoch = 0
while loss > 0.01:
    loss = train_step(epoch)
    epoch += 1
    if epoch > 1000:
        print("early stopping")
        break

# While-let pattern for option unwrapping
while let Some(batch) = dataloader.next():
    process(batch)
```

### Example 5: If/Elif/Else and Ternary

```nsl
# Standard conditional
if batch_size > 64:
    print("large batch")
elif batch_size > 16:
    print("medium batch")
else:
    print("small batch")

# If as an expression (ternary equivalent)
let lr = if warmup then base_lr * step / warmup_steps else base_lr

# Guard clauses
fn validate_shape(t: Tensor<[_, _], fp32>) -> bool:
    if t.shape[0] == 0:
        return false
    if t.shape[1] > MAX_SEQ:
        return false
    return true
```

### Example 6: Pattern Matching

```nsl
# Pattern matching with 'match' — exhaustiveness checked at compile time

match optimizer:
    case Adam(lr, betas):
        print(f"Adam with lr={lr}")
    case SGD(lr, momentum):
        print(f"SGD with lr={lr}")
    case _:
        print("unknown optimizer")

# Pattern matching on tensor shapes (compile-time shape dispatch)
fn apply_norm(x: Tensor) -> Tensor:
    match x.rank:
        case 2 => LayerNorm(x)         # [batch, features]
        case 3 => RMSNorm(x, dim=-1)   # [batch, seq, features]
        case 4 => BatchNorm2d(x)        # [batch, C, H, W]
        case _ => panic("unsupported rank")

# Pattern matching with guards
match loss.item():
    case v: if v.is_nan()  => panic("NaN loss detected!")
    case v: if v < 0.001   => print("converged")
    case v: if v > 100.0   => print("diverging, reducing lr")
    case v                  => print(f"loss: {v}")
```

### Example 7: Closures and First-Class Functions

```nsl
# Lambda syntax uses |params| body
let double = |x| x * 2
let add = |a, b| a + b

# Multi-line closures
let transform = |batch| {
    let normed = normalize(batch)
    let augmented = augment(normed)
    return augmented
}

# Functions are first-class values
fn apply(f: (int) -> int, x: int) -> int:
    return f(x)

let result = apply(double, 21)   # result = 42

# Closures capture by reference (with move semantics for tensors)
let scale = 0.1
let scale_fn = |x| x * scale    # captures 'scale' from outer scope

# Higher-order functions on collections
let squared = [1, 2, 3, 4].map(|x| x ** 2)        # [1, 4, 9, 16]
let evens = (0..100).filter(|x| x % 2 == 0)        # lazy iterator
let total = losses.reduce(|acc, x| acc + x, 0.0)    # fold/reduce
```

### Example 8: Pipe Operator for ML Chains

```nsl
# The |> operator passes the left value as the first argument to the right function.
# This makes model pipelines read left-to-right instead of inside-out.

let output = input
    |> embedding
    |> positional_encode
    |> transformer_block
    |> layer_norm
    |> lm_head
    |> softmax(dim=-1)

# Without pipe (nested calls — harder to read):
# let output = softmax(lm_head(layer_norm(transformer_block(positional_encode(embedding(input))))), dim=-1)

# Pipe with partial application
let preprocessed = raw_text
    |> tokenizer.encode
    |> pad(max_len=2048)
    |> to_tensor(dtype=int32)
    |> to_device(cuda)
```

### Example 9: Destructuring and Structs

```nsl
# Struct definition
struct TrainConfig:
    lr: f64 = 3e-4
    batch_size: int = 32
    epochs: int = 10
    warmup_ratio: f64 = 0.1

# Struct instantiation and destructuring
let config = TrainConfig(lr=1e-4, epochs=5)
let {lr, batch_size, ..rest} = config    # destructure with rest

# Enum types (algebraic data types)
enum Activation:
    ReLU
    GELU
    SiLU
    Custom(fn: (Tensor) -> Tensor)

# Destructure enum in match
fn apply_activation(act: Activation, x: Tensor) -> Tensor:
    match act:
        case Activation.ReLU       => relu(x)
        case Activation.GELU       => gelu(x)
        case Activation.SiLU       => silu(x)
        case Activation.Custom(f)  => f(x)
```

### Example 10: Modules, Imports, and Visibility

```nsl
# File: models/attention.nsl

# Public items are accessible outside the module
pub fn scaled_dot_product(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    let d_k = q.shape[-1].to_f32()
    let scores = (q @ k.transpose(-2, -1)) / d_k.sqrt()
    let weights = softmax(scores, dim=-1)
    return weights @ v

# Private helper — only accessible within this module
fn _apply_mask(scores: Tensor, mask: Tensor) -> Tensor:
    return scores.masked_fill(mask == 0, -1e9)

# Importing from another module
import nsl.nn.{Linear, LayerNorm, Dropout}
from models.feedforward import MLP

# Re-export pattern
pub import models.attention.{scaled_dot_product as sdp}

# Conditional import (compile-time feature flag)
#[cfg(feature = "flash_attn")]
import nsl.nn.flash_attention as flash_attn
```

## Design Tensions & Tradeoffs

1. **Indentation vs Braces**: We chose indentation for Python familiarity, but this
   makes copy-paste from web sources error-prone. Mitigation: the formatter (`nsl fmt`)
   auto-fixes indentation, and the compiler gives precise column-level errors.

2. **Type Inference vs Explicit Types**: Heavy inference risks "spooky action at a distance"
   where changing one line changes inferred types elsewhere. Mitigation: the compiler always
   infers the most specific type and warns when inference crosses module boundaries. Public
   API functions require explicit return type annotations.

3. **`let` mutability**: Unlike Rust's `let` (immutable by default), NSL's `let` is mutable
   by default for Python ergonomics. `const` provides immutability. This is a conscious
   tradeoff of safety for approachability. We may revisit this in v2.

4. **Pipe operator precedence**: `|>` binds looser than all arithmetic operators but tighter
   than comparison operators. This matches Elixir/F# conventions and reads naturally for
   ML pipelines.
