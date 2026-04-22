<!-- owner: @bwiemz -->

# Syntax Reference

One-page cheat sheet. For complete semantics, every section links to the authoritative [`spec/`](../../spec/) file.

## Bindings — `let`, `const`

```python
let x = 3          # mutable
const PI = 3.14    # immutable
```

Full reference: [`spec/01-syntax-fundamentals.nsl.md`](../../spec/01-syntax-fundamentals.nsl.md).

## Functions — `fn`

```python
fn add(a: int, b: int) -> int:
    return a + b
```

Full reference: [`spec/01-syntax-fundamentals.nsl.md`](../../spec/01-syntax-fundamentals.nsl.md).

## Tensor types

```python
let x: Tensor<[32, 768], fp32, cpu> = randn([32, 768])
let y: Tensor<[batch="B", heads="H", seq="S"], fp8>
```

Full reference: [`spec/02-tensor-type-system.nsl.md`](../../spec/02-tensor-type-system.nsl.md).

## Pipe operator — `|>`

```python
let y = x |> norm |> linear |> gelu
```

Full reference: [`spec/01-syntax-fundamentals.nsl.md`](../../spec/01-syntax-fundamentals.nsl.md).

## `model` blocks

```python
model MLP:
    w: Tensor = randn([512, 1408])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
```

Full reference: [`spec/04-model-definition.nsl.md`](../../spec/04-model-definition.nsl.md).

## `train` blocks

```python
train(model=m, epochs=100):
    optimizer: AdamW(lr=0.001)
    step(batch):
        let loss = mse_loss(m.forward(batch.x), batch.y)
```

Full reference: [`spec/05-training-loop.nsl.md`](../../spec/05-training-loop.nsl.md).

## Autodiff — `grad`, `@no_grad`, `@checkpoint`

```python
let g = grad(loss, params)
@no_grad
fn evaluate(m, x): ...
```

Full reference: [`spec/03-automatic-differentiation.nsl.md`](../../spec/03-automatic-differentiation.nsl.md).

## Quantization — `quant`

```python
let quantized = quant(
    scheme=awq,
    mode=weight_only,
    granularity=per_group(size=128)
):
    model = MyModel
    calibration_data = calib_data
```

Full reference: [`spec/06-quantization.nsl.md`](../../spec/06-quantization.nsl.md).

## GPU kernels — `kernel`

```python
kernel my_kernel(x: Tensor<[N]>, out: Tensor<[N]>):
    let i = thread.x
    out[i] = x[i] * 2
```

Full reference: [`spec/09-hardware-abstraction.nsl.md`](../../spec/09-hardware-abstraction.nsl.md).

## Decorators

| Decorator | Purpose | Spec |
|---|---|---|
| [`@export`](Glossary.md#dec-export) | Mark function as C-ABI exported | [`spec/11-interoperability.nsl.md`](../../spec/11-interoperability.nsl.md) |
| [`@inspect`](Glossary.md#dec-inspect) | Dev-time shape/value inspection | [`Development-Setup`](Development-Setup.md#devtools) |
| [`@flash_attention`](Glossary.md#dec-flash-attention) | Lower to FA-2 kernel | [`Optimization-Passes`](Optimization-Passes.md) |
| [`@cpdt`](Glossary.md#dec-cpdt) | Weight-aware compilation | [`spec/06-quantization.nsl.md`](../../spec/06-quantization.nsl.md) |
| [`@no_grad`](Glossary.md#dec-no-grad) | Disable autodiff tape | [`spec/03-automatic-differentiation.nsl.md`](../../spec/03-automatic-differentiation.nsl.md) |
| [`@checkpoint`](Glossary.md#dec-checkpoint) | Gradient checkpointing | [`spec/03-automatic-differentiation.nsl.md`](../../spec/03-automatic-differentiation.nsl.md) |
| [`@fuse`](Glossary.md#dec-fuse) | Elementwise fusion hint | [`spec/09-hardware-abstraction.nsl.md`](../../spec/09-hardware-abstraction.nsl.md) |
| [`@tie_weights`](Glossary.md#dec-tie-weights) | Share parameters across modules | [`spec/04-model-definition.nsl.md`](../../spec/04-model-definition.nsl.md) |
| [`@shard`](Glossary.md#dec-shard) | Tensor parallelism | M30 milestone spec |
| [`@autotune`](Glossary.md#dec-autotune) | Build-time kernel tuning | M26 milestone spec |

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21. If the crate graph or pass order in this page no longer matches reality, open an issue tagged `docs-rot`.*
