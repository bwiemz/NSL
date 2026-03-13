# NeuralScript Examples

## Getting Started
- [hello.nsl](hello.nsl) -- Hello world with functions and loops
- [features.nsl](features.nsl) -- Language feature showcase

## Tensor Operations
- [m9_tensors.nsl](m9_tensors.nsl) -- Tensor creation and arithmetic
- [m10_shape_check.nsl](m10_shape_check.nsl) -- Compile-time shape verification
- [m10_symbolic_dims.nsl](m10_symbolic_dims.nsl) -- Named/symbolic dimensions

## Building Models
- [m11_model_basic.nsl](m11_model_basic.nsl) -- Simple model definition
- [m11_model_compose.nsl](m11_model_compose.nsl) -- Model composition
- [m11_model_tensor.nsl](m11_model_tensor.nsl) -- Model with tensor fields
- [m18_transformer.nsl](m18_transformer.nsl) -- Multi-layer transformer

## Automatic Differentiation
- [m12_grad_basic.nsl](m12_grad_basic.nsl) -- Basic gradient computation
- [m12_grad_matmul.nsl](m12_grad_matmul.nsl) -- Gradients through matmul
- [m12_grad_model.nsl](m12_grad_model.nsl) -- Model parameter gradients
- [m12_no_grad.nsl](m12_no_grad.nsl) -- @no_grad decorator

## Training
- [m14_sgd_basic.nsl](m14_sgd_basic.nsl) -- SGD optimizer training loop
- [m14_adam_scheduler.nsl](m14_adam_scheduler.nsl) -- Adam with learning rate scheduler
- [m15_tiny_lm.nsl](m15_tiny_lm.nsl) -- Complete tiny language model pipeline

## Standard Library
- [m13_stdlib_import.nsl](m13_stdlib_import.nsl) -- Importing stdlib modules
- [m15_nn_stdlib_test.nsl](m15_nn_stdlib_test.nsl) -- Neural network layers
- [m15_activations_test.nsl](m15_activations_test.nsl) -- Activation functions
- [m15_layernorm_test.nsl](m15_layernorm_test.nsl) -- LayerNorm
- [m15_embedding_test.nsl](m15_embedding_test.nsl) -- Embedding layer
- [m15_tokenizer_test.nsl](m15_tokenizer_test.nsl) -- Tokenization

## Quantization
- [m16_quantize.nsl](m16_quantize.nsl) -- INT8 weight quantization

## Interop
- [m18b_interop.nsl](m18b_interop.nsl) -- Safetensors + ONNX export

## Full Pipelines
- [gpt2.nsl](gpt2.nsl) -- GPT-2 model definition and training
- [codeforge_nano.nsl](codeforge_nano.nsl) -- Small code generation model
