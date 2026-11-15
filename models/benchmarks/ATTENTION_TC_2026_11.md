# Tensor-core attention: speed and agreement (2026-11-14)

NSL's own flash attention cost 673 ms of a Coder-1B optimizer update against
PyTorch's 74 (`NSL_VS_PYTORCH_2026_11.md`): an 8.6 ms backward per call against
FlashAttention-2's 0.77, and a 1.76 ms forward, launched once per batch row,
against 0.30. This records what replacing both with the KIR tensor-core kernels
(`crates/nsl-kir/src/kernels/flash_attn/`, launched by
`crates/nsl-runtime/src/attention_tc.rs`) changed. Every number is measured on this
machine (RTX PRO 4500 Blackwell, sm_120, 200 W cap); nothing is estimated.

## What serves which call

Unpacked `scaled_dot_product_attention` at head_dim 64 and 128 (the production
recipes' path: `--source-ad`, causal, K/V expanded by `nsl.nn.gqa`), forward and
backward, any sequence length, K/V with fewer heads than Q read in place. Packed
batches (segment ids) and other head dims keep the previous kernels.
`NSL_TC_ATTENTION=0` restores the previous kernels everywhere; the comparisons
below use it, so both arms are the same binary.

## Kernel time per call

`tc_attention_kernel_times` (nsl-runtime, release build), causal, mean of 20 calls
after warm-up. FLOPs are the forward's two products (backward kernels: dQ 1.5x,
dK/dV 2x of that). Run-to-run spread on this shared machine is ~5-10%.

| shape | forward | backward (pre-pass + dQ + dK/dV) |
|---|---|---|
| Coder-1B (2 x 32 heads x 2048, d 64), K/V expanded | 0.42-0.45 ms | 1.38-1.42 ms (0.08 + 0.53-0.57 + 0.76-0.80) |
| Coder-1B, grouped K/V (8 heads) | 0.385 ms | 1.45-1.53 ms |
| Coder-500M (2 x 20 x 1024, d 64) | 0.074 ms | 0.27 ms |
| 2 x 16 x 2048, d 128 | 0.47-0.54 ms | 1.43-1.44 ms |

At the Coder-1B shape the kernels these replace took 1.76 ms (forward, two
launches) and 8.6 ms (backward) per call, and PyTorch's FlashAttention-2 takes 0.30
and 0.77 ms (`NSL_VS_PYTORCH_2026_11.md`, Nsight Systems). The tilings were chosen
by sweeping 4 and 8 warps against 16-128 rows per tile (the same diagnostic prints
the sweep); the defaults are in `FwdConfig::new` / `BwdConfig::new`.

## End to end

`tools/perfcheck`, production flags (`--source-ad --checkpoint-blocks
--checkpoint-selective --fuse-rmsnorm-backward --fuse-wgrad-accum`), 24 updates per
run, 15 timed, each arm a fresh process, arm order rotated per round. `nsl_bf16` is
Coder-1B's production posture (`NSL_MATMUL_BF16=1`), `nsl_tf32` the f32/TF32 one
(Coder-500M's recipe precision). `_oldattn` = the same binary under
`NSL_TC_ATTENTION=0`. Ratios are paired within a round, then the median.

| size, posture | tok/s, new kernels (median [min-max]) | tok/s, previous kernels | paired ratio, per round | median |
|---|---|---|---|---|
| Coder-1B, `nsl_bf16` (production) | 7,919 [7,855-7,998] | 6,319 [6,163-6,322] | 1.274 1.253 1.266 | **1.266** |
| Coder-1B, `nsl_tf32` | 5,627 [5,405-5,638] | 4,679 [4,601-4,715] | 1.175 1.205 1.193 | **1.193** |
| Coder-500M, `nsl_bf16` | 12,798 [12,764-13,539] | 11,144 [10,892-11,623] | 1.175 1.145 1.165 | **1.165** |
| Coder-500M, `nsl_tf32` (production) | 9,461 [9,416-9,577] | 8,428 [8,404-8,428] | 1.120 1.122 1.136 | **1.122** |

Seconds per update (median): 1B bf16 2.060 against 2.597, 1B TF32 2.913 against
3.499, 500M bf16 1.278 against 1.476, 500M TF32 1.734 against 1.946. The arms with
the new kernels run at a lower SM clock (1B bf16: 2,336-2,426 against 2,471-2,519
MHz); both hold the card's 200 W cap (199.0-200.0 W mean), so the new arm's work draws
more power per clock.

The previous-kernel arms measured here are slower than the same posture in
`NSL_VS_PYTORCH_2026_11.md` (1B bf16 6,319 against 6,747 tok/s): the machine was
shared with other GPU work queued between runs on this day. The paired ratios cancel
that; the absolute numbers should be compared within this table only.

### Where the time went (Nsight Systems)

`perfcheck.py nsys` of the Coder-1B `nsl_bf16` arm and its `_oldattn` twin, the same
day, summarised by `nsys_summary.py --boundary fase_fused_adamw --fwd-start
nsl_embedding_i32idx --bwd-start lce_chunk_dlogits` (15 steady updates; tracing adds
host time, kernel durations are GPU-timed):

| per optimizer update | new kernels | previous kernels |
|---|---|---|
| attention kernels | **124.5 ms** | 661.7 ms |
| forward, per call | 0.395 ms (`nsl_flash_tc_fwd_d64_causal_w8_n32`, 1 launch) | 1.75 ms (`flash_attn_..._v2_mma`, 2 launches of 0.877) |
| backward, per call | 0.636 ms dQ + 0.798 ms dK/dV + ~0.12 ms pre-pass | 8.44 ms (`flash_attn_bwd_main_c1_q32_kv32_w4`) |
| memcpy/memset | 14.8 ms | 30.1 ms |
| GEMM / everything else (memcpy aside) | 986 / 838 ms | 1,000 / 847 ms |
| wall (traced) | 2,081 ms | 2,661 ms |

PyTorch's FlashAttention-2 on this card and shape: 0.30 ms forward and 0.77 ms
backward per call, 74-81 ms per update (`NSL_VS_PYTORCH_2026_11.md`). What remains
is ~50 ms per update: the forward at 1.3x and the backward at 2.0x FlashAttention-2.

Memory: the allocator, reserved and driver peaks are identical between the arms
(the kernels allocate the same output and workspace tensors).

Loss: on the same tokens and initialisation, the final (24th-update) loss of the two
arms agrees to 7e-5 (1B bf16: 10.974113-10.974118 against 10.974050-10.974082), 4e-6
(1B TF32), 2.3e-4 (500M bf16: 10.981030 against 10.981214-10.981260; 500M runs
dropout) and 2.4e-5 (500M TF32).

## Agreement with PyTorch

`tools/refcheck/grad_check.py --size 50m` with the production flags less
`--fuse-wgrad-accum` (a one-step check has no accumulation): relative error of every
parameter's gradient against PyTorch autograd.

| build | new kernels: median / worst | previous kernels: median / worst |
|---|---|---|
| TF32 (default) | 1.21e-3 / 2.10e-3 | 1.22e-3 / 2.95e-3 |
| `NSL_MATMUL_BF16=1` | 3.93e-3 / 5.62e-3 | 3.92e-3 / 5.74e-3 |

The bf16 build exceeds the check's default tolerance (5e-3) with either kernel set:
its bf16 GEMM operands set that floor.

## Reproduce

```bash
cargo test --release -p nsl-runtime --features cuda --lib attention_tc -- --ignored --nocapture --test-threads=1
P="VENV/bin/python tools/perfcheck/perfcheck.py"
$P build --size 1b --arms nsl_tf32,nsl_bf16 --out OUT     # and --size 500m
$P run --size 1b --arms nsl_bf16,nsl_bf16_oldattn,nsl_tf32,nsl_tf32_oldattn --round R --out OUT
tools/refcheck/grad_check.py --size 50m --tokens train.bin --nsl target/release/nsl -- \
    --source-ad --checkpoint-blocks --checkpoint-selective --fuse-rmsnorm-backward
```
