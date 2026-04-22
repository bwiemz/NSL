# WRGA B.3.2 trigger - FUSED forward vs UNFUSED backward

Captures the trigger condition from `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md`: `backward_time > 2.5 x forward_time` at seq=2048, rank=16, Llama-3-8B-proxy dims. Forward fires the B.3.1 fused GatedLoRA kernel; backward goes through source-AD's adapter-triple unfused path (this is what B.3.2 would replace).

Warmup iters: 3 discarded (via N=3 baseline for slope extraction). Timed iters: 10 per measurement. Wall-clock via `Instant` around subprocess; fixed setup/compile cost cancels in the slope.

Prerequisite: 2026-04-19 K-loop rewrite (commit converting the emitter's Rust-side for-loop to a PTX runtime loop). Without it, fused forward produces 20 MB PTX at k=4096 and ptxas rejects it - the fused path silently CPU-falls-back and this measurement degenerates to the unfused side-measurement.

## prescribed_b32_r16 (batch=32, seq=2048, dim=4096, rank=16, alpha=32)

- fused forward: 3597.12ms/iter
- fused fwd + unfused bwd: 220387.80ms/iter
- backward-only: 216790.68ms/iter
- ratio: 61.268x
- **Verdict:** **TRIGGER FIRES (ratio 61.268x > 2.5x) - B.3.2 should be scheduled**

## smaller_batch_b8_r16 (batch=8, seq=2048, dim=4096, rank=16, alpha=32)

- fused forward: 1497.03ms/iter
- fused fwd + unfused bwd: 51084.59ms/iter
- backward-only: 49587.56ms/iter
- ratio: 34.124x
- **Verdict:** **TRIGGER FIRES (ratio 34.124x > 2.5x) - B.3.2 should be scheduled**

