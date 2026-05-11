# BitNet b1.58 packed-byte layout

Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §2.1.

## Trit-within-byte ordering — HIGH-BITS-FIRST

Verified against Microsoft's bitnet.cpp `quantize_i2_s` / `ggml_vec_dot_i2_i8_s_1x1`
functions during pre-implementation step PI.1 (per spec §10.1).

**Source citations:**
- Pack (AVX): https://github.com/microsoft/BitNet/blob/01eb415772c342d9f20dc42772f1583ae1e5b102/src/ggml-bitnet-mad.cpp#L83-L86
- Unpack (AVX): https://github.com/microsoft/BitNet/blob/01eb415772c342d9f20dc42772f1583ae1e5b102/src/ggml-bitnet-mad.cpp#L223-L234

**Pack formula:** `byte = (trit[0] << 6) | (trit[1] << 4) | (trit[2] << 2) | trit[3]`

| Trit index | Bit positions | 2-bit extraction |
|-----------|---------------|------------------|
| trit[0]   | bits [7:6]    | `(byte >> 6) & 0b11` |
| trit[1]   | bits [5:4]    | `(byte >> 4) & 0b11` |
| trit[2]   | bits [3:2]    | `(byte >> 2) & 0b11` |
| trit[3]   | bits [1:0]    | `byte & 0b11`        |

## Trit-value encoding

2-bit value → trit value (per spec §2.1; verified at
https://github.com/microsoft/BitNet/blob/01eb415772c342d9f20dc42772f1583ae1e5b102/src/ggml-bitnet-mad.cpp#L67-L71):

| Bits  | Trit value |
|-------|------------|
| 0b00  | -1         |
| 0b01  | 0          |
| 0b10  | +1         |
| 0b11  | (unused / reserved) |

`0b11` is an invalid encoding. `try_unpack_byte` rejects any input
containing a `0b11` 2-bit field; the unchecked `unpack_byte` is for
hot paths where the producer (NSL's own packer) guarantees validity.

## Element-stride layout

NSL packs **4 consecutive trits per byte** — simpler than bitnet.cpp's
I2_S stride-32 layout. For interoperability with bitnet.cpp GGUF files,
the loader (Task 9) may need a separate `gguf_to_nsl_repack` step that
reshapes the stride-32 layout into the consecutive layout. The intra-byte
ordering (high-bits-first) is identical in both layouts.

## Canonical test vector

Trits `[-1, 0, +1, +1]` pack to byte `0x1A` (binary `0b00_01_10_10`).
The wrong answer (low-bits-first ordering) would be `0x58`. This is the
hand-constructed-byte invariant test in `tests/bitnet_packed_repr.rs`.
