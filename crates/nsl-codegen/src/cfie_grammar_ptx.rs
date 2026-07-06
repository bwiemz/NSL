//! CFIE Feature 6 (audit gap G11): grammar DFA lowered to an
//! initialized PTX module-scope constant.
//!
//! The paper's claim: "The DFA transition table is embedded as a
//! compile-time constant in the kernel -- no runtime FSM interpreter."
//! What the fused-sample kernel actually reads is not the full u32
//! transition table but a valid-token BITSET derived from it:
//! `[num_states]` rows of `ceil(vocab / 8)` bytes, where bit
//! `(state, token)` is 1 iff the DFA has a live (non-REJECT)
//! transition from `state` on `token` -- the same predicate the
//! runtime's `nsl_cfie_grammar_transition` host simulator answers with
//! `!= -1`.  Bit order matches the sampler's mask hook in
//! `cfie_sample_ptx`: byte `state * row_bytes + token / 8`, bit
//! `token & 7` (LSB first).
//!
//! Storage class is `.global`, NOT `.const`: the paper's example DFA
//! (47 states x 49152 tokens) is ~288 KB of mask data, far over the
//! 64 KB `.const` bank.  The initializer bakes the DATA into the
//! module image at compile time; the launch path (lands with G16's
//! decode loop) resolves the device address of
//! [`MASK_GLOBAL_NAME`] via `cuModuleGetGlobal` and binds it to the
//! sampler's `grammar_mask_ptr` param.  Until then the sampler's
//! runtime null-pointer guard keeps the hook inert.

use crate::cfie_grammar::CompiledDfa;
use std::fmt::Write;

/// Module-scope symbol the decode loop binds to `grammar_mask_ptr`.
pub const MASK_GLOBAL_NAME: &str = "nsl_cfie_grammar_mask";

/// Initializer bytes per emitted line: 24 worst-case three-digit bytes
/// occupy 4 (indent) + 24*4 (digits + comma) + 23 (spaces) = 123 ASCII
/// columns, under the 132-column line invariant the structural test
/// asserts.  (32/line breaks it at 163 columns on dense masks.)
const BYTES_PER_LINE: usize = 24;

/// Bytes per DFA-state row: one bit per vocab token, byte-padded.
pub fn mask_row_bytes(dfa: &CompiledDfa) -> usize {
    (dfa.vocab_size as usize).div_ceil(8)
}

/// Total mask size: `num_states * ceil(vocab / 8)` bytes.
pub fn mask_len(dfa: &CompiledDfa) -> usize {
    dfa.num_states as usize * mask_row_bytes(dfa)
}

/// Host-side mask bytes (the exact bytes the PTX initializer bakes).
/// Bit `(state, token)` = 1 iff `dfa.transition(state, token)` is not
/// the REJECT sentinel; byte `state * row_bytes + token / 8`, bit
/// `token & 7` -- LSB-first, mirroring `cfie_sample_ptx`'s hook.
pub fn mask_bytes(dfa: &CompiledDfa) -> Vec<u8> {
    let row = mask_row_bytes(dfa);
    let mut bytes = vec![0u8; mask_len(dfa)];
    for s in 0..dfa.num_states {
        for t in 0..dfa.vocab_size {
            if dfa.is_valid(s, t) {
                bytes[s as usize * row + (t as usize >> 3)] |= 1 << (t & 7);
            }
        }
    }
    bytes
}

/// Emit the initialized module-scope PTX fragment.  Concatenated into
/// the sampler's module ahead of the kernel entry; the fragment starts
/// directly with the `.global` directive so callers can splice it
/// after their own module header.
pub fn emit_mask_global(dfa: &CompiledDfa) -> String {
    assert!(
        dfa.num_states >= 1 && dfa.vocab_size >= 1,
        "grammar mask requires a non-degenerate DFA (num_states and vocab_size >= 1)"
    );
    let bytes = mask_bytes(dfa);
    let mut p = String::with_capacity(bytes.len() * 4 + 256);
    let w = &mut p;
    write!(
        w,
        ".global .align 1 .b8 {}[{}] = {{",
        MASK_GLOBAL_NAME,
        bytes.len()
    )
    .unwrap();
    for (i, b) in bytes.iter().enumerate() {
        if i % BYTES_PER_LINE == 0 {
            w.push_str("\n    ");
        } else {
            w.push(' ');
        }
        write!(w, "{}", b).unwrap();
        if i + 1 != bytes.len() {
            w.push(',');
        }
    }
    w.push_str("\n};\n");
    // Baked-constants trailer (decode-attention house style; kept after
    // the directive so the fragment still starts with `.global`).
    writeln!(
        w,
        "// {}: {} states x {} mask bytes/row (1 bit/token, LSB first), density {:.4}",
        MASK_GLOBAL_NAME,
        dfa.num_states,
        mask_row_bytes(dfa),
        dfa.density()
    )
    .unwrap();
    p
}

/// Splice the initialized mask fragment into a sampler module (CFIE
/// Cycle 6): the fragment is inserted on its own lines immediately
/// after the `.address_size` directive — the last line of the module
/// header, before any `.shared` / `.visible .entry` declarations — so
/// the loaded module carries the mask data and the host can resolve
/// [`MASK_GLOBAL_NAME`] with `cuModuleGetGlobal` at engine finalize.
///
/// Panics when `sampler_ptx` has no `.address_size` line: splicing
/// module-scope data into a headerless fragment would produce invalid
/// PTX, and every NSL emitter writes the three-line header.
pub fn splice_mask_into_module(sampler_ptx: &str, mask_fragment: &str) -> String {
    assert!(
        sampler_ptx
            .lines()
            .any(|l| l.trim_start().starts_with(".address_size")),
        "sampler module has no .address_size directive to splice after"
    );
    let mut out = String::with_capacity(sampler_ptx.len() + mask_fragment.len() + 1);
    let mut spliced = false;
    for line in sampler_ptx.lines() {
        out.push_str(line);
        out.push('\n');
        if !spliced && line.trim_start().starts_with(".address_size") {
            out.push_str(mask_fragment);
            if !mask_fragment.ends_with('\n') {
                out.push('\n');
            }
            spliced = true;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_grammar::{compile, GrammarSpec};

    fn sequence_dfa(tokens: &[u32], vocab: u32) -> CompiledDfa {
        compile(&GrammarSpec::sequence(tokens, vocab)).unwrap()
    }

    #[test]
    fn bit_set_exactly_where_transition_table_accepts() {
        // Round-trip against the runtime FFI's semantics: the runtime's
        // nsl_cfie_grammar_transition returns -1 exactly when the table
        // cell is the REJECT sentinel -- the same predicate is_valid()
        // answers here.
        let dfa = sequence_dfa(&[5, 7, 3], 10);
        let bytes = mask_bytes(&dfa);
        let row = mask_row_bytes(&dfa);
        for s in 0..dfa.num_states {
            for t in 0..dfa.vocab_size {
                let bit = (bytes[s as usize * row + (t as usize >> 3)] >> (t & 7)) & 1;
                assert_eq!(
                    bit == 1,
                    dfa.is_valid(s, t),
                    "mask bit (state {s}, token {t}) must mirror the transition table"
                );
            }
        }
    }

    #[test]
    fn popcount_matches_dfa_density() {
        let dfa = sequence_dfa(&[1, 2, 3], 1000);
        let bytes = mask_bytes(&dfa);
        let set: u32 = bytes.iter().map(|b| b.count_ones()).sum();
        assert_eq!(set, dfa.live_transitions);
        let total = (dfa.num_states as f64) * (dfa.vocab_size as f64);
        assert!((set as f64 / total - dfa.density()).abs() < 1e-12);
    }

    #[test]
    fn bit_order_matches_sampler_hook() {
        // cfie_sample_ptx's CPU-reference test allows token 5 with mask
        // byte 0b0010_0000 -- the emitter must produce the same byte.
        let dfa = sequence_dfa(&[5], 8);
        let bytes = mask_bytes(&dfa);
        assert_eq!(mask_row_bytes(&dfa), 1);
        assert_eq!(bytes, vec![0b0010_0000, 0]);
    }

    #[test]
    fn mask_len_is_states_times_padded_row() {
        let dfa = sequence_dfa(&[5, 7, 3], 10);
        // 4 states x ceil(10 / 8) = 8 bytes.
        assert_eq!(mask_row_bytes(&dfa), 2);
        assert_eq!(mask_len(&dfa), 8);
        assert_eq!(mask_bytes(&dfa).len(), 8);
        // Paper example: 47 states x 49152 tokens -> 47 * 6144 bytes.
        let paper = sequence_dfa(&(0..46).collect::<Vec<_>>(), 49_152);
        assert_eq!(paper.num_states, 47);
        assert_eq!(mask_len(&paper), 47 * 6144);
    }

    #[test]
    fn fragment_is_structurally_valid_ptx() {
        let dfa = sequence_dfa(&[5, 7, 3], 100);
        let frag = emit_mask_global(&dfa);
        assert!(
            frag.starts_with(".global .align 1 .b8 nsl_cfie_grammar_mask["),
            "fragment must start with the .global directive"
        );
        // Declared byte count == mask_len.
        let open = frag.find('[').unwrap();
        let close = frag.find(']').unwrap();
        let declared: usize = frag[open + 1..close].parse().unwrap();
        assert_eq!(declared, mask_len(&dfa));
        // Initializer entry count == declared size.
        let init_open = frag.find('{').unwrap();
        let init_close = frag.find('}').unwrap();
        let entries = frag[init_open + 1..init_close]
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .count();
        assert_eq!(entries, declared);
        assert!(frag.contains("};"));
        assert!(
            frag.bytes().all(|b| b < 128),
            "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
        );
        assert!(
            frag.lines().all(|l| l.len() <= 132),
            "initializer lines must stay wrapped"
        );
    }

    #[test]
    fn fragment_bytes_round_trip_mask_bytes() {
        let dfa = sequence_dfa(&[0, 3, 6], 17);
        let frag = emit_mask_global(&dfa);
        let init_open = frag.find('{').unwrap();
        let init_close = frag.find('}').unwrap();
        let parsed: Vec<u8> = frag[init_open + 1..init_close]
            .split(',')
            .map(|s| s.trim().parse::<u8>().unwrap())
            .collect();
        assert_eq!(parsed, mask_bytes(&dfa));
    }

    #[test]
    #[should_panic(expected = "non-degenerate")]
    fn degenerate_dfa_panics() {
        let dfa = CompiledDfa {
            num_states: 0,
            vocab_size: 0,
            transitions: vec![],
            is_accept: vec![],
            start_state: 0,
            live_transitions: 0,
        };
        let _ = emit_mask_global(&dfa);
    }

    // ── Cycle 6: splice into a sampler module ────────────────────────

    #[test]
    fn splice_inserts_fragment_after_address_size_before_entry() {
        let dfa = sequence_dfa(&[1, 2], 64);
        let frag = emit_mask_global(&dfa);
        let module = ".version 8.4\n.target sm_90\n.address_size 64\n\n\
                      .shared .align 4 .b8 smem[16];\n\n\
                      .visible .entry k()\n{\n    ret;\n}\n";
        let out = splice_mask_into_module(module, &frag);
        let addr = out.find(".address_size 64").unwrap();
        let mask = out.find("nsl_cfie_grammar_mask").unwrap();
        let entry = out.find(".visible .entry").unwrap();
        assert!(addr < mask && mask < entry, "mask must sit in module scope");
        // The fragment starts on its own line directly after the header.
        let after_addr = &out[addr..];
        let line_end = after_addr.find('\n').unwrap();
        assert!(after_addr[line_end + 1..]
            .starts_with(".global .align 1 .b8 nsl_cfie_grammar_mask["));
        // Nothing from the original module is lost or reordered.
        assert!(out.contains(".shared .align 4 .b8 smem[16];"));
        assert!(out.ends_with("{\n    ret;\n}\n"));
        assert!(out.bytes().all(|b| b < 128), "PTX must stay ASCII-only");
    }

    #[test]
    fn splice_is_pure_concatenation_of_module_and_fragment() {
        // Both inputs newline-terminated => output length is exactly
        // the sum (no dropped or duplicated lines).
        let dfa = sequence_dfa(&[3], 32);
        let frag = emit_mask_global(&dfa);
        let module = ".version 7.0\n.target sm_80\n.address_size 64\n\
                      .visible .entry k()\n{\n    ret;\n}\n";
        let out = splice_mask_into_module(module, &frag);
        assert_eq!(out.len(), module.len() + frag.len());
    }

    #[test]
    #[should_panic(expected = ".address_size")]
    fn splice_refuses_module_without_address_size() {
        let dfa = sequence_dfa(&[1], 8);
        let _ = splice_mask_into_module(".version 8.4\n.target sm_90\n", &emit_mask_global(&dfa));
    }

    // ── ptxas validation (skips silently when no validator present) ──

    #[test]
    fn ptxas_validates_spliced_sampler_module() {
        use crate::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams};
        use crate::cfie_sample_ptx::{emit as emit_sampler, FusedSampleKernelConfig};

        let dfa = sequence_dfa(&[5, 7, 3], 256);
        let params = SamplingParams {
            grammar_masked: true,
            top_k: 8,
            ..Default::default()
        };
        let shape = LmHeadShape {
            d_model: 64,
            vocab_size: 256,
            vocab_tile: 128,
            dtype_bytes: 2,
        };
        let program = emit_program(params, shape);
        let cfg = FusedSampleKernelConfig {
            d_model: 64,
            vocab_size: 256,
            vocab_tile: 128,
            top_k: 8,
            sm_version: 80,
            grammar_states: dfa.num_states,
        };
        let (sampler, _) = emit_sampler(&program, &cfg);
        let spliced = splice_mask_into_module(&sampler, &emit_mask_global(&dfa));
        assert!(spliced.contains("nsl_cfie_grammar_mask["));
        match crate::ptxas_validation::validate_ptx(&spliced) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                eprintln!(
                    "[skip] cfie grammar-splice ptxas validation - no validator: {msg}"
                );
            }
            Err(msg) => panic!(
                "spliced sampler module rejected by ptxas:\n{msg}\n\nEmitted PTX:\n{spliced}"
            ),
        }
    }
}
