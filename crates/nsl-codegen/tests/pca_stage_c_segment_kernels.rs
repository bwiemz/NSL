//! PCA Stage C — segment-masked plain attention kernel synthesis.
//!
//! Structural gates for the Stage C kernel family:
//!   * the `_segmask` two-phase backward declares the two extra params,
//!     carries the segment predicate at BOTH P-computation sites, and the
//!     monotone q-tile break;
//!   * the plain (segment_masked=false) backward is byte-identical in
//!     structure — no segment artifacts leak into it (SASS baselines and
//!     the 12-arg runtime marshal depend on this);
//!   * the v2 scalar forward with `csha: None` + `segment_masked` (the
//!     fused-forward variant-table config) declares `segment_ids_ptr` and
//!     admits tiles for every head_dim in the table;
//!   * everything is ASCII (the driver JIT rejects non-ASCII PTX even when
//!     offline ptxas tolerates it — see the maxpool em-dash incident);
//!   * ptxas assembles the segment variants cleanly when available.

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::flash_attention::{
    backward_select_blocks, flash_attention_bwd_d_kernel_name,
    flash_attention_bwd_main_kernel_name, synthesize_flash_attention_backward_ptx,
    FlashAttentionBackwardConfig, FlashAttentionConfig, RopeStyle,
};

const FWD_TABLE_HEAD_DIMS: &[i64] = &[32, 64, 128];
const FWD_TILE_CANDIDATES: &[(i64, i64)] = &[(64, 64), (32, 32), (32, 16), (16, 16)];

fn bwd_config(head_dim: i64, causal: bool, segment_masked: bool) -> FlashAttentionBackwardConfig {
    let (block_q, block_kv) = backward_select_blocks(head_dim);
    FlashAttentionBackwardConfig {
        block_q,
        block_kv,
        head_dim,
        causal,
        gpu_sm: 90,
        segment_masked,
    }
}

fn fwd_config(head_dim: i64, bq: i64, bkv: i64, segment_masked: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bkv,
        head_dim,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 90,
        segment_masked,
        csha: None,
        checkpoint: None,
    }
}

fn admitted_fwd_tiles(head_dim: i64, segment_masked: bool) -> Option<(i64, i64)> {
    FWD_TILE_CANDIDATES.iter().copied().find(|&(bq, bkv)| {
        nsl_codegen::flash_attention_v2::smem_layout::validate_scalar_v2_config(
            &fwd_config(head_dim, bq, bkv, segment_masked),
            nsl_codegen::flash_attention_v2::smem_layout::Direction::Forward,
        )
        .is_ok()
    })
}

fn ptx_str(bytes: &[u8]) -> &str {
    // Synthesized PTX is NUL-terminated for cuModuleLoadData.
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).expect("PTX must be valid UTF-8")
}

#[test]
fn segmask_backward_declares_params_predicates_and_break() {
    for &hd in FWD_TABLE_HEAD_DIMS {
        let cfg = bwd_config(hd, true, true);
        let (p1, p2) = synthesize_flash_attention_backward_ptx(&cfg);
        let p2s = ptx_str(&p2);

        // Kernel name carries the variant marker (runtime cache separation).
        let name = flash_attention_bwd_main_kernel_name(&cfg);
        assert!(
            name.ends_with("_segmask"),
            "hd={hd}: main kernel name must end in _segmask, got {name}"
        );
        assert!(p2s.contains(&name), "hd={hd}: PTX must define {name}");

        // The two extra params, in order, at the END of the param list.
        assert!(
            p2s.contains(".param .u64 param_segment_ids"),
            "hd={hd}: missing param_segment_ids"
        );
        assert!(
            p2s.contains(".param .u64 param_heads"),
            "hd={hd}: missing param_heads"
        );

        // Segment predicate present (both bodies emit through the shared
        // helpers, so these markers cover MMA and scalar paths alike).
        assert!(
            p2s.contains("%p_seg"),
            "hd={hd}: segment predicate register missing"
        );
        assert!(
            p2s.contains("ld.global.u16 %r_seg_i")
                && p2s.contains("ld.global.u16 %r_seg_j"),
            "hd={hd}: u16 segment-id loads missing"
        );
        // Belt-and-braces force-zero after the exp (NaN shield for
        // fully-masked rows whose LSE is -inf).
        assert!(
            p2s.contains("@%p_seg mov.f32 %f_p, 0f00000000"),
            "hd={hd}: post-exp force-zero missing"
        );
        // Monotone q-tile break.
        assert!(
            p2s.contains("@%p_seg_break bra BWD_MAIN_Q_LOOP_END"),
            "hd={hd}: q-tile monotone break missing"
        );

        // Phase 1 (D kernel) is mask-independent and shared with the plain
        // variant: same name, no segment artifacts.
        let p1s = ptx_str(&p1);
        let d_name = flash_attention_bwd_d_kernel_name(&cfg);
        assert!(!d_name.contains("segmask"), "D kernel must be shared");
        assert!(p1s.contains(&d_name));
        assert!(!p1s.contains("param_segment_ids"));

        // ASCII only (driver JIT strictness).
        assert!(p2s.is_ascii(), "hd={hd}: non-ASCII byte in segmask PTX");
    }
}

#[test]
fn plain_backward_has_no_segment_artifacts() {
    for &hd in FWD_TABLE_HEAD_DIMS {
        for causal in [false, true] {
            let cfg = bwd_config(hd, causal, false);
            let (_p1, p2) = synthesize_flash_attention_backward_ptx(&cfg);
            let p2s = ptx_str(&p2);
            let name = flash_attention_bwd_main_kernel_name(&cfg);
            assert!(
                !name.contains("segmask"),
                "plain name must not carry the marker"
            );
            for marker in [
                "param_segment_ids",
                "param_heads",
                "%p_seg",
                "%rd_seg",
                "%r_seg",
            ] {
                assert!(
                    !p2s.contains(marker),
                    "hd={hd} causal={causal}: plain PTX leaked segment artifact {marker}"
                );
            }
        }
    }
}

#[test]
fn fwd_table_configs_admit_and_declare_segment_param() {
    for &hd in FWD_TABLE_HEAD_DIMS {
        for segment_masked in [false, true] {
            let (bq, bkv) = admitted_fwd_tiles(hd, segment_masked).unwrap_or_else(|| {
                panic!("hd={hd} segmask={segment_masked}: no admissible tiles — the \
                        fused-forward variant table would silently skip this head_dim")
            });
            let cfg = fwd_config(hd, bq, bkv, segment_masked);
            let ptx = nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2(&cfg);
            let s = ptx_str(&ptx);
            assert!(s.is_ascii(), "hd={hd}: non-ASCII forward PTX");
            let has_seg = s.contains(".param .u64 segment_ids_ptr");
            assert_eq!(
                has_seg, segment_masked,
                "hd={hd}: segment_ids_ptr param presence must track segment_masked"
            );
            // LSE save slot always declared (forward-with-saves channel).
            assert!(
                s.contains("param_logsumexp"),
                "hd={hd}: logsumexp save param missing"
            );
        }
    }
}

// ── ptxas clean-assembly gate (skips when no ptxas on PATH) ─────────────

fn find_ptxas() -> Option<String> {
    if let Ok(p) = std::env::var("PTXAS") {
        if std::path::Path::new(&p).is_file() {
            return Some(p);
        }
    }
    for cand in ["ptxas", "/usr/local/cuda/bin/ptxas", "/opt/cuda/bin/ptxas"] {
        if Command::new(cand).arg("--version").output().is_ok() {
            return Some(cand.into());
        }
    }
    None
}

fn assemble(ptxas: &str, ptx: &[u8], sm: &str, tag: &str) {
    let out = std::env::temp_dir().join(format!("stage_c_{tag}_{sm}.cubin"));
    let mut cmd = Command::new(ptxas)
        .arg(format!("--gpu-name={sm}"))
        .arg("-o")
        .arg(&out)
        .arg("-")
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn ptxas");
    let end = ptx.iter().position(|&b| b == 0).unwrap_or(ptx.len());
    cmd.stdin
        .as_mut()
        .unwrap()
        .write_all(&ptx[..end])
        .unwrap();
    let fin = cmd.wait_with_output().unwrap();
    assert!(
        fin.status.success(),
        "ptxas rejected {tag} for {sm}:\n{}",
        String::from_utf8_lossy(&fin.stderr)
    );
    let _ = std::fs::remove_file(&out);
}

#[test]
fn ptxas_assembles_segment_variants() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("[skip] ptxas not found — segment-variant assembly gate skipped");
        return;
    };
    for &hd in FWD_TABLE_HEAD_DIMS {
        // Backward _segmask pair.
        let (p1, p2) = synthesize_flash_attention_backward_ptx(&bwd_config(hd, true, true));
        assemble(&ptxas, &p1, "sm_90", &format!("bwd_p1_hd{hd}"));
        assemble(&ptxas, &p2, "sm_90", &format!("bwd_p2_segmask_hd{hd}"));

        // Forward segment-masked v2 (csha: None) at table tiles.
        if let Some((bq, bkv)) = admitted_fwd_tiles(hd, true) {
            let ptx = nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2(
                &fwd_config(hd, bq, bkv, true),
            );
            assemble(&ptxas, &ptx, "sm_90", &format!("fwd_segmask_hd{hd}"));
        }
    }
}
