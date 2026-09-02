//! Execution fingerprint — the compile-time flag set that decides training
//! arithmetic and compiler mode, carried into the checkpoint sidecar so a
//! resume cannot silently change it.
//!
//! WHY THIS EXISTS. Item 8 made training state resumable: `θ`, the optimizer
//! moments, the data position and every RNG stream are restored, and the
//! loader/seed checks in `checkpoint.rs` refuse a resume that would continue
//! into different data or a different dither stream. None of that covers the
//! **flags the program was compiled with**. `--source-ad` vs the tape,
//! `--deterministic`, `--dtype bf16-sr`, `--disable-fusion` and the fused
//! backward kernels all change the arithmetic that produced the checkpoint,
//! and they live on the command line — exactly where the seed lived before
//! item 8 guarded it. The documented resume workflow is "re-run the recipe
//! unchanged", so a dropped flag is a routine operator slip, and today it
//! resumes without a word.
//!
//! WHAT IS AND IS NOT A REFUSAL. Two classes, because treating them alike
//! would be wrong in both directions:
//!
//!   * **arithmetic** — refuse. Changing these mid-run means the second half
//!     of training is not a continuation of the first, and nothing downstream
//!     can tell.
//!   * **placement** — warn, do not refuse. `--transient-arena`,
//!     `--cuda-graphs`, `--checkpoint-blocks` and `--optim-state-offload`
//!     move bytes without changing the value computed; the arena's own
//!     byte-identity gate (`scripts/arena-parity.sh`) exists to certify
//!     exactly that. The production 1B recipe ships resume x
//!     `--optim-state-offload` as a supported workflow, so refusing it would
//!     break a validated recipe to enforce a property it does not violate.
//!
//! A warning that does not gate is not a guard — which is why the arithmetic
//! class aborts rather than printing. The placement class prints because the
//! change is legitimate and the operator still wants to know it happened.

use std::sync::Mutex;

/// The live program's fingerprint, installed by compiled `main()` before user
/// code runs. Empty when the program predates this call, which is why every
/// consumer treats "empty" as "unknown" rather than as a mismatch.
static EXEC_FINGERPRINT: Mutex<String> = Mutex::new(String::new());

/// Keys whose disagreement means the resumed run computes different numbers
/// than the run that wrote the checkpoint. Kept as an explicit list, not a
/// negation of the placement list, so a newly added key defaults to the
/// SILENT class only by deliberate omission from both.
const ARITHMETIC_KEYS: &[&str] = &[
    "ad", "det", "dtype", "fusion", "fuse_rms", "fuse_wgrad", "zero", "zero_elem", "ws",
    "muon_bns", "muon_resmom", "lmhead",
];

/// Keys that move bytes without changing the value computed.
const PLACEMENT_KEYS: &[&str] = &["arena", "graphs", "ckpt", "offload"];

/// Install the fingerprint. Called from compiled `main()`.
///
/// # Safety
/// `ptr` must point to `len` initialised bytes that stay valid for the call.
/// Codegen passes a `.rodata` string constant, so this holds for the life of
/// the process.
#[no_mangle]
pub unsafe extern "C" fn nsl_set_exec_fingerprint(ptr: *const u8, len: i64) -> i64 {
    if ptr.is_null() || len <= 0 {
        return 0;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let Ok(s) = std::str::from_utf8(bytes) else {
        return 0;
    };
    if let Ok(mut guard) = EXEC_FINGERPRINT.lock() {
        *guard = s.to_string();
    }
    0
}

/// The live fingerprint, or empty when the program never installed one.
pub fn exec_fingerprint() -> String {
    EXEC_FINGERPRINT.lock().map(|g| g.clone()).unwrap_or_default()
}

/// Split a `k=v,k=v` fingerprint into pairs. Unparseable segments are dropped
/// rather than guessed at: a malformed fingerprint must not manufacture a
/// mismatch on a key that does not exist.
fn parse(fp: &str) -> Vec<(&str, &str)> {
    fp.split(',')
        .filter_map(|seg| seg.split_once('='))
        .collect()
}

/// One disagreeing key.
pub struct FieldDiff {
    pub key: String,
    pub saved: String,
    pub live: String,
}

/// Keys present in both fingerprints with different values, plus keys present
/// in exactly one (reported with `<absent>`), restricted to `class`.
pub fn diff(saved: &str, live: &str, class: &[&str]) -> Vec<FieldDiff> {
    let (s, l) = (parse(saved), parse(live));
    let mut out = Vec::new();
    for key in class {
        let sv = s.iter().find(|(k, _)| k == key).map(|(_, v)| *v);
        let lv = l.iter().find(|(k, _)| k == key).map(|(_, v)| *v);
        // Absent on BOTH sides is not a difference — it is a key this pair of
        // builds never emitted, which is the back-compatible case.
        if sv.is_none() && lv.is_none() {
            continue;
        }
        if sv != lv {
            out.push(FieldDiff {
                key: (*key).to_string(),
                saved: sv.unwrap_or("<absent>").to_string(),
                live: lv.unwrap_or("<absent>").to_string(),
            });
        }
    }
    out
}

/// Arithmetic-class disagreements — the refusal set.
pub fn arithmetic_diff(saved: &str, live: &str) -> Vec<FieldDiff> {
    diff(saved, live, ARITHMETIC_KEYS)
}

/// Placement-class disagreements — the warn set.
pub fn placement_diff(saved: &str, live: &str) -> Vec<FieldDiff> {
    diff(saved, live, PLACEMENT_KEYS)
}

/// Render diffs as `key: saved -> live` lines for an operator message.
pub fn render(diffs: &[FieldDiff]) -> String {
    diffs
        .iter()
        .map(|d| format!("    {}: checkpoint {} -> this run {}", d.key, d.saved, d.live))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_fingerprints_have_no_diff() {
        let fp = "ad=source,det=1,dtype=f32";
        assert!(arithmetic_diff(fp, fp).is_empty());
        assert!(placement_diff(fp, fp).is_empty());
    }

    #[test]
    fn ad_mode_flip_is_arithmetic() {
        let d = arithmetic_diff("ad=source,det=1", "ad=tape,det=1");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].key, "ad");
        assert_eq!((d[0].saved.as_str(), d[0].live.as_str()), ("source", "tape"));
    }

    #[test]
    fn offload_flip_is_placement_not_arithmetic() {
        // The shipped production-1B workflow resumes with --optim-state-offload
        // toggled. It must warn, never refuse.
        let (a, b) = ("ad=source,offload=0", "ad=source,offload=1");
        assert!(arithmetic_diff(a, b).is_empty());
        assert_eq!(placement_diff(a, b).len(), 1);
    }

    #[test]
    fn a_key_absent_from_both_sides_is_not_a_difference() {
        // Back-compat: a sidecar written before a key existed must not be
        // read as "that flag changed".
        assert!(arithmetic_diff("ad=source", "ad=source").is_empty());
    }

    #[test]
    fn a_key_present_on_one_side_only_is_reported() {
        let d = arithmetic_diff("ad=source", "ad=source,dtype=bf16-sr");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].saved, "<absent>");
        assert_eq!(d[0].live, "bf16-sr");
    }

    #[test]
    fn malformed_segments_do_not_manufacture_diffs() {
        assert!(arithmetic_diff("ad=source,garbage", "ad=source,other").is_empty());
    }
}
