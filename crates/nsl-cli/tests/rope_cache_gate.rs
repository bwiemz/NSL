//! `RotaryEmbedding` cached-table gates.
//!
//! `RotaryEmbedding` used to rebuild `arange -> outer product -> cat ->
//! cos/sin` on every call — twice per attention layer (Q and K), so 32 times
//! per forward for a 16-layer model — while the `max_seq_len` constructor
//! argument sat unused. It now precomputes cos/sin over the full
//! `[max_seq_len, dim]` grid at construction and `forward` gathers the first
//! `seq_len` rows.
//!
//! Two properties have to hold, and neither is self-evident:
//!
//! 1. **Bit-exactness.** The cached tables must equal what the old path
//!    computed, for every `seq_len <= max_seq_len`. A cache that is merely
//!    close would silently perturb every model's attention.
//! 2. **Loud refusal past the table.** Before this change RoPE was correct at
//!    ANY sequence length, because the table was built per call. Now the
//!    table has `max_seq_len` rows, so `seq_len > max_seq_len` is a real
//!    constraint — and it must abort, not read past the table. The
//!    host-resident bounds check in `nsl_tensor_embedding_lookup` is what
//!    enforces it; this gate is what stops that check from being deleted as
//!    dead code.
//!
//! Both run through `nsl run`, because the failure mode in (2) is a process
//! abort that cannot be observed from inside a `#[test]`.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn nsl_binary() -> PathBuf {
    // $CARGO_TARGET_DIR first, repo-local target/ second — the certification
    // lane and CI redirect the target dir, and a probe that only knows
    // repo-local target/ panics there while the built binary sits in the
    // redirected dir (weight_decay_groups_gate hit exactly that in the
    // 2026-08-24 lane run; this file carried the same latent probe).
    let mut roots = Vec::new();
    if let Ok(t) = std::env::var("CARGO_TARGET_DIR") {
        roots.push(PathBuf::from(t));
    }
    roots.push(repo_root().join("target"));
    for root in &roots {
        for profile in ["release", "debug"] {
            let mut p = root.join(profile).join("nsl");
            if cfg!(windows) {
                p.set_extension("exe");
            }
            if p.exists() {
                return p;
            }
        }
    }
    panic!("no nsl binary under $CARGO_TARGET_DIR or target/ ({{release,debug}}) — build it first");
}

/// Run an NSL source string through `nsl run`, returning (stdout, stderr, ok).
fn run_nsl(name: &str, src: &str, extra_args: &[&str]) -> (String, String, bool) {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_rope_cache_gate_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join(format!("{name}.nsl"));
    std::fs::write(&path, src).expect("write fixture");

    let out = Command::new(nsl_binary())
        .arg("run")
        .args(extra_args)
        .arg(&path)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("run nsl");

    let _ = std::fs::remove_dir_all(&dir);
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.success(),
    )
}

/// The literal pre-change `forward` body, kept here as the reference oracle so
/// the gate does not compare the new implementation against itself.
const RECOMPUTE_REFERENCE: &str = r#"
model RopeRecompute(dim: int, max_seq_len: int, theta: float):
    inv_freq: Tensor = exp(arange(0.0, float(dim), 2.0) * (-1.0 * log(theta) / float(dim)))
    _dim: Tensor = full([1], float(dim))
    _half_dim: Tensor = full([1], float(dim) / 2.0)

    fn forward(self, x: Tensor, seq_len: int) -> Tensor:
        let half_d = int(self._half_dim.item())
        let d = int(self._dim.item())
        let t = arange(0.0, float(seq_len))
        let t2 = t.reshape([seq_len, 1])
        let inv2 = self.inv_freq.reshape([1, half_d])
        let freqs = t2 @ inv2
        let tensors = [freqs, freqs]
        let emb = tensor_cat(tensors, 1)
        let cos_emb = tensor_cos(emb)
        let sin_emb = tensor_sin(emb)
        let cos_4d = cos_emb.reshape([1, 1, seq_len, d])
        let sin_4d = sin_emb.reshape([1, 1, seq_len, d])
        return x * cos_4d + rotate_half(x) * sin_4d
"#;

/// (1) The cached tables reproduce the old per-call recompute EXACTLY, across
/// several `seq_len <= max_seq_len` and both thetas the models use.
#[test]
fn cached_tables_are_bit_exact_with_the_old_recompute() {
    let mut body = String::from("from nsl.nn.rope import RotaryEmbedding\n");
    body.push_str(RECOMPUTE_REFERENCE);
    body.push('\n');

    // dim, max_seq_len, theta, seq_len
    let cases: &[(i32, i32, &str, i32)] = &[
        (8, 32, "10000.0", 1),
        (8, 32, "10000.0", 7),
        (8, 32, "10000.0", 32), // full table — the boundary
        (64, 64, "500000.0", 33),
        (128, 40, "500000.0", 40),
    ];

    for (i, (dim, msl, theta, seq)) in cases.iter().enumerate() {
        let n = seq * dim;
        body.push_str(&format!(
            "let cached{i} = RotaryEmbedding({dim}, {msl}, {theta})\n\
             let ref{i} = RopeRecompute({dim}, {msl}, {theta})\n\
             let x{i} = arange(0.0, {n}.0).reshape([1, 1, {seq}, {dim}])\n\
             let a{i} = cached{i}.forward(x{i}, {seq})\n\
             let b{i} = ref{i}.forward(x{i}, {seq})\n\
             let d{i} = (a{i} - b{i}) * (a{i} - b{i})\n\
             print(d{i}.sum().item())\n"
        ));
    }

    let (stdout, stderr, ok) = run_nsl("bitexact", &body, &[]);
    assert!(ok, "nsl run failed:\nstdout:\n{stdout}\nstderr:\n{stderr}");

    let diffs: Vec<f64> = stdout
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert_eq!(
        diffs.len(),
        cases.len(),
        "expected one squared-difference line per case, got {diffs:?}\n\
         stdout:\n{stdout}"
    );
    for (case, diff) in cases.iter().zip(&diffs) {
        assert_eq!(
            *diff, 0.0,
            "cached RoPE differs from the recompute reference for \
             (dim, max_seq_len, theta, seq_len) = {case:?}: sum of squared \
             differences {diff}"
        );
    }
}

/// (1b) Anti-vacuity for the oracle above: the reference must be able to
/// DISAGREE. Two different thetas must produce a nonzero difference —
/// otherwise `cached_tables_are_bit_exact_with_the_old_recompute` would pass
/// against a broken comparison that always reports 0.
///
/// This doubles as the liveness check for the `rope_theta` threading: before
/// the fix, `GroupedQueryAttention` hardcoded theta=10000.0, so a model
/// declaring `ROPE_THETA = 500000.0` got 10000.0 and this difference would be
/// unobservable through the attention layer.
#[test]
fn differing_theta_is_observable() {
    let src = "from nsl.nn.rope import RotaryEmbedding\n\
               let a = RotaryEmbedding(8, 32, 10000.0)\n\
               let b = RotaryEmbedding(8, 32, 500000.0)\n\
               let x = ones([1, 1, 6, 8])\n\
               let ya = a.forward(x, 6)\n\
               let yb = b.forward(x, 6)\n\
               let d = (ya - yb) * (ya - yb)\n\
               print(d.sum().item())\n";

    let (stdout, stderr, ok) = run_nsl("theta", src, &[]);
    assert!(ok, "nsl run failed:\nstdout:\n{stdout}\nstderr:\n{stderr}");
    let diff: f64 = stdout
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .next_back()
        .unwrap_or_else(|| panic!("no numeric output:\n{stdout}"));
    assert!(
        diff > 1e-6,
        "theta=10000 and theta=500000 produced indistinguishable RoPE \
         (squared difference {diff}) — the theta argument is not reaching the \
         frequency table"
    );
}

/// (2) `seq_len > max_seq_len` must ABORT, loudly, rather than gather past the
/// end of the cached table.
#[test]
fn sequence_longer_than_max_seq_len_refuses() {
    let src = "from nsl.nn.rope import RotaryEmbedding\n\
               let r = RotaryEmbedding(8, 16, 10000.0)\n\
               let x = ones([1, 1, 24, 8])\n\
               let y = r.forward(x, 24)\n\
               print(y.sum().item())\n";

    let (stdout, stderr, ok) = run_nsl("overrun", src, &[]);
    assert!(
        !ok,
        "seq_len=24 against max_seq_len=16 SUCCEEDED — it read past the \
         rotary table instead of refusing.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("out of bounds"),
        "expected an out-of-bounds refusal naming the overrun, got:\n\
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

/// (3) The tables must not become trainable parameters. They are derived
/// constants; if they entered the parameter set they would collect gradients,
/// AdamW moments and weight decay, and drift away from cos/sin during
/// training. The `_` prefix is what keeps them out
/// (`is_trainable_param_leaf_name`), and the source-AD summary line is where
/// that is observable.
#[test]
fn cached_tables_are_not_trainable_parameters() {
    let src = "from nsl.nn.gqa import GroupedQueryAttention\n\
               from nsl.nn.losses import cross_entropy\n\
               \n\
               model TinyGqa:\n\
               \x20   embed: Tensor = randn([32, 8])\n\
               \x20   attn: GroupedQueryAttention = GroupedQueryAttention(8, 2, 1, 4, 0.0, 64, 10000.0, 1)\n\
               \x20   w_out: Tensor = randn([8, 32])\n\
               \n\
               \x20   fn forward(self, input_ids: Tensor) -> Tensor:\n\
               \x20       let s = input_ids.shape\n\
               \x20       let batch = s[0]\n\
               \x20       let seq_len = s[1]\n\
               \x20       let flat_ids = input_ids.reshape([batch * seq_len])\n\
               \x20       let emb = embedding_lookup(self.embed, flat_ids)\n\
               \x20       let x = emb.reshape([batch, seq_len, 8])\n\
               \x20       let h = self.attn.forward(x, true)\n\
               \x20       let flat_h = h.reshape([batch * seq_len, 8])\n\
               \x20       let logits = flat_h @ self.w_out\n\
               \x20       return logits.reshape([batch, seq_len, 32])\n\
               \n\
               let m = TinyGqa()\n\
               let input_ids = full([1, 4], 1.0)\n\
               let labels = full([1, 4], 0.0)\n\
               \n\
               train(model = m, epochs = 1):\n\
               \x20   optimizer: SGD(lr = 0.001)\n\
               \x20   step(batch):\n\
               \x20       let logits = m.forward(input_ids)\n\
               \x20       let ls = logits.shape\n\
               \x20       let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])\n\
               \x20       let flat_labels = labels.reshape([ls[0] * ls[1]])\n\
               \x20       let loss = cross_entropy(flat_logits, flat_labels)\n";

    let (stdout, stderr, ok) = run_nsl("params", src, &["--source-ad"]);
    let combined = format!("{stdout}{stderr}");
    assert!(ok, "nsl run failed:\nstdout:\n{stdout}\nstderr:\n{stderr}");

    let summary = combined
        .lines()
        .find(|l| l.contains("source AD gradient summary"))
        .unwrap_or_else(|| {
            panic!("no source-AD gradient summary — did source AD silently fall back to tape?\n{combined}")
        });

    // Every trainable param must be connected: a table that leaked into the
    // parameter set would show up as an unconnected trainable, not as an
    // "ignored config-tensor".
    assert!(
        summary.contains("0 missing-from-forward")
            && summary.contains("0 no-primal")
            && summary.contains("0 no-adjoint"),
        "source-AD summary reports disconnected parameters: {summary}"
    );
    assert!(
        summary.contains("ignored config-tensor"),
        "expected the RoPE tables to be classified as ignored config tensors: {summary}"
    );
}
