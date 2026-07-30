//! Member/subscript-assign statements must adopt owned RHS temporaries
//! and drain the statement's tensor temporaries (2026-07-29, PR #433
//! review LOW-2).
//!
//! `compile_assign`'s Ident arm ends with the ownership transfer +
//! `free_tensor_temporaries` pair, but the Subscript arms (dict/list
//! set, tensor multi-dim set) and every MemberAccess store path had
//! NEITHER:
//!
//!   - The stored value: `d["k"] = t * 2.0` left the owned temp in
//!     `tensor_temporaries`, so the NEXT statement's sweep freed it
//!     while the container still held the raw handle — dict reads clone
//!     a freed tensor (bad-magic abort), list reads hand out the
//!     dangling pointer itself. The fix consumes ownership of the
//!     stored value (the container's reference outlives the statement;
//!     per the dict_lifetime.rs borrow-store convention nothing retains
//!     — an Owned temp transfers, everything else stays untouched).
//!   - Sub-expression temporaries: `d["k"] = t * 2.0 + 1.0` left the
//!     inner `t * 2.0` stranded past the statement. Harmless until a
//!     region that frees the temporaries list WITHOUT draining it runs
//!     twice — a train-block step loop frees the straddler at step 1
//!     and again at step 2: double free (the review's scenario). The
//!     fix drains at every assign exit, exactly like the Ident arm.
//!
//! CPU-only on purpose: the statement lowering is device-independent
//! and these gates run in the plain workspace suite.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_src(src: &str, tag: &str) -> String {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_atd_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "[{tag}] run failed (pre-fix failure mode is a bad-magic abort or \
         use-after-free):\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("DONE"), "[{tag}] incomplete:\n{stdout}");
    std::fs::remove_dir_all(&tmp).ok();
    stdout
}

fn printed_value(stdout: &str) -> String {
    stdout
        .lines()
        .take_while(|l| *l != "DONE")
        .last()
        .unwrap_or("")
        .to_string()
}

/// Dict store of an owned temp: the next statement's sweep freed it
/// pre-fix; the read then cloned a freed tensor.
#[test]
fn dict_store_of_owned_temp_survives_later_sweeps() {
    let src = r#"
let t = ones([4])
let d = {"a": t}
d["b"] = t * 2.0
let q = ones([2])
let y = d["b"]
print(sum(y).item())
print("DONE")
"#;
    let stdout = run_src(src, "dict");
    assert_eq!(printed_value(&stdout), "8", "dict-stored temp died:\n{stdout}");
}

/// List store of an owned temp: list reads return the RAW slot value
/// (no clone), so pre-fix the read handed out the dangling pointer.
#[test]
fn list_store_of_owned_temp_survives_later_sweeps() {
    let src = r#"
let t = ones([4])
let l = [t]
l[0] = t * 3.0
let q = ones([2])
let y = l[0]
print(sum(y).item())
print("DONE")
"#;
    let stdout = run_src(src, "list");
    assert_eq!(printed_value(&stdout), "12", "list-stored temp died:\n{stdout}");
}

/// Struct-field store of an owned temp — the MemberAccess path had an
/// early `return Ok(())` skipping both transfer and drain.
#[test]
fn struct_field_store_of_owned_temp_survives_later_sweeps() {
    let src = r#"
struct Holder:
    w: Tensor

let t = ones([4])
let s = Holder(w = t)
s.w = t * 5.0
let q = ones([2])
print(sum(s.w).item())
print("DONE")
"#;
    let stdout = run_src(src, "structf");
    assert_eq!(
        printed_value(&stdout),
        "20",
        "struct-stored temp died:\n{stdout}"
    );
}

/// The review's train scenario: a SUB-EXPRESSION temp (`t * 2.0` inside
/// `t * 2.0 + 1.0` — not the stored value, which ownership transfer
/// already removes) straddled into the train step loop, which frees the
/// temporaries list without draining it — freed at step 1, freed again
/// at step 2. The statement-exit drain removes the straddler entirely.
#[test]
fn subexpr_temp_before_train_block_does_not_double_free() {
    let src = r#"
from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

let t = ones([4])
let d = {"a": t}
d["b"] = t * 2.0 + 1.0

train(model = m, epochs = 2):
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

let z = d["b"]
print(sum(z).item())
print("DONE")
"#;
    let stdout = run_src(src, "train");
    assert_eq!(
        printed_value(&stdout),
        "12",
        "straddler reached the train step loop:\n{stdout}"
    );
}

/// Borrow-store regression pin: storing a VARIABLE stays untouched (no
/// retain — per the dict_lifetime.rs convention nothing ever releases a
/// dict-stored borrow) and the variable itself stays fully usable.
#[test]
fn dict_borrowed_store_and_var_still_usable() {
    let src = r#"
let t = ones([4])
let d = {"a": t}
d["b"] = t
let q = ones([2])
let y = d["b"]
print(sum(y).item() + sum(t).item())
print("DONE")
"#;
    let stdout = run_src(src, "borrow");
    assert_eq!(printed_value(&stdout), "8", "borrowed store broke:\n{stdout}");
}

/// Plain int container writes are untouched by the tensor machinery.
#[test]
fn int_list_set_unchanged() {
    let src = r#"
let l = [1, 2, 3]
l[0] = 9
print(l[0] + l[1])
print("DONE")
"#;
    let stdout = run_src(src, "intlist");
    assert_eq!(printed_value(&stdout), "11", "int list set broke:\n{stdout}");
}
