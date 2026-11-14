//! A named argument that is not in its parameter's position is refused.
//!
//! Calls bind arguments by position; codegen ignores an argument's name. So
//! `sub2(b = 1.0, a = 10.0)` ran as `sub2(a = 1.0, b = 10.0)` and returned
//! -9 where the call as written is 9 — for free functions, methods and model
//! constructors alike, with no diagnostic. Until calls bind by name, the
//! checker refuses the call where it knows the declaration's parameter
//! names: the module's own callables, and those it imports (the module
//! loader hands the checker the imported declarations' parameter names).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const DECLS: &str = "fn sub2(a: float, b: float) -> float:\n\
                     \x20   return a - b\n\
                     \n\
                     model M(scale: float, shift: float):\n\
                     \x20   w: Tensor = full([1], scale - shift)\n\
                     \n\
                     \x20   fn f(self, a: float, b: float) -> float:\n\
                     \x20       return a - b\n\
                     \n\
                     model Other(k: float):\n\
                     \x20   w: Tensor = full([1], k)\n\
                     \n\
                     \x20   fn f(self, x: float, mask: float) -> float:\n\
                     \x20       return x - mask\n\
                     \n";

fn check(tag: &str, body: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("nsl_namedarg_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(&prog, format!("{DECLS}{body}")).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("check")
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl check");
    let _ = std::fs::remove_dir_all(&dir);
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

#[test]
fn named_arguments_in_declaration_order_are_accepted() {
    let (ok, text) = check(
        "ok",
        "print(sub2(10.0, 1.0))\n\
         print(sub2(a = 10.0, b = 1.0))\n\
         print(sub2(10.0, b = 1.0))\n\
         let m = M(scale = 3.0, shift = 1.0)\n\
         print(m.f(a = 10.0, b = 1.0))\n\
         let o = Other(1.0)\n\
         print(o.f(2.0, mask = 1.0))\n\
         print(o.f(x = 2.0, mask = 1.0))\n",
    );
    // `Other.f(x, mask)` and `M.f(a, b)` share a name: each call is judged
    // against the model its receiver is, never against the other's list.
    assert!(ok, "in-order named arguments must check clean:\n{text}");
}

#[test]
fn a_named_argument_out_of_position_is_refused() {
    for (tag, call, needle) in [
        ("function", "print(sub2(b = 1.0, a = 10.0))\n", "argument `b` of `sub2` is passed in position 1, which binds parameter `a`"),
        ("method", "let m = M(3.0, 1.0)\nprint(m.f(b = 1.0, a = 10.0))\n", "argument `b` of `f` is passed in position 1, which binds parameter `a`"),
        ("constructor", "let m = M(shift = 1.0, scale = 3.0)\n", "argument `shift` of `M` is passed in position 1, which binds parameter `scale`"),
        ("skipped", "print(sub2(b = 1.0))\n", "argument `b` of `sub2` is passed in position 1, which binds parameter `a`"),
        ("other_model", "let o = Other(1.0)\nprint(o.f(mask = 1.0, x = 2.0))\n", "argument `mask` of `f` is passed in position 1, which binds parameter `x`"),
    ] {
        let (ok, text) = check(tag, call);
        assert!(!ok, "[{tag}] the call checked clean and would have bound by position:\n{text}");
        assert!(text.contains(needle), "[{tag}] expected {needle:?} in:\n{text}");
    }
}

// ---------------------------------------------------------------------------
// Imported callables
// ---------------------------------------------------------------------------

const LIB: &str = "fn sub2(a: float, b: float) -> float:\n\
                   \x20   return a - b\n\
                   \n\
                   model Scaler(scale: float, shift: float):\n\
                   \x20   w: Tensor = full([1], scale - shift)\n\
                   \n\
                   \x20   fn f(self, a: float, b: float) -> float:\n\
                   \x20       return a - b\n";

/// Builds and runs `main` beside a module `lib.nsl` holding [`LIB`]. The
/// program is run, not just checked: `nsl check` analyses one file and does
/// not see what an import declares.
fn run_with_lib(tag: &str, main: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("nsl_namedarg_import_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lib.nsl"), LIB).unwrap();
    let prog = dir.join("main.nsl");
    std::fs::write(&prog, main).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .arg(&prog)
        .args(["--target", "cpu"])
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// In declaration order the names are accepted, and the call computes what
/// it says: `sub2(a = 10, b = 1)` is 9.
#[test]
fn named_arguments_to_imported_callables_in_order_are_accepted() {
    let (ok, text) = run_with_lib(
        "ok",
        "from lib import sub2, Scaler\n\
         print(sub2(a = 10.0, b = 1.0))\n\
         let m = Scaler(scale = 3.0, shift = 1.0)\n\
         print(m.f(a = 7.0, b = 2.0))\n",
    );
    assert!(ok, "in-order named arguments to imports must compile and run:\n{text}");
    let printed: Vec<f64> = text.lines().filter_map(|l| l.trim().parse().ok()).collect();
    assert_eq!(printed, [9.0, 5.0], "{text}");
}

/// Out of position, an imported callable is refused exactly as a local one
/// is. Before, `sub2(b = 1.0, a = 10.0)` on an imported `sub2` compiled and
/// printed -9.
#[test]
fn a_named_argument_to_an_imported_callable_out_of_position_is_refused() {
    for (tag, main, needle) in [
        (
            "function",
            "from lib import sub2\nprint(sub2(b = 1.0, a = 10.0))\n",
            "argument `b` of `sub2` is passed in position 1, which binds parameter `a`",
        ),
        (
            "aliased",
            "from lib import sub2 as minus\nprint(minus(b = 1.0, a = 10.0))\n",
            "argument `b` of `minus` is passed in position 1, which binds parameter `a`",
        ),
        (
            "constructor",
            "from lib import Scaler\nlet m = Scaler(shift = 1.0, scale = 3.0)\nprint(m.w)\n",
            "argument `shift` of `Scaler` is passed in position 1, which binds parameter `scale`",
        ),
        (
            "method",
            "from lib import Scaler\nlet m = Scaler(3.0, 1.0)\nprint(m.f(b = 1.0, a = 10.0))\n",
            "argument `b` of `f` is passed in position 1, which binds parameter `a`",
        ),
        (
            "module_alias",
            "import lib as L\nprint(L.sub2(b = 1.0, a = 10.0))\n",
            "argument `b` of `sub2` is passed in position 1, which binds parameter `a`",
        ),
        (
            "stdlib",
            "from nsl.nn.losses import mse_loss\nlet a = ones([2])\nlet b = zeros([2])\nprint(mse_loss(target = b, pred = a))\n",
            "argument `target` of `mse_loss` is passed in position 1, which binds parameter `pred`",
        ),
    ] {
        let (ok, text) = run_with_lib(tag, main);
        assert!(!ok, "[{tag}] the call compiled and would have bound by position:\n{text}");
        assert!(text.contains(needle), "[{tag}] expected {needle:?} in:\n{text}");
    }
}
