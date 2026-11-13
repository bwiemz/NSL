//! A named argument that is not in its parameter's position is refused.
//!
//! Calls bind arguments by position; codegen ignores an argument's name. So
//! `sub2(b = 1.0, a = 10.0)` ran as `sub2(a = 1.0, b = 10.0)` and returned
//! -9 where the call as written is 9 — for free functions, methods and model
//! constructors alike, with no diagnostic. Until calls bind by name, the
//! checker refuses the call where it knows the declaration's parameter
//! names (the module's own callables).

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
         print(m.f(a = 10.0, b = 1.0))\n",
    );
    assert!(ok, "in-order named arguments must check clean:\n{text}");
}

#[test]
fn a_named_argument_out_of_position_is_refused() {
    for (tag, call, needle) in [
        ("function", "print(sub2(b = 1.0, a = 10.0))\n", "argument `b` of `sub2` is passed in position 1, which binds parameter `a`"),
        ("method", "let m = M(3.0, 1.0)\nprint(m.f(b = 1.0, a = 10.0))\n", "argument `b` of `f` is passed in position 1, which binds parameter `a`"),
        ("constructor", "let m = M(shift = 1.0, scale = 3.0)\n", "argument `shift` of `M` is passed in position 1, which binds parameter `scale`"),
        ("skipped", "print(sub2(b = 1.0))\n", "argument `b` of `sub2` is passed in position 1, which binds parameter `a`"),
    ] {
        let (ok, text) = check(tag, call);
        assert!(!ok, "[{tag}] the call checked clean and would have bound by position:\n{text}");
        assert!(text.contains(needle), "[{tag}] expected {needle:?} in:\n{text}");
    }
}
