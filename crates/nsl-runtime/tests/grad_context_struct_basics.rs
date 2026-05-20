//! Spec B §4.1 — GradContext is Send, holds owned ops, has a consumed
//! flag for double-backward detection.

use nsl_runtime::grad_context::GradContext;

fn _assert_send<T: Send>() {}

#[test]
fn grad_context_is_send() {
    _assert_send::<GradContext>();
}

#[test]
fn grad_context_consumed_flag_starts_false() {
    let ctx = GradContext::new(vec![], vec![], vec![], vec![]);
    assert!(!ctx.consumed());
}

#[test]
fn grad_context_mark_consumed_idempotent_returns_first_value() {
    let mut ctx = GradContext::new(vec![], vec![], vec![], vec![]);
    let first = ctx.mark_consumed();
    assert!(!first, "first mark returns prior value (false)");
    let second = ctx.mark_consumed();
    assert!(second, "subsequent mark returns prior value (true)");
}
