//! Phase 5 Task 5: `pinned_until_inspect_sync` extends tensor death past
//! the inspect memcpy completion so allocations aren't reused mid-step.

use std::collections::{BTreeSet, HashMap};

use nsl_codegen::wengert::{PrimalOp, VarId, WengertList, WengertOp};
use nsl_codegen::wrga_memory::{plan_memory, plan_memory_with_pin};

fn op(id: u32, result: u32, o: PrimalOp, inputs: Vec<u32>) -> WengertOp {
    WengertOp {
        id,
        result,
        op: o,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    }
}

/// Chain graph x → a → b → c → d — the first activation (var 1) naturally
/// dies early (last consumed at pp 2), so pinning it should extend its death.
fn build_chain_inputs() -> (
    WengertList,
    BTreeSet<VarId>,
    HashMap<VarId, u64>,
    BTreeSet<VarId>,
) {
    let list = WengertList {
        ops: vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Relu, vec![0]),
            op(2, 2, PrimalOp::Relu, vec![1]),
            op(3, 3, PrimalOp::Relu, vec![2]),
            op(4, 4, PrimalOp::Relu, vec![3]),
        ],
        output: 4,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };
    let live: BTreeSet<_> = [1u32, 2, 3, 4].iter().copied().collect();
    (list, live, HashMap::new(), BTreeSet::new())
}

#[test]
fn pinned_var_has_extended_death() {
    let (list, live, sizes, extra) = build_chain_inputs();
    let no_pin = plan_memory(&list, &live, &sizes, &extra);

    // var 1 is consumed by var 2 at pp 2, so its natural death is 2.
    let pin_var: VarId = 1;
    let mut pinned = BTreeSet::new();
    pinned.insert(pin_var);
    let with_pin = plan_memory_with_pin(&list, &live, &sizes, &extra, &pinned);

    let no_pin_death = no_pin
        .assignments
        .iter()
        .find(|s| s.var == pin_var)
        .expect("var should be in plan")
        .death;
    let pin_death = with_pin
        .assignments
        .iter()
        .find(|s| s.var == pin_var)
        .expect("var should be in plan")
        .death;

    assert!(
        pin_death > no_pin_death,
        "pinned death {} should exceed unpinned death {}",
        pin_death,
        no_pin_death
    );

    // And it should be past the max program-point death seen in the unpinned plan.
    let max_pp = no_pin
        .assignments
        .iter()
        .map(|s| s.death)
        .max()
        .unwrap_or(0);
    assert!(pin_death > max_pp);
}

#[test]
fn unpinned_plan_is_unchanged_when_set_is_empty() {
    let (list, live, sizes, extra) = build_chain_inputs();
    let no_pin = plan_memory(&list, &live, &sizes, &extra);
    let with_empty_pin = plan_memory_with_pin(&list, &live, &sizes, &extra, &BTreeSet::new());

    assert_eq!(no_pin.assignments.len(), with_empty_pin.assignments.len());
    for (a, b) in no_pin
        .assignments
        .iter()
        .zip(with_empty_pin.assignments.iter())
    {
        assert_eq!(a.var, b.var);
        assert_eq!(a.death, b.death);
        assert_eq!(a.birth, b.birth);
        assert_eq!(a.slot, b.slot);
    }
}
