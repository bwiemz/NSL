use crate::kernel_skeleton::indexing::{
    emit_thread_lane_warp_register_decl, emit_thread_lane_warp_register_init,
};

#[test]
fn thread_lane_warp_register_decl_test() {
    let mut ptx = String::new();
    emit_thread_lane_warp_register_decl(&mut ptx);
    assert_eq!(
        ptx,
        include_str!("snapshots/thread_lane_warp_register_decl.snap"),
        "tid/warp/lane register decl drift"
    );
}

#[test]
fn thread_lane_warp_register_init_test() {
    let mut ptx = String::new();
    emit_thread_lane_warp_register_init(&mut ptx);
    assert_eq!(
        ptx,
        include_str!("snapshots/thread_lane_warp_register_init.snap"),
        "tid/warp/lane register init drift"
    );
}
