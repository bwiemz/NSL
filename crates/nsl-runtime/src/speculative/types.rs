/// Tree node for tree-structured speculation.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub parent: i32,
    pub depth: i32,
    pub token_id: i64,
    pub log_prob: f32,
    pub accepted: bool,
}

/// Result of tree construction from draft model.
#[derive(Debug)]
pub struct SpeculativeTree {
    pub nodes: Vec<TreeNode>,
    pub dfs_enter: Vec<i32>,
    pub dfs_exit: Vec<i32>,
    pub children: Vec<Vec<usize>>,
    pub tree_depth: usize,
    pub tree_width: usize,
}

/// Verification result from rejection sampling.
#[derive(Debug)]
pub struct VerifyResult {
    pub accepted_tokens: Vec<i64>,
    pub num_accepted: usize,
    pub has_bonus: bool,
}

impl VerifyResult {
    pub fn empty() -> Self {
        Self {
            accepted_tokens: Vec::new(),
            num_accepted: 0,
            has_bonus: false,
        }
    }
}
