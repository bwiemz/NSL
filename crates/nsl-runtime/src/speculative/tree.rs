use super::types::{SpeculativeTree, TreeNode};

/// Build a speculation tree.
///
/// `depth`: number of token levels
/// `width`: branching factor at each level
/// `tokens`: flat array of token IDs in breadth-first order
/// `log_probs`: corresponding log-probabilities
pub fn build_tree(
    depth: usize,
    width: usize,
    tokens: &[i64],
    log_probs: &[f32],
) -> SpeculativeTree {
    let mut nodes = vec![TreeNode {
        parent: -1,
        depth: 0,
        token_id: -1,
        log_prob: 0.0,
        value: 0.0,
        accepted: false,
        is_leaf: true,
    }];
    let mut children: Vec<Vec<usize>> = vec![vec![]];

    let mut token_idx = 0;
    let mut current_level_start = 0;
    let mut current_level_end = 1;

    for d in 1..=depth {
        let _next_level_start = nodes.len();
        for parent_idx in current_level_start..current_level_end {
            for _w in 0..width {
                if token_idx >= tokens.len() {
                    break;
                }
                let node_idx = nodes.len();
                let parent_value = nodes[parent_idx].value;
                nodes[parent_idx].is_leaf = false;
                nodes.push(TreeNode {
                    parent: parent_idx as i32,
                    depth: d as i32,
                    token_id: tokens[token_idx],
                    log_prob: log_probs[token_idx],
                    value: parent_value + log_probs[token_idx],
                    accepted: false,
                    is_leaf: true,
                });
                children.push(vec![]);
                children[parent_idx].push(node_idx);
                token_idx += 1;
            }
        }
        current_level_start = current_level_end;
        current_level_end = nodes.len();
    }

    let (dfs_enter, dfs_exit) = compute_dfs_timestamps(&children, nodes.len());

    SpeculativeTree {
        nodes,
        dfs_enter,
        dfs_exit,
        children,
        tree_depth: depth,
        tree_width: width,
    }
}

fn compute_dfs_timestamps(children: &[Vec<usize>], num_nodes: usize) -> (Vec<i32>, Vec<i32>) {
    let mut enter = vec![0i32; num_nodes];
    let mut exit = vec![0i32; num_nodes];
    let mut time = 0i32;

    fn dfs(
        node: usize,
        children: &[Vec<usize>],
        enter: &mut Vec<i32>,
        exit: &mut Vec<i32>,
        time: &mut i32,
    ) {
        enter[node] = *time;
        *time += 1;
        for &child in &children[node] {
            dfs(child, children, enter, exit, time);
        }
        exit[node] = *time;
        *time += 1;
    }

    dfs(0, children, &mut enter, &mut exit, &mut time);
    (enter, exit)
}

/// O(1) ancestor check using DFS timestamps.
pub fn is_ancestor(tree: &SpeculativeTree, ancestor: usize, descendant: usize) -> bool {
    tree.dfs_enter[ancestor] <= tree.dfs_enter[descendant]
        && tree.dfs_exit[ancestor] >= tree.dfs_exit[descendant]
}

/// Select the longest path of accepted tokens from root.
pub fn select_longest_accepted_path(tree: &SpeculativeTree) -> Vec<i64> {
    fn dfs_longest(tree: &SpeculativeTree, node: usize) -> Vec<i64> {
        let mut best_path: Vec<i64> = Vec::new();
        for &child in &tree.children[node] {
            if !tree.nodes[child].accepted {
                continue;
            }
            let mut child_path = vec![tree.nodes[child].token_id];
            child_path.extend(dfs_longest(tree, child));
            if child_path.len() > best_path.len() {
                best_path = child_path;
            }
        }
        best_path
    }

    if !tree.nodes[0].accepted {
        return vec![];
    }
    dfs_longest(tree, 0)
}

// ---------------------------------------------------------------------------
// EAGLE-2: Dynamic confidence-scored tree expansion
// ---------------------------------------------------------------------------

/// Build a speculation tree dynamically based on confidence scores.
///
/// Unlike the static `build_tree()` which expands all branches uniformly,
/// `build_dynamic_tree()` expands the highest-value leaf at each step,
/// allocating the token budget to the most promising branches.
///
/// Candidate generator: given context tokens, returns (token_id, log_prob) pairs.
type CandidateFn<'a> = &'a dyn Fn(&[i64]) -> Vec<(i64, f32)>;

/// `candidates_fn`: given a path from root to a leaf (token IDs), returns
///   top-k (token_id, log_prob) candidates for the next position.
/// `token_budget`: maximum number of nodes to add to the tree.
/// `expansion_k`: top-k children to expand at each step.
pub fn build_dynamic_tree(
    candidates_fn: CandidateFn<'_>,
    token_budget: usize,
    expansion_k: usize,
) -> SpeculativeTree {
    let mut nodes = vec![TreeNode {
        parent: -1, depth: 0, token_id: -1,
        log_prob: 0.0, value: 0.0, accepted: false, is_leaf: true,
    }];
    let mut children: Vec<Vec<usize>> = vec![vec![]];
    let mut remaining = token_budget;

    while remaining > 0 {
        // Find leaf with highest cumulative value
        let best_leaf = match nodes.iter().enumerate()
            .filter(|(_, n)| n.is_leaf)
            .max_by(|a, b| a.1.value.partial_cmp(&b.1.value).unwrap_or(std::cmp::Ordering::Equal))
        {
            Some((idx, _)) => idx,
            None => break, // no more leaves
        };

        // Get path from root to this leaf (for context)
        let context = path_to_root(&nodes, best_leaf);

        // Get top-k candidates from the draft model
        let candidates = candidates_fn(&context);

        // Expand top-k children
        let k = candidates.len().min(expansion_k).min(remaining);
        for &(token_id, log_prob) in candidates.iter().take(k) {
            let node_idx = nodes.len();
            let parent_value = nodes[best_leaf].value;
            nodes.push(TreeNode {
                parent: best_leaf as i32,
                depth: nodes[best_leaf].depth + 1,
                token_id,
                log_prob,
                value: parent_value + log_prob,
                accepted: false,
                is_leaf: true,
            });
            children.push(vec![]);
            children[best_leaf].push(node_idx);
            remaining -= 1;
        }
        nodes[best_leaf].is_leaf = false;
    }

    let (dfs_enter, dfs_exit) = compute_dfs_timestamps(&children, nodes.len());
    let max_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0) as usize;

    SpeculativeTree {
        nodes,
        dfs_enter,
        dfs_exit,
        children,
        tree_depth: max_depth,
        tree_width: expansion_k,
    }
}

/// Get the path of token IDs from root to the given node.
fn path_to_root(nodes: &[TreeNode], mut node_idx: usize) -> Vec<i64> {
    let mut path = Vec::new();
    while nodes[node_idx].parent >= 0 {
        path.push(nodes[node_idx].token_id);
        node_idx = nodes[node_idx].parent as usize;
    }
    path.reverse();
    path
}

// ---------------------------------------------------------------------------
// Tree attention mask
// ---------------------------------------------------------------------------

/// Build a tree-structured causal attention mask.
///
/// `mask[i][j] = true` if node j is an ancestor of node i (or i == j).
/// This is used for tree-structured attention in the target model's
/// verification forward pass.
pub fn build_tree_attention_mask(tree: &SpeculativeTree) -> Vec<Vec<bool>> {
    let n = tree.nodes.len();
    let mut mask = vec![vec![false; n]; n];

    for (i, row) in mask.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = is_ancestor(tree, j, i);
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_tree_depth3() {
        let tree = build_tree(3, 1, &[10, 20, 30], &[-0.1, -0.2, -0.3]);
        assert_eq!(tree.nodes.len(), 4); // root + 3
        assert_eq!(tree.nodes[0].parent, -1);
        assert_eq!(tree.nodes[1].parent, 0);
        assert_eq!(tree.nodes[2].parent, 1);
        assert_eq!(tree.nodes[3].parent, 2);
        assert_eq!(tree.nodes[1].token_id, 10);
        assert_eq!(tree.nodes[2].token_id, 20);
        assert_eq!(tree.nodes[3].token_id, 30);
    }

    #[test]
    fn test_binary_tree_depth2() {
        let tokens = vec![10, 11, 20, 21, 22, 23];
        let probs = vec![-0.1; 6];
        let tree = build_tree(2, 2, &tokens, &probs);
        assert_eq!(tree.nodes.len(), 7); // root(1) + level1(2) + level2(4)
        assert_eq!(tree.tree_depth, 2);
        assert_eq!(tree.tree_width, 2);
        assert_eq!(tree.nodes[1].parent, 0);
        assert_eq!(tree.nodes[2].parent, 0);
        assert_eq!(tree.nodes[1].depth, 1);
    }

    #[test]
    fn test_dfs_ancestor_check() {
        let tree = build_tree(3, 1, &[10, 20, 30], &[-0.1, -0.2, -0.3]);
        assert!(is_ancestor(&tree, 0, 3)); // root -> C
        assert!(is_ancestor(&tree, 1, 3)); // A -> C
        assert!(is_ancestor(&tree, 2, 3)); // B -> C
        assert!(is_ancestor(&tree, 3, 3)); // C -> C (self)
        assert!(!is_ancestor(&tree, 3, 0)); // C NOT -> root
    }

    #[test]
    fn test_longest_path_linear() {
        let mut tree = build_tree(3, 1, &[10, 20, 30], &[-0.1, -0.2, -0.3]);
        tree.nodes[0].accepted = true;
        tree.nodes[1].accepted = true;
        tree.nodes[2].accepted = true;
        tree.nodes[3].accepted = false;
        let path = select_longest_accepted_path(&tree);
        assert_eq!(path, vec![10, 20]);
    }

    #[test]
    fn test_longest_path_branching() {
        let mut tree = build_tree(2, 2, &[10, 11, 20, 21, 22, 23], &[-0.1; 6]);
        tree.nodes[0].accepted = true;
        tree.nodes[1].accepted = true; // A(10) left
        tree.nodes[2].accepted = false; // B(11) right rejected
        tree.nodes[3].accepted = true; // C(20) child of A
        tree.nodes[4].accepted = false;
        let path = select_longest_accepted_path(&tree);
        assert_eq!(path, vec![10, 20]); // root -> A -> C
    }

    // --- EAGLE-2 dynamic tree tests ---

    #[test]
    fn test_dynamic_tree_budget_respected() {
        // candidates_fn always returns 3 candidates
        let candidates = |_ctx: &[i64]| -> Vec<(i64, f32)> {
            vec![(100, -0.1), (101, -0.5), (102, -0.9)]
        };
        let tree = build_dynamic_tree(&candidates, 5, 3);
        // Budget=5: should have root + 5 nodes = 6 total
        assert_eq!(tree.nodes.len(), 6);
    }

    #[test]
    fn test_dynamic_tree_expands_highest_value() {
        // First call returns two candidates with different confidence
        let call_count = std::cell::Cell::new(0);
        let candidates = |_ctx: &[i64]| -> Vec<(i64, f32)> {
            let n = call_count.get();
            call_count.set(n + 1);
            if n == 0 {
                // First expansion: two children with different values
                vec![(10, -0.1), (11, -2.0)] // token 10 is much more confident
            } else {
                // Second expansion: should expand from token 10 (higher value)
                vec![(20, -0.1)]
            }
        };
        let tree = build_dynamic_tree(&candidates, 3, 2);
        // Should have: root → {10(-0.1), 11(-2.0)} → 10→{20(-0.2)}
        assert_eq!(tree.nodes.len(), 4); // root + 3 nodes
        // Node 3 (token 20) should be child of node 1 (token 10), not node 2 (token 11)
        assert_eq!(tree.nodes[3].token_id, 20);
        assert_eq!(tree.nodes[3].parent, 1); // parent is token 10
    }

    #[test]
    fn test_tree_attention_mask_linear() {
        let tree = build_tree(2, 1, &[10, 20], &[-0.1, -0.2]);
        let mask = build_tree_attention_mask(&tree);
        // 3 nodes: root(0), A(1), B(2)
        assert_eq!(mask.len(), 3);
        // root attends to itself
        assert!(mask[0][0]);
        // A attends to root and itself
        assert!(mask[1][0]);
        assert!(mask[1][1]);
        assert!(!mask[1][2]); // A does NOT attend to B
        // B attends to root, A, and itself
        assert!(mask[2][0]);
        assert!(mask[2][1]);
        assert!(mask[2][2]);
    }

    #[test]
    fn test_tree_attention_mask_branching() {
        let tree = build_tree(1, 2, &[10, 11], &[-0.1, -0.2]);
        let mask = build_tree_attention_mask(&tree);
        // 3 nodes: root(0), A(1), B(2) — A and B are siblings
        // A attends to root and self, NOT to sibling B
        assert!(mask[1][0]);
        assert!(mask[1][1]);
        assert!(!mask[1][2]);
        // B attends to root and self, NOT to sibling A
        assert!(mask[2][0]);
        assert!(!mask[2][1]);
        assert!(mask[2][2]);
    }

    #[test]
    fn test_cumulative_value_computation() {
        let tree = build_tree(3, 1, &[10, 20, 30], &[-0.1, -0.2, -0.3]);
        // Values should be cumulative: -0.1, -0.3, -0.6
        assert!((tree.nodes[1].value - (-0.1)).abs() < 1e-6);
        assert!((tree.nodes[2].value - (-0.3)).abs() < 1e-6);
        assert!((tree.nodes[3].value - (-0.6)).abs() < 1e-6);
    }
}
