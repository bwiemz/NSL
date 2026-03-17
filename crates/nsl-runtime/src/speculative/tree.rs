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
        accepted: false,
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
                nodes.push(TreeNode {
                    parent: parent_idx as i32,
                    depth: d as i32,
                    token_id: tokens[token_idx],
                    log_prob: log_probs[token_idx],
                    accepted: false,
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
}
