use nsl_errors::{Diagnostic, Span};

use crate::types::{Dim, Shape};

/// Check if two shapes are compatible for element-wise operations.
pub fn check_elementwise(lhs: &Shape, rhs: &Shape, op_span: Span) -> Result<Shape, Diagnostic> {
    // If either shape is unknown (rank 0 = dynamic), skip validation
    if lhs.rank() == 0 || rhs.rank() == 0 {
        return Ok(Shape::unknown());
    }
    if lhs.rank() != rhs.rank() {
        return Err(Diagnostic::error(format!(
            "shape mismatch: left has shape {}, right has shape {}",
            fmt_shape(lhs),
            fmt_shape(rhs)
        ))
        .with_label(op_span, "incompatible shapes for element-wise operation"));
    }

    let mut result_dims = Vec::new();
    for (i, (l, r)) in lhs.dims.iter().zip(rhs.dims.iter()).enumerate() {
        match unify_dim(l, r) {
            Some(d) => result_dims.push(d),
            None => {
                return Err(Diagnostic::error(format!(
                    "shape mismatch at dimension {i}: {} vs {} in shapes {} and {}",
                    fmt_dim(l),
                    fmt_dim(r),
                    fmt_shape(lhs),
                    fmt_shape(rhs)
                ))
                .with_label(op_span, format!("dimension {i} incompatible")));
            }
        }
    }

    Ok(Shape { dims: result_dims })
}

/// Check matmul compatibility: `[.., M, K] @ [.., K, N] -> [.., M, N]`
pub fn check_matmul(lhs: &Shape, rhs: &Shape, op_span: Span) -> Result<Shape, Diagnostic> {
    // If either shape is unknown (rank 0 = dynamic), skip validation
    if lhs.rank() == 0 || rhs.rank() == 0 {
        return Ok(Shape::unknown());
    }
    if lhs.rank() < 2 || rhs.rank() < 2 {
        return Err(
            Diagnostic::error("matmul requires tensors with at least 2 dimensions")
                .with_label(op_span, "rank too low for matmul"),
        );
    }

    let l_inner = &lhs.dims[lhs.rank() - 1]; // K in left
    let r_inner = &rhs.dims[rhs.rank() - 2]; // K in right

    if unify_dim(l_inner, r_inner).is_none() {
        return Err(Diagnostic::error(format!(
            "matmul inner dimensions don't match: left is {}, right is {}",
            fmt_shape(lhs),
            fmt_shape(rhs)
        ))
        .with_label(op_span, format!(
            "inner dims: {} vs {}",
            fmt_dim(l_inner),
            fmt_dim(r_inner)
        )));
    }

    // Verify batch dimensions are compatible
    let l_batch = lhs.rank() - 2;
    let r_batch = rhs.rank() - 2;
    let min_batch = l_batch.min(r_batch);
    for i in 0..min_batch {
        let l_dim = &lhs.dims[l_batch - 1 - i];
        let r_dim = &rhs.dims[r_batch - 1 - i];
        if unify_dim(l_dim, r_dim).is_none() {
            return Err(Diagnostic::error(format!(
                "matmul batch dimensions don't match: {} vs {}",
                fmt_shape(lhs),
                fmt_shape(rhs)
            ))
            .with_label(op_span, format!(
                "batch dim mismatch: {} vs {}",
                fmt_dim(l_dim),
                fmt_dim(r_dim)
            )));
        }
    }

    // Result: [.., M, N] — take all but last from lhs, append last from rhs
    let mut result = lhs.dims[..lhs.rank() - 1].to_vec();
    result.push(rhs.dims[rhs.rank() - 1].clone());

    Ok(Shape { dims: result })
}

/// Try to unify two dimensions. Returns Some(unified) if compatible, None otherwise.
pub fn unify_dim(a: &Dim, b: &Dim) -> Option<Dim> {
    match (a, b) {
        // Wildcards unify with anything
        (Dim::Wildcard, other) | (other, Dim::Wildcard) => Some(other.clone()),

        // Same concrete values match
        (Dim::Concrete(x), Dim::Concrete(y)) if x == y => Some(Dim::Concrete(*x)),
        (Dim::Concrete(_), Dim::Concrete(_)) => None,

        // Same symbolic names unify
        (Dim::Symbolic(a), Dim::Symbolic(b)) if a == b => Some(Dim::Symbolic(*a)),

        // Symbolic unifies with concrete (the symbolic "becomes" that concrete value)
        (Dim::Symbolic(_), Dim::Concrete(n)) | (Dim::Concrete(n), Dim::Symbolic(_)) => {
            Some(Dim::Concrete(*n))
        }

        // Different symbolic names: treat as incompatible to catch shape errors
        (Dim::Symbolic(_), Dim::Symbolic(_)) => None,

        // Named dims: unify the inner size
        (Dim::Named { name: n1, size: s1 }, Dim::Named { name: n2, size: s2 }) if n1 == n2 => {
            unify_dim(s1, s2).map(|s| Dim::Named {
                name: *n1,
                size: Box::new(s),
            })
        }

        // Named vs non-named: strip the name and unify the size
        (Dim::Named { size, .. }, other) | (other, Dim::Named { size, .. }) => {
            unify_dim(size, other)
        }
    }
}

/// Format a shape for display in error messages.
pub fn fmt_shape(s: &Shape) -> String {
    if s.rank() == 0 {
        return "[?]".into();
    }
    let dims: Vec<String> = s.dims.iter().map(|d| fmt_dim(d)).collect();
    format!("[{}]", dims.join(", "))
}

pub fn fmt_dim(d: &Dim) -> String {
    match d {
        Dim::Concrete(n) => n.to_string(),
        Dim::Symbolic(_) => "<symbolic>".into(),
        Dim::Named { .. } => "<named>".into(),
        Dim::Wildcard => "_".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::Symbol;

    fn make_sym(n: u32) -> Symbol {
        use string_interner::Symbol as SI;
        Symbol(SI::try_from_usize(n as usize).unwrap())
    }

    fn concrete_shape(dims: &[i64]) -> Shape {
        Shape {
            dims: dims.iter().map(|&d| Dim::Concrete(d)).collect(),
        }
    }

    #[test]
    fn elementwise_same_shape() {
        let s = concrete_shape(&[3, 4]);
        let result = check_elementwise(&s, &s, Span::DUMMY).unwrap();
        assert_eq!(result.dims.len(), 2);
    }

    #[test]
    fn elementwise_rank_mismatch() {
        let a = concrete_shape(&[3, 4]);
        let b = concrete_shape(&[3, 4, 5]);
        assert!(check_elementwise(&a, &b, Span::DUMMY).is_err());
    }

    #[test]
    fn elementwise_dim_mismatch() {
        let a = concrete_shape(&[3, 4]);
        let b = concrete_shape(&[3, 5]);
        assert!(check_elementwise(&a, &b, Span::DUMMY).is_err());
    }

    #[test]
    fn matmul_valid() {
        let a = concrete_shape(&[3, 4]);
        let b = concrete_shape(&[4, 5]);
        let result = check_matmul(&a, &b, Span::DUMMY).unwrap();
        assert_eq!(
            result.dims,
            vec![Dim::Concrete(3), Dim::Concrete(5)]
        );
    }

    #[test]
    fn matmul_inner_mismatch() {
        let a = concrete_shape(&[3, 4]);
        let b = concrete_shape(&[5, 6]);
        assert!(check_matmul(&a, &b, Span::DUMMY).is_err());
    }

    #[test]
    fn matmul_rank_too_low() {
        let a = concrete_shape(&[3]);
        let b = concrete_shape(&[3, 4]);
        assert!(check_matmul(&a, &b, Span::DUMMY).is_err());
    }

    #[test]
    fn matmul_batch() {
        let a = concrete_shape(&[2, 3, 4]);
        let b = concrete_shape(&[2, 4, 5]);
        let result = check_matmul(&a, &b, Span::DUMMY).unwrap();
        assert_eq!(
            result.dims,
            vec![Dim::Concrete(2), Dim::Concrete(3), Dim::Concrete(5)]
        );
    }

    #[test]
    fn unify_wildcards() {
        assert_eq!(
            unify_dim(&Dim::Wildcard, &Dim::Concrete(5)),
            Some(Dim::Concrete(5))
        );
        assert_eq!(
            unify_dim(&Dim::Concrete(5), &Dim::Wildcard),
            Some(Dim::Concrete(5))
        );
    }

    #[test]
    fn unify_symbolic_concrete() {
        let sym = make_sym(0);
        assert_eq!(
            unify_dim(&Dim::Symbolic(sym), &Dim::Concrete(32)),
            Some(Dim::Concrete(32))
        );
    }

    #[test]
    fn unify_same_symbolic() {
        let sym = make_sym(0);
        assert_eq!(
            unify_dim(&Dim::Symbolic(sym), &Dim::Symbolic(sym)),
            Some(Dim::Symbolic(sym))
        );
    }

    #[test]
    fn elementwise_with_wildcards() {
        let a = Shape {
            dims: vec![Dim::Wildcard, Dim::Concrete(4)],
        };
        let b = concrete_shape(&[3, 4]);
        let result = check_elementwise(&a, &b, Span::DUMMY).unwrap();
        assert_eq!(
            result.dims,
            vec![Dim::Concrete(3), Dim::Concrete(4)]
        );
    }
}
