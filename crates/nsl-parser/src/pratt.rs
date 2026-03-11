use nsl_ast::operator::BinOp;
use nsl_lexer::TokenKind;

/// Returns (left_bp, right_bp) for infix operators.
/// Left < right means left-associative; left > right means right-associative.
pub fn infix_binding_power(op: &TokenKind) -> Option<(u8, u8)> {
    match op {
        // Pipe (lowest precedence)
        TokenKind::Pipe => Some((2, 3)),

        // Logical or
        TokenKind::Or => Some((4, 5)),

        // Logical and
        TokenKind::And => Some((6, 7)),

        // is, in
        TokenKind::Is | TokenKind::In => Some((8, 9)),

        // Comparison
        TokenKind::EqEq
        | TokenKind::NotEq
        | TokenKind::Lt
        | TokenKind::Gt
        | TokenKind::LtEq
        | TokenKind::GtEq => Some((10, 11)),

        // Bitwise or / Union
        TokenKind::Bar => Some((12, 13)),

        // Bitwise and
        TokenKind::Ampersand => Some((14, 15)),

        // Range
        TokenKind::DotDot | TokenKind::DotDotEq => Some((16, 17)),

        // Addition/subtraction
        TokenKind::Plus | TokenKind::Minus => Some((18, 19)),

        // Multiplication/division/modulo/floor-div
        TokenKind::Star
        | TokenKind::Slash
        | TokenKind::DoubleSlash
        | TokenKind::Percent => Some((20, 21)),

        // Matrix multiply
        TokenKind::At => Some((22, 23)),

        // Exponentiation (right-associative)
        TokenKind::DoubleStar => Some((25, 24)),

        _ => None,
    }
}

/// Returns ((), right_bp) for prefix operators.
pub fn prefix_binding_power(op: &TokenKind) -> Option<u8> {
    match op {
        TokenKind::Minus => Some(27),
        // `not` must bind tighter than `and` (lbp=6) and `or` (lbp=4) but looser
        // than comparisons (lbp=10), matching Python's precedence:
        // `not a and b` → `(not a) and b`; `not a < b` → `not (a < b)`
        TokenKind::Not => Some(8),
        _ => None,
    }
}

/// Returns the binding power for postfix operators (member access, subscript, call).
pub fn postfix_binding_power(op: &TokenKind) -> Option<u8> {
    match op {
        TokenKind::Dot | TokenKind::LeftBracket | TokenKind::LeftParen => Some(30),
        _ => None,
    }
}

/// Convert a token kind to a BinOp.
pub fn token_to_binop(kind: &TokenKind) -> Option<BinOp> {
    match kind {
        TokenKind::Plus => Some(BinOp::Add),
        TokenKind::Minus => Some(BinOp::Sub),
        TokenKind::Star => Some(BinOp::Mul),
        TokenKind::Slash => Some(BinOp::Div),
        TokenKind::DoubleSlash => Some(BinOp::FloorDiv),
        TokenKind::Percent => Some(BinOp::Mod),
        TokenKind::DoubleStar => Some(BinOp::Pow),
        TokenKind::At => Some(BinOp::MatMul),
        TokenKind::EqEq => Some(BinOp::Eq),
        TokenKind::NotEq => Some(BinOp::NotEq),
        TokenKind::Lt => Some(BinOp::Lt),
        TokenKind::Gt => Some(BinOp::Gt),
        TokenKind::LtEq => Some(BinOp::LtEq),
        TokenKind::GtEq => Some(BinOp::GtEq),
        TokenKind::And => Some(BinOp::And),
        TokenKind::Or => Some(BinOp::Or),
        TokenKind::Is => Some(BinOp::Is),
        TokenKind::In => Some(BinOp::In),
        TokenKind::Bar => Some(BinOp::BitOr),
        TokenKind::Ampersand => Some(BinOp::BitAnd),
        _ => None,
    }
}
