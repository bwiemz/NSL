use nsl_errors::Span;
use string_interner::DefaultSymbol;

pub type Symbol = DefaultSymbol;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // === Literals ===
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    FStringStart,
    FStringText(String),
    FStringExprStart,
    FStringExprEnd,
    FStringEnd,

    // === Identifiers ===
    Ident(Symbol),

    // === Keywords: Variable declarations ===
    Let,
    Const,
    Mut,

    // === Keywords: Functions & returns ===
    Fn,
    Return,
    Yield,
    Async,
    Await,

    // === Keywords: Control flow ===
    If,
    Elif,
    Else,
    For,
    In,
    While,
    Break,
    Continue,
    Match,
    Case,

    // === Keywords: ML blocks ===
    Model,
    Train,
    Grad,
    Quant,
    Kernel,
    Device,
    Tokenizer,
    Dataset,
    Datatype,

    // === Keywords: Module system ===
    Import,
    From,
    As,
    Pub,
    Priv,

    // === Keywords: Literals ===
    True,
    False,
    None,

    // === Keywords: Logical ===
    And,
    Or,
    Not,
    Is,

    // === Keywords: Type-related ===
    Typeof,
    Sizeof,
    Ref,

    // === Keywords: Compound types ===
    Struct,
    Enum,
    Trait,
    SelfKw,

    // === Operators: Arithmetic ===
    Plus,
    Minus,
    Star,
    Slash,
    DoubleSlash,
    Percent,
    DoubleStar,
    At,

    // === Operators: Comparison ===
    EqEq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,

    // === Operators: Assignment ===
    Eq,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,

    // === Operators: Pipe & Bitwise ===
    Pipe,       // |>
    Bar,        // |
    Ampersand,  // &

    // === Operators: Range ===
    DotDot,     // ..
    DotDotEq,   // ..=

    // === Operators: Arrows ===
    Arrow,      // ->
    FatArrow,   // =>

    // === Delimiters ===
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,

    // === Punctuation ===
    Dot,
    Comma,
    Colon,
    Semicolon,
    Hash,
    Underscore,
    Ellipsis,
    Backslash,

    // === Indentation (synthetic) ===
    Newline,
    Indent,
    Dedent,

    // === Special ===
    DocComment(String),
    Eof,
    Error(String),
}

impl TokenKind {
    pub fn is_assignment_op(&self) -> bool {
        matches!(
            self,
            TokenKind::Eq
                | TokenKind::PlusEq
                | TokenKind::MinusEq
                | TokenKind::StarEq
                | TokenKind::SlashEq
        )
    }
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenKind::IntLiteral(v) => write!(f, "{v}"),
            TokenKind::FloatLiteral(v) => write!(f, "{v}"),
            TokenKind::StringLiteral(s) => write!(f, "\"{s}\""),
            TokenKind::Ident(_) => write!(f, "identifier"),
            TokenKind::Let => write!(f, "let"),
            TokenKind::Const => write!(f, "const"),
            TokenKind::Fn => write!(f, "fn"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Elif => write!(f, "elif"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::For => write!(f, "for"),
            TokenKind::In => write!(f, "in"),
            TokenKind::While => write!(f, "while"),
            TokenKind::Break => write!(f, "break"),
            TokenKind::Continue => write!(f, "continue"),
            TokenKind::Match => write!(f, "match"),
            TokenKind::Case => write!(f, "case"),
            TokenKind::Model => write!(f, "model"),
            TokenKind::Train => write!(f, "train"),
            TokenKind::Grad => write!(f, "grad"),
            TokenKind::Quant => write!(f, "quant"),
            TokenKind::Kernel => write!(f, "kernel"),
            TokenKind::Datatype => write!(f, "datatype"),
            TokenKind::Import => write!(f, "import"),
            TokenKind::From => write!(f, "from"),
            TokenKind::As => write!(f, "as"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::None => write!(f, "none"),
            TokenKind::And => write!(f, "and"),
            TokenKind::Or => write!(f, "or"),
            TokenKind::Not => write!(f, "not"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Eq => write!(f, "="),
            TokenKind::EqEq => write!(f, "=="),
            TokenKind::NotEq => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::LtEq => write!(f, "<="),
            TokenKind::GtEq => write!(f, ">="),
            TokenKind::Pipe => write!(f, "|>"),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::FatArrow => write!(f, "=>"),
            TokenKind::LeftParen => write!(f, "("),
            TokenKind::RightParen => write!(f, ")"),
            TokenKind::LeftBracket => write!(f, "["),
            TokenKind::RightBracket => write!(f, "]"),
            TokenKind::LeftBrace => write!(f, "{{"),
            TokenKind::RightBrace => write!(f, "}}"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Newline => write!(f, "newline"),
            TokenKind::Indent => write!(f, "INDENT"),
            TokenKind::Dedent => write!(f, "DEDENT"),
            TokenKind::Eof => write!(f, "EOF"),
            TokenKind::Error(msg) => write!(f, "error: {msg}"),
            _ => write!(f, "{self:?}"),
        }
    }
}
