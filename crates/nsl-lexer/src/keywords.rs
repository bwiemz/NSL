use crate::token::TokenKind;

pub fn lookup_keyword(s: &str) -> Option<TokenKind> {
    match s {
        // Variable declarations
        "let" => Some(TokenKind::Let),
        "const" => Some(TokenKind::Const),
        "mut" => Some(TokenKind::Mut),

        // Functions & returns
        "fn" => Some(TokenKind::Fn),
        "return" => Some(TokenKind::Return),
        "yield" => Some(TokenKind::Yield),
        "async" => Some(TokenKind::Async),
        "await" => Some(TokenKind::Await),

        // Control flow
        "if" => Some(TokenKind::If),
        "elif" => Some(TokenKind::Elif),
        "else" => Some(TokenKind::Else),
        "for" => Some(TokenKind::For),
        "in" => Some(TokenKind::In),
        "while" => Some(TokenKind::While),
        "break" => Some(TokenKind::Break),
        "continue" => Some(TokenKind::Continue),
        "match" => Some(TokenKind::Match),
        "case" => Some(TokenKind::Case),

        // ML blocks
        "model" => Some(TokenKind::Model),
        "train" => Some(TokenKind::Train),
        "grad" => Some(TokenKind::Grad),
        "quant" => Some(TokenKind::Quant),
        "kernel" => Some(TokenKind::Kernel),
        "device" => Some(TokenKind::Device),
        "tokenizer" => Some(TokenKind::Tokenizer),
        "dataset" => Some(TokenKind::Dataset),

        // Module system
        "import" => Some(TokenKind::Import),
        "from" => Some(TokenKind::From),
        "as" => Some(TokenKind::As),
        "pub" => Some(TokenKind::Pub),
        "priv" => Some(TokenKind::Priv),

        // Literals
        "true" => Some(TokenKind::True),
        "false" => Some(TokenKind::False),
        "none" => Some(TokenKind::None),

        // Logical
        "and" => Some(TokenKind::And),
        "or" => Some(TokenKind::Or),
        "not" => Some(TokenKind::Not),
        "is" => Some(TokenKind::Is),

        // Type-related
        "typeof" => Some(TokenKind::Typeof),
        "sizeof" => Some(TokenKind::Sizeof),
        "ref" => Some(TokenKind::Ref),

        // Compound types
        "struct" => Some(TokenKind::Struct),
        "enum" => Some(TokenKind::Enum),
        "trait" => Some(TokenKind::Trait),
        "self" => Some(TokenKind::SelfKw),

        _ => None,
    }
}
