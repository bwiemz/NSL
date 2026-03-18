// crates/nsl-codegen/src/schema_convert.rs
//! M44: JSON Schema and BNF grammar conversion to abstract Grammar type.
//!
//! Converts JSON Schema objects and BNF grammar strings into the unified
//! `Grammar` representation used by the grammar compiler pipeline.

use crate::grammar_compiler::{
    Alternative, CharRange, Grammar, GrammarElement, RepeatMode, Rule,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during grammar conversion.
#[derive(Debug, Clone)]
pub enum GrammarError {
    /// The JSON schema is missing a required field.
    MissingField(String),
    /// An unsupported JSON Schema type was encountered.
    UnsupportedType(String),
    /// A BNF grammar parsing error.
    BnfParseError(String),
    /// Invalid schema structure.
    InvalidSchema(String),
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrammarError::MissingField(field) => write!(f, "missing required field: {field}"),
            GrammarError::UnsupportedType(ty) => write!(f, "unsupported JSON Schema type: {ty}"),
            GrammarError::BnfParseError(msg) => write!(f, "BNF parse error: {msg}"),
            GrammarError::InvalidSchema(msg) => write!(f, "invalid schema: {msg}"),
        }
    }
}

impl std::error::Error for GrammarError {}

// ---------------------------------------------------------------------------
// JSON Schema -> Grammar
// ---------------------------------------------------------------------------

/// Convert a JSON Schema to a Grammar.
///
/// Handles the following JSON Schema types:
/// - `boolean` -> `"true" | "false"`
/// - `null` -> `"null"`
/// - `integer` -> optional `-` followed by one or more digits
/// - `number` -> integer part with optional `.digits` decimal part
/// - `string` -> `"` followed by any characters, then `"`
/// - `array` -> `[` items `,` ... `]`
/// - `object` -> `{` key-value pairs `}`
/// - `enum` -> choice between literal values
/// - `oneOf` -> choice between sub-schemas
pub fn json_schema_to_grammar(schema: &serde_json::Value) -> Result<Grammar, GrammarError> {
    let mut rules = Vec::new();
    let start_element = convert_schema_value(schema, &mut rules, "root")?;

    rules.push(Rule {
        name: "start".into(),
        alternatives: vec![Alternative {
            elements: vec![start_element],
        }],
    });

    Ok(Grammar {
        rules,
        start_rule: "start".into(),
    })
}

/// Convert a single JSON Schema value to a GrammarElement, potentially adding
/// helper rules to the rule list.
fn convert_schema_value(
    schema: &serde_json::Value,
    rules: &mut Vec<Rule>,
    prefix: &str,
) -> Result<GrammarElement, GrammarError> {
    // Handle enum first (no "type" needed)
    if let Some(enum_values) = schema.get("enum") {
        return convert_enum(enum_values);
    }

    // Handle oneOf
    if let Some(one_of) = schema.get("oneOf") {
        return convert_one_of(one_of, rules, prefix);
    }

    let ty = schema.get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| GrammarError::MissingField("type".into()))?;

    match ty {
        "boolean" => Ok(GrammarElement::Choice(vec![
            GrammarElement::Literal("true".into()),
            GrammarElement::Literal("false".into()),
        ])),

        "null" => Ok(GrammarElement::Literal("null".into())),

        "integer" => Ok(integer_element()),

        "number" => Ok(number_element()),

        "string" => Ok(string_element()),

        "array" => convert_array(schema, rules, prefix),

        "object" => convert_object(schema, rules, prefix),

        other => Err(GrammarError::UnsupportedType(other.into())),
    }
}

/// Integer: optional `-` followed by one or more digits `[0-9]+`.
fn integer_element() -> GrammarElement {
    let digit = GrammarElement::CharClass(vec![CharRange { lo: b'0', hi: b'9' }]);
    GrammarElement::Sequence(vec![
        GrammarElement::Repeat(
            Box::new(GrammarElement::Literal("-".into())),
            RepeatMode::Optional,
        ),
        GrammarElement::Repeat(
            Box::new(digit),
            RepeatMode::OneOrMore,
        ),
    ])
}

/// Number: integer part with optional `.digits` decimal portion.
fn number_element() -> GrammarElement {
    let digit = GrammarElement::CharClass(vec![CharRange { lo: b'0', hi: b'9' }]);
    let decimal_part = GrammarElement::Sequence(vec![
        GrammarElement::Literal(".".into()),
        GrammarElement::Repeat(
            Box::new(digit.clone()),
            RepeatMode::OneOrMore,
        ),
    ]);

    GrammarElement::Sequence(vec![
        integer_element(),
        GrammarElement::Repeat(
            Box::new(decimal_part),
            RepeatMode::Optional,
        ),
    ])
}

/// String: `"` followed by zero or more non-quote, non-backslash chars or
/// escaped sequences, then `"`.
///
/// Simplified: `"` [^"\]* `"` (does not handle escape sequences for M44a).
fn string_element() -> GrammarElement {
    // Characters allowed inside a JSON string (simplified: any printable ASCII except " and \)
    let string_char = GrammarElement::CharClass(vec![
        CharRange { lo: 0x20, hi: 0x21 },  // space through !
        CharRange { lo: 0x23, hi: 0x5B },  // # through [
        CharRange { lo: 0x5D, hi: 0x7E },  // ] through ~
    ]);

    GrammarElement::Sequence(vec![
        GrammarElement::Literal("\"".into()),
        GrammarElement::Repeat(
            Box::new(string_char),
            RepeatMode::ZeroOrMore,
        ),
        GrammarElement::Literal("\"".into()),
    ])
}

/// Convert a JSON Schema `enum` to a choice between literal values.
fn convert_enum(enum_values: &serde_json::Value) -> Result<GrammarElement, GrammarError> {
    let arr = enum_values.as_array()
        .ok_or_else(|| GrammarError::InvalidSchema("enum must be an array".into()))?;

    if arr.is_empty() {
        return Err(GrammarError::InvalidSchema("enum must have at least one value".into()));
    }

    let choices: Vec<GrammarElement> = arr.iter()
        .map(|v| {
            match v {
                serde_json::Value::String(s) => {
                    // Enum string values are emitted as JSON strings with quotes
                    GrammarElement::Literal(format!("\"{}\"", s))
                }
                serde_json::Value::Number(n) => {
                    GrammarElement::Literal(n.to_string())
                }
                serde_json::Value::Bool(b) => {
                    GrammarElement::Literal(b.to_string())
                }
                serde_json::Value::Null => {
                    GrammarElement::Literal("null".into())
                }
                _ => GrammarElement::Literal(v.to_string()),
            }
        })
        .collect();

    Ok(GrammarElement::Choice(choices))
}

/// Convert a JSON Schema `oneOf` to a choice between sub-schemas.
fn convert_one_of(
    one_of: &serde_json::Value,
    rules: &mut Vec<Rule>,
    prefix: &str,
) -> Result<GrammarElement, GrammarError> {
    let arr = one_of.as_array()
        .ok_or_else(|| GrammarError::InvalidSchema("oneOf must be an array".into()))?;

    let mut choices = Vec::new();
    for (i, sub_schema) in arr.iter().enumerate() {
        let sub_prefix = format!("{prefix}_oneof{i}");
        let elem = convert_schema_value(sub_schema, rules, &sub_prefix)?;
        choices.push(elem);
    }

    Ok(GrammarElement::Choice(choices))
}

/// Convert a JSON Schema `array` to a grammar element.
///
/// `[ item, item, ... ]` with optional items schema.
fn convert_array(
    schema: &serde_json::Value,
    rules: &mut Vec<Rule>,
    prefix: &str,
) -> Result<GrammarElement, GrammarError> {
    let item_element = if let Some(items) = schema.get("items") {
        let item_prefix = format!("{prefix}_item");
        convert_schema_value(items, rules, &item_prefix)?
    } else {
        // Any value
        GrammarElement::Choice(vec![
            string_element(),
            number_element(),
            GrammarElement::Choice(vec![
                GrammarElement::Literal("true".into()),
                GrammarElement::Literal("false".into()),
            ]),
            GrammarElement::Literal("null".into()),
        ])
    };

    // Array: "[" ws (item ("," ws item)*)? ws "]"
    let ws = GrammarElement::Repeat(
        Box::new(GrammarElement::Literal(" ".into())),
        RepeatMode::ZeroOrMore,
    );

    let comma_item = GrammarElement::Sequence(vec![
        GrammarElement::Literal(",".into()),
        ws.clone(),
        item_element.clone(),
    ]);

    let items_list = GrammarElement::Sequence(vec![
        item_element,
        GrammarElement::Repeat(
            Box::new(comma_item),
            RepeatMode::ZeroOrMore,
        ),
    ]);

    Ok(GrammarElement::Sequence(vec![
        GrammarElement::Literal("[".into()),
        ws.clone(),
        GrammarElement::Repeat(
            Box::new(items_list),
            RepeatMode::Optional,
        ),
        ws,
        GrammarElement::Literal("]".into()),
    ]))
}

/// Convert a JSON Schema `object` to a grammar element.
///
/// `{ "key": value, ... }`
fn convert_object(
    schema: &serde_json::Value,
    rules: &mut Vec<Rule>,
    prefix: &str,
) -> Result<GrammarElement, GrammarError> {
    let properties = schema.get("properties")
        .and_then(|v| v.as_object());

    let ws = GrammarElement::Repeat(
        Box::new(GrammarElement::Literal(" ".into())),
        RepeatMode::ZeroOrMore,
    );

    if let Some(props) = properties {
        if props.is_empty() {
            // Empty object: "{}"
            return Ok(GrammarElement::Sequence(vec![
                GrammarElement::Literal("{".into()),
                ws.clone(),
                GrammarElement::Literal("}".into()),
            ]));
        }

        let mut kv_elements = Vec::new();
        for (i, (key, value_schema)) in props.iter().enumerate() {
            let value_prefix = format!("{prefix}_{key}");
            let value_element = convert_schema_value(value_schema, rules, &value_prefix)?;

            let kv = GrammarElement::Sequence(vec![
                GrammarElement::Literal(format!("\"{}\"", key)),
                ws.clone(),
                GrammarElement::Literal(":".into()),
                ws.clone(),
                value_element,
            ]);

            if i > 0 {
                kv_elements.push(GrammarElement::Sequence(vec![
                    GrammarElement::Literal(",".into()),
                    ws.clone(),
                    kv,
                ]));
            } else {
                kv_elements.push(kv);
            }
        }

        Ok(GrammarElement::Sequence(vec![
            GrammarElement::Literal("{".into()),
            ws.clone(),
            GrammarElement::Sequence(kv_elements),
            ws,
            GrammarElement::Literal("}".into()),
        ]))
    } else {
        // Object with no properties defined: just `{}`
        Ok(GrammarElement::Sequence(vec![
            GrammarElement::Literal("{".into()),
            ws.clone(),
            GrammarElement::Literal("}".into()),
        ]))
    }
}

// ---------------------------------------------------------------------------
// BNF -> Grammar
// ---------------------------------------------------------------------------

/// Parse a simple BNF grammar into the abstract Grammar type.
///
/// Format:
/// ```text
/// rule_name: alt1 | alt2
/// another_rule: "literal" rule_ref "literal"
/// ```
///
/// - Rules are separated by newlines
/// - Alternatives are separated by `|`
/// - Quoted strings (`"..."`) are literals
/// - Unquoted identifiers are rule references
/// - The first rule is the start rule
pub fn parse_bnf_grammar(text: &str) -> Result<Grammar, GrammarError> {
    let mut rules = Vec::new();
    let mut start_rule = String::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let (name, body) = line.split_once(':')
            .ok_or_else(|| GrammarError::BnfParseError(
                format!("expected 'rule_name: body', got: {line}")
            ))?;

        let name = name.trim().to_string();
        let body = body.trim();

        if start_rule.is_empty() {
            start_rule = name.clone();
        }

        let alternatives = parse_bnf_alternatives(body)?;

        rules.push(Rule {
            name,
            alternatives,
        });
    }

    if rules.is_empty() {
        return Err(GrammarError::BnfParseError("empty grammar".into()));
    }

    Ok(Grammar {
        rules,
        start_rule,
    })
}

/// Parse BNF alternatives separated by `|`.
fn parse_bnf_alternatives(body: &str) -> Result<Vec<Alternative>, GrammarError> {
    let mut alternatives = Vec::new();

    // Split on `|` but not inside quotes
    let parts = split_bnf_alternatives(body);

    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        let elements = parse_bnf_elements(part)?;
        alternatives.push(Alternative { elements });
    }

    if alternatives.is_empty() {
        return Err(GrammarError::BnfParseError("empty alternative".into()));
    }

    Ok(alternatives)
}

/// Split a BNF body on `|` characters, respecting quoted strings.
fn split_bnf_alternatives(body: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for ch in body.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
            current.push(ch);
        } else if ch == '|' && !in_quotes {
            parts.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

/// Parse a single BNF alternative into grammar elements.
///
/// Elements are separated by whitespace. Quoted strings become Literal elements,
/// and unquoted identifiers become RuleRef elements.
fn parse_bnf_elements(text: &str) -> Result<Vec<GrammarElement>, GrammarError> {
    let mut elements = Vec::new();
    let mut chars = text.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }

        if ch == '"' {
            // Quoted literal
            chars.next(); // consume opening quote
            let mut literal = String::new();
            loop {
                match chars.next() {
                    Some('"') => break,
                    Some(c) => literal.push(c),
                    None => return Err(GrammarError::BnfParseError(
                        "unterminated string literal".into()
                    )),
                }
            }
            elements.push(GrammarElement::Literal(literal));
        } else if ch.is_alphanumeric() || ch == '_' {
            // Rule reference (identifier)
            let mut ident = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_alphanumeric() || c == '_' {
                    ident.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            elements.push(GrammarElement::RuleRef(ident));
        } else {
            // Skip unexpected characters
            chars.next();
        }
    }

    Ok(elements)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar_compiler::compile_grammar;

    #[test]
    fn test_json_schema_boolean() {
        let schema: serde_json::Value = serde_json::json!({"type": "boolean"});
        let grammar = json_schema_to_grammar(&schema).unwrap();
        let dfa = compile_grammar(&grammar);
        assert!(dfa.accepts(b"true"));
        assert!(dfa.accepts(b"false"));
        assert!(!dfa.accepts(b"True"));
        assert!(!dfa.accepts(b"yes"));
        assert!(!dfa.accepts(b""));
    }

    #[test]
    fn test_json_schema_integer() {
        let schema: serde_json::Value = serde_json::json!({"type": "integer"});
        let grammar = json_schema_to_grammar(&schema).unwrap();
        let dfa = compile_grammar(&grammar);
        assert!(dfa.accepts(b"0"));
        assert!(dfa.accepts(b"42"));
        assert!(dfa.accepts(b"123"));
        assert!(dfa.accepts(b"-7"));
        assert!(dfa.accepts(b"-100"));
        assert!(!dfa.accepts(b""));
        assert!(!dfa.accepts(b"abc"));
        assert!(!dfa.accepts(b"-"));
        assert!(!dfa.accepts(b"3.14"));
    }

    #[test]
    fn test_json_schema_string() {
        let schema: serde_json::Value = serde_json::json!({"type": "string"});
        let grammar = json_schema_to_grammar(&schema).unwrap();
        let dfa = compile_grammar(&grammar);
        assert!(dfa.accepts(b"\"hello\""));
        assert!(dfa.accepts(b"\"\""));
        assert!(dfa.accepts(b"\"abc 123\""));
        assert!(!dfa.accepts(b"hello"));
        assert!(!dfa.accepts(b"\"unterminated"));
        assert!(!dfa.accepts(b""));
    }

    #[test]
    fn test_json_schema_object() {
        let schema: serde_json::Value = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        });
        let grammar = json_schema_to_grammar(&schema).unwrap();
        let dfa = compile_grammar(&grammar);
        // serde_json uses BTreeMap so properties are in alphabetical order: age before name
        assert!(dfa.accepts(b"{\"age\":25,\"name\":\"alice\"}"));
        assert!(dfa.accepts(b"{\"age\":-1,\"name\":\"bob\"}"));
        // With optional whitespace (no space before comma — grammar puts ws after comma)
        assert!(dfa.accepts(b"{ \"age\" : 42, \"name\" : \"x\" }"));
        // Reject non-objects
        assert!(!dfa.accepts(b"{}"));
        assert!(!dfa.accepts(b"\"hello\""));
    }

    #[test]
    fn test_json_schema_enum() {
        let schema: serde_json::Value = serde_json::json!({"enum": ["a", "b", "c"]});
        let grammar = json_schema_to_grammar(&schema).unwrap();
        let dfa = compile_grammar(&grammar);
        assert!(dfa.accepts(b"\"a\""));
        assert!(dfa.accepts(b"\"b\""));
        assert!(dfa.accepts(b"\"c\""));
        assert!(!dfa.accepts(b"\"d\""));
        assert!(!dfa.accepts(b"a"));
    }

    #[test]
    fn test_bnf_simple() {
        let bnf = r#"start: "true" | "false""#;
        let grammar = parse_bnf_grammar(bnf).unwrap();
        let dfa = compile_grammar(&grammar);
        assert!(dfa.accepts(b"true"));
        assert!(dfa.accepts(b"false"));
        assert!(!dfa.accepts(b"null"));
    }

    #[test]
    fn test_bnf_with_rule_refs() {
        let bnf = r#"
            start: greeting name
            greeting: "hello " | "hi "
            name: "alice" | "bob"
        "#;
        let grammar = parse_bnf_grammar(bnf).unwrap();
        let dfa = compile_grammar(&grammar);
        assert!(dfa.accepts(b"hello alice"));
        assert!(dfa.accepts(b"hello bob"));
        assert!(dfa.accepts(b"hi alice"));
        assert!(dfa.accepts(b"hi bob"));
        assert!(!dfa.accepts(b"hey alice"));
        assert!(!dfa.accepts(b"hello charlie"));
    }
}
