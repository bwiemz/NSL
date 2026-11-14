// crates/nsl-codegen/src/schema_convert.rs
//! M44: JSON Schema and BNF grammar conversion to abstract Grammar type.
//!
//! Converts JSON Schema objects and BNF grammar strings into the unified
//! `Grammar` representation used by the grammar compiler pipeline.

use crate::grammar_compiler::{Alternative, CharRange, Grammar, GrammarElement, RepeatMode, Rule};

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
/// Everything the grammar accepts is JSON (RFC 8259) that satisfies the
/// schema. It does not accept every such document: it fixes a layout.
///
/// - `boolean` -> `true | false`
/// - `null` -> `null`
/// - `integer` -> `-? (0 | [1-9][0-9]*)` — no leading zeros
/// - `number` -> an integer, an optional `.digits` fraction and an
///   optional exponent
/// - `string` -> `"`, then characters and escapes as JSON defines them
///   (any byte sequence that is well-formed UTF-8 except `"`, `\` and
///   the control characters; `\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`,
///   `\t`, `\uXXXX`), then `"`
/// - `array` -> `[` items `,` ... `]`
/// - `object` -> `{` members `}`: the properties named in `required` are
///   mandatory and the rest optional; members appear in ONE order, the
///   properties' names sorted (the order `serde_json` iterates them in).
///   Other orders and additional properties are valid JSON for the
///   schema and are not accepted.
/// - `enum` -> choice between the values, each as JSON
/// - `oneOf` -> choice between sub-schemas
///
/// Whitespace: at most one space where JSON allows whitespace between
/// tokens (after `[`, `{`, `,` and `:`, before `:`, `]` and `}`). JSON
/// allows any amount, and a model held to a grammar that does too can
/// emit spaces without end.
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

    let ty = schema
        .get("type")
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

fn digit() -> GrammarElement {
    GrammarElement::CharClass(vec![CharRange { lo: b'0', hi: b'9' }])
}

fn literal(text: &str) -> GrammarElement {
    GrammarElement::Literal(text.into())
}

/// Integer, as JSON writes one: an optional `-`, then `0` or a non-zero
/// digit followed by digits. `00`, `-01` and `007` are not JSON.
fn integer_element() -> GrammarElement {
    let nonzero = GrammarElement::CharClass(vec![CharRange { lo: b'1', hi: b'9' }]);
    GrammarElement::Sequence(vec![
        GrammarElement::Repeat(Box::new(literal("-")), RepeatMode::Optional),
        GrammarElement::Choice(vec![
            literal("0"),
            GrammarElement::Sequence(vec![
                nonzero,
                GrammarElement::Repeat(Box::new(digit()), RepeatMode::ZeroOrMore),
            ]),
        ]),
    ])
}

/// Number: an integer, an optional `.digits` fraction, an optional
/// exponent (`e` or `E`, an optional sign, digits).
fn number_element() -> GrammarElement {
    let fraction = GrammarElement::Sequence(vec![
        literal("."),
        GrammarElement::Repeat(Box::new(digit()), RepeatMode::OneOrMore),
    ]);
    let exponent = GrammarElement::Sequence(vec![
        GrammarElement::Choice(vec![literal("e"), literal("E")]),
        GrammarElement::Repeat(
            Box::new(GrammarElement::Choice(vec![literal("+"), literal("-")])),
            RepeatMode::Optional,
        ),
        GrammarElement::Repeat(Box::new(digit()), RepeatMode::OneOrMore),
    ]);

    GrammarElement::Sequence(vec![
        integer_element(),
        GrammarElement::Repeat(Box::new(fraction), RepeatMode::Optional),
        GrammarElement::Repeat(Box::new(exponent), RepeatMode::Optional),
    ])
}

/// One character of a JSON string that is not an escape: any Unicode
/// scalar value except `"`, `\` and U+0000..U+001F, as its UTF-8 bytes.
/// The multi-byte alternatives are the well-formed UTF-8 sequences
/// (Unicode 15, table 3-7), so no overlong form and no surrogate is
/// accepted.
fn string_char() -> GrammarElement {
    let range = |lo: u8, hi: u8| GrammarElement::CharClass(vec![CharRange { lo, hi }]);
    let tail = || range(0x80, 0xBF);
    let seq = |parts: Vec<GrammarElement>| GrammarElement::Sequence(parts);
    GrammarElement::Choice(vec![
        GrammarElement::CharClass(vec![
            CharRange { lo: 0x20, hi: 0x21 }, // space, !
            CharRange { lo: 0x23, hi: 0x5B }, // # through [
            CharRange { lo: 0x5D, hi: 0x7F }, // ] through DEL
        ]),
        seq(vec![range(0xC2, 0xDF), tail()]),
        seq(vec![range(0xE0, 0xE0), range(0xA0, 0xBF), tail()]),
        seq(vec![range(0xE1, 0xEC), tail(), tail()]),
        seq(vec![range(0xED, 0xED), range(0x80, 0x9F), tail()]),
        seq(vec![range(0xEE, 0xEF), tail(), tail()]),
        seq(vec![range(0xF0, 0xF0), range(0x90, 0xBF), tail(), tail()]),
        seq(vec![range(0xF1, 0xF3), tail(), tail(), tail()]),
        seq(vec![range(0xF4, 0xF4), range(0x80, 0x8F), tail(), tail()]),
    ])
}

/// A JSON escape: `\` then one of `"\/bfnrt`, or `u` and four hex digits.
fn string_escape() -> GrammarElement {
    let hex = || {
        GrammarElement::CharClass(vec![
            CharRange { lo: b'0', hi: b'9' },
            CharRange { lo: b'A', hi: b'F' },
            CharRange { lo: b'a', hi: b'f' },
        ])
    };
    let simple = GrammarElement::CharClass(
        b"\"\\/bfnrt".iter().map(|&c| CharRange { lo: c, hi: c }).collect(),
    );
    GrammarElement::Sequence(vec![
        literal("\\"),
        GrammarElement::Choice(vec![
            simple,
            GrammarElement::Sequence(vec![literal("u"), hex(), hex(), hex(), hex()]),
        ]),
    ])
}

/// String: `"`, zero or more characters or escapes, `"`.
fn string_element() -> GrammarElement {
    GrammarElement::Sequence(vec![
        literal("\""),
        GrammarElement::Repeat(
            Box::new(GrammarElement::Choice(vec![string_char(), string_escape()])),
            RepeatMode::ZeroOrMore,
        ),
        literal("\""),
    ])
}

/// Where JSON allows whitespace between tokens: nothing, or one space.
fn ws() -> GrammarElement {
    GrammarElement::Repeat(Box::new(literal(" ")), RepeatMode::Optional)
}

/// `text` as a JSON string literal, quoted and escaped.
fn json_string(text: &str) -> String {
    serde_json::Value::String(text.to_string()).to_string()
}

/// Convert a JSON Schema `enum` to a choice between literal values.
fn convert_enum(enum_values: &serde_json::Value) -> Result<GrammarElement, GrammarError> {
    let arr = enum_values
        .as_array()
        .ok_or_else(|| GrammarError::InvalidSchema("enum must be an array".into()))?;

    if arr.is_empty() {
        return Err(GrammarError::InvalidSchema(
            "enum must have at least one value".into(),
        ));
    }

    let choices: Vec<GrammarElement> = arr
        .iter()
        .map(|v| {
            match v {
                serde_json::Value::String(s) => {
                    // As a JSON string: quoted, and escaped where the
                    // value holds a quote, a backslash or a control
                    // character.
                    GrammarElement::Literal(json_string(s))
                }
                serde_json::Value::Number(n) => GrammarElement::Literal(n.to_string()),
                serde_json::Value::Bool(b) => GrammarElement::Literal(b.to_string()),
                serde_json::Value::Null => GrammarElement::Literal("null".into()),
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
    let arr = one_of
        .as_array()
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

    // Array: "[" ws ((item ("," ws item)*) ws)? "]"
    let comma_item = GrammarElement::Sequence(vec![literal(","), ws(), item_element.clone()]);

    let items_list = GrammarElement::Sequence(vec![
        item_element,
        GrammarElement::Repeat(Box::new(comma_item), RepeatMode::ZeroOrMore),
    ]);

    Ok(GrammarElement::Sequence(vec![
        literal("["),
        ws(),
        GrammarElement::Repeat(
            Box::new(GrammarElement::Sequence(vec![items_list, ws()])),
            RepeatMode::Optional,
        ),
        literal("]"),
    ]))
}

/// Convert a JSON Schema `object` to a grammar element.
///
/// `{ "key": value, ... }`: the properties in name order, those listed in
/// `required` mandatory and the others optional, with a comma between the
/// members that are present and nowhere else.
fn convert_object(
    schema: &serde_json::Value,
    rules: &mut Vec<Rule>,
    prefix: &str,
) -> Result<GrammarElement, GrammarError> {
    let empty = serde_json::Map::new();
    let props = match schema.get("properties") {
        None => &empty,
        Some(v) => v
            .as_object()
            .ok_or_else(|| GrammarError::InvalidSchema("properties must be an object".into()))?,
    };
    let required: Vec<&str> = match schema.get("required") {
        None => Vec::new(),
        Some(v) => v
            .as_array()
            .ok_or_else(|| GrammarError::InvalidSchema("required must be an array".into()))?
            .iter()
            .map(|name| {
                name.as_str()
                    .ok_or_else(|| GrammarError::InvalidSchema("required must list strings".into()))
            })
            .collect::<Result<_, _>>()?,
    };
    if let Some(missing) = required.iter().find(|name| !props.contains_key(**name)) {
        return Err(GrammarError::InvalidSchema(format!(
            "required names `{missing}`, which is not in properties"
        )));
    }

    // (member, is it required), in the order members are written.
    let mut members: Vec<(GrammarElement, bool)> = Vec::with_capacity(props.len());
    for (key, value_schema) in props {
        let value_prefix = format!("{prefix}_{key}");
        let value_element = convert_schema_value(value_schema, rules, &value_prefix)?;
        let member = GrammarElement::Sequence(vec![
            GrammarElement::Literal(json_string(key)),
            ws(),
            literal(":"),
            ws(),
            value_element,
        ]);
        members.push((member, required.contains(&key.as_str())));
    }

    // After a first member, each later one is `, member`, optional when
    // the property is.
    let rest_after = |from: usize| -> GrammarElement {
        GrammarElement::Sequence(
            members[from..]
                .iter()
                .map(|(member, required)| {
                    let item = GrammarElement::Sequence(vec![literal(","), ws(), member.clone()]);
                    if *required {
                        item
                    } else {
                        GrammarElement::Repeat(Box::new(item), RepeatMode::Optional)
                    }
                })
                .collect(),
        )
    };
    // The first member present is member `j` when every member before it
    // is optional and absent; there may be none when all are optional.
    let mut bodies: Vec<GrammarElement> = Vec::new();
    for (j, (member, required)) in members.iter().enumerate() {
        bodies.push(GrammarElement::Sequence(vec![member.clone(), rest_after(j + 1)]));
        if *required {
            break;
        }
    }
    let all_optional = members.iter().all(|(_, required)| !required);
    let body = GrammarElement::Choice(bodies);

    let mut object = vec![literal("{"), ws()];
    if members.is_empty() {
        // `{}` (or `{ }`): no member to write.
    } else if all_optional {
        object.push(GrammarElement::Repeat(
            Box::new(GrammarElement::Sequence(vec![body, ws()])),
            RepeatMode::Optional,
        ));
    } else {
        object.push(body);
        object.push(ws());
    }
    object.push(literal("}"));
    Ok(GrammarElement::Sequence(object))
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

        let (name, body) = line.split_once(':').ok_or_else(|| {
            GrammarError::BnfParseError(format!("expected 'rule_name: body', got: {line}"))
        })?;

        let name = name.trim().to_string();
        let body = body.trim();

        if start_rule.is_empty() {
            start_rule = name.clone();
        }

        let alternatives = parse_bnf_alternatives(body)?;

        rules.push(Rule { name, alternatives });
    }

    if rules.is_empty() {
        return Err(GrammarError::BnfParseError("empty grammar".into()));
    }

    Ok(Grammar { rules, start_rule })
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
                    None => {
                        return Err(GrammarError::BnfParseError(
                            "unterminated string literal".into(),
                        ))
                    }
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
            },
            "required": ["name", "age"]
        });
        let grammar = json_schema_to_grammar(&schema).unwrap();
        let dfa = compile_grammar(&grammar);
        // Members are written in name order: age before name.
        assert!(dfa.accepts(b"{\"age\":25,\"name\":\"alice\"}"));
        assert!(dfa.accepts(b"{\"age\":-1,\"name\":\"bob\"}"));
        // With the one optional space where whitespace may go.
        assert!(dfa.accepts(b"{ \"age\" : 42, \"name\" : \"x\" }"));
        // Both are required.
        assert!(!dfa.accepts(b"{}"));
        assert!(!dfa.accepts(b"{\"age\":25}"));
        assert!(!dfa.accepts(b"\"hello\""));
    }

    fn dfa_for(schema: serde_json::Value) -> crate::grammar_compiler::DFA {
        compile_grammar(&json_schema_to_grammar(&schema).unwrap())
    }

    /// Every string of at most `max_len` bytes drawn from `alphabet` that
    /// `dfa` accepts.
    fn accepted_strings(dfa: &crate::grammar_compiler::DFA, alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
        fn walk(
            dfa: &crate::grammar_compiler::DFA,
            alphabet: &[u8],
            left: usize,
            state: u32,
            text: &mut Vec<u8>,
            out: &mut Vec<Vec<u8>>,
        ) {
            if dfa.accept.contains(&state) {
                out.push(text.clone());
            }
            if left == 0 {
                return;
            }
            for &b in alphabet {
                if let Some(next) = dfa.transitions[state as usize][b as usize] {
                    text.push(b);
                    walk(dfa, alphabet, left - 1, next, text, out);
                    text.pop();
                }
            }
        }
        let mut out = Vec::new();
        walk(dfa, alphabet, max_len, dfa.start, &mut Vec::new(), &mut out);
        out
    }

    /// Soundness against an independent JSON implementation: everything
    /// the grammar accepts (over a sampled alphabet, up to a length) is
    /// parsed by `serde_json`, and is a value of the schema's type.
    fn assert_all_accepted_are_json(
        schema: serde_json::Value,
        alphabet: &[u8],
        max_len: usize,
        is_right_type: impl Fn(&serde_json::Value) -> bool,
    ) -> usize {
        let dfa = dfa_for(schema.clone());
        let accepted = accepted_strings(&dfa, alphabet, max_len);
        for text in &accepted {
            let shown = String::from_utf8_lossy(text).into_owned();
            let value: serde_json::Value = serde_json::from_slice(text)
                .unwrap_or_else(|e| panic!("{schema}: the grammar accepts {shown:?} ({text:?}), which is not JSON: {e}"));
            assert!(is_right_type(&value), "{schema}: the grammar accepts {shown:?}, a {value:?}");
        }
        accepted.len()
    }

    #[test]
    fn integers_have_no_leading_zeros() {
        let dfa = dfa_for(serde_json::json!({"type": "integer"}));
        for ok in ["0", "-0", "7", "42", "-7", "100", "9007199254740993"] {
            assert!(dfa.accepts(ok.as_bytes()), "{ok}");
        }
        // `00` used to be accepted; no JSON parser reads it.
        for bad in ["00", "01", "-01", "007", "", "-", "--1", "+1", "1.0", "1e3"] {
            assert!(!dfa.accepts(bad.as_bytes()), "{bad}");
            if !bad.is_empty() && bad != "1.0" && bad != "1e3" {
                assert!(serde_json::from_str::<serde_json::Value>(bad).is_err(), "{bad} is not JSON");
            }
        }
        // `-0` is a JSON number with no fraction, so an integer; serde_json
        // reads it as the float -0.0.
        let n = assert_all_accepted_are_json(
            serde_json::json!({"type": "integer"}),
            b"0129-.e+",
            6,
            |v| v.is_i64() || v.is_u64() || v.as_f64() == Some(0.0),
        );
        assert!(n > 1000, "the enumeration found {n} integers");
    }

    #[test]
    fn numbers_are_json_numbers() {
        let dfa = dfa_for(serde_json::json!({"type": "number"}));
        for ok in ["0", "-0", "1.5", "-0.25", "10", "1e10", "1E-3", "2.5e+7", "0.0"] {
            assert!(dfa.accepts(ok.as_bytes()), "{ok}");
            assert!(serde_json::from_str::<serde_json::Value>(ok).is_ok(), "{ok}");
        }
        for bad in ["1.", ".5", "1e", "01.5", "1.e3", "1e+", "--1", "1.5.2", "e3", "Infinity", "NaN"] {
            assert!(!dfa.accepts(bad.as_bytes()), "{bad}");
            assert!(serde_json::from_str::<serde_json::Value>(bad).is_err(), "{bad} is not JSON");
        }
        // Five bytes over these digits keep every exponent below 223:
        // serde_json refuses a number past f64's range (`1e999`), which is
        // its limit and not a rule of JSON's grammar.
        let n = assert_all_accepted_are_json(serde_json::json!({"type": "number"}), b"012-.eE+", 5, |v| v.is_number());
        assert!(n > 300, "the enumeration found {n} numbers");
    }

    #[test]
    fn strings_take_json_escapes_and_utf8_and_nothing_else() {
        let dfa = dfa_for(serde_json::json!({"type": "string"}));
        let ok: [&[u8]; 9] = [
            b"\"\"",
            b"\"plain\"",
            b"\"a\\\"b\"",        // an escaped quote
            b"\"back\\\\slash\"", // an escaped backslash
            b"\"tab\\there\"",    // \t
            b"\"\\u00e9\"",       // \u00e9
            "\"caf\u{e9}\"".as_bytes(), // e-acute as UTF-8
            "\"\u{4e2d}\u{6587}\"".as_bytes(), // three-byte sequences
            "\"\u{1f600}\"".as_bytes(), // a four-byte sequence
        ];
        for text in ok {
            assert!(dfa.accepts(text), "{:?}", String::from_utf8_lossy(text));
            assert!(serde_json::from_slice::<serde_json::Value>(text).is_ok(), "{text:?} is JSON");
        }
        let bad: [&[u8]; 10] = [
            b"\"line\nbreak\"",      // a raw control character
            b"\"tab\there\"",        // a raw tab
            b"\"lone\\\"",           // the closing quote is escaped
            b"\"\\x41\"",            // not a JSON escape
            b"\"\\u12\"",            // a short \u
            b"\"\\u12G4\"",          // not hex
            b"\"\xff\"",             // not UTF-8
            b"\"\xc0\x80\"",         // an overlong encoding
            b"\"\xed\xa0\x80\"",     // a surrogate
            b"\"\xe4\xb8\"",         // a truncated sequence
        ];
        for text in bad {
            assert!(!dfa.accepts(text), "{:?}", String::from_utf8_lossy(text));
            assert!(serde_json::from_slice::<serde_json::Value>(text).is_err(), "{text:?} is not JSON");
        }
        // 0xC3 0xA9 is e-acute; 0x09 a raw tab; 0x80 a stray continuation.
        let n = assert_all_accepted_are_json(
            serde_json::json!({"type": "string"}),
            b"\"\\/un0Fa \xc3\xa9\x09\x80",
            7,
            |v| v.is_string(),
        );
        assert!(n > 1000, "the enumeration found {n} strings");
    }

    #[test]
    fn required_properties_are_required_and_the_others_optional() {
        let schema = |required: serde_json::Value| {
            serde_json::json!({
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}, "c": {"type": "null"}},
                "required": required
            })
        };
        let dfa = dfa_for(schema(serde_json::json!(["b"])));
        for ok in [
            r#"{"b":true}"#,
            r#"{"a":1,"b":true}"#,
            r#"{"b":false,"c":null}"#,
            r#"{"a":1,"b":false,"c":null}"#,
            r#"{ "a" : 1, "b" : true }"#,
        ] {
            assert!(dfa.accepts(ok.as_bytes()), "{ok}");
        }
        for bad in [
            "{}",
            r#"{"a":1}"#,
            r#"{"a":1,"c":null}"#,
            r#"{"b":true,"a":1}"#, // members are written in name order
            r#"{,"b":true}"#,
            r#"{"a":1,"b":true,}"#,
            r#"{"a":1 "b":true}"#,
            r#"{"b":true,"b":true}"#,
        ] {
            assert!(!dfa.accepts(bad.as_bytes()), "{bad}");
        }

        // No `required`: every property is optional, `{}` included. They
        // all used to be mandatory.
        let dfa = dfa_for(serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}, "c": {"type": "null"}}
        }));
        for ok in ["{}", "{ }", r#"{"a":1}"#, r#"{"c":null}"#, r#"{"a":1,"c":null}"#, r#"{"a":1,"b":true,"c":null}"#] {
            assert!(dfa.accepts(ok.as_bytes()), "{ok}");
        }
        for bad in [r#"{,"a":1}"#, r#"{"a":1,}"#, r#"{"c":null,"a":1}"#] {
            assert!(!dfa.accepts(bad.as_bytes()), "{bad}");
        }

        // Everything accepted is JSON, an object, and carries `b`.
        let n = assert_all_accepted_are_json(schema(serde_json::json!(["b"])), b"{}\"abc:,1truefalsn", 30, |v| {
            v.as_object().is_some_and(|o| o.contains_key("b") && o.keys().all(|k| ["a", "b", "c"].contains(&k.as_str())))
        });
        assert!(n >= 8, "the enumeration found {n} objects");

        let err = json_schema_to_grammar(&schema(serde_json::json!(["zzz"]))).unwrap_err();
        assert!(err.to_string().contains("required names `zzz`"), "{err}");
    }

    #[test]
    fn enum_strings_and_property_names_are_escaped() {
        let dfa = dfa_for(serde_json::json!({"enum": ["a\"b", "back\\slash"]}));
        assert!(dfa.accepts(br#""a\"b""#));
        assert!(dfa.accepts(br#""back\\slash""#));
        assert!(!dfa.accepts(br#""a"b""#), "the unescaped form is not JSON");
        let dfa = dfa_for(serde_json::json!({
            "type": "object",
            "properties": {"k\"": {"type": "null"}},
            "required": ["k\""]
        }));
        assert!(dfa.accepts(br#"{"k\"":null}"#));
    }

    /// At most one space where whitespace may go. An unbounded run let a
    /// constrained model emit spaces for ever.
    #[test]
    fn whitespace_is_one_space_at_most() {
        let dfa = dfa_for(serde_json::json!({"type": "array", "items": {"type": "integer"}}));
        for ok in ["[]", "[ ]", "[1]", "[1,2]", "[1, 2]", "[ 1, 2 ]"] {
            assert!(dfa.accepts(ok.as_bytes()), "{ok}");
        }
        for bad in ["[1,  2]", "[  1]", "[  ]", "[1 ,2]", "[1,\t2]", "[1,\n2]"] {
            assert!(!dfa.accepts(bad.as_bytes()), "{bad}");
        }
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
