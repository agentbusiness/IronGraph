//! Multiline-safe Cypher tokenizer with comments, escaped strings, and quoted names.

use crate::{Error, ErrorCode, Result};

/// Byte and source-position range for precise driver-visible diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: u32,
    pub column: u32,
}

/// Cypher punctuation and operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symbol {
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Colon,
    Semicolon,
    Dollar,
    Pipe,
    DoublePipe,
    Ampersand,
    Bang,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    RegexMatch,
    ArrowLeft,
    ArrowRight,
    Range,
}

/// Lexical value. Keywords remain identifiers and are matched case-insensitively by the parser.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    Identifier(String),
    String(String),
    /// Unsigned lexical magnitude. A leading sign is a separate token and signed range
    /// validation belongs to the parser.
    Integer(u64),
    Float(f64),
    Parameter(String),
    Symbol(Symbol),
    End,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub fn lex(source: &str) -> Result<Vec<Token>> {
    Lexer::new(source).collect()
}

/// Stateful UTF-8 lexer. It advances by Unicode scalar boundaries, never raw bytes.
pub struct Lexer<'a> {
    source: &'a str,
    offset: usize,
    line: u32,
    column: u32,
    ended: bool,
}

impl<'a> Lexer<'a> {
    #[must_use]
    pub const fn new(source: &'a str) -> Self {
        Self {
            source,
            offset: 0,
            line: 1,
            column: 1,
            ended: false,
        }
    }

    pub fn collect(mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            let end = matches!(token.kind, TokenKind::End);
            tokens.push(token);
            if end {
                return Ok(tokens);
            }
        }
    }

    pub fn next_token(&mut self) -> Result<Token> {
        if self.ended {
            return Ok(self.token(TokenKind::End, self.offset, self.line, self.column));
        }
        self.skip_layout()?;
        let start = self.offset;
        let line = self.line;
        let column = self.column;
        let Some(character) = self.peek() else {
            self.ended = true;
            return Ok(self.token(TokenKind::End, start, line, column));
        };
        if character == '`' {
            return self.quoted_identifier(start, line, column);
        }
        if character == '\'' || character == '"' {
            return self.string(character, start, line, column);
        }
        if character == '$' {
            if self.remaining().starts_with("$(") {
                self.bump();
                return Ok(self.token_at(TokenKind::Symbol(Symbol::Dollar), start, line, column));
            }
            self.bump();
            let name = self.take_identifier();
            if name.is_empty() {
                return self.syntax("parameter name is missing", start, line, column);
            }
            return Ok(self.token_at(TokenKind::Parameter(name), start, line, column));
        }
        // A decimal literal may omit its integer digits (`.1`, `.1e-5`). Keep `..` reserved for
        // list slicing by requiring the following character to be a digit before taking this
        // numeric path.
        if character.is_ascii_digit()
            || (character == '.'
                && self
                    .remaining()
                    .as_bytes()
                    .get(1)
                    .is_some_and(u8::is_ascii_digit))
        {
            return self.number(start, line, column);
        }
        if is_identifier_start(character) {
            let identifier = self.take_identifier();
            return Ok(self.token_at(TokenKind::Identifier(identifier), start, line, column));
        }
        self.symbol(start, line, column)
    }

    fn skip_layout(&mut self) -> Result<()> {
        loop {
            while self.peek().is_some_and(char::is_whitespace) {
                self.bump();
            }
            if self.remaining().starts_with("//") {
                while self.peek().is_some_and(|character| character != '\n') {
                    self.bump();
                }
                continue;
            }
            if self.remaining().starts_with("/*") {
                let start = self.offset;
                let line = self.line;
                let column = self.column;
                self.bump();
                self.bump();
                let mut depth = 1_u32;
                while depth != 0 {
                    if self.remaining().starts_with("/*") {
                        self.bump();
                        self.bump();
                        depth = depth.saturating_add(1);
                    } else if self.remaining().starts_with("*/") {
                        self.bump();
                        self.bump();
                        depth = depth.saturating_sub(1);
                    } else if self.peek().is_some() {
                        self.bump();
                    } else {
                        return self.syntax("unterminated block comment", start, line, column);
                    }
                }
                continue;
            }
            return Ok(());
        }
    }

    fn quoted_identifier(&mut self, start: usize, line: u32, column: u32) -> Result<Token> {
        self.bump();
        let mut value = String::new();
        loop {
            let Some(character) = self.peek() else {
                return self.syntax("unterminated quoted identifier", start, line, column);
            };
            self.bump();
            if character == '`' {
                if self.peek() == Some('`') {
                    self.bump();
                    value.push('`');
                    continue;
                }
                break;
            }
            value.push(character);
        }
        // A quoted symbolic name may be empty.  In particular, an empty map key (`{``: ...}`)
        // is grammatically valid in the openCypher TCK; its surrounding expression is still
        // responsible for any later type error.  Rejecting it here incorrectly masks those
        // semantic diagnostics as a lexer error.
        Ok(self.token_at(TokenKind::Identifier(value), start, line, column))
    }

    fn string(&mut self, quote: char, start: usize, line: u32, column: u32) -> Result<Token> {
        self.bump();
        let mut value = String::new();
        loop {
            let Some(character) = self.peek() else {
                return self.syntax("unterminated string literal", start, line, column);
            };
            self.bump();
            if character == quote {
                if self.peek() == Some(quote) {
                    self.bump();
                    value.push(quote);
                    continue;
                }
                break;
            }
            if character == '\\' {
                let Some(escaped) = self.peek() else {
                    return self.syntax("unterminated string escape", start, line, column);
                };
                self.bump();
                match escaped {
                    'n' => value.push('\n'),
                    'r' => value.push('\r'),
                    't' => value.push('\t'),
                    'b' => value.push('\u{0008}'),
                    'f' => value.push('\u{000c}'),
                    '\\' => value.push('\\'),
                    '\'' => value.push('\''),
                    '"' => value.push('"'),
                    'u' => value.push(self.unicode_escape(4, start, line, column)?),
                    'U' => value.push(self.unicode_escape(8, start, line, column)?),
                    _ => return self.syntax("invalid string escape", start, line, column),
                }
            } else {
                value.push(character);
            }
        }
        Ok(self.token_at(TokenKind::String(value), start, line, column))
    }

    fn number(&mut self, start: usize, line: u32, column: u32) -> Result<Token> {
        if self.remaining().starts_with("0x") || self.remaining().starts_with("0X") {
            return self.radix_integer(16, "hexadecimal", start, line, column);
        }
        if self.remaining().starts_with("0o") || self.remaining().starts_with("0O") {
            return self.radix_integer(8, "octal", start, line, column);
        }
        let integer = self.take_while(|character| character.is_ascii_digit() || character == '_');
        let mut text = integer;
        let mut floating = false;
        if self.peek() == Some('.') && !self.remaining().starts_with("..") {
            floating = true;
            text.push('.');
            self.bump();
            text.push_str(
                &self.take_while(|character| character.is_ascii_digit() || character == '_'),
            );
        }
        if self
            .peek()
            .is_some_and(|character| character == 'e' || character == 'E')
        {
            floating = true;
            if let Some(character) = self.bump() {
                text.push(character);
            }
            if self
                .peek()
                .is_some_and(|character| character == '+' || character == '-')
            {
                if let Some(character) = self.bump() {
                    text.push(character);
                }
            }
            text.push_str(
                &self.take_while(|character| character.is_ascii_digit() || character == '_'),
            );
        }
        let cleaned = text.replace('_', "");
        if floating {
            let value = cleaned
                .parse::<f64>()
                .map_err(|_| self.error("invalid floating-point literal", start, line, column))?;
            if !value.is_finite() {
                return self.syntax(
                    "FloatingPointOverflow: floating-point literal is not finite",
                    start,
                    line,
                    column,
                );
            }
            Ok(self.token_at(TokenKind::Float(value), start, line, column))
        } else {
            if self.peek().is_some_and(is_identifier_continue) {
                let _ = self.take_while(is_identifier_continue);
                if self.remaining().trim_start().starts_with(':') {
                    return self.syntax(
                        "a symbolic name cannot start with a decimal digit",
                        start,
                        line,
                        column,
                    );
                }
                return Self::syntax_detail(
                    "InvalidNumberLiteral",
                    "invalid decimal integer literal",
                    start,
                    line,
                    column,
                );
            }
            let magnitude = cleaned.parse::<u64>().map_err(|_| {
                Self::detail_error(
                    "IntegerOverflow",
                    "integer literal is out of range",
                    start,
                    line,
                    column,
                )
            })?;
            Ok(self.token_at(TokenKind::Integer(magnitude), start, line, column))
        }
    }

    fn radix_integer(
        &mut self,
        radix: u32,
        name: &'static str,
        start: usize,
        line: u32,
        column: u32,
    ) -> Result<Token> {
        self.bump();
        self.bump();

        // A radix literal and an immediately adjacent identifier-like suffix form one numeric
        // token. Consuming that entire boundary prevents a valid prefix (`0x1A2b3`) from hiding
        // an invalid suffix (`j4D5E6f7`) behind a later generic parser error.
        let digits = self.take_while(is_identifier_continue);
        let valid = digits.chars().any(|character| character != '_')
            && digits.chars().all(|character| {
                character == '_' || (character.is_ascii() && character.is_digit(radix))
            });
        if !valid {
            return Self::syntax_detail(
                "InvalidNumberLiteral",
                match name {
                    "hexadecimal" => "invalid hexadecimal integer literal",
                    "octal" => "invalid octal integer literal",
                    _ => "invalid radix integer literal",
                },
                start,
                line,
                column,
            );
        }

        let cleaned = digits.replace('_', "");
        let value = u64::from_str_radix(&cleaned, radix).map_err(|_| {
            Self::detail_error(
                "IntegerOverflow",
                "integer literal is out of range",
                start,
                line,
                column,
            )
        })?;
        Ok(self.token_at(TokenKind::Integer(value), start, line, column))
    }

    fn unicode_escape(
        &mut self,
        width: usize,
        start: usize,
        line: u32,
        column: u32,
    ) -> Result<char> {
        let mut digits = String::with_capacity(width);
        for _ in 0..width {
            let Some(character) = self.peek() else {
                return Self::syntax_detail(
                    "InvalidUnicodeLiteral",
                    "truncated Unicode escape",
                    start,
                    line,
                    column,
                );
            };
            if !character.is_ascii_hexdigit() {
                return Self::syntax_detail(
                    "InvalidUnicodeLiteral",
                    "invalid Unicode escape",
                    start,
                    line,
                    column,
                );
            }
            self.bump();
            digits.push(character);
        }
        let scalar = u32::from_str_radix(&digits, 16).map_err(|_| {
            Self::detail_error(
                "InvalidUnicodeLiteral",
                "invalid Unicode escape",
                start,
                line,
                column,
            )
        })?;
        char::from_u32(scalar).ok_or_else(|| {
            Self::detail_error(
                "InvalidUnicodeLiteral",
                "Unicode escape is not a scalar value",
                start,
                line,
                column,
            )
        })
    }

    fn symbol(&mut self, start: usize, line: u32, column: u32) -> Result<Token> {
        let (symbol, width) = if self.remaining().starts_with("<-") {
            (Symbol::ArrowLeft, 2)
        } else if self.remaining().starts_with("->") {
            (Symbol::ArrowRight, 2)
        } else if self.remaining().starts_with("<=") {
            (Symbol::LessOrEqual, 2)
        } else if self.remaining().starts_with(">=") {
            (Symbol::GreaterOrEqual, 2)
        } else if self.remaining().starts_with("<>") || self.remaining().starts_with("!=") {
            (Symbol::NotEqual, 2)
        } else if self.remaining().starts_with("=~") {
            (Symbol::RegexMatch, 2)
        } else if self.remaining().starts_with("||") {
            (Symbol::DoublePipe, 2)
        } else if self.remaining().starts_with("..") {
            (Symbol::Range, 2)
        } else {
            let symbol = match self.peek() {
                Some('(') => Symbol::LeftParen,
                Some(')') => Symbol::RightParen,
                Some('[') => Symbol::LeftBracket,
                Some(']') => Symbol::RightBracket,
                Some('{') => Symbol::LeftBrace,
                Some('}') => Symbol::RightBrace,
                Some(',') => Symbol::Comma,
                Some('.') => Symbol::Dot,
                Some(':') => Symbol::Colon,
                Some(';') => Symbol::Semicolon,
                Some('|') => Symbol::Pipe,
                Some('&') => Symbol::Ampersand,
                Some('!') => Symbol::Bang,
                Some('+') => Symbol::Plus,
                Some('-') => Symbol::Minus,
                Some('*') => Symbol::Star,
                Some('/') => Symbol::Slash,
                Some('%') => Symbol::Percent,
                Some('^') => Symbol::Caret,
                Some('=') => Symbol::Equal,
                Some('<') => Symbol::Less,
                Some('>') => Symbol::Greater,
                Some(character) if !character.is_ascii() => {
                    return Self::syntax_detail(
                        "InvalidUnicodeCharacter",
                        "Unicode character is not valid Cypher punctuation",
                        start,
                        line,
                        column,
                    );
                }
                _ => return self.syntax("unexpected character", start, line, column),
            };
            (symbol, 1)
        };
        for _ in 0..width {
            self.bump();
        }
        Ok(self.token_at(TokenKind::Symbol(symbol), start, line, column))
    }

    fn take_identifier(&mut self) -> String {
        self.take_while(is_identifier_continue)
    }

    fn take_while(&mut self, predicate: impl Fn(char) -> bool) -> String {
        let mut result = String::new();
        while self.peek().is_some_and(&predicate) {
            if let Some(character) = self.bump() {
                result.push(character);
            }
        }
        result
    }

    fn remaining(&self) -> &'a str {
        &self.source[self.offset..]
    }

    fn peek(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.offset += character.len_utf8();
        if character == '\n' {
            self.line = self.line.saturating_add(1);
            self.column = 1;
        } else {
            self.column = self.column.saturating_add(1);
        }
        Some(character)
    }

    fn token(&self, kind: TokenKind, start: usize, line: u32, column: u32) -> Token {
        Token {
            kind,
            span: Span {
                start,
                end: start,
                line,
                column,
            },
        }
    }

    fn token_at(&self, kind: TokenKind, start: usize, line: u32, column: u32) -> Token {
        Token {
            kind,
            span: Span {
                start,
                end: self.offset,
                line,
                column,
            },
        }
    }

    fn error(&self, message: &'static str, start: usize, line: u32, column: u32) -> Error {
        Error::new(
            ErrorCode::QuerySyntax,
            format!("UnexpectedSyntax: {message} at {line}:{column} (byte {start})"),
        )
    }

    fn detail_error(
        detail: &'static str,
        message: &'static str,
        start: usize,
        line: u32,
        column: u32,
    ) -> Error {
        Error::new(
            ErrorCode::QuerySyntax,
            format!("{detail}: {message} at {line}:{column} (byte {start})"),
        )
    }

    fn syntax<T>(&self, message: &'static str, start: usize, line: u32, column: u32) -> Result<T> {
        Err(self.error(message, start, line, column))
    }

    fn syntax_detail<T>(
        detail: &'static str,
        message: &'static str,
        start: usize,
        line: u32,
        column: u32,
    ) -> Result<T> {
        Err(Self::detail_error(detail, message, start, line, column))
    }
}

fn is_identifier_start(character: char) -> bool {
    character == '_' || character.is_alphabetic()
}

fn is_identifier_continue(character: char) -> bool {
    character == '_' || character.is_alphanumeric()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{Symbol, TokenKind, lex};

    #[test]
    fn multiline_layout_strings_and_numeric_forms_are_lossless() {
        let tokens =
            lex("/* outer /* nested */ */ RETURN 0x2a, 0o52, 'line\\n\\u03bb', a =~ 'x' || 'y'");
        assert!(tokens.is_ok());
        let tokens = tokens.unwrap_or_default();
        assert!(
            tokens
                .iter()
                .any(|token| token.kind == TokenKind::Integer(42))
        );
        assert!(
            tokens
                .iter()
                .any(|token| { token.kind == TokenKind::String("line\nλ".to_owned()) })
        );
        assert!(
            tokens
                .iter()
                .any(|token| { token.kind == TokenKind::Symbol(Symbol::RegexMatch) })
        );
        assert!(
            tokens
                .iter()
                .any(|token| { token.kind == TokenKind::Symbol(Symbol::DoublePipe) })
        );
    }

    #[test]
    fn malformed_unicode_and_unterminated_comments_are_syntax_errors() {
        let unicode = lex("RETURN '\\uH'");
        assert!(matches!(unicode, Err(error) if error.message.contains("InvalidUnicodeLiteral")));
        let surrogate = lex("RETURN '\\uD800'");
        assert!(matches!(surrogate, Err(error) if error.message.contains("InvalidUnicodeLiteral")));
        assert!(lex("RETURN 1 /*").is_err());
    }

    #[test]
    fn digit_prefixed_symbolic_names_and_number_literals_have_distinct_details() {
        let map_key = lex("RETURN {1B2c3e67: 1}");
        assert!(matches!(map_key, Err(error) if error.message.starts_with("UnexpectedSyntax:")));
        let number = lex("RETURN 1B2c3e67");
        assert!(matches!(number, Err(error) if error.message.starts_with("InvalidNumberLiteral:")));
    }

    #[test]
    fn quoted_identifier_may_be_empty() {
        let tokens = lex("RETURN {``: null}").unwrap_or_default();
        assert!(
            tokens
                .iter()
                .any(|token| token.kind == TokenKind::Identifier(String::new()))
        );
    }

    proptest! {
        #[test]
        fn arbitrary_unicode_never_escapes_source_boundaries(
            characters in proptest::collection::vec(any::<char>(), 0..512),
        ) {
            let source: String = characters.into_iter().collect();
            if let Ok(tokens) = lex(&source) {
                prop_assert!(matches!(tokens.last().map(|token| &token.kind), Some(TokenKind::End)));
                for token in tokens {
                    prop_assert!(token.span.start <= token.span.end);
                    prop_assert!(token.span.end <= source.len());
                    prop_assert!(source.is_char_boundary(token.span.start));
                    prop_assert!(source.is_char_boundary(token.span.end));
                    prop_assert!(token.span.line >= 1);
                    prop_assert!(token.span.column >= 1);
                }
            }
        }
    }
}
