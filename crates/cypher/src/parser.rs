//! Recursive-descent clause parser and Pratt expression parser.

use std::sync::Arc;

use crate::{Error, ErrorCode, Layer, Result, ScalarValue, graph::LayerMask};

use super::{
    ast::*,
    lexer::{Span, Symbol, Token, TokenKind, lex},
};

pub fn parse(source: &str) -> Result<Query> {
    Parser::new(source, lex(source)?).parse()
}

struct Parser<'source> {
    source: &'source str,
    tokens: Vec<Token>,
    cursor: usize,
    parsed_label_predicates: usize,
    pattern_predicates_allowed: bool,
    existential_subquery_depth: usize,
    expression_depth: usize,
}

const I64_MIN_MAGNITUDE: u64 = 1_u64 << (i64::BITS - 1);

/// Deepest expression nesting the parser will accept before failing the query.
///
/// Every nesting construct — parentheses, list and map literals, subscripts, `NOT`, unary sign,
/// comprehensions — recurses through `Parser::expression`, so bounding that one function bounds
/// the whole recursive descent. Queries parse on `spawn_blocking` workers with a 2 MiB stack, and a
/// stack overflow there is a guard-page abort that no `panic` setting can catch, so an unbounded
/// parser turns a few hundred bytes of client text into the loss of a process that holds every
/// project. The limit is far above any expression a person or generator
/// writes and far below the depth at which the deepest profile runs out of stack.
const MAXIMUM_EXPRESSION_DEPTH: usize = 128;

/// Longest chain of postfix and infix operators the parser will accept in one expression.
///
/// `a.b.c`, `x[0][1][2]`, and `1 + 1 + 1` extend the expression iteratively, so they never re-enter
/// [`Parser::expression`] and the nesting bound cannot see them. The tree they build is still
/// left-nested, and releasing or walking it recurses once per link — a chain long enough to fit in
/// the query-text limit overflows the stack when the AST is dropped, after parsing has already
/// reported success. This bound sits far above any generated predicate list and far below the
/// shortest chain that has been observed to exhaust a worker stack.
const MAXIMUM_EXPRESSION_CHAIN: usize = 4_096;

struct ParsedCall {
    clause: CallClause,
    implicit_arguments: bool,
    yield_all: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PatternUse {
    Read,
    Predicate,
    Comprehension,
    Merge,
    Write,
}

impl<'source> Parser<'source> {
    fn new(source: &'source str, tokens: Vec<Token>) -> Self {
        Self {
            source,
            tokens,
            cursor: 0,
            parsed_label_predicates: 0,
            pattern_predicates_allowed: false,
            existential_subquery_depth: 0,
            expression_depth: 0,
        }
    }

    fn parse(mut self) -> Result<Query> {
        let mut project = None;
        let mut read_layers = LayerMask::AUTHORITY;
        let mut write_layer = Layer::Observed;
        let mut at_time = None;
        loop {
            if self.eat_word("USE") {
                if self.eat_word("LAYER") {
                    read_layers = self.parse_layer_mask()?;
                } else {
                    if project.is_some() {
                        return self.error("project USE appears more than once");
                    }
                    project = Some(self.identifier()?);
                }
            } else if self.eat_word("WRITE") {
                self.expect_word("LAYER")?;
                write_layer = self.parse_layer()?;
            } else if self.eat_word("AT") {
                self.expect_word("TIME")?;
                at_time = Some(self.expression(0)?);
            } else {
                break;
            }
        }
        let statement = self.statement()?;
        self.eat_symbol(Symbol::Semicolon);
        if !matches!(self.peek().kind, TokenKind::End) {
            return self.error("only one Cypher statement is accepted per execution");
        }
        // The write layer must be visible only when the query actually writes to it; that is enforced
        // in the binder, where read-only vs. write is known. A read-only query may legitimately scope
        // its reads to a single layer (e.g. `USE LAYER KNOWLEDGE`) even though the default write layer
        // is Observed.
        Ok(Query {
            project,
            read_layers,
            write_layer,
            at_time,
            statement,
        })
    }

    fn statement(&mut self) -> Result<Statement> {
        if self.peek_word("IMPORT") {
            self.advance();
            self.expect_word("DATASET")?;
            return Ok(Statement::ImportDataset {
                name: self.identifier()?,
            });
        }
        if self.peek_word("CHECK") {
            self.advance();
            self.expect_word("READ")?;
            self.expect_word("ONLY")?;
            return Ok(Statement::CheckReadOnly);
        }
        if self.peek_word("SHOW") {
            self.advance();
            if self.eat_word("PROJECTS") {
                return Ok(Statement::ShowProjects);
            }
            if self.eat_word("INDEXES") {
                return Ok(Statement::ShowIndexes);
            }
            if self.eat_word("CONSTRAINTS") {
                return Ok(Statement::ShowConstraints);
            }
            if self.eat_word("TOPICS") {
                return Ok(Statement::ShowTopics);
            }
            if self.eat_word("QUEUES") {
                return Ok(Statement::ShowQueues);
            }
            if self.eat_word("EXCHANGES") {
                return Ok(Statement::ShowExchanges);
            }
            if self.eat_word("CONSUMER") {
                self.expect_word("LAG")?;
                return Ok(Statement::ShowConsumerLag);
            }
            return self.error(
                "SHOW supports PROJECTS, INDEXES, CONSTRAINTS, TOPICS, QUEUES, EXCHANGES, or CONSUMER LAG",
            );
        }
        if self.peek_word("REBUILD") {
            self.advance();
            self.expect_word("INDEX")?;
            return Ok(Statement::RebuildIndex {
                name: self.identifier()?,
            });
        }
        if self.peek_word("ALTER") {
            self.advance();
            if self.eat_word("PROJECT") {
                let name = self.identifier()?;
                self.expect_word("RENAME")?;
                self.expect_word("TO")?;
                return Ok(Statement::AlterProjectRename {
                    name,
                    new_name: self.identifier()?,
                });
            }
            if self.eat_word("TOPIC") {
                let name = self.identifier()?;
                self.expect_word("RETENTION")?;
                let retention_days = self
                    .eat_integer_u32()?
                    .ok_or_else(|| Error::invalid_data("topic retention days are required"))?;
                self.expect_word("DAYS")?;
                return Ok(Statement::AlterTopicRetention {
                    name,
                    retention_days,
                });
            }
            if self.eat_word("QUEUE") {
                let name = self.identifier()?;
                self.expect_word("RETENTION")?;
                let retention_days = self
                    .eat_integer_u32()?
                    .ok_or_else(|| Error::invalid_data("queue retention days are required"))?;
                self.expect_word("DAYS")?;
                return Ok(Statement::AlterQueueRetention {
                    name,
                    retention_days,
                });
            }
            let target = if self.eat_word("NODE") {
                TemporalTarget::Node
            } else if self.eat_word("RELATIONSHIP") {
                TemporalTarget::Relationship
            } else {
                return self.error(
                    "ALTER supports PROJECT, TOPIC, QUEUE, NODE PROPERTY, or RELATIONSHIP PROPERTY",
                );
            };
            self.expect_word("PROPERTY")?;
            let label_or_type = self.identifier()?;
            self.expect_symbol(Symbol::Dot)?;
            let property = self.identifier()?;
            self.expect_word("SET")?;
            self.expect_word("TEMPORAL")?;
            let scalar_type = self.temporal_type()?;
            self.expect_word("RETENTION")?;
            let retention = self.expression(0)?;
            return Ok(Statement::DeclareTemporal(TemporalPropertyDeclaration {
                target,
                label_or_type,
                property,
                scalar_type,
                retention,
            }));
        }
        if self.peek_word("DROP") {
            self.advance();
            if self.eat_word("PROJECT") {
                let if_exists = self.eat_word("IF") && {
                    self.expect_word("EXISTS")?;
                    true
                };
                let name = self.identifier()?;
                let cascade = self.eat_word("CASCADE");
                return Ok(Statement::DropProject {
                    name,
                    if_exists,
                    cascade,
                });
            }
            if self.eat_word("INDEX") {
                let if_exists = self.eat_word("IF") && {
                    self.expect_word("EXISTS")?;
                    true
                };
                return Ok(Statement::DropIndex {
                    name: self.identifier()?,
                    if_exists,
                });
            }
            if self.eat_word("CONSTRAINT") {
                let if_exists = self.eat_word("IF") && {
                    self.expect_word("EXISTS")?;
                    true
                };
                return Ok(Statement::DropConstraint {
                    name: self.identifier()?,
                    if_exists,
                });
            }
            if self.eat_word("TOPIC") {
                return Ok(Statement::DropTopic {
                    name: self.identifier()?,
                });
            }
            if self.eat_word("QUEUE") {
                return Ok(Statement::DropQueue {
                    name: self.identifier()?,
                });
            }
            if self.eat_word("EXCHANGE") {
                return Ok(Statement::DropExchange {
                    name: self.identifier()?,
                });
            }
            return self
                .error("DROP supports PROJECT, INDEX, CONSTRAINT, TOPIC, QUEUE, or EXCHANGE");
        }
        if self.peek_word("CLEAR") {
            self.advance();
            self.expect_word("TOPIC")?;
            return Ok(Statement::ClearTopic {
                name: self.identifier()?,
            });
        }
        if self.peek_word("PURGE") {
            self.advance();
            self.expect_word("QUEUE")?;
            return Ok(Statement::PurgeQueue {
                name: self.identifier()?,
            });
        }
        if self.peek_word("BIND") || self.peek_word("UNBIND") {
            let unbind = self.peek_word("UNBIND");
            self.advance();
            self.expect_word("QUEUE")?;
            let queue = self.identifier()?;
            self.expect_word("TO")?;
            self.expect_word("EXCHANGE")?;
            let exchange = self.identifier()?;
            self.expect_word("KEY")?;
            let routing_key = self.identifier()?;
            return Ok(if unbind {
                Statement::UnbindQueue {
                    queue,
                    exchange,
                    routing_key,
                }
            } else {
                Statement::BindQueue {
                    queue,
                    exchange,
                    routing_key,
                }
            });
        }
        if self.peek_word("CREATE") {
            if self.peek_word_at(1, "PROJECT") {
                self.advance();
                self.advance();
                let mut if_not_exists = false;
                if self.eat_word("IF") {
                    self.expect_word("NOT")?;
                    self.expect_word("EXISTS")?;
                    if_not_exists = true;
                }
                return Ok(Statement::CreateProject {
                    name: self.identifier()?,
                    if_not_exists,
                });
            }
            if self.peek_word_at(1, "ROLLUP") {
                self.advance();
                self.advance();
                return self.create_rollup().map(Statement::CreateRollup);
            }
            if self.peek_word_at(1, "TOPIC") {
                self.advance();
                self.advance();
                let name = self.identifier()?;
                self.expect_word("PARTITIONS")?;
                let partitions = self
                    .eat_integer_u32()?
                    .ok_or_else(|| Error::invalid_data("topic partition count is required"))?;
                let partitions = u16::try_from(partitions)
                    .map_err(|_| Error::invalid_data("topic partition count exceeds u16"))?;
                let retention_days = if self.eat_word("RETENTION") {
                    let days = self
                        .eat_integer_u32()?
                        .ok_or_else(|| Error::invalid_data("topic retention days are required"))?;
                    self.expect_word("DAYS")?;
                    Some(days)
                } else {
                    None
                };
                return Ok(Statement::CreateTopic {
                    name,
                    partitions,
                    retention_days,
                });
            }
            if self.peek_word_at(1, "QUEUE") {
                self.advance();
                self.advance();
                let name = self.identifier()?;
                let stream = self.eat_word("STREAM");
                if !stream {
                    let _ = self.eat_word("CLASSIC");
                }
                let retention_days = if self.eat_word("RETENTION") {
                    let days = self
                        .eat_integer_u32()?
                        .ok_or_else(|| Error::invalid_data("queue retention days are required"))?;
                    self.expect_word("DAYS")?;
                    Some(days)
                } else {
                    None
                };
                return Ok(Statement::CreateQueue {
                    name,
                    stream,
                    retention_days,
                });
            }
            if self.peek_word_at(1, "EXCHANGE") {
                self.advance();
                self.advance();
                let name = self.identifier()?;
                self.expect_word("TYPE")?;
                return Ok(Statement::CreateExchange {
                    name,
                    kind: self.identifier()?,
                });
            }
            if self.peek_word_at(1, "EMBEDDING") {
                self.advance();
                self.advance();
                self.expect_word("INDEX")?;
                return self.create_embedding().map(Statement::CreateEmbedding);
            }
            if self.peek_word_at(1, "CONSTRAINT") {
                self.advance();
                self.advance();
                return self
                    .create_unique_constraint()
                    .map(Statement::CreateConstraint);
            }
            if self.peek_word_at(1, "INDEX")
                || self.peek_word_at(1, "RANGE")
                || self.peek_word_at(1, "TEXT")
                || self.peek_word_at(1, "VECTOR")
            {
                self.advance();
                return self.create_index().map(Statement::CreateIndex);
            }
        }
        self.query_body().map(Statement::Query)
    }

    fn query_body(&mut self) -> Result<QueryBody> {
        let clauses = self.clauses_until_union()?;
        if clauses.is_empty() {
            return self.error("query has no clauses");
        }
        let mut unions = Vec::new();
        let mut union_all = None;
        while self.eat_word("UNION") {
            let all = self.eat_word("ALL");
            if union_all.is_some_and(|expected| expected != all) {
                return self.syntax_detail(
                    "InvalidClauseComposition",
                    "UNION and UNION ALL cannot be combined in one query",
                );
            }
            union_all = Some(all);
            let body = self.clauses_until_union()?;
            if body.is_empty() {
                return self.error("UNION branch has no clauses");
            }
            unions.push(UnionBranch { all, body });
        }
        Ok(QueryBody { clauses, unions })
    }

    fn clauses_until_union(&mut self) -> Result<Vec<Clause>> {
        let mut clauses = Vec::new();
        while !self.peek_word("UNION")
            && !self.peek_symbol(Symbol::RightBrace)
            && !self.at_end_or_semicolon()
        {
            let clause = if self.eat_word("OPTIONAL") {
                self.expect_word("MATCH")?;
                Clause::Match {
                    optional: true,
                    patterns: self.patterns(PatternUse::Read)?,
                }
            } else if self.eat_word("MATCH") {
                Clause::Match {
                    optional: false,
                    patterns: self.patterns(PatternUse::Read)?,
                }
            } else if self.eat_word("WHERE") {
                Clause::Where(self.predicate_expression()?)
            } else if self.eat_word("UNWIND") {
                let expression = self.expression(0)?;
                self.expect_word("AS")?;
                Clause::Unwind {
                    expression,
                    variable: self.identifier()?,
                }
            } else if self.eat_word("FOR") {
                let variable = self.identifier()?;
                self.expect_word("IN")?;
                Clause::For {
                    variable,
                    expression: self.expression(0)?,
                }
            } else if self.eat_word("LET") {
                Clause::Let(self.let_items()?)
            } else if self.eat_word("FILTER") {
                Clause::Filter(self.predicate_expression()?)
            } else if self.eat_word("CREATE") || self.eat_word("INSERT") {
                Clause::Create(self.patterns(PatternUse::Write)?)
            } else if self.eat_word("MERGE") {
                self.merge_clause()?
            } else if self.eat_word("SET") {
                Clause::Set(self.set_items()?)
            } else if self.eat_word("REMOVE") {
                Clause::Remove(self.remove_items()?)
            } else if self.eat_word("DETACH") {
                self.expect_word("DELETE")?;
                Clause::Delete {
                    detach: true,
                    expressions: self.delete_expressions()?,
                }
            } else if self.eat_word("NODETACH") {
                self.expect_word("DELETE")?;
                Clause::Delete {
                    detach: false,
                    expressions: self.delete_expressions()?,
                }
            } else if self.eat_word("DELETE") {
                Clause::Delete {
                    detach: false,
                    expressions: self.delete_expressions()?,
                }
            } else if self.eat_word("WITH") {
                Clause::With(self.projection()?)
            } else if self.eat_word("RETURN") {
                Clause::Return(self.projection()?)
            } else if self.eat_word("ORDER") {
                self.expect_word("BY")?;
                Clause::OrderBy(self.sort_items()?)
            } else if self.eat_word("SKIP") || self.eat_word("OFFSET") {
                Clause::Skip(self.expression(0)?)
            } else if self.eat_word("LIMIT") {
                Clause::Limit(self.expression(0)?)
            } else if self.eat_word("HISTORY") {
                Clause::History(self.history()?)
            } else if self.eat_word("WINDOW") {
                Clause::Window(self.window()?)
            } else if self.eat_word("SEARCH") {
                Clause::Search(self.search()?)
            } else if self.eat_word("CALL") {
                let mut call = self.call()?;
                let standalone = clauses.is_empty() && self.at_end_or_semicolon();
                if call.implicit_arguments && !standalone {
                    return self.error(
                        "InvalidArgumentPassingMode: omitted CALL arguments are only valid for a standalone procedure call",
                    );
                }
                if call.yield_all && !standalone {
                    return self.error("YIELD * is only valid for a standalone procedure call");
                }
                call.clause.standalone = standalone;
                Clause::Call(call.clause)
            } else if self.eat_word("FINISH") {
                Clause::Finish
            } else {
                return self.error("unsupported or misplaced Cypher clause");
            };
            clauses.push(clause);
        }
        Ok(clauses)
    }

    fn patterns(&mut self, usage: PatternUse) -> Result<Vec<Pattern>> {
        let mut result = vec![self.pattern(usage)?];
        while self.eat_symbol(Symbol::Comma) {
            result.push(self.pattern(usage)?);
        }
        Ok(result)
    }

    fn let_items(&mut self) -> Result<Vec<LetItem>> {
        let mut result = Vec::new();
        loop {
            let variable = self.identifier()?;
            self.expect_symbol(Symbol::Equal)?;
            result.push(LetItem {
                variable,
                expression: self.expression(0)?,
            });
            if !self.eat_symbol(Symbol::Comma) {
                break;
            }
        }
        Ok(result)
    }

    fn pattern(&mut self, usage: PatternUse) -> Result<Pattern> {
        let variable = if self.peek_identifier() && self.peek_symbol_at(1, Symbol::Equal) {
            let variable = self.identifier()?;
            self.expect_symbol(Symbol::Equal)?;
            Some(variable)
        } else {
            None
        };
        let (selector, mode) = self.path_prefix()?;
        let start = self.node_pattern()?;
        let mut steps = Vec::new();
        while self.peek_symbol(Symbol::Minus) || self.peek_symbol(Symbol::ArrowLeft) {
            let has_left_arrowhead = self.eat_symbol(Symbol::ArrowLeft);
            let direction_prefix = if has_left_arrowhead {
                Direction::Incoming
            } else {
                self.expect_symbol(Symbol::Minus)?;
                Direction::Undirected
            };
            let mut relationship = if self.eat_symbol(Symbol::LeftBracket) {
                self.relationship_pattern(direction_prefix)?
            } else {
                RelationshipPattern {
                    variable: None,
                    types: Vec::new(),
                    direction: direction_prefix,
                    variable_length: false,
                    min_hops: None,
                    max_hops: None,
                    properties: Vec::new(),
                }
            };
            let has_right_arrowhead = self.eat_symbol(Symbol::ArrowRight);
            if has_right_arrowhead {
                relationship.direction = if has_left_arrowhead {
                    // In a read pattern, arrowheads at both ends mean that either stored
                    // relationship direction may satisfy the pattern.
                    Direction::Undirected
                } else {
                    Direction::Outgoing
                };
            } else {
                self.expect_symbol(Symbol::Minus)?;
                relationship.direction = direction_prefix;
            }
            if usage == PatternUse::Write && relationship.direction == Direction::Undirected {
                return self.syntax_detail(
                    "RequiresDirectedRelationship",
                    "write patterns require exactly one relationship direction",
                );
            }
            self.relationship_quantifier(&mut relationship)?;
            steps.push(PatternStep {
                relationship,
                node: self.node_pattern()?,
            });
        }
        Ok(Pattern {
            variable,
            selector,
            mode,
            start,
            steps,
        })
    }

    fn path_prefix(&mut self) -> Result<(PathSelector, PathMode)> {
        if self.eat_word("REPEATABLE") {
            self.expect_word("ELEMENTS")?;
            return Ok((PathSelector::All, PathMode::RepeatableElements));
        }
        if self.eat_word("DIFFERENT") {
            self.expect_word("RELATIONSHIPS")?;
            return Ok((PathSelector::All, PathMode::DifferentRelationships));
        }
        if self.eat_word("ACYCLIC") {
            return Ok((PathSelector::All, PathMode::Acyclic));
        }
        if self.eat_word("ANY") {
            return Ok((
                if self.eat_word("SHORTEST") {
                    PathSelector::AnyShortest
                } else {
                    PathSelector::Any
                },
                PathMode::DifferentRelationships,
            ));
        }
        if self.eat_word("ALL") {
            self.expect_word("SHORTEST")?;
            return Ok((PathSelector::AllShortest, PathMode::DifferentRelationships));
        }
        if self.eat_word("SHORTEST") {
            let count = self.eat_integer_u32()?.unwrap_or(1);
            if count == 0 {
                return self.error("SHORTEST count must be positive");
            }
            let groups = self.eat_word("GROUP") || self.eat_word("GROUPS");
            return Ok((
                PathSelector::Shortest { count, groups },
                PathMode::DifferentRelationships,
            ));
        }
        Ok((PathSelector::All, PathMode::DifferentRelationships))
    }

    fn node_pattern(&mut self) -> Result<NodePattern> {
        self.expect_symbol(Symbol::LeftParen)?;
        let variable = self
            .peek_identifier()
            .then(|| self.identifier())
            .transpose()?;
        let mut labels = Vec::new();
        while self.eat_symbol(Symbol::Colon) {
            labels.push(self.identifier()?);
        }
        let property_predicate_present = self.eat_symbol(Symbol::LeftBrace);
        let properties = if property_predicate_present {
            self.map_entries(Symbol::RightBrace)?
        } else {
            Vec::new()
        };
        self.reject_parameter_pattern_predicate("node")?;
        self.expect_symbol(Symbol::RightParen)?;
        Ok(NodePattern {
            variable,
            labels,
            property_predicate_present,
            properties,
        })
    }

    fn relationship_pattern(&mut self, direction: Direction) -> Result<RelationshipPattern> {
        let variable = self
            .peek_identifier()
            .then(|| self.identifier())
            .transpose()?;
        let mut types = Vec::new();
        if self.eat_symbol(Symbol::Colon) {
            types.push(self.identifier()?);
            while self.eat_symbol(Symbol::Pipe) {
                self.eat_symbol(Symbol::Colon);
                types.push(self.identifier()?);
            }
        }
        let (variable_length, min_hops, max_hops) = if self.eat_symbol(Symbol::Star) {
            let minimum = self.eat_integer_u32()?;
            if self.eat_symbol(Symbol::Range) {
                (true, Some(minimum.unwrap_or(1)), self.eat_integer_u32()?)
            } else {
                match minimum {
                    Some(exact) => (true, Some(exact), Some(exact)),
                    None => (true, Some(1), None),
                }
            }
        } else {
            (false, None, None)
        };
        let properties = if self.eat_symbol(Symbol::LeftBrace) {
            self.map_entries(Symbol::RightBrace)?
        } else {
            Vec::new()
        };
        self.reject_parameter_pattern_predicate("relationship")?;
        if !self.eat_symbol(Symbol::RightBracket) {
            return self.syntax_detail(
                "InvalidRelationshipPattern",
                "malformed relationship pattern or variable-length bound",
            );
        }
        Ok(RelationshipPattern {
            variable,
            types,
            direction,
            variable_length,
            min_hops,
            max_hops,
            properties,
        })
    }

    fn relationship_quantifier(&mut self, relationship: &mut RelationshipPattern) -> Result<()> {
        if relationship.variable_length {
            return Ok(());
        }
        if self.eat_symbol(Symbol::Plus) {
            relationship.variable_length = true;
            relationship.min_hops = Some(1);
            relationship.max_hops = None;
            return Ok(());
        }
        if self.eat_symbol(Symbol::Star) {
            relationship.variable_length = true;
            relationship.min_hops = Some(0);
            relationship.max_hops = None;
            return Ok(());
        }
        if !self.eat_symbol(Symbol::LeftBrace) {
            return Ok(());
        }
        relationship.variable_length = true;
        let minimum = self.eat_integer_u32()?;
        if self.eat_symbol(Symbol::Comma) {
            let maximum = self.eat_integer_u32()?;
            if minimum.is_none() && maximum.is_none() {
                return self.error("relationship quantifier requires a bound");
            }
            relationship.min_hops = Some(minimum.unwrap_or(0));
            relationship.max_hops = maximum;
        } else {
            let exact = minimum
                .ok_or_else(|| self.make_error("relationship quantifier requires an integer"))?;
            relationship.min_hops = Some(exact);
            relationship.max_hops = Some(exact);
        }
        self.expect_symbol(Symbol::RightBrace)?;
        if relationship
            .max_hops
            .is_some_and(|maximum| maximum < relationship.min_hops.unwrap_or(0))
        {
            return self.error("relationship quantifier maximum is below minimum");
        }
        Ok(())
    }

    fn set_items(&mut self) -> Result<Vec<SetItem>> {
        let mut result = Vec::new();
        loop {
            let parenthesized = self.eat_symbol(Symbol::LeftParen);
            let variable = self.identifier()?;
            let property_target = if parenthesized {
                // The openCypher property-target grammar permits a simple entity selector such
                // as `(n).name`. Keep this deliberately narrower than a general parenthesized
                // expression: only one variable may occur between the parentheses.
                if !self.eat_symbol(Symbol::RightParen) {
                    return self.error("parenthesized SET target requires one variable");
                }
                if !self.eat_symbol(Symbol::Dot) {
                    return self.error("parenthesized SET target requires a property");
                }
                true
            } else {
                self.eat_symbol(Symbol::Dot)
            };
            if property_target {
                let property = self.identifier()?;
                self.expect_symbol(Symbol::Equal)?;
                let value = self.expression(0)?;
                let event_time = if self.eat_word("AT") {
                    self.expect_word("TIME")?;
                    Some(self.expression(0)?)
                } else {
                    None
                };
                result.push(SetItem::Property {
                    target: PropertyAccess { variable, property },
                    value,
                    event_time,
                });
            } else if self.eat_symbol(Symbol::Colon) {
                result.push(SetItem::Labels {
                    variable,
                    labels: self.label_names(Symbol::Colon)?,
                });
            } else if self.eat_word("IS") {
                result.push(SetItem::Labels {
                    variable,
                    labels: self.label_names(Symbol::Ampersand)?,
                });
            } else if self.eat_symbol(Symbol::Plus) {
                self.expect_symbol(Symbol::Equal)?;
                result.push(SetItem::MergeMap {
                    variable,
                    value: self.expression(0)?,
                });
            } else {
                self.expect_symbol(Symbol::Equal)?;
                result.push(SetItem::ReplaceMap {
                    variable,
                    value: self.expression(0)?,
                });
            }
            if !self.eat_symbol(Symbol::Comma) {
                break;
            }
        }
        Ok(result)
    }

    fn merge_clause(&mut self) -> Result<Clause> {
        let pattern = self.pattern(PatternUse::Merge)?;
        let mut on_create = None;
        let mut on_match = None;
        while self.eat_word("ON") {
            let target = if self.eat_word("CREATE") {
                &mut on_create
            } else if self.eat_word("MATCH") {
                &mut on_match
            } else {
                return self.error("MERGE ON requires CREATE or MATCH");
            };
            if target.is_some() {
                return self.error("MERGE action is declared more than once");
            }
            self.expect_word("SET")?;
            *target = Some(self.set_items()?);
        }
        Ok(Clause::Merge {
            pattern,
            on_create: on_create.unwrap_or_default(),
            on_match: on_match.unwrap_or_default(),
        })
    }

    fn remove_items(&mut self) -> Result<Vec<RemoveItem>> {
        let mut result = Vec::new();
        loop {
            let variable = self.identifier()?;
            if self.eat_symbol(Symbol::Dot) {
                result.push(RemoveItem::Property(PropertyAccess {
                    variable,
                    property: self.identifier()?,
                }));
            } else if self.eat_symbol(Symbol::Colon) {
                result.push(RemoveItem::Labels {
                    variable,
                    labels: self.label_names(Symbol::Colon)?,
                });
            } else if self.eat_word("IS") {
                result.push(RemoveItem::Labels {
                    variable,
                    labels: self.label_names(Symbol::Ampersand)?,
                });
            } else {
                return self.error("REMOVE requires a property or node label");
            }
            if !self.eat_symbol(Symbol::Comma) {
                break;
            }
        }
        Ok(result)
    }

    fn label_names(&mut self, separator: Symbol) -> Result<Vec<LabelName>> {
        let mut labels = vec![self.label_name()?];
        while self.eat_symbol(separator) {
            labels.push(self.label_name()?);
        }
        Ok(labels)
    }

    fn label_name(&mut self) -> Result<LabelName> {
        if self.eat_symbol(Symbol::Dollar) {
            self.expect_symbol(Symbol::LeftParen)?;
            let expression = self.expression(0)?;
            self.expect_symbol(Symbol::RightParen)?;
            return Ok(LabelName::Dynamic(expression));
        }
        if matches!(
            self.peek().kind,
            TokenKind::Parameter(ref value) if value.eq_ignore_ascii_case("all")
        ) && self.peek_symbol_at(1, Symbol::LeftParen)
        {
            self.advance();
            self.expect_symbol(Symbol::LeftParen)?;
            let expression = self.expression(0)?;
            self.expect_symbol(Symbol::RightParen)?;
            return Ok(LabelName::DynamicAll(expression));
        }
        self.identifier().map(LabelName::Static)
    }

    fn expression_list(&mut self) -> Result<Vec<Expression>> {
        let mut result = vec![self.expression(0)?];
        while self.eat_symbol(Symbol::Comma) {
            result.push(self.expression(0)?);
        }
        Ok(result)
    }

    fn delete_expressions(&mut self) -> Result<Vec<Expression>> {
        let label_predicates_before = self.parsed_label_predicates;
        let expressions = self.expression_list()?;
        if self.parsed_label_predicates != label_predicates_before
            || self.peek_symbol(Symbol::Colon)
        {
            return self.error(
                "InvalidDelete: a node label or relationship type is not a deletable expression",
            );
        }
        Ok(expressions)
    }

    fn projection(&mut self) -> Result<Projection> {
        let distinct = self.eat_word("DISTINCT");
        let mut items = Vec::new();
        loop {
            let source_start = self.peek().span.start;
            let expression = if self.eat_symbol(Symbol::Star) {
                Expression::Star
            } else {
                self.expression(0)?
            };
            let source_text = Some(self.consumed_source_from(source_start)?);
            let alias = if self.eat_word("AS") {
                Some(self.identifier()?)
            } else {
                None
            };
            items.push(ProjectionItem {
                expression,
                alias,
                source_text,
            });
            if !self.eat_symbol(Symbol::Comma) {
                break;
            }
        }
        Ok(Projection { distinct, items })
    }

    fn sort_items(&mut self) -> Result<Vec<SortItem>> {
        let mut result = Vec::new();
        loop {
            let expression = self.expression(0)?;
            let ascending = if self.eat_word("DESC") || self.eat_word("DESCENDING") {
                false
            } else {
                self.eat_word("ASC");
                self.eat_word("ASCENDING");
                true
            };
            result.push(SortItem {
                expression,
                ascending,
            });
            if !self.eat_symbol(Symbol::Comma) {
                break;
            }
        }
        Ok(result)
    }

    fn history(&mut self) -> Result<HistoryClause> {
        let variable = self.identifier()?;
        self.expect_symbol(Symbol::Dot)?;
        let property = self.identifier()?;
        self.expect_word("FROM")?;
        let from = self.expression(0)?;
        self.expect_word("TO")?;
        let to = self.expression(0)?;
        self.expect_word("AS")?;
        Ok(HistoryClause {
            target: PropertyAccess { variable, property },
            from,
            to,
            variable: self.identifier()?,
        })
    }

    fn window(&mut self) -> Result<WindowClause> {
        let kind = if self.eat_word("TUMBLING") {
            WindowSyntax::Tumbling
        } else if self.eat_word("HOPPING") {
            WindowSyntax::Hopping
        } else {
            return self.error("WINDOW requires TUMBLING or HOPPING");
        };
        let width = self.expression(0)?;
        let every = if self.eat_word("EVERY") {
            Some(self.expression(0)?)
        } else {
            None
        };
        if kind == WindowSyntax::Tumbling && every.is_some() {
            return self.error("EVERY is legal only for HOPPING");
        }
        self.expect_word("ON")?;
        let event_expression = self.expression(0)?;
        let align = if self.eat_word("ALIGN") {
            self.expect_word("TO")?;
            Some(self.expression(0)?)
        } else {
            None
        };
        let timezone = if self.eat_word("TIME") {
            self.expect_word("ZONE")?;
            Some(self.string_literal()?)
        } else {
            None
        };
        let emit_empty = if self.eat_word("EMIT") {
            self.expect_word("EMPTY")?;
            true
        } else {
            false
        };
        self.expect_word("AS")?;
        Ok(WindowClause {
            kind,
            width,
            every,
            event_expression,
            align,
            timezone,
            emit_empty,
            variable: self.identifier()?,
        })
    }

    fn search(&mut self) -> Result<SearchClause> {
        let variable = self.identifier()?;
        self.expect_word("IN")?;
        self.expect_symbol(Symbol::LeftParen)?;
        self.expect_word("EMBEDDING")?;
        self.expect_word("INDEX")?;
        let index = self.identifier()?;
        let input = if self.eat_word("FOR") {
            if self.eat_word("TEXT") {
                SearchInput::Text(self.expression(0)?)
            } else if self.eat_word("VECTOR") {
                SearchInput::Vector(self.expression(0)?)
            } else {
                return self.error("SEARCH FOR requires TEXT or VECTOR");
            }
        } else {
            return self.error("SEARCH requires FOR TEXT or FOR VECTOR");
        };
        self.expect_word("LIMIT")?;
        let limit = self.expression(0)?;
        self.expect_symbol(Symbol::RightParen)?;
        self.expect_word("SCORE")?;
        self.expect_word("AS")?;
        Ok(SearchClause {
            variable,
            index,
            input,
            limit,
            score_variable: self.identifier()?,
        })
    }

    fn call(&mut self) -> Result<ParsedCall> {
        let mut name = vec![self.identifier()?];
        while self.eat_symbol(Symbol::Dot) {
            name.push(self.identifier()?);
        }
        let implicit_arguments = !self.eat_symbol(Symbol::LeftParen);
        let arguments = if implicit_arguments || self.eat_symbol(Symbol::RightParen) {
            Vec::new()
        } else {
            let args = self.expression_list()?;
            self.expect_symbol(Symbol::RightParen)?;
            args
        };
        let mut yields = Vec::new();
        let mut yield_all = false;
        if self.eat_word("YIELD") {
            if self.eat_symbol(Symbol::Star) {
                yield_all = true;
                if self.eat_symbol(Symbol::Comma) {
                    return self.error("YIELD * cannot be combined with named yield items");
                }
            } else {
                loop {
                    let expression = Expression::Variable(self.identifier()?);
                    let alias = if self.eat_word("AS") {
                        Some(self.identifier()?)
                    } else {
                        None
                    };
                    yields.push(ProjectionItem {
                        expression,
                        alias,
                        source_text: None,
                    });
                    if !self.eat_symbol(Symbol::Comma) {
                        break;
                    }
                }
            }
        }
        Ok(ParsedCall {
            clause: CallClause {
                name,
                argument_mode: if implicit_arguments {
                    CallArgumentMode::Implicit
                } else {
                    CallArgumentMode::Explicit
                },
                arguments,
                yield_mode: if yield_all {
                    CallYieldMode::All
                } else if yields.is_empty() {
                    CallYieldMode::Omitted
                } else {
                    CallYieldMode::Explicit
                },
                yields,
                standalone: false,
            },
            implicit_arguments,
            yield_all,
        })
    }

    fn create_index(&mut self) -> Result<IndexDefinition> {
        let kind = if self.eat_word("INDEX") {
            IndexKind::Equality
        } else if self.eat_word("RANGE") {
            self.expect_word("INDEX")?;
            IndexKind::Range
        } else if self.eat_word("TEXT") {
            self.expect_word("INDEX")?;
            IndexKind::Text
        } else if self.eat_word("VECTOR") {
            self.expect_word("INDEX")?;
            IndexKind::Vector
        } else {
            return self.error("invalid CREATE INDEX form");
        };
        let name = self.identifier()?;
        self.expect_word("FOR")?;
        self.expect_symbol(Symbol::LeftParen)?;
        let variable = self.identifier()?;
        self.expect_symbol(Symbol::Colon)?;
        let label = self.identifier()?;
        self.expect_symbol(Symbol::RightParen)?;
        self.expect_word("ON")?;
        self.expect_symbol(Symbol::LeftParen)?;
        let mut properties = Vec::new();
        loop {
            let owner = self.identifier()?;
            if owner != variable {
                return self.error("index property variable differs from FOR variable");
            }
            self.expect_symbol(Symbol::Dot)?;
            properties.push(self.identifier()?);
            if !self.eat_symbol(Symbol::Comma) {
                break;
            }
        }
        self.expect_symbol(Symbol::RightParen)?;
        Ok(IndexDefinition {
            name,
            kind,
            variable,
            label,
            properties,
        })
    }

    fn create_unique_constraint(&mut self) -> Result<UniqueConstraintDefinition> {
        let name = self.identifier()?;
        self.expect_word("FOR")?;
        self.expect_symbol(Symbol::LeftParen)?;
        let variable = self.identifier()?;
        self.expect_symbol(Symbol::Colon)?;
        let label = self.identifier()?;
        self.expect_symbol(Symbol::RightParen)?;
        self.expect_word("REQUIRE")?;
        let owner = self.identifier()?;
        if owner != variable {
            return self.error("constraint property variable differs from FOR variable");
        }
        self.expect_symbol(Symbol::Dot)?;
        let property = self.identifier()?;
        self.expect_word("IS")?;
        self.expect_word("UNIQUE")?;
        Ok(UniqueConstraintDefinition {
            name,
            variable,
            label,
            property,
        })
    }

    fn create_rollup(&mut self) -> Result<RollupDefinition> {
        let name = self.identifier()?;
        self.expect_word("FOR")?;
        self.expect_symbol(Symbol::LeftParen)?;
        let variable = self.identifier()?;
        self.expect_symbol(Symbol::Colon)?;
        let label = self.identifier()?;
        self.expect_symbol(Symbol::RightParen)?;
        self.expect_word("ON")?;
        let owner = self.identifier()?;
        if owner != variable {
            return self.error("rollup property variable differs from FOR variable");
        }
        self.expect_symbol(Symbol::Dot)?;
        let property = self.identifier()?;
        self.expect_word("WINDOW")?;
        let window = if self.eat_word("TUMBLING") {
            WindowSyntax::Tumbling
        } else if self.eat_word("HOPPING") {
            WindowSyntax::Hopping
        } else {
            return self.error("rollup WINDOW requires TUMBLING or HOPPING");
        };
        let width = self.expression(0)?;
        let every = if self.eat_word("EVERY") {
            Some(self.expression(0)?)
        } else {
            None
        };
        if window == WindowSyntax::Tumbling && every.is_some() {
            return self.error("EVERY is legal only for a HOPPING rollup");
        }
        if window == WindowSyntax::Hopping && every.is_none() {
            return self.error("a HOPPING rollup requires EVERY");
        }
        let align = if self.eat_word("ALIGN") {
            self.expect_word("TO")?;
            Some(self.expression(0)?)
        } else {
            None
        };
        let timezone = if self.eat_word("TIME") {
            self.expect_word("ZONE")?;
            Some(self.string_literal()?)
        } else {
            None
        };
        self.expect_word("AGGREGATE")?;
        let mut aggregates = vec![self.identifier()?];
        while self.eat_symbol(Symbol::Comma) {
            aggregates.push(self.identifier()?);
        }
        Ok(RollupDefinition {
            name,
            variable,
            label,
            property,
            window,
            width,
            every,
            align,
            timezone,
            aggregates,
        })
    }

    fn create_embedding(&mut self) -> Result<EmbeddingDefinition> {
        let name = self.identifier()?;
        self.expect_word("FOR")?;
        self.expect_symbol(Symbol::LeftParen)?;
        let variable = self.identifier()?;
        self.expect_symbol(Symbol::Colon)?;
        let label = self.identifier()?;
        self.expect_symbol(Symbol::RightParen)?;
        self.expect_word("FROM")?;
        if self.identifier()? != variable {
            return self.error("embedding source variable differs from FOR variable");
        }
        self.expect_symbol(Symbol::Dot)?;
        let source_property = self.identifier()?;
        self.expect_word("INTO")?;
        if self.identifier()? != variable {
            return self.error("embedding target variable differs from FOR variable");
        }
        self.expect_symbol(Symbol::Dot)?;
        let target_property = self.identifier()?;
        self.expect_word("USING")?;
        self.expect_word("MODEL")?;
        let model = self.identifier()?;
        self.expect_word("SIMILARITY")?;
        let similarity = self.identifier()?;
        Ok(EmbeddingDefinition {
            name,
            variable,
            label,
            source_property,
            target_property,
            model,
            similarity,
        })
    }

    /// Parses one expression, refusing nesting deeper than [`MAXIMUM_EXPRESSION_DEPTH`].
    ///
    /// Every recursive expression construct re-enters this function, so accounting here bounds the
    /// entire descent. The depth is restored on both the success and failure paths so a rejected
    /// subexpression does not leak budget into the rest of the query.
    fn expression(&mut self, minimum_binding: u8) -> Result<Expression> {
        if self.expression_depth >= MAXIMUM_EXPRESSION_DEPTH {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "expression nesting is too deep",
            ));
        }
        self.expression_depth = self.expression_depth.saturating_add(1);
        // The depth limit bounds how much nesting a query may express; `maybe_grow` bounds how much
        // of it lands on one stack segment. Both are required: the debug profile's frames are large
        // enough to exhaust a 2 MiB worker stack well before the limit, and growing without a limit
        // would trade a stack abort for unbounded allocation. The binder, planner, and executor
        // already guard their matching descents the same way.
        let parsed = stacker::maybe_grow(256 * 1024, 2 * 1024 * 1024, || {
            self.nested_expression(minimum_binding)
        });
        self.expression_depth = self.expression_depth.saturating_sub(1);
        parsed
    }

    fn nested_expression(&mut self, minimum_binding: u8) -> Result<Expression> {
        let mut left = if self.eat_word("NOT") {
            Expression::Unary {
                operation: UnaryOperator::Not,
                // Cypher evaluates comparison, membership, and `IS [NOT] NULL` before logical
                // negation. Keeping NOT above AND/XOR/OR but below those predicates makes
                // `NOT false >= false` parse as `NOT (false >= false)`, rather than
                // `(NOT false) >= false`.
                operand: Box::new(self.expression(25)?),
            }
        } else if self.eat_symbol(Symbol::Plus) {
            Expression::Unary {
                operation: UnaryOperator::Positive,
                operand: Box::new(self.expression(80)?),
            }
        } else if self.eat_symbol(Symbol::Minus) {
            self.negative_expression()?
        } else {
            self.primary()?
        };
        let mut previous_comparison_operand = None;
        let mut chained = 0_usize;
        loop {
            // Postfix and infix operators extend `left` iteratively rather than by recursing, so
            // the depth guard above never sees them. They still build a left-nested tree, and
            // dropping or walking that tree does recurse: an unbounded chain aborts the process
            // when the AST is released, long after parsing appeared to succeed. Bound the spine.
            chained = chained.saturating_add(1);
            if chained > MAXIMUM_EXPRESSION_CHAIN {
                return Err(Error::new(
                    ErrorCode::QuerySyntax,
                    "expression operator chain is too long",
                ));
            }
            if self.eat_symbol(Symbol::Dot) {
                left = Expression::Property(Box::new(left), self.identifier()?);
                continue;
            }
            if self.peek_symbol(Symbol::LeftBrace) {
                left = self.map_projection(left)?;
                continue;
            }
            if self.eat_symbol(Symbol::LeftBracket) {
                if self.eat_symbol(Symbol::Range) {
                    let end = (!self.peek_symbol(Symbol::RightBracket))
                        .then(|| self.expression(0))
                        .transpose()?
                        .map(Box::new);
                    self.expect_symbol(Symbol::RightBracket)?;
                    left = Expression::Slice {
                        expression: Box::new(left),
                        start: None,
                        end,
                    };
                } else {
                    let index = self.expression(0)?;
                    if self.eat_symbol(Symbol::Range) {
                        let end = (!self.peek_symbol(Symbol::RightBracket))
                            .then(|| self.expression(0))
                            .transpose()?
                            .map(Box::new);
                        self.expect_symbol(Symbol::RightBracket)?;
                        left = Expression::Slice {
                            expression: Box::new(left),
                            start: Some(Box::new(index)),
                            end,
                        };
                    } else {
                        self.expect_symbol(Symbol::RightBracket)?;
                        left = Expression::Index {
                            expression: Box::new(left),
                            index: Box::new(index),
                        };
                    }
                }
                continue;
            }
            if self.peek_symbol(Symbol::Colon) {
                left = self.entity_label_predicate(left)?;
                continue;
            }
            if self.peek_word("IS") {
                if 35 < minimum_binding {
                    break;
                }
                self.advance();
                let negated = self.eat_word("NOT");
                self.expect_word("NULL")?;
                left = Expression::IsNull {
                    expression: Box::new(left),
                    negated,
                };
                continue;
            }
            let Some((operation, left_binding, right_binding, words)) = self.binary_operator()
            else {
                break;
            };
            if left_binding < minimum_binding {
                break;
            }
            for _ in 0..words {
                self.advance();
            }
            let right = self.expression(right_binding)?;
            if Self::is_simple_comparison(operation) {
                left = if let Some(previous_operand) = previous_comparison_operand.take() {
                    let comparison = Expression::Binary {
                        left: Box::new(previous_operand),
                        operation,
                        right: Box::new(right.clone()),
                    };
                    Expression::Binary {
                        left: Box::new(left),
                        operation: BinaryOperator::And,
                        right: Box::new(comparison),
                    }
                } else {
                    Expression::Binary {
                        left: Box::new(left),
                        operation,
                        right: Box::new(right.clone()),
                    }
                };
                previous_comparison_operand = Some(right);
            } else {
                previous_comparison_operand = None;
                left = Expression::Binary {
                    left: Box::new(left),
                    operation,
                    right: Box::new(right),
                };
            }
        }
        Ok(left)
    }

    /// Parse an expression in a syntactic predicate position.
    ///
    /// Legacy openCypher pattern predicates are legal in WHERE/FILTER predicate trees, but not as
    /// general values in RETURN, WITH, SET, or ordinary function arguments. The scoped flag is
    /// restored on both success and failure so one malformed predicate cannot leak permissions to
    /// the following clause.
    fn predicate_expression(&mut self) -> Result<Expression> {
        let previous = self.pattern_predicates_allowed;
        self.pattern_predicates_allowed = true;
        let result = self.expression(0);
        self.pattern_predicates_allowed = previous;
        result
    }

    /// Parse the openCypher existential-subquery expression as an owned query AST.
    ///
    /// The simple form is normalized to `MATCH [WHERE]`; the full form retains its ordinary
    /// clauses. Both therefore share binder, planner, matcher, aggregation, and row semantics.
    fn existential_subquery_expression(&mut self) -> Result<Expression> {
        stacker::maybe_grow(256 * 1024, 2 * 1024 * 1024, || {
            self.parse_existential_subquery_expression()
        })
    }

    fn parse_existential_subquery_expression(&mut self) -> Result<Expression> {
        self.expect_word("EXISTS")?;
        self.expect_symbol(Symbol::LeftBrace)?;
        let prior_depth = self.existential_subquery_depth;
        self.existential_subquery_depth = prior_depth.saturating_add(1);
        let result = (|| {
            let body = if self.peek_symbol(Symbol::LeftParen) {
                let pattern = self.pattern(PatternUse::Read)?;
                let mut clauses = vec![Clause::Match {
                    optional: false,
                    patterns: vec![pattern],
                }];
                if self.eat_word("WHERE") {
                    clauses.push(Clause::Where(self.predicate_expression()?));
                }
                QueryBody {
                    clauses,
                    unions: Vec::new(),
                }
            } else {
                self.query_body()?
            };
            self.expect_symbol(Symbol::RightBrace)?;
            self.validate_existential_subquery_body(&body)?;
            Expression::existential_subquery(ExistentialSubquery { body })
        })();
        self.existential_subquery_depth = prior_depth;
        result
    }

    fn validate_existential_subquery_body(&self, body: &QueryBody) -> Result<()> {
        if !body.unions.is_empty() {
            return Err(Error::new(
                crate::ErrorCode::QuerySyntax,
                "InvalidClauseComposition: UNION is not supported inside an existential subquery",
            ));
        }
        for clause in &body.clauses {
            if matches!(
                clause,
                Clause::Create(_)
                    | Clause::Merge { .. }
                    | Clause::Set(_)
                    | Clause::Remove(_)
                    | Clause::Delete { .. }
            ) {
                return Err(Error::new(
                    crate::ErrorCode::QuerySyntax,
                    "InvalidClauseComposition: existential subqueries are read-only",
                ));
            }
            if !matches!(
                clause,
                Clause::Match { .. }
                    | Clause::Where(_)
                    | Clause::Unwind { .. }
                    | Clause::For { .. }
                    | Clause::Let(_)
                    | Clause::Filter(_)
                    | Clause::With(_)
                    | Clause::Return(_)
                    | Clause::OrderBy(_)
                    | Clause::Skip(_)
                    | Clause::Limit(_)
                    | Clause::Finish
            ) {
                return Err(Error::new(
                    crate::ErrorCode::QuerySyntax,
                    "InvalidClauseComposition: clause is not supported inside an existential subquery",
                ));
            }
        }
        Ok(())
    }

    /// Parse the polymorphic openCypher postfix label/type-predicate grammar.
    ///
    /// The parser cannot choose node versus relationship semantics: `node:A` checks labels while
    /// `relationship:A` checks its exact type, and NULL propagates. Preserve that distinction in
    /// one fail-closed AST intrinsic for binder and backend compilers to recognize explicitly.
    fn entity_label_predicate(&mut self, source: Expression) -> Result<Expression> {
        self.expect_symbol(Symbol::Colon)?;
        let mut names = vec![self.identifier()?];
        while self.eat_symbol(Symbol::Colon) {
            names.push(self.identifier()?);
        }
        self.parsed_label_predicates = self.parsed_label_predicates.saturating_add(1);
        Ok(Expression::entity_label_predicate(source, names))
    }

    fn primary(&mut self) -> Result<Expression> {
        if self.pattern_predicates_allowed && self.looks_like_pattern_predicate() {
            let pattern = self.pattern(PatternUse::Predicate)?;
            debug_assert!(!pattern.steps.is_empty());
            return Ok(Expression::pattern_predicate(pattern));
        }
        let token = self.peek().clone();
        match token.kind {
            TokenKind::Integer(value) => {
                let Ok(value) = i64::try_from(value) else {
                    return self
                        .syntax_detail("IntegerOverflow", "integer literal is out of range");
                };
                self.advance();
                Ok(Expression::integer(value))
            }
            TokenKind::Float(value) => {
                self.advance();
                Ok(Expression::float(value))
            }
            TokenKind::String(value) => {
                self.advance();
                Ok(Expression::string(Arc::<str>::from(value)))
            }
            TokenKind::Parameter(value) => {
                self.advance();
                Ok(Expression::Parameter(value))
            }
            TokenKind::Identifier(ref value) if value.eq_ignore_ascii_case("NULL") => {
                self.advance();
                Ok(Expression::Literal(ScalarValue::Null))
            }
            TokenKind::Identifier(ref value) if value.eq_ignore_ascii_case("TRUE") => {
                self.advance();
                Ok(Expression::Literal(ScalarValue::Boolean(true)))
            }
            TokenKind::Identifier(ref value) if value.eq_ignore_ascii_case("FALSE") => {
                self.advance();
                Ok(Expression::Literal(ScalarValue::Boolean(false)))
            }
            TokenKind::Identifier(ref value) if value.eq_ignore_ascii_case("CASE") => {
                self.case_expression()
            }
            TokenKind::Identifier(ref value)
                if value.eq_ignore_ascii_case("EXISTS")
                    && self.peek_symbol_at(1, Symbol::LeftBrace) =>
            {
                self.existential_subquery_expression()
            }
            TokenKind::Identifier(ref value)
                if value.eq_ignore_ascii_case("REDUCE")
                    && self.peek_symbol_at(1, Symbol::LeftParen) =>
            {
                self.reduce_expression()
            }
            TokenKind::Identifier(ref value)
                if matches!(
                    value.to_ascii_uppercase().as_str(),
                    "ALL" | "ANY" | "NONE" | "SINGLE"
                ) && self.peek_symbol_at(1, Symbol::LeftParen) =>
            {
                self.list_predicate_expression()
            }
            TokenKind::Identifier(_) => {
                let mut name = vec![self.identifier()?];
                while self.peek_symbol(Symbol::Dot)
                    && self.peek_identifier_at(1)
                    && self.peek_symbol_at(2, Symbol::LeftParen)
                {
                    self.advance();
                    name.push(self.identifier()?);
                }
                if self.eat_symbol(Symbol::LeftParen) {
                    let distinct = self.eat_word("DISTINCT");
                    let arguments = if self.eat_symbol(Symbol::RightParen) {
                        Vec::new()
                    } else if self.eat_symbol(Symbol::Star) {
                        self.expect_symbol(Symbol::RightParen)?;
                        vec![Expression::Star]
                    } else {
                        let values = self.expression_list()?;
                        self.expect_symbol(Symbol::RightParen)?;
                        values
                    };
                    Ok(Expression::Function {
                        name,
                        distinct,
                        arguments,
                    })
                } else {
                    Ok(Expression::Variable(name.remove(0)))
                }
            }
            TokenKind::Symbol(Symbol::LeftParen) => {
                self.advance();
                let value = self.expression(0)?;
                self.expect_symbol(Symbol::RightParen)?;
                Ok(value)
            }
            TokenKind::Symbol(Symbol::LeftBracket) => {
                self.advance();
                if self.eat_symbol(Symbol::RightBracket) {
                    return Ok(Expression::List(Vec::new()));
                }
                if self.peek_identifier() && self.peek_word_at(1, "IN") {
                    let variable = self.identifier()?;
                    self.expect_word("IN")?;
                    let list = self.expression(0)?;
                    let predicate = if self.eat_word("WHERE") {
                        Some(Box::new(self.predicate_expression()?))
                    } else {
                        None
                    };
                    let projection = if self.eat_symbol(Symbol::Pipe) {
                        Some(Box::new(self.expression(0)?))
                    } else {
                        None
                    };
                    self.expect_symbol(Symbol::RightBracket)?;
                    return Ok(Expression::ListComprehension {
                        variable,
                        list: Box::new(list),
                        predicate,
                        projection,
                    });
                }
                if self.looks_like_pattern_comprehension() {
                    let pattern = self.pattern(PatternUse::Comprehension)?;
                    if pattern.steps.is_empty() {
                        return self.syntax_detail(
                            "UnexpectedSyntax",
                            "a pattern comprehension requires a relationship pattern",
                        );
                    }
                    let predicate = if self.eat_word("WHERE") {
                        Some(self.predicate_expression()?)
                    } else {
                        None
                    };
                    self.expect_symbol(Symbol::Pipe)?;
                    let projection = self.expression(0)?;
                    self.expect_symbol(Symbol::RightBracket)?;
                    return Ok(Expression::pattern_comprehension(
                        pattern, predicate, projection,
                    ));
                }
                let values = self.expression_list()?;
                self.expect_symbol(Symbol::RightBracket)?;
                Ok(Expression::List(values))
            }
            TokenKind::Symbol(Symbol::LeftBrace) => {
                self.advance();
                Ok(Expression::Map(self.map_entries(Symbol::RightBrace)?))
            }
            TokenKind::Symbol(Symbol::Star) => {
                self.advance();
                Ok(Expression::Star)
            }
            _ => self.error("expected expression"),
        }
    }

    /// Distinguish a node-shaped parenthesized value from the start of a pattern predicate without
    /// speculative parsing. A relationship must immediately follow the matching node parenthesis:
    /// `-[]-`, `-[]->`, `<-[]-`, `--`, `-->`, or `<--`. Ordinary subtraction such as `(n) - 1`
    /// therefore remains an arithmetic expression.
    fn looks_like_pattern_predicate(&self) -> bool {
        self.relationship_follows_parenthesized_node(self.cursor)
    }

    /// Pattern comprehensions have their own bracketed grammar and may optionally declare a path
    /// variable before the first node. Requiring a relationship immediately after that first node
    /// keeps ordinary lists such as `[(n), 1]` and arithmetic such as `[(n) - 1]` unambiguous.
    fn looks_like_pattern_comprehension(&self) -> bool {
        let node_index = if self.peek_identifier() && self.peek_symbol_at(1, Symbol::Equal) {
            self.cursor.saturating_add(2)
        } else {
            self.cursor
        };
        self.relationship_follows_parenthesized_node(node_index)
    }

    fn relationship_follows_parenthesized_node(&self, node_index: usize) -> bool {
        if !matches!(
            self.tokens.get(node_index).map(|token| &token.kind),
            Some(TokenKind::Symbol(Symbol::LeftParen))
        ) {
            return false;
        }
        let mut depth = 0_usize;
        for index in node_index..self.tokens.len() {
            match self.tokens[index].kind {
                TokenKind::Symbol(Symbol::LeftParen) => depth = depth.saturating_add(1),
                TokenKind::Symbol(Symbol::RightParen) => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return self.relationship_follows_node(index.saturating_add(1));
                    }
                }
                TokenKind::End => return false,
                _ => {}
            }
        }
        false
    }

    fn relationship_follows_node(&self, index: usize) -> bool {
        match self.tokens.get(index).map(|token| &token.kind) {
            Some(TokenKind::Symbol(Symbol::Minus)) => matches!(
                self.tokens
                    .get(index.saturating_add(1))
                    .map(|token| &token.kind),
                Some(TokenKind::Symbol(
                    Symbol::LeftBracket | Symbol::Minus | Symbol::ArrowRight
                ))
            ),
            Some(TokenKind::Symbol(Symbol::ArrowLeft)) => matches!(
                self.tokens
                    .get(index.saturating_add(1))
                    .map(|token| &token.kind),
                Some(TokenKind::Symbol(Symbol::LeftBracket | Symbol::Minus))
            ),
            _ => false,
        }
    }

    fn negative_expression(&mut self) -> Result<Expression> {
        if let TokenKind::Integer(magnitude) = self.peek().kind {
            if magnitude == I64_MIN_MAGNITUDE {
                self.advance();
                return Ok(Expression::integer(i64::MIN));
            }
            if magnitude > I64_MIN_MAGNITUDE {
                return self.syntax_detail("IntegerOverflow", "integer literal is out of range");
            }
        }
        Ok(Expression::Unary {
            operation: UnaryOperator::Negative,
            operand: Box::new(self.expression(80)?),
        })
    }

    fn binary_operator(&self) -> Option<(BinaryOperator, u8, u8, usize)> {
        let word = |offset: usize, expected: &str| self.peek_word_at(offset, expected);
        if word(0, "OR") {
            Some((BinaryOperator::Or, 10, 11, 1))
        } else if word(0, "XOR") {
            Some((BinaryOperator::Xor, 15, 16, 1))
        } else if word(0, "AND") {
            Some((BinaryOperator::And, 20, 21, 1))
        } else if word(0, "IN") {
            // The TCK treats membership as a list predicate with higher precedence than the
            // ordinary comparison operators: `false = true IN [true, false]` means
            // `false = (true IN [true, false])`.
            Some((BinaryOperator::In, 35, 36, 1))
        } else if word(0, "STARTS") && word(1, "WITH") {
            Some((BinaryOperator::StartsWith, 35, 36, 2))
        } else if word(0, "ENDS") && word(1, "WITH") {
            Some((BinaryOperator::EndsWith, 35, 36, 2))
        } else if word(0, "CONTAINS") {
            Some((BinaryOperator::Contains, 35, 36, 1))
        } else {
            match self.peek().kind {
                TokenKind::Symbol(Symbol::Equal) => Some((BinaryOperator::Equal, 30, 31, 1)),
                TokenKind::Symbol(Symbol::NotEqual) => Some((BinaryOperator::NotEqual, 30, 31, 1)),
                TokenKind::Symbol(Symbol::Less) => Some((BinaryOperator::Less, 30, 31, 1)),
                TokenKind::Symbol(Symbol::LessOrEqual) => {
                    Some((BinaryOperator::LessOrEqual, 30, 31, 1))
                }
                TokenKind::Symbol(Symbol::Greater) => Some((BinaryOperator::Greater, 30, 31, 1)),
                TokenKind::Symbol(Symbol::GreaterOrEqual) => {
                    Some((BinaryOperator::GreaterOrEqual, 30, 31, 1))
                }
                TokenKind::Symbol(Symbol::RegexMatch) => {
                    Some((BinaryOperator::RegexMatch, 30, 31, 1))
                }
                TokenKind::Symbol(Symbol::DoublePipe) => Some((BinaryOperator::Concat, 40, 41, 1)),
                TokenKind::Symbol(Symbol::Plus) => Some((BinaryOperator::Add, 40, 41, 1)),
                TokenKind::Symbol(Symbol::Minus) => Some((BinaryOperator::Subtract, 40, 41, 1)),
                TokenKind::Symbol(Symbol::Star) => Some((BinaryOperator::Multiply, 50, 51, 1)),
                TokenKind::Symbol(Symbol::Slash) => Some((BinaryOperator::Divide, 50, 51, 1)),
                TokenKind::Symbol(Symbol::Percent) => Some((BinaryOperator::Modulo, 50, 51, 1)),
                // openCypher exponentiation is left-associative. As with the other
                // left-associative binary operators, the right operand therefore uses a
                // binding power one greater than the operator itself. Unary signs remain
                // tighter at binding power 80.
                TokenKind::Symbol(Symbol::Caret) => Some((BinaryOperator::Power, 70, 71, 1)),
                _ => None,
            }
        }
    }

    fn is_simple_comparison(operation: BinaryOperator) -> bool {
        matches!(
            operation,
            BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessOrEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterOrEqual
        )
    }

    fn case_expression(&mut self) -> Result<Expression> {
        self.expect_word("CASE")?;
        let operand = if self.peek_word("WHEN") {
            None
        } else {
            Some(Box::new(self.expression(0)?))
        };
        let mut alternatives = Vec::new();
        while self.eat_word("WHEN") {
            let when = self.expression(0)?;
            self.expect_word("THEN")?;
            alternatives.push(CaseAlternative {
                when,
                then: self.expression(0)?,
            });
        }
        if alternatives.is_empty() {
            return self.error("CASE requires at least one WHEN branch");
        }
        let default = if self.eat_word("ELSE") {
            Some(Box::new(self.expression(0)?))
        } else {
            None
        };
        self.expect_word("END")?;
        Ok(Expression::Case {
            operand,
            alternatives,
            default,
        })
    }

    fn reduce_expression(&mut self) -> Result<Expression> {
        self.expect_word("REDUCE")?;
        self.expect_symbol(Symbol::LeftParen)?;
        let accumulator = self.identifier()?;
        self.expect_symbol(Symbol::Equal)?;
        let initial = self.expression(0)?;
        self.expect_symbol(Symbol::Comma)?;
        let variable = self.identifier()?;
        self.expect_word("IN")?;
        let list = self.expression(0)?;
        self.expect_symbol(Symbol::Pipe)?;
        let expression = self.expression(0)?;
        self.expect_symbol(Symbol::RightParen)?;
        Ok(Expression::Reduce {
            accumulator,
            initial: Box::new(initial),
            variable,
            list: Box::new(list),
            expression: Box::new(expression),
        })
    }

    fn list_predicate_expression(&mut self) -> Result<Expression> {
        let kind = if self.eat_word("ALL") {
            ListPredicateKind::All
        } else if self.eat_word("ANY") {
            ListPredicateKind::Any
        } else if self.eat_word("NONE") {
            ListPredicateKind::None
        } else if self.eat_word("SINGLE") {
            ListPredicateKind::Single
        } else {
            return self.error("expected list predicate");
        };
        self.expect_symbol(Symbol::LeftParen)?;
        let variable = self.identifier()?;
        self.expect_word("IN")?;
        let list = self.expression(0)?;
        self.expect_word("WHERE")?;
        let predicate = self.predicate_expression()?;
        self.expect_symbol(Symbol::RightParen)?;
        Ok(Expression::ListPredicate {
            kind,
            variable,
            list: Box::new(list),
            predicate: Box::new(predicate),
        })
    }

    fn map_projection(&mut self, source: Expression) -> Result<Expression> {
        self.expect_symbol(Symbol::LeftBrace)?;
        let mut items = Vec::new();
        if self.eat_symbol(Symbol::RightBrace) {
            return Ok(Expression::MapProjection {
                source: Box::new(source),
                items,
            });
        }
        loop {
            if self.eat_symbol(Symbol::Dot) {
                if self.eat_symbol(Symbol::Star) {
                    items.push(MapProjectionItem::AllProperties);
                } else {
                    items.push(MapProjectionItem::Property(self.identifier()?));
                }
            } else {
                let name = self.identifier()?;
                if self.eat_symbol(Symbol::Colon) {
                    items.push(MapProjectionItem::Entry(name, self.expression(0)?));
                } else {
                    items.push(MapProjectionItem::Variable(name));
                }
            }
            if !self.eat_symbol(Symbol::Comma) {
                break;
            }
        }
        self.expect_symbol(Symbol::RightBrace)?;
        Ok(Expression::MapProjection {
            source: Box::new(source),
            items,
        })
    }

    fn map_entries(&mut self, closing: Symbol) -> Result<Vec<(String, Expression)>> {
        if self.eat_symbol(closing) {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        loop {
            let key = self.identifier()?;
            self.expect_symbol(Symbol::Colon)?;
            entries.push((key, self.expression(0)?));
            if !self.eat_symbol(Symbol::Comma) {
                break;
            }
        }
        self.expect_symbol(closing)?;
        Ok(entries)
    }

    fn parse_layer_mask(&mut self) -> Result<LayerMask> {
        let mut mask = LayerMask::empty();
        loop {
            let layer = self.parse_layer()?;
            let bit = match layer {
                Layer::Observed => LayerMask::OBSERVED,
                Layer::Knowledge => LayerMask::KNOWLEDGE,
                Layer::Workspace => LayerMask::WORKSPACE,
            };
            if mask.contains(bit) {
                return self.error("duplicate layer in USE LAYER");
            }
            mask |= bit;
            if !self.eat_symbol(Symbol::Comma) {
                break;
            }
        }
        Ok(mask)
    }

    fn parse_layer(&mut self) -> Result<Layer> {
        if self.eat_word("OBSERVED") {
            Ok(Layer::Observed)
        } else if self.eat_word("KNOWLEDGE") {
            Ok(Layer::Knowledge)
        } else if self.eat_word("WORKSPACE") {
            Ok(Layer::Workspace)
        } else {
            self.error("invalid layer name")
        }
    }

    fn temporal_type(&mut self) -> Result<String> {
        let first = self.identifier()?;
        let upper = first.to_ascii_uppercase();
        if matches!(upper.as_str(), "LOCAL" | "ZONED") {
            let second = self.identifier()?;
            Ok(format!("{upper} {}", second.to_ascii_uppercase()))
        } else {
            Ok(upper)
        }
    }

    fn string_literal(&mut self) -> Result<String> {
        match self.peek().kind.clone() {
            TokenKind::String(value) => {
                self.advance();
                Ok(value)
            }
            _ => self.error("expected string literal"),
        }
    }

    fn identifier(&mut self) -> Result<String> {
        match self.peek().kind.clone() {
            TokenKind::Identifier(value) => {
                self.advance();
                Ok(value)
            }
            _ => self.error("expected identifier"),
        }
    }

    fn eat_integer_u32(&mut self) -> Result<Option<u32>> {
        match self.peek().kind {
            TokenKind::Integer(value) => {
                self.advance();
                Ok(Some(
                    u32::try_from(value).map_err(|_| self.make_error("hop bound exceeds u32"))?,
                ))
            }
            _ => Ok(None),
        }
    }

    fn reject_parameter_pattern_predicate(&self, owner: &'static str) -> Result<()> {
        if !matches!(self.peek().kind, TokenKind::Parameter(_)) {
            return Ok(());
        }
        let message = match owner {
            "node" => "a parameter cannot be used as a node pattern predicate",
            "relationship" => "a parameter cannot be used as a relationship pattern predicate",
            _ => "a parameter cannot be used as a pattern predicate",
        };
        self.syntax_detail("InvalidParameterUse", message)
    }

    fn expect_word(&mut self, expected: &str) -> Result<()> {
        if self.eat_word(expected) {
            Ok(())
        } else {
            self.error("expected Cypher keyword")
        }
    }

    fn eat_word(&mut self, expected: &str) -> bool {
        if self.peek_word(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn peek_word(&self, expected: &str) -> bool {
        self.peek_word_at(0, expected)
    }

    fn peek_word_at(&self, offset: usize, expected: &str) -> bool {
        matches!(self.tokens.get(self.cursor + offset).map(|token| &token.kind), Some(TokenKind::Identifier(value)) if value.eq_ignore_ascii_case(expected))
    }

    fn expect_symbol(&mut self, expected: Symbol) -> Result<()> {
        if self.eat_symbol(expected) {
            Ok(())
        } else {
            self.error("expected Cypher punctuation")
        }
    }

    fn eat_symbol(&mut self, expected: Symbol) -> bool {
        if self.peek_symbol(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn peek_symbol(&self, expected: Symbol) -> bool {
        self.peek_symbol_at(0, expected)
    }

    fn peek_symbol_at(&self, offset: usize, expected: Symbol) -> bool {
        matches!(self.tokens.get(self.cursor + offset).map(|token| &token.kind), Some(TokenKind::Symbol(actual)) if *actual == expected)
    }

    fn peek_identifier(&self) -> bool {
        self.peek_identifier_at(0)
    }

    fn peek_identifier_at(&self, offset: usize) -> bool {
        matches!(
            self.tokens
                .get(self.cursor + offset)
                .map(|token| &token.kind),
            Some(TokenKind::Identifier(_))
        )
    }

    fn at_end_or_semicolon(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::End | TokenKind::Symbol(Symbol::Semicolon)
        )
    }

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.cursor)
            .or_else(|| self.tokens.last())
            .unwrap_or(&FALLBACK_END)
    }

    fn advance(&mut self) {
        if self.cursor < self.tokens.len() {
            self.cursor += 1;
        }
    }

    fn consumed_source_from(&self, start: usize) -> Result<Arc<str>> {
        let Some(last_token) = self.tokens.get(self.cursor.saturating_sub(1)) else {
            return Err(Error::internal(
                "projection source capture had no consumed token",
            ));
        };
        let Some(source) = self.source.get(start..last_token.span.end) else {
            return Err(Error::internal("projection source span was invalid"));
        };
        Ok(Arc::from(source))
    }

    fn make_error(&self, message: &'static str) -> Error {
        let span = self.peek().span;
        Error::new(
            ErrorCode::QuerySyntax,
            format!(
                "UnexpectedSyntax: {message} at {}:{} (byte {})",
                span.line, span.column, span.start
            ),
        )
    }

    fn syntax_detail<T>(&self, detail: &'static str, message: &'static str) -> Result<T> {
        let span = self.peek().span;
        Err(Error::new(
            ErrorCode::QuerySyntax,
            format!(
                "{detail}: {message} at {}:{} (byte {})",
                span.line, span.column, span.start
            ),
        ))
    }

    fn error<T>(&self, message: &'static str) -> Result<T> {
        Err(self.make_error(message))
    }
}

static FALLBACK_END: Token = Token {
    kind: TokenKind::End,
    span: Span {
        start: 0,
        end: 0,
        line: 1,
        column: 1,
    },
};

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn parses_native_broker_administration_statements() -> Result<()> {
        assert!(matches!(
            parse("SHOW TOPICS")?.statement,
            Statement::ShowTopics
        ));
        assert!(matches!(
            parse("CREATE TOPIC events PARTITIONS 12")?.statement,
            Statement::CreateTopic { name, partitions, retention_days: None } if name == "events" && partitions == 12
        ));
        assert!(matches!(
            parse("CLEAR TOPIC events")?.statement,
            Statement::ClearTopic { name } if name == "events"
        ));
        assert!(matches!(
            parse("CREATE QUEUE jobs STREAM")?.statement,
            Statement::CreateQueue { name, stream: true, retention_days: None } if name == "jobs"
        ));
        assert!(matches!(
            parse("CREATE TOPIC audit PARTITIONS 3 RETENTION 30 DAYS")?.statement,
            Statement::CreateTopic { name, partitions: 3, retention_days: Some(30) } if name == "audit"
        ));
        assert!(matches!(
            parse("ALTER QUEUE jobs RETENTION 7 DAYS")?.statement,
            Statement::AlterQueueRetention { name, retention_days: 7 } if name == "jobs"
        ));
        assert!(matches!(
            parse("CREATE EXCHANGE routing TYPE TOPIC")?.statement,
            Statement::CreateExchange { name, kind } if name == "routing" && kind == "TOPIC"
        ));
        assert!(matches!(
            parse("BIND QUEUE jobs TO EXCHANGE routing KEY work")?.statement,
            Statement::BindQueue { queue, exchange, routing_key }
                if queue == "jobs" && exchange == "routing" && routing_key == "work"
        ));
        assert!(matches!(
            parse("SHOW CONSUMER LAG")?.statement,
            Statement::ShowConsumerLag
        ));
        Ok(())
    }

    #[test]
    fn deeply_nested_expressions_are_rejected_instead_of_exhausting_the_stack() {
        // Each of these constructs recursed without a bound. Queries parse on a 2 MiB
        // `spawn_blocking` stack, so a few hundred bytes reached the guard page and aborted the
        // process — an abort no `panic` setting intercepts. Every construct must now fail as an
        // ordinary syntax error, and the parser must survive to answer the next query.
        let depth = MAXIMUM_EXPRESSION_DEPTH * 8;
        for source in [
            format!("RETURN {}1{}", "(".repeat(depth), ")".repeat(depth)),
            format!("RETURN {}1{}", "[".repeat(depth), "]".repeat(depth)),
            format!("RETURN {}true", "NOT ".repeat(depth)),
            format!("RETURN {}1", "-".repeat(depth)),
        ] {
            let error = parse(&source).expect_err("deep nesting must be refused");
            assert_eq!(error.code, ErrorCode::QuerySyntax);
        }
    }

    #[test]
    fn long_operator_chains_are_rejected_instead_of_aborting_on_drop() {
        // These extend the expression iteratively, so they parse without recursing and the depth
        // guard never sees them. The tree is still left-nested, and releasing it recurses once per
        // link — before this bound a long enough chain aborted the process while dropping the AST,
        // after parsing had already reported success.
        let chain = MAXIMUM_EXPRESSION_CHAIN + 1;
        for source in [
            format!("RETURN [1]{}", "[0]".repeat(chain)),
            format!("RETURN a{}", ".b".repeat(chain)),
            format!("RETURN 1{}", " + 1".repeat(chain)),
            format!("RETURN true{}", " AND true".repeat(chain)),
        ] {
            let error = parse(&source).expect_err("long operator chains must be refused");
            assert_eq!(error.code, ErrorCode::QuerySyntax);
        }
    }

    #[test]
    fn operator_chains_within_the_limit_still_parse() {
        let chain = MAXIMUM_EXPRESSION_CHAIN / 2;
        parse(&format!("RETURN 1{}", " + 1".repeat(chain))).expect("ordinary chains parse");
        parse("RETURN a.b.c.d[0][1] + 1 + 2 + 3").expect("everyday chains parse");
    }

    #[test]
    fn expression_nesting_within_the_limit_still_parses() {
        // The bound only exists to keep the descent inside the stack; it must sit far above any
        // expression a person or query generator actually writes.
        let depth = MAXIMUM_EXPRESSION_DEPTH - 4;
        let source = format!("RETURN {}1{}", "(".repeat(depth), ")".repeat(depth));
        parse(&source).expect("nesting within the limit must still parse");
        parse("RETURN NOT (1 + (2 * (3 - (4 / 5)))) = false").expect("ordinary nesting parses");
    }

    #[test]
    fn rejected_nesting_does_not_consume_budget_from_later_expressions() {
        // The depth counter is restored on the failure path, so one deep subexpression inside a
        // larger query cannot make an unrelated sibling expression fail.
        let deep = format!("{}1{}", "(".repeat(8), ")".repeat(8));
        let source = format!("RETURN {deep} AS a, {deep} AS b, {deep} AS c");
        parse(&source).expect("sibling expressions each get the full depth budget");
    }

    fn only_call(source: &str) -> Result<CallClause> {
        let query = parse(source)?;
        let Statement::Query(body) = query.statement else {
            return Err(Error::internal("query body required"));
        };
        let [Clause::Call(call)] = body.clauses.as_slice() else {
            return Err(Error::internal("one CALL clause required"));
        };
        Ok(call.clone())
    }

    fn parse_failure(source: &str) -> Result<Error> {
        match parse(source) {
            Ok(_) => Err(Error::internal("query was accepted")),
            Err(error) => Ok(error),
        }
    }

    fn only_match_pattern(source: &str) -> Result<Pattern> {
        let query = parse(source)?;
        let Statement::Query(body) = query.statement else {
            return Err(Error::internal("query body required"));
        };
        let [Clause::Match { patterns, .. }] = body.clauses.as_slice() else {
            return Err(Error::internal("one MATCH clause required"));
        };
        let [pattern] = patterns.as_slice() else {
            return Err(Error::internal("one pattern required"));
        };
        Ok(pattern.clone())
    }

    #[test]
    fn preserves_node_property_predicate_syntax_and_internal_round_trip() -> Result<()> {
        let cases = [
            ("MATCH (n)", false, None),
            ("MATCH (n {})", true, None),
            (
                "MATCH (n {x: 1})",
                true,
                Some(Expression::Literal(ScalarValue::Integer(1))),
            ),
            (
                "MATCH (n {x: $value})",
                true,
                Some(Expression::Parameter("value".to_owned())),
            ),
        ];

        for (source, predicate_present, expected_value) in cases {
            let pattern = only_match_pattern(source)?;
            assert_eq!(
                pattern.start.property_predicate_present, predicate_present,
                "{source}"
            );
            match expected_value {
                None => assert!(pattern.start.properties.is_empty(), "{source}"),
                Some(expected_value) => assert_eq!(
                    pattern.start.properties,
                    [("x".to_owned(), expected_value)],
                    "{source}"
                ),
            }

            let encoded = Expression::pattern_predicate(pattern.clone());
            let decoded = encoded
                .pattern_predicate_pattern()
                .ok_or_else(|| Error::internal("pattern intrinsic failed to decode"))?;
            assert_eq!(decoded, pattern, "{source}");
        }

        let error = parse_failure("MATCH (n $props)")?;
        assert_eq!(error.code, ErrorCode::QuerySyntax);
        assert!(error.message.contains("InvalidParameterUse"));
        Ok(())
    }

    #[test]
    fn parses_standalone_call_with_implicit_arguments() -> Result<()> {
        let call = only_call("CALL test.doNothing")?;
        assert_eq!(call.name, ["test", "doNothing"]);
        assert_eq!(call.argument_mode, CallArgumentMode::Implicit);
        assert!(call.arguments.is_empty());
        assert!(call.yields.is_empty());
        assert!(call.standalone);

        let call = only_call("CALL test.labels YIELD label AS value")?;
        assert_eq!(call.yields.len(), 1);
        assert!(matches!(
            &call.yields[0],
            ProjectionItem {
                expression: Expression::Variable(output),
                alias: Some(alias),
                source_text: None,
            } if output == "label" && alias == "value"
        ));

        let error = parse_failure("MATCH (n) CALL test.doNothing")?;
        assert_eq!(error.code, ErrorCode::QuerySyntax);
        assert!(error.message.contains("InvalidArgumentPassingMode"));

        let error = parse_failure("CALL test.labels YIELD label RETURN label")?;
        assert_eq!(error.code, ErrorCode::QuerySyntax);
        assert!(error.message.contains("InvalidArgumentPassingMode"));
        Ok(())
    }

    #[test]
    fn parses_yield_all_only_for_a_standalone_call() -> Result<()> {
        let call = only_call("CALL test.my.proc('Stefan', 1) YIELD *")?;
        assert_eq!(call.arguments.len(), 2);
        assert_eq!(call.argument_mode, CallArgumentMode::Explicit);
        assert_eq!(call.yield_mode, CallYieldMode::All);
        assert!(
            call.yields.is_empty(),
            "standalone YIELD * normalizes to all procedure outputs"
        );

        let error =
            parse_failure("CALL test.my.proc('Stefan', 1) YIELD * RETURN city, country_code")?;
        assert_eq!(error.code, ErrorCode::QuerySyntax);
        assert!(error.message.contains("YIELD * is only valid"));

        let error = parse_failure("WITH 1 AS seed CALL test.my.proc(seed) YIELD *")?;
        assert_eq!(error.code, ErrorCode::QuerySyntax);
        assert!(error.message.contains("YIELD * is only valid"));
        assert!(parse("CALL test.my.proc('Stefan', 1) YIELD *, city").is_err());
        Ok(())
    }

    #[test]
    fn parses_named_yield_projection_and_aliases() -> Result<()> {
        let call =
            only_call("CALL test.my.proc('Stefan', 1) YIELD city, country_code AS countryCode")?;
        assert_eq!(call.yields.len(), 2);
        assert!(matches!(
            &call.yields[0],
            ProjectionItem {
                expression: Expression::Variable(output),
                alias: None,
                source_text: None,
            } if output == "city"
        ));
        assert!(matches!(
            &call.yields[1],
            ProjectionItem {
                expression: Expression::Variable(output),
                alias: Some(alias),
                source_text: None,
            } if output == "country_code" && alias == "countryCode"
        ));

        assert!(parse("CALL test.my.proc(1) YIELD out RETURN out").is_ok());
        Ok(())
    }

    #[test]
    fn parses_cypher_25_pipeline_clauses() -> Result<()> {
        let query = parse(
            "FOR x IN [1, 2, 3]\n\
             FILTER x > 1\n\
             LET doubled = x * 2, text = toString(x)\n\
             FINISH",
        )?;
        let Statement::Query(body) = query.statement else {
            return Err(Error::internal("expected query body"));
        };
        assert!(matches!(body.clauses[0], Clause::For { .. }));
        assert!(matches!(body.clauses[1], Clause::Filter(_)));
        assert!(matches!(body.clauses[2], Clause::Let(_)));
        assert!(matches!(body.clauses[3], Clause::Finish));
        Ok(())
    }

    #[test]
    fn parses_case_comprehension_reduce_predicate_and_map_projection() -> Result<()> {
        let query = parse(
            "WITH {name: 'Ada'} AS person\n\
             RETURN CASE person.name WHEN 'Ada' THEN 1 ELSE 0 END AS selected,\n\
                    [x IN [1, 2, 3] WHERE x > 1 | x * 2] AS mapped,\n\
                    reduce(total = 0, x IN [1, 2, 3] | total + x) AS total,\n\
                    all(x IN [1, 2] WHERE x > 0) AS valid,\n\
                    person{.*, .name, copy: person.name} AS projected,\n\
                    'abc' =~ 'a.*' AS regex,\n\
                    [1] || [2] AS concatenated",
        )?;
        let Statement::Query(body) = query.statement else {
            return Err(Error::internal("expected query body"));
        };
        let Some(Clause::Return(projection)) = body.clauses.last() else {
            return Err(Error::internal("expected RETURN"));
        };
        assert_eq!(projection.items.len(), 7);
        assert!(matches!(
            projection.items[0].expression,
            Expression::Case { .. }
        ));
        assert!(matches!(
            projection.items[1].expression,
            Expression::ListComprehension { .. }
        ));
        assert!(matches!(
            projection.items[2].expression,
            Expression::Reduce { .. }
        ));
        assert!(matches!(
            projection.items[3].expression,
            Expression::ListPredicate { .. }
        ));
        assert!(matches!(
            projection.items[4].expression,
            Expression::MapProjection { .. }
        ));
        Ok(())
    }

    #[test]
    fn parses_predicates_before_not_and_comparison() -> Result<()> {
        let query = parse(
            "RETURN NOT false >= false AS negated, \
             false = true IN [true, false] AS membership, \
             NOT false IS NULL AS nullable",
        )?;
        let Statement::Query(body) = query.statement else {
            return Err(Error::internal("expected query body"));
        };
        let Some(Clause::Return(projection)) = body.clauses.last() else {
            return Err(Error::internal("expected RETURN"));
        };
        assert!(matches!(
            &projection.items[0].expression,
            Expression::Unary {
                operation: UnaryOperator::Not,
                operand,
            } if matches!(
                operand.as_ref(),
                Expression::Binary { operation: BinaryOperator::GreaterOrEqual, .. }
            )
        ));
        assert!(matches!(
            &projection.items[1].expression,
            Expression::Binary {
                operation: BinaryOperator::Equal,
                right,
                ..
            } if matches!(
                right.as_ref(),
                Expression::Binary { operation: BinaryOperator::In, .. }
            )
        ));
        assert!(matches!(
            &projection.items[2].expression,
            Expression::Unary {
                operation: UnaryOperator::Not,
                operand,
            } if matches!(operand.as_ref(), Expression::IsNull { .. })
        ));
        Ok(())
    }

    #[test]
    fn parses_merge_on_create_and_on_match_actions() -> Result<()> {
        let query = parse(
            "MERGE (n:Item {id: $id})\n\
             ON CREATE SET n.created = true, n.count = 1\n\
             ON MATCH SET n.count = n.count + 1\n\
             RETURN n",
        )?;
        let Statement::Query(body) = query.statement else {
            return Err(Error::internal("expected query body"));
        };
        let Some(Clause::Merge {
            on_create,
            on_match,
            ..
        }) = body.clauses.first()
        else {
            return Err(Error::internal("expected MERGE"));
        };
        assert_eq!(on_create.len(), 2);
        assert_eq!(on_match.len(), 1);
        Ok(())
    }

    #[test]
    fn parses_cross_layer_identity_relationship_write() -> Result<()> {
        let query = parse(
            "USE LAYER OBSERVED, KNOWLEDGE WRITE LAYER KNOWLEDGE \
             MATCH (p:Person {person_id: $person_id}), (h:Handle {value: $value}) \
             MERGE (p)-[:HAS_HANDLE]->(h)",
        )?;
        assert!(query.read_layers.contains_layer(Layer::Observed));
        assert!(query.read_layers.contains_layer(Layer::Knowledge));
        assert_eq!(query.write_layer, Layer::Knowledge);
        Ok(())
    }

    #[test]
    fn parses_enforced_single_property_uniqueness_constraint() -> Result<()> {
        let query = parse(
            "CREATE CONSTRAINT company_code FOR (company:Company) REQUIRE company.code IS UNIQUE",
        )?;
        let Statement::CreateConstraint(definition) = query.statement else {
            return Err(Error::internal("expected uniqueness constraint"));
        };
        assert_eq!(definition.name, "company_code");
        assert_eq!(definition.label, "Company");
        assert_eq!(definition.property, "code");
        assert!(matches!(
            parse("SHOW CONSTRAINTS")?.statement,
            Statement::ShowConstraints
        ));
        assert!(matches!(
            parse("DROP CONSTRAINT IF EXISTS company_code")?.statement,
            Statement::DropConstraint {
                if_exists: true,
                ..
            }
        ));
        assert!(parse("CREATE CONSTRAINT bad FOR (n:Company) REQUIRE x.code IS UNIQUE").is_err());
        Ok(())
    }

    #[test]
    fn parses_read_only_contract_statement_exactly() -> Result<()> {
        assert!(matches!(
            parse("CHECK READ ONLY")?.statement,
            Statement::CheckReadOnly
        ));
        assert!(parse("CHECK READ").is_err());
        assert!(parse("CHECK READ ONLY trailing").is_err());
        Ok(())
    }

    proptest! {
        #[test]
        fn arbitrary_unicode_parse_is_total(
            characters in proptest::collection::vec(any::<char>(), 0..384),
        ) {
            let source: String = characters.into_iter().collect();
            let _ = parse(&source);
        }
    }
}
