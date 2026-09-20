//! Bounded process-local cache for already bound and optimized physical plans.
//!
//! Plans are derived state. They are deliberately excluded from checkpoints and canonical data,
//! and every semantic input which can change binding or physical selection is part of the key.

use std::{
    collections::BTreeMap,
    sync::{Arc, LazyLock, Mutex},
};

use crate::{
    ProjectId, Result,
    execution::BackendKind,
    graph::{LayerMask, NameCatalog},
};

use super::{BindCapabilities, PhysicalPlan, ResultValue, Symbol, TokenKind, lex};

const MAX_CACHE_BYTES: usize = 16 * 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 512;

static CACHE: LazyLock<Mutex<PlanCache>> = LazyLock::new(|| Mutex::new(PlanCache::default()));

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlanCacheKey {
    project: ProjectId,
    normalized_source: Arc<[u8]>,
    schema_generation: [u8; 32],
    index_generation: [u8; 32],
    parameter_shape_hash: [u8; 32],
    statistics_revision: u64,
    backend: u8,
    backend_capability_bits: u64,
    bind_capability_bits: u8,
    layer_bits: u8,
    temporal_mode: u8,
    runtime_checkpoint_eligible: bool,
    scratch_budget_bytes: usize,
    max_result_rows: usize,
}

/// Inputs available before lexing, parsing, and binding. Exact raw source plus every external
/// generation/capability guard points to the normalized full key, so the ordinary repeated-query
/// path returns the prepared plan without repeating frontend or statistics work. A differently
/// formatted equivalent query still converges at the normalized full-key lookup after parsing.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EarlyPlanCacheKey {
    project: ProjectId,
    raw_source: Arc<[u8]>,
    schema_generation: [u8; 32],
    index_generation: [u8; 32],
    parameter_shape_hash: [u8; 32],
    statistics_revision: u64,
    backend: u8,
    backend_capability_bits: u64,
    bind_capability_bits: u8,
    runtime_checkpoint_eligible: bool,
    scratch_budget_bytes: usize,
    max_result_rows: usize,
}

impl PlanCacheKey {
    pub fn new(
        project: ProjectId,
        source: &str,
        catalog: &NameCatalog,
        index_generation: [u8; 32],
        parameters: &BTreeMap<String, ResultValue>,
        statistics_revision: u64,
        backend: BackendKind,
        backend_capability_bits: u64,
        capabilities: BindCapabilities,
        layers: LayerMask,
        temporal_mode: u8,
        runtime_checkpoint_eligible: bool,
        scratch_budget_bytes: usize,
        max_result_rows: usize,
    ) -> Result<Self> {
        let normalized_source = normalized_source(source)?;
        let mut parameter_shape = Vec::with_capacity(parameters.len().saturating_mul(16));
        for (name, value) in parameters {
            push_bytes(&mut parameter_shape, name.as_bytes());
            append_value_shape(&mut parameter_shape, value, 0);
        }
        Ok(Self {
            project,
            normalized_source: Arc::from(normalized_source),
            schema_generation: catalog.optimizer_generation(),
            index_generation,
            parameter_shape_hash: *blake3::hash(&parameter_shape).as_bytes(),
            statistics_revision,
            backend: match backend {
                BackendKind::Cpu => 0,
                BackendKind::Metal => 1,
                BackendKind::Cuda => 2,
            },
            backend_capability_bits,
            bind_capability_bits: u8::from(capabilities.write)
                | (u8::from(capabilities.schema) << 1)
                | (u8::from(capabilities.knowledge_write) << 2)
                | (u8::from(capabilities.workspace_write) << 3)
                | (u8::from(capabilities.require_native_execution) << 4),
            layer_bits: layers.bits(),
            temporal_mode,
            runtime_checkpoint_eligible,
            scratch_budget_bytes,
            max_result_rows,
        })
    }
}

impl EarlyPlanCacheKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        project: ProjectId,
        source: &str,
        catalog: &NameCatalog,
        index_generation: [u8; 32],
        parameters: &BTreeMap<String, ResultValue>,
        statistics_revision: u64,
        backend: BackendKind,
        backend_capability_bits: u64,
        capabilities: BindCapabilities,
        runtime_checkpoint_eligible: bool,
        scratch_budget_bytes: usize,
        max_result_rows: usize,
    ) -> Result<Self> {
        let mut parameter_shape = Vec::with_capacity(parameters.len().saturating_mul(16));
        for (name, value) in parameters {
            push_bytes(&mut parameter_shape, name.as_bytes());
            append_value_shape(&mut parameter_shape, value, 0);
        }
        Ok(Self {
            project,
            raw_source: Arc::from(source.as_bytes()),
            schema_generation: catalog.optimizer_generation(),
            index_generation,
            parameter_shape_hash: *blake3::hash(&parameter_shape).as_bytes(),
            statistics_revision,
            backend: match backend {
                BackendKind::Cpu => 0,
                BackendKind::Metal => 1,
                BackendKind::Cuda => 2,
            },
            backend_capability_bits,
            bind_capability_bits: u8::from(capabilities.write)
                | (u8::from(capabilities.schema) << 1)
                | (u8::from(capabilities.knowledge_write) << 2)
                | (u8::from(capabilities.workspace_write) << 3)
                | (u8::from(capabilities.require_native_execution) << 4),
            runtime_checkpoint_eligible,
            scratch_budget_bytes,
            max_result_rows,
        })
    }
}

#[derive(Clone)]
struct CachedPlan {
    plan: PhysicalPlan,
    /// Pre-optimization adaptive decision. Optimized plans intentionally lose some source-shape
    /// detail, so execution must consume this cached decision rather than trying to infer it again.
    force_host_adaptive: bool,
    estimated_bytes: usize,
    last_used: u64,
    /// Parameter bindings this plan was specialized against, and therefore only valid for.
    ///
    /// The key deliberately records parameters by *shape* — type and a coarse magnitude or length
    /// class — so a single plan serves every binding of the same shape. That is what makes the
    /// cache useful, and it is sound as long as planning never reads a parameter's value. A few
    /// rewrites do: resolving `n[$key]` to a static property access folds the parameter's text into
    /// the operator tree. Two different property names of the same length share a key, so without
    /// this guard such a plan would be handed back for the wrong property and silently return the
    /// wrong column. Guarding only the specialized bindings keeps every other plan fully shared.
    specialized_parameters: BTreeMap<String, ResultValue>,
}

#[derive(Default)]
struct PlanCache {
    entries: BTreeMap<PlanCacheKey, CachedPlan>,
    early: BTreeMap<EarlyPlanCacheKey, PlanCacheKey>,
    bytes: usize,
    clock: u64,
}

impl PlanCache {
    fn get_early(
        &mut self,
        key: &EarlyPlanCacheKey,
        parameters: &BTreeMap<String, ResultValue>,
    ) -> Option<(PhysicalPlan, bool)> {
        let full = self.early.get(key)?.clone();
        self.get(&full, parameters)
    }

    fn get(
        &mut self,
        key: &PlanCacheKey,
        parameters: &BTreeMap<String, ResultValue>,
    ) -> Option<(PhysicalPlan, bool)> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(key)?;
        if entry
            .specialized_parameters
            .iter()
            .any(|(name, folded)| parameters.get(name) != Some(folded))
        {
            // The plan was built for different bindings of these parameters. Treat it as a miss and
            // re-plan rather than returning a plan that answers a different question.
            return None;
        }
        entry.last_used = self.clock;
        Some((entry.plan.clone(), entry.force_host_adaptive))
    }

    fn insert(
        &mut self,
        early_key: EarlyPlanCacheKey,
        key: PlanCacheKey,
        plan: PhysicalPlan,
        force_host_adaptive: bool,
        specialized_parameters: BTreeMap<String, ResultValue>,
    ) {
        let estimated_bytes =
            estimated_plan_bytes(&key, &plan).saturating_add(early_key.raw_source.len());
        if estimated_bytes > MAX_CACHE_BYTES {
            return;
        }
        self.clock = self.clock.wrapping_add(1);
        if let Some(previous) = self.entries.remove(&key) {
            self.bytes = self.bytes.saturating_sub(previous.estimated_bytes);
        }
        self.bytes = self.bytes.saturating_add(estimated_bytes);
        self.early.retain(|_, full| full != &key);
        self.early.insert(early_key, key.clone());
        self.entries.insert(
            key,
            CachedPlan {
                plan,
                force_host_adaptive,
                estimated_bytes,
                last_used: self.clock,
                specialized_parameters,
            },
        );
        while self.bytes > MAX_CACHE_BYTES || self.entries.len() > MAX_CACHE_ENTRIES {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(key, entry)| (entry.last_used, *key))
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.early.retain(|_, full| full != &oldest);
                self.bytes = self.bytes.saturating_sub(removed.estimated_bytes);
            }
        }
    }
}

pub fn get(
    key: &PlanCacheKey,
    parameters: &BTreeMap<String, ResultValue>,
) -> Option<(PhysicalPlan, bool)> {
    cache().get(key, parameters)
}

pub fn get_early(
    key: &EarlyPlanCacheKey,
    parameters: &BTreeMap<String, ResultValue>,
) -> Option<(PhysicalPlan, bool)> {
    cache().get_early(key, parameters)
}

pub fn insert(
    early_key: EarlyPlanCacheKey,
    key: PlanCacheKey,
    plan: PhysicalPlan,
    force_host_adaptive: bool,
    specialized_parameters: BTreeMap<String, ResultValue>,
) {
    cache().insert(
        early_key,
        key,
        plan,
        force_host_adaptive,
        specialized_parameters,
    );
}

fn cache() -> std::sync::MutexGuard<'static, PlanCache> {
    CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn estimated_plan_bytes(key: &PlanCacheKey, plan: &PhysicalPlan) -> usize {
    let operator_count = plan.operators.len().saturating_add(
        plan.unions
            .iter()
            .map(|(_, operators)| operators.len())
            .sum::<usize>(),
    );
    std::mem::size_of::<PlanCacheKey>()
        .saturating_add(key.normalized_source.len().saturating_mul(4))
        .saturating_add(std::mem::size_of::<PhysicalPlan>())
        .saturating_add(
            operator_count.saturating_mul(std::mem::size_of::<super::PhysicalOperator>()),
        )
        .saturating_add(1_024)
}

fn normalized_source(source: &str) -> Result<Vec<u8>> {
    let tokens = lex(source)?;
    let mut output = Vec::with_capacity(source.len());
    for token in tokens {
        match token.kind {
            TokenKind::Identifier(value) => {
                output.push(0);
                push_bytes(&mut output, value.as_bytes());
            }
            TokenKind::String(value) => {
                output.push(1);
                push_bytes(&mut output, value.as_bytes());
            }
            TokenKind::Integer(value) => {
                output.push(2);
                output.extend_from_slice(&value.to_be_bytes());
            }
            TokenKind::Float(value) => {
                output.push(3);
                output.extend_from_slice(&value.to_bits().to_be_bytes());
            }
            TokenKind::Parameter(value) => {
                output.push(4);
                push_bytes(&mut output, value.as_bytes());
            }
            TokenKind::Symbol(symbol) => output.extend_from_slice(&[5, symbol_byte(symbol)]),
            TokenKind::End => output.push(6),
        }
    }
    Ok(output)
}

const fn symbol_byte(symbol: Symbol) -> u8 {
    match symbol {
        Symbol::LeftParen => 0,
        Symbol::RightParen => 1,
        Symbol::LeftBracket => 2,
        Symbol::RightBracket => 3,
        Symbol::LeftBrace => 4,
        Symbol::RightBrace => 5,
        Symbol::Comma => 6,
        Symbol::Dot => 7,
        Symbol::Colon => 8,
        Symbol::Semicolon => 9,
        Symbol::Dollar => 10,
        Symbol::Pipe => 11,
        Symbol::DoublePipe => 12,
        Symbol::Ampersand => 13,
        Symbol::Bang => 14,
        Symbol::Plus => 15,
        Symbol::Minus => 16,
        Symbol::Star => 17,
        Symbol::Slash => 18,
        Symbol::Percent => 19,
        Symbol::Caret => 20,
        Symbol::Equal => 21,
        Symbol::NotEqual => 22,
        Symbol::Less => 23,
        Symbol::LessOrEqual => 24,
        Symbol::Greater => 25,
        Symbol::GreaterOrEqual => 26,
        Symbol::RegexMatch => 27,
        Symbol::ArrowLeft => 28,
        Symbol::ArrowRight => 29,
        Symbol::Range => 30,
    }
}

fn append_value_shape(output: &mut Vec<u8>, value: &ResultValue, depth: usize) {
    if depth >= 16 {
        output.push(u8::MAX);
        return;
    }
    output.push(match value.column_type() {
        super::ColumnType::Any => 0,
        super::ColumnType::Null => 1,
        super::ColumnType::Boolean => 2,
        super::ColumnType::Integer => 3,
        super::ColumnType::Float => 4,
        super::ColumnType::String => 5,
        super::ColumnType::Bytes => 6,
        super::ColumnType::Temporal => 7,
        super::ColumnType::Duration => 8,
        super::ColumnType::Node => 9,
        super::ColumnType::Relationship => 10,
        super::ColumnType::Path => 11,
        super::ColumnType::Vector => 12,
        super::ColumnType::List => 13,
        super::ColumnType::Map => 14,
    });
    match value {
        ResultValue::Scalar(crate::ScalarValue::Null) => output.push(0),
        ResultValue::Scalar(crate::ScalarValue::Boolean(value)) => {
            output.push(1 + u8::from(*value));
        }
        ResultValue::Scalar(crate::ScalarValue::Integer(value)) => {
            output.push(3);
            output.push(if *value < 0 {
                0
            } else if *value == 0 {
                1
            } else {
                2
            });
            output.push(integer_selectivity_bucket(*value));
        }
        ResultValue::Scalar(crate::ScalarValue::Float(value)) => {
            output.push(4);
            output.extend_from_slice(&float_selectivity_class(value.into_inner()));
        }
        ResultValue::Scalar(crate::ScalarValue::String(value)) => {
            output.push(length_bucket(value.len()));
        }
        ResultValue::Scalar(crate::ScalarValue::Bytes(value)) => {
            output.push(length_bucket(value.len()));
        }
        ResultValue::Scalar(crate::ScalarValue::List(value)) => {
            output.push(length_bucket(value.as_bytes().len()));
        }
        ResultValue::Scalar(crate::ScalarValue::Map(value)) => {
            output.push(length_bucket(value.as_bytes().len()));
        }
        ResultValue::Vector(values) => output.push(length_bucket(values.len())),
        ResultValue::List(values) => {
            output.push(length_bucket(values.len()));
            for value in values.iter().take(16) {
                append_value_shape(output, value, depth + 1);
            }
        }
        ResultValue::Map(values) => {
            output.push(length_bucket(values.len()));
            for (name, value) in values.iter().take(16) {
                push_bytes(output, name.as_bytes());
                append_value_shape(output, value, depth + 1);
            }
        }
        _ => {}
    }
}

fn integer_selectivity_bucket(value: i64) -> u8 {
    let magnitude = value.unsigned_abs();
    if magnitude == 0 {
        0
    } else {
        ((u64::BITS - magnitude.leading_zeros() - 1) / 4) as u8
    }
}

fn float_selectivity_class(value: f64) -> [u8; 3] {
    if value.is_nan() {
        return [0, 0, 0];
    }
    if value == f64::NEG_INFINITY {
        return [1, 0, 0];
    }
    if value == f64::INFINITY {
        return [2, 0, 0];
    }
    if value == 0.0 {
        return [3, u8::from(value.is_sign_negative()), 0];
    }
    let bits = value.abs().to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as u16;
    [4, u8::from(value.is_sign_negative()), (exponent / 16) as u8]
}

fn length_bucket(length: usize) -> u8 {
    if length == 0 {
        0
    } else {
        (usize::BITS - length.leading_zeros()).min(u32::from(u8::MAX)) as u8
    }
}

fn push_bytes(output: &mut Vec<u8>, bytes: &[u8]) {
    output.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    output.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{ProjectId, ScalarValue, cypher::ResultValue, execution::BackendKind};

    use super::{BindCapabilities, EarlyPlanCacheKey, PlanCacheKey};

    #[test]
    fn prepared_plan_cache_hot_path_guards_early_lookup() -> crate::Result<()> {
        let project = ProjectId(uuid::Uuid::nil());
        let catalog = crate::graph::NameCatalog::default();
        let parameters = BTreeMap::from([(
            "value".to_owned(),
            ResultValue::Scalar(ScalarValue::Integer(7)),
        )]);
        let full = PlanCacheKey::new(
            project,
            "RETURN $value",
            &catalog,
            [4; 32],
            &parameters,
            19,
            BackendKind::Cpu,
            super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
            BindCapabilities::default(),
            crate::graph::LayerMask::AUTHORITY,
            0,
            true,
            4096,
            100,
        )?;
        let early = EarlyPlanCacheKey::new(
            project,
            "RETURN $value",
            &catalog,
            [4; 32],
            &parameters,
            19,
            BackendKind::Cpu,
            super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
            BindCapabilities::default(),
            true,
            4096,
            100,
        )?;
        let plan = super::PhysicalPlan {
            project: None,
            read_layers: crate::graph::LayerMask::AUTHORITY,
            write_layer: crate::Layer::Observed,
            at_time: None,
            operators: Vec::new(),
            unions: Vec::new(),
            read_only: true,
            dependencies: Vec::new(),
        };
        let mut cache = super::PlanCache::default();
        cache.insert(early.clone(), full, plan, true, BTreeMap::new());
        let (_, force_host_adaptive) = cache
            .get_early(&early, &parameters)
            .ok_or_else(|| crate::Error::internal("prepared plan cache missed its hot key"))?;
        assert!(force_host_adaptive);

        let stale = EarlyPlanCacheKey::new(
            project,
            "RETURN $value",
            &catalog,
            [4; 32],
            &parameters,
            20,
            BackendKind::Cpu,
            super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
            BindCapabilities::default(),
            true,
            4096,
            100,
        )?;
        assert!(cache.get_early(&stale, &parameters).is_none());

        let spaced_early = EarlyPlanCacheKey::new(
            project,
            "  RETURN   $value  ",
            &catalog,
            [4; 32],
            &parameters,
            19,
            BackendKind::Cpu,
            super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
            BindCapabilities::default(),
            true,
            4096,
            100,
        )?;
        let spaced_full = PlanCacheKey::new(
            project,
            "  RETURN   $value  ",
            &catalog,
            [4; 32],
            &parameters,
            19,
            BackendKind::Cpu,
            super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
            BindCapabilities::default(),
            crate::graph::LayerMask::AUTHORITY,
            0,
            true,
            4096,
            100,
        )?;
        assert_ne!(spaced_early, early);
        assert_eq!(cache.early.get(&early), Some(&spaced_full));
        Ok(())
    }

    #[test]
    fn parameter_shape_separates_types_without_keying_scalar_values() -> crate::Result<()> {
        let project = ProjectId(uuid::Uuid::nil());
        let catalog = crate::graph::NameCatalog::default();
        let key = |value| {
            PlanCacheKey::new(
                project,
                "RETURN $value",
                &catalog,
                [0; 32],
                &BTreeMap::from([("value".into(), value)]),
                0,
                BackendKind::Cpu,
                super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
                BindCapabilities::default(),
                crate::graph::LayerMask::AUTHORITY,
                0,
                true,
                usize::MAX,
                1_000,
            )
        };
        let one = key(ResultValue::Scalar(ScalarValue::Integer(1)))?;
        let other_integer = key(ResultValue::Scalar(ScalarValue::Integer(9)))?;
        let distant_integer = key(ResultValue::Scalar(ScalarValue::Integer(1_000_000)))?;
        let boolean = key(ResultValue::Scalar(ScalarValue::Boolean(true)))?;
        assert_eq!(one, other_integer);
        assert_ne!(one, distant_integer);
        assert_ne!(one, boolean);
        Ok(())
    }

    #[test]
    fn a_value_specialized_plan_is_not_reused_for_a_different_binding() -> crate::Result<()> {
        // `"name"` and `"type"` are both four bytes, so they share a length class and therefore a
        // cache key — that sharing is intentional and is what makes the cache effective. It is only
        // unsound when planning folded the value into the plan, as the dynamic node-property
        // rewrite does for `n[$key]`. Without the guard the second lookup returned the plan built
        // for `name` and the query silently answered with the wrong property.
        let project = ProjectId(uuid::Uuid::nil());
        let catalog = crate::graph::NameCatalog::default();
        let binding = |value: &str| {
            BTreeMap::from([(
                "key".to_owned(),
                ResultValue::Scalar(ScalarValue::String(value.into())),
            )])
        };
        let cache_key = |parameters: &BTreeMap<String, ResultValue>| {
            PlanCacheKey::new(
                project,
                "MATCH (n) RETURN n[$key] AS value",
                &catalog,
                [0; 32],
                parameters,
                0,
                BackendKind::Cpu,
                super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
                BindCapabilities::default(),
                crate::graph::LayerMask::AUTHORITY,
                0,
                true,
                usize::MAX,
                1_000,
            )
        };

        let name = binding("name");
        let kind = binding("type");
        let key = cache_key(&name)?;
        let early = EarlyPlanCacheKey::new(
            project,
            "MATCH (n) RETURN n[$key] AS value",
            &catalog,
            [0; 32],
            &name,
            0,
            BackendKind::Cpu,
            super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
            BindCapabilities::default(),
            true,
            usize::MAX,
            1_000,
        )?;
        assert_eq!(
            key,
            cache_key(&kind)?,
            "same-length string parameters must still share a key"
        );

        let empty_plan = || super::PhysicalPlan {
            project: None,
            read_layers: crate::graph::LayerMask::AUTHORITY,
            write_layer: crate::Layer::Observed,
            at_time: None,
            operators: Vec::new(),
            unions: Vec::new(),
            read_only: true,
            dependencies: Vec::new(),
        };
        let mut cache = super::PlanCache::default();
        cache.insert(
            early.clone(),
            key.clone(),
            empty_plan(),
            false,
            name.clone(),
        );

        assert!(
            cache.get(&key, &name).is_some(),
            "the binding the plan was specialized for must still hit"
        );
        assert!(
            cache.get(&key, &kind).is_none(),
            "a plan specialized for one binding must not be reused for another"
        );

        // A plan that folded nothing keeps sharing across every binding of the same shape.
        let mut unspecialized = super::PlanCache::default();
        unspecialized.insert(early, key.clone(), empty_plan(), false, BTreeMap::new());
        assert!(unspecialized.get(&key, &name).is_some());
        assert!(unspecialized.get(&key, &kind).is_some());
        Ok(())
    }

    #[test]
    fn source_key_ignores_layout_and_comments_but_not_literals() -> crate::Result<()> {
        let project = ProjectId(uuid::Uuid::nil());
        let catalog = crate::graph::NameCatalog::default();
        let key = |source| {
            PlanCacheKey::new(
                project,
                source,
                &catalog,
                [0; 32],
                &BTreeMap::new(),
                0,
                BackendKind::Cpu,
                super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
                BindCapabilities::default(),
                crate::graph::LayerMask::AUTHORITY,
                0,
                true,
                usize::MAX,
                1_000,
            )
        };
        let one = key("RETURN 1")?;
        assert_eq!(one, key("/* layout */\n RETURN   1")?);
        assert_ne!(one, key("RETURN 2")?);
        Ok(())
    }

    #[test]
    fn physical_generations_and_execution_class_never_alias() -> crate::Result<()> {
        let base = PlanCacheKey::new(
            ProjectId(uuid::Uuid::nil()),
            "MATCH (n) RETURN n",
            &crate::graph::NameCatalog::default(),
            [0; 32],
            &BTreeMap::new(),
            7,
            BackendKind::Cpu,
            super::super::optimizer::backend_capability_bits(BackendKind::Cpu),
            BindCapabilities::default(),
            crate::graph::LayerMask::AUTHORITY,
            0,
            true,
            usize::MAX,
            1_000,
        )?;
        let mut variants = Vec::new();
        let mut value = base.clone();
        value.index_generation[0] = 1;
        variants.push(value);
        let mut value = base.clone();
        value.statistics_revision += 1;
        variants.push(value);
        let mut value = base.clone();
        value.backend_capability_bits ^= 1;
        variants.push(value);
        let mut value = base.clone();
        value.layer_bits ^= 1;
        variants.push(value);
        let mut value = base.clone();
        value.temporal_mode = 1;
        variants.push(value);
        let mut value = base.clone();
        value.runtime_checkpoint_eligible = false;
        variants.push(value);
        let mut value = base.clone();
        value.scratch_budget_bytes = 4096;
        variants.push(value);
        assert!(variants.iter().all(|variant| variant != &base));
        Ok(())
    }
}
