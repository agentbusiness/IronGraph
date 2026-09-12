// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg(all(feature = "accelerator", target_os = "macos"))]

use std::{
    sync::{Mutex, MutexGuard, OnceLock},
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, ProjectId, Result,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, MetalBackend, ResidentDeviceCompletion,
        ResidentExecutionId, ResidentProjectImage, ResidentQuantifierBinary,
        ResidentQuantifierExpression as Expr, ResidentQuantifierGeneration, ResidentQuantifierKind,
        ResidentQuantifierOutput, ResidentQuantifierProgram, ResidentQuantifierProgramRequest,
        ResidentQuantifierProgramResult, ResidentQuantifierProjection, ResidentQuantifierSlot,
        ResidentQuantifierStage, ResidentQuantifierValue as Value,
    },
    graph::{GraphStore, IndexCatalog, TemporalStore},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(0x5041_5241_4c4c_454c_5155_414e_54));
const MEMORY_LIMIT: usize = 512 * 1024 * 1024;
const RESERVED: usize = 16 * 1024 * 1024;

const FANOUTS: [usize; 3] = [16, 32, 64];
const WARMUP_ROUNDS: usize = 2;
const SAMPLE_ROUNDS: usize = 7;

// This is deliberately a relative contract. Each step admits four times as many rows, so a
// parallel implementation must improve throughput by at least 1 / 0.85 = 1.176x at each step.
// Across the complete 16x range, throughput must improve by at least 1 / 0.70 = 1.429x.
const MAX_STEP_NORMALIZED_COST: f64 = 0.85;
const MAX_END_TO_END_NORMALIZED_COST: f64 = 0.70;
const MAX_RELATIVE_MAD: f64 = 0.35;

fn metal_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Fixture {
    cpu: CpuBackend,
    metal: MetalBackend,
    generation: ResidentQuantifierGeneration,
}

impl Fixture {
    fn new() -> Result<Self> {
        let graph = GraphStore::default();
        let temporal = TemporalStore::default();
        let indexes = IndexCatalog::default();
        let bookmark = Bookmark {
            term: 29,
            index: graph.revision(),
        };
        let generation = ResidentQuantifierGeneration {
            project: PROJECT,
            bookmark,
            graph_revision: graph.revision(),
            layout_version: graph.layout_version(),
            catalog_generation: graph.catalog().optimizer_generation(),
        };

        let mut cpu = CpuBackend::new(MEMORY_LIMIT, RESERVED);
        cpu.admit_project(ResidentProjectImage::build(
            PROJECT, bookmark, &graph, &temporal, &indexes,
        )?)?;

        let mut metal = MetalBackend::new(0, MEMORY_LIMIT, RESERVED)?;
        metal.admit_project(ResidentProjectImage::build(
            PROJECT, bookmark, &graph, &temporal, &indexes,
        )?)?;

        assert!(cpu.supports_native_quantifier_program());
        assert!(metal.supports_native_quantifier_program());
        Ok(Self {
            cpu,
            metal,
            generation,
        })
    }

    fn request(
        &self,
        fanout: usize,
        execution_low: u64,
    ) -> Result<ResidentQuantifierProgramRequest> {
        let admitted_rows = fanout * fanout;
        ResidentQuantifierProgramRequest::build(
            self.generation,
            ResidentExecutionId {
                high: 0x5041_5241_4c4c_454c,
                low: execution_low,
            },
            scaling_program(fanout),
            admitted_rows,
            fanout,
            0x5155_414e_5449_4649 ^ fanout as u64,
        )
    }
}

fn integer(value: i64) -> Expr {
    Expr::Literal(Value::Integer(value))
}

fn slot(index: u16) -> Expr {
    Expr::Slot(ResidentQuantifierSlot(index))
}

fn list(values: impl IntoIterator<Item = Expr>) -> Expr {
    Expr::List(values.into_iter().collect())
}

fn binary(left: Expr, operation: ResidentQuantifierBinary, right: Expr) -> Expr {
    Expr::Binary {
        left: Box::new(left),
        operation,
        right: Box::new(right),
    }
}

fn project(output: u16, expression: Expr) -> ResidentQuantifierProjection {
    ResidentQuantifierProjection {
        output: ResidentQuantifierSlot(output),
        expression,
    }
}

fn sequence(fanout: usize) -> Expr {
    list((0..fanout).map(|value| integer(value as i64)))
}

fn add_offset(source: Expr, offset: i64) -> Expr {
    if offset == 0 {
        source
    } else {
        binary(source, ResidentQuantifierBinary::Add, integer(offset))
    }
}

/// Produces a non-empty list comprehension and then evaluates an ALL whose body contains an ANY.
/// The value is deterministically true, but every retained relation row performs the complete
/// comprehension and both quantifier reductions on the selected backend.
fn nested_quantifier_over_comprehension() -> Expr {
    let row_sum = binary(slot(1), ResidentQuantifierBinary::Add, slot(3));
    let comprehension = Expr::ListComprehension {
        variable: ResidentQuantifierSlot(7),
        list: Box::new(list(
            (0..8).map(|offset| add_offset(row_sum.clone(), offset)),
        )),
        predicate: Some(Box::new(binary(
            slot(7),
            ResidentQuantifierBinary::GreaterOrEqual,
            integer(0),
        ))),
        projection: Some(Box::new(binary(
            slot(7),
            ResidentQuantifierBinary::Modulo,
            integer(7),
        ))),
    };
    let inner = Expr::Predicate {
        kind: ResidentQuantifierKind::Any,
        variable: ResidentQuantifierSlot(9),
        list: Box::new(list((0..8).map(|offset| add_offset(slot(8), offset)))),
        predicate: Box::new(binary(
            slot(9),
            ResidentQuantifierBinary::GreaterOrEqual,
            slot(8),
        )),
    };
    Expr::Predicate {
        kind: ResidentQuantifierKind::All,
        variable: ResidentQuantifierSlot(8),
        list: Box::new(comprehension),
        predicate: Box::new(inner),
    }
}

fn scaling_program(fanout: usize) -> ResidentQuantifierProgram {
    ResidentQuantifierProgram {
        slot_count: 10,
        stages: vec![
            ResidentQuantifierStage::Project {
                keep_scope: false,
                bindings: vec![project(0, sequence(fanout))],
            },
            ResidentQuantifierStage::Unwind {
                expression: slot(0),
                output: ResidentQuantifierSlot(1),
            },
            ResidentQuantifierStage::Project {
                keep_scope: true,
                bindings: vec![project(2, sequence(fanout))],
            },
            ResidentQuantifierStage::Unwind {
                expression: slot(2),
                output: ResidentQuantifierSlot(3),
            },
            ResidentQuantifierStage::Filter {
                predicate: binary(
                    binary(slot(3), ResidentQuantifierBinary::Modulo, integer(2)),
                    ResidentQuantifierBinary::Equal,
                    integer(0),
                ),
            },
            ResidentQuantifierStage::Project {
                keep_scope: false,
                bindings: vec![
                    project(4, slot(1)),
                    project(5, nested_quantifier_over_comprehension()),
                ],
            },
            ResidentQuantifierStage::GroupCount {
                groups: vec![project(4, slot(4)), project(5, slot(5))],
                count_outputs: vec![ResidentQuantifierSlot(6)],
            },
        ],
        outputs: vec![
            ResidentQuantifierOutput {
                name: "outer".to_owned(),
                source: ResidentQuantifierSlot(4),
            },
            ResidentQuantifierOutput {
                name: "nested".to_owned(),
                source: ResidentQuantifierSlot(5),
            },
            ResidentQuantifierOutput {
                name: "count".to_owned(),
                source: ResidentQuantifierSlot(6),
            },
        ],
    }
}

fn exact_expected_rows(fanout: usize) -> Vec<Vec<Value>> {
    assert_eq!(fanout % 2, 0, "scaling fanout must stay even");
    (0..fanout)
        .map(|outer| {
            vec![
                Value::Integer(outer as i64),
                Value::Boolean(true),
                Value::Integer((fanout / 2) as i64),
            ]
        })
        .collect()
}

fn expected_receipt_cardinalities(fanout: usize) -> Vec<(u64, u64)> {
    let fanout = fanout as u64;
    let expanded = fanout * fanout;
    let retained = expanded / 2;
    vec![
        (1, 1),
        (1, fanout),
        (fanout, fanout),
        (fanout, expanded),
        (expanded, retained),
        (retained, retained),
        (retained, fanout),
        (fanout, fanout),
    ]
}

fn validate_result(
    result: ResidentQuantifierProgramResult,
    request: &ResidentQuantifierProgramRequest,
    backend: BackendKind,
    completion: ResidentDeviceCompletion,
    expected_rows: &[Vec<Value>],
    expected_cardinalities: &[(u64, u64)],
) -> Result<()> {
    let result = result.validate(request, backend)?;
    assert_eq!(result.rows(), expected_rows);

    let obligations = request.obligations();
    assert_eq!(result.receipts().len(), obligations.len());
    assert_eq!(result.receipts().len(), expected_cardinalities.len());
    for ((receipt, obligation), (expected_input, expected_output)) in result
        .receipts()
        .iter()
        .zip(&obligations)
        .zip(expected_cardinalities)
    {
        assert_eq!(receipt.execution, request.execution);
        assert_eq!(receipt.obligation, *obligation);
        assert_eq!(receipt.completion, completion);
        assert_eq!(receipt.input_cardinality, *expected_input);
        assert_eq!(receipt.output_cardinality, *expected_output);
    }
    Ok(())
}

struct WorkloadCase {
    fanout: usize,
    admitted_rows: usize,
    request: ResidentQuantifierProgramRequest,
    oracle_rows: Vec<Vec<Value>>,
    receipt_cardinalities: Vec<(u64, u64)>,
}

fn prepare_cases(fixture: &Fixture) -> Result<Vec<WorkloadCase>> {
    FANOUTS
        .iter()
        .enumerate()
        .map(|(index, fanout)| {
            let request = fixture.request(*fanout, 0x1000 + index as u64)?;
            let expected_rows = exact_expected_rows(*fanout);
            let receipt_cardinalities = expected_receipt_cardinalities(*fanout);
            let cpu_result = fixture
                .cpu
                .execute_quantifier_program(&request, &CancellationToken::new())?;
            validate_result(
                cpu_result,
                &request,
                BackendKind::Cpu,
                ResidentDeviceCompletion::CpuReference,
                &expected_rows,
                &receipt_cardinalities,
            )?;
            Ok(WorkloadCase {
                fanout: *fanout,
                admitted_rows: *fanout * *fanout,
                request,
                oracle_rows: expected_rows,
                receipt_cardinalities,
            })
        })
        .collect()
}

fn execute_metal_case(fixture: &Fixture, case: &WorkloadCase) -> Result<Duration> {
    let started = Instant::now();
    let result = fixture
        .metal
        .execute_quantifier_program(&case.request, &CancellationToken::new())?;
    // execute_quantifier_program includes the device-to-host synchronization and packet decode.
    // Result validation is intentionally outside the timed interval.
    let elapsed = started.elapsed();
    validate_result(
        result,
        &case.request,
        BackendKind::Metal,
        ResidentDeviceCompletion::Metal,
        &case.oracle_rows,
        &case.receipt_cardinalities,
    )?;
    Ok(elapsed)
}

fn median(mut samples: Vec<Duration>) -> Duration {
    assert!(!samples.is_empty());
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn absolute_difference(left: Duration, right: Duration) -> Duration {
    if left >= right {
        left - right
    } else {
        right - left
    }
}

struct ScalingMeasurement {
    fanout: usize,
    admitted_rows: usize,
    samples: Vec<Duration>,
    median: Duration,
    median_absolute_deviation: Duration,
}

impl ScalingMeasurement {
    fn new(fanout: usize, admitted_rows: usize, samples: Vec<Duration>) -> Self {
        let measured_median = median(samples.clone());
        let deviations = samples
            .iter()
            .map(|sample| absolute_difference(*sample, measured_median))
            .collect();
        Self {
            fanout,
            admitted_rows,
            samples,
            median: measured_median,
            median_absolute_deviation: median(deviations),
        }
    }

    fn normalized_nanoseconds_per_row(&self) -> f64 {
        self.median.as_secs_f64() * 1_000_000_000.0 / self.admitted_rows as f64
    }

    fn relative_mad(&self) -> f64 {
        self.median_absolute_deviation.as_secs_f64() / self.median.as_secs_f64()
    }
}

fn measure_scaling(fixture: &Fixture) -> Result<Vec<ScalingMeasurement>> {
    let cases = prepare_cases(fixture)?;

    // Warm the library, pipeline state, command allocator, and all three arena sizes before any
    // sample is retained. Largest-first avoids making its first allocation a measured outlier.
    for _ in 0..WARMUP_ROUNDS {
        for index in [2, 1, 0] {
            let _ = execute_metal_case(fixture, &cases[index])?;
        }
    }

    // Rotate the order to spread thermal/frequency drift across sizes. Seven samples make the
    // median and median absolute deviation robust to isolated scheduling noise.
    const ORDERS: [[usize; 3]; 3] = [[0, 2, 1], [2, 1, 0], [1, 0, 2]];
    let mut samples = vec![Vec::with_capacity(SAMPLE_ROUNDS); cases.len()];
    for round in 0..SAMPLE_ROUNDS {
        for index in ORDERS[round % ORDERS.len()] {
            samples[index].push(execute_metal_case(fixture, &cases[index])?);
        }
    }

    let measurements = cases
        .iter()
        .zip(samples)
        .map(|(case, samples)| ScalingMeasurement::new(case.fanout, case.admitted_rows, samples))
        .collect::<Vec<_>>();
    eprintln!("fanout rows samples median_ms mad_pct ns_per_row");
    for measurement in &measurements {
        eprintln!(
            "{} {} {} {:.3} {:.1} {:.1}",
            measurement.fanout,
            measurement.admitted_rows,
            measurement.samples.len(),
            measurement.median.as_secs_f64() * 1_000.0,
            measurement.relative_mad() * 100.0,
            measurement.normalized_nanoseconds_per_row(),
        );
    }
    Ok(measurements)
}

/// Characterization is useful before and after kernel work. It proves semantic parity and prints
/// normalized scaling, but intentionally makes no performance assertion.
#[test]
#[ignore = "requires exclusive access to a real Metal device; characterization only"]
fn real_metal_quantifier_parallelism_characterization_matches_cpu_oracle() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let measurements = measure_scaling(&fixture)?;
    assert_eq!(measurements.len(), FANOUTS.len());
    assert!(
        measurements
            .iter()
            .all(|measurement| measurement.samples.len() == SAMPLE_ROUNDS)
    );
    Ok(())
}

/// GPU-throughput acceptance gate for the native multistage interpreter.
///
/// Limitation: the public result packet proves Metal completion, ordered stage obligations, and
/// cardinalities, but exposes no device-authored work-item/threadgroup count. Timing therefore
/// cannot prove a particular dispatch topology. Once such provenance is added to the ABI, this
/// test should also require it. Until then, normalized scaling is the least machine-specific
/// observable: no absolute latency is asserted, and medians are taken from interleaved samples.
#[test]
#[ignore = "requires exclusive access to a real Metal device"]
fn real_metal_quantifier_parallel_scaling_requires_rising_throughput() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let measurements = measure_scaling(&fixture)?;
    let mut violations = Vec::new();

    for measurement in &measurements {
        if measurement.relative_mad() > MAX_RELATIVE_MAD {
            violations.push(format!(
                "{} rows had {:.1}% relative MAD (limit {:.1}%): measurement is too noisy",
                measurement.admitted_rows,
                measurement.relative_mad() * 100.0,
                MAX_RELATIVE_MAD * 100.0,
            ));
        }
    }

    for pair in measurements.windows(2) {
        let smaller = &pair[0];
        let larger = &pair[1];
        let work_ratio = larger.admitted_rows as f64 / smaller.admitted_rows as f64;
        let latency_ratio = larger.median.as_secs_f64() / smaller.median.as_secs_f64();
        let normalized_ratio =
            larger.normalized_nanoseconds_per_row() / smaller.normalized_nanoseconds_per_row();
        if normalized_ratio > MAX_STEP_NORMALIZED_COST {
            violations.push(format!(
                "{} -> {} rows ({work_ratio:.1}x work) took {latency_ratio:.2}x as long; \
                 normalized cost was {normalized_ratio:.3}x, but must be <= \
                 {MAX_STEP_NORMALIZED_COST:.2}x",
                smaller.admitted_rows, larger.admitted_rows,
            ));
        }
    }

    let smallest = &measurements[0];
    let largest = measurements.last().expect("three scaling measurements");
    let end_to_end_normalized =
        largest.normalized_nanoseconds_per_row() / smallest.normalized_nanoseconds_per_row();
    if end_to_end_normalized > MAX_END_TO_END_NORMALIZED_COST {
        violations.push(format!(
            "{} -> {} rows had {end_to_end_normalized:.3}x normalized cost; the 16x workload \
             range must reduce median time per row to <= \
             {MAX_END_TO_END_NORMALIZED_COST:.2}x",
            smallest.admitted_rows, largest.admitted_rows,
        ));
    }

    assert!(
        violations.is_empty(),
        "native Metal quantifier parallel-scaling contract failed:\n{}",
        violations.join("\n"),
    );
    Ok(())
}
