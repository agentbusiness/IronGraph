// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Bit-exact CPU/Metal agreement for binary64 arithmetic evaluated on the device.
//!
//! Apple Metal has no native `double`, so every floating-point Cypher operator that runs natively
//! is implemented as a software binary64 lane over 64-bit integers. That lane is invisible to the
//! openCypher corpus: its modulo and power feature files contain no scenarios at all, and its one
//! `sqrt()` scenario uses a value that even an incorrect implementation reproduces. A full
//! conformance run therefore cannot distinguish a correctly rounded lane from one that is a unit in
//! the last place off for a quarter of its inputs — which is exactly the defect this suite exists to
//! catch, after a Newton-Raphson square root shipped with no final rounding correction.
//!
//! Equality here is bit-for-bit on purpose. A tolerance would accept precisely the drift that makes
//! the CPU reference and the GPU backend disagree about a query's result.

use irongraph::{
    Result,
    gpu::{
        CpuBackend, ExecutionBackend, ResidentScalarCell, ResidentScalarCellTag,
        ResidentScalarProgramInstruction, ResidentScalarProgramOpcode,
        ResidentScalarProgramOperand, ResidentScalarProgramRequest,
    },
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const NULL_CELL: u16 = 0;

/// Builds `left <op> right` over two immediate FLOAT cells and returns the single output value.
fn binary_request(
    opcode: ResidentScalarProgramOpcode,
    left: f64,
    right: f64,
) -> ResidentScalarProgramRequest {
    let float = |value: f64| ResidentScalarCell {
        tag: ResidentScalarCellTag::Float,
        payload: value.to_bits(),
        auxiliary: 0,
    };
    let null = ResidentScalarCell {
        tag: ResidentScalarCellTag::Null,
        payload: 0,
        auxiliary: 0,
    };
    ResidentScalarProgramRequest {
        // Cell 0 is the shared NULL, 1 and 2 are the operands, 3 receives the instruction result.
        scalar_cells: vec![null, float(left), float(right), null],
        map_entries: Vec::new(),
        scalar_list_entries: Vec::new(),
        string_offsets: vec![0, 0],
        string_bytes: Vec::new(),
        null_cell: NULL_CELL,
        false_cell: None,
        true_cell: None,
        instructions: vec![ResidentScalarProgramInstruction {
            opcode,
            left: ResidentScalarProgramOperand::Cell(1),
            right: ResidentScalarProgramOperand::Cell(2),
            third: None,
            fourth: None,
        }],
        output_values: vec![ResidentScalarProgramOperand::Register(0)],
    }
}

/// Runs the program and returns the raw bits of the FLOAT it produced, or `None` if it refused.
///
/// A refusal is a legitimate answer — the backend declining to execute natively is visible to the
/// caller and falls back to the host reference — so it is compared as its own outcome rather than
/// treated as a failure.
fn evaluate(backend: &dyn ExecutionBackend, request: &ResidentScalarProgramRequest) -> Option<u64> {
    let result = backend
        .execute_scalar_program(request, &CancellationToken::new())
        .ok()?;
    let index = *result.output_cells.first()? as usize;
    let cell = result.scalar_cells.get(index)?;
    (cell.tag == ResidentScalarCellTag::Float).then_some(cell.payload)
}

/// Inputs chosen to exercise rounding boundaries rather than typical magnitudes.
fn operand_matrix() -> Vec<f64> {
    let mut values = vec![
        0.0,
        1.0,
        2.0,
        3.0,
        0.5,
        12.96,
        1e17,
        1e-17,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::EPSILON,
        2.5,
        4.0,
        9.0,
        1.0e300,
        1.0e-300,
    ];
    // A deterministic spread across the exponent range, including subnormals and near-ties.
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    for _ in 0..512 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let candidate = f64::from_bits(state >> 1);
        if candidate.is_finite() {
            values.push(candidate);
        }
        values.push(f64::from(u32::try_from(state % 4_096).unwrap_or(1)));
    }
    values
}

/// Asserts the invariant that matters: whenever the candidate backend *answers*, it answers with
/// exactly the reference's bits.
///
/// Declining to execute is a legitimate outcome rather than a divergence — an admission failure is
/// visible to the planner, which then evaluates the expression on the host reference, so the query
/// still returns the reference's answer. Silently returning a different number is the failure this
/// guards against, so refusals are counted and reported but not treated as disagreement. The
/// comparison count is asserted so a backend that refused everything could not pass vacuously.
fn assert_backends_agree(reference: &dyn ExecutionBackend, candidate: &dyn ExecutionBackend) {
    let operands = operand_matrix();
    let mut compared = 0_usize;
    let mut refused = 0_usize;
    for opcode in [
        ResidentScalarProgramOpcode::NumericAdd,
        ResidentScalarProgramOpcode::NumericSubtract,
        ResidentScalarProgramOpcode::NumericMultiply,
        ResidentScalarProgramOpcode::NumericDivide,
        ResidentScalarProgramOpcode::NumericModulo,
        ResidentScalarProgramOpcode::NumericPower,
    ] {
        for (index, &left) in operands.iter().enumerate() {
            // Pair each operand with a rotating partner instead of forming the full cross product,
            // which keeps the suite fast while still covering every operand in both positions.
            let right = operands[(index * 7 + 3) % operands.len()];
            let request = binary_request(opcode, left, right);
            let Some(actual) = evaluate(candidate, &request) else {
                refused += 1;
                continue;
            };
            let expected = evaluate(reference, &request).unwrap_or_else(|| {
                panic!(
                    "{opcode:?} was answered for {left:e} and {right:e} but the reference refused"
                )
            });
            compared += 1;
            assert_eq!(
                f64::from_bits(actual).to_bits(),
                f64::from_bits(expected).to_bits(),
                "{opcode:?} diverged for {left:e} and {right:e}: reference {} candidate {}",
                f64::from_bits(expected),
                f64::from_bits(actual),
            );
        }
    }
    assert!(
        compared > 1_000,
        "differential matrix collapsed to nothing: {compared} compared, {refused} refused"
    );
}

#[test]
fn the_cpu_reference_is_stable_across_repeated_evaluation() {
    // Guards the harness itself: if the reference were nondeterministic, agreement would be
    // meaningless. This also keeps the suite meaningful on hosts without a Metal device.
    let cpu = CpuBackend::new(usize::MAX / 4, 0);
    assert_backends_agree(&cpu, &CpuBackend::new(usize::MAX / 4, 0));
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_binary64_arithmetic_matches_the_cpu_reference_bit_for_bit() -> Result<()> {
    let cpu = CpuBackend::new(usize::MAX / 4, 0);
    let metal = MetalBackend::new(0, 128 * 1024 * 1024, 16 * 1024 * 1024)?;
    assert_backends_agree(&cpu, &metal);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_integer_exponent_power_matches_the_reference_or_declines() -> Result<()> {
    // Integer exponents are the case the corpus exercises and the case repeated squaring got
    // wrong: multiplying step by step rounds at every step, so the result drifted from `powf` for
    // most bases even though the handful of small values in the corpus stayed exact. The general
    // operand matrix almost never produces an exact integer exponent, so this walks them directly.
    let cpu = CpuBackend::new(usize::MAX / 4, 0);
    let metal = MetalBackend::new(0, 128 * 1024 * 1024, 16 * 1024 * 1024)?;
    let mut answered = 0_usize;
    for base in [
        0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, 7.0, 10.0, 1024.0, 4096.0, 2.5, -0.5, 1.5,
        123.0, -47.0, 99999.0,
    ] {
        for exponent in [
            0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 53, 54, 64, -1, -2, -3,
        ] {
            let request = binary_request(
                ResidentScalarProgramOpcode::NumericPower,
                base,
                f64::from(exponent),
            );
            let Some(actual) = evaluate(&metal, &request) else {
                continue;
            };
            answered += 1;
            let expected = evaluate(&cpu, &request)
                .unwrap_or_else(|| panic!("reference refused {base} ^ {exponent}"));
            assert_eq!(
                f64::from_bits(actual),
                f64::from_bits(expected),
                "{base} ^ {exponent} diverged from the reference",
            );
        }
    }
    // The values the conformance corpus depends on must still be answered natively, not declined.
    for (base, exponent, expected) in [
        (4.0_f64, 3.0_f64, 64.0_f64),
        (2.0, 3.0, 8.0),
        (-3.0, 2.0, 9.0),
        (4096.0, 3.0, 68_719_476_736.0),
        (1024.0, 3.0, 1_073_741_824.0),
    ] {
        let request = binary_request(ResidentScalarProgramOpcode::NumericPower, base, exponent);
        assert_eq!(
            evaluate(&metal, &request).map(f64::from_bits),
            Some(expected),
            "{base} ^ {exponent} must stay natively answerable and exact",
        );
    }
    assert!(
        answered > 50,
        "integer-exponent matrix collapsed to nothing"
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_float_modulo_is_the_exact_ieee_remainder() -> Result<()> {
    // The former implementation computed `left - trunc(left / right) * right` through three
    // separately rounded software operations. These are the two shapes that exposed it: a dividend
    // far larger than the divisor, where the rounded quotient absorbed the remainder entirely, and
    // an exponent spread wide enough that the intermediate quotient could not be held in a signed
    // 64-bit integer, which made the backend refuse a query the reference answers.
    let metal = MetalBackend::new(0, 128 * 1024 * 1024, 16 * 1024 * 1024)?;
    for (left, right) in [
        (1.0e17_f64, 3.0_f64),
        (1.0e18, 3.0),
        (12.96, 2.024_034_889_738_657_7e-19),
        (-731_271_511_775.197_5, 694.867),
        (1.0e300, 7.0),
        (5.5, 2.0),
        (-7.5, 2.5),
    ] {
        let request = binary_request(ResidentScalarProgramOpcode::NumericModulo, left, right);
        assert_eq!(
            evaluate(&metal, &request).map(f64::from_bits),
            Some(left % right),
            "{left:e} % {right:e} must be the exact remainder",
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_square_root_is_correctly_rounded() -> Result<()> {
    // `sqrt()` lowers to `x ^ 0.5`, so this is the operator that regressed. Squares must come back
    // exactly, and every other input must match the platform's correctly rounded result.
    let metal = MetalBackend::new(0, 128 * 1024 * 1024, 16 * 1024 * 1024)?;
    let mut checked = 0_usize;
    for value in operand_matrix() {
        if !value.is_finite() || value < 0.0 {
            continue;
        }
        let request = binary_request(ResidentScalarProgramOpcode::NumericPower, value, 0.5);
        let Some(actual) = evaluate(&metal, &request) else {
            continue;
        };
        checked += 1;
        assert_eq!(
            actual,
            value.sqrt().to_bits(),
            "sqrt({value:e}) was not correctly rounded: got {}, want {}",
            f64::from_bits(actual),
            value.sqrt(),
        );
    }
    assert_eq!(
        evaluate(
            &metal,
            &binary_request(ResidentScalarProgramOpcode::NumericPower, 12.96, 0.5)
        ),
        Some(3.6_f64.to_bits()),
        "the conformance corpus value must remain exact"
    );
    assert!(checked > 100, "square-root matrix collapsed to nothing");
    Ok(())
}
