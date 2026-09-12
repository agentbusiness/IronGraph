//! Hardware execution backends for the backend-neutral resident execution contract.

#![allow(
    clippy::collapsible_if,
    clippy::err_expect,
    clippy::filter_map_bool_then,
    clippy::if_same_then_else,
    clippy::iter_skip_next,
    clippy::manual_contains,
    clippy::manual_ignore_case_cmp,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::manual_saturating_arithmetic,
    clippy::manual_slice_size_calculation,
    clippy::match_like_matches_macro,
    clippy::needless_borrow,
    clippy::needless_lifetimes,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::nonminimal_bool,
    clippy::obfuscated_if_else,
    clippy::only_used_in_recursion,
    clippy::question_mark,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_filter_map,
    clippy::unnecessary_lazy_evaluations,
    clippy::useless_conversion,
    clippy::useless_vec
)]

pub use irongraph_execution as execution;
pub use irongraph_execution::*;
pub use irongraph_graph as graph;
pub use irongraph_types as types;
pub use irongraph_types::document;
pub use irongraph_types::{
    Bookmark, DocumentItem, DocumentList, DocumentMap, EdgeId, Error, ErrorCode, Layer, NodeId,
    ProjectId, Result, ScalarValue,
};

#[cfg(any(
    all(feature = "accelerator", any(target_os = "macos", target_os = "ios")),
    feature = "cuda"
))]
mod accelerator;
#[cfg(feature = "accelerator")]
pub mod device;
#[cfg(feature = "accelerator")]
pub use device::{ResolvedCandleDevice, resolve_candle_device};
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
mod metal;
#[cfg(feature = "accelerator")]
pub mod metal_gate;
#[cfg(feature = "accelerator")]
pub use metal_gate::{lock_metal_device, metal_device_gate};

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub use accelerator::{
    resident_named_zone_table_bytes, stable_device_argsort, stable_device_top_k,
};
#[cfg(all(feature = "cuda", not(any(target_os = "macos", target_os = "ios"))))]
pub use accelerator::{stable_device_argsort, stable_device_top_k};
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub use metal::MetalBackend;

/// Serialises tests that allocate on the process-wide Metal device.
#[cfg(test)]
pub fn metal_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static METAL_DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    match METAL_DEVICE.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Returns the shared Metal device when this test host exposes one. CI and sandboxed macOS
/// processes may have Metal compiled in while device enumeration is unavailable; hardware tests
/// skip in that case instead of panicking inside Candle's constructor.
#[cfg(all(
    test,
    feature = "accelerator",
    any(target_os = "macos", target_os = "ios")
))]
pub fn metal_test_device() -> Option<candle_core::Device> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        candle_core::Device::new_metal(0)
    }))
    .ok()
    .and_then(std::result::Result::ok)
}

/// Constructs exactly the resolved backend or fails; it never changes execution class.
pub fn create_execution_backend(
    resolved: ResolvedComputeDevice,
    memory_limit_bytes: usize,
    reserved_bytes: usize,
) -> Result<Box<dyn ExecutionBackend>> {
    create_execution_backend_with_governor(
        resolved,
        DeviceMemoryGovernor::new(memory_limit_bytes, reserved_bytes),
    )
}

/// Constructs a backend participating in the caller's process-wide graph/encoder byte ledger.
pub fn create_execution_backend_with_governor(
    resolved: ResolvedComputeDevice,
    governor: DeviceMemoryGovernor,
) -> Result<Box<dyn ExecutionBackend>> {
    match resolved.backend {
        BackendKind::Cpu => Ok(Box::new(CpuBackend::with_governor(governor))),
        BackendKind::Metal => {
            #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
            return Ok(Box::new(MetalBackend::with_governor(
                resolved.ordinal as usize,
                governor,
            )?));
            #[cfg(not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))))]
            Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "Metal support is not enabled",
            ))
        }
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            return Ok(Box::new(CudaBackend::with_governor(
                resolved.ordinal as usize,
                governor,
            )?));
            #[cfg(not(feature = "cuda"))]
            Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "CUDA support is not enabled",
            ))
        }
    }
}
