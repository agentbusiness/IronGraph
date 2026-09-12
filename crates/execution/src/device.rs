//! Fail-closed compute-device resolution shared by graph execution and text embedding.

use serde::{Deserialize, Serialize};

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Mutex,
};

#[cfg(feature = "accelerator")]
use crate::{Error, ErrorCode, Result};

use super::BackendKind;

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
fn selected_metal_registry_id(enumerated: &[u64], ordinal: u32) -> Option<u64> {
    enumerated.get(ordinal as usize).copied()
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
fn validate_metal_registry_identity(
    ordinal: u32,
    candidate_id: u64,
    expected_id: u64,
) -> std::result::Result<(), String> {
    if candidate_id != expected_id {
        return Err(format!(
            "Metal device {ordinal} resolved registry {candidate_id}, expected {expected_id}"
        ));
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
fn construct_metal_device_once(ordinal: u32) -> std::result::Result<candle_core::Device, String> {
    let enumerated_ids = catch_unwind(AssertUnwindSafe(candle_metal_kernels::metal::Device::all))
        .map_err(|_| "Metal device enumeration panicked".to_owned())?
        .iter()
        .map(candle_metal_kernels::metal::Device::registry_id)
        .collect::<Vec<_>>();
    let expected_id = selected_metal_registry_id(&enumerated_ids, ordinal).ok_or_else(|| {
        format!(
            "Metal device ordinal {ordinal} is unavailable; detected {} device(s)",
            enumerated_ids.len()
        )
    })?;
    let candidate = catch_unwind(AssertUnwindSafe(|| {
        candle_core::Device::new_metal(ordinal as usize)
    }))
    .map_err(|_| format!("Metal device {ordinal} construction panicked"))?
    .map_err(|error| error.to_string())?;
    let candidate_id = candidate
        .as_metal_device()
        .map_err(|error| error.to_string())?
        .metal_device()
        .registry_id();
    validate_metal_registry_identity(ordinal, candidate_id, expected_id)?;
    Ok(candidate)
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
pub fn construct_metal_device(ordinal: u32) -> std::result::Result<candle_core::Device, String> {
    static DEVICE_CONSTRUCTION: Mutex<()> = Mutex::new(());
    let _construction = match DEVICE_CONSTRUCTION.lock() {
        Ok(construction) => construction,
        Err(poisoned) => poisoned.into_inner(),
    };
    construct_metal_device_once(ordinal)
}

#[cfg(all(feature = "accelerator", feature = "cuda"))]
fn construct_cuda_device(ordinal: u32) -> std::result::Result<candle_core::Device, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        candle_core::Device::new_cuda(ordinal as usize)
    }))
    .map_err(|_| format!("CUDA device {ordinal} construction panicked"))?
    .map_err(|error| error.to_string())
}

/// Requested compute device. `Auto` selects a real accelerator and never silently degrades.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputeDeviceRequest {
    #[default]
    Auto,
    Cpu,
    Metal(u32),
    Cuda(u32),
}

/// Device identity after construction. Cluster admission must advertise this value, not a flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResolvedComputeDevice {
    pub backend: BackendKind,
    pub ordinal: u32,
}

impl ResolvedComputeDevice {
    #[must_use]
    pub const fn execution_class(self) -> BackendKind {
        self.backend
    }
}

/// A verified Candle device and the exact execution class it represents.
#[cfg(feature = "accelerator")]
pub struct ResolvedCandleDevice {
    pub device: candle_core::Device,
    pub dtype: candle_core::DType,
    pub identity: ResolvedComputeDevice,
}

/// Constructs the requested device and reports what was actually constructed.
///
/// CPU fallback is deliberately explicit. `Auto` failing to find an accelerator is an admission
/// error so a GPU-backed member cannot accidentally announce itself as equivalent to a CPU node.
#[cfg(feature = "accelerator")]
pub fn resolve_candle_device(
    requested: ComputeDeviceRequest,
    error_code: ErrorCode,
) -> Result<ResolvedCandleDevice> {
    use candle_core::{DType, Device};

    let resolved = match requested {
        ComputeDeviceRequest::Cpu => ResolvedCandleDevice {
            device: Device::Cpu,
            dtype: DType::F32,
            identity: ResolvedComputeDevice {
                backend: BackendKind::Cpu,
                ordinal: 0,
            },
        },
        ComputeDeviceRequest::Metal(ordinal) => {
            #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
            {
                let device = construct_metal_device(ordinal).map_err(|error| {
                    Error::new(
                        error_code,
                        format!("Metal device {ordinal} is unavailable: {error}"),
                    )
                })?;
                ResolvedCandleDevice {
                    device,
                    dtype: DType::F16,
                    identity: ResolvedComputeDevice {
                        backend: BackendKind::Metal,
                        ordinal,
                    },
                }
            }
            #[cfg(not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))))]
            {
                return Err(Error::new(
                    error_code,
                    format!("Metal device {ordinal} was requested but Metal is not compiled in"),
                ));
            }
        }
        ComputeDeviceRequest::Cuda(ordinal) => {
            #[cfg(feature = "cuda")]
            {
                let device = construct_cuda_device(ordinal).map_err(|error| {
                    Error::new(
                        error_code,
                        format!("CUDA device {ordinal} is unavailable: {error}"),
                    )
                })?;
                ResolvedCandleDevice {
                    device,
                    dtype: DType::F16,
                    identity: ResolvedComputeDevice {
                        backend: BackendKind::Cuda,
                        ordinal,
                    },
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::new(
                    error_code,
                    format!("CUDA device {ordinal} was requested but CUDA is not compiled in"),
                ));
            }
        }
        ComputeDeviceRequest::Auto => {
            #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
            if let Ok(device) = construct_metal_device(0) {
                return Ok(ResolvedCandleDevice {
                    device,
                    dtype: DType::F16,
                    identity: ResolvedComputeDevice {
                        backend: BackendKind::Metal,
                        ordinal: 0,
                    },
                });
            }
            #[cfg(feature = "cuda")]
            if let Ok(device) = construct_cuda_device(0) {
                return Ok(ResolvedCandleDevice {
                    device,
                    dtype: DType::F16,
                    identity: ResolvedComputeDevice {
                        backend: BackendKind::Cuda,
                        ordinal: 0,
                    },
                });
            }
            return Err(Error::new(
                error_code,
                "automatic device selection found no compiled, usable accelerator; select CPU explicitly for degraded operation",
            ));
        }
    };
    Ok(resolved)
}

#[cfg(all(
    test,
    feature = "accelerator",
    any(target_os = "macos", target_os = "ios")
))]
mod tests {
    use super::*;

    #[test]
    fn metal_registry_selection_requires_the_requested_enumerated_ordinal() {
        assert_eq!(selected_metal_registry_id(&[], 0), None);
        assert_eq!(selected_metal_registry_id(&[7, 9], 0), Some(7));
        assert_eq!(selected_metal_registry_id(&[7, 9], 1), Some(9));
        assert_eq!(selected_metal_registry_id(&[7, 9], 2), None);
    }

    #[test]
    fn constructed_device_must_match_the_preflight_identity() {
        assert!(validate_metal_registry_identity(0, 41, 41).is_ok());
        assert!(validate_metal_registry_identity(0, 41, 42).is_err());
    }

    #[test]
    fn invalid_metal_ordinal_fails_without_unwinding() {
        let resolved = std::panic::catch_unwind(|| {
            resolve_candle_device(
                ComputeDeviceRequest::Metal(u32::MAX),
                ErrorCode::EmbeddingUnavailable,
            )
        });
        assert!(resolved.is_ok());
        assert!(resolved.expect("resolution must not unwind").is_err());
    }
}
