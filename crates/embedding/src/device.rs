use crate::gpu::ComputeDeviceRequest;

/// Device selected for automatic local text embedding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingDevice {
    #[default]
    Auto,
    Cpu,
    Metal(u32),
    Cuda(u32),
}

impl From<EmbeddingDevice> for ComputeDeviceRequest {
    fn from(value: EmbeddingDevice) -> Self {
        match value {
            EmbeddingDevice::Auto => Self::Auto,
            EmbeddingDevice::Cpu => Self::Cpu,
            EmbeddingDevice::Metal(ordinal) => Self::Metal(ordinal),
            EmbeddingDevice::Cuda(ordinal) => Self::Cuda(ordinal),
        }
    }
}

#[cfg(feature = "accelerator")]
pub(super) fn select_device(
    requested: EmbeddingDevice,
) -> crate::Result<crate::gpu::ResolvedCandleDevice> {
    select_with(requested, |device| {
        crate::gpu::resolve_candle_device(device.into(), crate::ErrorCode::EmbeddingUnavailable)
    })
}

#[cfg(feature = "accelerator")]
fn select_with<T>(
    requested: EmbeddingDevice,
    mut resolve: impl FnMut(EmbeddingDevice) -> crate::Result<T>,
) -> crate::Result<T> {
    match resolve(requested) {
        Ok(selected) => Ok(selected),
        Err(error) if requested == EmbeddingDevice::Auto => {
            tracing::warn!(%error, "no usable GPU for automatic embedding; using CPU");
            resolve(EmbeddingDevice::Cpu)
        }
        Err(error) => Err(error),
    }
}

#[cfg(all(test, feature = "accelerator"))]
mod tests {
    use super::*;

    #[test]
    fn automatic_selection_tries_gpu_before_cpu() {
        let mut calls = Vec::new();
        let selected = select_with(EmbeddingDevice::Auto, |device| {
            calls.push(device);
            if device == EmbeddingDevice::Auto {
                Err(crate::Error::new(
                    crate::ErrorCode::EmbeddingUnavailable,
                    "GPU unavailable",
                ))
            } else {
                Ok(device)
            }
        })
        .unwrap();
        assert_eq!(calls, [EmbeddingDevice::Auto, EmbeddingDevice::Cpu]);
        assert_eq!(selected, EmbeddingDevice::Cpu);
        let mut calls = Vec::new();
        assert_eq!(
            select_with(EmbeddingDevice::Auto, |device| {
                calls.push(device);
                Ok(EmbeddingDevice::Metal(0))
            })
            .unwrap(),
            EmbeddingDevice::Metal(0)
        );
        assert_eq!(calls, [EmbeddingDevice::Auto]);
    }

    #[test]
    fn explicit_gpu_selection_does_not_silently_change_devices() {
        let mut calls = 0;
        let result: crate::Result<()> = select_with(EmbeddingDevice::Metal(u32::MAX), |_| {
            calls += 1;
            Err(crate::Error::new(
                crate::ErrorCode::EmbeddingUnavailable,
                "GPU unavailable",
            ))
        });
        assert!(result.is_err());
        assert_eq!(calls, 1);
    }
}
