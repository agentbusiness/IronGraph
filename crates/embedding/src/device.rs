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
    crate::gpu::resolve_candle_device(requested.into(), crate::ErrorCode::EmbeddingUnavailable)
}
