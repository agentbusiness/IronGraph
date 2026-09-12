//! Stable error taxonomy shared by every IronGraph crate and public protocol.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Machine-stable error code. Protocol adapters map it without parsing messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    StaleLocalRead,
    CorruptStorage,
    GpuAdmissionFailure,
    ProjectNotFound,
    ProjectFenced,
    LayerNotAllowed,
    QuerySyntax,
    QueryType,
    TransactionConflict,
    TransactionExpired,
    TransactionSequencerChanged,
    WriteAdmissionFull,
    TemporalRange,
    EmbeddingProfileMismatch,
    EmbeddingProfileImmutable,
    IndexUnavailable,
    ResultBudgetExceeded,
    EmbeddingUnavailable,
    AuthenticationFailed,
    AuthorizationDenied,
    Backpressure,
    RetentionExpired,
    ProtocolViolation,
    Cancelled,
    DeadlineExceeded,
    InvalidData,
    Io,
    Internal,
}

/// Product error with stable code, safe message, and retry classification.
#[derive(Debug, Error, Serialize, Deserialize)]
#[error("{code:?}: {message}")]
pub struct Error {
    pub code: ErrorCode,
    pub message: Cow<'static, str>,
    pub retryable: bool,
    pub retry_after_ms: Option<u64>,
}

impl Error {
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: false,
            retry_after_ms: None,
        }
    }

    #[must_use]
    pub fn retryable(
        code: ErrorCode,
        message: impl Into<Cow<'static, str>>,
        retry_after_ms: Option<u64>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: true,
            retry_after_ms,
        }
    }

    #[must_use]
    pub fn invalid_data(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorCode::InvalidData, message)
    }

    #[must_use]
    pub fn internal(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::new(ErrorCode::Io, value.to_string())
    }
}

/// Crate-wide result type.
pub type Result<T, E = Error> = std::result::Result<T, E>;
