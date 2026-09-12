use std::{
    convert::Infallible,
    pin::Pin,
    task::{Context, Poll},
};

use axum::body::Body;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

/// Bounded sender used by execution workers to preserve response backpressure.
#[derive(Clone)]
pub struct NdjsonSender<T> {
    sender: mpsc::Sender<T>,
}

impl<T> NdjsonSender<T> {
    /// Completes when the response body/receiver has been dropped by the client.
    pub async fn closed(&self) {
        self.sender.closed().await;
    }

    pub async fn send(&self, value: T) -> Result<()> {
        self.sender
            .send(value)
            .await
            .map_err(|_| Error::new(crate::ErrorCode::Cancelled, "response stream was closed"))
    }

    pub fn blocking_send(&self, value: T) -> Result<()> {
        self.sender
            .blocking_send(value)
            .map_err(|_| Error::new(crate::ErrorCode::Cancelled, "response stream was closed"))
    }
}

/// NDJSON response stream whose items cannot fail after headers are committed.
pub struct NdjsonBody {
    inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, Infallible>> + Send>>,
    cancellation: Option<CancellationToken>,
}

impl NdjsonBody {
    /// Cancels the producer when the HTTP body is dropped by a disconnected client. Keeping the
    /// cancellation hook on the body avoids retaining a sender clone that would prevent a
    /// naturally completed channel from ever reaching EOF.
    #[must_use]
    pub fn cancel_on_drop(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    #[must_use]
    pub fn into_body(self) -> Body {
        Body::from_stream(self)
    }
}

impl Drop for NdjsonBody {
    fn drop(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            cancellation.cancel();
        }
    }
}

impl Stream for NdjsonBody {
    type Item = std::result::Result<Bytes, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(context)
    }
}

#[must_use]
pub fn ndjson_channel<T: serde::Serialize + Send + 'static>(
    capacity: usize,
) -> (NdjsonSender<T>, NdjsonBody) {
    let (sender, receiver) = mpsc::channel::<T>(capacity.max(1));
    let stream = ReceiverStream::new(receiver).filter_map(|value| async move {
        match serde_json::to_vec(&value) {
            Ok(mut bytes) => {
                bytes.push(b'\n');
                Some(Ok(Bytes::from(bytes)))
            }
            Err(_) => None,
        }
    });
    (
        NdjsonSender { sender },
        NdjsonBody {
            inner: Box::pin(stream),
            cancellation: None,
        },
    )
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    #[test]
    fn dropping_a_cancelable_body_notifies_its_producer() {
        let cancellation = CancellationToken::new();
        let (_sender, body) = super::ndjson_channel::<u8>(1);
        let body = body.cancel_on_drop(cancellation.clone());

        assert!(!cancellation.is_cancelled());
        drop(body);
        assert!(cancellation.is_cancelled());
    }
}
