use std::{
    convert::Infallible,
    marker::PhantomData,
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
    sender: mpsc::Sender<Bytes>,
    value: PhantomData<fn(T)>,
}

impl<T: serde::Serialize + Send + 'static> NdjsonSender<T> {
    /// Completes when the response body/receiver has been dropped by the client.
    pub async fn closed(&self) {
        self.sender.closed().await;
    }

    pub async fn send(&self, value: T) -> Result<()> {
        let permit =
            self.sender.clone().reserve_owned().await.map_err(|_| {
                Error::new(crate::ErrorCode::Cancelled, "response stream was closed")
            })?;
        let bytes = tokio::task::spawn_blocking(move || encode(value))
            .await
            .map_err(|error| {
                Error::internal(format!("result encoding worker failed: {error}"))
            })??;
        permit.send(bytes);
        Ok(())
    }

    pub fn blocking_send(&self, value: T) -> Result<()> {
        if self.sender.is_closed() {
            return Err(Error::new(
                crate::ErrorCode::Cancelled,
                "response stream was closed",
            ));
        }
        self.sender
            .blocking_send(encode(value)?)
            .map_err(|_| Error::new(crate::ErrorCode::Cancelled, "response stream was closed"))
    }
}

fn encode(value: impl serde::Serialize) -> Result<Bytes> {
    let mut bytes = serde_json::to_vec(&value)
        .map_err(|error| Error::internal(format!("result encoding failed: {error}")))?;
    bytes.push(b'\n');
    Ok(Bytes::from(bytes))
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
    let (sender, receiver) = mpsc::channel::<Bytes>(capacity.max(1));
    let stream = ReceiverStream::new(receiver).map(Ok);
    (
        NdjsonSender {
            sender,
            value: PhantomData,
        },
        NdjsonBody {
            inner: Box::pin(stream),
            cancellation: None,
        },
    )
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use serde::{Serialize, Serializer};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio_util::sync::CancellationToken;

    struct Counted {
        calls: Arc<AtomicUsize>,
        runtime_thread: std::thread::ThreadId,
        fail: bool,
    }

    impl Serialize for Counted {
        fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
            assert_ne!(std::thread::current().id(), self.runtime_thread);
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(serde::ser::Error::custom(
                    "controlled serialization failure",
                ));
            }
            serializer.serialize_str("complete result")
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn async_send_encodes_once_off_runtime_and_propagates_errors() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (sender, mut body) = super::ndjson_channel::<Counted>(1);
        let value = |fail| Counted {
            calls: Arc::clone(&calls),
            runtime_thread: std::thread::current().id(),
            fail,
        };
        sender.send(value(false)).await.unwrap();
        let bytes = body.next().await.unwrap().unwrap();
        assert_eq!(&bytes[..], b"\"complete result\"\n");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(sender.send(value(true)).await.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        drop(body);
        assert!(sender.send(value(false)).await.is_err());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "a disconnected stream does no encoding work"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn response_capacity_waits_without_blocking_runtime_or_encoding_queued_values() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (sender, mut body) = super::ndjson_channel::<Counted>(1);
        let runtime_thread = std::thread::current().id();
        sender
            .send(Counted {
                calls: Arc::clone(&calls),
                runtime_thread,
                fail: false,
            })
            .await
            .unwrap();
        let pending = tokio::spawn(async move {
            sender
                .send(Counted {
                    calls,
                    runtime_thread,
                    fail: false,
                })
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        assert!(!pending.is_finished());
        assert!(body.next().await.is_some());
        pending.await.unwrap().unwrap();
        assert!(body.next().await.is_some());
    }

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
