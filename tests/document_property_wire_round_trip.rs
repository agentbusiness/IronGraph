// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! List and map properties must survive every encoding a write travels through.
//!
//! A write reaches durable state as a self-describing mutation payload and a length-prefixed log
//! record. A self-describing encoding keeps byte strings and element sequences distinct, so a
//! document type that writes one shape and reads the other encodes successfully and then fails to
//! decode — which rejected every list and map property, and with them every stored vector, at the
//! point of commit rather than at parse time. These round trips hold that contract in both
//! encodings and in JSON, and pin the length-prefixed bytes so records already on disk still read.

use std::{collections::BTreeMap, sync::Arc};

use irongraph::{DocumentItem, DocumentList, DocumentMap, Result, ScalarValue};

fn vector_document() -> Result<DocumentList> {
    DocumentList::new(vec![
        DocumentItem::Scalar(ScalarValue::Float(0.25.into())),
        DocumentItem::Scalar(ScalarValue::Float((-0.5).into())),
        DocumentItem::Scalar(ScalarValue::Float(1.0.into())),
    ])
}

fn tag_document() -> Result<DocumentList> {
    DocumentList::new(vec![
        DocumentItem::Scalar(ScalarValue::String(Arc::from("graph"))),
        DocumentItem::Scalar(ScalarValue::String(Arc::from("temporal"))),
    ])
}

fn map_document() -> Result<DocumentMap> {
    DocumentMap::new(BTreeMap::from([
        (
            Arc::from("iata"),
            DocumentItem::Scalar(ScalarValue::String(Arc::from("LHR"))),
        ),
        (
            Arc::from("runways"),
            DocumentItem::Scalar(ScalarValue::Integer(2)),
        ),
    ]))
}

#[test]
fn cbor_round_trips_list_and_map_documents() -> Result<()> {
    for document in [vector_document()?, tag_document()?] {
        let mut encoded = Vec::new();
        ciborium::ser::into_writer(&document, &mut encoded).expect("CBOR encodes a list document");
        let decoded: DocumentList =
            ciborium::de::from_reader(encoded.as_slice()).expect("CBOR decodes a list document");
        assert_eq!(decoded, document);
    }

    let document = map_document()?;
    let mut encoded = Vec::new();
    ciborium::ser::into_writer(&document, &mut encoded).expect("CBOR encodes a map document");
    let decoded: DocumentMap =
        ciborium::de::from_reader(encoded.as_slice()).expect("CBOR decodes a map document");
    assert_eq!(decoded, document);
    Ok(())
}

#[test]
fn postcard_round_trips_list_and_map_documents() -> Result<()> {
    let document = vector_document()?;
    let encoded = postcard::to_stdvec(&document).expect("postcard encodes a list document");
    let decoded: DocumentList =
        postcard::from_bytes(&encoded).expect("postcard decodes a list document");
    assert_eq!(decoded, document);

    let document = map_document()?;
    let encoded = postcard::to_stdvec(&document).expect("postcard encodes a map document");
    let decoded: DocumentMap =
        postcard::from_bytes(&encoded).expect("postcard decodes a map document");
    assert_eq!(decoded, document);
    Ok(())
}

#[test]
fn json_round_trips_list_and_map_documents() -> Result<()> {
    let document = vector_document()?;
    let encoded = serde_json::to_vec(&document).expect("JSON encodes a list document");
    let decoded: DocumentList =
        serde_json::from_slice(&encoded).expect("JSON decodes a list document");
    assert_eq!(decoded, document);

    let document = map_document()?;
    let encoded = serde_json::to_vec(&document).expect("JSON encodes a map document");
    let decoded: DocumentMap =
        serde_json::from_slice(&encoded).expect("JSON decodes a map document");
    assert_eq!(decoded, document);
    Ok(())
}

#[test]
fn decoding_rejects_bytes_that_are_not_a_canonical_document() {
    let mut encoded = Vec::new();
    ciborium::ser::into_writer(&serde_bytes_stub(), &mut encoded).expect("CBOR encodes raw bytes");
    let decoded: std::result::Result<DocumentList, _> =
        ciborium::de::from_reader(encoded.as_slice());
    assert!(
        decoded.is_err(),
        "non-canonical bytes must not decode as a list document"
    );
}

/// One CBOR byte string that is deliberately not a canonical document encoding.
fn serde_bytes_stub() -> ciborium::Value {
    ciborium::Value::Bytes(vec![0xff, 0x00, 0x01])
}

#[test]
fn cbor_round_trips_documents_nested_in_a_scalar_value() -> Result<()> {
    // Snapshots and mutation payloads never carry a bare document: the document is the content of
    // a `ScalarValue` enum variant. CBOR encodes that as a single-entry map, and the value inside
    // is read through the variant deserializer rather than the top-level one, so a document type
    // has to decode correctly in both positions.
    for value in [
        ScalarValue::List(vector_document()?),
        ScalarValue::List(tag_document()?),
        ScalarValue::Map(map_document()?),
    ] {
        let mut encoded = Vec::new();
        ciborium::ser::into_writer(&value, &mut encoded).expect("CBOR encodes a scalar value");
        let decoded: ScalarValue = ciborium::de::from_reader(encoded.as_slice())
            .expect("CBOR decodes a document nested in a scalar value");
        assert_eq!(decoded, value);
    }
    Ok(())
}

#[test]
fn cbor_round_trips_a_document_larger_than_the_decoder_scratch_buffer() -> Result<()> {
    // An embedding vector is far larger than the fixed scratch space a borrowed byte read is
    // served from. A document that only decodes while it fits in that buffer passes every small
    // fixture and then makes the first snapshot holding a real vector unreadable, so the size that
    // matters here is one comfortably past it.
    let vector = DocumentList::new(
        (0..4096)
            .map(|index| DocumentItem::Scalar(ScalarValue::Float(f64::from(index).into())))
            .collect(),
    )?;
    assert!(
        vector.as_bytes().len() > 8192,
        "fixture must exceed the decoder scratch buffer"
    );

    let mut encoded = Vec::new();
    ciborium::ser::into_writer(&vector, &mut encoded).expect("CBOR encodes a large document");
    let decoded: DocumentList =
        ciborium::de::from_reader(encoded.as_slice()).expect("CBOR decodes a large document");
    assert_eq!(decoded, vector);

    let mut encoded = Vec::new();
    ciborium::ser::into_writer(&ScalarValue::List(vector.clone()), &mut encoded)
        .expect("CBOR encodes a large document in a scalar value");
    let decoded: ScalarValue = ciborium::de::from_reader(encoded.as_slice())
        .expect("CBOR decodes a large document in a scalar value");
    assert_eq!(decoded, ScalarValue::List(vector));
    Ok(())
}

#[test]
fn postcard_encoding_is_unchanged_by_the_document_wire_shape() -> Result<()> {
    // Snapshots and log records already on disk were written before the wire shape was corrected.
    // The length-prefixed encoding renders a byte string and a byte sequence identically, so this
    // pins that equivalence: a length varint followed by the canonical bytes, exactly as before.
    let document = vector_document()?;
    let encoded = postcard::to_stdvec(&document).expect("postcard encodes a list document");
    let mut expected = postcard::to_stdvec(&(document.as_bytes().len() as u64))
        .expect("postcard encodes the length varint");
    expected.extend_from_slice(document.as_bytes());
    assert_eq!(encoded, expected);
    Ok(())
}
