//! Explicit CPU retrieval strategies over canonical vector owners.
//!
//! Prefix and binary searches are candidate generators; final scores always use the complete
//! current owner vectors. RaBitQ uses a seeded Gaussian Gram–Schmidt orthogonal rotation and
//! its alignment-corrected dot-product estimator (Gao/Long, SIGMOD 2024, arXiv:2405.12497,
//! sections 3.1–3.2 and appendix B.3). No empirical error bound or speedup is promised here.

use super::*;
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

/// Explicit approximate methods never replace the ordinary exact search contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    Exact,
    /// The caller must establish that the model was trained for this prefix dimension.
    Prefix {
        dimensions: usize,
        candidates: usize,
    },
    Quantized {
        candidates: usize,
    },
    Adaptive {
        candidates: usize,
    },
}

// Legacy non-cancellable entry points reuse a private token per worker thread. Initialization
// cannot wait on another thread, and scoring does not allocate a token for every owner.
thread_local! {
    static UNCANCELLED: CancellationToken = CancellationToken::new();
}

/// Rebuildable binary codes only: no second full vector or graph representation.
#[derive(Default)]
pub(super) struct RetrievalIndex {
    codes: ConcurrentMap<u64, Arc<CodedOwner>>,
    code_bytes: AtomicUsize,
    rotation: std::sync::OnceLock<Rotation>,
    quantization_building: AtomicBool,
    quantization_ready: AtomicBool,
    #[cfg(test)]
    span_scores: AtomicU64,
    #[cfg(test)]
    encoded_spans: AtomicU64,
    #[cfg(test)]
    first_span_gate: std::sync::OnceLock<Arc<std::sync::Barrier>>,
    #[cfg(test)]
    first_code_gate: std::sync::OnceLock<Arc<std::sync::Barrier>>,
}

#[derive(Clone)]
struct BinaryCode {
    signs: Vec<u64>,
    norm: f64,
    alignment: f64,
}

struct CodedOwner {
    stamp: u64,
    codes: Vec<BinaryCode>,
}
impl CodedOwner {
    fn resident_bytes(&self) -> usize {
        size_of::<Self>()
            + self
                .codes
                .iter()
                .map(BinaryCode::resident_bytes)
                .sum::<usize>()
    }
}

fn cancelled(token: &CancellationToken) -> Result<()> {
    if token.is_cancelled() {
        Err(Error::new(
            ErrorCode::Cancelled,
            "vector retrieval cancelled",
        ))
    } else {
        Ok(())
    }
}

fn mix(mut n: u64) -> u64 {
    n = (n ^ (n >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    n = (n ^ (n >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    n ^ (n >> 31)
}

struct Rotation {
    dimension: usize,
    rows: Vec<f64>,
}

impl Rotation {
    fn new(dimension: usize, cancellation: &CancellationToken) -> Result<Self> {
        let cells = dimension.checked_mul(dimension).ok_or_else(|| {
            Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "RaBitQ rotation shape overflows",
            )
        })?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(cells).map_err(|_| {
            Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "cannot allocate RaBitQ rotation",
            )
        })?;
        let mut counter = 0x1782_956b_71e4_833a_u64 ^ dimension as u64;
        let mut uniform = || {
            counter = counter.wrapping_add(0x9e37_79b9_7f4a_7c15);
            // Open interval (0,1), avoiding log(0) and preserving 53 random mantissa bits.
            (((mix(counter) >> 11) as f64) + 0.5) / ((1_u64 << 53) as f64)
        };
        for row in 0..dimension {
            cancelled(cancellation)?;
            let mut gaussian = Vec::with_capacity(dimension);
            while gaussian.len() < dimension {
                let radius = (-2.0 * uniform().ln()).sqrt();
                let angle = std::f64::consts::TAU * uniform();
                gaussian.push(radius * angle.cos());
                if gaussian.len() < dimension {
                    gaussian.push(radius * angle.sin());
                }
            }
            // Reorthogonalization reduces floating-point drift without changing the distribution
            // of the exact Gaussian QR construction described in the paper.
            for _ in 0..2 {
                for previous in rows.chunks_exact(dimension) {
                    cancelled(cancellation)?;
                    let dot: f64 = gaussian.iter().zip(previous).map(|(x, y)| x * y).sum();
                    for (value, basis) in gaussian.iter_mut().zip(previous) {
                        *value -= dot * basis;
                    }
                }
            }
            let norm = gaussian.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !norm.is_finite() || norm <= f64::EPSILON {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "RaBitQ orthogonalization is singular",
                ));
            }
            rows.extend(gaussian.into_iter().map(|value| value / norm));
            debug_assert_eq!(rows.len(), (row + 1) * dimension);
        }
        Ok(Self { dimension, rows })
    }

    fn rotate(&self, vector: &[f32]) -> Vec<f64> {
        self.rows
            .chunks_exact(self.dimension)
            .map(|row| row.iter().zip(vector).map(|(a, b)| a * f64::from(*b)).sum())
            .collect()
    }

    fn resident_bytes(&self) -> usize {
        size_of::<Self>() + self.rows.capacity() * size_of::<f64>()
    }
}

impl BinaryCode {
    fn resident_bytes(&self) -> usize {
        size_of::<Self>() + self.signs.capacity() * size_of::<u64>()
    }
    fn new(vector: &[f32], rotation: &Rotation) -> Self {
        let transformed = rotation.rotate(vector);
        let norm = vector
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        let mut signs = vec![0; transformed.len().div_ceil(64)];
        for (i, value) in transformed.iter().enumerate() {
            if *value >= 0.0 {
                signs[i / 64] |= 1 << (i % 64);
            }
        }
        let alignment = if norm > f64::EPSILON {
            transformed.iter().map(|v| v.abs()).sum::<f64>()
                / (norm * (transformed.len() as f64).sqrt())
        } else {
            1.0
        };
        Self {
            signs,
            norm,
            alignment,
        }
    }

    fn score(&self, transformed_query: &[f64], query_norm: f64, similarity: Similarity) -> f32 {
        let dot = transformed_query
            .iter()
            .enumerate()
            .map(|(i, value)| {
                if self.signs[i / 64] & (1 << (i % 64)) != 0 {
                    *value
                } else {
                    -*value
                }
            })
            .sum::<f64>()
            * self.norm
            / ((transformed_query.len() as f64).sqrt() * self.alignment);
        if similarity == Similarity::Euclidean {
            (self.norm * self.norm + query_norm * query_norm - 2.0 * dot)
                .max(0.0)
                .sqrt() as f32
        } else {
            dot as f32
        }
    }
}

fn better(similarity: Similarity, left: f32, right: f32) -> bool {
    if similarity == Similarity::Euclidean {
        left < right
    } else {
        left > right
    }
}

fn push_hit(hits: &mut Vec<VectorHit>, hit: VectorHit, limit: usize, similarity: Similarity) {
    let at = hits.partition_point(|current| {
        let order = if similarity == Similarity::Euclidean {
            current.score.total_cmp(&hit.score)
        } else {
            hit.score.total_cmp(&current.score)
        };
        order
            .then_with(|| current.entity_id.cmp(&hit.entity_id))
            .is_lt()
    });
    if at < limit {
        hits.insert(at, hit);
        hits.truncate(limit);
    }
}

impl VectorIndex {
    fn coordinate_score(&self, query: &[f32], coordinates: impl Iterator<Item = u16>) -> f32 {
        if self.similarity == Similarity::Euclidean {
            query
                .iter()
                .zip(coordinates)
                .map(|(q, x)| {
                    let delta = q - decode_coordinate(self.dtype, x);
                    delta * delta
                })
                .sum::<f32>()
                .sqrt()
        } else {
            query
                .iter()
                .zip(coordinates)
                .map(|(q, x)| q * decode_coordinate(self.dtype, x))
                .sum()
        }
    }

    pub(super) fn score_payload(
        &self,
        query: &[f32],
        payload: &VectorPayload,
    ) -> Result<Option<f32>> {
        UNCANCELLED.with(|token| self.score_payload_cancellable(query, payload, token))
    }

    fn score_payload_cancellable(
        &self,
        query: &[f32],
        payload: &VectorPayload,
        token: &CancellationToken,
    ) -> Result<Option<f32>> {
        cancelled(token)?;
        if let VectorCoordinates::Spans(spans) = &payload.coordinates {
            let mut best = None;
            for (position, span) in spans.iter().enumerate() {
                #[cfg(test)]
                if position == 0 {
                    if let Some(gate) = self.store.retrieval.first_span_gate.get() {
                        gate.wait();
                        gate.wait();
                    }
                }
                #[cfg(not(test))]
                let _ = position;
                cancelled(token)?;
                let score = self.coordinate_score(query, span.iter().copied());
                #[cfg(test)]
                self.store
                    .retrieval
                    .span_scores
                    .fetch_add(1, AtomicOrdering::Relaxed);
                if best.is_none_or(|old| better(self.similarity, score, old)) {
                    best = Some(score);
                }
            }
            Ok(best)
        } else {
            Ok(self
                .payload_coordinates(payload.entity_id, payload)?
                .map(|coordinates| self.coordinate_score(query, coordinates.iter())))
        }
    }

    pub(super) fn score_owner(&self, query: &[f32], owner: u64) -> Result<Option<f32>> {
        UNCANCELLED.with(|token| self.score_owner_cancellable(query, owner, token))
    }

    fn score_owner_cancellable(
        &self,
        query: &[f32],
        owner: u64,
        token: &CancellationToken,
    ) -> Result<Option<f32>> {
        cancelled(token)?;
        let Some(row) = self.store.rows.get(&owner) else {
            return Ok(None);
        };
        let Some(slot) = self.store.slots.get(row) else {
            return Ok(None);
        };
        let payload = slot.payload.load();
        let Some(payload) = payload.as_ref().filter(|p| p.entity_id == owner) else {
            return Ok(None);
        };
        self.score_payload_cancellable(query, payload, token)
    }

    fn refresh_codes(&self, owner: u64) -> Result<()> {
        self.refresh_codes_cancellable(owner, &CancellationToken::new())
    }

    fn refresh_codes_cancellable(&self, owner: u64, token: &CancellationToken) -> Result<()> {
        cancelled(token)?;
        let Some(rotation) = self.store.retrieval.rotation.get() else {
            return Ok(());
        };
        let Some(row) = self.store.rows.get(&owner) else {
            return Ok(());
        };
        let Some(slot) = self.store.slots.get(row) else {
            return Ok(());
        };
        let payload = slot.payload.load();
        let Some(payload) = payload.as_ref() else {
            return Ok(());
        };
        let codes = if let VectorCoordinates::Spans(spans) = &payload.coordinates {
            spans
                .iter()
                .enumerate()
                .map(|(position, span)| {
                    #[cfg(test)]
                    if position == 0 {
                        if let Some(gate) = self.store.retrieval.first_code_gate.get() {
                            gate.wait();
                            gate.wait();
                        }
                    }
                    #[cfg(not(test))]
                    let _ = position;
                    cancelled(token)?;
                    let code = BinaryCode::new(
                        &span
                            .iter()
                            .map(|bits| decode_coordinate(self.dtype, *bits))
                            .collect::<Vec<_>>(),
                        rotation,
                    );
                    #[cfg(test)]
                    self.store
                        .retrieval
                        .encoded_spans
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    Ok(code)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            self.decode_payload(payload)
                .map(|vector| vec![BinaryCode::new(&vector, rotation)])
                .unwrap_or_default()
        };
        let coded = Arc::new(CodedOwner {
            stamp: payload.stamp,
            codes,
        });
        self.store
            .retrieval
            .code_bytes
            .fetch_add(coded.resident_bytes(), AtomicOrdering::Relaxed);
        if let Some(previous) = self
            .store
            .retrieval
            .codes
            .swap_value(owner, Arc::clone(&coded))
        {
            self.store
                .retrieval
                .code_bytes
                .fetch_sub(previous.resident_bytes(), AtomicOrdering::Relaxed);
        }
        if self.store.rows.get(&owner) != Some(row)
            || slot
                .payload
                .load()
                .as_ref()
                .is_none_or(|current| !Arc::ptr_eq(current, payload))
        {
            if let Some(removed) = self
                .store
                .retrieval
                .codes
                .remove_if_value(&owner, |current| Arc::ptr_eq(current, &coded))
            {
                self.store
                    .retrieval
                    .code_bytes
                    .fetch_sub(removed.resident_bytes(), AtomicOrdering::Relaxed);
            }
        }
        Ok(())
    }

    // A competing reader never waits for a cold builder: it uses exact scoring while the first
    // caller builds. Writers see no rotation until it is fully initialized and never wait on QR.
    fn ensure_quantization(&self, token: &CancellationToken) -> Result<bool> {
        let derived = &self.store.retrieval;
        if derived.quantization_ready.load(AtomicOrdering::Acquire) {
            return Ok(true);
        }
        if derived
            .quantization_building
            .compare_exchange(false, true, AtomicOrdering::AcqRel, AtomicOrdering::Acquire)
            .is_err()
        {
            return Ok(false);
        }
        struct Building<'a>(&'a AtomicBool);
        impl Drop for Building<'_> {
            fn drop(&mut self) {
                self.0.store(false, AtomicOrdering::Release);
            }
        }
        let _building = Building(&derived.quantization_building);
        if derived.rotation.get().is_none() {
            let rotation = Rotation::new(self.dimension, token)?;
            let _ = derived.rotation.set(rotation);
        }
        for row in 0..self.row_count() {
            cancelled(token)?;
            if self.row_active(row) {
                if let Some(owner) = self.row_entity(row) {
                    self.refresh_codes_cancellable(owner, token)?;
                }
            }
        }
        derived
            .quantization_ready
            .store(true, AtomicOrdering::Release);
        Ok(true)
    }

    pub(super) fn refresh_retrieval(&self, owner: u64) -> Result<()> {
        self.refresh_codes(owner)
    }

    pub(super) fn remove_retrieval(&self, owner: u64) {
        if let Some(codes) = self.store.retrieval.codes.remove(&owner) {
            self.store
                .retrieval
                .code_bytes
                .fetch_sub(codes.resident_bytes(), AtomicOrdering::Relaxed);
        }
    }

    /// Retained payload/codec/link bytes and fixed slot metadata; allocator overhead is excluded.
    /// Source graph properties referenced by this index are counted only by the canonical graph.
    pub fn resident_bytes(&self) -> usize {
        size_of::<Self>()
            + self.row_count() * size_of::<VectorSlot>()
            + self.store.payload_bytes.load(AtomicOrdering::Relaxed)
            + self
                .store
                .retrieval
                .code_bytes
                .load(AtomicOrdering::Relaxed)
            + self
                .store
                .retrieval
                .rotation
                .get()
                .map_or(0, Rotation::resident_bytes)
    }

    pub(super) fn rebuild_retrieval(&self) -> Result<()> {
        for row in 0..self.row_count() {
            if self.row_active(row) {
                if let Some(owner) = self.row_entity(row) {
                    self.refresh_retrieval(owner)?;
                }
            }
        }
        Ok(())
    }

    pub fn search_retrieval(
        &self,
        query: &[f32],
        limit: usize,
        strategy: RetrievalStrategy,
        visible: impl Fn(u64) -> bool,
    ) -> Result<Vec<VectorHit>> {
        self.search_retrieval_cancellable(
            query,
            limit,
            strategy,
            visible,
            &CancellationToken::new(),
        )
    }

    pub fn search_retrieval_cancellable(
        &self,
        query: &[f32],
        limit: usize,
        strategy: RetrievalStrategy,
        visible: impl Fn(u64) -> bool,
        cancellation: &CancellationToken,
    ) -> Result<Vec<VectorHit>> {
        cancelled(cancellation)?;
        let query = self.prepare(query)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut selected = strategy;
        if let RetrievalStrategy::Adaptive { candidates } = strategy {
            // Stop selectivity sampling after the exact route's crossover. No owner ID list or
            // vector copy is built. Tight filters use exact scoring rather than ANN post-filtering.
            let crossover = candidates.max(limit).saturating_mul(2);
            let mut eligible = 0usize;
            for row in 0..self.row_count() {
                cancelled(cancellation)?;
                if self.row_active(row) && self.row_entity(row).is_some_and(&visible) {
                    eligible += 1;
                    if eligible > crossover {
                        break;
                    }
                }
            }
            selected = if eligible <= crossover {
                RetrievalStrategy::Exact
            } else {
                RetrievalStrategy::Quantized { candidates }
            };
        }
        let budget = match selected {
            RetrievalStrategy::Exact => self.len(),
            RetrievalStrategy::Prefix { candidates, .. }
            | RetrievalStrategy::Quantized { candidates }
            | RetrievalStrategy::Adaptive { candidates } => candidates.max(limit),
        }
        .min(self.len());
        let limit = limit.min(self.len());
        if limit == 0 {
            return Ok(Vec::new());
        }
        if let RetrievalStrategy::Prefix { dimensions, .. } = selected {
            if dimensions == 0 || dimensions > self.dimension {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "retrieval prefix dimension is invalid",
                ));
            }
        }
        let mut candidates = Vec::new();
        if matches!(selected, RetrievalStrategy::Quantized { .. })
            && !self.ensure_quantization(cancellation)?
        {
            selected = RetrievalStrategy::Exact;
        }
        let transformed = if matches!(selected, RetrievalStrategy::Quantized { .. }) {
            self.store
                .retrieval
                .rotation
                .get()
                .map(|rotation| rotation.rotate(&query))
        } else {
            None
        };
        let query_norm = query
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();

        for row in 0..self.row_count() {
            cancelled(cancellation)?;
            let Some(slot) = self.store.slots.get(row) else {
                continue;
            };
            let payload = slot.payload.load();
            let Some(payload) = payload.as_ref() else {
                continue;
            };
            if !visible(payload.entity_id) {
                continue;
            }
            let score = match selected {
                RetrievalStrategy::Prefix { dimensions, .. } => {
                    let score_span = |span: &[u16]| {
                        let prefix = span
                            .iter()
                            .take(dimensions)
                            .map(|v| decode_coordinate(self.dtype, *v))
                            .collect::<Vec<_>>();
                        prefix_score(&query[..dimensions], &prefix, self.similarity)
                    };
                    if let VectorCoordinates::Spans(spans) = &payload.coordinates {
                        let mut best = None;
                        for span in spans {
                            cancelled(cancellation)?;
                            let score = score_span(span);
                            if best.is_none_or(|old| better(self.similarity, score, old)) {
                                best = Some(score);
                            }
                        }
                        best
                    } else {
                        self.payload_coordinates(payload.entity_id, payload)?
                            .map(|coordinates| {
                                let bits: Vec<_> = coordinates.iter().take(dimensions).collect();
                                score_span(&bits)
                            })
                    }
                }
                RetrievalStrategy::Quantized { .. } => {
                    let codes = self
                        .store
                        .retrieval
                        .codes
                        .get(&payload.entity_id)
                        .filter(|coded| coded.stamp == payload.stamp);
                    let mut approximate = None;
                    if let Some(codes) = codes {
                        for code in &codes.codes {
                            cancelled(cancellation)?;
                            let score = code.score(
                                transformed.as_deref().unwrap_or_default(),
                                query_norm,
                                self.similarity,
                            );
                            if approximate.is_none_or(|old| better(self.similarity, score, old)) {
                                approximate = Some(score);
                            }
                        }
                    }
                    match approximate {
                        Some(value) => Some(value),
                        None => self.score_payload_cancellable(&query, payload, cancellation)?,
                    }
                }
                _ => self.score_payload_cancellable(&query, payload, cancellation)?,
            };
            if let Some(score) = score {
                push_hit(
                    &mut candidates,
                    VectorHit {
                        entity_id: payload.entity_id,
                        score,
                    },
                    if selected == RetrievalStrategy::Exact {
                        limit
                    } else {
                        budget
                    },
                    self.similarity,
                );
            }
        }

        // Reacquire canonical payloads: approximate scores and cached revision data never become
        // final evidence, including after update/delete races or owner-slot reuse.
        let mut hits = Vec::new();
        for hit in candidates {
            cancelled(cancellation)?;
            if visible(hit.entity_id) {
                if let Some(score) =
                    self.score_owner_cancellable(&query, hit.entity_id, cancellation)?
                {
                    push_hit(
                        &mut hits,
                        VectorHit {
                            entity_id: hit.entity_id,
                            score,
                        },
                        limit,
                        self.similarity,
                    );
                }
            }
        }
        Ok(hits)
    }
}

fn prefix_score(query: &[f32], vector: &[f32], similarity: Similarity) -> f32 {
    if similarity == Similarity::Euclidean {
        query
            .iter()
            .zip(vector)
            .map(|(q, x)| (q - x).powi(2))
            .sum::<f32>()
            .sqrt()
    } else {
        let dot = query.iter().zip(vector).map(|(q, x)| q * x).sum::<f32>();
        if similarity == Similarity::Cosine {
            let norm = (query.iter().map(|v| v * v).sum::<f32>()
                * vector.iter().map(|v| v * v).sum::<f32>())
            .sqrt();
            if norm > f32::EPSILON { dot / norm } else { 0.0 }
        } else {
            dot
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Result<VectorIndex> {
        let index = VectorIndex::new(64, Similarity::Cosine)?;
        for owner in 0..512_u64 {
            let vector: Vec<_> = (0..64)
                .map(|d| ((mix(owner * 71 + d) % 2001) as f32 - 1000.0) / 1000.0)
                .collect();
            index.upsert(owner, &vector, owner + 1)?;
        }
        Ok(index)
    }

    #[test]
    fn retrieval_prefix_reranks_full_current_vectors() -> Result<()> {
        let index = fixture()?;
        let query = index
            .vector_for(321)
            .ok_or_else(|| Error::internal("missing query"))?;
        let expected = index.exact_search(&query, 10)?;
        let hits = index.search_retrieval(
            &query,
            10,
            RetrievalStrategy::Prefix {
                dimensions: 16,
                candidates: 512,
            },
            |_| true,
        )?;
        assert_eq!(hits, expected);
        assert_eq!(hits[0].entity_id, 321);
        assert!(
            index
                .search_retrieval(
                    &query,
                    10,
                    RetrievalStrategy::Prefix {
                        dimensions: 0,
                        candidates: 10
                    },
                    |_| true
                )
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn retrieval_filter_adapts_without_owner_leakage() -> Result<()> {
        let index = fixture()?;
        let query = index
            .vector_for(321)
            .ok_or_else(|| Error::internal("missing query"))?;
        let visible = |id| id % 71 == 0;
        let expected = index.exact_search_where(&query, 10, visible)?;
        assert_eq!(
            index.search_retrieval(
                &query,
                10,
                RetrievalStrategy::Adaptive { candidates: 32 },
                visible
            )?,
            expected
        );
        let hits = index.search_retrieval(
            &query,
            10,
            RetrievalStrategy::Quantized { candidates: 128 },
            visible,
        )?;
        assert!(hits.iter().all(|hit| visible(hit.entity_id)));
        assert_eq!(hits.len(), expected.len());
        Ok(())
    }

    #[test]
    fn retrieval_quant_transform_estimator_and_metrics() -> Result<()> {
        let vector: Vec<_> = (0..384).map(|v| (v as f32 * 0.17).sin()).collect();
        let rotation = Rotation::new(vector.len(), &CancellationToken::new())?;
        let transformed = rotation.rotate(&vector);
        let norm = vector.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
        assert!((norm - transformed.iter().map(|v| v * v).sum::<f64>()).abs() < 1e-8);
        let code = BinaryCode::new(&vector, &rotation);
        assert!(
            (f64::from(code.score(&transformed, norm.sqrt(), Similarity::Dot)) - norm).abs()
                < 0.001
        );
        assert!(code.score(&transformed, norm.sqrt(), Similarity::Euclidean) < 0.01);
        let zero_rotation = Rotation::new(64, &CancellationToken::new())?;
        let zero = BinaryCode::new(&[0.0; 64], &zero_rotation);
        assert_eq!(zero.score(&[0.0; 64], 0.0, Similarity::Dot), 0.0);
        let index = fixture()?;
        let query = index
            .vector_for(17)
            .ok_or_else(|| Error::internal("missing query"))?;
        let hits = index.search_retrieval(
            &query,
            10,
            RetrievalStrategy::Quantized { candidates: 128 },
            |_| true,
        )?;
        let expected = index.exact_search(&query, 10)?;
        assert_eq!(hits[0].entity_id, 17);
        assert!(
            hits.iter()
                .filter(|hit| expected.iter().any(|e| e.entity_id == hit.entity_id))
                .count()
                >= 8
        );
        Ok(())
    }

    #[test]
    fn retrieval_quant_rejects_encoded_overflow_and_cancels() -> Result<()> {
        let index = VectorIndex::new(2, Similarity::Dot)?;
        assert!(index.upsert(1, &[f32::MAX, 0.0], 1).is_err());
        assert!(index.quantize(&[f32::NAN, 1.0]).is_err());
        index.upsert(1, &[0.0, 0.0], 1)?;
        assert_eq!(
            index.search_retrieval(
                &[0.0, 0.0],
                1,
                RetrievalStrategy::Quantized { candidates: 1 },
                |_| true
            )?[0]
                .score,
            0.0
        );
        let token = CancellationToken::new();
        token.cancel();
        assert!(
            index
                .search_retrieval_cancellable(
                    &[0.0, 0.0],
                    1,
                    RetrievalStrategy::Quantized { candidates: 1 },
                    |_| true,
                    &token
                )
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn retrieval_quant_cold_reader_does_not_wait_for_builder() -> Result<()> {
        let index = fixture()?;
        let query = index
            .vector_for(23)
            .ok_or_else(|| Error::internal("missing query"))?;
        assert!(index.store.retrieval.rotation.get().is_none());
        index
            .store
            .retrieval
            .quantization_building
            .store(true, AtomicOrdering::Release);
        let hits = index.search_retrieval(
            &query,
            10,
            RetrievalStrategy::Quantized { candidates: 64 },
            |_| true,
        )?;
        assert_eq!(hits, index.exact_search(&query, 10)?);
        assert!(index.store.retrieval.rotation.get().is_none());
        index
            .store
            .retrieval
            .quantization_building
            .store(false, AtomicOrdering::Release);
        let before = index.resident_bytes();
        index.search_retrieval(
            &query,
            10,
            RetrievalStrategy::Quantized { candidates: 64 },
            |_| true,
        )?;
        assert!(index.resident_bytes() >= before + 64 * 64 * size_of::<f64>());
        let codes_before = index.store.retrieval.codes.len();
        index.remove(23, 999);
        assert_eq!(index.store.retrieval.codes.len(), codes_before - 1);
        assert!(index.store.retrieval.codes.get(&23).is_none());
        Ok(())
    }

    #[test]
    fn retrieval_quant_gaussian_rotation_is_orthonormal_and_reproducible() -> Result<()> {
        let token = CancellationToken::new();
        let rotation = Rotation::new(31, &token)?;
        let repeat = Rotation::new(31, &token)?;
        assert_eq!(rotation.rows, repeat.rows);
        for (i, row) in rotation.rows.chunks_exact(31).enumerate() {
            for (j, other) in rotation.rows.chunks_exact(31).enumerate() {
                let dot = row.iter().zip(other).map(|(a, b)| a * b).sum::<f64>();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        token.cancel();
        assert!(Rotation::new(384, &token).is_err());
        Ok(())
    }

    #[test]
    fn retrieval_span_single_large_owner_cancels_inside_scoring() -> Result<()> {
        let index = VectorIndex::new(64, Similarity::Cosine)?;
        let span = index.quantize(&[1.0; 64])?;
        index.upsert_spans(1, &vec![span; 4096], 1)?;
        let gate = Arc::new(std::sync::Barrier::new(2));
        let _ = index.store.retrieval.first_span_gate.set(Arc::clone(&gate));
        let token = CancellationToken::new();
        let result = std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                index.search_retrieval_cancellable(
                    &[1.0; 64],
                    1,
                    RetrievalStrategy::Exact,
                    |_| true,
                    &token,
                )
            });
            gate.wait();
            token.cancel();
            gate.wait();
            worker
                .join()
                .map_err(|_| Error::internal("scoring worker panicked"))?
        });
        assert_eq!(result.err().map(|e| e.code), Some(ErrorCode::Cancelled));
        assert!(
            index
                .store
                .retrieval
                .span_scores
                .load(AtomicOrdering::Relaxed)
                < 4096
        );
        Ok(())
    }

    #[test]
    fn retrieval_quant_single_large_owner_cancels_inside_cold_codes() -> Result<()> {
        let index = VectorIndex::new(64, Similarity::Cosine)?;
        let span = index.quantize(&[1.0; 64])?;
        index.upsert_spans(1, &vec![span; 4096], 1)?;
        let gate = Arc::new(std::sync::Barrier::new(2));
        let _ = index.store.retrieval.first_code_gate.set(Arc::clone(&gate));
        let token = CancellationToken::new();
        let result = std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                index.search_retrieval_cancellable(
                    &[1.0; 64],
                    1,
                    RetrievalStrategy::Quantized { candidates: 1 },
                    |_| true,
                    &token,
                )
            });
            gate.wait();
            token.cancel();
            gate.wait();
            worker
                .join()
                .map_err(|_| Error::internal("code worker panicked"))?
        });
        assert_eq!(result.err().map(|e| e.code), Some(ErrorCode::Cancelled));
        assert!(
            index
                .store
                .retrieval
                .encoded_spans
                .load(AtomicOrdering::Relaxed)
                < 4096
        );
        assert!(
            !index
                .store
                .retrieval
                .quantization_ready
                .load(AtomicOrdering::Acquire)
        );
        assert!(
            !index
                .store
                .retrieval
                .quantization_building
                .load(AtomicOrdering::Acquire)
        );
        Ok(())
    }

    #[test]
    fn retrieval_span_large_owner_mutation_wire_discriminants_remain_stable() -> Result<()> {
        let property = PropertyId(1);
        let upsert = ResolvedVectorMutation::Upsert {
            property,
            entity_id: 1,
            coordinates: vec![0; 384],
            revision: 1,
        };
        let remove = ResolvedVectorMutation::Remove {
            property,
            entity_id: 1,
            revision: 1,
        };
        let spans = ResolvedVectorMutation::UpsertSpans {
            property,
            entity_id: 1,
            coordinates: vec![vec![0; 384]; 128],
            revision: 1,
        };
        for (tag, mutation) in [upsert, remove, spans].into_iter().enumerate() {
            let wire =
                postcard::to_allocvec(&mutation).map_err(|e| Error::internal(e.to_string()))?;
            assert_eq!(wire[0], tag as u8);
            let restored: ResolvedVectorMutation =
                postcard::from_bytes(&wire).map_err(|e| Error::internal(e.to_string()))?;
            assert_eq!(restored, mutation);
        }
        Ok(())
    }

    #[test]
    #[ignore = "release-mode vector primitive diagnostic, not a complete-query benchmark"]
    fn retrieval_diagnostic_release() -> Result<()> {
        use std::time::Instant;
        let index = VectorIndex::new(384, Similarity::Cosine)?;
        let owners = std::env::var("IRONGRAPH_RETRIEVAL_BENCH_OWNERS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(1000);
        let build = Instant::now();
        for owner in 0..owners {
            let vector: Vec<_> = (0..384)
                .map(|d| ((mix(owner * 71 + d) % 2001) as f32 - 1000.0) / 1000.0)
                .collect();
            index.upsert(owner, &vector, owner + 1)?;
        }
        println!(
            "synthetic {owners}x384 canonical build_ms={} canonical_bytes={}",
            build.elapsed().as_millis(),
            index.resident_bytes()
        );
        let queries: Vec<_> = (0..owners)
            .step_by((owners / 40).max(1) as usize)
            .filter_map(|id| index.vector_for(id))
            .collect();
        let ground: Vec<_> = queries
            .iter()
            .map(|q| index.exact_search(q, 10))
            .collect::<Result<_>>()?;
        let selected_method = std::env::var("IRONGRAPH_RETRIEVAL_BENCH_METHOD").ok();
        if selected_method
            .as_deref()
            .is_some_and(|method| !matches!(method, "Exact" | "Prefix" | "Quantized" | "Adaptive"))
        {
            return Err(Error::invalid_data("unknown retrieval benchmark method"));
        }
        let selected_candidates = std::env::var("IRONGRAPH_RETRIEVAL_BENCH_CANDIDATES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let batch_only = std::env::var_os("IRONGRAPH_RETRIEVAL_BENCH_BATCH_ONLY").is_some();
        let legacy_exact = std::env::var_os("IRONGRAPH_RETRIEVAL_BENCH_LEGACY_EXACT").is_some();
        for strategy in [
            RetrievalStrategy::Exact,
            RetrievalStrategy::Prefix {
                dimensions: 128,
                candidates: 512,
            },
            RetrievalStrategy::Quantized { candidates: 512 },
            RetrievalStrategy::Adaptive { candidates: 128 },
        ] {
            let strategy = match (strategy, selected_candidates) {
                (RetrievalStrategy::Prefix { dimensions, .. }, Some(candidates)) => {
                    RetrievalStrategy::Prefix {
                        dimensions,
                        candidates,
                    }
                }
                (RetrievalStrategy::Quantized { .. }, Some(candidates)) => {
                    RetrievalStrategy::Quantized { candidates }
                }
                (RetrievalStrategy::Adaptive { .. }, Some(candidates)) => {
                    RetrievalStrategy::Adaptive { candidates }
                }
                (strategy, _) => strategy,
            };
            if selected_method
                .as_ref()
                .is_some_and(|name| !format!("{strategy:?}").starts_with(name))
            {
                continue;
            }
            let cold = Instant::now();
            let _ = index.search_retrieval(&queries[0], 10, strategy, |_| true)?;
            println!(
                "{strategy:?}: first_query_us={} retained_bytes={}",
                cold.elapsed().as_micros(),
                index.resident_bytes()
            );
            let start = Instant::now();
            let mut latencies = Vec::new();
            let mut matched = 0;
            let mut returned = 0;
            let mut checksum = 0_u64;
            for (query, expected) in queries.iter().zip(&ground).cycle().take(queries.len() * 5) {
                let query_start = (!batch_only).then(Instant::now);
                let hits = if legacy_exact && strategy == RetrievalStrategy::Exact {
                    index.exact_search(query, 10)?
                } else {
                    index.search_retrieval(query, 10, strategy, |_| true)?
                };
                latencies.push(query_start.map_or(0, |start| start.elapsed().as_micros()));
                matched += hits
                    .iter()
                    .filter(|hit| expected.iter().any(|e| e.entity_id == hit.entity_id))
                    .count();
                returned += hits.len();
                for hit in hits {
                    checksum ^= hit.entity_id ^ u64::from(hit.score.to_bits());
                }
            }
            assert_eq!(returned, queries.len() * 5 * 10);
            latencies.sort_unstable();
            println!(
                "{strategy:?}: queries={} returned={returned} total_us={} p50_us={} p95_us={} recall10={:.4} checksum={checksum}",
                latencies.len(),
                start.elapsed().as_micros(),
                latencies[latencies.len() / 2],
                latencies[latencies.len() * 95 / 100],
                matched as f64 / returned as f64
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "concurrent retrieval torture qualification; run explicitly"]
    fn retrieval_concurrent_torture_qualification() -> Result<()> {
        use std::sync::{Arc, Barrier};
        use std::time::Instant;
        let index = Arc::new(VectorIndex::new(32, Similarity::Cosine)?);
        let epochs = std::env::var("IRONGRAPH_RETRIEVAL_TORTURE_EPOCHS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(20);
        let reader_iterations = epochs as usize * 10;
        let vector = |id: u64| -> Vec<f32> {
            (0..32)
                .map(|d| (mix(id.wrapping_mul(71).wrapping_add(d)) % 2001) as f32 / 1000.0 - 1.0)
                .collect()
        };
        for owner in 0..256 {
            index.upsert(owner, &vector(owner), owner + 1)?;
        }
        let start = Instant::now();
        let gate = Arc::new(Barrier::new(5));
        let reads = std::thread::scope(|scope| -> Result<usize> {
            let mut readers = Vec::new();
            for reader in 0..4 {
                let index = Arc::clone(&index);
                let gate = Arc::clone(&gate);
                readers.push(scope.spawn(move || -> Result<usize> {
                    gate.wait();
                    for iteration in 0..reader_iterations {
                        let method = match (iteration + reader) % 4 {
                            0 => RetrievalStrategy::Exact,
                            1 => RetrievalStrategy::Quantized { candidates: 64 },
                            2 => RetrievalStrategy::Prefix {
                                dimensions: 16,
                                candidates: 64,
                            },
                            _ => RetrievalStrategy::Adaptive { candidates: 32 },
                        };
                        let hits = index.search_retrieval(
                            &vector(iteration as u64),
                            10,
                            method,
                            |id| id % 3 == 0,
                        )?;
                        let mut owners = std::collections::HashSet::new();
                        for hit in hits {
                            assert!(hit.score.is_finite());
                            assert_eq!(hit.entity_id % 3, 0);
                            assert!(owners.insert(hit.entity_id), "duplicate owner during churn");
                        }
                        // This fixture has 256 owners, at most eight large owners, and bounded
                        // codes/links. Sample accounting during publication, not just at rest.
                        assert!(
                            index.resident_bytes() < 1024 * 1024,
                            "fixture accounting exceeded its bounded shape during churn"
                        );
                    }
                    Ok(reader_iterations)
                }));
            }
            gate.wait();
            for epoch in 0..epochs {
                for owner in 0..64_u64 {
                    let revision = 10_000 + epoch * 1000 + owner * 2;
                    index.remove(owner, revision);
                    let quantized = index.quantize(&vector(owner + epoch))?;
                    if owner % 8 == 0 {
                        index.upsert_spans(owner, &vec![quantized; 64], revision + 1)?;
                    } else {
                        index.upsert(owner, &vector(owner + epoch), revision + 1)?;
                    }
                }
            }
            let mut total = 0;
            for reader in readers {
                total += reader
                    .join()
                    .map_err(|_| Error::internal("retrieval torture reader panicked"))??;
            }
            Ok(total)
        })?;
        assert_eq!(reads, reader_iterations * 4);
        assert_eq!(index.len(), 256);
        let query = vector(7);
        let expected = index.exact_search(&query, 256)?;
        let actual = index.search_retrieval(
            &query,
            256,
            RetrievalStrategy::Quantized { candidates: 256 },
            |_| true,
        )?;
        assert_eq!(
            actual, expected,
            "stable full reranking after concurrent updates"
        );
        let retained = index.resident_bytes();
        for epoch in 0..10_u64 {
            for owner in 0..64 {
                index.upsert(
                    owner,
                    &vector(owner),
                    10_000 + epochs * 1000 + epoch * 1000 + owner,
                )?;
            }
        }
        let final_bytes = index.resident_bytes();
        let codes_bytes: usize = index
            .store
            .retrieval
            .codes
            .values()
            .map(|coded| coded.resident_bytes())
            .sum();
        assert_eq!(
            index
                .store
                .retrieval
                .code_bytes
                .load(AtomicOrdering::Relaxed),
            codes_bytes,
            "code accounting after cold-build/write races"
        );
        assert!(
            final_bytes <= retained,
            "retained state grew after replacing large owners: {retained} -> {final_bytes}"
        );
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert_eq!(
            index
                .search_retrieval_cancellable(
                    &query,
                    usize::MAX,
                    RetrievalStrategy::Quantized {
                        candidates: usize::MAX
                    },
                    |_| true,
                    &cancelled
                )
                .err()
                .map(|e| e.code),
            Some(ErrorCode::Cancelled)
        );
        println!(
            "TORTURE_OK readers={reads} remove_reinsert={} final_replacements=640 elapsed_ms={} retained_large={retained} retained_final={final_bytes}",
            epochs * 64,
            start.elapsed().as_millis()
        );
        Ok(())
    }

    #[test]
    fn retrieval_span_max_scoring_roundtrip_update_and_delete() -> Result<()> {
        let index = VectorIndex::new(2, Similarity::Cosine)?;
        index.upsert_spans(
            7,
            &[index.quantize(&[1.0, 0.0])?, index.quantize(&[0.0, 1.0])?],
            1,
        )?;
        index.upsert(8, &[0.5, 0.5], 1)?;
        assert!(index.has_multi_span_owners());
        assert_eq!(index.exact_search(&[0.0, 1.0], 1)?[0].entity_id, 7);
        let bytes = postcard::to_allocvec(&index).map_err(|e| Error::internal(e.to_string()))?;
        let restored: VectorIndex =
            postcard::from_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
        assert!(restored.has_multi_span_owners());
        assert_eq!(restored.exact_search(&[0.0, 1.0], 1)?[0].entity_id, 7);
        restored.upsert(7, &[1.0, 0.0], 2)?;
        assert!(!restored.has_multi_span_owners());
        assert_eq!(restored.exact_search(&[0.0, 1.0], 1)?[0].entity_id, 8);
        restored.remove(8, 3);
        assert_eq!(restored.exact_search(&[0.0, 1.0], 2)?.len(), 1);
        Ok(())
    }

    #[test]
    fn retrieval_huge_limit_is_not_a_preallocation_request() -> Result<()> {
        let index = VectorIndex::new(2, Similarity::Cosine)?;
        for id in 0..3 {
            index.upsert(id, &[1.0, id as f32], 1)?;
        }
        for strategy in [
            RetrievalStrategy::Exact,
            RetrievalStrategy::Prefix {
                dimensions: 1,
                candidates: usize::MAX,
            },
            RetrievalStrategy::Quantized {
                candidates: usize::MAX,
            },
            RetrievalStrategy::Adaptive {
                candidates: usize::MAX,
            },
        ] {
            let hits = index.search_retrieval(&[1.0, 0.0], usize::MAX, strategy, |_| true)?;
            assert_eq!(hits.len(), 3);
            assert_eq!(
                hits.iter()
                    .map(|hit| hit.entity_id)
                    .collect::<BTreeSet<_>>()
                    .len(),
                3
            );
        }
        Ok(())
    }

    #[test]
    fn retrieval_span_single_window_marker_survives_recovery_and_replacement() -> Result<()> {
        let index = VectorIndex::new(2, Similarity::Cosine)?;
        index.upsert(1, &[1.0, 0.0], 1)?;
        assert!(!index.has_owner_spans(1));
        index.upsert_spans(1, &[index.quantize(&[1.0, 0.0])?], 2)?;
        assert!(index.has_owner_spans(1));
        assert!(!index.has_multi_span_owners());
        let bytes = postcard::to_allocvec(&index).map_err(|e| Error::internal(e.to_string()))?;
        let restored: VectorIndex =
            postcard::from_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
        assert!(restored.has_owner_spans(1));
        restored.upsert(1, &[1.0, 0.0], 3)?;
        assert!(!restored.has_owner_spans(1));
        restored.remove(1, 4);
        assert!(!restored.has_owner_spans(1));
        Ok(())
    }
}
