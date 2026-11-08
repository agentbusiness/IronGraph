//! Lazy changed-owner lexical statistics on existing canonical vector allocations.
use super::*;

struct LexicalBuild<'a>(&'a std::sync::atomic::AtomicUsize);
impl Drop for LexicalBuild<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}

/// Query-independent term counts. Original text remains on the graph owner.
#[derive(Debug)]
pub struct LexicalDocument {
    stamp: [u64; 5],
    epoch: u64,
    length: usize,
    frequencies: Vec<(String, usize)>,
}

impl LexicalDocument {
    /// Uncached fallback retains only requested counts, matching the original lexical scan.
    pub fn for_query(
        text: &str,
        terms: &[String],
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        let mut frequencies: Vec<_> = terms.iter().map(|term| (term.clone(), 0_usize)).collect();
        frequencies.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        frequencies.dedup_by(|a, b| a.0 == b.0);
        let mut length = 0_usize;
        for (position, token) in text.split(|c: char| !c.is_alphanumeric()).enumerate() {
            if position & 1023 == 0 && cancellation.is_cancelled() {
                return Err(Error::new(
                    ErrorCode::Cancelled,
                    "lexical retrieval cancelled",
                ));
            }
            if !token.is_empty() {
                length = length.saturating_add(1);
                let token = token.to_lowercase();
                if let Ok(at) = frequencies.binary_search_by(|(word, _)| word.cmp(&token)) {
                    frequencies[at].1 = frequencies[at].1.saturating_add(1);
                }
            }
        }
        Ok(Self {
            stamp: [0; 5],
            epoch: 0,
            length,
            frequencies,
        })
    }
    fn reusable(text: &str, cancellation: &CancellationToken) -> Result<Option<Self>> {
        let mut counts = BTreeMap::<String, usize>::new();
        let mut length = 0_usize;
        let mut bytes = size_of::<Self>();
        for (position, token) in text.split(|c: char| !c.is_alphanumeric()).enumerate() {
            if position & 1023 == 0 && cancellation.is_cancelled() {
                return Err(Error::new(
                    ErrorCode::Cancelled,
                    "lexical retrieval cancelled",
                ));
            }
            if !token.is_empty() {
                length = length.saturating_add(1);
                match counts.entry(token.to_lowercase()) {
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        let count = entry.get_mut();
                        *count = count.saturating_add(1);
                    }
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        bytes = bytes
                            .saturating_add(size_of::<(String, usize)>())
                            .saturating_add(entry.key().capacity());
                        // Abort only the cache build, not the query. Even an unrepresentative
                        // repetitive prefix cannot amplify a long unique-token tail.
                        if bytes > text.len() / 8 {
                            return Ok(None);
                        }
                        entry.insert(1);
                    }
                }
            }
        }
        Ok(Some(Self {
            stamp: [0; 5],
            epoch: 0,
            length,
            frequencies: counts.into_iter().collect(),
        }))
    }
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }
    #[must_use]
    pub fn frequency(&self, term: &str) -> usize {
        self.frequencies
            .binary_search_by(|(word, _)| word.as_str().cmp(term))
            .ok()
            .map_or(0, |at| self.frequencies[at].1)
    }
    fn resident_bytes(&self) -> usize {
        size_of::<Self>()
            + self.frequencies.capacity() * size_of::<(String, usize)>()
            + self
                .frequencies
                .iter()
                .map(|(word, _)| word.capacity())
                .sum::<usize>()
    }
}

impl VectorIndex {
    /// Reuse counts only for this exact graph/source/endpoint revision tuple. Transaction
    /// overlays use statement-local counts instead of publishing derived data.
    pub fn lexical_document(
        &self,
        entity_id: u64,
        stamp: [u64; 5],
        terms: &[String],
        load: impl FnOnce() -> Result<Option<String>>,
        cancellation: &CancellationToken,
    ) -> Result<Option<Arc<LexicalDocument>>> {
        if cancellation.is_cancelled() {
            return Err(Error::new(
                ErrorCode::Cancelled,
                "lexical retrieval cancelled",
            ));
        }
        self.store
            .lexical_building
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _building = LexicalBuild(&self.store.lexical_building);
        // Pair with invalidation's fence before inspecting canonical text. A first cold build
        // cannot miss a writer that observed no retained counts before starting its mutation.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let payload = self
            .store
            .rows
            .get(&entity_id)
            .and_then(|row| self.store.slots.get(row))
            .and_then(|slot| slot.payload.load_full())
            .filter(|payload| payload.entity_id == entity_id);
        let epoch = payload.as_ref().map_or(0, |payload| {
            payload
                .lexical_epoch
                .load(std::sync::atomic::Ordering::Acquire)
        });
        if let Some(cached) = payload
            .as_ref()
            .and_then(|payload| payload.lexical.load_full())
        {
            if cached.stamp == stamp && cached.epoch == epoch {
                return Ok(Some(cached));
            }
        }
        let Some(text) = load()? else { return Ok(None) };
        // The sample affects cache retention only; every token still contributes to scoring.
        let mut reusable = text.len() >= 1024 && {
            let sample: Vec<_> = text
                .split(|c: char| !c.is_alphanumeric())
                .filter(|term| !term.is_empty())
                .take(64)
                .map(str::to_lowercase)
                .collect();
            sample.iter().collect::<BTreeSet<_>>().len() * 4 <= sample.len()
        };
        let mut document = if reusable {
            match LexicalDocument::reusable(&text, cancellation)? {
                Some(document) => document,
                None => {
                    reusable = false;
                    LexicalDocument::for_query(&text, terms, cancellation)?
                }
            }
        } else {
            LexicalDocument::for_query(&text, terms, cancellation)?
        };
        document.stamp = stamp;
        document.epoch = epoch;
        let document = Arc::new(document);
        if let Some(payload) = payload {
            // Replacement retires this exact allocation. A concurrent reader cannot publish
            // into its replacement or resurrect a deleted owner through this handle.
            // Unique-token prose can cost more as a count dictionary than as source text.
            // Do not retain amplified derived data. This never limits or rejects a query.
            let retained = reusable && document.resident_bytes() <= text.len() / 8;
            if retained {
                self.store
                    .lexical_active
                    .store(true, std::sync::atomic::Ordering::Release);
            }
            payload.lexical.store(retained.then(|| document.clone()));
        }
        Ok(Some(document))
    }

    /// Current derived counts only; this diagnostic scans metadata, never source text/vectors.
    #[must_use]
    pub fn lexical_resident_bytes(&self) -> usize {
        if !self
            .store
            .lexical_active
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return 0;
        }
        (0..self.row_count())
            .filter_map(|row| self.store.slots.get(row))
            .filter_map(|slot| slot.payload.load_full())
            .filter_map(|payload| payload.lexical.load_full())
            .map(|document| document.resident_bytes())
            .sum()
    }

    fn invalidate_lexical(&self, entity_id: u64) {
        if let Some(payload) = self
            .store
            .rows
            .get(&entity_id)
            .and_then(|row| self.store.slots.get(row))
            .and_then(|slot| slot.payload.load_full())
        {
            payload
                .lexical_epoch
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            payload.lexical.store(None);
        }
    }

    fn lexical_in_use(&self) -> bool {
        // Read the transient count first: a completed builder publishes retention before its
        // release decrement. Reading these in the opposite order could miss that handoff.
        self.store
            .lexical_building
            .load(std::sync::atomic::Ordering::Acquire)
            != 0
            || self
                .store
                .lexical_active
                .load(std::sync::atomic::Ordering::Acquire)
    }
}

impl IndexCatalog {
    /// Per-mutation fencing also handles multiple changes sharing one publication revision.
    /// Only affected owners/incident relationships are inspected, never unrelated source text.
    pub(super) fn invalidate_lexical_mutation(
        &self,
        graph: &GraphStore,
        mutation: &GraphMutation,
    ) -> Result<()> {
        if self.vector_columns.is_empty() {
            return Ok(());
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        if !self
            .vector_columns
            .values()
            .any(|column| column.lexical_in_use())
        {
            return Ok(());
        }
        let mut nodes = BTreeSet::new();
        let mut edges = BTreeSet::new();
        // Declared text indexes can explicitly select fields excluded from automatic meaning.
        // Fence every directly edited owner as well, even when its automatic text is unchanged.
        match mutation {
            GraphMutation::InsertNode(node) => {
                nodes.insert(node.id);
            }
            GraphMutation::DeleteNode { node, .. }
            | GraphMutation::AddNodeLabels { node, .. }
            | GraphMutation::RemoveNodeLabels { node, .. }
            | GraphMutation::SetNodeProperty { node, .. } => {
                nodes.insert(*node);
            }
            GraphMutation::InsertEdge(edge) => {
                edges.insert(edge.id);
            }
            GraphMutation::DeleteEdge { edge, .. }
            | GraphMutation::SetEdgeProperty { edge, .. } => {
                edges.insert(*edge);
            }
            _ => {}
        }
        if self
            .vector_columns
            .get(&crate::SEMANTIC_RELATIONSHIP_PROPERTY)
            .is_some_and(|column| column.lexical_in_use())
        {
            edges.extend(
                crate::semantic_affected_owners(graph, &[], std::slice::from_ref(mutation))?.1,
            );
        }
        for (property, column) in self.vector_columns.iter() {
            if !column.lexical_in_use() {
                continue;
            }
            if property == crate::SEMANTIC_RELATIONSHIP_PROPERTY {
                for edge in &edges {
                    column.invalidate_lexical(edge.0);
                }
            } else {
                for node in &nodes {
                    column.invalidate_lexical(node.0);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retrieval_lexical_cold_build_racing_write_cannot_leave_valid_old_counts() -> Result<()> {
        let graph = GraphStore::default();
        let property = graph.catalog().intern_property("body")?;
        graph.insert_node(crate::NodeInput {
            id: crate::NodeId(1),
            layer: crate::Layer::Observed,
            revision: 1,
            labels: vec![],
            properties: vec![],
        })?;
        let indexes = IndexCatalog::default();
        indexes.initialize_semantic(EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            2,
            EmbeddingDType::F16,
            true,
            Similarity::Cosine,
        )?)?;
        let (vectors, _) = indexes
            .vector_search_source(crate::SEMANTIC_NODE_INDEX)
            .ok_or_else(|| Error::internal("fixture index absent"))?;
        vectors.upsert(1, &[1.0, 0.0], 1)?;
        let entered = std::sync::Barrier::new(2);
        let release = std::sync::Barrier::new(2);
        std::thread::scope(|scope| -> Result<()> {
            let reader = scope.spawn(|| {
                vectors.lexical_document(
                    1,
                    [1; 5],
                    &["needle".to_owned()],
                    || {
                        let old = "needle ".repeat(512);
                        entered.wait();
                        release.wait();
                        Ok(Some(old))
                    },
                    &CancellationToken::new(),
                )
            });
            entered.wait();
            let mutation = GraphMutation::SetNodeProperty {
                node: crate::NodeId(1),
                property,
                value: ScalarValue::String("replacement ".repeat(512).into()),
                revision: 1,
            };
            let write = (|| -> Result<()> {
                indexes.before_graph_apply(&graph, &mutation)?;
                graph.apply(mutation.clone())?;
                indexes.after_graph_apply(&graph, &mutation)
            })();
            release.wait();
            reader
                .join()
                .map_err(|_| Error::internal("cold reader panicked"))??;
            write
        })?;
        let current = vectors
            .lexical_document(
                1,
                [1; 5],
                &["needle".to_owned()],
                || Ok(Some("replacement ".repeat(512))),
                &CancellationToken::new(),
            )?
            .ok_or_else(|| Error::internal("current counts absent"))?;
        assert_eq!(current.frequency("needle"), 0);
        Ok(())
    }
    #[test]
    fn retrieval_lexical_explicit_metadata_field_is_fenced_within_one_revision() -> Result<()> {
        let graph = GraphStore::default();
        let property = graph.catalog().intern_property("_private")?;
        graph.insert_node(crate::NodeInput {
            id: crate::NodeId(1),
            layer: crate::Layer::Observed,
            revision: 1,
            labels: vec![],
            properties: vec![],
        })?;
        let indexes = IndexCatalog::default();
        indexes.initialize_semantic(EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            2,
            EmbeddingDType::F16,
            true,
            Similarity::Cosine,
        )?)?;
        let (vectors, _) = indexes
            .vector_search_source(crate::SEMANTIC_NODE_INDEX)
            .ok_or_else(|| Error::internal("fixture index absent"))?;
        vectors.upsert(1, &[1.0, 0.0], 1)?;
        let token = CancellationToken::new();
        let terms = vec!["needle".to_owned()];
        let first = vectors
            .lexical_document(
                1,
                [1; 5],
                &terms,
                || Ok(Some("needle ".repeat(512))),
                &token,
            )?
            .ok_or_else(|| Error::internal("fixture counts absent"))?;
        assert_eq!(first.frequency("needle"), 512);
        let mutation = GraphMutation::SetNodeProperty {
            node: crate::NodeId(1),
            property,
            value: ScalarValue::String("replacement ".repeat(512).into()),
            revision: 1,
        };
        indexes.before_graph_apply(&graph, &mutation)?;
        graph.apply(mutation.clone())?;
        indexes.after_graph_apply(&graph, &mutation)?;
        let next = vectors
            .lexical_document(
                1,
                [1; 5],
                &terms,
                || Ok(Some("replacement ".repeat(512))),
                &token,
            )?
            .ok_or_else(|| Error::internal("updated counts absent"))?;
        assert_eq!(next.frequency("needle"), 0);
        Ok(())
    }
    #[test]
    fn retrieval_lexical_diverse_text_does_not_retain_amplified_or_query_specific_counts()
    -> Result<()> {
        let index = VectorIndex::new(2, Similarity::Cosine)?;
        index.upsert(1, &[1.0, 0.0], 1)?;
        let text = (0..5000).map(|i| format!("token{i} ")).collect::<String>();
        for term in ["token1", "token4999"] {
            let terms = vec![term.to_owned()];
            let document = index
                .lexical_document(
                    1,
                    [1; 5],
                    &terms,
                    || Ok(Some(text.clone())),
                    &CancellationToken::new(),
                )?
                .ok_or_else(|| Error::internal("diverse fixture absent"))?;
            assert_eq!(document.frequency(term), 1);
            assert_eq!(document.length(), 5000);
            assert_eq!(index.lexical_resident_bytes(), 0);
            assert!(
                !index.lexical_in_use(),
                "uncached queries must not leave write-path work active"
            );
        }
        let adversarial = format!("{}{}", "repeated ".repeat(128), text);
        let document = index
            .lexical_document(
                1,
                [2; 5],
                &["token4999".to_owned()],
                || Ok(Some(adversarial)),
                &CancellationToken::new(),
            )?
            .ok_or_else(|| Error::internal("adversarial fixture absent"))?;
        assert_eq!(document.frequency("token4999"), 1);
        assert_eq!(document.length(), 5128);
        assert_eq!(index.lexical_resident_bytes(), 0);
        Ok(())
    }
    #[test]
    fn retrieval_lexical_reuses_only_unchanged_owner_counts_and_reclaims_on_delete() -> Result<()> {
        let index = VectorIndex::new(2, Similarity::Cosine)?;
        let token = CancellationToken::new();
        let terms = vec!["needle".to_owned()];
        index.upsert(1, &[1.0, 0.0], 1)?;
        let text = "NEEDLE needle 日本語 ".repeat(65_536);
        let first = index
            .lexical_document(
                1,
                [1, 1, 0, 0, 0],
                &terms,
                || Ok(Some(text.clone())),
                &token,
            )?
            .ok_or_else(|| Error::internal("lexical fixture owner absent"))?;
        assert_eq!(first.frequency("needle"), 131_072);
        let bytes = index.lexical_resident_bytes();
        assert!(
            bytes < 1024,
            "large repeated text retains counts only: {bytes}"
        );
        for _ in 0..128 {
            let same = index
                .lexical_document(
                    1,
                    [1, 1, 0, 0, 0],
                    &terms,
                    || Err(Error::internal("unchanged owner reparsed")),
                    &token,
                )?
                .ok_or_else(|| Error::internal("lexical fixture owner absent"))?;
            assert!(Arc::ptr_eq(&first, &same));
        }
        let next = index
            .lexical_document(
                1,
                [1, 2, 0, 0, 0],
                &terms,
                || Ok(Some("replacement".into())),
                &token,
            )?
            .ok_or_else(|| Error::internal("lexical fixture owner absent"))?;
        assert_eq!(next.frequency("needle"), 0);
        assert_eq!(next.length(), 1);
        index.remove(1, 3);
        assert_eq!(index.lexical_resident_bytes(), 0);
        assert_eq!(first.frequency("needle"), 131_072);
        token.cancel();
        assert!(
            index
                .lexical_document(1, [0; 5], &terms, || Ok(Some(text)), &token)
                .is_err()
        );
        Ok(())
    }
}
