//! Record-wise durability adapter for the existing canonical database objects.
use super::*;
use crate::graph::PersistenceOp;
use crate::types::EntityKind;

const HEADER: &[u8] = b"!database";
const PROJECTS: &[u8] = b"p/";
const PROJECT_METADATA: &[u8] = b"m/";
const REQUESTS: &[u8] = b"r/";
pub(super) const EMBEDDING_PENDING: &[u8] = b"e/";

#[derive(Serialize, Deserialize)]
struct Header {
    embedding_activation: Option<EmbeddingProfileActivation>,
    applied: Bookmark,
    last_payload_checksum: Option<[u8; 32]>,
    last_response: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct ProjectMetadata {
    id: ProjectId,
    display_name: String,
    next_node_id: u64,
    next_edge_id: u64,
    authority_revision: u64,
    predicate_versions: BTreeMap<DependencyStamp, u64>,
}

pub(super) struct Capture {
    project: Option<ProjectId>,
    dropped: bool,
    owners: Vec<super::super::embedding_jobs::EmbeddingOwner>,
    seed_project: bool,
    security: bool,
    completed_owners: Vec<super::super::embedding_jobs::EmbeddingOwner>,
}

pub(super) fn project_prefix(id: ProjectId) -> Vec<u8> {
    let mut key = PROJECTS.to_vec();
    key.extend_from_slice(id.0.as_bytes());
    key.push(b'/');
    key
}

fn project_metadata_key(id: ProjectId) -> Vec<u8> {
    let mut key = PROJECT_METADATA.to_vec();
    key.extend_from_slice(id.0.as_bytes());
    key
}

pub(super) fn prepare(state: &DatabaseState, mutation: &DatabaseMutation) -> Result<Capture> {
    let project = match mutation {
        DatabaseMutation::CreateProject { id, .. }
        | DatabaseMutation::RenameProject { id, .. }
        | DatabaseMutation::DropProject { id, .. } => Some(*id),
        DatabaseMutation::Graph { project, .. } => Some(*project),
        DatabaseMutation::EmbeddingProfile { command } => Some(match command {
            EmbeddingProfileCommand::Begin { project, .. }
            | EmbeddingProfileCommand::Acknowledge { project, .. }
            | EmbeddingProfileCommand::Activate { project, .. }
            | EmbeddingProfileCommand::Abort { project, .. } => *project,
        }),
        _ => None,
    };
    let mut owners = Vec::new();
    let mut seed_project = false;
    let mut completed_owners = Vec::new();
    if let DatabaseMutation::Graph {
        project,
        graph,
        administrative,
        ..
    } = mutation
    {
        if let Some(live) = state
            .projects
            .get(project)
            .filter(|live| live.indexes.profile().is_some())
        {
            let (nodes, edges) = embedding_changed_owners(live, graph)?;
            owners.extend(nodes.into_iter().map(|id| {
                super::super::embedding_jobs::EmbeddingOwner {
                    project: *project,
                    kind: EntityKind::Node,
                    entity_id: id.0,
                }
            }));
            owners.extend(edges.into_iter().map(|id| {
                super::super::embedding_jobs::EmbeddingOwner {
                    project: *project,
                    kind: EntityKind::Relationship,
                    entity_id: id.0,
                }
            }));
        }
        seed_project = matches!(
            administrative,
            Some(
                AdministrativeMutation::CreateEmbedding { .. }
                    | AdministrativeMutation::InitializeSemantic { .. }
                    | AdministrativeMutation::AlterAutomaticSemantic { .. }
            )
        );
    }
    if let DatabaseMutation::Graph {
        project, vectors, ..
    } = mutation
    {
        for vector in vectors {
            let (property, entity_id) = match vector {
                ResolvedVectorMutation::Upsert {
                    property,
                    entity_id,
                    ..
                }
                | ResolvedVectorMutation::UpsertSpans {
                    property,
                    entity_id,
                    ..
                }
                | ResolvedVectorMutation::Remove {
                    property,
                    entity_id,
                    ..
                } => (*property, *entity_id),
            };
            if property != crate::graph::SEMANTIC_NODE_PROPERTY
                && property != crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY
            {
                continue;
            }
            let kind = if property == crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY {
                EntityKind::Relationship
            } else {
                EntityKind::Node
            };
            completed_owners.push(super::super::embedding_jobs::EmbeddingOwner {
                project: *project,
                kind,
                entity_id,
            });
        }
    }
    Ok(Capture {
        project,
        dropped: matches!(mutation, DatabaseMutation::DropProject { .. }),
        owners,
        seed_project,
        security: matches!(mutation, DatabaseMutation::Security { .. }),
        completed_owners,
    })
}

pub(super) fn put<T: Serialize>(key: Vec<u8>, value: &T) -> Result<RecordOp> {
    Ok(RecordOp::Put {
        key,
        value: encode_database_value(value, "durable record")?,
    })
}

pub(super) fn append_project_operations(
    output: &mut Vec<RecordOp>,
    project: ProjectId,
    operations: Vec<PersistenceOp>,
) {
    let prefix = project_prefix(project);
    for operation in operations {
        let qualify = |suffix: Vec<u8>| {
            let mut key = prefix.clone();
            key.extend(suffix);
            key
        };
        output.push(match operation {
            PersistenceOp::Put { key, value } => RecordOp::Put {
                key: qualify(key),
                value,
            },
            PersistenceOp::Delete { key } => RecordOp::Delete { key: qualify(key) },
            PersistenceOp::DeletePrefix { prefix } => RecordOp::DeletePrefix {
                prefix: qualify(prefix),
            },
        });
    }
}

pub(super) fn pending_key(owner: super::super::embedding_jobs::EmbeddingOwner) -> Vec<u8> {
    let mut key = EMBEDDING_PENDING.to_vec();
    key.extend_from_slice(owner.project.0.as_bytes());
    key.push(owner.kind as u8);
    key.extend_from_slice(&owner.entity_id.to_be_bytes());
    key
}

pub(super) fn decode_pending(
    key: &[u8],
    value: &[u8],
) -> Result<super::super::embedding_jobs::EmbeddingJob> {
    if !key.starts_with(EMBEDDING_PENDING) || key.len() != 27 || value.len() != 8 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "invalid pending embedding owner",
        ));
    }
    let project = Uuid::from_slice(&key[2..18]).map_err(|e| Error::invalid_data(e.to_string()))?;
    let kind = match key[18] {
        x if x == EntityKind::Node as u8 => EntityKind::Node,
        x if x == EntityKind::Relationship as u8 => EntityKind::Relationship,
        _ => {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "invalid pending owner kind",
            ));
        }
    };
    let entity_id = u64::from_be_bytes(
        key[19..27]
            .try_into()
            .map_err(|_| Error::invalid_data("owner ID"))?,
    );
    let revision = u64::from_be_bytes(
        value
            .try_into()
            .map_err(|_| Error::invalid_data("owner revision"))?,
    );
    Ok(super::super::embedding_jobs::EmbeddingJob {
        owner: super::super::embedding_jobs::EmbeddingOwner {
            project: ProjectId(project),
            kind,
            entity_id,
        },
        revision,
        relationship_scope_revision: None,
    })
}

pub(super) fn finish(
    state: &DatabaseState,
    capture: Capture,
    mut operations: Vec<RecordOp>,
    request: Option<Uuid>,
    removed_requests: &[Uuid],
) -> Result<RecordBatch> {
    operations.push(put(
        HEADER.to_vec(),
        &Header {
            embedding_activation: state.embedding_activation.clone(),
            applied: state.applied,
            last_payload_checksum: state.last_payload_checksum,
            last_response: state.last_response.clone(),
        },
    )?);
    if capture.security {
        operations.push(put(b"!security".to_vec(), &state.security)?);
    }
    if let Some(id) = capture.project {
        let prefix = project_prefix(id);
        if capture.dropped {
            operations.push(RecordOp::DeletePrefix { prefix });
            operations.push(RecordOp::Delete {
                key: project_metadata_key(id),
            });
            let mut pending = EMBEDDING_PENDING.to_vec();
            pending.extend_from_slice(id.0.as_bytes());
            operations.push(RecordOp::DeletePrefix { prefix: pending });
            let mut seed = b"s/".to_vec();
            seed.extend_from_slice(id.0.as_bytes());
            operations.push(RecordOp::Delete { key: seed });
        } else if let Some(project) = state.projects.get(&id) {
            let key = project_metadata_key(id);
            operations.push(put(
                key,
                &ProjectMetadata {
                    id,
                    display_name: project.display_name.get().to_string(),
                    next_node_id: project.next_node_id.get(),
                    next_edge_id: project.next_edge_id.get(),
                    authority_revision: project.authority_revision.get(),
                    predicate_versions: project.predicate_versions.clone(),
                },
            )?);
            for owner in capture.owners {
                let key = pending_key(owner);
                match embedding_owner_dependencies(project, owner).0 {
                    Some(revision) => operations.push(RecordOp::Put {
                        key,
                        value: revision.to_be_bytes().to_vec(),
                    }),
                    None => operations.push(RecordOp::Delete { key }),
                }
            }
            if capture.seed_project {
                let mut key = b"s/".to_vec();
                key.extend_from_slice(id.0.as_bytes());
                operations.push(RecordOp::Put {
                    key,
                    value: state.applied.index.to_be_bytes().to_vec(),
                });
            }
        }
    }
    if let Some(request) = request {
        if let Some(record) = state.request_results.get(&request) {
            let mut key = REQUESTS.to_vec();
            key.extend_from_slice(request.as_bytes());
            operations.push(put(key, record)?);
        }
    }
    for request in removed_requests {
        let mut key = REQUESTS.to_vec();
        key.extend_from_slice(request.as_bytes());
        operations.push(RecordOp::Delete { key });
    }
    for owner in capture.completed_owners {
        operations.push(RecordOp::Delete {
            key: pending_key(owner),
        });
    }
    Ok(RecordBatch {
        first_index: state.applied.index,
        bookmark: state.applied,
        operations,
    })
}

pub(super) fn restore(records: &RecordStore) -> Result<DatabaseState> {
    let mut state = DatabaseState::default();
    if let Some(bytes) = records.get(HEADER)? {
        let header: Header = decode_database_value(&bytes, "database metadata")?;
        if header.applied != records.bookmark()? {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "durable metadata position differs from records",
            ));
        }
        state.embedding_activation = header.embedding_activation;
        state.applied = header.applied;
        state.last_payload_checksum = header.last_payload_checksum;
        state.last_response = header.last_response;
    } else if records.bookmark()? != Bookmark::default() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database metadata missing",
        ));
    }
    if let Some(bytes) = records.get(b"!security")? {
        state.security = decode_database_value(&bytes, "security metadata")?;
    }
    records.scan_prefix(PROJECT_METADATA, |key, bytes| {
        if key.len() == 18 {
            let metadata: ProjectMetadata = decode_database_value(bytes, "project metadata")?;
            if project_metadata_key(metadata.id) != key {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "project record identity differs",
                ));
            }
            let normalized = normalize_name(&metadata.display_name);
            if state.names.insert(normalized, metadata.id).is_some() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "duplicate project name",
                ));
            }
            state.projects.insert(
                metadata.id,
                Arc::new(ProjectState {
                    id: metadata.id,
                    display_name: metadata.display_name.into(),
                    graph: GraphStore::default(),
                    temporal: TemporalStore::default(),
                    indexes: IndexCatalog::default(),
                    predicate_versions: metadata.predicate_versions,
                    next_node_id: metadata.next_node_id.into(),
                    next_edge_id: metadata.next_edge_id.into(),
                    authority_revision: metadata.authority_revision.into(),
                    optimizer_statistics: OnceLock::new().into(),
                }),
            );
        } else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "invalid project metadata key",
            ));
        }
        Ok(())
    })?;
    for project in state.projects.values() {
        let prefix = project_prefix(project.id);
        let started = Instant::now();
        let mut record_count = 0_u64;
        let mut record_bytes = 0_u64;
        records.scan_prefix(&prefix, |key, bytes| {
            record_count = record_count.saturating_add(1);
            record_bytes = record_bytes.saturating_add(bytes.len() as u64);
            let suffix = &key[prefix.len()..];
            if project.graph.restore_persistence_record(suffix, bytes)?
                || project.indexes.restore_persistence_record(suffix, bytes)?
                || project.temporal.restore_persistence_record(suffix, bytes)?
            {
                return Ok(());
            }
            Err(Error::new(
                ErrorCode::CorruptStorage,
                "unknown project persistence record",
            ))
        })?;
        tracing::info!(phase = "project_records", project = %project.id,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            records = record_count, bytes = record_bytes, "startup recovery phase completed");
        let started = Instant::now();
        project.graph.finish_persistence_restore()?;
        tracing::info!(phase = "graph_validation", project = %project.id,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0, "startup recovery phase completed");
        let started = Instant::now();
        project.indexes.finish_persistence_restore(&project.graph)?;
        tracing::info!(phase = "index_validation", project = %project.id,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0, "startup recovery phase completed");
        let started = Instant::now();
        project.temporal.finish_persistence_restore()?;
        tracing::info!(phase = "temporal_validation", project = %project.id,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0, "startup recovery phase completed");
    }
    records.scan_prefix(REQUESTS, |key, bytes| {
        let id = Uuid::from_slice(&key[REQUESTS.len()..])
            .map_err(|e| Error::invalid_data(e.to_string()))?;
        let record: RequestResultRecord = decode_database_value(bytes, "request result")?;
        state.request_result_bytes = state
            .request_result_bytes
            .checked_add(record.response.len() as u64)
            .ok_or_else(|| Error::invalid_data("request bytes overflow"))?;
        state.request_result_order.insert(record.index, id);
        state.request_results.insert(id, record);
        Ok(())
    })?;
    // Broker restores individual current records into its existing canonical cells.
    records.scan_prefix(b"b/", |key, bytes| {
        state.broker.restore_persistence_record(key, bytes)
    })?;
    state.broker.finish_persistence_restore()?;
    validate_durable_state(&state)?;
    Ok(state)
}
