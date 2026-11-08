//! Offline primitive qualification. Experimental codecs are never production strategies.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "release filtered recall, cold cost and total-resident codec comparison"]
fn retrieval_cost_scorecard() -> Result<()> {
    let owners: u64 = std::env::var("IG_RETRIEVAL_SCORECARD_OWNERS")
        .unwrap_or_else(|_| "10000".into())
        .parse()
        .map_err(|e| Error::invalid_data(format!("owners: {e}")))?;
    assert!(owners >= 1000);
    let dimensions = 384;
    let index = VectorIndex::new(dimensions, Similarity::Cosine)?;
    for id in 0..owners {
        let vector: Vec<_> = (0..dimensions)
            .map(|d| {
                ((mix(id.wrapping_mul(719).wrapping_add(d as u64)) % 2001) as f32 - 1000.0) / 1000.0
            })
            .collect();
        index.upsert(id, &vector, 1)?;
    }
    let queries: Vec<_> = (0..20)
        .map(|at| {
            index
                .vector_for(at * (owners / 20))
                .ok_or_else(|| Error::internal("query fixture absent"))
        })
        .collect::<Result<_>>()?;
    let canonical_bytes = index.resident_bytes();
    // Experimental symmetric four-bit scalar codec. No runtime dependency/API/default change.
    // Read one canonical vector transiently; retain only packed codes for this experiment.
    {
        let build = Instant::now();
        let step = 0.5 / (dimensions as f32).sqrt();
        let mut packed = Vec::with_capacity(owners as usize * dimensions / 2);
        for id in 0..owners {
            let vector = index
                .vector_for(id)
                .ok_or_else(|| Error::internal("codec fixture absent"))?;
            for pair in vector.chunks_exact(2) {
                let code = |value: f32| ((value / step).round().clamp(-7.0, 8.0) as i8 + 7) as u8;
                packed.push(code(pair[0]) | (code(pair[1]) << 4));
            }
        }
        println!(
            "{{\"event\":\"codec_build\",\"method\":\"experimental_scalar4\",\"build_ms\":{},\"canonical_bytes\":{canonical_bytes},\"total_derived_and_canonical_bytes\":{},\"production\":false}}",
            build.elapsed().as_secs_f64() * 1000.0,
            canonical_bytes + packed.capacity()
        );
        let mut times = Vec::new();
        let mut matched = 0;
        let cancellation = CancellationToken::new();
        for query in &queries {
            let truth = index.exact_search(query, 10)?;
            let started = Instant::now();
            let mut candidates = Vec::new();
            for (id, codes) in packed.chunks_exact(dimensions / 2).enumerate() {
                let score = codes
                    .iter()
                    .zip(query.chunks_exact(2))
                    .map(|(bits, q)| {
                        let a = (bits & 15) as i8 - 7;
                        let b = (bits >> 4) as i8 - 7;
                        step * (q[0] * f32::from(a) + q[1] * f32::from(b))
                    })
                    .sum();
                push_hit(
                    &mut candidates,
                    VectorHit {
                        entity_id: id as u64,
                        score,
                    },
                    512,
                    Similarity::Cosine,
                );
            }
            let mut results = Vec::new();
            for hit in candidates {
                if let Some(score) =
                    index.score_owner_cancellable(query, hit.entity_id, &cancellation)?
                {
                    push_hit(
                        &mut results,
                        VectorHit {
                            entity_id: hit.entity_id,
                            score,
                        },
                        10,
                        Similarity::Cosine,
                    );
                }
            }
            times.push(started.elapsed().as_secs_f64() * 1000.0);
            matched += results
                .iter()
                .filter(|hit| truth.iter().any(|t| t.entity_id == hit.entity_id))
                .count();
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{{\"event\":\"codec_queries\",\"method\":\"experimental_scalar4\",\"queries\":{},\"p50_ms\":{},\"p95_ms\":{},\"recall10\":{},\"production\":false}}",
            queries.len(),
            times[times.len() / 2],
            times[(times.len() * 95).div_ceil(100) - 1],
            matched as f64 / (queries.len() * 10) as f64
        );
    }
    // The experimental codes are now dropped; production-code totals exclude that experiment.
    for share in [1_u64, 10, 100, 1000] {
        let visible = |id: u64| id % 1000 < share;
        let truth: Vec<_> = queries
            .iter()
            .map(|q| index.search_retrieval(q, 10, RetrievalStrategy::Exact, visible))
            .collect::<Result<_>>()?;
        for (name, strategy) in [
            ("exact", RetrievalStrategy::Exact),
            (
                "prefix128",
                RetrievalStrategy::Prefix {
                    dimensions: 128,
                    candidates: 512,
                },
            ),
            (
                "quantized",
                RetrievalStrategy::Quantized { candidates: 512 },
            ),
            ("adaptive", RetrievalStrategy::Adaptive { candidates: 512 }),
        ] {
            let cold = Instant::now();
            index.search_retrieval(&queries[0], 10, strategy, visible)?;
            let first_ms = cold.elapsed().as_secs_f64() * 1000.0;
            let mut times = Vec::new();
            let mut matched = 0;
            let mut returned = 0;
            for (query, ground) in queries.iter().zip(&truth) {
                let started = Instant::now();
                let hits = index.search_retrieval(query, 10, strategy, visible)?;
                times.push(started.elapsed().as_secs_f64() * 1000.0);
                assert!(
                    hits.iter()
                        .all(|hit| visible(hit.entity_id) && hit.score.is_finite())
                );
                assert_eq!(hits.len(), ground.len());
                assert_eq!(
                    hits.iter()
                        .map(|h| h.entity_id)
                        .collect::<BTreeSet<_>>()
                        .len(),
                    hits.len()
                );
                matched += hits
                    .iter()
                    .filter(|hit| ground.iter().any(|g| g.entity_id == hit.entity_id))
                    .count();
                returned += hits.len();
            }
            times.sort_by(f64::total_cmp);
            println!(
                "{{\"event\":\"filtered_retrieval\",\"owners\":{owners},\"dimensions\":{dimensions},\"matching_per_thousand\":{share},\"method\":\"{name}\",\"queries\":{},\"first_query_ms\":{first_ms},\"p50_ms\":{},\"p95_ms\":{},\"recall10\":{},\"retained_bytes\":{},\"boundary\":\"primitive CPU retrieval, synthetic untrained-prefix corpus\"}}",
                queries.len(),
                times[times.len() / 2],
                times[(times.len() * 95).div_ceil(100) - 1],
                matched as f64 / returned.max(1) as f64,
                index.resident_bytes()
            );
        }
    }
    println!("RETRIEVAL_COST_SCORECARD_OK");
    Ok(())
}
