# StockNetV2 Graph Quality And Financial Meaning Plan

## 0. Project State And Version Baseline

### 0.1 Current formal research baseline

- Formal monthly evaluation pack: [evaluation_pack_2025_01_v2](/D:/DEV/stocknetwork/StockNet/StockNetV2/evaluation_pack_2025_01_v2)
- Formal artifact commit: `600b23d`
- Graph-build commit under evaluation: `cb4e44b72f22a1079d2745a1ff78e3a9c520af47`
- Evaluation-pack generator commit: `a14f4fd715088b809d763ab8b4cf2ef704bff740`

Baseline data scope:

- `2025-01-02` to `2025-01-31`
- 20 trade dates
- 1,560 snapshots
- 14,793,303 edges
- 149,601 layer communities
- 10,247,185 community memberships
- 6,379,952 active symbol-snapshot feature rows
- 6,379,952 forward-label rows

### 0.2 Required milestone wording

The current milestone must be labeled:

`Graph Evaluation Infrastructure Completed`

It must not be labeled:

- `Graph Quality Validated`
- `Theme Discovery Validated`
- `Financial Alpha Validated`

### 0.3 Relationship between latest code and formal data

The current branch already contains candidate implementation changes in code, but the formal January pack does not validate them.

Therefore:

`latest branch code != latest validated graph result`

Every candidate fix must pass:

1. Unit tests
2. Small integration tests
3. Six-day qualification run
4. Old-vs-new structural comparison
5. Financial-meaning validation
6. January rebuild
7. February out-of-sample validation

Only then can a research issue be treated as resolved.

## 1. Overall Goal

The goal is not merely to reduce edge counts or shrink communities.

The target graph system must satisfy all of the following:

1. All inputs are truly available at snapshot time and contain no future data.
2. Every edge has interpretable statistical support.
3. Edge weights distinguish weak and strong relationships without widespread saturation.
4. Communities are locally cohesive and externally separated, not giant sparse components created by bridge edges.
5. The six layers provide complementary evidence rather than duplicated scoring.
6. Communities have verifiable industry, event, flow, or price-resonance meaning.
7. Community quality exceeds matched controls and null models.
8. Edge weight and community quality calibrate sensibly to future resonance, persistence, or excess return.
9. Parameters, methods, dependencies, inputs, and artifacts are fully traceable.

## 2. Confirmed Completed Foundations

### 2.1 Evaluation-pack artifact structure

The following structure is already the correct foundation and should be retained:

- `graph/all_edges/`
- `graph/snapshot_layer_diagnostics.csv`
- `graph/node_layer_metrics/`
- `graph/community_metrics.parquet`
- `graph/community_membership.parquet`
- `graph/layer_review_candidates.csv`
- `market/symbol_snapshot_features/`
- `market/symbol_forward_labels/`
- `market/symbol_master.csv`
- `market/benchmark_series/`
- `compare_old_vs_new/`
- `run_manifest.json`
- `README.md`
- `ASSESSMENT.md`

### 2.2 Provenance infrastructure

Already completed:

- graph-build commit capture
- generator commit capture
- config hash
- input database hashes
- metadata CSV hash
- dependency versions
- dirty paths
- relevant dirty state
- manifest self-size
- generated README quality-gate wording

This should be extended, not rebuilt.

### 2.3 Early graph-filtering direction

The following direction remains correct and should stay in place pending validation:

- `candidate_top_k = 8`
- `reciprocal_top_k = 3`
- `degree_cap = 6`

Flow-alignment direction also correctly moved toward:

- `min_joint_active_points`
- `activity_epsilon`
- cross-sectional residualization

## 3. P0 Blocking Issues

| ID | Issue | Current status | Risk | Requires graph rebuild |
| --- | --- | --- | --- | --- |
| P0-01 | 1m/5m availability-time contract | Not complete | Look-ahead bias | Yes |
| P0-02 | Forward-label starting point | Not complete | Evaluation contamination | Labels at minimum |
| P0-03 | DTW shared-timestamp alignment and support | Candidate only | Weight distortion | Yes |
| P0-04 | DTW distance normalization and calibration | Candidate only | Weight saturation | Yes |
| P0-05 | Weighted-Leiden runtime guarantee | Candidate only | Community semantics wrong | Yes |
| P0-06 | Return-corr full-day accumulation and premarket bleed | Candidate only | Giant market-mode clusters | Yes |
| P0-07 | `UNKNOWN` metadata pollutes concentration | Not fixed | Semantic review invalid | Pack rebuild |
| P0-08 | Market-mode denominator uses active nodes | Not fixed | Sparse-layer false positives | Pack rebuild |
| P0-09 | Evaluation features are not actual graph inputs | Not fixed | Edge causes cannot be explained | Pack rebuild |
| P0-10 | Only active symbols exported | Not fixed | Selection bias | Pack rebuild |
| P0-11 | Graph summary parameters remain null | Not fixed | Parameters not traceable | New graph runs |
| P0-12 | `member_weight` fixed at 1 | Not fixed | Core vs peripheral lost | Recompute or new run |
| P0-13 | Communities lack external metrics | Not fixed | Isolation cannot be judged | Post-process okay |
| P0-14 | Consensus can still absorb low-quality layers | Not fixed | False themes spread | Re-enrichment at minimum |
| P0-15 | No null models or controls | Not fixed | No proof of superiority | Research layer |

## 4. Time-Causality Contract

### 4.1 Unified time fields

All minute-level data must define:

- `event_time`
- `available_time`
- `snapshot_time`

Rule:

`available_time <= snapshot_time`

for every graph input row used by a snapshot.

### 4.2 1m data rule

If raw 1m timestamps represent bucket start, then:

- `09:35` means `09:35:00` to `09:35:59`
- `available_time = 09:36:00`

Therefore a `09:35` snapshot must not read the `09:35` 1m bucket.

### 4.3 5m data rule

If 5m bars are right-labeled:

- `timestamp = 09:35` means the `09:30-09:35` bar is complete
- `available_time = 09:35`

That bar may be used by a `09:35` snapshot.

### 4.4 Forward-label rule

Do not join labels by `timestamp == snapshot_time`.

Instead define:

- `label_start_price`: last completed price with `available_time <= snapshot_time`
- `label_end_price`: first completed price after the horizon ends

### 4.5 Required tests

- `test_1m_bucket_not_available_at_bucket_start`
- `test_5m_right_labeled_bar_available_at_bar_end`
- `test_snapshot_0935_cannot_read_0935_0936_bucket`
- `test_forward_label_starts_from_last_available_price`
- `test_no_feature_available_time_after_snapshot_time`

### 4.6 Acceptance conditions

- `lookahead_violation_count = 0`
- `label_start_mismatch_count = 0`
- `feature_available_time_coverage = 100%`

## 5. DTW Return

### 5.1 Baseline issue

The January pack shows:

- `weight_p50 ~= 1`
- `weight_p90 ~= 1`
- support frequently near `1` to `2`

So the current baseline weights do not carry useful ranking information.

### 5.2 Correct pair construction

Each pair must align on shared timestamps first:

```python
paired = pd.concat(
    [left_series.rename("left"), right_series.rename("right")],
    axis=1,
).dropna()
```

Never use independent `dropna()` lists per symbol.

### 5.3 Minimum support

Recommended:

- `min_overlap_points = 8`
- preferred full-window support `>= 12`

Reject the edge if support is below threshold.

### 5.4 Variance gate

- `left_std < 1e-8 -> reject`
- `right_std < 1e-8 -> reject`

### 5.5 Distance normalization

Use:

- `normalized_dtw_distance = accumulated_dtw_distance / warping_path_length`
- `similarity = 1 / (1 + normalized_dtw_distance)`

But recalibrate thresholds from real distributions rather than assuming `0.9` is final.

### 5.6 Candidate prescreen

Prescreening must apply:

- minimum overlap
- minimum variance
- constant-series rejection

before top-k candidate selection.

### 5.7 Diagnostics

Per snapshot:

- `candidate_pair_count`
- `aligned_pair_count`
- `rejected_low_overlap_count`
- `rejected_low_variance_count`
- `support_points_p10/p50/p90`
- `weight_p10/p50/p90/p99`
- `exact_weight_1_ratio`

### 5.8 Acceptance

- `support_points_p50 >= 8`
- `exact_weight_1_ratio < 5%`
- `weight_p50 < weight_p90 < weight_p99`
- no low-variance edge output
- no zero-overlap edge output

## 6. DTW Trade Flow

### 6.1 Align three components separately

Components:

- `flow_impulse_score`
- `imbalance_z`
- `large_trade_ratio_z`

Each must align and validate on shared timestamps.

### 6.2 Validity gate

Suggested:

- at least 2 valid components
- each valid component support `>= 8`

### 6.3 Composite score

Base weights may begin as:

- flow `0.50`
- imbalance `0.30`
- large trade `0.20`

If a component is invalid, re-normalize across valid components only, while recording:

- `valid_component_count`
- `component_support_min`
- `component_support_mean`

### 6.4 Financial meaning

This layer may describe:

- flow-activity alignment
- imbalance alignment
- large-trade participation alignment

It does not justify claims like institutional net buying without richer microstructure data.

### 6.5 Acceptance

- same DTW acceptance rules as DTW Return
- `valid_component_count_p50 >= 2`
- `single_component_edge_ratio = 0`

## 7. Return Correlation

### 7.1 Baseline issue

The formal January pack shows:

- `5,230,407` edges
- max community `3,470`
- early-session max communities covering about `98%` to `99%` of active nodes

This is the main giant-cluster failure mode.

### 7.2 Fixed regular-session window

Must use:

- regular session only
- fixed rolling bars

Grid:

- `lookback_bars: 12 / 18 / 24`
- `min_correlation: 0.65 / 0.70 / 0.75 / 0.80`
- `min_overlap: 8 / 10 / 12`

Do not treat current candidate defaults as final.

### 7.3 Premarket separation

Use separate layers or modes:

- `premarket_return_corr_graph`
- `regular_return_corr_graph`

### 7.4 Residual return variants

Minimum version:

- stock return minus cross-sectional median return

Later variants:

- SPY beta residual
- sector ETF residual
- market-cap bucket residual

### 7.5 Graph-structure guardrails

Degree cap alone is not enough.

Need:

- weighted Leiden
- resolution grid
- market-mode identification
- component diagnostics

### 7.6 Acceptance

- largest non-market-mode community below about `10%-15%` of eligible universe
- connected components may still exist, but Leiden communities must not collapse into one market blob
- edge counts should vary with day structure, not be mechanically pinned by top-k alone

## 8. Flow Alignment

### 8.1 Confirmed progress

Formal January results show edge count contraction from:

- `20,825,310` to `8,449,084`

which confirms that active gating, reciprocal top-k, degree cap, and residualization helped.

### 8.2 Remaining issue

The layer still produces giant clusters:

- `community_size_p95 ~= 3,587`
- `community_size_max ~= 3,774`

### 8.3 Variance gate

Require:

- `min_variance > 0`

and export:

- `constant_series_count`
- `invalid_variance_pair_count`
- `joint_active_points distribution`
- `active_symbol_count`

### 8.4 Shared factor reduction

Retain cross-sectional median residualization and test:

- sector-neutral flow
- liquidity-bucket-neutral flow

### 8.5 Edge decomposition

Per edge, preserve:

- `correlation_component`
- `same_direction_component`
- `joint_active_points`
- `left_active_ratio`
- `right_active_ratio`

### 8.6 Acceptance

- `joint_active_points_p50 >= 8`
- `constant-series edges = 0`
- largest non-market-mode community below about `15%` of eligible universe
- overlap with DTW-flow edges must not be near-complete duplication

## 9. Volume Expansion

### 9.1 Current status

This is one of the most improved layers structurally, but early-session support remains weak.

### 9.2 Event-support logic

Add:

- `min_co_expansion_events >= 3`
- `min_support_points`
- `event_jaccard`

Suggested score family:

`continuous similarity + co-expansion event Jaccard + event-timing similarity`

### 9.3 Intraday seasonality

Standardize by minute-of-session history rather than only same-day rolling behavior.

### 9.4 Acceptance

- median shared expansion events `>= 3`
- edge-weight saturation materially reduced
- opening auction effects standardized without eliminating local communities

## 10. Large Trade

### 10.1 Current status

The formal January layer is too sparse to judge quality from counts alone.

### 10.2 Feature-scale validation

Need diagnostics for:

- raw `large_trade_ratio`
- `large_trade_ratio_z`
- non-null coverage
- non-zero coverage
- threshold pass rate

### 10.3 Event-style graph

Suggested event definition:

- `ratio_z >= 2`
- or per-symbol historical minute-of-session percentile `>= 98`

Pair metrics:

- event Jaccard
- shared event count
- event lag
- post-event direction consistency

Minimum shared-event count: `>= 3`

### 10.4 Language discipline

This may be called `large-trade participation alignment`.

Do not call it institutional buying without stronger market microstructure evidence.

## 11. Community Detection

### 11.1 Weighted-Leiden dependencies

Formal dependencies must include:

- `python-igraph`
- `leidenalg`

If config requests `weighted_leiden`, missing dependencies must fail the run.

### 11.2 Method recording

Store per layer and snapshot:

- `requested_community_method`
- `actual_community_method`
- `resolution`
- `fallback_used`
- `fallback_reason`

Fallback count must be zero in formal research runs.

### 11.3 Resolution grid

Test at least:

- `0.5`
- `0.75`
- `0.9`
- `1.1`
- `1.3`
- `1.5`

Choose based on structure health, not just community count.

### 11.4 Role of connected components

Keep connected components only for:

- percolation diagnostics
- giant-component diagnostics

not as the final theme community definition.

## 12. Market Mode

### 12.1 Correct denominator

Market-mode judgments must use:

- `eligible_universe_symbol_count`

not only active nodes in the current layer.

### 12.2 Minimum scale gate

Only judge market mode when:

- `eligible_universe >= 100`
- `active_nodes >= 50`

### 12.3 Dual ratios

Persist both:

- `community_ratio_of_active_nodes`
- `community_ratio_of_eligible_universe`

Use eligible-universe ratio as the final market-mode decision metric.

## 13. Symbol Metadata

### 13.1 `UNKNOWN` exclusion

Concentration must be computed on known metadata only.

### 13.2 Coverage report

Add:

- `metadata_coverage_report.csv`

with coverage for company name, sector, industry, market cap, security type, exchange, and country.

### 13.3 Symbol-master expansion

At minimum add:

- `security_type`
- `is_etf`
- `is_adr`
- `is_warrant`
- `is_unit`
- `is_preferred`
- `is_spac`
- `exchange`
- `country`
- `sector`
- `industry`
- `subindustry`
- `market_cap`
- `float`

### 13.4 Acceptance

- sector coverage `>= 90%-95%`
- industry coverage `>= 85%-90%`
- market-cap coverage `>= 90%`
- `UNKNOWN` must never trigger concentration flags

## 14. Evaluation-Pack Input Consistency

### 14.1 Export two feature families

Add:

- `market/graph_input_features_1m/`
- `market/review_features_derived/`

Graph-input export must mirror the real graph features.

Review features may remain as analyst-friendly derived tables but must not pretend to explain actual edge generation.

### 14.2 Consistency report

Add:

- `graph_input_consistency_report.csv`

covering row counts, missingness, distributions, and version/hashing identity.

## 15. Negative Samples And Selection Bias

### 15.1 Current issue

Current evaluation exports focus on graph-active symbols, so they cannot answer whether in-graph symbols are better than contemporaneous non-members.

### 15.2 Eligible-universe export

Per snapshot export:

- `snapshot_id`
- `symbol`
- `is_eligible`
- `is_graph_active`
- `degree_by_layer`
- `community_id_by_layer`

### 15.3 Matched controls

For each community, match controls by:

- same snapshot
- similar market cap
- similar dollar volume
- similar realized volatility
- similar same-day return
- similar security type
- not belonging to the community

### 15.4 Outputs

- `community_matched_controls.parquet`
- `matched_control_balance_report.csv`

## 16. Community Metrics

### 16.1 Keep current internal metrics

Retain:

- member count
- edge count
- density
- average/min/max weight
- degree summary

### 16.2 Add external structure metrics

- `external_edge_count`
- `external_weight_sum`
- `internal_weight_sum`
- `conductance`
- `cut_ratio`
- `expansion`
- `coverage`
- `modularity_contribution`

### 16.3 Hub dependency

Add:

- `top1_member_internal_degree_share`
- `top5_member_internal_degree_share`
- `bridge_member_count`
- `hub_dependency_score`

### 16.4 Member roles

Recompute roles such as:

- core
- peripheral
- bridge
- hub
- leader
- follower

`member_weight` must not remain uniformly equal to `1`.

## 17. Temporal Stability And Lifecycle

Add:

- `community_temporal_matches.parquet`
- `community_lifecycle_summary.parquet`

Track:

- `member_jaccard`
- `overlap_small`
- `core_member_retention`
- `weighted_overlap`
- `member_churn`

Lifecycle events:

- birth
- continuation
- split
- merge
- death
- revival

Distinguish:

- active market minutes
- wall-clock minutes
- session count

## 18. Cross-Layer Independence

Add:

- `cross_layer_overlap.csv`

Per snapshot and layer pair:

- `active_node_jaccard`
- `edge_jaccard`
- `weighted_edge_similarity`
- community co-assignment similarity
- adjusted Rand index
- normalized mutual information

If related layers are near-duplicates, consensus must not count them as independent evidence.

## 19. Consensus

### 19.1 Family principle

Keep the three evidence families:

- Price
- Flow
- Activity

Formal themes should require at least two different families.

### 19.2 Paused layers

Until qualification passes, keep these out of formal consensus:

- DTW Return
- DTW Flow
- Return Corr

### 19.3 Layer eligibility gate

A layer may contribute only if it passes:

- actual method is weighted Leiden
- edge weights valid
- support valid
- no pathological giant-mode behavior
- metadata-independent structure valid

### 19.4 Large-community discounting

Large communities must be penalized rather than allowed to spread equal support across all member pairs.

## 20. Financial-Meaning Outputs

### 20.1 Edge predictive calibration

Add:

- `edge_predictive_calibration.csv`

by layer, weight bucket, and minute-of-session:

- future same-direction rate
- future return correlation
- future excess-return correlation
- future flow coherence

### 20.2 Community financial metrics

Add:

- `community_financial_metrics.parquet`

with:

- equal-weight future return
- market-cap-weighted future return
- SPY excess return
- sector excess return
- positive-return breadth
- return dispersion
- flow breadth
- volume breadth
- large-trade breadth
- matched-control excess

### 20.3 Lead-lag

Evaluate:

- cross-correlation lag
- DTW optimal lag
- leader persistence
- follower persistence
- lagged predictive coefficient

## 21. Null Models

Add null-model infrastructure for:

- degree-preserving rewiring
- symbol permutation
- time shuffling

Outputs:

- `null_model_results.parquet`
- `null_model_significance_summary.csv`

## 22. Diagnostics Expansion

Per snapshot and layer export at least:

- `eligible_universe_count`
- `active_node_count`
- `active_node_ratio`
- `isolated_node_count`
- `edge_count`
- `average_degree`
- `degree_p50/p90/p95/p99`
- `max_degree`
- `hub_top1_share`
- `hub_top10_share`
- `weight_p01/p10/p25/p50/p75/p90/p95/p99`
- `exact_weight_1_ratio`
- `support_p10/p50/p90`
- `low_support_ratio`
- `connected_component_count`
- `largest_component_ratio_active`
- `largest_component_ratio_universe`
- `community_count`
- `community_size_p50/p95/max`
- `modularity`
- `conductance_p50`
- `actual_method`
- `resolution`

## 23. Database And Provenance Enhancements

### 23.1 Graph-edge summary

Persist real values for:

- `threshold`
- `candidate_top_k`
- `reciprocal_top_k`
- `degree_cap`
- `min_overlap`
- `min_variance`
- `lookback`

### 23.2 Community method

Persist:

- `requested_method`
- `actual_method`
- `resolution`
- `fallback_count`

### 23.3 Artifact integrity

Per artifact store:

- path
- size
- row count
- sha256
- schema hash

Directory-style artifacts should also emit `artifact_inventory.csv`.

### 23.4 Version relations

Manifest should continue to separate:

- `graph_build_commit`
- `evaluation_generator_commit`
- `input_database_hash`
- `config_hash`
- `dependency_lock_hash`

## 24. Test Plan

### 24.1 Unit tests

Need unit coverage for:

- time availability
- forward-label alignment
- timezone and session boundaries
- DTW shared-timestamp alignment
- no-overlap rejection
- low-variance rejection
- support computation
- path normalization
- weight saturation
- Leiden dependency failure
- actual-method recording
- toy graph clustering
- `UNKNOWN` exclusion
- coverage calculation
- market-mode denominator behavior

### 24.2 Integration tests

Build a semantic fixture with:

- two clear industry clusters
- one market factor
- one low-liquidity noise cluster
- one constant-series invalid group

### 24.3 Six-day qualification

Use:

- three high-activity days
- three ordinary days

and not only favorable examples.

## 25. Six-Day Qualification Gates

### Data correctness

- look-ahead violations `= 0`
- label-alignment mismatch `= 0`
- graph-input export coverage `>= 99%`

### Metadata

- sector coverage `>= 90%-95%`
- market-cap coverage `>= 90%`
- `UNKNOWN` concentration count `= 0`

### DTW

- support `p50 >= 8`
- exact weight `= 1` ratio `< 5%`
- weight quantiles clearly separated

### Community structure

- `actual_method = weighted_leiden` for all intended layers
- `fallback_count = 0`
- largest non-market-mode community below about `10%-15%` of universe

### Financial signal

- higher-weight edges outperform lower-weight edges on future coherence
- communities improve relative to matched controls
- direction of results is consistent across most qualification days

If any blocking gate fails, do not rerun the full month yet.

## 26. Execution Phases

### Phase A: version freeze and regression recovery

Goal:

- select a candidate-fix commit
- install formal dependencies
- run all Python and Node tests
- capture failures
- keep January v2 pack untouched
- do not claim the new algorithm is already effective

Deliverable:

- `candidate_fix_test_report.md`

### Phase B: time and algorithm correctness

Focus:

- availability contract
- forward labels
- DTW
- return correlation
- Leiden
- constant-series handling
- summary-parameter persistence

Deliverable:

- `graph_algorithm_v3_candidate`

### Phase C: evaluation-pack v3 schema

Focus:

- metadata coverage
- `UNKNOWN` exclusion
- eligible universe
- graph-input features
- inactive symbols
- matched controls
- community external metrics

Deliverable:

- `evaluation_pack_schema_v3`

### Phase D: six-day qualification run

Deliverable:

- `qualification_pack_6d_v1`

including old-vs-new comparisons.

### Phase E: parameter grid

Only after correctness is restored:

- return-corr window x threshold
- Leiden resolution
- DTW threshold
- flow activity threshold
- volume-event threshold

Choose parameters on structural health first, then inspect financial calibration.

### Phase F: rerun January

After qualification passes:

- `evaluation_pack_2025_01_v3`

Keep it alongside v2, never overwrite v2.

### Phase G: February out-of-sample

Freeze January-selected parameters and run:

- `evaluation_pack_2025_02_oos_v1`

Do not retune to fit February.

### Phase H: theme discovery

Only after January and February both pass:

- consensus theme
- semantic naming
- lifecycle
- theme quality

### Phase I: backtest and TGNN

Only after the above:

- edge predictive model
- community predictive model
- lead-lag strategy
- TGNN

## 27. Suggested Commit Splits

Do not collapse the remediation into one giant commit.

Suggested split:

1. `Formalize market-data availability timestamps`
2. `Correct DTW alignment and support semantics`
3. `Enforce weighted Leiden runtime contract`
4. `Use fixed regular-session return-correlation windows`
5. `Persist graph parameters and actual community method`
6. `Correct metadata coverage and UNKNOWN concentration`
7. `Export graph inputs and eligible-universe controls`
8. `Add community external and temporal metrics`
9. `Add financial calibration and null-model outputs`
10. `Add six-day qualification runner`

Each commit should carry its own tests and should not mix large binary packs with core algorithm changes.

## 28. Final Approval Conditions

Full theme discovery should be approved only after all of the following are proven:

- time-causality contract passes
- DTW weights pass calibration gates
- weighted Leiden actually executes in 100% of intended runs
- return-corr giant clusters are controlled
- metadata coverage is above threshold
- `UNKNOWN` does not pollute semantics
- community external isolation is healthy
- temporal stability is healthy
- matched controls are beaten
- null-model significance passes
- January and February behave consistently out of sample

Before that point:

- the evaluation pack may be used for diagnosis
- the graph should not be used for formal theme claims
- themes should not be used for backtest conclusions
- TGNN training should not start

## Final Conclusion

`evaluation_pack_2025_01_v2` remains a valid and valuable baseline.

It is a problem baseline, not a validation certificate for the latest candidate code.

The correct next route is:

candidate-code freeze  
-> full regression tests  
-> evaluation-pack bias fixes  
-> six-day qualification  
-> parameter grid  
-> January v3  
-> February out-of-sample  
-> theme discovery

Current post-`600b23d` code changes should be uniformly labeled `candidate fix / unvalidated` until they complete that route.
