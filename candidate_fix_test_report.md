# Candidate Fix Test Report

Generated at: `2026-06-19T19:21:20+08:00`

Candidate commit under test: `6432754b2b0532c6e412c913b65c00a490fc245b`

Formal baseline reminder:

- Formal January evaluation pack artifact: `600b23d`
- Graph-build commit under formal evaluation: `cb4e44b72f22a1079d2745a1ff78e3a9c520af47`
- The current branch contains candidate fixes and must still be treated as `candidate fix / unvalidated`

## Why Graph Rebuild Was Not Started Yet

The current branch has candidate graph-quality fixes, but Phase A regression screening is not yet green.

Under the validation roadmap, candidate code must pass:

1. local regression tests
2. targeted graph-layer tests
3. qualification-run readiness checks

before any six-day or monthly rebuild is treated as a meaningful validation run.

At the time this report was generated, that gate was not satisfied.

## Commands Run

Python graph-quality regression subset:

```bash
python -m pytest D:\DEV\stocknetwork\StockNet\StockNetV2\tests\test_dtw_similarity.py D:\DEV\stocknetwork\StockNet\StockNetV2\tests\test_layer_execution_service.py D:\DEV\stocknetwork\StockNet\StockNetV2\tests\test_remaining_graph_layers.py D:\DEV\stocknetwork\StockNet\StockNetV2\tests\test_community_and_consensus.py -q
```

Node route regression:

```bash
npm test
```

## Current Result

### Python graph-layer regression

Status: `failed`

Summary:

- `22 passed`
- `1 failed`

Failing test:

- `StockNetV2/tests/test_layer_execution_service.py::test_layer_execution_service_builds_all_six_layer_outputs`

Observed failure:

- `large_trade_alignment_graph` returned `0` edges in the fixture, while the test still expects at least one edge.

This means the current candidate branch has not yet restored a stable graph-layer regression baseline.

### Node regression

Status: `passed`

Summary:

- `server exposes read-only T1 run and snapshot routes`
- `1 passed`
- `0 failed`

## Immediate Interpretation

- The current branch has meaningful candidate graph fixes in code, but it is not yet cleared for rebuild-based validation.
- A graph rerun performed before fixing this regression would mix algorithm changes with known failing local expectations.
- The correct next step is to resolve the `large_trade_alignment_graph` regression, then rerun the targeted graph-layer suite, and only then proceed to six-day qualification.

## Next Required Action

1. Inspect why `large_trade_alignment_graph` now returns zero edges in the integration fixture.
2. Decide whether the fixture is outdated or the candidate implementation introduced a real regression.
3. Restore a green graph-layer regression gate.
4. Start the six-day qualification runner only after that gate is green.
