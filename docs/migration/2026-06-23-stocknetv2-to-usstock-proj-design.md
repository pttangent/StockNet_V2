# USStock_Proj Migration Design

**Date:** 2026-06-23

**Goal:** Move the StockNetV2 working project, market data estate, and symbol metadata into `D:\DEV\USStock_Proj` without git history, while keeping old source paths documented for later historical lookup.

## History Sources

- Code history source: `D:\DEV\stocknetwork\StockNet\StockNetV2`
- Data history source: `D:\DEV\stocknetwork\StockNet\data`
- Symbol metadata history source: `D:\DEV\stocknetwork\StockNet\artifacts\input_symbols.csv`

## Target Layout

```text
USStock_Proj/
  src/
  scripts/
  tests/
  data/
    raw_1m/
    trade_flow_1m/
    bars_5m/
    labels_1m/
    stocknet_us.duckdb
    artifacts/
      input_symbols.csv
    distributed_packs/
    distributed_runs/
```

## Required Changes

1. Move the StockNetV2 working tree into `USStock_Proj`, excluding `.git`.
2. Move the market `data` directory into `USStock_Proj/data`.
3. Move `input_symbols.csv` into `USStock_Proj/data/artifacts/input_symbols.csv`.
4. Replace scripts that still default to the old sibling `data` and `artifacts` paths.
5. Add month-pack build/validate/run entrypoints so distributed graph builds can work from the new project-local data layout.

## Compatibility Strategy

- The project keeps the existing `stocknetv2` package name and runtime behavior.
- New path resolution is centralized so future scripts stop hard-coding legacy sibling paths.
- Distributed month packs preserve current string-symbol compatibility while adding `symbol_id` columns and manifests required by the migration plan.
