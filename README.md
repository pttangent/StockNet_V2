# StockNetV2

`StockNetV2` is the new backend-first T1 mainline for theme discovery quality research.

## Purpose

- consume the existing `stocknet-research` data estate
- produce 5-minute theme-discovery snapshots
- persist graph, community, theme, semantic, lifecycle, and audit outputs into DuckDB
- power a read-only observer through Node query endpoints

## Runtime model

- Python: offline batch execution only
- DuckDB: system-of-record for T1 outputs
- Node: only online service
- Frontend: read-only observer

## Status

This project is being bootstrapped from the approved T1 design on 2026-06-15.

Current formal research baseline:

- [evaluation_pack_2025_01_v2](/D:/DEV/stocknetwork/StockNet/StockNetV2/evaluation_pack_2025_01_v2) is the validated problem baseline artifact from commit `600b23d`.
- It evaluates graph-build outputs from `cb4e44b72f22a1079d2745a1ff78e3a9c520af47`, not every newer candidate code change on the branch.
- The active remediation roadmap is documented in [GRAPH_QUALITY_AND_FINANCIAL_MEANING_PLAN.md](/D:/DEV/stocknetwork/StockNet/StockNetV2/GRAPH_QUALITY_AND_FINANCIAL_MEANING_PLAN.md).
