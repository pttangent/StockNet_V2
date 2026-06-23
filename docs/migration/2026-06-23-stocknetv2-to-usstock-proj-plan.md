# USStock_Proj Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `D:\DEV\USStock_Proj` into the new StockNetV2 project root with project-local data paths and distributed month-pack graph-build entrypoints.

**Architecture:** Keep the current `stocknetv2` package and graph-build services, then layer a project-path resolver plus month-pack sources and scripts on top. This keeps existing code working while adding the distributed-pack workflow the migration plan requires.

**Tech Stack:** Python 3.11, DuckDB, pandas, pyarrow, Node test runner

---

### Task 1: Migrate the working tree and document history sources

**Files:**
- Create: `docs/migration/2026-06-23-stocknetv2-to-usstock-proj-design.md`
- Create: `docs/migration/2026-06-23-stocknetv2-to-usstock-proj-plan.md`

- [x] Move StockNetV2 working files into `D:\DEV\USStock_Proj` without `.git`
- [x] Move legacy `data` into `D:\DEV\USStock_Proj\data`
- [x] Move `input_symbols.csv` into `D:\DEV\USStock_Proj\data\artifacts\input_symbols.csv`
- [x] Record legacy source paths for later history lookups

### Task 2: Centralize project-local path resolution

**Files:**
- Create: `src/stocknetv2/infrastructure/project_paths.py`
- Modify: `scripts/*.py`
- Test: `tests/test_project_paths.py`

- [ ] Add a single resolver for the new project-local `data` layout
- [ ] Update scripts that default to old sibling `data` and `artifacts` paths
- [ ] Verify the new defaults point to `USStock_Proj/data`

### Task 3: Add month-pack repository support

**Files:**
- Modify: `src/stocknetv2/infrastructure/repositories/market_read_repository.py`
- Modify: `src/stocknetv2/interfaces/cli/run_theme_discovery_t1.py`
- Modify: `src/stocknetv2/application/services/graph_build_range_service.py`
- Test: `tests/test_market_read_repository.py`

- [ ] Add a `MonthPackSourceLayout`
- [ ] Teach graph-build entrypoints to run against month packs
- [ ] Keep legacy-layout behavior unchanged

### Task 4: Add distributed month-pack scripts

**Files:**
- Create: `scripts/build_month_pack.py`
- Create: `scripts/validate_month_pack.py`
- Create: `scripts/run_distributed_month.py`
- Test: `tests/test_month_pack_scripts.py`

- [ ] Build month packs under `data/distributed_packs/month=YYYY-MM`
- [ ] Validate pack structure and manifest coverage
- [ ] Run graph-build shards into `data/distributed_runs/run=*/month=*`

### Task 5: Verify end-to-end path and runtime behavior

**Files:**
- Test: `tests/test_graph_build_range_service.py`

- [ ] Run targeted pytest coverage for the new path and month-pack layers
- [ ] Run Node route tests if migration changed server-facing defaults
- [ ] Summarize validation results and remaining gaps
