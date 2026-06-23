from __future__ import annotations

from pathlib import Path

from stocknetv2.infrastructure.project_paths import ProjectPaths


def test_project_paths_resolve_project_local_data_layout(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(tmp_path)

    assert paths.root == tmp_path.resolve()
    assert paths.data_root == tmp_path.resolve() / "data"
    assert paths.ready_root == tmp_path.resolve() / "data" / "ready"
    assert paths.not_ready_root == tmp_path.resolve() / "data" / "not_ready"
    assert paths.other_root == tmp_path.resolve() / "data" / "other"
    assert paths.artifacts_root == tmp_path.resolve() / "data" / "other" / "artifacts"
    assert paths.market_db_path == tmp_path.resolve() / "data" / "other" / "stocknet_us.duckdb"
    assert paths.symbol_metadata_csv_path == tmp_path.resolve() / "data" / "other" / "artifacts" / "input_symbols.csv"
    assert paths.distributed_packs_root == tmp_path.resolve() / "data" / "ready"
    assert paths.distributed_runs_root == tmp_path.resolve() / "data" / "other" / "distributed_runs"
