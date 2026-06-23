from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path
    ready_root: Path
    not_ready_root: Path
    other_root: Path
    artifacts_root: Path
    metadata_root: Path
    distributed_packs_root: Path
    distributed_runs_root: Path
    market_db_path: Path
    symbol_metadata_csv_path: Path

    @classmethod
    def discover(cls, root: Path | str | None = None) -> "ProjectPaths":
        resolved_root = (
            Path(root).expanduser().resolve()
            if root is not None
            else Path(__file__).resolve().parents[3]
        )
        data_root = resolved_root / "data"
        ready_root = data_root / "ready"
        not_ready_root = data_root / "not_ready"
        other_root = data_root / "other"
        return cls(
            root=resolved_root,
            data_root=data_root,
            ready_root=ready_root,
            not_ready_root=not_ready_root,
            other_root=other_root,
            artifacts_root=other_root / "artifacts",
            metadata_root=resolved_root / "metadata",
            distributed_packs_root=ready_root,
            distributed_runs_root=other_root / "distributed_runs",
            market_db_path=other_root / "stocknet_us.duckdb",
            symbol_metadata_csv_path=other_root / "artifacts" / "input_symbols.csv",
        )
