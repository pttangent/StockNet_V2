from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class GraphEdge:
    graph_layer: str
    edge_type: str
    source_symbol: str
    target_symbol: str
    snapshot_time: pd.Timestamp
    weight: float
    raw_score: float
    support_points: int
    edge_confidence: float = 1.0
    effective_lookback_minutes: int | None = None
    calculation_backend: str = "cpu_python"
