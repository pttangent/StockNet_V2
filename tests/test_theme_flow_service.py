from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from stocknetv2.application.services.consensus_service import ConsensusThemeCandidate
from stocknetv2.application.services.theme_flow_service import ThemeFlowService
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


def test_theme_flow_service_excludes_unfinished_1m_buckets() -> None:
    snapshot_time = datetime(2026, 1, 2, 14, 35, tzinfo=UTC)
    inputs = TradeDateInputs(
        trade_date="2026-01-02",
        bars_5m=pd.DataFrame(),
        trade_flow_1m=pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "timestamp": [
                    datetime(2026, 1, 2, 14, 34, tzinfo=UTC),
                    datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
                ],
                "available_time": [
                    datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
                    datetime(2026, 1, 2, 14, 36, tzinfo=UTC),
                ],
                "flow_impulse_score": [1.0, 100.0],
                "imbalance_z": [1.0, 1.0],
                "large_trade_ratio_z": [0.0, 1.0],
            }
        ),
        features_1m=pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "timestamp": [
                    datetime(2026, 1, 2, 14, 34, tzinfo=UTC),
                    datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
                ],
                "available_time": [
                    datetime(2026, 1, 2, 14, 35, tzinfo=UTC),
                    datetime(2026, 1, 2, 14, 36, tzinfo=UTC),
                ],
                "ret_1m": [0.01, 0.99],
            }
        ),
        data_version="unit-test",
    )
    candidates = [
        ConsensusThemeCandidate(
            theme_instance_id="theme-1",
            theme_path_id="path-1",
            members=["AAA"],
            source_layers=["flow_alignment_graph"],
            consensus_score=0.8,
            structure_score=0.8,
            cross_layer_consensus_score=0.8,
            flow_support_score=0.8,
            dtw_flow_support_score=0.0,
            volume_support_score=0.0,
            large_trade_support_score=0.0,
            stability_score=0.0,
            semantic_coherence_score=0.0,
            theme_quality_score=0.8,
            theme_quality_breakdown_json="{}",
        )
    ]

    records = ThemeFlowService().build_theme_flow_records(
        candidates=candidates,
        inputs=inputs,
        snapshot_time=pd.Timestamp(snapshot_time),
    )

    assert len(records) == 1
    assert records[0].theme_net_flow == 1.0
    assert records[0].price_breadth == 1.0
    assert records[0].large_trade_breadth == 0.0
