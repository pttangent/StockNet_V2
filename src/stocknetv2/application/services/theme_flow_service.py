from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stocknetv2.application.services.consensus_service import ConsensusThemeCandidate
from stocknetv2.domain.graph.series_utils import select_time_window
from stocknetv2.infrastructure.repositories.market_read_repository import TradeDateInputs


@dataclass(frozen=True)
class ThemeFlowRecord:
    theme_instance_id: str
    theme_path_id: str
    timestamp: pd.Timestamp
    theme_net_flow: float
    theme_inflow: float
    theme_outflow: float
    flow_breadth: float
    price_breadth: float
    dtw_flow_coherence: float
    large_trade_breadth: float
    member_count: int


class ThemeFlowService:
    """Aggregate per-theme flow and breadth metrics."""

    def build_theme_flow_records(
        self,
        *,
        candidates: list[ConsensusThemeCandidate],
        inputs: TradeDateInputs,
        snapshot_time: pd.Timestamp,
    ) -> list[ThemeFlowRecord]:
        trade_flow_window = select_time_window(
            inputs.trade_flow_1m,
            snapshot_time=snapshot_time,
            minutes=60,
        ).copy()
        features_window = select_time_window(
            inputs.features_1m,
            snapshot_time=snapshot_time,
            minutes=60,
        ).copy()

        records: list[ThemeFlowRecord] = []
        for candidate in candidates:
            members = set(candidate.members)
            flow_slice = trade_flow_window[trade_flow_window["symbol"].isin(members)].copy()
            feature_slice = features_window[features_window["symbol"].isin(members)].copy()

            if flow_slice.empty:
                theme_net_flow = 0.0
                theme_inflow = 0.0
                theme_outflow = 0.0
                flow_breadth = 0.0
                large_trade_breadth = 0.0
            else:
                signed_flow = (
                    flow_slice["flow_impulse_score"].astype(float)
                    * flow_slice["imbalance_z"].astype(float).apply(lambda value: 1.0 if value >= 0 else -1.0)
                )
                theme_net_flow = float(signed_flow.sum())
                theme_inflow = float(signed_flow[signed_flow > 0].sum())
                theme_outflow = float((-signed_flow[signed_flow < 0]).sum())
                flow_breadth = float((signed_flow > 0).sum() / len(signed_flow))
                if "large_trade_ratio_z" in flow_slice.columns:
                    large_trade_breadth = float((flow_slice["large_trade_ratio_z"].astype(float) > 0).sum() / len(flow_slice))
                else:
                    large_trade_breadth = 0.0

            if feature_slice.empty or "ret_1m" not in feature_slice.columns:
                price_breadth = 0.0
            else:
                latest_returns = (
                    feature_slice.sort_values("timestamp")
                    .groupby("symbol", as_index=False)
                    .tail(1)["ret_1m"]
                    .astype(float)
                )
                price_breadth = float((latest_returns > 0).sum() / len(latest_returns)) if len(latest_returns) else 0.0

            records.append(
                ThemeFlowRecord(
                    theme_instance_id=candidate.theme_instance_id,
                    theme_path_id=candidate.theme_path_id,
                    timestamp=snapshot_time,
                    theme_net_flow=theme_net_flow,
                    theme_inflow=theme_inflow,
                    theme_outflow=theme_outflow,
                    flow_breadth=flow_breadth,
                    price_breadth=price_breadth,
                    dtw_flow_coherence=candidate.consensus_score,
                    large_trade_breadth=large_trade_breadth,
                    member_count=len(candidate.members),
                )
            )
        return records
