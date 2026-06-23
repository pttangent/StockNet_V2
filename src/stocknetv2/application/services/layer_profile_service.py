from __future__ import annotations

from dataclasses import dataclass

from stocknetv2.domain.graph.layer_config import ThemeDiscoverySettings


@dataclass(frozen=True)
class LayerProfile:
    name: str
    selected_layers: tuple[str, ...]
    feature_columns: tuple[str, ...]
    max_feature_lookback_minutes: int
    max_return_lookback_minutes: int
    dtw_backend: str


def resolve_layer_profile(name: str, settings: ThemeDiscoverySettings) -> LayerProfile:
    activity_columns = ("timestamp", "available_time", "symbol", "symbol_id", "volume_z_12", "large_trade_ratio_z")
    flow_columns = (*activity_columns, "imbalance_z", "flow_impulse_score")
    dtw_columns = (*flow_columns, "ret_1m")
    full_layers = (
        "return_corr_graph",
        "dtw_return_similarity_graph",
        "flow_alignment_graph",
        "dtw_trade_flow_similarity_graph",
        "volume_expansion_graph",
        "large_trade_alignment_graph",
    )
    dtw_only_layers = ("dtw_return_similarity_graph", "dtw_trade_flow_similarity_graph")
    profiles = {
        "cpu_no_dtw": LayerProfile(
            name="cpu_no_dtw",
            selected_layers=(
                "return_corr_graph",
                "flow_alignment_graph",
                "volume_expansion_graph",
                "large_trade_alignment_graph",
            ),
            feature_columns=flow_columns,
            max_feature_lookback_minutes=max(
                settings.flow_alignment.lookback_minutes,
                60,
            ),
            max_return_lookback_minutes=(settings.return_corr.lookback_bars + 1) * 5,
            dtw_backend=settings.dtw_return.backend,
        ),
        "cpu_full": LayerProfile(
            name="cpu_full",
            selected_layers=full_layers,
            feature_columns=dtw_columns,
            max_feature_lookback_minutes=max(
                settings.flow_alignment.lookback_minutes,
                settings.dtw_return.max_lookback_minutes,
                settings.dtw_trade_flow.max_lookback_minutes,
                60,
            ),
            max_return_lookback_minutes=(settings.return_corr.lookback_bars + 1) * 5,
            dtw_backend=settings.dtw_return.backend,
        ),
        "cpu_dtw_only": LayerProfile(
            name="cpu_dtw_only",
            selected_layers=dtw_only_layers,
            feature_columns=dtw_columns,
            max_feature_lookback_minutes=max(
                settings.dtw_return.max_lookback_minutes,
                settings.dtw_trade_flow.max_lookback_minutes,
            ),
            max_return_lookback_minutes=0,
            dtw_backend=settings.dtw_return.backend,
        ),
        "cpu_only_dtw": LayerProfile(
            name="cpu_only_dtw",
            selected_layers=dtw_only_layers,
            feature_columns=dtw_columns,
            max_feature_lookback_minutes=max(
                settings.dtw_return.max_lookback_minutes,
                settings.dtw_trade_flow.max_lookback_minutes,
            ),
            max_return_lookback_minutes=0,
            dtw_backend=settings.dtw_return.backend,
        ),
        "hybird_full": LayerProfile(
            name="hybird_full",
            selected_layers=full_layers,
            feature_columns=dtw_columns,
            max_feature_lookback_minutes=max(
                settings.flow_alignment.lookback_minutes,
                settings.dtw_return.max_lookback_minutes,
                settings.dtw_trade_flow.max_lookback_minutes,
                60,
            ),
            max_return_lookback_minutes=(settings.return_corr.lookback_bars + 1) * 5,
            dtw_backend=settings.dtw_return.backend,
        ),
        "gpu_only_dtw": LayerProfile(
            name="gpu_only_dtw",
            selected_layers=dtw_only_layers,
            feature_columns=dtw_columns,
            max_feature_lookback_minutes=max(
                settings.dtw_return.max_lookback_minutes,
                settings.dtw_trade_flow.max_lookback_minutes,
            ),
            max_return_lookback_minutes=0,
            dtw_backend=settings.dtw_return.backend,
        ),
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported layer profile: {name}") from exc
