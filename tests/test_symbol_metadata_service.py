from __future__ import annotations

import pandas as pd

from stocknetv2.application.services import symbol_metadata_service
from stocknetv2.application.services.symbol_metadata_service import (
    SymbolMetadataRefreshConfig,
    refresh_symbol_metadata_cache,
)


class _FakeTicker:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @property
    def info(self) -> dict[str, object]:
        if self.symbol == "NVDA":
            return {
                "shortName": "NVIDIA Corporation",
                "sector": "Technology",
                "industry": "Semiconductors",
                "marketCap": 123456789,
                "exchange": "NMS",
                "country": "United States",
                "quoteType": "EQUITY",
                "sharesOutstanding": 1000,
                "enterpriseValue": 222222222,
            }
        return {
            "shortName": "Unknown Corp",
            "sector": None,
            "industry": None,
            "marketCap": None,
            "exchange": "NYQ",
            "country": "United States",
            "quoteType": "EQUITY",
        }

    @property
    def fast_info(self) -> dict[str, object]:
        if self.symbol == "NVDA":
            return {
                "lastPrice": 140.25,
                "marketCap": 123456789,
            }
        return {
            "lastPrice": 11.5,
            "marketCap": None,
        }


class _FakeYFinanceModule:
    def Ticker(self, symbol: str) -> _FakeTicker:
        return _FakeTicker(symbol)


def test_refresh_symbol_metadata_cache_writes_standardized_outputs(tmp_path, monkeypatch):
    symbols_csv = tmp_path / "symbols.csv"
    output_csv = tmp_path / "symbol_metadata.csv"
    output_parquet = tmp_path / "symbol_metadata.parquet"
    symbols_csv.write_text("Ticker\nNVDA\nMISSING\n", encoding="utf-8")
    monkeypatch.setattr(
        symbol_metadata_service,
        "_load_yfinance_module",
        lambda: _FakeYFinanceModule(),
    )
    monkeypatch.setattr(
        symbol_metadata_service,
        "_fetch_nasdaq_screener_snapshot",
        lambda: symbol_metadata_service.empty_symbol_metadata_frame(),
    )

    summary = refresh_symbol_metadata_cache(
        SymbolMetadataRefreshConfig(
            symbols_csv_path=symbols_csv,
            output_csv_path=output_csv,
            output_parquet_path=output_parquet,
            max_workers=2,
        )
    )

    assert summary.symbol_count == 2
    assert output_csv.exists()
    assert output_parquet.exists()

    frame = pd.read_csv(output_csv).sort_values("symbol").reset_index(drop=True)
    assert {"symbol", "company_name", "sector_code", "industry_code", "exchange", "country", "quote_type"}.issubset(
        frame.columns
    )
    assert {"Ticker", "Name", "SectorCode", "IndCode", "Last", "MktCap"}.issubset(frame.columns)

    missing_row = frame.loc[frame["symbol"] == "MISSING"].iloc[0]
    assert pd.isna(missing_row["sector_code"])
    assert pd.isna(missing_row["industry_code"])
    assert missing_row["exchange"] == "NYQ"
    assert missing_row["quote_type"] == "EQUITY"

    nvda_row = frame.loc[frame["symbol"] == "NVDA"].iloc[0]
    assert nvda_row["company_name"] == "NVIDIA Corporation"
    assert nvda_row["sector_code"] == "Technology"
    assert nvda_row["industry_code"] == "Semiconductors"
    assert nvda_row["market_cap"] == 123456789
    assert nvda_row["last_price"] == 140.25
    assert nvda_row["country"] == "United States"
