from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import importlib
import math
import time

import pandas as pd
import requests


CANONICAL_SYMBOL_METADATA_COLUMNS = [
    "symbol",
    "source_symbol",
    "company_name",
    "sector_code",
    "industry_code",
    "last_price",
    "rank",
    "market_cap",
    "exchange",
    "country",
    "quote_type",
    "shares_outstanding",
    "enterprise_value",
    "currency",
    "security_type",
    "is_etf",
    "fetch_status",
    "fetch_error",
]

LEGACY_SYMBOL_METADATA_COLUMNS = {
    "Ticker": "symbol",
    "Name": "company_name",
    "SectorCode": "sector_code",
    "IndCode": "industry_code",
    "Last": "last_price",
    "Rank": "rank",
    "MktCap": "market_cap",
    "Exchange": "exchange",
    "Country": "country",
    "QuoteType": "quote_type",
}


@dataclass(frozen=True)
class SymbolMetadataRefreshConfig:
    output_csv_path: Path | str
    symbols: tuple[str, ...] | None = None
    symbols_csv_path: Path | str | None = None
    output_parquet_path: Path | str | None = None
    max_workers: int = 8
    retries: int = 2
    refresh_existing: bool = False


@dataclass(frozen=True)
class SymbolMetadataRefreshSummary:
    output_csv_path: Path
    output_parquet_path: Path | None
    symbol_count: int
    fetched_count: int
    reused_count: int


def refresh_symbol_metadata_cache(
    config: SymbolMetadataRefreshConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> SymbolMetadataRefreshSummary:
    logger = log or (lambda message: None)
    output_csv_path = Path(config.output_csv_path).expanduser().resolve()
    output_parquet_path = (
        Path(config.output_parquet_path).expanduser().resolve()
        if config.output_parquet_path is not None
        else None
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if output_parquet_path is not None:
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    symbols = _resolve_symbols(config)
    existing_frame = (
        read_symbol_metadata_csv(output_csv_path)
        if output_csv_path.exists()
        else empty_symbol_metadata_frame()
    )
    existing_records = {
        str(row.symbol): row._asdict()
        for row in existing_frame.itertuples(index=False)
        if isinstance(row.symbol, str) and row.symbol
    }

    symbols_to_fetch: list[str] = []
    reused_count = 0
    for symbol in symbols:
        if (
            not config.refresh_existing
            and symbol in existing_records
            and _row_has_usable_metadata(existing_records[symbol])
        ):
            reused_count += 1
            continue
        symbols_to_fetch.append(symbol)

    fetched_rows: list[dict[str, Any]] = []
    nasdaq_rows_by_symbol: dict[str, dict[str, Any]] = {}
    if symbols_to_fetch:
        logger(f"Fetching metadata for {len(symbols_to_fetch)} symbols with {config.max_workers} workers.")
        nasdaq_frame = _fetch_nasdaq_screener_snapshot()
        if not nasdaq_frame.empty:
            nasdaq_rows_by_symbol = {
                str(row.symbol): row._asdict()
                for row in nasdaq_frame.itertuples(index=False)
                if isinstance(row.symbol, str) and row.symbol
            }
            logger(f"Nasdaq screener resolved {sum(1 for symbol in symbols_to_fetch if symbol in nasdaq_rows_by_symbol)}/{len(symbols_to_fetch)} symbols.")
        fetched_rows.extend(
            nasdaq_rows_by_symbol[symbol]
            for symbol in symbols_to_fetch
            if symbol in nasdaq_rows_by_symbol
        )
        yfinance_symbols = [symbol for symbol in symbols_to_fetch if symbol not in nasdaq_rows_by_symbol]
        if yfinance_symbols:
            logger(f"Falling back to yfinance for {len(yfinance_symbols)} symbols.")
        with ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as executor:
            futures = {
                executor.submit(_fetch_symbol_metadata_with_retries, symbol, config.retries): symbol
                for symbol in yfinance_symbols
            }
            completed_count = 0
            for future in as_completed(futures):
                fetched_rows.append(future.result())
                completed_count += 1
                if completed_count == len(yfinance_symbols) or completed_count % 100 == 0:
                    logger(f"Metadata progress: {completed_count}/{len(yfinance_symbols)} yfinance fallbacks")

    fetched_frame = pd.DataFrame(fetched_rows) if fetched_rows else empty_symbol_metadata_frame()
    if not existing_frame.empty and not config.refresh_existing:
        kept_existing = existing_frame.loc[existing_frame["symbol"].isin(symbols)].copy()
        combined = pd.concat([kept_existing, fetched_frame], ignore_index=True)
    else:
        combined = fetched_frame

    combined = standardize_symbol_metadata_frame(combined)
    combined = combined.drop_duplicates(subset=["symbol"], keep="last").sort_values("symbol").reset_index(drop=True)
    csv_frame = combined.copy()
    for legacy_column, canonical_column in LEGACY_SYMBOL_METADATA_COLUMNS.items():
        csv_frame[legacy_column] = csv_frame[canonical_column]

    csv_frame.to_csv(output_csv_path, index=False)
    if output_parquet_path is not None:
        combined.to_parquet(output_parquet_path, index=False, compression="zstd")

    return SymbolMetadataRefreshSummary(
        output_csv_path=output_csv_path,
        output_parquet_path=output_parquet_path,
        symbol_count=len(symbols),
        fetched_count=len(symbols_to_fetch),
        reused_count=reused_count,
    )


def read_symbol_metadata_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    return standardize_symbol_metadata_frame(frame)


def empty_symbol_metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_SYMBOL_METADATA_COLUMNS)


def standardize_symbol_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_symbol_metadata_frame()

    normalized = pd.DataFrame(index=frame.index)
    normalized["symbol"] = _normalize_symbol(_pick_column(frame, "symbol", "Ticker"))
    normalized["source_symbol"] = _normalize_symbol(
        _pick_column(frame, "source_symbol", "Ticker", "symbol")
    )
    normalized["company_name"] = _normalize_text(_pick_column(frame, "company_name", "Name", "short_name"))
    normalized["sector_code"] = _normalize_text(_pick_column(frame, "sector_code", "SectorCode", "sector"))
    normalized["industry_code"] = _normalize_text(_pick_column(frame, "industry_code", "IndCode", "industry"))
    normalized["last_price"] = _normalize_numeric(_pick_column(frame, "last_price", "Last"))
    normalized["rank"] = _normalize_integer(_pick_column(frame, "rank", "Rank"))
    normalized["market_cap"] = _normalize_numeric(_pick_column(frame, "market_cap", "MktCap"))
    normalized["exchange"] = _normalize_text(_pick_column(frame, "exchange", "Exchange"))
    normalized["country"] = _normalize_text(_pick_column(frame, "country", "Country"))
    normalized["quote_type"] = _normalize_text(_pick_column(frame, "quote_type", "QuoteType"))
    normalized["shares_outstanding"] = _normalize_numeric(_pick_column(frame, "shares_outstanding"))
    normalized["enterprise_value"] = _normalize_numeric(_pick_column(frame, "enterprise_value"))
    normalized["currency"] = _normalize_text(_pick_column(frame, "currency", "Currency"))
    normalized["security_type"] = _normalize_text(_pick_column(frame, "security_type", "SecurityType"))
    normalized["is_etf"] = _normalize_bool(_pick_column(frame, "is_etf", "IsETF"))
    normalized["fetch_status"] = _normalize_text(_pick_column(frame, "fetch_status"))
    normalized["fetch_error"] = _normalize_text(_pick_column(frame, "fetch_error"))

    normalized["source_symbol"] = normalized["source_symbol"].fillna(normalized["symbol"])
    normalized["is_etf"] = normalized["is_etf"].fillna(
        normalized["quote_type"].fillna("").str.upper().eq("ETF")
        | normalized["security_type"].fillna("").str.upper().eq("ETF")
    )

    return normalized.loc[:, CANONICAL_SYMBOL_METADATA_COLUMNS]


def _resolve_symbols(config: SymbolMetadataRefreshConfig) -> list[str]:
    if config.symbols is not None:
        return sorted({symbol.strip().upper() for symbol in config.symbols if symbol and symbol.strip()})
    if config.symbols_csv_path is None:
        raise ValueError("Either symbols or symbols_csv_path must be provided.")
    frame = pd.read_csv(Path(config.symbols_csv_path))
    for candidate in ("symbol", "Ticker", "ticker"):
        if candidate in frame.columns:
            values = frame[candidate]
            break
    else:
        values = frame.iloc[:, 0]
    symbols = sorted({str(value).strip().upper() for value in values if str(value).strip()})
    return symbols


def _fetch_symbol_metadata_with_retries(symbol: str, retries: int) -> dict[str, Any]:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            return _fetch_symbol_metadata(symbol)
        except Exception as exc:
            last_error = str(exc)
            if attempt >= retries:
                break
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
    return {
        "symbol": symbol,
        "source_symbol": symbol,
        "company_name": None,
        "sector_code": None,
        "industry_code": None,
        "last_price": None,
        "rank": None,
        "market_cap": None,
        "exchange": None,
        "country": None,
        "quote_type": None,
        "shares_outstanding": None,
        "enterprise_value": None,
        "currency": None,
        "security_type": None,
        "is_etf": None,
        "fetch_status": "error",
        "fetch_error": last_error[:500],
    }


def _fetch_symbol_metadata(symbol: str) -> dict[str, Any]:
    yf = _load_yfinance_module()
    ticker = yf.Ticker(symbol)
    info = _coerce_mapping(getattr(ticker, "info", None))
    fast_info = _coerce_mapping(getattr(ticker, "fast_info", None))
    quote_type = _first_present(info, "quoteType", "quote_type")
    security_type = _first_present(info, "typeDisp", "securityType", "instrumentType")
    return {
        "symbol": symbol,
        "source_symbol": symbol,
        "company_name": _first_present(info, "shortName", "longName", "displayName", "name"),
        "sector_code": _first_present(info, "sector"),
        "industry_code": _first_present(info, "industry"),
        "last_price": _first_present(fast_info, "lastPrice", "last_price") or _first_present(info, "currentPrice", "regularMarketPrice", "previousClose"),
        "rank": None,
        "market_cap": _first_present(fast_info, "marketCap", "market_cap") or _first_present(info, "marketCap"),
        "exchange": _first_present(info, "exchange"),
        "country": _first_present(info, "country"),
        "quote_type": quote_type,
        "shares_outstanding": _first_present(info, "sharesOutstanding"),
        "enterprise_value": _first_present(info, "enterpriseValue"),
        "currency": _first_present(info, "currency", "financialCurrency"),
        "security_type": security_type,
        "is_etf": str(quote_type).upper() == "ETF" if quote_type is not None else None,
        "fetch_status": "ok",
        "fetch_error": None,
    }


def _fetch_nasdaq_screener_snapshot() -> pd.DataFrame:
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&offset=0&download=true"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
    }
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("rows", [])
    if not rows:
        return empty_symbol_metadata_frame()
    frame = pd.DataFrame(rows)
    normalized = pd.DataFrame(
        {
            "symbol": _normalize_symbol(_pick_column(frame, "symbol")),
            "source_symbol": _normalize_symbol(_pick_column(frame, "symbol")),
            "company_name": _normalize_text(_pick_column(frame, "name")),
            "sector_code": _normalize_text(_pick_column(frame, "sector")),
            "industry_code": _normalize_text(_pick_column(frame, "industry")),
            "last_price": _normalize_currency_numeric(_pick_column(frame, "lastsale")),
            "rank": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "market_cap": _normalize_currency_numeric(_pick_column(frame, "marketCap")),
            "exchange": pd.Series([pd.NA] * len(frame), dtype="string"),
            "country": _normalize_text(_pick_column(frame, "country")),
            "quote_type": pd.Series([pd.NA] * len(frame), dtype="string"),
            "shares_outstanding": pd.Series([float("nan")] * len(frame), dtype="float64"),
            "enterprise_value": pd.Series([float("nan")] * len(frame), dtype="float64"),
            "currency": pd.Series(["USD"] * len(frame), dtype="string"),
            "security_type": pd.Series([pd.NA] * len(frame), dtype="string"),
            "is_etf": pd.Series([pd.NA] * len(frame), dtype="boolean"),
            "fetch_status": pd.Series(["ok"] * len(frame), dtype="string"),
            "fetch_error": pd.Series([pd.NA] * len(frame), dtype="string"),
        }
    )
    return standardize_symbol_metadata_frame(normalized)


def _row_has_usable_metadata(row: dict[str, Any]) -> bool:
    sector = row.get("sector_code")
    industry = row.get("industry_code")
    market_cap = row.get("market_cap")
    company_name = row.get("company_name")
    for text_value in (sector, industry):
        if isinstance(text_value, str) and text_value.strip() and text_value.strip().upper() != "UNKNOWN":
            return True
    if pd.notna(market_cap) and float(market_cap) > 0:
        return True
    if isinstance(company_name, str) and company_name.strip() and company_name.strip().upper() != str(row.get("symbol", "")).strip().upper():
        return True
    return False


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _pick_column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    available = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        resolved = available.get(candidate.strip().lower())
        if resolved is not None:
            return frame[resolved]
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


def _normalize_symbol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA})


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA})


def _normalize_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_currency_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_integer(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64")


def _normalize_bool(series: pd.Series) -> pd.Series:
    def _coerce(value: Any) -> bool | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        return None

    return series.map(_coerce).astype("boolean")


def _load_yfinance_module() -> Any:
    return importlib.import_module("yfinance")
