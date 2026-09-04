"""Research-only categories for the OKX tokenized-stock universe.

The exchange universe does not include a sector field.  These groups are
manually curated from each underlying's primary business or instrument type,
and are intentionally broad.  They are used for pooled diagnostics only;
single-ticker conclusions are never promoted.
"""

from __future__ import annotations

import pandas as pd


CATEGORY_GROUPS: dict[str, tuple[str, ...]] = {
    "etf_index": (
        "EWJ", "EWT", "EWY", "EWZ", "IWM", "KR200", "QQQ", "SPY", "SMH", "URNM", "USO", "XBI", "XLE",
    ),
    "leveraged_inverse": (
        "CSOPSAMSUNG2L", "CSOPSKHYNIX2L", "DRAM", "KORU", "MUU", "MVLL", "NVDL", "SKDD", "SKHY",
        "SKUU", "SOXL", "SOXS", "SQQQ", "TQQQ", "TMF", "TSLL", "UVXY",
    ),
    "semiconductor_hardware": (
        "AAOI", "AEHR", "ALAB", "AMAT", "AMD", "ARM", "ASML", "AVGO", "AXTI", "CGNX", "CIEN",
        "COHR", "CRDO", "CXMT", "GLW", "INTC", "KIOXIA", "KLAC", "LITE", "LRCX", "MRVL", "MU",
        "ON", "POET", "QCOM", "SAMSUNG", "SIMO", "SKHYNIX", "SMCI", "SNDK", "TER", "TSEM", "TTMI",
        "TSM", "WDC",
    ),
    "mega_cap_tech": (
        "AAPL", "ADBE", "AMZN", "APP", "CRM", "CRWD", "CSCO", "DDOG", "DELL", "GOOGL", "HPE",
        "IBM", "META", "MSFT", "NET", "NFLX", "NOW", "NVDA", "OKTA", "ORCL", "PLTR", "RDDT",
        "SHOP", "SNOW", "TWLO", "ZM",
    ),
    "ai_infrastructure_energy_space": (
        "APLD", "ASTS", "BE", "CBRS", "CRWV", "FLNC", "GEV", "IONQ", "IREN", "LUNR", "NBIS", "ONDS",
        "OUST", "RDW", "RKLB", "ROK", "USAR", "VRT",
    ),
    "crypto_native": (
        "BMNR", "COIN", "CRCL", "MARA", "MSTR", "RIOT", "STRC",
    ),
    "healthcare_biotech": (
        "HIMS", "ISRG", "JNJ", "LLY", "MRNA", "MRK", "OSCR", "UNH",
    ),
    "consumer_retail_media": (
        "COST", "DKNG", "GME", "KO", "POPMART", "TTWO", "WEN", "WMT",
    ),
    "auto_industrial_foreign": (
        "BB", "HYUNDAI", "NOK", "RIVN", "SOFTBANK", "SONY", "TSLA", "XIAOMI",
    ),
    "foreign_private_tech": (
        "ANTHROPIC", "MINIMAX", "MOONSHOT", "OPENAI", "SHEIN", "UNITREE", "ZHIPU",
    ),
    "other_speculative": (
        "BOT", "BSP", "FLY", "FWDI", "INFQ", "INTW", "LYTE", "PENG", "PURR", "QNT", "RAM", "SHAZ",
        "SHLD", "SNXX", "SPCH", "SPCX",
    ),
    "financial_services": ("BRKB", "BX", "HOOD", "PYPL"),
}


_TICKER_TO_CATEGORY = {
    ticker: category
    for category, tickers in CATEGORY_GROUPS.items()
    for ticker in tickers
}


def classify_universe(universe: pd.DataFrame) -> pd.DataFrame:
    """Attach a broad category and its universe count to an instrument table."""
    out = universe.copy()
    out["category"] = out["ticker"].astype(str).map(_TICKER_TO_CATEGORY).fillna("unclassified")
    counts = out["category"].value_counts()
    out["category_size"] = out["category"].map(counts).astype(int)
    return out


def validate_categories(universe: pd.DataFrame) -> None:
    """Fail loudly if the manually curated mapping misses or duplicates a ticker."""
    tickers = set(universe["ticker"].astype(str))
    mapped = set(_TICKER_TO_CATEGORY)
    missing = tickers - mapped
    unknown = mapped - tickers
    if missing or unknown:
        raise ValueError(f"category mapping mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if len(_TICKER_TO_CATEGORY) != sum(len(values) for values in CATEGORY_GROUPS.values()):
        raise ValueError("duplicate ticker in category mapping")
