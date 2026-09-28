import pandas as pd

from scripts.analysis.selected_tech_universe_study import select_pools


def pools():
    tickers = ["NVDA", "AMD", "TSLA", "VRT", "KORU", "SKDD", "SKHY", "COIN", "HIMS", "ISRG",
               "MINIMAX", "ZHIPU", "XIAOMI", "POPMART", "CSOPSKHYNIX2L", "KIOXIA", "SOFTBANK",
               "SAMSUNG", "SKHYNIX", "HYUNDAI", "OPENAI"]
    universe = pd.DataFrame({"ticker": tickers, "instId": [f"{t}-USDT-SWAP" for t in tickers]})
    ordinary = {"NVDA", "AMD", "TSLA", "VRT", "SKHY", "COIN", "HIMS", "ISRG"}
    identity = pd.DataFrame({"ticker": tickers, "us_ordinary": [t in ordinary for t in tickers]})
    return {name: {inst.removesuffix("-USDT-SWAP") for inst in ids}
            for name, ids in select_pools(universe, identity).items()}


def test_regional_selection_is_exact_outside_us():
    selected = pools()["broad_tech"]
    assert {"MINIMAX", "ZHIPU", "XIAOMI", "SAMSUNG", "SKHYNIX"} <= selected
    assert not selected & {"POPMART", "CSOPSKHYNIX2L", "KIOXIA", "SOFTBANK", "HYUNDAI", "OPENAI"}


def test_tech_pools_exclude_funds_crypto_and_healthcare():
    for name in ("core_tech", "broad_tech"):
        assert not pools()[name] & {"KORU", "SKDD", "COIN", "HIMS", "ISRG"}
        assert {"NVDA", "AMD", "SKHY"} <= pools()[name]


def test_definition_sensitivity_is_not_a_profitability_filter():
    result = pools()
    assert {"TSLA", "VRT"} <= result["broad_tech"]
    assert not result["core_tech"] & {"TSLA", "VRT"}
    assert {"KORU", "SKDD", "COIN", "HIMS"} <= result["us_all_hk_tech_kr_memory"]
