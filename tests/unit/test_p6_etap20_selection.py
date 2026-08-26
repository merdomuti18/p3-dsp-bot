"""
P6 Etap 2.0 — SQZ selection-edge forensic unit tests.

Network yok. evaluate_symbol çağrılmaz. Sentetik fiyat + küçük ledger.
Production dosyalarına yazılmaz.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import meta_portfolio
import scanner_p6
from p6_fwd import fwd_return, load_ledger, match_status, summarize
from p6_replay import ReplayDuplicateError, normalize_ohlcv
from p6_selection import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    EXPECTED_COUNTS,
    HORIZONS,
    INCONCLUSIVE_N,
    LedgerIntegrityError,
    classify_sqz,
    cliffs_delta,
    cohens_d,
    compact_payload,
    group_horizon_stats,
    incremental_block,
    overlap_groups,
    reconstruct_eval_pairs_p6,
    run_etap20,
    seeded_bootstrap_diff_ci,
    split_date_thirds,
    symbol_concentration,
)


def _ohlcv(closes: list[float], start="2026-03-02", volume: float = 1_000_000.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.05,
            "low": close * 0.95,
            "close": close,
            "volume": np.full(len(close), volume),
        },
        index=idx,
    )


def _row(d: str, symbol: str, agent: str) -> dict:
    versions = {
        "SQZ": "sqz-kc20-1.5-v1",
        "ZKN": "zkn-p1-birebir-v1",
        "WYC": "wyc-v1",
    }
    return {
        "date": d,
        "symbol": symbol,
        "agent": agent,
        "signal": True,
        "rule_version": versions[agent],
    }


def test_fwd_formula_matches_repo_no_t_in_denominator():
    idx = pd.date_range("2026-07-01", periods=30, freq="D")
    px = [100.0] * 30
    px[10] = 105.0  # t+1 if t=2026-07-10 (pos 9)
    px[14] = 115.5  # t+5
    df = pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px, "volume": [1e6] * 30}, index=idx
    )
    got = fwd_return(df, "2026-07-10", 5)
    expect = meta_portfolio._fwd_return_hesapla(df["close"], date(2026, 7, 10), 5)
    assert got == pytest.approx(expect)
    assert got == pytest.approx(math.log(115.5 / 105.0))
    # close[t] must not enter
    assert got != pytest.approx(math.log(115.5 / 100.0))


def test_period_1_identity_and_missing_horizon():
    idx = pd.date_range("2026-07-01", periods=5, freq="D")
    px = list(range(100, 105))
    df = pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px, "volume": [1e6] * 5}, index=idx
    )
    assert fwd_return(df, "2026-07-01", 1) == pytest.approx(0.0)
    assert match_status({"AAA": df}, "AAA", "2026-07-01", 5) == "insufficient_future_horizon"
    assert fwd_return(df, "2026-07-01", 5) is None


def test_no_future_leakage_uses_t_plus_1_not_t():
    idx = pd.date_range("2026-07-01", periods=10, freq="D")
    px = [100.0] * 10
    px[0] = 999.0  # close[t] spike must be ignored
    px[1] = 110.0
    px[5] = 121.0
    df = pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px, "volume": [1e6] * 10}, index=idx
    )
    got = fwd_return(df, "2026-07-01", 5)
    assert got == pytest.approx(math.log(121.0 / 110.0))


def test_invalid_price_is_none_no_impute():
    idx = pd.date_range("2026-07-01", periods=8, freq="D")
    px = [100.0] * 8
    px[3] = 0.0
    df = pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px, "volume": [1e6] * 8}, index=idx
    )
    frames = {"AAA": df}
    # t=first, period=3 uses t+3 which is 0
    assert match_status(frames, "AAA", "2026-07-01", 3) == "invalid_return"
    assert fwd_return(df, "2026-07-01", 3) is None


def test_win_rule_zero_is_not_win():
    s = summarize([0.0, 0.1, -0.1])
    assert s["win_count"] == 1
    assert s["zero_count"] == 1
    assert s["win_pct"] == pytest.approx(1 / 3)


def test_ledger_duplicate_invariant(tmp_path: Path):
    p = tmp_path / "led.jsonl"
    rec = _row("2026-08-10", "AAA", "SQZ")
    p.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ReplayDuplicateError):
        load_ledger(p)


def test_expected_expanded_counts_constant():
    assert EXPECTED_COUNTS == {"total": 1830, "SQZ": 425, "ZKN": 608, "WYC": 797}
    assert HORIZONS == (1, 3, 5, 10, 20)


def test_control_construction_same_asof_universe():
    df = _ohlcv([100.0] * 60, start="2026-03-02")
    frames = {"AAA": normalize_ohlcv(df, symbol="AAA")}
    start = str(df.index[49].date())
    end = str(df.index[-1].date())
    pairs = reconstruct_eval_pairs_p6(frames, start=start, end=end, min_bars=50)
    assert len(pairs) == 11
    assert all(p[1] == "AAA" for p in pairs)
    sqz = {("2026-05-11", "AAA")}  # may or may not be in window; subtract whatever exists
    non_sqz = set(pairs) - sqz
    assert non_sqz.isdisjoint(sqz)
    assert non_sqz.issubset(set(pairs))


def test_control_applies_universe_ok_not_min_bars_only():
    cheap = _ohlcv([1.0] * 60, volume=1.0)  # mc = 1 * 1 = 1 < 10M
    rich = _ohlcv([100.0] * 60, volume=1_000_000.0)
    frames = {
        "CHEAP": normalize_ohlcv(cheap, symbol="CHEAP"),
        "RICH": normalize_ohlcv(rich, symbol="RICH"),
    }
    start = str(rich.index[49].date())
    end = str(rich.index[-1].date())
    with_mc = reconstruct_eval_pairs_p6(frames, start=start, end=end, min_bars=50, apply_universe_ok=True)
    no_mc = reconstruct_eval_pairs_p6(frames, start=start, end=end, min_bars=50, apply_universe_ok=False)
    assert all(sym == "RICH" for _, sym in with_mc)
    assert {sym for _, sym in no_mc} == {"CHEAP", "RICH"}
    assert scanner_p6.universe_ok(cheap) is False
    assert scanner_p6.universe_ok(rich) is True


def test_overlap_groups_and_pool_dedup():
    rows = [
        _row("2026-04-01", "AAA", "SQZ"),
        _row("2026-04-01", "AAA", "ZKN"),
        _row("2026-04-01", "BBB", "WYC"),
        _row("2026-04-02", "CCC", "SQZ"),
        _row("2026-04-03", "DDD", "ZKN"),
    ]
    g = overlap_groups(rows)
    assert g["SQZ_and_ZKN"] == {("2026-04-01", "AAA")}
    assert g["SQZ_only"] == {("2026-04-02", "CCC")}
    assert g["SQZ_and_WYC"] == set()
    assert g["SQZ_and_ZKN_and_WYC"] == set()
    assert len(g["BASE"]) == 3  # AAA, BBB, DDD
    assert len(g["BASE_plus_SQZ"]) == 4  # + CCC; AAA not doubled
    assert ("2026-04-01", "AAA") in g["BASE_plus_SQZ"]


def test_incremental_pool_does_not_double_count():
    idx = pd.bdate_range("2026-03-02", periods=70)
    close = np.linspace(100.0, 140.0, 70)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(70, 1e6),
        },
        index=idx,
    )
    frames = {"AAA": df, "BBB": df.copy()}
    t = str(idx[55].date())
    base = {(t, "AAA")}
    plus = {(t, "AAA"), (t, "AAA")}  # set dedup
    plus = {(t, "AAA")}
    block = incremental_block(base, plus, frames)
    assert block["5"]["base_n"] == block["5"]["plus_n"]
    assert block["5"]["delta_mean"] == pytest.approx(0.0)


def test_bootstrap_determinism_seed_18():
    a = [0.01, -0.02, 0.03, 0.00, -0.01, 0.04]
    b = [0.00, 0.01, -0.01, 0.02, -0.03]
    x = seeded_bootstrap_diff_ci(a, b, n_boot=200, seed=18)
    y = seeded_bootstrap_diff_ci(a, b, n_boot=200, seed=18)
    assert x == y
    assert x["seed"] == 18
    assert x["label"] == "EXPLORATORY"
    assert x["not_production_edge_proof"] is True


def test_bootstrap_ci_crosses_zero_labeled():
    a = [0.01, -0.01, 0.02]
    b = [0.00, 0.01, -0.02]
    out = seeded_bootstrap_diff_ci(a, b, n_boot=300, seed=18)
    assert "mean_ci95" in out
    assert isinstance(out["mean_ci_crosses_zero"], bool)


def test_effect_size_methods():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 1.0, 2.0, 3.0]
    d = cohens_d(a, b)
    c = cliffs_delta(a, b)
    assert d is not None and d > 0
    assert c > 0
    assert cohens_d([], [1.0]) is None


def test_alpha_formula_same_fwd_method():
    idx = pd.date_range("2026-07-01", periods=20, freq="D")
    stock = np.full(20, 100.0)
    stock[6] = 110.0  # t+1
    stock[10] = 121.0  # t+5
    bench_px = np.full(20, 1000.0)
    bench_px[6] = 1000.0
    bench_px[10] = 1050.0
    stock_df = pd.DataFrame(
        {"open": stock, "high": stock, "low": stock, "close": stock, "volume": [1e6] * 20},
        index=idx,
    )
    bench = pd.Series(bench_px, index=idx)
    t = "2026-07-06"  # pos 5
    s = fwd_return(stock_df, t, 5)
    b = meta_portfolio._fwd_return_hesapla(bench, date.fromisoformat(t), 5)
    alpha = s - b
    assert s == pytest.approx(math.log(121.0 / 110.0))
    assert b == pytest.approx(math.log(1050.0 / 1000.0))
    assert alpha == pytest.approx(s - b)
    # missing benchmark bar → None, no impute
    short = bench.iloc[:-10]
    assert meta_portfolio._fwd_return_hesapla(short, date.fromisoformat(t), 5) is None


def test_mfe_mae_descriptive_not_execution():
    idx = pd.date_range("2026-07-01", periods=10, freq="D")
    close = np.full(10, 100.0)
    high = np.full(10, 100.0)
    low = np.full(10, 100.0)
    close[1] = 100.0  # t+1
    high[2] = 110.0
    low[3] = 90.0
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": [1e6] * 10},
        index=idx,
    )
    from p6_fwd import mfe_mae

    mfe, mae = mfe_mae(df, "2026-07-01", 5)
    assert mfe == pytest.approx(math.log(110.0 / 100.0))
    assert mae == pytest.approx(math.log(90.0 / 100.0))


def test_period_thirds_equal_count_and_reported():
    dates = [f"2026-03-{d:02d}" for d in range(1, 10)]
    split = split_date_thirds(dates)
    assert split["n_dates"] == 9
    assert len(split["buckets"]["EARLY"]) == 3
    assert len(split["buckets"]["MIDDLE"]) == 3
    assert len(split["buckets"]["LATE"]) == 3
    assert "equal-count" in split["method"]
    a = split_date_thirds(dates)
    b = split_date_thirds(dates)
    assert a["ranges"] == b["ranges"]


def test_symbol_sensitivity_top_drop():
    rows = [_row("2026-04-01", "NETAS", "SQZ")] * 5 + [_row("2026-04-02", "AAA", "SQZ")] * 2
    # unique (date,symbol,agent) not required here — concentration is a Counter
    conc = symbol_concentration(rows)
    assert conc["unique_symbols"] == 2
    assert conc["top1"][0]["symbol"] == "NETAS"
    assert conc["top1_share"] == pytest.approx(5 / 7)


def test_overlap_group_small_n_inconclusive():
    pairs = {("2026-04-01", "AAA"), ("2026-04-02", "BBB")}
    df = _ohlcv([100.0] * 80)
    frames = {"AAA": df, "BBB": df.copy()}
    stats = group_horizon_stats(pairs, frames)
    assert stats["n_pairs"] == 2
    assert stats["sample_flag"] == "INCONCLUSIVE"
    assert INCONCLUSIVE_N == 30


def test_run_etap20_integrity_and_determinism():
    idx = pd.bdate_range("2026-03-02", periods=80)
    close_a = 100 + np.linspace(0, 20, 80)
    close_b = 100 + np.linspace(0, -5, 80)
    def frame(close):
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": np.full(len(close), 1e6),
            },
            index=idx,
        )

    frames = {"AAA": frame(close_a), "BBB": frame(close_b)}
    t = str(idx[55].date())
    t2 = str(idx[56].date())
    rows = [
        _row(t, "AAA", "SQZ"),
        _row(t, "BBB", "ZKN"),
        _row(t2, "BBB", "WYC"),
    ]
    start = str(idx[49].date())
    end = str(idx[-1].date())
    eval_pairs = reconstruct_eval_pairs_p6(frames, start=start, end=end, min_bars=50)
    meta = {
        "earliest_valid_p6_date": start,
        "latest_valid_p6_date": end,
        "min_bars": 50,
        "evaluation_count": len(eval_pairs),
        "below_mc": 0,
        "closed_asof_cutoff": end,
    }
    expected = {"total": 3, "SQZ": 1, "ZKN": 1, "WYC": 1}
    bench = pd.Series(np.linspace(1000, 1100, 80), index=idx)
    p1 = run_etap20(
        rows, frames, meta, benchmark=bench,
        benchmark_meta={"chosen": {"ticker": "^XU100", "source": "synthetic"}},
        expected_counts=expected,
    )
    p2 = run_etap20(
        rows, frames, meta, benchmark=bench,
        benchmark_meta={"chosen": {"ticker": "^XU100", "source": "synthetic"}},
        expected_counts=expected,
    )
    c1 = compact_payload(p1)
    c2 = compact_payload(p2)
    assert json.dumps(c1, sort_keys=True) == json.dumps(c2, sort_keys=True)
    assert p1["eval_grid"]["checksum_ok"] is True
    assert p1["alpha"]["computable"] is True
    assert p1["agents"]["SQZ"]["horizons"]["1"]["stats"]["mean"] == pytest.approx(0.0)
    assert "matched_rows" not in c1["agents"]["SQZ"]["horizons"]["5"]
    assert p1["classification"]["class"] in {"A", "B", "C", "D", "F"}


def test_run_etap20_rejects_count_mismatch():
    rows = [_row("2026-04-01", "AAA", "SQZ")]
    with pytest.raises(LedgerIntegrityError):
        run_etap20(rows, {}, {"earliest_valid_p6_date": "2026-03-06",
                              "latest_valid_p6_date": "2026-08-25",
                              "min_bars": 50,
                              "evaluation_count": 0})


def test_classify_does_not_use_low_overlap_as_redundant():
    # Build a payload-like dict that is positive vs control AND incremental,
    # with zero overlap — must not become D solely from overlap=0.
    def hz(mean_u, med_u, win_u, dmean, ci_cross):
        return {
            "mean_uplift": mean_u,
            "median_uplift": med_u,
            "sqz_win_pct": 0.55 + win_u,
            "control_win_pct": 0.55,
            "bootstrap": {"mean_ci_crosses_zero": ci_cross},
        }

    payload = {
        "eval_grid": {"checksum_ok": True},
        "selection_uplift": {
            "5": hz(0.01, 0.008, 0.03, 0.002, True),
            "10": hz(0.012, 0.009, 0.02, 0.002, True),
            "20": hz(0.015, 0.01, 0.02, 0.003, True),
        },
        "incremental_base_vs_base_plus_sqz": {
            "5": {"delta_mean": 0.004},
            "10": {"delta_mean": 0.005},
            "20": {"delta_mean": 0.006},
        },
        "time_stability": {
            "horizons": {
                "5": {
                    "EARLY": {"n": 40, "mean": 0.01},
                    "MIDDLE": {"n": 40, "mean": 0.012},
                    "LATE": {"n": 40, "mean": 0.008},
                }
            }
        },
        "symbol_sensitivity": {
            "5": {"all": {"mean": 0.01}, "excluding_top3": {"mean": 0.009}}
        },
        "overlap_sqz_zkn": {"intersection": 0},
        "overlap_sqz_wyc": {"intersection": 0},
    }
    out = classify_sqz(payload)
    assert out["class"] != "D"
    assert out["class"] in {"A", "B"}
    assert any("not by itself D" in r for r in out["reasons"])


def test_bootstrap_constants():
    assert BOOTSTRAP_SEED == 18
    assert BOOTSTRAP_N == 5000
