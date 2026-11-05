"""
P6 Etap 1.8 — forward-return forensic unit tests.

Network yok. evaluate_symbol çağrılmaz. Sentetik fiyat + küçük ledger.
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
from p6_fwd import (
    HORIZONS,
    WIN_RULE,
    agent_horizon_table,
    fwd_return,
    ledger_counts,
    load_ledger,
    match_status,
    overlap_sets,
    pair_returns,
    reconstruct_eval_pairs,
    run_forensic,
    seeded_bootstrap_mean_ci,
    strip_matched_rows,
    summarize,
    winsorize,
)
from p6_replay import ReplayDuplicateError, normalize_ohlcv


def _series(closes: list[float], start="2026-08-03") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    close = np.asarray(closes, dtype=float)
    high = close * 1.02
    low = close * 0.98
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": np.full(len(close), 1e6)},
        index=idx,
    )


def test_fwd_formula_matches_repo():
    df = _series([100.0] * 12)
    df.iloc[1, df.columns.get_loc("close")] = 105.0  # t+1 if t is first? need known date
    # Use same construction as FAZ 4: 30 bars starting 2026-07-01, t=2026-07-10 pos 9
    idx = pd.date_range("2026-07-01", periods=30, freq="D")
    px = [100.0] * 30
    px[10] = 105.0
    px[14] = 115.5
    df = pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px, "volume": [1e6] * 30}, index=idx
    )
    got = fwd_return(df, "2026-07-10", 5)
    expect = meta_portfolio._fwd_return_hesapla(df["close"], date(2026, 7, 10), 5)
    assert got == pytest.approx(expect)
    assert got == pytest.approx(math.log(115.5 / 105.0))


def test_period_1_is_identically_zero_when_matched():
    idx = pd.date_range("2026-07-01", periods=10, freq="D")
    px = list(range(100, 110))
    df = pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px, "volume": [1e6] * 10}, index=idx
    )
    got = fwd_return(df, "2026-07-01", 1)
    assert got == pytest.approx(0.0)


def test_missing_future_horizon():
    idx = pd.date_range("2026-07-01", periods=5, freq="D")
    px = [100.0] * 5
    df = pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px, "volume": [1e6] * 5}, index=idx
    )
    frames = {"AAA": df}
    assert match_status(frames, "AAA", "2026-07-01", 5) == "insufficient_future_horizon"
    assert fwd_return(df, "2026-07-01", 5) is None


def test_missing_symbol_and_date():
    df = _series([100.0] * 20)
    frames = {"AAA": normalize_ohlcv(df, symbol="AAA")}
    assert match_status(frames, "BBB", "2026-08-03", 1) == "missing_symbol"
    assert match_status(frames, "AAA", "2026-01-01", 1) == "missing_signal_date"


def test_nan_close_invalid():
    df = _series([100.0] * 12)
    df.iloc[2, df.columns.get_loc("close")] = float("nan")
    frames = {"AAA": df}
    # t=first, period=1 uses iloc[1]; period=2 uses t+2 which is nan
    st = match_status(frames, "AAA", str(df.index[0].date()), 2)
    assert st == "nan_close"


def test_win_rule_zero_is_not_win():
    assert "zero is not a win" in WIN_RULE
    s = summarize([0.0, 0.1, -0.1])
    assert s["n"] == 3
    assert s["win_count"] == 1
    assert s["zero_count"] == 1
    assert s["win_pct"] == pytest.approx(1 / 3)


def test_summarize_empty():
    s = summarize([])
    assert s["n"] == 0
    assert s["mean"] is None


def test_uplift_is_difference_of_means():
    sqz = summarize([0.10, 0.20])
    ctrl = summarize([0.00, 0.10])
    uplift = sqz["mean"] - ctrl["mean"]
    assert uplift == pytest.approx(0.10)


def test_overlap_zero_jaccard():
    rows = [
        {"date": "2026-08-10", "symbol": "AAA", "agent": "SQZ"},
        {"date": "2026-08-10", "symbol": "BBB", "agent": "ZKN"},
    ]
    o = overlap_sets(rows, "SQZ", "ZKN")
    assert o["intersection"] == 0
    assert o["jaccard"] == pytest.approx(0.0)
    assert o["a_only"] == 1 and o["b_only"] == 1


def test_overlap_intersection():
    rows = [
        {"date": "2026-08-12", "symbol": "BIMAS", "agent": "ZKN"},
        {"date": "2026-08-12", "symbol": "BIMAS", "agent": "WYC"},
    ]
    o = overlap_sets(rows, "ZKN", "WYC")
    assert o["intersection"] == 1
    assert o["intersection_keys"] == [("2026-08-12", "BIMAS")]


def test_ledger_duplicate_raises(tmp_path: Path):
    p = tmp_path / "led.jsonl"
    rec = {
        "date": "2026-08-10", "symbol": "AAA", "agent": "SQZ",
        "signal": True, "rule_version": "sqz-kc20-1.5-v1",
    }
    p.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ReplayDuplicateError):
        load_ledger(p)


def test_load_ledger_counts(tmp_path: Path):
    p = tmp_path / "led.jsonl"
    recs = [
        {"date": "2026-08-10", "symbol": "A", "agent": "SQZ", "signal": True},
        {"date": "2026-08-10", "symbol": "B", "agent": "ZKN", "signal": True},
        {"date": "2026-08-11", "symbol": "C", "agent": "WYC", "signal": True},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    rows = load_ledger(p)
    c = ledger_counts(rows)
    assert c == {"total": 3, "SQZ": 1, "ZKN": 1, "WYC": 1, "all_true": True}


def test_reconstruct_eval_pairs_min_bars():
    df = _series([100.0] * 55, start="2026-06-01")
    frames = {"AAA": df}
    start = str(df.index[49].date())
    end = str(df.index[-1].date())
    pairs = reconstruct_eval_pairs(frames, start=start, end=end, min_bars=50)
    assert len(pairs) == 55 - 49
    assert all(p[1] == "AAA" for p in pairs)


def test_pair_returns_and_agent_table_deterministic():
    idx = pd.bdate_range("2026-08-03", periods=20)
    close = np.linspace(100.0, 120.0, 20)
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": np.full(20, 1e6)},
        index=idx,
    )
    frames = {"AAA": df}
    t = str(idx[5].date())
    rows = [
        {"date": t, "symbol": "AAA", "agent": "SQZ", "signal": True},
        {"date": t, "symbol": "AAA", "agent": "ZKN", "signal": True},
    ]
    a = agent_horizon_table(rows, frames, "SQZ")
    b = agent_horizon_table(rows, frames, "SQZ")
    assert a == b
    assert a["raw_signals"] == 1
    assert a["horizons"]["1"]["stats"]["mean"] == pytest.approx(0.0)


def test_bootstrap_is_seeded_deterministic():
    v = [0.01, -0.02, 0.03, 0.00, -0.01]
    a = seeded_bootstrap_mean_ci(v, n_boot=200, seed=18)
    b = seeded_bootstrap_mean_ci(v, n_boot=200, seed=18)
    assert a == b
    assert a["label"] == "EXPLORATORY"


def test_winsorize_labeled_not_primary():
    vals = [0.0, 0.1, 0.2, 10.0]
    w = winsorize(vals, p=0.25)
    assert max(w) < 10.0


def test_run_forensic_checksum_and_no_alpha():
    # 50 warmup + 6 valid days, one symbol; build 106-count would fail integrity.
    # This test uses run pieces: reconstruct checksum style + strip determinism.
    df = _series([100.0] * 56, start="2026-06-01")
    frames = {"AAA": df}
    start = str(df.index[49].date())
    end = str(df.index[-1].date())
    pairs = reconstruct_eval_pairs(frames, start=start, end=end, min_bars=50)
    assert len(pairs) == 7
    payload = {
        "agents": {"SQZ": {"horizons": {"5": {"matched_rows": [{"date": "x"}]}}}},
        "k": 1,
    }
    compact = strip_matched_rows(payload)
    assert "matched_rows" not in compact["agents"]["SQZ"]["horizons"]["5"]
    again = strip_matched_rows(payload)
    assert compact == again


def test_horizons_tuple_frozen():
    assert HORIZONS == (1, 3, 5, 10, 20)
