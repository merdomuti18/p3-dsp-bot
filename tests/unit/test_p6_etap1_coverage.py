"""
P6 Etap 1.9 — expanded coverage replay tests.

Network yok. evaluate_symbol mevcut P6 fonksiyonu; Boolean değişmez.
Forward return hesaplanmaz. Orijinal 1.7 ledger path'i overwrite edilmez.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scanner_p6
from p6_replay import (
    LEDGER_PURPOSE,
    frames_look_ahead_equal,
    ledgers_equal,
    replay_asof,
    write_ledger,
)
from p6_replay_expanded import (
    EXPANDED_LEDGER_NAME,
    EXPANDED_META_NAME,
    HORIZONS,
    ORIGINAL_LEDGER,
    apply_closed_cutoff,
    future_horizon_available,
    horizon_coverage,
    last_closed_asof_cutoff,
    look_ahead_scan,
)
from p6_scenarios import SQZ_NO, SQZ_YES, WYC_NO, WYC_YES, ZKN_NO, ZKN_YES


def _ohlcv(n=200, seed=7, vol_scale=5_000_000.0, start="2025-01-06") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 100 + rng.normal(0, 1.0, n).cumsum()
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.5, n)
    volume = rng.integers(200_000, 800_000, n).astype(float) * (vol_scale / 1_000_000)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_min_bars_and_versions_frozen():
    assert scanner_p6.MIN_BARS == 50
    assert scanner_p6.RULE_VERSION_SQZ == "sqz-kc20-1.5-v1"
    assert scanner_p6.RULE_VERSION_ZKN == "zkn-p1-birebir-v1"
    assert scanner_p6.RULE_VERSION_WYC == "wyc-v1"


@pytest.mark.parametrize("etiket,ind,beklenen", SQZ_YES + SQZ_NO)
def test_sqz_5plus5(etiket, ind, beklenen):
    assert scanner_p6.strategy_sqz(ind) is beklenen, etiket


@pytest.mark.parametrize("etiket,ind,beklenen", ZKN_YES + ZKN_NO)
def test_zkn_5plus5(etiket, ind, beklenen):
    assert scanner_p6.strategy_zkn(ind) is beklenen, etiket


@pytest.mark.parametrize("etiket,ind,beklenen", WYC_YES + WYC_NO)
def test_wyc_5plus5(etiket, ind, beklenen):
    assert scanner_p6.strategy_wyc(ind) is beklenen, etiket


def test_boundary_squeeze():
    assert scanner_p6.squeeze_on_bar(100.0, 100.0, 90.0, 80.0) is False
    assert scanner_p6.squeeze_on_bar(110.0, 120.0, 80.0, 80.0) is False
    assert scanner_p6.squeeze_on_bar(110.0, 120.0, 90.0, 80.0) is True


def test_expanded_filenames_do_not_overwrite_17():
    assert EXPANDED_LEDGER_NAME != "p6_signal_ledger.jsonl"
    assert EXPANDED_META_NAME != "p6_replay_meta.json"
    assert ORIGINAL_LEDGER.name == "p6_signal_ledger.jsonl"


def test_schema_and_unique_pk_on_long_window(tmp_path: Path):
    frames = {"AAA": _ohlcv(n=200, seed=1), "BBB": _ohlcv(n=200, seed=2)}
    rows, stats = replay_asof(frames, universe=["AAA", "BBB"])
    assert stats["valid_replay_dates"] > 11
    keys = [(r["date"], r["symbol"], r["agent"]) for r in rows]
    assert len(keys) == len(set(keys))
    for r in rows:
        assert r["signal"] is True
        assert set(r) >= {
            "date", "symbol", "agent", "signal", "rule_version",
            "close", "bb_up", "recent_squeeze", "support",
        }
        assert r["purpose"] == LEDGER_PURPOSE
        assert "score" not in r and "rank" not in r
    path = tmp_path / EXPANDED_LEDGER_NAME
    write_ledger(path, rows)
    assert not (tmp_path / "p6_signal_ledger.jsonl").exists()


def test_determinism_long_window():
    frames = {"AAA": _ohlcv(n=180, seed=3), "BBB": _ohlcv(n=180, seed=4)}
    a, sa = replay_asof(frames, universe=["AAA", "BBB"])
    b, sb = replay_asof(frames, universe=["AAA", "BBB"])
    assert ledgers_equal(a, b)
    assert sa["signal_count"] == sb["signal_count"]
    assert sa["evaluation_count"] == sb["evaluation_count"]


def test_asof_lookahead_long_window():
    df = _ohlcv(n=120, seed=9)
    df2 = df.copy()
    last = df2.index[-1]
    df2.loc[last, ["open", "high", "low", "close", "volume"]] = [9999, 10000, 0.01, 9999.5, 1.0]
    asof = df.index[-2]
    chk = frames_look_ahead_equal("AAA", df2, asof)
    assert chk["agents_equal"]
    assert chk["sqz_signal_equal"]
    assert chk["recent_squeeze_equal"]
    assert chk["bb_up_equal"]
    assert chk["squeeze_on_prefix_equal"]
    assert chk["zkn_equal"]
    assert chk["wyc_equal"]
    fails = look_ahead_scan({"AAA": df2})
    assert fails == []


def test_missing_bar_not_stale_evaluated():
    df = _ohlcv(n=80, seed=5)
    gap_day = df.index[60]
    df2 = df.drop(gap_day)
    frames = {"AAA": df2}
    rows, _stats = replay_asof(frames, universe=["AAA"])
    assert all(r["date"] != str(gap_day.date()) for r in rows)


def test_horizon_coverage_counts_not_returns():
    df = _ohlcv(n=80, seed=6)
    last = str(df.index[-1].date())
    mid = str(df.index[-6].date())
    early = str(df.index[-25].date())
    rows = [
        {"date": last, "symbol": "AAA", "agent": "SQZ"},
        {"date": mid, "symbol": "AAA", "agent": "SQZ"},
        {"date": early, "symbol": "AAA", "agent": "SQZ"},
    ]
    frames = {"AAA": df}
    cov = horizon_coverage(rows, frames, agent="SQZ")
    assert cov["raw_signals"] == 3
    assert cov["horizons"]["1"]["usable"] == 2
    assert cov["horizons"]["5"]["usable"] == 2
    assert cov["horizons"]["20"]["usable"] == 1
    assert "mean" not in json.dumps(cov)
    assert future_horizon_available(df, last, 1) == "insufficient_future"


def test_horizons_tuple():
    assert HORIZONS == (1, 3, 5, 10, 20)


def test_closed_cutoff_drops_unclosed_today():
    df = _ohlcv(n=10, start="2026-08-17")
    frames = {"AAA": df}
    now = pd.Timestamp("2026-08-26 16:00", tz="Europe/Istanbul")
    cut = last_closed_asof_cutoff(now)
    assert cut == pd.Timestamp("2026-08-25")
    clipped, used = apply_closed_cutoff(frames, cutoff=cut)
    assert used == cut
    assert clipped["AAA"].index.max() <= cut


def test_no_fwd_return_in_expanded_module():
    src = Path(__file__).resolve().parents[1] / "p6_replay_expanded.py"
    text = src.read_text(encoding="utf-8")
    assert "_fwd_return_hesapla" not in text
    assert "np.log" not in text
    assert EXPANDED_LEDGER_NAME in text
