"""
P6 Etap 1.7 — as-of replay / signal ledger.

Network yok. Production state yazılmaz. Forward return / ranking / score yok.
Mevcut evaluate_symbol + truncate_to_asof replay edilir.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scanner_p6
from p6_replay import (
    LEDGER_PURPOSE,
    ReplayDataError,
    ReplayDuplicateError,
    assert_no_duplicates,
    frames_look_ahead_equal,
    ledger_row_from_record,
    ledgers_equal,
    normalize_ohlcv,
    replay_asof,
    sort_ledger,
    valid_replay_dates,
    write_ledger,
)
from p6_scenarios import SQZ_NO, SQZ_YES, ZKN_NO, ZKN_YES, WYC_NO, WYC_YES


def _ohlcv(n=64, seed=42, vol_scale=1_000_000.0, start="2025-01-06") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 100 + rng.normal(0, 1.0, n).cumsum()
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.5, n)
    volume = rng.integers(
        int(200_000 * vol_scale / 1_000_000),
        int(800_000 * vol_scale / 1_000_000),
        n,
    ).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _append_extreme_bar(df: pd.DataFrame) -> pd.DataFrame:
    nxt = df.index[-1] + pd.offsets.BDay()
    extra = pd.DataFrame(
        {
            "open": [9_999.0],
            "high": [10_000.0],
            "low": [0.01],
            "close": [9_999.5],
            "volume": [1.0],
        },
        index=[nxt],
    )
    return pd.concat([df, extra])


def _frames(n=70, symbols=("AAA", "BBB", "CCC")) -> dict[str, pd.DataFrame]:
    return {sym: _ohlcv(n=n, seed=10 + i, vol_scale=5_000_000.0) for i, sym in enumerate(symbols)}


# ---------------------------------------------------------------------------
# Contract: replay does not invent ranking / fwd / download
# ---------------------------------------------------------------------------

def test_replay_module_has_no_fwd_rank_yfinance():
    src = Path(__file__).resolve().parents[1] / "p6_replay.py"
    text = src.read_text(encoding="utf-8")
    assert "yfinance" not in text
    assert "_fwd_return" not in text
    assert "meta_portfolio" not in text
    assert "STRATEGY_WEIGHTS" not in text
    assert "Top-N" not in text
    assert LEDGER_PURPOSE in text


def test_min_bars_unchanged():
    assert scanner_p6.MIN_BARS == 50


def test_rule_versions_frozen():
    assert scanner_p6.RULE_VERSION_SQZ == "sqz-kc20-1.5-v1"
    assert scanner_p6.RULE_VERSION_ZKN == "zkn-p1-birebir-v1"
    assert scanner_p6.RULE_VERSION_WYC == "wyc-v1"


# ---------------------------------------------------------------------------
# 5+5 regression still imported (replay must not retune agents)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("etiket,ind,beklenen", SQZ_YES + SQZ_NO)
def test_sqz_5plus5_still_holds(etiket, ind, beklenen):
    assert scanner_p6.strategy_sqz(ind) is beklenen, etiket


@pytest.mark.parametrize("etiket,ind,beklenen", ZKN_YES + ZKN_NO)
def test_zkn_5plus5_still_holds(etiket, ind, beklenen):
    assert scanner_p6.strategy_zkn(ind) is beklenen, etiket


@pytest.mark.parametrize("etiket,ind,beklenen", WYC_YES + WYC_NO)
def test_wyc_5plus5_still_holds(etiket, ind, beklenen):
    assert scanner_p6.strategy_wyc(ind) is beklenen, etiket


# ---------------------------------------------------------------------------
# Normalize / duplicate source
# ---------------------------------------------------------------------------

def test_normalize_title_case_columns():
    df = _ohlcv(n=55)
    titled = df.rename(columns=str.title)
    out = normalize_ohlcv(titled, symbol="X")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_normalize_duplicate_timestamps_raises():
    df = _ohlcv(n=55)
    bad = pd.concat([df, df.iloc[[-1]]])
    with pytest.raises(ReplayDataError, match="duplicate timestamps"):
        normalize_ohlcv(bad, symbol="X")


# ---------------------------------------------------------------------------
# As-of / look-ahead
# ---------------------------------------------------------------------------

def test_evaluate_asof_matches_prefix():
    df = _ohlcv(n=80, seed=11, vol_scale=6_000_000)
    asof = df.index[-2]
    prefix = scanner_p6.truncate_to_asof(df, asof)
    rec_a = scanner_p6.evaluate_symbol("Y", prefix)
    rec_b = scanner_p6.evaluate_symbol("Y", df, asof=asof)
    assert [r["strategy"] for r in rec_a] == [r["strategy"] for r in rec_b]
    assert all(r["asof_date"] == pd.Timestamp(asof).date().isoformat() for r in rec_b)


def test_lookahead_full_vs_prefix_evaluate_and_sqz_state():
    df_t = _ohlcv(n=80, seed=11, vol_scale=6_000_000)
    df_full = _append_extreme_bar(df_t)
    asof = df_t.index[-1]
    chk = frames_look_ahead_equal("Y", df_full, asof)
    assert chk["agents_equal"]
    assert chk["sqz_signal_equal"]
    assert chk["recent_squeeze_equal"]
    assert chk["bb_up_equal"]
    assert chk["squeeze_on_prefix_equal"]
    assert chk["zkn_equal"]
    assert chk["wyc_equal"]
    prefix = scanner_p6.truncate_to_asof(df_full, asof)
    assert prefix.index.max() == asof
    assert df_full.index.max() > asof


def test_lookahead_tplus1_does_not_enter_indicator_at_t():
    df = _ohlcv(n=81, seed=12, vol_scale=6_000_000)
    asof = df.index[-2]
    a = frames_look_ahead_equal("Y", df, asof)
    df_mut = df.copy()
    last = df_mut.index[-1]
    df_mut.loc[last, ["open", "high", "low", "close", "volume"]] = [
        9_999.0, 10_000.0, 0.01, 9_999.5, 1.0,
    ]
    b = frames_look_ahead_equal("Y", df_mut, asof)
    assert a["agents_prefix"] == b["agents_prefix"]
    prefix = scanner_p6.truncate_to_asof(df_mut, asof)
    ind = scanner_p6.get_indicators_p6(prefix)
    assert ind["close"] != 9_999.5


def test_sqz_zkn_fields_not_overwritten_during_replay_prefix():
    df = _ohlcv(n=80, seed=4, vol_scale=6_000_000)
    import scanner_p1
    p1 = scanner_p1.get_indicators(df)
    p6 = scanner_p6.get_indicators_p6(df)
    for k in scanner_p6.ZKN_CRITICAL_KEYS:
        va, vb = p1[k], p6[k]
        if pd.isna(va) and pd.isna(vb):
            continue
        assert va == pytest.approx(vb, rel=0, abs=1e-12), k
    support_before = p6["support"]
    scanner_p6._fill_sqz_from_df(p6, df)
    assert p6["support"] == support_before or (pd.isna(p6["support"]) and pd.isna(support_before))


# ---------------------------------------------------------------------------
# Replay ledger
# ---------------------------------------------------------------------------

def test_replay_true_only_and_denominator():
    frames = _frames(n=70)
    rows, stats = replay_asof(frames, universe=list(frames))
    assert stats["evaluation_count"] > 0
    assert stats["signal_count"] == len(rows)
    assert all(r["signal"] is True for r in rows)
    assert all(r["purpose"] == LEDGER_PURPOSE for r in rows)
    assert all(r["analysis_only"] is True for r in rows)
    for r in rows:
        for k in scanner_p6.FORBIDDEN_RECORD_KEYS:
            assert k not in r


def test_replay_asof_date_is_t():
    frames = _frames(n=60)
    rows, _stats = replay_asof(frames, universe=list(frames))
    dates = {r["date"] for r in rows}
    for d in dates:
        assert len(d) == 10


def test_determinism_two_runs():
    frames = _frames(n=65)
    a, sa = replay_asof(frames, universe=list(frames))
    b, sb = replay_asof(frames, universe=list(frames))
    assert ledgers_equal(a, b)
    assert sa["signal_count"] == sb["signal_count"]
    assert sa["evaluation_count"] == sb["evaluation_count"]
    assert sa["sqz_true"] == sb["sqz_true"]


def test_duplicate_primary_key_raises():
    rec = {
        "asof_date": "2025-03-03",
        "symbol": "AAA",
        "strategy": "SQZ",
        "rule_version": scanner_p6.RULE_VERSION_SQZ,
        "indicators": {"close": 1.0, "bb_up": 0.5, "recent_squeeze": True},
    }
    row = ledger_row_from_record(rec)
    with pytest.raises(ReplayDuplicateError):
        assert_no_duplicates([row, dict(row)])


def test_write_ledger_roundtrip(tmp_path: Path):
    frames = _frames(n=60)
    rows, _stats = replay_asof(frames, universe=list(frames))
    path = tmp_path / "ledger.jsonl"
    write_ledger(path, rows)
    loaded = [__import__("json").loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert sort_ledger(loaded) == sort_ledger(rows)


def test_warmup_not_evaluated():
    frames = {"AAA": _ohlcv(n=49, seed=1, vol_scale=5_000_000)}
    rows, stats = replay_asof(frames, universe=["AAA"])
    assert rows == []
    assert stats["evaluation_count"] == 0
    assert stats["insufficient_bars"] > 0
    assert stats["valid_replay_dates"] == 0


def test_valid_dates_start_at_min_bars():
    frames = {"AAA": _ohlcv(n=55, seed=1, vol_scale=5_000_000)}
    valid = valid_replay_dates(frames)
    assert len(valid) == 6  # bars 50..55
    assert len(scanner_p6.truncate_to_asof(frames["AAA"], valid[0])) == scanner_p6.MIN_BARS


def test_missing_universe_symbol_counted():
    frames = {"AAA": _ohlcv(n=55, seed=1, vol_scale=5_000_000)}
    _rows, stats = replay_asof(frames, universe=["AAA", "ZZZ"])
    assert stats["missing_ohlcv"] > 0
    assert "ZZZ" in stats["universe_missing_in_frames"]


def test_below_mc_not_evaluated():
    df = _ohlcv(n=55, seed=2, vol_scale=5_000_000)
    df["volume"] = 1.0
    frames = {"LOW": df}
    rows, stats = replay_asof(frames, universe=["LOW"])
    assert rows == []
    assert stats["evaluation_count"] == 0
    assert stats["below_mc"] > 0


def test_evaluate_signature_still_asof():
    params = list(inspect.signature(scanner_p6.evaluate_symbol).parameters)
    assert params == ["symbol", "df", "asof"]


def test_no_score_in_evaluate_records():
    df = _ohlcv(n=60, seed=7, vol_scale=8_000_000)
    recs = scanner_p6.evaluate_symbol("THYAO", df)
    for rec in recs:
        assert "score" not in rec
        assert "rank" not in rec
        assert "strategies" not in rec
