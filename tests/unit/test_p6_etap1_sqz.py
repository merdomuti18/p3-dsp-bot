"""
P6 Etap 1.5 — SQZ-v1 Boolean, squeeze penceresi, look-ahead, kenar.

Boolean katmanı: sentetik ind → strategy_sqz(ind).
Pencere katmanı: sentetik squeeze_on serisi → recent_squeeze_from_series.
Gösterge: OHLCV → get_indicators_p6 (bb_* P1 ile aynı; ZKN/support overwrite yok).
Look-ahead: t önekini kes, t+1 ekle, t önekini yeniden hesapla.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import scanner_p1
import scanner_p6
from p6_scenarios import SQZ_NO, SQZ_YES, sqz_base_ind

ZKN_KEYS = scanner_p6.ZKN_CRITICAL_KEYS


def _ohlcv(n=64, seed=42, vol_scale=1_000_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-06", periods=n)
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


def _squeeze_flags(*, n=10, true_at=()):
    """n bar; son bar t. true_at: t'ye göre negatif ofset (1=t-1, 5=t-5, 0=t)."""
    flags = [False] * n
    for off in true_at:
        flags[n - 1 - off] = True
    return pd.Series(flags, dtype=bool)


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


def _tight_range_ohlcv(n=60, last_close=None) -> pd.DataFrame:
    """Düz fiyat + geniş wick → BB dar, KC görece geniş (squeeze)."""
    dates = pd.bdate_range("2025-01-06", periods=n)
    close = np.full(n, 100.0)
    close = close + np.linspace(0.0, 0.02, n)
    if last_close is not None:
        close[-1] = last_close
    open_ = close.copy()
    high = close + 1.0
    low = close - 1.0
    high[-1] = max(close[-1], open_[-1]) + 0.5
    low[-1] = min(close[-1], open_[-1]) - 0.5
    vol = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


# ---------------------------------------------------------------------------
# İmza / rel_vol yok / dönüş tipi
# ---------------------------------------------------------------------------

def test_strategy_sqz_imzasi_ind_only():
    params = list(inspect.signature(scanner_p6.strategy_sqz).parameters)
    assert params == ["ind"]


def test_strategy_sqz_return_is_python_bool():
    assert type(scanner_p6.strategy_sqz(sqz_base_ind())) is bool
    assert type(scanner_p6.strategy_sqz(sqz_base_ind(bb_up=float("nan")))) is bool


def test_strategy_sqz_ignores_rel_vol():
    assert scanner_p6.strategy_sqz(sqz_base_ind(rel_vol=0.0)) is True
    src = inspect.getsource(scanner_p6.strategy_sqz)
    body = src.split("try:", 1)[1]
    assert "rel_vol" not in body
    assert "vol_ma" not in body
    assert "close > kc" not in src


def test_min_bars_still_50():
    assert scanner_p6.MIN_BARS == 50


# ---------------------------------------------------------------------------
# 5+5 Boolean
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("etiket,ind,beklenen", SQZ_YES + SQZ_NO)
def test_sqz_5plus5_direkt_ind(etiket, ind, beklenen):
    assert scanner_p6.strategy_sqz(ind) is beklenen, etiket


def test_yes5_squeeze_on_t_not_required():
    ind = sqz_base_ind(recent_squeeze=True, squeeze_on=False, close=101.0, bb_up=100.0)
    assert scanner_p6.strategy_sqz(ind) is True


def test_no3_only_current_squeeze_false():
    ind = sqz_base_ind(recent_squeeze=False, squeeze_on=True, close=101.0, bb_up=100.0)
    assert scanner_p6.strategy_sqz(ind) is False


def test_no1_close_eq_bb_up_false():
    ind = sqz_base_ind(close=100.0, bb_up=100.0, recent_squeeze=True)
    assert scanner_p6.strategy_sqz(ind) is False


# ---------------------------------------------------------------------------
# Pencere: iloc[-6:-1]
# ---------------------------------------------------------------------------

def test_window_tminus5_only_true():
    s = _squeeze_flags(n=12, true_at=(5,))
    assert bool(s.iloc[-6]) is True
    assert bool(s.iloc[-1]) is False
    assert scanner_p6.recent_squeeze_from_series(s) is True
    ind = sqz_base_ind(recent_squeeze=True, close=101.0, bb_up=100.0)
    assert scanner_p6.strategy_sqz(ind) is True


def test_window_tminus1_only_true():
    s = _squeeze_flags(n=12, true_at=(1,))
    assert bool(s.iloc[-2]) is True
    assert bool(s.iloc[-1]) is False
    assert scanner_p6.recent_squeeze_from_series(s) is True


def test_window_t_only_false():
    s = _squeeze_flags(n=12, true_at=(0,))
    assert bool(s.iloc[-1]) is True
    assert scanner_p6.recent_squeeze_from_series(s) is False
    ind = sqz_base_ind(recent_squeeze=False, squeeze_on=True, close=101.0, bb_up=100.0)
    assert scanner_p6.strategy_sqz(ind) is False


def test_window_all_false_false():
    s = _squeeze_flags(n=12, true_at=())
    assert scanner_p6.recent_squeeze_from_series(s) is False


def test_window_insufficient_length_false():
    s = pd.Series([True, True, True, True, True], dtype=bool)
    assert len(s) < 6
    assert scanner_p6.recent_squeeze_from_series(s) is False


def test_window_slice_is_iloc_m6_m1():
    flags = [False] * 10
    flags[4] = True  # t-5 when n=10
    s = pd.Series(flags, dtype=bool)
    window = s.iloc[-6:-1]
    assert list(window) == [True, False, False, False, False]
    assert scanner_p6.SQZ_RECENT_START == -6
    assert scanner_p6.SQZ_RECENT_END == -1


# ---------------------------------------------------------------------------
# Boundary squeeze_on
# ---------------------------------------------------------------------------

def test_boundary_bb_up_eq_kc_up_false():
    assert scanner_p6.squeeze_on_bar(100.0, 100.0, 90.0, 80.0) is False


def test_boundary_bb_lo_eq_kc_lo_false():
    assert scanner_p6.squeeze_on_bar(110.0, 120.0, 80.0, 80.0) is False


def test_boundary_strict_inside_true():
    assert scanner_p6.squeeze_on_bar(110.0, 120.0, 90.0, 80.0) is True


def test_boundary_bb_lo_eq_only_false():
    assert scanner_p6.squeeze_on_bar(110.0, 120.0, 80.0, 80.0) is False


def test_boundary_bb_up_eq_only_false():
    assert scanner_p6.squeeze_on_bar(120.0, 120.0, 90.0, 80.0) is False


def test_boundary_nan_false():
    assert scanner_p6.squeeze_on_bar(float("nan"), 120.0, 90.0, 80.0) is False
    assert scanner_p6.squeeze_on_bar(110.0, float("nan"), 90.0, 80.0) is False


# ---------------------------------------------------------------------------
# Gösterge: P1 BB last-bar, ZKN/support overwrite yok
# ---------------------------------------------------------------------------

def test_p1_bb_last_bar_not_overwritten():
    df = _ohlcv(n=80, seed=3)
    p1 = scanner_p1.get_indicators(df)
    p6 = scanner_p6.get_indicators_p6(df)
    for key in ("bb_mid", "bb_up", "bb_lo"):
        assert p6[key] == pytest.approx(p1[key], rel=0, abs=1e-12), key
    _mid, bb_up_s, bb_lo_s = scanner_p1._bbands(df["close"])
    assert p6["bb_up"] == pytest.approx(float(bb_up_s.iloc[-1]), rel=0, abs=1e-12)
    assert p6["bb_lo"] == pytest.approx(float(bb_lo_s.iloc[-1]), rel=0, abs=1e-12)


def test_zkn_seven_fields_and_support_unchanged_vs_p1():
    df = _ohlcv(n=80, seed=4)
    p1 = scanner_p1.get_indicators(df)
    p6 = scanner_p6.get_indicators_p6(df)
    for key in ZKN_KEYS:
        va, vb = p1[key], p6[key]
        if pd.isna(va) and pd.isna(vb):
            continue
        assert va == pytest.approx(vb, rel=0, abs=1e-12), key
    expected_support = float(
        df["low"].rolling(10, min_periods=10).min().shift(1).iloc[-1]
    )
    assert p6["support"] == pytest.approx(expected_support, rel=0, abs=1e-12)


def test_sqz_fill_does_not_mutate_zkn_keys():
    df = _ohlcv(n=80, seed=5)
    p6 = scanner_p6.get_indicators_p6(df)
    out = {k: p6[k] for k in ZKN_KEYS}
    scanner_p6._fill_sqz_from_df(p6, df)
    for key in ZKN_KEYS:
        va, vb = out[key], p6[key]
        if pd.isna(va) and pd.isna(vb):
            continue
        assert va == pytest.approx(vb, rel=0, abs=1e-12), key


def test_pipeline_recent_squeeze_matches_series_helper():
    df = _ohlcv(n=80, seed=6)
    ind = scanner_p6.get_indicators_p6(df)
    sq = scanner_p6.squeeze_on_series(df)
    assert ind["recent_squeeze"] is scanner_p6.recent_squeeze_from_series(sq)
    assert scanner_p6.strategy_sqz(ind) is (
        bool(ind["recent_squeeze"]) and ind["close"] > ind["bb_up"]
    )


def test_nan_bb_up_false():
    assert scanner_p6.strategy_sqz(sqz_base_ind(bb_up=float("nan"))) is False


def test_nan_close_false():
    assert scanner_p6.strategy_sqz(sqz_base_ind(close=float("nan"))) is False


def test_nan_recent_false():
    assert scanner_p6.strategy_sqz(sqz_base_ind(recent_squeeze=float("nan"))) is False


def test_empty_dict_false():
    assert scanner_p6.strategy_sqz({}) is False


def test_short_history_not_true():
    df = _ohlcv(n=5)
    ind = scanner_p6.get_indicators_p6(df)
    assert scanner_p6.strategy_sqz(ind) is False
    assert scanner_p6.universe_ok(df) is False


# ---------------------------------------------------------------------------
# Look-ahead
# ---------------------------------------------------------------------------

def test_lookahead_append_tplus1_does_not_change_t_prefix():
    df_t = _ohlcv(n=80, seed=11)
    ind_t = scanner_p6.get_indicators_p6(df_t)
    sqz_t = scanner_p6.strategy_sqz(ind_t)
    snap = {k: ind_t[k] for k in list(ZKN_KEYS) + ["support", "bb_up", "recent_squeeze"]}
    df_t1 = _append_extreme_bar(df_t)
    ind_again = scanner_p6.get_indicators_p6(df_t1.iloc[: len(df_t)])
    assert scanner_p6.strategy_sqz(ind_again) is sqz_t
    for k, v in snap.items():
        va, vb = v, ind_again[k]
        if pd.isna(va) and pd.isna(vb):
            continue
        if isinstance(va, (bool, np.bool_)) or isinstance(vb, (bool, np.bool_)):
            assert bool(va) is bool(vb), k
        else:
            assert va == pytest.approx(float(vb), rel=0, abs=1e-12), k


def test_lookahead_window_not_reinterpreted_on_full_series():
    """Tam serideki son-bar penceresi t'den farklı olabilir; t öneki korunur."""
    df = _ohlcv(n=81, seed=12)
    asof = df.index[-2]
    prefix = scanner_p6.truncate_to_asof(df, asof)
    a = scanner_p6.strategy_sqz(scanner_p6.get_indicators_p6(prefix))
    df_mut = df.copy()
    last = df_mut.index[-1]
    df_mut.loc[last, ["open", "high", "low", "close", "volume"]] = [
        9_999.0, 10_000.0, 0.01, 9_999.5, 1.0,
    ]
    prefix_b = scanner_p6.truncate_to_asof(df_mut, asof)
    b = scanner_p6.strategy_sqz(scanner_p6.get_indicators_p6(prefix_b))
    assert a is b


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

def test_runtime_sqz_bool_no_exception():
    rng = np.random.default_rng(2)
    for i in range(300):
        if i % 7 == 0:
            ind = sqz_base_ind(bb_up=float("nan"))
        elif i % 7 == 1:
            ind = sqz_base_ind(recent_squeeze=False)
        else:
            ind = sqz_base_ind(
                close=float(rng.uniform(90, 110)),
                bb_up=float(rng.uniform(90, 110)),
                recent_squeeze=bool(rng.integers(0, 2)),
            )
        out = scanner_p6.strategy_sqz(ind)
        assert type(out) is bool


def test_evaluate_emits_sqz_independently():
    df = _tight_range_ohlcv(n=50, last_close=108.0)
    recs = scanner_p6.evaluate_symbol("X", df)
    sqz_recs = [r for r in recs if r["strategy"] == "SQZ"]
    ind = scanner_p6.get_indicators_p6(df)
    if scanner_p6.strategy_sqz(ind):
        assert len(sqz_recs) == 1
        assert sqz_recs[0]["rule_version"] == scanner_p6.RULE_VERSION_SQZ
        assert "score" not in sqz_recs[0]
        assert "mc" not in sqz_recs[0]
    else:
        assert sqz_recs == []
