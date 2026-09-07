"""
P6 Etap 1.3 — WYC (wyc-v1) Boolean, support göstergesi, look-ahead, kenar.

Boolean katmanı: sentetik ind → strategy_wyc(ind). OHLCV yok.
Gösterge katmanı: sentetik OHLCV → get_indicators_p6 → support / strategy_wyc.
Look-ahead: t önekini kes, t+1 ekle/mutate et, t önekini yeniden hesapla.
P1 scanner_p1 / portfoy_yonetici çağrılarak state yazılmaz.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import scanner_p1
import scanner_p6
from p6_scenarios import WYC_NO, WYC_YES, wyc_base_ind

ZKN_KEYS = scanner_p6.ZKN_CRITICAL_KEYS
SUPPORT_WINDOW = scanner_p6.SUPPORT_WINDOW


def _wick(ind: dict) -> float:
    rng = ind["high"] - ind["low"]
    return (min(ind["open"], ind["close"]) - ind["low"]) / rng


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


def _spring_ohlcv(
    *,
    n=60,
    window_low=100.0,
    unique_min=95.0,
    unique_offset=5,
    t_open=96.0,
    t_high=110.0,
    t_low=90.0,
    t_close=105.0,
    volume=1_000_000.0,
) -> pd.DataFrame:
    """
    Son bar t. support(t) = min(low[t-10..t-1]).
    Pencere içindeki tek düşük bar unique_min; diğerleri window_low.
    t mumu varsayılan geçerli yeşil spring.
    unique_offset: t-k (1..10); 5 → t-5.
    """
    assert 1 <= unique_offset <= SUPPORT_WINDOW
    dates = pd.bdate_range("2025-01-06", periods=n)
    open_ = np.full(n, 101.0)
    high = np.full(n, 102.0)
    low = np.full(n, window_low)
    close = np.full(n, 101.0)
    vol = np.full(n, volume)
    min_i = n - 1 - unique_offset
    low[min_i] = unique_min
    open_[min_i] = unique_min + 1.0
    high[min_i] = unique_min + 2.0
    close[min_i] = unique_min + 1.0
    t = n - 1
    open_[t] = t_open
    high[t] = t_high
    low[t] = t_low
    close[t] = t_close
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
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


def _snapshot_existing(ind: dict) -> dict:
    keys = list(ZKN_KEYS) + ["open", "high", "low", "close", "support"]
    out = {}
    for k in keys:
        v = ind[k]
        out[k] = v if pd.isna(v) else float(v)
    return out


def _snapshots_equal(a: dict, b: dict) -> None:
    for k in a:
        va, vb = a[k], b[k]
        if pd.isna(va) and pd.isna(vb):
            continue
        assert va == pytest.approx(vb, rel=0, abs=1e-12), k


# ---------------------------------------------------------------------------
# İmza / dönüş tipi / MC yok
# ---------------------------------------------------------------------------

def test_strategy_wyc_imzasi_ind_only():
    params = list(inspect.signature(scanner_p6.strategy_wyc).parameters)
    assert params == ["ind"]


def test_strategy_wyc_return_is_python_bool():
    assert type(scanner_p6.strategy_wyc(wyc_base_ind())) is bool
    assert type(scanner_p6.strategy_wyc(wyc_base_ind(support=float("nan")))) is bool


def test_strategy_wyc_mc_parametresi_yok():
    src = inspect.getsource(scanner_p6.strategy_wyc)
    assert "mc" not in src.split("(")[1].split(")")[0]


# ---------------------------------------------------------------------------
# 5+5 Boolean (direkt ind; OHLCV yok)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("etiket,ind,beklenen", WYC_YES + WYC_NO)
def test_wyc_5plus5_direkt_ind(etiket, ind, beklenen):
    assert scanner_p6.strategy_wyc(ind) is beklenen, etiket


def test_wyc_yes3_wick_exactly_030():
    ind = dict(WYC_YES[2][1])
    assert _wick(ind) == pytest.approx(0.30)
    assert scanner_p6.strategy_wyc(ind) is True


def test_wyc_yes2_low_strictly_below_support():
    ind = dict(WYC_YES[1][1])
    assert ind["low"] < ind["support"]
    assert ind["low"] != ind["support"]
    assert scanner_p6.strategy_wyc(ind) is True


def test_wyc_yes4_green_yes5_red():
    green = dict(WYC_YES[3][1])
    red = dict(WYC_YES[4][1])
    assert green["close"] > green["open"]
    assert red["close"] < red["open"]
    assert scanner_p6.strategy_wyc(green) is True
    assert scanner_p6.strategy_wyc(red) is True


# ---------------------------------------------------------------------------
# Kenar (Boolean)
# ---------------------------------------------------------------------------

def test_edge_support_nan_false():
    assert scanner_p6.strategy_wyc(wyc_base_ind(support=float("nan"))) is False


def test_edge_high_eq_low_false():
    assert scanner_p6.strategy_wyc(
        wyc_base_ind(open=100.0, high=100.0, low=100.0, close=100.0, support=99.0)
    ) is False


def test_edge_low_eq_support_false():
    # close>support ve wick>=0.30 korunur; yalnız eşitlik kırılır
    ind = wyc_base_ind(low=100.0, open=106.0, high=110.0, close=105.0, support=100.0)
    assert ind["low"] == ind["support"]
    assert ind["close"] > ind["support"]
    assert _wick(ind) >= 0.30
    assert scanner_p6.strategy_wyc(ind) is False


def test_edge_close_eq_support_false():
    ind = wyc_base_ind(low=90.0, open=110.0, high=110.0, close=100.0, support=100.0)
    assert ind["close"] == ind["support"]
    assert ind["low"] < ind["support"]
    assert _wick(ind) >= 0.30
    assert scanner_p6.strategy_wyc(ind) is False


def test_edge_wick_eq_030_true():
    ind = wyc_base_ind()
    assert _wick(ind) == pytest.approx(0.30)
    assert scanner_p6.strategy_wyc(ind) is True


def test_edge_wick_lt_030_false():
    ind = wyc_base_ind(open=92.0, high=110.0, low=90.0, close=105.0, support=100.0)
    assert _wick(ind) == pytest.approx(0.10)
    assert ind["low"] < ind["support"]
    assert ind["close"] > ind["support"]
    assert scanner_p6.strategy_wyc(ind) is False


def test_edge_low_below_but_close_le_support_false():
    ind = wyc_base_ind(open=110.0, high=120.0, low=90.0, close=99.0, support=100.0)
    assert ind["low"] < ind["support"]
    assert ind["close"] <= ind["support"]
    assert scanner_p6.strategy_wyc(ind) is False


def test_edge_valid_green_spring_true():
    ind = wyc_base_ind(open=96.0, close=105.0)
    assert ind["close"] > ind["open"]
    assert scanner_p6.strategy_wyc(ind) is True


def test_edge_valid_red_spring_true():
    ind = wyc_base_ind(open=110.0, close=101.0, high=110.0, low=90.0)
    assert ind["close"] < ind["open"]
    assert scanner_p6.strategy_wyc(ind) is True


# ---------------------------------------------------------------------------
# NaN / eksik alan fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["open", "high", "low", "close", "support"])
def test_wyc_required_nan_false(key):
    assert scanner_p6.strategy_wyc(wyc_base_ind(**{key: float("nan")})) is False


def test_wyc_empty_dict_false():
    assert scanner_p6.strategy_wyc({}) is False


def test_wyc_non_dict_false():
    assert scanner_p6.strategy_wyc(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Gösterge: support formülü, off-by-one, 10-bar min
# ---------------------------------------------------------------------------

def test_support_matches_independent_t_minus_10_to_t_minus_1():
    df = _spring_ohlcv(n=40, unique_min=87.3, unique_offset=10, t_low=80.0)
    ind = scanner_p6.get_indicators_p6(df)
    lows = df["low"].to_numpy()
    expected = float(np.min(lows[-11:-1]))  # t-10 … t-1; t hariç
    assert ind["support"] == pytest.approx(expected)
    assert expected == pytest.approx(87.3)
    assert ind["low"] < ind["support"]
    assert scanner_p6.strategy_wyc(ind) is True


def test_support_current_bar_low_not_included():
    df = _spring_ohlcv(n=40, unique_min=95.0, t_low=90.0)
    before = scanner_p6.get_indicators_p6(df)["support"]
    df = df.copy()
    df.iloc[-1, df.columns.get_loc("low")] = 0.01
    after = scanner_p6.get_indicators_p6(df)["support"]
    assert before == pytest.approx(after)
    assert before == pytest.approx(95.0)
    assert after != pytest.approx(0.01)


def test_support_ten_prior_bars_min_enters():
    df = _spring_ohlcv(n=40, window_low=100.0, unique_min=88.0, unique_offset=7)
    ind = scanner_p6.get_indicators_p6(df)
    t = len(df) - 1
    window = df["low"].iloc[t - 10 : t]  # t-10 … t-1
    assert len(window) == 10
    assert float(window.min()) == pytest.approx(88.0)
    assert ind["support"] == pytest.approx(88.0)
    assert df["low"].iloc[t] == pytest.approx(90.0)
    assert df["low"].iloc[t] != ind["support"]


def test_support_nan_when_history_short():
    df = _ohlcv(n=10)
    ind = scanner_p6.get_indicators_p6(df)
    assert pd.isna(ind["support"])
    assert scanner_p6.strategy_wyc(ind) is False


def test_support_finite_at_eleven_bars():
    df = _spring_ohlcv(n=11, unique_min=95.0, unique_offset=3)
    ind = scanner_p6.get_indicators_p6(df)
    assert not pd.isna(ind["support"])
    assert ind["support"] == pytest.approx(95.0)


def test_indicator_pipeline_spring_true():
    df = _spring_ohlcv(n=40)
    ind = scanner_p6.get_indicators_p6(df)
    assert ind["support"] == pytest.approx(95.0)
    assert ind["low"] == pytest.approx(90.0)
    assert ind["close"] == pytest.approx(105.0)
    assert _wick(ind) == pytest.approx(0.30)
    assert scanner_p6.strategy_wyc(ind) is True


def test_support_not_unshifted_rolling_min():
    """low.rolling(10).min() (mevcut bar dahil) yasak; shift(1) zorunlu."""
    df = _spring_ohlcv(n=30, window_low=100.0, unique_min=100.0, t_low=1.0)
    # Pencere t-10..t-1 hepsi 100; t low=1. Unshifted rolling min = 1; shifted = 100.
    unshifted = float(df["low"].rolling(10, min_periods=10).min().iloc[-1])
    shifted = float(df["low"].rolling(10, min_periods=10).min().shift(1).iloc[-1])
    ind = scanner_p6.get_indicators_p6(df)
    assert unshifted == pytest.approx(1.0)
    assert shifted == pytest.approx(100.0)
    assert ind["support"] == pytest.approx(shifted)
    assert ind["support"] != pytest.approx(unshifted)


# ---------------------------------------------------------------------------
# Look-ahead: t öneki vs t+1 (son barı mutate edip aynı son-bar sinyali YOK)
# ---------------------------------------------------------------------------

def test_lookahead_append_tplus1_does_not_change_t_prefix():
    df_t = _spring_ohlcv(n=60)
    ind_t = scanner_p6.get_indicators_p6(df_t)
    snap_t = _snapshot_existing(ind_t)
    wyc_t = scanner_p6.strategy_wyc(ind_t)

    df_t1 = _append_extreme_bar(df_t)
    prefix_again = df_t1.iloc[: len(df_t)]
    ind_again = scanner_p6.get_indicators_p6(prefix_again)
    snap_again = _snapshot_existing(ind_again)

    _snapshots_equal(snap_t, snap_again)
    assert scanner_p6.strategy_wyc(ind_again) is wyc_t
    # t+1 eklendikten sonra t öneki değişmez; son-bar (t+1) sinyali bu testin konusu değil
    assert len(df_t1) == len(df_t) + 1
    assert prefix_again.equals(df_t)


def test_lookahead_mutate_tplus1_does_not_change_t_prefix():
    df = _spring_ohlcv(n=61)
    asof = df.index[-2]
    prefix = scanner_p6.truncate_to_asof(df, asof)
    ind_a = scanner_p6.get_indicators_p6(prefix)
    snap_a = _snapshot_existing(ind_a)
    wyc_a = scanner_p6.strategy_wyc(ind_a)

    df_mut = df.copy()
    last = df_mut.index[-1]
    df_mut.loc[last, ["open", "high", "low", "close", "volume"]] = [
        9_999.0, 10_000.0, 0.01, 9_999.5, 1.0,
    ]
    prefix_b = scanner_p6.truncate_to_asof(df_mut, asof)
    ind_b = scanner_p6.get_indicators_p6(prefix_b)
    _snapshots_equal(snap_a, _snapshot_existing(ind_b))
    assert scanner_p6.strategy_wyc(ind_b) is wyc_a


def test_zkn_seven_fields_unchanged_on_tplus1_append():
    df_t = _ohlcv(n=80, seed=3)
    p1_t = scanner_p1.get_indicators(df_t)
    p6_t = scanner_p6.get_indicators_p6(df_t)
    df_t1 = _append_extreme_bar(df_t)
    p6_again = scanner_p6.get_indicators_p6(df_t1.iloc[: len(df_t)])
    for key in ZKN_KEYS:
        va, vb = p6_t[key], p6_again[key]
        if pd.isna(va) and pd.isna(vb):
            continue
        assert va == pytest.approx(vb, rel=0, abs=1e-12), key
        if p1_t is not None:
            p1v = p1_t[key]
            if pd.isna(p6_t[key]) and pd.isna(p1v):
                continue
            assert p6_t[key] == pytest.approx(p1v, rel=0, abs=1e-12), key


# ---------------------------------------------------------------------------
# Orkestratör: MIN_BARS=50 kalır; 11 bar matematik ısınması eşik değiştirmez
# ---------------------------------------------------------------------------

def test_min_bars_still_50_despite_support_warmup_11():
    df11 = _spring_ohlcv(n=11)
    ind11 = scanner_p6.get_indicators_p6(df11)
    assert not pd.isna(ind11["support"])
    assert scanner_p6.strategy_wyc(ind11) is True
    assert scanner_p6.universe_ok(df11) is False
    assert scanner_p6.evaluate_symbol("X", df11) == []

    df49 = _spring_ohlcv(n=49, volume=1_000_000.0)
    assert scanner_p6.universe_ok(df49) is False
    assert scanner_p6.evaluate_symbol("X", df49) == []

    df50 = _spring_ohlcv(n=50, volume=1_000_000.0)
    # close[-1]=105, vol_ma20=1e6 → mc = 105e6 >= 10M
    assert scanner_p6.universe_ok(df50) is True
    recs = scanner_p6.evaluate_symbol("X", df50)
    wyc_recs = [r for r in recs if r["strategy"] == "WYC"]
    assert len(wyc_recs) == 1
    assert wyc_recs[0]["rule_version"] == scanner_p6.RULE_VERSION_WYC
    assert "score" not in wyc_recs[0]
    assert "mc" not in wyc_recs[0]


def test_wyc_independent_of_zkn():
    ind = wyc_base_ind()
    assert scanner_p6.strategy_wyc(ind) is True
    # ZKN alanları yok → ZKN False; WYC etkilenmez
    assert scanner_p6.strategy_zkn(ind) is False


# ---------------------------------------------------------------------------
# Runtime: exception yok, dönüş bool
# ---------------------------------------------------------------------------

def test_runtime_wyc_bool_no_exception():
    rng = np.random.default_rng(1)
    for i in range(400):
        if i % 8 == 0:
            ind = wyc_base_ind(support=float("nan"))
        elif i % 8 == 1:
            ind = wyc_base_ind(high=90.0, low=90.0)
        else:
            ind = wyc_base_ind(
                open=float(rng.uniform(80, 120)),
                high=float(rng.uniform(90, 130)),
                low=float(rng.uniform(70, 100)),
                close=float(rng.uniform(80, 120)),
                support=float(rng.uniform(80, 120)),
            )
        out = scanner_p6.strategy_wyc(ind)
        assert type(out) is bool
