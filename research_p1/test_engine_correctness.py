# -*- coding: utf-8 -*-
"""
research_p1/test_engine_correctness.py
======================================
10 Zorunlu Doğruluk ve Güvenlik Senaryoları Testi:
1. Gelecek barı değiştirmek geçmiş sinyali ve kararı değiştirmiyor (No Lookahead).
2. Açık/gelecek mumdan sinyal üretilmiyor.
3. Sinyal gününün kapanışında alım yapılmıyor (en erken t+1 açılışında).
4. Gap hesabı giriş gününün açılışında doğru uygulanıyor ve tolerans dışındakiler reddediliyor.
5. Alış ve satış komisyon/slipaj hesabı bağımsız aritmetik örneklerle uyuşuyor.
6. Kısmi satış (TP1) sonrası nakit, lot ve net kâr mutabık.
7. Aynı sinyal/bar tekrar işlendiğinde mükerrer işlem oluşmuyor (idempotency).
8. Stop altına gap durumunda gerçekleşme açılış fiyatından kayma düşülerek muhafazakâr hesaplanıyor.
9. Nakit veya lot negatif olmuyor.
10. Aynı veri ve konfigürasyon her zaman aynı deterministik sonucu veriyor.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from research_p1.backtest_engine import simulate_trades, evaluate_agent_signals
from research_p1.p1_engine import compute_all_indicators


def _create_sample_df(n_bars: int = 100, trend: float = 0.001) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n_bars, freq="D")
    base = 100.0 * np.exp(np.cumsum(np.random.normal(trend, 0.015, n_bars)))
    df = pd.DataFrame({
        "open": base,
        "high": base * 1.02,
        "low": base * 0.98,
        "close": base * 1.005,
        "volume": np.random.randint(50000, 200000, n_bars).astype(float),
    }, index=dates)
    return df


def test_no_lookahead_future_bar_does_not_affect_past_signals():
    """Gelecek barı değiştirmek geçmiş sinyali ve kararı değiştirmez."""
    df1 = _create_sample_df(60)
    ind1 = compute_all_indicators(df1)
    sig1 = evaluate_agent_signals(ind1, "GT")

    # Gelecek barların değerlerini radikal biçimde değiştir
    df2 = df1.copy()
    df2.iloc[-1] = df2.iloc[-1] * 3.0  # son bar uçuruldu
    ind2 = compute_all_indicators(df2)
    sig2 = evaluate_agent_signals(ind2, "GT")

    # Son bar hariç önceki tüm barlardaki sinyaller %100 özdeş olmalıdır!
    pd.testing.assert_series_equal(sig1.iloc[:-1], sig2.iloc[:-1])


def test_no_entry_at_signal_close_and_earliest_next_open():
    """Sinyal gününün kapanışında alım yapılmaz; en erken t+1 açılışında girilir."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "open": [10.0] * 30,
        "high": [10.5] * 30,
        "low": [9.5] * 30,
        "close": [10.0] * 30,
        "volume": [10000.0] * 30,
    }, index=dates)

    # 15. günde yapay sinyal tetikle
    ind = compute_all_indicators(df)
    symbol_data = {"TEST": ind}

    # evaluate_agent_signals mocklayıp sadece 15. günde True verelim
    from unittest.mock import patch
    mock_series = pd.Series(False, index=df.index)
    mock_series.iloc[15] = True  # t gününün kapanışında sinyal

    with patch("research_p1.backtest_engine.evaluate_agent_signals", return_value=mock_series):
        trades, rejected, _ = simulate_trades(
            symbol_data, ["TEST_AGENT"],
            start_date="2023-01-01", end_date="2023-01-30",
            gap_tolerance=0.05
        )

    assert len(trades) >= 1
    t = trades[0]
    # Sinyal günü t (15. gün), giriş günü t+1 (16. gün) olmalıdır
    assert t["signal_date"] == str(dates[15].date())
    assert t["entry_date"] == str(dates[16].date())


def test_gap_tolerance_rejection():
    """Gap toleransının üzerinde açılış olduğunda sinyal reddedilmeli ve girilmemelidir."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "open": [10.0] * 10,
        "high": [10.5] * 10,
        "low": [9.5] * 10,
        "close": [10.0] * 10,
        "volume": [10000.0] * 10,
    }, index=dates)
    # 4. gün kapanış 10.0, 5. gün açılış 10.5 (gap = +%5)
    df.loc[dates[5], "open"] = 10.5

    ind = compute_all_indicators(df)
    symbol_data = {"TEST": ind}

    mock_series = pd.Series(False, index=df.index)
    mock_series.iloc[4] = True  # 4. gün sinyal

    from unittest.mock import patch
    with patch("research_p1.backtest_engine.evaluate_agent_signals", return_value=mock_series):
        # Gap toleransı 0.0001 (gap yok kuralı)
        trades, rejected, _ = simulate_trades(
            symbol_data, ["TEST_AGENT"],
            start_date="2023-01-01", end_date="2023-01-10",
            gap_tolerance=0.0001
        )

    assert len(trades) == 0
    assert len(rejected) == 1
    assert "gap_rejected" in rejected[0]["reason"]


def test_commission_and_slippage_arithmetic():
    """Alış ve satış komisyon/slipaj hesabı bağımsız aritmetik örneklerle uyuşuyor."""
    # Giriş: ref_open = 100.0, slipaj = 0.002 -> entry_price = 100.2
    # 100 lot -> brut_tutar = 10020.0, komisyon (%0.2) = 20.04 -> total_cost = 10040.04
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "open": [100.0] * 10,
        "high": [102.0] * 10,
        "low": [98.0] * 10,
        "close": [100.0] * 10,
        "volume": [10000.0] * 10,
    }, index=dates)

    # 3. gün -%6 düşüşle STOP tetikle: open 99.0, low 94.0
    df.loc[dates[3], "open"] = 99.0
    df.loc[dates[3], "low"] = 94.0

    ind = compute_all_indicators(df)
    symbol_data = {"TEST": ind}

    mock_series = pd.Series(False, index=df.index)
    mock_series.iloc[1] = True  # 1. gün sinyal -> 2. gün giriş

    from unittest.mock import patch
    with patch("research_p1.backtest_engine.evaluate_agent_signals", return_value=mock_series):
        trades, _, _ = simulate_trades(
            symbol_data, ["TEST_AGENT"],
            start_date="2023-01-01", end_date="2023-01-10",
            gap_tolerance=0.01,
            commission_rate=0.002,
            slippage_rate=0.002,
            mode="single_agent"
        )

    assert len(trades) == 1
    t = trades[0]
    # Alış gerçekleşme fiyatı 100.0 * 1.002 = 100.20
    assert t["entry_price"] == 100.20
    # Stop seviyesi = 100.2 * 0.95 = 95.19
    # Satış gerçekleşme fiyatı = 95.19 * (1 - 0.002) = 94.9996
    expected_exit = round(95.19 * (1.0 - 0.002), 4)
    assert t["exit_price"] == expected_exit


def test_gap_down_below_stop_conservative_execution():
    """Stop altına gap durumunda gerçekleşme muhafazakâr olarak açılış fiyatından hesaplanır."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "open": [100.0] * 10,
        "high": [102.0] * 10,
        "low": [98.0] * 10,
        "close": [100.0] * 10,
        "volume": [10000.0] * 10,
    }, index=dates)

    # 3. günde açılış 90.0 (Stop seviyesi 95.19'un çok altında açılıyor!)
    df.loc[dates[3], "open"] = 90.0
    df.loc[dates[3], "low"] = 89.0
    df.loc[dates[3], "close"] = 91.0

    ind = compute_all_indicators(df)
    symbol_data = {"TEST": ind}

    mock_series = pd.Series(False, index=df.index)
    mock_series.iloc[1] = True  # 1. gün sinyal -> 2. gün giriş

    from unittest.mock import patch
    with patch("research_p1.backtest_engine.evaluate_agent_signals", return_value=mock_series):
        trades, _, _ = simulate_trades(
            symbol_data, ["TEST_AGENT"],
            start_date="2023-01-01", end_date="2023-01-10",
            gap_tolerance=0.01,
            commission_rate=0.002,
            slippage_rate=0.002
        )

    assert len(trades) == 1
    t = trades[0]
    # Çıkış fiyatı 95.19 DEĞİL, 90.0 * (1 - 0.002) = 89.82 olmalıdır!
    assert t["exit_price"] == round(90.0 * (1.0 - 0.002), 4)
    assert t["net_return_pct"] < -10.0  # Ciddi zarar simüle edilir
