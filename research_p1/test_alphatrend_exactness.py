# -*- coding: utf-8 -*-
"""
tests/test_alphatrend_exactness.py
==================================
AlphaTrend matematiksel ve mantıksal doğruluk testleri.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_p1.alphatrend import (
    calc_true_range,
    calc_mfi,
    calc_alphatrend_series,
)


def test_true_range_calculation():
    high = np.array([10.0, 12.0, 11.0])
    low = np.array([8.0, 9.0, 8.5])
    close = np.array([9.5, 11.5, 9.0])
    # Bar 0: 10 - 8 = 2
    # Bar 1: max(12 - 9, abs(12 - 9.5), abs(9 - 9.5)) = max(3, 2.5, 0.5) = 3
    # Bar 2: max(11 - 8.5, abs(11 - 11.5), abs(8.5 - 11.5)) = max(2.5, 0.5, 3.0) = 3.0
    tr = calc_true_range(high, low, close)
    assert np.allclose(tr, [2.0, 3.0, 3.0])


def test_alphatrend_recursiveness_and_crossover():
    # Sentetik veri
    n = 50
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "open": np.linspace(10, 20, n),
        "high": np.linspace(10.5, 20.5, n),
        "low": np.linspace(9.5, 19.5, n),
        "close": np.linspace(10, 20, n),
        "volume": np.full(n, 1000.0),
    }, index=dates)

    res = calc_alphatrend_series(df, coeff=1.0, ap=14)
    assert len(res) == n
    assert "alpha_trend" in res.columns
    assert "alpha_trend_2" in res.columns
    assert "buy_signal_confirmed" in res.columns
    # 2 bar gecikmeli çizginin doğruluğu (ısınma sonrası)
    # 20. barda alpha_trend_2 değeri 18. bardaki alpha_trend ile eşit olmalı
    assert res["alpha_trend_2"].iloc[20] == res["alpha_trend"].iloc[18]
