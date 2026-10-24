# -*- coding: utf-8 -*-
"""
research_p1/alphatrend.py
=========================
Kıvanç Özbilgiç'in özgün AlphaTrend göstergesinin (Pine Script v5)
birebir Python/NumPy uygulaması.

Kaynak: https://www.tradingview.com/script/o50NYLAZ-AlphaTrend/
Sürüm: Pine Script v5, author & developer: KivancOzbilgic
SHA-256 (AlphaTrend_KivancOzbilgic.pine): kaydedilmiştir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Pine Script ta.tr hesabı:
    ta.tr = max(high - low, abs(high - close[1]), abs(low - close[1]))
    İlk bar için: high[0] - low[0]
    """
    n = len(high)
    tr = np.zeros(n, dtype=float)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    return tr


def calc_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Pine Script ta.mfi(hlc3, period) hesabı:
    hlc3 = (high + low + close) / 3
    upper = sum(volume * (change(hlc3) <= 0 ? 0 : hlc3), period)
    lower = sum(volume * (change(hlc3) >= 0 ? 0 : hlc3), period)
    mfi = rsi(upper, lower) -> 100 - (100 / (1 + upper / lower))
    """
    n = len(high)
    mfi = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return mfi

    hlc3 = (high + low + close) / 3.0
    pos_flow = np.zeros(n, dtype=float)
    neg_flow = np.zeros(n, dtype=float)

    for i in range(1, n):
        diff = hlc3[i] - hlc3[i - 1]
        if diff > 0:
            pos_flow[i] = hlc3[i] * volume[i]
        elif diff < 0:
            neg_flow[i] = hlc3[i] * volume[i]

    # Rolling sum with period
    pos_sum = pd.Series(pos_flow).rolling(period).sum().values
    neg_sum = pd.Series(neg_flow).rolling(period).sum().values

    for i in range(period, n):
        ps = pos_sum[i]
        ns = neg_sum[i]
        if np.isnan(ps) or np.isnan(ns):
            continue
        if ns == 0.0:
            mfi[i] = 100.0 if ps > 0 else 50.0
        else:
            mr = ps / ns
            mfi[i] = 100.0 - (100.0 / (1.0 + mr))

    return mfi


def calc_alphatrend_series(
    df: pd.DataFrame,
    coeff: float = 1.0,
    ap: int = 14,
) -> pd.DataFrame:
    """Özgün AlphaTrend göstergesini hesaplar.

    Parametreler:
      df: 'high', 'low', 'close', 'volume' sütunlarını içeren DataFrame
      coeff: 1.0 (sabit katsayı)
      ap: 14 (ortak periyot)

    Dönen DataFrame:
      - 'atr': ta.sma(ta.tr, AP)
      - 'mfi': ta.mfi(hlc3, AP)
      - 'up_t': low - atr * coeff
      - 'down_t': high + atr * coeff
      - 'alpha_trend': AlphaTrend çizgisi
      - 'alpha_trend_2': AlphaTrend[2] (2 bar gecikmeli çizgi)
      - 'buy_signal_confirmed': Kapanışta teyit edilmiş alış sinyali (O1 > K2 filtreli)
      - 'sell_signal_confirmed': Kapanışta teyit edilmiş satış sinyali (O2 > K1 filtreli)
      - 'has_volume': Hacim verisinin geçerli olup olmadığı
    """
    n = len(df)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float) if "volume" in df.columns else np.zeros(n, dtype=float)

    has_volume = bool((volume > 0).sum() > (n * 0.5))

    # 1. TR & ATR (ta.sma(ta.tr, ap))
    tr = calc_true_range(high, low, close)
    atr = pd.Series(tr).rolling(ap).mean().values

    # 2. MFI
    mfi = calc_mfi(high, low, close, volume, period=ap)

    # 3. upT & downT
    up_t = low - atr * coeff
    down_t = high + atr * coeff

    # 4. AlphaTrend özyinelemeli çizgi
    alpha_trend = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(mfi[i]):
            continue
        prev_at = 0.0 if (i == 0 or np.isnan(alpha_trend[i - 1])) else alpha_trend[i - 1]

        # ta.mfi(hlc3, AP) >= 50 ?
        #   (upT < nz(AlphaTrend[1]) ? nz(AlphaTrend[1]) : upT) :
        #   (downT > nz(AlphaTrend[1]) ? nz(AlphaTrend[1]) : downT)
        if mfi[i] >= 50.0:
            if prev_at > 0.0 and up_t[i] < prev_at:
                alpha_trend[i] = prev_at
            else:
                alpha_trend[i] = up_t[i]
        else:
            if prev_at > 0.0 and down_t[i] > prev_at:
                alpha_trend[i] = prev_at
            else:
                alpha_trend[i] = down_t[i]

    # 5. AlphaTrend[2]
    alpha_trend_2 = np.full(n, np.nan, dtype=float)
    if n >= 3:
        alpha_trend_2[2:] = alpha_trend[:-2]

    # 6. Kesişimler: buySignalk = ta.crossover(AlphaTrend, AlphaTrend[2])
    # crossover(x, y) -> x[1] <= y[1] and x > y
    buy_signalk = np.zeros(n, dtype=bool)
    sell_signalk = np.zeros(n, dtype=bool)

    for i in range(3, n):
        at = alpha_trend[i]
        at_prev = alpha_trend[i - 1]
        at2 = alpha_trend_2[i]
        at2_prev = alpha_trend_2[i - 1]

        if not np.isnan(at) and not np.isnan(at_prev) and not np.isnan(at2) and not np.isnan(at2_prev):
            if at_prev <= at2_prev and at > at2:
                buy_signalk[i] = True
            if at_prev >= at2_prev and at < at2:
                sell_signalk[i] = True

    # 7. Sinyal süzme (O1 > K2, O2 > K1):
    # K1 = ta.barssince(buySignalk)
    # K2 = ta.barssince(sellSignalk)
    # O1 = ta.barssince(buySignalk[1])
    # O2 = ta.barssince(sellSignalk[1])
    # Alış sinyali: buySignalk and O1 > K2
    # Bu koşul sinyalin ilk defa gelmesini (re-arm) ve SELL'den sonra ilk BUY olmasını sağlar.
    buy_confirmed = np.zeros(n, dtype=bool)
    sell_confirmed = np.zeros(n, dtype=bool)

    state = 0  # 1 = long'da, -1 = short'ta
    for i in range(n):
        if buy_signalk[i]:
            if state != 1:
                buy_confirmed[i] = True
                state = 1
        elif sell_signalk[i]:
            if state != -1:
                sell_confirmed[i] = True
                state = -1

    return pd.DataFrame({
        "atr": atr,
        "mfi": mfi,
        "up_t": up_t,
        "down_t": down_t,
        "alpha_trend": alpha_trend,
        "alpha_trend_2": alpha_trend_2,
        "buy_signal_raw": buy_signalk,
        "sell_signal_raw": sell_signalk,
        "buy_signal_confirmed": buy_confirmed,
        "sell_signal_confirmed": sell_confirmed,
        "has_volume": has_volume,
    }, index=df.index)
