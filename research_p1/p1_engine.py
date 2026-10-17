# -*- coding: utf-8 -*-
"""
research_p1/p1_engine.py
========================
P1 Momentum araştırma simülasyon ve backtest motoru.

Kapsam:
- 6 Ajan sinyalleri: ZKN, GTD, GT, ALPHA (Özgün AlphaTrend 14-1), ZT3, KBM.
- İki ALPHA varyantı:
    A: Özgün AlphaTrend 14-1 alış sinyali (filtresiz)
    B: Özgün AlphaTrend 14-1 + mevcut P1 ALPHA ajanının ek filtreleri
- Gap hesabı ve ana kural: "gap yok" (|gap| <= 0.0001, yuvarlama toleransı).
- Gap duyarlılık alternatifleri: %0.25, %0.50, %1.00.
- Masraf ve nakit hesabı:
    Komisyon: %0.20 (%0.002) alışta ve satışta
    Slipaj: %0.20 (%0.002) yatırımcı aleyhine alışta ve satışta
    Dayanıklılık senaryoları: %0.30 + %0.30 ve %0.40 + %0.40
- Çıkış kuralları (Mevcut P1):
    STOP: -%5
    TP1: +%8 (yarısı satılır, tek lot ise tamamı satılır)
    TRAILING: -%5 (TP1 sonrası tepe fiyattan)
    MAX_GUN: 10 gün
- Tarihsel dönemler:
    Geliştirme: 2021-10-06 – 2024-10-05 (~3 yıl)
    Doğrulama:  2024-10-06 – 2025-10-05 (~1 yıl)
    Nihai Test: 2025-10-06 – 2026-10-05 (~1 yıl, dokunulmamış)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research_p1.alphatrend import calc_alphatrend_series

log = logging.getLogger(__name__)


# ── Teknik İndikatör Hesaplayıcıları ─────────────────────────────────────────

def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def _sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()


def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(com=p - 1, adjust=False).mean()
    ma_dn = dn.ewm(com=p - 1, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(s: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    l = _ema(s, fast) - _ema(s, slow)
    return l, _ema(l, sig)


def _bbands(s: pd.Series, p: int = 20):
    mid = s.rolling(p).mean()
    sd = s.rolling(p).std()
    return mid, mid + 2 * sd, mid - 2 * sd


def _cmf(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, p: int = 20) -> pd.Series:
    hl_diff = h - l
    mfv = np.where(hl_diff == 0, 0, ((c - l) - (h - c)) / (hl_diff + 1e-9)) * v
    return pd.Series(mfv, index=c.index).rolling(p).sum() / (v.rolling(p).sum() + 1e-9)


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, p: int = 14):
    up = h.diff()
    dn = -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=p - 1, adjust=False).mean()
    pdi = 100 * pdm.ewm(com=p - 1, adjust=False).mean() / (atr + 1e-9)
    ndi = 100 * ndm.ewm(com=p - 1, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    return dx.ewm(com=p - 1, adjust=False).mean(), pdi, ndi


def _stochrsi(s: pd.Series, p: int = 14, k: int = 3) -> pd.Series:
    r = _rsi(s, p)
    lo = r.rolling(p).min()
    hi = r.rolling(p).max()
    return (100 * (r - lo) / (hi - lo + 1e-9)).rolling(k).mean()


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm indikatörleri tarihsel barlar üzerinde cross-sectional lookahead olmaksızın hesaplar."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

    ind = pd.DataFrame(index=df.index)
    ind["open"] = df["open"]
    ind["high"] = h
    ind["low"] = l
    ind["close"] = c
    ind["volume"] = v

    ind["ema8"] = _ema(c, 8)
    ind["ema21"] = _ema(c, 21)
    ind["ema50"] = _ema(c, 50)
    ind["ema200"] = _ema(c, 200)

    ind["sma20"] = _sma(c, 20)
    ind["rsi"] = _rsi(c, 14)

    macd_val, macd_sig = _macd(c, 12, 26, 9)
    ind["macd"] = macd_val
    ind["macd_sig"] = macd_sig
    ind["macd_prev"] = macd_val.shift(1)
    ind["macd_sprev"] = macd_sig.shift(1)

    bb_mid, bb_up, bb_lo = _bbands(c, 20)
    ind["bb_mid"] = bb_mid
    ind["bb_up"] = bb_up
    ind["bb_lo"] = bb_lo

    ind["cmf"] = _cmf(h, l, c, v, 20)
    adx_val, pdi, ndi = _adx(h, l, c, 14)
    ind["adx"] = adx_val
    ind["di_p"] = pdi
    ind["di_n"] = ndi

    ind["stochrsi"] = _stochrsi(c, 14, 3)
    ind["vol20"] = v.rolling(20).mean()
    ind["rel_vol"] = v / (ind["vol20"] + 1e-9)
    ind["change_pct"] = c.pct_change() * 100
    ind["islem_tl"] = c * v

    # AlphaTrend hesaplama
    at_df = calc_alphatrend_series(df, coeff=1.0, ap=14)
    ind["at_alpha_trend"] = at_df["alpha_trend"]
    ind["at_alpha_trend_2"] = at_df["alpha_trend_2"]
    ind["at_buy_confirmed"] = at_df["buy_signal_confirmed"]
    ind["at_sell_confirmed"] = at_df["sell_signal_confirmed"]
    ind["at_has_volume"] = at_df["has_volume"]

    # Mevcut P1 içindeki basitleştirilmiş AlphaTrend çizgisi
    # (ind['alpha_bull'] ve ind['alpha_trend_bull'])
    alpha_atr = _sma(pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1), 14)
    src = (h + l) / 2
    up_t = src - 1.5 * alpha_atr
    dn_t = src + 1.5 * alpha_atr
    # Quick rolling approx for scanner_p1 legacy compatibility
    alpha_legacy = pd.Series(index=df.index, dtype=float)
    if len(df) > 0:
        alpha_legacy.iloc[0] = up_t.iloc[0]
        c_vals = c.values
        u_vals = up_t.values
        d_vals = dn_t.values
        a_vals = np.zeros(len(df))
        a_vals[0] = u_vals[0]
        for i in range(1, len(df)):
            if c_vals[i - 1] > a_vals[i - 1]:
                a_vals[i] = max(u_vals[i], a_vals[i - 1])
            else:
                a_vals[i] = min(d_vals[i], a_vals[i - 1])
        alpha_legacy[:] = a_vals
    ind["alpha_legacy"] = alpha_legacy
    ind["alpha_bull_legacy"] = (c > alpha_legacy) & (c.shift(1) <= alpha_legacy.shift(1))
    ind["alpha_trend_bull_legacy"] = c > alpha_legacy

    return ind
