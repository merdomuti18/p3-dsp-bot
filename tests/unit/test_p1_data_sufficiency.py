# -*- coding: utf-8 -*-
"""
tests/unit/test_p1_data_sufficiency.py
======================================
P1 Veri Yeterliliği ve EMA200 Görünürlüğü Testleri:
1. EMA200 için yetersiz veri (< 200 bar) olduğunda indikatörde ema200_yetersiz_veri bayrağı
2. Sinyal kayıtlarında ema200_filtre_devre_disi ve bar sayısının görünür hale gelmesi
3. Sinyal eşiklerinin ve strateji mantığının korunması
4. Yeterli veri (>= 200 bar) olduğunda EMA200'ün normal hesaplanması ve filtrenin aktif olması
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scanner_p1 as sp1


def _generate_synthetic_df(n_bars: int = 150) -> pd.DataFrame:
    """Ağdan bağımsız sentetik OHLCV DataFrame üretici."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=n_bars, freq="D")
    base_price = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = base_price + np.random.rand(n_bars) * 2.0
    low = base_price - np.random.rand(n_bars) * 2.0
    close = base_price + np.random.randn(n_bars) * 0.3
    volume = np.random.randint(10000, 50000, size=n_bars)
    return pd.DataFrame({
        "open": base_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


class TestP1DataSufficiency:
    def test_ema200_yetersiz_veri_gorunurlugu(self):
        """150 barlık veride EMA200 hesaplanamaz ve yetersiz veri bayrağı True olur."""
        df_150 = _generate_synthetic_df(150)
        ind = sp1.get_indicators(df_150)
        assert ind is not None
        assert np.isnan(ind["ema200"])
        assert ind["ema200_yetersiz_veri"] is True
        assert ind["bar_sayisi"] == 150

    def test_ema200_yeterli_veri_normal_hesaplama(self):
        """220 barlık veride EMA200 normal hesaplanır ve bayrak False olur."""
        df_220 = _generate_synthetic_df(220)
        ind = sp1.get_indicators(df_220)
        assert ind is not None
        assert not np.isnan(ind["ema200"])
        assert ind["ema200"] > 0
        assert ind["ema200_yetersiz_veri"] is False
        assert ind["bar_sayisi"] == 220

    def test_sinyal_kayitlarinda_ema200_devre_disi_gorunurlugu(self):
        """EMA200 verisi yetersiz olduğunda üretilen sinyal kaydında ema200_filtre_devre_disi=True olmalıdır."""
        df_150 = _generate_synthetic_df(150)
        ind_150 = sp1.get_indicators(df_150)

        strategy_results = {
            "GT": [{"symbol": "THYAO", "ind": ind_150}],
        }
        records = sp1._build_signal_records("06.10.2026 19:05", "19:05 Kapanış", strategy_results)
        assert len(records) == 1
        rec = records[0]
        assert rec["symbol"] == "THYAO"
        assert rec["ema200_filtre_devre_disi"] is True
        assert rec["ema200_bar_sayisi"] == 150

    def test_strateji_mantigi_ve_esikleri_korunur(self):
        """EMA200 NaN olduğunda (pd.isna(ind['ema200']) or ...) koşulu strateji mantığını aynen korur."""
        # GT koşulları: ema8 > ema21 > ema50, rsi 50..65, adx > 20, di_p > di_n, cmf > 0, rel_vol >= 1
        ind = {
            "ema8": 105.0, "ema21": 103.0, "ema50": 101.0, "ema200": float("nan"),
            "rsi": 55.0, "adx": 25.0, "di_p": 25.0, "di_n": 15.0,
            "cmf": 0.10, "rel_vol": 1.5, "close": 106.0
        }
        # Piyasa değeri > 10M
        assert sp1.strategy_guclu_trend(ind, 50_000_000) is True

        # rsi eşik dışı (70) -> False
        ind_bad = dict(ind, rsi=70.0)
        assert sp1.strategy_guclu_trend(ind_bad, 50_000_000) is False
