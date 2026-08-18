"""
DSP-P3 FAZ 4 — IC hesaplama sözleşmesi (tarih hizalamalı)
==========================================================
strateji_ic_hesapla'nın deterministik girdiyle deterministik çıktı üretmesi
test edilir. FAZ 4 (Paket 1) sonrası sözleşme:

  * fiyat_cache tarih eksenlidir (pd.Series + DatetimeIndex).
  * fwd_return(t) = log(close[t+period] / close[t+1]) — sinyal tarihinden
    sonraki bar ile period bar sonrası arası.
  * Sinyal tarihi seride yoksa veya t+period yoksa kayıt atlanır.

Network YOK: fiyat_cache elle verilir, fiyat_cek ÇAĞRILMAZ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta_portfolio import strateji_ic_hesapla


def _fiyat_cache(n_sembol: int, taban: float = 100.0) -> dict[str, pd.Series]:
    """Her sembol için tarih eksenli 30 barlık seri (07-01..07-30).
    Sinyal tarihi 07-10 (pos 9) → fwd = log(close[14] / close[10])."""
    idx = pd.date_range("2026-07-01", periods=30, freq="D")
    cache = {}
    for i in range(n_sembol):
        degerler = np.full(30, taban, dtype=float)
        degerler[10] = taban
        degerler[14] = taban + (i + 1) * 10.0
        cache[f"S{i:02d}"] = pd.Series(degerler, index=idx, dtype=float)
    return cache


def _sinyal_logu(n: int) -> list[dict]:
    return [
        {"symbol": f"S{i:02d}", "score": float(i + 1), "tarih": "2026-07-10"}
        for i in range(n)
    ]


def test_strateji_ic_yetersiz_sinyal_sifir():
    # < 15 sinyal → 0.0 (mevcut koruma)
    assert strateji_ic_hesapla("P1", _sinyal_logu(5), _fiyat_cache(5)) == 0.0


def test_strateji_ic_bos_log_sifir():
    assert strateji_ic_hesapla("P2", [], {}) == 0.0


def test_strateji_ic_fiyat_yoksa_sifir():
    # 15 sinyal ama fiyat_cache boş → hiçbir kayıt eşleşmez → yetersiz veri → 0.0
    assert strateji_ic_hesapla("P1", _sinyal_logu(15), {}) == 0.0


def test_strateji_ic_monoton_veri_mukemmel_korelasyon():
    # skor artan (1..15), fwd_return log((100+i*10)/100) monoton artan
    # → Spearman rank ≈ 1.0 (log monoton dönüşüm olduğu için rank korunur)
    ic = strateji_ic_hesapla("P3", _sinyal_logu(15), _fiyat_cache(15), period=5)
    assert 0.95 <= ic <= 1.0


def test_strateji_ic_deterministik_tekrar():
    log = _sinyal_logu(15)
    cache = _fiyat_cache(15)
    a = strateji_ic_hesapla("P1", log, cache, period=5)
    b = strateji_ic_hesapla("P1", log, cache, period=5)
    assert a == b
