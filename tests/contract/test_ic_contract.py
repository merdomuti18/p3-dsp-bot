"""
DSP-P3 FAZ 1 — IC hesaplama sözleşmesi
======================================
strateji_ic_hesapla'nın deterministik girdiyle deterministik çıktı üretmesi
test edilir. Bu test MEVCUT davranışı kilitler (P0-1 bulgusu: tarih
hizalaması yok — fwd_return fiyat dizisinin sonundan hesaplanır).
FAZ 4'te IC düzeltilirse bu test yeni sözleşmeye göre güncellenecek.

Network YOK: fiyat_cache elle verilir, fiyat_cek ÇAĞRILMAZ.
"""

from __future__ import annotations

import numpy as np
import pytest

from meta_portfolio import strateji_ic_hesapla


def _fiyat_cache(n_sembol: int, taban: float = 100.0) -> dict[str, np.ndarray]:
    """Her sembol için son fiyatı artan şekilde farklılaşan 30 barlık dizi."""
    cache = {}
    for i in range(n_sembol):
        dizi = np.full(29, taban, dtype=float)
        dizi = np.append(dizi, taban + (i + 1) * 10.0)
        cache[f"S{i:02d}"] = dizi
    return cache


def _sinyal_logu(n: int) -> list[dict]:
    return [
        {"symbol": f"S{i:02d}", "score": float(i + 1), "tarih": "2026-08-01"}
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
