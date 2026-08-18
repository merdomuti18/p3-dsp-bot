"""
DSP-P3 FAZ 4 — Paket 1: IC tarih hizalama + look-ahead guard
============================================================
B1/B2/B3 düzeltmesinin sözleşme testleri:

  * fwd_return(t) = log(close[t+period] / close[t+1])
    — sinyal gününden SONRAKİ bar ile period bar sonrası arası.
  * Sinyal tarihinden ÖNCEKİ hiçbir fiyat forward return hesabına giremez.
  * t+period seride yoksa sinyal hizalanmış örneklerden çıkar.
  * Aynı sembolün farklı tarihlerdeki sinyalleri farklı forward return üretir.
  * fiyat_cek() tarih eksenini (DatetimeIndex) korur.
  * Mevcut guard'lar korunur: <15 sinyal → 0.0, <10 hizalı örnek → 0.0.

Network YOK: fiyat_cek yalnız mock ile test edilir.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _clean_cache():
    """FAZ 6.1: Her test öncesi fiyat cache'ini temizle."""
    import mott_fiyat
    mott_fiyat.clear_cache()
    yield
    mott_fiyat.clear_cache()

import meta_portfolio


def _gunluk_seri(fiyatlar: list[float], baslangic: str = "2026-07-01") -> pd.Series:
    idx = pd.date_range(baslangic, periods=len(fiyatlar), freq="D")
    return pd.Series(fiyatlar, index=idx, dtype=float)


def _sinyal_logu(n: int, tarih: str = "2026-07-10", skor_baz: float = 1.0) -> list[dict]:
    return [
        {"symbol": f"S{i:02d}", "score": float(skor_baz * (i + 1)), "tarih": tarih}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _sinyal_tarihi — tarih çözümleme (ISO + DD.MM.YYYY)
# ---------------------------------------------------------------------------

def test_sinyal_tarihi_iso():
    assert meta_portfolio._sinyal_tarihi({"tarih": "2026-07-10"}) == date(2026, 7, 10)


def test_sinyal_tarihi_scan_time_ddmmyyyy():
    # P1/P2 state kayıtları scan_time formatı: "14.08.2026 20:02"
    assert meta_portfolio._sinyal_tarihi(
        {"scan_time": "14.08.2026 20:02"}
    ) == date(2026, 8, 14)


def test_sinyal_tarihi_bozuk_none():
    assert meta_portfolio._sinyal_tarihi({}) is None
    assert meta_portfolio._sinyal_tarihi({"scan_time": "bozuk-veri"}) is None


# ---------------------------------------------------------------------------
# _fwd_return_hesapla — hizalama + look-ahead guard
# ---------------------------------------------------------------------------

def test_fwd_return_tarih_hizalama_formulu():
    # 30 bar; sinyal t = 07-10 (pos 9); period=5
    fiyatlar = [100.0] * 30
    fiyatlar[10] = 105.0    # close[t+1]
    fiyatlar[14] = 115.5    # close[t+period]
    seri = _gunluk_seri(fiyatlar)
    fwd = meta_portfolio._fwd_return_hesapla(seri, date(2026, 7, 10), 5)
    assert fwd == pytest.approx(math.log(115.5 / 105.0))


def test_fwd_return_ayni_sembol_farkli_tarih_farkli_deger():
    fiyatlar = [100.0] * 30
    fiyatlar[10] = 100.0; fiyatlar[14] = 120.0   # t1=07-10 → +%20
    fiyatlar[20] = 110.0; fiyatlar[24] = 99.0    # t2=07-20 → −%10
    seri = _gunluk_seri(fiyatlar)
    f1 = meta_portfolio._fwd_return_hesapla(seri, date(2026, 7, 10), 5)
    f2 = meta_portfolio._fwd_return_hesapla(seri, date(2026, 7, 20), 5)
    assert f1 != f2
    assert f1 == pytest.approx(math.log(120.0 / 100.0))
    assert f2 == pytest.approx(math.log(99.0 / 110.0))


def test_fwd_return_ileri_veri_yoksa_none():
    seri = _gunluk_seri([100.0] * 12)   # 07-01 .. 07-12
    # t=07-10 pos 9 → t+5 = pos 14 seride yok
    assert meta_portfolio._fwd_return_hesapla(seri, date(2026, 7, 10), 5) is None
    # t=07-12 son bar → t+1 yok
    assert meta_portfolio._fwd_return_hesapla(seri, date(2026, 7, 12), 5) is None


def test_fwd_return_tarih_seride_yoksa_none():
    # Sinyal tarihi seride karşılığı yoksa hizalanamaz → None
    seri = _gunluk_seri([100.0] * 20)   # 07-01 .. 07-20
    assert meta_portfolio._fwd_return_hesapla(seri, date(2026, 6, 30), 5) is None
    assert meta_portfolio._fwd_return_hesapla(seri, date(2026, 8, 1), 5) is None


def test_fwd_return_lookahead_t_oncesi_fiyata_erismez():
    fiyatlar = [100.0] * 30
    fiyatlar[10] = 105.0; fiyatlar[14] = 115.5
    seri = _gunluk_seri(fiyatlar)
    temel = meta_portfolio._fwd_return_hesapla(seri, date(2026, 7, 10), 5)
    # Sinyal günü kapanışı (close[t]) ve t-1 değiştirilirse sonuç DEĞİŞMEMELİ:
    # forward return t öncesi hiçbir bara erişemez (look-ahead guard).
    for pos in (9, 8):
        seri2 = seri.copy()
        seri2.iloc[pos] = 0.5
        assert meta_portfolio._fwd_return_hesapla(
            seri2, date(2026, 7, 10), 5
        ) == pytest.approx(temel)


def test_fwd_return_t1_ve_tp_kullanilir():
    fiyatlar = [100.0] * 30
    fiyatlar[10] = 105.0; fiyatlar[14] = 115.5
    seri = _gunluk_seri(fiyatlar)
    temel = meta_portfolio._fwd_return_hesapla(seri, date(2026, 7, 10), 5)
    # close[t+1] değişirse sonuç değişmeli
    seri2 = seri.copy(); seri2.iloc[10] = 200.0
    assert meta_portfolio._fwd_return_hesapla(
        seri2, date(2026, 7, 10), 5
    ) != pytest.approx(temel)
    # close[t+period] değişirse sonuç değişmeli
    seri3 = seri.copy(); seri3.iloc[14] = 50.0
    assert meta_portfolio._fwd_return_hesapla(
        seri3, date(2026, 7, 10), 5
    ) != pytest.approx(temel)


# ---------------------------------------------------------------------------
# strateji_ic_hesapla — pipeline düzeyinde hizalama
# ---------------------------------------------------------------------------

def test_strateji_ic_hizalama_eski_kod_ayirt_eder():
    # Sinyal t=07-10; close[t+1] ve close[t+period] skorla monoton artan;
    # t+period SONRASI barlar SABİT 100 → eski kod (son bar penceresi) tüm
    # fwd'leri 0 üretir (IC=0), yeni kod hizalı fwd üretir (IC≈1).
    n = 15
    cache = {}
    for i in range(n):
        f = [100.0] * 30
        f[10] = 100.0 + i       # close[t+1]
        f[14] = 100.0 + 2 * i   # close[t+period]
        cache[f"S{i:02d}"] = _gunluk_seri(f)
    log = _sinyal_logu(n)
    ic = meta_portfolio.strateji_ic_hesapla("P1", log, cache, period=5)
    assert 0.95 <= ic <= 1.0


def test_strateji_ic_ayni_sembol_farkli_tarih():
    # S00 iki farklı tarihte sinyal veriyor; her kayıt KENDİ tarihinden
    # hizalanmalı (eski kod iki kayda da aynı değeri verirdi → IC bozulurdu).
    cache = {}
    for i in range(1, 15):
        f = [100.0] * 30
        f[10] = 100.0; f[14] = 100.0 + i
        cache[f"S{i:02d}"] = _gunluk_seri(f)
    f_s00 = [100.0] * 30
    f_s00[10] = 100.0; f_s00[14] = 105.0      # S00 @ t1=07-10 → +%5
    f_s00[20] = 110.0; f_s00[24] = 99.0       # S00 @ t2=07-20 → −%10
    cache["S00"] = _gunluk_seri(f_s00)

    kayitlar = [
        {"symbol": "S00", "score": 1.0, "tarih": "2026-07-20"},   # en düşük fwd
        {"symbol": "S00", "score": 2.0, "tarih": "2026-07-10"},   # ikinci düşük
    ]
    for i in range(1, 15):
        kayitlar.append({"symbol": f"S{i:02d}", "score": float(i + 2), "tarih": "2026-07-10"})
    # fwd sırası: S00@07-20 (−) < S00@07-10 (+%5) < S01 (+%1) < ... < S14 (+%14)
    # skor sırası 1..16 ile monoton → IC ≈ 1.0
    ic = meta_portfolio.strateji_ic_hesapla("P2", kayitlar, cache, period=5)
    assert 0.95 <= ic <= 1.0


def test_strateji_ic_ileri_veri_yetmezse_sifir():
    # 15 sinyal ama seriler t+period'u taşımıyor → 0 hizalı örnek → 0.0
    cache = {}
    for i in range(15):
        f = [100.0] * 12
        f[10] = 100.0; f[11] = 101.0
        cache[f"S{i:02d}"] = _gunluk_seri(f)
    log = _sinyal_logu(15)   # t=07-10 → t+5 seride yok
    assert meta_portfolio.strateji_ic_hesapla("P1", log, cache, period=5) == 0.0


def test_strateji_ic_deterministik_tekrar_hizali():
    n = 15
    cache = {}
    for i in range(n):
        f = [100.0] * 30
        f[10] = 100.0 + i; f[14] = 100.0 + 2 * i
        cache[f"S{i:02d}"] = _gunluk_seri(f)
    log = _sinyal_logu(n)
    a = meta_portfolio.strateji_ic_hesapla("P1", log, cache, period=5)
    b = meta_portfolio.strateji_ic_hesapla("P1", log, cache, period=5)
    assert a == b


# ---------------------------------------------------------------------------
# fiyat_cek — tarih ekseni sözleşmesi (mock; network YOK)
# ---------------------------------------------------------------------------

def test_fiyat_cek_tarih_ekseni_korur(monkeypatch):
    class FakeTicker:
        def __init__(self, sym: str):
            self.sym = sym

        def history(self, period=None):
            idx = pd.date_range("2026-07-01", periods=35, freq="D")
            return pd.DataFrame({"Close": np.linspace(100.0, 130.0, 35)}, index=idx)

    monkeypatch.setattr(meta_portfolio.yf, "Ticker", FakeTicker)
    monkeypatch.setattr("mott_fiyat.tv_fiyatlar", lambda semboller: {})

    cache = meta_portfolio.fiyat_cek(["GARAN"], bars=30)
    assert "GARAN" in cache
    seri = cache["GARAN"]
    assert isinstance(seri, pd.Series)
    assert isinstance(seri.index, pd.DatetimeIndex)
    assert len(seri) == 30
    assert seri.index.is_monotonic_increasing
    # Son bar tüketici uyumu: .iloc[-1] ile okunabilir
    assert float(seri.iloc[-1]) == pytest.approx(130.0)


def test_fiyat_cek_tv_son_bari_ezdirir(monkeypatch):
    class FakeTicker:
        def __init__(self, sym: str):
            self.sym = sym

        def history(self, period=None):
            idx = pd.date_range("2026-07-01", periods=35, freq="D")
            return pd.DataFrame({"Close": np.linspace(100.0, 130.0, 35)}, index=idx)

    monkeypatch.setattr(meta_portfolio.yf, "Ticker", FakeTicker)
    monkeypatch.setattr("mott_fiyat.tv_fiyatlar", lambda semboller: {"GARAN": 999.0})

    cache = meta_portfolio.fiyat_cek(["GARAN"], bars=30)
    seri = cache["GARAN"]
    assert isinstance(seri, pd.Series)
    assert isinstance(seri.index, pd.DatetimeIndex)
    assert float(seri.iloc[-1]) == pytest.approx(999.0)
