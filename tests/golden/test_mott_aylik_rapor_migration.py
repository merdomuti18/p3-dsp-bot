"""
DSP-P3 FAZ 3.2 — M1..M4: mott_aylik_rapor migration testleri
==============================================================
Test-first: bu testler production değişikliğinden ÖNCE yazılır (RED).

  M1 — Migration equivalence: yeni normalize-tabanlı rapor blokları, eski
       okuyucularla (mott_performans_analiz — DOKUNULMAZ) aynı mock altında
       aynı sonucu üretir.
  M2 — _acik_pozisyon_enrich() birim testi (mock 100.0; IZMDC özel kontrol).
  M3 — P3 bozuk_kayit + donem rapor katmanında korunur (normalize'a eklenmez).
  M4 — P3 current_equity = baslangic + realized + unrealized(enrichment);
       normalize("P3")["equity"] ASLA değiştirilmez (None kalır).

Mock noktası (T4 ile AYNI): mpa.get_price = lambda sym: 100.0.
Enrichment içinde mpa.get_price(sym) — runtime lookup (import kopyası YOK).
"""

from __future__ import annotations

import pytest

import mott_aylik_rapor as mar
import mott_performans_analiz as mpa
from mott_state import normalize

POS_TL = 20_000


@pytest.fixture
def mock_fiyat(monkeypatch):
    monkeypatch.setattr(mpa, "get_price", lambda sym: 100.0)


# ---------------------------------------------------------------------------
# M1 — Migration equivalence (eski vs yeni, aynı mock)
# ---------------------------------------------------------------------------

def test_m1_p3_eski_yeni_esdeger(mock_fiyat):
    eski = mpa.analiz_p3()
    yeni = mar._p3_rapor_blok()
    assert yeni["equity_est"] == pytest.approx(eski["equity_est"], abs=1.0)
    assert yeni["getiri_pct"] == pytest.approx(eski["getiri_pct"], abs=0.1)
    assert yeni["kapanan"] == eski["kapanan"]
    assert yeni["kapanan_wr"] == eski["kapanan_wr"]
    assert yeni["bozuk_kayit"] == eski["bozuk_kayit"]
    assert len(yeni["acik"]) == len(eski["acik"])
    assert yeni["equity_est"] == pytest.approx(274000, abs=1.0)   # T4 golden
    assert yeni["getiri_pct"] == pytest.approx(174.0, abs=0.1)    # T4 golden
    assert yeni["kapanan"] == 16 and len(yeni["acik"]) == 4


def test_m1_p4_eski_yeni_esdeger(mock_fiyat):
    eski = mpa.analiz_p4()
    yeni = mar._p4_rapor_blok()
    assert yeni["equity_est"] == pytest.approx(eski["equity_est"], abs=1.0)
    assert yeni["getiri_pct"] == pytest.approx(eski["getiri_pct"], abs=0.1)
    assert yeni["kapanan"] == eski["kapanan"]
    assert yeni["equity_est"] == pytest.approx(96998, abs=1.0)    # T4 golden
    assert yeni["getiri_pct"] == pytest.approx(-3.0, abs=0.1)     # T4 golden
    assert yeni["kapanan"] == 36


def test_m1_p5_eski_yeni_esdeger(mock_fiyat):
    eski = mar._p5_analiz()
    yeni = mar._p5_rapor_blok()
    assert yeni["equity_est"] == pytest.approx(eski["equity_est"], abs=1.0)
    assert yeni["getiri_pct"] == pytest.approx(eski["getiri_pct"], abs=0.1)
    assert yeni["kapanan"] == eski["kapanan"]
    assert yeni["equity_est"] == pytest.approx(101900, abs=1.0)   # T4 golden
    assert yeni["getiri_pct"] == pytest.approx(1.9, abs=0.1)      # T4 golden
    assert yeni["kapanan"] == 10


def test_m1_p1_eski_yeni_esdeger(mock_fiyat):
    yeni = mar._p1_p2_rapor_blok("P1")
    # Mevcut davranış KORUNUR: sermaye_mevcut yok → başlangıç sermayesi, %0.
    assert yeni["equity_est"] == 100000
    assert yeni["getiri_pct"] == 0.0
    # FAZ 3.2 T10 (onaylı contract evolution): mott_state._p1_p2 artık
    # `trade_history` anahtarını da okur → normalize("P1") islem_gecmisi = 10
    # (portfoy.json trade_history) — eski gerçek okuyucu semantiğiyle aynı.
    assert yeni["kapanan"] == 10
    assert yeni["acik_sayisi"] == 5       # portfoy.json pozisyonlar
    assert yeni["nakit"] == pytest.approx(13183.105270965556)


def test_m1_p2_eski_yeni_esdeger(mock_fiyat):
    yeni = mar._p1_p2_rapor_blok("P2")
    assert yeni["equity_est"] == 100000
    assert yeni["getiri_pct"] == 0.0
    # Aynı T10 düzeltmesi: normalize("P2") islem_gecmisi = 14 (portfoy_p2.json
    # trade_history) — eski gerçek okuyucu semantiğiyle aynı.
    assert yeni["kapanan"] == 14
    assert yeni["acik_sayisi"] == 7       # portfoy_p2.json pozisyonlar
    assert yeni["nakit"] == pytest.approx(6774.107740554809)


# ---------------------------------------------------------------------------
# M2 — Enrichment unit (mock 100.0, P3 fixture)
# ---------------------------------------------------------------------------

def test_m2_acik_pozisyon_enrich_p3(mock_fiyat):
    pozisyonlar = normalize("P3")["pozisyonlar"]
    enriched, unrealized = mar._acik_pozisyon_enrich(pozisyonlar)

    by_sym = {p["symbol"]: p for p in enriched}
    izmdc = by_sym["IZMDC"]
    assert izmdc["entry"] == pytest.approx(9.25)
    assert izmdc["now"] == 100.0
    assert izmdc["pnl_pct"] == pytest.approx((100.0 - 9.25) / 9.25 * 100, abs=0.01)
    assert izmdc["tl"] == pytest.approx(POS_TL * (100.0 - 9.25) / 9.25, abs=1.0)

    # Toplam unrealized, T4 golden'la matematiksel uyumlu (274000 = 100000 - 9390 + U)
    assert unrealized == pytest.approx(183390.0, abs=5.0)
    assert len(enriched) == 4


def test_m2_enrich_network_kullanmaz_ama_mock_gerekli(mock_fiyat):
    """Enrichment yalnız raporlama katmanında; mock'suz NETWORK=OFF engeller."""
    pozisyonlar = normalize("P3")["pozisyonlar"]
    enriched, unrealized = mar._acik_pozisyon_enrich(pozisyonlar)
    assert unrealized is not None


# ---------------------------------------------------------------------------
# M3 — P3 bozuk_kayit + donem (rapor katmanı; normalize'a EKLENMEZ)
# ---------------------------------------------------------------------------

def test_m3_p3_bozuk_kayit_donem(mock_fiyat):
    r = mar.rapor_olustur()
    assert r["p3"]["bozuk_kayit"] == 0
    assert r["p3"]["donem"] == "2026-07-02 → bugün"
    # normalize() bu alanları TAŞIMAZ (kullanıcı kuralı 15)
    n = normalize("P3")
    assert "bozuk_kayit" not in n and "donem" not in n and "created" not in n


# ---------------------------------------------------------------------------
# M4 — P3 current equity formülü (rapor katmanı)
# ---------------------------------------------------------------------------

def test_m4_p3_current_equity_formulu(mock_fiyat):
    n = normalize("P3")
    # normalize çıktısı: deterministik, enrichment öncesi
    assert n["baslangic_sermayesi"] == 100000
    assert n["realized_pnl"] == pytest.approx(-9390.0, abs=0.01)
    assert n["unrealized_pnl"] is None
    assert n["equity"] is None  # ASLA 274000'e çevrilmez (kullanıcı kuralı)

    enriched, unrealized = mar._acik_pozisyon_enrich(n["pozisyonlar"])
    current_equity = n["baslangic_sermayesi"] + n["realized_pnl"] + unrealized
    getiri = (current_equity - 100000) / 100000 * 100

    assert current_equity == pytest.approx(274000.0, abs=1.0)   # T4 golden
    assert getiri == pytest.approx(174.0, abs=0.1)              # T4 golden
    # normalize çıktısı enrichment'ten ETKİLENMEZ (kopya üzerinde çalışılır)
    assert normalize("P3")["equity"] is None
