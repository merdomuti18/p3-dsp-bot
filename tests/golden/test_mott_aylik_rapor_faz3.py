"""
DSP-P3 FAZ 3.2 — T4: mott_aylik_rapor migration/golden testleri
================================================================
Amaç: mott_aylik_rapor'un MEVCUT davranışını, canlı fiyatı mock'layarak
deterministik biçimde kilitlemek. FAZ 3.2 migrasyonu (rapor → normalize())
sonrasında aynı mock girdisiyle AYNI çıktının üretilmesi bu testlerin
beklentisidir — davranış bozulması burada yakalanır.

Kritik invariantlar:
  * mott_state.normalize() canlı fiyat ÇEKMEZ (yalnız state okur).
  * Canlı fiyat gereksinimi (P3/P1/P2 açık pozisyon pnl'si) raporlama
    katmanında AYRI bir enrichment adımıdır — burada get_price mock'lanır.
  * NETWORK = OFF (conftest autouse): mock olmadan yfinance çağrısı imkânsız.
  * mott_portfoy_deger karar hattına DOKUNULMAZ (bu testler onu çağırmaz).

Mock stratejisi: mott_performans_analiz.get_price → sabit 100.0. analiz_p3
ve _portfoy_analiz kendi modül global'ini kullandığı için monkeypatch yeterlidir.
"""

from __future__ import annotations

import json

import pytest

import mott_aylik_rapor as mar
import mott_performans_analiz as mpa
from mott_state import normalize


@pytest.fixture
def mock_fiyat_sabit(monkeypatch):
    """T4 mock stratejisi: tüm semboller için sabit 100.0 — deterministik."""
    monkeypatch.setattr(mpa, "get_price", lambda sym: 100.0)


# ---------------------------------------------------------------------------
# 1. Rapor şeması (yapı kilidi)
# ---------------------------------------------------------------------------

def test_rapor_olustur_sema_anahtarlari(mock_fiyat_sabit):
    r = mar.rapor_olustur()
    assert set(r.keys()) == {"tarih", "sermaye_portfoy", "portfoyler", "p3", "p4", "p5", "p1_p2"}
    assert r["sermaye_portfoy"] == 100000
    assert len(r["portfoyler"]) == 5
    assert {p["kod"] for p in r["portfoyler"]} == {"P1", "P2", "P3", "P4", "P5"}


def test_rapor_portfoyler_alan_kumeleri(mock_fiyat_sabit):
    r = mar.rapor_olustur()
    alanlar = {p["kod"]: set(p.keys()) for p in r["portfoyler"]}
    # FAZ 0 artefaktındaki alan kümeleri (ast.literal_eval ile doğrulandı)
    assert {"kod", "ad", "equity_est", "getiri_pct", "kapanan", "kapanan_wr", "sermaye"} <= alanlar["P3"]
    assert "acik" in alanlar["P3"] and "bozuk_kayit" in alanlar["P3"]
    assert {"kod", "ad", "equity_est", "getiri_pct", "kapanan", "kapanan_wr", "sermaye"} <= alanlar["P5"]
    assert {"kod", "ad", "equity_est", "getiri_pct", "sermaye", "durum", "sinyal_sayisi"} <= alanlar["P1"]
    assert {"kod", "ad", "equity_est", "getiri_pct", "sermaye", "durum", "sinyal_sayisi"} <= alanlar["P2"]
    assert "worst" in alanlar["P4"] and "best" in alanlar["P4"] and "donem" in alanlar["P4"]


# ---------------------------------------------------------------------------
# 2. Determinism (mock fiyatla replay edilebilirlik)
# ---------------------------------------------------------------------------

def test_rapor_olustur_deterministik_mock_fiyat(mock_fiyat_sabit):
    r1 = mar.rapor_olustur()
    r2 = mar.rapor_olustur()
    t1, t2 = r1.pop("tarih"), r2.pop("tarih")  # timestamp akışkan — hariç tutulur
    assert isinstance(t1, str) and isinstance(t2, str)
    s1 = json.dumps(r1, sort_keys=True, ensure_ascii=False, default=str)
    s2 = json.dumps(r2, sort_keys=True, ensure_ascii=False, default=str)
    assert s1 == s2


# ---------------------------------------------------------------------------
# 3. Migration golden — mock 100.0 ile beklenen değerler
# ---------------------------------------------------------------------------

def test_rapor_mock_fiyat_golden_degerler(mock_fiyat_sabit):
    """FAZ 3.2 migrasyonu sonrasında AYNI mock girdisiyle AYNI değerler beklenir."""
    r = mar.rapor_olustur()
    beklenen = {
        "P3": (274000, 174.0, 16, 12.5),
        "P4": (96998, -3.0, 36, 36.1),
        "P5": (101900, 1.9, 10, 40.0),
        "P1": (100000, 0.0, None, None),
        "P2": (100000, 0.0, None, None),
    }
    for p in r["portfoyler"]:
        eq, g, kapanan, wr = beklenen[p["kod"]]
        assert p["equity_est"] == eq, f"{p['kod']} equity_est"
        assert p["getiri_pct"] == g, f"{p['kod']} getiri_pct"
        assert p.get("kapanan") == kapanan, f"{p['kod']} kapanan"
        assert p.get("kapanan_wr") == wr, f"{p['kod']} kapanan_wr"


def test_rapor_siralama_azalan_getiri(mock_fiyat_sabit):
    r = mar.rapor_olustur()
    degerler = [p["getiri_pct"] for p in r["portfoyler"]]
    assert degerler == sorted(degerler, reverse=True)
    assert [p["kod"] for p in r["portfoyler"]] == ["P3", "P5", "P1", "P2", "P4"]


def test_p1_p2_equity_mevcut_davranis_kilit(mock_fiyat_sabit):
    """P1/P2: state'te sermaye_mevcut yok → equity_est=başlangıç, getiri=%0.
    FAZ 0 artefaktında '+0.00% → ~100,000 TL' olarak kayıtlı mevcut davranış."""
    assert mar._portfoy_getiri("portfoy.json", "P1 Momentum")["equity_est"] == 100000
    assert mar._portfoy_getiri("portfoy_p2.json", "P2 SMC")["getiri_pct"] == 0.0
    assert mar._portfoy_getiri("yok_dosya.json", "X") is None  # dosya yok → None


# ---------------------------------------------------------------------------
# 4. Telegram format kilidi
# ---------------------------------------------------------------------------

def test_telegram_metin_format(mock_fiyat_sabit):
    r = mar.rapor_olustur()
    satirlar = mar.telegram_metin(r).splitlines()
    assert satirlar[0] == "📊 *MOTT — Aylık Portföy Karşılaştırması*"
    assert satirlar[1].startswith("📅 ") and "TSİ" in satirlar[1]
    assert "Başlangıç sermayesi: *100,000 TL*" in satirlar[2]
    assert satirlar[4] == "*Sıralama (tahmini getiri):*"
    # 5 sıralama satırı: "N. emoji `kod` ad: *%* → ~TL"
    for i, satir in enumerate(satirlar[5:10], 1):
        assert satir.startswith(f"{i}. "), satir
        assert ("🟢" in satir or "🔴" in satir) and "`" in satir and "%" in satir
    assert satirlar[11] == "_P1/P2: portfoy dosyası yoksa getiri hesaplanamaz._"
    assert satirlar[12] == "_P4/P5: P1+P2+P3 sinyallerinden; birbirinden bağımsız._"
    assert len(satirlar) == 13


def test_telegram_veri_yok_dali():
    """getiri_pct=None → 'veri yok (sadece tarama)' satırı (mevcut format)."""
    rapor = {
        "tarih": "test",
        "sermaye_portfoy": 100000,
        "portfoyler": [
            {"kod": "P1", "ad": "Momentum", "getiri_pct": None, "equity_est": 100000},
        ],
    }
    metin = mar.telegram_metin(rapor)
    assert "`P1` Momentum — veri yok (sadece tarama)" in metin


# ---------------------------------------------------------------------------
# 5. Migration köprüsü — normalize() alan kapsamı (dokümantasyon testi)
# ---------------------------------------------------------------------------

def test_normalize_alan_kapsami_rapor_ihtiyacini_karsiliyor():
    """Raporun ihtiyaç duyduğu pozisyon/islem alanları normalize'de mevcut:
    symbol, giris_fiyat, guncel_fiyat, pnl_pct, neden, tarihler."""
    poz_alanlar = {"symbol", "giris_fiyat", "guncel_fiyat", "pnl_pct", "giris_tarih", "lot"}
    islem_alanlar = {"symbol", "giris_fiyat", "cikis_fiyat", "pnl_pct", "neden", "giris_tarih", "cikis_tarih"}
    for kod in ("P1", "P2", "P3", "P4", "P5"):
        n = normalize(kod)
        for poz in n["pozisyonlar"]:
            assert poz_alanlar <= set(poz.keys()), f"{kod} pozisyon: {poz}"
        for t in n["islem_gecmisi"]:
            assert islem_alanlar <= set(t.keys()), f"{kod} islem: {t}"
    # nakit/equity semantiği (FAZ 3.1 sözleşmesi)
    assert normalize("P1")["nakit"] is not None and normalize("P1")["equity"] is None
    assert normalize("P3")["nakit"] is None and normalize("P3")["equity"] is None
    assert normalize("P4")["equity"] == normalize("P4")["nakit"]  # sermaye_mevcut
