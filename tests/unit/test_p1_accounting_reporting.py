# -*- coding: utf-8 -*-
"""
tests/unit/test_p1_accounting_reporting.py
=========================================
P1 Muhasebe ve Raporlama Güvenilirliği Testleri:
1. Pozisyon kimliği (position_id) ve olay kimliği (event_id) üretimi
2. TP1 ve nihai çıkışın (STOP/TRAILING/MAX_GUN) aynı position_id'ye bağlanması
3. Nakit ve lot hareketlerinin işlem defterinden (islem_defteri) %100 mutabakatla yeniden hesaplanabilmesi
4. P1 aylık raporunda sabit 100.000 TL ve %0.0 değerlerinin kaldırılması, dinamik equity ve getiri hesabı
5. Başlangıçtan getiri ile aylık getirinin ayrılması, ay başı kaydı yoksa None olması
6. Satış olayı sayısı ve tamamlanan pozisyon sayısının ayrı raporlanması, parasal kâr faktörü
7. İleriye dönük günlük equity kaydı oluşturulması
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

import portfoy_yonetici as py
import mott_aylik_rapor as mar


@pytest.fixture(autouse=True)
def _isolate_audit_file(tmp_path, monkeypatch):
    monkeypatch.setattr(py, "PORTFOY_AUDIT_FILE", tmp_path / "portfolio_actions.jsonl")
    monkeypatch.setattr(py, "PORTFOY_FILE", tmp_path / "test_portfoy.json")


class TestP1AccountingReporting:
    def test_yeni_islem_position_id_ve_event_id_uretir(self, monkeypatch):
        portfoy = {"nakit": 100000.0, "pozisyonlar": {}}
        adaylar = [{"symbol": "THYAO", "final_score": 80.0}]
        viop_bias = {"size_factor": 1.0}

        monkeypatch.setattr(py, "guncel_fiyat_detayli", lambda sym, **kw: {
            "symbol": sym, "price": 100.0, "source": "tradingview", "time": "2026-10-06T10:00:00", "valid": True, "trade_eligible": True
        })

        p_sonuc, mesajlar, alinan, alinmayan = py.yeni_pozisyon_ac(
            portfoy, adaylar, "NORMAL", viop_bias
        )

        assert "THYAO" in p_sonuc["pozisyonlar"]
        pos = p_sonuc["pozisyonlar"]["THYAO"]
        assert "position_id" in pos
        assert pos["position_id"].startswith("P1_THYAO_")
        assert "entry_event_id" in pos

        # İşlem defteri (islem_defteri) kaydı
        assert "islem_defteri" in p_sonuc
        defter = p_sonuc["islem_defteri"]
        assert len(defter) == 1
        entry = defter[0]
        assert entry["symbol"] == "THYAO"
        assert entry["islem_tipi"] == "ALIS"
        assert entry["position_id"] == pos["position_id"]
        assert entry["nakit_etkisi"] < 0  # Nakit çıkışı

    def test_tp1_ve_son_cikis_ayni_position_idye_baglanir(self, monkeypatch):
        ref_dt = datetime(2026, 10, 6, 11, 0)
        pos_id = "P1_THYAO_20261001_99999"
        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "position_id": pos_id,
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": "01.10.2026 10:00",
                    "max_gun_date": "15.10.2026",
                    "tp1_yapildi": False,
                }
            },
            "trade_history": [],
            "islem_defteri": [],
        }

        # 1. Bar: TP1 tetiklenir (High 109)
        tp_bar = {
            "open": 102.0, "high": 109.0, "low": 101.0, "close": 107.0, "volume": 1000,
            "bar_time": "2026-10-06T10:00:00"
        }
        monkeypatch.setattr(py, "saatlik_bar", lambda sym, now=None: tp_bar)
        p_sonuc, _ = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert len(p_sonuc["trade_history"]) == 1
        tp1_trade = p_sonuc["trade_history"][0]
        assert tp1_trade["position_id"] == pos_id
        assert tp1_trade["neden"] == "TP1"
        assert tp1_trade["lotlar"] == 50

        # 2. Bar: Kalan 50 lot Trailing ile kapanır
        ref_dt_2 = ref_dt + timedelta(hours=1)
        trail_bar = {
            "open": 106.0, "high": 107.0, "low": 103.0, "close": 103.5, "volume": 1000,
            "bar_time": "2026-10-06T11:00:00"
        }
        monkeypatch.setattr(py, "saatlik_bar", lambda sym, now=None: trail_bar)
        p_sonuc, _ = py.pozisyon_guncelle_saatlik(p_sonuc, "NORMAL", now=ref_dt_2)

        assert "THYAO" not in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 2
        trail_trade = p_sonuc["trade_history"][1]
        assert trail_trade["position_id"] == pos_id
        assert trail_trade["neden"] == "TRAILING"
        assert trail_trade["lotlar"] == 50

        # TP1 ve Trailing aynı position_id'ye sahip olmalı
        assert tp1_trade["position_id"] == trail_trade["position_id"] == pos_id

    def test_islem_defterinden_nakit_ve_lot_mutabakati(self, monkeypatch):
        """İşlem defterindeki tüm nakit_etkisi ve lot hareketleri mevcut nakit ve açık lotlarla birebir tutmalıdır."""
        portfoy = {"baslangic": 100000.0, "nakit": 100000.0, "pozisyonlar": {}, "islem_defteri": []}
        adaylar = [{"symbol": "THYAO", "final_score": 80.0}]
        viop_bias = {"size_factor": 1.0}

        # 1. Alış işlemi
        monkeypatch.setattr(py, "guncel_fiyat_detayli", lambda sym, **kw: {
            "symbol": sym, "price": 100.0, "source": "tradingview", "time": "2026-10-06T10:00:00", "valid": True, "trade_eligible": True
        })
        p, _, alinan, _ = py.yeni_pozisyon_ac(portfoy, adaylar, "NORMAL", viop_bias)
        assert "THYAO" in alinan
        alinan_lot = p["pozisyonlar"]["THYAO"]["lotlar"]

        # Defterden nakit doğrulaması (Alış sonrası)
        defter_nakit_etkisi = sum(x["nakit_etkisi"] for x in p["islem_defteri"])
        assert round(100000.0 + defter_nakit_etkisi, 2) == round(p["nakit"], 2)

        # 2. TP1 satışı
        ref_dt = datetime(2026, 10, 6, 11, 0)
        tp_bar = {
            "open": 102.0, "high": 109.0, "low": 101.0, "close": 107.0, "volume": 1000,
            "bar_time": "2026-10-06T10:00:00"
        }
        monkeypatch.setattr(py, "saatlik_bar", lambda sym, now=None: tp_bar)
        p, _ = py.pozisyon_guncelle_saatlik(p, "NORMAL", now=ref_dt)

        # Defterden nakit doğrulaması (TP1 sonrası)
        defter_nakit_etkisi = sum(x["nakit_etkisi"] for x in p["islem_defteri"])
        assert round(100000.0 + defter_nakit_etkisi, 2) == round(p["nakit"], 2)

        # Defterden lot doğrulaması: Alınan lot - satılan lot == kalan açık lot
        toplam_alis_lot = sum(x["lot"] for x in p["islem_defteri"] if x["islem_tipi"] == "ALIS")
        toplam_satis_lot = sum(x["lot"] for x in p["islem_defteri"] if x["islem_tipi"].startswith("SATIS"))
        kalan_lot = toplam_alis_lot - toplam_satis_lot
        assert kalan_lot == p["pozisyonlar"]["THYAO"]["lotlar"]

    def test_aylik_rapor_p1_dinamik_equity_ve_getiri(self, monkeypatch):
        """P1 aylık rapor bloğu sabit %0.0 yerine gerçek nakit ve açık pozisyon piyasa değerini hesaplamalıdır."""
        mock_normalized_p1 = {
            "strateji": "P1",
            "baslangic_sermayesi": 100000.0,
            "nakit": 70000.0,
            "pozisyonlar": [
                {"symbol": "THYAO", "lot": 200, "guncel_fiyat": 150.0, "giris_fiyat": 140.0},
                {"symbol": "ASELS", "lot": 100, "guncel_fiyat": 100.0, "giris_fiyat": 90.0},
            ],
            "islem_gecmisi": [
                {"symbol": "GARAN", "giris_fiyat": 100.0, "cikis_fiyat": 110.0, "lotlar": 100, "tl_kar": 1000.0, "neden": "TP1"},
                {"symbol": "GARAN", "giris_fiyat": 100.0, "cikis_fiyat": 115.0, "lotlar": 100, "tl_kar": 1500.0, "neden": "TRAILING"},
                {"symbol": "KCHOL", "giris_fiyat": 200.0, "cikis_fiyat": 190.0, "lotlar": 50, "tl_kar": -500.0, "neden": "STOP"},
            ],
        }

        monkeypatch.setattr(mar, "normalize", lambda kod: mock_normalized_p1 if kod == "P1" else mar.normalize(kod))
        monkeypatch.setattr(mar, "_yukle", lambda fname: {"ay_basi_equity": 105000.0})
        monkeypatch.setattr(py, "guncel_fiyat_detayli", lambda symbol, **kw: {"price": 150 if symbol == "THYAO" else 100, "valuation_valid": True})

        p1_blok = mar._p1_p2_rapor_blok("P1")

        # Açık pozisyon değeri = (200 * 150) + (100 * 100) = 30.000 + 10.000 = 40.000 TL
        # Toplam equity = 70.000 + 40.000 = 110.000 TL
        assert p1_blok["equity_est"] == 110000
        # Başlangıçtan getiri = (110.000 - 100.000) / 100.000 = +%10.0
        assert p1_blok["baslangictan_getiri_pct"] == 10.0
        # Aylık getiri = (110.000 - 105.000) / 105.000 = +%4.76
        assert p1_blok["aylik_getiri_pct"] == pytest.approx(4.76, abs=0.01)

        # Satış olayı sayısı: 3 satış olayı
        assert p1_blok["satis_olayi_sayisi"] == 3
        # Tamamlanan pozisyon sayısı: TP1 hariç 2 pozisyon tamamlandı (GARAN trailing, KCHOL stop)
        assert p1_blok["tamamlanan_pozisyon_sayisi"] == 2
        # Kâr faktörü: Toplam kazanç (1000 + 1500 = 2500 TL) / Toplam kayıp (500 TL) = 5.0
        assert p1_blok["kar_faktoru"] == 5.0

    def test_aylik_rapor_ay_basi_equity_yoksa_hesaplanamiyor(self, monkeypatch):
        mock_normalized_p1 = {
            "strateji": "P1",
            "baslangic_sermayesi": 100000.0,
            "nakit": 100000.0,
            "pozisyonlar": [],
            "islem_gecmisi": [],
        }
        monkeypatch.setattr(mar, "normalize", lambda kod: mock_normalized_p1 if kod == "P1" else mar.normalize(kod))
        # ay_basi_equity yok
        monkeypatch.setattr(mar, "_yukle", lambda fname: {})

        p1_blok = mar._p1_p2_rapor_blok("P1")
        assert p1_blok["aylik_getiri_pct"] is None

    def test_gunluk_equity_kaydi_ileriye_donuk_olusturulur(self):
        ref_dt = datetime(2026, 10, 6, 18, 0)
        portfoy = {
            "nakit": 60000.0,
            "pozisyonlar": {
                "THYAO": {"giris_f": 100.0, "lotlar": 200, "guncel_fiyat": 105.0}
            },
            "gunluk_equity_tarihcesi": [
                {"tarih": "2026-10-05", "equity": 80000.0, "nakit": 60000.0, "acik_deger": 20000.0}
            ],
        }

        with patch.object(py, "guncel_fiyat_detayli", return_value={"price":105.0,"valuation_valid":True}):
            p = py.gunluk_equity_kaydet(portfoy, now=ref_dt)

        tarihce = p["gunluk_equity_tarihcesi"]
        assert len(tarihce) == 2
        bugun_kayit = tarihce[1]
        assert bugun_kayit["tarih"] == "2026-10-06"
        # Equity = 60.000 + (200 * 105) = 81.000 TL
        assert bugun_kayit["equity"] == 81000.0
        assert bugun_kayit["nakit"] == 60000.0
        assert bugun_kayit["acik_deger"] == 21000.0

        # Aynı gün tekrar çağrıldığında mükerrer kayıt oluşturmaz, mevcut bugünkü kaydı günceller (idempotent)
        with patch.object(py, "guncel_fiyat_detayli", return_value={"price":110.0,"valuation_valid":True}):
            p = py.gunluk_equity_kaydet(p, now=ref_dt + timedelta(hours=1))

        assert len(p["gunluk_equity_tarihcesi"]) == 2
        # Equity = 60.000 + (200 * 110) = 82.000 TL
        assert p["gunluk_equity_tarihcesi"][1]["equity"] == 82000.0
