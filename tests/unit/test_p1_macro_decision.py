# -*- coding: utf-8 -*-
"""
tests/unit/test_p1_macro_decision.py
===================================
P1 Makro Karar Güvenilirliği Testleri:
1. Sabah kararı metadata (zaman, geçerlilik, kaynak, run_id) paketleme
2. get_aktif_makro_karar doğrulama (geçerli, eksik, bozuk, süresi geçmiş)
3. Geçersiz/süresi geçmiş kararda yeni alımların engellenmesi
4. Geçersiz/GIRME kararda mevcut pozisyonların bağımsız STOP/TRAILING kontrollerinin sürmesi
5. Tutarlı acil tasfiye kararı (karar ve skorun aynı değerlendirmeden gelmesi)
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

import portfoy_yonetici as py


@pytest.fixture(autouse=True)
def _isolate_audit_file(tmp_path, monkeypatch):
    monkeypatch.setattr(py, "PORTFOY_AUDIT_FILE", tmp_path / "portfolio_actions.jsonl")


def _mock_bar(open_p=100.0, high_p=102.0, low_p=98.0, close_p=100.0):
    return {"open": open_p, "high": high_p, "low": low_p, "close": close_p}


class TestP1MacroDecision:
    def test_makro_karar_olustur_metadata(self):
        ref_dt = datetime(2026, 10, 6, 9, 0)
        payload = py.makro_karar_olustur(
            skor=25.0,
            karar="NORMAL",
            detaylar=[{"endeks": "^GSPC", "tetiklendi": False}],
            piyasa_ret={"^GSPC": 0.5},
            kaynak="sabah_09_akisi",
            now=ref_dt,
        )

        assert payload["karar"] == "NORMAL"
        assert payload["skor"] == 25.0
        assert payload["gecerlilik_tarih"] == "2026-10-06"
        assert payload["kaynak"] == "sabah_09_akisi"
        assert payload["valid"] is True
        assert "run_id" in payload

    def test_get_aktif_makro_karar_gecerli(self):
        ref_dt = datetime(2026, 10, 6, 11, 0)
        portfoy = {
            "makro_karar": {
                "karar": "NORMAL",
                "skor": 15.0,
                "zaman": datetime(2026, 10, 6, 9, 0).isoformat(),
                "gecerlilik_tarih": "2026-10-06",
                "kaynak": "sabah_09_akisi",
                "run_id": "run_test_1",
            }
        }
        karar, skor, meta = py.get_aktif_makro_karar(portfoy, now=ref_dt)
        assert karar == "NORMAL"
        assert skor == 15.0
        assert meta["valid"] is True
        assert meta["run_id"] == "run_test_1"

    def test_get_aktif_makro_karar_eksik(self, tmp_path, monkeypatch):
        ref_dt = datetime(2026, 10, 6, 11, 0)
        # Portföyde makro karar yok, dosyalar da boş
        monkeypatch.setattr(py, "PORTFOY_FILE", tmp_path / "empty_portfoy.json")
        monkeypatch.setattr(py, "STATE_P1_FILE", tmp_path / "empty_state.json")
        monkeypatch.setattr(py, "DURUM_FILE", tmp_path / "empty_durum.json")

        portfoy = {}
        karar, skor, meta = py.get_aktif_makro_karar(portfoy, now=ref_dt)
        # Eksik kararda güvenli taraf (GIRME) seçilmeli ve valid=False olmalı
        assert karar == "GIRME"
        assert meta["valid"] is False
        assert meta["reason"] == "makro_karar_yok"

    def test_get_aktif_makro_karar_bozuk(self):
        ref_dt = datetime(2026, 10, 6, 11, 0)
        portfoy = {
            "makro_karar": {
                "karar": "GECERSIZ_KARAR",
                "skor": 10.0,
                "gecerlilik_tarih": "2026-10-06",
            }
        }
        karar, skor, meta = py.get_aktif_makro_karar(portfoy, now=ref_dt)
        assert karar == "GIRME"
        assert meta["valid"] is False
        assert meta["reason"] == "bozuk_karar_degeri"

    def test_get_aktif_makro_karar_suresi_gecmis_dun(self):
        ref_dt = datetime(2026, 10, 6, 11, 0)
        # Dünden kalma karar (5 Ekim)
        portfoy = {
            "makro_karar": {
                "karar": "NORMAL",
                "skor": 15.0,
                "zaman": datetime(2026, 10, 5, 9, 0).isoformat(),
                "gecerlilik_tarih": "2026-10-05",
                "kaynak": "sabah_09_akisi",
            }
        }
        karar, skor, meta = py.get_aktif_makro_karar(portfoy, now=ref_dt)
        assert karar == "GIRME"
        assert meta["valid"] is False
        assert meta["reason"] == "suresi_gecmis_karar"

    def test_gecersiz_karar_yeni_alimi_engeller(self):
        """Makro karar eksik/geçersiz olduğunda adaylar olsa dahi alım engellenmeli."""
        portfoy = {"nakit": 100000.0, "pozisyonlar": {}}
        adaylar = [{"symbol": "THYAO", "final_score": 85.0}]
        viop_bias = {"size_factor": 1.0}
        meta_invalid = {"valid": False, "reason": "suresi_gecmis_karar"}

        p_sonuc, mesajlar, alinan, alinmayan = py.yeni_pozisyon_ac(
            portfoy, adaylar, "GIRME", viop_bias, makro_meta=meta_invalid
        )

        assert len(alinan) == 0
        assert len(alinmayan) == 1
        assert alinmayan[0]["symbol"] == "THYAO"
        assert alinmayan[0]["reason"] == "makro_gecersiz"

    def test_girme_kararinda_mevcut_pozisyon_stop_calisir(self, tmp_path):
        """Makro karar GIRME olduğunda açık pozisyonların STOP kontrolleri bağımsız olarak çalışır."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "GARAN": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": "01.10.2026 10:00",
                }
            },
            "trade_history": [],
        }

        # Stop tetikleyen bar (%6 düşüş: low 94)
        stop_bar = {"open": 98.0, "high": 99.0, "low": 94.0, "close": 94.5}
        with patch.object(py, "saatlik_bar", return_value=stop_bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(
                portfoy, makro_karar="GIRME", makro_skor=35.0, now=ref_dt
            )

        assert "GARAN" not in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 1
        assert p_sonuc["trade_history"][0]["neden"] == "STOP"

    def test_tutarsiz_acil_tasfiye_engellenir(self, tmp_path):
        """Makro karar GIRME fakat skor acil tasfiye eşiğinin (75) altındaysa acil tasfiye yapılmamalı."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "GARAN": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": "01.10.2026 10:00",
                }
            },
            "trade_history": [],
        }

        # Normal bar, stop/trailing tetiklenmiyor
        normal_bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        with patch.object(py, "saatlik_bar", return_value=normal_bar), \
             patch.object(py, "guncel_makro_skoru", return_value=90.0):
            # makro_skor parametresi 60.0 (tasfiye eşiği 75'in altında)
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(
                portfoy, makro_karar="GIRME", makro_skor=60.0, now=ref_dt
            )

        # Pozisyon kapatılmamalıdır (eski/tutarsız guncel_makro_skoru ile acil tasfiye üretilmez)
        assert "GARAN" in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 0

    def test_tutarlı_acil_tasfiye_calisir(self, tmp_path):
        """Makro karar GIRME ve skor tutarlı olarak >= 75 ise acil nakit tasfiyesi çalışır."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "GARAN": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": "01.10.2026 10:00",
                }
            },
            "trade_history": [],
        }

        normal_bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        mock_fiyat_detay = {"symbol": "GARAN", "price": 102.0, "source": "tradingview", "time": "2026-10-06T11:00:00", "valid": True, "trade_eligible": True}
        with patch.object(py, "saatlik_bar", return_value=normal_bar), \
             patch.object(py, "guncel_fiyat_detayli", return_value=mock_fiyat_detay):
            # makro_skor 80.0 >= EMERGENCY_LIQUIDATION_SCORE (75.0)
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(
                portfoy, makro_karar="GIRME", makro_skor=80.0, now=ref_dt
            )

        assert "GARAN" not in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 1
        assert p_sonuc["trade_history"][0]["neden"] == "ACIL_NAKIT"
        assert p_sonuc["nakit"] == pytest.approx(60184.7051)  # 100*101.898 less 5.0949 sell commission
