# -*- coding: utf-8 -*-
"""
tests/unit/test_p1_holding_period.py
===================================
P1 Tutma Süresi (Holding Period) ve MAX_GUN Rolling Extension Testleri:
1. Uzatılmış max_gun_date gelmeden MAX_GUN satışı yapılmaması (11., 14. gün)
2. Süre uzatması yapıldığında dahi STOP, TRAILING risk kontrollerinin korunması
3. 10. gün ilk uzatma, 15. gün ikinci uzatma
4. Aday listesinde bulunmama durumunda MAX_GUN çıkışı
5. Aday listesinin eksik veya eski olması ile geçerli listede adayın bulunmamasının ayrılması
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


class TestP1HoldingPeriod:
    def test_elde_tutma_gunu_hesaplama(self):
        bugun = date(2026, 10, 6)
        # 10 takvim günü önce
        giris_10 = (bugun - timedelta(days=10)).strftime("%d.%m.%Y 10:00")
        assert py._elde_tutma_gunu(giris_10, now_date=bugun) == 10

        # ISO format desteği
        giris_iso = (bugun - timedelta(days=14)).strftime("%Y-%m-%d")
        assert py._elde_tutma_gunu(giris_iso, now_date=bugun) == 14

    def test_gun_10_aday_listesinde_varsa_uzatilir(self, tmp_path):
        ref_dt = datetime(2026, 10, 6, 14, 0)
        ref_date = ref_dt.date()
        giris_t = (ref_date - timedelta(days=10)).strftime("%d.%m.%Y 10:00")

        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": giris_t,
                    "max_gun_date": ref_date.strftime("%d.%m.%Y"),
                }
            },
            "trade_history": [],
        }

        # Aday listesinde THYAO var ve tarama taze
        tarama_file = tmp_path / "tarama_listesi.json"
        tarama_data = {
            "scan_time": ref_dt.strftime("%d.%m.%Y 09:00"),
            "signals": [{"symbol": "THYAO"}, {"symbol": "ASELS"}],
        }
        tarama_file.write_text(json.dumps(tarama_data), encoding="utf-8")

        with patch.object(py, "TARAMA_FILE", tarama_file), \
             patch.object(py, "saatlik_bar", return_value=_mock_bar()):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert "THYAO" in p_sonuc["pozisyonlar"]
        beklenen_mgd = (ref_date + timedelta(days=py.MAX_GUN_EXTENSION)).strftime("%d.%m.%Y")
        assert p_sonuc["pozisyonlar"]["THYAO"]["max_gun_date"] == beklenen_mgd
        assert len(p_sonuc["trade_history"]) == 0

    def test_gun_11_uzatilmis_tarihe_kadar_satilmaz(self, tmp_path):
        """11. gün: max_gun_date 15. güne uzatılmışsa, aday listesinde bugün olmasa dahi satılmamalı."""
        ref_dt = datetime(2026, 10, 6, 14, 0)
        ref_date = ref_dt.date()
        giris_t = (ref_date - timedelta(days=11)).strftime("%d.%m.%Y 10:00")
        uzatilmis_mgd = (ref_date + timedelta(days=4)).strftime("%d.%m.%Y")  # 15. gün

        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": giris_t,
                    "max_gun_date": uzatilmis_mgd,
                }
            },
            "trade_history": [],
        }

        # Tarama listesinde THYAO YOK
        tarama_file = tmp_path / "tarama_listesi.json"
        tarama_data = {
            "scan_time": ref_dt.strftime("%d.%m.%Y 09:00"),
            "signals": [{"symbol": "GARAN"}],
        }
        tarama_file.write_text(json.dumps(tarama_data), encoding="utf-8")

        with patch.object(py, "TARAMA_FILE", tarama_file), \
             patch.object(py, "saatlik_bar", return_value=_mock_bar()):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        # 11. günde henüz 15. gün gelmediği için pozisyon korunmalı!
        assert "THYAO" in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 0

    def test_gun_14_uzatilmis_tarihe_kadar_satilmaz(self, tmp_path):
        """14. gün: max_gun_date 15. gün; pozisyon korunmalı."""
        ref_dt = datetime(2026, 10, 6, 14, 0)
        ref_date = ref_dt.date()
        giris_t = (ref_date - timedelta(days=14)).strftime("%d.%m.%Y 10:00")
        uzatilmis_mgd = (ref_date + timedelta(days=1)).strftime("%d.%m.%Y")  # yarın 15. gün

        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": giris_t,
                    "max_gun_date": uzatilmis_mgd,
                }
            },
            "trade_history": [],
        }

        tarama_file = tmp_path / "tarama_listesi.json"
        tarama_data = {
            "scan_time": ref_dt.strftime("%d.%m.%Y 09:00"),
            "signals": [],
        }
        tarama_file.write_text(json.dumps(tarama_data), encoding="utf-8")

        with patch.object(py, "TARAMA_FILE", tarama_file), \
             patch.object(py, "saatlik_bar", return_value=_mock_bar()):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert "THYAO" in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 0

    def test_gun_15_ikinci_kez_uzatma(self, tmp_path):
        """15. gün: max_gun_date gününe gelindi, aday listesinde tekrar varsa 20. güne uzatılır."""
        ref_dt = datetime(2026, 10, 6, 14, 0)
        ref_date = ref_dt.date()
        giris_t = (ref_date - timedelta(days=15)).strftime("%d.%m.%Y 10:00")
        bugun_mgd = ref_date.strftime("%d.%m.%Y")

        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": giris_t,
                    "max_gun_date": bugun_mgd,
                    "extension_count": 1,
                }
            },
            "trade_history": [],
        }

        tarama_file = tmp_path / "tarama_listesi.json"
        tarama_data = {
            "scan_time": ref_dt.strftime("%d.%m.%Y 09:00"),
            "signals": [{"symbol": "THYAO"}],
        }
        tarama_file.write_text(json.dumps(tarama_data), encoding="utf-8")

        with patch.object(py, "TARAMA_FILE", tarama_file), \
             patch.object(py, "saatlik_bar", return_value=_mock_bar()):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert "THYAO" in p_sonuc["pozisyonlar"]
        beklenen_mgd = (ref_date + timedelta(days=py.MAX_GUN_EXTENSION)).strftime("%d.%m.%Y")
        assert p_sonuc["pozisyonlar"]["THYAO"]["max_gun_date"] == beklenen_mgd
        assert p_sonuc["pozisyonlar"]["THYAO"]["extension_count"] == 2
        assert len(p_sonuc["trade_history"]) == 0

    def test_gun_15_aday_listesinde_yoksa_satilir(self, tmp_path):
        """15. gün: max_gun_date geldi, aday listesinde yoksa MAX_GUN ile kapatılır."""
        ref_dt = datetime(2026, 10, 6, 14, 0)
        ref_date = ref_dt.date()
        giris_t = (ref_date - timedelta(days=15)).strftime("%d.%m.%Y 10:00")
        bugun_mgd = ref_date.strftime("%d.%m.%Y")

        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": giris_t,
                    "max_gun_date": bugun_mgd,
                }
            },
            "trade_history": [],
        }

        # Geçerli listede THYAO yok
        tarama_file = tmp_path / "tarama_listesi.json"
        tarama_data = {
            "scan_time": ref_dt.strftime("%d.%m.%Y 09:00"),
            "signals": [{"symbol": "KCHOL"}],
        }
        tarama_file.write_text(json.dumps(tarama_data), encoding="utf-8")

        with patch.object(py, "TARAMA_FILE", tarama_file), \
             patch.object(py, "saatlik_bar", return_value=_mock_bar(close_p=105.0)):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert "THYAO" not in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 1
        trade = p_sonuc["trade_history"][0]
        assert trade["symbol"] == "THYAO"
        assert trade["neden"] == "MAX_GUN"
        assert trade["cikis_fiyat"] == 105.0
        net_tutar, _ = py.hesapla_net_tutar(100, 105.0)
        assert p_sonuc["nakit"] == pytest.approx(50000.0 + net_tutar, abs=0.01)

    def test_uzatma_stop_risk_kontrolunu_atlamaz(self, tmp_path):
        """Uzatılmış pozisyon 11. günde STOP seviyesine düşerse derhal STOP ile kapanmalıdır."""
        ref_dt = datetime(2026, 10, 6, 14, 0)
        ref_date = ref_dt.date()
        giris_t = (ref_date - timedelta(days=11)).strftime("%d.%m.%Y 10:00")
        uzatilmis_mgd = (ref_date + timedelta(days=4)).strftime("%d.%m.%Y")

        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": giris_t,
                    "max_gun_date": uzatilmis_mgd,
                }
            },
            "trade_history": [],
        }

        tarama_file = tmp_path / "tarama_listesi.json"
        tarama_data = {
            "scan_time": ref_dt.strftime("%d.%m.%Y 09:00"),
            "signals": [{"symbol": "THYAO"}],  # Listede olsa dahi stop önceliklidir
        }
        tarama_file.write_text(json.dumps(tarama_data), encoding="utf-8")

        # Stop tetikleyen bar: low %6 düştü (94 TL)
        stop_bar = {"open": 98.0, "high": 99.0, "low": 94.0, "close": 94.5}
        with patch.object(py, "TARAMA_FILE", tarama_file), \
             patch.object(py, "saatlik_bar", return_value=stop_bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert "THYAO" not in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 1
        assert p_sonuc["trade_history"][0]["neden"] == "STOP"

    def test_uzatma_trailing_risk_kontrolunu_atlamaz(self, tmp_path):
        """TP1 yapılmış ve uzatılmış pozisyon trailing seviyesini delerse TRAILING ile kapanır."""
        ref_dt = datetime(2026, 10, 6, 14, 0)
        ref_date = ref_dt.date()
        giris_t = (ref_date - timedelta(days=12)).strftime("%d.%m.%Y 10:00")
        uzatilmis_mgd = (ref_date + timedelta(days=3)).strftime("%d.%m.%Y")

        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 120.0,  # zirve 120
                    "lotlar": 50,
                    "tp1_yapildi": True,
                    "giris_t": giris_t,
                    "max_gun_date": uzatilmis_mgd,
                }
            },
            "trade_history": [],
        }

        # Trailing tetikleyen bar: tepe 120'den %5'ten fazla düştü (low=113, -%5.8)
        trail_bar = {"open": 116.0, "high": 117.0, "low": 113.0, "close": 113.5}
        with patch.object(py, "saatlik_bar", return_value=trail_bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert "THYAO" not in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 1
        assert p_sonuc["trade_history"][0]["neden"] == "TRAILING"

    def test_aday_listesi_stale_ile_aday_yok_ayrimi(self, tmp_path):
        """Eski/bayat tarama listesi durumunda data_error kaydedilmesi."""
        ref_dt = datetime(2026, 10, 6, 14, 0)
        ref_date = ref_dt.date()
        giris_t = (ref_date - timedelta(days=10)).strftime("%d.%m.%Y 10:00")

        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": giris_t,
                    "max_gun_date": ref_date.strftime("%d.%m.%Y"),
                }
            },
            "trade_history": [],
        }

        # 10 gün önceki bayat tarama
        eski_tarama_dt = ref_dt - timedelta(days=10)
        tarama_file = tmp_path / "tarama_listesi.json"
        tarama_data = {
            "scan_time": eski_tarama_dt.strftime("%d.%m.%Y 09:00"),
            "signals": [{"symbol": "THYAO"}],
        }
        tarama_file.write_text(json.dumps(tarama_data), encoding="utf-8")

        audit_events = []
        def _mock_append_jsonl(path, payload):
            audit_events.append(payload)

        with patch.object(py, "TARAMA_FILE", tarama_file), \
             patch.object(py, "saatlik_bar", return_value=_mock_bar()), \
             patch.object(py, "append_jsonl", side_effect=_mock_append_jsonl):
            p_sonuc, _ = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        # Bayat tarama nedeniyle uzatma VERİLMEZ, güvenli çıkış yapılır
        assert "THYAO" not in p_sonuc["pozisyonlar"]
        # Audit kaydında veri hatası ("stale") ayrıştırılmış olmalı
        error_events = [e for e in audit_events if e.get("event") == "max_gun_data_error"]
        assert len(error_events) >= 1
        assert error_events[0]["status"] == "stale"
