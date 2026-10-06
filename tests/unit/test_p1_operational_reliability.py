# -*- coding: utf-8 -*-
"""
tests/unit/test_p1_operational_reliability.py
=============================================
P1 Kayıt ve Çalışma Güvenilirliği Testleri:
1. Atomik state yazımı ve _gen / _updated_at damgalaması
2. Optimistik concurrency: eski sürümün yeni sürümü ezmesini önleyen merge guard
3. BIST işlem günü, resmi tatil ve yarım seans kontrolleri
4. BIST son kapanmış saatlik mum zamanı tespiti
5. Yapılandırılmış veri hatası kaydı (sembol, veri zamanı, run_id, sebep)
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from unittest.mock import patch

import pytz
import pytest

import portfoy_yonetici as py
import mott_bist_takvim as mbt


@pytest.fixture(autouse=True)
def _isolate_audit_file(tmp_path, monkeypatch):
    monkeypatch.setattr(py, "PORTFOY_AUDIT_FILE", tmp_path / "portfolio_actions.jsonl")


class TestP1OperationalReliability:
    def test_portfoy_kaydet_atomik_ve_gen_damgali(self, tmp_path, monkeypatch):
        portfoy_path = tmp_path / "portfoy.json"
        monkeypatch.setattr(py, "PORTFOY_FILE", portfoy_path)

        portfoy = {
            "baslangic": 100000.0,
            "nakit": 95000.0,
            "pozisyonlar": {"THYAO": {"giris_f": 100.0, "lotlar": 50}},
        }

        p = py.portfoy_kaydet(portfoy)
        assert portfoy_path.exists()
        saved = json.loads(portfoy_path.read_text(encoding="utf-8"))

        assert saved["_gen"] >= 1
        assert "_updated_at" in saved
        assert "_initial_gen" not in saved  # Geçici anahtar temizlenmiş olmalı

        # İkinci kez kaydetme _gen'i artırır
        py.portfoy_kaydet(p)
        saved_2 = json.loads(portfoy_path.read_text(encoding="utf-8"))
        assert saved_2["_gen"] == saved["_gen"] + 1

    def test_optimistik_concurrency_surum_ezme_korumasi(self, tmp_path, monkeypatch):
        """Diskte daha yeni bir sürüm (_gen) varsa, eski süreç diskteki yeni veriyi ezmeyip birleştirmelidir."""
        portfoy_path = tmp_path / "portfoy.json"
        monkeypatch.setattr(py, "PORTFOY_FILE", portfoy_path)

        # Süreç 1: portföyü yüklediğinde gen=5
        portfoy_surec_1 = {
            "baslangic": 100000.0,
            "nakit": 90000.0,
            "pozisyonlar": {"THYAO": {"giris_f": 100.0, "lotlar": 100}},
            "trade_history": [{"event_id": "EVT_1", "symbol": "GARAN", "tl_kar": 500.0}],
            "_gen": 5,
            "_initial_gen": 5,
        }

        # Bu sırada başka bir süreç (Süreç 2) diske yeni bir pozisyon ve trade ile gen=6 yazdı
        portfoy_surec_2 = {
            "baslangic": 100000.0,
            "nakit": 80000.0,
            "pozisyonlar": {
                "THYAO": {"giris_f": 100.0, "lotlar": 100},
                "ASELS": {"giris_f": 50.0, "lotlar": 200},  # Yeni pozisyon
            },
            "trade_history": [
                {"event_id": "EVT_1", "symbol": "GARAN", "tl_kar": 500.0},
                {"event_id": "EVT_2", "symbol": "KCHOL", "tl_kar": -200.0},  # Yeni trade
            ],
            "_gen": 6,
        }
        portfoy_path.write_text(json.dumps(portfoy_surec_2), encoding="utf-8")

        # Süreç 1 işini bitirip kaydettiğinde Süreç 2'nin yazdığı ASELS ve EVT_2 korunmalıdır
        p_sonuc = py.portfoy_kaydet(portfoy_surec_1)
        saved = json.loads(portfoy_path.read_text(encoding="utf-8"))

        assert "ASELS" in saved["pozisyonlar"]
        trade_event_ids = [t["event_id"] for t in saved["trade_history"]]
        assert "EVT_2" in trade_event_ids
        assert saved["_gen"] >= 7

    def test_bist_islem_gunu_ve_tatiller(self):
        # Hafta sonu
        cumartesi = date(2026, 10, 10)
        pazar = date(2026, 10, 11)
        assert mbt.is_bist_islem_gunu(cumartesi) is False
        assert mbt.is_bist_islem_gunu(pazar) is False

        # Hafta içi işlem günü (Salı)
        sali = date(2026, 10, 6)
        assert mbt.is_bist_islem_gunu(sali) is True
        assert mbt.is_bist_yarim_gun(sali) is False
        assert mbt.bist_seans_saatleri(sali) == (time(10, 0), time(18, 0))

        # 29 Ekim Cumhuriyet Bayramı (Resmi Tatil)
        tatil_29_ekim = date(2026, 10, 29)
        assert mbt.is_bist_islem_gunu(tatil_29_ekim) is False
        assert mbt.is_bist_tatil(tatil_29_ekim) is True

        # 28 Ekim Cumhuriyet Bayramı Arefesi (Yarım Gün)
        arefe_28_ekim = date(2026, 10, 28)
        assert mbt.is_bist_islem_gunu(arefe_28_ekim) is True
        assert mbt.is_bist_yarim_gun(arefe_28_ekim) is True
        assert mbt.bist_seans_saatleri(arefe_28_ekim) == (time(10, 0), time(12, 40))

    def test_bist_son_kapanan_saatlik_mum(self):
        IST = pytz.timezone("Europe/Istanbul")
        # 1. Seans açılmadan önce (09:45)
        dt_sabah = IST.localize(datetime(2026, 10, 6, 9, 45))
        assert mbt.son_kapanan_saatlik_mum_zamani(dt_sabah) is None

        # 2. Gün içi (14:35) -> Son kapanan mum 13:00 - 14:00 mumudur
        dt_ogle = IST.localize(datetime(2026, 10, 6, 14, 35))
        son_mum = mbt.son_kapanan_saatlik_mum_zamani(dt_ogle)
        assert son_mum is not None
        assert son_mum.hour == 13

        # 3. Kapanış sonrası (19:05) -> Son kapanan mum 17:00 - 18:00 mumudur
        dt_aksam = IST.localize(datetime(2026, 10, 6, 19, 5))
        son_mum_aksam = mbt.son_kapanan_saatlik_mum_zamani(dt_aksam)
        assert son_mum_aksam is not None
        assert son_mum_aksam.hour == 17

        # 4. Yarım gün (28 Ekim 13:15) -> 12:40 kapanışından sonraki son mum 11:00 mumu
        dt_arefe = IST.localize(datetime(2026, 10, 28, 13, 15))
        son_mum_arefe = mbt.son_kapanan_saatlik_mum_zamani(dt_arefe)
        assert son_mum_arefe is not None
        assert son_mum_arefe.hour == 11

    def test_veri_hatasi_kaydet(self):
        portfoy = {"pozisyonlar": {}, "veri_hatalari": []}
        ref_dt = datetime(2026, 10, 6, 11, 0)
        p = py.veri_hatasi_kaydet(
            portfoy,
            symbol="THYAO",
            data_time="2026-10-06T10:00:00",
            stage="price_fetch",
            reason="timeout_yfinance",
            now=ref_dt,
        )

        assert len(p["veri_hatalari"]) == 1
        err = p["veri_hatalari"][0]
        assert err["symbol"] == "THYAO"
        assert err["stage"] == "price_fetch"
        assert err["reason"] == "timeout_yfinance"
        assert "run_id" in err
        assert err["timestamp"] == ref_dt.isoformat()
