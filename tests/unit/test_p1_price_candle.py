# -*- coding: utf-8 -*-
"""
tests/unit/test_p1_price_candle.py
==================================
P1 Fiyat ve Mum Güvenilirliği Testleri:
1. guncel_fiyat_detayli kaynak ve zaman takibi
2. Eksik/geçersiz fiyatta yeni alım engelleme
3. Zaman bazlı son kapanmış saatlik mum seçimi (iloc[-2] varsayımına dayanmama)
4. Mum idempotency: aynı mum için tekrar çalıştırmada mükerrer işlem engelleme
5. Acil tasfiyede fiyat yoksa giriş fiyatından satmama, pozisyonu koruyup beklemeye alma
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import portfoy_yonetici as py


@pytest.fixture(autouse=True)
def _isolate_audit_file(tmp_path, monkeypatch):
    monkeypatch.setattr(py, "PORTFOY_AUDIT_FILE", tmp_path / "portfolio_actions.jsonl")


class TestP1PriceCandle:
    def test_guncel_fiyat_detayli_kaynak_ve_zaman(self, monkeypatch):
        ref_dt = datetime(2026, 10, 6, 11, 30)
        # Mock TradingView
        mock_tv = MagicMock()
        mock_tv.tv_fiyatlar.return_value = {"THYAO": 320.5}
        monkeypatch.setitem(sys.modules, "mott_fiyat", mock_tv)

        res = py.guncel_fiyat_detayli("THYAO", cache=False, now=ref_dt)
        assert res["valid"] is True
        assert res["price"] == 320.5
        assert res["source"] == "tradingview"
        assert res["time"] == ref_dt.isoformat()
        assert py.guncel_fiyat("THYAO") == 320.5

    def test_eksik_fiyatla_yeni_alim_engellenir(self, monkeypatch):
        # Tüm fiyat kaynakları None/geçersiz döner
        monkeypatch.setattr(py, "guncel_fiyat_detayli", lambda sym, **kw: {
            "symbol": sym, "price": None, "source": "none", "time": "", "valid": False, "reason": "fiyat_yok"
        })

        portfoy = {"nakit": 100000.0, "pozisyonlar": {}}
        adaylar = [{"symbol": "THYAO", "final_score": 80.0}]
        viop_bias = {"size_factor": 1.0}

        p_sonuc, mesajlar, alinan, alinmayan = py.yeni_pozisyon_ac(
            portfoy, adaylar, "NORMAL", viop_bias
        )

        assert len(alinan) == 0
        assert len(alinmayan) == 1
        assert alinmayan[0]["symbol"] == "THYAO"
        assert alinmayan[0]["reason"] == "veri_yok"
        assert "THYAO" not in p_sonuc["pozisyonlar"]

    def test_saatlik_bar_zaman_bazli_kapanmis_mum_secimi(self, monkeypatch):
        """Saat 12:30'da 12:00 mumu henüz tamamlanmamıştır; son kapanmış mum 11:00-12:00 mumudur."""
        ref_now = datetime(2026, 10, 6, 12, 30)

        # 3 adet bar oluştur: 10:00, 11:00, 12:00
        t10 = datetime(2026, 10, 6, 10, 0)
        t11 = datetime(2026, 10, 6, 11, 0)
        t12 = datetime(2026, 10, 6, 12, 0)

        df = pd.DataFrame([
            {"Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 1000},
            {"Open": 101.0, "High": 105.0, "Low": 100.5, "Close": 104.0, "Volume": 2000},
            {"Open": 104.0, "High": 106.0, "Low": 103.0, "Close": 105.5, "Volume": 1500},
        ], index=[t10, t11, t12])

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        monkeypatch.setattr(py, "_ticker", lambda sym: mock_ticker)

        bar = py.saatlik_bar("THYAO", now=ref_now)
        assert bar is not None
        # 12:30 itibariyle son KAPANMIŞ bar 11:00 barıdır (Close: 104.0)
        assert bar["open"] == 101.0
        assert bar["close"] == 104.0
        assert bar["bar_time"] == t11.isoformat()

    def test_mum_idempotency_ayni_mumda_tekrar_calisma_engellenir(self, monkeypatch):
        """Aynı mum daha önce değerlendirildiyse, TP1/STOP kontrolleri tekrar çalıştırılmaz."""
        ref_dt = datetime(2026, 10, 6, 11, 30)
        bar_t10 = datetime(2026, 10, 6, 10, 0).isoformat()

        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "tp1_yapildi": False,
                    "giris_t": "01.10.2026 10:00",
                    "max_gun_date": "15.10.2026",
                    "last_bar_time": bar_t10,  # 10:00 mumu daha önce işlenmiş!
                }
            },
            "trade_history": [],
        }

        # High 112 (+%12, TP1 seviyesi) olan aynı 10:00 mumu
        tp_bar = {
            "open": 100.0, "high": 112.0, "low": 99.0, "close": 111.0,
            "bar_time": bar_t10,
        }
        monkeypatch.setattr(py, "saatlik_bar", lambda sym, now=None: tp_bar)

        p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(
            portfoy, makro_karar="NORMAL", now=ref_dt
        )

        # Mükerrer çalıştırmada TP1 tekrar tetiklenmemeli, lotlar 100 olarak kalmalı!
        assert p_sonuc["pozisyonlar"]["THYAO"]["lotlar"] == 100
        assert p_sonuc["pozisyonlar"]["THYAO"]["tp1_yapildi"] is False
        assert len(p_sonuc["trade_history"]) == 0

    def test_acil_tasfiye_fiyat_yoksa_pozisyon_korunur_beklemeye_alinir(self, monkeypatch):
        """Acil tasfiyede fiyat bulunamazsa giriş fiyatından satılmaz; pozisyon korunur ve beklemeye alınır."""
        ref_dt = datetime(2026, 10, 6, 11, 30)
        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 100,
                    "giris_t": "01.10.2026 10:00",
                    "max_gun_date": "15.10.2026",
                }
            },
            "trade_history": [],
        }

        # Normal bar, stop yok
        normal_bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "bar_time": "2026-10-06T10:00:00"}
        monkeypatch.setattr(py, "saatlik_bar", lambda sym, now=None: normal_bar)

        # Canlı fiyat YOK
        monkeypatch.setattr(py, "guncel_fiyat_detayli", lambda sym, **kw: {
            "symbol": sym, "price": None, "source": "none", "time": "", "valid": False, "reason": "fiyat_yok"
        })

        p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(
            portfoy, makro_karar="GIRME", makro_skor=85.0, now=ref_dt
        )

        # Pozisyon silinmemeli!
        assert "THYAO" in p_sonuc["pozisyonlar"]
        pos = p_sonuc["pozisyonlar"]["THYAO"]
        # Pending liquidation bayrağı konmalı
        assert "pending_liquidation" in pos
        assert pos["pending_liquidation"]["reason"] == "acil_nakit_fiyat_yok"
        # Nakit değişmemeli
        assert p_sonuc["nakit"] == 50000.0
        # Trade history'ye hayali satış yazılmamalı
        assert len(p_sonuc["trade_history"]) == 0
