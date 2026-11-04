# -*- coding: utf-8 -*-
"""
tests/unit/test_p1_trade_execution.py
====================================
P1 İşlem Gerçekleşmesi Testleri:
1. Stop altı açılışta (gap down) stop fiyatının garanti edilmemesi, açılış + kayma simülasyonu
2. Normal stop (open > stop, low <= stop) seviyesinde kayma simülasyonu
3. Intrabar trailing lookahead düzeltmesi (mevcut mumun high değerinin aynı mumun geçmiş low değerine uygulanmaması)
4. Aynı mumda STOP ve TP1 görülmesi: belirsizliğin kaydedilmesi ve muhafazakâr STOP önceliği
5. TP1 tetik seviyesi, gerçekleşme fiyatı ve gerçek hesaplanan getiri alanlarının ayrımı ve Telegram mesajı
6. Tek lot (1 lot -> 0 kalan lot ve tam kapanış), tek sayıda lot (3 lot -> 1 sat, 2 kal), tekrar TP1 yapılmaması
7. Merkezi komisyon ve kayma hesabı ile likidite teyidi
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

import portfoy_yonetici as py


@pytest.fixture(autouse=True)
def _isolate_audit_file(tmp_path, monkeypatch):
    monkeypatch.setattr(py, "PORTFOY_AUDIT_FILE", tmp_path / "portfolio_actions.jsonl")


class TestP1TradeExecution:
    def test_stop_alti_acilis_gap_down_kayma(self):
        """Açılış stop seviyesinin altındaysa, satış stop fiyatından değil açılış - kayma ile simüle edilir."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
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

        # Stop seviyesi 95.0. Mum 92.0 açılıyor (gap down), low 91.0, close 92.5
        gap_bar = {
            "open": 92.0, "high": 93.0, "low": 91.0, "close": 92.5, "volume": 1000,
            "bar_time": "2026-10-06T10:00:00"
        }
        with patch.object(py, "saatlik_bar", return_value=gap_bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert "THYAO" not in p_sonuc["pozisyonlar"]
        trade = p_sonuc["trade_history"][0]
        assert trade["neden"] == "STOP"
        # Çıkış fiyatı 95.0 DEĞİL, 92.0 * (1 - 0.001) = 91.908 olmalıdır!
        beklenen_cikis = round(92.0 * (1.0 - py.VARSAYILAN_KAYMA_ORANI), 4)
        assert trade["cikis_fiyat"] == beklenen_cikis
        assert trade["cikis_fiyat"] < 95.0

    def test_normal_stop_kayma(self):
        """Açılış stop seviyesinin üzerindeyken low stop seviyesini delerse stop - kayma ile simüle edilir."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
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

        # Open 98.0, Stop seviyesi 95.0, low 94.0
        normal_stop_bar = {
            "open": 98.0, "high": 98.5, "low": 94.0, "close": 94.5, "volume": 1000,
            "bar_time": "2026-10-06T10:00:00"
        }
        with patch.object(py, "saatlik_bar", return_value=normal_stop_bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        trade = p_sonuc["trade_history"][0]
        beklenen_cikis = round(95.0 * (1.0 - py.VARSAYILAN_KAYMA_ORANI), 4)
        assert trade["cikis_fiyat"] == beklenen_cikis

    def test_intrabar_trailing_onceki_tepeyi_kullanir(self):
        """Mevcut mumun yüksek değeri (high) aynı mumun geçmiş low değerine trailing olarak uygulanamaz."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 105.0,  # Önceki zirve 105.0
                    "lotlar": 50,
                    "tp1_yapildi": True,
                    "giris_t": "01.10.2026 10:00",
                    "max_gun_date": "15.10.2026",
                }
            },
            "trade_history": [],
        }

        # Bu mumda: open 104.0, low 101.0, high 108.0, close 107.0
        # Önceki zirve 105.0'a göre: (101 - 105) / 105 = -%3.8 (Trailing eşiği -%5'e ulaşmadı, tetiklenmemeli!)
        # Hatalı eski kodda: tepe_f önce 108 yapılırsa: (101 - 108) / 108 = -%6.48 -> YANLIŞ trailing satışı yapılırdı!
        bar = {
            "open": 104.0, "high": 108.0, "low": 101.0, "close": 107.0, "volume": 1000,
            "bar_time": "2026-10-06T10:00:00"
        }
        with patch.object(py, "saatlik_bar", return_value=bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        # Pozisyon trailing ile SATILMAMALI, tepe_f ise mum kapandıktan sonra 108'e güncellenmeli
        assert "THYAO" in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 0
        assert p_sonuc["pozisyonlar"]["THYAO"]["tepe_f"] == 108.0

    def test_ayni_mumda_stop_ve_tp_belirsizlik_muhafazakar_stop(self):
        """Aynı barda hem STOP (-%5) hem TP1 (+%8) görülürse belirsizlik kaydedilir ve muhafazakâr STOP uygulanır."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
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

        # Hem low 94 (-%6, stop), hem high 110 (+%10, TP1)
        wild_bar = {
            "open": 98.0, "high": 110.0, "low": 94.0, "close": 95.0, "volume": 1000,
            "bar_time": "2026-10-06T10:00:00"
        }
        with patch.object(py, "saatlik_bar", return_value=wild_bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert "THYAO" not in p_sonuc["pozisyonlar"]
        trade = p_sonuc["trade_history"][0]
        assert trade["neden"] == "STOP"
        assert trade["ambiguity"] == "both_stop_and_tp_in_bar"
        assert any("Belirsiz Mum" in m for m in mesajlar)

    def test_tp1_alanlar_ve_telegram_gerceklesen_getiri(self):
        """TP1 tetik seviyesi, simüle edilen satış fiyatı ve gerçek getiri ayrı alanlar olarak kaydedilir."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
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

        # TP1 tetik seviyesi: 100 * (1 + 0.08) = 108.0
        # Mum: open 102.0, high 109.0, low 101.0, close 107.0
        tp_bar = {
            "open": 102.0, "high": 109.0, "low": 101.0, "close": 107.0, "volume": 1000,
            "bar_time": "2026-10-06T10:00:00"
        }
        with patch.object(py, "saatlik_bar", return_value=tp_bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        # 50 lot satıldı, 50 lot kaldı
        assert p_sonuc["pozisyonlar"]["THYAO"]["lotlar"] == 50
        assert p_sonuc["pozisyonlar"]["THYAO"]["tp1_yapildi"] is True
        trade = p_sonuc["trade_history"][0]
        assert trade["neden"] == "TP1"
        assert trade["tp1_trigger_price"] == 108.0
        # Gerçekleşen çıkış: 108 * (1 - 0.001) = 107.892
        assert trade["cikis_fiyat"] == round(108.0 * (1.0 - py.VARSAYILAN_KAYMA_ORANI), 4)
        assert trade["lotlar"] == 50
        # Unknown legacy buy fee is not invented. Selling fee is deducted.
        # (50*107.892 - 2.6973 - 5000)/5000*100 = 7.838054%
        assert trade["net_pnl_pct"] == pytest.approx(7.8381, abs=.0001)
        assert trade["maliyet_bilgisi_tam"] is False
        assert any("net +7.84%" in m for m in mesajlar)
        assert any("Tetik: 108.00" in m for m in mesajlar)

    def test_tek_lot_tp1_tam_kapanis(self):
        """1 lot olan pozisyonda TP1 yapıldığında, kalan lot 0 olacağından pozisyon tamamen kapatılmalıdır."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "UFUK": {
                    "giris_f": 1000.0,
                    "tepe_f": 1000.0,
                    "lotlar": 1,  # Tek lot!
                    "giris_t": "01.10.2026 10:00",
                    "max_gun_date": "15.10.2026",
                }
            },
            "trade_history": [],
        }

        tp_bar = {
            "open": 1020.0, "high": 1090.0, "low": 1010.0, "close": 1085.0, "volume": 1000,
            "bar_time": "2026-10-06T10:00:00"
        }
        with patch.object(py, "saatlik_bar", return_value=tp_bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        # 0 lot kalan pozisyon açık tutulmamalı, tamamen kapanmalı!
        assert "UFUK" not in p_sonuc["pozisyonlar"]
        assert len(p_sonuc["trade_history"]) == 1
        assert p_sonuc["trade_history"][0]["lotlar"] == 1
        assert p_sonuc["trade_history"][0]["neden"] == "TP1"

    def test_tek_sayida_lot_ve_tekrar_tp1_yapilmama(self):
        """3 lot olan pozisyon: 1 lot satılır, 2 lot kalır. Sonraki mumlarda tekrar TP1 yapılmaz."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
        portfoy = {
            "nakit": 50000.0,
            "pozisyonlar": {
                "THYAO": {
                    "giris_f": 100.0,
                    "tepe_f": 100.0,
                    "lotlar": 3,  # 3 lot
                    "giris_t": "01.10.2026 10:00",
                    "max_gun_date": "15.10.2026",
                }
            },
            "trade_history": [],
        }

        # İlk TP1 mumu (10:00)
        bar_1 = {
            "open": 102.0, "high": 109.0, "low": 101.0, "close": 108.0, "volume": 1000,
            "bar_time": "2026-10-06T10:00:00"
        }
        with patch.object(py, "saatlik_bar", return_value=bar_1):
            p_sonuc, _ = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        assert p_sonuc["pozisyonlar"]["THYAO"]["lotlar"] == 2
        assert p_sonuc["pozisyonlar"]["THYAO"]["tp1_yapildi"] is True
        assert len(p_sonuc["trade_history"]) == 1

        # İkinci saat mumu (11:00) — fiyat daha da yükseliyor (115)
        bar_2 = {
            "open": 109.0, "high": 115.0, "low": 108.0, "close": 114.0, "volume": 1000,
            "bar_time": "2026-10-06T11:00:00"
        }
        with patch.object(py, "saatlik_bar", return_value=bar_2):
            p_sonuc, _ = py.pozisyon_guncelle_saatlik(p_sonuc, "NORMAL", now=ref_dt + timedelta(hours=1))

        # Tekrar TP1 tetiklenmemeli, lot sayısı 2 olarak kalmalı
        assert p_sonuc["pozisyonlar"]["THYAO"]["lotlar"] == 2
        assert len(p_sonuc["trade_history"]) == 1

    def test_hacim_yokken_likidite_teyitsiz_isaretlenir(self):
        """Mumda hacim bilgisi yoksa veya sıfırsa likidite_teyitli=False olarak kaydedilir."""
        ref_dt = datetime(2026, 10, 6, 11, 0)
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

        # Hacim = 0 olan stop barı
        zero_vol_bar = {
            "open": 98.0, "high": 98.5, "low": 94.0, "close": 94.5, "volume": 0,
            "bar_time": "2026-10-06T10:00:00"
        }
        with patch.object(py, "saatlik_bar", return_value=zero_vol_bar):
            p_sonuc, mesajlar = py.pozisyon_guncelle_saatlik(portfoy, "NORMAL", now=ref_dt)

        trade = p_sonuc["trade_history"][0]
        assert trade["likidite_teyitli"] is False
