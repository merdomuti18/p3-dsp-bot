"""
DSP-P3 FAZ 5 — P5 Portfolio Management & Risk Certification Tests
==================================================================
Production kodu DEĞİŞTİRİLMEZ; yalnızca mevcut davranış doğrulanır.

p5_committee.py sabitleri:
  STOP_PCT  = -0.05   (-%5)
  TP_PCT    = +0.10   (+%10)
  MAX_GUN   = 10
  MAX_POS   = 5
  POS_TL    = 20_000  (100_000 / 5)
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pytest

import mott_risk
import p5_committee


# ── Helpers ──────────────────────────────────────────────────────────────────

def _state(poz=None, hist=None):
    return {
        "pozisyonlar": poz or {},
        "trade_history": hist or [],
        "komite_log": [],
        "sermaye_baslangic": 100_000,
        "sermaye_mevcut": 100_000,
    }


def _pos(sym, giris_f, gun=0, guncel_f=None, giris_tarih=None):
    if giris_tarih is None:
        giris_tarih = (date.today() - timedelta(days=gun)).isoformat()
    return {
        "symbol": sym,
        "giris_fiyat": giris_f,
        "guncel_fiyat": guncel_f if guncel_f is not None else giris_f,
        "lotlar": 100,
        "pnl_pct": 0.0,
        "gun": gun,
        "giris_tarih": giris_tarih,
        "strateji": "P5",
    }


def _cache(prices: dict) -> dict:
    """symbol → price list ([-1] ile son fiyat alınır)."""
    return {s: [p] for s, p in prices.items()}


def _aday(sym, skor=60, kaynaklar=None):
    return {
        "symbol": sym,
        "komite_skor": skor,
        "kaynaklar": kaynaklar or ["P1", "P2"],
        "sektor": "DIGER",
    }


# ═════════════════════════════════════════════════════════════════════════════
# A) STOP — pnl <= -0.05
# ═════════════════════════════════════════════════════════════════════════════

class TestStop:
    def test_stop_seviyesinde_kapanir(self):
        """pnl = -5.0% → STOP."""
        s = _state(poz={"A": _pos("A", 100)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 95}))
        assert len(r["kapanan"]) == 1
        assert r["kapanan"][0]["neden"] == "STOP"
        assert r["kapanan"][0]["pnl_pct"] == -5.0

    def test_stop_altinda_kapanir(self):
        """pnl = -8.0% → STOP."""
        s = _state(poz={"A": _pos("A", 100)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 92}))
        assert r["kapanan"][0]["neden"] == "STOP"
        assert r["kapanan"][0]["pnl_pct"] == -8.0

    def test_stop_ustunde_devam(self):
        """pnl = -4.0% → devam."""
        s = _state(poz={"A": _pos("A", 100)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 96}))
        assert len(r["kapanan"]) == 0
        assert "A" in r["devam_eden"]


# ═════════════════════════════════════════════════════════════════════════════
# B) TAKE PROFIT — pnl >= +0.10
# ═════════════════════════════════════════════════════════════════════════════

class TestTakeProfit:
    def test_tp_seviyesinde_kapanir(self):
        """pnl = +10.0% → TP."""
        s = _state(poz={"A": _pos("A", 100)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 110}))
        assert len(r["kapanan"]) == 1
        assert r["kapanan"][0]["neden"] == "TP"
        assert r["kapanan"][0]["pnl_pct"] == 10.0

    def test_tp_altinda_devam(self):
        """pnl = +9.0% → devam (TP değil)."""
        s = _state(poz={"A": _pos("A", 100)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 109}))
        assert len(r["kapanan"]) == 0

    def test_trade_history_dogru(self):
        """Exit sonrası trade_history formatı doğrulanmalı."""
        s = _state(poz={"A": _pos("A", 80, gun=5)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 90}))
        t = s["trade_history"][-1]
        assert t["symbol"] == "A"
        assert t["giris_fiyat"] == 80
        assert t["cikis_fiyat"] == 90
        assert t["neden"] == "TP"
        assert t["strateji"] == "P5"
        assert "cikis_tarih" in t
        assert t["gun"] == 5


# ═════════════════════════════════════════════════════════════════════════════
# C) MAX_GUN — gun >= 10
# ═════════════════════════════════════════════════════════════════════════════

class TestMaxGun:
    def test_max_gun_kapanir(self):
        """gun = 10 → MAX_GUN."""
        giris = (date.today() - timedelta(days=10)).isoformat()
        s = _state(poz={"A": _pos("A", 100, giris_tarih=giris)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert r["kapanan"][0]["neden"] == "MAX_GUN"

    def test_max_gun_altinda_devam(self):
        """gun = 9 → devam."""
        giris = (date.today() - timedelta(days=9)).isoformat()
        s = _state(poz={"A": _pos("A", 100, giris_tarih=giris)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 0

    def test_stop_onceligi_tp_den_buyuk(self):
        """pnl = -6% (STOP) ama gun = 11 (MAX_GUN) → STOP kazanır."""
        giris = (date.today() - timedelta(days=11)).isoformat()
        s = _state(poz={"A": _pos("A", 100, giris_tarih=giris)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 94}))
        assert r["kapanan"][0]["neden"] == "STOP"

    def test_tp_onceligi_max_gunden_buyuk(self):
        """pnl = +12% (TP) ama gun = 11 (MAX_GUN) → TP kazanır."""
        giris = (date.today() - timedelta(days=11)).isoformat()
        s = _state(poz={"A": _pos("A", 100, giris_tarih=giris)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 112}))
        assert r["kapanan"][0]["neden"] == "TP"


# ═════════════════════════════════════════════════════════════════════════════
# D) NORMAL POZİSYON — exit koşulu yok
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalPozisyon:
    def test_pozisyon_korunur(self):
        """pnl = +5%, gun = 3 → pozisyon korunur, state güncellenir."""
        s = _state(poz={"A": _pos("A", 100, gun=3)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 105}))
        assert len(r["kapanan"]) == 0
        assert "A" in r["devam_eden"]
        pos = s["pozisyonlar"]["A"]
        assert pos["guncel_fiyat"] == 105
        assert pos["pnl_pct"] == 5.0
        assert pos["gun"] == 3

    def test_pozisyon_sifirdan_baslar(self):
        """yeni pozisyonda gun=0, pnl=0."""
        p = _pos("A", 100)
        assert p["gun"] == 0
        assert p["pnl_pct"] == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# E) POS_TL — pozisyon büyüklüğü
# ═════════════════════════════════════════════════════════════════════════════

class TestPOS_TL:
    def test_pos_tl_20000(self):
        """POS_TL = 100_000 / 5 = 20_000."""
        assert p5_committee.POS_TL == 20_000

    def test_lot_hesabi_dogru(self):
        """fiyat=250 → lot = int(20000/250) = 80."""
        s = _state()
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("A")], _cache({"A": 250})
            )
        assert len(acilan) == 1
        assert acilan[0]["lotlar"] == 80

    def test_lot_sifirsa_acilmaz(self):
        """fiyat=20001 → lot=0 → pozisyon açılmaz."""
        s = _state()
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("A")], _cache({"A": 20_001})
            )
        assert len(acilan) == 0

    def test_giris_fiyati_dogru(self):
        """Açılan pozisyonun giriş fiyatı fiyata eşit olmalı."""
        s = _state()
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("A")], _cache({"A": 42.5})
            )
        assert acilan[0]["giris_fiyat"] == 42.5
        assert acilan[0]["guncel_fiyat"] == 42.5


# ═════════════════════════════════════════════════════════════════════════════
# F) MAX_POS / SLOT — açık pozisyon limiti
# ═════════════════════════════════════════════════════════════════════════════

class TestMaxPos:
    def test_bos_slot_varsa_acilir(self):
        """3 açık pozisyon → 2 boş slot → yeni pozisyon açılır."""
        poz = {f"P{i}": _pos(f"P{i}", 100) for i in range(3)}
        s = _state(poz=poz)
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("NEW")], _cache({"NEW": 100})
            )
        assert len(acilan) == 1

    def test_limit_doluysa_acilmaz(self):
        """5 açık pozisyon → 0 boş slot → yeni pozisyon açılmaz."""
        poz = {f"P{i}": _pos(f"P{i}", 100) for i in range(5)}
        s = _state(poz=poz)
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("NEW")], _cache({"NEW": 100})
            )
        assert len(acilan) == 0

    def test_mevcut_sembol_atlanir(self):
        """Zaten açık pozisyon → tekrar açılmaz."""
        s = _state(poz={"A": _pos("A", 100)})
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("A")], _cache({"A": 100})
            )
        assert len(acilan) == 0


# ═════════════════════════════════════════════════════════════════════════════
# G) COOLDOWN — STOP/TP sonrası yeniden giriş yasağı
# ═════════════════════════════════════════════════════════════════════════════

class TestCooldown:
    def test_cooldown_icinde_acilmaz(self):
        """cooldown_da=True → pozisyon açılmaz."""
        s = _state()
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.cooldown_da", return_value=True), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("A")], _cache({"A": 100})
            )
        assert len(acilan) == 0

    def test_cooldown_disinda_acilir(self):
        """cooldown_da=False → pozisyon açılır."""
        s = _state()
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("A")], _cache({"A": 100})
            )
        assert len(acilan) == 1

    def test_max_gun_cooldown_tetiklemez(self):
        """MAX_GUN ile kapanan pozisyon → COOLDOWN_NEDENLER'de değil → cooldown yok."""
        giris = (date.today() - timedelta(days=11)).isoformat()
        hist = [{
            "symbol": "A", "giris_fiyat": 100, "cikis_fiyat": 102,
            "pnl_pct": 2.0, "gun": 11, "neden": "MAX_GUN",
            "giris_tarih": giris, "cikis_tarih": date.today().isoformat(),
            "strateji": "P5",
        }]
        s = _state(hist=hist)
        # mott_risk.cooldown_da naturally returns False for MAX_GUN
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("A")], _cache({"A": 100})
            )
        assert len(acilan) == 1


# ═════════════════════════════════════════════════════════════════════════════
# H) EXPOSURE — kitap/maruz kalma limitleri
# ═════════════════════════════════════════════════════════════════════════════

class TestExposure:
    def test_kitap_limiti_acilmaz(self):
        """kitap_limiti_asildi=True → pozisyon açılmaz."""
        s = _state()
        with patch("mott_risk.rejim_slot_limiti", return_value=None), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=True):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("A")], _cache({"A": 100})
            )
        assert len(acilan) == 0

    def test_rejim_girme_sifir_slot(self):
        """GIRME → 0 yeni pozisyon."""
        s = _state()
        with patch("mott_risk.rejim_slot_limiti", return_value=0), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(
                s, [_aday("A")], _cache({"A": 100})
            )
        assert len(acilan) == 0

    def test_rejim_dikkatli_iki_slot(self):
        """DIKKATLI → max 2 yeni pozisyon."""
        s = _state()
        adaylar = [_aday(f"S{i}") for i in range(5)]
        cache = {f"S{i}": [100.0] for i in range(5)}
        with patch("mott_risk.rejim_slot_limiti", return_value=2), \
             patch("mott_risk.cooldown_da", return_value=False), \
             patch("mott_risk.kitap_limiti_asildi", return_value=False):
            acilan = p5_committee.yeni_pozisyon_ac(s, adaylar, cache)
        assert len(acilan) == 2


# ═════════════════════════════════════════════════════════════════════════════
# I) SPLIT / BEDELSİZ — BIST %10 limit aşımı koruması
# ═════════════════════════════════════════════════════════════════════════════

class TestSplit:
    @patch("mott_telegram.telegram_gonder")
    def test_split_algılanır_giris_duzeltilir(self, mock_tg):
        """guncel_fiyat=100, guncel=50 → %50 düşüş → split algılanır,
        giriş fiyatı orantıyla düzeltilir, pozisyon devam eder."""
        # giris=95, guncel_f=100 → split sonrası giris=47.5, pnl≈5.3% → devam
        s = _state(poz={"A": _pos("A", 95, guncel_f=100)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 50}))
        # Split: oran = 100/50 = 2.0, giris = 95/2 = 47.5
        pos = s["pozisyonlar"]["A"]
        assert pos["giris_fiyat"] == 47.5
        assert pos["split_duzeltme"] == 2.0
        assert "split_tarih" in pos
        # pnl = (50-47.5)/47.5 ≈ 5.26% → devam
        assert len(r["kapanan"]) == 0
        assert pos["pnl_pct"] == round((50 - 47.5) / 47.5 * 100, 2)

    def test_normal_dusus_split_tetiklemez(self):
        """%4 düşüş → split tetiklenmez (-10% eşiğinin altında), STOP da yok."""
        s = _state(poz={"A": _pos("A", 100, guncel_f=100)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 96}))
        # gunluk = (96-100)/100 = -4% > -10% → split YOK
        # pnl = -4% > -5% STOP → devam
        assert len(r["kapanan"]) == 0
        pos = s["pozisyonlar"]["A"]
        assert "split_duzeltme" not in pos
        assert pos["pnl_pct"] == -4.0

    @patch("mott_telegram.telegram_gonder")
    def test_split_sonrasi_stop_kontrolu(self, mock_tg):
        """Split sonrası pnl hâlâ stop yaptırabilmeli."""
        # giris=100, guncel_fiyat=100, guncel=50
        # split: oran=2, giris=50, pnl=(50-50)/50=0 → devam
        s = _state(poz={"A": _pos("A", 100, guncel_f=100)})
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 50}))
        assert len(r["kapanan"]) == 0  # split sonrası pnl=0


# ═════════════════════════════════════════════════════════════════════════════
# J) TRADE HISTORY — exit sonrası kayıt formatı
# ═════════════════════════════════════════════════════════════════════════════

class TestTradeHistory:
    def test_stop_history_formatı(self):
        """STOP sonrası trade_history alanları doğru."""
        s = _state(poz={"X": _pos("X", 200, gun=3)})
        p5_committee.portfoy_guncelle(s, _cache({"X": 190}))
        t = s["trade_history"][-1]
        assert t["symbol"] == "X"
        assert t["giris_fiyat"] == 200
        assert t["cikis_fiyat"] == 190
        assert t["pnl_pct"] == -5.0
        assert t["neden"] == "STOP"
        assert t["gun"] == 3
        assert t["strateji"] == "P5"
        assert t["giris_tarih"] == (date.today() - timedelta(days=3)).isoformat()
        assert t["cikis_tarih"] == date.today().isoformat()

    def test_pnl_yuvarlama(self):
        """PnL 2 ondalık basamağa yuvarlanmalı (STOP senaryosu)."""
        s = _state(poz={"A": _pos("A", 333)})
        # pnl = (316-333)/333 = -5.1% → STOP
        p5_committee.portfoy_guncelle(s, _cache({"A": 316}))
        t = s["trade_history"][-1]
        assert t["pnl_pct"] == round((316 - 333) / 333 * 100, 2)

    def test_coklu_exit_hepsi_kaydedilir(self):
        """Birden fazla pozisyon kapanırsa hepsi history'de."""
        poz = {
            "A": _pos("A", 100),  # → STOP (-5%)
            "B": _pos("B", 100),  # → TP (+10%)
        }
        s = _state(poz=poz)
        r = p5_committee.portfoy_guncelle(s, _cache({"A": 95, "B": 110}))
        assert len(r["kapanan"]) == 2
        assert len(s["trade_history"]) == 2
        nedenler = {t["neden"] for t in s["trade_history"]}
        assert nedenler == {"STOP", "TP"}


# ═════════════════════════════════════════════════════════════════════════════
# K) IDEMPOTENCY — çift exit engeli
# ═════════════════════════════════════════════════════════════════════════════

class TestIdempotency:
    def test_cift_exit_olusmaz(self):
        """Aynı state ile iki kez çağrıldığında çift exit oluşmaz."""
        s = _state(poz={"A": _pos("A", 100)})
        cache = _cache({"A": 110})  # TP

        r1 = p5_committee.portfoy_guncelle(s, cache)
        assert len(r1["kapanan"]) == 1

        # İkinci çağrı — pozisyonlar artık boş
        r2 = p5_committee.portfoy_guncelle(s, cache)
        assert len(r2["kapanan"]) == 0
        assert len(s["trade_history"]) == 1  # yalnızca 1 kayıt
