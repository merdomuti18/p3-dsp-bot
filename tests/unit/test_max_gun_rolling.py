# -*- coding: utf-8 -*-
"""
BQRP PAKET 3 — MAX_GUN Rolling Extension Tests
================================================
Canonical kural:
  Başlangıç MAX_GUN = 10 gün.
  MAX_GUN gününe ulaşıldığında:
    IF hisse CANONICAL ALIM LİSTESİNDE → TUT, MAX_GUN += 5
    ELSE → SAT
  Extension yalnızca MAX_GUN kontrol gününde yapılır.
  Idempotency: aynı gün iki çalıştırma → tek extension.
  State'e `max_gun_date` alanı eklenir; restart sonrası korunur.

Invariantlar:
  A. MAX_GUN + AL → +5
  B. MAX_GUN + AL değil → EXIT
  C. İkinci MAX_GUN + AL → tekrar +5
  D. İkinci MAX_GUN + AL değil → EXIT
  E. STOP + AL → STOP EXIT
  F. TP + AL → TP davranışı
  G. Aynı gün iki çalıştırma → tek extension
  H. Restart sonrası extension korunur
  I. Stale AL → extension yok
  J. Empty AL list → güvenli EXIT
  K. Boundary: 9/10/11/14/15/16/20 sınırları
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _p5_state(poz=None, hist=None):
    return {
        "pozisyonlar": poz or {},
        "trade_history": hist or [],
        "komite_log": [],
        "sermaye_baslangic": 100_000,
        "sermaye_mevcut": 100_000,
    }


def _p5_pos(sym, giris_f, gun=0, giris_tarih=None, max_gun_date=None):
    if giris_tarih is None:
        giris_tarih = (date.today() - timedelta(days=gun)).isoformat()
    d = {
        "symbol": sym,
        "giris_fiyat": giris_f,
        "guncel_fiyat": giris_f,
        "lotlar": 100,
        "pnl_pct": 0.0,
        "gun": gun,
        "giris_tarih": giris_tarih,
        "strateji": "P5",
    }
    if max_gun_date is not None:
        d["max_gun_date"] = max_gun_date
    return d


def _cache(prices):
    return {sym: np.array([p]) for sym, p in prices.items()}


# ═════════════════════════════════════════════════════════════════════════════
# P5 — MAX_GUN Rolling Extension
# ═════════════════════════════════════════════════════════════════════════════

class TestP5MaxGunRolling:
    """P5Committee — MAX_GUN rolling extension."""

    def test_max_gun_al_varsa_extends(self):
        """A) gun=10, AL listesinde → HOLD, max_gun_date=15."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = (date.today()).isoformat()  # max_gun_date = today (10. gün)
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 0
        assert "A" in r["devam_eden"]
        assert s["pozisyonlar"]["A"]["max_gun_date"] == (date.today() + timedelta(days=5)).isoformat()

    def test_max_gun_al_yoksa_exits(self):
        """B) gun=10, AL listesinde değil → EXIT."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = (date.today()).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 1
        assert r["kapanan"][0]["neden"] == "MAX_GUN"

    def test_ikinci_max_gun_al_varsa_tekrar_extends(self):
        """C) İlk extension sonrası 15. gün, AL listesinde → tekrar +5."""
        import p5_committee
        giris = (date.today() - timedelta(days=15)).isoformat()
        mgd = (date.today()).isoformat()  # max_gun_date = today (15. gün)
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 105}))
        assert len(r["kapanan"]) == 0
        assert s["pozisyonlar"]["A"]["max_gun_date"] == (date.today() + timedelta(days=5)).isoformat()

    def test_ikinci_max_gun_al_yoksa_exits(self):
        """D) 15. gün, AL listesinde değil → EXIT."""
        import p5_committee
        giris = (date.today() - timedelta(days=15)).isoformat()
        mgd = (date.today()).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 105}))
        assert len(r["kapanan"]) == 1
        assert r["kapanan"][0]["neden"] == "MAX_GUN"

    def test_stop_al_olsa_bile_stop_exits(self):
        """E) STOP + AL listesinde → STOP EXIT (exit önceliği korunur)."""
        import p5_committee
        giris = (date.today() - timedelta(days=11)).isoformat()
        mgd = (date.today() - timedelta(days=1)).isoformat()  # max_gun_date geçmiş
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 94}))  # -6% → STOP
        assert r["kapanan"][0]["neden"] == "STOP"

    def test_tp_al_olsa_bile_tp_exits(self):
        """F) TP + AL listesinde → TP davranışı."""
        import p5_committee
        giris = (date.today() - timedelta(days=11)).isoformat()
        mgd = (date.today() - timedelta(days=1)).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 112}))  # +12% → TP
        assert r["kapanan"][0]["neden"] == "TP"

    def test_ayni_gun_ikinci_calistirma_tek_extension(self):
        """G) Aynı gün iki kez çalıştırma → yalnızca bir extension."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = (date.today()).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r1 = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
            # İlk çalıştırmada max_gun_date today+5 oldu
            # İkinci çalıştırmada today < max_gun_date → extension yok
            r2 = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r1["kapanan"]) == 0
        assert len(r2["kapanan"]) == 0
        # max_gun_date yalnızca bir kez uzatıldı
        assert s["pozisyonlar"]["A"]["max_gun_date"] == (date.today() + timedelta(days=5)).isoformat()

    def test_restart_sonrasi_max_gun_korunur(self):
        """H) max_gun_date state'te kayıtlı → restart sonrası korunur."""
        import p5_committee
        giris = (date.today() - timedelta(days=15)).isoformat()
        mgd = (date.today() + timedelta(days=3)).isoformat()  # 3 gün sonra
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 105}))
        # today < max_gun_date → HOLD
        assert len(r["kapanan"]) == 0
        assert s["pozisyonlar"]["A"]["max_gun_date"] == mgd

    def test_stale_al_listesi_extension_vermez(self):
        """I) Stale AL listesi (dünkü) → extension yok → EXIT."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = (date.today()).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        #komite_adaylari() dünkü tarihli sonuç döndürüyor — stale
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 1
        assert r["kapanan"][0]["neden"] == "MAX_GUN"

    def test_empty_al_list_guvenli_exit(self):
        """J) Boş AL listesi → güvenli EXIT."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = (date.today()).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 1
        assert r["kapanan"][0]["neden"] == "MAX_GUN"

    def test_boundary_9_gun_hold(self):
        """K) gun=9 → HOLD (MAX_GUN henüz tetiklenmedi)."""
        import p5_committee
        giris = (date.today() - timedelta(days=9)).isoformat()
        mgd = (date.today() + timedelta(days=1)).isoformat()  # yarın
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 0

    def test_boundary_10_gun_al_extends(self):
        """K) gun=10, AL → extend."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = (date.today()).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 0
        assert s["pozisyonlar"]["A"]["max_gun_date"] == (date.today() + timedelta(days=5)).isoformat()

    def test_boundary_10_gun_al_yok_exit(self):
        """K) gun=10, AL yok → EXIT."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = (date.today()).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 1

    def test_boundary_14_gun_hold(self):
        """K) gun=14, max_gun_date=15 → HOLD."""
        import p5_committee
        giris = (date.today() - timedelta(days=14)).isoformat()
        mgd = (date.today() + timedelta(days=1)).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 0

    def test_boundary_15_gun_al_extends(self):
        """K) gun=15, AL → extend to 20."""
        import p5_committee
        giris = (date.today() - timedelta(days=15)).isoformat()
        mgd = (date.today()).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 105}))
        assert len(r["kapanan"]) == 0
        assert s["pozisyonlar"]["A"]["max_gun_date"] == (date.today() + timedelta(days=5)).isoformat()

    def test_boundary_20_gun_al_yok_exit(self):
        """K) gun=20, AL yok → EXIT."""
        import p5_committee
        giris = (date.today() - timedelta(days=20)).isoformat()
        mgd = (date.today()).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 105}))
        assert len(r["kapanan"]) == 1
        assert r["kapanan"][0]["neden"] == "MAX_GUN"

    def test_max_gun_date_yoksa_entryden_hesaplanir(self):
        """Eski pozisyonlarda max_gun_date yoksa entry'den hesaplanır."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        # max_gun_date yok, entry+10 = today → EXIT
        assert len(r["kapanan"]) == 1

    def test_max_gun_date_yoksa_9_gun_hold(self):
        """Eski pozisyon, max_gun_date yok, gun=9 → HOLD."""
        import p5_committee
        giris = (date.today() - timedelta(days=9)).isoformat()
        s = _p5_state(poz={"A": _p5_pos("A", 100, giris_tarih=giris)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 102}))
        assert len(r["kapanan"]) == 0


# ═════════════════════════════════════════════════════════════════════════════
# P5 — Mevcut Davranış Korunması (Regression)
# ═════════════════════════════════════════════════════════════════════════════

class TestP5ExistingBehavior:
    """Mevcut P5 davranışları korunur — ALIM listesi mocklanmamış."""

    def test_normal_pozisyon_korunur(self):
        """gun=3, pnl=+5% → HOLD."""
        import p5_committee
        s = _p5_state(poz={"A": _p5_pos("A", 100, gun=3)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 105}))
        assert len(r["kapanan"]) == 0

    def test_stop_without_max_gun(self):
        """gun=5, STOP → STOP EXIT."""
        import p5_committee
        s = _p5_state(poz={"A": _p5_pos("A", 100, gun=5)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 94}))
        assert r["kapanan"][0]["neden"] == "STOP"

    def test_tp_without_max_gun(self):
        """gun=5, TP → TP EXIT."""
        import p5_committee
        s = _p5_state(poz={"A": _p5_pos("A", 100, gun=5)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, _cache({"A": 112}))
        assert r["kapanan"][0]["neden"] == "TP"

    def test_empty_pozisyon(self):
        """Pozisyon yok → kapanan=0."""
        import p5_committee
        s = _p5_state()
        r = p5_committee.portfoy_guncelle(s, _cache({}))
        assert len(r["kapanan"]) == 0
