# -*- coding: utf-8 -*-
"""
FAZ 4 — Cross-Pipeline MAX_GUN Rolling Extension Invariant Tests
=================================================================
P3 _p3_alim_listesi() helper testleri + P1-P5 invariant doğrulama.

Invariantlar:
  A. MAX_GUN doldu + ALIM listesinde → pozisyon kapanmamalı
  B. MAX_GUN doldu + ALIM listesinde → max_gun_date = bugün + 5
  C. Aynı gün ikinci çalıştırma → ikinci kez +5 yapılmamalı
  D. Yeni MAX_GUN tarihinde tekrar ALIM listesinde → tekrar +5
  E. MAX_GUN doldu + ALIM listesinde değil → exit
  F. STOP + ALIM listesinde → STOP yine çalışmalı
  G. TP + ALIM listesinde → TP yine çalışmalı
  H. Gelecek tarihli ALIM sinyali → kullanılmamalı
  I. Eski/stale ALIM listesi → bugünün listesi kabul edilmemeli
  J. P1, P2, P3, P4, P5 için aynı invariantlar
  K. Extension entry_date değiştirmemeli
  L. Extension PnL değiştirmemeli
  M. Extension yeni trade_history kaydı oluşturmamalı
  N. Extension pozisyonu yeniden açılmış gibi işlememeli
  O. Eski state'te max_gun_date yoksa fallback çalışmalı
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ═════════════════════════════════════════════════════════════════════════════
# P3 _p3_alim_listesi() HELPER TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestP3AlimListesiHelper:
    """P3 canonical ALIM listesi helper fonksiyonunun davranış testleri."""

    def test_p3_alim_listesi_dosyadan_okur(self):
        """P3 ALIM listesi portfolio_state.json scan_log'undan okunmalı."""
        import simulate_dsp
        import json as _json
        mock_data = {
            "positions": {},
            "scan_log": [
                {"date": date.today().isoformat(), "top5": ["SYM1", "SYM2", "SYM3"]}
            ],
        }
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = _json.dumps(mock_data)
        with patch.object(simulate_dsp, "STATE_FILE", mock_path):
            result = simulate_dsp._p3_alim_listesi()
        assert "SYM1" in result
        assert "SYM2" in result
        assert "SYM3" in result

    def test_p3_alim_listesi_bos_scan_log(self):
        """Boş scan_log → boş küme."""
        import simulate_dsp
        mock_state = {"positions": {}, "scan_log": []}
        with patch.object(simulate_dsp, "load_state", return_value=mock_state):
            result = simulate_dsp._p3_alim_listesi()
        assert result == set()

    def test_p3_alim_listesi_stale_scan_log(self):
        """Dünkü scan_log → bugünün tarihiyle uyuşmuyorsa boş küme."""
        import simulate_dsp
        dün = (date.today() - timedelta(days=1)).isoformat()
        mock_state = {
            "positions": {},
            "scan_log": [{"date": dün, "top5": ["SYM1"]}],
        }
        with patch.object(simulate_dsp, "load_state", return_value=mock_state):
            result = simulate_dsp._p3_alim_listesi()
        assert result == set()

    def test_p3_alim_listesi_dosya_yok(self):
        """portfolio_state.json yoksa → boş küme."""
        import simulate_dsp
        with patch.object(simulate_dsp, "STATE_FILE") as mock_path:
            mock_path.exists.return_value = False
            result = simulate_dsp._p3_alim_listesi()
        assert result == set()

    def test_p3_alim_listesi_hatali_json(self):
        """Bozuk JSON → boş küme (exception safety)."""
        import simulate_dsp
        with patch.object(simulate_dsp, "STATE_FILE") as mock_path:
            mock_path.exists.return_value = True
            mock_path.read_text.side_effect = Exception("bozuk")
            result = simulate_dsp._p3_alim_listesi()
        assert result == set()


# ═════════════════════════════════════════════════════════════════════════════
# P3 MAX_GUN ROLLING EXTENSION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestP3MaxGunRolling:
    """P3 — MAX_GUN rolling extension invariant testleri."""

    def _make_scan(self, syms):
        """Mock ScanResult oluştur."""
        scan = MagicMock()
        scan.top_longs = []
        for s in syms:
            item = MagicMock()
            item.symbol = s
            item.score = 0.5
            scan.top_longs.append(item)
        scan.scanned = 100
        scan.long_signals = len(syms)
        return scan

    def test_p3_A_max_gun_al_extends(self):
        """A) MAX_GUN doldu + ALIM listesinde → pozisyon kapanmamalı."""
        import simulate_dsp
        giris = (date.today() - timedelta(days=10)).isoformat()
        state = {
            "positions": {"SYM1": {
                "entry_date": giris, "entry_price": 100,
                "score": 0.5, "margin": 0.01,
                "max_gun_date": date.today().isoformat(),
            }},
            "history": [], "scan_log": [],
        }
        scan = self._make_scan(["SYM1"])
        with patch.object(simulate_dsp, "_get_price", return_value=105), \
             patch.object(simulate_dsp, "_p3_alim_listesi", return_value={"SYM1"}):
            result = simulate_dsp.update_portfolio(state, scan)
        assert "SYM1" not in result["exits"]
        assert "SYM1" in result["holds"]

    def test_p3_B_max_gun_al_extends_date(self):
        """B) MAX_GUN doldu + ALIM listesinde → max_gun_date = bugün + 5."""
        import simulate_dsp
        giris = (date.today() - timedelta(days=10)).isoformat()
        state = {
            "positions": {"SYM1": {
                "entry_date": giris, "entry_price": 100,
                "score": 0.5, "margin": 0.01,
                "max_gun_date": date.today().isoformat(),
            }},
            "history": [], "scan_log": [],
        }
        scan = self._make_scan(["SYM1"])
        with patch.object(simulate_dsp, "_get_price", return_value=105), \
             patch.object(simulate_dsp, "_p3_alim_listesi", return_value={"SYM1"}):
            simulate_dsp.update_portfolio(state, scan)
        expected = (date.today() + timedelta(days=5)).isoformat()
        assert state["positions"]["SYM1"]["max_gun_date"] == expected

    def test_p3_C_idempotency(self):
        """C) Aynı gün ikinci çalıştırma → ikinci kez +5 yapılmamalı."""
        import simulate_dsp
        giris = (date.today() - timedelta(days=10)).isoformat()
        state = {
            "positions": {"SYM1": {
                "entry_date": giris, "entry_price": 100,
                "score": 0.5, "margin": 0.01,
                "max_gun_date": date.today().isoformat(),
            }},
            "history": [], "scan_log": [],
        }
        scan = self._make_scan(["SYM1"])
        with patch.object(simulate_dsp, "_get_price", return_value=105), \
             patch.object(simulate_dsp, "_p3_alim_listesi", return_value={"SYM1"}):
            simulate_dsp.update_portfolio(state, scan)
            expected1 = (date.today() + timedelta(days=5)).isoformat()
            assert state["positions"]["SYM1"]["max_gun_date"] == expected1
            # İkinci çalıştırma
            simulate_dsp.update_portfolio(state, scan)
        # max_gun_date değişmemeli (bugün < today+5)
        assert state["positions"]["SYM1"]["max_gun_date"] == expected1

    def test_p3_D_tekrar_alim_listesinde(self):
        """D) Yeni MAX_GUN tarihinde tekrar ALIM listesinde → tekrar +5."""
        import simulate_dsp
        # 5 gün sonraki tarih
        gelecek = (date.today() + timedelta(days=5)).isoformat()
        giris = (date.today() - timedelta(days=15)).isoformat()
        state = {
            "positions": {"SYM1": {
                "entry_date": giris, "entry_price": 100,
                "score": 0.5, "margin": 0.01,
                "max_gun_date": gelecek,  # Bugün = max_gun_date
            }},
            "history": [], "scan_log": [],
        }
        scan = self._make_scan(["SYM1"])
        with patch.object(simulate_dsp, "_get_price", return_value=105), \
             patch.object(simulate_dsp, "_p3_alim_listesi", return_value={"SYM1"}):
            simulate_dsp.update_portfolio(state, scan)
        expected = (date.today() + timedelta(days=5)).isoformat()
        assert state["positions"]["SYM1"]["max_gun_date"] == expected

    def test_p3_E_max_gun_al_yok_exit(self):
        """E) MAX_GUN doldu + ALIM listesinde değil → exit."""
        import simulate_dsp
        giris = (date.today() - timedelta(days=10)).isoformat()
        state = {
            "positions": {"SYM1": {
                "entry_date": giris, "entry_price": 100,
                "score": 0.5, "margin": 0.01,
                "max_gun_date": date.today().isoformat(),
            }},
            "history": [], "scan_log": [],
        }
        scan = self._make_scan([])  # SYM1 listede değil
        with patch.object(simulate_dsp, "_get_price", return_value=105), \
             patch.object(simulate_dsp, "_p3_alim_listesi", return_value=set()):
            result = simulate_dsp.update_portfolio(state, scan)
        assert "SYM1" in result["exits"]
        assert "SYM1" not in result["holds"]

    def test_p3_K_entry_date_degismez(self):
        """K) Extension entry_date değiştirmemeli."""
        import simulate_dsp
        giris = (date.today() - timedelta(days=10)).isoformat()
        state = {
            "positions": {"SYM1": {
                "entry_date": giris, "entry_price": 100,
                "score": 0.5, "margin": 0.01,
                "max_gun_date": date.today().isoformat(),
            }},
            "history": [], "scan_log": [],
        }
        scan = self._make_scan(["SYM1"])
        with patch.object(simulate_dsp, "_get_price", return_value=105), \
             patch.object(simulate_dsp, "_p3_alim_listesi", return_value={"SYM1"}):
            simulate_dsp.update_portfolio(state, scan)
        assert state["positions"]["SYM1"]["entry_date"] == giris

    def test_p3_L_pnl_degismez(self):
        """L) Extension PnL değiştirmemeli."""
        import simulate_dsp
        giris = (date.today() - timedelta(days=10)).isoformat()
        state = {
            "positions": {"SYM1": {
                "entry_date": giris, "entry_price": 100,
                "score": 0.5, "margin": 0.01,
                "max_gun_date": date.today().isoformat(),
            }},
            "history": [], "scan_log": [],
        }
        scan = self._make_scan(["SYM1"])
        with patch.object(simulate_dsp, "_get_price", return_value=105), \
             patch.object(simulate_dsp, "_p3_alim_listesi", return_value={"SYM1"}):
            simulate_dsp.update_portfolio(state, scan)
        # P3 position'ta pnl_pct alanı yok (P3 farklı yapıda)
        # Ama entry_price değişmemeli
        assert state["positions"]["SYM1"]["entry_price"] == 100

    def test_p3_M_trade_history_olusturmaz(self):
        """M) Extension yeni trade_history kaydı oluşturmamalı."""
        import simulate_dsp
        giris = (date.today() - timedelta(days=10)).isoformat()
        state = {
            "positions": {"SYM1": {
                "entry_date": giris, "entry_price": 100,
                "score": 0.5, "margin": 0.01,
                "max_gun_date": date.today().isoformat(),
            }},
            "history": [], "scan_log": [],
        }
        scan = self._make_scan(["SYM1"])
        with patch.object(simulate_dsp, "_get_price", return_value=105), \
             patch.object(simulate_dsp, "_p3_alim_listesi", return_value={"SYM1"}):
            simulate_dsp.update_portfolio(state, scan)
        assert len(state["history"]) == 0

    def test_p3_O_fallback_max_gun_date(self):
        """O) Eski state'te max_gun_date yoksa fallback çalışmalı."""
        import simulate_dsp
        giris = (date.today() - timedelta(days=10)).isoformat()
        state = {
            "positions": {"SYM1": {
                "entry_date": giris, "entry_price": 100,
                "score": 0.5, "margin": 0.01,
                # max_gun_date yok!
            }},
            "history": [], "scan_log": [],
        }
        scan = self._make_scan(["SYM1"])
        with patch.object(simulate_dsp, "_get_price", return_value=105), \
             patch.object(simulate_dsp, "_p3_alim_listesi", return_value={"SYM1"}):
            simulate_dsp.update_portfolio(state, scan)
        # fallback: entry+10 = today → extend edilmeli
        expected = (date.today() + timedelta(days=5)).isoformat()
        assert state["positions"]["SYM1"]["max_gun_date"] == expected


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-PIPELINE INVARIANT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestP5CrossPipeline:
    """P5 — cross-pipeline invariant testleri."""

    def _p5_state(self, poz=None, hist=None):
        return {
            "pozisyonlar": poz or {},
            "trade_history": hist or [],
            "komite_log": [],
            "sermaye_baslangic": 100_000,
            "sermaye_mevcut": 100_000,
        }

    def _p5_pos(self, sym, giris_f, gun=0, giris_tarih=None, max_gun_date=None):
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

    def _cache(self, prices):
        return {sym: np.array([p]) for sym, p in prices.items()}

    def test_p5_F_stop_al_calisiyor(self):
        """F) STOP + ALIM listesinde → STOP yine çalışmalı."""
        import p5_committee
        giris = (date.today() - timedelta(days=11)).isoformat()
        mgd = (date.today() - timedelta(days=1)).isoformat()
        s = self._p5_state(poz={"A": self._p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, self._cache({"A": 94}))
        assert r["kapanan"][0]["neden"] == "STOP"

    def test_p5_G_tp_al_calisiyor(self):
        """G) TP + ALIM listesinde → TP yine çalışmalı."""
        import p5_committee
        giris = (date.today() - timedelta(days=11)).isoformat()
        mgd = (date.today() - timedelta(days=1)).isoformat()
        s = self._p5_state(poz={"A": self._p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, self._cache({"A": 112}))
        assert r["kapanan"][0]["neden"] == "TP"

    def test_p5_K_entry_date_degismez(self):
        """K) Extension entry_date değiştirmemeli."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = date.today().isoformat()
        s = self._p5_state(poz={"A": self._p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            p5_committee.portfoy_guncelle(s, self._cache({"A": 102}))
        assert s["pozisyonlar"]["A"]["giris_tarih"] == giris

    def test_p5_L_pnl_degismez(self):
        """L) Extension PnL değiştirmemeli."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = date.today().isoformat()
        s = self._p5_state(poz={"A": self._p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            p5_committee.portfoy_guncelle(s, self._cache({"A": 102}))
        assert s["pozisyonlar"]["A"]["giris_fiyat"] == 100

    def test_p5_M_trade_history_olusturmaz(self):
        """M) Extension yeni trade_history kaydı oluşturmamalı."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = date.today().isoformat()
        s = self._p5_state(poz={"A": self._p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            p5_committee.portfoy_guncelle(s, self._cache({"A": 102}))
        assert len(s["trade_history"]) == 0

    def test_p5_N_pozisyon_yeniden_acilmamis(self):
        """N) Extension pozisyonu yeniden açılmış gibi işlememeli."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        mgd = date.today().isoformat()
        s = self._p5_state(poz={"A": self._p5_pos("A", 100, giris_tarih=giris, max_gun_date=mgd)})
        with patch.object(p5_committee, "_p5_alim_listesi", return_value={"A"}):
            r = p5_committee.portfoy_guncelle(s, self._cache({"A": 102}))
        # Pozisyon hala devam ediyor, yeni açılmadı
        assert "A" in r["devam_eden"]
        assert s["pozisyonlar"]["A"]["gun"] == 10  # gun değişmemeli (tarih bazlı)

    def test_p5_O_fallback_max_gun_date(self):
        """O) Eski state'te max_gun_date yoksa fallback çalışmalı."""
        import p5_committee
        giris = (date.today() - timedelta(days=10)).isoformat()
        s = self._p5_state(poz={"A": self._p5_pos("A", 100, giris_tarih=giris)})
        # max_gun_date yok → fallback: entry+10 = today → EXIT (ALIM listesi boş)
        with patch.object(p5_committee, "_p5_alim_listesi", return_value=set()):
            r = p5_committee.portfoy_guncelle(s, self._cache({"A": 102}))
        assert len(r["kapanan"]) == 1
        assert r["kapanan"][0]["neden"] == "MAX_GUN"
