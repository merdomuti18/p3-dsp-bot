"""
DSP-P3 FAZ 6.2 — State Coordination Unit Tests
=================================================
T2  kitaplar_arasi_taze_sayi — bayat state koruması
T11 Workflow YAML contract — P5 job has state_p4.json checkout

Ek testler: stamp_state, is_state_fresh, atomic_write_json
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import mott_state_coordination as msc
import mott_risk

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fresh_state(gen: int = 1, hours_ago: float = 1.0) -> dict:
    """Taze state dict — _gen ve _updated_at ile."""
    ts = (msc.now_tsi() - timedelta(hours=hours_ago)).isoformat()
    return {
        "_gen": gen,
        "_updated_at": ts,
        "pozisyonlar": {"SYM_A": {"giris_fiyat": 100.0}},
        "trade_history": [],
    }


def _stale_state(hours_ago: float = 48.0) -> dict:
    """Bayat state dict — _updated_at eski."""
    ts = (msc.now_tsi() - timedelta(hours=hours_ago)).isoformat()
    return {
        "_gen": 5,
        "_updated_at": ts,
        "pozisyonlar": {"SYM_A": {"giris_fiyat": 100.0}},
        "trade_history": [],
    }


def _no_gen_state() -> dict:
    """_gen ve _updated_at olmayan eski-surum state."""
    return {
        "pozisyonlar": {"SYM_A": {"giris_fiyat": 100.0}},
        "trade_history": [],
    }


# ── T2: stamp_state ─────────────────────────────────────────────────────────

class TestStampState:
    """_gen ve _updated_at dogru artar/mi."""

    def test_first_stamp(self):
        """Ilk cagirida _gen=1, _updated_at dolu olmali."""
        state = {"pozisyonlar": {}}
        msc.stamp_state(state)
        assert state["_gen"] == 1
        assert "_updated_at" in state

    def test_incremental_stamp(self):
        """Her cagirida _gen 1 artmali."""
        state = {"_gen": 3, "_updated_at": "old"}
        msc.stamp_state(state)
        assert state["_gen"] == 4

    def test_stamp_returns_same_dict(self):
        """stamp_state ayni dict'i dondurur."""
        state = {}
        result = msc.stamp_state(state)
        assert result is state


# ── T2: is_state_fresh ──────────────────────────────────────────────────────

class TestIsStateFresh:
    """Taze/bayat state tespiti."""

    def test_fresh_1h(self):
        """1 saat once yazilmis state taze olmali."""
        assert msc.is_state_fresh(_fresh_state(hours_ago=1.0)) is True

    def test_stale_48h(self):
        """48 saat once yazilmis state bayat olmali."""
        assert msc.is_state_fresh(_stale_state(hours_ago=48.0)) is False

    def test_no_gen_is_stale(self):
        """_gen olmayan state bayat sayilmali."""
        assert msc.is_state_fresh(_no_gen_state()) is False

    def test_gen_zero_is_stale(self):
        """_gen=0 olan state bayat sayilmali."""
        state = {"_gen": 0, "_updated_at": msc.now_tsi().isoformat()}
        assert msc.is_state_fresh(state) is False

    def test_no_timestamp_is_stale(self):
        """_updated_at olmayan state bayat sayilmali."""
        state = {"_gen": 1}
        assert msc.is_state_fresh(state) is False

    def test_custom_threshold(self):
        """Threshold ozel deger ile test."""
        # 2 saat once, threshold 1 saat -> stale
        state = _fresh_state(hours_ago=2.0)
        assert msc.is_state_fresh(state, threshold_hours=1) is False
        # Ayni state, threshold 3 saat -> taze
        assert msc.is_state_fresh(state, threshold_hours=3) is True


# ── T2: kitaplar_arasi_taze_sayi ────────────────────────────────────────────

class TestKitaplarArasiTazeSayi:
    """Bayat state koruması — L1a risk fix."""

    def test_fresh_state_counted(self, tmp_path):
        """Taze state'de SYM_A sayilmali."""
        mock_kitap = {
            "P4": ("state_p4.json", "pozisyonlar"),
        }
        # Taze state_p4.json olustur
        state = _fresh_state(gen=5, hours_ago=1.0)
        (tmp_path / "state_p4.json").write_text(json.dumps(state), encoding="utf-8")

        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = msc.kitaplar_arasi_taze_sayi(
                "SYM_A", mock_kitap, threshold_hours=14
            )
        assert sayi == 1
        assert bayat == []

    def test_stale_state_counted_conservative(self, tmp_path):
        """Bayat state'de SYM_A SAYILIR (konservatif: bilinmeyen risk = dolu)."""
        mock_kitap = {
            "P4": ("state_p4.json", "pozisyonlar"),
        }
        state = _stale_state(hours_ago=48.0)
        (tmp_path / "state_p4.json").write_text(json.dumps(state), encoding="utf-8")

        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = msc.kitaplar_arasi_taze_sayi(
                "SYM_A", mock_kitap, threshold_hours=14
            )
        assert sayi == 1  # bayat olsa bile sayilir (N1 invariant korunur)
        assert bayat == ["P4"]

    def test_no_gen_state_counted_conservative(self, tmp_path):
        """_gen olmayan state SAYILIR ama bayat listesine eklenir (konservatif)."""
        mock_kitap = {
            "P4": ("state_p4.json", "pozisyonlar"),
        }
        state = _no_gen_state()
        (tmp_path / "state_p4.json").write_text(json.dumps(state), encoding="utf-8")

        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = msc.kitaplar_arasi_taze_sayi(
                "SYM_A", mock_kitap, threshold_hours=14
            )
        assert sayi == 1  # _gen yok -> bayat ama sayilir
        assert bayat == ["P4"]

    def test_mixed_fresh_and_stale(self, tmp_path):
        """1 taze + 1 bayat: toplam sayisi=2 (bayat da sayılır, N1 korunur)."""
        mock_kitap = {
            "P4": ("state_p4.json", "pozisyonlar"),
            "P5": ("state_p5.json", "pozisyonlar"),
        }
        # P4 taze, P5 bayat
        fresh = _fresh_state(gen=3, hours_ago=1.0)
        stale = _stale_state(hours_ago=48.0)
        (tmp_path / "state_p4.json").write_text(json.dumps(fresh), encoding="utf-8")
        (tmp_path / "state_p5.json").write_text(json.dumps(stale), encoding="utf-8")

        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = msc.kitaplar_arasi_taze_sayi(
                "SYM_A", mock_kitap, threshold_hours=14
            )
        assert sayi == 2  # her ikisi de sayılır
        assert bayat == ["P5"]  # sadece P5 bayat

    def test_excluded_kitap_not_counted(self, tmp_path):
        """haric parametresi ile dislanan kitap sayilmamali."""
        mock_kitap = {
            "P4": ("state_p4.json", "pozisyonlar"),
        }
        state = _fresh_state(gen=3, hours_ago=1.0)
        (tmp_path / "state_p4.json").write_text(json.dumps(state), encoding="utf-8")

        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = msc.kitaplar_arasi_taze_sayi(
                "SYM_A", mock_kitap, haric="P4", threshold_hours=14
            )
        assert sayi == 0
        assert bayat == []

    def test_missing_file_skipped(self, tmp_path):
        """Dosya mevcut degilse o kitap atlanmali."""
        mock_kitap = {
            "P4": ("state_p4.json", "pozisyonlar"),
        }
        # state_p4.json dosyasi yok
        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = msc.kitaplar_arasi_taze_sayi(
                "SYM_A", mock_kitap, threshold_hours=14
            )
        assert sayi == 0
        assert bayat == []

    def test_symbol_not_in_positions(self, tmp_path):
        """Sembol pozisyonlarda yoksa sayi=0."""
        mock_kitap = {
            "P4": ("state_p4.json", "pozisyonlar"),
        }
        state = _fresh_state(gen=3, hours_ago=1.0)
        state["pozisyonlar"] = {"DIFFERENT_SYM": {}}
        (tmp_path / "state_p4.json").write_text(json.dumps(state), encoding="utf-8")

        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = msc.kitaplar_arasi_taze_sayi(
                "SYM_A", mock_kitap, threshold_hours=14
            )
        assert sayi == 0
        assert bayat == []


# ── T2: mott_risk wrapper ───────────────────────────────────────────────────

class TestMottRiskTazeSayi:
    """mott_risk.kitaplar_arasi_taze_sayi wrapper."""

    def test_fresh_p4_counted(self, tmp_path):
        """Fresh P4 state'te SYM sayilmali."""
        state = _fresh_state(gen=5, hours_ago=1.0)
        (tmp_path / "state_p4.json").write_text(json.dumps(state), encoding="utf-8")

        # kitaplar_arasi_taze_sayi mott_state_coordination.BASE_DIR kullanir
        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = mott_risk.kitaplar_arasi_taze_sayi(
                "SYM_A", haric=None, threshold_hours=14
            )
        assert sayi == 1
        assert bayat == []

    def test_stale_p4_counted_conservative(self, tmp_path):
        """Bayat P4 state'te SYM SAYILIR (konservatif: bilinmeyen risk = dolu)."""
        state = _stale_state(hours_ago=48.0)
        (tmp_path / "state_p4.json").write_text(json.dumps(state), encoding="utf-8")

        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = mott_risk.kitaplar_arasi_taze_sayi(
                "SYM_A", haric=None, threshold_hours=14
            )
        assert sayi == 1  # bayat ama sayılır
        assert bayat == ["P4"]

    def test_backward_compat_old_caller(self, tmp_path):
        """Eski kitaplar_arasi_sayi() degismemis olmali."""
        state = _stale_state(hours_ago=48.0)
        (tmp_path / "state_p4.json").write_text(json.dumps(state), encoding="utf-8")

        with patch.object(mott_risk, "BASE_DIR", tmp_path):
            # Eski fonksiyon hala sayar (bayat fark etmez)
            sayi = mott_risk.kitaplar_arasi_sayi("SYM_A", haric=None)
        assert sayi == 1  # bayat olsa bile eski sayi sayar

    def test_stale_book_blocks_new_position(self, tmp_path):
        """Bayat kitapta SYM_A varsa MAX_KITAP asimi sayilmali (N1 invariant)."""
        # Senaryo: P3'te SYM_A acik ama state bayat (48 saat once)
        # P1'de SYM_A acik (taze)
        # MAX_KITAP=2 ise, P4/P5 icin SYM_A acilmamali
        mock_kitap = {
            "P1": ("portfoy.json", "pozisyonlar"),
            "P3": ("portfolio_state.json", "positions"),
        }
        # P1 taze
        fresh_p1 = _fresh_state(gen=5, hours_ago=1.0)
        (tmp_path / "portfoy.json").write_text(json.dumps(fresh_p1), encoding="utf-8")
        # P3 bayat
        stale_p3 = {
            "_gen": 3,
            "_updated_at": (msc.now_tsi() - timedelta(hours=48)).isoformat(),
            "positions": {"SYM_A": {"giris_fiyat": 100.0}},
        }
        (tmp_path / "portfolio_state.json").write_text(json.dumps(stale_p3), encoding="utf-8")

        with patch.object(msc, "BASE_DIR", tmp_path):
            sayi, bayat = msc.kitaplar_arasi_taze_sayi(
                "SYM_A", mock_kitap, threshold_hours=14
            )
        # sayi=2: P1 taze + P3 bayat ama sayılır → kitap_limiti_asildi=True
        assert sayi == 2
        assert "P3" in bayat


# ── T11: Workflow YAML Contract ─────────────────────────────────────────────

class TestWorkflowYAMLContract:
    """P5 job'da state_p4.json checkout var mi (FAZ 6.2 M1a)."""

    def _load_workflow(self) -> str:
        wf_path = Path(__file__).resolve().parent.parent.parent / \
            ".github" / "workflows" / "mott_daily.yml"
        return wf_path.read_text(encoding="utf-8")

    def test_p5_komite_has_state_p4_checkout(self):
        """p5-komite job'inda state_p4.json checkout adimi olmali."""
        content = self._load_workflow()

        # p5-komite section'ini bul
        p5_start = content.find("p5-komite:")
        assert p5_start >= 0, "p5-komite job not found in workflow"

        # P5 section'inin sonunu bul (bir sonraki job)
        p5_section = content[p5_start:]

        # state_p4.json checkout adimi olmali
        assert "state_p4.json" in p5_section, \
            "p5-komite job missing state_p4.json checkout (FAZ 6.2 M1a)"

    def test_p5_monitor_has_state_p4_checkout(self):
        """p5-monitor job'inda state_p4.json checkout adimi olmali."""
        content = self._load_workflow()

        # p5-monitor section'ini bul
        p5m_start = content.find("p5-monitor:")
        assert p5m_start >= 0, "p5-monitor job not found in workflow"

        p5m_section = content[p5m_start:]

        assert "state_p4.json" in p5m_section, \
            "p5-monitor job missing state_p4.json checkout (FAZ 6.2 M1a)"

    def test_p4_meta_has_p1_p2_p3_checkout(self):
        """p4-meta job'inda P1/P2/P3 state checkout hala mevcut olmali."""
        content = self._load_workflow()
        p4_start = content.find("p4-meta:")
        assert p4_start >= 0
        p4_section = content[p4_start:]
        for f in ["state_p1.json", "state_p2.json", "portfolio_state.json"]:
            assert f in p4_section, f"p4-meta missing {f}"

    def test_p5_komite_depends_on_p4_meta(self):
        """p5-komite needs listesi p4-meta'yi icermeli (FAZ 6.2 race fix)."""
        content = self._load_workflow()
        p5_start = content.find("p5-komite:")
        assert p5_start >= 0
        # Bir sonraki job'a kadar olan bolumu al
        p5m_start = content.find("p5-monitor:", p5_start)
        p5_section = content[p5_start:p5m_start] if p5m_start > p5_start else content[p5_start:]
        assert "p4-meta" in p5_section, \
            "p5-komite needs listesi p4-meta iciermiyor — race condition devam eder"


# ── atomic_write_json ────────────────────────────────────────────────────────

class TestAtomicWriteJSON:
    """Atomik yazma dogrulugu."""

    def test_write_and_read(self, tmp_path):
        """Yazilan dosya dogru okunmali."""
        data = {"test": True, "nested": {"a": 1}}
        path = tmp_path / "test.json"
        msc.atomic_write_json(path, data)

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_no_tmp_file_left(self, tmp_path):
        """Basarili yazma sonrasi .tmp dosyasi kalmamali."""
        data = {"key": "value"}
        path = tmp_path / "test.json"
        msc.atomic_write_json(path, data)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_existing_file_replaced(self, tmp_path):
        """Mevcut dosya uzerine yazma — eski icerik gitmeli."""
        path = tmp_path / "test.json"
        path.write_text('{"old": true}', encoding="utf-8")

        msc.atomic_write_json(path, {"new": True})
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == {"new": True}
        assert "old" not in loaded
