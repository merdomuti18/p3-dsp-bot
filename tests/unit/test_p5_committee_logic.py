"""
DSP-P3 FAZ 5 — P5 Committee Business Logic Certification Tests
================================================================
Bu takım, p5_committee.py'nin iş mantığı invariantlarını test eder.
Production kodu DEĞİŞTİRİLMEZ; yalnızca mevcut davranış doğrulanır.

Network YOK: tüm girdiler monkeypatch ile kontrol edilir.

Scoring formülü (p5_committee.py Ln 181-195):
  base = 60 (2 kaynak) | 100 (3 kaynak)
  + 15  → P2 verdict == "GÜÇLÜ AL"
  + 10  → P1 score_count >= 3
  + 10  → sym in p3_top_set(3)
  MAX = 135
"""

from __future__ import annotations

import p5_committee


# ── Autouse default mocks ────────────────────────────────────────────────────
# Her test fonksiyonu için boş state, driftsiz, NORMAL makro.
# Tek tek test monkeypatch ile override eder.

import pytest


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: set())
    monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: set())
    monkeypatch.setattr(p5_committee, "p3_top_set", lambda limit=15: set())
    monkeypatch.setattr(p5_committee, "p1_detay", lambda: {})
    monkeypatch.setattr(p5_committee, "p2_detay", lambda: {})
    monkeypatch.setattr(p5_committee, "p3_detay", lambda: {})
    monkeypatch.setattr(p5_committee, "drift_sembolleri", lambda: set())
    monkeypatch.setattr(p5_committee, "makro_karar", lambda: "NORMAL")
    monkeypatch.setattr(p5_committee, "get_sector", lambda sym: "DIGER")


def _skor(secilen: list[dict], sym: str) -> int:
    """secilen listesinden sembolün komite_skor'unu bul."""
    return next(s["komite_skor"] for s in secilen if s["symbol"] == sym)


# ═════════════════════════════════════════════════════════════════════════════
# A) MIN_KAYNAK — en az 2 bağımsız kaynak gerekli
# ═════════════════════════════════════════════════════════════════════════════

class TestMinKaynak:
    def test_tek_kaynak_p1_reddedilir(self, monkeypatch):
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"X"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"X": {"score_count": 1}})
        secilen, elenen = p5_committee.komite_adaylari()
        assert len(secilen) == 0
        assert any(e["symbol"] == "X" and "tek kaynak" in e["neden"] for e in elenen)

    def test_tek_kaynak_p2_reddedilir(self, monkeypatch):
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"Y"})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"Y": {"verdict": "AL"}})
        secilen, elenen = p5_committee.komite_adaylari()
        assert len(secilen) == 0
        assert any(e["symbol"] == "Y" for e in elenen)

    def test_iki_kaynak_kabul(self, monkeypatch):
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert any(s["symbol"] == "A" for s in secilen)

    def test_uc_kaynak_kabul(self, monkeypatch):
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p3_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert any(s["symbol"] == "A" for s in secilen)


# ═════════════════════════════════════════════════════════════════════════════
# B) SCORING — mevcut skor formülünün doğruluğu
# ═════════════════════════════════════════════════════════════════════════════

class TestScoring:
    def test_iki_kaynak_base_60(self, monkeypatch):
        """P1+P2, bonus yok → 60."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert _skor(secilen, "A") == 60

    def test_uc_kaynak_base_100(self, monkeypatch):
        """P1+P2+P3, bonus yok → 100.
        A top3'te değil (p3_top_set(3) = {B,C,D}) → P3 bonus uygulanmaz."""
        def _p3(limit=15):
            return set(["B", "C", "D", "A"][:limit])

        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p3_top_set", _p3)
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert _skor(secilen, "A") == 100

    def test_p2_guclu_al_bonus_15(self, monkeypatch):
        """P2 verdict='GÜÇLÜ AL' → +15."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "GÜÇLÜ AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert _skor(secilen, "A") == 75   # 60 + 15

    def test_p1_uc_strateji_bonus_10(self, monkeypatch):
        """P1 score_count >= 3 → +10."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 3}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert _skor(secilen, "A") == 70   # 60 + 10

    def test_p1_iki_strateji_bonus_yok(self, monkeypatch):
        """P1 score_count = 2 (< 3) → bonus yok."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 2}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert _skor(secilen, "A") == 60   # base, bonus yok

    def test_p3_top3_bonus_10(self, monkeypatch):
        """P3 top3'te → +10."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p3_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        # 3 kaynak = 100, P3 top3 = +10 → 110
        assert _skor(secilen, "A") == 110

    def test_p3_top3_disinda_bonus_yok(self, monkeypatch):
        """P3 set'te ama top3'te değil → P3 bonus yok."""
        # p3_top_set(3) = {"B","C","D"} → A top3'te değil
        def _p3(limit=15):
            return set(["B", "C", "D", "A"][:limit])

        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p3_top_set", _p3)
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert _skor(secilen, "A") == 100  # 3 kaynak base, P3 bonus yok

    def test_maksimum_skor_135(self, monkeypatch):
        """3 kaynak + tüm bonuslar → 135."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p3_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 5}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "GÜÇLÜ AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        # 100 + 15 (P2) + 10 (P1≥3) + 10 (P3 top3) = 135
        assert _skor(secilen, "A") == 135


# ═════════════════════════════════════════════════════════════════════════════
# C) MAX_SEKTOR — sektör başına en fazla 2 pozisyon
# ═════════════════════════════════════════════════════════════════════════════

class TestMaxSektor:
    def test_ayni_sektorden_ucuncu_elendi(self, monkeypatch):
        """Aynı sektörden 3 aday → en fazla 2 kabul."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A", "B", "C"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A", "B", "C"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {
            s: {"score_count": 1} for s in ("A", "B", "C")
        })
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {
            s: {"verdict": "AL"} for s in ("A", "B", "C")
        })
        monkeypatch.setattr(p5_committee, "get_sector", lambda sym: "AYNI")
        secilen, elenen = p5_committee.komite_adaylari()
        assert len(secilen) == 2
        assert any("sektör dolu" in e["neden"] for e in elenen)

    def test_farkli_sektorlerde_hepsi_kabul(self, monkeypatch):
        """Farklı sektörler → sektör limiti dolmaz."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A", "B"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A", "B"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {
            "A": {"score_count": 1}, "B": {"score_count": 1}
        })
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {
            "A": {"verdict": "AL"}, "B": {"verdict": "AL"}
        })
        monkeypatch.setattr(p5_committee, "get_sector", lambda sym: f"S_{sym}")
        secilen, _ = p5_committee.komite_adaylari()
        assert len(secilen) == 2


# ═════════════════════════════════════════════════════════════════════════════
# D) MAKRO KAPISI — makro kararın komiteye etkisi
# ═════════════════════════════════════════════════════════════════════════════

class TestMakroKapisi:
    def test_girme_tumunu_elender(self, monkeypatch):
        """GIRME → tüm adaylar elenmeli."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "makro_karar", lambda: "GIRME")
        secilen, elenen = p5_committee.komite_adaylari()
        assert len(secilen) == 0
        assert any("makro GIRME" in e["neden"] for e in elenen)

    def test_normal_limitsiz(self, monkeypatch):
        """NORMAL → adaylar normal değerlendirilir."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert len(secilen) == 1

    def test_dikkatli_komiteyi_etkilemez(self, monkeypatch):
        """DIKKATLI → komite aday seçimi ETKİLENMEZ
        (slot limiti mott_risk.rejim_slot_limiti'nde, komite seçiminde değil)."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        monkeypatch.setattr(p5_committee, "makro_karar", lambda: "DIKKATLI")
        secilen, _ = p5_committee.komite_adaylari()
        assert len(secilen) == 1  # DIKKATLI komiteyi etkilemez


# ═════════════════════════════════════════════════════════════════════════════
# E) DRIFT — P3 drift uyarısı filtresi
# ═════════════════════════════════════════════════════════════════════════════

class TestDrift:
    def test_drift_reddeder(self, monkeypatch):
        """P3 drift uyarısı olan sembol → elenmeli."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p3_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "drift_sembolleri", lambda: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, elenen = p5_committee.komite_adaylari()
        assert len(secilen) == 0
        assert any("P3 drift" in e["neden"] for e in elenen)

    def test_drift_digerini_etkilemez(self, monkeypatch):
        """Drift olan sembol elenir, diğerleri korunur."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A", "B"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A", "B"})
        monkeypatch.setattr(p5_committee, "drift_sembolleri", lambda: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {
            "A": {"score_count": 1}, "B": {"score_count": 1}
        })
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {
            "A": {"verdict": "AL"}, "B": {"verdict": "AL"}
        })
        secilen, elenen = p5_committee.komite_adaylari()
        assert any(s["symbol"] == "B" for s in secilen)
        assert not any(s["symbol"] == "A" for s in secilen)


# ═════════════════════════════════════════════════════════════════════════════
# F) SIRALAMA — deterministik ve skor bazlı
# ═════════════════════════════════════════════════════════════════════════════

class TestSiralama:
    def test_deterministik(self, monkeypatch):
        """Aynı girdi → her çalıştırmada aynı sıralama."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A", "B"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A", "B"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {
            "A": {"score_count": 5}, "B": {"score_count": 1}
        })
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {
            "A": {"verdict": "GÜÇLÜ AL"}, "B": {"verdict": "AL"}
        })
        monkeypatch.setattr(p5_committee, "get_sector", lambda sym: f"S_{sym}")

        s1, _ = p5_committee.komite_adaylari()
        s2, _ = p5_committee.komite_adaylari()
        assert [x["symbol"] for x in s1] == [x["symbol"] for x in s2]

    def test_yuksek_skor_once(self, monkeypatch):
        """Yüksek skorlu aday önce gelmeli."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"HI", "LO"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"HI", "LO"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {
            "HI": {"score_count": 5}, "LO": {"score_count": 1}
        })
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {
            "HI": {"verdict": "GÜÇLÜ AL"}, "LO": {"verdict": "AL"}
        })
        monkeypatch.setattr(p5_committee, "get_sector", lambda sym: f"S_{sym}")

        secilen, _ = p5_committee.komite_adaylari()
        assert secilen[0]["symbol"] == "HI"
        assert secilen[0]["komite_skor"] > secilen[1]["komite_skor"]


# ═════════════════════════════════════════════════════════════════════════════
# G) EMPTY / MALFORMED — boş veya eksik girdi güvenliği
# ═════════════════════════════════════════════════════════════════════════════

class TestEmptyMalformed:
    def test_tam_bos(self):
        """Tüm state boş → aday yok, exception yok."""
        secilen, elenen = p5_committee.komite_adaylari()
        assert secilen == []
        assert elenen == []

    def test_yalnizca_p1(self, monkeypatch):
        """Yalnızca P1 → tek kaynak → elenmeli."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"X"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"X": {"score_count": 1}})
        secilen, elenen = p5_committee.komite_adaylari()
        assert secilen == []
        assert len(elenen) == 1

    def test_yalnizca_p3(self, monkeypatch):
        """Yalnızca P3 → tek kaynak → elenmeli."""
        monkeypatch.setattr(p5_committee, "p3_top_set", lambda limit=15: {"Z"})
        secilen, elenen = p5_committee.komite_adaylari()
        assert secilen == []
        assert len(elenen) == 1

    def test_detay_bos_ama_kaynak_var(self, monkeypatch):
        """Kaynaklar var ama detay boş → bonus yok, base skor."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        # p1_detay, p2_detay _defaults'ta zaten {} dönüyor
        secilen, _ = p5_committee.komite_adaylari()
        assert len(secilen) == 1
        assert secilen[0]["komite_skor"] == 60

    def test_p1_detay_eksik_sembol(self, monkeypatch):
        """P1 set'te var ama p1_detay'da yok → p1 score_count bonusu yok."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {})  # A yok
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"verdict": "AL"}})
        secilen, _ = p5_committee.komite_adaylari()
        assert _skor(secilen, "A") == 60  # base, bonus yok

    def test_p2_detay_bos_verdict(self, monkeypatch):
        """P2 detay'da verdict yok → GÜÇLÜ AL bonusu yok."""
        monkeypatch.setattr(p5_committee, "p1_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p2_top_set", lambda limit=15: {"A"})
        monkeypatch.setattr(p5_committee, "p1_detay", lambda: {"A": {"score_count": 1}})
        monkeypatch.setattr(p5_committee, "p2_detay", lambda: {"A": {"score": 8}})  # verdict yok
        secilen, _ = p5_committee.komite_adaylari()
        assert _skor(secilen, "A") == 60
