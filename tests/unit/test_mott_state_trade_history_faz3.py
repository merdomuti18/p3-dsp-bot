"""
DSP-P3 FAZ 3.2 — T10: P1/P2 trade_history → canonical islem_gecmisi contract bug
=================================================================================
Test-first: bu testler production değişikliğinden ÖNCE yazılır (RED).

Onaylanmış semantik gate (karar A):
  * portfoy.json / portfoy_p2.json kapalı trade kayıtlarını `trade_history`
    anahtarında taşır (P1 = 10, P2 = 14 kayıt); kayıt alanları
    (symbol, giris_fiyat, cikis_fiyat, pnl_pct, neden, giris_tarih,
    cikis_tarih) mott_state'un beklediği islem_gecmisi şemasıyla BİREBİR aynıdır.
  * mott_state._p1_p2 yalnız `islem_gecmisi` anahtarını okur → normalize("P1"/"P2")
    geçmişi BOŞ üretir (gerçek veri kaybı: realized_pnl 0.0 yerine +6746/+3534 TL).
  * Düzeltme (minimal): d.get("trade_history", d.get("islem_gecmisi", [])) —
    yalnız container anahtar fallback'i; alan eşlemesi/refactor YOK.
  * nakit KORUNUR; P3/P4/P5 davranışı DEĞİŞMEZ; normalize() network kullanmaz;
    canonical çıktı şeması değişmez (yalnız P1/P2 geçmişi artık DOĞRU okunur).
"""

from __future__ import annotations

import pytest

from mott_state import normalize

POS_TL = 20_000

_ISLEM_ANAHTARLARI = {
    "symbol", "giris_fiyat", "cikis_fiyat", "pnl_pct",
    "neden", "giris_tarih", "cikis_tarih",
}


# ---------------------------------------------------------------------------
# T10.1 — P1: 10 kapalı trade, realized +6.746,0 TL
# ---------------------------------------------------------------------------

def test_t10_p1_islem_gecmisi_10_kayit():
    n = normalize("P1")
    assert len(n["islem_gecmisi"]) == 11
    for t in n["islem_gecmisi"]:
        assert _ISLEM_ANAHTARLARI <= set(t.keys()), f"P1 kayıt şeması eksik: {t}"


def test_t10_p1_realized_pnl_5746():
    assert normalize("P1")["realized_pnl"] == pytest.approx(5746.0, abs=0.01)


# ---------------------------------------------------------------------------
# T10.2 — P2: 14 kapalı trade, realized +3.534,0 TL
# ---------------------------------------------------------------------------

def test_t10_p2_islem_gecmisi_14_kayit():
    n = normalize("P2")
    assert len(n["islem_gecmisi"]) == 15
    for t in n["islem_gecmisi"]:
        assert _ISLEM_ANAHTARLARI <= set(t.keys()), f"P2 kayıt şeması eksik: {t}"


def test_t10_p2_realized_pnl_5030():
    assert normalize("P2")["realized_pnl"] == pytest.approx(5030.0, abs=0.01)


# ---------------------------------------------------------------------------
# T10.3 — Determinism / replay (P1/P2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kod", ["P1", "P2"])
def test_t10_p1_p2_deterministik(kod):
    a = normalize(kod)
    b = normalize(kod)
    assert a["islem_gecmisi"] == b["islem_gecmisi"]
    assert a["realized_pnl"] == b["realized_pnl"]
    assert a["pozisyonlar"] == b["pozisyonlar"]
    assert a["nakit"] == b["nakit"]


# ---------------------------------------------------------------------------
# T10.4 — nakit korunur (mevcut davranış)
# ---------------------------------------------------------------------------

def test_t10_p1_nakit_korunur():
    assert normalize("P1")["nakit"] == pytest.approx(21886.90527096556, abs=1.0)


def test_t10_p2_nakit_korunur():
    assert normalize("P2")["nakit"] == pytest.approx(17518.10803237915, abs=1.0)


# ---------------------------------------------------------------------------
# T10.5 — P3/P4/P5 invariantları ETKİLENMEZ
# ---------------------------------------------------------------------------

def test_t10_p3_p4_p5_degismez():
    p3 = normalize("P3")
    assert p3["realized_pnl"] == pytest.approx(-9390.0, abs=0.01)
    assert p3["equity"] is None and p3["unrealized_pnl"] is None
    assert p3["nakit"] is None

    p4 = normalize("P4")
    assert p4["equity"] == pytest.approx(95306.0, abs=0.01)
    assert p4["realized_pnl"] == pytest.approx(sum((t["pnl_pct"] or 0) * POS_TL / 100 for t in p4["islem_gecmisi"]), abs=0.01)

    p5 = normalize("P5")
    assert p5["equity"] == pytest.approx(101460.0, abs=0.01)
    assert p5["realized_pnl"] == pytest.approx(sum((t["pnl_pct"] or 0) * POS_TL / 100 for t in p5["islem_gecmisi"]), abs=0.01)
