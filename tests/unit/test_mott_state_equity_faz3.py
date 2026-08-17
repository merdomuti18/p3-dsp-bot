"""
DSP-P3 FAZ 3.2 — T5..T9: Paper equity contract testleri
=========================================================
Canonical state sözleşmesi (FAZ 3.2 — onaylı tasarım):

  * baslangic_sermayesi = 100_000 (5 strateji — paper modeli)
  * realized_pnl  = Σ(pnl_pct × POS_TL / 100), POS_TL = 20_000 — DETERMINISTİK
  * unrealized_pnl: P4/P5 → state pozisyon pnl'lerinden deterministik;
                    P1/P2/P3 → None (state'te açık pozisyon fiyatı yok)
  * equity: P4/P5 = sermaye_mevcut (mevcut sözleşme KORUNUR);
            P1/P2/P3 = None — anlamı: "current equity deterministik state'ten
            hesaplanamıyor; canlı fiyat enrichment'ı gerekir" — "sermaye yok" DEĞİL.
  * normalize() asla yfinance/network kullanmaz (canlı fiyat raporlama katmanında).

Test-first: bu testler production değişikliğinden ÖNCE yazılır ve RED olur.
"""

from __future__ import annotations

import pytest

from mott_state import normalize

POS_TL = 20_000
BASLANGIC = 100_000


# ---------------------------------------------------------------------------
# T5 — baslangic_sermayesi (paper invariant)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kod", ["P1", "P2", "P3", "P4", "P5"])
def test_t5_baslangic_sermayesi_100000(kod):
    assert normalize(kod)["baslangic_sermayesi"] == BASLANGIC


# ---------------------------------------------------------------------------
# T6 — realized_pnl: determinizm + formül
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kod", ["P1", "P2", "P3", "P4", "P5"])
def test_t6_realized_pnl_deterministik_ve_formul(kod):
    n = normalize(kod)
    # Formül çapraz kontrolü: islem_gecmisi pnl'lerinden bağımsız hesap
    beklenen = sum((t["pnl_pct"] or 0) * POS_TL / 100 for t in n["islem_gecmisi"])
    assert isinstance(n["realized_pnl"], (int, float))
    assert n["realized_pnl"] == pytest.approx(beklenen, abs=0.01)
    # Determinizm: iki çağrı aynı değer
    assert n["realized_pnl"] == normalize(kod)["realized_pnl"]


def test_t6_p3_realized_pnl_gercek_deger():
    """P3 gerçek değer: 16 kapalı trade, pnl_pct toplamı −46.95 pp → −9.390 TL."""
    assert normalize("P3")["realized_pnl"] == pytest.approx(-9390.0, abs=0.01)


# ---------------------------------------------------------------------------
# T7 — unrealized_pnl contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kod", ["P1", "P2", "P3"])
def test_t7_unrealized_pnl_none_p1_p2_p3(kod):
    """State'te açık pozisyon fiyatı yok → None (canlı fiyat yasağı)."""
    assert normalize(kod)["unrealized_pnl"] is None


@pytest.mark.parametrize("kod", ["P4", "P5"])
def test_t7_unrealized_pnl_deterministik_p4_p5(kod):
    """P4/P5: state pozisyon pnl'lerinden deterministik hesaplanır (canlı yok)."""
    n = normalize(kod)
    beklenen = sum((p["pnl_pct"] or 0) * POS_TL / 100 for p in n["pozisyonlar"])
    assert n["unrealized_pnl"] is not None
    assert n["unrealized_pnl"] == pytest.approx(beklenen, abs=0.01)
    assert n["unrealized_pnl"] == normalize(kod)["unrealized_pnl"]


# ---------------------------------------------------------------------------
# T8 — Enrichment ayrımı: current_equity formülü + T4 golden koruması
# ---------------------------------------------------------------------------

def test_t8_enrichment_ayri_katman_current_equity():
    """normalize canlı fiyat çekmez (unrealized=None); raporlama katmanında
    mock fiyatla current_equity = baslangic + realized + unrealized.
    Mock 100.0 → P3 current_equity = 274.000 (%174) — T4 golden ile tutarlı."""
    n = normalize("P3")
    assert n["unrealized_pnl"] is None  # normalize network kullanmadı

    # Raporlama katmanı enrichment (mock fiyat 100.0 — ayrı adım).
    # pnl_pct = (cp - ep)/ep × 100 ve tl = POS_TL × pnl_pct / 100 → POS_TL × (cp - ep)/ep
    unrealized = sum(
        POS_TL * (100.0 - p["giris_fiyat"]) / p["giris_fiyat"]
        for p in n["pozisyonlar"]
    )
    current_equity = n["baslangic_sermayesi"] + n["realized_pnl"] + unrealized
    getiri = (current_equity - BASLANGIC) / BASLANGIC * 100

    assert current_equity == pytest.approx(274000.0, abs=1.0)   # T4 golden
    assert getiri == pytest.approx(174.0, abs=0.1)              # T4 golden
    # Deterministik çekirdek: baslangic + realized (açık pozisyonlar hariç)
    assert n["baslangic_sermayesi"] + n["realized_pnl"] == pytest.approx(90610.0, abs=0.01)


# ---------------------------------------------------------------------------
# T9 — "None != sermaye yok" (semantic regression guard)
# ---------------------------------------------------------------------------

def test_t9_p3_none_sermaye_yok_degildir():
    """P3'te equity=None AMA baslangic_sermayesi=100000 VE realized_pnl sayısal.
    Bu, 'None = current equity hesaplanamadı' anlamının kalıcı kilididir."""
    n = normalize("P3")
    assert n["equity"] is None
    assert n["baslangic_sermayesi"] == BASLANGIC
    assert isinstance(n["realized_pnl"], (int, float))
    assert n["unrealized_pnl"] is None
