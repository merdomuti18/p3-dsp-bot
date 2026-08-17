"""
DSP-P3 FAZ 3.1 — mott_state aktivasyon testleri (T1/T2/T3)
==========================================================
Test-first: bu testler ÖNCE yazılır, production'a dokunulmadan koşulur.

  T1 — Determinism / replay: normalize(kod) aynı state ile iki kez çağrılırsa
       bit-bit aynı çıktıyı üretir (canlı fiyat / network YOK).
  T2 — Equity semantiği: normalize çıktısındaki `equity` alanı yalnız
       P4/P5 için sermaye_mevcut değerini taşır; P1/P2/P3 için None.
       `nakit` alanı mevcut davranışını KORUR (bu fazda karar yok).
  T3 — Dosya haritası tek kaynak: mott_state.DOSYALAR dosya adları,
       mott_risk._KITAP_DOSYALARI ile birebir eşleşir (P1..P5).

SADECE state dosyalarını okur; hiçbir yazma / network yok.
"""

from __future__ import annotations

import json

import pytest

from conftest import REPO_ROOT  # type: ignore
from mott_state import DOSYALAR, normalize


def _state_json(fname: str) -> dict:
    return json.loads((REPO_ROOT / fname).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# T1 — Determinism / replay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kod", ["P1", "P2", "P3", "P4", "P5"])
def test_t1_normalize_deterministik_replay(kod):
    """Aynı state × 2 çağrı → bit-bit aynı çıktı (replay edilebilirlik)."""
    a = normalize(kod)
    b = normalize(kod)
    assert a == b
    # Bit-bit eşitlik (anahtar sırasından bağımsız, deterministik serileştirme)
    sa = json.dumps(a, sort_keys=True, ensure_ascii=False)
    sb = json.dumps(b, sort_keys=True, ensure_ascii=False)
    assert sa == sb


# ---------------------------------------------------------------------------
# T2 — Equity semantiği (nakit davranışı KORUNUR)
# ---------------------------------------------------------------------------

def test_t2_equity_p4_p5_sermaye_mevcut():
    """P4/P5: equity == state'teki sermaye_mevcut (equity tahmini)."""
    for kod, fname in (("P4", "state_p4.json"), ("P5", "state_p5.json")):
        beklenen = _state_json(fname).get("sermaye_mevcut")
        assert beklenen is not None, f"{fname} sermaye_mevcut eksik"
        d = normalize(kod)
        assert d["equity"] == beklenen, f"{kod}: equity != sermaye_mevcut"
        # Mevcut davranış korunuyor: nakit alanı hâlâ sermaye_mevcut taşıyor
        # (nakit'i değiştirme kararı bu fazda YOK).
        assert d["nakit"] == beklenen


def test_t2_equity_p1_p2_p3_none():
    """P1/P2/P3: equity alanı None (equity hesabı yalnız P4/P5 için)."""
    for kod in ("P1", "P2", "P3"):
        assert normalize(kod)["equity"] is None, f"{kod}: equity None olmalı"


def test_t2_nakit_davranisi_korunuyor():
    """P1/P2 gerçek nakit; P3 None — FAZ 0 davranışı değişmedi."""
    p1 = _state_json("portfoy.json")
    p2 = _state_json("portfoy_p2.json")
    assert normalize("P1")["nakit"] == p1["nakit"]
    assert normalize("P2")["nakit"] == p2["nakit"]
    assert normalize("P3")["nakit"] is None


# ---------------------------------------------------------------------------
# T3 — Dosya haritası tek kaynak
# ---------------------------------------------------------------------------

def test_t3_dosya_haritasi_tek_kaynak():
    """mott_state.DOSYALAR dosya adları == mott_risk._KITAP_DOSYALARI."""
    from mott_risk import _KITAP_DOSYALARI
    assert set(DOSYALAR) == set(_KITAP_DOSYALARI), "strateji seti uyuşmuyor"
    for kod in DOSYALAR:
        assert DOSYALAR[kod] == _KITAP_DOSYALARI[kod][0], (
            f"{kod}: {DOSYALAR[kod]} != {_KITAP_DOSYALARI[kod][0]}"
        )


def test_t3_dosya_haritasi_bes_strateji():
    assert set(DOSYALAR) == {"P1", "P2", "P3", "P4", "P5"}
