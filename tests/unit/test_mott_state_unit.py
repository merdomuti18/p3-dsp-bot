"""
DSP-P3 FAZ 1 — mott_state unit testleri
=======================================
- _tarih_iso / _sayi: saf dönüşüm fonksiyonları (NaN/None dayanıklılık)
- normalize(kod): repo state'lerinden SADECE okur (MOTT_BASE_DIR=repo kökü),
  hiçbir dosyaya yazmaz.
"""

from __future__ import annotations

import pytest

from mott_state import (
    DOSYALAR,
    _sayi,
    _tarih_iso,
    hepsi,
    normalize,
)

_ORTAK_ANAHTARLAR = {"strateji", "dosya", "pozisyonlar", "islem_gecmisi", "nakit", "equity"}
_POZ_ANAHTARLAR = {"symbol", "giris_fiyat", "guncel_fiyat", "pnl_pct", "giris_tarih", "lot"}


# ---------------------------------------------------------------------------
# Saf dönüşümler
# ---------------------------------------------------------------------------

def test_tarih_iso_ddmmyyyy_saatli():
    assert _tarih_iso("03.07.2026 13:23") == "2026-07-03"


def test_tarih_iso_zaten_iso():
    assert _tarih_iso("2026-07-03") == "2026-07-03"


def test_tarih_iso_bos():
    assert _tarih_iso("") == ""
    assert _tarih_iso(None) == ""


def test_sayi_nan_none_dayaniklilik():
    assert _sayi("12.5") == 12.5
    assert _sayi(None, varsayilan=-1) == -1
    assert _sayi(float("nan"), varsayilan=7) == 7  # NaN → varsayılan
    assert _sayi("abc", varsayilan=3) == 3


# ---------------------------------------------------------------------------
# normalize — ortak şema sözleşmesi
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kod", ["P1", "P2", "P3", "P4", "P5"])
def test_normalize_ortak_sema_anahtarlari(kod):
    d = normalize(kod)
    assert set(d.keys()) == _ORTAK_ANAHTARLAR
    assert d["strateji"] == kod
    assert isinstance(d["pozisyonlar"], list)
    assert isinstance(d["islem_gecmisi"], list)


@pytest.mark.parametrize("kod", ["P1", "P2", "P3", "P4", "P5"])
def test_normalize_pozisyon_anahtarlari(kod):
    d = normalize(kod)
    for poz in d["pozisyonlar"]:
        assert set(poz.keys()) == _POZ_ANAHTARLAR, f"{kod} pozisyon şeması bozuk: {poz}"
        assert isinstance(poz["symbol"], str) and poz["symbol"]


def test_normalize_bilinmeyen_kod_valueerror():
    with pytest.raises(ValueError):
        normalize("P9")


def test_normalize_kucuk_harf_kabul():
    assert normalize("p3")["strateji"] == "P3"


def test_hepsi_bes_strateji():
    d = hepsi()
    assert set(d.keys()) == {"P1", "P2", "P3", "P4", "P5"}


def test_dosyalar_haritasi():
    assert DOSYALAR == {
        "P1": "portfoy.json",
        "P2": "portfoy_p2.json",
        "P3": "portfolio_state.json",
        "P4": "state_p4.json",
        "P5": "state_p5.json",
    }
