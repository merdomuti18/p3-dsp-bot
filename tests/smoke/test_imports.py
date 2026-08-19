"""
DSP-P3 FAZ 1 — Import / Smoke testleri
======================================
18 kritik production modülü import edilebilir mi?

Güvenlik garantileri:
  * NETWORK OFF (conftest autouse): herhangi bir ağ çağrısı anında fail.
  * Import'lar module-level yan etki içermez (FAZ 0 taraması doğruladı:
    yfinance/TV/state yazma çağrıları yalnızca fonksiyon içindedir).
  * Hiçbir fonksiyon ÇAĞRILMAZ — yalnızca import edilir.
"""

from __future__ import annotations

import importlib

import pytest

_18_MODUL = [
    "dsp_core", "dsp_strategies", "dsp_trend", "scanner_dsp", "simulate_dsp",
    "meta_portfolio", "mott_state", "mott_risk", "mott_telegram", "mott_fiyat",
    "mott_portfoy_deger", "mott_performans_analiz", "mott_aylik_rapor",
    "p5_committee", "portfoy_yonetici", "scanner_p1", "scanner_smc",
    "mott_state_coordination",
]


@pytest.mark.parametrize("modul", _18_MODUL)
def test_modul_import_edilebilir(modul):
    importlib.import_module(modul)  # network yok; yalnızca import


def test_tum_moduller_listesi_manifest_ile_ayni():
    """Smoke listesi FAZ 0 frozen sabitlerindeki kritik modül setiyle birebir aynı."""
    from baseline_constants import KRITIK_MODULLER
    assert {f"{m}.py" for m in _18_MODUL} == set(KRITIK_MODULLER)
