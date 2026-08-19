"""
DSP-P3 FAZ 1 — Look-ahead koruma otomasyonu
===========================================
P0-2 bulgusu: `check_no_lookahead` production'da hiçbir yerden çağrılmıyordu.
Bu test, dsp_core.check_no_lookahead'i 17 kritik modülün TÜMÜNE uygular:
  * filtfilt import'u içeren modül → AssertionError → test FAIL
  * yorum/string içinde 'filtfilt' geçmesi sorun değil (yalnızca AST import)

SADECE dosya okur; hiçbir modül import edilmez (import yan etkisi yok).
"""

from __future__ import annotations

import pytest

from conftest import REPO_ROOT  # type: ignore
from dsp_core import check_no_lookahead

_18_MODUL = [
    "dsp_core.py", "dsp_strategies.py", "dsp_trend.py", "scanner_dsp.py",
    "simulate_dsp.py", "meta_portfolio.py", "mott_state.py", "mott_risk.py",
    "mott_telegram.py", "mott_fiyat.py", "mott_portfoy_deger.py",
    "mott_performans_analiz.py", "mott_aylik_rapor.py", "p5_committee.py",
    "portfoy_yonetici.py", "scanner_p1.py", "scanner_smc.py",
    "mott_state_coordination.py",
]


@pytest.mark.parametrize("modul", _18_MODUL)
def test_modulde_yasakli_import_yok(modul):
    check_no_lookahead(str(REPO_ROOT / modul))
