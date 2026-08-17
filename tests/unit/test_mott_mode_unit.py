"""
DSP-P3 FAZ 1 — mott_mode unit testleri
======================================
detect_mode(now) deterministik saat penceresi testleri.
Pencereler (TSİ): sabah 08:45–09:45, alim 10:00–11:20, takip 11:20–16:59,
kapani 17:00–18:15, aksam 18:30–21:30; diğer/hafta sonu → skip.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mott_mode import detect_mode

# 2026-08-17 = Pazartesi (iş günü)
# 2026-08-15 = Cumartesi (hafta sonu)
PT = pytest.param


@pytest.mark.parametrize(
    "now,beklenen",
    [
        (datetime(2026, 8, 17, 9, 0), "sabah"),     # 08:45–09:45
        (datetime(2026, 8, 17, 10, 0), "alim"),     # 10:00 sınır (dahil)
        (datetime(2026, 8, 17, 11, 20), "alim"),    # 11:20 sınır (dahil — alim)
        (datetime(2026, 8, 17, 11, 21), "takip"),   # 11:21 → takip
        (datetime(2026, 8, 17, 13, 0), "takip"),    # 11:20–16:59
        (datetime(2026, 8, 17, 17, 30), "kapani"),  # 17:00–18:15
        (datetime(2026, 8, 17, 19, 0), "aksam"),    # 18:30–21:30
        (datetime(2026, 8, 17, 8, 0), "skip"),      # pencere dışı (08:00)
        (datetime(2026, 8, 17, 23, 0), "skip"),     # pencere dışı (23:00)
        (datetime(2026, 8, 15, 10, 0), "skip"),     # Cumartesi → skip
        (datetime(2026, 8, 16, 19, 0), "skip"),     # Pazar → skip
    ],
)
def test_detect_mode_tsi_pencereleri(now, beklenen):
    assert detect_mode(now) == beklenen


def test_detect_mode_naive_datetime_localize_edilir():
    # naive datetime → TSİ'ye localize edilir; 12:00 IST → takip
    assert detect_mode(datetime(2026, 8, 17, 12, 0)) == "takip"


def test_detect_mode_utc_datetime_tsi_ye_cevrilir():
    # UTC 07:00 → IST 10:00 → alim (UTC+3)
    utc = datetime(2026, 8, 17, 7, 0, tzinfo=__import__("pytz").timezone("UTC"))
    assert detect_mode(utc) == "alim"


def test_detect_mode_varsayilan_gecerli_mod():
    # Varsayılan (now=None) her zaman geçerli bir mod döndürmeli
    assert detect_mode() in {"sabah", "alim", "takip", "kapani", "aksam", "skip"}
