"""
mott_mode.py — GitHub Actions çalışma modu (TSİ)
=================================================
Tüm zamanlar Europe/Istanbul (TSİ) üzerinden belirlenir.
GitHub Actions gecikmeleri için geniş zaman pencereleri kullanılır.
"""

from __future__ import annotations

import sys
from datetime import datetime

import pytz

IST = pytz.timezone("Europe/Istanbul")

# (dakika, mod) — gün içi sırayla, ilk eşleşen kazanır
_WINDOWS: list[tuple[tuple[int, int], str]] = [
    ((8 * 60 + 45, 9 * 60 + 45), "sabah"),       # 08:45–09:45 TSİ
    ((10 * 60 + 45, 11 * 60 + 20), "alim"),      # 10:45–11:20 TSİ
    ((11 * 60 + 20, 16 * 60 + 45), "takip"),     # 11:20–16:45 TSİ
    ((17 * 60 + 0, 18 * 60 + 15), "kapani"),     # 17:00–18:15 TSİ
    ((18 * 60 + 30, 21 * 60 + 30), "aksam"),     # 18:30–21:30 TSİ
]


def detect_mode(now: datetime | None = None) -> str:
    """TSİ saatine göre çalışma modunu döndür. Hafta sonu → skip."""
    now = now or datetime.now(IST)
    if now.tzinfo is None:
        now = IST.localize(now)
    else:
        now = now.astimezone(IST)

    if now.weekday() >= 5:
        return "skip"

    t = now.hour * 60 + now.minute
    for (lo, hi), mode in _WINDOWS:
        if lo <= t <= hi:
            return mode
    return "skip"


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "auto"
    if arg in ("auto", ""):
        mode = detect_mode()
    else:
        mode = arg
    print(mode)
