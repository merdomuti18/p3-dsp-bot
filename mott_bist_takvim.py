# -*- coding: utf-8 -*-
"""
mott_bist_takvim.py — Borsa İstanbul (BIST) Takvim ve Seans Yönetimi
====================================================================
BIST işlem günleri, resmi tatiller, yarım seanslar ve Europe/Istanbul saat kontrolleri.
"""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta

import pytz

IST = pytz.timezone("Europe/Istanbul")

# ── Resmi Tatiller (Yıl bazında sabitler) ────────────────────────────────────
# BIST resmi tatilleri (tam gün kapalı)
BIST_SABIT_TATILLER = {
    (1, 1): "Yılbaşı",
    (4, 23): "Ulusal Egemenlik ve Çocuk Bayramı",
    (5, 1): "Emek ve Dayanışma Günü",
    (5, 19): "Atatürk'ü Anma, Gençlik ve Spor Bayramı",
    (7, 15): "Demokrasi ve Milli Birlik Günü",
    (8, 30): "Zafer Bayramı",
    (10, 29): "Cumhuriyet Bayramı",
}

# Yarım seans günleri (10:00 - 12:40 işlem, 13:00 sonrası tatil)
BIST_SABIT_YARIM_GUNLER = {
    (10, 28): "Cumhuriyet Bayramı Arefesi",
}

# Değişken dini bayram tatilleri (2026 takvimi)
BIST_DINI_TATILLER_2026 = {
    # Ramazan Bayramı 2026: 19 Mart Arefe (yarım gün), 20-22 Mart bayram
    date(2026, 3, 19): ("Ramazan Bayramı Arefesi", True),
    date(2026, 3, 20): ("Ramazan Bayramı 1. Gün", False),
    # Kurban Bayramı 2026: 26 Mayıs Arefe (yarım gün), 27-30 Mayıs bayram
    date(2026, 5, 26): ("Kurban Bayramı Arefesi", True),
    date(2026, 5, 27): ("Kurban Bayramı 1. Gün", False),
    date(2026, 5, 28): ("Kurban Bayramı 2. Gün", False),
    date(2026, 5, 29): ("Kurban Bayramı 3. Gün", False),
}


def to_tsi(dt: datetime | None = None) -> datetime:
    """Verilen datetime'ı veya şu anı Europe/Istanbul saat dilimine çevirir."""
    if dt is None:
        return datetime.now(IST)
    if dt.tzinfo is None:
        return IST.localize(dt)
    return dt.astimezone(IST)


def is_bist_yarim_gun(d: date) -> bool:
    """Tarihin BIST için yarım seans günü olup olmadığını kontrol eder."""
    if (d.month, d.day) in BIST_SABIT_YARIM_GUNLER:
        return True
    if d in BIST_DINI_TATILLER_2026:
        _, is_half = BIST_DINI_TATILLER_2026[d]
        return is_half
    return False


def is_bist_tatil(d: date) -> bool:
    """Tarihin tam gün BIST tatili olup olmadığını kontrol eder (Hafta sonu HARİÇ)."""
    if (d.month, d.day) in BIST_SABIT_TATILLER:
        return True
    if d in BIST_DINI_TATILLER_2026:
        _, is_half = BIST_DINI_TATILLER_2026[d]
        return not is_half
    return False


def is_bist_islem_gunu(d: date) -> bool:
    """Tarihin BIST işlem günü (hafta içi ve tam gün tatil değil) olup olmadığını kontrol eder."""
    # Hafta sonu: Cumartesi (5), Pazar (6)
    if d.weekday() >= 5:
        return False
    if is_bist_tatil(d):
        return False
    return True


def bist_seans_saatleri(d: date) -> tuple[time, time]:
    """İşlem gününün seans başlangıç ve bitiş saatlerini döndürür."""
    if is_bist_yarim_gun(d):
        return time(10, 0), time(12, 40)
    return time(10, 0), time(18, 0)


def is_bist_seans_acik(dt: datetime | None = None) -> bool:
    """Verilen anda veya şu anda BIST pay piyasası sürekli müzayedesinin açık olup olmadığını kontrol eder."""
    dt_tsi = to_tsi(dt)
    d = dt_tsi.date()
    if not is_bist_islem_gunu(d):
        return False

    t = dt_tsi.time()
    seans_basla, seans_bitir = bist_seans_saatleri(d)
    return seans_basla <= t <= seans_bitir


def son_kapanan_saatlik_mum_zamani(dt: datetime | None = None) -> datetime | None:
    """Verilen an itibariyle en son tamamlanmış (kapanmış) saatlik mum başlangıç zamanını döndürür."""
    dt_tsi = to_tsi(dt)
    d = dt_tsi.date()
    if not is_bist_islem_gunu(d):
        return None

    seans_basla, seans_bitir = bist_seans_saatleri(d)
    t = dt_tsi.time()

    # Seans açılışından önce (10:00 öncesi) bugün için kapanmış mum yoktur
    if t < time(11, 0):
        return None

    # Normal seans saatleri: 10:00-11:00, 11:00-12:00, ..., 17:00-18:00
    if is_bist_yarim_gun(d):
        if t >= time(12, 40):
            # Yarım gün 12:40'ta biter, son saatlik mum 11:00-12:00
            return IST.localize(datetime.combine(d, time(11, 0)))
        elif t >= time(12, 0):
            return IST.localize(datetime.combine(d, time(11, 0)))
        elif t >= time(11, 0):
            return IST.localize(datetime.combine(d, time(10, 0)))
        return None

    # Normal gün
    kapanis_saat = min(dt_tsi.hour, 18)
    if t >= time(18, 0):
        # Seans kapandı, son saatlik mum 17:00-18:00
        return IST.localize(datetime.combine(d, time(17, 0)))

    # Gün içi: örn 14:35 ise son kapanmış saatlik mum 13:00 mumu (13:00 - 14:00)
    son_saat = dt_tsi.hour - 1
    if son_saat < 10:
        return None
    return IST.localize(datetime.combine(d, time(son_saat, 0)))
