"""
DSP-P3 FAZ 1 — State bütünlük testleri
======================================
9 state dosyası: mevcut mu, JSON okunabiliyor mu, FAZ 0 hash'leri korunuyor
mu, beklenen şema alanları var mı? SADECE okuma.
"""

from __future__ import annotations

import json

import pytest

from conftest import REPO_ROOT, sha256_of  # type: ignore

# Frozen FAZ 0 sabitleri — CI'da baseline olmadan da çalışır.
from baseline_constants import STATE_SHA256 as _BEKLENEN_STATE_HASH


# 9 state dosyası + beklenen üst düzey anahtar yapısı
_STATE_SEMA = {
    "state_p1.json": ({"tarama", "last_scan"}, True),
    "state_p2.json": ({"tarama", "last_scan", "bekleyen_al"}, True),
    "portfolio_state.json": ({"positions", "history", "scan_log"}, True),
    "state_p4.json": ({"pozisyonlar", "trade_history", "ic_history", "sermaye_mevcut"}, True),
    "state_p5.json": ({"pozisyonlar", "trade_history", "komite_log"}, True),
    "portfoy.json": ({"pozisyonlar", "nakit", "trade_history"}, True),
    "portfoy_p2.json": ({"pozisyonlar", "nakit", "trade_history"}, True),
    "tarama_listesi.json": (None, False),     # list veya dict — sadece JSON
    "tarama_listesi_p2.json": (None, False),  # list veya dict — sadece JSON
}


def test_tum_state_dosyalari_mevcut():
    for fname in _STATE_SEMA:
        assert (REPO_ROOT / fname).exists(), f"Eksik state dosyası: {fname}"


def test_tum_state_dosyalari_json():
    for fname in _STATE_SEMA:
        data = json.loads((REPO_ROOT / fname).read_text(encoding="utf-8"))
        assert data is not None


def test_state_hashleri_9_9_eslesir():
    """Diskteki 9 state dosyası FAZ 0 frozen hash sabitleriyle birebir eşleşir."""
    for fname in _STATE_SEMA:
        assert sha256_of(REPO_ROOT / fname) == _BEKLENEN_STATE_HASH[fname], fname


def test_state_sema_alanlari():
    for fname, (anahtarlar, dict_mi) in _STATE_SEMA.items():
        data = json.loads((REPO_ROOT / fname).read_text(encoding="utf-8"))
        if dict_mi:
            assert isinstance(data, dict), fname
            assert anahtarlar.issubset(data.keys()), (
                f"{fname} şeması bozuk: eksik {anahtarlar - set(data.keys())}"
            )


def test_state_signal_log_yok_durumu_belgelendi():
    """P0-1 bulgusu: state'lerde signal_log YOK (mevcut davranışın kilidi)."""
    for fname in ("state_p1.json", "state_p2.json", "portfolio_state.json"):
        data = json.loads((REPO_ROOT / fname).read_text(encoding="utf-8"))
        assert "signal_log" not in data, f"{fname} signal_log içeriyor — FAZ 0'dan farklı!"
