"""
DSP-P3 FAZ 4 — Paket 3 / B5: P1 kanonik score
=============================================
scanner_p1._build_signal_records kayıtlarında:

  score == score_count

(mevcut strateji sayısı — yeni hesaplama YOK; yalnız taşıma).

Ayrıca üretilen kaydın Paket 2 kanonik JSONL hattına uyduğu doğrulanır:
_persisted_sinyal_logu("P1") → {symbol, score, tarih} okur.
"""

from __future__ import annotations

import json

import pytest

import meta_portfolio


def _ind():
    """_build_signal_records'un kullandığı tüm indikatör alanları."""
    return {
        "rsi": 60.0, "rel_vol": 1.1, "change_pct": 0.5, "close": 100.0,
        "adx": 30.0, "cmf": 0.2, "alpha_bull": True, "alpha_trend_bull": False,
    }


def test_p1_score_score_count_esit():
    import scanner_p1
    results = {
        "GT":  [{"symbol": "GARAN", "ind": _ind()}],
        "ZKN": [{"symbol": "GARAN", "ind": _ind()},
                {"symbol": "THYAO", "ind": _ind()}],
        "ALPHA": [{"symbol": "THYAO", "ind": _ind()}],
    }
    records = scanner_p1._build_signal_records("14.08.2026 20:02", "akşam", results)
    skorlar = {r["symbol"]: r for r in records}
    assert set(skorlar) == {"GARAN", "THYAO"}
    # GARAN: GT+ZKN → 2 strateji; THYAO: ZKN+ALPHA → 2 strateji
    for sym in ("GARAN", "THYAO"):
        assert skorlar[sym]["score"] == skorlar[sym]["score_count"] == 2


def test_p1_score_jsonl_hattina_uyar(monkeypatch, tmp_path):
    import scanner_p1
    monkeypatch.setattr(meta_portfolio, "BASE_DIR", tmp_path)
    monkeypatch.setattr(scanner_p1, "SIGNAL_LOG", tmp_path / "scan_history_p1.jsonl")
    results = {
        "GT": [{"symbol": "GARAN", "ind": _ind()}],
        "ZKN": [{"symbol": "GARAN", "ind": _ind()},
                {"symbol": "THYAO", "ind": _ind()}],
    }
    records = scanner_p1._build_signal_records("14.08.2026 20:02", "akşam", results)
    scanner_p1._append_signal_log(records)

    kayitlar = meta_portfolio._persisted_sinyal_logu("P1")
    assert len(kayitlar) == 2
    by_sym = {k["symbol"]: k for k in kayitlar}
    assert by_sym["GARAN"]["score"] == pytest.approx(2.0)
    assert by_sym["THYAO"]["score"] == pytest.approx(1.0)
    assert by_sym["GARAN"]["tarih"] == "2026-08-14"
    assert set(by_sym["GARAN"].keys()) == {"symbol", "score", "tarih"}
