"""
DSP-P3 FAZ 4 — Paket 3 / B6: P3 gerçek skor (hardcode 0.0 kaldırma)
==================================================================
meta_portfolio.p3_sinyalleri_yukle, P3 sinyallerini üretirken skoru:

  1. kanonik persisted JSONL'den (Paket 2: {symbol, score, tarih})
  2. yoksa mevcut 0.0 fallback'i

ile doldurmalıdır. Yeni skor hesaplaması YOK — sc.score zaten üretim skoru.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

import meta_portfolio


def test_p3_sinyalleri_yukle_persisted_score(monkeypatch, tmp_path):
    sabit_bugun = date(2026, 8, 18)
    monkeypatch.setattr(meta_portfolio, "bugun_tsi", lambda: sabit_bugun)
    monkeypatch.setattr(meta_portfolio, "BASE_DIR", tmp_path)

    # Kanonik persisted JSONL (Paket 2 formatı)
    yol = tmp_path / "scan_history_p3.jsonl"
    yol.write_text(
        json.dumps({"symbol": "TEST", "score": 7.25, "tarih": "2026-08-18"}) + "\n" +
        json.dumps({"symbol": "DIGER", "score": 3.5, "tarih": "2026-08-18"}) + "\n",
        encoding="utf-8",
    )
    # Bugünkü scan_log fixture
    state = {"scan_log": [{
        "date": "2026-08-18", "scanned": 10, "long_signals": 2,
        "top5": ["TEST", "DIGER"], "entries": [], "exits": [],
    }]}
    (tmp_path / "portfolio_state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8",
    )

    sinyaller, _ = meta_portfolio.p3_sinyalleri_yukle()
    assert len(sinyaller) == 2
    skorlar = {s["symbol"]: s["score"] for s in sinyaller}
    assert skorlar["TEST"] == pytest.approx(7.25)
    assert skorlar["DIGER"] == pytest.approx(3.5)
    # Hardcode 0.0 üretilmedi
    assert all(s["score"] != 0.0 for s in sinyaller)


def test_p3_sinyalleri_yukle_skor_yoksa_fallback_sifir(monkeypatch, tmp_path):
    sabit_bugun = date(2026, 8, 18)
    monkeypatch.setattr(meta_portfolio, "bugun_tsi", lambda: sabit_bugun)
    monkeypatch.setattr(meta_portfolio, "BASE_DIR", tmp_path)

    # JSONL yok / başka sembol → fallback 0.0 (mevcut davranış korunur)
    state = {"scan_log": [{
        "date": "2026-08-18", "scanned": 10, "long_signals": 1,
        "top5": ["YOK"], "entries": [], "exits": [],
    }]}
    (tmp_path / "portfolio_state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8",
    )

    sinyaller, _ = meta_portfolio.p3_sinyalleri_yukle()
    assert len(sinyaller) == 1
    assert sinyaller[0]["symbol"] == "YOK"
    assert sinyaller[0]["score"] == 0.0


def test_p3_sinyalleri_yukle_bayat_scan_bos(monkeypatch, tmp_path):
    sabit_bugun = date(2026, 8, 18)
    monkeypatch.setattr(meta_portfolio, "bugun_tsi", lambda: sabit_bugun)
    monkeypatch.setattr(meta_portfolio, "BASE_DIR", tmp_path)

    state = {"scan_log": [{
        "date": "2026-08-10", "scanned": 10, "long_signals": 1,
        "top5": ["ESKI"], "entries": [], "exits": [],
    }]}
    (tmp_path / "portfolio_state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8",
    )

    sinyaller, _ = meta_portfolio.p3_sinyalleri_yukle()
    assert sinyaller == []
