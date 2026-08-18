"""
DSP-P3 FAZ 4 — Paket 2: Sinyal geçmişi kalıcılığı (B4, Seçenek B)
=================================================================
Repo-içi JSONL persistence sözleşmesi:

  * P1/P2/P3 sinyal geçmişi /tmp'ye değil repo-içi JSONL dosyalarına yazılır.
  * P4 okuyucusu (meta_portfolio._persisted_sinyal_logu) aynı kanonik
    formatı okur: {symbol, score, tarih}.
  * Dosya yoksa [] döner — çağıran mevcut state fallback'ini kullanır
    (persisted log önceliklidir; state fallback yalnız log yokken).
  * State schema DEĞİŞMEZ (9/9 korunur).

Network YOK: writer'lar doğrudan tmp_path'e yazar; hiçbir canlı veri çağrılmaz.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import meta_portfolio


def _tmp_base(monkeypatch, tmp_path):
    """meta_portfolio.BASE_DIR'i izole tmp_path'e yönlendir — repo'ya YAZMA."""
    monkeypatch.setattr(meta_portfolio, "BASE_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# P1 / P2 / P3 JSONL append → P4 read (writer → reader entegrasyonu)
# ---------------------------------------------------------------------------

def test_p1_jsonl_append_read(monkeypatch, tmp_path):
    import scanner_p1
    base = _tmp_base(monkeypatch, tmp_path)
    monkeypatch.setattr(scanner_p1, "SIGNAL_LOG", base / "scan_history_p1.jsonl")
    scanner_p1._append_signal_log([
        {"symbol": "GARAN", "scan_time": "14.08.2026 20:02", "scan_label": "akşam",
         "strategies": ["GT"], "score_count": 1},
        {"symbol": "THYAO", "scan_time": "14.08.2026 20:02", "scan_label": "akşam",
         "strategies": ["GT", "ZKN"], "score_count": 2},
    ])
    kayitlar = meta_portfolio._persisted_sinyal_logu("P1")
    assert len(kayitlar) == 2
    assert kayitlar[0]["symbol"] == "GARAN"
    assert kayitlar[0]["tarih"] == "2026-08-14"
    assert kayitlar[1]["symbol"] == "THYAO"


def test_p2_jsonl_append_read(monkeypatch, tmp_path):
    import scanner_smc
    base = _tmp_base(monkeypatch, tmp_path)
    monkeypatch.setattr(scanner_smc, "SIGNAL_LOG", base / "scan_smc_history.jsonl")
    scanner_smc._append_signal_log([
        {"symbol": "SKTAS", "scan_time": "14.08.2026 20:02", "score": 7.5, "verdict": "GÜÇLÜ AL"},
        {"symbol": "BFREN", "scan_time": "14.08.2026 20:02", "score": 6.0, "verdict": "AL"},
    ])
    kayitlar = meta_portfolio._persisted_sinyal_logu("P2")
    assert len(kayitlar) == 2
    assert kayitlar[0]["score"] == pytest.approx(7.5)
    assert kayitlar[1]["score"] == pytest.approx(6.0)
    assert kayitlar[0]["tarih"] == "2026-08-14"


def test_p3_jsonl_append_read(monkeypatch, tmp_path):
    import simulate_dsp
    base = _tmp_base(monkeypatch, tmp_path)
    monkeypatch.setattr(simulate_dsp, "SINYAL_LOG", base / "scan_history_p3.jsonl")
    simulate_dsp._append_sinyal_log([
        SimpleNamespace(symbol="TKFEN", score=0.6123),
        SimpleNamespace(symbol="ARDYZ", score=0.45),
    ])
    kayitlar = meta_portfolio._persisted_sinyal_logu("P3")
    assert len(kayitlar) == 2
    assert kayitlar[0]["symbol"] == "TKFEN"
    assert kayitlar[0]["score"] == pytest.approx(0.6123)
    assert kayitlar[1]["symbol"] == "ARDYZ"


# ---------------------------------------------------------------------------
# Kanonik kayıt şeması + tarih + skor okuma
# ---------------------------------------------------------------------------

def test_kanonik_kayit_semasi(monkeypatch, tmp_path):
    base = _tmp_base(monkeypatch, tmp_path)
    yol = base / "scan_history_p1.jsonl"
    yol.write_text(
        json.dumps({"symbol": "GARAN", "score": 3.0, "tarih": "2026-08-14"}) + "\n",
        encoding="utf-8",
    )
    kayitlar = meta_portfolio._persisted_sinyal_logu("P1")
    assert len(kayitlar) == 1
    assert set(kayitlar[0].keys()) == {"symbol", "score", "tarih"}


def test_tarih_alani_iso_ve_scan_time(monkeypatch, tmp_path):
    base = _tmp_base(monkeypatch, tmp_path)
    yol = base / "scan_smc_history.jsonl"
    yol.write_text(
        json.dumps({"symbol": "A", "score": 5.0, "tarih": "2026-08-14"}) + "\n" +
        json.dumps({"symbol": "B", "score": 6.0, "scan_time": "13.08.2026 20:02"}) + "\n",
        encoding="utf-8",
    )
    kayitlar = meta_portfolio._persisted_sinyal_logu("P2")
    assert kayitlar[0]["tarih"] == "2026-08-14"
    assert kayitlar[1]["tarih"] == "2026-08-13"


def test_score_fallback_final_score(monkeypatch, tmp_path):
    base = _tmp_base(monkeypatch, tmp_path)
    yol = base / "scan_history_p1.jsonl"
    yol.write_text(
        json.dumps({"symbol": "A", "scan_time": "2026-08-14 20:00"}) + "\n" +
        json.dumps({"symbol": "B", "scan_time": "2026-08-14 20:00", "final_score": 4.0}) + "\n",
        encoding="utf-8",
    )
    kayitlar = meta_portfolio._persisted_sinyal_logu("P1")
    # P1'de score yok → 0.0 (Paket 3 B5'te eklenecek); final_score fallback çalışır
    assert kayitlar[0]["score"] == 0.0
    assert kayitlar[1]["score"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Persisted log önceliği / fallback / izolasyon / determinizm
# ---------------------------------------------------------------------------

def test_p4_okuyucu_persisted_log_kullanir(monkeypatch, tmp_path):
    # JSONL doluysa okuyucu persisted log'u döndürür — state fallback'e düşmez
    base = _tmp_base(monkeypatch, tmp_path)
    yol = base / "scan_history_p3.jsonl"
    for i in range(15):
        with yol.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"symbol": f"S{i:02d}", "score": 0.5, "tarih": "2026-08-14"}
            ) + "\n")
    kayitlar = meta_portfolio._persisted_sinyal_logu("P3")
    assert len(kayitlar) == 15
    assert kayitlar[0]["symbol"] == "S00"


def test_dosya_yoksa_bos_fallback(monkeypatch, tmp_path):
    # Log yoksa [] → çağıran mevcut state fallback'ini kullanır
    base = _tmp_base(monkeypatch, tmp_path)
    assert meta_portfolio._persisted_sinyal_logu("P1") == []
    assert meta_portfolio._persisted_sinyal_logu("P2") == []
    assert meta_portfolio._persisted_sinyal_logu("P3") == []


def test_sinyal_log_yolu_tmp_basedir(monkeypatch, tmp_path):
    base = _tmp_base(monkeypatch, tmp_path)
    assert meta_portfolio._sinyal_log_yolu("P1") == base / "scan_history_p1.jsonl"
    assert meta_portfolio._sinyal_log_yolu("P2") == base / "scan_smc_history.jsonl"
    assert meta_portfolio._sinyal_log_yolu("P3") == base / "scan_history_p3.jsonl"


def test_bozuk_satir_atlanir(monkeypatch, tmp_path):
    base = _tmp_base(monkeypatch, tmp_path)
    yol = base / "scan_history_p1.jsonl"
    yol.write_text(
        "bu-json-degil\n" +
        json.dumps({"symbol": "GARAN", "score": 1.0, "tarih": "2026-08-14"}) + "\n",
        encoding="utf-8",
    )
    kayitlar = meta_portfolio._persisted_sinyal_logu("P1")
    assert len(kayitlar) == 1
    assert kayitlar[0]["symbol"] == "GARAN"


def test_deterministik_tekrar(monkeypatch, tmp_path):
    base = _tmp_base(monkeypatch, tmp_path)
    yol = base / "scan_history_p1.jsonl"
    for i in range(20):
        with yol.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"symbol": f"S{i:02d}", "score": 1.0, "tarih": "2026-08-14"}
            ) + "\n")
    a = meta_portfolio._persisted_sinyal_logu("P1")
    b = meta_portfolio._persisted_sinyal_logu("P1")
    assert a == b
    assert len(a) == 20


def test_sinir_son_kayitlar(monkeypatch, tmp_path):
    base = _tmp_base(monkeypatch, tmp_path)
    yol = base / "scan_smc_history.jsonl"
    for i in range(25):
        with yol.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"symbol": f"S{i:02d}", "score": 1.0, "tarih": "2026-08-14"}
            ) + "\n")
    kayitlar = meta_portfolio._persisted_sinyal_logu("P2", sinir=10)
    assert len(kayitlar) == 10
    assert kayitlar[0]["symbol"] == "S15"
