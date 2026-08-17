"""
DSP-P3 FAZ 1 — meta_portfolio (P4) unit testleri
================================================
Yalnızca SAF fonksiyonlar test edilir:
  compute_ic, normalize_skor, ic_agirlikli_birlestir, half_kelly_boyut,
  _parse_tarama_tarih.
Network gerektiren (fiyat_cek) ve state YAZAN (state_kaydet, calistir)
fonksiyonlar BURADA çağrılmaz.
"""

from __future__ import annotations

import numpy as np
import pytest

from meta_portfolio import (
    MAX_POS,
    SERMAYE,
    _parse_tarama_tarih,
    compute_ic,
    half_kelly_boyut,
    ic_agirlikli_birlestir,
    normalize_skor,
)


# ---------------------------------------------------------------------------
# compute_ic — Spearman IC
# ---------------------------------------------------------------------------

def test_compute_ic_yetersiz_veri_sifir():
    assert compute_ic([1.0, 2.0], [1.0, 2.0]) == 0.0  # 10'dan az → 0.0


def test_compute_ic_mukemmel_pozitif():
    sinyaller = list(range(1, 21))
    getiriler = list(range(1, 21))
    assert compute_ic(sinyaller, getiriler) == pytest.approx(1.0)


def test_compute_ic_mukemmel_negatif():
    sinyaller = list(range(1, 21))
    getiriler = list(range(20, 0, -1))
    assert compute_ic(sinyaller, getiriler) == pytest.approx(-1.0)


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_compute_ic_sabit_seri_nan_guard():
    # Varyans 0 → spearman NaN → fonksiyon 0.0 dönmeli (NaN değil)
    # (spearmanr sabit girdide ConstantInputWarning verir — production davranışı)
    sabit = [1.0] * 20
    ic = compute_ic(sabit, list(range(1, 21)))
    assert ic == 0.0
    assert not np.isnan(ic)


def test_compute_ic_farkli_uzunluklar():
    # n = min(len) ile hizalama; 10+ veride çalışır
    ic = compute_ic(list(range(1, 15)), list(range(1, 12)))
    assert -1.0 <= ic <= 1.0


# ---------------------------------------------------------------------------
# normalize_skor — min-max normalizasyon
# ---------------------------------------------------------------------------

def test_normalize_skor_bos():
    assert normalize_skor([], "P1") == []


def test_normalize_skor_tek_deger_1():
    out = normalize_skor([{"symbol": "GARAN", "score": 5.0}], "P1")
    assert out[0]["norm_score"] == 1.0
    assert out[0]["strateji"] == "P1"


def test_normalize_skor_min_max():
    sinyaller = [
        {"symbol": "A", "score": 2.0},
        {"symbol": "B", "score": 6.0},
        {"symbol": "C", "score": 4.0},
    ]
    out = normalize_skor(sinyaller, "P2")
    skorlar = {s["symbol"]: s["norm_score"] for s in out}
    assert skorlar["A"] == pytest.approx(0.0)   # min → 0
    assert skorlar["B"] == pytest.approx(1.0)   # max → 1
    assert skorlar["C"] == pytest.approx(0.5)   # orta → 0.5


# ---------------------------------------------------------------------------
# ic_agirlikli_birlestir — IC ağırlıklı birleştirme
# ---------------------------------------------------------------------------

def test_ic_birlestir_tum_ic_sifir_esit_agirlik():
    p1 = [{"symbol": "GARAN", "score": 5.0}]
    p2 = [{"symbol": "THYAO", "score": 8.0}]
    p3 = [{"symbol": "AKBNK", "score": 0.03}]
    out = ic_agirlikli_birlestir(p1, p2, p3, 0.0, 0.0, 0.0)
    assert len(out) == 3
    # Eşit ağırlık → her strateji 1/3; en yüksek norm skor 1.0 → meta 1/3
    for s in out:
        assert 0.0 <= s["meta_score"] <= 1.0
    en_iyi = out[0]
    assert en_iyi["meta_score"] == pytest.approx(1.0 / 3)


def test_ic_birlestir_tek_guclu_ic_baskin():
    p1 = [{"symbol": "GARAN", "score": 5.0}, {"symbol": "AKBNK", "score": 3.0}]
    out = ic_agirlikli_birlestir(p1, [], [], ic_p1=0.2, ic_p2=0.0, ic_p3=0.0)
    assert len(out) == 2
    assert out[0]["symbol"] == "GARAN"  # en yüksek skor
    assert out[0]["meta_score"] > out[1]["meta_score"]
    assert "P1" in out[0]["kaynaklar"]


def test_ic_birlestir_coklu_onay_bonus():
    p1 = [{"symbol": "GARAN", "score": 5.0}]
    p2 = [{"symbol": "GARAN", "score": 8.0}]
    p3 = []
    out = ic_agirlikli_birlestir(p1, p2, p3, ic_p1=0.1, ic_p2=0.1, ic_p3=0.0)
    assert len(out) == 1
    assert out[0]["symbol"] == "GARAN"
    assert out[0]["coklu_onay"] is True
    assert set(out[0]["kaynaklar"]) == {"P1", "P2"}
    # Bonus: 2 kaynak → meta_score * 1.1
    # norm GARAN(P1)=1.0*w1 + GARAN(P2)=1.0*w2 (w1=w2=0.5) → 1.0 → *1.1 = 1.1
    assert out[0]["meta_score"] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# half_kelly_boyut
# ---------------------------------------------------------------------------

def test_half_kelly_gecersiz_girdi_sermaye_dagilimi():
    assert half_kelly_boyut(0.6, 0.1, 0.0, SERMAYE) == SERMAYE / MAX_POS
    assert half_kelly_boyut(0.6, 0.0, 0.05, SERMAYE) == SERMAYE / MAX_POS


def test_half_kelly_normal():
    # b = 0.10/0.05 = 2; kelly = (0.6*2 - 0.4)/2 = 0.4 → clip 0.25 → half 0.125
    boyut = half_kelly_boyut(0.6, 0.10, 0.05, SERMAYE)
    assert boyut == pytest.approx(SERMAYE * 0.125)


def test_half_kelly_negatif_kelly_sifira_iner():
    # Kazanma oranı düşükse kelly ≤ 0 → max(0, ...) → 0 TL
    boyut = half_kelly_boyut(0.2, 0.05, 0.10, SERMAYE)
    assert boyut == 0.0


# ---------------------------------------------------------------------------
# _parse_tarama_tarih — tarih formatı toleransı (DD.MM.YYYY + ISO)
# ---------------------------------------------------------------------------

def test_parse_tarama_tarih_ddmmyyyy():
    assert _parse_tarama_tarih({"last_scan": "14.08.2026 20:02"}).isoformat() == "2026-08-14"


def test_parse_tarama_tarih_iso():
    assert _parse_tarama_tarih({"tarama": {"scan_time": "2026-08-14T20:02:00"}}).isoformat() == "2026-08-14"


def test_parse_tarama_tarih_bozuk_none():
    assert _parse_tarama_tarih({"last_scan": "bozuk-veri"}) is None
    assert _parse_tarama_tarih({}) is None
