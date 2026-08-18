"""
DSP-P3 FAZ 5 — Final temporal / look-ahead safety sertifikasyonu
================================================================
Bu takım, FAZ 5'in temporal invariantlarını otomatik olarak kilitler:

  E) warmup tamamlanmadan sinyal üretilemez
     (generate_xover_signal valid_from / warm-up barları sıfır).
  F) stale (bayat) tarama sinyali kabul edilmez
     (meta_portfolio.sinyaller_taze_filtre).
  G) gelecek bar değiştirilirse geçmiş sinyal DEĞİŞMEZ
     (causal filtre — look-ahead yok).
  H) geçmiş bar değiştirilirse ilgili çıktı DEĞİŞEBİLİR
     (causal filtre geçmişe bağımlıdır).

Halihazırda kapsanan (tekrar edilmez, burada referans):
  * C) fwd_return(t) = log(close[t+period]/close[t+1]) — tests/unit/test_ic_hizalama_faz4.py
  * D) target_date < signal_date değerlendirmesi yapılamaz — aynı dosya
  * check_no_lookahead + 17 modül filtfilt — tests/smoke/test_lookahead_guard.py
  * warmup_length hesabı — tests/unit/test_dsp_core_unit.py::test_warmup_length

Network YOK: hiçbir canlı veri çağrılmaz; fixture'lar deterministiktir.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dsp_core import warmup_length
from dsp_strategies import XoverParams, generate_xover_signal
from meta_portfolio import sinyaller_taze_filtre


# ---------------------------------------------------------------------------
# E) Warmup gate — sinyal üretiminde uygulanışı
# ---------------------------------------------------------------------------

def test_warmup_sinyal_barlari_sifir():
    params = XoverParams(fast_period=15, slow_period=40, order=3)
    rng = np.random.default_rng(7)
    prices = 100.0 + np.cumsum(rng.normal(0, 1.0, 200))

    sig = generate_xover_signal(prices, params)
    vf = warmup_length(params.order, params.slow_period)
    assert sig.valid_from == vf
    # Warm-up barları sinyal olarak kullanılamaz (hepsi 0)
    assert np.all(sig.signal[:vf] == 0.0)


def test_warmup_kisa_veri_sinyal_yok():
    # Warm-up tamamlanmadan (veri < valid_from) hiçbir bar sinyal üretemez
    params = XoverParams(fast_period=15, slow_period=40, order=3)
    vf = warmup_length(params.order, params.slow_period)
    kisa = np.linspace(100.0, 110.0, vf)  # valid_from barlık veri — sınırda

    sig = generate_xover_signal(kisa, params)
    assert sig.valid_from == vf
    assert np.all(sig.signal == 0.0)


# ---------------------------------------------------------------------------
# F) Stale-data gate — bayat tarama kabul edilmez
# ---------------------------------------------------------------------------

def test_stale_bugun_tarama_sinyaller_kabul(monkeypatch):
    sabit_bugun = date(2026, 8, 18)
    monkeypatch.setattr("meta_portfolio.bugun_tsi", lambda: sabit_bugun)
    state = {"tarama": {"scan_time": "18.08.2026 20:02"}, "last_scan": "18.08.2026 20:02"}
    sinyaller = [{"symbol": "GARAN", "score": 5.0}]
    out = sinyaller_taze_filtre(state, sinyaller, "P1")
    assert out == sinyaller


def test_stale_bayat_tarama_sinyaller_red(monkeypatch):
    sabit_bugun = date(2026, 8, 18)
    monkeypatch.setattr("meta_portfolio.bugun_tsi", lambda: sabit_bugun)
    state = {"tarama": {"scan_time": "14.08.2026 20:02"}, "last_scan": "14.08.2026 20:02"}
    sinyaller = [{"symbol": "GARAN", "score": 5.0}]
    out = sinyaller_taze_filtre(state, sinyaller, "P1")
    assert out == []


def test_stale_tarihsiz_state_sinyaller_red(monkeypatch):
    sabit_bugun = date(2026, 8, 18)
    monkeypatch.setattr("meta_portfolio.bugun_tsi", lambda: sabit_bugun)
    out = sinyaller_taze_filtre({}, [{"symbol": "GARAN", "score": 5.0}], "P1")
    assert out == []


# ---------------------------------------------------------------------------
# G) Gelecek bar mutasyonu — geçmiş sinyal değişmemeli (look-ahead yok)
# ---------------------------------------------------------------------------

def test_gelecek_bar_mutasyonu_gecmis_sinyali_degistirmez():
    params = XoverParams(fast_period=15, slow_period=40, order=3)
    rng = np.random.default_rng(11)
    prices = 100.0 + np.cumsum(rng.normal(0, 1.0, 200))

    t = 50
    sig_a = generate_xover_signal(prices, params)

    # t ve sonrası barları değiştir (gelecek)
    prices2 = prices.copy()
    prices2[t:] *= 2.0
    sig_b = generate_xover_signal(prices2, params)

    # t'den ÖNCEKİ sinyal/çizgiler gelecek barlardan ETKİLENEMEZ
    assert np.array_equal(sig_a.signal[:t], sig_b.signal[:t])
    assert np.allclose(sig_a.fast_line[:t], sig_b.fast_line[:t], atol=1e-9)
    assert np.allclose(sig_a.slow_line[:t], sig_b.slow_line[:t], atol=1e-9)


def test_gelecek_bar_mutasyonu_gelecek_cikti_degisebilir():
    params = XoverParams(fast_period=15, slow_period=40, order=3)
    rng = np.random.default_rng(13)
    prices = 100.0 + np.cumsum(rng.normal(0, 1.0, 200))

    t = 50
    sig_a = generate_xover_signal(prices, params)
    prices2 = prices.copy()
    prices2[t:] *= 2.0
    sig_b = generate_xover_signal(prices2, params)

    # Mutasyon sonrası en az bir gelecek bar çıktısı değişmeli (filtre tepkisi)
    assert not np.allclose(sig_a.fast_line[t:], sig_b.fast_line[t:], atol=1e-9)


# ---------------------------------------------------------------------------
# H) Geçmiş bar mutasyonu — çıktı DEĞİŞEBİLİR (causal bağımlılık)
# ---------------------------------------------------------------------------

def test_gecmis_bar_mutasyonu_cikti_degisebilir():
    params = XoverParams(fast_period=15, slow_period=40, order=3)
    rng = np.random.default_rng(17)
    prices = 100.0 + np.cumsum(rng.normal(0, 1.0, 200))

    sig_a = generate_xover_signal(prices, params)
    prices2 = prices.copy()
    prices2[5] *= 0.5  # geçmiş barı değiştir
    sig_b = generate_xover_signal(prices2, params)

    # IIR (lfilter) geçmişe bağımlıdır — en az bir çıktı değişmeli
    assert not np.allclose(sig_a.fast_line, sig_b.fast_line, atol=1e-9)
