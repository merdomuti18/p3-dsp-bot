"""
DSP-P3 FAZ 1 — dsp_core unit testleri
=====================================
Saf, deterministik fonksiyonlar. Network yok, state yazma yok.
Kritik: check_no_lookahead gerçekten çalışıyor mu (P0-2 bulgusu).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dsp_core import (
    AdaptiveFilterState,
    FORBIDDEN_IN_PROD,
    apply_causal,
    check_no_lookahead,
    design_butterworth,
    design_chebyshev,
    detect_dominant_cycle,
    warmup_length,
)


# ---------------------------------------------------------------------------
# Filtre tasarımı
# ---------------------------------------------------------------------------

def test_design_butterworth_katsayi_uretir():
    b, a = design_butterworth(period=20, order=3)
    assert isinstance(b, np.ndarray) and isinstance(a, np.ndarray)
    assert len(b) == len(a)
    assert a[0] == pytest.approx(1.0)


def test_design_butterworth_gecersiz_period_valueerror():
    # normalized_cutoff >= 1 olduğunda fırlatmalı (period=1 → 1/(1)/0.5 = 2)
    with pytest.raises(ValueError):
        design_butterworth(period=1, order=3)


def test_design_chebyshev_katsayi_uretir():
    b, a = design_chebyshev(period=20, order=4, ripple_db=0.5)
    assert isinstance(b, np.ndarray) and isinstance(a, np.ndarray)
    assert a[0] == pytest.approx(1.0)


def test_design_chebyshev_gecersiz_period_valueerror():
    with pytest.raises(ValueError):
        design_chebyshev(period=1, order=4)


# ---------------------------------------------------------------------------
# Causal filtre uygulama
# ---------------------------------------------------------------------------

def test_apply_causal_cikti_uzunlugu():
    b, a = design_butterworth(period=20, order=3)
    data = np.linspace(100, 200, 100)
    out = apply_causal(b, a, data)
    assert len(out) == len(data)
    assert np.all(np.isfinite(out))


def test_apply_causal_az_veri_valueerror():
    b, a = design_butterworth(period=20, order=3)
    with pytest.raises(ValueError):
        apply_causal(b, a, np.array([1.0, 2.0]))


def test_apply_causal_deterministik():
    b, a = design_butterworth(period=20, order=3)
    data = np.linspace(100, 200, 100)
    out1 = apply_causal(b, a, data)
    out2 = apply_causal(b, a, data.copy())
    np.testing.assert_array_equal(out1, out2)


def test_warmup_length():
    assert warmup_length(order=3, period=20) == 60
    assert warmup_length(order=2, period=15) == 30


# ---------------------------------------------------------------------------
# Cycle detektörü
# ---------------------------------------------------------------------------

def test_detect_dominant_cycle_az_veri_valueerror():
    with pytest.raises(ValueError):
        detect_dominant_cycle(np.linspace(100, 110, 31))


def test_detect_dominant_cycle_deterministik_sinüs():
    t = np.arange(128)
    # 32 günlük periyotlu sinüs → dominant periyot ~32 civarı bulunmalı
    prices = 100 + 5 * np.sin(2 * np.pi * t / 32) + 0.01 * t
    r = detect_dominant_cycle(prices)
    assert r.n_samples == 128
    assert 15 <= r.dominant_period <= 60
    # production numpy bool döndürür (np.True_); bool() ile normalleştirilir
    assert isinstance(r.is_reliable, (bool, np.bool_))
    assert bool(r.is_reliable) is True  # peakiness 20.4 >> 2.5 → güvenilir
    assert len(r.top_periods) == 3 and len(r.top_powers) == 3


# ---------------------------------------------------------------------------
# AdaptiveFilterState
# ---------------------------------------------------------------------------

def test_adaptive_state_refit_akis():
    st = AdaptiveFilterState(refit_interval_days=7)
    assert st.needs_refit(date(2026, 8, 17)) is True  # ilk çağrı → True

    prices = 100 + 5 * np.sin(2 * np.pi * np.arange(128) / 32)
    c = st.refit("GARAN", prices, date(2026, 8, 17))
    assert c.refit_date == date(2026, 8, 17)
    assert st.needs_refit(date(2026, 8, 18)) is False  # 1 gün sonra → False
    assert st.needs_refit(date(2026, 8, 24)) is True   # 7 gün sonra → True

    b, a = st.get_coeffs("GARAN")
    assert len(b) == len(a)


def test_adaptive_state_refit_yoksa_valueerror():
    st = AdaptiveFilterState()
    with pytest.raises(ValueError):
        st.get_coeffs("OLMAYAN")


# ---------------------------------------------------------------------------
# LOOK-AHEAD GUARD — check_no_lookahead gerçekten çalışıyor mu
# ---------------------------------------------------------------------------

def test_forbidden_liste_filtfilt_icerir():
    assert FORBIDDEN_IN_PROD == ["filtfilt"]


def test_check_no_lookahead_temiz_dosya_gecer(tmp_path):
    temiz = tmp_path / "temiz.py"
    temiz.write_text("from scipy.signal import lfilter\n", encoding="utf-8")
    check_no_lookahead(str(temiz))  # fırlatmamalı


def test_check_no_lookahead_filtfilt_import_yakalar(tmp_path):
    kirli = tmp_path / "kirli.py"
    kirli.write_text(
        "from scipy.signal import filtfilt\nx = filtfilt(b, a, data)\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        check_no_lookahead(str(kirli))


def test_check_no_lookahead_yorum_ve_string_gecer(tmp_path):
    # Yorumlarda/string'lerde 'filtfilt' geçmesi SORUN DEĞİL (yalnızca import).
    dosya = tmp_path / "yorum.py"
    dosya.write_text(
        '# filtfilt KULLANILMAZ\nKRITIK = "bu dosyada filtfilt yasak"\n',
        encoding="utf-8",
    )
    check_no_lookahead(str(dosya))  # fırlatmamalı
