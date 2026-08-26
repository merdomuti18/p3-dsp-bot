"""
P6 Etap 1 — ZKN parity, orkestratör kapıları, 5+5, look-ahead.

Network yok. P1 scanner_p1 / portfoy_yonetici çağrılarak state yazılmaz
(yalnız get_indicators + strategy_zkn okuma).
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import scanner_p1
import scanner_p6
from p6_scenarios import AGENT_IMPLEMENTED, ZKN_NO, ZKN_YES, zkn_base_ind

ZKN_KEYS = scanner_p6.ZKN_CRITICAL_KEYS


def _ohlcv(n=64, seed=42, vol_scale=1_000_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-06", periods=n)
    close = 100 + rng.normal(0, 1.0, n).cumsum()
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.5, n)
    volume = rng.integers(int(200_000 * vol_scale / 1_000_000),
                          int(800_000 * vol_scale / 1_000_000), n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _ind_close(a, b, key):
    va, vb = a[key], b[key]
    if pd.isna(va) and pd.isna(vb):
        return True
    if pd.isna(va) or pd.isna(vb):
        return False
    return va == pytest.approx(vb, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# A1 imza / yasak ajanlar
# ---------------------------------------------------------------------------

def test_scanner_p6_filtfilt_yok():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / "scanner_p6.py").read_text(encoding="utf-8")
    assert "filtfilt" not in text


def test_a1_strategy_zkn_imzasi_ind_only():
    params = list(inspect.signature(scanner_p6.strategy_zkn).parameters)
    assert params == ["ind"]


def test_diger_ajan_stratejileri_yazilmamis():
    for ad in ("strategy_wyc", "strategy_sqz", "strategy_crsi", "strategy_md"):
        assert not hasattr(scanner_p6, ad)
    assert AGENT_IMPLEMENTED["ZKN"] is True
    assert AGENT_IMPLEMENTED["WYC"] is False
    assert AGENT_IMPLEMENTED["SQZ"] is False
    assert AGENT_IMPLEMENTED["CRSI"] is False
    assert AGENT_IMPLEMENTED["MD"] is False


# ---------------------------------------------------------------------------
# A7 ZKN 5+5 (direkt ind)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("etiket,ind,beklenen", ZKN_YES + ZKN_NO)
def test_a7_zkn_5plus5_direkt_ind(etiket, ind, beklenen):
    assert scanner_p6.strategy_zkn(ind) is beklenen, etiket
    assert bool(scanner_p1.strategy_zkn(ind, scanner_p6.MC_MIN)) is beklenen, etiket


def test_a3_zkn_nan_ind_false_exception_yok():
    assert scanner_p6.strategy_zkn(zkn_base_ind(rsi=float("nan"))) is False
    assert scanner_p6.strategy_zkn({}) is False


# ---------------------------------------------------------------------------
# C1 / C4 gösterge parity
# ---------------------------------------------------------------------------

def test_c1_yedi_kritik_gosterge_p1_ile_esit():
    df = _ohlcv(n=80)
    p1 = scanner_p1.get_indicators(df)
    p6 = scanner_p6.get_indicators_p6(df)
    assert p1 is not None and p6 is not None
    for key in ZKN_KEYS:
        assert _ind_close(p1, p6, key), key


def test_c4_ema200_kisa_seride_nan():
    df = _ohlcv(n=60)  # 50 <= n < 200
    p1 = scanner_p1.get_indicators(df)
    p6 = scanner_p6.get_indicators_p6(df)
    assert pd.isna(p1["ema200"]) and pd.isna(p6["ema200"])
    # isna kolu ZKN'yi tek başına öldürmez
    p6["close"] = p6["ema50"] + 1
    p6["rsi"] = 50
    p6["stochrsi"] = 10
    p6["cmf"] = 0
    p6["rel_vol"] = 1
    assert scanner_p6.strategy_zkn(p6) is True


def test_b5_extras_zkn_dict_none_yapmaz():
    df = _ohlcv(n=50)
    out = scanner_p6.get_indicators_p6(df)
    assert out is not None
    for key in ZKN_KEYS:
        assert key in out


# ---------------------------------------------------------------------------
# C2 True kümesi — aynı OHLCV, evren ∩ Boolean
# ---------------------------------------------------------------------------

def test_c2_zkn_true_kumesı_ayni_ohlcv():
    hits_p1 = []
    hits_p6 = []
    for seed in range(30):
        df = _ohlcv(n=80, seed=seed, vol_scale=5_000_000)
        mc = scanner_p6.mc_hesapla(df)
        eligible = scanner_p6.universe_ok(df, mc)
        p1_ind = scanner_p1.get_indicators(df)
        p6_ind = scanner_p6.get_indicators_p6(df)
        if p1_ind is None or p6_ind is None:
            continue
        p1_bool = scanner_p1.strategy_zkn(p1_ind, mc)
        p6_bool = scanner_p6.strategy_zkn(p6_ind)
        if eligible and p1_bool:
            hits_p1.append(seed)
        if eligible and p6_bool:
            hits_p6.append(seed)
        # Evren dışı: P6 evaluate boş; P1 fonksiyonu mc ile False
        if not eligible:
            assert scanner_p6.evaluate_symbol(f"S{seed}", df) == []
    assert hits_p1 == hits_p6


def test_c2_evaluate_p1_boolean_ile_uyumlu():
    df = _ohlcv(n=80, seed=7, vol_scale=8_000_000)
    recs = scanner_p6.evaluate_symbol("THYAO", df)
    p1_ind = scanner_p1.get_indicators(df)
    mc = scanner_p6.mc_hesapla(df)
    p1_hit = bool(scanner_p1.strategy_zkn(p1_ind, mc))
    assert (len(recs) == 1) == p1_hit
    if recs:
        assert recs[0]["strategy"] == "ZKN"
        assert recs[0]["rule_version"] == scanner_p6.RULE_VERSION_ZKN
        assert "score" not in recs[0]
        assert "mc" not in recs[0]


# ---------------------------------------------------------------------------
# B2 / B3 orkestratör
# ---------------------------------------------------------------------------

def test_b3_elli_bar_altinda_evren_disi():
    df = _ohlcv(n=49)
    assert scanner_p6.universe_ok(df) is False
    assert scanner_p6.evaluate_symbol("X", df) == []


def test_b2_mc_altinda_evren_disi_zkn_true_olsa_bile():
    ind = zkn_base_ind()
    assert scanner_p6.strategy_zkn(ind) is True
    df = _ohlcv(n=60)
    df["close"] = 1.0
    df["volume"] = 1.0  # mc = 1 * 1 << 10M
    assert scanner_p6.mc_hesapla(df) < scanner_p6.MC_MIN
    assert scanner_p6.evaluate_symbol("X", df) == []


def test_b2_mc_esik_dahil():
    df = _ohlcv(n=60)
    df["close"] = 10.0
    df["volume"] = 1_000_000.0  # mc = 10M
    assert scanner_p6.mc_hesapla(df) == pytest.approx(10_000_000.0)
    assert scanner_p6.universe_ok(df) is True


# ---------------------------------------------------------------------------
# B7 kayıt yasak alanlar / A5 A6
# ---------------------------------------------------------------------------

def test_b7_kayitta_score_yok():
    rec = scanner_p6.build_signal_record(
        symbol="GARAN",
        asof_date="2026-08-20",
        strategy="ZKN",
        rule_version=scanner_p6.RULE_VERSION_ZKN,
        indicators={"close": 1.0},
        trigger_conditions={"rsi_in_range": True},
    )
    for k in scanner_p6.FORBIDDEN_RECORD_KEYS:
        assert k not in rec


# ---------------------------------------------------------------------------
# A8 look-ahead: t+1 mutasyonu t asof sinyalini değiştirmez
# ---------------------------------------------------------------------------

def test_a8_lookahead_tplus1_mutasyonu():
    df = _ohlcv(n=81, seed=11, vol_scale=6_000_000)
    asof = df.index[-2]
    prefix = scanner_p6.truncate_to_asof(df, asof)
    rec_a = scanner_p6.evaluate_symbol("Y", prefix)
    df_mut = df.copy()
    last = df_mut.index[-1]
    df_mut.loc[last, ["open", "high", "low", "close", "volume"]] = [
        9_999.0, 10_000.0, 9_998.0, 9_999.5, 1.0,
    ]
    prefix_b = scanner_p6.truncate_to_asof(df_mut, asof)
    rec_b = scanner_p6.evaluate_symbol("Y", prefix_b)
    assert rec_a == rec_b


def _assert_p6_contract_dict(out):
    assert out is not None
    assert isinstance(out, dict)
    assert out != {}
    for key in scanner_p6.P6_INDICATOR_SCHEMA:
        assert key in out


# ---------------------------------------------------------------------------
# Seçenek B — get_indicators_p6 contract
# ---------------------------------------------------------------------------

def test_contract_valid_df_dict_not_empty_not_none():
    df = _ohlcv(n=80)
    out = scanner_p6.get_indicators_p6(df)
    _assert_p6_contract_dict(out)
    p1 = scanner_p1.get_indicators(df)
    for key in ZKN_KEYS:
        assert _ind_close(p1, out, key), key


def test_contract_short_df_still_dict():
    df = _ohlcv(n=10)
    out = scanner_p6.get_indicators_p6(df)
    _assert_p6_contract_dict(out)


def test_contract_empty_df_still_dict():
    out = scanner_p6.get_indicators_p6(pd.DataFrame())
    _assert_p6_contract_dict(out)
    for key in ZKN_KEYS:
        assert pd.isna(out[key])


def test_contract_p1_none_simulation_dict_nan(monkeypatch):
    monkeypatch.setattr(scanner_p1, "get_indicators", lambda df: None)
    df = _ohlcv(n=80)
    out = scanner_p6.get_indicators_p6(df)
    _assert_p6_contract_dict(out)
    assert pd.isna(out["rsi"])
    assert pd.isna(out["ema50"])
    assert pd.isna(out["stochrsi"])
    assert pd.isna(out["cmf"])
    assert pd.isna(out["rel_vol"])
    assert scanner_p6.strategy_zkn(out) is False


def test_contract_never_none_never_empty_on_paths(monkeypatch):
    for df in (_ohlcv(n=80), _ohlcv(n=10), pd.DataFrame()):
        out = scanner_p6.get_indicators_p6(df)
        assert isinstance(out, dict)
        assert out is not None
        assert out != {}
    monkeypatch.setattr(scanner_p1, "get_indicators", lambda df: None)
    out = scanner_p6.get_indicators_p6(_ohlcv(n=50))
    assert isinstance(out, dict)
    assert out is not None
    assert out != {}


def test_strategy_zkn_return_is_python_bool():
    assert type(scanner_p6.strategy_zkn(zkn_base_ind())) is bool
    assert type(scanner_p6.strategy_zkn(zkn_base_ind(rsi=float("nan")))) is bool


def test_c1_ema200_numeric_220_bar():
    df = _ohlcv(n=220, seed=5)
    p1 = scanner_p1.get_indicators(df)
    p6 = scanner_p6.get_indicators_p6(df)
    assert not pd.isna(p1["ema200"]) and not pd.isna(p6["ema200"])
    for key in ZKN_KEYS:
        assert _ind_close(p1, p6, key), key


def test_c2_signal_parity_33_asof():
    p1_true, p6_true = set(), set()
    for seed in (0, 1):
        df = _ohlcv(n=220, seed=100 + seed, vol_scale=8_000_000)
        for asof in df.index[-33:]:
            prefix = scanner_p6.truncate_to_asof(df, asof)
            mc = scanner_p6.mc_hesapla(prefix)
            if not scanner_p6.universe_ok(prefix, mc):
                continue
            p1i = scanner_p1.get_indicators(prefix)
            p6i = scanner_p6.get_indicators_p6(prefix)
            key = (seed, str(asof.date()))
            if p1i is not None and bool(scanner_p1.strategy_zkn(p1i, mc)):
                p1_true.add(key)
            if bool(scanner_p6.strategy_zkn(p6i)):
                p6_true.add(key)
    assert p1_true == p6_true


def test_runtime_zkn_bool_no_exception():
    rng = np.random.default_rng(0)
    for i in range(400):
        if i % 10 == 0:
            ind = zkn_base_ind(rsi=float("nan"))
        else:
            ind = zkn_base_ind(
                close=float(rng.uniform(50, 150)),
                ema50=float(rng.uniform(50, 150)),
                ema200=float(rng.uniform(50, 150)),
                rsi=float(rng.uniform(0, 100)),
                stochrsi=float(rng.uniform(0, 100)),
                cmf=float(rng.uniform(-0.5, 0.5)),
                rel_vol=float(rng.uniform(0, 3)),
            )
        out = scanner_p6.strategy_zkn(ind)
        assert type(out) is bool

