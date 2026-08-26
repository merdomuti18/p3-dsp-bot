# -*- coding: utf-8 -*-
"""
P6 Etap 1 — Director altyapısı + ZKN (zkn-p1-birebir-v1)

P1 scanner_p1.py / portfoy_yonetici.py DOKUNULMAZ.
ZKN Boolean kopyalanır (import edilmez). WYC-v1 support + strategy_wyc(ind).

Bu modül SQZ / CRSI / MD strateji fonksiyonları içermez.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

import pandas as pd

import scanner_p1

log = logging.getLogger(__name__)

RULE_VERSION_ZKN = "zkn-p1-birebir-v1"
RULE_VERSION_WYC = "wyc-v1"
STRATEGY_ZKN = "ZKN"
STRATEGY_WYC = "WYC"
MC_MIN = 10_000_000
MIN_BARS = 50
SUPPORT_WINDOW = 10
ZKN_CRITICAL_KEYS = (
    "close", "ema50", "ema200", "rsi", "stochrsi", "cmf", "rel_vol",
)
WYC_CRITICAL_KEYS = ("open", "high", "low", "close", "support")
# P1 get_indicators anahtarları + P6 OHLC + wyc-v1 support. Başka yeni gösterge yok.
P6_INDICATOR_SCHEMA = (
    "close", "ema8", "ema21", "ema50", "ema200", "sma20", "rsi",
    "macd", "macd_sig", "macd_prev", "macd_sprev",
    "bb_mid", "bb_up", "bb_lo", "cmf", "adx", "di_p", "di_n",
    "stochrsi", "atr", "rel_vol", "change_pct",
    "alpha_bull", "alpha_trend_bull",
    "open", "high", "low",
    "support",
)
FORBIDDEN_RECORD_KEYS = frozenset(
    {"score", "weight", "rank", "final_score", "strategies", "mc"}
)
_NAN = float("nan")


def aktif_hisse_listesi() -> list[str]:
    return scanner_p1.aktif_hisse_listesi()


def mc_hesapla(df: pd.DataFrame) -> float:
    """P1 fetch_ohlcv: close[-1] * volume[-20:].mean()."""
    try:
        return float(df["close"].iloc[-1] * df["volume"].iloc[-20:].mean())
    except Exception:
        return 0.0


def truncate_to_asof(df: pd.DataFrame, asof: Optional[Any] = None) -> pd.DataFrame:
    """asof'tan sonraki barları at. asof None ise df olduğu gibi (son bar = asof)."""
    if asof is None or df.empty:
        return df
    ts = pd.Timestamp(asof)
    return df.loc[df.index <= ts]


def asof_date_from_df(df: pd.DataFrame) -> str:
    idx = df.index[-1]
    if hasattr(idx, "date"):
        return idx.date().isoformat()
    return str(idx)[:10]


def _empty_indicator_dict() -> dict:
    """Complete-schema dict; değerler NaN. Asla {}."""
    return {k: _NAN for k in P6_INDICATOR_SCHEMA}


def _fill_ohlc_from_df(out: dict, df: pd.DataFrame) -> None:
    try:
        if df is None or len(df) < 1:
            return
        out["open"] = float(df["open"].iloc[-1])
        out["high"] = float(df["high"].iloc[-1])
        out["low"] = float(df["low"].iloc[-1])
        out["close"] = float(df["close"].iloc[-1])
    except Exception as exc:
        log.debug("P6 OHLC doldurulamadi: %s", exc)


def _fill_support_from_df(out: dict, df: pd.DataFrame) -> None:
    """support(t) = low.rolling(10, min_periods=10).min().shift(1) — pencere t-10…t-1."""
    try:
        if df is None or "low" not in getattr(df, "columns", []) or len(df) < 1:
            return
        series = df["low"].rolling(window=SUPPORT_WINDOW, min_periods=SUPPORT_WINDOW).min().shift(1)
        val = series.iloc[-1]
        out["support"] = float(val) if pd.notna(val) else _NAN
    except Exception as exc:
        log.debug("P6 support doldurulamadi: %s", exc)
        out["support"] = _NAN


def get_indicators_p6(df: pd.DataFrame) -> dict:
    """
    P6 gösterge sözleşmesi (Seçenek B):
    her zaman complete-schema dict; None yok; {} yok;
    hesaplanamayan alan NaN. P1 değiştirilmez.
    """
    out = _empty_indicator_dict()
    base = None
    try:
        base = scanner_p1.get_indicators(df)
    except Exception as exc:
        log.debug("P1 get_indicators exception (P6 NaN semaya duser): %s", exc)
        base = None
    if isinstance(base, dict) and base:
        out.update(base)
    _fill_ohlc_from_df(out, df)
    _fill_support_from_df(out, df)
    return out


def strategy_zkn(ind: dict) -> bool:
    """zkn-p1-birebir-v1 — P1 Boolean, mc yok. Eksik/NaN zorunlu alan → False."""
    if not isinstance(ind, dict):
        return False
    try:
        rsi = ind.get("rsi", _NAN)
        stochrsi = ind.get("stochrsi", _NAN)
        cmf = ind.get("cmf", _NAN)
        rel_vol = ind.get("rel_vol", _NAN)
        close = ind.get("close", _NAN)
        ema50 = ind.get("ema50", _NAN)
        ema200 = ind.get("ema200", _NAN)
        if any(pd.isna(x) for x in (rsi, stochrsi, cmf, rel_vol, close, ema50)):
            return False
        return bool(
            close > ema50
            and (pd.isna(ema200) or close > ema200)
            and 40 <= rsi <= 58
            and stochrsi < 40
            and cmf > -0.1
            and rel_vol >= 0.8
        )
    except Exception:
        return False


def strategy_wyc(ind: dict) -> bool:
    """wyc-v1 — tek bar spring. mc yok. NaN/range<=0 → False."""
    if not isinstance(ind, dict):
        return False
    try:
        o = ind.get("open", _NAN)
        h = ind.get("high", _NAN)
        low = ind.get("low", _NAN)
        c = ind.get("close", _NAN)
        support = ind.get("support", _NAN)
        if any(pd.isna(x) for x in (o, h, low, c, support)):
            return False
        rng = h - low
        if rng <= 0:
            return False
        wick = (min(o, c) - low) / rng
        return bool(low < support and c > support and wick >= 0.30)
    except Exception:
        return False


def wyc_trigger_conditions(ind: dict) -> dict[str, bool]:
    o = ind.get("open", _NAN)
    h = ind.get("high", _NAN)
    low = ind.get("low", _NAN)
    c = ind.get("close", _NAN)
    support = ind.get("support", _NAN)
    rng = h - low if not any(pd.isna(x) for x in (h, low)) else _NAN
    wick = (min(o, c) - low) / rng if (not pd.isna(rng) and rng > 0) else _NAN
    return {
        "support_finite": bool(not pd.isna(support)),
        "range_positive": bool(not pd.isna(rng) and rng > 0),
        "low_lt_support": bool(not pd.isna(support) and not pd.isna(low) and low < support),
        "close_gt_support": bool(not pd.isna(support) and not pd.isna(c) and c > support),
        "wick_ge_030": bool(not pd.isna(wick) and wick >= 0.30),
    }


def zkn_trigger_conditions(ind: dict) -> dict[str, bool]:
    ema200 = ind.get("ema200")
    return {
        "close_gt_ema50": bool(ind["close"] > ind["ema50"]),
        "close_gt_ema200": bool(pd.isna(ema200) or ind["close"] > ema200),
        "rsi_in_range": bool(40 <= ind["rsi"] <= 58),
        "stochrsi_low": bool(ind["stochrsi"] < 40),
        "cmf_ok": bool(ind["cmf"] > -0.1),
        "rel_vol_adequate": bool(ind["rel_vol"] >= 0.8),
    }


def build_signal_record(
    *,
    symbol: str,
    asof_date: str,
    strategy: str,
    rule_version: str,
    indicators: dict,
    trigger_conditions: dict[str, bool],
    scan_time: str = "",
) -> dict:
    rec = {
        "symbol": symbol,
        "asof_date": asof_date,
        "strategy": strategy,
        "rule_version": rule_version,
        "indicators": indicators,
        "trigger_conditions": trigger_conditions,
    }
    if scan_time:
        rec["scan_time"] = scan_time
    overlap = FORBIDDEN_RECORD_KEYS.intersection(rec)
    if overlap:
        raise ValueError(f"yasak kayit alanlari: {sorted(overlap)}")
    return rec


def universe_ok(df: pd.DataFrame, mc: Optional[float] = None) -> bool:
    if df is None or len(df) < MIN_BARS:
        return False
    if mc is None:
        mc = mc_hesapla(df)
    return mc >= MC_MIN


def evaluate_symbol(symbol: str, df: pd.DataFrame, asof: Optional[Any] = None) -> list[dict]:
    """Tek sembol, kapanmış barlar. Network yok. ZKN ve WYC bağımsız."""
    df = truncate_to_asof(df, asof)
    if not universe_ok(df):
        return []
    ind = get_indicators_p6(df)
    records: list[dict] = []
    if strategy_zkn(ind):
        zkn_keys = {k: ind[k] for k in ZKN_CRITICAL_KEYS if k in ind}
        records.append(
            build_signal_record(
                symbol=symbol,
                asof_date=asof_date_from_df(df),
                strategy=STRATEGY_ZKN,
                rule_version=RULE_VERSION_ZKN,
                indicators=zkn_keys,
                trigger_conditions=zkn_trigger_conditions(ind),
            )
        )
    if strategy_wyc(ind):
        wyc_keys = {k: ind[k] for k in WYC_CRITICAL_KEYS if k in ind}
        records.append(
            build_signal_record(
                symbol=symbol,
                asof_date=asof_date_from_df(df),
                strategy=STRATEGY_WYC,
                rule_version=RULE_VERSION_WYC,
                indicators=wyc_keys,
                trigger_conditions=wyc_trigger_conditions(ind),
            )
        )
    return records


def _assert_agent_signatures() -> None:
    for fn in (strategy_zkn, strategy_wyc):
        params = list(inspect.signature(fn).parameters)
        if params != ["ind"]:
            raise RuntimeError(f"{fn.__name__} imzasi bozuk: {params}")


_assert_agent_signatures()
