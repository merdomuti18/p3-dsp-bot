# -*- coding: utf-8 -*-
"""
P6 Etap 1 — Director altyapısı + ZKN + WYC + SQZ-v1

P1 scanner_p1.py / portfoy_yonetici.py DOKUNULMAZ.
ZKN Boolean kopyalanır. WYC-v1 support. SQZ-v1 recent_squeeze + close>bb_up.

Bu modül CRSI / MD strateji fonksiyonları içermez.
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
RULE_VERSION_SQZ = "sqz-kc20-1.5-v1"
STRATEGY_ZKN = "ZKN"
STRATEGY_WYC = "WYC"
STRATEGY_SQZ = "SQZ"
MC_MIN = 10_000_000
MIN_BARS = 50
SUPPORT_WINDOW = 10
KC_EMA_SPAN = 20
KC_ATR_PERIOD = 14
KC_MULT = 1.5
# squeeze_on.iloc[-6:-1] → t-5…t-1 (5 bar); t hariç. Başka dilim yok.
SQZ_RECENT_START = -6
SQZ_RECENT_END = -1
SQZ_RECENT_MIN_BARS = 6
ZKN_CRITICAL_KEYS = (
    "close", "ema50", "ema200", "rsi", "stochrsi", "cmf", "rel_vol",
)
WYC_CRITICAL_KEYS = ("open", "high", "low", "close", "support")
SQZ_CRITICAL_KEYS = ("close", "bb_up", "recent_squeeze")
# P1 get_indicators + P6 OHLC + support + SQZ recent_squeeze.
P6_INDICATOR_SCHEMA = (
    "close", "ema8", "ema21", "ema50", "ema200", "sma20", "rsi",
    "macd", "macd_sig", "macd_prev", "macd_sprev",
    "bb_mid", "bb_up", "bb_lo", "cmf", "adx", "di_p", "di_n",
    "stochrsi", "atr", "rel_vol", "change_pct",
    "alpha_bull", "alpha_trend_bull",
    "open", "high", "low",
    "support",
    "recent_squeeze",
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


def squeeze_on_bar(bb_up, kc_up, bb_lo, kc_lo) -> bool:
    """Frozen: (bb_up < kc_up) AND (bb_lo > kc_lo). Eşitlik ve NaN → False."""
    if any(pd.isna(x) for x in (bb_up, kc_up, bb_lo, kc_lo)):
        return False
    return bool(bb_up < kc_up and bb_lo > kc_lo)


def squeeze_on_series(df: pd.DataFrame) -> pd.Series:
    """BB = P1 _bbands; KC = EMA20 ± 1.5 * P1 _atr(h,l,c,14). bb_* overwrite yok."""
    c = df["close"]
    h = df["high"]
    low = df["low"]
    _bb_mid, bb_up, bb_lo = scanner_p1._bbands(c)
    kc_mid = scanner_p1._ema(c, KC_EMA_SPAN)
    atr14 = scanner_p1._atr(h, low, c, KC_ATR_PERIOD)
    kc_up = kc_mid + KC_MULT * atr14
    kc_lo = kc_mid - KC_MULT * atr14
    valid = bb_up.notna() & bb_lo.notna() & kc_up.notna() & kc_lo.notna()
    return valid & (bb_up < kc_up) & (bb_lo > kc_lo)


def recent_squeeze_from_series(squeeze_on: pd.Series) -> bool:
    """squeeze_on.iloc[-6:-1].any() — t-5…t-1; t yok. len<6 → False."""
    if squeeze_on is None or len(squeeze_on) < SQZ_RECENT_MIN_BARS:
        return False
    window = squeeze_on.iloc[SQZ_RECENT_START:SQZ_RECENT_END]
    return bool(window.fillna(False).any())


def _fill_sqz_from_df(out: dict, df: pd.DataFrame) -> None:
    """Yalnız recent_squeeze skaler. bb_*/ZKN/support üzerine yazılmaz."""
    try:
        if df is None or len(df) < 1:
            return
        cols = getattr(df, "columns", [])
        if not all(k in cols for k in ("high", "low", "close")):
            return
        sq = squeeze_on_series(df)
        out["recent_squeeze"] = recent_squeeze_from_series(sq)
    except Exception as exc:
        log.debug("P6 SQZ doldurulamadi: %s", exc)
        out["recent_squeeze"] = _NAN


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
    _fill_sqz_from_df(out, df)
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


def strategy_sqz(ind: dict) -> bool:
    """sqz-kc20-1.5-v1 — recent_squeeze AND close > bb_up. mc/rel_vol/squeeze_on[t] yok."""
    if not isinstance(ind, dict):
        return False
    try:
        close = ind.get("close", _NAN)
        bb_up = ind.get("bb_up", _NAN)
        recent = ind.get("recent_squeeze", _NAN)
        if any(pd.isna(x) for x in (close, bb_up, recent)):
            return False
        if not recent:
            return False
        return bool(close > bb_up)
    except Exception:
        return False


def sqz_trigger_conditions(ind: dict) -> dict[str, bool]:
    close = ind.get("close", _NAN)
    bb_up = ind.get("bb_up", _NAN)
    recent = ind.get("recent_squeeze", _NAN)
    return {
        "recent_squeeze": bool(not pd.isna(recent) and recent),
        "close_gt_bb_up": bool(
            not pd.isna(close) and not pd.isna(bb_up) and close > bb_up
        ),
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
    """Tek sembol, kapanmış barlar. Network yok. ZKN, WYC, SQZ bağımsız."""
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
    if strategy_sqz(ind):
        sqz_keys = {k: ind[k] for k in SQZ_CRITICAL_KEYS if k in ind}
        records.append(
            build_signal_record(
                symbol=symbol,
                asof_date=asof_date_from_df(df),
                strategy=STRATEGY_SQZ,
                rule_version=RULE_VERSION_SQZ,
                indicators=sqz_keys,
                trigger_conditions=sqz_trigger_conditions(ind),
            )
        )
    return records


def _assert_agent_signatures() -> None:
    for fn in (strategy_zkn, strategy_wyc, strategy_sqz):
        params = list(inspect.signature(fn).parameters)
        if params != ["ind"]:
            raise RuntimeError(f"{fn.__name__} imzasi bozuk: {params}")


_assert_agent_signatures()
