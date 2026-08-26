"""
P6 Etap 1 — ZKN 5+5 sentetik ind senaryoları + ajan kayıt defteri.

Boolean testleri doğrudan strategy_*(ind) çağırır (OHLCV yok).
CRSI/MD stratejisi henüz yok.
"""
from __future__ import annotations

import math

# Ajan kodu: ZKN + WYC + SQZ. CRSI/MD yok.
AGENT_IMPLEMENTED = {
    "ZKN": True,
    "WYC": True,
    "SQZ": True,
    "CRSI": False,
    "MD": False,
}


def zkn_base_ind(**overrides):
    ind = {
        "close": 100.0,
        "ema50": 99.0,
        "ema200": 90.0,
        "rsi": 50.0,
        "stochrsi": 30.0,
        "cmf": 0.0,
        "rel_vol": 1.0,
    }
    ind.update(overrides)
    return ind


ZKN_YES = [
    ("YES-1", zkn_base_ind(), True),
    ("YES-2", zkn_base_ind(rsi=40.0), True),
    ("YES-3", zkn_base_ind(rsi=58.0), True),
    ("YES-4", zkn_base_ind(rel_vol=0.8), True),
    ("YES-5", zkn_base_ind(ema200=float("nan")), True),
]

ZKN_NO = [
    ("NO-1", zkn_base_ind(rsi=39.0), False),
    ("NO-2", zkn_base_ind(rsi=59.0), False),
    ("NO-3", zkn_base_ind(stochrsi=40.0), False),
    ("NO-4", zkn_base_ind(cmf=-0.1), False),
    ("NO-5", zkn_base_ind(rel_vol=0.799), False),
]


def wyc_base_ind(**overrides):
    """Geçerli yeşil spring: support=100, low=90, close=105, wick>=0.30."""
    ind = {
        "open": 96.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "support": 100.0,
    }
    ind.update(overrides)
    return ind


WYC_YES = [
    # YES-1: geçerli normal spring. wick=(96-90)/(110-90)=0.30
    ("YES-1", wyc_base_ind(), True),
    # YES-2: low < support sıkı (eşitlik yok). support=100, low=99.99
    ("YES-2", wyc_base_ind(low=99.99, open=100.5, high=101.5, close=101.2), True),
    # YES-3: lower_wick_ratio == 0.30
    ("YES-3", wyc_base_ind(open=96.0, high=110.0, low=90.0, close=105.0), True),
    # YES-4: geçerli yeşil mum (close > open)
    ("YES-4", wyc_base_ind(open=96.0, close=105.0), True),
    # YES-5: geçerli kırmızı mum (close < open); wick=(101-90)/20=0.55
    ("YES-5", wyc_base_ind(open=110.0, close=101.0, high=110.0, low=90.0), True),
]

WYC_NO = [
    # NO-1: low >= support; wick=(104-101)/(111-101)=0.30, close>support
    ("NO-1", wyc_base_ind(low=101.0, open=104.0, high=111.0, close=105.0, support=100.0), False),
    # NO-2: close <= support; wick=(99-90)/(120-90)=0.30, low<support
    ("NO-2", wyc_base_ind(open=110.0, high=120.0, low=90.0, close=99.0, support=100.0), False),
    # NO-3: wick=(92-90)/(110-90)=0.10 < 0.30; low<support ve close>support
    ("NO-3", wyc_base_ind(open=92.0, high=110.0, low=90.0, close=105.0, support=100.0), False),
    # NO-4: high == low → range=0. (düz mumda low==close olduğu için close>support
    # ile low<support aynı anda mümkün değildir; kural range<=0 → False)
    ("NO-4", wyc_base_ind(open=100.0, high=100.0, low=100.0, close=100.0, support=99.0), False),
    ("NO-5", wyc_base_ind(support=float("nan")), False),
]


def sqz_base_ind(**overrides):
    """Geçerli kırılım: recent_squeeze True, close > bb_up. squeeze_on[t] şart değil."""
    ind = {
        "close": 101.0,
        "bb_up": 100.0,
        "recent_squeeze": True,
        "squeeze_on": False,
    }
    ind.update(overrides)
    return ind


SQZ_YES = [
    ("YES-1", sqz_base_ind(), True),
    ("YES-2", sqz_base_ind(close=100.0 + 1e-9, bb_up=100.0), True),
    ("YES-3", sqz_base_ind(recent_squeeze=True, squeeze_on=False), True),
    ("YES-4", sqz_base_ind(recent_squeeze=True, squeeze_on=False), True),
    ("YES-5", sqz_base_ind(recent_squeeze=True, squeeze_on=False), True),
]

SQZ_NO = [
    ("NO-1", sqz_base_ind(close=100.0, bb_up=100.0), False),
    ("NO-2", sqz_base_ind(close=99.0, bb_up=100.0), False),
    ("NO-3", sqz_base_ind(recent_squeeze=False, squeeze_on=True, close=101.0, bb_up=100.0), False),
    ("NO-4", sqz_base_ind(bb_up=float("nan")), False),
    ("NO-5", sqz_base_ind(recent_squeeze=False, squeeze_on=False), False),
]


def is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False
