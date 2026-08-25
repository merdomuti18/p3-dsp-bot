"""
P6 Etap 1 — ZKN 5+5 sentetik ind senaryoları + ajan kayıt defteri.

Boolean testleri doğrudan strategy_*(ind) çağırır (OHLCV yok).
WYC/SQZ/CRSI/MD stratejisi henüz yok; kayıt defteri False.
"""
from __future__ import annotations

import math

# Ajan kodu bu PR'de yalnızca ZKN için vardır.
AGENT_IMPLEMENTED = {
    "ZKN": True,
    "WYC": False,
    "SQZ": False,
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


def is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False
