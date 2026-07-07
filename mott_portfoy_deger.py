"""
mott_portfoy_deger.py — P1–P5 portföy değeri (100k başlangıç)
"""
from __future__ import annotations

from typing import Any

SERMAYE = 100_000
MAX_POS = 5
POS_TL = SERMAYE / MAX_POS


def _sayi(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def equity_hesapla(strateji: str, portfoy: dict) -> dict:
    """
    Portföy sözlüğünden tahmini equity döndürür.
    Returns: {baslangic, equity, getiri_pct}
    """
    bas = _sayi(
        portfoy.get("baslangic", portfoy.get("sermaye_baslangic", SERMAYE)),
        SERMAYE,
    )

    # P1/P2: nakit + lot değeri (equity alanı varsa öncelik)
    if "nakit" in portfoy:
        if portfoy.get("equity") is not None:
            eq = _sayi(portfoy["equity"], bas)
        else:
            eq = _sayi(portfoy.get("nakit"), bas)
            for pos in portfoy.get("pozisyonlar", {}).values():
                if not isinstance(pos, dict):
                    continue
                f = _sayi(
                    pos.get("guncel_fiyat", pos.get("giris_f", pos.get("giris_fiyat"))),
                    0,
                )
                lot = int(pos.get("lotlar", 0) or 0)
                eq += lot * f
        getiri = (eq - bas) / bas * 100 if bas else 0.0
        return {
            "baslangic": round(bas),
            "equity": round(eq),
            "getiri_pct": round(getiri, 2),
        }

    # P3: history pnl_pct + açık pozisyonlar (entry_price / canlı yoksa pnl_pct)
    if strateji == "P3" or portfoy.get("positions") and "scan_log" in portfoy:
        trades = [
            t for t in portfoy.get("history", [])
            if isinstance(t.get("pnl_pct"), (int, float)) and t["pnl_pct"] == t["pnl_pct"]
        ]
        closed_tl = sum(POS_TL * _sayi(t["pnl_pct"]) / 100 for t in trades)
        open_tl = 0.0
        for sym, pos in portfoy.get("positions", {}).items():
            if not isinstance(pos, dict):
                continue
            ep = _sayi(pos.get("entry_price"), 0)
            cp = _sayi(pos.get("current_price"), 0)
            if ep > 0 and cp > 0:
                open_tl += POS_TL * (cp - ep) / ep
            elif pos.get("pnl_pct") is not None:
                open_tl += POS_TL * _sayi(pos["pnl_pct"]) / 100
        eq = bas + closed_tl + open_tl
        getiri = (eq - bas) / bas * 100 if bas else 0.0
        return {
            "baslangic": round(bas),
            "equity": round(eq),
            "getiri_pct": round(getiri, 2),
        }

    # P4/P5 ve benzeri: trade_history + pozisyonlar.pnl_pct
    trades = portfoy.get("trade_history", portfoy.get("history", []))
    closed_tl = sum(
        POS_TL * _sayi(t.get("pnl_pct")) / 100
        for t in trades
        if isinstance(t, dict)
    )
    open_tl = sum(
        POS_TL * _sayi(pos.get("pnl_pct")) / 100
        for pos in portfoy.get("pozisyonlar", portfoy.get("positions", {})).values()
        if isinstance(pos, dict)
    )
    eq = bas + closed_tl + open_tl
    if portfoy.get("sermaye_mevcut") is not None and not trades and not portfoy.get("pozisyonlar"):
        eq = _sayi(portfoy["sermaye_mevcut"], eq)
    getiri = (eq - bas) / bas * 100 if bas else 0.0
    return {
        "baslangic": round(bas),
        "equity": round(eq),
        "getiri_pct": round(getiri, 2),
    }


def portfoy_degeri_satiri(strateji: str, portfoy: dict) -> str:
    o = equity_hesapla(strateji, portfoy)
    emoji = "🟢" if o["getiri_pct"] >= 0 else "🔴"
    return (
        f"{emoji} *Portföy değeri:* `{o['equity']:,} TL` "
        f"({o['getiri_pct']:+.1f}%) · başlangıç `{o['baslangic']:,} TL`"
    )
