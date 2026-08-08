"""Geçmiş state dosyalarından portföy performans özeti."""
from __future__ import annotations

import json
from pathlib import Path

import yfinance as yf

BASE = Path(__file__).parent
SERMAYE = 100_000
POS = 20_000


def get_price(sym: str) -> float | None:
    try:
        h = yf.Ticker(f"{sym}.IS").history(period="5d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None


def analiz_p3() -> dict:
    d = json.loads((BASE / "portfolio_state.json").read_text(encoding="utf-8"))
    closed = [
        t for t in d.get("history", [])
        if isinstance(t.get("pnl_pct"), (int, float)) and t["pnl_pct"] == t["pnl_pct"]
    ]
    open_rows = []
    open_tl = 0.0
    for sym, pos in d.get("positions", {}).items():
        ep = pos.get("entry_price")
        cp = get_price(sym)
        if ep and cp:
            pnl = (cp - ep) / ep * 100
            tl = POS * pnl / 100
            open_tl += tl
            open_rows.append({"symbol": sym, "entry": ep, "now": cp, "pnl_pct": round(pnl, 2), "tl": round(tl)})

    closed_tl = sum(POS * t["pnl_pct"] / 100 for t in closed)
    equity = SERMAYE + closed_tl + open_tl
    return {
        "strateji": "P3 DSP",
        "donem": f"{d.get('created', '?')} → bugün",
        "sermaye": SERMAYE,
        "equity_est": round(equity),
        "getiri_pct": round((equity - SERMAYE) / SERMAYE * 100, 2),
        "kapanan": len(closed),
        "kapanan_wr": round(sum(1 for t in closed if t["pnl_pct"] > 0) / len(closed) * 100, 1) if closed else 0,
        "bozuk_kayit": len(d.get("history", [])) - len(closed),
        "acik": open_rows,
        "closed_detail": closed,
    }


def analiz_p4() -> dict:
    d = json.loads((BASE / "state_p4.json").read_text(encoding="utf-8"))
    trades = d.get("trade_history", [])
    closed_tl = sum(POS * t["pnl_pct"] / 100 for t in trades)
    open_rows = []
    open_tl = 0.0
    for sym, pos in d.get("pozisyonlar", {}).items():
        pnl = pos.get("pnl_pct", 0) or 0
        tl = POS * pnl / 100
        open_tl += tl
        open_rows.append({"symbol": sym, "pnl_pct": pnl, "tl": round(tl)})
    equity = SERMAYE + closed_tl + open_tl
    return {
        "strateji": "P4 Meta",
        "donem": "2026-06-25 → bugün",
        "sermaye": SERMAYE,
        "equity_est": round(equity),
        "getiri_pct": round((equity - SERMAYE) / SERMAYE * 100, 2),
        "kapanan": len(trades),
        "kapanan_wr": round(sum(1 for t in trades if t["pnl_pct"] > 0) / len(trades) * 100, 1) if trades else 0,
        "acik": open_rows,
        "worst": min(trades, key=lambda x: x["pnl_pct"]) if trades else None,
        "best": max(trades, key=lambda x: x["pnl_pct"]) if trades else None,
    }


def _portfoy_analiz(dosya: str, strateji: str) -> dict | None:
    """portfoy.json / portfoy_p2.json: trade_history + açık pozisyon özeti."""
    path = BASE / dosya
    if not path.exists():
        return None
    p = json.loads(path.read_text(encoding="utf-8"))
    trades = p.get("trade_history", [])
    closed = [t for t in trades if isinstance(t.get("pnl_pct"), (int, float))]
    wins = [t for t in closed if t["pnl_pct"] > 0]
    acik = []
    for sym, pos in p.get("pozisyonlar", {}).items():
        ep = pos.get("giris_f")
        cp = get_price(sym)
        pnl = round((cp - ep) / ep * 100, 2) if ep and cp else None
        acik.append({"symbol": sym, "entry": ep, "now": cp, "pnl_pct": pnl})
    return {
        "strateji": strateji,
        "nakit": round(p.get("nakit", 0), 2),
        "baslangic": p.get("baslangic", SERMAYE),
        "kapanan": len(closed),
        "kapanan_wr": round(sum(1 for t in closed if t["pnl_pct"] > 0) / len(closed) * 100, 1) if closed else None,
        "sum_pnl_pct": round(sum(t["pnl_pct"] for t in closed), 2),
        "acik": acik,
        "closed_detail": closed,
    }


def analiz_p1_p2() -> list[dict]:
    out = []
    r = _portfoy_analiz("portfoy.json", "P1 Momentum")
    if r:
        out.append(r)
    r = _portfoy_analiz("portfoy_p2.json", "P2 SMC")
    if r:
        out.append(r)
    return out


if __name__ == "__main__":
    import pprint
    r = {"p3": analiz_p3(), "p4": analiz_p4(), "p1_p2": analiz_p1_p2()}
    pprint.pp(r)
