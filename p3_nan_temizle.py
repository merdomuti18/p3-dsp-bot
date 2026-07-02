"""
p3_nan_temizle.py — portfolio_state.json'daki bozuk (NaN) trade kayıtlarını temizler.

Eski _get_price() fonksiyonu, yfinance'in son barında Close=NaN dönmesi
durumunda bunu geçerli fiyat sanıp kaydediyordu. Bu script:
  1. NaN içeren kayıtlar için entry_date/exit_date civarından gerçek
     kapanış fiyatını yeniden çekmeyi dener.
  2. Fiyat bulunursa pnl_pct'i yeniden hesaplar.
  3. Bulunamazsa (delisted, tatil günü, vb.) kaydı geçmişten siler.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf

STATE_FILE = Path(__file__).parent / "portfolio_state.json"


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _fiyat_bul(symbol: str, gun: str) -> float | None:
    """gun (YYYY-MM-DD) civarında (±3 gün) gerçek kapanış fiyatını bul."""
    try:
        d = date.fromisoformat(gun)
    except Exception:
        return None
    ticker = f"{symbol}.IS"
    start = (d - timedelta(days=5)).isoformat()
    end = (d + timedelta(days=3)).isoformat()
    try:
        df = yf.Ticker(ticker).history(start=start, end=end)
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        # d gününe en yakın kapanışı bul
        target = str(d)
        exact = df[df.index.strftime("%Y-%m-%d") <= target]
        row = exact.iloc[-1] if not exact.empty else df.iloc[0]
        price = float(row["Close"])
        return round(price, 3) if price == price else None
    except Exception:
        return None


def temizle() -> dict:
    if not STATE_FILE.exists():
        return {"durum": "portfolio_state.json yok"}
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    history = state.get("history", [])

    duzeltilen, silinen, degismeyen = [], [], 0
    yeni_history = []

    for rec in history:
        entry_p, exit_p = rec.get("entry_price"), rec.get("exit_price")
        pnl = rec.get("pnl_pct")
        if not (_nan(entry_p) or _nan(exit_p) or _nan(pnl)):
            yeni_history.append(rec)
            degismeyen += 1
            continue

        sym = rec["symbol"]
        if _nan(entry_p):
            entry_p = _fiyat_bul(sym, rec["entry_date"])
        if _nan(exit_p):
            exit_p = _fiyat_bul(sym, rec["exit_date"])

        if entry_p and exit_p and entry_p > 0:
            yeni_pnl = round((exit_p - entry_p) / entry_p * 100, 2)
            rec["entry_price"] = entry_p
            rec["exit_price"] = exit_p
            rec["pnl_pct"] = yeni_pnl
            yeni_history.append(rec)
            duzeltilen.append({"symbol": sym, "entry": entry_p, "exit": exit_p, "pnl_pct": yeni_pnl})
        else:
            silinen.append({"symbol": sym, "entry_date": rec.get("entry_date"), "exit_date": rec.get("exit_date")})

    state["history"] = yeni_history
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "toplam_kayit": len(history),
        "degismeyen": degismeyen,
        "duzeltilen_sayisi": len(duzeltilen),
        "duzeltilen": duzeltilen,
        "silinen_sayisi": len(silinen),
        "silinen": silinen,
    }


if __name__ == "__main__":
    import pprint
    sonuc = temizle()
    pprint.pp(sonuc)
