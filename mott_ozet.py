"""
mott_ozet.py — MOTT Günlük Özet Rapor
=======================================
P1 + P2 + P3 + P4 state dosyalarını okur,
karşılaştırmalı özet Telegram mesajı gönderir.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pytz
import requests

BASE_DIR   = Path(os.environ.get("MOTT_BASE_DIR", "."))
TG_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
P1_STATUS  = os.environ.get("P1_STATUS", "unknown")
P2_STATUS  = os.environ.get("P2_STATUS", "unknown")
P3_STATUS  = os.environ.get("P3_STATUS", "unknown")
P4_STATUS  = os.environ.get("P4_STATUS", "unknown")

IST = pytz.timezone("Europe/Istanbul")


def load_state(path: Path) -> dict:
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def portfoy_metrikleri(state: dict) -> dict:
    """State'den portföy metriklerini çıkar."""
    pozlar  = state.get("pozisyonlar", state.get("positions", {}))
    trades  = state.get("trade_history", state.get("closed_trades", []))
    n_pos   = len(pozlar)
    n_trade = len(trades)
    wins    = sum(1 for t in trades if t.get("pnl_pct", t.get("pnl", 0)) > 0)
    wr      = wins / n_trade if n_trade else 0.0
    pnl     = sum(t.get("pnl_pct", t.get("pnl", 0)) for t in trades)
    return {"n_pos": n_pos, "n_trade": n_trade, "wr": wr, "pnl": pnl}


def send_telegram(message: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        print(message)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "Markdown"},
            timeout=15,
        )
        print("Telegram OK" if r.ok else f"Telegram hata: {r.status_code}")
    except Exception as e:
        print(f"Telegram exception: {e}")


def main():
    now   = datetime.now(IST)
    zaman = now.strftime("%d.%m.%Y | %H:%M TSİ")

    s_p1 = load_state(BASE_DIR / "state_p1.json")
    s_p2 = load_state(BASE_DIR / "state_p2.json")
    s_p3 = load_state(BASE_DIR / "portfolio_state.json")
    s_p4 = load_state(BASE_DIR / "state_p4.json")

    m1 = portfoy_metrikleri(s_p1)
    m2 = portfoy_metrikleri(s_p2)
    m3 = portfoy_metrikleri(s_p3)
    m4 = portfoy_metrikleri(s_p4)

    status_e = {"success": "✅", "failure": "❌", "skipped": "⏭️", "unknown": "❓"}

    def satir(emoji, label, status, metriks):
        durum  = status_e.get(status, "❓")
        pnl_s  = f"{metriks['pnl']:+.1f}%" if metriks['n_trade'] else "—"
        wr_s   = f"{metriks['wr']:.0%}" if metriks['n_trade'] else "—"
        return f"{durum} {emoji} *{label}*  P&L:{pnl_s}  WR:{wr_s}  Pos:{metriks['n_pos']}"

    # IC bilgisi P4'ten
    ic_history = s_p4.get("ic_history", [])
    ic_str = ""
    if ic_history:
        son_ic = ic_history[-1]
        ic_str = (
            f"\n🧠 *IC (sinyal kalitesi):*\n"
            f"  P1:{son_ic.get('ic_p1',0):+.3f}  "
            f"P2:{son_ic.get('ic_p2',0):+.3f}  "
            f"P3:{son_ic.get('ic_p3',0):+.3f}"
        )

    # En iyi P&L
    pnller = {
        "P1": m1["pnl"], "P2": m2["pnl"],
        "P3": m3["pnl"], "P4": m4["pnl"],
    }
    en_iyi = max(pnller, key=lambda k: pnller[k])
    emoji_map = {"P1": "🟡", "P2": "🔵", "P3": "🟣", "P4": "🟢"}

    lines = [
        f"📊 ────────────────────────",
        f"*MOTT — GÜNLÜK ÖZET*",
        f"────────────────────────",
        f"📅 {zaman}",
        "",
        satir("🟡", "P1 Momentum", P1_STATUS, m1),
        satir("🔵", "P2 SMC     ", P2_STATUS, m2),
        satir("🟣", "P3 DSP     ", P3_STATUS, m3),
        satir("🟢", "P4 Meta    ", P4_STATUS, m4),
        ic_str,
        "",
        f"🏆 *Günün Lideri:* {emoji_map[en_iyi]} {en_iyi}",
        "",
        f"📊 ────────────────────────",
    ]

    mesaj = "\n".join(l for l in lines if l is not None)
    send_telegram(mesaj)
    print(mesaj)


if __name__ == "__main__":
    main()
