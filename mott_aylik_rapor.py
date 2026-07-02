"""
mott_aylik_rapor.py — 5 portföy aylık karşılaştırma raporu
P1, P2, P3, P4, P5 — her biri 100k TL başlangıç sermayesi.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mott_performans_analiz import SERMAYE, analiz_p1_p2, analiz_p3, analiz_p4

BASE = Path(os.environ.get("MOTT_BASE_DIR", "."))
TSI = ZoneInfo("Europe/Istanbul")


def _portfoy_getiri(fname: str, label: str) -> dict | None:
    path = BASE / fname
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        sermaye = d.get("sermaye_baslangic", SERMAYE)
        mevcut = d.get("sermaye_mevcut", sermaye)
        return {
            "strateji": label,
            "sermaye": sermaye,
            "equity_est": round(mevcut),
            "getiri_pct": round((mevcut - sermaye) / sermaye * 100, 2),
            "kapanan": len(d.get("trade_history", d.get("islem_gecmisi", []))),
            "acik": len(d.get("pozisyonlar", d.get("positions", {}))),
        }
    except Exception:
        return None


def _p5_analiz() -> dict:
    path = BASE / "state_p5.json"
    if not path.exists():
        return {
            "strateji": "P5 Komite",
            "durum": "Henüz çalışmadı",
            "sermaye": SERMAYE,
            "equity_est": SERMAYE,
            "getiri_pct": 0.0,
            "kapanan": 0,
            "kapanan_wr": 0,
            "acik": 0,
        }
    d = json.loads(path.read_text(encoding="utf-8"))
    trades = d.get("trade_history", [])
    pos_tl = SERMAYE / 5
    closed_tl = sum(pos_tl * t["pnl_pct"] / 100 for t in trades)
    open_tl = sum(pos_tl * (p.get("pnl_pct") or 0) / 100 for p in d.get("pozisyonlar", {}).values())
    equity = SERMAYE + closed_tl + open_tl
    return {
        "strateji": "P5 Komite",
        "sermaye": SERMAYE,
        "equity_est": round(equity),
        "getiri_pct": round((equity - SERMAYE) / SERMAYE * 100, 2),
        "kapanan": len(trades),
        "kapanan_wr": round(sum(1 for t in trades if t["pnl_pct"] > 0) / len(trades) * 100, 1) if trades else 0,
        "acik": len(d.get("pozisyonlar", {})),
    }


def rapor_olustur() -> dict:
    p3 = analiz_p3()
    p4 = analiz_p4()
    p5 = _p5_analiz()
    p1p2_scan = analiz_p1_p2()
    p1_eq = _portfoy_getiri("portfoy.json", "P1 Momentum")
    p2_eq = _portfoy_getiri("portfoy_p2.json", "P2 SMC")

    satirlar = [
        {"kod": "P3", "ad": "DSP", **{k: v for k, v in p3.items() if k != "strateji"}},
        {"kod": "P4", "ad": "Meta Optimizer", **{k: v for k, v in p4.items() if k != "strateji"}},
        {"kod": "P5", "ad": "Komite", **{k: v for k, v in p5.items() if k != "strateji"}},
    ]

    for scan in p1p2_scan:
        kod = "P1" if "P1" in scan["strateji"] else "P2"
        eq = p1_eq if kod == "P1" else p2_eq
        satirlar.append({
            "kod": kod,
            "ad": "Momentum" if kod == "P1" else "SMC",
            "sermaye": SERMAYE,
            "equity_est": eq["equity_est"] if eq else SERMAYE,
            "getiri_pct": eq["getiri_pct"] if eq else None,
            "durum": scan.get("durum"),
            "sinyal_sayisi": scan.get("sinyal_sayisi"),
        })

    satirlar.sort(
        key=lambda x: x.get("getiri_pct") if x.get("getiri_pct") is not None else -999,
        reverse=True,
    )
    return {
        "tarih": datetime.now(TSI).strftime("%d.%m.%Y %H:%M TSİ"),
        "sermaye_portfoy": SERMAYE,
        "portfoyler": satirlar,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p1_p2": p1p2_scan,
    }


def telegram_metin(rapor: dict) -> str:
    lines = [
        "📊 *MOTT — Aylık Portföy Karşılaştırması*",
        f"📅 {rapor['tarih']}",
        f"💰 Başlangıç sermayesi: *{rapor['sermaye_portfoy']:,} TL* / portföy",
        "",
        "*Sıralama (tahmini getiri):*",
    ]
    for i, p in enumerate(rapor["portfoyler"], 1):
        kod = p["kod"]
        ad = p.get("ad", "")
        g = p.get("getiri_pct")
        eq = p.get("equity_est", SERMAYE)
        if g is None:
            lines.append(f"{i}. `{kod}` {ad} — veri yok (sadece tarama)")
        else:
            emoji = "🟢" if g >= 0 else "🔴"
            lines.append(f"{i}. {emoji} `{kod}` {ad}: *{g:+.2f}%* → ~{eq:,} TL")
    lines += [
        "",
        "_P1/P2: portfoy dosyası yoksa getiri hesaplanamaz._",
        "_P4/P5: P1+P2+P3 sinyallerinden; birbirinden bağımsız._",
    ]
    return "\n".join(lines)


def telegram_gonder_rapor() -> bool:
    try:
        from mott_telegram import telegram_gonder
        return telegram_gonder(telegram_metin(rapor_olustur()))
    except Exception:
        return False


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="MOTT aylık 5 portföy raporu")
    parser.add_argument("--telegram", action="store_true", help="Telegram'a gönder")
    args = parser.parse_args()
    rapor = rapor_olustur()
    if args.telegram:
        ok = telegram_gonder_rapor()
        print("Telegram:", "OK" if ok else "BAŞARISIZ")
    else:
        print(telegram_metin(rapor))
        print("\n--- JSON ---")
        pprint.pp(rapor)
