"""
mott_aylik_rapor.py — 5 portföy aylık karşılaştırma raporu
P1, P2, P3, P4, P5 — her biri 100k TL başlangıç sermayesi.

FAZ 3.2 MİGRASYON MİMARİSİ:
    state → mott_state.normalize() → canonical state
         → derived metrikler (_pX_rapor_blok)
         → ayrı live enrichment (_acik_pozisyon_enrich — YALNIZ P1/P2/P3)
         → Telegram / aylık rapor

Kurallar:
  * mott_state.normalize() canlı fiyat bilmez; deterministiktir.
  * Canlı fiyat yalnız bu modüldeki açık enrichment adımında kullanılır
    (mpa.get_price — runtime lookup; T4 mock noktası korunur).
  * P1/P2 mevcut "%0 → 100.000 TL" davranışı KORUNUR (bu migration'da düzeltilmez).
  * P4/P5 equity = normalize()["equity"] (sermaye_mevcut) — mevcut sözleşme.
  * normalize()["equity"] ASLA enrichment sonucuyla değiştirilmez.
  * _portfoy_getiri / _p5_analiz eski okuyucular olarak KORUNUR (migration
    equivalence testleri ve mevcut T4 kilitleri bunlara bağlıdır).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mott_performans_analiz as mpa  # runtime lookup (mock noktası: mpa.get_price)
from mott_performans_analiz import SERMAYE
from mott_state import normalize

BASE = Path(os.environ.get("MOTT_BASE_DIR", "."))
TSI = ZoneInfo("Europe/Istanbul")

# Paper modeli canonical sabiti (POS_SIZE_PCT %20 × 100_000 başlangıç).
POS_TL = SERMAYE // 5  # 20.000


def _portfoy_getiri(fname: str, label: str) -> dict | None:
    """ESKİ okuyucu (P1/P2) — KORUNUR (migration equivalence + T4 kilidi).
    FAZ 3.2'de rapor_olustur bunu kullanmaz; _p1_p2_rapor_blok kullanır."""
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
    """ESKİ okuyucu (P5) — KORUNUR (migration equivalence referansı).
    FAZ 3.2'de rapor_olustur bunu kullanmaz; _p5_rapor_blok kullanır."""
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


# ---------------------------------------------------------------------------
# FAZ 3.2 — YENİ: live enrichment (yalnız raporlama katmanı)
# ---------------------------------------------------------------------------

def _acik_pozisyon_enrich(pozisyonlar: list[dict]) -> tuple[list[dict], float | None]:
    """Açık pozisyonlara now/pnl_pct/tl ekler ve toplam unrealized PnL (TL)
    döndürür. YALNIZ P1/P2/P3 (state'te fiyat yok). P4/P5 KULLANMAZ.

    Canlı fiyat: mpa.get_price(sym) — runtime lookup (import kopyası YOK),
    böylece T4 mock noktası (monkeypatch mpa.get_price) korunur.
    NETWORK=OFF altında mock'suz çağrı get_price'ı bloklar (try/except → None).
    """
    enriched: list[dict] = []
    toplam = 0.0
    for poz in pozisyonlar:
        ep = poz["giris_fiyat"]
        cp = mpa.get_price(poz["symbol"])
        if ep and cp:
            pnl_pct = (cp - ep) / ep * 100
            tl = POS_TL * (cp - ep) / ep
            toplam += tl
        else:
            pnl_pct = None
            tl = 0.0
        enriched.append({
            "symbol": poz["symbol"],
            "entry": ep,
            "now": cp,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "tl": round(tl),
        })
    return enriched, round(toplam, 2)


def _kapanan_wr(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    kazanan = sum(1 for t in trades if (t["pnl_pct"] or 0) > 0)
    return round(kazanan / len(trades) * 100, 1)


# ---------------------------------------------------------------------------
# FAZ 3.2 — YENİ: normalize tabanlı rapor blokları (canonical + derived)
# ---------------------------------------------------------------------------

def _p3_rapor_blok() -> dict:
    n = normalize("P3")
    acik_enriched, unrealized = _acik_pozisyon_enrich(n["pozisyonlar"])
    equity = n["baslangic_sermayesi"] + n["realized_pnl"] + unrealized
    getiri = (equity - n["baslangic_sermayesi"]) / n["baslangic_sermayesi"] * 100
    trades = n["islem_gecmisi"]
    # bozuk_kayit + donem: rapor katmanında minimal ham-state okuması
    # (normalize() contract'ına EKLENMEZ — FAZ 3.2 kuralı).
    raw = {}
    try:
        raw = json.loads((BASE / "portfolio_state.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    bozuk = max(0, len(raw.get("history", [])) - len(trades))
    donem = f"{raw.get('created', '?')} → bugün"
    return {
        "strateji": "P3 DSP",
        "donem": donem,
        "sermaye": n["baslangic_sermayesi"],
        "equity_est": round(equity),
        "getiri_pct": round(getiri, 2),
        "kapanan": len(trades),
        "kapanan_wr": _kapanan_wr(trades),
        "bozuk_kayit": bozuk,
        "acik": acik_enriched,
        "closed_detail": trades,
    }


def _p4_rapor_blok() -> dict:
    n = normalize("P4")
    equity = n["equity"]  # sermaye_mevcut — canonical (değiştirilmez)
    getiri = (equity - n["baslangic_sermayesi"]) / n["baslangic_sermayesi"] * 100
    trades = n["islem_gecmisi"]
    acik = [
        {"symbol": p["symbol"], "pnl_pct": p["pnl_pct"],
         "tl": round(POS_TL * (p["pnl_pct"] or 0) / 100)}
        for p in n["pozisyonlar"]
    ]
    return {
        "strateji": "P4 Meta",
        "donem": "2026-06-25 → bugün",
        "sermaye": n["baslangic_sermayesi"],
        "equity_est": round(equity),
        "getiri_pct": round(getiri, 2),
        "kapanan": len(trades),
        "kapanan_wr": _kapanan_wr(trades),
        "acik": acik,
        "worst": min(trades, key=lambda x: x["pnl_pct"] or 0) if trades else None,
        "best": max(trades, key=lambda x: x["pnl_pct"] or 0) if trades else None,
    }


def _p5_rapor_blok() -> dict:
    n = normalize("P5")
    equity = n["equity"]  # sermaye_mevcut — canonical (değiştirilmez)
    getiri = (equity - n["baslangic_sermayesi"]) / n["baslangic_sermayesi"] * 100
    trades = n["islem_gecmisi"]
    return {
        "strateji": "P5 Komite",
        "sermaye": n["baslangic_sermayesi"],
        "equity_est": round(equity),
        "getiri_pct": round(getiri, 2),
        "kapanan": len(trades),
        "kapanan_wr": _kapanan_wr(trades),
        "acik": len(n["pozisyonlar"]),
    }


def _p1_p2_rapor_blok(kod: str) -> dict:
    n = normalize(kod)
    ad = "Momentum" if kod == "P1" else "SMC"
    return {
        "strateji": f"{kod} {ad}",
        "sermaye": n["baslangic_sermayesi"],
        # Mevcut davranış KORUNUR: state'te sermaye_mevcut yok → başlangıç
        # sermayesi, %0 (bu migration'da DÜZELTİLMEZ — ayrı karar).
        "equity_est": n["baslangic_sermayesi"],
        "getiri_pct": 0.0,
        "kapanan": len(n["islem_gecmisi"]),
        "acik_sayisi": len(n["pozisyonlar"]),
        "nakit": n["nakit"],
    }


def rapor_olustur() -> dict:
    p3 = _p3_rapor_blok()
    p4 = _p4_rapor_blok()
    p5 = _p5_rapor_blok()
    p1 = _p1_p2_rapor_blok("P1")
    p2 = _p1_p2_rapor_blok("P2")

    satirlar = [
        {"kod": "P3", "ad": "DSP", **{k: v for k, v in p3.items() if k != "strateji"}},
        {"kod": "P4", "ad": "Meta Optimizer", **{k: v for k, v in p4.items() if k != "strateji"}},
        {"kod": "P5", "ad": "Komite", **{k: v for k, v in p5.items() if k != "strateji"}},
    ]
    for kod, ad, b in (("P1", "Momentum", p1), ("P2", "SMC", p2)):
        satirlar.append({
            "kod": kod,
            "ad": ad,
            "sermaye": b["sermaye"],
            "equity_est": b["equity_est"],
            "getiri_pct": b["getiri_pct"],
            "durum": None,
            "sinyal_sayisi": None,
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
        "p1_p2": [p1, p2],
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
