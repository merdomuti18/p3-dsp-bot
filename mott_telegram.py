"""
mott_telegram.py — MOTT Telegram Mesaj Formatları
==================================================
Her strateji için görsel kimlik:
  P1 — Momentum  : 🟡 ━━━━━━━━━━━━━━
  P2 — SMC       : 🔵 〰〰〰〰〰〰〰〰
  P3 — DSP       : 🟣 ◆◆◆◆◆◆◆◆◆◆◆
  P4 — Meta      : 🟢 ════════════════
  Özet           : 📊 ────────────────

Her mesajda tarih ve saat (TSİ) zorunlu.
"""

from __future__ import annotations

import os
import requests
import logging
from datetime import datetime
from typing import Optional
import pytz

log = logging.getLogger(__name__)
IST = pytz.timezone("Europe/Istanbul")

# ─── Strateji Görsel Kimlikleri ───────────────────────────────────────────────

STYLES = {
    "P1": {
        "emoji":  "🟡",
        "char":   "━",
        "title":  "P1 — MOMENTUM TARAMA",
        "width":  24,
    },
    "P2": {
        "emoji":  "🔵",
        "char":   "〰",
        "title":  "P2 — SMC TARAMA",
        "width":  13,   # 〰 çift genişlik — 13 yeterli
    },
    "P3": {
        "emoji":  "🟣",
        "char":   "◆",
        "title":  "P3 — DSP TARAMA",
        "width":  24,
    },
    "P4": {
        "emoji":  "🟢",
        "char":   "═",
        "title":  "P4 — META PORTFÖY",
        "width":  24,
    },
    "OZET": {
        "emoji":  "📊",
        "char":   "─",
        "title":  "MOTT — GÜNLÜK ÖZET",
        "width":  24,
    },
}


def _simdi() -> str:
    """Şu anki TSİ zaman damgası."""
    return datetime.now(IST).strftime("%d.%m.%Y | %H:%M TSİ")


def _cizgi(style: dict) -> str:
    """Strateji stiline göre dekoratif çizgi."""
    return style["emoji"] + " " + style["char"] * style["width"]


def header(strateji: str, zaman: Optional[str] = None) -> str:
    """Mesaj başlığı bloğu."""
    s = STYLES.get(strateji, STYLES["OZET"])
    cizgi = _cizgi(s)
    zaman = zaman or _simdi()
    return (
        f"{cizgi}\n"
        f"*{s['title']}*\n"
        f"{cizgi}\n"
        f"📅 {zaman}\n"
    )


def footer(strateji: str) -> str:
    """Mesaj alt çizgisi."""
    s = STYLES.get(strateji, STYLES["OZET"])
    return _cizgi(s)


# ─── P1 Momentum Mesajı ──────────────────────────────────────────────────────

def p1_mesaj(
    sinyaller: list[dict],
    portfoy:   dict,
    zaman:     Optional[str] = None,
) -> str:
    """
    P1 Telegram mesajı.
    sinyaller: [{"symbol": "ALKIM", "score": 0.048, "strategies": [...]}]
    portfoy:   {"pozisyonlar": {...}, "trade_history": [...]}
    """
    zaman = zaman or _simdi()
    lines = [header("P1", zaman)]

    # Sinyaller
    if sinyaller:
        lines.append("📈 *Güçlü AL sinyalleri:*")
        for s in sinyaller[:8]:
            sym   = s.get("symbol", "?")
            skor  = s.get("score", s.get("final_score", 0))
            strat = ", ".join(s.get("strategies", [])[:2])
            lines.append(f"  • `{sym:<8}` skor={skor:+.3f}  _{strat}_")
    else:
        lines.append("📭 Güçlü AL sinyali yok")

    lines.append("")

    # Portföy özeti
    pozlar  = portfoy.get("pozisyonlar", {})
    trades  = portfoy.get("trade_history", [])
    n_pos   = len(pozlar)
    n_trade = len(trades)
    wins    = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
    wr      = f"{wins/n_trade:.0%}" if n_trade else "—"
    pnl     = sum(t.get("pnl_pct", 0) for t in trades)
    pnl_str = f"{pnl:+.1f}%" if n_trade else "—"

    semboller = list(pozlar.keys())[:4]
    pos_str   = " ".join(f"`{s}`" for s in semboller) or "—"

    lines += [
        f"💼 *Portföy:* {n_pos} pozisyon",
        f"   {pos_str}",
        f"📊 WR: {wr} | P&L: {pnl_str} | İşlem: {n_trade}",
        "",
        footer("P1"),
    ]
    return "\n".join(lines)


# ─── P2 SMC Mesajı ───────────────────────────────────────────────────────────

def p2_mesaj(
    sinyaller: list[dict],
    portfoy:   dict,
    zaman:     Optional[str] = None,
) -> str:
    """
    P2 SMC Telegram mesajı.
    sinyaller: [{"symbol": "EREGL", "score": 7.5, "verdict": "GUCLU_AL", "signals": [...]}]
    """
    zaman = zaman or _simdi()
    lines = [header("P2", zaman)]

    if sinyaller:
        lines.append("📈 *SMC Sinyalleri:*")
        for s in sinyaller[:8]:
            sym     = s.get("symbol", "?")
            skor    = s.get("score", s.get("final_score", 0))
            verdict = s.get("verdict", "")
            sigs    = ", ".join(s.get("signals", [])[:2])
            verdict_emoji = "🔵" if "GUCLU" in str(verdict) else "⚪"
            lines.append(f"  {verdict_emoji} `{sym:<8}` skor={skor:.1f}  _{sigs}_")
    else:
        lines.append("📭 SMC sinyali yok")

    lines.append("")

    pozlar  = portfoy.get("pozisyonlar", {})
    trades  = portfoy.get("trade_history", [])
    n_pos   = len(pozlar)
    n_trade = len(trades)
    wins    = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
    wr      = f"{wins/n_trade:.0%}" if n_trade else "—"
    pnl     = sum(t.get("pnl_pct", 0) for t in trades)
    pnl_str = f"{pnl:+.1f}%" if n_trade else "—"

    semboller = list(pozlar.keys())[:4]
    pos_str   = " ".join(f"`{s}`" for s in semboller) or "—"

    lines += [
        f"💼 *Portföy:* {n_pos} pozisyon",
        f"   {pos_str}",
        f"📊 WR: {wr} | P&L: {pnl_str} | İşlem: {n_trade}",
        "",
        footer("P2"),
    ]
    return "\n".join(lines)


# ─── P3 DSP Mesajı ───────────────────────────────────────────────────────────

def p3_mesaj(
    top_longs:     list,
    portfoy:       dict,
    monitor_alert: bool = False,
    corr_risk:     str  = "LOW",
    zaman:         Optional[str] = None,
) -> str:
    """
    P3 DSP Telegram mesajı.
    top_longs: SymbolScore listesi veya dict listesi
    """
    zaman = zaman or _simdi()
    lines = [header("P3", zaman)]

    if top_longs:
        lines.append("📈 *Top Long Adayları:*")
        for s in top_longs[:8]:
            if hasattr(s, "symbol"):
                sym   = s.symbol
                skor  = s.score
                marj  = s.crossover_margin
                slope = getattr(s, "trend_slope", 0)
            else:
                sym   = s.get("symbol", "?")
                skor  = s.get("score", 0)
                marj  = s.get("crossover_margin", 0)
                slope = s.get("trend_slope", 0)
            slope_emoji = "↑" if slope > 0 else "↓"
            lines.append(
                f"  • `{sym:<8}` {skor:+.3f}  "
                f"marj={marj:+.3f} {slope_emoji}"
            )
    else:
        lines.append("📭 Long sinyal yok")

    lines.append("")

    pozlar  = portfoy.get("positions", portfoy.get("pozisyonlar", {}))
    trades  = portfoy.get("trade_history", portfoy.get("closed_trades", []))
    n_pos   = len(pozlar)
    n_trade = len(trades)
    wins    = sum(1 for t in trades if t.get("pnl_pct", t.get("pnl", 0)) > 0)
    wr      = f"{wins/n_trade:.0%}" if n_trade else "—"
    pnl     = sum(t.get("pnl_pct", t.get("pnl", 0)) for t in trades)
    pnl_str = f"{pnl:+.1f}%" if n_trade else "—"

    semboller = list(pozlar.keys())[:4]
    pos_str   = " ".join(f"`{s}`" for s in semboller) or "—"

    corr_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}.get(corr_risk, "⚪")
    monitor_str = "⚠️ Drift var!" if monitor_alert else "✅ Stabil"

    lines += [
        f"💼 *Portföy:* {n_pos} pozisyon",
        f"   {pos_str}",
        f"📊 WR: {wr} | P&L: {pnl_str} | İşlem: {n_trade}",
        f"🔬 Non-stat: {monitor_str}",
        f"🔗 Korelasyon: {corr_emoji} {corr_risk}",
        "",
        footer("P3"),
    ]
    return "\n".join(lines)


# ─── P4 Meta Portföy Mesajı ──────────────────────────────────────────────────

def p4_mesaj(
    secilen:  list[dict],
    portfoy:  dict,
    ic_scores: dict[str, float],
    zaman:    Optional[str] = None,
) -> str:
    """
    P4 Meta Portföy Telegram mesajı.
    secilen:  [{"symbol": "ALKIM", "kaynak": "P1", "ic_agirlik": 0.42}]
    ic_scores: {"P1": 0.08, "P2": 0.06, "P3": 0.10}
    """
    zaman = zaman or _simdi()
    lines = [header("P4", zaman)]

    # IC ağırlıkları
    lines.append("🧠 *IC Ağırlıkları (sinyal kalitesi):*")
    for strateji, ic in sorted(ic_scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(ic * 100)
        lines.append(f"  {strateji}: {ic:+.3f}  {bar}")

    lines.append("")

    if secilen:
        lines.append("📈 *Seçilen Pozisyonlar:*")
        for s in secilen[:8]:
            sym     = s.get("symbol", "?")
            kaynak  = s.get("kaynak", "?")
            agirlik = s.get("ic_agirlik", 0)
            lines.append(f"  • `{sym:<8}` ← {kaynak}  (IC={agirlik:.3f})")
    else:
        lines.append("📭 Seçilen pozisyon yok")

    lines.append("")

    pozlar  = portfoy.get("positions", portfoy.get("pozisyonlar", {}))
    trades  = portfoy.get("trade_history", [])
    n_pos   = len(pozlar)
    n_trade = len(trades)
    wins    = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
    wr      = f"{wins/n_trade:.0%}" if n_trade else "—"
    pnl     = sum(t.get("pnl_pct", 0) for t in trades)
    pnl_str = f"{pnl:+.1f}%" if n_trade else "—"

    lines += [
        f"💼 *Portföy:* {n_pos} pozisyon",
        f"📊 WR: {wr} | P&L: {pnl_str} | İşlem: {n_trade}",
        "",
        footer("P4"),
    ]
    return "\n".join(lines)


# ─── Günlük Özet Mesajı ──────────────────────────────────────────────────────

def ozet_mesaj(
    p1_pnl: float, p1_wr: float, p1_ok: bool,
    p2_pnl: float, p2_wr: float, p2_ok: bool,
    p3_pnl: float, p3_wr: float, p3_ok: bool,
    p4_pnl: Optional[float] = None,
    p4_wr:  Optional[float] = None,
    p4_ok:  bool = False,
    zaman:  Optional[str] = None,
) -> str:
    """Tüm stratejilerin günlük karşılaştırma özeti."""
    zaman = zaman or _simdi()
    lines = [header("OZET", zaman)]

    def satir(label, pnl, wr, ok):
        durum = "✅" if ok else "❌"
        pnl_s = f"{pnl:+.1f}%" if pnl is not None else "—"
        wr_s  = f"{wr:.0%}" if wr is not None and wr > 0 else "—"
        return f"{durum} {label}  P&L: {pnl_s}  WR: {wr_s}"

    lines += [
        satir("🟡 P1 Momentum", p1_pnl, p1_wr, p1_ok),
        satir("🔵 P2 SMC     ", p2_pnl, p2_wr, p2_ok),
        satir("🟣 P3 DSP     ", p3_pnl, p3_wr, p3_ok),
    ]

    if p4_pnl is not None:
        lines.append(satir("🟢 P4 Meta    ", p4_pnl, p4_wr, p4_ok))

    # En iyi strateji
    aktif = {"P1": p1_pnl, "P2": p2_pnl, "P3": p3_pnl}
    if p4_pnl is not None:
        aktif["P4"] = p4_pnl
    en_iyi = max(aktif, key=lambda k: aktif[k] or -999)

    emoji_map = {"P1": "🟡", "P2": "🔵", "P3": "🟣", "P4": "🟢"}
    lines += [
        "",
        f"🏆 *Günün Lideri:* {emoji_map[en_iyi]} {en_iyi}",
        "",
        footer("OZET"),
    ]
    return "\n".join(lines)


# ─── Gönderici ───────────────────────────────────────────────────────────────

def telegram_gonder(
    mesaj:    str,
    token:    Optional[str] = None,
    chat_id:  Optional[str] = None,
) -> bool:
    """Telegram mesajı gönder."""
    token   = token   or os.environ.get("TELEGRAM_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        log.warning("Telegram token/chat_id eksik")
        return False

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id":    chat_id,
                "text":       mesaj,
                "parse_mode": "Markdown",
            },
            timeout=15,
        )
        if r.ok:
            return True
        log.warning("Telegram hata: %s %s", r.status_code, r.text[:100])
        return False
    except Exception as e:
        log.warning("Telegram exception: %s", e)
        return False
