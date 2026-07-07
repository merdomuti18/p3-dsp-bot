"""
mott_sabah_telegram.py — Sabah günlük plan (Telegram)
=====================================================
Küresel makro özeti + P1/P2/P3 gün içi plan + P4/P5 önerileri (tek mesaj).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)
BASE_DIR = Path(os.environ.get("MOTT_BASE_DIR", "."))
DURUM_FILE = BASE_DIR / "son_durum.json"
SERMAYE = 100_000


def _fmt_pct(v: float) -> str:
    return f"{v:+.1f}%" if v else "—"


def _makro_bolumu() -> list[str]:
    if not DURUM_FILE.exists():
        return ["🌍 *Küresel görünüm:* veri yok (sabah akışı henüz çalışmadı)"]
    try:
        d = json.loads(DURUM_FILE.read_text(encoding="utf-8"))
    except Exception:
        return ["🌍 *Küresel görünüm:* okunamadı"]

    skor = float(d.get("makro_skor", 0))
    karar = d.get("makro_karar", "NORMAL")
    pr = d.get("piyasa_ret", {})
    emoji = {"GIRME": "🔴", "DIKKATLI": "🟡", "NORMAL": "🟢"}.get(karar, "⚪")
    lines = [
        "🌍 *Küresel Görünüm*",
        f"  🇺🇸 S&P500: {_fmt_pct(pr.get('^GSPC', 0))} | Nasdaq: {_fmt_pct(pr.get('^IXIC', 0))} | VIX: {_fmt_pct(pr.get('^VIX', 0))}",
        f"  🌏 Nikkei: {_fmt_pct(pr.get('^N225', 0))} | Hang Seng: {_fmt_pct(pr.get('^HSI', 0))} | Shanghai: {_fmt_pct(pr.get('000001.SS', 0))}",
        f"  🥇 Altın: {_fmt_pct(pr.get('GC=F', 0))}",
        f"  📊 Makro skor: *{skor:.1f}* → {emoji} *{karar}*",
    ]
    viop = d.get("viop", {})
    if viop:
        lines.append(
            f"  📌 VİOP: *{viop.get('label', '?')}* ({viop.get('score', 0):+.0f}p)"
        )
    return lines


def _portfoy_satir(strateji: str, portfoy: dict) -> str:
    try:
        from mott_portfoy_deger import portfoy_degeri_satiri
        return portfoy_degeri_satiri(strateji, portfoy)
    except Exception:
        return f"💰 Portföy: ~{SERMAYE:,} TL (başlangıç)"


def _p1_bolumu() -> list[str]:
    path = BASE_DIR / "portfoy.json"
    if not path.exists():
        return ["🟡 *P1 — Momentum*", "  _Portföy dosyası yok_"]
    p = json.loads(path.read_text(encoding="utf-8"))
    bekleyen = p.get("bekleyen_al", [])
    lines = [
        "🟡 *P1 — Momentum*",
        _portfoy_satir("P1", p),
    ]
    if p.get("pozisyonlar"):
        acik = ", ".join(p["pozisyonlar"].keys())
        lines.append(f"  📂 Açık: `{acik}`")
    else:
        lines.append("  📂 Açık pozisyon yok")
    if bekleyen:
        lines.append(
            "  🎯 Plan (11:00 alım): "
            + ", ".join(f"`{b['symbol']}`" for b in bekleyen[:6])
        )
        for b in bekleyen[:4]:
            lines.append(
                f"    • `{b.get('symbol','?')}` skor:{b.get('final_score', 0):.0f} "
                f"[{','.join(b.get('strategies', [])[:3])}]"
            )
    else:
        lines.append("  🎯 Bugün P1 bekleyen alım yok")
    return lines


def _p2_bolumu() -> list[str]:
    path = BASE_DIR / "portfoy_p2.json"
    if not path.exists():
        return ["🔵 *P2 — SMC*", "  _Portföy dosyası yok_"]
    p = json.loads(path.read_text(encoding="utf-8"))
    bekleyen = p.get("bekleyen_al", [])
    lines = [
        "🔵 *P2 — SMC*",
        _portfoy_satir("P2", p),
    ]
    if p.get("pozisyonlar"):
        lines.append(f"  📂 Açık: {', '.join('`'+s+'`' for s in p['pozisyonlar'])}")
    if bekleyen:
        lines.append(
            "  🎯 Gün içi aday: "
            + ", ".join(f"`{b['symbol']}`" for b in bekleyen[:6])
        )
        for b in bekleyen[:4]:
            teyit = "✓" if b.get("teyit_var") else "?"
            lines.append(
                f"    • `{b.get('symbol','?')}` {b.get('verdict','')} "
                f"skor:{b.get('final_score', b.get('score', 0)):.1f} teyit:{teyit}"
            )
    else:
        lines.append("  🎯 Bekleyen SMC adayı yok (akşam taramasından)")
    scan = ""
    s2 = BASE_DIR / "state_p2.json"
    if s2.exists():
        scan = json.loads(s2.read_text(encoding="utf-8")).get("last_scan", "")
    if scan:
        lines.append(f"  _Son tarama: {scan}_")
    return lines


def _p3_bolumu() -> list[str]:
    path = BASE_DIR / "portfolio_state.json"
    if not path.exists():
        return ["🟣 *P3 — DSP*", "  _State yok_"]
    p = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "🟣 *P3 — DSP*",
        _portfoy_satir("P3", p),
    ]
    if p.get("positions"):
        lines.append(f"  📂 Açık: {', '.join(f'`{s}`' for s in p['positions'])}")
    log_items = p.get("scan_log", [])
    if log_items:
        son = log_items[-1]
        top = son.get("top5", son.get("top_longs", []))
        if top:
            lines.append(
                "  👁 İzleme (son tarama): "
                + ", ".join(f"`{s}`" for s in top[:5])
            )
        lines.append(f"  _Son tarama: {son.get('date', '?')}_")
    else:
        lines.append("  👁 Son DSP taraması kaydı yok")
    return lines


def _p4_oneri_bolumu() -> list[str]:
    try:
        from meta_portfolio import (
            ic_agirlikli_birlestir,
            p1_sinyalleri_yukle,
            p2_sinyalleri_yukle,
            p3_sinyalleri_yukle,
        )
        p1, _ = p1_sinyalleri_yukle()
        p2, _ = p2_sinyalleri_yukle()
        p3, _ = p3_sinyalleri_yukle()
        if not p1 and not p2 and not p3:
            p1, p2, p3 = _ham_sinyaller_fallback()
        adaylar = ic_agirlikli_birlestir(p1, p2, p3, 0.0, 0.0, 0.0)[:5]
    except Exception as exc:
        log.warning("P4 öneri: %s", exc)
        adaylar = []

    lines = ["🟢 *P4 — Meta Optimizer (günlük öneri)*"]
    p4 = _portfoy_yukle("state_p4.json")
    if p4:
        lines.append(_portfoy_satir("P4", p4))
    if adaylar:
        for a in adaylar:
            kay = ",".join(a.get("kaynaklar", []))
            lines.append(
                f"  • `{a.get('symbol','?')}` meta={a.get('meta_score', 0):.3f} _{kay}_"
                + (" _çoklu onay_" if a.get("coklu_onay") else "")
            )
    else:
        lines.append("  _Taze P1/P2/P3 sinyali yok — akşam taraması sonrası dolacak_")
    lines.append("  _Not: P4 akşam işlem yapar; bu liste planlama önerisidir._")
    return lines


def _p5_oneri_bolumu() -> list[str]:
    lines = ["⚖️ *P5 — Komite (günlük öneri)*"]
    p5_path = BASE_DIR / "state_p5.json"
    if p5_path.exists():
        p5 = json.loads(p5_path.read_text(encoding="utf-8"))
        lines.append(_portfoy_satir("P5", p5))
    try:
        from p5_committee import komite_adaylari, makro_karar
        secilen, elenen = komite_adaylari()
        lines.append(f"  🧭 Makro kapı: *{makro_karar()}*")
        if secilen:
            lines.append("  ✅ Öncelikli:")
            for a in secilen[:5]:
                lines.append(
                    f"    • `{a['symbol']}` skor:{a.get('komite_skor', 0)} "
                    f"_{','.join(a.get('kaynaklar', []))}_"
                )
        if elenen:
            lines.append(
                "  🚫 Elenen (örnek): "
                + ", ".join(f"`{e['symbol']}`" for e in elenen[:4])
            )
    except Exception as exc:
        lines.append(f"  _Komite önerisi üretilemedi: {exc}_")
    lines.append("  _Not: P5 akşam işlem yapar; bu liste planlama önerisidir._")
    return lines


def _portfoy_yukle(fname: str) -> dict:
    path = BASE_DIR / fname
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _ham_sinyaller_fallback() -> tuple[list, list, list]:
    """Sabah: taze tarama yoksa son kayıtlı taramayı kullan (planlama)."""
    p1, p2, p3 = [], [], []
    p1f = BASE_DIR / "state_p1.json"
    if p1f.exists():
        d = json.loads(p1f.read_text(encoding="utf-8"))
        p1 = d.get("tarama", {}).get("signals", [])
    p2f = BASE_DIR / "state_p2.json"
    if p2f.exists():
        d = json.loads(p2f.read_text(encoding="utf-8"))
        p2 = d.get("tarama", {}).get("signals", [])
    p3f = BASE_DIR / "portfolio_state.json"
    if p3f.exists():
        d = json.loads(p3f.read_text(encoding="utf-8"))
        log_items = d.get("scan_log", [])
        if log_items:
            for sym in log_items[-1].get("top5", log_items[-1].get("top_longs", [])):
                p3.append({"symbol": sym, "score": 0.0, "strateji": "P3"})
    return p1, p2, p3


def gunluk_plan_mesaji() -> str:
    from mott_telegram import simdi

    lines = [
        "🌅 *MOTT — Günlük Plan*",
        f"📅 {simdi()}",
        "",
        *_makro_bolumu(),
        "",
        *_p1_bolumu(),
        "",
        *_p2_bolumu(),
        "",
        *_p3_bolumu(),
        "",
        *_p4_oneri_bolumu(),
        "",
        *_p5_oneri_bolumu(),
        "",
        "ℹ️ _Gün içi: yalnızca işlem yapılan stratejiden anlık bildirim gelir._",
        "ℹ️ _Akşam: P1→P5 tarama sonuçları sırayla iletilir._",
    ]
    return "\n".join(lines)


def gunluk_plan_gonder() -> bool:
    from mott_telegram import telegram_gonder

    mesaj = gunluk_plan_mesaji()
    ok = telegram_gonder(mesaj)
    if ok:
        log.info("Sabah günlük plan Telegram gönderildi")
    return ok


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="MOTT sabah günlük plan")
    parser.add_argument("--gonder", action="store_true")
    args = parser.parse_args()
    if args.gonder:
        print("OK" if gunluk_plan_gonder() else "FAIL")
    else:
        print(gunluk_plan_mesaji())
