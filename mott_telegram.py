"""
mott_telegram.py — MOTT Telegram Mesaj Formatları
==================================================
Her strateji için görsel kimlik + zorunlu alanlar:
  sinyal listesi, açık pozisyonlar, P&L, kazanma oranı, strateji, tarih/saat (TSİ)

Telegram yalnızca alım/satım (işlem) olduğunda gönderilir — çağıran kod kontrol eder.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz
import requests

log = logging.getLogger(__name__)
IST = pytz.timezone("Europe/Istanbul")
BASE_DIR = Path(os.environ.get("MOTT_BASE_DIR", "."))

STYLES = {
    "P1": {"emoji": "🟡", "char": "━", "title": "P1 — MOMENTUM", "width": 24},
    "P2": {"emoji": "🔵", "char": "〰", "title": "P2 — SMC", "width": 13},
    "P3": {"emoji": "🟣", "char": "◆", "title": "P3 — DSP", "width": 24},
    "P4": {"emoji": "🟢", "char": "═", "title": "P4 — META OPTİMİZER", "width": 24},
    "P5": {"emoji": "⚖️", "char": "▬", "title": "P5 — KOMİTE", "width": 24},
}


def get_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN", "")


def get_chat_id() -> str:
    return os.environ.get("TELEGRAM_CHAT_ID", "")


def simdi() -> str:
    return datetime.now(IST).strftime("%d.%m.%Y | %H:%M TSİ")


def islem_var(
    giris: list | None = None,
    cikis: list | None = None,
    mesajlar: list | None = None,
) -> bool:
    """Alım/satım veya işlem mesajı var mı?"""
    if giris or cikis:
        return bool(giris or cikis)
    return bool(mesajlar)


def _cizgi(style: dict) -> str:
    return style["emoji"] + " " + style["char"] * style["width"]


def header(strateji: str, zaman: Optional[str] = None) -> str:
    s = STYLES.get(strateji, STYLES["P1"])
    return (
        f"{_cizgi(s)}\n"
        f"*{s['title']}*\n"
        f"{_cizgi(s)}\n"
        f"📅 {zaman or simdi()}\n"
    )


def footer(strateji: str) -> str:
    return _cizgi(STYLES.get(strateji, STYLES["P1"]))


def _islem_satirlari(giris: list | None, cikis: list | None) -> list[str]:
    lines: list[str] = []
    if giris:
        lines.append(f"🟢 *Alım:* {', '.join(f'`{s}`' for s in giris)}")
    if cikis:
        lines.append(f"⬜ *Satım:* {', '.join(f'`{s}`' for s in cikis)}")
    return lines


def _md(text: str) -> str:
    """HTML <b> etiketlerini Markdown'a çevir (Telegram parse_mode=Markdown)."""
    return text.replace("<b>", "*").replace("</b>", "*")


def _detay_satirlari(mesajlar: list[str] | None) -> list[str]:
    """STOP/TP1/TRAILING/MAX GÜN gibi işlem detaylarını mesaja ekle.

    Bu olmadan mesajlar yalnızca gönderim/gönderilmeme kararı için
    kullanılıyor, gerçek sebep (neden satıldığı) hiç görünmüyordu.
    """
    if not mesajlar:
        return []
    lines = ["🔔 *İşlem Detayı:*"]
    for m in mesajlar:
        for parca in _md(m).split("\n"):
            parca = parca.strip()
            if parca:
                lines.append(f"  {parca}")
    return lines


def _portfoy_istatistik(portfoy: dict) -> tuple[int, str, str, list[str]]:
    """n_pos, wr_str, pnl_str, acik_satirlar"""
    pozlar = portfoy.get("positions", portfoy.get("pozisyonlar", {}))
    trades = portfoy.get("trade_history", portfoy.get("history", portfoy.get("closed_trades", [])))
    n_pos = len(pozlar)

    if trades:
        pnls = [t.get("pnl_pct", t.get("pnl", 0)) for t in trades]
        pnls = [p for p in pnls if p is not None and p == p]
        wins = sum(1 for p in pnls if p > 0)
        wr_str = f"{wins / len(pnls):.0%}" if pnls else "—"
        pnl_str = f"{sum(pnls):+.1f}%" if pnls else "—"
    elif portfoy.get("baslangic") and portfoy.get("equity"):
        bas = float(portfoy["baslangic"])
        eq = float(portfoy["equity"])
        pnl_str = f"{(eq - bas) / bas * 100:+.1f}%" if bas else "—"
        wr_str = "—"
    else:
        wr_str = "—"
        pnl_str = "—"

    acik: list[str] = []
    detay = portfoy.get("acik_detay", [])
    if detay:
        for d in detay[:8]:
            sym = d.get("symbol", "?")
            pnl = d.get("pnl_pct")
            if pnl is not None:
                acik.append(f"  • `{sym}` {pnl:+.1f}%")
            else:
                acik.append(f"  • `{sym}`")
    elif pozlar:
        for sym, pos in list(pozlar.items())[:8]:
            if isinstance(pos, dict):
                pnl = pos.get("pnl_pct")
                if pnl is not None:
                    acik.append(f"  • `{sym}` {pnl:+.1f}%")
                elif pos.get("giris_f"):
                    acik.append(f"  • `{sym}` @ {pos['giris_f']:.2f}")
                else:
                    acik.append(f"  • `{sym}`")
            else:
                acik.append(f"  • `{sym}`")
    if not acik:
        acik.append("  _Açık pozisyon yok_")

    return n_pos, wr_str, pnl_str, acik


def _sinyal_satirlari(sinyaller: list[dict], limit: int = 8) -> list[str]:
    if not sinyaller:
        return ["📭 _Sinyal yok_"]
    lines = ["📈 *Sinyaller:*"]
    for s in sinyaller[:limit]:
        sym = s.get("symbol", "?")
        skor = s.get("score", s.get("final_score", s.get("score_count", 0)))
        strat = s.get("strategies", s.get("kaynaklar", []))
        if isinstance(strat, list):
            strat_s = ", ".join(str(x) for x in strat[:3])
        else:
            strat_s = str(strat) if strat else ""
        if isinstance(skor, (int, float)):
            lines.append(f"  • `{sym:<8}` skor={skor:+.3f}  _{strat_s}_")
        else:
            lines.append(f"  • `{sym:<8}` _{strat_s}_")
    return lines


def yukle_p1_sinyaller() -> list[dict]:
    path = BASE_DIR / "tarama_listesi.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("signals", [])
        except Exception:
            pass
    return []


def yukle_p2_sinyaller() -> list[dict]:
    path = BASE_DIR / "tarama_listesi_p2.json"
    if not path.exists():
        path = BASE_DIR / "state_p2.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            t = data.get("tarama", data)
            sigs = t.get("signals", [])
            out = []
            for s in sigs:
                ai = s.get("ai_analysis", {})
                out.append({
                    "symbol": s.get("symbol", "?"),
                    "score": ai.get("score", s.get("score", 0)),
                    "verdict": ai.get("verdict", ""),
                    "signals": [x.get("type", "") for x in s.get("signals", [])[:2]],
                })
            return out
        except Exception:
            pass
    return []


def strateji_mesaj(
    strateji: str,
    sinyaller: list[dict],
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    ekstra: list[str] | None = None,
    mesajlar: list[str] | None = None,
    zaman: Optional[str] = None,
) -> str:
    """Tüm stratejiler için standart işlem mesajı."""
    zaman = zaman or simdi()
    lines = [header(strateji, zaman)]
    lines.extend(_islem_satirlari(giris, cikis))
    if lines[-1] != header(strateji, zaman).split("\n")[-1]:
        lines.append("")
    detay = _detay_satirlari(mesajlar)
    if detay:
        lines.extend(detay)
        lines.append("")
    lines.extend(_sinyal_satirlari(sinyaller))
    lines.append("")
    n_pos, wr_str, pnl_str, acik = _portfoy_istatistik(portfoy)
    lines.append(f"💼 *Açık pozisyon ({n_pos}):*")
    lines.extend(acik)
    lines.append(f"📊 *WR:* {wr_str}  |  *P&L:* {pnl_str}")
    if ekstra:
        lines.append("")
        lines.extend(ekstra)
    lines.extend(["", footer(strateji)])
    return "\n".join(lines)


def p1_mesaj(
    sinyaller: list[dict],
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    mesajlar: list[str] | None = None,
    zaman: Optional[str] = None,
) -> str:
    return strateji_mesaj("P1", sinyaller, portfoy, giris=giris, cikis=cikis,
                           mesajlar=mesajlar, zaman=zaman)


def p2_mesaj(
    sinyaller: list[dict],
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    mesajlar: list[str] | None = None,
    zaman: Optional[str] = None,
) -> str:
    return strateji_mesaj("P2", sinyaller, portfoy, giris=giris, cikis=cikis,
                           mesajlar=mesajlar, zaman=zaman)


def p3_mesaj(
    top_longs: list,
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    monitor_alert: bool = False,
    corr_risk: str = "LOW",
    mesajlar: list[str] | None = None,
    zaman: Optional[str] = None,
) -> str:
    sinyaller = []
    for s in top_longs[:8]:
        if hasattr(s, "symbol"):
            sinyaller.append({
                "symbol": s.symbol,
                "score": s.score,
                "strategies": [f"marj={s.crossover_margin:+.3f}"],
            })
        else:
            sinyaller.append({
                "symbol": s.get("symbol", "?"),
                "score": s.get("score", 0),
                "strategies": [f"marj={s.get('crossover_margin', 0):+.3f}"],
            })
    ekstra = [
        f"🔬 Non-stat: {'⚠️ Drift' if monitor_alert else '✅ Stabil'}",
        f"🔗 Korelasyon: {corr_risk}",
    ]
    return strateji_mesaj(
        "P3", sinyaller, portfoy,
        giris=giris, cikis=cikis, ekstra=ekstra, mesajlar=mesajlar, zaman=zaman,
    )


def p4_mesaj(
    secilen: list[dict],
    portfoy: dict,
    ic_scores: dict[str, float],
    giris: list | None = None,
    cikis: list | None = None,
    mesajlar: list[str] | None = None,
    zaman: Optional[str] = None,
) -> str:
    sinyaller = [
        {
            "symbol": s.get("symbol", "?"),
            "score": s.get("meta_score", s.get("ic_agirlik", 0)),
            "strategies": s.get("kaynaklar", [s.get("kaynak", "")]),
        }
        for s in secilen[:8]
    ]
    ic_lines = ["🧠 *IC ağırlıkları:*"]
    for k, v in sorted(ic_scores.items(), key=lambda x: x[1], reverse=True):
        ic_lines.append(f"  {k}: {v:+.3f}")
    return strateji_mesaj(
        "P4", sinyaller, portfoy,
        giris=giris, cikis=cikis, ekstra=ic_lines, mesajlar=mesajlar, zaman=zaman,
    )


def p5_mesaj(
    secilen: list[dict],
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    elenen: list | None = None,
    mesajlar: list[str] | None = None,
    zaman: Optional[str] = None,
) -> str:
    sinyaller = [
        {
            "symbol": s.get("symbol", "?"),
            "score": s.get("komite_skor", s.get("score", 0)),
            "strategies": s.get("kaynaklar", []),
        }
        for s in secilen[:8]
    ]
    ekstra = []
    if elenen:
        ekstra.append("🚫 *Elenen (örnek):*")
        for e in elenen[:5]:
            ekstra.append(f"  `{e.get('symbol','?')}` — {e.get('neden','')}")
    return strateji_mesaj(
        "P5", sinyaller, portfoy,
        giris=giris, cikis=cikis, ekstra=ekstra or None, mesajlar=mesajlar, zaman=zaman,
    )


MAX_TG_LEN = 4000       # Telegram sınırı 4096 — güvenlik payıyla
RETRY_SAYISI = 3        # başarısız gönderim yeniden denenir
RETRY_BEKLE = 5         # denemeler arası saniye


def _parcala(mesaj: str) -> list[str]:
    """4096 karakter sınırını aşan mesajları satır sınırından böl."""
    if len(mesaj) <= MAX_TG_LEN:
        return [mesaj]
    parcalar, mevcut = [], ""
    for satir in mesaj.split("\n"):
        if len(mevcut) + len(satir) + 1 > MAX_TG_LEN:
            parcalar.append(mevcut)
            mevcut = satir
        else:
            mevcut = f"{mevcut}\n{satir}" if mevcut else satir
    if mevcut:
        parcalar.append(mevcut)
    return parcalar


def _tek_gonder(parca: str, token: str, chat_id: str, parse_mode: str) -> bool:
    """Tek parçayı RETRY_SAYISI denemeyle gönder; Markdown hatasında düz metne düş."""
    import time
    for deneme in range(1, RETRY_SAYISI + 1):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": parca, "parse_mode": parse_mode},
                timeout=15,
            )
            if r.ok:
                return True
            # 400: genellikle Markdown parse hatası — düz metin olarak dene
            if r.status_code == 400 and parse_mode:
                log.warning("Telegram 400 (%s) — düz metin deneniyor", r.text[:100])
                parse_mode = ""
                continue
            # 429: rate limit — Telegram'ın önerdiği süre kadar bekle
            if r.status_code == 429:
                try:
                    bekle = int(r.json().get("parameters", {}).get("retry_after", RETRY_BEKLE))
                except Exception:
                    bekle = RETRY_BEKLE
                log.warning("Telegram 429 — %ds bekleniyor (deneme %d/%d)", bekle, deneme, RETRY_SAYISI)
                time.sleep(bekle)
                continue
            log.warning("Telegram hata: %s %s (deneme %d/%d)",
                        r.status_code, r.text[:120], deneme, RETRY_SAYISI)
        except Exception as e:
            log.warning("Telegram exception: %s (deneme %d/%d)", e, deneme, RETRY_SAYISI)
        if deneme < RETRY_SAYISI:
            time.sleep(RETRY_BEKLE * deneme)
    log.error("Telegram: %d denemede gönderilemedi — mesaj kaybedildi:\n%s",
              RETRY_SAYISI, parca[:500])
    return False


def telegram_gonder(
    mesaj: str,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "Markdown",
) -> bool:
    """
    Güvenilir Telegram gönderimi:
      - 3 deneme (ağ hatası, 5xx)
      - 429 rate-limit'te retry_after kadar bekleme
      - Markdown parse hatasında (400) düz metne düşme
      - 4096 karakter üstünde otomatik parçalama
    """
    token = token or get_token()
    chat_id = chat_id or get_chat_id()
    if not token or not chat_id:
        log.warning("Telegram token/chat_id eksik — mesaj gönderilemedi")
        return False
    return all(_tek_gonder(p, token, chat_id, parse_mode) for p in _parcala(mesaj))


def telegram_islem_gonder(
    strateji: str,
    sinyaller: list[dict],
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    mesajlar: list | None = None,
    **kwargs,
) -> bool:
    """İşlem yoksa gönderme; varsa standart formatta gönder."""
    if not islem_var(giris=giris, cikis=cikis, mesajlar=mesajlar):
        log.info("[%s] İşlem yok — Telegram atlandı", strateji)
        return False
    builders = {"P1": p1_mesaj, "P2": p2_mesaj, "P3": p3_mesaj, "P4": p4_mesaj, "P5": p5_mesaj}
    fn = builders.get(strateji, p1_mesaj)
    if strateji == "P3":
        msg = fn(
            kwargs.get("top_longs", sinyaller),
            portfoy, giris=giris, cikis=cikis,
            monitor_alert=kwargs.get("monitor_alert", False),
            corr_risk=kwargs.get("corr_risk", "LOW"),
            mesajlar=mesajlar,
        )
    elif strateji == "P4":
        msg = fn(
            kwargs.get("secilen", sinyaller),
            portfoy, ic_scores=kwargs.get("ic_scores", {}),
            giris=giris, cikis=cikis, mesajlar=mesajlar,
        )
    elif strateji == "P5":
        msg = p5_mesaj(
            secilen=kwargs.get("secilen", sinyaller),
            portfoy=portfoy,
            giris=giris, cikis=cikis,
            elenen=kwargs.get("elenen", []),
            mesajlar=mesajlar,
        )
    else:
        msg = fn(sinyaller, portfoy, giris=giris, cikis=cikis, mesajlar=mesajlar)
    return telegram_gonder(msg)
