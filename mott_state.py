"""
mott_state.py — Ortak state normalizasyon katmanı
==================================================
5 stratejinin state dosyaları tarihsel nedenlerle farklı şemalar kullanır:

  P1  portfoy.json           pozisyonlar{giris_f, giris_t...}, islem yok listesi
  P2  portfoy_p2.json        pozisyonlar{giris_f, giris_t...}, islem_gecmisi[]
  P3  portfolio_state.json   positions{entry_price, entry_date}, history[]
  P4  state_p4.json          pozisyonlar{giris_fiyat, giris_tarih}, trade_history[]
  P5  state_p5.json          pozisyonlar{giris_fiyat, giris_tarih}, trade_history[]

Bu modül hepsini TEK ortak şemaya çevirir; raporlama araçları
(mott_aylik_rapor, mott_performans_analiz vb.) strateji başına özel kod
yazmak yerine `normalize(kod)` çağırır.

Ortak şema:
    {
      "strateji": "P3",
      "dosya": "portfolio_state.json",
      "pozisyonlar": [
        {"symbol", "giris_fiyat", "guncel_fiyat", "pnl_pct", "giris_tarih", "lot"}
      ],
      "islem_gecmisi": [
        {"symbol", "giris_fiyat", "cikis_fiyat", "pnl_pct", "neden",
         "giris_tarih", "cikis_tarih"}
      ],
      "nakit": float | None,
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BASE = Path(os.environ.get("MOTT_BASE_DIR", "."))

DOSYALAR = {
    "P1": "portfoy.json",
    "P2": "portfoy_p2.json",
    "P3": "portfolio_state.json",
    "P4": "state_p4.json",
    "P5": "state_p5.json",
}


def _yukle(fname: str) -> dict:
    path = BASE / fname
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sayi(v, varsayilan=None):
    """NaN ve None'a dayanıklı float dönüşümü."""
    try:
        f = float(v)
        return f if f == f else varsayilan  # NaN != NaN
    except (TypeError, ValueError):
        return varsayilan


def _tarih_iso(t: str) -> str:
    """'03.07.2026 13:23' veya '2026-07-03' → '2026-07-03'."""
    t = (t or "").split(" ")[0]
    if "." in t:
        p = t.split(".")
        if len(p) == 3:
            return f"{p[2]}-{p[1]}-{p[0]}"
    return t


def _p1_p2(kod: str) -> dict:
    d = _yukle(DOSYALAR[kod])
    pozlar = []
    for sym, p in d.get("pozisyonlar", {}).items():
        giris = _sayi(p.get("giris_f"))
        guncel = _sayi(p.get("guncel_f"), giris)
        pnl = None
        if giris and guncel:
            pnl = round((guncel - giris) / giris * 100, 2)
        pozlar.append({
            "symbol": sym,
            "giris_fiyat": giris,
            "guncel_fiyat": guncel,
            "pnl_pct": pnl,
            "giris_tarih": _tarih_iso(p.get("giris_t", "")),
            "lot": p.get("lotlar"),
        })
    gecmis = []
    for t in d.get("islem_gecmisi", []):
        gecmis.append({
            "symbol": t.get("symbol", t.get("sym", "")),
            "giris_fiyat": _sayi(t.get("giris_f", t.get("giris_fiyat"))),
            "cikis_fiyat": _sayi(t.get("cikis_f", t.get("cikis_fiyat"))),
            "pnl_pct": _sayi(t.get("pnl_pct", t.get("kar_pct"))),
            "neden": t.get("neden", t.get("sebep", "")),
            "giris_tarih": _tarih_iso(t.get("giris_t", t.get("giris_tarih", ""))),
            "cikis_tarih": _tarih_iso(t.get("cikis_t", t.get("cikis_tarih", ""))),
        })
    return {
        "strateji": kod,
        "dosya": DOSYALAR[kod],
        "pozisyonlar": pozlar,
        "islem_gecmisi": gecmis,
        "nakit": _sayi(d.get("nakit")),
    }


def _p3() -> dict:
    d = _yukle(DOSYALAR["P3"])
    pozlar = []
    for sym, p in d.get("positions", {}).items():
        giris = _sayi(p.get("entry_price"))
        guncel = _sayi(p.get("current_price"), giris)
        pnl = None
        if giris and guncel:
            pnl = round((guncel - giris) / giris * 100, 2)
        pozlar.append({
            "symbol": sym,
            "giris_fiyat": giris,
            "guncel_fiyat": guncel,
            "pnl_pct": pnl,
            "giris_tarih": _tarih_iso(p.get("entry_date", "")),
            "lot": None,
        })
    gecmis = []
    for t in d.get("history", []):
        pnl = _sayi(t.get("pnl_pct"))
        if pnl is None:
            continue  # bozuk (NaN) kayıtları rapora sokma
        gecmis.append({
            "symbol": t.get("symbol", ""),
            "giris_fiyat": _sayi(t.get("entry_price")),
            "cikis_fiyat": _sayi(t.get("exit_price")),
            "pnl_pct": pnl,
            "neden": t.get("reason", t.get("neden", "")),
            "giris_tarih": _tarih_iso(t.get("entry_date", "")),
            "cikis_tarih": _tarih_iso(t.get("exit_date", "")),
        })
    return {
        "strateji": "P3",
        "dosya": DOSYALAR["P3"],
        "pozisyonlar": pozlar,
        "islem_gecmisi": gecmis,
        "nakit": None,
    }


def _p4_p5(kod: str) -> dict:
    d = _yukle(DOSYALAR[kod])
    pozlar = []
    for sym, p in d.get("pozisyonlar", {}).items():
        pozlar.append({
            "symbol": sym,
            "giris_fiyat": _sayi(p.get("giris_fiyat")),
            "guncel_fiyat": _sayi(p.get("guncel_fiyat")),
            "pnl_pct": _sayi(p.get("pnl_pct")),
            "giris_tarih": _tarih_iso(p.get("giris_tarih", "")),
            "lot": p.get("lotlar"),
        })
    gecmis = []
    for t in d.get("trade_history", []):
        gecmis.append({
            "symbol": t.get("symbol", ""),
            "giris_fiyat": _sayi(t.get("giris_fiyat")),
            "cikis_fiyat": _sayi(t.get("cikis_fiyat")),
            "pnl_pct": _sayi(t.get("pnl_pct")),
            "neden": t.get("neden", ""),
            "giris_tarih": _tarih_iso(t.get("giris_tarih", "")),
            "cikis_tarih": _tarih_iso(t.get("cikis_tarih", "")),
        })
    return {
        "strateji": kod,
        "dosya": DOSYALAR[kod],
        "pozisyonlar": pozlar,
        "islem_gecmisi": gecmis,
        "nakit": _sayi(d.get("sermaye_mevcut")),
    }


def normalize(kod: str) -> dict:
    """Bir stratejinin state'ini ortak şemada döndür. kod: P1..P5."""
    kod = kod.upper()
    if kod in ("P1", "P2"):
        return _p1_p2(kod)
    if kod == "P3":
        return _p3()
    if kod in ("P4", "P5"):
        return _p4_p5(kod)
    raise ValueError(f"Bilinmeyen strateji: {kod}")


def hepsi() -> dict[str, dict]:
    """Tüm stratejilerin normalize edilmiş state'leri."""
    return {kod: normalize(kod) for kod in DOSYALAR}


if __name__ == "__main__":
    print(json.dumps(hepsi(), indent=2, ensure_ascii=False))
