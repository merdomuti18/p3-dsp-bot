# -*- coding: utf-8 -*-
"""Ortak risk kuralları — tüm kitaplar (P1–P5) için tek kaynak.

Üç kural:
  1. Cooldown: Bir hisse STOP/TP ile kapandıysa N gün boyunca aynı kitapta
     yeniden alınmaz (DAGI/KRDMA/ICBCT tarzı zincir zararları keser).
  2. Kitaplar arası maruziyet: Bir hisse zaten MAX_KITAP kitapta açıksa
     yeni kitapta alınmaz ("5 bağımsız portföy"ün aynı long basket'e
     dönüşmesini engeller).
  3. Rejim kapısı: Makro karar DIKKATLI iken tek turda en fazla 2 yeni
     pozisyon, GIRME iken 0.

Ayarlar env ile değiştirilebilir: MOTT_COOLDOWN_GUN, MOTT_MAX_KITAP.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

BASE_DIR = Path(os.environ.get("MOTT_BASE_DIR", str(Path(__file__).parent)))

COOLDOWN_GUN = int(os.environ.get("MOTT_COOLDOWN_GUN", "3"))
MAX_KITAP    = int(os.environ.get("MOTT_MAX_KITAP", "2"))

# Cooldown'u tetikleyen çıkış nedenleri. MAX_GUN nötr bir çıkış olduğu
# için yeniden girişe engel değildir.
COOLDOWN_NEDENLER = {"STOP", "TP", "TP1", "TRAILING", "ACIL_STOP", "ACIL_NAKIT"}

# kitap -> (dosya, açık pozisyon anahtarı)
_KITAP_DOSYALARI = {
    "P1": ("portfoy.json",          "pozisyonlar"),
    "P2": ("portfoy_p2.json",       "pozisyonlar"),
    "P3": ("portfolio_state.json",  "positions"),
    "P4": ("state_p4.json",         "pozisyonlar"),
    "P5": ("state_p5.json",         "pozisyonlar"),
}


def _json_oku(path: Path) -> dict:
    try:
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as exc:
        log.debug("mott_risk json okunamadı %s: %s", path, exc)
    return {}


def _cikis_tarihi(trade: dict):
    """Trade kaydından çıkış tarihini çöz (ISO veya dd.mm.yyyy)."""
    raw = str(trade.get("cikis_tarih") or trade.get("exit_date") or "").strip()
    if not raw:
        return None
    raw = raw[:10]
    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass
    try:
        return datetime.strptime(raw, "%d.%m.%Y").date()
    except ValueError:
        return None


def cooldown_da(trade_history: list | None, symbol: str,
                gun: int | None = None,
                nedenler: set | None = COOLDOWN_NEDENLER) -> bool:
    """Sembol son `gun` gün içinde (STOP/TP vb. ile) kapandıysa True.

    nedenler=None -> tüm çıkış nedenleri cooldown sayılır (P3 gibi
    neden alanı olmayan history'ler için).
    """
    gun = COOLDOWN_GUN if gun is None else gun
    bugun = date.today()
    for t in reversed(trade_history or []):
        if t.get("symbol") != symbol:
            continue
        neden = str(t.get("neden") or t.get("reason") or "").upper()
        if nedenler is not None and neden and neden not in nedenler:
            continue
        ct = _cikis_tarihi(t)
        if ct is not None and (bugun - ct).days < gun:
            return True
    return False


def kitaplar_arasi_sayi(symbol: str, haric: str | None = None) -> int:
    """Sembolün şu anda kaç kitapta açık pozisyon olduğu (haric dışında)."""
    n = 0
    for kitap, (dosya, anahtar) in _KITAP_DOSYALARI.items():
        if kitap == haric:
            continue
        data = _json_oku(BASE_DIR / dosya)
        if symbol in (data.get(anahtar) or {}):
            n += 1
    return n


def kitap_limiti_asildi(symbol: str, haric: str | None = None) -> bool:
    """Sembol zaten MAX_KITAP (veya daha fazla) kitapta açıksa True."""
    sayi = kitaplar_arasi_sayi(symbol, haric=haric)
    if sayi >= MAX_KITAP:
        log.info("mott_risk: %s zaten %d kitapta açık (limit %d) — atlanıyor",
                 symbol, sayi, MAX_KITAP)
        return True
    return False


def rejim_slot_limiti(makro_karar: str) -> int | None:
    """Tek turda açılabilecek yeni pozisyon limiti.

    GIRME -> 0, DIKKATLI -> 2, NORMAL/diğer -> None (limitsiz).
    """
    if makro_karar == "GIRME":
        return 0
    if makro_karar == "DIKKATLI":
        return 2
    return None


def makro_karar_oku() -> str:
    """son_durum.json'dan güncel makro kararı (yoksa NORMAL)."""
    d = _json_oku(BASE_DIR / "son_durum.json")
    return d.get("makro_karar", "NORMAL")
