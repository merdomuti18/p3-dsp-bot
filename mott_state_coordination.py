# -*- coding: utf-8 -*-
"""
mott_state_coordination.py — FAZ 6.2 State Coordination
========================================================
Tek kaynak: atomik yazma, _gen sayacı, _updated_at tazeliği,
çapraz kitap bayat okuma koruması.

FAZ 6.2 Pipeline:
  M1a  → P5 workflow'a state_p4.json checkout eklendi
  M1b  → Tum state dosyalarina _gen + _updated_at eklendi
  M1c  → kitaplar_arasi_taze_sayi() bayat state korumasi

Backward-compat: var olan `kitaplar_arasi_sayi()` DEGISMEZ;
 yeni `kitaplar_arasi_taze_sayi()` TAZE state'leri sayar.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, date, timedelta
from pathlib import Path

import pytz

log = logging.getLogger(__name__)

BASE_DIR = Path(os.environ.get("MOTT_BASE_DIR", str(Path(__file__).parent)))
IST = pytz.timezone("Europe/Istanbul")

# --- Staleness threshold ---
STALENESS_THRESHOLD_HOURS = int(os.environ.get("MOTT_STALENESS_HOURS", "14"))


# ── TSI (Turkey Standard Time) helpers ──────────────────────────────────────

def now_tsi() -> datetime:
    """Current time in Europe/Istanbul."""
    return datetime.now(IST)


def bugun_tsi() -> date:
    """Today's date in Europe/Istanbul."""
    return now_tsi().date()


# ── Atomic Write ─────────────────────────────────────────────────────────────

def atomic_write_json(path: Path, data: dict) -> None:
    """Atomik JSON yazma: temp dosyasina yaz, flush, close, rename.

    Crash-safe: os.replace() POSIX'te atomiktir. Yarim yazilmis dosya
    olusmaz; ya eski dosya kalir ya da yeni dosya tam olarak visible olur.
    """
    dir_ = path.parent
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=str(dir_))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        # Temizlik: basarisiz temp dosyasini sil
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── State Stamp (FAZ 6.2 — _gen + _updated_at) ─────────────────────────────

def stamp_state(state: dict) -> dict:
    """State dict'ine _gen sayaci + _updated_at TSİ timestamp'i ekle/artir.

    Callers: her state_kaydet() / save_state() fonksiyonu SONUNDA cagrilir.

    _gen: Her yazma dongusunde 1 artar. 0 veya eksik = onceki surumden kalma.
    _updated_at: ISO-8601 TSİ timestamp (Europe/Istanbul).
    """
    state["_gen"] = state.get("_gen", 0) + 1
    state["_updated_at"] = now_tsi().isoformat()
    return state


# ── Freshness Guard ─────────────────────────────────────────────────────────

def is_state_fresh(
    state: dict,
    threshold_hours: int | None = None,
) -> bool:
    """State'in _updated_at'i son `threshold_hours` icinde mi?

    * _updated_at yoksa veya parse edilemiyorsa → STALE (güvenli taraf).
    * _gen yoksa veya 0 ise → STALE (onceki surumden kalma varsayilir).

    Threshold varsayilan: STALENESS_THRESHOLD_HOURS (default 14).
    """
    if threshold_hours is None:
        threshold_hours = STALENESS_THRESHOLD_HOURS

    gen = state.get("_gen", 0)
    if not gen or gen < 1:
        return False

    raw_ts = state.get("_updated_at", "")
    if not raw_ts:
        return False

    try:
        ts = datetime.fromisoformat(str(raw_ts))
        # TSİ timezone-aware; comparisons icin normalize et
        if ts.tzinfo is None:
            ts = IST.localize(ts)
        now = now_tsi()
        diff = now - ts
        return diff <= timedelta(hours=threshold_hours)
    except (ValueError, TypeError):
        return False


# ── Cross-Book Fresh Read (M1c) ─────────────────────────────────────────────

def kitaplar_arasi_taze_sayi(
    symbol: str,
    kitap_dosyalari: dict,
    haric: str | None = None,
    threshold_hours: int | None = None,
) -> tuple[int, list[str]]:
    """Sembolun kac kitapta acik pozisyon oldugu (FAZ 6.2 — bayat-safe).

    DAVRANIS:
      * Taze kitap → sayac artar (dogru veri).
      * Bayat kitap → sayac artar (konservatif: bilinmeyen risk = dolu varsayilir).
      * Eksik/dosyasi-olmayan kitap → sayac artmaz (dosya yok = pozisyon yok).

    Neden "hariç tutma" degil: Eger bayat bir kitapta X'in pozisyonu olup
    olmadigi bilinmiyorsa, o kitap "dolu" gibi muamele gorur. Aksi halde
    tazelenme aninda MAX_KITAP ihlali olusur (N1 invariant kırılır).

    Args:
        symbol: Aranan hisse sembolu
        kitap_dosyalari: mott_risk._KITAP_DOSYALARI formatinda
            {kitap_adi: (dosya_adi, pozisyon_anahtari)}
        haric: Bu kitap sayilmaz (orn. P4 kendi kitabini haric birakir)
        threshold_hours: Tazellik esigi (saat). None -> STALENESS_THRESHOLD_HOURS

    Returns:
        (taze_sayi, bayat_kitaplar): (kac kitapta acik, bayat olanlarin listesi)
    """
    taze = 0
    bayat = []
    for kitap, (dosya, anahtar) in kitap_dosyalari.items():
        if kitap == haric:
            continue
        try:
            state_path = BASE_DIR / dosya
            if not state_path.exists():
                continue
            with open(state_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if symbol not in (data.get(anahtar) or {}):
                continue
            # Sembol bu kitapta acik — dogrudan say.
            taze += 1
            if not is_state_fresh(data, threshold_hours):
                bayat.append(kitap)
                log.info(
                    "mott_risk taze: %s acik ama %s state'i BAYAT "
                    "(son guncelleme: %s) — sayildi (konservatif)",
                    symbol, kitap, data.get("_updated_at", "bilinmiyor"),
                )
        except Exception as exc:
            log.debug("mott_risk taze: %s/%s okunamadi: %s", kitap, dosya, exc)
    return taze, bayat
