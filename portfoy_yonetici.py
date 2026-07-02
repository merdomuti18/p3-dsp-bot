# -*- coding: utf-8 -*-
"""
BIST paper trading + portfolio manager.

Gün içi uygulama akışı:
  09:00 -> dünkü sinyaller + makro/VIOP/model -> "bugün şunları al" mesajı
  11:00 -> ilk alım denemesi
  11:30 -> portföy özeti
  12:00-17:00 -> saatlik risk kontrolü + tekrar alım denemesi
  17:30 -> kapanış özeti
  20:50 -> Cloud tarama (değişmez)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")

try:
    _tz_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "yf_tz"
    yf.set_tz_cache_location(str(_tz_cache))
except Exception:
    pass


def _ticker(sym: str):
    return yf.Ticker(sym)


def _market_ticker(sym: str):
    return yf.Ticker(sym)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("portfoy_yonetici.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

BOT_TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_TOKEN", ""))
CHAT_ID       = os.environ.get("TELEGRAM_CHAT_ID", "")
# FLASK_PORT kaldırıldı — GitHub Actions Flask sunucusu kullanmaz
FLASK_API_KEY = os.environ.get("FLASK_API_KEY", "")
VIOP_TICKER   = os.environ.get("VIOP_TICKER", "XU030.IS")

BASE_DIR             = Path(__file__).parent
PORTFOY_FILE         = BASE_DIR / "portfoy.json"
TARAMA_FILE          = BASE_DIR / "tarama_listesi.json"
LGBM_MODEL_FILE      = BASE_DIR / "lgbm_model.pkl"
LGBM_META_FILE       = BASE_DIR / "lgbm_ozellikler.json"
DURUM_FILE           = BASE_DIR / "son_durum.json"
SIGNAL_AUDIT_FILE    = BASE_DIR / "signal_audit.jsonl"
PORTFOY_AUDIT_FILE   = BASE_DIR / "portfolio_actions.jsonl"
HISSELER_FILE        = BASE_DIR / "hisseler.txt"
TELEGRAM_OFFSET_FILE = BASE_DIR / "telegram_offset.txt"
PORTFOY_P2_FILE       = BASE_DIR / "portfoy_p2.json"
TARAMA_P2_FILE        = BASE_DIR / "tarama_listesi_p2.json"
SIGNAL_AUDIT_P2_FILE  = BASE_DIR / "signal_audit_p2.jsonl"
PORTFOY_AUDIT_P2_FILE = BASE_DIR / "portfolio_actions_p2.jsonl"


SERMAYE_BASLANGIC        = 100_000
P2_SERMAYE_BASLANGIC = 100_000
P2_MAX_HISSE         = 7
P2_HISSE_LIMIT       = 20_000
P2_STOP_PCT          = -0.05
P2_TP1_PCT           = 0.08
P2_TRAILING_PCT      = -0.05
P2_MAX_GUN           = 10
P2_MIN_SCORE         = 3.0
P2_TEYIT_BONUS_ESIK  = 2.0

MAX_HISSE                = 7
HEDEF_HISSE              = 5
HISSE_LIMIT              = 20_000
STOP_PCT                 = -0.05
TP1_PCT                  = 0.08
TRAILING_PCT             = -0.05
MAX_GUN                  = 10
LGBM_MIN_SKOR            = 60
WAITING_EXPIRES_HOUR     = 17
EMERGENCY_LIQUIDATION_SCORE = 80
RETRY_REASONS = {"veri_yok", "exception", "lot_yetersiz", "nakit_yetersiz"}


def _elde_tutma_gunu(giris_t: str) -> int:
    """Pozisyonun kaç takvim günüdür açık olduğunu giriş tarihinden hesapla.

    Not: pos['gun'] sayacı yalnızca main() içindeki gün-sonu döngüsünde
    artıyordu; GitHub Actions bu döngüyü hiç çalıştırmadığı için MAX_GUN
    çıkışı hiçbir zaman tetiklenmiyordu. Bunun yerine giriş tarihinden
    itibaren geçen gerçek gün sayısını hesaplıyoruz — kontrol sıklığından
    (saatlik / 15 dk) bağımsız, her zaman doğru sonuç verir.
    """
    try:
        giris_tarih = datetime.strptime(giris_t.split(" ")[0], "%d.%m.%Y").date()
        return (date.today() - giris_tarih).days
    except Exception:
        return 0

# ── LGBM global (uygulama başında bir kez yüklenir) ──────────────────────────
_LGBM_MODEL: object  = None
_LGBM_STATUS: str    = "pasif"

STRATEGY_WEIGHTS = {
    "GT":    18, "ZT3":   18, "ALPHA": 16,
    "GTD":   14, "ZKN":   14, "DIP":   12,
    "MR":    10, "KBM":    8,
}

MAKRO_KURALLAR = [
    ("^GSPC",      1, "down", -2.0, 15.0),
    ("^IXIC",      1, "down", -3.0, 12.0),
    ("^VIX",       1, "up",    3.0, 20.0),
    ("^N225",      0, "down", -2.0, 12.0),
    ("^HSI",        0, "down", -2.0, 10.0),
    ("000001.SS",  0, "down", -2.0,  8.0),
    ("GC=F",       1, "up",    2.0,  8.0),
]
ASYA_TICKERLAR = ["^N225", "^HSI", "000001.SS"]
ASYA_ESIK      = -2.0
ASYA_MIN_TETIK = 2


# ── Makro skor cache (60 dakika) ─────────────────────────────────────────────
_MAKRO_CACHE: dict = {"value": None, "fetched_at": None}
MAKRO_CACHE_TTL = 3600  # saniye


def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def sonraki_islem_gunu(ref_date: date) -> date:
    nxt = ref_date + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def lgbm_model_yukle():
    """Global cache'den döner — disk okuma sadece ilk çağrıda yapılır."""
    global _LGBM_MODEL, _LGBM_STATUS
    if _LGBM_STATUS != "pasif" or _LGBM_MODEL is not None:
        return _LGBM_MODEL, _LGBM_STATUS
    if not LGBM_MODEL_FILE.exists():
        _LGBM_STATUS = "pasif"
        return None, "pasif"
    try:
        with open(LGBM_MODEL_FILE, "rb") as fh:
            model = pickle.load(fh)
        if not hasattr(model, "predict_proba"):
            _LGBM_STATUS = "pasif"
            return None, "pasif"
        _LGBM_MODEL  = model
        _LGBM_STATUS = "aktif"
        log.info("LGBM model yuklendi (global cache)")
        return _LGBM_MODEL, _LGBM_STATUS
    except Exception as exc:
        log.warning("LGBM model yuklenemedi: %s", exc)
        _LGBM_STATUS = "pasif"
        return None, "pasif"


def guncel_makro_skoru() -> float:
    if not DURUM_FILE.exists():
        return 0.0
    try:
        with open(DURUM_FILE, encoding="utf-8") as fh:
            return float(json.load(fh).get("makro_skor", 0.0) or 0.0)
    except Exception:
        return 0.0


def guncel_fiyat(symbol: str):
    """
    Fiyat çekme — üç katmanlı fallback:
      1. Intraday 1m chart (piyasa açıkken en güncel)
      2. fast_info.last_price (hızlı canlı alan)
      3. Son günlük Close (her zaman çalışır)
    """
    ticker_obj = _ticker(f"{symbol}.IS")
    # 1. Intraday
    try:
        df = ticker_obj.history(period="1d", interval="1m")
        if len(df) >= 1:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    # 2. fast_info
    try:
        price = ticker_obj.fast_info.get("last_price") or ticker_obj.fast_info.get("lastPrice")
        if price and float(price) > 0:
            return float(price)
    except Exception:
        pass
    # 3. Günlük Close
    try:
        df = ticker_obj.history(period="5d", interval="1d")
        if len(df) >= 1:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


def saatlik_bar(symbol: str):
    try:
        df = _ticker(f"{symbol}.IS").history(period="5d", interval="60m")
        if len(df) < 2:
            return None
        bar = df.iloc[-2]
        return {"open": float(bar["Open"]), "high": float(bar["High"]),
                "low": float(bar["Low"]), "close": float(bar["Close"])}
    except Exception as exc:
        log.debug("saatlik_bar %s: %s", symbol, exc)
        return None




# ─────────────────────────────────────────────────────────────────────────────
# P2-SMC Portföy Fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────

def p2_portfoy_yukle() -> dict:
    if PORTFOY_P2_FILE.exists():
        with open(PORTFOY_P2_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"pozisyonlar": {}, "nakit": P2_SERMAYE_BASLANGIC, "baslangic": P2_SERMAYE_BASLANGIC}
    data.setdefault("bekleyen_al", [])
    data.setdefault("open_attempts_today", [])
    data.setdefault("last_hourly_check_time", "")
    return data


def p2_portfoy_kaydet(portfoy: dict):
    with open(PORTFOY_P2_FILE, "w", encoding="utf-8") as fh:
        json.dump(portfoy, fh, indent=2, ensure_ascii=False)


def p2_portfolio_preview(portfoy: dict) -> dict:
    nakit  = portfoy.get("nakit", 0)
    equity = nakit
    for sym, pos in portfoy.get("pozisyonlar", {}).items():
        f = guncel_fiyat(sym)
        equity += pos["lotlar"] * (f if f else pos["giris_f"])
    kazanc_pct = (equity - portfoy.get("baslangic", P2_SERMAYE_BASLANGIC)) / portfoy.get("baslangic", P2_SERMAYE_BASLANGIC) * 100
    return {
        "cash": round(nakit, 2), "equity": round(equity, 2),
        "n_positions": len(portfoy.get("pozisyonlar", {})),
        "baslangic": portfoy.get("baslangic", P2_SERMAYE_BASLANGIC),
        "kazanc_pct": round(kazanc_pct, 2),
    }


def p2_yeni_pozisyon_ac(portfoy: dict, adaylar: list, makro_karar: str) -> tuple:
    if makro_karar == "GIRME":
        return portfoy, [], [], [{"symbol": a["symbol"], "reason": "makro_girme"} for a in adaylar]
    mesajlar, alinan, alinmayan = [], [], []
    mevcut   = portfoy["pozisyonlar"]
    nakit    = portfoy["nakit"]
    bos_slot = max(0, P2_MAX_HISSE - len(mevcut))
    if bos_slot == 0 or not adaylar:
        return portfoy, [], [], []
    adaylar_s  = sorted(adaylar, key=lambda x: -x.get("final_score", 0))
    secilenler = [a for a in adaylar_s if a.get("final_score", 0) >= P2_MIN_SCORE][:bos_slot]
    toplam_skor = sum(max(a.get("final_score", 1), 1) for a in secilenler) or 1
    for aday in secilenler:
        sym = aday["symbol"]
        if sym in mevcut:
            alinmayan.append({"symbol": sym, "reason": "already_open"})
            continue
        try:
            giris_f = guncel_fiyat(sym)
            if giris_f is None:
                alinmayan.append({"symbol": sym, "reason": "veri_yok"})
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": "veri_yok"})
                continue
            alloc  = min(P2_HISSE_LIMIT, nakit * (aday.get("final_score", 1) / toplam_skor))
            lotlar = int(alloc / max(giris_f, 0.01))
            if lotlar < 1:
                alinmayan.append({"symbol": sym, "reason": "lot_yetersiz"})
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": "lot_yetersiz"})
                continue
            maliyet = lotlar * giris_f
            if maliyet > nakit:
                alinmayan.append({"symbol": sym, "reason": "nakit_yetersiz"})
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": "nakit_yetersiz"})
                continue
            nakit -= maliyet
            mevcut[sym] = {
                "giris_f": round(giris_f, 4), "giris_t": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "tepe_f": round(giris_f, 4), "lotlar": lotlar, "gun": 0, "tp1_yapildi": False,
                "smc_score": aday.get("score", 0), "teyit_skoru": aday.get("teyit_skoru", 0),
                "signals": aday.get("signals", []),
            }
            mesajlar.append(
                f"\U0001f6a8 <b>P2-AL - {sym}</b>\n"
                f"   {lotlar} lot @ {giris_f:.2f} TL\n"
                f"   SMC:{aday.get('score',0):.1f} teyit:+{aday.get('teyit_skoru',0):.1f}"
            )
            append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_success", "symbol": sym,
                                                   "price": giris_f, "lots": lotlar,
                                                   "smc_score": aday.get("score", 0)})
            alinan.append(sym)
        except Exception as exc:
            log.exception("P2 pozisyon HATA %s", sym)
            alinmayan.append({"symbol": sym, "reason": "exception"})
            append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "buy_failed", "symbol": sym, "reason": f"exception:{exc}"})
    portfoy["nakit"] = nakit
    return portfoy, mesajlar, alinan, alinmayan


def p2_pozisyon_kontrol(portfoy: dict) -> tuple:
    mesajlar, kapatilacak = [], []
    for sym, pos in list(portfoy["pozisyonlar"].items()):
        try:
            bar = saatlik_bar(sym)
            if bar is None:
                continue
            high, low, close = bar["high"], bar["low"], bar["close"]
            giris_f = pos["giris_f"]
            lotlar  = pos["lotlar"]
            pos["tepe_f"] = max(pos.get("tepe_f", giris_f), high)
            if (low - giris_f) / giris_f <= P2_STOP_PCT:
                cikis_f = round(giris_f * (1 + P2_STOP_PCT), 4)
                portfoy["nakit"] += lotlar * cikis_f
                kapatilacak.append(sym)
                mesajlar.append(f"\U0001f6d1 <b>P2-STOP - {sym}</b>\n   {giris_f:.2f} \u2192 {cikis_f:.2f}")
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "stop", "symbol": sym,
                                                      "exit_price": cikis_f, "return_pct": P2_STOP_PCT * 100})
                continue
            if (high - giris_f) / giris_f >= P2_TP1_PCT and not pos.get("tp1_yapildi"):
                yari = max(1, lotlar // 2)
                portfoy["nakit"] += yari * close
                pos["lotlar"]    -= yari
                pos["tp1_yapildi"] = True
                mesajlar.append(f"\U0001f3af <b>P2-TP1 - {sym}</b>\n   {yari} lot @ {close:.2f} (+{P2_TP1_PCT*100:.0f}%)")
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "tp1", "symbol": sym,
                                                      "price": close, "remaining": pos["lotlar"]})
            elif pos.get("tp1_yapildi"):
                trail_ret = (low - pos["tepe_f"]) / pos["tepe_f"]
                if trail_ret <= P2_TRAILING_PCT:
                    cikis_f = round(pos["tepe_f"] * (1 + P2_TRAILING_PCT), 4)
                    portfoy["nakit"] += pos["lotlar"] * cikis_f
                    kapatilacak.append(sym)
                    ret_g = (cikis_f - giris_f) / giris_f
                    mesajlar.append(f"\U0001f4c9 <b>P2-TRAIL - {sym}</b>\n   {cikis_f:.2f} | {ret_g*100:+.1f}%")
                    append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "trailing", "symbol": sym,
                                                          "exit_price": cikis_f, "return_pct": ret_g * 100})
                    continue
            if _elde_tutma_gunu(pos.get("giris_t", "")) >= P2_MAX_GUN:
                portfoy["nakit"] += pos["lotlar"] * close
                kapatilacak.append(sym)
                gun_ret = (close - giris_f) / giris_f
                mesajlar.append(f"\u23f0 <b>P2-MAXGUN - {sym}</b>\n   {close:.2f} | {gun_ret*100:+.1f}%")
                append_jsonl(PORTFOY_AUDIT_P2_FILE, {"event": "max_day", "symbol": sym,
                                                      "exit_price": close, "return_pct": gun_ret * 100})
        except Exception as exc:
            log.debug("P2 kontrol %s: %s", sym, exc)
    for sym in kapatilacak:
        portfoy["pozisyonlar"].pop(sym, None)
    return portfoy, mesajlar


def p2_ozet_mesaji(portfoy: dict, saat_label: str) -> str:
    prev = p2_portfolio_preview(portfoy)
    lines = [
        f"\U0001f7e3 <b>P2-SMC PORTFOY - {saat_label}</b>",
        "--------------------",
        f"\U0001f4b0 Baslangic : {prev['baslangic']:,.0f} TL",
        f"\U0001f4ca Guncel    : {prev['equity']:,.0f} TL ({prev['kazanc_pct']:+.1f}%)",
        f"\U0001f4b5 Nakit     : {prev['cash']:,.0f} TL",
    ]
    if portfoy["pozisyonlar"]:
        for sym, pos in portfoy["pozisyonlar"].items():
            f = guncel_fiyat(sym)
            ret = ((f - pos["giris_f"]) / pos["giris_f"] * 100) if f else 0
            tp1_tag = " TP1✓" if pos.get("tp1_yapildi") else ""
            lines.append(
                f"   \u2022 <b>{sym}</b> {pos['lotlar']}lot @ {pos['giris_f']:.2f}"
                f" \u2192 {f:.2f if f else '?'} ({ret:+.1f}%) gun:{pos.get('gun',0)}{tp1_tag}"
            )
    else:
        lines.append("   - Acik pozisyon yok")
    bekleyen = portfoy.get("bekleyen_al", [])
    if bekleyen:
        lines.append(f"\U0001f4cb Bekleyen: {', '.join(b['symbol'] for b in bekleyen[:7])}")
    return "\n".join(lines)


def _p2_acik_detay(portfoy: dict) -> list[dict]:
    detay = []
    for sym, pos in portfoy.get("pozisyonlar", {}).items():
        f = guncel_fiyat(sym)
        ret = ((f - pos["giris_f"]) / pos["giris_f"] * 100) if f and pos.get("giris_f") else pos.get("pnl_pct")
        detay.append({"symbol": sym, "pnl_pct": ret})
    return detay


def _p2_mott_portfoy(portfoy: dict) -> dict:
    prev = p2_portfolio_preview(portfoy)
    return {
        "pozisyonlar":   portfoy.get("pozisyonlar", {}),
        "trade_history": portfoy.get("trade_history", []),
        "baslangic":     portfoy.get("baslangic", P2_SERMAYE_BASLANGIC),
        "equity":        prev["equity"],
        "acik_detay":    _p2_acik_detay(portfoy),
    }


def _p2_telegram_islem(
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    mesajlar: list | None = None,
) -> None:
    """Yalnızca alım/satım varsa P2 formatında Telegram gönder."""
    try:
        from mott_telegram import telegram_islem_gonder, yukle_p2_sinyaller
        telegram_islem_gonder(
            "P2",
            sinyaller=yukle_p2_sinyaller(),
            portfoy=_p2_mott_portfoy(portfoy),
            giris=giris or [],
            cikis=cikis or [],
            mesajlar=mesajlar,
        )
    except Exception as exc:
        log.warning("P2 Telegram hatasi: %s", exc)


def p2_saatlik_kontrol(makro_karar: str):
    now = datetime.now()
    p2_teyit_senkronize_et()
    portfoy = p2_portfoy_yukle()
    onceki = set(portfoy["pozisyonlar"].keys())
    mesajlar = []
    if portfoy["pozisyonlar"]:
        portfoy, islem_msg = p2_pozisyon_kontrol(portfoy)
        mesajlar.extend(islem_msg)
    bekleyen = portfoy.get("bekleyen_al", [])
    if bekleyen and makro_karar != "GIRME":
        portfoy, al_msg, alinan, _ = p2_yeni_pozisyon_ac(portfoy, bekleyen, makro_karar)
        mesajlar.extend(al_msg)
        portfoy["bekleyen_al"] = [b for b in bekleyen if b["symbol"] not in set(alinan)]
    portfoy["last_hourly_check_time"] = now.strftime("%d.%m.%Y %H:%M")
    p2_portfoy_kaydet(portfoy)
    if mesajlar:
        sonra = set(portfoy["pozisyonlar"].keys())
        _p2_telegram_islem(
            portfoy,
            giris=list(sonra - onceki),
            cikis=list(onceki - sonra),
            mesajlar=mesajlar,
        )


def p2_gun_sonu_guncelle():
    portfoy = p2_portfoy_yukle()
    if portfoy["pozisyonlar"]:
        for pos in portfoy["pozisyonlar"].values():
            pos["gun"] = pos.get("gun", 0) + 1
        p2_portfoy_kaydet(portfoy)


def p2_adaylari_yukle_ve_hazirla():
    """
    Akşam SMC taramasından (state_p2.json) adayları oku,
    portfoy_p2.json'un bekleyen_al listesine aktar.

    scanner_smc.py yalnızca tarama yapar (state_p2.json yazar) — gerçek
    pozisyon açma/kapama işlemleri burada, portfoy_p2.json üzerinden yapılır.
    Bu köprü olmadan P2 hiçbir zaman gerçek işlem yapamaz.
    """
    state_file = BASE_DIR / "state_p2.json"
    if not state_file.exists():
        log.info("P2 aday hazırlama: state_p2.json yok")
        return
    try:
        with open(state_file, encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception as exc:
        log.warning("P2 state okunamadı: %s", exc)
        return

    sinyaller = state.get("tarama", {}).get("signals", [])
    portfoy = p2_portfoy_yukle()
    mevcut = set(portfoy["pozisyonlar"].keys())
    bekleyen = []
    for s in sinyaller:
        sym = s.get("symbol")
        if not sym or sym in mevcut:
            continue
        skor = float(s.get("score", 0))
        if skor < P2_MIN_SCORE:
            continue
        bekleyen.append({
            "symbol": sym,
            "score": skor,
            "final_score": skor,
            "verdict": s.get("verdict", ""),
            "signals": s.get("signals", []),
            "teyit_skoru": 0.0,
            "teyit_var": False,
            "queued_at": state.get("last_scan", ""),
            "attempt_count": 0,
            "last_attempt_reason": "",
        })
    portfoy["bekleyen_al"] = bekleyen
    portfoy["open_attempts_today"] = []
    p2_portfoy_kaydet(portfoy)
    log.info("P2 aday hazırlandı: %d aday bekleyen_al'a eklendi", len(bekleyen))


def p2_teyit_senkronize_et():
    """state_p2.json'daki (scanner_smc.py teyit modu) teyit skorlarını
    portfoy_p2.json'un bekleyen_al listesine aktarır."""
    state_file = BASE_DIR / "state_p2.json"
    if not state_file.exists():
        return
    try:
        with open(state_file, encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception:
        return
    teyit_map = {b["symbol"]: b for b in state.get("bekleyen_al", [])}
    if not teyit_map:
        return
    portfoy = p2_portfoy_yukle()
    for item in portfoy.get("bekleyen_al", []):
        t = teyit_map.get(item["symbol"])
        if t:
            item["teyit_skoru"] = t.get("teyit_skoru", 0)
            item["teyit_var"] = t.get("teyit_var", False)
            item["final_score"] = item.get("score", 0) + t.get("teyit_skoru", 0)
    p2_portfoy_kaydet(portfoy)

def portfoy_yukle() -> dict:
    if PORTFOY_FILE.exists():
        with open(PORTFOY_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"pozisyonlar": {}, "nakit": SERMAYE_BASLANGIC, "baslangic": SERMAYE_BASLANGIC}
    data.setdefault("bekleyen_al", [])
    data.setdefault("open_attempts_today", [])
    data.setdefault("last_open_attempt_summary", {})
    data.setdefault("last_hourly_check_time", "")
    return data


def portfoy_kaydet(portfoy: dict):
    with open(PORTFOY_FILE, "w", encoding="utf-8") as fh:
        json.dump(portfoy, fh, indent=2, ensure_ascii=False)


def tarama_listesi_yukle() -> dict:
    if TARAMA_FILE.exists():
        with open(TARAMA_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"signals": [], "scan_time": "", "scan_label": ""}
    if isinstance(data, list):
        return {"signals": [{"symbol": x, "score_count": 1, "strategies": [],
                             "scan_time": "", "scan_label": ""} for x in data]}
    if "signals" not in data and "semboller" in data:
        data["signals"] = [{"symbol": x, "score_count": 1, "strategies": [],
                            "scan_time": "", "scan_label": ""} for x in data["semboller"]]
    data.setdefault("signals", [])
    return data


def durum_kaydet(durum: dict):
    with open(DURUM_FILE, "w", encoding="utf-8") as fh:
        json.dump(durum, fh, indent=2, ensure_ascii=False)


def hisse_listesi_yukle() -> list:
    if not HISSELER_FILE.exists():
        return []
    try:
        return list(dict.fromkeys(
            line.strip().upper()
            for line in HISSELER_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ))
    except Exception as exc:
        log.warning("hisseler.txt okunamadi: %s", exc)
        return []


def hisse_listesi_kaydet(semboller: list) -> list:
    unique = list(dict.fromkeys(str(x).strip().upper() for x in semboller if str(x).strip()))
    HISSELER_FILE.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")
    return unique


def bekleyen_adayi_hazirla(aday: dict, ref_dt=None) -> dict:
    ref_dt = ref_dt or datetime.now()
    valid_for = sonraki_islem_gunu(ref_dt.date())
    hazir = dict(aday)
    hazir["queued_at"]           = ref_dt.strftime("%d.%m.%Y %H:%M")
    hazir["valid_for_date"]      = valid_for.isoformat()
    hazir["expires_at_hour"]     = WAITING_EXPIRES_HOUR
    hazir["attempt_count"]       = 0
    hazir["last_attempt_time"]   = ""
    hazir["last_attempt_reason"] = ""
    return hazir


def ayikla_suresi_dolan_bekleyenler(bekleyenler: list, now=None):
    now = now or datetime.now()
    active, expired = [], []
    current_date = now.date()
    current_hour = now.hour
    for item in bekleyenler:
        valid_for_str = item.get("valid_for_date")
        try:
            valid_for = date.fromisoformat(valid_for_str) if valid_for_str else current_date
        except Exception:
            valid_for = current_date
        expires_hour = item.get("expires_at_hour", WAITING_EXPIRES_HOUR)
        is_expired = (current_date > valid_for or
                      (current_date == valid_for and current_hour >= expires_hour))
        if is_expired:
            expired.append({"symbol": item.get("symbol", "?"), "reason": "expired"})
        else:
            active.append(item)
    return active, expired


def retry_bekleyenleri_filtrele(bekleyenler: list) -> list:
    return [b for b in bekleyenler
            if b.get("last_attempt_reason", "") in RETRY_REASONS
            or b.get("attempt_count", 0) == 0]


def _p1_acik_detay(portfoy: dict) -> list[dict]:
    detay = []
    for sym, pos in portfoy.get("pozisyonlar", {}).items():
        f = guncel_fiyat(sym)
        ret = ((f - pos["giris_f"]) / pos["giris_f"] * 100) if f and pos.get("giris_f") else None
        detay.append({"symbol": sym, "pnl_pct": ret})
    return detay


def _p1_mott_portfoy(portfoy: dict) -> dict:
    prev = portfolio_preview(portfoy)
    return {
        "pozisyonlar":   portfoy.get("pozisyonlar", {}),
        "trade_history": portfoy.get("trade_history", []),
        "baslangic":     portfoy.get("baslangic", SERMAYE_BASLANGIC),
        "equity":        prev["equity"],
        "acik_detay":    _p1_acik_detay(portfoy),
    }


def _p1_telegram_islem(
    portfoy: dict,
    giris: list | None = None,
    cikis: list | None = None,
    mesajlar: list | None = None,
) -> None:
    """Yalnızca alım/satım varsa P1 formatında Telegram gönder."""
    try:
        from mott_telegram import telegram_islem_gonder, yukle_p1_sinyaller
        telegram_islem_gonder(
            "P1",
            sinyaller=yukle_p1_sinyaller(),
            portfoy=_p1_mott_portfoy(portfoy),
            giris=giris or [],
            cikis=cikis or [],
            mesajlar=mesajlar,
        )
    except Exception as exc:
        log.warning("P1 Telegram hatasi: %s", exc)


def send_telegram(msg: str, parse_mode: str = "HTML"):
    from datetime import datetime
    import pytz
    _now = datetime.now(pytz.timezone("Europe/Istanbul"))
    _header = f"[PORTF\u00d6Y] {_now.strftime('%d.%m.%Y | %H:%M')}\n"
    msg = _header + msg
    if not BOT_TOKEN or not CHAT_ID:
        log.info("[TELEGRAM SIMULE]\n%s", msg)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": parse_mode,
                  "disable_web_page_preview": True},
            timeout=15,
        )
        if not r.ok:
            log.error("Telegram %s: %s", r.status_code, r.text[:200])
    except Exception as exc:
        log.error("Telegram hatasi: %s", exc)


def telegram_api_call(method: str, payload=None):
    if not BOT_TOKEN:
        return None
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/{method}",
                          json=payload or {}, timeout=20)
        return r.json() if r.ok else None
    except Exception:
        return None


def _authorized_chat(update: dict) -> bool:
    try:
        return str(update["message"]["chat"]["id"]) == str(CHAT_ID)
    except Exception:
        return False


def _telegram_offset_yukle() -> int:
    if TELEGRAM_OFFSET_FILE.exists():
        try:
            return int(TELEGRAM_OFFSET_FILE.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return 0


def _telegram_offset_kaydet(offset: int):
    TELEGRAM_OFFSET_FILE.write_text(str(int(offset)), encoding="utf-8")


def telegram_komut_isle(text: str) -> str:
    raw = (text or "").strip()
    cmd, _, arg = raw.partition(" ")
    symbol = arg.strip().upper()
    mevcut = hisse_listesi_yukle()
    if cmd == "/ekle":
        if not symbol:
            return "Kullanim: /ekle ASELS"
        if symbol in mevcut:
            return f"{symbol} zaten listede."
        mevcut.append(symbol)
        hisse_listesi_kaydet(mevcut)
        return f"\u2705 {symbol} eklendi.\nToplam: {len(mevcut)} hisse"
    if cmd == "/sil":
        if not symbol:
            return "Kullanim: /sil ASELS"
        if symbol not in mevcut:
            return f"{symbol} listede yok."
        hisse_listesi_kaydet([x for x in mevcut if x != symbol])
        return f"\U0001f5d1\ufe0f {symbol} silindi."
    if cmd == "/liste":
        if not mevcut:
            return "Liste bos."
        preview = ", ".join(mevcut[:40])
        extra = f"\n... +{len(mevcut)-40} hisse" if len(mevcut) > 40 else ""
        return f"\U0001f4cb {len(mevcut)} hisse:\n{preview}{extra}"
    if cmd == "/yardim":
        return "Komutlar:\n/ekle ASELS\n/sil ASELS\n/liste"
    return ""


def telegram_komutlarini_kontrol_et():
    if not BOT_TOKEN or not CHAT_ID:
        return
    offset = _telegram_offset_yukle()
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params={"offset": offset + 1, "timeout": 1, "allowed_updates": ["message"]},
            timeout=5,
        )
        if not r.ok:
            return
        for update in r.json().get("result", []):
            update_id = int(update.get("update_id", 0))
            offset = max(offset, update_id)
            if not _authorized_chat(update):
                continue
            text = (update.get("message", {}).get("text") or "").strip()
            if not text.startswith("/"):
                continue
            yanit = telegram_komut_isle(text)
            if yanit:
                send_telegram(yanit)
                append_jsonl(PORTFOY_AUDIT_FILE, {"event": "telegram_command", "command": text})
        _telegram_offset_kaydet(offset)
    except Exception as exc:
        log.debug("Telegram komut kontrol hatasi: %s", exc)


def makro_risk_skoru():
    global _MAKRO_CACHE
    now = datetime.now()
    # Cache geçerliyse direkt dön
    if (_MAKRO_CACHE["value"] is not None and _MAKRO_CACHE["fetched_at"] is not None
            and (now - _MAKRO_CACHE["fetched_at"]).total_seconds() < MAKRO_CACHE_TTL):
        log.debug("Makro skor cache'den dondu")
        return _MAKRO_CACHE["value"]

    detaylar = []
    piyasa_ret = {}
    for ticker, lag, yon, esik, agirlik in MAKRO_KURALLAR:
        try:
            df = _market_ticker(ticker).history(period="5d", interval="1d")
            if len(df) < 2:
                continue
            ret = float(df["Close"].pct_change().dropna().iloc[-(lag + 1)]) * 100
            piyasa_ret[ticker] = ret
            tetiklendi = ret < esik if yon == "down" else ret > esik
            detaylar.append({"endeks": ticker, "deger_pct": round(ret, 2),
                             "esik": esik, "agirlik": agirlik, "tetiklendi": tetiklendi})
        except Exception as exc:
            log.debug("Makro %s: %s", ticker, exc)

    sp_ret     = piyasa_ret.get("^GSPC",     0.0)
    nasdaq_ret = piyasa_ret.get("^IXIC",     0.0)
    vix_ret    = piyasa_ret.get("^VIX",      0.0)
    altin_ret  = piyasa_ret.get("GC=F",      0.0)
    abd_kotu   = sp_ret <= -2.0 or nasdaq_ret <= -3.0
    abd_orta   = -2.0 < sp_ret <= -1.0 or -3.0 < nasdaq_ret <= -1.5
    vix_sert   = vix_ret >= 3.0
    vix_hafif  = 2.0 <= vix_ret < 3.0
    altin_risk = altin_ret >= 2.0 and (sp_ret <= -1.0 or vix_ret >= 2.0)
    asya_kirmizi = sum(1 for t in ASYA_TICKERLAR if piyasa_ret.get(t, 0.0) <= ASYA_ESIK)

    if abd_kotu and vix_sert and asya_kirmizi >= ASYA_MIN_TETIK:
        karar = "GIRME"
    elif abd_orta or vix_hafif or asya_kirmizi >= 1 or altin_risk:
        karar = "DIKKATLI"
    else:
        karar = "NORMAL"

    toplam_skor = sum(d["agirlik"] for d in detaylar if d["tetiklendi"])
    if karar == "GIRME":
        toplam_skor = max(toplam_skor, 60.0)
    elif karar == "DIKKATLI":
        toplam_skor = max(toplam_skor, 30.0)

    sonuc = round(toplam_skor, 1), detaylar, karar, piyasa_ret
    _MAKRO_CACHE["value"]      = sonuc
    _MAKRO_CACHE["fetched_at"] = now
    return sonuc


def _makro_cache_temizle():
    """Sabah 09:00 akışında cache'i zorla yenile."""
    global _MAKRO_CACHE
    _MAKRO_CACHE = {"value": None, "fetched_at": None}


def viop_bias_hesapla() -> dict:
    """
    BIST yön tahmini için proxy zinciri:
      1. VIOP_TICKER (.env override — kullanıcı tanımlı)
      2. ^XU100      (Yahoo Finance BIST100 endeks sembolü)
      3. XU100.IS    (alternatif format)
      4. XU030.IS    (BIST30 ETF)
      5. basket      (10 büyük hisse ortalaması — son çare)
    Her adımda len(df) >= 2 ve NaN kontrolü yapılır.
    """
    def _ret_from_ticker(sym: str):
        try:
            df = _ticker(sym).history(period="5d", interval="1d")
            df = df.dropna(subset=["Close"])
            if len(df) < 2:
                return None
            ret = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
            if not np.isfinite(ret):
                return None
            return ret
        except Exception as exc:
            log.debug("VIOP proxy %s: %s", sym, exc)
            return None

    def _bias_from_ret(ret: float, ticker: str, source: str) -> dict:
        if ret >= 1.5:
            return {"label": "Long bias g\u00fc\u00e7l\u00fc", "score": 5, "size_factor": 1.15,
                    "ret": round(ret, 2), "ticker": ticker, "source": source}
        if ret >= 0.5:
            return {"label": "Normal", "score": 2, "size_factor": 1.05,
                    "ret": round(ret, 2), "ticker": ticker, "source": source}
        if ret <= -1.5:
            return {"label": "K\u00fc\u00e7\u00fck boyut / nakit a\u011f\u0131rl\u0131k", "score": -5, "size_factor": 0.55,
                    "ret": round(ret, 2), "ticker": ticker, "source": source}
        if ret <= -0.5:
            return {"label": "Temkinli long", "score": -2, "size_factor": 0.80,
                    "ret": round(ret, 2), "ticker": ticker, "source": source}
        return {"label": "Normal", "score": 0, "size_factor": 1.0,
                "ret": round(ret, 2), "ticker": ticker, "source": source}

    # Proxy zinciri — sırayla dene
    proxy_zinciri = []
    # 1. Kullanıcı override (.env) — boş veya varsayılan değilse ekle
    if VIOP_TICKER and VIOP_TICKER not in ("XU100.IS", "XU030.IS"):
        proxy_zinciri.append((VIOP_TICKER, "env_override"))
    # 2-4. Sabit fallback'ler
    proxy_zinciri += [
        ("^XU100",   "bist100_index"),
        ("XU100.IS", "bist100_etf"),
        ("XU030.IS", "bist30_etf"),
    ]

    for ticker, source in proxy_zinciri:
        ret = _ret_from_ticker(ticker)
        if ret is not None:
            log.debug("VIOP proxy basarili: %s (%s) ret=%.2f", ticker, source, ret)
            return _bias_from_ret(ret, ticker, source)

    log.warning("VIOP proxy zinciri basarisiz, basket'e geciliyor")

    # 5. Basket proxy — son çare
    basket = ["AKBNK.IS", "ASELS.IS", "BIMAS.IS", "EREGL.IS", "KCHOL.IS",
              "PGSUS.IS", "SAHOL.IS", "SISE.IS",  "THYAO.IS", "TUPRS.IS"]
    rets = []
    for ticker in basket:
        ret = _ret_from_ticker(ticker)
        if ret is not None:
            rets.append(ret)

    if rets:
        ret = float(np.mean(rets))
        log.info("VIOP basket proxy: %d/%d hisse, ret=%.2f", len(rets), len(basket), ret)
        if ret >= 1.2:
            return {"label": "Long bias g\u00fc\u00e7l\u00fc", "score": 4, "size_factor": 1.10,
                    "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}
        if ret >= 0.3:
            return {"label": "Normal", "score": 1, "size_factor": 1.03,
                    "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}
        if ret <= -1.2:
            return {"label": "K\u00fc\u00e7\u00fck boyut / nakit a\u011f\u0131rl\u0131k", "score": -4, "size_factor": 0.65,
                    "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}
        if ret <= -0.3:
            return {"label": "Temkinli long", "score": -1, "size_factor": 0.85,
                    "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}
        return {"label": "Normal", "score": 0, "size_factor": 1.0,
                "ret": round(ret, 2), "ticker": "BIST_BASKET", "source": "basket_proxy"}

    log.error("VIOP: tum proxy'ler basarisiz, Veri yok donuyor")
    return {"label": "Veri yok", "score": 0, "size_factor": 1.0,
            "ret": None, "ticker": None, "source": "none"}


def ema_calc(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_angle(fast_ma: pd.Series, lookback: int = 3) -> pd.Series:
    delta = (fast_ma / fast_ma.shift(lookback) - 1) * 100
    return np.degrees(np.arctan(delta / lookback))


def calc_distance(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    return (slow_ma - fast_ma) / (slow_ma + 1e-9) * 100


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def ma_motor_skoru(symbol: str):
    try:
        df = _ticker(f"{symbol}.IS").history(period="6mo", interval="1d")
        if len(df) < 60:
            return None
        c, v  = df["Close"], df["Volume"]
        vol_ratio = v / (v.rolling(20).mean() + 1e-9)
        ema50 = ema_calc(c, 50)
        scores = {"MotorA": 0.0, "MotorB": 0.0}
        for motor_ad, fast, slow, a_min, a_max, pre_cross in [
            ("MotorA", 12, 26, 20, 30, False),
            ("MotorB",  8, 21, 30, 45, True),
        ]:
            fast_ma = ema_calc(c, fast)
            slow_ma = ema_calc(c, slow)
            angle   = calc_angle(fast_ma)
            dist    = calc_distance(fast_ma, slow_ma)
            speed   = dist.shift(1) - dist
            above   = (fast_ma > slow_ma).astype(int)
            crossover = ((above == 1) & (above.shift(1) == 0)).astype(int)
            has_signal = (bool(crossover.iloc[-1]) if not pre_cross
                         else float(dist.iloc[-1]) > 0 and float(dist.iloc[-1]) <= 1.0 and float(speed.iloc[-1]) > 0)
            if not has_signal:
                continue
            aci = float(angle.iloc[-1])
            vol_r = float(vol_ratio.iloc[-1])
            if a_min <= aci < a_max and vol_r >= 1.5 and float(c.iloc[-1]) > float(ema50.iloc[-1]):
                scores[motor_ad] = round(min(aci, 90) * min(vol_r, 5), 2)
        toplam = scores["MotorA"] + scores["MotorB"]
        if scores["MotorA"] > 0 and scores["MotorB"] > 0:
            toplam *= 1.5
        scores["toplam"] = round(toplam, 2)
        return scores
    except Exception as exc:
        log.debug("MA motor %s: %s", symbol, exc)
        return None


def lgbm_skor_hesapla(symbol: str, model):
    if model is None:
        return None
    try:
        df = _ticker(f"{symbol}.IS").history(period="1y", interval="1d")
        if len(df) < 60:
            return None
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        feats = {}
        for p in [5,8,9,12,13,20,21,26,34,50,100,200]:
            feats[f"ema{p}_ratio"] = float((c / (ema_calc(c, p) + 1e-9)).iloc[-1])
        e9,e21,e12,e26,e50,e200 = (ema_calc(c,p) for p in [9,21,12,26,50,200])
        feats.update({
            "ema9_gt_21": int(e9.iloc[-1]>e21.iloc[-1]),
            "ema12_gt_26": int(e12.iloc[-1]>e26.iloc[-1]),
            "ema21_gt_50": int(e21.iloc[-1]>e50.iloc[-1]),
            "ema50_gt_200": int(e50.iloc[-1]>e200.iloc[-1]),
        })
        for p,lb in [(9,3),(12,3),(21,5),(50,10)]:
            feats[f"angle{p}"] = float(calc_angle(ema_calc(c,p),lb).iloc[-1])
        def _dist(fp,sp):
            return (ema_calc(c,sp)-ema_calc(c,fp))/(ema_calc(c,sp)+1e-9)*100
        feats.update({
            "dist_9_21": float(_dist(9,21).iloc[-1]),
            "dist_12_26": float(_dist(12,26).iloc[-1]),
            "dist_8_21": float(_dist(8,21).iloc[-1]),
            "speed_9_21": float(_dist(9,21).diff().iloc[-1]),
            "speed_12_26": float(_dist(12,26).diff().iloc[-1]),
            "rsi14": float(_rsi(c,14).iloc[-1]),
            "rsi7":  float(_rsi(c,7).iloc[-1]),
        })
        ml = ema_calc(c,12)-ema_calc(c,26); ms = ema_calc(ml,9)
        feats.update({
            "macd": float(ml.iloc[-1]), "macd_sig": float(ms.iloc[-1]),
            "macd_hist": float((ml-ms).iloc[-1]),
            "macd_gt_0": int(ml.iloc[-1]>0), "macd_gt_sig": int(ml.iloc[-1]>ms.iloc[-1]),
        })
        mid=c.rolling(20).mean(); std=c.rolling(20).std()
        bb_l=mid-2*std; bb_u=mid+2*std
        feats.update({
            "bb_pct": float(((c-bb_l)/(bb_u-bb_l+1e-9)).iloc[-1]),
            "bb_width": float(((bb_u-bb_l)/(mid+1e-9)).iloc[-1]),
            "bb_gt_mid": int(c.iloc[-1]>mid.iloc[-1]),
        })
        vm10,vm20 = v.rolling(10).mean(),v.rolling(20).mean()
        feats.update({
            "rel_vol_10": float((v/(vm10+1e-9)).iloc[-1]),
            "rel_vol_20": float((v/(vm20+1e-9)).iloc[-1]),
            "vol_trend":  float((vm10/(vm20+1e-9)).iloc[-1]),
        })
        tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        a14=tr.rolling(14).mean()
        feats.update({
            "atr_pct": float((a14/(c+1e-9)*100).iloc[-1]),
            "atr_ratio": float((a14/(a14.rolling(20).mean()+1e-9)).iloc[-1]),
            "ret1d": float(c.pct_change(1).iloc[-1]*100),
            "ret3d": float(c.pct_change(3).iloc[-1]*100),
            "ret5d": float(c.pct_change(5).iloc[-1]*100),
            "ret10d": float(c.pct_change(10).iloc[-1]*100),
        })
        mfv=((c-l)-(h-c))/(h-l+1e-9)*v
        feats["cmf20"] = float((mfv.rolling(20).sum()/(v.rolling(20).sum()+1e-9)).iloc[-1])
        r14=_rsi(c,14); lo=r14.rolling(14).min(); hi=r14.rolling(14).max()
        feats["stochrsi"] = float((100*(r14-lo)/(hi-lo+1e-9)).rolling(3).mean().iloc[-1])
        pivot=(h.shift(1)+l.shift(1)+c.shift(1))/3
        feats.update({
            "price_gt_fibr1": int(c.iloc[-1]>(pivot+0.382*(h.shift(1)-l.shift(1))).iloc[-1]),
            "price_gt_pivot": int(c.iloc[-1]>pivot.iloc[-1]),
            "motorA_sinyal": int(e12.iloc[-1]>e26.iloc[-1]),
            "motorB_sinyal": int(e9.iloc[-1]>e21.iloc[-1]),
        })
        feat_cols = (json.loads(LGBM_META_FILE.read_text(encoding="utf-8")).get("feature_cols")
                     if LGBM_META_FILE.exists() else None)
        X = (pd.DataFrame([[feats.get(col,0.0) for col in feat_cols]],columns=feat_cols)
             if feat_cols else pd.DataFrame([feats]))
        X = X.replace([np.inf,-np.inf],np.nan).fillna(0)
        prob = model.predict_proba(X)[0][1]
        return None if np.isnan(prob) else round(float(prob)*100,1)
    except Exception as exc:
        log.debug("LGBM %s: %s", symbol, exc)
        return None


def alpha_trend_analiz(symbol: str) -> dict:
    try:
        df = _ticker(f"{symbol}.IS").history(period="8mo", interval="1d")
        if len(df) < 60:
            return {"state": "none", "flip_up": False, "bonus": 0, "tag": ""}
        c,h,l,v = df["Close"],df["High"],df["Low"],df["Volume"]
        ap,coeff = 14,1.0
        atr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1).rolling(ap).mean()
        tp = (h+l+c)/3.0; rmf = tp*v
        pos_mf = rmf.where(tp>tp.shift(1),0.0); neg_mf = rmf.where(tp<tp.shift(1),0.0)
        mfi = 100-(100/(1+pos_mf.rolling(ap).sum()/(neg_mf.rolling(ap).sum()+1e-9)))
        up_t,dn_t = l-atr*coeff, h+atr*coeff
        alpha = pd.Series(index=df.index, dtype=float)
        alpha.iloc[0] = c.iloc[0]
        for i in range(1, len(df)):
            prev = alpha.iloc[i-1]
            alpha.iloc[i] = max(up_t.iloc[i],prev) if mfi.iloc[i]>=50 else min(dn_t.iloc[i],prev)
        bullish = bool(alpha.iloc[-1]>alpha.iloc[-3])
        prev_bullish = bool(alpha.iloc[-2]>alpha.iloc[-4]) if len(alpha)>=4 else False
        flip_up = bullish and not prev_bullish
        ema21,ema50 = ema_calc(c,21),ema_calc(c,50)
        above = bool(c.iloc[-1]>ema21.iloc[-1] and ema21.iloc[-1]>=ema50.iloc[-1])
        if flip_up and above:
            return {"state":"bullish","flip_up":True,"bonus":4,"tag":"AT\u2191"}
        if bullish:
            return {"state":"bullish","flip_up":False,"bonus":2,"tag":"AT+"}
        return {"state":"bearish","flip_up":False,"bonus":0,"tag":""}
    except Exception as exc:
        log.debug("Alpha Trend %s: %s", symbol, exc)
        return {"state":"none","flip_up":False,"bonus":0,"tag":""}


def _strategy_combo_score(strategies: list) -> float:
    return sum(STRATEGY_WEIGHTS.get(x, 0) for x in strategies)


def final_signal_score(signal: dict, ma_score, lgbm_score, viop_bias: dict, alpha_bonus: float) -> float:
    base = _strategy_combo_score(signal.get("strategies", []))
    base += min(signal.get("score_count", 1) * 6, 20)
    if ma_score:
        base += min(ma_score / 10, 20)
    if lgbm_score is not None:
        base += max((lgbm_score - 50) / 3, -10)
    base += viop_bias.get("score", 0)
    base += min(alpha_bonus, 4)
    rsi = signal.get("rsi", 50)
    rel_vol = signal.get("rel_vol", 1)
    strategies = signal.get("strategies", [])
    is_dip_only = len(strategies) >= 1 and all(s == "DIP" for s in strategies)
    if is_dip_only:
        if rsi >= 75:
            base -= 3
    else:
        if rsi >= 82:
            base -= 6
        elif rsi >= 75:
            base -= 3
    if rel_vol < 1:
        base -= 4
    if signal.get("islem_tl", 0) < 10_000_000:
        base -= 4
    return round(max(0, min(base, 100)), 2)


def build_candidate(symbol: str, signal: dict, model, viop_bias: dict) -> dict:
    ma_result  = ma_motor_skoru(symbol)
    ma_total   = ma_result["toplam"] if ma_result else 0.0
    lgbm_score = lgbm_skor_hesapla(symbol, model) if model else None
    alpha      = alpha_trend_analiz(symbol)
    return {
        "symbol":            symbol,
        "score_count":       signal.get("score_count", 1),
        "strategies":        signal.get("strategies", []),
        "tarama_score":      _strategy_combo_score(signal.get("strategies", [])),
        "ma_score":          round(ma_total, 2),
        "lgbm_score":        lgbm_score,
        "viop_bias":         viop_bias["label"],
        "viop_score":        viop_bias["score"],
        "alpha_trend_state": alpha["state"],
        "alpha_trend_flip":  alpha["flip_up"],
        "alpha_trend_bonus": alpha["bonus"],
        "alpha_tag":         alpha["tag"],
        "final_score":       final_signal_score(signal, ma_total, lgbm_score, viop_bias, alpha["bonus"]),
        "change_pct":        signal.get("change_pct", 0),
        "rsi":               signal.get("rsi", 0),
        "rel_vol":           signal.get("rel_vol", 0),
        "cmf":               signal.get("cmf", 0),
        "islem_tl":          signal.get("islem_tl", 0),
        "scan_time":         signal.get("scan_time", ""),
        "scan_label":        signal.get("scan_label", ""),
    }


def tarama_calistir(saat_label: str, makro_karar: str, viop_bias: dict):
    model, model_status = lgbm_model_yukle()
    signals  = tarama_listesi_yukle().get("signals", [])
    adaylar, elinenler = [], []
    for signal in signals:
        aday = build_candidate(signal["symbol"], signal, model, viop_bias)
        if len(aday.get("strategies", [])) == 1 and "DIP" in aday.get("strategies", []):
            append_jsonl(SIGNAL_AUDIT_FILE, {**aday, "event": "dip_info_only"})
            continue
        if aday["lgbm_score"] is not None and aday["lgbm_score"] < LGBM_MIN_SKOR:
            elinenler.append({"symbol": signal["symbol"], "reason": "score_low"})
            continue
        adaylar.append(aday)
    adaylar.sort(key=lambda x: (-x["final_score"], x["symbol"]))
    for aday in adaylar:
        append_jsonl(SIGNAL_AUDIT_FILE, {**aday, "event": "candidate"})
    return adaylar, model_status, elinenler


def portfolio_preview(portfoy: dict) -> dict:
    nakit = portfoy.get("nakit", 0)
    equity = nakit
    for sym, pos in portfoy.get("pozisyonlar", {}).items():
        f = guncel_fiyat(sym)
        equity += pos["lotlar"] * (f if f else pos["giris_f"])
    return {"cash": round(nakit, 2), "equity": round(equity, 2),
            "n_positions": len(portfoy.get("pozisyonlar", {}))}


def yeni_pozisyon_ac(portfoy: dict, adaylar: list, makro_karar: str, viop_bias: dict):
    if makro_karar == "GIRME":
        return portfoy, [], [], [{"symbol": a["symbol"], "reason": "makro_girme"} for a in adaylar]
    mesajlar, alinan, alinmayan = [], [], []
    mevcut   = portfoy["pozisyonlar"]
    nakit    = portfoy["nakit"]
    bos_slot = max(0, MAX_HISSE - len(mevcut))
    if bos_slot == 0 or not adaylar:
        return portfoy, [], [], []
    secilenler  = [a for a in adaylar if a["final_score"] >= 30][:bos_slot]
    toplam_skor = sum(max(a["final_score"], 1) for a in secilenler) or 1
    size_factor = viop_bias.get("size_factor", 1.0) * (0.6 if makro_karar == "DIKKATLI" else 1.0)
    for aday in secilenler:
        sym = aday["symbol"]
        if sym in mevcut:
            alinmayan.append({"symbol": sym, "reason": "already_open"})
            continue
        try:
            giris_f = guncel_fiyat(sym)
            if giris_f is None:
                alinmayan.append({"symbol": sym, "reason": "veri_yok"})
                append_jsonl(PORTFOY_AUDIT_FILE, {"event":"buy_failed","symbol":sym,"reason":"veri_yok"})
                continue
            alloc  = min(HISSE_LIMIT, nakit * size_factor * (aday["final_score"] / toplam_skor))
            lotlar = int(alloc / max(giris_f, 0.01))
            if lotlar < 1:
                alinmayan.append({"symbol": sym, "reason": "lot_yetersiz"})
                append_jsonl(PORTFOY_AUDIT_FILE, {"event":"buy_failed","symbol":sym,"reason":"lot_yetersiz"})
                continue
            maliyet = lotlar * giris_f
            if maliyet > nakit:
                alinmayan.append({"symbol": sym, "reason": "nakit_yetersiz"})
                append_jsonl(PORTFOY_AUDIT_FILE, {"event":"buy_failed","symbol":sym,"reason":"nakit_yetersiz"})
                continue
            nakit -= maliyet
            mevcut[sym] = {
                "giris_f": round(giris_f, 4), "giris_t": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "tepe_f": round(giris_f, 4), "lotlar": lotlar, "gun": 0, "tp1_yapildi": False,
                "final_score": aday["final_score"], "ma_score": aday["ma_score"],
                "lgbm_score": aday["lgbm_score"],
                "source_signal": {"score_count": aday["score_count"], "strategies": aday["strategies"],
                                  "viop_bias": aday["viop_bias"], "alpha_tag": aday.get("alpha_tag",""),
                                  "alpha_trend_bonus": aday.get("alpha_trend_bonus",0)},
            }
            mesajlar.append(
                f"\U0001f6a8 <b>AL - {sym}</b>\n"
                f"   {lotlar} lot @ {giris_f:.2f} TL\n"
                f"   Skor: {aday['final_score']:.1f} {aday.get('alpha_tag','')} | V\u0130OP: {aday['viop_bias']}"
            )
            append_jsonl(PORTFOY_AUDIT_FILE, {"event":"buy_success","symbol":sym,"price":giris_f,
                                               "lots":lotlar,"final_score":aday["final_score"]})
            alinan.append(sym)
        except Exception as exc:
            log.exception("Yeni pozisyon HATA %s", sym)
            alinmayan.append({"symbol": sym, "reason": "exception"})
            append_jsonl(PORTFOY_AUDIT_FILE, {"event":"buy_failed","symbol":sym,"reason":f"exception:{exc}"})
    portfoy["nakit"] = nakit
    return portfoy, mesajlar, alinan, alinmayan


def pozisyon_guncelle_saatlik(portfoy: dict, makro_karar: str):
    mesajlar, kapatilacak = [], []
    for sym, pos in list(portfoy["pozisyonlar"].items()):
        try:
            bar = saatlik_bar(sym)
            if bar is None:
                continue
            high, low, close = bar["high"], bar["low"], bar["close"]
            giris_f = pos["giris_f"]
            lotlar  = pos["lotlar"]
            pos["tepe_f"] = max(pos.get("tepe_f", giris_f), high)
            # STOP
            if (low - giris_f) / giris_f <= STOP_PCT:
                cikis_f = round(giris_f * (1 + STOP_PCT), 4)
                portfoy["nakit"] += lotlar * cikis_f
                kapatilacak.append(sym)
                mesajlar.append(f"\U0001f6d1 <b>STOP - {sym}</b>\n   Giri\u015f: {giris_f:.2f} \u2192 \u00c7\u0131k\u0131\u015f: {cikis_f:.2f}")
                append_jsonl(PORTFOY_AUDIT_FILE, {"event":"stop","symbol":sym,"exit_price":cikis_f,"return_pct":STOP_PCT*100})
                continue
            # TP1
            if (high - giris_f) / giris_f >= TP1_PCT and not pos.get("tp1_yapildi"):
                yari = max(1, lotlar // 2)
                portfoy["nakit"] += yari * close
                pos["lotlar"] -= yari
                pos["tp1_yapildi"] = True
                mesajlar.append(f"\U0001f3af <b>TP1 - {sym}</b>\n   {yari} lot @ {close:.2f} sat\u0131ld\u0131 (+{TP1_PCT*100:.0f}%)")
                append_jsonl(PORTFOY_AUDIT_FILE, {"event":"tp1","symbol":sym,"price":close,"remaining":pos["lotlar"]})
            # TRAILING
            elif pos.get("tp1_yapildi"):
                trail_ret = (low - pos["tepe_f"]) / pos["tepe_f"]
                if trail_ret <= TRAILING_PCT:
                    cikis_f = round(pos["tepe_f"] * (1 + TRAILING_PCT), 4)
                    portfoy["nakit"] += pos["lotlar"] * cikis_f
                    kapatilacak.append(sym)
                    ret_g = (cikis_f - giris_f) / giris_f
                    mesajlar.append(f"\U0001f4c9 <b>TRAILING - {sym}</b>\n   \u00c7\u0131k\u0131\u015f: {cikis_f:.2f} | Getiri: {ret_g*100:+.1f}%")
                    append_jsonl(PORTFOY_AUDIT_FILE, {"event":"trailing","symbol":sym,"exit_price":cikis_f,"return_pct":ret_g*100})
                    continue
            # MAX GUN
            if _elde_tutma_gunu(pos.get("giris_t", "")) >= MAX_GUN:
                portfoy["nakit"] += pos["lotlar"] * close
                kapatilacak.append(sym)
                gun_ret = (close - giris_f) / giris_f
                mesajlar.append(f"\u23f0 <b>MAX G\u00dcN - {sym}</b>\n   \u00c7\u0131k\u0131\u015f: {close:.2f} | Getiri: {gun_ret*100:+.1f}%")
                append_jsonl(PORTFOY_AUDIT_FILE, {"event":"max_day","symbol":sym,"exit_price":close,"return_pct":gun_ret*100})
        except Exception as exc:
            log.debug("Pozisyon guncelle %s: %s", sym, exc)
    for sym in kapatilacak:
        portfoy["pozisyonlar"].pop(sym, None)
    # Acil likidasyon
    makro_skor = guncel_makro_skoru()
    if makro_karar == "GIRME" and portfoy["pozisyonlar"] and makro_skor >= EMERGENCY_LIQUIDATION_SCORE:
        semboller = list(portfoy["pozisyonlar"].keys())
        for sym in semboller:
            pos = portfoy["pozisyonlar"].pop(sym)
            f   = guncel_fiyat(sym) or pos["giris_f"]
            portfoy["nakit"] += pos["lotlar"] * f
        mesajlar.append(f"\U0001f6a8 <b>AC\u0130L NAK\u0130T</b>\n   Makro skor {makro_skor:.1f} \u2192 {len(semboller)} pozisyon kapat\u0131ld\u0131")
        append_jsonl(PORTFOY_AUDIT_FILE, {"event":"risk_off_liquidation","symbols":semboller,"makro_skor":makro_skor})
    return portfoy, mesajlar


def makro_ozet_metni(skor: float, karar: str, detaylar: list, piyasa_ret: dict) -> str:
    def fmt(v): return f"{v:+.1f}%" if v != 0.0 else "\u2014"
    sp   = piyasa_ret.get("^GSPC",    0.0)
    nq   = piyasa_ret.get("^IXIC",    0.0)
    vix  = piyasa_ret.get("^VIX",     0.0)
    n225 = piyasa_ret.get("^N225",    0.0)
    hsi  = piyasa_ret.get("^HSI",      0.0)
    sse  = piyasa_ret.get("000001.SS",0.0)
    altin = piyasa_ret.get("GC=F",    0.0)
    emoji = {"GIRME":"\U0001f534","DIKKATLI":"\U0001f7e1","NORMAL":"\U0001f7e2"}.get(karar,"\u26aa")
    return (
        "\U0001f30d <b>K\u00fcresel G\u00f6r\u00fcn\u00fcm</b>\n"
        f"   \U0001f1fa\U0001f1f8 S&P500: {fmt(sp)} | Nasdaq: {fmt(nq)} | VIX: {fmt(vix)}\n"
        f"   \U0001f30f Nikkei: {fmt(n225)} | Hang Seng: {fmt(hsi)} | Shanghai: {fmt(sse)}\n"
        f"   \U0001f947 Alt\u0131n: {fmt(altin)}\n"
        f"\U0001f4ca Makro Skor: {skor:.1f} \u2192 {emoji} <b>{karar}</b>"
    )


def portfoy_ozet_mesaji(portfoy: dict, saat_label: str, model_status: str = "pasif") -> str:
    prev = portfolio_preview(portfoy)
    lines = [
        f"\U0001f4bc <b>PORTF\u00d6Y - {saat_label}</b>",
        "--------------------",
        f"\U0001f4b0 Ba\u015flang\u0131\u00e7 : {portfoy.get('baslangic', SERMAYE_BASLANGIC):,.0f} TL",
        f"\U0001f4ca G\u00fcncel    : {prev['equity']:,.0f} TL",
        f"\U0001f4b5 Nakit     : {prev['cash']:,.0f} TL",
        f"\U0001f916 Model     : {model_status}",
    ]
    if portfoy["pozisyonlar"]:
        for sym, pos in portfoy["pozisyonlar"].items():
            f = guncel_fiyat(sym)
            ret = ((f - pos["giris_f"]) / pos["giris_f"] * 100) if f else 0
            lines.append(
                f"   \u2022 <b>{sym}</b> {pos['lotlar']}lot @ {pos['giris_f']:.2f}"
                f" \u2192 {f:.2f if f else '?'} ({ret:+.1f}%) g\u00fcn:{pos.get('gun',0)}"
            )
    else:
        lines.append("   - A\u00e7\u0131k pozisyon yok")
    waiting = portfoy.get("bekleyen_al", [])
    if waiting:
        lines.append(f"\U0001f4cb Bekleyen: {', '.join(x['symbol'] for x in waiting[:7])}")
    summary = portfoy.get("last_open_attempt_summary", {})
    if summary:
        lines.append(
            f"\U0001fa9f Son al\u0131m: bekleyen={summary.get('pending',0)} | "
            f"al\u0131nd\u0131={summary.get('bought',0)} | "
            f"al\u0131namad\u0131={summary.get('failed',0)} | "
            f"s\u00fcresi dolan={summary.get('expired',0)}"
        )
    return "\n".join(lines)


def alim_denemesi(portfoy: dict, makro_karar: str, viop_bias: dict, now: datetime):
    aktif, expired = ayikla_suresi_dolan_bekleyenler(portfoy.get("bekleyen_al", []), now)
    portfoy["bekleyen_al"] = aktif
    portfoy, al_mesajlari, alinanlar, alinmayanlar = yeni_pozisyon_ac(
        portfoy, aktif, makro_karar, viop_bias)
    portfoy["bekleyen_al"] = [x for x in aktif if x["symbol"] not in set(alinanlar)]
    # Deneme kaydını güncelle
    for item in portfoy["bekleyen_al"]:
        sym = item["symbol"]
        neden = next((a["reason"] for a in alinmayanlar if a["symbol"] == sym), "")
        item["attempt_count"]       = item.get("attempt_count", 0) + 1
        item["last_attempt_time"]   = now.strftime("%d.%m.%Y %H:%M")
        item["last_attempt_reason"] = neden
    summary = {
        "pending":  len(aktif),
        "bought":   len(alinanlar),
        "failed":   len(alinmayanlar),
        "expired":  len(expired),
        "failed_reasons": (alinmayanlar + expired)[:10],
        "time":     now.strftime("%d.%m.%Y %H:%M"),
    }
    portfoy["last_open_attempt_summary"] = summary
    portfoy.setdefault("open_attempts_today", []).append({
        "saat":     now.strftime("%H:%M"),
        "tarih":    now.strftime("%Y-%m-%d"),
        "alinan":   alinanlar,
        "alinmayan": [a["symbol"] for a in alinmayanlar],
        "nedenler": {a["symbol"]: a["reason"] for a in alinmayanlar},
    })
    # Son 7 günü tut, eskiyi sil
    yedi_gun_once = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    portfoy["open_attempts_today"] = [
        a for a in portfoy["open_attempts_today"]
        if a.get("tarih", "9999-99-99") >= yedi_gun_once
    ]
    mesajlar = al_mesajlari
    if expired:
        mesajlar += [f"\u23f3 <b>S\u00fcresi dolan:</b> {', '.join(x['symbol'] for x in expired[:7])}"]
    return portfoy, mesajlar, summary


def sabah_09_akisi():
    now = datetime.now()
    saat_label = now.strftime("%d.%m.%Y %H:%M")
    _makro_cache_temizle()  # her sabah taze veri çek
    skor, detaylar, karar, piyasa_ret = makro_risk_skoru()
    viop_bias = viop_bias_hesapla()
    _, model_status = lgbm_model_yukle()
    adaylar, _, elinenler = tarama_calistir(saat_label, karar, viop_bias)
    portfoy = portfoy_yukle()
    bekleyen = []
    for aday in adaylar[:MAX_HISSE]:
        if aday["symbol"] in portfoy["pozisyonlar"]:
            continue
        bekleyen.append(bekleyen_adayi_hazirla(aday, now))
    portfoy["bekleyen_al"] = bekleyen
    portfoy["open_attempts_today"] = []
    portfoy_kaydet(portfoy)
    durum_kaydet({"tarih": saat_label, "makro_skor": skor, "makro_karar": karar,
                  "makro_detaylar": detaylar})
    lines = [
        f"\u2600\ufe0f <b>09:00 Sabah De\u011flendirmesi \u2014 {saat_label}</b>", "",
        makro_ozet_metni(skor, karar, detaylar, piyasa_ret), "",
        f"\U0001f4cc V\u0130OP: <b>{viop_bias['label']}</b> ({viop_bias['score']:+.0f}p) | "
        f"Kaynak: {viop_bias.get('ticker','?')}", "",
    ]
    if karar == "GIRME":
        lines.append("\U0001f534 <b>GIRME karar\u0131 \u2014 bug\u00fcn yeni al\u0131m yap\u0131lmayacak.</b>")
    elif bekleyen:
        lines.append(f"\U0001f4cb <b>Bug\u00fcn \u015funlar\u0131 alal\u0131m</b> ({len(bekleyen)} aday, 11:00'da i\u015flem):")
        for aday in bekleyen[:7]:
            lines.append(
                f"   \u2022 <b>{aday['symbol']}</b> {aday.get('alpha_tag','')} "
                f"skor:{aday['final_score']:.1f} puan:{aday['score_count']} "
                f"LGBM:{aday['lgbm_score'] if aday['lgbm_score'] is not None else '-'} [{','.join(aday.get('strategies',[]))}]"
            )
    else:
        lines.append("   - Bug\u00fcn g\u00fc\u00e7l\u00fc aday yok")
    if elinenler:
        lines.append(f"\n\U0001f6ab LGBM filtresiyle elinen: {', '.join(x['symbol'] for x in elinenler[:5])}")
    log.info("Sabah degerlendirmesi tamamlandi (Telegram: islem yok, atlandi)")
    return karar


def saat_11_alim(makro_karar: str):
    now = datetime.now()
    viop_bias = viop_bias_hesapla()
    portfoy = portfoy_yukle()
    onceki = set(portfoy["pozisyonlar"].keys())
    mesajlar = []
    # Piyasa açılışından itibaren (10:00 TSİ) STOP/TP kontrolü de burada
    # yapılır — yalnızca "takip" penceresini (11:20+) beklemek, sabah erken
    # saatlerde taşınan pozisyonların saatlerce izlenmeden kalmasına yol açardı.
    if portfoy["pozisyonlar"]:
        portfoy, islem_msg = pozisyon_guncelle_saatlik(portfoy, makro_karar)
        mesajlar.extend(islem_msg)
    portfoy, al_msg, summary = alim_denemesi(portfoy, makro_karar, viop_bias, now)
    mesajlar.extend(al_msg)
    portfoy["last_hourly_check_time"] = now.strftime("%d.%m.%Y %H:%M")
    portfoy_kaydet(portfoy)
    if mesajlar:
        sonra = set(portfoy["pozisyonlar"].keys())
        _p1_telegram_islem(
            portfoy,
            giris=list(sonra - onceki),
            cikis=list(onceki - sonra),
            mesajlar=mesajlar,
        )


def saat_1130_ozeti():
    """11:30 özeti — işlem yoksa Telegram gönderilmez."""
    log.info("11:30 ozet: islem bazli politika nedeniyle Telegram atlanir")


def saatlik_kontrol(makro_karar: str):
    now = datetime.now()
    viop_bias = viop_bias_hesapla()
    portfoy = portfoy_yukle()
    onceki = set(portfoy["pozisyonlar"].keys())
    mesajlar = []
    if portfoy["pozisyonlar"]:
        portfoy, islem_msg = pozisyon_guncelle_saatlik(portfoy, makro_karar)
        mesajlar.extend(islem_msg)
    portfoy["bekleyen_al"] = retry_bekleyenleri_filtrele(portfoy.get("bekleyen_al", []))
    if portfoy["bekleyen_al"]:
        portfoy, al_msg, _ = alim_denemesi(portfoy, makro_karar, viop_bias, now)
        mesajlar.extend(al_msg)
    portfoy["last_hourly_check_time"] = now.strftime("%d.%m.%Y %H:%M")
    portfoy_kaydet(portfoy)
    append_jsonl(PORTFOY_AUDIT_FILE, {
        "event": "hourly_check", "saat": now.strftime("%H:%M"),
        "pozisyon_sayisi": len(portfoy["pozisyonlar"]),
        "bekleyen_sayisi": len(portfoy.get("bekleyen_al", [])),
    })
    if mesajlar:
        sonra = set(portfoy["pozisyonlar"].keys())
        _p1_telegram_islem(
            portfoy,
            giris=list(sonra - onceki),
            cikis=list(onceki - sonra),
            mesajlar=mesajlar,
        )


def kapanis_ozeti_1730():
    """17:30 — bekleyenleri temizle; Telegram yalnızca işlem anında gönderilir."""
    now = datetime.now()
    portfoy = portfoy_yukle()
    _, expired = ayikla_suresi_dolan_bekleyenler(portfoy.get("bekleyen_al", []), now)
    portfoy["bekleyen_al"] = []
    portfoy_kaydet(portfoy)
    log.info("17:30 kapanis: bekleyen temizlendi (Telegram: islem yoksa atlanir)")


def preview_from_signals(signals: list, scan_time: str, scan_label: str) -> dict:
    model, model_status = lgbm_model_yukle()
    viop_bias = viop_bias_hesapla()
    preview = []
    for signal in signals[:15]:
        preview.append(build_candidate(signal["symbol"], signal, model, viop_bias))
    preview.sort(key=lambda x: (-x["final_score"], x["symbol"]))
    portfoy = portfoy_yukle()
    return {
        "preview_candidates": preview[:5],
        "portfolio":          portfolio_preview(portfoy),
        "waiting_symbols":    [x["symbol"] for x in preview[:MAX_HISSE]],
        "viop_bias":          viop_bias,
        "scan_time":          scan_time,
        "scan_label":         scan_label,
        "model_status":       model_status,
    }


def flask_baslat():
    from flask import Flask, jsonify, request as freq
    app = Flask(__name__)

    def api_key_ok():
        gelen = freq.headers.get("X-API-Key") or freq.args.get("api_key", "")
        return not FLASK_API_KEY or gelen == FLASK_API_KEY

    @app.route("/tarama", methods=["POST"])
    def tarama_al():
        if not api_key_ok():
            return jsonify({"status": "error", "msg": "yetkisiz"}), 401
        data    = freq.get_json(silent=True) or {}
        signals = data.get("signals")
        if signals is None:
            semboller = data.get("semboller", [])
            signals   = [{"symbol": s, "score_count": 1, "strategies": [],
                          "scan_time": data.get("scan_time",""), "scan_label": data.get("scan_label","")}
                         for s in semboller]
        clean = []
        for s in signals[:200]:
            symbol = str(s.get("symbol","")).upper().strip()
            if not symbol:
                continue
            clean.append({
                "symbol":     symbol,
                "score_count": int(s.get("score_count",1)),
                "strategies":  list(s.get("strategies",[])),
                "change_pct":  float(s.get("change_pct",0)),
                "rsi":         float(s.get("rsi",0)),
                "rel_vol":     float(s.get("rel_vol",0)),
                "cmf":         float(s.get("cmf",0)),
                "islem_tl":    float(s.get("islem_tl",0)),
                "scan_time":   s.get("scan_time", data.get("scan_time","")),
                "scan_label":  s.get("scan_label", data.get("scan_label","")),
            })
        payload = {"scan_time": data.get("scan_time",""), "scan_label": data.get("scan_label",""), "signals": clean}
        with open(TARAMA_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        for item in clean:
            append_jsonl(SIGNAL_AUDIT_FILE, {**item, "event": "scanner_signal"})
        summary = preview_from_signals(clean, payload["scan_time"], payload["scan_label"])
        return jsonify({"status": "ok", "n": len(clean), **summary})

    
    @app.route("/tarama_p2", methods=["POST"])
    def tarama_p2_al():
        if not api_key_ok():
            return jsonify({"status": "error", "msg": "yetkisiz"}), 401
        data = freq.get_json(silent=True) or {}
        mod  = data.get("mod", "aksam")

        if mod == "teyit":
            teyitler = data.get("teyitler", {})
            portfoy  = p2_portfoy_yukle()
            guncellenen = []
            for item in portfoy.get("bekleyen_al", []):
                sym = item["symbol"]
                if sym in teyitler:
                    t = teyitler[sym]
                    item["teyit_skoru"]     = t.get("teyit_skoru", 0)
                    item["teyit_var"]       = t.get("teyit_var", False)
                    item["teyit_sinyaller"] = t.get("teyit_sinyaller", [])
                    item["final_score"]     = item.get("score", 0) + t.get("teyit_skoru", 0)
                    guncellenen.append(sym)
            p2_portfoy_kaydet(portfoy)
            log.info("P2 teyit guncellendi: %s", guncellenen)
            return jsonify({"status": "ok", "mod": "teyit", "guncellenen": guncellenen})

        else:
            signals = data.get("signals", [])
            clean   = []
            for s in signals[:200]:
                symbol = str(s.get("symbol", "")).upper().strip()
                if not symbol:
                    continue
                clean.append({
                    "symbol":          symbol,
                    "score":           float(s.get("score", 0)),
                    "final_score":     float(s.get("score", 0)),
                    "verdict":         s.get("verdict", "AL"),
                    "pd_zone":         s.get("pd_zone", ""),
                    "rvol":            float(s.get("rvol", 1)),
                    "rsi":             float(s.get("rsi", 50)),
                    "htf_aligned":     bool(s.get("htf_aligned", False)),
                    "signals":         list(s.get("signals", [])),
                    "islem_tl":        float(s.get("islem_tl", 0)),
                    "change_pct":      float(s.get("change_pct", 0)),
                    "scan_time":       s.get("scan_time", data.get("scan_time", "")),
                    "scan_label":      s.get("scan_label", data.get("scan_label", "")),
                    "teyit_skoru":     0.0,
                    "teyit_var":       False,
                    "teyit_sinyaller": [],
                })
            payload = {
                "scan_time":  data.get("scan_time", ""),
                "scan_label": data.get("scan_label", ""),
                "signals":    clean,
            }
            with open(TARAMA_P2_FILE, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            portfoy = p2_portfoy_yukle()
            mevcut_semboller = set(portfoy["pozisyonlar"].keys())
            bekleyen = []
            for s in clean:
                if s["symbol"] not in mevcut_semboller:
                    item = dict(s)
                    item["queued_at"]           = data.get("scan_time", "")
                    item["attempt_count"]       = 0
                    item["last_attempt_reason"] = ""
                    bekleyen.append(item)
            portfoy["bekleyen_al"]         = bekleyen
            portfoy["open_attempts_today"] = []
            p2_portfoy_kaydet(portfoy)
            for item in clean:
                append_jsonl(SIGNAL_AUDIT_P2_FILE, {**item, "event": "p2_scanner_signal"})
            log.info("P2 tarama alindi: %s sinyal, mod=%s", len(clean), mod)
            return jsonify({
                "status": "ok", "n": len(clean), "mod": mod,
                "portfolio": p2_portfolio_preview(portfoy),
            })

    @app.route("/durum_p2", methods=["GET"])
    def durum_p2():
        if not api_key_ok():
            return jsonify({"status": "error", "msg": "yetkisiz"}), 401
        portfoy = p2_portfoy_yukle()
        return jsonify({
            "status":      "ok",
            "portfolio":   p2_portfolio_preview(portfoy),
            "pozisyonlar": list(portfoy["pozisyonlar"].keys()),
            "bekleyen_al": portfoy.get("bekleyen_al", []),
        })

    @app.route("/durum", methods=["GET"])
    def durum():
        p = portfoy_yukle()
        return jsonify(portfolio_preview(p) | {"baslangic": SERMAYE_BASLANGIC,
                                                "semboller": list(p["pozisyonlar"].keys())})

    @app.route("/semboller", methods=["GET"])
    def semboller():
        if not api_key_ok():
            return jsonify({"status": "error", "msg": "yetkisiz"}), 401
        syms = hisse_listesi_yukle()
        return jsonify({"status": "ok", "symbols": syms, "n": len(syms)})

    @app.route("/saglik", methods=["GET"])
    def saglik():
        return jsonify({"status": "ok", "zaman": datetime.now().strftime("%H:%M:%S")})

    log.info("Flask basliyor: port %s", FLASK_PORT)
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False)


def main():
    log.info("portfoy_yonetici.py basliyor")
    # Flask kaldırıldı — GitHub Actions tek seferlik çalışır
    makro_karar       = "NORMAL"
    son_saat          = ""
    saat_11_yapildi   = False
    saat_1130_yapildi = False
    saat_1730_yapildi = False

    while True:
        now  = datetime.now()
        saat = now.strftime("%H:%M")
        gun  = now.weekday()

        telegram_komutlarini_kontrol_et()

        if gun >= 5:
            time.sleep(300)
            continue

        # Gün başı sıfırla
        if saat == "00:01" and son_saat != "00:01":
            saat_11_yapildi = saat_1130_yapildi = saat_1730_yapildi = False
            son_saat = saat

        # 09:00 Sabah değerlendirmesi
        if saat == "09:00" and son_saat != "09:00":
            makro_karar = sabah_09_akisi()
            son_saat = saat
            time.sleep(61)

        # 11:00 İlk alım
        elif saat == "11:00" and not saat_11_yapildi:
            saat_11_alim(makro_karar)
            p2_saatlik_kontrol(makro_karar)
            saat_11_yapildi = True
            son_saat = saat
            time.sleep(61)

        # 11:30 Portföy özeti
        elif saat == "11:30" and not saat_1130_yapildi:
            saat_1130_ozeti()
            saat_1130_yapildi = True
            son_saat = saat
            time.sleep(61)

        # 12:00-17:00 Saatlik kontrol
        elif now.minute == 0 and 12 <= now.hour <= 17 and son_saat != saat:
            saatlik_kontrol(makro_karar)
            p2_saatlik_kontrol(makro_karar)
            son_saat = saat
            time.sleep(61)

        # 17:30 Kapanış özeti
        elif saat == "17:30" and not saat_1730_yapildi:
            kapanis_ozeti_1730()
            saat_1730_yapildi = True
            son_saat = saat
            time.sleep(61)

        # 21:00 Gün sonu pozisyon güncelle
        elif saat == "21:00" and son_saat != "21:00":
            portfoy = portfoy_yukle()
            if portfoy["pozisyonlar"]:
                for pos in portfoy["pozisyonlar"].values():
                    pos["gun"] = pos.get("gun", 0) + 1
                portfoy_kaydet(portfoy)
            p2_gun_sonu_guncelle()
            son_saat = saat
            time.sleep(61)

        else:
            time.sleep(20)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        cmds = {
            "sabah":  sabah_09_akisi,
            "alim":   lambda: saat_11_alim("NORMAL"),
            "ozet":   saat_1130_ozeti,
            "kapani": kapanis_ozeti_1730,
            "takip":  lambda: saatlik_kontrol("NORMAL"),
            "durum":  lambda: print(json.dumps(portfoy_yukle(), indent=2, ensure_ascii=False)),
            "viop":   lambda: print(json.dumps(viop_bias_hesapla(), ensure_ascii=False)),
            "makro":  lambda: [print(f"Skor: {s} -> {k}") or [print(f"  {t}: {v:+.2f}%") for t,v in pr.items()]
                               for s,_,k,pr in [makro_risk_skoru()]],
            # ── P2 — SMC portföy köprüsü ──────────────────────────────────
            "p2_aksam": p2_adaylari_yukle_ve_hazirla,
            "p2_takip": lambda: p2_saatlik_kontrol("NORMAL"),
            "p2_durum": lambda: print(json.dumps(p2_portfoy_yukle(), indent=2, ensure_ascii=False)),
        }
        if cmd in cmds:
            cmds[cmd]()
        else:
            print("Kullanim: python portfoy_yonetici.py [sabah|alim|ozet|kapani|takip|durum|viop|makro|p2_aksam|p2_takip|p2_durum]")
    else:
        main()