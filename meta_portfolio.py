"""
meta_portfolio.py — P4 Meta Portföy Yöneticisi
================================================
ML4AT (Stefan Jansen) kitabından:
  - Spearman IC ile her stratejinin sinyal kalitesi ölçülür
  - IC ağırlıklı sinyal birleştirme — yüksek IC → daha fazla ağırlık
  - Half-Kelly pozisyon boyutlandırma
  - HRP (Hierarchical Risk Parity) çeşitlendirme

Çalışma akışı (GitHub Actions):
  1. state_p1.json + state_p2.json + state_p3.json oku
  2. Her stratejinin rolling IC'sini hesapla
  3. IC ağırlıklı sinyal listesi üret
  4. En iyi 5 sinyali seç, pozisyon aç/kapat
  5. state_p4.json kaydet
  6. Telegram bildir

Çalıştır:
    python meta_portfolio.py
    python meta_portfolio.py --durum
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR   = Path(os.environ.get("MOTT_BASE_DIR", "."))
MAX_POS    = 5          # Maksimum açık pozisyon
SERMAYE    = 100_000    # 100k TL
POS_TL     = SERMAYE / MAX_POS   # Pozisyon başı 20k TL
STOP_PCT   = -0.05      # %5 stop-loss
TP_PCT     = 0.10       # %10 take-profit
MAX_GUN    = 10         # Maksimum elde tutma süresi
IC_WINDOW  = 60         # Rolling IC penceresi (bar)
IC_PERIODS = [1, 5, 10] # IC hesabı için ileriye bakış dönemleri

# ---------------------------------------------------------------------------
# IC Hesaplama — ML4AT Ch. 4
# ---------------------------------------------------------------------------

def compute_ic(
    signals: list[float],
    forward_returns: list[float],
) -> float:
    """
    Spearman rank korelasyonu ile IC hesapla.
    IC > 0.05 → anlamlı sinyal
    IC > 0.10 → güçlü sinyal
    """
    from scipy.stats import spearmanr
    if len(signals) < 10 or len(forward_returns) < 10:
        return 0.0
    n = min(len(signals), len(forward_returns))
    ic, _ = spearmanr(signals[:n], forward_returns[:n])
    return float(ic) if not np.isnan(ic) else 0.0


def strateji_ic_hesapla(
    strateji: str,
    signal_log: list[dict],
    fiyat_cache: dict[str, np.ndarray],
    period: int = 5,
) -> float:
    """
    Bir stratejinin geçmiş sinyal logunu kullanarak IC hesaplar.

    signal_log: [{"symbol": "GARAN", "score": 0.05, "tarih": "2026-01-15"}, ...]
    fiyat_cache: {symbol: prices_array}

    Returns: IC değeri (-1 ile +1 arası)
    """
    if len(signal_log) < 15:
        log.info("%s: IC için yeterli sinyal yok (%d)", strateji, len(signal_log))
        return 0.0

    sinyal_skorlar = []
    ileriki_getiriler = []

    for kayit in signal_log:
        sym   = kayit.get("symbol", "")
        skor  = float(kayit.get("score", kayit.get("final_score", 0)))
        tarih = kayit.get("tarih", kayit.get("scan_time", ""))[:10]

        if sym not in fiyat_cache or not tarih:
            continue

        prices = fiyat_cache[sym]

        # Tarihten itibaren period bar sonraki getiri
        # Basitleştirilmiş: son N bar üzerinden
        if len(prices) < period + 2:
            continue

        fwd_return = float(
            np.log(prices[-1] / prices[-period]) if prices[-period] > 0 else 0.0
        )

        sinyal_skorlar.append(skor)
        ileriki_getiriler.append(fwd_return)

    ic = compute_ic(sinyal_skorlar, ileriki_getiriler)
    log.info("%s IC (period=%d): %.4f (%d sinyal)", strateji, period, ic, len(sinyal_skorlar))
    return ic


# ---------------------------------------------------------------------------
# Fiyat Çekici
# ---------------------------------------------------------------------------

def fiyat_cek(semboller: list[str], bars: int = 30) -> dict[str, np.ndarray]:
    """Semboller için son N bar kapanış fiyatı."""
    cache = {}
    for sym in semboller:
        try:
            ticker = f"{sym}.IS" if not sym.endswith(".IS") else sym
            df = yf.Ticker(ticker).history(period="3mo")
            prices = df["Close"].dropna().values
            if len(prices) >= bars:
                cache[sym] = prices[-bars:]
        except Exception as e:
            log.debug("%s fiyat hatası: %s", sym, e)
    return cache


# ---------------------------------------------------------------------------
# State Yönetimi
# ---------------------------------------------------------------------------

def state_yukle() -> dict:
    path = BASE_DIR / "state_p4.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "pozisyonlar": {},
        "trade_history": [],
        "ic_history": [],
        "last_run": "",
        "sermaye_baslangic": SERMAYE,
        "sermaye_mevcut": SERMAYE,
    }


def state_kaydet(state: dict):
    path = BASE_DIR / "state_p4.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    log.info("P4 state kaydedildi")


def p1_sinyalleri_yukle() -> tuple[list[dict], list[dict]]:
    """state_p1.json'dan sinyal ve portföy yükle."""
    state = {}
    path  = BASE_DIR / "state_p1.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    sinyaller = state.get("tarama", {}).get("signals", [])
    return sinyaller, state


def p2_sinyalleri_yukle() -> tuple[list[dict], list[dict]]:
    """state_p2.json'dan sinyal ve portföy yükle."""
    state = {}
    path  = BASE_DIR / "state_p2.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    sinyaller = state.get("tarama", {}).get("signals", [])
    return sinyaller, state


def p3_sinyalleri_yukle() -> tuple[list[dict], dict]:
    """portfolio_state.json'dan P3 sinyal ve portföy yükle."""
    state = {}
    path  = BASE_DIR / "portfolio_state.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

    # Son tarama logundaki top longs
    scan_log = state.get("scan_log", [])
    sinyaller = []
    if scan_log:
        son_tarama = scan_log[-1]
        for sym in son_tarama.get("top_longs", []):
            sinyaller.append({
                "symbol":       sym,
                "score":        0.0,
                "final_score":  0.0,
                "strateji":     "P3",
            })

    return sinyaller, state


# ---------------------------------------------------------------------------
# Sinyal Normalizasyonu — Karşılaştırılabilir skor
# ---------------------------------------------------------------------------

def normalize_skor(sinyaller: list[dict], strateji: str) -> list[dict]:
    """
    Farklı stratejilerin skor ölçeklerini 0-1 arasına normalize et.
    P1: final_score genellikle 3-8 arası
    P2: score genellikle 4-10 arası
    P3: score genellikle 0.01-0.06 arası
    """
    if not sinyaller:
        return []

    skorlar = [float(s.get("score", s.get("final_score", 0))) for s in sinyaller]
    min_s   = min(skorlar)
    max_s   = max(skorlar)
    aralik  = max_s - min_s

    normalized = []
    for i, s in enumerate(sinyaller):
        s_copy = dict(s)
        if aralik > 0:
            s_copy["norm_score"] = (skorlar[i] - min_s) / aralik
        else:
            s_copy["norm_score"] = 1.0
        s_copy["strateji"] = strateji
        normalized.append(s_copy)

    return normalized


# ---------------------------------------------------------------------------
# IC Ağırlıklı Sinyal Birleştirme — ML4AT Ch. 3
# ---------------------------------------------------------------------------

def ic_agirlikli_birlestir(
    p1_sinyaller: list[dict],
    p2_sinyaller: list[dict],
    p3_sinyaller: list[dict],
    ic_p1: float,
    ic_p2: float,
    ic_p3: float,
) -> list[dict]:
    """
    ML4AT Fundamental Law: IR = IC × √Breadth
    Yüksek IC → bu stratejinin sinyaline daha fazla ağırlık ver.

    IC negatif veya çok düşükse (< 0.02) → o stratejinin ağırlığı sıfırlanır.
    """
    # Negatif IC'yi 0'a çek — zararlı sinyali yoksay
    w1 = max(ic_p1, 0.0)
    w2 = max(ic_p2, 0.0)
    w3 = max(ic_p3, 0.0)
    toplam = w1 + w2 + w3

    if toplam < 0.01:
        # Hiçbir strateji anlamlı IC göstermiyorsa eşit ağırlık
        w1 = w2 = w3 = 1.0 / 3
        log.warning("IC değerleri çok düşük — eşit ağırlık kullanılıyor")
    else:
        w1 /= toplam
        w2 /= toplam
        w3 /= toplam

    log.info("IC ağırlıkları: P1=%.2f P2=%.2f P3=%.2f", w1, w2, w3)

    # Her stratejinin sinyallerini normalize et
    n1 = normalize_skor(p1_sinyaller, "P1")
    n2 = normalize_skor(p2_sinyaller, "P2")
    n3 = normalize_skor(p3_sinyaller, "P3")

    # Sembol bazında birleştir
    sembol_skorlar: dict[str, dict] = {}

    for sinyal_grubu, agirlik in [(n1, w1), (n2, w2), (n3, w3)]:
        for s in sinyal_grubu:
            sym  = s["symbol"]
            nskor = s.get("norm_score", 0) * agirlik
            if sym not in sembol_skorlar:
                sembol_skorlar[sym] = {
                    "symbol":      sym,
                    "meta_score":  0.0,
                    "kaynaklar":   [],
                    "ic_agirlik":  0.0,
                }
            sembol_skorlar[sym]["meta_score"]  += nskor
            sembol_skorlar[sym]["kaynaklar"].append(s["strateji"])
            sembol_skorlar[sym]["ic_agirlik"] = max(
                sembol_skorlar[sym]["ic_agirlik"], agirlik
            )

    # Skora göre sırala
    sirali = sorted(
        sembol_skorlar.values(),
        key=lambda x: x["meta_score"],
        reverse=True,
    )

    # Bonus: birden fazla strateji aynı hisseyi seçiyorsa ek puan
    for s in sirali:
        n_kaynak = len(set(s["kaynaklar"]))
        if n_kaynak >= 2:
            s["meta_score"] *= (1 + 0.1 * (n_kaynak - 1))
            s["coklu_onay"] = True
        else:
            s["coklu_onay"] = False

    # Yeniden sırala (bonus sonrası)
    sirali.sort(key=lambda x: x["meta_score"], reverse=True)
    return sirali


# ---------------------------------------------------------------------------
# Half-Kelly Pozisyon Boyutlandırma — ML4AT Ch. 8
# ---------------------------------------------------------------------------

def half_kelly_boyut(
    win_rate: float,
    avg_win:  float,
    avg_loss: float,
    sermaye:  float,
) -> float:
    """
    Half-Kelly pozisyon büyüklüğü (TL).
    Kelly = (p × b - q) / b
    b = avg_win / |avg_loss| (kazanç/kayıp oranı)
    p = win_rate, q = 1 - win_rate
    Half-Kelly = Kelly × 0.5 (daha muhafazakâr)
    """
    if avg_loss <= 0 or avg_win <= 0:
        return sermaye / MAX_POS

    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate

    kelly = (p * b - q) / b
    kelly = max(0, min(kelly, 0.25))  # Max %25 bir pozisyona
    half_kelly_pct = kelly * 0.5

    return sermaye * half_kelly_pct


# ---------------------------------------------------------------------------
# Portföy Güncelleme
# ---------------------------------------------------------------------------

def portfoy_guncelle(state: dict, fiyat_cache: dict[str, np.ndarray]) -> dict:
    """
    Açık pozisyonları kontrol et: stop/TP/max gün.
    Returns: {"kapanan": [...], "devam_eden": [...]}
    """
    bugun      = date.today().isoformat()
    kapananlar = []
    devam      = {}

    for sym, pos in list(state["pozisyonlar"].items()):
        gun       = pos.get("gun", 0) + 1
        giris_fiy = pos.get("giris_fiyat", 0)

        # Güncel fiyat
        if sym in fiyat_cache and len(fiyat_cache[sym]) > 0:
            guncel = float(fiyat_cache[sym][-1])
        else:
            guncel = giris_fiy

        pnl_pct = (guncel - giris_fiy) / giris_fiy if giris_fiy > 0 else 0

        # Stop / TP / Max gün kontrolü
        neden = None
        if pnl_pct <= STOP_PCT:
            neden = "STOP"
        elif pnl_pct >= TP_PCT:
            neden = "TP"
        elif gun >= MAX_GUN:
            neden = "MAX_GUN"

        if neden:
            trade = {
                "symbol":       sym,
                "giris_fiyat":  giris_fiy,
                "cikis_fiyat":  guncel,
                "pnl_pct":      round(pnl_pct * 100, 2),
                "gun":          gun,
                "neden":        neden,
                "giris_tarih":  pos.get("giris_tarih", ""),
                "cikis_tarih":  bugun,
                "strateji":     pos.get("strateji", "P4"),
            }
            state["trade_history"].append(trade)
            kapananlar.append(trade)
            log.info("P4 pozisyon kapandı: %s %s pnl=%.1f%%", sym, neden, pnl_pct * 100)
        else:
            pos["gun"]           = gun
            pos["guncel_fiyat"]  = guncel
            pos["pnl_pct"]       = round(pnl_pct * 100, 2)
            devam[sym]           = pos

    state["pozisyonlar"] = devam
    return {"kapanan": kapananlar, "devam_eden": list(devam.keys())}


def yeni_pozisyon_ac(
    state:     dict,
    adaylar:   list[dict],
    fiyat_cache: dict[str, np.ndarray],
) -> list[dict]:
    """En iyi adaylardan boş slotları doldur."""
    bugun    = date.today().isoformat()
    acilan   = []
    mevcut   = set(state["pozisyonlar"].keys())
    bos_slot = MAX_POS - len(mevcut)

    if bos_slot <= 0:
        return []

    # Win rate ve avg win/loss hesapla (Half-Kelly için)
    trades   = state.get("trade_history", [])
    wins     = [t for t in trades if t.get("pnl_pct", 0) > 0]
    losses   = [t for t in trades if t.get("pnl_pct", 0) <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.55
    avg_win  = abs(np.mean([t["pnl_pct"] for t in wins])) / 100 if wins else 0.08
    avg_loss = abs(np.mean([t["pnl_pct"] for t in losses])) / 100 if losses else 0.05

    pos_tl = half_kelly_boyut(win_rate, avg_win, avg_loss, SERMAYE)
    pos_tl = max(pos_tl, SERMAYE / (MAX_POS * 2))   # Min 10k TL
    pos_tl = min(pos_tl, SERMAYE / MAX_POS * 1.5)   # Max 30k TL

    for aday in adaylar:
        if bos_slot <= 0:
            break
        sym = aday["symbol"]
        if sym in mevcut:
            continue

        fiyat = float(fiyat_cache[sym][-1]) if sym in fiyat_cache else 0
        if fiyat <= 0:
            continue

        lotlar = int(pos_tl / fiyat) if fiyat > 0 else 0
        if lotlar <= 0:
            continue

        state["pozisyonlar"][sym] = {
            "symbol":       sym,
            "giris_fiyat":  fiyat,
            "guncel_fiyat": fiyat,
            "lotlar":       lotlar,
            "pnl_pct":      0.0,
            "gun":          0,
            "giris_tarih":  bugun,
            "meta_score":   aday.get("meta_score", 0),
            "kaynaklar":    aday.get("kaynaklar", []),
            "coklu_onay":   aday.get("coklu_onay", False),
            "strateji":     "P4",
        }
        mevcut.add(sym)
        bos_slot -= 1
        acilan.append(state["pozisyonlar"][sym])
        log.info("P4 yeni pozisyon: %s fiyat=%.2f lot=%d", sym, fiyat, lotlar)

    return acilan


# ---------------------------------------------------------------------------
# Ana Akış
# ---------------------------------------------------------------------------

def calistir():
    log.info("P4 Meta Portföy başlıyor...")

    state = state_yukle()
    state["last_run"] = datetime.now().isoformat()

    # 1. Tüm stratejilerden sinyal ve state yükle
    p1_sig, p1_state = p1_sinyalleri_yukle()
    p2_sig, p2_state = p2_sinyalleri_yukle()
    p3_sig, p3_state = p3_sinyalleri_yukle()

    log.info("Sinyaller: P1=%d P2=%d P3=%d", len(p1_sig), len(p2_sig), len(p3_sig))

    # 2. Fiyat verisi çek
    tum_semboller = list(set(
        [s["symbol"] for s in p1_sig] +
        [s["symbol"] for s in p2_sig] +
        [s["symbol"] for s in p3_sig] +
        list(state["pozisyonlar"].keys())
    ))
    log.info("Fiyat çekiliyor: %d sembol", len(tum_semboller))
    fiyat_cache = fiyat_cek(tum_semboller)

    # 3. Her stratejinin IC'sini hesapla
    # Sinyal logunu state'den çek (geçmiş sinyaller varsa)
    p1_log = p1_state.get("signal_log", p1_sig)
    p2_log = p2_state.get("signal_log", p2_sig)
    p3_log = p3_state.get("signal_log", p3_sig)

    ic_p1 = strateji_ic_hesapla("P1", p1_log, fiyat_cache)
    ic_p2 = strateji_ic_hesapla("P2", p2_log, fiyat_cache)
    ic_p3 = strateji_ic_hesapla("P3", p3_log, fiyat_cache)

    # IC geçmişini state'e kaydet
    state["ic_history"].append({
        "tarih": date.today().isoformat(),
        "ic_p1": round(ic_p1, 4),
        "ic_p2": round(ic_p2, 4),
        "ic_p3": round(ic_p3, 4),
    })
    state["ic_history"] = state["ic_history"][-90:]  # Son 90 gün

    # 4. IC ağırlıklı sinyal birleştirme
    adaylar = ic_agirlikli_birlestir(
        p1_sig, p2_sig, p3_sig,
        ic_p1, ic_p2, ic_p3,
    )
    log.info("Birleşik aday: %d sembol", len(adaylar))

    # 5. Mevcut pozisyonları güncelle
    sonuc = portfoy_guncelle(state, fiyat_cache)
    log.info("Kapanan: %d | Devam: %d",
             len(sonuc["kapanan"]), len(sonuc["devam_eden"]))

    # 6. Yeni pozisyon aç
    acilan = yeni_pozisyon_ac(state, adaylar, fiyat_cache)
    log.info("Açılan: %d yeni pozisyon", len(acilan))

    # 7. State kaydet
    state_kaydet(state)

    # 8. Telegram bildirimi
    try:
        from mott_telegram import p4_mesaj, telegram_gonder
        mesaj = p4_mesaj(
            secilen  = adaylar[:MAX_POS],
            portfoy  = state,
            ic_scores = {"P1": ic_p1, "P2": ic_p2, "P3": ic_p3},
        )
        telegram_gonder(mesaj)
    except Exception as e:
        log.warning("Telegram hatası: %s", e)

    return {
        "ic":     {"P1": ic_p1, "P2": ic_p2, "P3": ic_p3},
        "adaylar": len(adaylar),
        "acilan":  len(acilan),
        "kapanan": len(sonuc["kapanan"]),
        "portfoy": len(state["pozisyonlar"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="P4 Meta Portföy")
    parser.add_argument("--durum", action="store_true", help="Portföy durumunu göster")
    args = parser.parse_args()

    if args.durum:
        state = state_yukle()
        print(json.dumps(state, indent=2, ensure_ascii=False))
    else:
        sonuc = calistir()
        print(f"\nP4 tamamlandı:")
        print(f"  IC: P1={sonuc['ic']['P1']:.3f} P2={sonuc['ic']['P2']:.3f} P3={sonuc['ic']['P3']:.3f}")
        print(f"  Aday: {sonuc['adaylar']} | Açılan: {sonuc['acilan']} | Kapanan: {sonuc['kapanan']}")
        print(f"  Portföy: {sonuc['portfoy']}/{MAX_POS} pozisyon")
