"""
p5_committee.py — P5 Komite Portföy Yöneticisi
================================================
P4'ten bağımsız; yalnızca P1 + P2 + P3 sinyallerine bakar.

Felsefe: Muhafazakâr konsensüs — min 2 strateji onayı, makro kapısı,
sektör limiti, eşit ağırlık pozisyon.

Çalıştır:
    python p5_committee.py
    python p5_committee.py --durum
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np
import yfinance as yf

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(os.environ.get("MOTT_BASE_DIR", "."))
MAX_POS = 5
SERMAYE = 100_000
POS_TL = SERMAYE / MAX_POS
STOP_PCT = -0.05
TP_PCT = 0.10
MAX_GUN = 10
MIN_KAYNAK = 2
MAX_SEKTOR = 2

# scanner_dsp sektör haritası
try:
    from scanner_dsp import get_sector, SECTOR_MAP
except ImportError:
    def get_sector(symbol: str) -> str:
        return "DIGER"
    SECTOR_MAP = {}


# ---------------------------------------------------------------------------
# Veri yükleme — yalnızca P1, P2, P3
# ---------------------------------------------------------------------------

def _json_yukle(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def p1_top_set(limit: int = 15) -> set[str]:
    s = _json_yukle(BASE_DIR / "state_p1.json")
    sigs = s.get("tarama", {}).get("signals", [])
    ranked = sorted(sigs, key=lambda x: -x.get("score_count", 0))
    return {x["symbol"] for x in ranked[:limit]}


def p2_top_set(limit: int = 15) -> set[str]:
    s = _json_yukle(BASE_DIR / "state_p2.json")
    sigs = s.get("tarama", {}).get("signals", [])
    ranked = sorted(sigs, key=lambda x: -float(x.get("score", 0)))
    return {x["symbol"] for x in ranked[:limit]}


def p3_top_set(limit: int = 15) -> set[str]:
    s = _json_yukle(BASE_DIR / "portfolio_state.json")
    log_items = s.get("scan_log", [])
    if log_items:
        top = log_items[-1].get("top5", [])
        return set(top[:limit])
    return {p for p in s.get("positions", {})}


def p1_detay() -> dict[str, dict]:
    s = _json_yukle(BASE_DIR / "state_p1.json")
    return {x["symbol"]: x for x in s.get("tarama", {}).get("signals", [])}


def p2_detay() -> dict[str, dict]:
    s = _json_yukle(BASE_DIR / "state_p2.json")
    return {x["symbol"]: x for x in s.get("tarama", {}).get("signals", [])}


def p3_detay() -> dict[str, dict]:
    s = _json_yukle(BASE_DIR / "portfolio_state.json")
    out = {}
    for sym in s.get("positions", {}):
        out[sym] = {"symbol": sym, "kaynak": "P3"}
    log_items = s.get("scan_log", [])
    if log_items:
        for sym in log_items[-1].get("top5", []):
            out.setdefault(sym, {"symbol": sym, "kaynak": "P3"})
    return out


def makro_karar() -> str:
    d = _json_yukle(BASE_DIR / "son_durum.json")
    return d.get("makro_karar", "NORMAL")


def drift_sembolleri() -> set[str]:
    s = _json_yukle(BASE_DIR / "portfolio_state.json")
    alerts: set[str] = set()
    for entry in s.get("cycle_monitor", {}).get("log", []):
        alerts.update(entry.get("alert_symbols", []))
    return alerts


def fiyat_cek(semboller: list[str]) -> dict[str, np.ndarray]:
    """yfinance geçmişi + son bar TradingView canlı fiyatıyla güncellenir."""
    cache = {}
    for sym in semboller:
        try:
            ticker = f"{sym}.IS" if not sym.endswith(".IS") else sym
            df = yf.Ticker(ticker).history(period="3mo")
            prices = df["Close"].dropna().values
            if len(prices) >= 5:
                cache[sym] = prices.copy()
        except Exception as e:
            log.debug("%s fiyat: %s", sym, e)

    try:
        from mott_fiyat import tv_fiyatlar
        canli = tv_fiyatlar(semboller)
        for sym, p in canli.items():
            if sym in cache:
                cache[sym][-1] = p
            else:
                cache[sym] = np.array([p])
    except Exception as e:
        log.warning("TV canlı fiyat alınamadı, yfinance değerleri kullanılacak: %s", e)
    return cache


# ---------------------------------------------------------------------------
# Komite skorlama
# ---------------------------------------------------------------------------

def komite_adaylari() -> tuple[list[dict], list[dict]]:
    """
    Returns: (secilen_adaylar, elenen_adaylar)
    """
    p1_set = p1_top_set()
    p2_set = p2_top_set()
    p3_set = p3_top_set()
    p1d, p2d, p3d = p1_detay(), p2_detay(), p3_detay()
    drift = drift_sembolleri()
    makro = makro_karar()

    tum = p1_set | p2_set | p3_set
    adaylar: list[dict] = []
    elenen: list[dict] = []

    for sym in tum:
        kaynaklar = []
        if sym in p1_set:
            kaynaklar.append("P1")
        if sym in p2_set:
            kaynaklar.append("P2")
        if sym in p3_set:
            kaynaklar.append("P3")

        n = len(kaynaklar)
        if n < MIN_KAYNAK:
            elenen.append({"symbol": sym, "neden": f"tek kaynak ({','.join(kaynaklar)})"})
            continue
        if makro == "GIRME":
            elenen.append({"symbol": sym, "neden": "makro GIRME"})
            continue
        if sym in drift:
            elenen.append({"symbol": sym, "neden": "P3 drift uyarısı"})
            continue

        skor = 60 if n == 2 else 100
        p2i = p2d.get(sym, {})
        if p2i.get("verdict") == "GÜÇLÜ AL":
            skor += 15
        p1i = p1d.get(sym, {})
        if p1i.get("score_count", 0) >= 3:
            skor += 10
        if sym in p3_set and sym in list(p3_top_set(3)):
            skor += 10

        adaylar.append({
            "symbol": sym,
            "kaynaklar": kaynaklar,
            "komite_skor": skor,
            "sektor": get_sector(sym),
        })

    # Sektör limiti ile sırala ve seç
    adaylar.sort(key=lambda x: -x["komite_skor"])
    secilen: list[dict] = []
    sektor_say = Counter()
    for ad in adaylar:
        sec = ad["sektor"]
        if sektor_say[sec] >= MAX_SEKTOR:
            elenen.append({"symbol": ad["symbol"], "neden": f"sektör dolu ({sec})"})
            continue
        secilen.append(ad)
        sektor_say[sec] += 1
        if len(secilen) >= MAX_POS:
            break

    return secilen, elenen


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def state_yukle() -> dict:
    path = BASE_DIR / "state_p5.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "pozisyonlar": {},
        "trade_history": [],
        "komite_log": [],
        "last_run": "",
        "sermaye_baslangic": SERMAYE,
        "sermaye_mevcut": SERMAYE,
    }


def state_kaydet(state: dict) -> None:
    try:
        from mott_portfoy_deger import equity_hesapla
        state["sermaye_mevcut"] = equity_hesapla("P5", state)["equity"]
    except Exception:
        pass
    with open(BASE_DIR / "state_p5.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _elde_tutma_gunu(giris_tarih: str) -> int:
    """Giriş tarihinden bugüne kadar geçen takvim günü (bkz. meta_portfolio.py)."""
    try:
        return (date.today() - date.fromisoformat(giris_tarih)).days
    except Exception:
        return 0


def portfoy_guncelle(state: dict, fiyat_cache: dict) -> dict:
    bugun = date.today().isoformat()
    kapananlar = []
    devam = {}
    for sym, pos in list(state.get("pozisyonlar", {}).items()):
        gun = _elde_tutma_gunu(pos.get("giris_tarih", ""))
        giris = pos.get("giris_fiyat", 0)
        guncel = float(fiyat_cache[sym][-1]) if sym in fiyat_cache else giris
        pnl = (guncel - giris) / giris if giris > 0 else 0

        # Split/bedelsiz koruması: BIST günlük taban limiti %10 — son kayıtlı
        # fiyata göre %10'u aşan düşüş piyasa hareketi olamaz (bkz. meta_portfolio.py)
        onceki = pos.get("guncel_fiyat", giris)
        gunluk = (guncel - onceki) / onceki if onceki > 0 else 0
        if gunluk < -0.10 and guncel > 0:
            oran = onceki / guncel
            log.warning("P5 %s: son kayittan bu yana %.1f%% düşüş — muhtemel split (oran~%.2f), düzeltiliyor.",
                        sym, gunluk * 100, oran)
            giris = round(giris / oran, 4) if oran > 0 else giris
            pos["giris_fiyat"] = giris
            pos["split_duzeltme"] = round(oran, 4)
            pos["split_tarih"] = bugun
            pnl = (guncel - giris) / giris if giris > 0 else 0
            try:
                from mott_telegram import telegram_gonder
                telegram_gonder(
                    f"⚠️ *P5 — Bölünme şüphesi: {sym}*\n"
                    f"Son kayıtlı fiyat {onceki:.2f} → güncel {guncel:.2f} ({gunluk*100:+.1f}%).\n"
                    f"BIST günlük limiti aşıldığı için bedelsiz/split varsayıldı; "
                    f"giriş fiyatı {oran:.2f} oranıyla düzeltildi. Lütfen KAP'tan doğrulayın."
                )
            except Exception:
                pass

        neden = None
        if pnl <= STOP_PCT:
            neden = "STOP"
        elif pnl >= TP_PCT:
            neden = "TP"
        elif gun >= MAX_GUN:
            neden = "MAX_GUN"
        if neden:
            trade = {
                "symbol": sym,
                "giris_fiyat": giris,
                "cikis_fiyat": guncel,
                "pnl_pct": round(pnl * 100, 2),
                "gun": gun,
                "neden": neden,
                "giris_tarih": pos.get("giris_tarih", ""),
                "cikis_tarih": bugun,
                "strateji": "P5",
            }
            state.setdefault("trade_history", []).append(trade)
            kapananlar.append(trade)
        else:
            pos["gun"] = gun
            pos["guncel_fiyat"] = guncel
            pos["pnl_pct"] = round(pnl * 100, 2)
            devam[sym] = pos
    state["pozisyonlar"] = devam
    return {"kapanan": kapananlar, "devam_eden": list(devam.keys())}


def yeni_pozisyon_ac(state: dict, adaylar: list[dict], fiyat_cache: dict) -> list[dict]:
    bugun = date.today().isoformat()
    acilan = []
    mevcut = set(state.get("pozisyonlar", {}).keys())
    bos = MAX_POS - len(mevcut)
    for ad in adaylar:
        if bos <= 0:
            break
        sym = ad["symbol"]
        if sym in mevcut:
            continue
        if sym not in fiyat_cache:
            continue
        fiyat = float(fiyat_cache[sym][-1])
        if fiyat <= 0:
            continue
        lotlar = int(POS_TL / fiyat)
        if lotlar < 1:
            continue
        state["pozisyonlar"][sym] = {
            "symbol": sym,
            "giris_fiyat": fiyat,
            "guncel_fiyat": fiyat,
            "lotlar": lotlar,
            "pnl_pct": 0.0,
            "gun": 0,
            "giris_tarih": bugun,
            "komite_skor": ad.get("komite_skor", 0),
            "kaynaklar": ad.get("kaynaklar", []),
            "strateji": "P5",
        }
        mevcut.add(sym)
        bos -= 1
        acilan.append(state["pozisyonlar"][sym])
    return acilan


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def calistir() -> dict:
    log.info("P5 Komite başlıyor...")
    state = state_yukle()
    state["last_run"] = datetime.now().isoformat()

    secilen, elenen = komite_adaylari()
    log.info("Komite aday: %d seçildi, %d elendi", len(secilen), len(elenen))

    semboller = list(
        set(s["symbol"] for s in secilen)
        | set(state.get("pozisyonlar", {}).keys())
    )
    fiyat_cache = fiyat_cek(semboller)

    sonuc = portfoy_guncelle(state, fiyat_cache)
    acilan = yeni_pozisyon_ac(state, secilen, fiyat_cache)

    state.setdefault("komite_log", []).append({
        "tarih": date.today().isoformat(),
        "secilen": [s["symbol"] for s in secilen],
        "elenen": elenen[:10],
        "acilan": [p["symbol"] for p in acilan],
        "kapanan": [t["symbol"] for t in sonuc["kapanan"]],
        "makro": makro_karar(),
    })
    state["komite_log"] = state["komite_log"][-60:]
    state_kaydet(state)

    if os.environ.get("MOTT_MODE", "aksam") == "aksam":
        try:
            from mott_telegram import telegram_islem_gonder
            sinyaller = [
                {"symbol": s["symbol"], "score": s["komite_skor"], "strategies": s["kaynaklar"]}
                for s in secilen
            ]
            cikis_mesaj = [
                f"{'🛑' if t['neden']=='STOP' else '🎯' if t['neden']=='TP' else '⏰'} "
                f"*{t['neden']} - {t['symbol']}*\n   Çıkış: {t['cikis_fiyat']:.2f} | Getiri: {t['pnl_pct']:+.1f}%"
                for t in sonuc["kapanan"]
            ]
            telegram_islem_gonder(
                "P5",
                sinyaller=sinyaller,
                portfoy=state,
                giris=[p["symbol"] for p in acilan],
                cikis=[t["symbol"] for t in sonuc["kapanan"]],
                secilen=secilen,
                elenen=elenen,
                mesajlar=cikis_mesaj or None,
            )
        except Exception as e:
            log.warning("P5 Telegram: %s", e)

    return {
        "secilen": len(secilen),
        "acilan": len(acilan),
        "kapanan": len(sonuc["kapanan"]),
        "portfoy": len(state.get("pozisyonlar", {})),
    }


def monitor() -> dict:
    """
    Gün içi hafif kontrol (her ~15 dk çağrılır): yalnızca AÇIK pozisyonlar
    için STOP/TP/MAX_GUN kontrolü yapar ve tetiklenirse ANINDA Telegram
    gönderir. Yeni komite adayı seçimi burada YAPILMAZ (o yalnızca akşam
    `calistir()` çağrısında olur).
    """
    state = state_yukle()
    pozisyonlar = state.get("pozisyonlar", {})
    if not pozisyonlar:
        log.info("P5 monitor: açık pozisyon yok, atlandı")
        return {"kapanan": 0}

    fiyat_cache = fiyat_cek(list(pozisyonlar.keys()))
    sonuc = portfoy_guncelle(state, fiyat_cache)
    state_kaydet(state)
    log.info("P5 monitor: kapanan=%d devam=%d", len(sonuc["kapanan"]), len(sonuc["devam_eden"]))

    if sonuc["kapanan"]:
        try:
            from mott_telegram import telegram_islem_gonder
            cikis = [t["symbol"] for t in sonuc["kapanan"]]
            mesajlar = [
                f"{'🛑' if t['neden']=='STOP' else '🎯' if t['neden']=='TP' else '⏰'} "
                f"*{t['neden']} - {t['symbol']}*\n   Çıkış: {t['cikis_fiyat']:.2f} | Getiri: {t['pnl_pct']:+.1f}%"
                for t in sonuc["kapanan"]
            ]
            telegram_islem_gonder(
                "P5", sinyaller=[], portfoy=state,
                giris=[], cikis=cikis, secilen=[], elenen=[],
                mesajlar=mesajlar,
            )
        except Exception as e:
            log.warning("P5 monitor Telegram: %s", e)

    return {"kapanan": len(sonuc["kapanan"]), "devam_eden": len(sonuc["devam_eden"])}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P5 Komite Portföy")
    parser.add_argument("--durum", action="store_true")
    parser.add_argument("--monitor", action="store_true",
                         help="Gün içi hafif kontrol: yalnızca açık pozisyonlar için STOP/TP/MAX_GUN")
    args = parser.parse_args()
    if args.durum:
        print(json.dumps(state_yukle(), indent=2, ensure_ascii=False))
    elif args.monitor:
        r = monitor()
        print(f"P5 monitor: kapanan={r.get('kapanan', 0)}")
    else:
        r = calistir()
        print(f"P5 tamamlandı: {r}")
