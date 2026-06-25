# -*- coding: utf-8 -*-
import json, logging, os, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# .env dosyasından değerleri yükle
# Ortam değişkenleri GitHub Actions Secrets'tan gelir
# Yerel test için .env dosyası opsiyonel
def _load_env():
    env_file = os.environ.get("ENV_FILE", ".env")
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass
_load_env()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# GitHub Actions'ta state dosyası repo kök dizininde
BASE_DIR  = Path(os.environ.get("MOTT_BASE_DIR", "."))
TMP_BASE  = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")
SIGNAL_LOG= TMP_BASE / "scan_history_p1.jsonl"

try:
    yf.set_tz_cache_location("/tmp/yf_tz")
except Exception:
    pass

try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="chrome110")
    def _ticker(symbol):
        return yf.Ticker(f"{symbol}.IS", session=_SESSION)
except Exception:
    _SESSION = None
    def _ticker(symbol):
        return yf.Ticker(f"{symbol}.IS")
        return yf.Ticker(f"{symbol}.IS")

_VARSAYILAN_SEMBOLLER = [
    "AKBNK", "AKSA", "AKSEN", "AKGRT", "ALKIM", "ANSGR", "ARCLK", "ARDYZ", "ARSAN", "ASELS",
    "ASTOR", "ASUZU", "AYGAZ", "BAGFS", "BERA", "BFREN", "BIMAS", "BORLS", "BRISA", "BRSAN",
    "BTCIM", "BUCIM", "BURCE", "BVSAN", "CASA", "CATES", "CCOLA", "CELHA", "CEMTS", "CIMSA",
    "CLEBI", "CWENE", "DAGI", "DENGE", "DEVA", "DITAS", "DOAS", "DOFER", "DOKTA", "ECILC",
    "ECZYT", "EFOR", "EGEEN", "EGSER", "EKGYO", "ELITE", "ENKAI", "ENSRI", "ENTRA", "EREGL",
    "ERSU", "FADE", "FENER", "FONET", "FROTO", "GARAN", "GARFA", "GEDZA", "GESAN", "GLCVY",
    "GLYHO", "GOODY", "GRSEL", "GSDHO", "GUBRF", "HATEK", "HALKB", "HATSN", "HEKTS", "HOROZ",
    "ICBCT", "IEYHO", "IHLGM", "IMASM", "INDES", "INFO", "INGRM", "INTEM", "INVEO", "ISBIR",
    "ISCTR", "ISDMR", "ISGSY", "ISKPL", "KAREL", "KARSN", "KARTN", "KCHOL", "KGYO", "KLGYO",
    "KLMSN", "KMPUR", "KNFRT", "KONYA", "KORDS", "KRDMA", "KRDMB", "KRDMD", "KRGYO", "KSTUR",
    "KTSKR", "KUYAS", "LIDFA", "LINK", "LOGO", "MAKIM", "MANAS", "MARTI", "MAVI", "MEGMT",
    "MGROS", "MNDRS", "MOBTL", "NATEN", "NETAS", "NTGAZ", "NUHCM", "OBASE", "ODAS", "ORGE",
    "PAGYO", "PENGD", "PETKM", "PGSUS", "PKENT", "PNLSN", "PRKAB", "RALYH", "RAYSG", "REEDR",
    "RUZYE", "SAFKR", "SAHOL", "SANKO", "SARKY", "SASA", "SILVR", "SKTAS", "SNICA", "TABGD",
    "TARKM", "THYAO", "TKFEN", "TKNSA", "TOASO", "TRGYO", "TRMET", "TSGYO", "TTKOM", "TTRAK",
    "TUPRS", "TUREX", "TURGG", "TURSG", "UFUK", "ULUFA", "ULUSE", "UNLU", "VAKBN", "VAKKO",
    "VKING", "YATAS", "YIGIT", "ZOREN", "ULKER", "BRKSN", "GWIND", "ALARK", "PAHOL", "BALSU",
    "SISE", "KCAER", "MIATK", "MOGAN", "GLRMK", "BORSK", "PATEK", "PSGYO", "UCAYM", "OBAMS",
    "KLYPV",
]

def _hisse_listesi_yukle():
    txt_yol = BASE_DIR / "hisseler.txt"
    if txt_yol.exists():
        try:
            semboller = [
                line.strip().upper()
                for line in txt_yol.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            return list(dict.fromkeys(semboller))
        except Exception as exc:
            log.warning("hisseler.txt okunamadi: %s", exc)
    return list(_VARSAYILAN_SEMBOLLER)

_VERI_CACHE = {}

def aktif_hisse_listesi():
    return _hisse_listesi_yukle()

def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / (l + 1e-9)))

def _macd(s, fast=12, slow=26, sig=9):
    ml = _ema(s, fast) - _ema(s, slow)
    return ml, _ema(ml, sig)

def _bbands(s, p=20):
    mid = _sma(s, p)
    std = s.rolling(p).std()
    return mid, mid + 2*std, mid - 2*std

def _cmf(h, l, c, v, p=20):
    mfv = ((c - l) - (h - c)) / (h - l + 1e-9) * v
    return mfv.rolling(p).sum() / (v.rolling(p).sum() + 1e-9)

def _adx(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(p).mean()
    up = h.diff(); dn = -l.diff()
    dmp = _ema(up.clip(lower=0), p)
    dmn = _ema(dn.clip(lower=0), p)
    di_p = 100 * dmp / (atr + 1e-9)
    di_n = 100 * dmn / (atr + 1e-9)
    dx = 100 * (di_p - di_n).abs() / (di_p + di_n + 1e-9)
    return dx.rolling(p).mean(), di_p, di_n

def _stochrsi(s, p=14, k=3):
    r = _rsi(s, p)
    lo = r.rolling(p).min(); hi = r.rolling(p).max()
    return (100 * (r - lo) / (hi - lo + 1e-9)).rolling(k).mean()

def _atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def veri_hazirla(semboller):
    global _VERI_CACHE
    _VERI_CACHE = {}
    tickers = [f"{s}.IS" for s in semboller]
    log.info("Toplu veri cekiliyor: %s hisse", len(tickers))
    try:
        kwargs = dict(tickers=tickers, period="8mo", interval="1d",
                      group_by="ticker", auto_adjust=True, progress=False, threads=True)
        if _SESSION is not None:
            kwargs["session"] = _SESSION
        raw = yf.download(**kwargs)
    except Exception as exc:
        log.error("Toplu indirme hatasi: %s", exc)
        return 0
    if raw is None or raw.empty:
        return 0
    ok = 0
    for sym in semboller:
        ticker = f"{sym}.IS"
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker][["Open","High","Low","Close","Volume"]].copy()
            else:
                df = raw[["Open","High","Low","Close","Volume"]].copy()
            df.columns = [c.lower() for c in df.columns]
            df["volume"] = df["volume"].ffill()
            df = df.dropna(subset=["open","high","low","close"])
            if len(df) >= 50:
                _VERI_CACHE[sym] = df
                ok += 1
        except Exception:
            continue
    log.info("Veri hazir: %s/%s", ok, len(semboller))
    return ok

def fetch_ohlcv(symbol):
    df = _VERI_CACHE.get(symbol)
    if df is None or len(df) < 50:
        return None, None
    try:
        mc = df["close"].iloc[-1] * df["volume"].iloc[-20:].mean()
    except Exception:
        mc = 0
    return df, mc

def get_indicators(df):
    try:
        c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]
        ema8   = _ema(c, 8).iloc[-1]
        ema21  = _ema(c, 21).iloc[-1]
        ema50  = _ema(c, 50).iloc[-1]
        ema200 = _ema(c, 200).iloc[-1] if len(df) >= 200 else float("nan")
        sma20  = _sma(c, 20).iloc[-1]
        rsi    = _rsi(c).iloc[-1]
        macd_l, macd_s = _macd(c)
        macd_val  = macd_l.iloc[-1]
        macd_sig  = macd_s.iloc[-1]
        macd_prev = macd_l.iloc[-2]
        macd_sprev= macd_s.iloc[-2]
        bb_mid, bb_up, bb_lo = _bbands(c)
        bb_mid = bb_mid.iloc[-1]; bb_up = bb_up.iloc[-1]; bb_lo = bb_lo.iloc[-1]
        cmf    = _cmf(h, l, c, v).iloc[-1]
        adx_s, di_p, di_n = _adx(h, l, c)
        adx    = adx_s.iloc[-1]; dip = di_p.iloc[-1]; din = di_n.iloc[-1]
        srsi   = _stochrsi(c).iloc[-1]
        atr_v  = _atr(h, l, c).iloc[-1]
        vol20  = v.iloc[-20:].mean()
        rel_vol= v.iloc[-1] / (vol20 + 1e-9)
        change_pct = (c.iloc[-1] - c.iloc[-2]) / (c.iloc[-2] + 1e-9) * 100
        close  = c.iloc[-1]
        # Alpha Trend
        alpha_atr = _atr(h, l, c, 14)
        src = (h + l) / 2
        up_t = src - 1.5 * alpha_atr; dn_t = src + 1.5 * alpha_atr
        alpha = pd.Series(index=df.index, dtype=float)
        alpha.iloc[0] = up_t.iloc[0]
        for i in range(1, len(df)):
            if c.iloc[i-1] > alpha.iloc[i-1]:
                alpha.iloc[i] = max(up_t.iloc[i], alpha.iloc[i-1])
            else:
                alpha.iloc[i] = min(dn_t.iloc[i], alpha.iloc[i-1])
        alpha_cur  = alpha.iloc[-1]; alpha_prev = alpha.iloc[-2]
        alpha_bull = (c.iloc[-1] > alpha_cur) and (c.iloc[-2] <= alpha_prev)
        alpha_trend_bull = c.iloc[-1] > alpha_cur
        return dict(
            close=close, ema8=ema8, ema21=ema21, ema50=ema50, ema200=ema200,
            sma20=sma20, rsi=rsi, macd=macd_val, macd_sig=macd_sig,
            macd_prev=macd_prev, macd_sprev=macd_sprev,
            bb_mid=bb_mid, bb_up=bb_up, bb_lo=bb_lo,
            cmf=cmf, adx=adx, di_p=dip, di_n=din,
            stochrsi=srsi, atr=atr_v, rel_vol=rel_vol,
            change_pct=change_pct, alpha_bull=alpha_bull,
            alpha_trend_bull=alpha_trend_bull,
        )
    except Exception as exc:
        log.debug("Indikatör hatası %s", exc)
        return None

def strategy_guclu_trend(ind, mc):
    if mc < 10_000_000: return False
    return (ind["ema8"] > ind["ema21"] > ind["ema50"] and
            (pd.isna(ind["ema200"]) or ind["close"] > ind["ema200"]) and
            50 <= ind["rsi"] <= 65 and ind["adx"] > 20 and
            ind["di_p"] > ind["di_n"] and ind["cmf"] > 0 and ind["rel_vol"] >= 1)

def strategy_guclu_trend_devam(ind, mc):
    if mc < 10_000_000: return False
    return (ind["ema8"] > ind["ema21"] > ind["ema50"] and
            (pd.isna(ind["ema200"]) or ind["close"] > ind["ema200"]) and
            65 <= ind["rsi"] <= 78 and ind["adx"] > 25 and
            ind["di_p"] > ind["di_n"] and ind["cmf"] > 0.05 and ind["rel_vol"] >= 1.2)

def strategy_alpha(ind, mc):
    if mc < 10_000_000: return False
    return (ind["alpha_bull"] and ind["rsi"] > 45 and
            ind["adx"] > 18 and ind["cmf"] > -0.05 and ind["rel_vol"] >= 1)

def strategy_zkn(ind, mc):
    if mc < 10_000_000: return False
    return (ind["close"] > ind["ema50"] and
            (pd.isna(ind["ema200"]) or ind["close"] > ind["ema200"]) and
            40 <= ind["rsi"] <= 58 and ind["stochrsi"] < 40 and
            ind["cmf"] > -0.1 and ind["rel_vol"] >= 0.8)

def strategy_ztan3(ind, mc):
    if mc < 10_000_000: return False
    return (ind["adx"] > 25 and ind["di_p"] > ind["di_n"] and
            ind["rsi"] > 50 and ind["macd"] > ind["macd_sig"] and
            ind["macd_prev"] <= ind["macd_sprev"] and
            ind["alpha_trend_bull"] and ind["rel_vol"] >= 1)

def strategy_mini_ralli(ind, mc):
    if mc < 10_000_000: return False
    return (ind["rsi"] > 55 and ind["macd"] > ind["macd_sig"] and
            ind["macd"] > 0 and ind["close"] > ind["sma20"] and
            ind["rel_vol"] >= 1.1 and ind["change_pct"] > 0)

def strategy_karisik_bb_macd(ind, mc):
    if mc < 10_000_000: return False
    return (ind["close"] > ind["bb_mid"] and
            ind["macd"] > ind["macd_sig"] and
            ind["macd_prev"] <= ind["macd_sprev"] and
            ind["rsi"] > 48 and ind["cmf"] > 0 and ind["rel_vol"] >= 0.9)

def strategy_dip(ind, mc):
    if mc < 10_000_000: return False
    return (ind["rsi"] < 30 and ind["stochrsi"] < 20 and
            ind["close"] < ind["bb_lo"] and ind["cmf"] < -0.1 and
            ind["change_pct"] > -1 and ind["rel_vol"] >= 1)

def _build_signal_records(scan_time, scan_label, strategy_results):
    seen = {}
    for strat, results in strategy_results.items():
        for r in results:
            sym = r["symbol"]
            if sym not in seen:
                seen[sym] = {"symbol": sym, "scan_time": scan_time,
                             "scan_label": scan_label, "strategies": [],
                             "ind": r["ind"]}
            seen[sym]["strategies"].append(strat)
    records = []
    for sym, data in seen.items():
        ind = data["ind"]
        records.append({
            "symbol": sym, "scan_time": scan_time, "scan_label": scan_label,
            "strategies": data["strategies"], "score_count": len(data["strategies"]),
            "rsi": round(ind["rsi"], 2), "rel_vol": round(ind["rel_vol"], 2),
            "change_pct": round(ind["change_pct"], 2), "close": round(ind["close"], 4),
            "adx": round(ind["adx"], 2), "cmf": round(ind["cmf"], 4),
            "alpha_bull": bool(ind["alpha_bull"]), "alpha_trend_bull": bool(ind["alpha_trend_bull"]),
        })
    return records

def _append_signal_log(records):
    try:
        with open(SIGNAL_LOG, "a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("Signal log yazilamadi: %s", exc)

def send_telegram(msg):
    from datetime import datetime
    import pytz
    _now = datetime.now(pytz.timezone("Europe/Istanbul"))
    _zaman = _now.strftime('%d.%m.%Y | %H:%M TSİ')
    _header = (
        f"🟡 ━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>P1 — MOMENTUM TARAMA</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {_zaman}\n\n"
    )
    _footer = "\n🟡 ━━━━━━━━━━━━━━━━━━━━━━━━"
    msg = _header + msg + _footer

    if not BOT_TOKEN or not CHAT_ID:
        log.warning("Telegram ayarları eksik")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as exc:
        log.warning("Telegram gönderilemedi: %s", exc)

def vm_gonder(signal_records, scan_time, scan_label):
    """
    Eski: VM Flask API'ye HTTP POST
    Yeni: Direkt JSON dosyasına yaz — GitHub Actions uyumlu
    """
    payload = {
        "scan_time":  scan_time,
        "scan_label": scan_label,
        "signals":    signal_records,
        "semboller":  [x["symbol"] for x in signal_records],
    }
    state_file = BASE_DIR / "state_p1.json"
    try:
        existing = {}
        if state_file.exists():
            with open(state_file, encoding="utf-8") as fh:
                existing = json.load(fh)
        existing["tarama"] = payload
        existing["last_scan"] = scan_time
        with open(state_file, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)
        log.info("P1 tarama state_p1.json'a yazıldı: %d sinyal", len(signal_records))
        return {"status": "ok", "n": len(signal_records)}
    except Exception as exc:
        log.warning("state_p1.json yazma hatası: %s", exc)
        return {}

def format_message(res_gt, res_gtd, res_alpha, res_zkn, res_ztan, res_mr, res_kbms, res_dip, scan_time, scan_label, n_symbols):
    set_gt   = {r["symbol"] for r in res_gt}
    set_gtd  = {r["symbol"] for r in res_gtd}
    set_alpha= {r["symbol"] for r in res_alpha}
    set_zkn  = {r["symbol"] for r in res_zkn}
    set_ztan = {r["symbol"] for r in res_ztan}
    set_mr   = {r["symbol"] for r in res_mr}
    set_kbms = {r["symbol"] for r in res_kbms}
    set_dip  = {r["symbol"] for r in res_dip}
    all_sets = [set_gt, set_gtd, set_alpha, set_zkn, set_ztan, set_mr, set_kbms, set_dip]
    all_syms = set_gt | set_gtd | set_alpha | set_zkn | set_ztan | set_mr | set_kbms | set_dip

    def count_s(sym):
        return sum(1 for s in all_sets if sym in s)

    ortak_4p = sorted(s for s in all_syms if count_s(s) >= 4)
    ortak_3  = sorted(s for s in all_syms if count_s(s) == 3)
    ortak_2  = sorted(s for s in all_syms if count_s(s) == 2)
    score_dist = {"4+": len(ortak_4p), "3": len(ortak_3), "2": len(ortak_2),
                  "1": len([s for s in all_syms if count_s(s) == 1])}

    hour = int(scan_time.split(" ")[1].split(":")[0])
    saat_emoji = "\U0001F319" if hour < 6 else ("\u2600\ufe0f" if hour < 13 else "\U0001F306")
    lines = [f"{saat_emoji} <b>BIST Tarama - {scan_label}</b>", f"\U0001F4C5 {scan_time}", ""]

    def bolum(baslik, icon, results):
        rows = [f"{icon} <b>{baslik}</b>  ({len(results)} hisse)"]
        if not results:
            rows.append("   - sinyal yok")
        else:
            for r in results[:5]:
                i = r["ind"]
                em = "\U0001F7E2" if i["change_pct"] > 0 else "\U0001F534"
                cnt = count_s(r["symbol"])
                star = " \U0001F3C6" if cnt >= 4 else (" \u2b50\u2b50" if cnt == 3 else (" \u2b50" if cnt == 2 else ""))
                rows.append(
                    f"   <b>{r['symbol']}</b>{star} "
                    f"{em}{i['change_pct']:+.1f}% RSI:{i['rsi']:.0f} RV:{i['rel_vol']:.1f}"
                )
            if len(results) > 5:
                rows.append(f"   ... +{len(results)-5} hisse daha")
        rows.append("")
        return rows

    lines += bolum("G\u00fc\u00e7l\u00fc Trend", "\U0001F535", res_gt)
    lines += bolum("G\u00fc\u00e7l\u00fc Trend Devam", "\U0001F537", res_gtd)
    lines += bolum("Alpha Trend Flip", "\U0001F31F", res_alpha)
    lines += bolum("ZKN \u0130yi Al\u0131m Yeri", "\U0001F7E1", res_zkn)
    lines += bolum("ZTAN 3", "\U0001F534", res_ztan)
    lines += bolum("Mini Ralli", "\U0001F7E0", res_mr)
    lines += bolum("Kar\u0131\u015f\u0131k BB MACD", "\u26aa", res_kbms)
    lines += bolum("Dip D\u00f6n\u00fc\u015f\u00fc", "\U0001F7E4", res_dip)
    lines.append("--------------------")
    if ortak_4p:
        lines.append(f"\U0001F3C6 <b>4+ stratejide:</b> {', '.join(ortak_4p)}")
    if ortak_3:
        lines.append(f"\u2b50\u2b50 <b>3 stratejide:</b> {', '.join(ortak_3)}")
    if ortak_2:
        lines.append(f"\u2b50 <b>2 stratejide:</b> {', '.join(ortak_2)}")
    if not ortak_4p and not ortak_3 and not ortak_2:
        lines.append("- Bug\u00fcn ortak hisse yok")
    lines += [
        "",
        f"\U0001F4CA Taranan: {n_symbols} hisse | Tekil sinyal: {len(all_syms)}",
        f"\U0001F9EE Puan da\u011f\u0131l\u0131m\u0131: 4+={score_dist['4+']} | 3={score_dist['3']} | 2={score_dist['2']} | 1={score_dist['1']}",
        "\U0001F535 GT  \U0001F537 GTD  \U0001F31F ALPHA  \U0001F7E1 ZKN  \U0001F534 ZTAN3  \U0001F7E0 MR  \u26aa KBM  \U0001F7E4 DIP",
        "",
        "\u26a0\ufe0f <i>Yat\u0131r\u0131m tavsiyesi de\u011fildir.</i>",
    ]
    return "\n".join(lines)


def run_scan():
    tz_turkey = timezone(timedelta(hours=3))
    now = datetime.now(tz_turkey)
    scan_time = now.strftime("%d.%m.%Y %H:%M")
    hour = now.hour
    if 0 <= hour < 6:
        scan_label = "Gece / Manuel Tetikleme"
    elif 6 <= hour < 13:
        scan_label = "11:05 Sabah Taramas\u0131"
    elif 19 <= hour <= 23:
        scan_label = "20:50 Kapan\u0131\u015f"
    else:
        scan_label = "G\u00fcn \u0130\u00e7i Manuel Tetikleme"

    active_symbols = aktif_hisse_listesi()
    ok = veri_hazirla(active_symbols)
    if ok == 0:
        msg = f"\u26a0\ufe0f <b>BIST Tarama - {scan_label}</b>\nVeri \u00e7ekilemedi."
        send_telegram(msg)
        return

    res_gt, res_gtd, res_alpha, res_zkn = [], [], [], []
    res_ztan, res_mr, res_kbms, res_dip = [], [], [], []

    for i, symbol in enumerate(active_symbols):
        df, mc = fetch_ohlcv(symbol)
        if df is None: continue
        ind = get_indicators(df)
        if ind is None: continue
        if strategy_guclu_trend(ind, mc):       res_gt.append({"symbol": symbol, "ind": ind})
        if strategy_guclu_trend_devam(ind, mc): res_gtd.append({"symbol": symbol, "ind": ind})
        if strategy_alpha(ind, mc):             res_alpha.append({"symbol": symbol, "ind": ind})
        if strategy_zkn(ind, mc):               res_zkn.append({"symbol": symbol, "ind": ind})
        if strategy_ztan3(ind, mc):             res_ztan.append({"symbol": symbol, "ind": ind})
        if strategy_mini_ralli(ind, mc):        res_mr.append({"symbol": symbol, "ind": ind})
        if strategy_karisik_bb_macd(ind, mc):   res_kbms.append({"symbol": symbol, "ind": ind})
        if strategy_dip(ind, mc):               res_dip.append({"symbol": symbol, "ind": ind})
        if (i + 1) % 10 == 0:
            time.sleep(0.6)

    for lst in [res_gt, res_gtd, res_alpha, res_zkn, res_ztan, res_mr, res_kbms]:
        lst.sort(key=lambda x: -x["ind"]["rsi"])
    res_dip.sort(key=lambda x: x["ind"]["rsi"])

    strategy_results = {"GT": res_gt, "GTD": res_gtd, "ALPHA": res_alpha, "ZKN": res_zkn,
                        "ZT3": res_ztan, "MR": res_mr, "KBM": res_kbms, "DIP": res_dip}
    signal_records = _build_signal_records(scan_time, scan_label, strategy_results)
    _append_signal_log(signal_records)
    vm_gonder(signal_records, scan_time, scan_label)
    try:
        msg = format_message(res_gt, res_gtd, res_alpha, res_zkn, res_ztan, res_mr, res_kbms,
                             res_dip, scan_time, scan_label, len(active_symbols))
        send_telegram(msg)
        log.info("P1 Telegram gonderildi")
    except Exception as e:
        log.error("P1 Telegram hatasi: %s", e, exc_info=True)
    log.info("P1 tarama tamamlandi: %s sinyal", len(signal_records))


if __name__ == "__main__":
    run_scan()
