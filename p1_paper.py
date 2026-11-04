# -*- coding: utf-8 -*-
"""
p1_paper.py — P1 Momentum Paper Trading Motoru ve Günlük Döngüsü
================================================================
P1 araştırma sonuçlarına dayalı sanal portföy işletim altyapısı:
- Sadece PAPER modunda çalışır; gerçek broker emirleri GÖNDERMEZ.
- Backtest motorunun sinyal ve portföy mantığıyla birebir AYNI kodu import eder.
- Konfigürasyon: 100k TL sermaye, 20k TL slot, en fazla 5 pozisyon (paralel 10k eşdeğeri).
- Ajan önceliği: GTD (1) > ALPHA_B_SIMPLE (2) > ZT3 (3), RelVol azalan sırayla.
- Gölge ajan: GT_NO_RSI sermaye almaz, shadow_signals.jsonl'e loglanır.
- Gün sonu sinyal üretimi, ertesi gün açılışta emir dolumu (lookahead yok).
- Reddedilen sinyaller (KAPASITE_REDDI / GAP_REDDI) loglanır ve sonradan getirisi takip edilir.
- Tamamen idempotent: aynı döngü tekrar çağrıldığında mükerrer kayıt oluşmaz.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Repo kök dizini
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from p1_paper_config import (
    ACILIS_VWAP_WINDOW_END,
    ACILIS_VWAP_WINDOW_START,
    AGENTS_PRIORITY,
    ALERTS_LOG_FILE,
    BASE_DIR,
    COMMISSION_RATE,
    GAP_TOLERANCE,
    HEARTBEAT_FILE,
    INITIAL_CAPITAL,
    MAX_HOLDING_DAYS,
    MAX_POSITIONS,
    PAPER_LOG_FILE,
    PAPER_STATE_FILE,
    POSITION_SIZE,
    POSITION_SIZE_10K_EQ,
    ROUNDTRIP_COST_PCT,
    SHADOW_AGENTS,
    SHADOW_SIGNALS_FILE,
    SLIPPAGE_RATE,
    STOP_PCT,
    TP1_PCT,
    TRAILING_PCT,
    TZ_ISTANBUL,
    gap_ratio,
    gap_rejected,
    is_p1_paper_enabled,
    stop_level_price,
    tp1_level_price,
    tp1_sold_lots,
    trail_level_price,
)
from p1_monitoring import check_drawdown_alerts, check_fill_cost_alerts, warn_vwap_missing
from mott_bist_takvim import is_bist_islem_gunu
from p1_signal_parity import compare_daily_signals, evaluate_backtest_signals_for_day
from research_p1.data_downloader import load_cached_ohlcv, BIST100_SYMBOLS
from research_p1.p1_engine import compute_all_indicators
from research_p1.run_p1_revision_backtest import evaluate_custom_agent_signals

log = logging.getLogger("p1_paper")

# Tarihsel replay: shadow jsonl / KACIRILAN_GUN taraması kapatılır (canlı davranışı değiştirmez).
_REPLAY_MODE = False

# Log tablosu başlıkları (research_p1/results/paper_trading_log_template.csv ile birebir aynı + 2 kolon)
PAPER_LOG_HEADERS = [
    "sinyal_tarihi",
    "giris_tarihi",
    "sembol",
    "birincil_ajan",
    "tum_ajanlar",
    "ajan_bazli_rel_vol_sirasi",
    "rel_vol",
    "sinyal_kapanis_fiyati",
    "beklenen_acilis_fiyati",
    "gerceklesen_acilis_fiyati",
    "simule_dolum_fiyati",
    "gap_orani_pct",
    "gerceklesen_slipaj_pct",
    "komisyon_pct",
    "dolum_vekili_tur_basi_maliyet_pct",
    "onaylanan_lot",
    "tutar_tl",
    "kapasite_durumu",
    "xu100_rejim_boga",
    "cikis_tarihi",
    "cikis_fiyati",
    "cikis_nedeni",
    "net_getiri_pct",
    "net_pnl_tl",
    "net_pnl_10k_esdegeri_tl",
    "reddedilen_sonradan_getiri_pct",
    "notlar",
]


def init_paper_log() -> None:
    """Paper trading log CSV dosyasını başlıklarıyla oluşturur."""
    if not PAPER_LOG_FILE.exists() or PAPER_LOG_FILE.stat().st_size == 0:
        PAPER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PAPER_LOG_FILE, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(PAPER_LOG_HEADERS)


def load_paper_log() -> List[dict]:
    init_paper_log()
    with open(PAPER_LOG_FILE, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def save_paper_log(rows: List[dict]) -> None:
    init_paper_log()
    with open(PAPER_LOG_FILE, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAPER_LOG_HEADERS)
        writer.writeheader()
        for r in rows:
            # Eksik alanları varsayılan boş bırak
            row_dict = {k: r.get(k, "") for k in PAPER_LOG_HEADERS}
            writer.writerow(row_dict)


def load_paper_state() -> dict:
    """Sanal portföy durumunu diskten yükler."""
    if PAPER_STATE_FILE.exists():
        try:
            with open(PAPER_STATE_FILE, "r", encoding="utf-8") as fh:
                state = json.load(fh)
                return state
        except Exception as exc:
            log.warning("Paper state okunamadı (%s), varsayılan oluşturuluyor.", exc)

    return {
        "_gen": 0,
        "_updated_at": datetime.now(TZ_ISTANBUL).isoformat(),
        "cash_20k": INITIAL_CAPITAL,
        "cash_10k": INITIAL_CAPITAL,
        "peak_equity_20k": INITIAL_CAPITAL,
        "peak_equity_10k": INITIAL_CAPITAL,
        "positions_20k": {},
        "positions_10k": {},
        "pending_candidates": [],
        "tracked_rejected_signals": [],
        "daily_equity": [],
        "processed_runs": {},  # {"YYYY-MM-DD_phase": timestamp}
        "paper_start_date": "",
    }


def save_paper_state(state: dict) -> None:
    """Sanal portföy durumunu atomik olarak diske kaydeder."""
    state["_gen"] = state.get("_gen", 0) + 1
    state["_updated_at"] = datetime.now(TZ_ISTANBUL).isoformat()
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = PAPER_STATE_FILE.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    import time as _time
    last_exc = None
    for _attempt in range(8):
        try:
            tmp_file.replace(PAPER_STATE_FILE)
            last_exc = None
            break
        except PermissionError as exc:
            last_exc = exc
            _time.sleep(0.15 * (_attempt + 1))
    if last_exc is not None:
        raise last_exc


HEARTBEAT_HEADERS = ["tarih", "faz", "baslangic", "bitis", "durum"]
EXPECTED_DAILY_PHASES = ("aksam", "acilis", "takip")


def init_heartbeat_log() -> None:
    if not HEARTBEAT_FILE.exists() or HEARTBEAT_FILE.stat().st_size == 0:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HEARTBEAT_FILE, "w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(HEARTBEAT_HEADERS)


def write_heartbeat(tarih: str, faz: str, baslangic: str, bitis: str, durum: str) -> None:
    if _REPLAY_MODE:
        return
    init_heartbeat_log()
    with open(HEARTBEAT_FILE, "a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow([tarih, faz, baslangic, bitis, durum])


def load_heartbeat_rows() -> List[dict]:
    init_heartbeat_log()
    with open(HEARTBEAT_FILE, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def check_missed_trading_days(as_of: date) -> List[dict]:
    """Önceki BIST işlem günlerinde beklenen faz yoksa KACIRILAN_GUN yazar. Geçmişi telafi etmez."""
    state = load_paper_state()
    start_s = state.get("paper_start_date") or ""
    if not start_s:
        return []
    try:
        start_d = date.fromisoformat(str(start_s)[:10])
    except ValueError:
        return []

    rows = load_heartbeat_rows()
    done: Set[Tuple[str, str]] = set()
    for r in rows:
        if r.get("durum") in ("success", "skipped_idempotent", "no_pending_signals"):
            done.add((str(r.get("tarih", "")), str(r.get("faz", ""))))

    missed: List[dict] = []
    cur = start_d
    while cur < as_of:
        if is_bist_islem_gunu(cur):
            t_str = cur.isoformat()
            for faz in EXPECTED_DAILY_PHASES:
                key = (t_str, faz)
                already_flagged = any(
                    r.get("tarih") == t_str and r.get("faz") == faz and r.get("durum") == "KACIRILAN_GUN"
                    for r in rows
                )
                if key not in done and not already_flagged:
                    now_iso = datetime.now(TZ_ISTANBUL).isoformat()
                    write_heartbeat(t_str, faz, now_iso, now_iso, "KACIRILAN_GUN")
                    log.warning("KACIRILAN_GUN: %s faz=%s (geçmiş telafi edilmez)", t_str, faz)
                    missed.append({"tarih": t_str, "faz": faz, "durum": "KACIRILAN_GUN"})
        cur += timedelta(days=1)
    return missed


def _phase_guard(dt_str: str, faz: str, state: dict) -> Optional[dict]:
    """İdempotent atlama + heartbeat. Atlanacaksa dict döner."""
    run_key = f"{dt_str}_{faz}"
    if run_key in state.get("processed_runs", {}):
        now_iso = datetime.now(TZ_ISTANBUL).isoformat()
        write_heartbeat(dt_str, faz, now_iso, now_iso, "skipped_idempotent")
        return {"status": "skipped_idempotent", "date": dt_str}
    return None


def wait_for_acilis_vwap_window(now: Optional[datetime] = None, max_wait_sec: int = 600) -> None:
    """TSİ 10:05'e kadar bekler (en fazla max_wait_sec). Testte PYTEST_CURRENT_TEST varken no-op."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    now_dt = now or datetime.now(TZ_ISTANBUL)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=TZ_ISTANBUL)
    else:
        now_dt = now_dt.astimezone(TZ_ISTANBUL)
    if now_dt.time() >= ACILIS_VWAP_WINDOW_START:
        return
    target = now_dt.replace(
        hour=ACILIS_VWAP_WINDOW_START.hour,
        minute=ACILIS_VWAP_WINDOW_START.minute,
        second=0,
        microsecond=0,
    )
    sleep_s = min(float(max_wait_sec), (target - now_dt).total_seconds())
    if sleep_s > 0:
        log.info("VWAP penceresi için %.0f sn bekleniyor (hedef 10:05 TSİ).", sleep_s)
        import time as _time
        _time.sleep(sleep_s)


def _should_fetch_vwap(fetch_vwap: Optional[bool]) -> bool:
    """Açık True/False çağıranı bağlar. None ise pytest'te False, canlıda True."""
    if fetch_vwap is not None:
        return bool(fetch_vwap)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


def get_xu100_regime(as_of_date: pd.Timestamp) -> bool:
    """XU100 > EMA200 boğa rejimi bayrağını döndürür."""
    bench_file = BASE_DIR / "research_p1" / "data" / "benchmarks" / "XU100.IS.csv"
    if not bench_file.exists():
        bench_file = BASE_DIR / "data" / "benchmarks" / "XU100.IS.csv"
    if bench_file.exists():
        try:
            df_xu = pd.read_csv(bench_file, index_col=0)
            df_xu.index = pd.to_datetime(df_xu.index)
            sub = df_xu.loc[:as_of_date]
            if len(sub) >= 200:
                ema200 = sub["close"].ewm(span=200, adjust=False).mean().iloc[-1]
                return bool(sub["close"].iloc[-1] > ema200)
        except Exception:
            pass
    return True  # Varsayılan


def generate_bot_signals_for_day(
    symbol_data: Dict[str, pd.DataFrame],
    dt: pd.Timestamp,
    precomputed: Optional[Dict[str, Dict[str, pd.Series]]] = None,
) -> Tuple[Set[Tuple[str, str]], List[dict]]:
    """Günün kapanışında botun sinyal üretim motorunu çalıştırır."""
    all_agents = AGENTS_PRIORITY if _REPLAY_MODE else (AGENTS_PRIORITY + SHADOW_AGENTS)
    bot_signals_set = set()
    active_candidates = []

    priority_rank = {ag: i for i, ag in enumerate(AGENTS_PRIORITY)}

    for sym, df in symbol_data.items():
        if dt not in df.index:
            continue

        triggered_active = []
        triggered_shadow = []

        for ag in all_agents:
            if precomputed is not None and sym in precomputed and ag in precomputed[sym]:
                is_on = bool(precomputed[sym][ag].loc[dt]) if dt in precomputed[sym][ag].index else False
            else:
                sig_s = evaluate_custom_agent_signals(df, ag)
                is_on = bool(sig_s.loc[dt])
            if is_on:
                bot_signals_set.add((sym, ag))
                if ag in AGENTS_PRIORITY:
                    triggered_active.append(ag)
                elif ag in SHADOW_AGENTS:
                    triggered_shadow.append(ag)

        c = float(df.loc[dt, "close"])
        rel_vol = float(df.loc[dt, "rel_vol"]) if "rel_vol" in df.columns else 1.0

        # Gölge ajan sinyali kaydı (sermaye almaz!)
        if triggered_shadow and not _REPLAY_MODE:
            for sh_ag in triggered_shadow:
                _log_shadow_signal(dt, sym, sh_ag, c, rel_vol, df)

        if triggered_active:
            best_rank = min(priority_rank[ag] for ag in triggered_active)
            primary_ag = [ag for ag in triggered_active if priority_rank[ag] == best_rank][0]
            active_candidates.append({
                "symbol": sym,
                "signal_date": dt,
                "signal_close": c,
                "rel_vol": rel_vol,
                "primary_agent": primary_ag,
                "all_agents": triggered_active,
                "priority_rank": best_rank,
            })

    # Sıralama: Öncelik rank (GTD=0 > ALPHA_B_SIMPLE=1 > ZT3=2) -> RelVol azalan -> Sembol deterministik
    active_candidates.sort(key=lambda x: (x["priority_rank"], -x["rel_vol"], x["symbol"]))

    for i, cand in enumerate(active_candidates, 1):
        cand["ajan_bazli_rel_vol_sirasi"] = i

    return bot_signals_set, active_candidates


def _log_shadow_signal(dt: pd.Timestamp, sym: str, agent: str, close: float, rel_vol: float, df: pd.DataFrame) -> None:
    """Gölge sinyali shadow_signals.jsonl dosyasına idempotent olarak kaydeder."""
    SHADOW_SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    t_str = dt.strftime("%Y-%m-%d")
    record = {
        "tarih": t_str,
        "sembol": sym,
        "ajan": agent,
        "kapanis": round(close, 4),
        "rel_vol": round(rel_vol, 2),
        "rsi": round(float(df.loc[dt, "rsi"]), 2) if "rsi" in df.columns else None,
        "timestamp": datetime.now(TZ_ISTANBUL).isoformat(),
    }
    # İdempotency: Aynı tarih, sembol ve ajan varsa tekrar yazma
    existing_keys = set()
    if SHADOW_SIGNALS_FILE.exists():
        with open(SHADOW_SIGNALS_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        d = json.loads(line)
                        existing_keys.add((d.get("tarih"), d.get("sembol"), d.get("ajan")))
                    except Exception:
                        pass
    if (t_str, sym, agent) not in existing_keys:
        with open(SHADOW_SIGNALS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("[SHADOW SİNYAL] %s %s @ %.2f (RelVol: %.2f)", t_str, sym, close, rel_vol)


# ── GÜNLÜK DÖNGÜ (PHASE A, B, C, D) ──────────────────────────────────────────

def run_phase_aksam(
    as_of_date: Optional[pd.Timestamp] = None,
    symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
    precomputed: Optional[Dict[str, Dict[str, pd.Series]]] = None,
    skip_signal_parity: bool = False,
) -> dict:
    """Phase A: Gün sonu kapanış taraması, sinyal üretimi ve parity kontrolü."""
    t0 = datetime.now(TZ_ISTANBUL)
    if symbol_data is None:
        symbol_data = load_all_indicators()

    all_dates = sorted(list(set().union(*[df.index for df in symbol_data.values()])))
    dt = as_of_date or all_dates[-1]
    dt_str = dt.strftime("%Y-%m-%d")
    if not _REPLAY_MODE:
        check_missed_trading_days(dt.date() if hasattr(dt, "date") else date.fromisoformat(dt_str))

    state = load_paper_state()
    if not state.get("paper_start_date"):
        state["paper_start_date"] = dt_str
    skipped = _phase_guard(dt_str, "aksam", state)
    if skipped:
        log.info("Phase Aksam [%s] zaten çalıştırılmış, idempotent olarak atlanıyor.", dt_str)
        return skipped

    # 1. Sinyal üretimi
    bot_sigs, candidates = generate_bot_signals_for_day(symbol_data, dt, precomputed=precomputed)

    # 2. Backtest parity kontrolü (tarihsel replay'de atlanabilir)
    if skip_signal_parity:
        parity_res = {"uyum_orani_pct": None}
        bt_sigs = set()
    else:
        bt_sigs = evaluate_backtest_signals_for_day(symbol_data, dt)
        parity_res = compare_daily_signals(dt_str, bot_sigs, bt_sigs)

    # 3. Adayları state'e kaydet
    serializable_candidates = []
    regime_bull = get_xu100_regime(dt)
    for c in candidates:
        serializable_candidates.append({
            "symbol": c["symbol"],
            "signal_date": dt_str,
            "signal_close": c["signal_close"],
            "rel_vol": c["rel_vol"],
            "primary_agent": c["primary_agent"],
            "all_agents": c["all_agents"],
            "ajan_bazli_rel_vol_sirasi": c["ajan_bazli_rel_vol_sirasi"],
            "xu100_rejim_boga": regime_bull,
        })
    state["pending_candidates"] = serializable_candidates
    state.setdefault("processed_runs", {})[f"{dt_str}_aksam"] = datetime.now(TZ_ISTANBUL).isoformat()

    # 4. Günlük MTM Equity kaydı ve DD uyarıları
    _record_daily_mtm_equity(state, dt, symbol_data)

    save_paper_state(state)
    t1 = datetime.now(TZ_ISTANBUL)
    write_heartbeat(dt_str, "aksam", t0.isoformat(), t1.isoformat(), "success")
    parity_txt = "n/a" if parity_res.get("uyum_orani_pct") is None else f"{parity_res['uyum_orani_pct']:.1f}"
    log.info("Phase Aksam tamamlandı [%s]: %d aktif aday, %d backtest sinyali (Uyum: %%%s)",
             dt_str, len(candidates), len(bt_sigs), parity_txt)

    return {
        "status": "success",
        "date": dt_str,
        "candidates_count": len(candidates),
        "parity_rate": parity_res["uyum_orani_pct"],
    }


def run_phase_acilis(
    as_of_date: Optional[pd.Timestamp] = None,
    open_prices: Optional[Dict[str, float]] = None,
    minute_bars: Optional[Dict[str, pd.DataFrame]] = None,
    symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
    fetch_vwap: Optional[bool] = None,
    wait_vwap_window: bool = False,
) -> dict:
    """Phase B: Ertesi gün açılış seansı, emirlerin doldurulması ve kapasite reddi loglaması.

    simule_dolum_fiyati = açılış × (1 + SLIPPAGE_RATE)  [paper simülasyonu]
    gerceklesen_slipaj_pct / dolum_vekili_* = yalnızca ölçülen 5dk VWAP vekili
    """
    t0 = datetime.now(TZ_ISTANBUL)
    if wait_vwap_window:
        wait_for_acilis_vwap_window()
    if symbol_data is None:
        symbol_data = load_all_indicators()

    all_dates = sorted(list(set().union(*[df.index for df in symbol_data.values()])))
    dt = as_of_date or all_dates[-1]
    dt_str = dt.strftime("%Y-%m-%d")
    if not _REPLAY_MODE:
        check_missed_trading_days(dt.date() if hasattr(dt, "date") else date.fromisoformat(dt_str))

    state = load_paper_state()
    if not state.get("paper_start_date"):
        state["paper_start_date"] = dt_str
    skipped = _phase_guard(dt_str, "acilis", state)
    if skipped:
        log.info("Phase Acilis [%s] zaten çalıştırılmış, idempotent olarak atlanıyor.", dt_str)
        return skipped

    pending = state.get("pending_candidates", [])
    if not pending:
        log.info("Phase Acilis [%s]: Bekleyen sinyal yok.", dt_str)
        state.setdefault("processed_runs", {})[f"{dt_str}_acilis"] = datetime.now(TZ_ISTANBUL).isoformat()
        save_paper_state(state)
        t1 = datetime.now(TZ_ISTANBUL)
        write_heartbeat(dt_str, "acilis", t0.isoformat(), t1.isoformat(), "no_pending_signals")
        return {"status": "no_pending_signals", "date": dt_str}

    paper_log = load_paper_log()
    positions_20k = state.setdefault("positions_20k", {})
    positions_10k = state.setdefault("positions_10k", {})
    tracked_rejected = state.setdefault("tracked_rejected_signals", [])

    opened_count = 0
    rejected_count = 0

    for cand in pending:
        sym = cand["symbol"]
        sig_date = cand["signal_date"]
        sig_close = float(cand["signal_close"])
        primary_ag = cand["primary_agent"]
        all_ag = cand["all_agents"]
        rel_vol_rank = cand["ajan_bazli_rel_vol_sirasi"]
        rel_vol = cand["rel_vol"]
        regime_bull = cand["xu100_rejim_boga"]

        # Açılış fiyatını belirle
        if open_prices and sym in open_prices:
            curr_open = float(open_prices[sym])
        elif sym in symbol_data and dt in symbol_data[sym].index:
            curr_open = float(symbol_data[sym].loc[dt, "open"])
        else:
            log.warning("%s için açılış fiyatı bulunamadı, atlanıyor.", sym)
            continue

        gap = gap_ratio(curr_open, sig_close)
        sim_fill = round(curr_open * (1.0 + SLIPPAGE_RATE), 4)

        # Gap kontrolü (oran: abs(open/close - 1) > 0.0001)
        if gap_rejected(curr_open, sig_close, GAP_TOLERANCE):
            rej_row = {
                "sinyal_tarihi": sig_date,
                "giris_tarihi": dt_str,
                "sembol": sym,
                "birincil_ajan": primary_ag,
                "tum_ajanlar": ",".join(all_ag),
                "ajan_bazli_rel_vol_sirasi": rel_vol_rank,
                "rel_vol": rel_vol,
                "sinyal_kapanis_fiyati": round(sig_close, 4),
                "beklenen_acilis_fiyati": round(sig_close, 4),
                "gerceklesen_acilis_fiyati": round(curr_open, 4),
                "simule_dolum_fiyati": "",
                "gap_orani_pct": round(gap * 100.0, 4),
                "gerceklesen_slipaj_pct": "",
                "komisyon_pct": round(COMMISSION_RATE * 100.0, 2),
                "dolum_vekili_tur_basi_maliyet_pct": "",
                "onaylanan_lot": 0,
                "tutar_tl": 0.0,
                "kapasite_durumu": "GAP_REDDI",
                "xu100_rejim_boga": regime_bull,
                "cikis_tarihi": "",
                "cikis_fiyati": "",
                "cikis_nedeni": "",
                "net_getiri_pct": "",
                "net_pnl_tl": "",
                "net_pnl_10k_esdegeri_tl": "",
                "reddedilen_sonradan_getiri_pct": "",
                "notlar": f"Gap reddi (|gap|={abs(gap*100):.3f}% > %0.01)",
            }
            _upsert_log_row(paper_log, rej_row)
            tracked_rejected.append({
                "symbol": sym, "signal_date": sig_date, "entry_date": dt_str,
                "entry_price": curr_open, "peak_price": curr_open, "tp1_done": False,
                "primary_agent": primary_ag, "reason": "GAP_REDDI"
            })
            rejected_count += 1
            continue

        # Kapasite kontrolü (Max 5 pozisyon)
        if len(positions_20k) >= MAX_POSITIONS or sym in positions_20k:
            rej_reason = "ALREADY_OPEN" if sym in positions_20k else "KAPASITE_REDDI"
            rej_row = {
                "sinyal_tarihi": sig_date,
                "giris_tarihi": dt_str,
                "sembol": sym,
                "birincil_ajan": primary_ag,
                "tum_ajanlar": ",".join(all_ag),
                "ajan_bazli_rel_vol_sirasi": rel_vol_rank,
                "rel_vol": rel_vol,
                "sinyal_kapanis_fiyati": round(sig_close, 4),
                "beklenen_acilis_fiyati": round(sig_close, 4),
                "gerceklesen_acilis_fiyati": round(curr_open, 4),
                "simule_dolum_fiyati": "",
                "gap_orani_pct": round(gap * 100.0, 4),
                "gerceklesen_slipaj_pct": "",
                "komisyon_pct": round(COMMISSION_RATE * 100.0, 2),
                "dolum_vekili_tur_basi_maliyet_pct": "",
                "onaylanan_lot": 0,
                "tutar_tl": 0.0,
                "kapasite_durumu": rej_reason,
                "xu100_rejim_boga": regime_bull,
                "cikis_tarihi": "",
                "cikis_fiyati": "",
                "cikis_nedeni": "",
                "net_getiri_pct": "",
                "net_pnl_tl": "",
                "net_pnl_10k_esdegeri_tl": "",
                "reddedilen_sonradan_getiri_pct": "",
                "notlar": "5 slot dolu, kapasite reddi" if rej_reason == "KAPASITE_REDDI" else "Hisse zaten acik",
            }
            _upsert_log_row(paper_log, rej_row)
            if rej_reason == "KAPASITE_REDDI":
                tracked_rejected.append({
                    "symbol": sym, "signal_date": sig_date, "entry_date": dt_str,
                    "entry_price": curr_open, "peak_price": curr_open, "tp1_done": False,
                    "primary_agent": primary_ag, "reason": "KAPASITE_REDDI"
                })
            rejected_count += 1
            continue

        # Pozisyon açılışı (20.000 TL ve 10.000 TL)
        fill_price = round(curr_open * (1.0 + SLIPPAGE_RATE), 4)
        alloc_20k = min(POSITION_SIZE, state["cash_20k"] / max(1, MAX_POSITIONS - len(positions_20k)))
        alloc_10k = min(POSITION_SIZE_10K_EQ, state["cash_10k"] / max(1, MAX_POSITIONS - len(positions_10k)))

        lots_20k = int((alloc_20k / (1.0 + COMMISSION_RATE)) / max(fill_price, 0.01))
        lots_10k = int((alloc_10k / (1.0 + COMMISSION_RATE)) / max(fill_price, 0.01))

        if lots_20k < 1:
            continue

        cost_20k = round(lots_20k * fill_price * (1.0 + COMMISSION_RATE), 4)
        cost_10k = round(lots_10k * fill_price * (1.0 + COMMISSION_RATE), 4)

        if cost_20k > state["cash_20k"]:
            # Küçük küsurat aşımı varsa 1 lot düşür
            lots_20k = max(1, lots_20k - 1)
            cost_20k = round(lots_20k * fill_price * (1.0 + COMMISSION_RATE), 4)
            if cost_20k > state["cash_20k"]:
                continue  # Sanal nakit yetersiz

        if cost_10k > state["cash_10k"]:
            lots_10k = max(1, lots_10k - 1)
            cost_10k = round(lots_10k * fill_price * (1.0 + COMMISSION_RATE), 4)

        state["cash_20k"] -= cost_20k
        state["cash_10k"] -= cost_10k

        # Dakikalık VWAP vekili (tipik fiyat = (H+L+C)/3). Simüle dolum ayrı kolon.
        slip_proxy_pct = ""
        maliyet_tur_basi_pct = ""
        notlar_str = ""
        m_df = None
        attempted_vwap = False
        if minute_bars and sym in minute_bars:
            m_df = minute_bars[sym]
            attempted_vwap = True
        elif _should_fetch_vwap(fetch_vwap):
            from p1_data_provider import fetch_opening_minute_bars
            session_day = dt.date() if hasattr(dt, "date") else date.fromisoformat(dt_str)
            m_df = fetch_opening_minute_bars(sym, session_day, n_bars=5)
            attempted_vwap = True

        if m_df is not None:
            from p1_data_provider import opening_vwap
            vwap5 = opening_vwap(m_df, n_bars=5)
            if vwap5 is not None and curr_open > 0:
                slip_proxy_pct = round((vwap5 - curr_open) / curr_open * 100.0, 4)
                maliyet_tur_basi_pct = round(
                    2.0 * (COMMISSION_RATE * 100.0 + abs(float(slip_proxy_pct))), 4
                )
        if slip_proxy_pct == "":
            notlar_str = "VWAP verisi yok"
            if attempted_vwap:
                log.error(
                    "VWAP verisi yok: %s %s — ölçülen vekil boş; sonradan doldurulamaz (1m geçmiş ~7 gün).",
                    sym, dt_str,
                )
                warn_vwap_missing(sym, dt_str)

        pos_id = f"P1_{sym}_{dt_str.replace('-', '')}_{len(paper_log)}"
        positions_20k[sym] = {
            "pos_id": pos_id, "symbol": sym, "entry_date": dt_str, "signal_date": sig_date,
            "entry_price": fill_price, "lots": lots_20k, "initial_lots": lots_20k,
            "total_cost": cost_20k, "net_sales_income": 0.0, "peak_price": fill_price,
            "tp1_done": False, "primary_agent": primary_ag, "all_agents": all_ag,
            "rel_vol": rel_vol, "regime_bull": regime_bull, "mfe": 0.0, "mae": 0.0
        }
        positions_10k[sym] = {
            "pos_id": pos_id, "symbol": sym, "entry_date": dt_str, "signal_date": sig_date,
            "entry_price": fill_price, "lots": lots_10k, "initial_lots": lots_10k,
            "total_cost": cost_10k, "net_sales_income": 0.0, "peak_price": fill_price,
            "tp1_done": False
        }

        trade_row = {
            "sinyal_tarihi": sig_date,
            "giris_tarihi": dt_str,
            "sembol": sym,
            "birincil_ajan": primary_ag,
            "tum_ajanlar": ",".join(all_ag),
            "ajan_bazli_rel_vol_sirasi": rel_vol_rank,
            "rel_vol": rel_vol,
            "sinyal_kapanis_fiyati": round(sig_close, 4),
            "beklenen_acilis_fiyati": round(sig_close, 4),
            "gerceklesen_acilis_fiyati": round(curr_open, 4),
            "simule_dolum_fiyati": sim_fill,
            "gap_orani_pct": round(gap * 100.0, 4),
            "gerceklesen_slipaj_pct": slip_proxy_pct,
            "komisyon_pct": round(COMMISSION_RATE * 100.0, 2),
            "dolum_vekili_tur_basi_maliyet_pct": maliyet_tur_basi_pct,
            "onaylanan_lot": lots_20k,
            "tutar_tl": round(cost_20k, 2),
            "kapasite_durumu": "ISLEME_ALINDI",
            "xu100_rejim_boga": regime_bull,
            "cikis_tarihi": "",
            "cikis_fiyati": "",
            "cikis_nedeni": "",
            "net_getiri_pct": "",
            "net_pnl_tl": "",
            "net_pnl_10k_esdegeri_tl": "",
            "reddedilen_sonradan_getiri_pct": "",
            "notlar": notlar_str,
        }
        _upsert_log_row(paper_log, trade_row)
        opened_count += 1
        log.info("[POZİSYON AÇILDI] %s %d lot @ %.2f TL (Tutar: %.2f TL)", sym, lots_20k, fill_price, cost_20k)

    # İşlenen adayları temizle
    state["pending_candidates"] = []
    state.setdefault("processed_runs", {})[f"{dt_str}_acilis"] = datetime.now(TZ_ISTANBUL).isoformat()

    save_paper_log(paper_log)
    save_paper_state(state)
    t1 = datetime.now(TZ_ISTANBUL)
    write_heartbeat(dt_str, "acilis", t0.isoformat(), t1.isoformat(), "success")
    log.info("Phase Acilis tamamlandı [%s]: %d açıldı, %d reddedildi.", dt_str, opened_count, rejected_count)

    return {"status": "success", "date": dt_str, "opened": opened_count, "rejected": rejected_count}


def run_phase_takip(
    as_of_date: Optional[pd.Timestamp] = None,
    current_bars: Optional[Dict[str, dict]] = None,
    symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> dict:
    """Phase C & D: Açık pozisyonların çıkış kuralları ile yönetimi ve reddedilen sinyallerin sonradan getiri takibi."""
    t0 = datetime.now(TZ_ISTANBUL)
    if symbol_data is None:
        symbol_data = load_all_indicators()

    all_dates = sorted(list(set().union(*[df.index for df in symbol_data.values()])))
    dt = as_of_date or all_dates[-1]
    dt_str = dt.strftime("%Y-%m-%d")
    if not _REPLAY_MODE:
        check_missed_trading_days(dt.date() if hasattr(dt, "date") else date.fromisoformat(dt_str))

    state = load_paper_state()
    if not state.get("paper_start_date"):
        state["paper_start_date"] = dt_str
    paper_log = load_paper_log()

    positions_20k = state.setdefault("positions_20k", {})
    positions_10k = state.setdefault("positions_10k", {})
    closed_syms = []

    # 1. Açık pozisyonların çıkış kontrolü
    for sym, pos_20 in list(positions_20k.items()):
        pos_10 = positions_10k.get(sym, {})
        bar = _get_bar(sym, dt, current_bars, symbol_data)
        if not bar:
            continue

        o_b, h_b, l_b, c_b = bar["open"], bar["high"], bar["low"], bar["close"]
        entry_p = pos_20["entry_price"]
        holding_days = (dt - pd.to_datetime(pos_20["entry_date"])).days

        pos_20["mfe"] = max(pos_20.get("mfe", 0.0), (h_b - entry_p) / entry_p)
        pos_20["mae"] = min(pos_20.get("mae", 0.0), (l_b - entry_p) / entry_p)

        stop_l = stop_level_price(entry_p)
        tp1_l = tp1_level_price(entry_p)
        tr_l = trail_level_price(pos_20["peak_price"]) if pos_20["tp1_done"] else 0.0

        is_stop = l_b <= stop_l
        is_tp1 = (h_b >= tp1_l) and not pos_20["tp1_done"]
        is_trailing = pos_20["tp1_done"] and (l_b <= tr_l)
        is_max_gun = holding_days >= MAX_HOLDING_DAYS

        exit_triggered = False
        exit_reason = ""
        exit_price = 0.0
        sold_lots_20k = 0
        sold_lots_10k = 0
        is_partial = False

        if is_stop and is_tp1:
            ref_exit = o_b if o_b <= stop_l else stop_l
            exit_price = round(ref_exit * (1.0 - SLIPPAGE_RATE), 4)
            exit_reason = "STOP_AMBIGUOUS"
            sold_lots_20k = pos_20["lots"]
            sold_lots_10k = pos_10.get("lots", 0)
            exit_triggered = True
        elif is_stop:
            ref_exit = o_b if o_b <= stop_l else stop_l
            exit_price = round(ref_exit * (1.0 - SLIPPAGE_RATE), 4)
            exit_reason = "STOP"
            sold_lots_20k = pos_20["lots"]
            sold_lots_10k = pos_10.get("lots", 0)
            exit_triggered = True
        elif is_tp1:
            ref_exit = max(o_b, tp1_l)
            exit_price = round(ref_exit * (1.0 - SLIPPAGE_RATE), 4)
            sold_lots_20k = tp1_sold_lots(pos_20["lots"])
            sold_lots_10k = tp1_sold_lots(pos_10.get("lots", 0))
            pos_20["lots"] -= sold_lots_20k
            if "lots" in pos_10:
                pos_10["lots"] -= sold_lots_10k
            pos_20["tp1_done"] = True
            pos_10["tp1_done"] = True

            inc_20k = round(sold_lots_20k * exit_price * (1.0 - COMMISSION_RATE), 4)
            inc_10k = round(sold_lots_10k * exit_price * (1.0 - COMMISSION_RATE), 4)
            pos_20["net_sales_income"] += inc_20k
            pos_10["net_sales_income"] += inc_10k
            state["cash_20k"] += inc_20k
            state["cash_10k"] += inc_10k

            if pos_20["lots"] == 0:
                exit_triggered = True
                exit_reason = "TP1_FULL"
            else:
                is_partial = True
        elif is_trailing:
            ref_exit = o_b if o_b <= tr_l else tr_l
            exit_price = round(ref_exit * (1.0 - SLIPPAGE_RATE), 4)
            exit_reason = "TRAILING"
            sold_lots_20k = pos_20["lots"]
            sold_lots_10k = pos_10.get("lots", 0)
            exit_triggered = True
        elif is_max_gun:
            exit_price = round(c_b * (1.0 - SLIPPAGE_RATE), 4)
            exit_reason = "MAX_GUN"
            sold_lots_20k = pos_20["lots"]
            sold_lots_10k = pos_10.get("lots", 0)
            exit_triggered = True

        if exit_triggered and not is_partial:
            inc_20k = round(sold_lots_20k * exit_price * (1.0 - COMMISSION_RATE), 4)
            inc_10k = round(sold_lots_10k * exit_price * (1.0 - COMMISSION_RATE), 4)
            pos_20["net_sales_income"] += inc_20k
            pos_10["net_sales_income"] += inc_10k
            state["cash_20k"] += inc_20k
            state["cash_10k"] += inc_10k

            net_pnl_20k = round(pos_20["net_sales_income"] - pos_20["total_cost"], 2)
            net_ret_pct = round((pos_20["net_sales_income"] - pos_20["total_cost"]) / pos_20["total_cost"] * 100.0, 4)
            net_pnl_10k_eq = round(net_pnl_20k / 2.0, 2)

            # Log tablosunu güncelle
            _update_log_exit(paper_log, sym, pos_20["signal_date"], dt_str, exit_price, exit_reason, net_ret_pct, net_pnl_20k, net_pnl_10k_eq)
            closed_syms.append(sym)
            log.info("[POZİSYON KAPANDI] %s Çıkış: %.2f (%s) | Net: %+.2f%% | PnL: %+.2f TL (10k Eq: %+.2f TL)",
                     sym, exit_price, exit_reason, net_ret_pct, net_pnl_20k, net_pnl_10k_eq)

        pos_20["peak_price"] = max(pos_20.get("peak_price", entry_p), h_b)

    for sym in closed_syms:
        positions_20k.pop(sym, None)
        positions_10k.pop(sym, None)

    # 2. Reddedilen Sinyallerin "Sonradan Getirisi" Takibi (Phase D)
    _evaluate_tracked_rejected_signals(state, dt, paper_log, current_bars, symbol_data)

    # 3. İzleme Uyarıları (Dolum Maliyeti MA)
    if not _REPLAY_MODE:
        completed_trades = [r for r in paper_log if r.get("cikis_nedeni") != ""]
        check_fill_cost_alerts(completed_trades, window=20)

    save_paper_log(paper_log)
    save_paper_state(state)
    t1 = datetime.now(TZ_ISTANBUL)
    write_heartbeat(dt_str, "takip", t0.isoformat(), t1.isoformat(), "success")

    return {"status": "success", "date": dt_str, "closed": len(closed_syms)}


def _evaluate_tracked_rejected_signals(state: dict, dt: pd.Timestamp, paper_log: List[dict], current_bars, symbol_data) -> None:
    """Kapasite veya gap nedeniyle reddedilen sinyallerin sanal getirisini takip eder."""
    tracked = state.get("tracked_rejected_signals", [])
    finished_indices = []

    for idx, cand in enumerate(tracked):
        sym = cand["symbol"]
        bar = _get_bar(sym, dt, current_bars, symbol_data)
        if not bar:
            continue

        o_b, h_b, l_b, c_b = bar["open"], bar["high"], bar["low"], bar["close"]
        entry_p = cand["entry_price"]
        holding_days = (dt - pd.to_datetime(cand["entry_date"])).days

        stop_l = stop_level_price(entry_p)
        tp1_l = tp1_level_price(entry_p)
        tr_l = trail_level_price(cand["peak_price"]) if cand["tp1_done"] else 0.0

        is_stop = l_b <= stop_l
        is_tp1 = (h_b >= tp1_l) and not cand["tp1_done"]
        is_tr = cand["tp1_done"] and (l_b <= tr_l)
        is_mg = holding_days >= MAX_HOLDING_DAYS

        if is_tp1:
            cand["tp1_done"] = True

        finished = False
        hypo_exit_p = 0.0
        if is_stop:
            ref_p = o_b if o_b <= stop_l else stop_l
            hypo_exit_p = round(ref_p * (1.0 - SLIPPAGE_RATE), 4)
            finished = True
        elif is_tr:
            ref_p = o_b if o_b <= tr_l else tr_l
            hypo_exit_p = round(ref_p * (1.0 - SLIPPAGE_RATE), 4)
            finished = True
        elif is_mg:
            hypo_exit_p = round(c_b * (1.0 - SLIPPAGE_RATE), 4)
            finished = True

        if finished:
            hypo_ret_pct = round((hypo_exit_p - entry_p) / entry_p * 100.0 - ROUNDTRIP_COST_PCT, 4)
            # Logdaki ilgili reddedilen satırı güncelle
            for row in paper_log:
                if row.get("sembol") == sym and row.get("sinyal_tarihi") == cand["signal_date"]:
                    row["reddedilen_sonradan_getiri_pct"] = str(hypo_ret_pct)
                    break
            finished_indices.append(idx)
            log.info("[REDDEDİLEN SONRADAN GETİRİ] %s (Sinyal: %s) -> Sanal Çıkış Getirisi: %+.2f%%",
                     sym, cand["signal_date"], hypo_ret_pct)

        cand["peak_price"] = max(cand["peak_price"], h_b)

    for idx in sorted(finished_indices, reverse=True):
        tracked.pop(idx)


def _record_daily_mtm_equity(state: dict, dt: pd.Timestamp, symbol_data: Dict[str, pd.DataFrame]) -> None:
    """Günlük MTM Equity değerini hem 20k hem 10k için kaydeder ve DD kontrollerini yapar."""
    dt_str = dt.strftime("%Y-%m-%d")
    pos_20 = state.get("positions_20k", {})
    pos_10 = state.get("positions_10k", {})

    open_val_20 = 0.0
    open_val_10 = 0.0

    for sym, p20 in pos_20.items():
        if sym in symbol_data and dt in symbol_data[sym].index:
            c = float(symbol_data[sym].loc[dt, "close"])
        else:
            c = p20["entry_price"]
        open_val_20 += p20["lots"] * c

    for sym, p10 in pos_10.items():
        if sym in symbol_data and dt in symbol_data[sym].index:
            c = float(symbol_data[sym].loc[dt, "close"])
        else:
            c = p10.get("entry_price", 0.0)
        open_val_10 += p10.get("lots", 0) * c

    eq_20 = round(state["cash_20k"] + open_val_20, 2)
    eq_10 = round(state["cash_10k"] + open_val_10, 2)

    state["peak_equity_20k"] = max(state.get("peak_equity_20k", INITIAL_CAPITAL), eq_20)
    state["peak_equity_10k"] = max(state.get("peak_equity_10k", INITIAL_CAPITAL), eq_10)

    # Drawdown uyarıları
    if not _REPLAY_MODE:
        check_drawdown_alerts(eq_20, state["peak_equity_20k"], size_label="20k")
        check_drawdown_alerts(eq_10, state["peak_equity_10k"], size_label="10k")

    daily_eq_list = state.setdefault("daily_equity", [])
    if _REPLAY_MODE:
        return
    # İdempotent: Aynı tarih varsa güncelle, yoksa ekle
    idx = next((i for i, r in enumerate(daily_eq_list) if r.get("tarih") == dt_str), None)
    entry = {
        "tarih": dt_str,
        "equity_20k": eq_20,
        "cash_20k": round(state["cash_20k"], 2),
        "open_val_20k": round(open_val_20, 2),
        "equity_10k": eq_10,
        "cash_10k": round(state["cash_10k"], 2),
        "open_val_10k": round(open_val_10, 2),
        "open_positions_count": len(pos_20),
    }
    if idx is not None:
        daily_eq_list[idx] = entry
    else:
        daily_eq_list.append(entry)


def _get_bar(sym: str, dt: pd.Timestamp, current_bars, symbol_data) -> Optional[dict]:
    if current_bars and sym in current_bars:
        return current_bars[sym]
    if symbol_data and sym in symbol_data and dt in symbol_data[sym].index:
        r = symbol_data[sym].loc[dt]
        return {"open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]), "close": float(r["close"])}
    return None


def _upsert_log_row(paper_log: List[dict], row_data: dict) -> None:
    """(sinyal_tarihi, sembol, birincil_ajan) anahtarıyla idempotent ekleme/güncelleme."""
    key = (row_data.get("sinyal_tarihi"), row_data.get("sembol"), row_data.get("birincil_ajan"))
    for idx, existing in enumerate(paper_log):
        ex_key = (existing.get("sinyal_tarihi"), existing.get("sembol"), existing.get("birincil_ajan"))
        if ex_key == key:
            paper_log[idx].update(row_data)
            return
    paper_log.append(row_data)


def _update_log_exit(paper_log: List[dict], sym: str, sig_date: str, exit_dt: str, exit_p: float, exit_r: str, net_ret: float, pnl_20k: float, pnl_10k: float) -> None:
    for row in paper_log:
        if row.get("sembol") == sym and row.get("sinyal_tarihi") == sig_date and row.get("kapasite_durumu") == "ISLEME_ALINDI":
            row["cikis_tarihi"] = exit_dt
            row["cikis_fiyati"] = str(round(exit_p, 4))
            row["cikis_nedeni"] = exit_r
            row["net_getiri_pct"] = str(round(net_ret, 4))
            row["net_pnl_tl"] = str(round(pnl_20k, 2))
            row["net_pnl_10k_esdegeri_tl"] = str(round(pnl_10k, 2))
            return


def load_all_indicators() -> Dict[str, pd.DataFrame]:
    """BIST100 önbellek verilerinden tüm indikatörleri yükler."""
    symbol_data = {}
    for s in BIST100_SYMBOLS:
        df, meta = load_cached_ohlcv(s)
        if df is not None and len(df) >= 50:
            symbol_data[s] = compute_all_indicators(df)
    return symbol_data


def run_historical_replay(
    symbol_data: Dict[str, pd.DataFrame],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    precomputed: Optional[Dict[str, Dict[str, pd.Series]]] = None,
) -> dict:
    """Paper fazlarını backtest sırasıyla (takip → acilis → aksam) tarihsel koşturur.

    Canlı sıra acilis sonra takip olduğu için aynı gün çıkış canlıda mümkün,
    bu replay'de yeni açılan pozisyon aynı bar'da çıkılmaz (backtest ile hizalı).
    VWAP çekilmez; simule_dolum_fiyati = open*(1+slippage).
    """
    all_dates = sorted(list(set().union(*[df.index for df in symbol_data.values()])))
    if start_date:
        start_ts = pd.to_datetime(start_date)
        all_dates = [d for d in all_dates if d >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date)
        all_dates = [d for d in all_dates if d <= end_ts]

    n_aksam = n_acilis = n_takip = 0
    global _REPLAY_MODE
    prev = _REPLAY_MODE
    _REPLAY_MODE = True
    prev_level = log.level
    log.setLevel(logging.WARNING)
    try:
        for dt in all_dates:
            run_phase_takip(as_of_date=dt, symbol_data=symbol_data)
            n_takip += 1
            run_phase_acilis(as_of_date=dt, symbol_data=symbol_data, fetch_vwap=False)
            n_acilis += 1
            run_phase_aksam(
                as_of_date=dt,
                symbol_data=symbol_data,
                precomputed=precomputed,
                skip_signal_parity=True,
            )
            n_aksam += 1
    finally:
        _REPLAY_MODE = prev
        log.setLevel(prev_level)

    log_rows = load_paper_log()
    taken = [r for r in log_rows if r.get("kapasite_durumu") == "ISLEME_ALINDI" and r.get("cikis_nedeni")]
    return {
        "n_days": len(all_dates),
        "n_aksam": n_aksam,
        "n_acilis": n_acilis,
        "n_takip": n_takip,
        "n_completed_trades": len(taken),
        "n_log_rows": len(log_rows),
        "start": str(all_dates[0].date()) if all_dates else "",
        "end": str(all_dates[-1].date()) if all_dates else "",
    }


# ── CLI KOMUTLARI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P1 Paper Trading Botu")
    parser.add_argument(
        "mode",
        choices=["aksam", "acilis", "takip", "replay", "status", "heartbeat-check"],
        help="Çalışma modu",
    )
    parser.add_argument("--date", help="Simüle edilecek tarih (YYYY-MM-DD)")
    parser.add_argument(
        "--wait-vwap-window",
        action="store_true",
        help="acilis: TSİ 10:05 VWAP penceresine kadar bekle (mott_daily alim)",
    )
    args = parser.parse_args()

    as_of = pd.to_datetime(args.date) if args.date else None

    if args.mode == "aksam":
        run_phase_aksam(as_of_date=as_of)
    elif args.mode == "acilis":
        run_phase_acilis(
            as_of_date=as_of,
            fetch_vwap=True,
            wait_vwap_window=bool(args.wait_vwap_window),
        )
    elif args.mode == "takip":
        run_phase_takip(as_of_date=as_of)
    elif args.mode == "heartbeat-check":
        d = as_of.date() if as_of is not None else datetime.now(TZ_ISTANBUL).date()
        missed = check_missed_trading_days(d)
        print(json.dumps({"missed": missed}, ensure_ascii=False, indent=2))
    elif args.mode == "replay":
        from p1_signal_parity import run_60_session_replay
        res = run_60_session_replay()
        print(json.dumps(res, indent=2))
    elif args.mode == "status":
        st = load_paper_state()
        print(f"20k Nakit: {st.get('cash_20k'):,.2f} TL | Açık Pozisyon: {len(st.get('positions_20k', {}))}")
        print(f"10k Nakit: {st.get('cash_10k'):,.2f} TL | Açık Pozisyon: {len(st.get('positions_10k', {}))}")
        print(f"Bekleyen Adaylar: {len(st.get('pending_candidates', []))}")
