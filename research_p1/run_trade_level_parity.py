# -*- coding: utf-8 -*-
"""
research_p1/run_trade_level_parity.py
=====================================
Paper tarihsel replay vs simulate_priority_portfolio işlem-düzeyi karşılaştırma.

Maliyet modeli (her iki taraf, aynı sabitler):
  alış fill  = open * (1 + 0.0030)
  satış fill = ref  * (1 - 0.0030)
  komisyon   = lots * fill * 0.0030
  tur başı   = %1.20

Lot formülü (BİLİNEN FARK, düzeltilmedi):
  backtest: lots = int(alloc / fill_p); tot_cost = lots*fill_p*(1+commission)
  paper:    lots = int(alloc / ((1+commission)*fill_p))

Çıktı: results/trade_level_parity.csv
Eşleşmezse satır satır kök neden yazılır; motor değiştirilmez.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from p1_paper_config import (
    AGENTS_PRIORITY,
    COMMISSION_RATE,
    GAP_TOLERANCE,
    INITIAL_CAPITAL,
    MAX_POSITIONS,
    POSITION_SIZE,
    RESULTS_DIR,
    SLIPPAGE_RATE,
    TRADE_LEVEL_PARITY_FILE,
)
from research_p1.data_downloader import BIST100_SYMBOLS, load_cached_ohlcv
from research_p1.p1_engine import compute_all_indicators
from research_p1.p1_exit_rules import gap_ratio
from research_p1.run_p1_revision_backtest import (
    evaluate_custom_agent_signals,
    simulate_priority_portfolio,
)

log = logging.getLogger("trade_level_parity")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REF_TRADES = REPO_ROOT / "research_p1" / "results" / "p1_rev_ALPHA_B_SIMPLE_20k_trades.csv"
REF_SKIPPED = REPO_ROOT / "research_p1" / "results" / "p1_rev_ALPHA_B_SIMPLE_20k_skipped.csv"
SUMMARY_FILE = RESULTS_DIR / "trade_level_parity_summary.json"
FULL_START = "2021-10-06"
FULL_END = "2026-10-05"

PARITY_COLS = [
    "donem",
    "sembol",
    "birincil_ajan",
    "bt_giris",
    "paper_giris",
    "bt_cikis",
    "paper_cikis",
    "bt_giris_fiyat",
    "paper_giris_fiyat",
    "bt_cikis_fiyat",
    "paper_cikis_fiyat",
    "bt_cikis_nedeni",
    "paper_cikis_nedeni",
    "bt_net_getiri_pct",
    "paper_net_getiri_pct",
    "bt_net_pnl_tl",
    "paper_net_pnl_tl",
    "eslesme",
    "kok_neden",
]


def _load_ohlcv_cache() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in BIST100_SYMBOLS:
        df, _meta = load_cached_ohlcv(s)
        if df is not None and len(df) >= 50:
            out[s] = compute_all_indicators(df)
    return out


def _synthetic_universe(n_sessions: int = 80, n_symbols: int = 8) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=n_sessions + 40)
    data = {}
    for i in range(n_symbols):
        sym = f"PAR{i+1}"
        base = 50.0 + i * 5.0
        rets = rng.normal(0.001, 0.018, len(dates))
        close = base * np.exp(np.cumsum(rets))
        open_ = np.r_[close[0], close[:-1]]
        # Çoğu gün gap yok (open ≈ prev close); birkaç günde bilinçli gap
        df = pd.DataFrame({
            "open": open_,
            "high": np.maximum(open_, close) * 1.015,
            "low": np.minimum(open_, close) * 0.985,
            "close": close,
            "volume": rng.integers(80_000, 250_000, len(dates)).astype(float),
        }, index=dates)
        data[sym] = compute_all_indicators(df)
    return data


def _precompute(symbol_data: Dict[str, pd.DataFrame], agents: List[str]) -> Dict[str, Dict[str, pd.Series]]:
    pre = {}
    for sym, df in symbol_data.items():
        pre[sym] = {ag: evaluate_custom_agent_signals(df, ag) for ag in agents}
    return pre


def _redirect_paper_io(target_dir: Path) -> None:
    """Replay çıktısını canlı paper dosyalarının üzerine yazmamak için yolları saptır."""
    import p1_paper as p1_pap
    import p1_monitoring as p1_mon
    import p1_paper_config as p1_cfg

    target_dir.mkdir(parents=True, exist_ok=True)
    p1_cfg.PAPER_LOG_FILE = target_dir / "paper_trading_log.csv"
    p1_cfg.PAPER_STATE_FILE = target_dir / "p1_paper_state.json"
    p1_cfg.HEARTBEAT_FILE = target_dir / "p1_run_heartbeat.csv"
    p1_cfg.ALERTS_LOG_FILE = target_dir / "p1_monitoring_alerts.log"
    p1_cfg.SHADOW_SIGNALS_FILE = target_dir / "shadow_signals.jsonl"
    p1_pap.PAPER_LOG_FILE = p1_cfg.PAPER_LOG_FILE
    p1_pap.PAPER_STATE_FILE = p1_cfg.PAPER_STATE_FILE
    p1_pap.HEARTBEAT_FILE = p1_cfg.HEARTBEAT_FILE
    p1_pap.SHADOW_SIGNALS_FILE = p1_cfg.SHADOW_SIGNALS_FILE
    p1_mon.ALERTS_LOG_FILE = p1_cfg.ALERTS_LOG_FILE
    for p in (p1_cfg.PAPER_LOG_FILE, p1_cfg.PAPER_STATE_FILE, p1_cfg.HEARTBEAT_FILE):
        if p.exists():
            p.unlink()


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _paper_trades_from_log(rows: List[dict]) -> pd.DataFrame:
    taken = [
        r for r in rows
        if r.get("kapasite_durumu") == "ISLEME_ALINDI" and str(r.get("cikis_nedeni", "")).strip()
    ]
    recs = []
    for r in taken:
        sim_fill = _to_float(r.get("simule_dolum_fiyati"))
        raw_open = _to_float(r.get("gerceklesen_acilis_fiyati"))
        recs.append({
            "symbol": r.get("sembol"),
            "primary_agent": r.get("birincil_ajan"),
            "entry_date": str(r.get("giris_tarihi")),
            "exit_date": str(r.get("cikis_tarihi")),
            "entry_price": sim_fill if sim_fill is not None else raw_open,
            "simule_dolum_fiyati": sim_fill,
            "exit_price": _to_float(r.get("cikis_fiyati")),
            "exit_reason": r.get("cikis_nedeni"),
            "net_return_pct": _to_float(r.get("net_getiri_pct")),
            "net_profit_tl": _to_float(r.get("net_pnl_tl")),
            "gap_orani_pct": _to_float(r.get("gap_orani_pct")),
            "onaylanan_lot": _to_float(r.get("onaylanan_lot")),
        })
    return pd.DataFrame(recs)


def _trade_key(row: dict) -> Tuple[str, str, str]:
    return (str(row.get("symbol")), str(row.get("primary_agent")), str(row.get("entry_date")))


def _classify(bt: Optional[dict], pap: Optional[dict]) -> str:
    if bt is None:
        return "EKSIK_BACKTEST (paper'da var, simulate_priority_portfolio'da yok)"
    if pap is None:
        return "EKSIK_PAPER (backtest'te var, paper replay'de yok)"
    reasons = []
    if str(bt.get("exit_date")) != str(pap.get("exit_date")):
        reasons.append("cikis_tarihi")
    if str(bt.get("exit_reason")) != str(pap.get("exit_reason")):
        reasons.append("cikis_nedeni")
    bt_ep = float(bt.get("entry_price") or 0)
    pap_ep = float(pap.get("entry_price") or 0)
    if abs(bt_ep - pap_ep) > 1e-4:
        reasons.append("giris_fiyati")
    bt_xp = float(bt.get("exit_price") or 0)
    pap_xp = float(pap.get("exit_price") or 0)
    if abs(bt_xp - pap_xp) > 1e-4:
        reasons.append("cikis_fiyati")
    bt_pnl = float(bt.get("net_profit_tl") or 0)
    pap_pnl = float(pap.get("net_profit_tl") or 0)
    if abs(bt_pnl - pap_pnl) > 0.05:
        reasons.append("net_pnl")
    bt_ret = float(bt.get("net_return_pct") or 0)
    pap_ret = float(pap.get("net_return_pct") or 0)
    if abs(bt_ret - pap_ret) > 0.01:
        reasons.append("net_getiri")
    lot_note = (
        "LOT_FORMULU (paper: int(alloc/((1+komisyon)*fill)); "
        "backtest: int(alloc/fill) sonra tot_cost=lots*fill*(1+komisyon))"
    )
    if not reasons:
        return "ESLESTI"
    if "net_pnl" in reasons or "giris_fiyati" in reasons:
        reasons.append(lot_note)
    return "; ".join(reasons)


def compare_trade_frames(bt_df: pd.DataFrame, pap_df: pd.DataFrame, donem: str) -> pd.DataFrame:
    bt_map = {_trade_key(r): r for r in bt_df.to_dict("records")} if len(bt_df) else {}
    pap_map = {_trade_key(r): r for r in pap_df.to_dict("records")} if len(pap_df) else {}
    keys = sorted(set(bt_map) | set(pap_map))
    rows = []
    for k in keys:
        bt = bt_map.get(k)
        pap = pap_map.get(k)
        kok = _classify(bt, pap)
        rows.append({
            "donem": donem,
            "sembol": k[0],
            "birincil_ajan": k[1],
            "bt_giris": k[2] if bt else "",
            "paper_giris": k[2] if pap else "",
            "bt_cikis": (bt or {}).get("exit_date", ""),
            "paper_cikis": (pap or {}).get("exit_date", ""),
            "bt_giris_fiyat": (bt or {}).get("entry_price", ""),
            "paper_giris_fiyat": (pap or {}).get("entry_price", ""),
            "bt_cikis_fiyat": (bt or {}).get("exit_price", ""),
            "paper_cikis_fiyat": (pap or {}).get("exit_price", ""),
            "bt_cikis_nedeni": (bt or {}).get("exit_reason", ""),
            "paper_cikis_nedeni": (pap or {}).get("exit_reason", ""),
            "bt_net_getiri_pct": (bt or {}).get("net_return_pct", ""),
            "paper_net_getiri_pct": (pap or {}).get("net_return_pct", ""),
            "bt_net_pnl_tl": (bt or {}).get("net_profit_tl", ""),
            "paper_net_pnl_tl": (pap or {}).get("net_profit_tl", ""),
            "eslesme": kok == "ESLESTI",
            "kok_neden": kok,
        })
    return pd.DataFrame(rows, columns=PARITY_COLS)


def _agent_breakdown(df: pd.DataFrame, agent_col: str, pnl_col: str) -> dict:
    if df is None or len(df) == 0:
        return {}
    out = {}
    for ag, g in df.groupby(agent_col):
        out[str(ag)] = {
            "n": int(len(g)),
            "pnl": round(float(pd.to_numeric(g[pnl_col], errors="coerce").fillna(0).sum()), 2),
        }
    return out


def gap_distribution(paper_log_rows: List[dict]) -> dict:
    gaps = []
    n_gap = n_cap = n_signal = 0
    for r in paper_log_rows:
        status = r.get("kapasite_durumu")
        if status in ("ISLEME_ALINDI", "GAP_REDDI", "KAPASITE_REDDI"):
            n_signal += 1
        if status == "GAP_REDDI":
            n_gap += 1
        if status == "KAPASITE_REDDI":
            n_cap += 1
        g = _to_float(r.get("gap_orani_pct"))
        if g is not None:
            gaps.append(g)
    if not gaps:
        return {
            "n_gap_orani": 0,
            "min": None, "median": None, "p90": None, "max": None,
            "n_gap_reddi": n_gap,
            "n_kapasite_reddi": n_cap,
            "n_sinyal": n_signal,
            "gap_reddi_orani": None,
        }
    arr = np.array(gaps, dtype=float)
    return {
        "n_gap_orani": int(len(arr)),
        "min": round(float(np.min(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "max": round(float(np.max(arr)), 4),
        "n_gap_reddi": n_gap,
        "n_kapasite_reddi": n_cap,
        "n_sinyal": n_signal,
        "gap_reddi_orani": round(n_gap / n_signal, 4) if n_signal else None,
        "gap_tolerance_oran": GAP_TOLERANCE,
        "gap_tolerance_pct_puan": GAP_TOLERANCE * 100.0,
        "gap_formulu": "gap = open[t+1]/close[t] - 1.0  (oran; 0.0001 = 0.01%)",
    }


def _run_pair(symbol_data, donem: str, start: str, end: str) -> Tuple[pd.DataFrame, dict, List[dict]]:
    import p1_paper as p1_pap

    _redirect_paper_io(RESULTS_DIR / f"_parity_replay_{donem}")

    agents = list(AGENTS_PRIORITY)
    pre = _precompute(symbol_data, agents)
    bt_trades, _eq, _skip = simulate_priority_portfolio(
        symbol_data,
        agent_priority=agents,
        precomputed=pre,
        start_date=start,
        end_date=end,
        gap_tolerance=GAP_TOLERANCE,
        commission_rate=COMMISSION_RATE,
        slippage_rate=SLIPPAGE_RATE,
        initial_capital=INITIAL_CAPITAL,
        max_positions=MAX_POSITIONS,
        position_size=POSITION_SIZE,
    )
    p1_pap.run_historical_replay(
        symbol_data,
        start_date=start,
        end_date=end,
        precomputed=pre,
    )
    log_rows = p1_pap.load_paper_log()
    pap_df = _paper_trades_from_log(log_rows)
    cmp_df = compare_trade_frames(bt_trades, pap_df, donem)
    n_match = int(cmp_df["eslesme"].sum()) if len(cmp_df) else 0
    summary = {
        "donem": donem,
        "start": start,
        "end": end,
        "bt_n": int(len(bt_trades)),
        "paper_n": int(len(pap_df)),
        "n_karsilastirma": int(len(cmp_df)),
        "n_eslesen": n_match,
        "n_uyusmayan": int(len(cmp_df) - n_match),
        "bt_pnl": round(float(bt_trades["net_profit_tl"].sum()), 2) if len(bt_trades) else 0.0,
        "paper_pnl": round(float(pd.to_numeric(pap_df["net_profit_tl"], errors="coerce").fillna(0).sum()), 2) if len(pap_df) else 0.0,
        "bt_by_agent": _agent_breakdown(bt_trades, "primary_agent", "net_profit_tl"),
        "paper_by_agent": _agent_breakdown(pap_df, "primary_agent", "net_profit_tl"),
        "gap": gap_distribution(log_rows),
        "maliyet_konvansiyonu": {
            "alis": "open * (1 + 0.0030)",
            "satis": "ref * (1 - 0.0030)",
            "komisyon": "0.0030",
            "tur_basi_pct": 1.20,
            "lot_farki": "paper komisyonu lot hesabına dahil eder; backtest fill üzerinden lot keser",
        },
    }
    return cmp_df, summary, log_rows


def _flush_parity_outputs(all_cmp, summaries, data_mode: str) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ref_meta = {}
    if REF_TRADES.exists():
        ref = pd.read_csv(REF_TRADES)
        ref_meta = {
            "dosya": str(REF_TRADES),
            "n": int(len(ref)),
            "pnl": round(float(ref["net_profit_tl"].sum()), 2),
            "by_agent": _agent_breakdown(ref, "primary_agent", "net_profit_tl"),
            "referans_beklenen_n": 543,
            "referans_beklenen_pnl": 82307.0,
        }
        if data_mode == "ohlcv_cache":
            ref_meta["not"] = (
                "Karşılaştırma OHLCV cache üzerinde simulate_priority_portfolio + paper replay."
            )
        else:
            ref_meta["not"] = (
                "5yıl satır satır paper replay yapılamadı (research_p1/data OHLCV yok). "
                "CSV referans sayıları: n=543, pnl≈82307 TL."
            )
    if REF_SKIPPED.exists():
        sk = pd.read_csv(REF_SKIPPED)
        ref_meta["capacity_skipped_n"] = int(len(sk))

    out_df = pd.concat(all_cmp, ignore_index=True) if all_cmp else pd.DataFrame(columns=PARITY_COLS)
    out_df.to_csv(TRADE_LEVEL_PARITY_FILE, index=False, encoding="utf-8")
    payload = {
        "data_mode": data_mode,
        "cikti": str(TRADE_LEVEL_PARITY_FILE),
        "donemler": summaries,
        "referans_csv": ref_meta,
        "gap_anlami": {
            "birim": "oran",
            "formula": "open[t+1]/close[t] - 1.0",
            "tolerance": 0.0001,
            "tolerance_pct_puan": 0.01,
            "paper_backtest_ayni_kaynak": "research_p1.p1_exit_rules.GAP_TOLERANCE",
        },
    }
    SUMMARY_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("Yazildi: %s (%d satir)", TRADE_LEVEL_PARITY_FILE, len(out_df))
    return payload


def run_trade_level_parity() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_ohlcv_cache()
    all_cmp = []
    summaries = []

    if cache:
        dates = sorted(list(set().union(*[df.index for df in cache.values()])))
        last60 = dates[-60:]
        s60, e60 = str(last60[0].date()), str(last60[-1].date())
        cmp60, sum60, _ = _run_pair(cache, "60_seans", s60, e60)
        all_cmp.append(cmp60)
        summaries.append(sum60)
        payload = _flush_parity_outputs(all_cmp, summaries, "ohlcv_cache")

        cmp5y, sum5y, _ = _run_pair(cache, "5yil", FULL_START, FULL_END)
        all_cmp.append(cmp5y)
        summaries.append(sum5y)
        data_mode = "ohlcv_cache"
    else:
        log.warning("OHLCV önbellek yok; sentetik evren ile motor karşılaştırması yapılıyor.")
        syn = _synthetic_universe()
        dates = sorted(list(set().union(*[df.index for df in syn.values()])))
        last60 = dates[-60:]
        cmp60, sum60, _ = _run_pair(syn, "60_seans_sentetik", str(last60[0].date()), str(last60[-1].date()))
        all_cmp.append(cmp60)
        summaries.append(sum60)
        data_mode = "sentetik_ohlcv_cache_yok"

    return _flush_parity_outputs(all_cmp, summaries, data_mode)


if __name__ == "__main__":
    print(json.dumps(run_trade_level_parity(), indent=2, ensure_ascii=False, default=str))
