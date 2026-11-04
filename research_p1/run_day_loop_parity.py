# -*- coding: utf-8 -*-
"""
Gün-gün paper faz döngüsü (takip → acilis → aksam) vs simulate_priority_portfolio.

Kesilen canlı döngüden farkı:
  - I/O results/_day_loop_full/ altına alınır (canlı results/ üzerine yazmaz)
  - p1_paper._REPLAY_MODE = True (shadow jsonl / KACIRILAN_GUN / daily_equity şişmesi kapalı)
  - Yıllık parça + progress.json; bir parça bitmeden sonrakine geçilmez
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
)
from research_p1.data_downloader import BIST100_SYMBOLS, load_cached_ohlcv
from research_p1.p1_engine import compute_all_indicators
from research_p1.run_p1_revision_backtest import (
    evaluate_custom_agent_signals,
    simulate_priority_portfolio,
)
from research_p1.run_trade_level_parity import (
    _agent_breakdown,
    _paper_trades_from_log,
    _redirect_paper_io,
    compare_trade_frames,
    gap_distribution,
)

log = logging.getLogger("day_loop_parity")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FULL_START = "2021-10-06"
FULL_END = "2026-10-05"
OUT_DIR = RESULTS_DIR / "_day_loop_full"
PROGRESS_FILE = OUT_DIR / "progress.json"
CMP_FILE = RESULTS_DIR / "day_loop_parity.csv"
SUMMARY_FILE = RESULTS_DIR / "day_loop_parity_summary.json"

YEAR_CHUNKS: List[Tuple[str, str, str]] = [
    ("2021", "2021-10-06", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-10-05"),
]


def _load_cache() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in BIST100_SYMBOLS:
        df, _meta = load_cached_ohlcv(s)
        if df is not None and len(df) >= 50:
            out[s] = compute_all_indicators(df)
    return out


def _reset_paper_state() -> None:
    import p1_paper as p1_pap

    p1_pap.save_paper_state({
        "_gen": 0,
        "_updated_at": "2021-10-06T00:00:00",
        "cash_20k": float(INITIAL_CAPITAL),
        "cash_10k": float(INITIAL_CAPITAL),
        "peak_equity_20k": float(INITIAL_CAPITAL),
        "peak_equity_10k": float(INITIAL_CAPITAL),
        "positions_20k": {},
        "positions_10k": {},
        "pending_candidates": [],
        "tracked_rejected_signals": [],
        "daily_equity": [],
        "processed_runs": {},
        "paper_start_date": FULL_START,
    })
    p1_pap.save_paper_log([])


def _write_progress(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run_day_loop(symbol_data: Dict[str, pd.DataFrame], start: str, end: str) -> dict:
    """p1_paper.run_phase_takip / acilis / aksam gün gün. Yıllık parça + progress."""
    import p1_paper as p1_pap

    _redirect_paper_io(OUT_DIR)
    _reset_paper_state()

    agents = list(AGENTS_PRIORITY)
    pre = {sym: {ag: evaluate_custom_agent_signals(df, ag) for ag in agents} for sym, df in symbol_data.items()}
    all_dates = sorted(list(set().union(*[df.index for df in symbol_data.values()])))
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    sim_dates = [d for d in all_dates if start_ts <= d <= end_ts]

    prev_replay = p1_pap._REPLAY_MODE
    p1_pap._REPLAY_MODE = True
    prev_level = p1_pap.log.level
    p1_pap.log.setLevel(logging.WARNING)

    chunks_done: List[dict] = []
    n_days = 0
    failed = None
    try:
        for label, c_start, c_end in YEAR_CHUNKS:
            c0 = pd.to_datetime(c_start)
            c1 = pd.to_datetime(c_end)
            chunk_dates = [d for d in sim_dates if c0 <= d <= c1]
            if not chunk_dates:
                chunks_done.append({"chunk": label, "n_days": 0, "status": "bos"})
                continue
            log.info("CHUNK %s basladi %s -> %s (%d gun)", label, chunk_dates[0].date(), chunk_dates[-1].date(), len(chunk_dates))
            try:
                for dt in chunk_dates:
                    p1_pap.run_phase_takip(as_of_date=dt, symbol_data=symbol_data)
                    p1_pap.run_phase_acilis(as_of_date=dt, symbol_data=symbol_data, fetch_vwap=False)
                    p1_pap.run_phase_aksam(
                        as_of_date=dt,
                        symbol_data=symbol_data,
                        precomputed=pre,
                        skip_signal_parity=True,
                    )
                    n_days += 1
            except Exception as exc:
                failed = {
                    "chunk": label,
                    "last_date": str(dt.date()) if chunk_dates else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                log.exception("CHUNK %s KIRILDI %s: %s", label, failed["last_date"], exc)
                chunks_done.append({
                    "chunk": label,
                    "n_days": n_days,
                    "start": str(chunk_dates[0].date()),
                    "end": str(chunk_dates[-1].date()),
                    "status": "hata",
                    "error": failed,
                })
                _write_progress({"n_days": n_days, "chunks": chunks_done, "failed": failed})
                break
            chunks_done.append({
                "chunk": label,
                "n_days": len(chunk_dates),
                "start": str(chunk_dates[0].date()),
                "end": str(chunk_dates[-1].date()),
                "status": "ok",
            })
            _write_progress({"n_days": n_days, "chunks": chunks_done, "failed": None})
            log.info("CHUNK %s bitti n_days_toplam=%d", label, n_days)
    finally:
        p1_pap._REPLAY_MODE = prev_replay
        p1_pap.log.setLevel(prev_level)

    log_rows = p1_pap.load_paper_log()
    pap_df = _paper_trades_from_log(log_rows)
    return {
        "n_days": n_days,
        "n_dates": len(sim_dates),
        "chunks": chunks_done,
        "failed": failed,
        "log_rows": log_rows,
        "pap_df": pap_df,
        "pre": pre,
    }


def compare_and_write(symbol_data, loop_out: dict) -> dict:
    bt_trades, _eq, _skip = simulate_priority_portfolio(
        symbol_data,
        agent_priority=list(AGENTS_PRIORITY),
        precomputed=loop_out["pre"],
        start_date=FULL_START,
        end_date=FULL_END,
        gap_tolerance=GAP_TOLERANCE,
        commission_rate=COMMISSION_RATE,
        slippage_rate=SLIPPAGE_RATE,
        initial_capital=INITIAL_CAPITAL,
        max_positions=MAX_POSITIONS,
        position_size=POSITION_SIZE,
    )
    pap_df = loop_out["pap_df"]
    cmp_df = compare_trade_frames(bt_trades, pap_df, "day_loop_5yil")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmp_df.to_csv(CMP_FILE, index=False, encoding="utf-8")
    n_match = int(cmp_df["eslesme"].sum()) if len(cmp_df) else 0
    summary = {
        "paper_yolu": "p1_paper.run_phase_takip / run_phase_acilis / run_phase_aksam (gün gün, _REPLAY_MODE)",
        "backtest_yolu": "research_p1.run_p1_revision_backtest.simulate_priority_portfolio",
        "start": FULL_START,
        "end": FULL_END,
        "n_days_run": loop_out["n_days"],
        "n_dates_expected": loop_out["n_dates"],
        "chunks": loop_out["chunks"],
        "failed": loop_out["failed"],
        "bt_n": int(len(bt_trades)),
        "paper_n": int(len(pap_df)),
        "n_karsilastirma": int(len(cmp_df)),
        "n_eslesen": n_match,
        "n_uyusmayan": int(len(cmp_df) - n_match),
        "bt_pnl": round(float(bt_trades["net_profit_tl"].sum()), 2) if len(bt_trades) else 0.0,
        "paper_pnl": round(float(pd.to_numeric(pap_df["net_profit_tl"], errors="coerce").fillna(0).sum()), 2) if len(pap_df) else 0.0,
        "bt_by_agent": _agent_breakdown(bt_trades, "primary_agent", "net_profit_tl"),
        "paper_by_agent": _agent_breakdown(pap_df, "primary_agent", "net_profit_tl"),
        "gap": gap_distribution(loop_out["log_rows"]),
        "cikti": str(CMP_FILE),
        "referans_beklenen_n": 543,
        "referans_beklenen_pnl": 82307.0,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("Yazildi %s eslesen=%d uyusmayan=%d", CMP_FILE, n_match, summary["n_uyusmayan"])
    return summary


def main() -> dict:
    cache = _load_cache()
    if not cache:
        raise RuntimeError("OHLCV cache yok; research_p1/data boş. Gün-gün döngü çalışmıyor.")
    loop_out = run_day_loop(cache, FULL_START, FULL_END)
    if loop_out["failed"] is not None:
        summary = {
            "status": "failed",
            "failed": loop_out["failed"],
            "chunks": loop_out["chunks"],
            "n_days_run": loop_out["n_days"],
        }
        SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return summary
    return compare_and_write(cache, loop_out)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False, default=str))
