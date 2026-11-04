# -*- coding: utf-8 -*-
"""PR #8 ölçüm: PnL ayrıştırma, gap yönü/duyarlılık, canlı vs replay sıra, eksik açılış.
Eşik değiştirmez. Çıktı: results/parity_decomposition_summary.json
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
)
from research_p1.data_downloader import BIST100_SYMBOLS, load_cached_ohlcv
from research_p1.p1_engine import compute_all_indicators
from research_p1.p1_exit_rules import gap_ratio, tp1_sold_lots
from research_p1.run_p1_revision_backtest import (
    evaluate_custom_agent_signals,
    simulate_priority_portfolio,
)
from research_p1.run_trade_level_parity import (
    _paper_trades_from_log,
    _redirect_paper_io,
    compare_trade_frames,
    gap_distribution,
)

log = logging.getLogger("parity_decomp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FULL_START = "2021-10-06"
FULL_END = "2026-10-05"
LIVE_START = "2025-10-06"
LIVE_END = "2026-10-05"
MISSING_SYMS = ["TAVHL", "HALKB", "ALBRK", "EUPWR", "BSOKE", "CIMSA", "RALYH", "IZENR", "TABGD"]
OUT = RESULTS_DIR / "parity_decomposition_summary.json"
PAPER_LOG = RESULTS_DIR / "_day_loop_full" / "paper_trading_log.csv"
BT_TRADES = REPO_ROOT / "research_p1" / "results" / "p1_rev_ALPHA_B_SIMPLE_20k_trades.csv"


def _f(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_cache() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in BIST100_SYMBOLS:
        df, _ = load_cached_ohlcv(s)
        if df is not None and len(df) >= 50:
            out[s] = compute_all_indicators(df)
    return out


def load_paper_taken() -> pd.DataFrame:
    rows = pd.read_csv(PAPER_LOG)
    taken = rows[rows["kapasite_durumu"] == "ISLEME_ALINDI"].copy()
    taken = taken[taken["cikis_nedeni"].astype(str).str.strip() != ""]
    taken["key"] = (
        taken["sembol"].astype(str) + "|" + taken["birincil_ajan"].astype(str) + "|" + taken["giris_tarihi"].astype(str)
    )
    return taken


def pnl_decomposition() -> dict:
    pap = load_paper_taken()
    bt = pd.read_csv(BT_TRADES)
    bt["key"] = bt["symbol"].astype(str) + "|" + bt["primary_agent"].astype(str) + "|" + bt["entry_date"].astype(str)
    pap_map = {r["key"]: r for r in pap.to_dict("records")}
    bt_map = {r["key"]: r for r in bt.to_dict("records")}
    keys = sorted(set(pap_map) | set(bt_map))

    buckets = {
        "lot_only_pnl": 0.0,
        "entry_px_pnl": 0.0,
        "exit_px_pnl": 0.0,
        "entry_slip_pnl": 0.0,
        "exit_slip_pnl": 0.0,
        "entry_comm_pnl": 0.0,
        "exit_comm_pnl": 0.0,
        "tp1_residual_pnl": 0.0,
        "unmatched_paper_pnl": 0.0,
        "unmatched_bt_pnl": 0.0,
        "recon_residual": 0.0,
    }
    by_agent = {}
    rows_out = []
    lot_pred_err = []
    n_paper_lots_lt = n_eq = n_gt = 0

    def ag_acc(agent, field, val):
        by_agent.setdefault(agent, {k: 0.0 for k in list(buckets) + ["d_pnl", "n_matched", "n_unmatched_p", "n_unmatched_b"]})
        by_agent[agent][field] = by_agent[agent].get(field, 0.0) + val

    for k in keys:
        p = pap_map.get(k)
        b = bt_map.get(k)
        agent = k.split("|")[1]
        if p is None:
            pnl_b = float(b["net_profit_tl"])
            buckets["unmatched_bt_pnl"] += -pnl_b  # paper-bt
            ag_acc(agent, "unmatched_bt_pnl", -pnl_b)
            ag_acc(agent, "d_pnl", -pnl_b)
            ag_acc(agent, "n_unmatched_b", 1)
            rows_out.append({"key": k, "d_pnl": -pnl_b, "kok": "EKSIK_PAPER"})
            continue
        if b is None:
            pnl_p = float(p["net_pnl_tl"])
            buckets["unmatched_paper_pnl"] += pnl_p
            ag_acc(agent, "unmatched_paper_pnl", pnl_p)
            ag_acc(agent, "d_pnl", pnl_p)
            ag_acc(agent, "n_unmatched_p", 1)
            rows_out.append({"key": k, "d_pnl": pnl_p, "kok": "EKSIK_BACKTEST"})
            continue

        lots_p = int(float(p["onaylanan_lot"]))
        lots_b = int(float(b["initial_lots"]))
        ep_p = float(p["simule_dolum_fiyati"] if pd.notna(p.get("simule_dolum_fiyati")) else p["gerceklesen_acilis_fiyati"])
        ep_b = float(b["entry_price"])
        xp_p = float(p["cikis_fiyati"])
        xp_b = float(b["exit_price"])
        pnl_p = float(p["net_pnl_tl"])
        pnl_b = float(b["net_profit_tl"])
        d_pnl = pnl_p - pnl_b
        open_p = float(p["gerceklesen_acilis_fiyati"])
        # backtest fill = open*(1+slip); reverse open
        open_b = ep_b / (1.0 + SLIPPAGE_RATE)

        entry_comm_p = lots_p * ep_p * COMMISSION_RATE
        entry_comm_b = lots_b * ep_b * COMMISSION_RATE
        exit_comm_p = lots_p * xp_p * COMMISSION_RATE
        exit_comm_b = lots_b * xp_b * COMMISSION_RATE
        # slip TL relative to raw open / raw ref (ref = exit/(1-slip))
        ref_p = xp_p / (1.0 - SLIPPAGE_RATE)
        ref_b = xp_b / (1.0 - SLIPPAGE_RATE)
        entry_slip_p = lots_p * (ep_p - open_p)
        entry_slip_b = lots_b * (ep_b - open_b)
        exit_slip_p = lots_p * (ref_p - xp_p)
        exit_slip_b = lots_b * (ref_b - xp_b)

        # Per-lot full-exit identity at BT prices (ignores TP1 split)
        per_lot_b = (xp_b * (1.0 - COMMISSION_RATE) - ep_b * (1.0 + COMMISSION_RATE))
        lot_effect = (lots_p - lots_b) * per_lot_b
        # Hold lots at BT, change entry px
        entry_px_effect = lots_p * (-(ep_p - ep_b) * (1.0 + COMMISSION_RATE))
        exit_px_effect = lots_p * ((xp_p - xp_b) * (1.0 - COMMISSION_RATE))
        predicted = lot_effect + entry_px_effect + exit_px_effect
        residual = d_pnl - predicted
        tp1_like = str(b.get("exit_reason", "")) in ("TRAILING", "TP1_FULL", "TP1") or str(p.get("cikis_nedeni", "")) in (
            "TRAILING",
            "TP1_FULL",
            "TP1",
        )

        if lots_p < lots_b:
            n_paper_lots_lt += 1
        elif lots_p == lots_b:
            n_eq += 1
        else:
            n_gt += 1

        same_px = abs(ep_p - ep_b) < 1e-4 and abs(xp_p - xp_b) < 1e-4
        if same_px:
            lot_pred_err.append(d_pnl - lot_effect)

        buckets["lot_only_pnl"] += lot_effect
        buckets["entry_px_pnl"] += entry_px_effect
        buckets["exit_px_pnl"] += exit_px_effect
        buckets["entry_comm_pnl"] += -(entry_comm_p - entry_comm_b)
        buckets["exit_comm_pnl"] += -((exit_comm_p) - (exit_comm_b))
        buckets["entry_slip_pnl"] += -(entry_slip_p - entry_slip_b)
        buckets["exit_slip_pnl"] += -(exit_slip_p - exit_slip_b)
        if tp1_like:
            buckets["tp1_residual_pnl"] += residual
        else:
            buckets["recon_residual"] += residual

        ag_acc(agent, "lot_only_pnl", lot_effect)
        ag_acc(agent, "entry_px_pnl", entry_px_effect)
        ag_acc(agent, "exit_px_pnl", exit_px_effect)
        ag_acc(agent, "tp1_residual_pnl", residual if tp1_like else 0.0)
        ag_acc(agent, "d_pnl", d_pnl)
        ag_acc(agent, "n_matched", 1)

        rows_out.append({
            "key": k,
            "agent": agent,
            "lots_p": lots_p,
            "lots_b": lots_b,
            "d_lots": lots_p - lots_b,
            "d_pnl": round(d_pnl, 4),
            "lot_effect": round(lot_effect, 4),
            "entry_px_effect": round(entry_px_effect, 4),
            "exit_px_effect": round(exit_px_effect, 4),
            "residual": round(residual, 4),
            "tp1_like": tp1_like,
            "same_px": same_px,
            "reason_p": p.get("cikis_nedeni"),
            "reason_b": b.get("exit_reason"),
        })

    pap_pnl = float(pd.to_numeric(pap["net_pnl_tl"]).sum())
    bt_pnl = float(pd.to_numeric(bt["net_profit_tl"]).sum())
    err = np.array(lot_pred_err) if lot_pred_err else np.array([0.0])
    return {
        "paper_n": int(len(pap)),
        "bt_n": int(len(bt)),
        "paper_pnl": round(pap_pnl, 2),
        "bt_pnl": round(bt_pnl, 2),
        "d_pnl": round(pap_pnl - bt_pnl, 2),
        "gtd_d_pnl": round(by_agent.get("GTD", {}).get("d_pnl", 0.0), 2),
        "buckets_paper_minus_bt": {k: round(v, 2) for k, v in buckets.items()},
        "by_agent": {a: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()} for a, d in by_agent.items()},
        "lot_formula": {
            "n_matched_same_px": int(len(lot_pred_err)),
            "n_paper_lots_lt": n_paper_lots_lt,
            "n_lots_eq": n_eq,
            "n_paper_lots_gt": n_gt,
            "same_px_lot_effect_sum": round(float(sum(r["lot_effect"] for r in rows_out if r.get("same_px"))), 2),
            "same_px_actual_d_pnl_sum": round(float(sum(r["d_pnl"] for r in rows_out if r.get("same_px"))), 2),
            "same_px_pred_err_mean": round(float(np.mean(err)), 4),
            "same_px_pred_err_sum": round(float(np.sum(err)), 2),
            "aciklama": (
                "paper lots=int(alloc/((1+komisyon)*fill)); bt lots=int(alloc/fill). "
                "lot_effect=(lots_p-lots_b)*(exit*(1-k)-entry*(1+k)) tam çıkış varsayımı."
            ),
        },
        "n_matched": int(sum(1 for r in rows_out if "lots_p" in r)),
    }


def gap_direction_and_extremes(cache: Dict[str, pd.DataFrame]) -> dict:
    rows = pd.read_csv(PAPER_LOG)
    gaps = pd.to_numeric(rows["gap_orani_pct"], errors="coerce")
    status = rows["kapasite_durumu"].astype(str)
    rej = rows[status == "GAP_REDDI"]
    g = pd.to_numeric(rej["gap_orani_pct"], errors="coerce")
    up = int((g > 0).sum())
    down = int((g < 0).sum())
    zero = int((g == 0).sum())
    extreme = rows[gaps < -30.0][["sembol", "giris_tarihi", "sinyal_tarihi", "gap_orani_pct", "kapasite_durumu", "gerceklesen_acilis_fiyati", "sinyal_kapanis_fiyati"]]
    extreme_list = extreme.sort_values("gap_orani_pct").to_dict("records")
    # also scan cache for open/close < -30% even if no signal
    cache_ext = []
    for sym, df in cache.items():
        if "open" not in df.columns or "close" not in df.columns:
            continue
        prev_c = df["close"].shift(1)
        gap = df["open"] / prev_c - 1.0
        m = gap < -0.30
        for dt, val in gap[m].items():
            cache_ext.append({
                "sembol": sym,
                "tarih": str(pd.Timestamp(dt).date()),
                "gap_orani_pct": round(float(val) * 100.0, 4),
                "open": float(df.loc[dt, "open"]),
                "prev_close": float(prev_c.loc[dt]),
            })
    cache_ext.sort(key=lambda x: x["gap_orani_pct"])
    return {
        "kural": "abs(gap) > GAP_TOLERANCE  (yön ayırmaz; aşağı ve yukarı)",
        "n_gap_reddi": int(len(rej)),
        "gap_reddi_yukari": up,
        "gap_reddi_asagi": down,
        "gap_reddi_sifir": zero,
        "n_gap_lt_minus_30_in_log": int(len(extreme)),
        "gap_lt_minus_30_log": extreme_list[:50],
        "gap_lt_minus_30_cache_n": len(cache_ext),
        "gap_lt_minus_30_cache": cache_ext[:80],
        "gap_distribution": gap_distribution(rows.to_dict("records")),
    }


def metrics_from_sim(trades: pd.DataFrame, eq: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"n_trades": 0, "mean_return_pct": None, "profit_factor": None, "total_profit_tl": 0.0, "max_daily_mtm_dd_pct": None}
    rets = pd.to_numeric(trades["net_return_pct"], errors="coerce")
    pnl = pd.to_numeric(trades["net_profit_tl"], errors="coerce")
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    pf = float(wins / abs(losses)) if losses < 0 else None
    max_dd = float(eq["dd_pct"].min()) if eq is not None and len(eq) and "dd_pct" in eq.columns else None
    return {
        "n_trades": int(len(trades)),
        "mean_return_pct": round(float(rets.mean()), 4),
        "profit_factor": round(pf, 4) if pf is not None else None,
        "total_profit_tl": round(float(pnl.sum()), 2),
        "max_daily_mtm_dd_pct": round(max_dd, 2) if max_dd is not None else None,
    }


def gap_sensitivity(cache, pre) -> List[dict]:
    tols = [
        ("mevcut_0.0001", 0.0001),
        ("le_0.5pct", 0.0050),
        ("le_1pct", 0.0100),
        ("le_2pct", 0.0200),
        ("sinirsiz", 10.0),
    ]
    out = []
    for label, tol in tols:
        log.info("gap sensitivity %s tol=%s", label, tol)
        trades, eq, _sk = simulate_priority_portfolio(
            cache,
            agent_priority=list(AGENTS_PRIORITY),
            precomputed=pre,
            start_date=FULL_START,
            end_date=FULL_END,
            gap_tolerance=tol,
            commission_rate=COMMISSION_RATE,
            slippage_rate=SLIPPAGE_RATE,
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_POSITIONS,
            position_size=POSITION_SIZE,
        )
        m = metrics_from_sim(trades, eq)
        m["label"] = label
        m["gap_tolerance"] = tol
        out.append(m)
    return out


def run_phase_loop(symbol_data, pre, start, end, order: str, out_dir: Path) -> pd.DataFrame:
    import p1_paper as p1_pap

    _redirect_paper_io(out_dir)
    p1_pap.save_paper_state({
        "_gen": 0, "_updated_at": start,
        "cash_20k": float(INITIAL_CAPITAL), "cash_10k": float(INITIAL_CAPITAL),
        "peak_equity_20k": float(INITIAL_CAPITAL), "peak_equity_10k": float(INITIAL_CAPITAL),
        "positions_20k": {}, "positions_10k": {},
        "pending_candidates": [], "tracked_rejected_signals": [],
        "daily_equity": [], "processed_runs": {}, "paper_start_date": start,
    })
    p1_pap.save_paper_log([])
    dates = sorted(list(set().union(*[df.index for df in symbol_data.values()])))
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    dates = [d for d in dates if s <= d <= e]
    prev = p1_pap._REPLAY_MODE
    p1_pap._REPLAY_MODE = True
    prev_level = p1_pap.log.level
    p1_pap.log.setLevel(logging.WARNING)
    try:
        for i, dt in enumerate(dates):
            if order == "replay":
                p1_pap.run_phase_takip(as_of_date=dt, symbol_data=symbol_data)
                p1_pap.run_phase_acilis(as_of_date=dt, symbol_data=symbol_data, fetch_vwap=False)
                p1_pap.run_phase_aksam(as_of_date=dt, symbol_data=symbol_data, precomputed=pre, skip_signal_parity=True)
            else:
                # canlı: gün t açılış (dünün aksam adayı) → aynı gün takip → aksam
                if i > 0:
                    p1_pap.run_phase_acilis(as_of_date=dt, symbol_data=symbol_data, fetch_vwap=False)
                    p1_pap.run_phase_takip(as_of_date=dt, symbol_data=symbol_data)
                p1_pap.run_phase_aksam(as_of_date=dt, symbol_data=symbol_data, precomputed=pre, skip_signal_parity=True)
    finally:
        p1_pap._REPLAY_MODE = prev
        p1_pap.log.setLevel(prev_level)
    return _paper_trades_from_log(p1_pap.load_paper_log())


def live_vs_replay(cache, pre) -> dict:
    log.info("live vs replay %s -> %s", LIVE_START, LIVE_END)
    replay = run_phase_loop(cache, pre, LIVE_START, LIVE_END, "replay", RESULTS_DIR / "_order_replay_1y")
    live = run_phase_loop(cache, pre, LIVE_START, LIVE_END, "live", RESULTS_DIR / "_order_live_1y")
    cmp = compare_trade_frames(
        replay.rename(columns={"primary_agent": "primary_agent"}) if len(replay) else replay,
        live, "live_vs_replay",
    )
    # compare_trade_frames expects bt vs paper column names from simulate. Both frames already have those names.
    n_match = int(cmp["eslesme"].sum()) if len(cmp) else 0
    mismatch = cmp.loc[~cmp["eslesme"].astype(bool)] if len(cmp) else cmp
    reasons = mismatch["kok_neden"].astype(str).value_counts().to_dict() if len(mismatch) else {}
    return {
        "start": LIVE_START,
        "end": LIVE_END,
        "replay_n": int(len(replay)),
        "live_n": int(len(live)),
        "replay_pnl": round(float(pd.to_numeric(replay["net_profit_tl"], errors="coerce").sum()), 2) if len(replay) else 0.0,
        "live_pnl": round(float(pd.to_numeric(live["net_profit_tl"], errors="coerce").sum()), 2) if len(live) else 0.0,
        "n_karsilastirma": int(len(cmp)),
        "n_eslesen": n_match,
        "n_uyusmayan": int(len(cmp) - n_match),
        "kok_neden_sayim": {str(k): int(v) for k, v in reasons.items()},
        "uyusmayan_ornek": mismatch.head(20).to_dict("records") if len(mismatch) else [],
        "replay_sira": "takip -> acilis -> aksam (aynı bar çıkış yok)",
        "live_sira": "acilis -> takip -> aksam (aynı gün çıkış mümkün)",
    }


def missing_opens(cache) -> dict:
    all_dates = sorted(list(set().union(*[df.index for df in cache.values()])))
    s, e = pd.to_datetime(FULL_START), pd.to_datetime(FULL_END)
    cal = [d for d in all_dates if s <= d <= e]
    out = {}
    for sym in MISSING_SYMS:
        df = cache.get(sym)
        info = {"in_cache": df is not None}
        if df is None:
            out[sym] = info
            continue
        idx = df.index[(df.index >= s) & (df.index <= e)]
        first = str(df.index.min().date())
        last = str(df.index.max().date())
        nan_open = []
        if "open" in df.columns:
            sub = df.loc[idx]
            bad = sub[sub["open"].isna() | (pd.to_numeric(sub["open"], errors="coerce") <= 0)]
            nan_open = [str(pd.Timestamp(i).date()) for i in bad.index]
        missing_cal = [str(d.date()) for d in cal if d not in df.index]
        info.update({
            "first_bar": first,
            "last_bar": last,
            "n_bars_in_window": int(len(idx)),
            "n_calendar_in_window": int(len(cal)),
            "n_missing_calendar": int(len(missing_cal)),
            "missing_calendar_dates": missing_cal,
            "n_nan_or_nonpos_open": int(len(nan_open)),
            "nan_or_nonpos_open_dates": nan_open,
            "pre_listing": first > FULL_START,
        })
        out[sym] = info
    return out


def cost_models() -> dict:
    return {
        "oranlar": {"commission": COMMISSION_RATE, "slippage": SLIPPAGE_RATE, "tur_basi_pct": 1.20},
        "backtest_giris": "fill=open*(1+slip); lots=int(alloc/fill); tot_cost=lots*fill*(1+comm)",
        "backtest_cikis": "exit=ref*(1-slip); net_inc=lots*exit - lots*exit*comm = lots*exit*(1-comm)",
        "paper_giris": "fill=open*(1+slip); lots=int((alloc/(1+comm))/fill); cost=lots*fill*(1+comm)",
        "paper_cikis": "exit=ref*(1-slip); inc=lots*exit*(1-comm)",
        "per_lot_maliyet_ayni": True,
        "lot_formulu_farkli": True,
        "cikis_net_formulu_ayni": True,
        "giris_fill_formulu_ayni": True,
    }


def main() -> dict:
    cache = _load_cache()
    pre = {sym: {ag: evaluate_custom_agent_signals(df, ag) for ag in AGENTS_PRIORITY} for sym, df in cache.items()}
    payload = {
        "maliyet_modelleri": cost_models(),
        "pnl_ayristirma": pnl_decomposition(),
        "gap": gap_direction_and_extremes(cache),
        "gap_duyarlilik_portfoy_20k_1p2": gap_sensitivity(cache, pre),
        "canli_vs_replay_1y": live_vs_replay(cache, pre),
        "eksik_acilis": missing_opens(cache),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("yazildi %s", OUT)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False, default=str))
