# -*- coding: utf-8 -*-
"""
research_p1/run_p1_revision_exact.py
====================================
Kullanıcının geri bildirimleri doğrultusunda tam ve kesin analiz:
1. Portföy motorunun ALPHA_B yerine ALPHA_B_SIMPLE ile koşulması:
   Ajanlar: GTD (1) > ALPHA_B_SIMPLE (2) > ZT3 (3), RelVol öncelikli, 5 slot.
   Senaryolar:
     - 20k Boyut, %1.2 Maliyet (Komisyon %0.30, Slipaj %0.30)
     - 10k Boyut, %1.2 Maliyet (Komisyon %0.30, Slipaj %0.30)
2. Yıl yıl döküm (2021..2026), IS / OOS dökümü ve Ajan katkı dökümü.
3. Ekonomik Çerçeve & Boştaki Nakdin Gerçek Dinamik Nemalandırılması (Compounding):
   - Önceki rapordaki hata düzeltildi: Boştaki nakit statik ~85k üzerinden değil,
     her gün biriken faizle birlikte dinamik olarak portföy nakit bakiyesinde büyütülerek hesaplandı.
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

from research_p1.data_downloader import load_cached_ohlcv, BIST100_SYMBOLS
from research_p1.p1_engine import compute_all_indicators
from research_p1.run_p1_revision_backtest import (
    evaluate_custom_agent_signals,
    simulate_priority_portfolio,
)
from research_p1.run_hurdle_and_robustness import get_daily_risk_free_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FULL_START = "2021-10-06"
FULL_END   = "2026-10-05"
IS_START   = "2021-10-06"
IS_END     = "2024-10-05"
OOS1_START = "2024-10-06"
OOS1_END   = "2025-10-05"
OOS2_START = "2025-10-06"
OOS2_END   = "2026-10-05"
OOS_COMB_START = "2024-10-06"
OOS_COMB_END   = "2026-10-05"


def calc_ci(returns: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    if len(returns) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot = [rng.choice(returns, size=len(returns), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def run_exact_analysis():
    log.info("Veriler yükleniyor...")
    symbol_data = {}
    for s in BIST100_SYMBOLS:
        df, meta = load_cached_ohlcv(s)
        if df is not None and len(df) >= 50:
            symbol_data[s] = compute_all_indicators(df)

    log.info("XU100 Rejim Serisi hazırlanıyor...")
    bench_dir = Path(__file__).parent / "data" / "benchmarks"
    df_xu = pd.read_csv(bench_dir / "XU100.IS.csv", index_col=0)
    df_xu.index = pd.to_datetime(df_xu.index)
    df_xu["ema200"] = df_xu["close"].ewm(span=200, adjust=False).mean()
    xu100_bull_regime = df_xu["close"] > df_xu["ema200"]

    # Sinyal Ön-Hesaplama (GTD, ALPHA_B_SIMPLE, ZT3)
    agents_needed = ["GTD", "ALPHA_B_SIMPLE", "ZT3"]
    precomputed = {}
    for sym, df in symbol_data.items():
        precomputed[sym] = {ag: evaluate_custom_agent_signals(df, ag) for ag in agents_needed}

    log.info("--- 1. Portföy Simülasyonu: GTD > ALPHA_B_SIMPLE > ZT3 (5 Slot, 100k TL) ---")

    # A) 20k Boyut, %1.2 Maliyet
    trades_20k, eq_20k, skip_20k = simulate_priority_portfolio(
        symbol_data,
        agent_priority=["GTD", "ALPHA_B_SIMPLE", "ZT3"],
        precomputed=precomputed,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=0.0030, slippage_rate=0.0030,
        initial_capital=100_000.0, max_positions=5,
        position_size=20_000.0,
        xu100_regime_series=xu100_bull_regime
    )
    trades_20k.to_csv(RESULTS_DIR / "p1_rev_ALPHA_B_SIMPLE_20k_trades.csv", index=False)
    eq_20k.to_csv(RESULTS_DIR / "p1_rev_ALPHA_B_SIMPLE_20k_daily_equity.csv", index=False)
    skip_20k.to_csv(RESULTS_DIR / "p1_rev_ALPHA_B_SIMPLE_20k_skipped.csv", index=False)

    # B) 10k Boyut, %1.2 Maliyet
    trades_10k, eq_10k, skip_10k = simulate_priority_portfolio(
        symbol_data,
        agent_priority=["GTD", "ALPHA_B_SIMPLE", "ZT3"],
        precomputed=precomputed,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001,
        commission_rate=0.0030, slippage_rate=0.0030,
        initial_capital=100_000.0, max_positions=5,
        position_size=10_000.0,
        xu100_regime_series=xu100_bull_regime
    )
    trades_10k.to_csv(RESULTS_DIR / "p1_rev_ALPHA_B_SIMPLE_10k_trades.csv", index=False)
    eq_10k.to_csv(RESULTS_DIR / "p1_rev_ALPHA_B_SIMPLE_10k_daily_equity.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Dökümler: 20k Boyut
    # ─────────────────────────────────────────────────────────────────────────
    # 1. Ajan Kırılımı
    by_agent = trades_20k.groupby("primary_agent").agg(
        n=("net_return_pct", "count"),
        win=("net_return_pct", lambda s: round((s > 0).mean() * 100, 2)),
        mean_ret=("net_return_pct", "mean"),
        median_ret=("net_return_pct", "median"),
        total_pnl=("net_profit_tl", "sum")
    ).reset_index()
    by_agent.to_csv(RESULTS_DIR / "p1_rev_SIMPLE_20k_by_agent.csv", index=False)

    # 2. Yıl Kırılımı
    trades_20k["entry_dt"] = pd.to_datetime(trades_20k["entry_date"])
    trades_20k["year"] = trades_20k["entry_dt"].dt.year
    by_year = trades_20k.groupby("year").agg(
        n=("net_return_pct", "count"),
        win=("net_return_pct", lambda s: round((s > 0).mean() * 100, 2)),
        mean_ret=("net_return_pct", "mean"),
        median_ret=("net_return_pct", "median"),
        total_pnl=("net_profit_tl", "sum")
    ).reset_index()
    by_year.to_csv(RESULTS_DIR / "p1_rev_SIMPLE_20k_by_year.csv", index=False)

    # 3. IS / OOS Kırılımı
    period_conditions = [
        ("IS_2021_2024", trades_20k["entry_date"].between(IS_START, IS_END)),
        ("OOS1_2024_2025", trades_20k["entry_date"].between(OOS1_START, OOS1_END)),
        ("OOS2_2025_2026", trades_20k["entry_date"].between(OOS2_START, OOS2_END)),
        ("OOS_COMB_2024_2026", trades_20k["entry_date"].between(OOS_COMB_START, OOS_COMB_END)),
        ("FULL_5Y", trades_20k["entry_date"].between(FULL_START, FULL_END)),
    ]
    period_rows = []
    for p_name, cond in period_conditions:
        grp = trades_20k[cond]
        rets = grp["net_return_pct"].values
        pnls = grp["net_profit_tl"].values
        gp = pnls[pnls > 0].sum()
        gl = abs(pnls[pnls < 0].sum())
        pf = round(gp / gl, 2) if gl > 0 else (999.0 if gp > 0 else 0.0)
        ci_lo, ci_hi = calc_ci(rets)
        period_rows.append({
            "period": p_name,
            "n_trades": len(grp),
            "win_rate": round(float((rets > 0).mean() * 100), 2),
            "mean_ret_pct": round(float(rets.mean()), 4),
            "median_ret_pct": round(float(np.median(rets)), 4),
            "profit_factor": pf,
            "total_pnl_tl": round(float(pnls.sum()), 2),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
        })
    df_by_period = pd.DataFrame(period_rows)
    df_by_period.to_csv(RESULTS_DIR / "p1_rev_SIMPLE_20k_by_period.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Dinamik Nemalandırma (Compounding Cash Interest) Hesaplaması
    # ─────────────────────────────────────────────────────────────────────────
    # df_eq içindeki cash ve open_value'dan adım adım gerçek PPF nemalandırması:
    # Her günün sonunda nakit bakiyesi = önceki nakit + gün içindeki net nakit akışı + O GÜNÜN REPO FAİZİ
    # Böylece kazanılan faiz ertesi gün nakde eklenir ve bileşik faiz kazanır!
    all_dates = sorted(list(set().union(*[df.index for df in symbol_data.values()])))
    start_ts = pd.to_datetime(FULL_START)
    end_ts = pd.to_datetime(FULL_END)
    sim_dates = [d for d in all_dates if start_ts <= d <= end_ts]

    # Portföy simülasyonunu doğrudan cash compounding ile adım adım çalıştıralım:
    def simulate_with_compound_cash(p_size: float = 20_000.0) -> Tuple[float, float, List[dict]]:
        active_pos = {}
        cash_balance = 100_000.0
        equity_hist = []
        priority_rank = {"GTD": 0, "ALPHA_B_SIMPLE": 1, "ZT3": 2}

        for t_idx, cur_dt in enumerate(all_dates):
            # Günlük gecelik repo faizi işletimi (boştaki nakit gece boyunca nemalanır)
            if t_idx > 0 and cur_dt >= start_ts:
                prev_dt = all_dates[t_idx - 1]
                days = (cur_dt - prev_dt).days
                rf_rate = get_daily_risk_free_rate(cur_dt)
                rf_factor = (1.0 + rf_rate) ** (days / 365.0) - 1.0
                interest = cash_balance * rf_factor
                cash_balance += interest

            # Pozisyon çıkışları
            closed_s = []
            for sym, pos in active_pos.items():
                df = symbol_data.get(sym)
                if df is None or cur_dt not in df.index:
                    continue
                row = df.loc[cur_dt]
                o_b, h_b, l_b, c_b = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                h_days = (cur_dt - pos["entry_date"]).days
                stop_l = round(pos["entry_price"] * 0.95, 4)
                tp1_l = round(pos["entry_price"] * 1.08, 4)
                tr_l = round(pos["peak"] * 0.95, 4) if pos["tp1"] else 0.0

                is_s = l_b <= stop_l
                is_tp = h_b >= tp1_l and not pos["tp1"]
                is_tr = pos["tp1"] and l_b <= tr_l
                is_mg = h_days >= 10

                if is_s and is_tp:
                    ref_p = o_b if o_b <= stop_l else stop_l
                    p_exit = round(ref_p * (1.0 - 0.0030), 4)
                    inc = round(pos["lots"] * p_exit * (1.0 - 0.0030), 4)
                    cash_balance += inc
                    closed_s.append(sym)
                    continue
                if is_s:
                    ref_p = o_b if o_b <= stop_l else stop_l
                    p_exit = round(ref_p * (1.0 - 0.0030), 4)
                    inc = round(pos["lots"] * p_exit * (1.0 - 0.0030), 4)
                    cash_balance += inc
                    closed_s.append(sym)
                    continue
                if is_tp:
                    ref_p = max(o_b, tp1_l)
                    p_exit = round(ref_p * (1.0 - 0.0030), 4)
                    s_lots = 1 if pos["lots"] <= 1 else pos["lots"] // 2
                    pos["lots"] -= s_lots
                    inc = round(s_lots * p_exit * (1.0 - 0.0030), 4)
                    cash_balance += inc
                    pos["tp1"] = True
                    if pos["lots"] == 0:
                        closed_s.append(sym)
                        continue
                elif is_tr:
                    ref_p = o_b if o_b <= tr_l else tr_l
                    p_exit = round(ref_p * (1.0 - 0.0030), 4)
                    inc = round(pos["lots"] * p_exit * (1.0 - 0.0030), 4)
                    cash_balance += inc
                    closed_s.append(sym)
                    continue
                if is_mg:
                    p_exit = round(c_b * (1.0 - 0.0030), 4)
                    inc = round(pos["lots"] * p_exit * (1.0 - 0.0030), 4)
                    cash_balance += inc
                    closed_s.append(sym)
                    continue

                pos["peak"] = max(pos["peak"], h_b)

            for sym in closed_s:
                active_pos.pop(sym)

            if start_ts <= cur_dt <= end_ts:
                o_val = sum(
                    p["lots"] * float(symbol_data[s].loc[cur_dt, "close"])
                    for s, p in active_pos.items()
                    if s in symbol_data and cur_dt in symbol_data[s].index
                )
                equity_hist.append({"date": str(cur_dt.date()), "equity": cash_balance + o_val, "cash": cash_balance})

            # Alım
            if t_idx == 0 or cur_dt < start_ts or cur_dt > end_ts:
                continue
            prv_dt = all_dates[t_idx - 1]
            cands = []
            for sym, df in symbol_data.items():
                if prv_dt not in df.index or cur_dt not in df.index:
                    continue
                trig = [ag for ag in agents_needed if precomputed[sym][ag].loc[prv_dt]]
                if not trig:
                    continue
                prev_c = float(df.loc[prv_dt, "close"])
                curr_o = float(df.loc[cur_dt, "open"])
                gap = (curr_o / prev_c) - 1.0
                if abs(gap) > 0.0001:
                    continue
                best_rank = min(priority_rank[ag] for ag in trig)
                prim_ag = [ag for ag in trig if priority_rank[ag] == best_rank][0]
                rel_v = float(df.loc[prv_dt, "rel_vol"]) if "rel_vol" in df.columns else 1.0
                cands.append({"symbol": sym, "open": curr_o, "rank": best_rank, "rel_vol": rel_v})

            cands.sort(key=lambda x: (x["rank"], -x["rel_vol"], x["symbol"]))

            for cand in cands:
                sym = cand["symbol"]
                if sym in active_pos or len(active_pos) >= 5:
                    continue
                alloc = min(p_size, cash_balance / max(1, 5 - len(active_pos)))
                fill_p = round(cand["open"] * 1.0030, 4)
                lots = int(alloc / max(fill_p, 0.01))
                if lots < 1:
                    continue
                tot = round(lots * fill_p * 1.0030, 4)
                if tot > cash_balance:
                    continue
                cash_balance -= tot
                active_pos[sym] = {"lots": lots, "entry_price": fill_p, "peak": fill_p, "tp1": False, "entry_date": cur_dt}

        fin_eq = equity_hist[-1]["equity"]
        cagr = ((fin_eq / 100_000.0) ** 0.2 - 1.0) * 100.0
        return fin_eq, cagr, equity_hist

    fin_20k_comp, cagr_20k_comp, eq_hist_20k_comp = simulate_with_compound_cash(20_000.0)
    fin_10k_comp, cagr_10k_comp, eq_hist_10k_comp = simulate_with_compound_cash(10_000.0)

    # 100% PPF bileşik getiri
    ppf_bal = 100_000.0
    for i in range(1, len(sim_dates)):
        d = sim_dates[i]
        pd_ = sim_dates[i - 1]
        days = (d - pd_).days
        r = get_daily_risk_free_rate(d)
        ppf_bal *= (1.0 + r) ** (days / 365.0)
    cagr_ppf = ((ppf_bal / 100_000.0) ** 0.2 - 1.0) * 100.0

    economic_comp = {
        "final_p1_20k_with_compounded_ppf_tl": round(fin_20k_comp, 2),
        "cagr_p1_20k_with_compounded_ppf_pct": round(cagr_20k_comp, 2),
        "final_p1_10k_with_compounded_ppf_tl": round(fin_10k_comp, 2),
        "cagr_p1_10k_with_compounded_ppf_pct": round(cagr_10k_comp, 2),
        "final_pure_100_ppf_tl": round(ppf_bal, 2),
        "cagr_pure_100_ppf_pct": round(cagr_ppf, 2),
        "gap_20k_vs_pure_ppf_tl": round(fin_20k_comp - ppf_bal, 2),
        "gap_10k_vs_pure_ppf_tl": round(fin_10k_comp - ppf_bal, 2),
    }

    with open(RESULTS_DIR / "economic_compounding_comparison.json", "w", encoding="utf-8") as f:
        json.dump(economic_comp, f, indent=2, ensure_ascii=False)

    log.info("Ekonomik Kıyaslama:\n%s", json.dumps(economic_comp, indent=2))
    log.info("20k Boyut Max DD: %.2f%% | 10k Boyut Max DD: %.2f%%", eq_20k["dd_pct"].min(), eq_10k["dd_pct"].min())


if __name__ == "__main__":
    run_exact_analysis()
