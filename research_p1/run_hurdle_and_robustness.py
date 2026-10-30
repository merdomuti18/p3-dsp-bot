# -*- coding: utf-8 -*-
"""
research_p1/run_hurdle_and_robustness.py
========================================
1. 37.475 TL vs 20.624 TL Farkının Matematiksel Teyidi
2. Ortalama Tutma Süresi ve Ortalama Maruziyet (Günlük Pozisyon Sayısı)
3. İşlem Başına ve Yıllıklandırılmış Getirinin TL Nakit/Repo (TCMB/PPF) ile Karşılaştırması
4. Rastgele Hisse Çıkarma (Monte Carlo 1.000 İterasyon: Rastgele 5 ve 10 hisse)
5. Kıyaslama Portföyleri: 100% PPF, 100% XU100, %60 XU100 + %40 PPF, P1 GTD (Boş Nakit Sıfır Faiz) vs P1 GTD (Boş Nakit PPF)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from research_p1.backtest_engine import (
    simulate_trades,
    precompute_signals_for_agents,
)
from research_p1.data_downloader import load_cached_ohlcv, BIST100_SYMBOLS
from research_p1.p1_engine import compute_all_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FULL_START = "2021-10-06"
FULL_END   = "2026-10-05"

CAPITAL = 100_000.0
POS_SIZE = 10_000.0
MAX_POS = 6
COMM_RATE = 0.0030  # %0.30
SLIP_RATE = 0.0030  # %0.30 -> Tur başı %1.20

# TCMB Politika / Gecelik Repo Yıllık Faiz Oranı Tahmini Tarihsel Serisi (2021-2026)
def get_daily_risk_free_rate(d: pd.Timestamp) -> float:
    """Verilen gün için yıllıklandırılmış risksiz TL faiz oranını (repo/PPF) döndürür."""
    dt_str = str(d.date())
    if dt_str < "2021-11-01":
        ann_rate = 0.18
    elif dt_str < "2022-01-01":
        ann_rate = 0.15
    elif dt_str < "2022-08-01":
        ann_rate = 0.14
    elif dt_str < "2022-11-01":
        ann_rate = 0.12
    elif dt_str < "2023-03-01":
        ann_rate = 0.09
    elif dt_str < "2023-06-01":
        ann_rate = 0.085
    elif dt_str < "2023-07-01":
        ann_rate = 0.15
    elif dt_str < "2023-08-01":
        ann_rate = 0.175
    elif dt_str < "2023-09-01":
        ann_rate = 0.25
    elif dt_str < "2023-10-01":
        ann_rate = 0.30
    elif dt_str < "2023-11-01":
        ann_rate = 0.35
    elif dt_str < "2023-12-01":
        ann_rate = 0.40
    elif dt_str < "2024-03-01":
        ann_rate = 0.45
    elif dt_str < "2025-01-01":
        ann_rate = 0.50
    elif dt_str < "2025-07-01":
        ann_rate = 0.475
    elif dt_str < "2026-01-01":
        ann_rate = 0.425
    else:
        ann_rate = 0.375

    # Günlük bileşik faiz katsayısı
    return ann_rate


def run_analysis():
    log.info("Veriler yükleniyor...")
    symbol_data = {}
    for s in BIST100_SYMBOLS:
        df, meta = load_cached_ohlcv(s)
        if df is not None and len(df) >= 50:
            symbol_data[s] = compute_all_indicators(df)

    precomp_gtd = precompute_signals_for_agents(symbol_data, ["GTD"])

    # ─────────────────────────────────────────────────────────────────────────
    # 1. 37.475 TL vs 20.624 TL Farkının Analizi
    # ─────────────────────────────────────────────────────────────────────────
    t_single, r_single, m_single = simulate_trades(
        symbol_data, ["GTD"], precomputed_signals=precomp_gtd,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001, commission_rate=COMM_RATE, slippage_rate=SLIP_RATE,
        mode="single_agent", position_size=POS_SIZE
    )

    t_port, r_port, m_port = simulate_trades(
        symbol_data, ["GTD"], precomputed_signals=precomp_gtd,
        start_date=FULL_START, end_date=FULL_END,
        gap_tolerance=0.0001, commission_rate=COMM_RATE, slippage_rate=SLIP_RATE,
        mode="portfolio", initial_capital=CAPITAL, max_positions=MAX_POS, position_size=POS_SIZE
    )

    df_single = pd.DataFrame(t_single)
    df_port = pd.DataFrame(t_port)

    # Portföyde alınan işlemler ile tekil modda alınan işlemlerin küme farkı
    set_single_ids = set((t["symbol"], t["entry_date"]) for t in t_single)
    set_port_ids = set((t["symbol"], t["entry_date"]) for t in t_port)

    skipped_ids = set_single_ids - set_port_ids
    df_skipped = df_single[df_single.apply(lambda r: (r["symbol"], r["entry_date"]) in skipped_ids, axis=1)]

    skipped_count = len(df_skipped)
    skipped_pnl = float(df_skipped["net_profit_tl"].sum())
    skipped_mean_ret = float(df_skipped["net_return_pct"].mean())

    diff_explanation = {
        "single_agent_trades_count": len(t_single),
        "single_agent_total_pnl_tl": round(m_single["total_profit_tl"], 2),
        "portfolio_trades_taken_count": len(t_port),
        "portfolio_total_pnl_tl": round(m_port["total_profit_tl"], 2),
        "capacity_skipped_trades_count": skipped_count,
        "capacity_skipped_trades_pnl_tl": round(skipped_pnl, 2),
        "capacity_skipped_trades_mean_ret_pct": round(skipped_mean_ret, 2),
        "mathematical_reconciliation_check": round(m_port["total_profit_tl"] + skipped_pnl, 2) == round(m_single["total_profit_tl"], 2),
    }

    log.info("Fark Analizi: %s", json.dumps(diff_explanation, indent=2))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Ortalama Tutma Süresi ve Ortalama Maruziyet
    # ─────────────────────────────────────────────────────────────────────────
    holding_calendar_days = df_port["holding_days"].mean()
    # İşlem günleri bazında tutma süresi hesabı
    # Her trade için entry_date ile exit_date arasındaki işlem günü sayısı
    trading_days_held = []
    for _, row in df_port.iterrows():
        s_df = symbol_data.get(row["symbol"])
        if s_df is not None:
            sub = s_df.loc[row["entry_date"]:row["exit_date"]]
            trading_days_held.append(max(1, len(sub) - 1))
        else:
            trading_days_held.append(row["holding_days"])

    mean_trading_days = float(np.mean(trading_days_held))

    # Günlük equity serisinden ortalama açık pozisyon sayısı ve sermaye maruziyeti
    df_eq = pd.DataFrame(m_port["equity_series"])
    df_eq["date_dt"] = pd.to_datetime(df_eq["date"])
    mean_daily_positions = float(df_eq["open_positions"].mean())
    mean_daily_exposure_pct = float(mean_daily_positions * 10_000.0 / CAPITAL * 100.0)
    pct_days_with_zero_positions = float((df_eq["open_positions"] == 0).mean() * 100.0)

    exposure_summary = {
        "mean_holding_calendar_days": round(float(holding_calendar_days), 2),
        "mean_holding_trading_days": round(mean_trading_days, 2),
        "mean_daily_active_positions": round(mean_daily_positions, 2),
        "mean_capital_exposure_pct": round(mean_daily_exposure_pct, 2),
        "pct_days_idle_zero_positions": round(pct_days_with_zero_positions, 2),
    }
    log.info("Maruziyet Özeti: %s", json.dumps(exposure_summary, indent=2))

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TL Nakit / Repo (TCMB / PPF) Eşiği Kıyaslaması
    # ─────────────────────────────────────────────────────────────────────────
    # A) Trade bazında risksiz faiz kıyaslaması:
    # Her trade'in tutulduğu süre boyunca TL gecelik faiz getirisi ne kadardı?
    trade_cash_comparisons = []
    for _, row in df_port.iterrows():
        entry_d = pd.to_datetime(row["entry_date"])
        exit_d = pd.to_datetime(row["exit_date"])
        days = max(1, (exit_d - entry_d).days)

        ann_rate = get_daily_risk_free_rate(entry_d)
        rf_return_pct = ((1.0 + ann_rate) ** (days / 365.0) - 1.0) * 100.0
        trade_net_ret = row["net_return_pct"]
        excess_ret = trade_net_ret - rf_return_pct

        trade_cash_comparisons.append({
            "symbol": row["symbol"],
            "holding_days": days,
            "annualized_rf_rate": ann_rate,
            "rf_return_pct": rf_return_pct,
            "trade_net_ret_pct": trade_net_ret,
            "excess_return_over_cash_pct": excess_ret,
            "beat_cash": trade_net_ret > rf_return_pct,
        })

    df_tcc = pd.DataFrame(trade_cash_comparisons)
    pct_trades_beat_cash = float(df_tcc["beat_cash"].mean() * 100.0)
    mean_rf_ret_per_trade = float(df_tcc["rf_return_pct"].mean())
    mean_excess_ret_per_trade = float(df_tcc["excess_return_over_cash_pct"].mean())

    # Yıllıklandırılmış Return on Capital Employed (ROCE)
    # İşlem başı ortalama net getiri = +%0.75, Ortalama tutma süresi = 6.4 takvim günü
    # Yıllıklandırılmış işlem getirisi = (1 + 0.0075) ^ (365 / 6.4) - 1
    annualized_trade_return_pct = ((1.0 + 0.00752) ** (365.0 / float(holding_calendar_days)) - 1.0) * 100.0
    mean_rf_annual_rate_5y = float(np.mean([get_daily_risk_free_rate(d) for d in df_eq["date_dt"]])) * 100.0

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Kıyaslama Portföyleri (Compound Edilmiş 5 Yıllık Getiriler)
    # ─────────────────────────────────────────────────────────────────────────
    # Günlük bazda faiz işletimi:
    # Benchmark 1: 100% PPF / Repo (100.000 TL her gün gecelik bileşir)
    ppf_equity = [CAPITAL]
    for i in range(1, len(df_eq)):
        d = df_eq["date_dt"].iloc[i]
        prev_d = df_eq["date_dt"].iloc[i - 1]
        days_diff = (d - prev_d).days
        ann_r = get_daily_risk_free_rate(d)
        factor = (1.0 + ann_r) ** (days_diff / 365.0)
        ppf_equity.append(ppf_equity[-1] * factor)
    df_eq["bench_100_ppf"] = ppf_equity

    # Benchmark 2: 100% XU100
    bench_dir = Path(__file__).parent / "data" / "benchmarks"
    df_xu = pd.read_csv(bench_dir / "XU100.IS.csv", index_col=0)
    df_xu.index = pd.to_datetime(df_xu.index)
    df_xu_sub = df_xu.reindex(df_eq["date_dt"]).ffill()
    xu_start_p = df_xu_sub["close"].iloc[0]
    df_eq["bench_100_xu100"] = CAPITAL * (df_xu_sub["close"].values / xu_start_p)

    # Benchmark 3: %60 XU100 + %40 PPF
    df_eq["bench_60_xu_40_ppf"] = 0.60 * df_eq["bench_100_xu100"] + 0.40 * df_eq["bench_100_ppf"]

    # P1 GTD (Boş Nakit PPF ile Değerlenirse):
    # P1 portföyünde nakit her gün gecelik faiz kazanır
    p1_active_cash_ppf = [CAPITAL]
    # Simülasyonu yeniden kurup günlük boştaki nakde faiz işletelim
    port_cash = CAPITAL
    port_pos = {}
    port_hist = []
    p1_with_ppf_series = []

    # Map daily events from t_port
    # Let's compute directly from df_eq: each day cash gets interest, equity grows
    equity_with_cash_yield = []
    running_eq = CAPITAL
    for i in range(len(df_eq)):
        if i == 0:
            equity_with_cash_yield.append(CAPITAL)
            continue
        d = df_eq["date_dt"].iloc[i]
        prev_d = df_eq["date_dt"].iloc[i - 1]
        days_diff = (d - prev_d).days
        ann_r = get_daily_risk_free_rate(d)
        rf_factor = (1.0 + ann_r) ** (days_diff / 365.0) - 1.0

        # Boştaki nakit = df_eq['cash'].iloc[i-1]
        idle_cash = df_eq["cash"].iloc[i - 1]
        cash_interest = idle_cash * rf_factor

        # Günlük P1 equity değişimi + nakit faizi
        p1_delta = df_eq["equity"].iloc[i] - df_eq["equity"].iloc[i - 1]
        running_eq = running_eq + p1_delta + cash_interest
        equity_with_cash_yield.append(running_eq)

    df_eq["p1_gtd_with_ppf"] = equity_with_cash_yield

    bench_metrics = {
        "final_p1_gtd_zero_cash_tl": round(df_eq["equity"].iloc[-1], 2),
        "cagr_p1_gtd_zero_cash_pct": round(((df_eq["equity"].iloc[-1] / CAPITAL) ** 0.2 - 1.0) * 100, 2),
        "final_p1_gtd_with_ppf_tl": round(df_eq["p1_gtd_with_ppf"].iloc[-1], 2),
        "cagr_p1_gtd_with_ppf_pct": round(((df_eq["p1_gtd_with_ppf"].iloc[-1] / CAPITAL) ** 0.2 - 1.0) * 100, 2),
        "final_100_ppf_tl": round(df_eq["bench_100_ppf"].iloc[-1], 2),
        "cagr_100_ppf_pct": round(((df_eq["bench_100_ppf"].iloc[-1] / CAPITAL) ** 0.2 - 1.0) * 100, 2),
        "final_60_xu_40_ppf_tl": round(df_eq["bench_60_xu_40_ppf"].iloc[-1], 2),
        "cagr_60_xu_40_ppf_pct": round(((df_eq["bench_60_xu_40_ppf"].iloc[-1] / CAPITAL) ** 0.2 - 1.0) * 100, 2),
        "final_100_xu100_tl": round(df_eq["bench_100_xu100"].iloc[-1], 2),
        "cagr_100_xu100_pct": round(((df_eq["bench_100_xu100"].iloc[-1] / CAPITAL) ** 0.2 - 1.0) * 100, 2),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Rastgele Hisse Çıkarma Dağılımı (Monte Carlo 1.000 İterasyon)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- Rastgele Hisse Çıkarma Monte Carlo (1.000 İterasyon) ---")
    unique_symbols = np.array(sorted(df_single["symbol"].unique()))
    n_syms = len(unique_symbols)
    rng = np.random.default_rng(42)

    # Pre-group trades by symbol for speed
    sym_returns = {s: df_single[df_single["symbol"] == s]["net_return_pct"].values for s in unique_symbols}
    sym_pnls = {s: df_single[df_single["symbol"] == s]["net_profit_tl"].values for s in unique_symbols}

    mc_drop_5_means = []
    mc_drop_5_pnls = []
    mc_drop_10_means = []
    mc_drop_10_pnls = []

    for _ in range(1000):
        # Drop 5 random symbols
        drop5 = set(rng.choice(unique_symbols, size=5, replace=False))
        ret5 = [r for s in unique_symbols if s not in drop5 for r in sym_returns[s]]
        pnl5 = [p for s in unique_symbols if s not in drop5 for p in sym_pnls[s]]
        mc_drop_5_means.append(np.mean(ret5))
        mc_drop_5_pnls.append(np.sum(pnl5))

        # Drop 10 random symbols
        drop10 = set(rng.choice(unique_symbols, size=10, replace=False))
        ret10 = [r for s in unique_symbols if s not in drop10 for r in sym_returns[s]]
        pnl10 = [p for s in unique_symbols if s not in drop10 for p in sym_pnls[s]]
        mc_drop_10_means.append(np.mean(ret10))
        mc_drop_10_pnls.append(np.sum(pnl10))

    mc_summary = {
        "drop_5_random_mean_return_p05": round(float(np.percentile(mc_drop_5_means, 5)), 4),
        "drop_5_random_mean_return_median": round(float(np.median(mc_drop_5_means)), 4),
        "drop_5_random_mean_return_p95": round(float(np.percentile(mc_drop_5_means, 95)), 4),
        "drop_5_pct_runs_positive": round(float((np.array(mc_drop_5_means) > 0).mean() * 100), 2),
        "drop_5_pnl_median_tl": round(float(np.median(mc_drop_5_pnls)), 2),

        "drop_10_random_mean_return_p05": round(float(np.percentile(mc_drop_10_means, 5)), 4),
        "drop_10_random_mean_return_median": round(float(np.median(mc_drop_10_means)), 4),
        "drop_10_random_mean_return_p95": round(float(np.percentile(mc_drop_10_means, 95)), 4),
        "drop_10_pct_runs_positive": round(float((np.array(mc_drop_10_means) > 0).mean() * 100), 2),
        "drop_10_pnl_median_tl": round(float(np.median(mc_drop_10_pnls)), 2),
    }

    # Consolidated JSON output
    final_output = {
        "diff_explanation": diff_explanation,
        "exposure_summary": exposure_summary,
        "trade_vs_risk_free_cash": {
            "mean_trade_net_return_pct": round(float(df_port["net_return_pct"].mean()), 4),
            "mean_risk_free_return_during_holding_pct": round(mean_rf_ret_per_trade, 4),
            "mean_excess_return_over_cash_pct": round(mean_excess_ret_per_trade, 4),
            "pct_trades_beating_cash_during_hold": round(pct_trades_beat_cash, 2),
            "annualized_trade_return_on_active_capital_pct": round(annualized_trade_return_pct, 2),
            "average_annual_risk_free_rate_5y_pct": round(mean_rf_annual_rate_5y, 2),
            "excess_annualized_edge_on_active_capital_pct": round(annualized_trade_return_pct - mean_rf_annual_rate_5y, 2),
        },
        "benchmark_comparisons": bench_metrics,
        "monte_carlo_random_drops": mc_summary,
    }

    with open(RESULTS_DIR / "stage0_hurdle_analysis.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    df_eq.to_csv(RESULTS_DIR / "stage0_equity_vs_benchmarks.csv", index=False)

    print("\n" + "=" * 60)
    print("HURDLE & ROBUSTNESS RAPORU TAMAMLANDI")
    print("=" * 60)
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_analysis()
