# -*- coding: utf-8 -*-
"""
research_p1/run_p1_revision_backtest.py
======================================
P1 Revizyon Birleşik Backtest & Hipotez Doğrulama:
1. ALPHA_B Sadeleştirme Testi (RSI ve ADX birlikte kaldırılınca ne oluyor?)
2. GT (RSI'sız) Hipotez Testi: GT_NO_RSI gerçekten GTD bölgesini mi yakalıyor?
3. 3 Ajanlı Birleşik Portföy Simülasyonu (GTD > ALPHA_B > ZT3, RelVol öncelikli):
   - 100.000 TL, 5 slot
   - 20.000 TL vs 10.000 TL pozisyon boyutu karşılaştırması (Günlük MTM Drawdown)
   - %0.80 vs %1.20 tur başı maliyet
   - Ajan bazlı ayrı kırılım (GTD kirlenmeden net ayrışıyor mu?)
   - Kapasite reddi analizi (Reddedilenlerin performansı)
   - XU100 > EMA200 rejim filtresi A/B incelemesi
4. Paper Trading Kayıt Şablonu Üretimi
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
from research_p1.p1_exit_rules import (
    GAP_TOLERANCE,
    MAX_HOLDING_DAYS,
    gap_ratio,
    gap_rejected,
    stop_level_price,
    tp1_level_price,
    tp1_sold_lots,
    trail_level_price,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FULL_START = "2021-10-06"
FULL_END   = "2026-10-05"


def evaluate_custom_agent_signals(ind: pd.DataFrame, agent_name: str) -> pd.Series:
    c = ind["close"]
    rsi = ind["rsi"]
    adx = ind["adx"]
    di_p = ind["di_p"]
    di_n = ind["di_n"]
    cmf = ind["cmf"]
    rel_vol = ind["rel_vol"]
    stochrsi = ind["stochrsi"]
    ema8 = ind["ema8"]
    ema21 = ind["ema21"]
    ema50 = ind["ema50"]
    ema200 = ind["ema200"]
    macd = ind["macd"]
    macd_sig = ind["macd_sig"]
    macd_prev = ind["macd_prev"]
    macd_sprev = ind["macd_sprev"]
    bb_mid = ind["bb_mid"]
    at_bull = ind["at_alpha_trend"] > ind["at_alpha_trend_2"]

    if agent_name == "GTD":
        return (
            (ema8 > ema21) & (ema21 > ema50) &
            (ema200.isna() | (c > ema200)) &
            (rsi >= 65) & (rsi <= 78) &
            (adx > 25) &
            (di_p > di_n) &
            (cmf > 0.05) &
            (rel_vol >= 1.2)
        )
    elif agent_name == "ALPHA_B":
        # Mevcut ALPHA_B: AlphaTrend + RSI>45 + ADX>18 + CMF>-0.05 + RelVol>=1.0
        return (
            ind["at_buy_confirmed"].fillna(False) &
            (rsi > 45) &
            (adx > 18) &
            (cmf > -0.05) &
            (rel_vol >= 1.0)
        )
    elif agent_name == "ALPHA_B_SIMPLE":
        # Sadeleştirilmiş ALPHA_B: RSI ve ADX BİRLİKTE kaldırıldı
        # AlphaTrend + CMF>-0.05 + RelVol>=1.0
        return (
            ind["at_buy_confirmed"].fillna(False) &
            (cmf > -0.05) &
            (rel_vol >= 1.0)
        )
    elif agent_name == "ZT3":
        return (
            (adx > 25) &
            (di_p > di_n) &
            (rsi > 50) &
            (macd > macd_sig) &
            (macd_prev <= macd_sprev) &
            at_bull &
            (rel_vol >= 1.0)
        )
    elif agent_name == "GT":
        return (
            (ema8 > ema21) & (ema21 > ema50) &
            (ema200.isna() | (c > ema200)) &
            (rsi >= 50) & (rsi <= 65) &
            (adx > 20) &
            (di_p > di_n) &
            (cmf > 0) &
            (rel_vol >= 1.0)
        )
    elif agent_name == "GT_NO_RSI":
        # GT RSI filtresiz
        return (
            (ema8 > ema21) & (ema21 > ema50) &
            (ema200.isna() | (c > ema200)) &
            (adx > 20) &
            (di_p > di_n) &
            (cmf > 0) &
            (rel_vol >= 1.0)
        )
    return pd.Series(False, index=ind.index)


def simulate_priority_portfolio(
    symbol_data: Dict[str, pd.DataFrame],
    agent_priority: List[str],  # Örn: ["GTD", "ALPHA_B", "ZT3"]
    precomputed: Dict[str, Dict[str, pd.Series]],
    start_date: str = FULL_START,
    end_date: str = FULL_END,
    gap_tolerance: float = GAP_TOLERANCE,
    commission_rate: float = 0.0030,  # %0.30
    slippage_rate: float = 0.0030,    # %0.30
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
    position_size: float = 20_000.0,
    xu100_regime_series: Optional[pd.Series] = None,
):
    all_dates = sorted(list(set().union(*[df.index for df in symbol_data.values()])))
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    active_positions: Dict[str, dict] = {}
    completed_trades: List[dict] = []
    rejected_signals: List[dict] = []
    capacity_skipped_signals: List[dict] = []

    cash = initial_capital
    portfolio_equity_series = []

    priority_rank = {ag: i for i, ag in enumerate(agent_priority)}

    for t_idx, current_date in enumerate(all_dates):
        # 1. Pozisyon çıkış kontrolleri (STOP, TP1, TRAILING, MAX_GUN)
        closed_syms = []
        for sym, pos in active_positions.items():
            df = symbol_data.get(sym)
            if df is None or current_date not in df.index:
                continue

            row = df.loc[current_date]
            o_bar, h_bar, l_bar, c_bar = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            pos["mfe"] = max(pos["mfe"], (h_bar - pos["entry_price"]) / pos["entry_price"])
            pos["mae"] = min(pos["mae"], (l_bar - pos["entry_price"]) / pos["entry_price"])

            holding_days = (current_date - pos["entry_date"]).days
            stop_level = stop_level_price(pos["entry_price"])
            tp1_level = tp1_level_price(pos["entry_price"])
            trail_level = trail_level_price(pos["peak_price"]) if pos["tp1_done"] else 0.0

            is_stop = l_bar <= stop_level
            is_tp1 = (h_bar >= tp1_level) and not pos["tp1_done"]
            is_trailing = pos["tp1_done"] and (l_bar <= trail_level)
            is_max_gun = holding_days >= MAX_HOLDING_DAYS

            # A. Çift Tetik
            if is_stop and is_tp1:
                ref_exit = o_bar if o_bar <= stop_level else stop_level
                exit_price = round(ref_exit * (1.0 - slippage_rate), 4)
                comm = round(pos["lots"] * exit_price * commission_rate, 4)
                net_income = round(pos["lots"] * exit_price - comm, 4)
                pos["exits"].append({"lots": pos["lots"], "price": exit_price, "reason": "STOP_AMBIGUOUS"})
                pos["net_income"] += net_income
                pos["comm"] += comm
                pos["exit_date"] = current_date
                pos["exit_price"] = exit_price
                pos["exit_reason"] = "STOP_AMBIGUOUS"
                cash += net_income
                closed_syms.append(sym)
                continue

            # B. Stop
            if is_stop:
                ref_exit = o_bar if o_bar <= stop_level else stop_level
                exit_price = round(ref_exit * (1.0 - slippage_rate), 4)
                comm = round(pos["lots"] * exit_price * commission_rate, 4)
                net_income = round(pos["lots"] * exit_price - comm, 4)
                pos["exits"].append({"lots": pos["lots"], "price": exit_price, "reason": "STOP"})
                pos["net_income"] += net_income
                pos["comm"] += comm
                pos["exit_date"] = current_date
                pos["exit_price"] = exit_price
                pos["exit_reason"] = "STOP"
                cash += net_income
                closed_syms.append(sym)
                continue

            # C. TP1
            if is_tp1:
                ref_exit = max(o_bar, tp1_level)
                exit_price = round(ref_exit * (1.0 - slippage_rate), 4)
                sold_lots = tp1_sold_lots(pos["lots"])
                pos["lots"] -= sold_lots
                comm = round(sold_lots * exit_price * commission_rate, 4)
                net_income = round(sold_lots * exit_price - comm, 4)
                pos["exits"].append({"lots": sold_lots, "price": exit_price, "reason": "TP1"})
                pos["net_income"] += net_income
                pos["comm"] += comm
                pos["tp1_done"] = True
                cash += net_income
                if pos["lots"] == 0:
                    pos["exit_date"] = current_date
                    pos["exit_price"] = exit_price
                    pos["exit_reason"] = "TP1_FULL"
                    closed_syms.append(sym)
                    continue

            # D. Trailing
            elif is_trailing:
                ref_exit = o_bar if o_bar <= trail_level else trail_level
                exit_price = round(ref_exit * (1.0 - slippage_rate), 4)
                comm = round(pos["lots"] * exit_price * commission_rate, 4)
                net_income = round(pos["lots"] * exit_price - comm, 4)
                pos["exits"].append({"lots": pos["lots"], "price": exit_price, "reason": "TRAILING"})
                pos["net_income"] += net_income
                pos["comm"] += comm
                pos["exit_date"] = current_date
                pos["exit_price"] = exit_price
                pos["exit_reason"] = "TRAILING"
                cash += net_income
                closed_syms.append(sym)
                continue

            # E. Max Gun
            if is_max_gun:
                exit_price = round(c_bar * (1.0 - slippage_rate), 4)
                comm = round(pos["lots"] * exit_price * commission_rate, 4)
                net_income = round(pos["lots"] * exit_price - comm, 4)
                pos["exits"].append({"lots": pos["lots"], "price": exit_price, "reason": "MAX_GUN"})
                pos["net_income"] += net_income
                pos["comm"] += comm
                pos["exit_date"] = current_date
                pos["exit_price"] = exit_price
                pos["exit_reason"] = "MAX_GUN"
                cash += net_income
                closed_syms.append(sym)
                continue

            pos["peak_price"] = max(pos["peak_price"], h_bar)

        for sym in closed_syms:
            p_closed = active_positions.pop(sym)
            net_ret_pct = (p_closed["net_income"] - p_closed["total_cost"]) / p_closed["total_cost"] * 100.0
            pnl_tl = p_closed["net_income"] - p_closed["total_cost"]
            completed_trades.append({
                "pos_id": p_closed["pos_id"],
                "symbol": sym,
                "primary_agent": p_closed["primary_agent"],
                "all_agents": ",".join(p_closed["agents"]),
                "entry_date": str(p_closed["entry_date"].date()),
                "exit_date": str(p_closed["exit_date"].date()),
                "entry_price": p_closed["entry_price"],
                "exit_price": p_closed["exit_price"],
                "initial_lots": p_closed["initial_lots"],
                "total_cost": round(p_closed["total_cost"], 2),
                "net_sales_income": round(p_closed["net_income"], 2),
                "exit_reason": p_closed["exit_reason"],
                "net_return_pct": round(net_ret_pct, 4),
                "net_profit_tl": round(pnl_tl, 2),
                "holding_days": (p_closed["exit_date"] - p_closed["entry_date"]).days,
                "mfe_pct": round(p_closed["mfe"] * 100, 2),
                "mae_pct": round(p_closed["mae"] * 100, 2),
                "regime_bull": p_closed.get("regime_bull", True),
                "rel_vol": round(p_closed.get("rel_vol", 1.0), 2),
            })

        # Günlük MTM Değerleme
        if start_ts <= current_date <= end_ts:
            open_val = sum(
                p["lots"] * float(symbol_data[s].loc[current_date, "close"])
                for s, p in active_positions.items()
                if s in symbol_data and current_date in symbol_data[s].index
            )
            eq = cash + open_val
            portfolio_equity_series.append({
                "date": str(current_date.date()),
                "cash": round(cash, 2),
                "open_value": round(open_val, 2),
                "equity": round(eq, 2),
                "open_positions": len(active_positions),
            })

        # 2. t gününün kapanış sinyallerinin t+1 açılışında alımı
        if t_idx == 0 or current_date < start_ts or current_date > end_ts:
            continue

        prev_date = all_dates[t_idx - 1]
        regime_bull = True
        if xu100_regime_series is not None and prev_date in xu100_regime_series.index:
            regime_bull = bool(xu100_regime_series.loc[prev_date])

        candidates = []
        for sym, df in symbol_data.items():
            if prev_date not in df.index or current_date not in df.index:
                continue

            trig = [ag for ag in agent_priority if precomputed[sym][ag].loc[prev_date]]
            if not trig:
                continue

            prev_c = float(df.loc[prev_date, "close"])
            curr_o = float(df.loc[current_date, "open"])
            gap = gap_ratio(curr_o, prev_c)

            if gap_rejected(curr_o, prev_c, gap_tolerance):
                rejected_signals.append({
                    "symbol": sym, "signal_date": str(prev_date.date()), "entry_date": str(current_date.date()),
                    "gap": gap, "agents": trig, "reason": "gap_rejection"
                })
                continue

            # Öncelik belirleme: En yüksek öncelikli ajan (en küçük rank)
            best_rank = min(priority_rank[ag] for ag in trig)
            primary_ag = [ag for ag in trig if priority_rank[ag] == best_rank][0]
            rel_vol = float(df.loc[prev_date, "rel_vol"]) if "rel_vol" in df.columns else 1.0

            candidates.append({
                "symbol": sym, "prev_date": prev_date, "prev_close": prev_c,
                "curr_open": curr_o, "gap": gap, "agents": trig,
                "primary_agent": primary_ag, "priority_rank": best_rank,
                "rel_vol": rel_vol, "regime_bull": regime_bull
            })

        # Slot Kuralı Sıralaması:
        # 1. Ajan Önceliği (GTD=0 > ALPHA_B=1 > ZT3=2)
        # 2. RelVol (Büyükten küçüğe)
        # 3. Sembol (Deterministik eşitlik bozucu)
        candidates.sort(key=lambda x: (x["priority_rank"], -x["rel_vol"], x["symbol"]))

        for cand in candidates:
            sym = cand["symbol"]
            if sym in active_positions:
                continue

            # Slot doluluğu kontrolü
            if len(active_positions) >= max_positions:
                capacity_skipped_signals.append({
                    "symbol": sym, "signal_date": str(cand["prev_date"].date()),
                    "entry_date": str(current_date.date()), "primary_agent": cand["primary_agent"],
                    "all_agents": cand["agents"], "rel_vol": cand["rel_vol"], "reason": "capacity_full_5slots"
                })
                continue

            alloc = min(position_size, cash / max(1, max_positions - len(active_positions)))
            fill_p = round(cand["curr_open"] * (1.0 + slippage_rate), 4)
            lots = int(alloc / max(fill_p, 0.01))
            if lots < 1:
                continue

            comm = round(lots * fill_p * commission_rate, 4)
            tot_cost = round(lots * fill_p + comm, 4)
            if tot_cost > cash:
                continue

            cash -= tot_cost
            active_positions[sym] = {
                "pos_id": f"P1_{sym}_{current_date.strftime('%Y%m%d')}_{len(completed_trades)+len(active_positions)}",
                "symbol": sym,
                "entry_date": current_date,
                "entry_price": fill_p,
                "lots": lots,
                "initial_lots": lots,
                "total_cost": tot_cost,
                "net_income": 0.0,
                "comm": comm,
                "peak_price": fill_p,
                "tp1_done": False,
                "primary_agent": cand["primary_agent"],
                "agents": cand["agents"],
                "rel_vol": cand["rel_vol"],
                "regime_bull": cand["regime_bull"],
                "mfe": 0.0,
                "mae": 0.0,
                "exits": [],
            }

    # Periyot sonu açık kalanları kapat
    for sym, pos in list(active_positions.items()):
        df = symbol_data.get(sym)
        last_dt = all_dates[-1]
        c_bar = float(df.loc[last_dt, "close"]) if (df is not None and last_dt in df.index) else pos["entry_price"]
        exit_p = round(c_bar * (1.0 - slippage_rate), 4)
        comm = round(pos["lots"] * exit_p * commission_rate, 4)
        net_inc = round(pos["lots"] * exit_p - comm, 4)
        pos["net_income"] += net_inc
        pos["comm"] += comm
        pos["exit_date"] = last_dt
        pos["exit_price"] = exit_p
        pos["exit_reason"] = "PERIOD_END"
        net_ret = (pos["net_income"] - pos["total_cost"]) / pos["total_cost"] * 100.0
        pnl = pos["net_income"] - pos["total_cost"]
        completed_trades.append({
            "pos_id": pos["pos_id"], "symbol": sym, "primary_agent": pos["primary_agent"],
            "all_agents": ",".join(pos["agents"]), "entry_date": str(pos["entry_date"].date()),
            "exit_date": str(last_dt.date()), "entry_price": pos["entry_price"],
            "exit_price": exit_p, "initial_lots": pos["initial_lots"],
            "total_cost": round(pos["total_cost"], 2), "net_sales_income": round(pos["net_income"], 2),
            "exit_reason": "PERIOD_END", "net_return_pct": round(net_ret, 4),
            "net_profit_tl": round(pnl, 2), "holding_days": (last_dt - pos["entry_date"]).days,
            "mfe_pct": round(pos["mfe"] * 100, 2), "mae_pct": round(pos["mae"] * 100, 2),
            "regime_bull": pos.get("regime_bull", True), "rel_vol": round(pos.get("rel_vol", 1.0), 2),
        })

    # Metrikler
    df_trades = pd.DataFrame(completed_trades)
    df_eq = pd.DataFrame(portfolio_equity_series)
    df_eq["peak"] = df_eq["equity"].cummax()
    df_eq["dd_pct"] = (df_eq["equity"] - df_eq["peak"]) / df_eq["peak"] * 100.0

    return df_trades, df_eq, pd.DataFrame(capacity_skipped_signals)


def run_full_p1_revision():
    log.info("Veriler yükleniyor...")
    symbol_data = {}
    for s in BIST100_SYMBOLS:
        df, meta = load_cached_ohlcv(s)
        if df is not None and len(df) >= 50:
            symbol_data[s] = compute_all_indicators(df)

    log.info("XU100 Rejim Serisi (EMA200) hazırlanıyor...")
    bench_dir = Path(__file__).parent / "data" / "benchmarks"
    df_xu = pd.read_csv(bench_dir / "XU100.IS.csv", index_col=0)
    df_xu.index = pd.to_datetime(df_xu.index)
    df_xu["ema200"] = df_xu["close"].ewm(span=200, adjust=False).mean()
    xu100_bull_regime = df_xu["close"] > df_xu["ema200"]

    # Sinyal Ön-Hesaplama
    agents_needed = ["GTD", "ALPHA_B", "ALPHA_B_SIMPLE", "ZT3", "GT", "GT_NO_RSI"]
    precomputed = {}
    for sym, df in symbol_data.items():
        precomputed[sym] = {ag: evaluate_custom_agent_signals(df, ag) for ag in agents_needed}

    # ─────────────────────────────────────────────────────────────────────────
    # 1. HİPOTEZ 1: ALPHA_B SADELEŞTİRME (RSI ve ADX Birlikte Kaldırılınca)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 1. ALPHA_B Sadeleştirme Testi (RSI & ADX Birlikte Kaldırıldı) ---")
    # Tekil modda karşılaştıralım (%1.2 maliyetle)
    from research_p1.backtest_engine import simulate_trades as sim_base
    t_ab_curr, _, m_ab_curr = sim_base(
        symbol_data, ["ALPHA_B"], precomputed_signals=precomputed,
        gap_tolerance=0.0001, commission_rate=0.003, slippage_rate=0.003, mode="single_agent"
    )
    t_ab_simp, _, m_ab_simp = sim_base(
        symbol_data, ["ALPHA_B_SIMPLE"], precomputed_signals=precomputed,
        gap_tolerance=0.0001, commission_rate=0.003, slippage_rate=0.003, mode="single_agent"
    )
    alpha_b_comp = pd.DataFrame([
        {"label": "ALPHA_B (Mevcut: RSI>45 + ADX>18 + CMF + RelVol)", "n": m_ab_curr["n_trades"], "win": m_ab_curr["win_rate"], "mean": m_ab_curr["mean_return_pct"], "pf": m_ab_curr["profit_factor"], "total_tl": m_ab_curr["total_profit_tl"]},
        {"label": "ALPHA_B_SIMPLE (Sade: CMF + RelVol; RSI & ADX Yok)", "n": m_ab_simp["n_trades"], "win": m_ab_simp["win_rate"], "mean": m_ab_simp["mean_return_pct"], "pf": m_ab_simp["profit_factor"], "total_tl": m_ab_simp["total_profit_tl"]},
    ])
    alpha_b_comp.to_csv(RESULTS_DIR / "alpha_b_simplification_comparison.csv", index=False)
    log.info("ALPHA_B Sadeleştirme:\n%s", alpha_b_comp.to_string())

    # ─────────────────────────────────────────────────────────────────────────
    # 2. HİPOTEZ 2: GT (RSI'sız) GTD BÖLGESİNİ Mİ YAKALIYOR?
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 2. GT (RSI'sız) Hipotez Analizi ---")
    # GT_NO_RSI sinyallerinin RSI dağılımı
    gt_no_rsi_signals = []
    for sym, df in symbol_data.items():
        sig = precomputed[sym]["GT_NO_RSI"].loc[FULL_START:FULL_END]
        for dt, is_on in sig.items():
            if is_on:
                rsi_val = float(df.loc[dt, "rsi"])
                gt_no_rsi_signals.append({"symbol": sym, "date": dt, "rsi": rsi_val})

    df_gt_sig = pd.DataFrame(gt_no_rsi_signals)
    pct_in_gtd_zone = float(((df_gt_sig["rsi"] >= 65) & (df_gt_sig["rsi"] <= 78)).mean() * 100)
    pct_under_65 = float((df_gt_sig["rsi"] < 65).mean() * 100)
    pct_over_78 = float((df_gt_sig["rsi"] > 78).mean() * 100)
    log.info("GT_NO_RSI Toplam Sinyal: %d", len(df_gt_sig))
    log.info("  RSI < 65 (Eski GT bölgesi): %.1f%%", pct_under_65)
    log.info("  RSI 65-78 (GTD bölgesi): %.1f%%", pct_in_gtd_zone)
    log.info("  RSI > 78 (Aşırı momentum): %.1f%%", pct_over_78)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. BİRLEŞİK 3-AJAN PORTFÖY TESTİ (GTD > ALPHA_B > ZT3, 5 Slot, 100k TL)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("--- 3. Birleşik 3-Ajan Portföy Testleri (5 Slot, Öncelik Kuralı) ---")
    core_3 = ["GTD", "ALPHA_B", "ZT3"]

    scenarios = [
        # (Pos Size, Cost Rate, Label)
        (20_000.0, 0.0030, "S1_20kBoyut_Maliyet1.2pct"),
        (10_000.0, 0.0030, "S2_10kBoyut_Maliyet1.2pct"),
        (20_000.0, 0.0020, "S3_20kBoyut_Maliyet0.8pct"),
        (10_000.0, 0.0020, "S4_10kBoyut_Maliyet0.8pct"),
    ]

    scenario_summaries = []

    for p_size, cost, s_name in scenarios:
        trades_df, eq_df, skipped_df = simulate_priority_portfolio(
            symbol_data, core_3, precomputed,
            start_date=FULL_START, end_date=FULL_END,
            gap_tolerance=0.0001,
            commission_rate=cost, slippage_rate=cost,
            initial_capital=100_000.0, max_positions=5,
            position_size=p_size,
            xu100_regime_series=xu100_bull_regime
        )

        trades_df.to_csv(RESULTS_DIR / f"p1_rev_{s_name}_trades.csv", index=False)
        eq_df.to_csv(RESULTS_DIR / f"p1_rev_{s_name}_daily_equity.csv", index=False)
        skipped_df.to_csv(RESULTS_DIR / f"p1_rev_{s_name}_capacity_skipped.csv", index=False)

        fin_eq = float(eq_df["equity"].iloc[-1])
        tot_ret = (fin_eq / 100_000.0 - 1.0) * 100.0
        cagr = ((fin_eq / 100_000.0) ** 0.2 - 1.0) * 100.0
        max_dd = float(eq_df["dd_pct"].min())
        avg_pos = float(eq_df["open_positions"].mean())

        # Ajan bazlı kırılım
        by_agent = trades_df.groupby("primary_agent").agg(
            n=("net_return_pct", "count"),
            win=("net_return_pct", lambda s: round((s > 0).mean() * 100, 2)),
            mean_ret=("net_return_pct", "mean"),
            pnl_tl=("net_profit_tl", "sum")
        ).reset_index()

        # Rejim kırılımı (XU100 > EMA200 boğa vs ayı)
        by_regime = trades_df.groupby("regime_bull").agg(
            n=("net_return_pct", "count"),
            win=("net_return_pct", lambda s: round((s > 0).mean() * 100, 2)),
            mean_ret=("net_return_pct", "mean"),
            pnl_tl=("net_profit_tl", "sum")
        ).reset_index()

        summary_item = {
            "scenario": s_name,
            "position_size": p_size,
            "roundtrip_cost_pct": round(cost * 4 * 100, 2),
            "trades_count": len(trades_df),
            "capacity_skipped_count": len(skipped_df),
            "final_equity_tl": round(fin_eq, 2),
            "total_return_pct": round(tot_ret, 2),
            "cagr_pct": round(cagr, 2),
            "max_daily_mtm_dd_pct": round(max_dd, 2),
            "avg_open_positions": round(avg_pos, 2),
            "agent_breakdown": by_agent.to_dict(orient="records"),
            "regime_breakdown": by_regime.to_dict(orient="records"),
        }
        scenario_summaries.append(summary_item)
        log.info("[%s] Equity: %.2f TL, CAGR: %.2f%%, MaxDD: %.2f%%, İşlem: %d, Kapasite Reddi: %d",
                 s_name, fin_eq, cagr, max_dd, len(trades_df), len(skipped_df))

    with open(RESULTS_DIR / "p1_revision_scenarios_summary.json", "w", encoding="utf-8") as f:
        json.dump(scenario_summaries, f, indent=2, ensure_ascii=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. PAPER TRADING KAYIT ŞABLONU
    # ─────────────────────────────────────────────────────────────────────────
    sample_paper_log = pd.DataFrame([
        {
            "tarih": "2026-10-07",
            "sembol": "ASELS",
            "birincil_ajan": "GTD",
            "tum_ajanlar": "GTD,ALPHA_B",
            "sinyal_kapanis_fiyati": 376.00,
            "beklenen_acilis_fiyati": 376.00,
            "gerceklesen_acilis_fiyati": 376.25,
            "gap_orani_pct": 0.07,
            "gerceklesen_slipaj_pct": 0.07,
            "onaylanan_lot": 26,
            "tutar_tl": 9782.50,
            "kapasite_durumu": "ISLEME_ALINDI",  # veya KAPASITE_DOLU_RED
            "xu100_rejim_boga": True,
            "rel_vol": 1.45,
            "cikis_tarihi": "",
            "cikis_fiyati": "",
            "cikis_nedeni": "",
            "net_getiri_pct": "",
            "net_pnl_tl": "",
            "notlar": "Ilk paper pozisyonu"
        }
    ])
    sample_paper_log.to_csv(RESULTS_DIR / "paper_trading_log_template.csv", index=False)
    log.info("Paper Trading Şablonu oluşturuldu -> results/paper_trading_log_template.csv")


if __name__ == "__main__":
    run_full_p1_revision()
