# -*- coding: utf-8 -*-
"""
research_p1/run_ablation.py
===========================
Ajan İçi Filtre Ablasyon Deneyleri (Deney B1 ve B2):
Temel sinyaller ile ek filtrelerin ayrıştırılması ve her bir filtrenin tek tek kaldırılması:
1. GT (Güçlü Trend):
   - Temel Sinyal: EMA8 > EMA21 > EMA50
   - Filtreler: EMA200, RSI (50-65), ADX (>20), DI (DI+ > DI-), CMF (>0), RelVol (>=1.0)
2. GTD (Güçlü Trend Devam):
   - Temel Sinyal: EMA8 > EMA21 > EMA50
   - Filtreler: EMA200, RSI (65-78), ADX (>25), DI (DI+ > DI-), CMF (>0.05), RelVol (>=1.2)
3. ZKN:
   - Temel Sinyal: Close > EMA50
   - Filtreler: EMA200, RSI (40-58), StochRSI (<40), CMF (>-0.1), RelVol (>=0.8)
4. ZT3:
   - Temel Sinyal: MACD Bullish Crossover (macd > macd_sig and macd_prev <= macd_sprev)
   - Filtreler: ADX (>25), DI (DI+ > DI-), RSI (>50), AlphaTrend Bull, RelVol (>=1.0)
5. KBM:
   - Temel Sinyal: MACD Bullish Crossover (macd > macd_sig and macd_prev <= macd_sprev)
   - Filtreler: BB_mid (Close > BB_mid), RSI (>48), CMF (>0), RelVol (>=0.9)
6. ALPHA:
   - Varyant A: Özgün AlphaTrend 14-1 Alış Sinyali (Kıvanç Özbilgiç)
   - Varyant B: Özgün AlphaTrend + P1 Ek Filtreleri (RSI > 45, ADX > 18, CMF > -0.05, RelVol >= 1.0)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

from research_p1.backtest_engine import simulate_trades
from research_p1.data_downloader import load_cached_ohlcv, BIST100_SYMBOLS
from research_p1.p1_engine import compute_all_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_custom_filter(ind: pd.DataFrame, agent_name: str, ablate_filter: str | None = None) -> pd.Series:
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

    if agent_name == "GT":
        base = (ema8 > ema21) & (ema21 > ema50)
        cond = base.copy()
        if ablate_filter != "ema200":
            cond &= (ema200.isna() | (c > ema200))
        if ablate_filter != "rsi":
            cond &= (rsi >= 50) & (rsi <= 65)
        if ablate_filter != "adx":
            cond &= (adx > 20)
        if ablate_filter != "di":
            cond &= (di_p > di_n)
        if ablate_filter != "cmf":
            cond &= (cmf > 0)
        if ablate_filter != "rel_vol":
            cond &= (rel_vol >= 1.0)
        return cond

    elif agent_name == "GTD":
        base = (ema8 > ema21) & (ema21 > ema50)
        cond = base.copy()
        if ablate_filter != "ema200":
            cond &= (ema200.isna() | (c > ema200))
        if ablate_filter != "rsi":
            cond &= (rsi >= 65) & (rsi <= 78)
        if ablate_filter != "adx":
            cond &= (adx > 25)
        if ablate_filter != "di":
            cond &= (di_p > di_n)
        if ablate_filter != "cmf":
            cond &= (cmf > 0.05)
        if ablate_filter != "rel_vol":
            cond &= (rel_vol >= 1.2)
        return cond

    elif agent_name == "ZKN":
        base = (c > ema50)
        cond = base.copy()
        if ablate_filter != "ema200":
            cond &= (ema200.isna() | (c > ema200))
        if ablate_filter != "rsi":
            cond &= (rsi >= 40) & (rsi <= 58)
        if ablate_filter != "stochrsi":
            cond &= (stochrsi < 40)
        if ablate_filter != "cmf":
            cond &= (cmf > -0.1)
        if ablate_filter != "rel_vol":
            cond &= (rel_vol >= 0.8)
        return cond

    elif agent_name == "ZT3":
        base = (macd > macd_sig) & (macd_prev <= macd_sprev)
        cond = base.copy()
        if ablate_filter != "adx":
            cond &= (adx > 25)
        if ablate_filter != "di":
            cond &= (di_p > di_n)
        if ablate_filter != "rsi":
            cond &= (rsi > 50)
        if ablate_filter != "alphatrend":
            cond &= at_bull
        if ablate_filter != "rel_vol":
            cond &= (rel_vol >= 1.0)
        return cond

    elif agent_name == "KBM":
        base = (macd > macd_sig) & (macd_prev <= macd_sprev)
        cond = base.copy()
        if ablate_filter != "bb_mid":
            cond &= (c > bb_mid)
        if ablate_filter != "rsi":
            cond &= (rsi > 48)
        if ablate_filter != "cmf":
            cond &= (cmf > 0)
        if ablate_filter != "rel_vol":
            cond &= (rel_vol >= 0.9)
        return cond

    elif agent_name == "ALPHA":
        # Varyant A = özgün (filtresiz)
        base = ind["at_buy_confirmed"].fillna(False)
        cond = base.copy()
        if ablate_filter == "none_full_filters":  # Varyant B
            cond &= (rsi > 45) & (adx > 18) & (cmf > -0.05) & (rel_vol >= 1.0)
        elif ablate_filter == "remove_rsi":
            cond &= (adx > 18) & (cmf > -0.05) & (rel_vol >= 1.0)
        elif ablate_filter == "remove_adx":
            cond &= (rsi > 45) & (cmf > -0.05) & (rel_vol >= 1.0)
        elif ablate_filter == "remove_cmf":
            cond &= (rsi > 45) & (adx > 18) & (rel_vol >= 1.0)
        elif ablate_filter == "remove_rel_vol":
            cond &= (rsi > 45) & (adx > 18) & (cmf > -0.05)
        return cond

    return pd.Series(False, index=ind.index)


def run_filter_ablation_experiments(symbol_data: dict):
    FULL_START = "2021-10-06"
    FULL_END = "2026-10-05"

    ablation_definitions = [
        # GT
        ("GT", None, "GT_Tam_Filtreler"),
        ("GT", "ema200", "GT_Kaldır_EMA200"),
        ("GT", "rsi", "GT_Kaldır_RSI"),
        ("GT", "adx", "GT_Kaldır_ADX"),
        ("GT", "di", "GT_Kaldır_DI"),
        ("GT", "cmf", "GT_Kaldır_CMF"),
        ("GT", "rel_vol", "GT_Kaldır_RelVol"),

        # GTD
        ("GTD", None, "GTD_Tam_Filtreler"),
        ("GTD", "ema200", "GTD_Kaldır_EMA200"),
        ("GTD", "rsi", "GTD_Kaldır_RSI"),
        ("GTD", "adx", "GTD_Kaldır_ADX"),
        ("GTD", "di", "GTD_Kaldır_DI"),
        ("GTD", "cmf", "GTD_Kaldır_CMF"),
        ("GTD", "rel_vol", "GTD_Kaldır_RelVol"),

        # ZKN
        ("ZKN", None, "ZKN_Tam_Filtreler"),
        ("ZKN", "ema200", "ZKN_Kaldır_EMA200"),
        ("ZKN", "rsi", "ZKN_Kaldır_RSI"),
        ("ZKN", "stochrsi", "ZKN_Kaldır_StochRSI"),
        ("ZKN", "cmf", "ZKN_Kaldır_CMF"),
        ("ZKN", "rel_vol", "ZKN_Kaldır_RelVol"),

        # ZT3
        ("ZT3", None, "ZT3_Tam_Filtreler"),
        ("ZT3", "adx", "ZT3_Kaldır_ADX"),
        ("ZT3", "di", "ZT3_Kaldır_DI"),
        ("ZT3", "rsi", "ZT3_Kaldır_RSI"),
        ("ZT3", "alphatrend", "ZT3_Kaldır_AlphaTrend"),
        ("ZT3", "rel_vol", "ZT3_Kaldır_RelVol"),

        # KBM
        ("KBM", None, "KBM_Tam_Filtreler"),
        ("KBM", "bb_mid", "KBM_Kaldır_BBMid"),
        ("KBM", "rsi", "KBM_Kaldır_RSI"),
        ("KBM", "cmf", "KBM_Kaldır_CMF"),
        ("KBM", "rel_vol", "KBM_Kaldır_RelVol"),

        # ALPHA Varyant A vs B ve Filtreleri
        ("ALPHA", None, "ALPHA_Varyant_A_Özgün"),
        ("ALPHA", "none_full_filters", "ALPHA_Varyant_B_Filtreli"),
        ("ALPHA", "remove_rsi", "ALPHA_Varyant_B_Kaldır_RSI"),
        ("ALPHA", "remove_adx", "ALPHA_Varyant_B_Kaldır_ADX"),
        ("ALPHA", "remove_cmf", "ALPHA_Varyant_B_Kaldır_CMF"),
        ("ALPHA", "remove_rel_vol", "ALPHA_Varyant_B_Kaldır_RelVol"),
    ]

    log.info("Filtre ablasyonu başlıyor (%d deney)...", len(ablation_definitions))
    results = []

    for ag, ablate_target, label in ablation_definitions:
        precomp = {}
        for sym, df in symbol_data.items():
            precomp[sym] = {label: evaluate_custom_filter(df, ag, ablate_target)}

        trades, _, m = simulate_trades(
            symbol_data, [label],
            precomputed_signals=precomp,
            start_date=FULL_START, end_date=FULL_END,
            gap_tolerance=0.0001,
            commission_rate=0.002, slippage_rate=0.002,
            mode="single_agent"
        )
        m["agent"] = ag
        m["ablated_filter"] = ablate_target or "NONE"
        m["label"] = label
        results.append(m)
        log.info("[%s] islem: %d, Ort Net: %+.2f%%, PF: %.2f",
                 label, m["n_trades"], m["mean_return_pct"], m["profit_factor"])

    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_DIR / "exp_b1_filter_ablation.csv", index=False)
    log.info("Ablasyon deneyleri tamamlandı -> exp_b1_filter_ablation.csv")


if __name__ == "__main__":
    from research_p1.run_experiments import load_and_prepare_all_data
    sym_data = load_and_prepare_all_data()
    run_filter_ablation_experiments(sym_data)
