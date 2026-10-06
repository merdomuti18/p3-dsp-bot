# -*- coding: utf-8 -*-
"""
research_p1/backtest_engine.py
==============================
P1 Momentum araştırma simülasyon ve backtest motoru.

Kapsam ve Kurallar:
1. Sinyal Tespiti:
   - t gününün kapanışında oluşan sinyal hesaplanır.
   - Sinyal günü t'nin kapanışında ASLA alım yapılmaz.
   - En erken giriş sonraki işlem günü t+1'in açılışındadır.
2. Gap Kuralı:
   - gap = open[t+1] / close[t] - 1.0
   - Ana kural: "gap yok" (|gap| <= 0.0001 yuvarlama toleransı).
   - Duyarlılık deneyleri: |gap| <= 0.0025 (%0.25), 0.0050 (%0.50), 0.0100 (%1.00).
   - Gap toleransı dışındaysa alım REDDEDİLİR ve kovalanmaz.
3. Masraf ve Slipaj Modeli:
   - Alış gerçekleşme fiyatı = open[t+1] * (1.0 + slipaj)
   - Alış komisyonu = lotlar * alis_fiyat * komisyon
   - Satış gerçekleşme fiyatı = ref_fiyat * (1.0 - slipaj)
   - Satış komisyonu = lotlar * satis_fiyat * komisyon
   - Temel senaryo: Komisyon = %0.20 (%0.002), Slipaj = %0.20 (%0.002)
   - Dayanıklılık: (%0.30, %0.30) ve (%0.40, %0.40)
4. Çıkış Kuralları:
   - STOP: -%5 (giris_f * 0.95). Gap down açılışta açılış fiyatından satılır.
   - TP1: +%8 (giris_f * 1.08). Lotların yarısı satılır (tek lot ise tamamı).
   - TRAILING: -%5 (önceki mumların tepe_f değerinden). Lookahead yapılmaz.
   - MAX_GUN: 10 takvim günü.
5. İki Test Düzeni:
   - Düzen A (Ajan Katkısı): Bağımsız pozisyonlar (aynı hisse açıkken yeni alım yok). Sabit ufuklar: 1, 5, 10, 20 gün + P1 çıkışı + MFE/MAE.
   - Düzen B (Portföy Testi): 100.000 TL başlangıç sermayesi, nakit yeterliliği, tam lot, slot sınırları.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

from research_p1.p1_engine import compute_all_indicators

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "ohlcv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Position:
    pos_id: str
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float        # Slipaj uygulanmış gerçekleşme fiyatı
    ref_open_price: float     # Ham açılış fiyatı
    signal_date: pd.Timestamp
    signal_close: float
    gap: float
    lots: int
    initial_lots: int
    entry_commission: float
    total_cost: float         # lots * entry_price + entry_commission
    peak_price: float
    tp1_done: bool = False
    closed: bool = False
    exit_date: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_commission: float = 0.0
    net_sales_income: float = 0.0
    exits_detail: List[dict] = field(default_factory=list)
    mfe: float = 0.0
    mae: float = 0.0
    agents: List[str] = field(default_factory=list)


def evaluate_agent_signals(ind: pd.DataFrame, agent_name: str, mc: float = 50_000_000) -> pd.Series:
    """Belirli bir ajanın sinyal maskesini (bool Series) üretir."""
    n = len(ind)
    signals = pd.Series(False, index=ind.index)

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

    if agent_name == "GT":
        # Güçlü Trend: ema8 > ema21 > ema50, (isna(ema200) or close > ema200), 50<=rsi<=65, adx>20, di_p>di_n, cmf>0, rel_vol>=1
        cond = (
            (ema8 > ema21) & (ema21 > ema50) &
            (ema200.isna() | (c > ema200)) &
            (rsi >= 50) & (rsi <= 65) &
            (adx > 20) &
            (di_p > di_n) &
            (cmf > 0) &
            (rel_vol >= 1.0)
        )
        return cond

    elif agent_name == "GTD":
        # Güçlü Trend Devam: ema8 > ema21 > ema50, (isna(ema200) or close > ema200), 65<=rsi<=78, adx>25, di_p>di_n, cmf>0.05, rel_vol>=1.2
        cond = (
            (ema8 > ema21) & (ema21 > ema50) &
            (ema200.isna() | (c > ema200)) &
            (rsi >= 65) & (rsi <= 78) &
            (adx > 25) &
            (di_p > di_n) &
            (cmf > 0.05) &
            (rel_vol >= 1.2)
        )
        return cond

    elif agent_name == "ALPHA_A":
        # Özgün AlphaTrend 14-1 Alış Sinyali (Kıvanç Özbilgiç Pine Script onaylı sinyal)
        return ind["at_buy_confirmed"].fillna(False)

    elif agent_name == "ALPHA_B":
        # Mevcut P1 ALPHA: AlphaTrend Alış Sinyali + Ek Filtreler (rsi > 45, adx > 18, cmf > -0.05, rel_vol >= 1)
        # Not: Özgün AlphaTrend al sinyaliyle teyit edilir
        cond = (
            ind["at_buy_confirmed"].fillna(False) &
            (rsi > 45) &
            (adx > 18) &
            (cmf > -0.05) &
            (rel_vol >= 1.0)
        )
        return cond

    elif agent_name == "ALPHA":
        # Varsayılan ALPHA olarak Varyant A (Özgün AlphaTrend) kullanılır
        return ind["at_buy_confirmed"].fillna(False)

    elif agent_name == "ZKN":
        # ZKN: close > ema50, (isna(ema200) or close > ema200), 40<=rsi<=58, stochrsi<40, cmf>-0.1, rel_vol>=0.8
        cond = (
            (c > ema50) &
            (ema200.isna() | (c > ema200)) &
            (rsi >= 40) & (rsi <= 58) &
            (stochrsi < 40) &
            (cmf > -0.1) &
            (rel_vol >= 0.8)
        )
        return cond

    elif agent_name == "ZT3":
        # ZT3: adx>25, di_p>di_n, rsi>50, macd>macd_sig, macd_prev<=macd_sprev, alpha_trend_bull, rel_vol>=1
        # alpha_trend_bull için özgün AlphaTrend çizgisi > AlphaTrend[2] kullanılır
        at_bull = ind["at_alpha_trend"] > ind["at_alpha_trend_2"]
        cond = (
            (adx > 25) &
            (di_p > di_n) &
            (rsi > 50) &
            (macd > macd_sig) &
            (macd_prev <= macd_sprev) &
            at_bull &
            (rel_vol >= 1.0)
        )
        return cond

    elif agent_name == "KBM":
        # KBM: close > bb_mid, macd > macd_sig, macd_prev <= macd_sprev, rsi > 48, cmf > 0, rel_vol >= 0.9
        cond = (
            (c > bb_mid) &
            (macd > macd_sig) &
            (macd_prev <= macd_sprev) &
            (rsi > 48) &
            (cmf > 0) &
            (rel_vol >= 0.9)
        )
        return cond

    return signals


def precompute_signals_for_agents(
    symbol_data: Dict[str, pd.DataFrame],
    agents: List[str]
) -> Dict[str, Dict[str, pd.Series]]:
    """Tüm hisseler ve ajanlar için sinyal serilerini bir kez önceden hesaplar."""
    precomputed = {}
    for sym, df in symbol_data.items():
        precomputed[sym] = {}
        for ag in agents:
            precomputed[sym][ag] = evaluate_agent_signals(df, ag)
    return precomputed


def simulate_trades(
    symbol_data: Dict[str, pd.DataFrame],
    agent_names: List[str],
    precomputed_signals: Optional[Dict[str, Dict[str, pd.Series]]] = None,
    start_date: str = "2021-10-06",
    end_date: str = "2026-10-05",
    gap_tolerance: float = 0.0001,  # Ana kural: gap yok (|gap| <= 0.0001)
    commission_rate: float = 0.0020,  # %0.20
    slippage_rate: float = 0.0020,    # %0.20
    slippage_rate_map: Optional[Dict[str, float]] = None,
    mode: str = "single_agent",       # 'single_agent' veya 'portfolio'
    initial_capital: float = 100_000.0,
    max_positions: int = 7,
    position_size: float = 20_000.0,
    fixed_holding_periods: Optional[List[int]] = None,
) -> Tuple[List[dict], List[dict], dict]:
    """Tarihsel simülasyonu çalıştırır.

    Dönen tuple:
      (completed_trades_list, rejected_signals_list, summary_metrics_dict)
    """
    fixed_holding_periods = fixed_holding_periods or [1, 5, 10, 20]

    # Ortak tarih takvimini oluştur
    all_dates = set()
    for df in symbol_data.values():
        all_dates.update(df.index)
    sorted_dates = sorted(list(all_dates))

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    active_positions: Dict[str, Position] = {}
    completed_trades: List[dict] = []
    rejected_signals: List[dict] = []

    cash = initial_capital
    portfolio_equity_series = []

    for t_idx, current_date in enumerate(sorted_dates):
        # 1. Mevcut açık pozisyonların güncellenmesi (STOP, TP1, TRAILING, MAX_GUN)
        closed_syms = []
        for sym, pos in active_positions.items():
            df = symbol_data.get(sym)
            if df is None or current_date not in df.index:
                continue

            sym_slip = slippage_rate_map.get(sym, slippage_rate) if slippage_rate_map else slippage_rate

            row = df.loc[current_date]
            o_bar = float(row["open"])
            h_bar = float(row["high"])
            l_bar = float(row["low"])
            c_bar = float(row["close"])

            # MFE ve MAE takibi (giriş fiyatına göre)
            mfe_p = (h_bar - pos.entry_price) / pos.entry_price
            mae_p = (l_bar - pos.entry_price) / pos.entry_price
            pos.mfe = max(pos.mfe, mfe_p)
            pos.mae = min(pos.mae, mae_p)

            holding_days = (current_date - pos.entry_date).days

            # STOP kontrolü (-%5)
            stop_level = round(pos.entry_price * 0.95, 4)
            is_stop = l_bar <= stop_level

            # TP1 kontrolü (+%8)
            tp1_level = round(pos.entry_price * 1.08, 4)
            is_tp1 = (h_bar >= tp1_level) and not pos.tp1_done

            # Trailing kontrolü (Önceki mumların zirvesinden -%5)
            trail_level = round(pos.peak_price * 0.95, 4) if pos.tp1_done else 0.0
            is_trailing = pos.tp1_done and (l_bar <= trail_level)

            # Max gün kontrolü (10 gün)
            is_max_gun = holding_days >= 10

            # A. Çift Tetik Belirsizliği: STOP ve TP1 aynı mumda
            if is_stop and is_tp1:
                # Muhafazakâr stop önceliği
                ref_exit = o_bar if o_bar <= stop_level else stop_level
                exit_price = round(ref_exit * (1.0 - sym_slip), 4)
                comm = round(pos.lots * exit_price * commission_rate, 4)
                net_income = round(pos.lots * exit_price - comm, 4)

                pos.exits_detail.append({
                    "date": str(current_date.date()),
                    "lots": pos.lots,
                    "price": exit_price,
                    "reason": "STOP_AMBIGUOUS",
                    "commission": comm,
                    "net_income": net_income,
                })
                pos.net_sales_income += net_income
                pos.exit_commission += comm
                pos.closed = True
                pos.exit_date = current_date
                pos.exit_price = exit_price
                pos.exit_reason = "STOP_AMBIGUOUS"
                cash += net_income
                closed_syms.append(sym)
                continue

            # B. Normal STOP
            if is_stop:
                ref_exit = o_bar if o_bar <= stop_level else stop_level
                exit_price = round(ref_exit * (1.0 - sym_slip), 4)
                comm = round(pos.lots * exit_price * commission_rate, 4)
                net_income = round(pos.lots * exit_price - comm, 4)

                pos.exits_detail.append({
                    "date": str(current_date.date()),
                    "lots": pos.lots,
                    "price": exit_price,
                    "reason": "STOP",
                    "commission": comm,
                    "net_income": net_income,
                })
                pos.net_sales_income += net_income
                pos.exit_commission += comm
                pos.closed = True
                pos.exit_date = current_date
                pos.exit_price = exit_price
                pos.exit_reason = "STOP"
                cash += net_income
                closed_syms.append(sym)
                continue

            # C. TP1
            if is_tp1:
                ref_exit = max(o_bar, tp1_level)
                tp_exit_price = round(ref_exit * (1.0 - sym_slip), 4)
                if pos.lots <= 1:
                    sold_lots = 1
                    pos.lots = 0
                else:
                    sold_lots = pos.lots // 2
                    pos.lots -= sold_lots

                comm = round(sold_lots * tp_exit_price * commission_rate, 4)
                net_income = round(sold_lots * tp_exit_price - comm, 4)
                pos.exits_detail.append({
                    "date": str(current_date.date()),
                    "lots": sold_lots,
                    "price": tp_exit_price,
                    "reason": "TP1",
                    "commission": comm,
                    "net_income": net_income,
                })
                pos.net_sales_income += net_income
                pos.exit_commission += comm
                pos.tp1_done = True
                cash += net_income

                if pos.lots == 0:
                    pos.closed = True
                    pos.exit_date = current_date
                    pos.exit_price = tp_exit_price
                    pos.exit_reason = "TP1_FULL"
                    closed_syms.append(sym)
                    continue

            # D. TRAILING
            elif is_trailing:
                ref_exit = o_bar if o_bar <= trail_level else trail_level
                exit_price = round(ref_exit * (1.0 - sym_slip), 4)
                comm = round(pos.lots * exit_price * commission_rate, 4)
                net_income = round(pos.lots * exit_price - comm, 4)

                pos.exits_detail.append({
                    "date": str(current_date.date()),
                    "lots": pos.lots,
                    "price": exit_price,
                    "reason": "TRAILING",
                    "commission": comm,
                    "net_income": net_income,
                })
                pos.net_sales_income += net_income
                pos.exit_commission += comm
                pos.closed = True
                pos.exit_date = current_date
                pos.exit_price = exit_price
                pos.exit_reason = "TRAILING"
                cash += net_income
                closed_syms.append(sym)
                continue

            # E. MAX_GUN
            if is_max_gun:
                exit_price = round(c_bar * (1.0 - sym_slip), 4)
                comm = round(pos.lots * exit_price * commission_rate, 4)
                net_income = round(pos.lots * exit_price - comm, 4)

                pos.exits_detail.append({
                    "date": str(current_date.date()),
                    "lots": pos.lots,
                    "price": exit_price,
                    "reason": "MAX_GUN",
                    "commission": comm,
                    "net_income": net_income,
                })
                pos.net_sales_income += net_income
                pos.exit_commission += comm
                pos.closed = True
                pos.exit_date = current_date
                pos.exit_price = exit_price
                pos.exit_reason = "MAX_GUN"
                cash += net_income
                closed_syms.append(sym)
                continue

            # Trailing için zirve güncelleme (mum kapandıktan sonra)
            pos.peak_price = max(pos.peak_price, h_bar)

        for sym in closed_syms:
            pos = active_positions.pop(sym)
            net_return_pct = (pos.net_sales_income - pos.total_cost) / pos.total_cost * 100
            net_profit_tl = pos.net_sales_income - pos.total_cost
            completed_trades.append({
                "pos_id": pos.pos_id,
                "symbol": pos.symbol,
                "agents": ",".join(pos.agents),
                "signal_date": str(pos.signal_date.date()),
                "entry_date": str(pos.entry_date.date()),
                "ref_open_price": pos.ref_open_price,
                "entry_price": pos.entry_price,
                "gap": round(pos.gap, 5),
                "initial_lots": pos.initial_lots,
                "entry_commission": pos.entry_commission,
                "exit_commission": pos.exit_commission,
                "total_commission": round(pos.entry_commission + pos.exit_commission, 4),
                "total_cost": round(pos.total_cost, 2),
                "net_sales_income": round(pos.net_sales_income, 2),
                "exit_date": str(pos.exit_date.date()) if pos.exit_date else "",
                "exit_price": pos.exit_price,
                "exit_reason": pos.exit_reason,
                "net_return_pct": round(net_return_pct, 4),
                "net_profit_tl": round(net_profit_tl, 2),
                "holding_days": (pos.exit_date - pos.entry_date).days if pos.exit_date else 0,
                "mfe_pct": round(pos.mfe * 100, 2),
                "mae_pct": round(pos.mae * 100, 2),
                "exits_count": len(pos.exits_detail),
            })

        # 2. Portföy günlük değerleme serisi
        if start_ts <= current_date <= end_ts:
            open_pos_val = 0.0
            for sym, pos in active_positions.items():
                df = symbol_data.get(sym)
                if df is not None and current_date in df.index:
                    open_pos_val += pos.lots * float(df.loc[current_date, "close"])
                else:
                    open_pos_val += pos.lots * pos.entry_price
            equity = cash + open_pos_val
            portfolio_equity_series.append({
                "date": str(current_date.date()),
                "cash": round(cash, 2),
                "open_value": round(open_pos_val, 2),
                "equity": round(equity, 2),
                "open_positions": len(active_positions),
            })

        # 3. Önceki gün t'de üretilen sinyallerin t+1 (current_date) açılışında alımı
        if t_idx == 0 or current_date < start_ts or current_date > end_ts:
            continue

        prev_date = sorted_dates[t_idx - 1]

        # t gününün kapanışında oluşan sinyalleri tara
        candidates_today = []
        for sym, df in symbol_data.items():
            if prev_date not in df.index or current_date not in df.index:
                continue

            # Ajan sinyali kontrolü (prev_date kapanışında)
            triggered_agents = []
            for ag in agent_names:
                if precomputed_signals and sym in precomputed_signals and ag in precomputed_signals[sym]:
                    if precomputed_signals[sym][ag].loc[prev_date]:
                        triggered_agents.append(ag)
                else:
                    sig_series = evaluate_agent_signals(df, ag)
                    if sig_series.loc[prev_date]:
                        triggered_agents.append(ag)

            if not triggered_agents:
                continue

            # Gap kontrolü
            prev_close = float(df.loc[prev_date, "close"])
            curr_open = float(df.loc[current_date, "open"])
            gap = (curr_open / prev_close) - 1.0

            if abs(gap) > gap_tolerance:
                rejected_signals.append({
                    "symbol": sym,
                    "signal_date": str(prev_date.date()),
                    "entry_date": str(current_date.date()),
                    "signal_close": prev_close,
                    "open_price": curr_open,
                    "gap": round(gap, 5),
                    "agents": ",".join(triggered_agents),
                    "reason": f"gap_rejected_{abs(gap):.4f}>tol_{gap_tolerance:.4f}",
                })
                continue

            candidates_today.append({
                "symbol": sym,
                "signal_date": prev_date,
                "signal_close": prev_close,
                "curr_open": curr_open,
                "gap": gap,
                "agents": triggered_agents,
                "score": len(triggered_agents),
            })

        # Adayları sırala (Skor sayısı büyükten küçüğe, eşitlikte sembol adına göre deterministik)
        candidates_today.sort(key=lambda x: (-x["score"], x["symbol"]))

        # Alım simülasyonu
        for cand in candidates_today:
            sym = cand["symbol"]
            if sym in active_positions:
                continue  # Aynı hissede açık pozisyon varken tekrar alım yapılmaz

            if mode == "portfolio":
                if len(active_positions) >= max_positions:
                    continue  # Boş slot yok
                alloc = min(position_size, cash / max(1, max_positions - len(active_positions)))
            else:
                # single_agent modu: position_size sanal slot büyüklüğü
                alloc = position_size

            sym_slip = slippage_rate_map.get(sym, slippage_rate) if slippage_rate_map else slippage_rate
            curr_open = cand["curr_open"]
            fill_price = round(curr_open * (1.0 + sym_slip), 4)
            lots = int(alloc / max(fill_price, 0.01))
            if lots < 1:
                continue

            entry_comm = round(lots * fill_price * commission_rate, 4)
            tot_cost = round(lots * fill_price + entry_comm, 4)

            if mode == "portfolio" and tot_cost > cash:
                continue  # Nakit yetersiz

            if mode == "portfolio":
                cash -= tot_cost

            pos_id = f"POS_{sym}_{current_date.strftime('%Y%m%d')}_{len(completed_trades)+len(active_positions)}"
            pos = Position(
                pos_id=pos_id,
                symbol=sym,
                entry_date=current_date,
                entry_price=fill_price,
                ref_open_price=curr_open,
                signal_date=cand["signal_date"],
                signal_close=cand["signal_close"],
                gap=cand["gap"],
                lots=lots,
                initial_lots=lots,
                entry_commission=entry_comm,
                total_cost=tot_cost,
                peak_price=fill_price,
                agents=cand["agents"],
            )
            active_positions[sym] = pos

    # Periyot sonunda açık kalan pozisyonları son fiyattan kapat
    for sym, pos in list(active_positions.items()):
        df = symbol_data.get(sym)
        last_dt = sorted_dates[-1]
        c_bar = float(df.loc[last_dt, "close"]) if (df is not None and last_dt in df.index) else pos.entry_price
        sym_slip = slippage_rate_map.get(sym, slippage_rate) if slippage_rate_map else slippage_rate
        exit_price = round(c_bar * (1.0 - sym_slip), 4)
        comm = round(pos.lots * exit_price * commission_rate, 4)
        net_income = round(pos.lots * exit_price - comm, 4)

        pos.exits_detail.append({
            "date": str(last_dt.date()),
            "lots": pos.lots,
            "price": exit_price,
            "reason": "PERIOD_END",
            "commission": comm,
            "net_income": net_income,
        })
        pos.net_sales_income += net_income
        pos.exit_commission += comm
        pos.closed = True
        pos.exit_date = last_dt
        pos.exit_price = exit_price
        pos.exit_reason = "PERIOD_END"

        net_return_pct = (pos.net_sales_income - pos.total_cost) / pos.total_cost * 100
        net_profit_tl = pos.net_sales_income - pos.total_cost
        completed_trades.append({
            "pos_id": pos.pos_id,
            "symbol": pos.symbol,
            "agents": ",".join(pos.agents),
            "signal_date": str(pos.signal_date.date()),
            "entry_date": str(pos.entry_date.date()),
            "ref_open_price": pos.ref_open_price,
            "entry_price": pos.entry_price,
            "gap": round(pos.gap, 5),
            "initial_lots": pos.initial_lots,
            "entry_commission": pos.entry_commission,
            "exit_commission": pos.exit_commission,
            "total_commission": round(pos.entry_commission + pos.exit_commission, 4),
            "total_cost": round(pos.total_cost, 2),
            "net_sales_income": round(pos.net_sales_income, 2),
            "exit_date": str(pos.exit_date.date()),
            "exit_price": pos.exit_price,
            "exit_reason": pos.exit_reason,
            "net_return_pct": round(net_return_pct, 4),
            "net_profit_tl": round(net_profit_tl, 2),
            "holding_days": (pos.exit_date - pos.entry_date).days,
            "mfe_pct": round(pos.mfe * 100, 2),
            "mae_pct": round(pos.mae * 100, 2),
            "exits_count": len(pos.exits_detail),
        })

    # İstatistiklerin hesaplanması
    metrics = compute_metrics(completed_trades, len(rejected_signals), portfolio_equity_series, initial_capital)
    return completed_trades, rejected_signals, metrics


def compute_metrics(trades: List[dict], rejected_count: int, equity_series: List[dict], initial_capital: float) -> dict:
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "mean_return_pct": 0.0, "median_return_pct": 0.0,
            "profit_factor": 0.0, "mean_profit_tl": 0.0, "rejected_signals": rejected_count,
            "avg_holding_days": 0.0, "avg_mfe_pct": 0.0, "avg_mae_pct": 0.0,
            "ci_lower_95": 0.0, "ci_upper_95": 0.0
        }

    returns = np.array([t["net_return_pct"] for t in trades])
    profits = np.array([t["net_profit_tl"] for t in trades])
    holding = np.array([t["holding_days"] for t in trades])
    mfes = np.array([t["mfe_pct"] for t in trades])
    maes = np.array([t["mae_pct"] for t in trades])

    wins = returns > 0
    win_rate = round(float(wins.mean() * 100), 2)
    mean_ret = round(float(returns.mean()), 4)
    med_ret = round(float(np.median(returns)), 4)

    gross_gain = float(profits[profits > 0].sum()) if (profits > 0).any() else 0.0
    gross_loss = abs(float(profits[profits < 0].sum())) if (profits < 0).any() else 0.0
    pf = round(gross_gain / gross_loss, 2) if gross_loss > 0 else (999.0 if gross_gain > 0 else 0.0)

    # 1000 Bootstrap güven aralığı (%95)
    ci_low, ci_high = 0.0, 0.0
    if n_trades >= 5:
        np.random.seed(42)
        boot_means = [np.random.choice(returns, size=n_trades, replace=True).mean() for _ in range(1000)]
        ci_low = round(float(np.percentile(boot_means, 2.5)), 4)
        ci_high = round(float(np.percentile(boot_means, 97.5)), 4)

    res = {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "mean_return_pct": mean_ret,
        "median_return_pct": med_ret,
        "profit_factor": pf,
        "mean_profit_tl": round(float(profits.mean()), 2),
        "total_profit_tl": round(float(profits.sum()), 2),
        "rejected_signals": rejected_count,
        "avg_holding_days": round(float(holding.mean()), 1),
        "avg_mfe_pct": round(float(mfes.mean()), 2),
        "avg_mae_pct": round(float(maes.mean()), 2),
        "ci_lower_95": ci_low,
        "ci_upper_95": ci_high,
    }

    if equity_series:
        eq_df = pd.DataFrame(equity_series)
        final_eq = eq_df["equity"].iloc[-1]
        cagr = ((final_eq / initial_capital) ** (1.0 / 5.0) - 1.0) * 100
        # Drawdown
        peak = eq_df["equity"].cummax()
        dd = (eq_df["equity"] - peak) / peak * 100
        max_dd = round(float(dd.min()), 2)
        res["final_equity"] = round(final_eq, 2)
        res["total_return_pct"] = round((final_eq / initial_capital - 1.0) * 100, 2)
        res["cagr_pct"] = round(cagr, 2)
        res["max_drawdown_pct"] = max_dd
        res["equity_series"] = equity_series

    return res
