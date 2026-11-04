# -*- coding: utf-8 -*-
"""
research_p1/p1_exit_rules.py
============================
P1 çıkış, gap ve maliyet sabitlerinin TEK kaynağı.

Paper motoru (p1_paper_config / p1_paper) ve backtest motorları
(backtest_engine, run_p1_revision_backtest) bu modülü import eder.
Değer kopyası yoktur; parametre değiştirilmez.

Gap tanımı (oran, yüzde değil):
    gap = open[t+1] / close[t] - 1.0
    kabul  <=>  abs(gap) <= GAP_TOLERANCE
    GAP_TOLERANCE = 0.0001  ==  0.01 yüzde puanı (1 bp) yuvarlama toleransı

TP1 lot kuralı (BIST tam lot, kesir yok):
    lots <= 1  -> tamamı
    lots >  1  -> lots // 2  (aşağı tam sayı)
"""
from __future__ import annotations

STOP_PCT = -0.05
TP1_PCT = 0.08
TRAILING_PCT = -0.05
MAX_HOLDING_DAYS = 10

GAP_TOLERANCE = 0.0001

COMMISSION_RATE = 0.0030
SLIPPAGE_RATE = 0.0030


def stop_level_price(entry_price: float) -> float:
    return round(float(entry_price) * (1.0 + STOP_PCT), 4)


def tp1_level_price(entry_price: float) -> float:
    return round(float(entry_price) * (1.0 + TP1_PCT), 4)


def trail_level_price(peak_price: float) -> float:
    return round(float(peak_price) * (1.0 + TRAILING_PCT), 4)


def tp1_sold_lots(lots: int) -> int:
    """BIST tam lot: tek lot (veya 0) ise tamamı, aksi halde floor(lots/2)."""
    n = int(lots)
    if n <= 1:
        return max(n, 0) if n <= 0 else 1
    return n // 2


def gap_ratio(curr_open: float, prev_close: float) -> float:
    """Açılış/önceki kapanış oranı eksi 1. Birim: oran (0.0001 = 0.01%)."""
    return (float(curr_open) / float(prev_close)) - 1.0


def gap_rejected(curr_open: float, prev_close: float, tolerance: float = GAP_TOLERANCE) -> bool:
    return abs(gap_ratio(curr_open, prev_close)) > tolerance
