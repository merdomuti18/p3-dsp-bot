# -*- coding: utf-8 -*-
"""
p1_paper_config.py — P1 Paper Trading Sabit Konfigürasyon ve Parametreler
========================================================================
P1 Momentum araştırma sonuçlarına dayalı sabit ve dondurulmuş paper trading parametreleri.
Tüm parametreler ve dosya yolları bu modülden okunur.
"""
from __future__ import annotations

import os
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from research_p1.p1_exit_rules import (
    COMMISSION_RATE,
    GAP_TOLERANCE,
    MAX_HOLDING_DAYS,
    SLIPPAGE_RATE,
    STOP_PCT,
    TP1_PCT,
    TRAILING_PCT,
    gap_ratio,
    gap_rejected,
    stop_level_price,
    tp1_level_price,
    tp1_sold_lots,
    trail_level_price,
)

# ── Temel Dizinler ───────────────────────────────────────────────────────────
BASE_DIR = Path(os.environ.get("MOTT_BASE_DIR", str(Path(__file__).resolve().parent)))
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESEARCH_DIR = BASE_DIR / "research_p1"
RESEARCH_RESULTS_DIR = RESEARCH_DIR / "results"
RESEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dosya Yolları ───────────────────────────────────────────────────────────
PAPER_LOG_FILE = RESULTS_DIR / "paper_trading_log.csv"
SHADOW_SIGNALS_FILE = RESULTS_DIR / "shadow_signals.jsonl"
SIGNAL_PARITY_FILE = RESULTS_DIR / "signal_parity_log.csv"
PAPER_STATE_FILE = RESULTS_DIR / "p1_paper_state.json"
ALERTS_LOG_FILE = RESULTS_DIR / "p1_monitoring_alerts.log"
HEARTBEAT_FILE = RESULTS_DIR / "p1_run_heartbeat.csv"
TRADE_LEVEL_PARITY_FILE = RESULTS_DIR / "trade_level_parity.csv"
TEMPLATE_FILE = RESEARCH_RESULTS_DIR / "paper_trading_log_template.csv"

# ── Zaman Dilimi ─────────────────────────────────────────────────────────────
TZ_ISTANBUL = ZoneInfo("Europe/Istanbul")

# ── Sermaye ve Pozisyon Boyutları ───────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0         # 100.000 TL sanal sermaye
POSITION_SIZE = 20_000.0           # 20.000 TL / işlem
POSITION_SIZE_10K_EQ = 10_000.0    # Paralel 10.000 TL eşdeğeri
MAX_POSITIONS = 5                  # En fazla 5 eşzamanlı pozisyon (maksimum maruziyet <= %100)

# ── Ajanlar ve Öncelik Kuralları ─────────────────────────────────────────────
# 1. Öncelik: GTD (RSI 65-78 dahil mevcut tüm filtreler)
# 2. Öncelik: ALPHA_B_SIMPLE (Özgün AlphaTrend + CMF > -0.05 + RelVol >= 1.0; RSI/ADX yok)
# 3. Öncelik: ZT3 (Mevcut haliyle)
AGENTS_PRIORITY = ["GTD", "ALPHA_B_SIMPLE", "ZT3"]

# Gölge Ajan: GT (RSI'sız, GT_NO_RSI) -> Sermaye ALMAZ, sadece shadow_signals.jsonl'e loglanır
SHADOW_AGENTS = ["GT_NO_RSI"]

# ── İşlem Maliyeti ve Kayma Ayarları ─────────────────────────────────────────
# Tek kaynak: research_p1.p1_exit_rules (paper ve backtest aynı nesneler)
# Tek yön: Komisyon %0.30 + Slipaj %0.30 -> Tur başı %1.20
ROUNDTRIP_COST_PCT = 2.0 * (COMMISSION_RATE + SLIPPAGE_RATE) * 100.0  # %1.20

# ── Açılış VWAP penceresi (TSİ) ──────────────────────────────────────────────
# acilis fazı mevcut mott_daily `alim` penceresi (10:00–11:20) ile tetiklenir.
# İlk 5 adet 1m bar 10:00–10:05 arasında oluşur; 10:05–10:10 beklenebilir.
ACILIS_VWAP_WINDOW_START = time(10, 5)
ACILIS_VWAP_WINDOW_END = time(10, 10)

# ── İzleme Uyarı Eşikleri ───────────────────────────────────────────────────
# Dolum maliyeti hareketli ortalama baremleri (n >= 20):
COST_ALERT_GREEN_MAX = 1.20    # <= %1.20 Yeşil
COST_ALERT_YELLOW_MAX = 1.50   # %1.20 - %1.50 Sarı
# > %1.50 Kırmızı

# Günlük MTM Drawdown uyarı eşikleri (açık pozisyonlar dahil):
DD_ALERT_THRESHOLDS = [-10.0, -15.0, -20.0]

# ── Feature Flag Kontrolü ────────────────────────────────────────────────────
def is_p1_paper_enabled() -> bool:
    """P1 paper trading modunun aktif olup olmadığını kontrol eder (P1_PAPER=on/1/true)."""
    val = os.environ.get("P1_PAPER", "").strip().lower()
    return val in ("on", "1", "true", "yes")
