# -*- coding: utf-8 -*-
"""
p1_paper_config.py — P1 Paper Trading Sabit Konfigürasyon ve Parametreler
========================================================================
P1 Momentum araştırma sonuçlarına dayalı sabit ve dondurulmuş paper trading parametreleri.
Tüm parametreler ve dosya yolları bu modülden okunur.
"""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

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
# Tek yön: Komisyon %0.30 + Slipaj %0.30 -> Tur başı %1.20
COMMISSION_RATE = 0.0030
SLIPPAGE_RATE = 0.0030
ROUNDTRIP_COST_PCT = 2.0 * (COMMISSION_RATE + SLIPPAGE_RATE) * 100.0  # %1.20

# ── Sinyal ve Fiyat Kuralları ────────────────────────────────────────────────
# Gap yok kuralı: |gap| <= 0.0001 (0.01% sayısal yuvarlama toleransı)
GAP_TOLERANCE = 0.0001

# ── Çıkış Kuralları (Mevcut Motor Kuralları) ──────────────────────────────────
STOP_PCT = -0.05           # -%5.0 zarar kes
TP1_PCT = 0.08             # +%8.0 kısmi kâr al (yarısı satılır, tek lot kalırsa tamamı)
TRAILING_PCT = -0.05       # -%5.0 iz süren stop (TP1 sonrası görülen zirveden)
MAX_HOLDING_DAYS = 10      # 10 takvim günü maksimum tutma süresi

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
