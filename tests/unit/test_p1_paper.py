# -*- coding: utf-8 -*-
"""
tests/unit/test_p1_paper.py
===========================
P1 Paper Trading Altyapısı Kapsamlı Test Paketi:
(a) Öncelik ve slot kuralı: GTD (1) > ALPHA_B_SIMPLE (2) > ZT3 (3), RelVol azalan, maks. 5 slot.
(b) Lookahead yokluğu: Sinyal günü t kapanış, emir t+1 açılışta dolar; gelecek bar mutasyonu geçmişi etkilemez.
(c) Log idempotency: Aynı gün/seans için tekrar çalıştırmada mükerrer log satırı veya çift işlem oluşmaz.
(d) Maliyet kademe eşikleri: Yeşil <= %1.20, Sarı %1.20 - %1.50, Kırmızı > %1.50 ve DD uyarı eşikleri.
(e) Ajan filtre tanımları: ALPHA_B_SIMPLE'da RSI/ADX yok; GT sermaye almaz (sadece shadow log).
(f) Reddedilen sinyallerin sonradan getiri takibi (KAPASITE_REDDI / GAP_REDDI -> reddedilen_sonradan_getiri_pct).
(g) 60-seanslık geçmiş replay testi (uyum >= %95).
(h) 10k eşdeğer PnL ve MTM equity takibi.
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import p1_monitoring as p1_mon
import p1_paper as p1_pap
import p1_paper_config as p1_cfg
import p1_signal_parity as p1_par
from research_p1.run_p1_revision_backtest import evaluate_custom_agent_signals


@pytest.fixture(autouse=True)
def _isolate_paper_environment(tmp_path, monkeypatch):
    """Testlerin tmp_path altında izole dosyalara yazmasını sağlar."""
    res_dir = tmp_path / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(p1_cfg, "RESULTS_DIR", res_dir)
    monkeypatch.setattr(p1_cfg, "PAPER_LOG_FILE", res_dir / "paper_trading_log.csv")
    monkeypatch.setattr(p1_cfg, "SHADOW_SIGNALS_FILE", res_dir / "shadow_signals.jsonl")
    monkeypatch.setattr(p1_cfg, "SIGNAL_PARITY_FILE", res_dir / "signal_parity_log.csv")
    monkeypatch.setattr(p1_cfg, "PAPER_STATE_FILE", res_dir / "p1_paper_state.json")
    monkeypatch.setattr(p1_cfg, "ALERTS_LOG_FILE", res_dir / "p1_monitoring_alerts.log")
    monkeypatch.setattr(p1_cfg, "HEARTBEAT_FILE", res_dir / "p1_run_heartbeat.csv")

    monkeypatch.setattr(p1_pap, "PAPER_LOG_FILE", res_dir / "paper_trading_log.csv")
    monkeypatch.setattr(p1_pap, "SHADOW_SIGNALS_FILE", res_dir / "shadow_signals.jsonl")
    monkeypatch.setattr(p1_pap, "PAPER_STATE_FILE", res_dir / "p1_paper_state.json")
    monkeypatch.setattr(p1_pap, "HEARTBEAT_FILE", res_dir / "p1_run_heartbeat.csv")

    monkeypatch.setattr(p1_mon, "ALERTS_LOG_FILE", res_dir / "p1_monitoring_alerts.log")
    monkeypatch.setattr(p1_par, "SIGNAL_PARITY_FILE", res_dir / "signal_parity_log.csv")

    p1_pap.init_paper_log()


def _make_dummy_df(n_bars: int = 50, start_price: float = 100.0, start_date: str = "2026-05-15") -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=n_bars, freq="D")
    prices = np.linspace(start_price, start_price * 1.2, n_bars)
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices * 1.005,
        "volume": np.full(n_bars, 10000.0),
        "ema8": prices * 1.02,
        "ema21": prices * 1.01,
        "ema50": prices * 0.99,
        "ema200": prices * 0.95,
        "rsi": np.full(n_bars, 70.0),
        "adx": np.full(n_bars, 30.0),
        "di_p": np.full(n_bars, 25.0),
        "di_n": np.full(n_bars, 15.0),
        "cmf": np.full(n_bars, 0.10),
        "rel_vol": np.full(n_bars, 1.5),
        "stochrsi": np.full(n_bars, 25.0),
        "macd": np.full(n_bars, 1.0),
        "macd_sig": np.full(n_bars, 0.5),
        "macd_prev": np.full(n_bars, 0.4),
        "macd_sprev": np.full(n_bars, 0.5),
        "bb_mid": prices * 0.99,
        "at_alpha_trend": prices * 1.02,
        "at_alpha_trend_2": prices * 0.98,
        "at_buy_confirmed": np.full(n_bars, False),
    }, index=dates)
    return df


# ── (a) ÖNCELİK VE SLOT KURALI TESTİ ────────────────────────────────────────

def test_priority_and_slot_rule():
    """Öncelik: GTD (1) > ALPHA_B_SIMPLE (2) > ZT3 (3), RelVol azalan, maks. 5 slot."""
    dt_sig = pd.to_datetime("2026-06-01")
    dt_open = pd.to_datetime("2026-06-02")

    # 7 farklı hisse hazırlayalım (slot tavanı 5 olduğu için 2'si reddedilmeli)
    symbol_data = {}
    for i in range(7):
        sym = f"SYM{i+1}"
        df = _make_dummy_df(30, start_price=100.0)
        symbol_data[sym] = df

    # Mock sinyalleri:
    # SYM1: GTD, RelVol = 2.0
    # SYM2: GTD, RelVol = 1.5
    # SYM3: ALPHA_B_SIMPLE, RelVol = 3.0
    # SYM4: ALPHA_B_SIMPLE, RelVol = 1.2
    # SYM5: ZT3, RelVol = 4.0
    # SYM6: ZT3, RelVol = 1.8  -> Kapasite reddi olmalı
    # SYM7: ZT3, RelVol = 1.1  -> Kapasite reddi olmalı
    cands = [
        {"symbol": "SYM1", "signal_date": "2026-06-01", "signal_close": 100.0, "rel_vol": 2.0, "primary_agent": "GTD", "all_agents": ["GTD"], "ajan_bazli_rel_vol_sirasi": 1, "xu100_rejim_boga": True},
        {"symbol": "SYM2", "signal_date": "2026-06-01", "signal_close": 100.0, "rel_vol": 1.5, "primary_agent": "GTD", "all_agents": ["GTD"], "ajan_bazli_rel_vol_sirasi": 2, "xu100_rejim_boga": True},
        {"symbol": "SYM3", "signal_date": "2026-06-01", "signal_close": 100.0, "rel_vol": 3.0, "primary_agent": "ALPHA_B_SIMPLE", "all_agents": ["ALPHA_B_SIMPLE"], "ajan_bazli_rel_vol_sirasi": 1, "xu100_rejim_boga": True},
        {"symbol": "SYM4", "signal_date": "2026-06-01", "signal_close": 100.0, "rel_vol": 1.2, "primary_agent": "ALPHA_B_SIMPLE", "all_agents": ["ALPHA_B_SIMPLE"], "ajan_bazli_rel_vol_sirasi": 2, "xu100_rejim_boga": True},
        {"symbol": "SYM5", "signal_date": "2026-06-01", "signal_close": 100.0, "rel_vol": 4.0, "primary_agent": "ZT3", "all_agents": ["ZT3"], "ajan_bazli_rel_vol_sirasi": 1, "xu100_rejim_boga": True},
        {"symbol": "SYM6", "signal_date": "2026-06-01", "signal_close": 100.0, "rel_vol": 1.8, "primary_agent": "ZT3", "all_agents": ["ZT3"], "ajan_bazli_rel_vol_sirasi": 2, "xu100_rejim_boga": True},
        {"symbol": "SYM7", "signal_date": "2026-06-01", "signal_close": 100.0, "rel_vol": 1.1, "primary_agent": "ZT3", "all_agents": ["ZT3"], "ajan_bazli_rel_vol_sirasi": 3, "xu100_rejim_boga": True},
    ]

    state = p1_pap.load_paper_state()
    state["pending_candidates"] = cands
    p1_pap.save_paper_state(state)

    open_prices = {f"SYM{i+1}": 100.0 for i in range(7)}
    res = p1_pap.run_phase_acilis(as_of_date=dt_open, open_prices=open_prices, symbol_data=symbol_data)

    assert res["opened"] == 5
    assert res["rejected"] == 2

    st_after = p1_pap.load_paper_state()
    pos_open = st_after["positions_20k"]
    assert len(pos_open) == 5

    # Alınan ilk 5: SYM1, SYM2 (GTD), SYM3, SYM4 (ALPHA_B_SIMPLE), SYM5 (ZT3)
    assert "SYM1" in pos_open
    assert "SYM2" in pos_open
    assert "SYM3" in pos_open
    assert "SYM4" in pos_open
    assert "SYM5" in pos_open

    # Kapasite reddi olanlar: SYM6, SYM7
    assert "SYM6" not in pos_open
    assert "SYM7" not in pos_open

    # Log tablosunu kontrol et
    log_rows = p1_pap.load_paper_log()
    assert len(log_rows) == 7
    status_map = {r["sembol"]: r["kapasite_durumu"] for r in log_rows}
    assert status_map["SYM1"] == "ISLEME_ALINDI"
    assert status_map["SYM6"] == "KAPASITE_REDDI"
    assert status_map["SYM7"] == "KAPASITE_REDDI"


# ── (b) LOOKAHEAD YOKLUĞU TESTİ ──────────────────────────────────────────────

def test_no_lookahead_and_timing():
    """Sinyal günü t kapanışta üretilir, t+1 açılışında girilir; gelecek bar geçmişi etkilemez."""
    dt_sig = pd.to_datetime("2026-06-01")
    dt_open = pd.to_datetime("2026-06-02")

    df1 = _make_dummy_df(30, start_price=100.0)
    # 15. günde GTD sinyali oluşturalım
    df1.loc[dt_sig, "rsi"] = 72.0
    df1.loc[dt_sig, "adx"] = 28.0
    df1.loc[dt_sig, "di_p"] = 30.0
    df1.loc[dt_sig, "di_n"] = 10.0
    df1.loc[dt_sig, "cmf"] = 0.15
    df1.loc[dt_sig, "rel_vol"] = 1.8

    sym_data1 = {"THYAO": df1}
    sigs1, cands1 = p1_pap.generate_bot_signals_for_day(sym_data1, dt_sig)
    assert len(cands1) == 1
    assert cands1[0]["symbol"] == "THYAO"

    # Gelecek barın (dt_open) fiyatını radikal şekilde değiştirelim
    df2 = df1.copy()
    df2.loc[dt_open, "open"] = 999.0
    df2.loc[dt_open, "close"] = 999.0
    sym_data2 = {"THYAO": df2}

    sigs2, cands2 = p1_pap.generate_bot_signals_for_day(sym_data2, dt_sig)
    assert sigs1 == sigs2
    assert cands1[0]["signal_close"] == cands2[0]["signal_close"]


# ── (c) LOG IDEMPOTENCY TESTİ ────────────────────────────────────────────────

def test_log_and_phase_idempotency():
    """Aynı seans/gün için fonksiyon iki kez çağrıldığında mükerrer kayıt oluşmaz."""
    dt_sig = pd.to_datetime("2026-06-01")
    dt_open = pd.to_datetime("2026-06-02")

    df = _make_dummy_df(30, start_price=100.0)
    df.loc[dt_sig, "rsi"] = 70.0
    df.loc[dt_sig, "adx"] = 30.0
    df.loc[dt_sig, "di_p"] = 25.0
    df.loc[dt_sig, "di_n"] = 15.0
    df.loc[dt_sig, "cmf"] = 0.10
    df.loc[dt_sig, "rel_vol"] = 1.5

    sym_data = {"ASELS": df}

    # Phase Aksam 1. kez çalıştır
    res1 = p1_pap.run_phase_aksam(as_of_date=dt_sig, symbol_data=sym_data)
    assert res1["status"] == "success"

    # Phase Aksam 2. kez çalıştır (idempotent atlamalı)
    res2 = p1_pap.run_phase_aksam(as_of_date=dt_sig, symbol_data=sym_data)
    assert res2["status"] == "skipped_idempotent"

    open_p = float(df.loc[dt_sig, "close"])
    # Phase Acilis 1. kez çalıştır
    res_open1 = p1_pap.run_phase_acilis(as_of_date=dt_open, open_prices={"ASELS": open_p}, symbol_data=sym_data)
    assert res_open1["status"] == "success"
    assert res_open1["opened"] == 1

    log1 = p1_pap.load_paper_log()
    assert len(log1) == 1

    # Phase Acilis 2. kez çalıştır (idempotent atlamalı)
    res_open2 = p1_pap.run_phase_acilis(as_of_date=dt_open, open_prices={"ASELS": open_p}, symbol_data=sym_data)
    assert res_open2["status"] == "skipped_idempotent"

    log2 = p1_pap.load_paper_log()
    assert len(log2) == 1  # Tekrar satır eklenmedi!


# ── (d) MALİYET KADEME VE DRAWDOWN EŞİKLERİ TESTİ ────────────────────────────

def test_cost_tier_thresholds_and_drawdown():
    """Yeşil <= %1.20, Sarı %1.20-%1.50, Kırmızı > %1.50 ve DD uyarı eşikleri."""
    # Maliyet Kademeleri
    tier_g, _ = p1_mon.evaluate_cost_tier(1.10)
    assert tier_g == "YESIL"

    tier_edge, _ = p1_mon.evaluate_cost_tier(1.20)
    assert tier_edge == "YESIL"

    tier_y, _ = p1_mon.evaluate_cost_tier(1.35)
    assert tier_y == "SARI"

    tier_r, _ = p1_mon.evaluate_cost_tier(1.55)
    assert tier_r == "KIRMIZI"

    # Hareketli ortalama dolum maliyeti kontrolü (n >= 20)
    trades_cheap = [{"dolum_vekili_tur_basi_maliyet_pct": 1.15} for _ in range(25)]
    alert_c = p1_mon.check_fill_cost_alerts(trades_cheap, window=20)
    assert alert_c is not None
    assert alert_c["tier"] == "YESIL"

    trades_expensive = [{"dolum_vekili_tur_basi_maliyet_pct": 1.65} for _ in range(25)]
    alert_e = p1_mon.check_fill_cost_alerts(trades_expensive, window=20)
    assert alert_e is not None
    assert alert_e["tier"] == "KIRMIZI"

    # Drawdown Eşikleri (-10%, -15%, -20%)
    dd_alerts_low = p1_mon.check_drawdown_alerts(current_equity=95_000.0, peak_equity=100_000.0)
    assert len(dd_alerts_low) == 0  # -%5 henüz eşiğe ulaşmadı

    dd_alerts_10 = p1_mon.check_drawdown_alerts(current_equity=89_000.0, peak_equity=100_000.0)
    assert len(dd_alerts_10) == 1
    assert dd_alerts_10[0]["threshold_pct"] == -10.0

    dd_alerts_20 = p1_mon.check_drawdown_alerts(current_equity=78_000.0, peak_equity=100_000.0)
    assert len(dd_alerts_20) == 3  # -10, -15 ve -20 üçü birden tetiklenir


# ── (e) AJAN FİLTRE TANIMLARI TESTİ ──────────────────────────────────────────

def test_agent_filter_definitions():
    """ALPHA_B_SIMPLE'da RSI/ADX yok; GT sermaye almaz (sadece shadow log)."""
    df = _make_dummy_df(30, start_price=100.0)
    dt = df.index[-1]

    # ALPHA_B_SIMPLE koşulları: AlphaTrend buy + CMF > -0.05 + RelVol >= 1.0
    # RSI ve ADX çok düşük olsa bile tetiklenmeli!
    df.loc[dt, "at_buy_confirmed"] = True
    df.loc[dt, "cmf"] = 0.05
    df.loc[dt, "rel_vol"] = 1.3
    df.loc[dt, "rsi"] = 30.0   # Normalde < 45 elerdi
    df.loc[dt, "adx"] = 10.0   # Normalde < 18 elerdi

    sig_simple = evaluate_custom_agent_signals(df, "ALPHA_B_SIMPLE")
    assert bool(sig_simple.loc[dt]) is True

    # Eski ALPHA_B ise RSI/ADX düşük olduğu için elenmeli
    sig_old = evaluate_custom_agent_signals(df, "ALPHA_B")
    assert bool(sig_old.loc[dt]) is False

    # GT (GT_NO_RSI): Sadece shadow log dosyasına yazılır, portföy adayı olmaz!
    df.loc[dt, "rsi"] = 70.0
    df.loc[dt, "adx"] = 25.0
    sym_data = {"EREGL": df}

    bot_sigs, cands = p1_pap.generate_bot_signals_for_day(sym_data, dt)

    # Shadow signals dosyasına yazılmış olmalı
    assert p1_cfg.SHADOW_SIGNALS_FILE.exists()
    lines = p1_cfg.SHADOW_SIGNALS_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    sh_record = json.loads(lines[-1])
    assert sh_record["sembol"] == "EREGL"
    assert sh_record["ajan"] == "GT_NO_RSI"

    # Ancak cands listesinde GT yer almaz (sermaye almaz!)
    assert all(c["primary_agent"] != "GT_NO_RSI" for c in cands)


# ── (f) REDDEDİLEN SİNYALLERİN SONRADAN GETİRİSİ TESTİ ───────────────────────

def test_rejected_signals_subsequent_return_tracking():
    """Kapasite reddi alan sinyalin sanal takibi yapılır ve reddedilen_sonradan_getiri_pct doldurulur."""
    dt_open = pd.to_datetime("2026-06-02")
    dt_exit = pd.to_datetime("2026-06-05")

    # State'e kapasite reddi almış bir hisse ekleyelim
    state = p1_pap.load_paper_state()
    state["tracked_rejected_signals"] = [
        {
            "symbol": "GARAN",
            "signal_date": "2026-06-01",
            "entry_date": "2026-06-02",
            "entry_price": 100.0,
            "peak_price": 100.0,
            "tp1_done": False,
            "primary_agent": "GTD",
            "reason": "KAPASITE_REDDI",
        }
    ]
    p1_pap.save_paper_state(state)

    # Log tablosuna başlangıç satırını yaz
    initial_log = [
        {
            "sinyal_tarihi": "2026-06-01", "giris_tarihi": "2026-06-02", "sembol": "GARAN",
            "birincil_ajan": "GTD", "kapasite_durumu": "KAPASITE_REDDI", "reddedilen_sonradan_getiri_pct": "",
        }
    ]
    p1_pap.save_paper_log(initial_log)

    # dt_exit gününde STOP tetiklensin (low 94.0 <= 95.0)
    current_bars = {
        "GARAN": {"open": 98.0, "high": 99.0, "low": 94.0, "close": 94.5}
    }

    p1_pap.run_phase_takip(as_of_date=dt_exit, current_bars=current_bars, symbol_data={})

    # Log tablosundaki reddedilen satır güncellenmiş olmalı
    updated_log = p1_pap.load_paper_log()
    assert len(updated_log) == 1
    sr_val = updated_log[0]["reddedilen_sonradan_getiri_pct"]
    assert sr_val != ""
    assert float(sr_val) < -5.0  # Stop zararı


# ── (g) 60-SEANSLIK REPLAY TESTİ ─────────────────────────────────────────────

def test_60_session_replay_parity():
    """60 seanslık replay testi en az %95 uyum sağlamalıdır."""
    res = p1_par.run_60_session_replay(n_sessions=60)
    assert res["status"] == "PASSED"
    assert res["overall_parity_pct"] >= 95.0
    assert res["n_sessions"] == 60


# ── (h) 10K EŞDEĞER PNL VE MTM EQUITY TESTİ ─────────────────────────────────

def test_10k_equivalent_pnl_and_equity():
    """net_pnl_10k_esdegeri_tl tam olarak net_pnl_tl'nin yarısı olmalıdır."""
    state = p1_pap.load_paper_state()
    state["positions_20k"]["THYAO"] = {
        "pos_id": "POS_THYAO", "symbol": "THYAO", "entry_date": "2026-06-02", "signal_date": "2026-06-01",
        "entry_price": 100.0, "lots": 200, "initial_lots": 200, "total_cost": 20000.0,
        "net_sales_income": 0.0, "peak_price": 100.0, "tp1_done": False,
    }
    state["positions_10k"]["THYAO"] = {
        "pos_id": "POS_THYAO", "symbol": "THYAO", "entry_date": "2026-06-02", "signal_date": "2026-06-01",
        "entry_price": 100.0, "lots": 100, "initial_lots": 100, "total_cost": 10000.0,
        "net_sales_income": 0.0, "peak_price": 100.0, "tp1_done": False,
    }
    p1_pap.save_paper_state(state)

    initial_log = [
        {
            "sinyal_tarihi": "2026-06-01", "giris_tarihi": "2026-06-02", "sembol": "THYAO",
            "birincil_ajan": "GTD", "kapasite_durumu": "ISLEME_ALINDI", "net_pnl_tl": "",
            "net_pnl_10k_esdegeri_tl": "",
        }
    ]
    p1_pap.save_paper_log(initial_log)

    # 10 gün sonra MAX_GUN çıkışı (fiyat 104.5, TP1 veya STOP tetiklenmez)
    dt_exit = pd.to_datetime("2026-06-15")
    current_bars = {
        "THYAO": {"open": 104.0, "high": 105.0, "low": 101.0, "close": 104.5}
    }
    p1_pap.run_phase_takip(as_of_date=dt_exit, current_bars=current_bars, symbol_data={})

    updated_log = p1_pap.load_paper_log()
    trade_row = updated_log[0]
    pnl_20k = float(trade_row["net_pnl_tl"])
    pnl_10k = float(trade_row["net_pnl_10k_esdegeri_tl"])
    assert round(pnl_10k, 2) == round(pnl_20k / 2.0, 2)


# ── (i) ÇIKIŞ SABİTLERİ TEK KAYNAK ──────────────────────────────────────────

def test_exit_rules_single_source_identity():
    import research_p1.p1_exit_rules as src
    import research_p1.backtest_engine as bt_eng
    import research_p1.run_p1_revision_backtest as bt_rev

    assert p1_cfg.STOP_PCT is src.STOP_PCT
    assert p1_cfg.TP1_PCT is src.TP1_PCT
    assert p1_cfg.TRAILING_PCT is src.TRAILING_PCT
    assert p1_cfg.MAX_HOLDING_DAYS is src.MAX_HOLDING_DAYS
    assert p1_cfg.GAP_TOLERANCE is src.GAP_TOLERANCE
    assert p1_pap.STOP_PCT is src.STOP_PCT
    assert p1_pap.MAX_HOLDING_DAYS is src.MAX_HOLDING_DAYS
    assert bt_eng.GAP_TOLERANCE is src.GAP_TOLERANCE
    assert bt_eng.MAX_HOLDING_DAYS is src.MAX_HOLDING_DAYS
    assert bt_rev.GAP_TOLERANCE is src.GAP_TOLERANCE
    assert bt_rev.stop_level_price is src.stop_level_price
    assert p1_cfg.tp1_sold_lots is src.tp1_sold_lots
    assert src.STOP_PCT == -0.05
    assert src.TP1_PCT == 0.08
    assert src.TRAILING_PCT == -0.05
    assert src.MAX_HOLDING_DAYS == 10
    assert src.GAP_TOLERANCE == 0.0001


def test_gap_rule_is_ratio_not_percent():
    """gap = open/close - 1.0 (oran). 0.0001 = 0.01 yüzde puanı. Paper = backtest."""
    import research_p1.p1_exit_rules as src

    assert p1_cfg.GAP_TOLERANCE == 0.0001
    assert src.gap_ratio(100.01, 100.0) == pytest.approx(0.0001)
    assert src.gap_rejected(100.01, 100.0) is False  # abs(gap) > 0.0001 değil
    assert src.gap_rejected(100.02, 100.0) is True
    assert src.gap_rejected(99.98, 100.0) is True
    # yüzde gibi yorumlansaydı 0.0001% olurdu; 100.02 reddi bunu çürütür
    paper_gap = p1_cfg.gap_ratio(100.02, 100.0)
    bt_gap = src.gap_ratio(100.02, 100.0)
    assert paper_gap == bt_gap
    assert p1_cfg.gap_rejected is src.gap_rejected


def test_tp1_integer_lot_rounding():
    """BIST tam lot: 1→1, 2→1, 3→1, 5→2. Kesirli yarı yok."""
    from research_p1.p1_exit_rules import tp1_sold_lots

    assert tp1_sold_lots(1) == 1
    assert tp1_sold_lots(2) == 1
    assert tp1_sold_lots(3) == 1
    assert tp1_sold_lots(4) == 2
    assert tp1_sold_lots(5) == 2
    assert tp1_sold_lots(0) == 0


def test_tp1_lot_rounding_in_paper_engine():
    """Paper TP1 tek lot/2 lot/tek sayı lot tam sayı keser."""
    dt_exit = pd.to_datetime("2026-06-03")
    cases = [(1, 0, "TP1_FULL"), (2, 1, None), (3, 2, None)]
    for lots, remaining, full_reason in cases:
        state = p1_pap.load_paper_state()
        state["positions_20k"] = {
            "THYAO": {
                "pos_id": "P", "symbol": "THYAO", "entry_date": "2026-06-02",
                "signal_date": "2026-06-01", "entry_price": 100.0, "lots": lots,
                "initial_lots": lots, "total_cost": 100.0 * lots,
                "net_sales_income": 0.0, "peak_price": 100.0, "tp1_done": False,
            }
        }
        state["positions_10k"] = {
            "THYAO": {
                "pos_id": "P", "lots": lots, "total_cost": 50.0 * lots,
                "net_sales_income": 0.0, "peak_price": 100.0, "entry_price": 100.0,
            }
        }
        p1_pap.save_paper_state(state)
        p1_pap.save_paper_log([{
            "sinyal_tarihi": "2026-06-01", "giris_tarihi": "2026-06-02", "sembol": "THYAO",
            "birincil_ajan": "GTD", "kapasite_durumu": "ISLEME_ALINDI",
        }])
        bars = {"THYAO": {"open": 108.0, "high": 109.0, "low": 107.0, "close": 108.5}}
        p1_pap.run_phase_takip(as_of_date=dt_exit, current_bars=bars, symbol_data={})
        st = p1_pap.load_paper_state()
        if remaining == 0:
            assert "THYAO" not in st["positions_20k"]
            log_row = p1_pap.load_paper_log()[0]
            assert log_row["cikis_nedeni"] == full_reason
        else:
            assert st["positions_20k"]["THYAO"]["lots"] == remaining
            assert st["positions_20k"]["THYAO"]["tp1_done"] is True


def test_cost_alerts_use_only_measured_proxy():
    """Simüle %1.20 ve boş VWAP kademeye girmez; n>=20 ölçülen şart."""
    empty = [{"dolum_vekili_tur_basi_maliyet_pct": "", "simule_dolum_fiyati": 100.3} for _ in range(30)]
    assert p1_mon.check_fill_cost_alerts(empty, window=20) is None

    mixed = (
        [{"dolum_vekili_tur_basi_maliyet_pct": ""} for _ in range(10)]
        + [{"dolum_vekili_tur_basi_maliyet_pct": 1.65} for _ in range(20)]
    )
    alert = p1_mon.check_fill_cost_alerts(mixed, window=20)
    assert alert is not None
    assert alert["tier"] == "KIRMIZI"
    assert alert["average_cost_pct"] == pytest.approx(1.65)

    nineteen = [{"dolum_vekili_tur_basi_maliyet_pct": 1.65} for _ in range(19)]
    nineteen += [{"dolum_vekili_tur_basi_maliyet_pct": ""} for _ in range(20)]
    assert p1_mon.check_fill_cost_alerts(nineteen, window=20) is None


def test_typical_price_vwap_and_separate_simulated_fill():
    from p1_data_provider import opening_vwap, TYPICAL_PRICE_DEFINITION

    assert TYPICAL_PRICE_DEFINITION == "(high + low + close) / 3"
    mdf = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.5] * 5,
        "volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    })
    # typical = (101+99+100.5)/3 = 100.1666...
    vwap = opening_vwap(mdf, n_bars=5)
    assert vwap == pytest.approx((101.0 + 99.0 + 100.5) / 3.0)

    dt_sig = pd.to_datetime("2026-06-01")
    dt_open = pd.to_datetime("2026-06-02")
    df = _make_dummy_df(30, start_price=100.0)
    state = p1_pap.load_paper_state()
    state["pending_candidates"] = [{
        "symbol": "ASELS", "signal_date": "2026-06-01", "signal_close": 100.0,
        "rel_vol": 1.5, "primary_agent": "GTD", "all_agents": ["GTD"],
        "ajan_bazli_rel_vol_sirasi": 1, "xu100_rejim_boga": True,
    }]
    p1_pap.save_paper_state(state)
    res = p1_pap.run_phase_acilis(
        as_of_date=dt_open,
        open_prices={"ASELS": 100.0},
        minute_bars={"ASELS": mdf},
        symbol_data={"ASELS": df},
        fetch_vwap=False,
    )
    assert res["opened"] == 1
    row = p1_pap.load_paper_log()[0]
    assert float(row["simule_dolum_fiyati"]) == pytest.approx(100.0 * 1.003, rel=1e-6)
    assert float(row["gerceklesen_slipaj_pct"]) == pytest.approx((vwap - 100.0) / 100.0 * 100.0, abs=1e-4)
    assert row["dolum_vekili_tur_basi_maliyet_pct"] != ""
    assert abs(float(row["gerceklesen_slipaj_pct"])) != pytest.approx(0.30)


def test_vwap_missing_sets_notes_and_alert():
    from p1_data_provider import MinuteBarProvider

    class EmptyProvider(MinuteBarProvider):
        name = "empty"

        def fetch_opening_1m_bars(self, symbol, session_day, n_bars=5):
            return None

    import p1_data_provider as prov
    prov.set_default_provider(EmptyProvider())
    dt_open = pd.to_datetime("2026-06-02")
    df = _make_dummy_df(30, start_price=100.0)
    state = p1_pap.load_paper_state()
    state["pending_candidates"] = [{
        "symbol": "GARAN", "signal_date": "2026-06-01", "signal_close": 100.0,
        "rel_vol": 1.2, "primary_agent": "GTD", "all_agents": ["GTD"],
        "ajan_bazli_rel_vol_sirasi": 1, "xu100_rejim_boga": True,
    }]
    p1_pap.save_paper_state(state)
    p1_pap.run_phase_acilis(
        as_of_date=dt_open,
        open_prices={"GARAN": 100.0},
        symbol_data={"GARAN": df},
        fetch_vwap=True,
    )
    row = p1_pap.load_paper_log()[0]
    assert row["notlar"] == "VWAP verisi yok"
    assert row["gerceklesen_slipaj_pct"] == ""
    assert row["dolum_vekili_tur_basi_maliyet_pct"] == ""
    alerts = p1_cfg.ALERTS_LOG_FILE.read_text(encoding="utf-8")
    assert "VWAP_VERISI_YOK" in alerts
    prov.set_default_provider(None)


def test_heartbeat_and_missed_day_no_catchup():
    dt = pd.to_datetime("2026-06-01")
    df = _make_dummy_df(30, start_price=100.0)
    p1_pap.run_phase_aksam(as_of_date=dt, symbol_data={"ASELS": df})
    rows = p1_pap.load_heartbeat_rows()
    assert any(r["faz"] == "aksam" and r["durum"] == "success" for r in rows)

    state = p1_pap.load_paper_state()
    state["paper_start_date"] = "2026-06-01"
    p1_pap.save_paper_state(state)
    missed = p1_pap.check_missed_trading_days(date(2026, 6, 4))
    assert any(m["durum"] == "KACIRILAN_GUN" for m in missed)
    hb = p1_pap.load_heartbeat_rows()
    assert any(r["durum"] == "KACIRILAN_GUN" for r in hb)


def test_trade_level_replay_records_mismatches(tmp_path, monkeypatch):
    """Paper vs simulate_priority_portfolio karşılaştırma CSV üretir; eşleşmezse kök neden yazar."""
    from research_p1 import run_trade_level_parity as par

    monkeypatch.setattr(par, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(par, "TRADE_LEVEL_PARITY_FILE", tmp_path / "trade_level_parity.csv")
    syn = par._synthetic_universe(n_sessions=40, n_symbols=4)
    dates = sorted(list(set().union(*[df.index for df in syn.values()])))
    start, end = str(dates[20].date()), str(dates[-1].date())
    cmp_df, summary, _ = par._run_pair(syn, "unit", start, end)
    assert "bt_n" in summary
    assert "paper_n" in summary
    assert "kok_neden" in cmp_df.columns
    assert summary["maliyet_konvansiyonu"]["tur_basi_pct"] == 1.20
    # Lot formülü farkı kök neden olarak raporlanabilir; test eşleşmeyi zorlamaz.
    if len(cmp_df) and not bool(cmp_df["eslesme"].all()):
        assert cmp_df.loc[~cmp_df["eslesme"].astype(bool), "kok_neden"].str.len().gt(0).all()


def test_day_loop_calls_three_phases_and_finishes(tmp_path, monkeypatch):
    """Gün-gün döngü run_phase_takip/acilis/aksam çağırır ve sentetik aralıkta bitirir."""
    from research_p1 import run_day_loop_parity as dl
    from research_p1 import run_trade_level_parity as par

    monkeypatch.setattr(dl, "OUT_DIR", tmp_path / "loop")
    monkeypatch.setattr(dl, "PROGRESS_FILE", tmp_path / "loop" / "progress.json")
    monkeypatch.setattr(dl, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(dl, "CMP_FILE", tmp_path / "day_loop_parity.csv")
    monkeypatch.setattr(dl, "SUMMARY_FILE", tmp_path / "day_loop_parity_summary.json")
    syn = par._synthetic_universe(n_sessions=25, n_symbols=4)
    dates = sorted(list(set().union(*[df.index for df in syn.values()])))
    start, end = str(dates[8].date()), str(dates[18].date())
    monkeypatch.setattr(dl, "FULL_START", start)
    monkeypatch.setattr(dl, "FULL_END", end)
    monkeypatch.setattr(dl, "YEAR_CHUNKS", [("u", start, end)])
    out = dl.run_day_loop(syn, start, end)
    assert out["failed"] is None
    assert out["n_days"] == 11
    assert out["chunks"][0]["status"] == "ok"

