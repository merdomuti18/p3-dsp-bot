"""
DSP-P3 FAZ 1 — Modül sözleşme testleri (contract)
=================================================
Production modüllerinin beklenen public API'leri mevcut mu?
Sadece import + varlık + imza kontrolü; hiçbir fonksiyon ÇAĞRILMAZ
(state yazma / network riski yok).
"""

from __future__ import annotations

import inspect

import pytest

# Modüller import edilir (MOTT_BASE_DIR=repo kökü, conftest'te set edildi)
import dsp_core
import meta_portfolio
import mott_aylik_rapor
import mott_fiyat
import mott_mode
import mott_performans_analiz
import mott_portfoy_deger
import mott_risk
import mott_state
import mott_telegram
import p5_committee
import portfoy_yonetici
import scanner_dsp
import scanner_smc
import simulate_dsp


def _var(mod, ad):
    assert hasattr(mod, ad), f"{mod.__name__}.{ad} eksik!"
    return getattr(mod, ad)


# ---------------------------------------------------------------------------
# dsp_core
# ---------------------------------------------------------------------------

def test_dsp_core_public_api():
    for ad in (
        "design_butterworth", "design_chebyshev", "apply_causal",
        "warmup_length", "detect_dominant_cycle", "AdaptiveFilterState",
        "FilterCoeffs", "CycleDetectionResult", "check_no_lookahead",
        "FORBIDDEN_IN_PROD", "CYCLE_PERIOD_MIN", "CYCLE_PERIOD_MAX",
    ):
        _var(dsp_core, ad)


def test_dsp_core_imzalar():
    sig = inspect.signature(dsp_core.apply_causal)
    params = list(sig.parameters)
    assert params == ["b", "a", "data"]
    sig = inspect.signature(dsp_core.warmup_length)
    assert list(sig.parameters) == ["order", "period"]
    sig = inspect.signature(dsp_core.design_butterworth)
    assert list(sig.parameters)[:2] == ["period", "order"]


# ---------------------------------------------------------------------------
# meta_portfolio (P4)
# ---------------------------------------------------------------------------

def test_meta_portfolio_public_api():
    for ad in (
        "compute_ic", "strateji_ic_hesapla", "fiyat_cek", "normalize_skor",
        "ic_agirlikli_birlestir", "half_kelly_boyut", "portfoy_guncelle",
        "yeni_pozisyon_ac", "state_yukle", "state_kaydet", "calistir", "monitor",
        "IC_PERIODS", "IC_WINDOW",
    ):
        _var(meta_portfolio, ad)


def test_meta_portfolio_ic_imzasi():
    sig = inspect.signature(meta_portfolio.compute_ic)
    assert list(sig.parameters) == ["signals", "forward_returns"]
    sig = inspect.signature(meta_portfolio.strateji_ic_hesapla)
    assert list(sig.parameters)[:3] == ["strateji", "signal_log", "fiyat_cache"]


# ---------------------------------------------------------------------------
# mott_state — kanonik normalizasyon katmanı
# ---------------------------------------------------------------------------

def test_mott_state_public_api():
    for ad in ("normalize", "hepsi", "DOSYALAR"):
        _var(mott_state, ad)


def test_mott_state_normalize_imzasi():
    sig = inspect.signature(mott_state.normalize)
    assert list(sig.parameters) == ["kod"]


# ---------------------------------------------------------------------------
# mott_mode
# ---------------------------------------------------------------------------

def test_mott_mode_public_api():
    sig = inspect.signature(mott_mode.detect_mode)
    assert list(sig.parameters) == ["now"]


# ---------------------------------------------------------------------------
# mott_risk — ortak risk omurgası
# ---------------------------------------------------------------------------

def test_mott_risk_public_api():
    for ad in (
        "cooldown_da", "kitaplar_arasi_sayi", "kitap_limiti_asildi",
        "rejim_slot_limiti", "makro_karar_oku", "COOLDOWN_GUN", "COOLDOWN_NEDENLER",
    ):
        _var(mott_risk, ad)


def test_mott_risk_cooldown_imzasi():
    sig = inspect.signature(mott_risk.cooldown_da)
    params = list(sig.parameters)
    assert params[0] == "trade_history" and "symbol" in params


# ---------------------------------------------------------------------------
# mott_telegram — gönderim sözleşmesi (import edilebilir, çağrılmaz)
# ---------------------------------------------------------------------------

def test_mott_telegram_public_api():
    for ad in (
        "get_token", "get_chat_id", "telegram_gonder", "telegram_islem_gonder",
        "MAX_TG_LEN",
    ):
        _var(mott_telegram, ad)


def test_mott_telegram_max_len_telegram_sinir_altinda():
    # Telegram 4096 karakter sınırı — güvenlik payıyla 4000
    assert mott_telegram.MAX_TG_LEN <= 4096


# ---------------------------------------------------------------------------
# scanner_dsp (P3 tarayıcı)
# ---------------------------------------------------------------------------

def test_scanner_dsp_public_api():
    for ad in (
        "load_symbols", "get_sector", "score_symbol", "DspScanner",
        "RealDataAdapter", "MockDataAdapter", "SymbolScore", "ScanResult",
        "SYMBOL_UNIVERSE_SAMPLE", "MAX_SECTOR_POSITIONS",
    ):
        _var(scanner_dsp, ad)


def test_scanner_dsp_load_symbols_imzasi():
    sig = inspect.signature(scanner_dsp.load_symbols)
    assert list(sig.parameters)[0] == "path"


def test_scanner_dsp_scanner_ve_adaptorlar_callable():
    assert callable(scanner_dsp.DspScanner)
    assert callable(scanner_dsp.RealDataAdapter)
    assert callable(scanner_dsp.MockDataAdapter)


# ---------------------------------------------------------------------------
# scanner_smc (P2 tarayıcı)
# ---------------------------------------------------------------------------

def test_scanner_smc_public_api():
    for ad in (
        "veri_hazirla", "run_scan_smc", "saatlik_teyit_smc", "calculate_verdict",
        "teyit_skoru_hesapla", "vm_gonder_p2", "format_aksam",
        "get_tv_ema200_bulk", "get_htf_trend", "send_telegram",
    ):
        _var(scanner_smc, ad)


def test_scanner_smc_veri_hazirla_imzasi():
    sig = inspect.signature(scanner_smc.veri_hazirla)
    params = list(sig.parameters)
    assert params[0] == "semboller" and params[1] == "period"


# ---------------------------------------------------------------------------
# simulate_dsp (P3 orkestratör)
# ---------------------------------------------------------------------------

def test_simulate_dsp_public_api():
    for ad in (
        "load_state", "save_state", "update_portfolio", "calc_performance",
        "monitor", "main", "build_report", "PARAMS", "POS_SIZE_PCT",
        "EMERGENCY_STOP_PCT", "MIN_ENTRY_SCORE",
    ):
        _var(simulate_dsp, ad)


def test_simulate_dsp_update_portfolio_imzasi():
    sig = inspect.signature(simulate_dsp.update_portfolio)
    assert list(sig.parameters)[:2] == ["state", "scan"]


# ---------------------------------------------------------------------------
# portfoy_yonetici (P1/P2 execution)
# ---------------------------------------------------------------------------

def test_portfoy_yonetici_public_api():
    for ad in (
        "append_jsonl", "sonraki_islem_gunu", "lgbm_model_yukle",
        "guncel_makro_skoru", "p2_portfoy_yukle", "p2_portfoy_kaydet",
        "p2_yeni_pozisyon_ac", "p2_pozisyon_kontrol", "p2_ozet_mesaji",
        "STRATEGY_WEIGHTS", "EMERGENCY_LIQUIDATION_SCORE",
    ):
        _var(portfoy_yonetici, ad)


def test_portfoy_yonetici_p2_imzasi():
    sig = inspect.signature(portfoy_yonetici.p2_yeni_pozisyon_ac)
    params = list(sig.parameters)
    assert params[:3] == ["portfoy", "adaylar", "makro_karar"]


# ---------------------------------------------------------------------------
# p5_committee (komite portföyü)
# ---------------------------------------------------------------------------

def test_p5_committee_public_api():
    for ad in (
        "komite_adaylari", "state_yukle", "state_kaydet", "portfoy_guncelle",
        "yeni_pozisyon_ac", "makro_karar", "drift_sembolleri", "p1_top_set",
        "p2_top_set", "p3_top_set", "MAX_POS", "MIN_KAYNAK", "MAX_SEKTOR",
        "STOP_PCT", "TP_PCT", "MAX_GUN",
    ):
        _var(p5_committee, ad)


def test_p5_committee_komite_adaylari_imzasi():
    sig = inspect.signature(p5_committee.komite_adaylari)
    assert list(sig.parameters) == []


# ---------------------------------------------------------------------------
# mott_fiyat (canlı fiyat katmanı)
# ---------------------------------------------------------------------------

def test_mott_fiyat_public_api():
    for ad in ("canli_fiyatlar", "canli_fiyat", "tv_fiyatlar"):
        _var(mott_fiyat, ad)


def test_mott_fiyat_imzalari():
    assert list(inspect.signature(mott_fiyat.canli_fiyatlar).parameters)[0] == "semboller"
    assert list(inspect.signature(mott_fiyat.canli_fiyat).parameters)[0] == "sym"


# ---------------------------------------------------------------------------
# mott_portfoy_deger (equity katmanı)
# ---------------------------------------------------------------------------

def test_mott_portfoy_deger_public_api():
    for ad in ("equity_hesapla", "portfoy_degeri_satiri"):
        _var(mott_portfoy_deger, ad)


def test_mott_portfoy_deger_equity_imzasi():
    sig = inspect.signature(mott_portfoy_deger.equity_hesapla)
    assert list(sig.parameters)[:2] == ["strateji", "portfoy"]


# ---------------------------------------------------------------------------
# mott_performans_analiz (performans katmanı)
# ---------------------------------------------------------------------------

def test_mott_performans_analiz_public_api():
    for ad in ("analiz_p3", "analiz_p4", "analiz_p1_p2"):
        _var(mott_performans_analiz, ad)


def test_mott_performans_analiz_imzalari():
    assert list(inspect.signature(mott_performans_analiz.analiz_p3).parameters) == []


# ---------------------------------------------------------------------------
# mott_aylik_rapor (aylık rapor katmanı)
# ---------------------------------------------------------------------------

def test_mott_aylik_rapor_public_api():
    for ad in ("rapor_olustur", "telegram_metin", "telegram_gonder_rapor"):
        _var(mott_aylik_rapor, ad)


def test_mott_aylik_rapor_rapor_imzasi():
    assert list(inspect.signature(mott_aylik_rapor.rapor_olustur).parameters) == []
