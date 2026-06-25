"""
scanner_dsp.py — P3-DSP Sembol Tarayıcı
=========================================
P2-SMC'nin scanner_smc.py'sine paralel yapı.

Görev:
  1. Sembol evrenini (181 sembol) tara
  2. Her sembol için DSP-XOVER sinyali üret
  3. Güç skorunu hesapla (crossover marjı + momentum)
  4. Top-N long adayını döndür

Veri bağımlılığı:
  - fetch_prices(symbol, lookback_days) → np.ndarray
  - Bu fonksiyon P2-SMC'nin mevcut veri pipeline'ına ince adaptör —
    VM'de dsp_data_adapter.py içinden import edilir.
  - Burada MockDataAdapter (test/geliştirme) ve gerçek adaptör için
    aynı arayüz tanımlanır.

Çalıştır (VM'de):
  python scanner_dsp.py                  # stdout özet
  python scanner_dsp.py --top 10         # top 10 sembol
  python scanner_dsp.py --symbol GARAN   # tek sembol debug
"""

from __future__ import annotations

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol, Optional
import sys
import os

# Proje kök dizini (VM'de ~/bist_ai/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dsp_core import (
    design_butterworth,
    apply_causal,
    warmup_length,
    AdaptiveFilterState,
)
from dsp_strategies import (
    XoverParams,
    XoverBacktest,
    generate_xover_signal,
)
from dsp_trend import (
    TrendParams,
    generate_trend_signal,
)

# DSP-TREND parametreleri (Faz 4 walk-forward kazananı: p40_s10)
TREND_PARAMS = TrendParams(period=40, slope_window=10, order=3)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sembol Evreni
# ---------------------------------------------------------------------------

def load_symbols(path: str = None) -> list[str]:
    """
    hisse_listesi.txt dosyasından sembol listesi yükler.

    Dosya formatı:
      - Her satırda bir sembol (AKBNK, GARAN vb. — .IS olmadan)
      - # ile başlayan satırlar yorum, atlanır
      - Boş satırlar atlanır
      - Hisse ekle: yeni satır ekle
      - Hisse çıkar: satır başına # koy veya sil

    Dosya bulunamazsa dahili yedek 20 sembolük liste kullanılır.
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, "hisse_listesi.txt")

    if not os.path.exists(path):
        logger.warning(f"Sembol listesi bulunamadı: {path} — yedek liste kullanılıyor")
        return _FALLBACK_SYMBOLS

    symbols = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            symbols.append(line.upper())

    if not symbols:
        return _FALLBACK_SYMBOLS

    return symbols


_FALLBACK_SYMBOLS = [
    "GARAN", "AKBNK", "YKBNK", "HALKB", "VAKBN",
    "THYAO", "PGSUS", "TUPRS", "EREGL", "KRDMD",
    "ASELS", "AEFES", "BIMAS", "MGROS", "SOKM",
    "KCHOL", "SAHOL", "SISE",  "PETKM", "TOASO",
]

SYMBOL_UNIVERSE_SAMPLE = load_symbols()

# ---------------------------------------------------------------------------
# Sektör Haritası — Faz 4 Korelasyon Analizi Bulgusu
# ---------------------------------------------------------------------------
# YKBNK ↔ AKBNK korelasyonu +0.92 → bankacılık içi konsantrasyon riski
# Kural: Aynı sektörden maksimum MAX_SECTOR_POSITIONS pozisyon

MAX_SECTOR_POSITIONS = 2   # Aynı sektörden max pozisyon sayısı

SECTOR_MAP: dict[str, str] = {
    # Bankacılık
    "AKBNK": "BANKA", "ALBRK": "BANKA", "GARAN": "BANKA",
    "HALKB": "BANKA", "ICBCT": "BANKA", "ISCTR": "BANKA",
    "KLNMA": "BANKA", "SKBNK": "BANKA", "TSKB":  "BANKA",
    "VAKBN": "BANKA", "YKBNK": "BANKA",
    # Enerji
    "AKSEN": "ENERJI", "AYDEM": "ENERJI", "BIOEN": "ENERJI",
    "CWENE": "ENERJI", "ENJSA": "ENERJI", "EUPWR": "ENERJI",
    "ODAS":  "ENERJI", "ORGE":  "ENERJI", "ZOREN": "ENERJI",
    # Petrol / Kimya
    "PETKM": "PETKIM", "TUPRS": "PETKIM",
    # Sanayi / Otomotiv
    "ASUZU": "OTOMOTIV", "DOAS": "OTOMOTIV", "FROTO": "OTOMOTIV",
    "OTKAR": "OTOMOTIV", "TOASO": "OTOMOTIV", "TTRAK": "OTOMOTIV",
    # Demir Çelik
    "EREGL": "CELIK", "IZMDC": "CELIK", "KRDMD": "CELIK", "TMSN": "CELIK",
    # Savunma / Teknoloji
    "ARDYZ": "SAVUNMA", "ASELS": "SAVUNMA", "ASTOR": "SAVUNMA",
    "LOGO":  "TEKNOLOJI", "NETAS": "TEKNOLOJI",
    # Holding
    "DOHOL": "HOLDING", "GLYHO": "HOLDING", "GSDHO": "HOLDING",
    "KCHOL": "HOLDING", "POLHO": "HOLDING", "SAHOL": "HOLDING",
    "SISE":  "HOLDING",
    # Perakende / Tüketici
    "AEFES": "TUKETICI", "BIMAS": "TUKETICI", "CCOLA": "TUKETICI",
    "MGROS": "TUKETICI", "MAVI":  "TUKETICI", "SOKM":  "TUKETICI",
    "ULKER": "TUKETICI",
    # Ulaşım / Turizm
    "CLEBI": "ULASIM", "PGSUS": "ULASIM", "TAVHL": "ULASIM",
    "THYAO": "ULASIM",
    # GYO / İnşaat
    "AKSGY": "GYO", "EKGYO": "GYO", "ENKAI": "GYO",
    "ISGYO": "GYO", "MPARK": "GYO",
    # Kimya / Gübre
    "AKCNS": "KIMYA", "AKSA": "KIMYA", "ALKIM": "KIMYA", "GUBRF": "KIMYA",
    # Telekom
    "TCELL": "TELEKOM", "TTKOM": "TELEKOM",
    # Sigorta / Finans
    "AGESA": "FINANS", "ISMEN": "FINANS", "SNPAM": "FINANS",
    # Diğer
    "ARCLK": "DIGER", "BRYAT": "DIGER", "CIMSA": "DIGER",
    "GESAN": "DIGER", "HEKTS": "DIGER", "JANTS": "DIGER",
    "KARSN": "DIGER", "TKFEN": "DIGER", "VESTL": "DIGER",
}


def get_sector(symbol: str) -> str:
    """Sembol için sektör kodu döndürür. Bilinmiyorsa 'DIGER'."""
    clean = symbol.replace(".IS", "").upper()
    return SECTOR_MAP.get(clean, "DIGER")

class DataAdapter(Protocol):
    """
    Scanner'ın beklediği veri arayüzü.
    P2-SMC'nin veri pipeline'ı bu arayüzü karşılayan bir sınıfla sarılır.
    """
    def fetch(self, symbol: str, lookback_days: int) -> Optional[np.ndarray]:
        """
        Sembol için kapanış fiyatlarını döndürür.
        Veri yoksa veya hata varsa None döner.
        """
        ...


class MockDataAdapter:
    """
    Geliştirme/test için sentetik veri üretici.
    VM'de gerçek DataAdapter ile değiştirilir.
    """
    def __init__(self, seed_base: int = 42):
        self.seed_base = seed_base

    def fetch(self, symbol: str, lookback_days: int) -> Optional[np.ndarray]:
        seed = self.seed_base + hash(symbol) % 10000
        rng = np.random.default_rng(seed)
        n = lookback_days
        vol = 0.018
        returns = []
        for i in range(n):
            shock = rng.normal(0, vol)
            vol = np.clip(0.0003 + 0.85 * vol + 0.12 * abs(shock), 0.008, 0.055)
            drift = 0.0008 * np.sin(i / 80) + rng.choice([-1, 1]) * 0.0002
            returns.append(drift + shock)
        return 10.0 * np.cumprod(1 + np.array(returns))


class RealDataAdapter:
    """
    VM'deki gerçek veri pipeline'ına adaptör.
    P2-SMC'nin mevcut fetch fonksiyonunu wrap eder.

    VM'de kullanım:
        from dsp_data_adapter import get_ohlc   # P2-SMC pipeline
        adapter = RealDataAdapter()
        prices = adapter.fetch("GARAN.IS", 300)
    """
    def fetch(self, symbol: str, lookback_days: int) -> Optional[np.ndarray]:
        try:
            # VM'de bu satır aktif edilir:
            # from dsp_data_adapter import get_close_prices
            # return get_close_prices(symbol, lookback_days)
            raise NotImplementedError(
                "RealDataAdapter: VM'de dsp_data_adapter.py import'u etkinleştir."
            )
        except Exception as e:
            logger.warning(f"{symbol} veri hatası: {e}")
            return None


# ---------------------------------------------------------------------------
# Sinyal Skoru
# ---------------------------------------------------------------------------

@dataclass
class SymbolScore:
    """
    Tek sembol için tarama çıktısı.
    """
    symbol: str
    signal: int                     # +1 long, -1 short/flat, 0 veri yok
    score: float                    # Sıralama skoru (yüksek = güçlü long)
    crossover_margin: float         # (fast - slow) / slow  — normalize marj
    momentum_3w: float              # Son 15 bar log getiri
    fast_line: float                # Son bar fast filtre değeri
    slow_line: float                # Son bar slow filtre değeri
    warmup_ok: bool                 # Yeterli veri var mı
    scan_time_ms: float
    trend_slope: float = 0.0        # Faz 4: DSP-TREND normalize eğim
    error: Optional[str] = None     # Hata mesajı (veri yoksa)

    @property
    def is_long(self) -> bool:
        return self.signal == 1 and self.warmup_ok

    @property
    def signal_str(self) -> str:
        if not self.warmup_ok:
            return "WARM-UP"
        return "LONG" if self.signal == 1 else "FLAT"


def score_symbol(
    symbol: str,
    prices: np.ndarray,
    params: XoverParams,
    trend_params: TrendParams = None,
) -> SymbolScore:
    """
    Tek sembol için DSP-XOVER + DSP-TREND birleşik skor hesapla.

    Skor bileşenleri (Faz 4 — hepsi normalize):
      1. crossover_margin : (fast - slow) / slow     → XOVER trend gücü   %50
      2. momentum_3w      : Son 15 bar log getiri    → kısa momentum       %30
      3. trend_slope      : TREND filtered eğim oranı → orta vade onay     %20

    Faz 2 formülü: 0.7 × margin + 0.3 × momentum
    Faz 4 formülü: 0.5 × margin + 0.3 × momentum + 0.2 × trend_slope

    10 sembol walk-forward sonucu: 7/10 sembolde TREND consistency üstün,
    0/10 sembolde XOVER üstün → birleşik skor entegrasyonu onaylandı.
    """
    t0 = time.perf_counter()

    prices = np.asarray(prices, dtype=float)
    tp = trend_params or TREND_PARAMS

    # --- XOVER sinyali ---
    sig = generate_xover_signal(prices, params)
    vf  = sig.valid_from

    warmup_ok   = len(prices) > vf + 5
    last_fast   = float(sig.fast_line[-1])
    last_slow   = float(sig.slow_line[-1])
    last_signal = int(sig.signal[-1])

    # Normalize crossover marjı
    crossover_margin = (last_fast - last_slow) / (last_slow + 1e-9)

    # 15 bar momentum
    if len(prices) >= 16:
        momentum_3w = float(np.log(prices[-1] / prices[-16]))
    else:
        momentum_3w = 0.0

    # --- DSP-TREND slope skoru ---
    trend_slope = 0.0
    try:
        trend_vf = warmup_ok and len(prices) > tp.slope_window + int(tp.order * tp.period) + 5
        if trend_vf:
            tsig = generate_trend_signal(prices, tp)
            trend_slope = float(tsig.slope_series[-1])
    except Exception:
        trend_slope = 0.0

    # --- Birleşik skor (Faz 4) ---
    score = (
        0.5 * crossover_margin
        + 0.3 * momentum_3w
        + 0.2 * trend_slope
    )
    # Flat sinyalde skoru negatife çek (long adaylarını üste taşı)
    if last_signal < 0:
        score -= 0.05

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return SymbolScore(
        symbol=symbol,
        signal=last_signal if warmup_ok else 0,
        score=score,
        crossover_margin=crossover_margin,
        momentum_3w=momentum_3w,
        fast_line=last_fast,
        slow_line=last_slow,
        warmup_ok=warmup_ok,
        scan_time_ms=round(elapsed_ms, 2),
        trend_slope=round(trend_slope, 4),
    )


# ---------------------------------------------------------------------------
# Ana Tarayıcı
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """
    Tüm evren taramasının çıktısı.
    """
    scan_date: date
    total_symbols: int
    scanned: int
    errors: int
    long_signals: int
    top_longs: list[SymbolScore]        # Skorla sıralı long adaylar
    all_scores: list[SymbolScore]       # Tüm semboller (debug için)
    elapsed_sec: float
    params: XoverParams

    def summary_line(self) -> str:
        return (
            f"[P3-DSP] {self.scan_date} | "
            f"Taranan: {self.scanned}/{self.total_symbols} | "
            f"Long sinyal: {self.long_signals} | "
            f"Top-5: {', '.join(s.symbol.replace('.IS','') for s in self.top_longs[:5])} | "
            f"{self.elapsed_sec:.1f}s"
        )

    def sector_summary(self) -> str:
        """Top long adaylarının sektör dağılımı."""
        from collections import Counter
        sectors = Counter(get_sector(s.symbol) for s in self.top_longs)
        return " | ".join(f"{sec}:{cnt}" for sec, cnt in sectors.most_common())


class DspScanner:
    """
    P3-DSP sembol tarayıcı.

    Kullanım:
        adapter = MockDataAdapter()          # veya RealDataAdapter()
        scanner = DspScanner(adapter)
        result  = scanner.run()
        print(result.summary_line())
        for s in result.top_longs[:10]:
            print(s)
    """

    def __init__(
        self,
        data_adapter: DataAdapter,
        params: Optional[XoverParams] = None,
        lookback_days: int = 300,
        top_n: int = 10,
        symbols: Optional[list[str]] = None,
        max_sector_positions: int = MAX_SECTOR_POSITIONS,
    ):
        self.adapter = data_adapter
        self.params  = params or XoverParams(fast_period=15, slow_period=40, order=3)
        self.lookback_days = lookback_days
        self.top_n   = top_n
        self.symbols = symbols or SYMBOL_UNIVERSE_SAMPLE
        self.max_sector_positions = max_sector_positions

    def run(self, verbose: bool = False) -> ScanResult:
        """
        Tüm sembol evrenini tara, skorla, sırala.
        """
        t0 = time.perf_counter()
        today = date.today()
        scores: list[SymbolScore] = []
        errors = 0

        for symbol in self.symbols:
            try:
                prices = self.adapter.fetch(symbol, self.lookback_days)
                if prices is None or len(prices) < 60:
                    sc = SymbolScore(
                        symbol=symbol, signal=0, score=-999,
                        crossover_margin=0, momentum_3w=0,
                        fast_line=0, slow_line=0, warmup_ok=False,
                        scan_time_ms=0, error="Yetersiz veri",
                    )
                    errors += 1
                else:
                    sc = score_symbol(symbol, prices, self.params)

                scores.append(sc)
                if verbose:
                    flag = "▲ LONG" if sc.is_long else "  flat"
                    print(f"  {flag}  {symbol:<15} skor={sc.score:+.4f}  "
                          f"marj={sc.crossover_margin:+.3f}  "
                          f"mom={sc.momentum_3w:+.3f}  "
                          f"tslope={sc.trend_slope:+.4f}")

            except Exception as e:
                logger.error(f"{symbol} tarama hatası: {e}")
                errors += 1

        # Long sinyalleri skora göre sırala
        long_scores = [s for s in scores if s.is_long]
        long_scores.sort(key=lambda x: x.score, reverse=True)
        all_sorted = sorted(scores, key=lambda x: x.score, reverse=True)

        # Sektör limiti uygula — Faz 4 korelasyon bulgusu
        # Aynı sektörden max self.max_sector_positions sembol seçilir
        sector_filtered: list[SymbolScore] = []
        sector_counts: dict[str, int] = {}
        sector_skipped: list[SymbolScore] = []

        for sc in long_scores:
            sector = get_sector(sc.symbol)
            count  = sector_counts.get(sector, 0)
            if count < self.max_sector_positions:
                sector_filtered.append(sc)
                sector_counts[sector] = count + 1
            else:
                sector_skipped.append(sc)

        if sector_skipped:
            skipped_str = ", ".join(
                f"{s.symbol}({get_sector(s.symbol)})" for s in sector_skipped[:5]
            )
            logger.info(f"Sektör limiti ({self.max_sector_positions}) nedeniyle atlandı: {skipped_str}")

        return ScanResult(
            scan_date=today,
            total_symbols=len(self.symbols),
            scanned=len(self.symbols) - errors,
            errors=errors,
            long_signals=len(long_scores),
            top_longs=sector_filtered[:self.top_n],
            all_scores=all_sorted,
            elapsed_sec=round(time.perf_counter() - t0, 2),
            params=self.params,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="P3-DSP Sembol Tarayıcı")
    parser.add_argument("--top",    type=int, default=10,   help="Top N long adayı")
    parser.add_argument("--symbol", type=str, default=None, help="Tek sembol debug")
    parser.add_argument("--real",   action="store_true",    help="Gerçek veri adaptörü")
    args = parser.parse_args()

    adapter = RealDataAdapter() if args.real else MockDataAdapter()
    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = load_symbols()

    scanner = DspScanner(
        data_adapter=adapter,
        top_n=args.top,
        symbols=symbols,
        # Faz 4 sonucu: XOVER(15/40) + TREND(p40/s10) birleşik skor
        params=XoverParams(fast_period=15, slow_period=40, order=3),
    )

    print(f"\nP3-DSP Tarama başlıyor... ({len(scanner.symbols)} sembol)\n")
    result = scanner.run(verbose=True)

    print(f"\n{'─'*65}")
    print(result.summary_line())
    print(f"{'─'*65}")
    print(f"\nTop {args.top} Long Adayı (Faz 4 — Birleşik Skor + Sektör Limiti):")
    print(f"  {'Sembol':<15} {'Sektör':<12} {'Skor':>8} {'Marj':>8} {'Mom':>8} {'TSlope':>8} {'Signal'}")
    print(f"  {'─'*15} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    for s in result.top_longs:
        sector = get_sector(s.symbol)
        print(f"  {s.symbol:<15} {sector:<12} {s.score:>+8.4f} "
              f"{s.crossover_margin:>+8.3f} "
              f"{s.momentum_3w:>+8.3f} "
              f"{s.trend_slope:>+8.4f}  {s.signal_str}")

    print(f"\n  Sektör dağılımı: {result.sector_summary()}")