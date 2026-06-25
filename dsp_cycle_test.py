"""
dsp_cycle_test.py — P3-DSP Faz 4: DSP-CYCLE Tekrar Test
=========================================================
Faz 0 bulgusu: BIST'te evrensel cycle yok (peakiness < 2.5)
Faz 4 sorusu: 79 sembol + 3 farklı dönem ile sonuç değişir mi?

Test yapısı:
  1. 79 sembol, sektör bazında gruplu cycle analizi
  2. 3 farklı piyasa dönemi:
       - 2022: Bear market (TL krizi, yüksek volatilite)
       - 2023: Toparlanma dönemi
       - 2024-2025: Bull market
  3. Karar kriterleri:
       - peakiness > 2.5 → güvenilir cycle
       - Güvenilir sembol oranı > %30 → DSP-CYCLE standalone strateji olarak ekle
       - Güvenilir sembol oranı ≤ %30 → XOVER/TREND'e filtre olarak kullan

Çalıştır:
    python dsp_cycle_test.py
    python dsp_cycle_test.py --period 2023
    python dsp_cycle_test.py --quick   (sadece 20 sembol)
"""

from __future__ import annotations

import logging
import warnings
import numpy as np
from dataclasses import dataclass
from datetime import date
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_core import detect_dominant_cycle
from scanner_dsp import SECTOR_MAP, get_sector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test Dönemleri
# ---------------------------------------------------------------------------

TEST_PERIODS = {
    "2022": ("2022-01-01", "2022-12-31", "Bear — TL krizi, yüksek volatilite"),
    "2023": ("2023-01-01", "2023-12-31", "Toparlanma — deprem sonrası"),
    "2024": ("2024-01-01", "2024-12-31", "Bull — seçim sonrası ralli"),
    "2025": ("2025-01-01", "2025-12-31", "Güncel"),
}

# Karar eşiği
RELIABLE_THRESHOLD    = 2.5    # peakiness > bu → güvenilir cycle
STANDALONE_THRESHOLD  = 0.30   # güvenilir sembol oranı → standalone strateji
PERIOD_MIN            = 15.0
PERIOD_MAX            = 60.0


# ---------------------------------------------------------------------------
# Veri Çekici
# ---------------------------------------------------------------------------

def fetch_period_prices(
    symbol: str,
    start: str,
    end: str,
) -> Optional[np.ndarray]:
    """Belirli tarih aralığı için kapanış fiyatları."""
    try:
        import yfinance as yf
        ticker = f"{symbol}.IS" if not symbol.endswith(".IS") else symbol
        df = yf.Ticker(ticker).history(start=start, end=end)
        prices = df["Close"].dropna().values
        return prices if len(prices) >= 60 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Veri Yapıları
# ---------------------------------------------------------------------------

@dataclass
class SymbolCycleResult:
    symbol:           str
    sector:           str
    period_label:     str
    n_bars:           int
    dominant_period:  float
    peakiness:        float
    is_reliable:      bool
    top_periods:      list[float]


@dataclass
class SectorCycleSummary:
    sector:           str
    n_symbols:        int
    n_reliable:       int
    reliable_ratio:   float
    avg_peakiness:    float
    avg_period:       float
    dominant_periods: list[float]   # Güvenilir sembollerin periyotları


@dataclass
class PeriodCycleReport:
    period_label:     str
    description:      str
    start:            str
    end:              str
    total_symbols:    int
    n_reliable:       int
    reliable_ratio:   float
    avg_peakiness:    float
    sector_summaries: list[SectorCycleSummary]
    symbol_results:   list[SymbolCycleResult]
    verdict:          str   # 'STANDALONE' | 'FILTER' | 'NO_CYCLE'


# ---------------------------------------------------------------------------
# Ana Test
# ---------------------------------------------------------------------------

class DspCycleTest:
    """
    DSP-CYCLE Faz 4 tekrar test motoru.
    """

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        reliable_threshold: float = RELIABLE_THRESHOLD,
        standalone_threshold: float = STANDALONE_THRESHOLD,
    ):
        if symbols is None:
            from scanner_dsp import load_symbols
            self.symbols = load_symbols()
        else:
            self.symbols = symbols

        self.reliable_threshold   = reliable_threshold
        self.standalone_threshold = standalone_threshold

    def run_period(
        self,
        period_label: str,
        verbose: bool = True,
    ) -> PeriodCycleReport:
        """Tek dönem için tüm sembolleri test et."""
        start, end, desc = TEST_PERIODS[period_label]

        if verbose:
            print(f"\n  Dönem: {period_label} ({start} → {end})")
            print(f"  {desc}")
            print(f"  {'─'*55}")

        results: list[SymbolCycleResult] = []

        for symbol in self.symbols:
            prices = fetch_period_prices(symbol, start, end)
            if prices is None:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cycle = detect_dominant_cycle(
                        prices,
                        period_min=PERIOD_MIN,
                        period_max=PERIOD_MAX,
                    )

                r = SymbolCycleResult(
                    symbol=symbol,
                    sector=get_sector(symbol),
                    period_label=period_label,
                    n_bars=len(prices),
                    dominant_period=round(cycle.dominant_period, 1),
                    peakiness=round(cycle.peakiness, 3),
                    is_reliable=cycle.is_reliable,
                    top_periods=[round(p, 1) for p in cycle.top_periods[:3]],
                )
                results.append(r)

                if verbose:
                    flag = "🔵" if r.is_reliable else "⚪"
                    print(f"    {flag} {symbol:<10} {r.sector:<12} "
                          f"period={r.dominant_period:5.1f}g  "
                          f"peak={r.peakiness:.2f}")

            except Exception as e:
                logger.debug(f"{symbol} cycle hatası: {e}")

        return self._build_report(period_label, desc, start, end, results)

    def run_all_periods(self, verbose: bool = True) -> dict[str, PeriodCycleReport]:
        """Tüm dönemler için test."""
        reports = {}
        for label in TEST_PERIODS:
            reports[label] = self.run_period(label, verbose=verbose)
        return reports

    def _build_report(
        self,
        period_label: str,
        desc: str,
        start: str,
        end: str,
        results: list[SymbolCycleResult],
    ) -> PeriodCycleReport:

        if not results:
            return PeriodCycleReport(
                period_label=period_label, description=desc,
                start=start, end=end, total_symbols=0,
                n_reliable=0, reliable_ratio=0.0, avg_peakiness=0.0,
                sector_summaries=[], symbol_results=[],
                verdict="NO_CYCLE",
            )

        n_reliable     = sum(1 for r in results if r.is_reliable)
        reliable_ratio = n_reliable / len(results)
        avg_peakiness  = float(np.mean([r.peakiness for r in results]))

        # Sektör bazında grupla
        sector_groups: dict[str, list[SymbolCycleResult]] = {}
        for r in results:
            sector_groups.setdefault(r.sector, []).append(r)

        sector_summaries = []
        for sector, group in sorted(sector_groups.items()):
            n_rel   = sum(1 for r in group if r.is_reliable)
            rel_rat = n_rel / len(group)
            avg_pk  = float(np.mean([r.peakiness for r in group]))
            periods = [r.dominant_period for r in group if r.is_reliable]
            avg_per = float(np.mean(periods)) if periods else 0.0

            sector_summaries.append(SectorCycleSummary(
                sector=sector,
                n_symbols=len(group),
                n_reliable=n_rel,
                reliable_ratio=rel_rat,
                avg_peakiness=round(avg_pk, 3),
                avg_period=round(avg_per, 1),
                dominant_periods=sorted(periods),
            ))

        # Sektör güvenilirlik oranına göre sırala
        sector_summaries.sort(key=lambda x: x.reliable_ratio, reverse=True)

        # Karar
        if reliable_ratio > self.standalone_threshold:
            verdict = "STANDALONE"
        elif reliable_ratio > 0.10:
            verdict = "FILTER"
        else:
            verdict = "NO_CYCLE"

        return PeriodCycleReport(
            period_label=period_label,
            description=desc,
            start=start,
            end=end,
            total_symbols=len(results),
            n_reliable=n_reliable,
            reliable_ratio=round(reliable_ratio, 3),
            avg_peakiness=round(avg_peakiness, 3),
            sector_summaries=sector_summaries,
            symbol_results=results,
            verdict=verdict,
        )

    # --- Raporlama ---

    def print_period_report(self, report: PeriodCycleReport) -> None:
        verdict_icons = {
            "STANDALONE": "✅ STANDALONE — DSP-CYCLE bağımsız strateji olarak eklenebilir",
            "FILTER":     "🔄 FİLTRE — XOVER/TREND'e ek filtre olarak kullanılabilir",
            "NO_CYCLE":   "❌ CYCLE YOK — Bu dönemde anlamlı cycle tespit edilemedi",
        }

        print(f"\n  {'═'*60}")
        print(f"  Dönem: {report.period_label} | {report.description}")
        print(f"  {'═'*60}")
        print(f"  Taranan  : {report.total_symbols} sembol")
        print(f"  Güvenilir: {report.n_reliable} sembol "
              f"({report.reliable_ratio:.0%})")
        print(f"  Ort. peakiness: {report.avg_peakiness:.2f} "
              f"(eşik: {self.reliable_threshold})")
        print(f"  Karar: {verdict_icons.get(report.verdict, report.verdict)}")

        print(f"\n  Sektör Bazında Cycle Güvenilirliği:")
        print(f"  {'Sektör':<14} {'N':>4} {'Güv':>5} {'Oran':>6} "
              f"{'Ort.Peak':>9} {'Ort.Per':>8}")
        print(f"  {'─'*14} {'─'*4} {'─'*5} {'─'*6} "
              f"{'─'*9} {'─'*8}")

        for s in report.sector_summaries:
            bar = "█" * int(s.reliable_ratio * 10)
            per_str = f"{s.avg_period:.0f}g" if s.avg_period > 0 else "—"
            print(f"  {s.sector:<14} {s.n_symbols:>4} {s.n_reliable:>5} "
                  f"{s.reliable_ratio:>5.0%}  "
                  f"{s.avg_peakiness:>8.2f}  {per_str:>7}  {bar}")

    def print_cross_period_summary(
        self,
        reports: dict[str, PeriodCycleReport],
    ) -> None:
        """Tüm dönemlerin karşılaştırmalı özeti."""
        print(f"\n{'='*65}")
        print("  DSP-CYCLE ÇAPRAZ DÖNEM ÖZET")
        print(f"{'='*65}")
        print(f"  {'Dönem':<8} {'Sembol':>7} {'Güvenilir':>10} "
              f"{'Oran':>6} {'Ort.Peak':>9} {'Karar'}")
        print(f"  {'─'*8} {'─'*7} {'─'*10} "
              f"{'─'*6} {'─'*9} {'─'*12}")

        verdicts = []
        for label, r in reports.items():
            verdict_short = {
                "STANDALONE": "✅ STANDALONE",
                "FILTER":     "🔄 FİLTRE",
                "NO_CYCLE":   "❌ YOK",
            }.get(r.verdict, r.verdict)
            verdicts.append(r.verdict)
            print(f"  {label:<8} {r.total_symbols:>7} {r.n_reliable:>10} "
                  f"{r.reliable_ratio:>5.0%}  "
                  f"{r.avg_peakiness:>8.2f}  {verdict_short}")

        # Nihai karar
        print(f"\n{'─'*65}")
        standalone_count = verdicts.count("STANDALONE")
        filter_count     = verdicts.count("FILTER")
        no_cycle_count   = verdicts.count("NO_CYCLE")

        print(f"  Dönem sonuçları: STANDALONE={standalone_count} | "
              f"FİLTRE={filter_count} | YOK={no_cycle_count}")

        if standalone_count >= 2:
            final = ("✅ NİHAİ KARAR: DSP-CYCLE bağımsız strateji olarak ekle\n"
                     "  → dsp_strategies.py'e CycleParams + CycleBacktest eklenecek")
        elif filter_count + standalone_count >= 2:
            final = ("🔄 NİHAİ KARAR: DSP-CYCLE XOVER/TREND filtresi olarak kullan\n"
                     "  → score_symbol()'da cycle güvenilirlik bonusu eklenecek")
        else:
            final = ("❌ NİHAİ KARAR: DSP-CYCLE eklenmeyecek\n"
                     "  → BIST'te cycle tabanlı strateji için yeterli sinyal yok\n"
                     "  → Faz 0 bulgusu teyit edildi")

        print(f"\n  {final}")
        print(f"{'='*65}\n")

    def export_reliable_symbols(
        self,
        reports: dict[str, PeriodCycleReport],
    ) -> dict[str, list[str]]:
        """
        Dönem bazında güvenilir cycle gösteren sembolleri döndürür.
        Birden fazla dönemde güvenilir → en tutarlı semboller.
        """
        from collections import Counter
        symbol_counts: Counter = Counter()

        for report in reports.values():
            for r in report.symbol_results:
                if r.is_reliable:
                    symbol_counts[r.symbol] += 1

        consistent = [s for s, c in symbol_counts.items() if c >= 2]
        any_period  = [s for s, c in symbol_counts.items() if c >= 1]

        return {
            "consistent_2plus": sorted(consistent),
            "any_period":       sorted(any_period),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="P3-DSP DSP-CYCLE Faz 4 Test")
    parser.add_argument(
        "--period", choices=list(TEST_PERIODS.keys()) + ["all"],
        default="all", help="Test dönemi (varsayılan: all)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Hızlı test — 20 sembol (GARAN,THYAO,EREGL,ASELS,BIMAS...)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Sembol detayları"
    )
    args = parser.parse_args()

    # Sembol listesi
    if args.quick:
        symbols = [
            "GARAN", "AKBNK", "YKBNK", "THYAO", "PGSUS",
            "EREGL", "KRDMD", "ASELS", "BIMAS", "MGROS",
            "KCHOL", "SAHOL", "TUPRS", "PETKM", "TCELL",
            "TTKOM", "ENKAI", "EKGYO", "AKCNS", "ALKIM",
        ]
        print(f"Hızlı mod: {len(symbols)} sembol")
    else:
        from scanner_dsp import load_symbols
        symbols = load_symbols()
        print(f"Tam evren: {len(symbols)} sembol")

    tester = DspCycleTest(symbols=symbols)

    print(f"\n{'='*65}")
    print("  P3-DSP DSP-CYCLE Faz 4 Tekrar Test")
    print(f"  Peakiness eşiği: {RELIABLE_THRESHOLD} | "
          f"Standalone eşiği: {STANDALONE_THRESHOLD:.0%}")
    print(f"{'='*65}")

    if args.period == "all":
        reports = tester.run_all_periods(verbose=args.verbose)

        for label, report in reports.items():
            tester.print_period_report(report)

        tester.print_cross_period_summary(reports)

        # Tutarlı semboller
        reliable = tester.export_reliable_symbols(reports)
        if reliable["consistent_2plus"]:
            print(f"  2+ dönemde güvenilir cycle: "
                  f"{', '.join(reliable['consistent_2plus'])}")
        else:
            print("  Hiçbir sembol 2+ dönemde tutarlı cycle göstermedi.")

    else:
        report = tester.run_period(args.period, verbose=args.verbose)
        tester.print_period_report(report)