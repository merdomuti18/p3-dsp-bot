"""
dsp_strategies.py — P3-DSP Strateji Katmanı
=============================================
Faz 2 çıktısı. DSP-XOVER prototipi + walk-forward validasyon.

Kapsam (Faz 2):
  - XoverSignal: Dual Butterworth crossover sinyal üreteci
  - XoverBacktest: Tek pencere backtest motoru
  - phase_lag_budget(): Her parametre seti için gecikme belgesi
  - walk_forward_validate(): Pencere bazlı istikrar testi

Gelecek (Faz 4):
  - DSP-TREND, DSP-CYCLE, DSP-MR buraya eklenir

KRİTİK KURAL: Bu dosyada filtfilt KULLANILMAZ.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys

sys.path.insert(0, "/home/claude")
from dsp_core import (
    design_butterworth,
    apply_causal,
    warmup_length,
)


# ---------------------------------------------------------------------------
# 1. DSP-XOVER — Sinyal Üreteci
# ---------------------------------------------------------------------------

@dataclass
class XoverParams:
    """
    DSP-XOVER parametre seti.

    Mimari doküman Bölüm 5.1 default'ları ile başlıyoruz;
    walk-forward'da bu parametreler taranır.
    """
    fast_period: float = 10.0   # gün
    slow_period: float = 50.0   # gün
    order: int = 3
    # Phase-lag bütçesi (Bölüm 6.2): oran tabanlı.
    # Ölçüm: lag ≈ order × period × 0.11  →  order=3 için ~period/3.
    # Kriter: lag ≤ slow_period × max_lag_ratio
    #   order=2 → ~0.23x  ✅   order=3 → ~0.34x  ✅   order=4 → ~0.44x  ⚠️
    # Eşik 0.40: order 2-3'ü geçirir, order 4+'ü reddeder.
    max_lag_ratio: float = 0.40


@dataclass
class XoverSignal:
    """
    Dual Butterworth crossover sinyal dizisi.

    Alanlar
    -------
    signal     : +1 (long) / -1 (short/flat), prices ile aynı uzunlukta
    fast_line  : Hızlı Butterworth filtre çıktısı
    slow_line  : Yavaş Butterworth filtre çıktısı
    valid_from : Warm-up tamamlandıktan sonraki ilk geçerli bar indeksi
    lag_bars   : Gerçekleşen faz gecikmesi (step response'tan ölçülür)
    params     : Kullanılan parametre seti
    """
    signal: np.ndarray
    fast_line: np.ndarray
    slow_line: np.ndarray
    valid_from: int
    lag_bars: float
    params: XoverParams


def generate_xover_signal(
    prices: np.ndarray,
    params: Optional[XoverParams] = None,
) -> XoverSignal:
    """
    Dual Butterworth crossover sinyali üretir.

    Tüm filtreler causal (lfilter) — look-ahead yok.
    Warm-up süresi: slow filtre için order × slow_period bar.

    Parameters
    ----------
    prices : Kapanış fiyatları dizisi (1-D, en eski → en yeni).
    params : XoverParams, None ise mimari doküman default'ları kullanılır.

    Returns
    -------
    XoverSignal
    """
    if params is None:
        params = XoverParams()

    prices = np.asarray(prices, dtype=float)

    b_fast, a_fast = design_butterworth(params.fast_period, params.order)
    b_slow, a_slow = design_butterworth(params.slow_period, params.order)

    fast_line = apply_causal(b_fast, a_fast, prices)
    slow_line = apply_causal(b_slow, a_slow, prices)

    signal = np.where(fast_line > slow_line, 1.0, -1.0)

    # Warm-up: yavaş filtre belirleyici (daha uzun geçici hata süresi)
    valid_from = warmup_length(params.order, params.slow_period)
    # Warm-up barlarını sıfırla (backtest'te bu barlar atlanır)
    signal[:valid_from] = 0.0

    # Faz gecikmesi ölçümü: basamak yanıtından
    lag_bars = _measure_lag(b_slow, a_slow, params.slow_period)

    return XoverSignal(
        signal=signal,
        fast_line=fast_line,
        slow_line=slow_line,
        valid_from=valid_from,
        lag_bars=lag_bars,
        params=params,
    )


def _measure_lag(b, a, period: float) -> float:
    """
    Causal filtre faz gecikmesini basamak yanıtından ölçer.

    Yöntem: 0→1 basamak fonksiyonu uygula, filtrelenmiş sinyalin
    0.5 eşiğini geçtiği bar = gecikme tahmini.
    """
    step = np.concatenate([np.zeros(int(period * 3)), np.ones(int(period * 3))])
    out = apply_causal(b, a, step)
    cross = np.where(out >= 0.5)[0]
    if len(cross) == 0:
        return float(period)  # çok yüksek order durumu
    # Basamak 0'da → geçiş noktası = gecikme
    nominal_cross = int(period * 3)
    return float(cross[0] - nominal_cross)


# ---------------------------------------------------------------------------
# 2. Phase-Lag Bütçesi Belgeleyici
# ---------------------------------------------------------------------------

@dataclass
class PhaseLagEntry:
    fast_period: float
    slow_period: float
    order: int
    fast_lag: float
    slow_lag: float
    fast_lag_ratio: float       # fast_lag / fast_period
    slow_lag_ratio: float       # slow_lag / slow_period
    within_budget: bool         # slow_lag_ratio ≤ max_lag_ratio
    note: str


def phase_lag_budget(
    param_sets: list[XoverParams],
) -> list[PhaseLagEntry]:
    """
    Mimari doküman Bölüm 6.2 — faz-lag bütçesi belgesi.

    Her parametre seti için fast/slow gecikmeyi ölçer ve bütçe
    içinde olup olmadığını bildirir.

    Bütçe kriteri: slow_lag ≤ params.max_allowed_lag_bars
    """
    entries = []
    for p in param_sets:
        b_f, a_f = design_butterworth(p.fast_period, p.order)
        b_s, a_s = design_butterworth(p.slow_period, p.order)

        fast_lag = _measure_lag(b_f, a_f, p.fast_period)
        slow_lag = _measure_lag(b_s, a_s, p.slow_period)

        fast_ratio = fast_lag / p.fast_period
        slow_ratio = slow_lag / p.slow_period

        within = slow_ratio <= p.max_lag_ratio

        note = (
            f"Order={p.order}: fast={p.fast_period}g→{fast_lag:.0f}bar ({fast_ratio:.2f}x), "
            f"slow={p.slow_period}g→{slow_lag:.0f}bar ({slow_ratio:.2f}x)  "
            + (f"✅ bütçe içinde (≤{p.max_lag_ratio:.2f}x)"
               if within
               else f"❌ bütçe aşıldı ({slow_ratio:.2f}x > {p.max_lag_ratio:.2f}x)")
        )
        entries.append(PhaseLagEntry(
            fast_period=p.fast_period,
            slow_period=p.slow_period,
            order=p.order,
            fast_lag=fast_lag,
            slow_lag=slow_lag,
            fast_lag_ratio=fast_ratio,
            slow_lag_ratio=slow_ratio,
            within_budget=within,
            note=note,
        ))
    return entries


# ---------------------------------------------------------------------------
# 3. Tek Pencere Backtest Motoru
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    XoverBacktest.run() dönüş tipi.
    """
    total_return_pct: float         # Strateji toplam getiri (%)
    buy_hold_return_pct: float      # Buy-and-hold karşılaştırma (%)
    sharpe: float                   # Günlük getirilerden yıllıklaştırılmış Sharpe
    max_drawdown_pct: float         # Maksimum drawdown (%)
    n_trades: int                   # Toplam pozisyon değişimi sayısı
    win_rate: float                 # Kazanan trade oranı (0-1)
    n_bars: int                     # Kullanılan bar sayısı (warm-up sonrası)
    params: XoverParams
    # Ara seriler (görselleştirme için)
    equity_curve: np.ndarray = field(repr=False)
    daily_returns: np.ndarray = field(repr=False)


class XoverBacktest:
    """
    DSP-XOVER için tek pencere backtest motoru.

    Varsayımlar:
    - Kapanışta sinyal üretilir, ertesi açılışta uygulama
      (gerçek implementasyonda; burada simplifikasyon: aynı bar kapanışında)
    - İşlem maliyeti: 0 (paper trading aşaması; Faz 3'te eklenecek)
    - Kısa pozisyon: flat (paper portföyde short yok)
    """

    def __init__(self, params: Optional[XoverParams] = None):
        self.params = params or XoverParams()

    def run(self, prices: np.ndarray) -> BacktestResult:
        """
        Fiyat dizisi üzerinde backtest çalıştırır.

        Returns
        -------
        BacktestResult — getiri, Sharpe, drawdown, trade istatistikleri.
        """
        prices = np.asarray(prices, dtype=float)
        sig = generate_xover_signal(prices, self.params)
        vf = sig.valid_from

        # Warm-up sonrası çalış
        p = prices[vf:]
        s = sig.signal[vf:]

        if len(p) < 10:
            raise ValueError(f"Warm-up sonrası yeterli bar yok (vf={vf}, n={len(prices)})")

        # Günlük log getiriler
        log_ret = np.diff(np.log(p))
        # Sinyal: t günü sinyali, t+1 günü getirisi (1-bar offset)
        position = s[:-1]  # long=+1, short/flat=-1 (flat olarak değerlendir)
        # Paper portföyde short flat → negatif sinyali 0 yap
        position_flat = np.where(position > 0, 1.0, 0.0)

        strat_ret = position_flat * log_ret
        bh_ret    = log_ret  # buy-and-hold

        # Kümülatif getiri
        equity    = np.exp(np.cumsum(strat_ret))
        bh_equity = np.exp(np.cumsum(bh_ret))

        total_ret = (equity[-1] - 1) * 100
        bh_ret_pct = (bh_equity[-1] - 1) * 100

        # Sharpe (yıllıklaştırılmış, günlük rf=0)
        sharpe = (
            strat_ret.mean() / (strat_ret.std() + 1e-9) * np.sqrt(252)
        )

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min() * 100

        # Trade sayısı ve win rate
        trades = self._count_trades(position_flat, strat_ret)

        return BacktestResult(
            total_return_pct=round(total_ret, 2),
            buy_hold_return_pct=round(bh_ret_pct, 2),
            sharpe=round(sharpe, 3),
            max_drawdown_pct=round(max_dd, 2),
            n_trades=trades["n_trades"],
            win_rate=round(trades["win_rate"], 3),
            n_bars=len(log_ret),
            params=self.params,
            equity_curve=equity,
            daily_returns=strat_ret,
        )

    @staticmethod
    def _count_trades(
        position: np.ndarray,
        daily_returns: np.ndarray,
    ) -> dict:
        """Trade geçişlerini ve win rate'i hesaplar."""
        changes = np.diff(np.concatenate([[0], position]))
        entry_bars = np.where(changes == 1)[0]
        exit_bars  = np.where(changes == -1)[0]

        # Her trade: entry → exit arası kümülatif getiri
        n_trades = len(entry_bars)
        if n_trades == 0:
            return {"n_trades": 0, "win_rate": 0.0}

        wins = 0
        for i, entry in enumerate(entry_bars):
            exit = exit_bars[i] if i < len(exit_bars) else len(daily_returns)
            trade_ret = daily_returns[entry:exit].sum()
            if trade_ret > 0:
                wins += 1

        return {
            "n_trades": n_trades,
            "win_rate": wins / n_trades,
        }


# ---------------------------------------------------------------------------
# 4. Walk-Forward Validasyon
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """
    Tek pencere (train+test) sonucu.
    """
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    test_return_pct: float
    test_sharpe: float
    test_max_dd_pct: float
    bh_return_pct: float
    params: XoverParams


@dataclass
class WalkForwardSummary:
    """
    Tüm pencereler için istatistiksel özet.
    """
    n_windows: int
    mean_test_return: float
    std_test_return: float
    positive_windows: int           # Pozitif getirili pencere sayısı
    mean_sharpe: float
    mean_max_dd: float
    consistency_ratio: float        # positive_windows / n_windows
    is_consistent: bool             # consistency_ratio > 0.55 ise tutarlı
    windows: list[WalkForwardResult]

    def print_summary(self):
        print(f"\n  Walk-Forward Özet ({self.n_windows} pencere)")
        print(f"  ─────────────────────────────────────────")
        print(f"  Ortalama test getirisi : {self.mean_test_return:+.2f}%")
        print(f"  Std test getirisi      : {self.std_test_return:.2f}%")
        print(f"  Pozitif pencere        : {self.positive_windows}/{self.n_windows} ({self.consistency_ratio:.0%})")
        print(f"  Ortalama Sharpe        : {self.mean_sharpe:.3f}")
        print(f"  Ortalama Max DD        : {self.mean_max_dd:.2f}%")
        print(f"  Tutarlılık değerlendirmesi: {'✅ Tutarlı (>55%)' if self.is_consistent else '⚠️  Tutarsız (≤55%)'}")


def walk_forward_validate(
    prices: np.ndarray,
    params: Optional[XoverParams] = None,
    train_window: int = 252,    # ~1 yıl
    test_window: int = 63,      # ~3 ay
    step: int = 21,             # ~1 ay
) -> WalkForwardSummary:
    """
    Rolling pencerelerle istikrar testi (Mimari Plan Bölüm 6.4).

    Faz 2'de: sabit parametreler (XoverParams.default), train bölümü
    gerçek parametre optimizasyonu için kullanılmıyor — bu intentional.
    Overfitting guard (Bölüm 6.3): parametreler ekonomik mantıkla sabitlendi,
    veri üzerinden optimize edilmiyor.

    Parameters
    ----------
    train_window : Train bölümü uzunluğu (bar). Şimdilik görselleştirme amaçlı;
                   parametreler dışarıdan sabit geliyor.
    test_window  : Her pencerenin test bölümü (bar).
    step         : Pencere kaydırma adımı (bar).

    Returns
    -------
    WalkForwardSummary
    """
    if params is None:
        params = XoverParams()

    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    bt = XoverBacktest(params)
    windows = []

    idx = 0
    window_num = 0
    while idx + train_window + test_window <= n:
        train_start = idx
        train_end   = idx + train_window
        test_start  = train_end
        test_end    = train_end + test_window

        # Backtest yalnızca test bölümünde ölçülür.
        # Warm-up için: test penceresinin öncesindeki veriyi de ver
        # (filtre baştan ısınır, test bölümü değerlendirme penceresi)
        context_start = max(0, test_start - warmup_length(params.order, params.slow_period))
        segment = prices[context_start:test_end]

        try:
            result = bt.run(segment)
            # Sonuç tüm segment için — test bölümünü izole etmek için
            # son test_window barı yeniden hesapla
            test_segment = prices[test_start:test_end]
            if len(test_segment) > warmup_length(params.order, params.slow_period) + 5:
                test_result = bt.run(test_segment)
                t_ret   = test_result.total_return_pct
                t_sharpe = test_result.sharpe
                t_dd    = test_result.max_drawdown_pct
                bh_ret  = test_result.buy_hold_return_pct
            else:
                t_ret   = result.total_return_pct
                t_sharpe = result.sharpe
                t_dd    = result.max_drawdown_pct
                bh_ret  = result.buy_hold_return_pct

        except ValueError:
            idx += step
            continue

        windows.append(WalkForwardResult(
            window_idx=window_num,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            test_return_pct=t_ret,
            test_sharpe=t_sharpe,
            test_max_dd_pct=t_dd,
            bh_return_pct=bh_ret,
            params=params,
        ))
        window_num += 1
        idx += step

    if not windows:
        raise ValueError(
            f"Hiç pencere oluşturulamadı. "
            f"prices uzunluğu ({n}) en az {train_window + test_window} olmalı."
        )

    returns = np.array([w.test_return_pct for w in windows])
    sharpes = np.array([w.test_sharpe for w in windows])
    dds     = np.array([w.test_max_dd_pct for w in windows])
    pos     = int((returns > 0).sum())
    ratio   = pos / len(windows)

    return WalkForwardSummary(
        n_windows=len(windows),
        mean_test_return=round(float(returns.mean()), 3),
        std_test_return=round(float(returns.std()), 3),
        positive_windows=pos,
        mean_sharpe=round(float(sharpes.mean()), 3),
        mean_max_dd=round(float(dds.mean()), 3),
        consistency_ratio=round(ratio, 3),
        is_consistent=(ratio > 0.55),
        windows=windows,
    )
