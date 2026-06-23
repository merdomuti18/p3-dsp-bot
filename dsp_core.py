"""
dsp_core.py — P3-DSP Çekirdek Motoru
=====================================
Faz 1 çıktısı. Üç ana bileşen:
  1. Causal Filtre Bankası  (Butterworth + Chebyshev, SADECE lfilter)
  2. FFT/Welch Cycle Detector (rolling window, parametrik seçim aracı)
  3. AdaptiveFilterState (haftalık/aylık periyodik refit)

KRİTİK KURAL: Bu dosyada `filtfilt` KULLANILMAZ.
filtfilt yalnızca dsp_research/diagnostics.py'de (offline analiz) serbesttir.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, cheby1, lfilter, welch
from scipy.signal.windows import hann
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
import warnings


# ---------------------------------------------------------------------------
# 1. CAUSAL FİLTRE BANKASI
# ---------------------------------------------------------------------------

def design_butterworth(
    period: float,
    order: int = 3,
    filter_type: str = "low",
    sampling_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Butterworth filtre katsayıları üretir (uygulamaz).

    Parameters
    ----------
    period        : Cutoff periyot (gün). Örn. 20 → 20 günlük low-pass.
    order         : Filtre mertebesi (2-5 arası tavsiye edilir).
    filter_type   : 'low', 'high' veya 'bandpass'.
    sampling_rate : Örnekleme hızı (günlük veri için 1.0).

    Returns
    -------
    (b, a) katsayı çifti — apply_causal()'a geçilir.

    Faz-gecikme rehberi:
        order=2 → ~period/4 bar  (düşük gecikme, düşük keskinlik)
        order=3 → ~period/3 bar  (denge noktası — DSP-XOVER default'u)
        order=5 → ~period/2 bar  (yüksek keskinlik, ciddi gecikme)
    """
    nyq = sampling_rate / 2.0
    normalized_cutoff = (1.0 / period) / nyq

    # Filtre tasarımı fiziksel sınırlar içinde olmalı
    if normalized_cutoff <= 0 or normalized_cutoff >= 1:
        raise ValueError(
            f"Geçersiz normalized_cutoff={normalized_cutoff:.4f}. "
            f"period={period} sampling_rate={sampling_rate} ile hesaplandı. "
            f"period değeri [2, ∞) aralığında olmalı."
        )

    b, a = butter(order, normalized_cutoff, btype=filter_type)
    return b, a


def design_chebyshev(
    period: float,
    order: int = 4,
    ripple_db: float = 0.5,
    filter_type: str = "low",
    sampling_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chebyshev Type-I filtre katsayıları üretir (uygulamaz).

    Butterworth'a göre daha keskin geçiş bandı, pasband'da küçük ripple
    kabul edilir. DSP-TREND'de alternatif olarak denenebilir.

    Parameters
    ----------
    ripple_db : Pasband maksimum dalgalanma (dB). 0.5 dB iyi başlangıç.
    Diğerleri : design_butterworth() ile aynı anlam.
    """
    nyq = sampling_rate / 2.0
    normalized_cutoff = (1.0 / period) / nyq

    if normalized_cutoff <= 0 or normalized_cutoff >= 1:
        raise ValueError(
            f"Geçersiz normalized_cutoff={normalized_cutoff:.4f}. "
            f"period={period} ile hesaplandı."
        )

    b, a = cheby1(order, ripple_db, normalized_cutoff, btype=filter_type)
    return b, a


def apply_causal(
    b: np.ndarray,
    a: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
    """
    Tek yönlü (causal) filtreleme — look-ahead YOK.

    Production'da sinyal üretmek için SADECE bu fonksiyon kullanılır.
    filtfilt bu dosyaya asla girmez.

    Parameters
    ----------
    b, a  : design_butterworth() veya design_chebyshev() çıktısı.
    data  : 1-D fiyat/log-return dizisi (en eski → en yeni).

    Returns
    -------
    Filtrelenmiş dizi, data ile aynı uzunlukta.

    NOT: lfilter başlangıçta (ilk ~order*period bar) geçici hata içerir
    (transient/warm-up etkisi). Bu barları sinyal olarak kullanma,
    yalnızca warm-up tamamlandıktan sonraki değerlere güven.
    Warm-up süresi tahmini: order * period bar.
    """
    if len(data) < 3:
        raise ValueError("Filtreleme için en az 3 veri noktası gerekli.")
    return lfilter(b, a, data)


def warmup_length(order: int, period: float) -> int:
    """
    Causal filtre transient (warm-up) süresi tahmini.
    Bu kadar bar'ı sinyalde kullanma — özellikle backtest başında önemli.
    """
    return int(order * period)


# ---------------------------------------------------------------------------
# 2. FFT / WELCH CYCLE DETECTOR
# ---------------------------------------------------------------------------

# Ekonomik sınırlar — Faz 0 bulgularıyla uyumlu (Mimari Plan Bölüm 6.3)
CYCLE_PERIOD_MIN = 15   # gün
CYCLE_PERIOD_MAX = 60   # gün


@dataclass
class CycleDetectionResult:
    """detect_dominant_cycle() dönüş tipi."""
    dominant_period: float          # En güçlü peak'in periyodu (gün)
    dominant_power: float           # O peak'in normalize güç değeri
    peakiness: float                # Faz 0'ın kalite metriği (>2.5 = anlamlı)
    top_periods: np.ndarray         # İlk top_n periyot (küçükten büyüğe)
    top_powers: np.ndarray          # Karşılık gelen güç değerleri
    is_reliable: bool               # peakiness > PEAKINESS_THRESHOLD mı?
    n_samples: int                  # Kullanılan pencere uzunluğu

    PEAKINESS_THRESHOLD: float = field(default=2.5, init=False, repr=False)


def detect_dominant_cycle(
    prices_window: np.ndarray,
    sampling_rate: float = 1.0,
    top_n: int = 3,
    nperseg_ratio: float = 0.5,
    period_min: float = CYCLE_PERIOD_MIN,
    period_max: float = CYCLE_PERIOD_MAX,
) -> CycleDetectionResult:
    """
    Rolling window üzerinde Welch yöntemiyle dominant cycle tespiti.

    Faz 0 metodolojisiyle uyumlu (Welch + log-return + peakiness).
    Yalnızca geçmiş veri kullanır — causal, look-ahead yok.

    Parameters
    ----------
    prices_window  : Kapanış fiyatları (en eski → en yeni, 1-D).
    sampling_rate  : Örnekleme hızı (günlük=1.0).
    top_n          : Kaç tane dominant periyot döndürülsün.
    nperseg_ratio  : Welch segment boyutu = len(prices) * ratio.
    period_min/max : İncelenecek periyot aralığı (gün).

    Returns
    -------
    CycleDetectionResult — dominant periyot, peakiness, güvenilirlik bayrağı.

    Kullanım notu:
        Bu fonksiyon bir *teşhis/parametre seçim aracıdır*.
        Çıktısı (dominant_period), band-pass filtre cutoff'unu belirlemek için
        AdaptiveFilterState.refit()'e beslenir; doğrudan sinyal üretmez.
    """
    prices_window = np.asarray(prices_window, dtype=float)
    n = len(prices_window)

    if n < 32:
        raise ValueError(f"Welch için en az 32 veri noktası gerekli (n={n}).")

    # Log-return (fiyat serisini durağanlaştır, Faz 0 ile aynı)
    log_returns = np.diff(np.log(prices_window))

    # Welch PSD
    nperseg = max(16, int(len(log_returns) * nperseg_ratio))
    freqs, psd = welch(
        log_returns,
        fs=sampling_rate,
        nperseg=nperseg,
        window="hann",
        scaling="density",
    )

    # DC bileşeni (freq=0) ve Nyquist'i atla
    valid_mask = (freqs > 0) & (freqs < sampling_rate / 2)
    freqs_v = freqs[valid_mask]
    psd_v = psd[valid_mask]

    # Periyot aralığı filtresi
    periods_v = 1.0 / freqs_v
    range_mask = (periods_v >= period_min) & (periods_v <= period_max)

    if range_mask.sum() < 3:
        # Aralıkta yeterli frekans çözünürlüğü yok — uyarı ver, dummy döndür
        warnings.warn(
            f"period_min={period_min}..period_max={period_max} aralığında "
            f"yalnızca {range_mask.sum()} Welch frekans noktası var. "
            f"Pencere boyutunu artır (n={n}).",
            UserWarning,
            stacklevel=2,
        )
        dummy = np.array([float(period_min)] * top_n)
        return CycleDetectionResult(
            dominant_period=float(period_min),
            dominant_power=0.0,
            peakiness=1.0,
            top_periods=dummy,
            top_powers=np.zeros(top_n),
            is_reliable=False,
            n_samples=n,
        )

    freqs_r = freqs_v[range_mask]
    psd_r = psd_v[range_mask]
    periods_r = periods_v[range_mask]

    # Peakiness: Faz 0 metriği (max_peak / mean_background)
    peak_power = psd_r.max()
    mean_power = psd_v.mean()  # tüm frekansların ortalaması (arka plan)
    peakiness = peak_power / mean_power if mean_power > 0 else 1.0

    # Top-N periyot
    top_idx = np.argsort(psd_r)[-top_n:][::-1]
    top_periods = periods_r[top_idx]
    top_powers = psd_r[top_idx]

    dominant_period = float(top_periods[0])
    dominant_power = float(top_powers[0])

    return CycleDetectionResult(
        dominant_period=dominant_period,
        dominant_power=dominant_power,
        peakiness=float(peakiness),
        top_periods=top_periods,
        top_powers=top_powers,
        is_reliable=(peakiness >= 2.5),
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# 3. ADAPTIVE REFIT MEKANİZMASI
# ---------------------------------------------------------------------------

@dataclass
class FilterCoeffs:
    """Bir sembol için saklanan filtre katsayıları."""
    b: np.ndarray
    a: np.ndarray
    dominant_period: float
    peakiness: float
    refit_date: date
    is_cycle_reliable: bool


class AdaptiveFilterState:
    """
    Periyodik olarak (haftalık/aylık) yeniden fit edilen filtre durumu.

    Walk-forward mantığının production yansıması: her refit döneminde
    son lookback_days günlük veriyle dominant cycle tespit edilir,
    bu periyot ekonomik sınırlar içinde clip'lenir ve filtre katsayıları
    güncellenir. Barlar arasında katsayılar sabittir (tutarlılık).

    Kullanım örneği:
        state = AdaptiveFilterState()
        # Her gün:
        if state.needs_refit(today):
            state.refit("GARAN", recent_prices, today)
        b, a = state.get_coeffs("GARAN")
        filtered = apply_causal(b, a, prices)
    """

    def __init__(
        self,
        refit_interval_days: int = 7,
        lookback_days: int = 120,
        filter_order: int = 3,
        period_min: float = CYCLE_PERIOD_MIN,
        period_max: float = CYCLE_PERIOD_MAX,
        fallback_period: float = 30.0,
    ):
        """
        Parameters
        ----------
        refit_interval_days : Kaç günde bir refit yapılsın (7=haftalık, 30=aylık).
        lookback_days       : Refit için kullanılacak geçmiş pencere (gün).
        filter_order        : Butterworth mertebesi.
        period_min/max      : Cycle tespit aralığı (ekonomik sınırlar).
        fallback_period     : Cycle güvenilir değilse varsayılan cutoff (gün).
        """
        self.refit_interval_days = refit_interval_days
        self.lookback_days = lookback_days
        self.filter_order = filter_order
        self.period_min = period_min
        self.period_max = period_max
        self.fallback_period = fallback_period

        self._coeffs: dict[str, FilterCoeffs] = {}
        self._last_refit: Optional[date] = None

    # --- Public API ---

    def needs_refit(self, current_date: date) -> bool:
        """Bu tarihte refit yapılması gerekiyor mu?"""
        if self._last_refit is None:
            return True
        return (current_date - self._last_refit).days >= self.refit_interval_days

    def refit(
        self,
        symbol: str,
        prices_window: np.ndarray,
        current_date: date,
    ) -> FilterCoeffs:
        """
        Tek sembol için filtre katsayılarını günceller.

        Adımlar:
          1. Welch cycle detector ile dominant periyot tespit
          2. period_min..period_max aralığına clip
          3. Cycle güvenilir değilse fallback_period kullan
          4. Butterworth tasarla ve sakla
        """
        prices_window = np.asarray(prices_window, dtype=float)

        # Yeterli veri var mı?
        lookback = min(self.lookback_days, len(prices_window))
        window = prices_window[-lookback:]

        try:
            result = detect_dominant_cycle(
                window,
                period_min=self.period_min,
                period_max=self.period_max,
            )
            if result.is_reliable:
                chosen_period = np.clip(
                    result.dominant_period, self.period_min, self.period_max
                )
            else:
                chosen_period = self.fallback_period
            peakiness = result.peakiness
            is_reliable = result.is_reliable
        except (ValueError, RuntimeWarning):
            chosen_period = self.fallback_period
            peakiness = 0.0
            is_reliable = False

        b, a = design_butterworth(
            period=chosen_period,
            order=self.filter_order,
        )

        coeffs = FilterCoeffs(
            b=b,
            a=a,
            dominant_period=float(chosen_period),
            peakiness=float(peakiness),
            refit_date=current_date,
            is_cycle_reliable=is_reliable,
        )
        self._coeffs[symbol] = coeffs
        self._last_refit = current_date
        return coeffs

    def refit_batch(
        self,
        symbols_prices: dict[str, np.ndarray],
        current_date: date,
    ) -> dict[str, FilterCoeffs]:
        """Birden fazla sembol için toplu refit."""
        results = {}
        for symbol, prices in symbols_prices.items():
            results[symbol] = self.refit(symbol, prices, current_date)
        self._last_refit = current_date
        return results

    def get_coeffs(self, symbol: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Sembol için (b, a) katsayı çifti döndürür.
        Henüz refit yapılmamışsa ValueError fırlatır.
        """
        if symbol not in self._coeffs:
            raise ValueError(
                f"'{symbol}' için filtre katsayısı yok. "
                f"Önce refit() çağır."
            )
        c = self._coeffs[symbol]
        return c.b, c.a

    def get_state(self, symbol: str) -> Optional[FilterCoeffs]:
        """Sembol için tam FilterCoeffs nesnesini döndürür (peakiness vb.)."""
        return self._coeffs.get(symbol)

    def summary(self) -> dict:
        """Tüm sembollerin mevcut durumu — monitoring için."""
        return {
            sym: {
                "dominant_period": c.dominant_period,
                "peakiness": round(c.peakiness, 3),
                "is_reliable": c.is_cycle_reliable,
                "refit_date": c.refit_date.isoformat(),
            }
            for sym, c in self._coeffs.items()
        }


# ---------------------------------------------------------------------------
# 4. LOOk-AHEAD GUARD (CI / import kontrolü)
# ---------------------------------------------------------------------------

FORBIDDEN_IN_PROD = ["filtfilt"]


def check_no_lookahead(module_path: str) -> None:
    """
    Bir prod modülünde yasaklı fonksiyon import edilmiş mi kontrol eder.
    Yorum satırları ve string literalleri goez ardi edilir;
    yalnizca gercek AST import node'lari kontrol edilir.

    Raises AssertionError eger yasak import tespit edilirse.
    """
    import ast as _ast

    with open(module_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = _ast.parse(source, filename=module_path)

    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom):
            for alias in node.names:
                for forbidden in FORBIDDEN_IN_PROD:
                    assert alias.name != forbidden, (
                        f"HATA: '{forbidden}' prod modulunde yasak import! "
                        f"Dosya: {module_path}, satir: {node.lineno}"
                    )
        elif isinstance(node, _ast.Import):
            for alias in node.names:
                for forbidden in FORBIDDEN_IN_PROD:
                    assert forbidden not in alias.name, (
                        f"HATA: '{forbidden}' prod modulunde yasak import! "
                        f"Dosya: {module_path}, satir: {node.lineno}"
                    )
