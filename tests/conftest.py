"""
DSP-P3 FAZ 1 — Test Harness ortak kurulumu (conftest)
=====================================================
Kurallar:
  * Production modüllerini IMPORT edebilir (salt-okunur), ASLA çağırarak
    state yazdırmaz (calistir / monitor / scan / run fonksiyonları çağrılmaz).
  * NETWORK = OFF: socket bağlantıları autouse fixture ile engellenir.
  * State / baseline / production dosyaları salt-okunurdur; testler yalnızca
    tmp_path altına yazar.
  * Session sonunda state + baseline hash'leri FAZ 0 değerleriyle karşılaştırılır
    ve immutability ihlali varsa FAIL olarak raporlanır.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
# Baseline dizini (repo DIŞINDA). DSP_P3_BASELINE_DIR env ile override
# edilebilir — CI davranışı simülasyonu ve farklı ortamlar için.
BASELINE_ROOT = Path(
    os.environ.get("DSP_P3_BASELINE_DIR", str(REPO_ROOT.parent / "dsp-p3-baseline"))
)
# CI ortamında dsp-p3-baseline mevcut değildir. Frozen FAZ 0 sabitleri
# tests/baseline_constants.py içindedir; baseline-dosyasına bağlı doğrulamalar
# yalnızca BASELINE_AVAILABLE iken çalışır (açık nedenle skip).
BASELINE_AVAILABLE = BASELINE_ROOT.exists()

# Modüller import edilmeden ÖNCE set edilmelidir (module-level BASE okunur).
# Tüm test koşularında tutarlılık için doğrudan atanır.
os.environ["MOTT_BASE_DIR"] = str(REPO_ROOT)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fixture'lar
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def baseline_root() -> Path:
    return BASELINE_ROOT


@pytest.fixture
def immutable_file() -> Path:
    """Varsayılan olarak repo kökündeki state dosyalarından biri.
    Testler bunu SADECE okumalıdır (okumadan önce hash'ini alabilir)."""
    return REPO_ROOT / "state_p1.json"


@pytest.fixture
def json_fixture(tmp_path: Path) -> Path:
    """Testlerin yazabileceği İZOLİ tmp_path içinde JSON dosyası."""
    p = tmp_path / "fixture.json"
    p.write_text(json.dumps({"ok": True}), encoding="utf-8")
    return p


@pytest.fixture
def deterministic_data() -> pd.DataFrame:
    """Network gerektirmeyen, sabit, küçük OHLCV verisi (BIST benzeri).
    Aynı girdi her çalıştırmada aynı çıktıyı üretmelidir (seed sabit)."""
    rng = np.random.default_rng(42)
    n = 64
    dates = pd.bdate_range("2026-01-05", periods=n)
    close = 100 + rng.normal(0, 1.0, n).cumsum()
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.5, n)
    volume = rng.integers(100_000, 5_000_000, n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "tarih"
    return df


@pytest.fixture
def isolated_tmp_path(tmp_path: Path) -> Path:
    """Testin kendi geçici dizini — production state'ine asla dokunmaz."""
    return tmp_path


# ---------------------------------------------------------------------------
# NETWORK = OFF — autouse
# ---------------------------------------------------------------------------

def _network_engel(*_args, **_kwargs):
    raise RuntimeError(
        "NETWORK OFF: FAZ 1 testleri network erişimi kullanamaz. "
        "Canlı veri (yfinance/TradingView/Telegram/broker) testlerde yasaktır."
    )


@pytest.fixture(autouse=True)
def _network_off(monkeypatch):
    """Her testten önce socket tabanlı tüm ağ erişimini engelle."""
    monkeypatch.setattr(socket.socket, "connect", _network_engel)
    monkeypatch.setattr(socket.socket, "connect_ex", _network_engel)
    monkeypatch.setattr(socket.socket, "sendto", _network_engel)


# ---------------------------------------------------------------------------
# Session sonu immutability raporu
# ---------------------------------------------------------------------------

# FAZ 0 kapanış doğrulamasında kaydedilen state hash'leri (değişmemeli).
# Tek kaynak: tests/baseline_constants.py (frozen FAZ 0 sabitleri — CI'da da mevcut).
from baseline_constants import STATE_SHA256 as _FAZ0_STATE_HASHES


def pytest_sessionfinish(session, exitstatus):
    """Session sonunda state ve baseline immutability'sini raporla."""
    ihlal = []

    for fname, beklenen in _FAZ0_STATE_HASHES.items():
        p = REPO_ROOT / fname
        if not p.exists():
            ihlal.append(f"STATE YOK: {fname}")
        elif sha256_of(p) != beklenen:
            ihlal.append(f"STATE DEGISTI: {fname}")

    if BASELINE_AVAILABLE:
        for artefakt in ("manifest.json", "p4_ic_baseline.txt", "state_sha256.txt"):
            p = BASELINE_ROOT / artefakt
            if not p.exists():
                ihlal.append(f"BASELINE YOK: {artefakt}")
    else:
        print(
            "[IMMUTABILITY] NOT: dsp-p3-baseline CI'da mevcut değil; "
            "baseline artefakt doğrulaması yerel ortamda yapılır. "
            "Production/state frozen hash kapıları (baseline_constants.py) "
            "integrity/golden testlerinde her ortamda çalışır."
        )

    if ihlal:
        print("\n\n[IMMUTABILITY] FAIL — FAZ 0 dondurması ihlal edildi:")
        for satir in ihlal:
            print(f"  [IMMUTABILITY] {satir}")
        print("[IMMUTABILITY] Test koşusu FAZ 1 başarı kriterlerini KARŞILAMIYOR.\n")
        session.exitstatus = 1
    else:
        if BASELINE_AVAILABLE:
            print("\n[IMMUTABILITY] PASS — state ve baseline dosyaları FAZ 0 ile aynı.\n")
        else:
            print(
                "\n[IMMUTABILITY] PASS — state dosyaları FAZ 0 frozen sabitleriyle aynı "
                "(baseline artefakt kontrolü yerel ortamda yapılır).\n"
            )
