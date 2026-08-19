"""
DSP-P3 FAZ 1/2 — Golden / Baseline testleri
===========================================
FAZ 0 artefaktlarını (dsp-p3-baseline/) SADECE okur, ASLA değiştirmez.

FAZ 2 (CI-güvenli): dsp-p3-baseline repo DIŞINDADIR ve CI'a kopyalanmaz.
  * Baseline dizini mevcutsa → artefakt dosyaları üzerinde TAM doğrulama
    (hash sabitliği, manifest tutarlılığı, normalize golden, state_sha256).
  * Baseline yoksa (CI) → bu dosyaya bağlı testler AÇIK NEDENLE skip edilir;
    asıl CI kapıları integrity katmanında frozen sabitlerle çalışır
    (production 17/17 + state 9/9 + lookahead + import smoke).
"""

from __future__ import annotations

import json

import pytest

from conftest import BASELINE_AVAILABLE, BASELINE_ROOT, REPO_ROOT, sha256_of  # type: ignore
from baseline_constants import (
    ARTEFAKT_SHA256,
    BASELINE_ID,
    HEAD_AT_FAZ0,
    KRITIK_MODULLER,
    PRODUCTION_SHA256,
    REQUIREMENTS_SHA256,
    STATE_DOSYALARI,
    STATE_SHA256,
)

_STATE_DOSYALARI = STATE_DOSYALARI
_MODULLER = KRITIK_MODULLER


@pytest.fixture(scope="module")
def baseline():
    """Baseline dizini — CI'da yoksa AÇIK NEDENLE skip."""
    if not BASELINE_AVAILABLE:
        pytest.skip(
            "dsp-p3-baseline (repo DIŞINDAKI artefakt dizini) CI ortamında "
            "mevcut değildir ve repo'ya kopyalanmaz. CI kapıları "
            "baseline_constants.py frozen sabitleriyle integrity katmanında "
            "çalışır; bu dosya-düzeyi golden doğrulaması yerel ortam içindir."
        )
    return BASELINE_ROOT


# ---------------------------------------------------------------------------
# 0. Frozen sabitlerin kendi tutarlılığı (her ortamda çalışır — CI dokümantasyonu)
# ---------------------------------------------------------------------------

def test_frozen_sabitler_boyutlari_gecerli():
    assert len(STATE_SHA256) == 9
    assert set(STATE_SHA256) == set(STATE_DOSYALARI)
    assert len(PRODUCTION_SHA256) == 18
    assert set(PRODUCTION_SHA256) == {f.replace(".py", ".py") for f in KRITIK_MODULLER}
    assert len(ARTEFAKT_SHA256) == 6
    assert BASELINE_ID.startswith("DSP-P3-FAZ0-")
    assert len(HEAD_AT_FAZ0) == 40
    assert len(REQUIREMENTS_SHA256) == 64


# ---------------------------------------------------------------------------
# 1. Artefakt varlığı + hash sabitliği (yerel; baseline mevcutken)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("artefakt", sorted(ARTEFAKT_SHA256))
def test_artefakt_mevcut_ve_hash_sabit(baseline, artefakt):
    p = baseline / artefakt
    assert p.exists(), f"Baseline artefaktı eksik: {artefakt}"
    assert sha256_of(p) == ARTEFAKT_SHA256[artefakt]


# ---------------------------------------------------------------------------
# 2. manifest.json tutarlılığı (yerel; baseline mevcutken)
# ---------------------------------------------------------------------------

def test_manifest_json_gecerli(baseline):
    m = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    assert m["baseline_id"] == BASELINE_ID
    assert m["head_commit"] == HEAD_AT_FAZ0


def test_manifest_state_hashleri_diskle_eslesir(baseline):
    m = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    for fname, kisa in m["state_dosyalari"].items():
        gercek = sha256_of(REPO_ROOT / fname)[:12]
        assert gercek == kisa, f"{fname}: manifest {kisa} != disk {gercek}"


def test_manifest_modul_hashleri_diskle_eslesir(baseline):
    m = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    for fname, kisa in m["kritik_modul_hashleri"].items():
        gercek = sha256_of(REPO_ROOT / fname)[:12]
        assert gercek == kisa, f"{fname}: manifest {kisa} != disk {gercek}"


def test_manifest_requirements_hash_eslesir(baseline):
    m = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    gercek = sha256_of(REPO_ROOT / "requirements.txt")
    assert gercek == m["requirements_txt_sha256"]


def test_manifest_frozen_sabitlerle_tutarli(baseline):
    """manifest.json (kısa hash'ler) ↔ baseline_constants (tam hash'ler)."""
    m = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    for fname, kisa in m["state_dosyalari"].items():
        assert STATE_SHA256[fname][:12] == kisa, fname
    for fname, kisa in m["kritik_modul_hashleri"].items():
        assert PRODUCTION_SHA256[fname][:12] == kisa, fname


# ---------------------------------------------------------------------------
# 3. mott_state normalize çıktısı FAZ 0 golden'ıyla eşleşir (yerel)
# ---------------------------------------------------------------------------

def test_normalize_output_golden_ile_eslesir(baseline):
    from mott_state import hepsi
    golden = json.loads(
        (baseline / "mott_state_normalize_output.txt").read_text(encoding="utf-8")
    )
    assert hepsi() == golden


# ---------------------------------------------------------------------------
# 4. state_sha256.txt — 9/9 tam hash eşleşmesi (yerel)
# ---------------------------------------------------------------------------

def test_state_sha256_txt_9_9_eslesir(baseline):
    kayitlar = {}
    for satir in (baseline / "state_sha256.txt").read_text(encoding="utf-8").splitlines():
        satir = satir.strip()
        if not satir:
            continue
        h, f = satir.split(" *", 1)
        kayitlar[f] = h
    assert set(kayitlar) == set(_STATE_DOSYALARI)
    for fname in _STATE_DOSYALARI:
        assert sha256_of(REPO_ROOT / fname) == kayitlar[fname], fname
