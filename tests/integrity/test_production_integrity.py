"""
DSP-P3 FAZ 1/2 — Production bütünlük testleri
=============================================
17 kritik production modülünün SHA-256 değerleri FAZ 0 frozen sabitleriyle
(baseline_constants.py) eşleşiyor mu? Workflow'lar yerinde mi?

FAZ 2 (CI-güvenli): hash doğrulaması, baseline dizini (repo dışı) olmadan da
çalışır — beklenen değerler baseline_constants.py içine gömülüdür.
HEAD sabitlemesi yalnızca yerel FAZ 0 ortamında anlamlıdır (CI'da HEAD,
test edilen commit'tir); baseline yoksa açık nedenle skip edilir.

SADECE okuma (subprocess: yalnızca salt-okunur `git rev-parse HEAD`).
"""

from __future__ import annotations

import subprocess

import pytest

from conftest import BASELINE_AVAILABLE, REPO_ROOT, sha256_of  # type: ignore
from baseline_constants import (
    HEAD_AT_FAZ0,
    KRITIK_MODULLER,
    PRODUCTION_SHA256,
    WORKFLOWLAR,
)


def test_17_modul_hash_frozen_sabitlerle_eslesir():
    """Diskteki 17 modül, FAZ 0 sertifikalı TAM SHA-256 değerleriyle birebir."""
    assert len(PRODUCTION_SHA256) == 17, "frozen sabit seti 17 modül olmalı"
    for fname in KRITIK_MODULLER:
        beklenen = PRODUCTION_SHA256[fname]
        gercek = sha256_of(REPO_ROOT / fname)
        assert gercek == beklenen, f"{fname}: frozen {beklenen} != disk {gercek}"


def test_head_commit_degismedi():
    """Yerel (baseline mevcut) ortamda HEAD, FAZ 0 commit'ine sabitlenir."""
    if not BASELINE_AVAILABLE:
        pytest.skip(
            "dsp-p3-baseline CI'da yok: HEAD sabitlemesi yalnızca FAZ 0 yerel "
            "ortamında anlamlıdır; CI'da production/state frozen hash kapıları "
            "çalışır (test_17_modul_hash... / test_state_hashleri_9_9_eslesir)."
        )
    cikti = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert cikti.returncode == 0, cikti.stderr
    assert cikti.stdout.strip() == HEAD_AT_FAZ0


def test_workflowlar_yerinde():
    for wf in WORKFLOWLAR:
        assert (REPO_ROOT / wf).exists(), f"Eksik workflow: {wf}"
