"""
DSP-P3 FAZ 1/2 — Production bütünlük testleri
=============================================
17 kritik production modülünün SHA-256 değerleri FAZ 0 frozen sabitleriyle
(baseline_constants.py) eşleşiyor mu? Workflow'lar yerinde mi?

FAZ 2 (CI-güvenli): hash doğrulaması, baseline dizini (repo dışı) olmadan da
çalışır — beklenen değerler baseline_constants.py içine gömülüdür.

FAZ 3.1: HEAD kontrolü SHA eşitliği yerine SÜRDÜRÜLEBİLİR invariant kullanır:
FAZ 0 baseline commit'i, mevcut HEAD'in ancestor'ı olmalıdır (baseline
korunmuştur, HEAD onun üzerine inşa edilmiştir). Yerelde ve CI'da çalışır.

SADECE okuma (subprocess: salt-okunur `git merge-base --is-ancestor`).
"""

from __future__ import annotations

import subprocess

from conftest import REPO_ROOT, sha256_of  # type: ignore
from baseline_constants import (
    HEAD_AT_FAZ0,
    KRITIK_MODULLER,
    PRODUCTION_SHA256,
    WORKFLOWLAR,
)


def test_17_modul_hash_frozen_sabitlerle_eslesir():
    """Diskteki 17 modül, FAZ 0 sertifikalı TAM SHA-256 değerleriyle birebir."""
    assert len(PRODUCTION_SHA256) == 18, "frozen sabit seti 18 modül olmalı (FAZ 6.2: mott_state_coordination eklendi)"
    for fname in KRITIK_MODULLER:
        beklenen = PRODUCTION_SHA256[fname]
        gercek = sha256_of(REPO_ROOT / fname)
        assert gercek == beklenen, f"{fname}: frozen {beklenen} != disk {gercek}"


def test_head_faz0_baseline_commitini_iceriyor():
    """Sürdürülebilir invariant: FAZ 0 baseline commit'i mevcut HEAD'in
    ancestor'ı olmalıdır (SHA eşitliği değil — her yeni commit'te geçerli).
    Production/state bütünlüğü, commit SHA'sından bağımsız olarak frozen
    hash testleriyle (17/17, 9/9) doğrulanır."""
    cikti = subprocess.run(
        ["git", "merge-base", "--is-ancestor", HEAD_AT_FAZ0, "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert cikti.returncode == 0, (
        f"FAZ 0 baseline commit'i ({HEAD_AT_FAZ0}) mevcut HEAD'in ancestor'ı "
        f"değil! {cikti.stderr}"
    )


def test_workflowlar_yerinde():
    for wf in WORKFLOWLAR:
        assert (REPO_ROOT / wf).exists(), f"Eksik workflow: {wf}"
