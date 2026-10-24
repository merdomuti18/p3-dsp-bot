# -*- coding: utf-8 -*-
"""P1_PAPER varsayılan kapalı; üretim modülleri importta paper/broker çağırmaz."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_p1_paper_flag_default_off(monkeypatch):
    monkeypatch.delenv("P1_PAPER", raising=False)
    import importlib
    import p1_paper_config as cfg
    importlib.reload(cfg)
    assert cfg.is_p1_paper_enabled() is False
    assert os.environ.get("P1_PAPER", "") == ""


def _module_level_p1_paper_imports(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "p1_paper" or alias.name.startswith("p1_paper."):
                    hits.append(alias.name)
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "p1_paper" or mod.startswith("p1_paper"):
                hits.append(mod)
    return hits


def test_production_modules_have_no_module_level_p1_paper_import():
    for fname in ("portfoy_yonetici.py", "scanner_p1.py"):
        hits = _module_level_p1_paper_imports(REPO / fname)
        assert hits == [], f"{fname} module-level p1_paper import: {hits}"


def test_production_hooks_are_feature_flag_gated():
    py = (REPO / "portfoy_yonetici.py").read_text(encoding="utf-8")
    sc = (REPO / "scanner_p1.py").read_text(encoding="utf-8")
    assert 'os.environ.get("P1_PAPER"' in py
    assert 'os.environ.get("P1_PAPER"' in sc
    assert py.count("import p1_paper") >= 1
    assert sc.count("import p1_paper") >= 1
    # Hook'lar if bloğunun içinde (satır başı import yok)
    for line in py.splitlines():
        if line.startswith("import p1_paper") or line.startswith("from p1_paper"):
            pytest.fail("portfoy_yonetici.py module-level p1_paper")
    for line in sc.splitlines():
        if line.startswith("import p1_paper") or line.startswith("from p1_paper"):
            pytest.fail("scanner_p1.py module-level p1_paper")


def test_import_production_does_not_call_p1_paper(monkeypatch):
    monkeypatch.delenv("P1_PAPER", raising=False)
    import sys
    import importlib

    for name in list(sys.modules):
        if name == "p1_paper" or name.startswith("p1_paper."):
            del sys.modules[name]

    import portfoy_yonetici
    import scanner_p1
    importlib.reload(portfoy_yonetici)
    importlib.reload(scanner_p1)

    assert "p1_paper" not in sys.modules
    assert not hasattr(portfoy_yonetici, "run_phase_aksam")
    assert not hasattr(scanner_p1, "run_phase_aksam")
