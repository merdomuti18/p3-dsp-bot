"""
DSP-P3 FAZ 2 — Frozen FAZ 0 Baseline Sabitleri
==============================================
Bu modül, FAZ 0'da sertifikalanmış (DSP-P3-FAZ0-2026-08-17) hash değerlerinin
test takımına gömülü (embedded) kopyasıdır. Amaç:

  * CI ortamında `dsp-p3-baseline/` (repo DIŞINDA) mevcut olmasa bile
    production 17/17 + state 9/9 immutability kapıları çalışsın.
  * Yerel ortamda baseline artefakt dosyaları varken golden testleri tam
    doğrulama yapar (bu sabitlerle çapraz kontrol).

Değerler FAZ 0 kapanış doğrulamasında ve FAZ 1 test koşusunda birebir
doğrulanmıştır. Production/state dosyalarında BİLİNÇLİ bir değişiklik
yapıldığında (ör. FAZ 4 — P4 IC düzeltmesi) bu sabitler strateji sahibinin
onayıyla yeni baseline kaydıyla birlikte güncellenir.
"""

from __future__ import annotations

BASELINE_ID = "DSP-P3-FAZ0-2026-08-17"
HEAD_AT_FAZ0 = "5349ca676f7b1ba0457f0a235643a06ed4a5185a"

# ---------------------------------------------------------------------------
# State dosyaları — 9/9 tam SHA-256 (state_sha256.txt / FAZ 0 kapanışı)
# ---------------------------------------------------------------------------
STATE_DOSYALARI = [
    "state_p1.json", "state_p2.json", "portfolio_state.json", "state_p4.json",
    "state_p5.json", "portfoy.json", "portfoy_p2.json",
    "tarama_listesi.json", "tarama_listesi_p2.json",
]

STATE_SHA256 = {
    "state_p1.json": "5b7202535ddf5194b4e9e3f09d771bf3ce31fe913ba9ed761f017f181f624670",
    "state_p2.json": "4d328bd864044302aab3bf78fc00aecea7c1f816a524a1713cbab3ac6f560a52",
    "portfolio_state.json": "7fdea0e8350bc3470e9a64cf241aacdc7e0b0669f9e947db82c3a0d7e692894b",
    "state_p4.json": "8bcf3f53b648bc03e261ce3a97ca4b150c9175cc6db1bc46ee0be188c86cb51c",
    "state_p5.json": "8e6e7467c9054b65fb1b4ad005a36cfa828df5d5e09b30de5bec2655c373bdae",
    "portfoy.json": "514929f49817f4db5532dda0d3a3c8a57bec5565349a2f7c426a497750317ee4",
    "portfoy_p2.json": "3234204d43d549ba911cf1d087410ca7cd57a29a2c930e44714802c44035fc0d",
    "tarama_listesi.json": "283f88197206c97b09894b21b9b30d7042adac3e5fdd679d60848ace9cbb9281",
    "tarama_listesi_p2.json": "e1212a9d0e9b2e9df0c11505490b75ee45237d36854e1adbba4e8fa0ae0107e1",
}

# ---------------------------------------------------------------------------
# Kritik production modülleri — 17/17 tam SHA-256 (manifest.json / FAZ 0)
# ---------------------------------------------------------------------------
KRITIK_MODULLER = [
    "dsp_core.py", "dsp_strategies.py", "dsp_trend.py", "scanner_dsp.py",
    "simulate_dsp.py", "meta_portfolio.py", "mott_state.py", "mott_risk.py",
    "mott_telegram.py", "mott_fiyat.py", "mott_portfoy_deger.py",
    "mott_performans_analiz.py", "mott_aylik_rapor.py", "p5_committee.py",
    "portfoy_yonetici.py", "scanner_p1.py", "scanner_smc.py",
]

PRODUCTION_SHA256 = {
    "dsp_core.py": "8e5fc88c9d90abb0bd893008b6d626357771a0d7c4b38bc3cbad867776405ea1",
    "dsp_strategies.py": "ae108ff65769d69b30b6a46c416fdfebd35cab0c20b34f5d0fe0ff6adcc44200",
    "dsp_trend.py": "dea1c9ad7f86158205322ac6fbef0e58bc14b111ece8d4ec9be232d6b00e143b",
    "scanner_dsp.py": "311ae33ae94a675dfe87c2a19b8f6e8ed6ab78b658bc616314331492b0d14ef6",
    "simulate_dsp.py": "8739aa36da43adc3159bf50a037d953f8fb7fbcdcbfa650367e1798f4251e372",
    "meta_portfolio.py": "8f42f8fdfaf2b8802cb2409878bfcf7d718eae9365fec96106c72c569a816446",
    "mott_state.py": "6ba583bccd17bc7d5b2c427e8832bd99c645eaaa4e5bbf70d39343d511ba4668",
    "mott_risk.py": "de5decc19316d80d1e524df708b60e0f2235a1c642b23947dd9efb1068fd546c",
    "mott_telegram.py": "53fab3e86fb9ed5f012f12c9abcf0d9ee5e42266424b6b75bff687e557a2b935",
    "mott_fiyat.py": "4016830abdb040c1d83115583c6c9a1b42683a771303343e1951196970bf685e",
    "mott_portfoy_deger.py": "818f19c582123d5816918f8cfd7f47380f05ccc0c96bf9eb76c091c056fce4e4",
    "mott_performans_analiz.py": "0268b9ac8bb70b5a7029b2308e5b58c01bb134102a0c84653ab986fe64251752",
    "mott_aylik_rapor.py": "abe7237612dc910ba16430184dead43f2b585dd6543b03bc0bf4bc071dffdf9e",
    "p5_committee.py": "5b0bef77b3873f86e0a5a338199a2aec50c0cac9e30011c7c29b8fde88a894eb",
    "portfoy_yonetici.py": "f64d59ba43b02adc3ae57ba6f70e8b7fe6ab1e5d62c3ec95af7bbd523f4f84e4",
    "scanner_p1.py": "59b8a427bb77b2d17468df4bb0c4e2c2027248d78e51c442fab2d2ffeace035d",
    "scanner_smc.py": "a815c470e65788d7a06a43522ec9233d35af092435798fd379298f08c88de489",
}

REQUIREMENTS_SHA256 = "ee81e5bd95bfb034beaf62bc7234abaf150dfcff372ebbe17a1ebb87a8b302fc"

# ---------------------------------------------------------------------------
# Baseline artefaktları — 6/6 tam SHA-256 (repo DIŞINDA: dsp-p3-baseline/)
# CI'da bu dosyalar bulunmaz; bu sabitler yerel golden doğrulamasının
# beklenen değerleridir.
# ---------------------------------------------------------------------------
ARTEFAKT_SHA256 = {
    "manifest.json": "f4f50bbf8beb5a40da7f99ca95e132beb4ada65e41e2316466ccb1dd78951975",
    "p4_ic_baseline.txt": "512e6d741ea250b9fec2d4bcbc8fc19480e8947bddff486b0d831f44bf7578e9",
    "state_sha256.txt": "82c8c4771463ba75bfe13b749cbf380ffef23ae4bf77c17740d2c8c2348f0098",
    "mott_state_normalize_output.txt": "ed252d0cbea8430c6b9b17725b5293342737e0bf73dbfcf8dec9ab95aae2c755",
    "mott_performans_analiz_output.txt": "5d41c287ab4c22fc849d06b7ddf9bdb675b8beba38ca10cf2881c8cae9cf51a0",
    "mott_aylik_rapor_output.txt": "7f95555439cf128bad8486c896cf39da953efbc19c112f35e826ce257b365f34",
}

# ---------------------------------------------------------------------------
# Workflow'lar (repo içinde, CI'da da mevcut)
# ---------------------------------------------------------------------------
WORKFLOWLAR = [
    ".github/workflows/mott_daily.yml",
    ".github/workflows/tv-auth-smoke.yml",
    ".github/workflows/bqrp-hourly-ema-paper.yml",
    ".github/workflows/test.yml",
]
