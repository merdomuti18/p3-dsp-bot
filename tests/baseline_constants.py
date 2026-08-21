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
    "state_p1.json": "eb76cb0ea7a85f7fceed83791c36f478405934b6be585569e89d4d9893e2dfae",
    "state_p2.json": "26ffc4555821210bb96d7d4dede1040978ccbfe97ffecc5b578c48dc58b0c415",
    "portfolio_state.json": "014d8033007a922a566fb63f46396f402587e05ba2901ecd02de9cb073865fcf",
    "state_p4.json": "f48b92e2de87918d15a0a5871172ff1f6d9de97e907f717dd516548f78d9cb75",
    "state_p5.json": "b22bb32a13e32167fc15c321b12933c93dd3a6e0350d852387cdb727e0337b07",
    "portfoy.json": "5e8829f2449257be372997e6085d51ae40289e316acd97ca87c32163ff721b42",
    "portfoy_p2.json": "06177b698d4739b05f9cd0421944275f6682f6d8d304c9992b8dfbfb91d3ff6d",
    "tarama_listesi.json": "9940efb60f919c3728d83ec58c9a119f83c20a018a842b90c8bbd319656550b4",
    "tarama_listesi_p2.json": "f037faa0aa86d3792549e2b8ac030e3e8f0142469c4dba31364fddf7992a0889",
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
    "mott_state_coordination.py",
]

PRODUCTION_SHA256 = {
    "dsp_core.py": "8e5fc88c9d90abb0bd893008b6d626357771a0d7c4b38bc3cbad867776405ea1",
    "dsp_strategies.py": "ae108ff65769d69b30b6a46c416fdfebd35cab0c20b34f5d0fe0ff6adcc44200",
    "dsp_trend.py": "dea1c9ad7f86158205322ac6fbef0e58bc14b111ece8d4ec9be232d6b00e143b",
    "scanner_dsp.py": "311ae33ae94a675dfe87c2a19b8f6e8ed6ab78b658bc616314331492b0d14ef6",
    "simulate_dsp.py": "37faf55571fe19664e4215fd91cbf8a7a473962efdd667b130ad631f1f588ff7",
    "meta_portfolio.py": "6fbdbccde94bf8adc057cc915de9cf31e6a53202ea6d4a3d881f3e943ec66ee0",
    "mott_state.py": "078398246937a464d4aac33ea566b379c5482fea1335b33ae3ab26662b1e1756",
    "mott_risk.py": "0d03197b313ed3cf78b259a7eea8b108fccdf16d1cc862cf345c9f6aeac2758f",
    "mott_telegram.py": "53fab3e86fb9ed5f012f12c9abcf0d9ee5e42266424b6b75bff687e557a2b935",
    "mott_fiyat.py": "9f598f015666619030c8aafef651e424f4df2fb3e168f26cba73577cd7227fa1",
    "mott_portfoy_deger.py": "818f19c582123d5816918f8cfd7f47380f05ccc0c96bf9eb76c091c056fce4e4",
    "mott_performans_analiz.py": "0268b9ac8bb70b5a7029b2308e5b58c01bb134102a0c84653ab986fe64251752",
    "mott_aylik_rapor.py": "6793f0dbb2138fe5fd5eba831bc7460c550a04d7f8a109557d5dd5574993d12e",
    "p5_committee.py": "7f9b48f93c560fe8269304c9b09237456e9c0f2b010787c055273cd022b06074",
    "portfoy_yonetici.py": "004d260f09764093c09c49742da1e90098bd5594896eb4cdf0f96015b5942d12",
    "scanner_p1.py": "a50c1469f24b365dcbd957d3ac3ff3a4bff7873a9bb497cf13447df997654fb2",
    "scanner_smc.py": "add62c77066d1ca92e1ab80277cb13bd9bc3a671ad93f5b5c54685147e232fe8",
    "mott_state_coordination.py": "f50023ab93ed285fc827d642cff2cf4a61021ec0d2d4a7223317bd3c62837b3a",
}

REQUIREMENTS_SHA256 = "ee81e5bd95bfb034beaf62bc7234abaf150dfcff372ebbe17a1ebb87a8b302fc"

# ---------------------------------------------------------------------------
# Baseline artefaktları — 6/6 tam SHA-256 (repo DIŞINDA: dsp-p3-baseline/)
# CI'da bu dosyalar bulunmaz; bu sabitler yerel golden doğrulamasının
# beklenen değerleridir.
# ---------------------------------------------------------------------------
ARTEFAKT_SHA256 = {
    "manifest.json": "7587c20f6b97b8a0c17a8377fb310ddb53c76eba885809898d2a14577b694848",
    "p4_ic_baseline.txt": "5b6658ecfe4f61f794ada2b046bdce0e78ff28df9feaf5fe45c297525bc21543",
    "state_sha256.txt": "285979cead461f61741e8eed4358811f677f2d733848071f72582f30c5c74808",
    "mott_state_normalize_output.txt": "045a55b0cb5dc40f9be8123f68316b1c47442418ab746e4bff20d0c587eca67d",
    "mott_performans_analiz_output.txt": "5d41c287ab4c22fc849d06b7ddf9bdb675b8beba38ca10cf2881c8cae9cf51a0",
    "mott_aylik_rapor_output.txt": "dcbee405a16ba79f3337cb272f89fb6c2d2c4f0be452a4c70f34780818ee1558",
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
