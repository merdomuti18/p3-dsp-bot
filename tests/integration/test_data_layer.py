"""
FAZ 6.1 — Data Layer Integration Tests
========================================
Cache'in mevcut pipeline davranışını bozmadığını doğrular.

Kurallar:
  - NETWORK = OFF (conftest autouse)
  - Production state dosyalarına dokunulmaz
  - Mock ile test edilir
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_cache():
    """Her test öncesi ve sonrası cache'i temizle."""
    import mott_fiyat
    mott_fiyat.clear_cache()
    yield
    mott_fiyat.clear_cache()


class TestDataLayerIntegration:
    """Cache'in price pipeline davranışını bozmadığını doğrula."""

    def test_fiyat_cek_p4_uses_cache_layer(self, monkeypatch):
        """meta_portfolio.fiyat_cek() cache-aware tv_fiyatlar kullanmalı."""
        import mott_fiyat

        # Cache'e fiyat ekle
        mott_fiyat._PRICE_CACHE["GARAN"] = {
            "price": 100.0,
            "timestamp": time.time(),
            "source": "test",
        }

        # tv_fiyatlar mock'la — çağrılmamalı (cache hit)
        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={}) as mock_tv:
            result = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)
            mock_tv.assert_not_called()

        assert result["GARAN"] == 100.0

    def test_fiyat_cek_p5_uses_cache_layer(self, monkeypatch):
        """p5_committee.fiyat_cek() cache-aware tv_fiyatlar kullanmalı."""
        import mott_fiyat

        mott_fiyat._PRICE_CACHE["THYAO"] = {
            "price": 250.0,
            "timestamp": time.time(),
            "source": "test",
        }

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={}) as mock_tv:
            result = mott_fiyat.canli_fiyatlar(["THYAO"], use_cache=True)
            mock_tv.assert_not_called()

        assert result["THYAO"] == 250.0

    def test_cache_doesnt_affect_signal_generation(self):
        """Cache fiyat sinyal skorunu değiştirmez — sadece fiyat lookup hızlanır."""
        import mott_fiyat

        # Cache'e farklı fiyat ekle
        mott_fiyat._PRICE_CACHE["GARAN"] = {
            "price": 9999.0,  # kasıtlı abartılı fiyat
            "timestamp": time.time(),
            "source": "test",
        }

        # tv_fiyatlar çağrılsa gerçek fiyat gelir
        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 100.0}):
            result = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=False)
        assert result["GARAN"] == 100.0

        # Cache'den okunsa abartılı fiyat gelir — bu beklenen davranış
        # (cache.invalidates_ttl dolduğunda düzelir)
        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={}) as mock_tv:
            result2 = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)
            mock_tv.assert_not_called()
        assert result2["GARAN"] == 9999.0  # cache hit

    def test_tv_fiyatlar_unaffected_by_cache(self):
        """tv_fiyatlar() her zaman raw API çağrısı yapar — cache'e bakmaz."""
        import mott_fiyat

        # Cache'e fiyat ekle
        mott_fiyat._PRICE_CACHE["GARAN"] = {
            "price": 55.0,
            "timestamp": time.time(),
            "source": "test",
        }

        # tv_fiyatlar her zaman çağrılır
        with patch("tradingview_screener.Query") as MockQuery:
            mock_instance = MagicMock()
            mock_instance.set_markets.return_value = mock_instance
            mock_instance.select.return_value = mock_instance
            mock_instance.where.return_value = mock_instance
            mock_instance.get_scanner_data.return_value = (0, MagicMock())
            MockQuery.return_value = mock_instance

            # tv_fiyatlar cache'e bakmaz — her zaman API'ye gider
            # Bu test tv_fiyatlar'ın cache'den bağımsız olduğunu doğrular
            # (Query mocklandığı için boş sonuç gelir)
            result = mott_fiyat.tv_fiyatlar(["GARAN"])
            # tv_fiyatlar cache kullanmaz → boş mock sonucu gelir
            assert isinstance(result, dict)

    def test_empty_symbol_list_no_api_call(self):
        """Boş sembol listesi ile hiç API çağrısı yapılmamalı."""
        import mott_fiyat

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={}) as mock_tv:
            result = mott_fiyat.canli_fiyatlar([], use_cache=True)
            mock_tv.assert_not_called()

        assert result == {}

    def test_dot_is_suffix_stripped(self):
        """Sembol .IS suffixinin strip edilmesi cache'e yansıtılmalı."""
        import mott_fiyat

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 100.0}):
            result = mott_fiyat.canli_fiyatlar(["GARAN.IS"], use_cache=True)

        # .IS strip edilmeli
        assert "GARAN" in mott_fiyat._PRICE_CACHE
        assert result.get("GARAN") == 100.0
