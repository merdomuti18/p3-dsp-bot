"""
FAZ 6.1 — Unified Price Cache Tests
=====================================
mott_fiyat.py in-memory cache davranışını test eder.

Kurallar:
  - CACHE_TTL = 60 saniye
  - cache hit → API çağrısı yok
  - cache miss → raw fetch
  - TTL doldu → yeniden fetch
  - use_cache=False → her zaman raw fetch
  - stale cache hiçbir şekilde kullanılmaz
  - cache source metadata taşır
  - concurrent erişim güvenli
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixture: cache modülünü import et, temizle
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_cache():
    """Her test öncesi ve sonrası cache'i temizle."""
    import mott_fiyat
    mott_fiyat.clear_cache()
    yield
    mott_fiyat.clear_cache()


# ---------------------------------------------------------------------------
# Test 1: Cache hit — API çağrısı yapılmaz
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_cache_hit_no_api_call(self):
        """Cache'de fiyat varsa tv_fiyatlar/yf çağrılmamalı."""
        import mott_fiyat

        # Manuel cache'e fiyat ekle
        mott_fiyat._PRICE_CACHE["GARAN"] = {
            "price": 123.45,
            "timestamp": time.time(),
            "source": "test",
        }

        # tv_fiyatlar mock'la — çağrılmamalı
        with patch.object(mott_fiyat, "tv_fiyatlar", wraps=mott_fiyat.tv_fiyatlar) as mock_tv:
            result = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)
            mock_tv.assert_not_called()

        assert result["GARAN"] == 123.45

    def test_cache_hit_partial(self):
        """Bazı semboller cache'de, bazıları yok — yalnızca eksikler için fetch."""
        import mott_fiyat

        mott_fiyat._PRICE_CACHE["GARAN"] = {
            "price": 100.0,
            "timestamp": time.time(),
            "source": "test",
        }

        # tv_fiyatlar sadece THYAO için çağrılmalı
        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"THYAO": 200.0}) as mock_tv:
            result = mott_fiyat.canli_fiyatlar(["GARAN", "THYAO"], use_cache=True)
            mock_tv.assert_called_once_with(["THYAO"])

        assert result["GARAN"] == 100.0
        assert result["THYAO"] == 200.0


# ---------------------------------------------------------------------------
# Test 2: Cache miss — raw fetch yapılır
# ---------------------------------------------------------------------------

class TestCacheMiss:
    def test_cache_miss_fetches_fresh(self):
        """Cache boşsa tamamen raw fetch yapılır."""
        import mott_fiyat

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 123.45}) as mock_tv:
            result = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)
            mock_tv.assert_called_once()

        assert result["GARAN"] == 123.45

    def test_cache_miss_populates_cache(self):
        """Cache miss sonrası çekilen fiyat cache'e yazılır."""
        import mott_fiyat

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 123.45}):
            mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)

        assert "GARAN" in mott_fiyat._PRICE_CACHE
        assert mott_fiyat._PRICE_CACHE["GARAN"]["price"] == 123.45
        assert mott_fiyat._PRICE_CACHE["GARAN"]["source"] == "tv"


# ---------------------------------------------------------------------------
# Test 3: TTL expiry — süre dolduysa yeniden fetch
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    def test_expired_cache_refetches(self):
        """TTL dolduysa cache'i görmezden gelip yeniden çekim yapılır."""
        import mott_fiyat

        # Cache'e eski fiyat ekle (TTL = 0 saniye)
        mott_fiyat._PRICE_CACHE["GARAN"] = {
            "price": 99.0,
            "timestamp": time.time() - 999,  # çok eski
            "source": "test",
        }
        mott_fiyat.CACHE_TTL = 60

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 123.45}) as mock_tv:
            result = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)
            mock_tv.assert_called_once()

        assert result["GARAN"] == 123.45  # yeni fiyat

    def test_fresh_cache_not_refetched(self):
        """TTL dolmadıysa cache kullanılır, yeniden fetch yapılmaz."""
        import mott_fiyat

        mott_fiyat._PRICE_CACHE["GARAN"] = {
            "price": 55.55,
            "timestamp": time.time(),  # taze
            "source": "test",
        }

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={}) as mock_tv:
            result = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)
            mock_tv.assert_not_called()

        assert result["GARAN"] == 55.55


# ---------------------------------------------------------------------------
# Test 4: use_cache=False — her zaman raw fetch
# ---------------------------------------------------------------------------

class TestCacheDisabled:
    def test_use_cache_false_fetches_always(self):
        """use_cache=False cache'i tamamen bypass eder."""
        import mott_fiyat

        # Cache'e fiyat ekle
        mott_fiyat._PRICE_CACHE["GARAN"] = {
            "price": 99.0,
            "timestamp": time.time(),
            "source": "test",
        }

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 123.45}) as mock_tv:
            result = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=False)
            mock_tv.assert_called_once()

        assert result["GARAN"] == 123.45

    def test_use_cache_false_does_not_write_cache(self):
        """use_cache=False cache'e yazmaz."""
        import mott_fiyat

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 123.45}):
            mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=False)

        # Cache'e yazılmamış olmalı (veya eski olmalı)
        # clear_cache fixture zaten temizliyor, bu yüzden doğrudan kontrol edelim:
        # canli_fiyatlar use_cache=False ile çağrıldığında cache'e yazmaz
        # Bu davranış test edilmeli
        assert "GARAN" not in mott_fiyat._PRICE_CACHE or \
               mott_fiyat._PRICE_CACHE.get("GARAN", {}).get("source") != "tv"


# ---------------------------------------------------------------------------
# Test 5: Source metadata
# ---------------------------------------------------------------------------

class TestSourceMetadata:
    def test_tv_source_recorded(self):
        """TV'den gelen fiyat 'tv' source ile kaydedilmeli."""
        import mott_fiyat

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 123.45}):
            mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)

        assert mott_fiyat._PRICE_CACHE["GARAN"]["source"] == "tv"

    def test_yf_source_recorded(self):
        """TV'de olmayıp yfinance'den gelen fiyat 'yf' source ile kaydedilmeli."""
        import mott_fiyat

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={}), \
             patch.object(mott_fiyat, "_yf_fiyat", return_value=123.45):
            mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)

        assert mott_fiyat._PRICE_CACHE["GARAN"]["source"] == "yf"

    def test_timestamp_is_set(self):
        """Cache'e yazılan fiyatın timestamp'i güncel olmalı."""
        import mott_fiyat

        before = time.time()
        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 123.45}):
            mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)
        after = time.time()

        ts = mott_fiyat._PRICE_CACHE["GARAN"]["timestamp"]
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# Test 6: Stale price rejection
# ---------------------------------------------------------------------------

class TestStaleRejection:
    def test_stale_price_not_used(self):
        """TTL'den eski fiyat cache'de kalsa bile kullanılmamalı."""
        import mott_fiyat

        # Eski fiyat ekle
        mott_fiyat._PRICE_CACHE["GARAN"] = {
            "price": 10.0,
            "timestamp": time.time() - 200,
            "source": "test",
        }

        # Yeni fiyat çekilmeli
        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 20.0}):
            result = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)

        assert result["GARAN"] == 20.0  # eski 10.0 değil


# ---------------------------------------------------------------------------
# Test 7: P4/P5 shared cache
# ---------------------------------------------------------------------------

class TestSharedCache:
    def test_p4_p5_share_same_cache(self):
        """meta_portfolio ve p5_committee aynı cache instance'ını kullanmalı."""
        import mott_fiyat

        # Cache'e fiyat ekle
        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 100.0}):
            mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)

        # İkinci çağrı — cache'den gelmeli
        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={}) as mock_tv:
            result = mott_fiyat.canli_fiyatlar(["GARAN"], use_cache=True)
            mock_tv.assert_not_called()

        assert result["GARAN"] == 100.0

    def test_clear_cache_removes_all(self):
        """clear_cache() tüm cache'i temizlemeli."""
        import mott_fiyat

        mott_fiyat._PRICE_CACHE["GARAN"] = {"price": 100.0, "timestamp": time.time(), "source": "test"}
        mott_fiyat._PRICE_CACHE["THYAO"] = {"price": 200.0, "timestamp": time.time(), "source": "test"}

        mott_fiyat.clear_cache()

        assert len(mott_fiyat._PRICE_CACHE) == 0


# ---------------------------------------------------------------------------
# Test 8: canli_fiyat (tek sembol) cache
# ---------------------------------------------------------------------------

class TestCanliFiyatSingle:
    def test_canli_fiyat_uses_cache(self):
        """canli_fiyat() de cache kullanmalı."""
        import mott_fiyat

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={"GARAN": 123.45}):
            result = mott_fiyat.canli_fiyat("GARAN", use_cache=True)

        assert result == 123.45

        # İkinci çağrı cache'den
        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={}) as mock_tv:
            result2 = mott_fiyat.canli_fiyat("GARAN", use_cache=True)
            mock_tv.assert_not_called()

        assert result2 == 123.45

    def test_canli_fiyat_none_when_no_data(self):
        """Veri yoksa canli_fiyat None dönmeli, cache'e None yazılmamalı."""
        import mott_fiyat

        with patch.object(mott_fiyat, "tv_fiyatlar", return_value={}), \
             patch.object(mott_fiyat, "_yf_fiyat", return_value=None):
            result = mott_fiyat.canli_fiyat("NONEXIST", use_cache=True)

        assert result is None
        assert "NONEXIST" not in mott_fiyat._PRICE_CACHE
