"""
Unit tests for tick rounding functionality.
"""

import pytest
import numpy as np

from ultra_signals.core.market_meta import (
    get_tick_size, round_to_tick, round_price_for_symbol,
    format_price_for_display, validate_tick_size,
    get_symbol_info, batch_round_prices
)


class TestTickSize:
    """Test tick size functionality."""
    
    def test_get_tick_size_major_pairs(self):
        """Test tick size for major pairs."""
        assert get_tick_size("BTCUSDT") == 0.1
        assert get_tick_size("ETHUSDT") == 0.01
        assert get_tick_size("BNBUSDT") == 0.01
        assert get_tick_size("SOLUSDT") == 0.01
        assert get_tick_size("XRPUSDT") == 0.0001
        assert get_tick_size("DOGEUSDT") == 0.00001
    
    def test_get_tick_size_minor_pairs(self):
        """Test tick size for minor pairs."""
        assert get_tick_size("1000SHIBUSDT") == 0.00000001
        assert get_tick_size("SHIBUSDT") == 0.00000001
        assert get_tick_size("PEPEUSDT") == 0.00000001
        assert get_tick_size("WIFUSDT") == 0.0001
        assert get_tick_size("JUPUSDT") == 0.0001
        assert get_tick_size("ORDIUSDT") == 0.01
    
    def test_get_tick_size_unknown_symbol(self):
        """Test tick size for unknown symbol."""
        assert get_tick_size("UNKNOWNUSDT") == 0.0001  # Default
    
    def test_get_tick_size_with_settings_override(self):
        """Test tick size with settings override."""
        settings = {
            'formatting': {
                'tick_size_overrides': {
                    'BTCUSDT': 0.05,
                    'CUSTOMUSDT': 0.001
                }
            }
        }
        
        assert get_tick_size("BTCUSDT", settings) == 0.05
        assert get_tick_size("CUSTOMUSDT", settings) == 0.001
        assert get_tick_size("ETHUSDT", settings) == 0.01  # Not overridden


class TestRoundToTick:
    """Test round to tick functionality."""
    
    def test_round_to_tick_btc(self):
        """Test rounding for BTCUSDT."""
        tick_size = 0.1
        
        # Test exact ticks
        assert round_to_tick(50000.0, tick_size) == 50000.0
        assert round_to_tick(50000.1, tick_size) == 50000.1
        assert round_to_tick(50000.2, tick_size) == 50000.2
        
        # Test rounding up
        assert round_to_tick(50000.06, tick_size) == 50000.1
        assert round_to_tick(50000.16, tick_size) == 50000.2
        
        # Test rounding down
        assert round_to_tick(50000.04, tick_size) == 50000.0
        assert round_to_tick(50000.14, tick_size) == 50000.1
        
        # Test exact halfway (rounds to even)
        assert round_to_tick(50000.05, tick_size) == 50000.0
    
    def test_round_to_tick_eth(self):
        """Test rounding for ETHUSDT."""
        tick_size = 0.01
        
        # Test exact ticks
        assert round_to_tick(3000.00, tick_size) == 3000.00
        assert round_to_tick(3000.01, tick_size) == 3000.01
        assert round_to_tick(3000.02, tick_size) == 3000.02
        
        # Test rounding up
        assert round_to_tick(3000.006, tick_size) == 3000.01
        assert round_to_tick(3000.016, tick_size) == 3000.02
        
        # Test rounding down
        assert round_to_tick(3000.004, tick_size) == 3000.00
        assert round_to_tick(3000.014, tick_size) == 3000.01
        
        # Test exact halfway (rounds to even)
        assert round_to_tick(3000.005, tick_size) == 3000.00
    
    def test_round_to_tick_doge(self):
        """Test rounding for DOGEUSDT."""
        tick_size = 0.00001
        
        # Test exact ticks
        assert round_to_tick(0.10000, tick_size) == 0.10000
        assert round_to_tick(0.10001, tick_size) == 0.10001
        assert round_to_tick(0.10002, tick_size) == 0.10002
        
        # Test rounding up
        assert round_to_tick(0.100006, tick_size) == 0.10001
        assert round_to_tick(0.100016, tick_size) == 0.10002
        
        # Test rounding down
        assert round_to_tick(0.100004, tick_size) == 0.10000
        assert round_to_tick(0.100014, tick_size) == 0.10001
        
        # Test exact halfway (rounds to even)
        assert round_to_tick(0.100005, tick_size) == 0.10000
    
    def test_round_to_tick_invalid_tick_size(self):
        """Test rounding with invalid tick size."""
        # Zero tick size
        assert round_to_tick(50000.0, 0.0) == 50000.0  # Uses default
        
        # Negative tick size
        assert round_to_tick(50000.0, -0.1) == 50000.0  # Uses default
        
        # Very large tick size
        assert round_to_tick(50000.0, 1000.0) == 50000.0


class TestRoundPriceForSymbol:
    """Test round price for symbol functionality."""
    
    def test_round_price_for_symbol_btc(self):
        """Test rounding for BTCUSDT."""
        price = 50000.123
        rounded = round_price_for_symbol(price, "BTCUSDT")
        assert rounded == 50000.1  # Rounded to 0.1 tick size
    
    def test_round_price_for_symbol_eth(self):
        """Test rounding for ETHUSDT."""
        price = 3000.123
        rounded = round_price_for_symbol(price, "ETHUSDT")
        assert rounded == 3000.12  # Rounded to 0.01 tick size
    
    def test_round_price_for_symbol_doge(self):
        """Test rounding for DOGEUSDT."""
        price = 0.123456
        rounded = round_price_for_symbol(price, "DOGEUSDT")
        assert rounded == 0.12346  # Rounded to 0.00001 tick size
    
    def test_round_price_for_symbol_with_settings(self):
        """Test rounding with settings override."""
        settings = {
            'formatting': {
                'tick_size_overrides': {
                    'BTCUSDT': 0.05
                }
            }
        }
        
        price = 50000.123
        rounded = round_price_for_symbol(price, "BTCUSDT", settings)
        assert rounded == 50000.10  # Rounded to 0.05 tick size


class TestFormatPriceForDisplay:
    """Test format price for display functionality."""
    
    def test_format_price_for_display_btc(self):
        """Test formatting for BTCUSDT."""
        price = 50000.123
        formatted = format_price_for_display(price, "BTCUSDT")
        assert formatted == "50000.1"  # 1 decimal place for 0.1 tick size
    
    def test_format_price_for_display_eth(self):
        """Test formatting for ETHUSDT."""
        price = 3000.123
        formatted = format_price_for_display(price, "ETHUSDT")
        assert formatted == "3000.12"  # 2 decimal places for 0.01 tick size
    
    def test_format_price_for_display_doge(self):
        """Test formatting for DOGEUSDT."""
        price = 0.123456
        formatted = format_price_for_display(price, "DOGEUSDT")
        assert formatted == "0.12346"  # 5 decimal places for 0.00001 tick size
    
    def test_format_price_for_display_with_settings(self):
        """Test formatting with settings override."""
        settings = {
            'formatting': {
                'tick_size_overrides': {
                    'BTCUSDT': 0.05
                }
            }
        }
        
        price = 50000.123
        formatted = format_price_for_display(price, "BTCUSDT", settings)
        assert formatted == "50000.10"  # 2 decimal places for 0.05 tick size


class TestValidateTickSize:
    """Test tick size validation."""
    
    def test_validate_tick_size_valid(self):
        """Test validation of valid tick sizes."""
        assert validate_tick_size(0.1) is True
        assert validate_tick_size(0.01) is True
        assert validate_tick_size(0.00001) is True
        assert validate_tick_size(1.0) is True
        assert validate_tick_size(10.0) is True
    
    def test_validate_tick_size_invalid(self):
        """Test validation of invalid tick sizes."""
        assert validate_tick_size(0.0) is False
        assert validate_tick_size(-0.1) is False
        assert validate_tick_size(10000.0) is False  # Too large
        assert validate_tick_size("invalid") is False
        assert validate_tick_size(None) is False
        assert validate_tick_size(float('nan')) is False
        assert validate_tick_size(float('inf')) is False


class TestGetSymbolInfo:
    """Test symbol info functionality."""
    
    def test_get_symbol_info_btc(self):
        """Test symbol info for BTCUSDT."""
        info = get_symbol_info("BTCUSDT")
        
        assert info['symbol'] == "BTCUSDT"
        assert info['tick_size'] == 0.1
        assert info['decimal_places'] == 1
        assert info['min_price'] == 0.1
        assert info['max_price'] == 1000000.0
        assert info['price_precision'] == 1
    
    def test_get_symbol_info_eth(self):
        """Test symbol info for ETHUSDT."""
        info = get_symbol_info("ETHUSDT")
        
        assert info['symbol'] == "ETHUSDT"
        assert info['tick_size'] == 0.01
        assert info['decimal_places'] == 2
        assert info['min_price'] == 0.01
        assert info['max_price'] == 1000000.0
        assert info['price_precision'] == 2
    
    def test_get_symbol_info_doge(self):
        """Test symbol info for DOGEUSDT."""
        info = get_symbol_info("DOGEUSDT")
        
        assert info['symbol'] == "DOGEUSDT"
        assert info['tick_size'] == 0.00001
        assert info['decimal_places'] == 5
        assert info['min_price'] == 0.00001
        assert info['max_price'] == 1000000.0
        assert info['price_precision'] == 5
    
    def test_get_symbol_info_with_settings(self):
        """Test symbol info with settings override."""
        settings = {
            'formatting': {
                'tick_size_overrides': {
                    'BTCUSDT': 0.05
                }
            }
        }
        
        info = get_symbol_info("BTCUSDT", settings)
        
        assert info['symbol'] == "BTCUSDT"
        assert info['tick_size'] == 0.05
        assert info['decimal_places'] == 2
        assert info['min_price'] == 0.05
        assert info['max_price'] == 1000000.0
        assert info['price_precision'] == 2


class TestBatchRoundPrices:
    """Test batch round prices functionality."""
    
    def test_batch_round_prices(self):
        """Test batch rounding of multiple prices."""
        prices = {
            "BTCUSDT": 50000.123,
            "ETHUSDT": 3000.123,
            "DOGEUSDT": 0.123456
        }
        
        rounded = batch_round_prices(prices)
        
        assert rounded["BTCUSDT"] == 50000.1
        assert rounded["ETHUSDT"] == 3000.12
        assert rounded["DOGEUSDT"] == 0.12346
    
    def test_batch_round_prices_with_settings(self):
        """Test batch rounding with settings override."""
        prices = {
            "BTCUSDT": 50000.123,
            "ETHUSDT": 3000.123
        }
        
        settings = {
            'formatting': {
                'tick_size_overrides': {
                    'BTCUSDT': 0.05
                }
            }
        }
        
        rounded = batch_round_prices(prices, settings)
        
        assert rounded["BTCUSDT"] == 50000.10  # Uses override
        assert rounded["ETHUSDT"] == 3000.12  # Uses default
    
    def test_batch_round_prices_empty(self):
        """Test batch rounding with empty price dict."""
        prices = {}
        rounded = batch_round_prices(prices)
        assert rounded == {}


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_extreme_prices(self):
        """Test rounding with extreme prices."""
        # Very large price
        assert round_to_tick(999999.999, 0.1) == 1000000.0
        
        # Very small price
        assert round_to_tick(0.000001, 0.00001) == 0.0
        
        # Zero price
        assert round_to_tick(0.0, 0.1) == 0.0
    
    def test_nan_inf_prices(self):
        """Test rounding with NaN and infinity."""
        # NaN price
        with pytest.raises(ValueError):
            round_to_tick(float('nan'), 0.1)
        
        # Infinity price
        with pytest.raises(ValueError):
            round_to_tick(float('inf'), 0.1)
    
    def test_very_small_tick_sizes(self):
        """Test rounding with very small tick sizes."""
        tick_size = 0.00000001  # 8 decimal places
        price = 0.123456789
        
        rounded = round_to_tick(price, tick_size)
        assert rounded == 0.12345679  # Rounded to tick size
    
    def test_very_large_tick_sizes(self):
        """Test rounding with very large tick sizes."""
        tick_size = 100.0
        price = 50000.123
        
        rounded = round_to_tick(price, tick_size)
        assert rounded == 50000.0  # Rounded to nearest 100


if __name__ == "__main__":
    pytest.main([__file__])
