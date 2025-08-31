"""
Unit tests for regime router functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from ultra_signals.engine.regime_router import RegimeRouter


class TestRegimeRouter:
    """Test regime router functionality."""
    
    def test_detect_trend_regime(self):
        """Test trend regime detection."""
        # Mock features with trend characteristics
        features = {
            'trend': Mock(
                adx=25.0,  # Above threshold
                ema_short=105.0,
                ema_medium=103.0,
                ema_long=100.0
            ),
            'volatility': Mock(atr=2.0, atr_percentile=0.6),
            'momentum': Mock(rsi=55.0, macd_hist=0.01),
            'flow_metrics': Mock(volume_z=0.5, cvd=0.02),
            'volume_flow': Mock()
        }
        
        # Mock current price
        features['trend'].current_price = 105.0
        
        settings = {
            'regimes': {
                'adx_trend_min': 18,
                'squeeze_bbkc_ratio_max': 1.1,
                'range_adx_max': 14,
                'breakout_vol_burst_z': 1.5
            }
        }
        
        regime = RegimeRouter.detect_regime(features, settings)
        assert regime == "trend"
    
    def test_detect_range_regime(self):
        """Test range regime detection."""
        # Mock features with range characteristics
        features = {
            'trend': Mock(adx=12.0),  # Below threshold
            'volatility': Mock(
                atr=1.0,
                atr_percentile=0.2,  # Low volatility
                bbands_upper=106.0,
                bbands_lower=94.0
            ),
            'momentum': Mock(rsi=50.0, macd_hist=0.0),
            'flow_metrics': Mock(volume_z=0.2, cvd=0.01),
            'volume_flow': Mock()
        }
        
        # Mock current price near BB edge
        features['trend'].current_price = 105.5  # Near upper edge
        
        settings = {
            'regimes': {
                'adx_trend_min': 18,
                'squeeze_bbkc_ratio_max': 1.1,
                'range_adx_max': 14,
                'breakout_vol_burst_z': 1.5
            }
        }
        
        regime = RegimeRouter.detect_regime(features, settings)
        assert regime == "range"
    
    def test_detect_breakout_regime(self):
        """Test breakout regime detection."""
        # Mock features with breakout characteristics
        features = {
            'trend': Mock(adx=20.0),
            'volatility': Mock(atr=3.0, atr_percentile=0.8),  # High volatility
            'momentum': Mock(rsi=60.0, macd_hist=0.02),
            'flow_metrics': Mock(volume_z=2.0, cvd=0.15),  # Volume burst + strong flow
            'volume_flow': Mock()
        }
        
        # Mock current price
        features['trend'].current_price = 105.0
        
        settings = {
            'regimes': {
                'adx_trend_min': 18,
                'squeeze_bbkc_ratio_max': 1.1,
                'range_adx_max': 14,
                'breakout_vol_burst_z': 1.5
            }
        }
        
        regime = RegimeRouter.detect_regime(features, settings)
        assert regime == "breakout"
    
    def test_detect_mean_revert_regime(self):
        """Test mean reversion regime detection."""
        # Mock features with mean reversion characteristics
        features = {
            'trend': Mock(adx=15.0),
            'volatility': Mock(atr=2.0, atr_percentile=0.5),
            'momentum': Mock(rsi=75.0, macd_hist=0.02),  # RSI extreme
            'flow_metrics': Mock(volume_z=0.8, cvd=0.05),
            'volume_flow': Mock()
        }
        
        # Mock current price
        features['trend'].current_price = 105.0
        
        settings = {
            'regimes': {
                'adx_trend_min': 18,
                'squeeze_bbkc_ratio_max': 1.1,
                'range_adx_max': 14,
                'breakout_vol_burst_z': 1.5
            }
        }
        
        regime = RegimeRouter.detect_regime(features, settings)
        assert regime == "mean_revert"
    
    def test_detect_chop_regime(self):
        """Test choppy regime detection."""
        # Mock features with choppy characteristics
        features = {
            'trend': Mock(adx=12.0),  # Low ADX
            'volatility': Mock(atr=1.0, atr_percentile=0.15),  # Very low volatility
            'momentum': Mock(rsi=50.0, macd_hist=0.0),
            'flow_metrics': Mock(volume_z=0.3, cvd=0.01),
            'volume_flow': Mock()
        }
        
        # Mock current price
        features['trend'].current_price = 105.0
        
        settings = {
            'regimes': {
                'adx_trend_min': 18,
                'squeeze_bbkc_ratio_max': 1.1,
                'range_adx_max': 14,
                'breakout_vol_burst_z': 1.5
            }
        }
        
        regime = RegimeRouter.detect_regime(features, settings)
        assert regime == "chop"
    
    def test_detect_mixed_regime(self):
        """Test mixed regime when no clear pattern."""
        # Mock features with mixed characteristics
        features = {
            'trend': Mock(adx=16.0),  # Medium ADX
            'volatility': Mock(atr=2.0, atr_percentile=0.4),  # Medium volatility
            'momentum': Mock(rsi=55.0, macd_hist=0.01),  # Neutral RSI
            'flow_metrics': Mock(volume_z=0.6, cvd=0.03),  # Medium flow
            'volume_flow': Mock()
        }
        
        # Mock current price
        features['trend'].current_price = 105.0
        
        settings = {
            'regimes': {
                'adx_trend_min': 18,
                'squeeze_bbkc_ratio_max': 1.1,
                'range_adx_max': 14,
                'breakout_vol_burst_z': 1.5
            }
        }
        
        regime = RegimeRouter.detect_regime(features, settings)
        assert regime == "mixed"
    
    def test_safe_float_conversion(self):
        """Test safe float conversion."""
        # Test valid values
        assert RegimeRouter._safe_float(10.5) == 10.5
        assert RegimeRouter._safe_float("10.5") == 10.5
        assert RegimeRouter._safe_float(10) == 10.0
        
        # Test invalid values
        assert RegimeRouter._safe_float(None) is None
        assert RegimeRouter._safe_float(float('nan')) is None
        assert RegimeRouter._safe_float(float('inf')) is None
        assert RegimeRouter._safe_float("invalid") is None
    
    def test_get_current_price(self):
        """Test current price extraction."""
        # Mock features with price
        features = {
            'trend': Mock(current_price=105.0),
            'volatility': Mock(),
            'momentum': Mock()
        }
        
        price = RegimeRouter._get_current_price(features)
        assert price == 105.0
        
        # Test fallback
        features = {
            'trend': Mock(),
            'volatility': Mock(),
            'momentum': Mock(price=104.0)
        }
        
        price = RegimeRouter._get_current_price(features)
        assert price == 104.0
        
        # Test no price found
        features = {
            'trend': Mock(),
            'volatility': Mock(),
            'momentum': Mock()
        }
        
        price = RegimeRouter._get_current_price(features)
        assert price is None
    
    def test_pick_alphas(self):
        """Test alpha strategy selection."""
        settings = {
            'alpha_profiles': {
                'trend': {
                    'alphas': ['breakout_v2', 'volume_surge'],
                    'weight_scale': 1.2,
                    'min_confidence': 0.65
                },
                'mean_revert': {
                    'alphas': ['bollinger_fade', 'rsi_extreme'],
                    'weight_scale': 1.1,
                    'min_confidence': 0.60
                },
                'chop': {
                    'alphas': [],
                    'weight_scale': 0.2,
                    'min_confidence': 0.7
                }
            }
        }
        
        # Test trend regime
        alphas, config = RegimeRouter.pick_alphas('trend', settings)
        assert alphas == ['breakout_v2', 'volume_surge']
        assert config['weight_scale'] == 1.2
        
        # Test mean_revert regime
        alphas, config = RegimeRouter.pick_alphas('mean_revert', settings)
        assert alphas == ['bollinger_fade', 'rsi_extreme']
        assert config['weight_scale'] == 1.1
        
        # Test fallback to trend for unknown regime
        alphas, config = RegimeRouter.pick_alphas('unknown', settings)
        assert alphas == ['breakout_v2', 'volume_surge']
    
    def test_route(self):
        """Test complete routing functionality."""
        # Mock features
        features = {
            'trend': Mock(
                adx=25.0,
                ema_short=105.0,
                ema_medium=103.0,
                ema_long=100.0,
                current_price=105.0
            ),
            'volatility': Mock(atr=2.0, atr_percentile=0.6),
            'momentum': Mock(rsi=55.0, macd_hist=0.01),
            'flow_metrics': Mock(volume_z=0.5, cvd=0.02),
            'volume_flow': Mock()
        }
        
        settings = {
            'regimes': {
                'adx_trend_min': 18,
                'squeeze_bbkc_ratio_max': 1.1,
                'range_adx_max': 14,
                'breakout_vol_burst_z': 1.5
            },
        'alpha_profiles': {
                'trend': {
                    'alphas': ['breakout_v2', 'volume_surge'],
                    'weight_scale': 1.2,
                    'min_confidence': 0.65
                }
            }
        }
        
        result = RegimeRouter.route(features, settings)
        
        assert result['regime'] == 'trend'
        assert result['detected'] == 'trend'
        assert result['alphas'] == ['breakout_v2', 'volume_surge']
        assert result['alphas_used'] == ['breakout_v2', 'volume_surge']
        assert result['weight_scale'] == 1.2
        assert result['confidence_boost'] == 1.2
        assert result['min_confidence'] == 0.65
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with missing features
        features = {}
        settings = {'regimes': {}}
        
        regime = RegimeRouter.detect_regime(features, settings)
        assert regime == "mixed"
        
        # Test with None values
        features = {
            'trend': Mock(adx=None, ema_short=None, ema_medium=None, ema_long=None),
            'volatility': Mock(atr=None, atr_percentile=None),
            'momentum': Mock(rsi=None, macd_hist=None),
            'flow_metrics': Mock(volume_z=None, cvd=None),
            'volume_flow': Mock()
        }
        
        regime = RegimeRouter.detect_regime(features, settings)
        assert regime == "mixed"
        
        # Test with NaN values
        features = {
            'trend': Mock(adx=float('nan')),
            'volatility': Mock(atr_percentile=float('nan')),
            'momentum': Mock(rsi=float('nan')),
            'flow_metrics': Mock(volume_z=float('nan')),
            'volume_flow': Mock()
        }
        
        regime = RegimeRouter.detect_regime(features, settings)
        assert regime == "mixed"


if __name__ == "__main__":
    pytest.main([__file__])
