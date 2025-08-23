import pytest
from ultra_signals.backtest.slippage import ATROrderSlippage, BookProxySlippage

def test_atr_slippage_buy():
    """Test ATR slippage for a buy order."""
    config = {"atr_slippage_multiplier": 0.5}
    model = ATROrderSlippage(config)
    
    price = 100
    atr = 2.0
    # For a buy, slippage should increase the price
    adjusted_price = model.calculate(price=price, atr=atr, side='BUY')
    
    # Expected slippage = 2.0 * 0.5 = 1.0
    assert adjusted_price == 101.0

def test_atr_slippage_sell():
    """Test ATR slippage for a sell order."""
    config = {"atr_slippage_multiplier": 0.5}
    model = ATROrderSlippage(config)
    
    price = 100
    atr = 2.0
    # For a sell, slippage should decrease the price
    adjusted_price = model.calculate(price=price, atr=atr, side='SELL')
    
    assert adjusted_price == 99.0

def test_book_proxy_slippage_buy():
    """Test book proxy slippage for a buy order."""
    config = {"book_base_slippage_bps": 1.0, "book_size_sensitivity": 2.0}
    model = BookProxySlippage(config)
    
    price = 100
    trade_size = 10
    book_depth = 20 # Trade is 50% of book depth
    
    # size_ratio = 10 / 20 = 0.5
    # slippage_bps = 1.0 * (1 + 0.5)^2 = 1.0 * 1.5^2 = 2.25 bps
    # slippage_pct = 0.000225
    # adjusted_price = 100 * (1 + 0.000225) = 100.0225
    adjusted_price = model.calculate(price=price, trade_size=trade_size, book_depth=book_depth, side='BUY')
    
    assert adjusted_price == pytest.approx(100.0225)

def test_book_proxy_slippage_sell():
    """Test book proxy slippage for a sell order."""
    config = {"book_base_slippage_bps": 1.0, "book_size_sensitivity": 2.0}
    model = BookProxySlippage(config)
    
    price = 100
    trade_size = 10
    book_depth = 20
    
    adjusted_price = model.calculate(price=price, trade_size=trade_size, book_depth=book_depth, side='SELL')
    
    assert adjusted_price == pytest.approx(99.9775)