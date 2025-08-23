import pytest
import pandas as pd
from ultra_signals.backtest.cost import TakerFeeModel, FundingCostModel

def test_taker_fee_model():
    """Test the taker fee calculation."""
    config = {"fee_bps": 5.0} # 5 bps fee
    model = TakerFeeModel(config)
    
    trade_value = 10000
    fee = model.calculate(trade_value=trade_value)
    
    # Expected fee = 10000 * (5 / 10000) = 5.0
    assert fee == 5.0

def test_funding_cost_model_long_position():
    """Test funding cost for a long position."""
    funding_data = {
        "timestamp": pd.to_datetime(["2023-01-01 08:00:00", "2023-01-01 16:00:00"]),
        "rate": [0.0001, -0.00005]
    }
    funding_df = pd.DataFrame(funding_data)
    model = FundingCostModel(funding_df)
    
    position_value = 50000 # Long position
    start_time = pd.Timestamp("2023-01-01 00:00:00")
    end_time = pd.Timestamp("2023-01-01 23:59:59")
    
    # The long position pays the funding rate
    # Expected cost = (50000 * 0.0001) + (50000 * -0.00005) = 5.0 - 2.5 = 2.5
    funding_cost = model.calculate(position_value=position_value, start_time=start_time, end_time=end_time)
    
    assert funding_cost == pytest.approx(2.5)

def test_funding_cost_model_short_position():
    """Test funding cost for a short position."""
    funding_data = {
        "timestamp": pd.to_datetime(["2023-01-01 08:00:00"]),
        "rate": [0.0001]
    }
    funding_df = pd.DataFrame(funding_data)
    model = FundingCostModel(funding_df)
    
    position_value = -20000 # Short position
    start_time = pd.Timestamp("2023-01-01 00:00:00")
    end_time = pd.Timestamp("2023-01-01 12:00:00")
    
    # The short position receives the funding rate, so the cost is negative
    # Expected cost = (-20000 * 0.0001) = -2.0
    funding_cost = model.calculate(position_value=position_value, start_time=start_time, end_time=end_time)
    
    assert funding_cost == pytest.approx(-2.0)

def test_funding_cost_no_events_in_bar():
    """Test that funding cost is zero if no funding events occur in the bar."""
    funding_df = pd.DataFrame({"timestamp": [], "rate": []})
    model = FundingCostModel(funding_df)
    
    cost = model.calculate(
        position_value=10000,
        start_time=pd.Timestamp("2023-01-01 00:00"),
        end_time=pd.Timestamp("2023-01-01 01:00")
    )
    
    assert cost == 0.0