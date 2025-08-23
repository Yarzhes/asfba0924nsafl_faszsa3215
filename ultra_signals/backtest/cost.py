from typing import List, Dict, Any
import pandas as pd

class CostModel:
    """Base class for cost models."""
    def calculate(self, **kwargs) -> float:
        raise NotImplementedError

class TakerFeeModel(CostModel):
    """
    Calculates transaction costs based on a fixed taker fee rate.
    """
    def __init__(self, config: Dict[str, Any]):
        self.fee_bps = float(config.get("fee_bps", 4.0)) # Default to 4 bps

    def calculate(self, trade_value: float, **kwargs) -> float:
        """
        Calculates the fee for a single trade.
        
        Args:
            trade_value: The notional value of the trade.
        
        Returns:
            The calculated fee amount.
        """
        return abs(trade_value) * (self.fee_bps / 10000.0)

class FundingCostModel(CostModel):
    """
    Calculates funding costs accrued over the duration of a bar.
    It uses a pre-computed trail of funding rates.
    """
    def __init__(self, funding_rates: pd.DataFrame):
        # Expects a DataFrame with 'timestamp' and 'rate' columns
        self.funding_rates = funding_rates.set_index('timestamp').sort_index()

    def calculate(self, position_value: float, start_time: pd.Timestamp, end_time: pd.Timestamp, **kwargs) -> float:
        """
        Calculates total funding costs for a position over a bar's duration.
        
        Args:
            position_value: The notional value of the open position.
            start_time: The start timestamp of the bar.
            end_time: The end timestamp of the bar.
            
        Returns:
            The total funding cost (positive) or payment (negative).
        """
        if self.funding_rates.empty:
            return 0.0
        
        # Find all funding events that occurred within this bar
        relevant_funding = self.funding_rates.loc[start_time:end_time]
        
        if relevant_funding.empty:
            return 0.0
            
        # Funding is paid on the notional value of the position
        # A positive position_value (long) pays the funding rate.
        # A negative position_value (short) receives the funding rate.
        total_funding_cost = (position_value * relevant_funding['rate']).sum()
        
        return total_funding_cost
