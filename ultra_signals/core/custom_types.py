"""
Custom Type Definitions
-----------------------

This module provides centralized, reusable type annotations to improve clarity
and maintain consistency across the aplication.

- Symbol: A simple string alias for a market symbol (e.g., "BTCUSDT").
- Timeframe: A simple string alias for a kline interval (e.g., "5m").
- FeatureVector: A structured Pydantic model for all computed features for a
  specific kline. This object is the primary input for the scoring engine.
"""
from enum import Enum
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel

# A simple type alias for a market symbol string (e.g., "BTCUSDT").
Symbol = str

# A simple type alias for a timeframe string (e.g., "5m").
Timeframe = str


class SignalType(str, Enum):
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"


@dataclass
class Signal:
    """
    Represents a trading signal.
    """

    symbol: str
    timeframe: str
    decision: Literal["LONG", "SHORT", "NO_TRADE"]
    signal_type: SignalType
    price: float = 0.0
    score: float = 0.0
    confidence: float = 0.0  # From 0 to 100
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)

    # Sizing information
    notional_size: float = (
        0.0  # The size of the position in quote currency (e.g., USDT)
    )
    quantity: float = 0.0  # The size of the position in base currency (e.g., BTC)


class DerivativesFeatures(BaseModel):
    """Features derived from derivatives market data like funding, OI, and liquidations."""

    funding_now: Optional[float] = None
    funding_trail: List[float] = []
    oi_delta_1m: Optional[float] = None
    oi_delta_5m: Optional[float] = None
    oi_delta_15m: Optional[float] = None
    liq_pulse: int = 0
    liq_notional_5m: Optional[float] = None


class VolatilityBucket(str, Enum):
    LOW = "low"
    MEDIUM = "med"
    HIGH = "high"


class RegimeMode(str, Enum):
    TREND = "trend"
    MEAN_REVERT = "mean_revert"


class RegimeProfile(str, Enum):
    TREND = "trend"
    MEAN_REVERT = "mean_revert"
    CHOP = "chop"


class RegimeFeatures(BaseModel):
    """Output of the market regime classification engine."""

    adx: Optional[float] = None
    vol_bucket: VolatilityBucket = VolatilityBucket.MEDIUM
    mode: RegimeMode = RegimeMode.TREND
    profile: RegimeProfile = RegimeProfile.TREND


class TrendFeatures(BaseModel):
    """Features from features/trend.py"""
    ema_short: Optional[float] = None
    ema_medium: Optional[float] = None
    ema_long: Optional[float] = None
    adx: Optional[float] = None

class MomentumFeatures(BaseModel):
    """Features from features/momentum.py"""
    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None

class VolatilityFeatures(BaseModel):
    """Features from features/volatility.py"""
    atr: Optional[float] = None
    atr_percentile: Optional[float] = None
    bbands_upper: Optional[float] = None
    bbands_lower: Optional[float] = None

class VolumeFlowFeatures(BaseModel):
    """Features from features/volume_flow.py"""
    vwap: Optional[float] = None
    volume_z_score: Optional[float] = None


class FeatureVector(BaseModel):
    """
    A structured container holding all computed features for a single kline event.
    Using Pydantic allows for optional fields and easier validation.
    """

    symbol: Symbol
    timeframe: Timeframe
    ohlcv: Dict[str, float] = {}
    orderbook: Optional[Dict[str, float]] = None
    derivatives: Optional[DerivativesFeatures] = None
    regime: Optional[RegimeFeatures] = None
    rs: Optional[Dict[str, list]] = None

    # Component Features
    trend: Optional[TrendFeatures] = None
    momentum: Optional[MomentumFeatures] = None
    volatility: Optional[VolatilityFeatures] = None
    volume_flow: Optional[VolumeFlowFeatures] = None


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeResult(str, Enum):
    TP = "TP"
    SL = "SL"
    BE = "BE"
    TRAILING_SL = "TrailingSL"
    MANUAL = "Manual"


class TradeRecord(BaseModel):
    """Represents a single completed trade for reporting and analysis."""

    ts_entry: int
    ts_exit: int
    symbol: str
    tf: str
    side: TradeSide
    entry: float
    exit: float
    sl: float
    tp1: float
    tp2: Optional[float] = None
    size: float
    fee_paid: float
    funding_paid: float
    slippage: float
    rr: float
    pnl: float
    pnl_pct: float
    result: TradeResult
    confidence_raw: float
    confidence_calibrated: Optional[float] = None
    regime: Optional[str] = None


class SubSignal(BaseModel):
    """Represents a potential trade signal from a single strategy."""

    ts: int
    symbol: str
    tf: str
    strategy_id: str
    direction: Literal["LONG", "SHORT", "FLAT"]
    confidence_calibrated: float
    reasons: Dict


class EnsembleDecision(BaseModel):
    """The result of combining multiple SubSignals."""

    ts: int
    symbol: str
    tf: str
    decision: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    subsignals: List[SubSignal]
    vote_detail: Dict
    vetoes: List[str]


class Position(BaseModel):
    """Represents an open position."""
    side: str
    size: float
    entry: float
    risk: float
    cluster: str


class PortfolioState(BaseModel):
    """Captures the current state of the trading portfolio."""

    ts: int
    equity: float
    positions: Dict[Symbol, Position]
    exposure: Dict


@dataclass
class RiskEvent:
    """Logged whenever a risk management rule is triggered."""

    ts: int
    symbol: str
    reason: str
    action: Literal["VETO", "DOWNSIZE", "COOLDOWN"]
    detail: Dict = field(default_factory=dict)


class EquityDataPoint(BaseModel):
    ts: int
    eq: float


class ReliabilityBin(BaseModel):
    p_hat: float
    p_obs: float
    n: int


class ReliabilityReport(BaseModel):
    bins: List[ReliabilityBin]
    brier: float


class BacktestReport(BaseModel):
    """Summary report object generated at the end of a backtest run."""

    symbols: List[str]
    period: Dict[str, int]
    kpis: Dict[str, float]
    equity_curve: List[EquityDataPoint]
    reliability: ReliabilityReport
    config_fingerprint: str