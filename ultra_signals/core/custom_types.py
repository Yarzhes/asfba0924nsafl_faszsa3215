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
from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel
from datetime import datetime

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
    CHOP = "chop"


class RegimeProfile(str, Enum):
    TREND = "trend"
    MEAN_REVERT = "mean_revert"
    CHOP = "chop"

class VolState(str, Enum):
    CRUSH = "crush"      # very low realized / implied volatility
    NORMAL = "normal"
    EXPANSION = "expansion"  # volatility expanding / elevated

class NewsState(str, Enum):
    QUIET = "quiet"
    NEWS = "news"

class LiquidityState(str, Enum):
    OK = "ok"
    THIN = "thin"


class RegimeFeatures(BaseModel):
    """Output of the market regime classification engine.

    Extended in Sprint 43 with probabilistic meta-regime fields. All new
    attributes are optional to preserve backward compatibility with any
    persisted state or historical records serialized prior to the upgrade.
    """

    adx: Optional[float] = None
    vol_bucket: VolatilityBucket = VolatilityBucket.MEDIUM
    mode: RegimeMode = RegimeMode.TREND
    profile: RegimeProfile = RegimeProfile.TREND
    atr_percentile: Optional[float] = None  # keep raw value for downstream sizing
    vol_state: VolState = VolState.NORMAL
    news_state: NewsState = NewsState.QUIET
    # Allow arbitrary gate values (bools, floats like size multipliers, counters)
    gates: Dict[str, object] = {}
    # Core legacy fields
    liquidity: LiquidityState = LiquidityState.OK
    confidence: float = 0.0
    since_ts: Optional[int] = None
    last_flip_ts: Optional[int] = None
    # Sprint 43 advanced meta-regime probabilistic outputs
    regime_label: Optional[str] = None                 # canonical label (trend_up, chop_lowvol, panic_deleverage ...)
    regime_probs: Optional[Dict[str, float]] = None     # soft distribution over configured regimes
    transition_hazard: Optional[float] = None           # flip risk proxy (0..1)
    exp_vol_h: Optional[float] = None                   # expected realized vol horizon h
    dir_bias: Optional[float] = None                    # directional drift (-1 .. +1)
    macro_risk_context: Optional[str] = None            # macro risk_on|risk_off|liquidity_squeeze|neutral
    sent_extreme_flag: Optional[int] = None             # 1 if sentiment extreme gating engaged
    whale_pressure: Optional[float] = None              # aggregated smart-money pressure score


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


class FlowMetricsFeatures(BaseModel):
    """Sprint 11 advanced order-flow & microstructure metrics.

    All fields are optional so downstream code remains resilient when a metric
    cannot be computed for the current bar (e.g. missing trades / depth).
    """
    # Core order-flow / aggressor tracking
    cvd: Optional[float] = None           # cumulative volume delta (rolling)
    cvd_chg: Optional[float] = None       # delta change this bar (buy_vol - sell_vol)
    buy_volume: Optional[float] = None
    sell_volume: Optional[float] = None

    # Order flow imbalance (top of book or synthetic)
    ofi: Optional[float] = None           # (bid_size - ask_size)/(bid_size+ask_size)

    # Open interest dynamics
    oi: Optional[float] = None            # latest OI level if available
    oi_prev: Optional[float] = None
    oi_rate: Optional[float] = None       # (oi - oi_prev)/oi_prev

    # Liquidation pulse
    liq_events: Optional[int] = None      # number of liquidations inside window
    liq_notional_sum: Optional[float] = None
    liq_cluster: Optional[int] = None     # 1 if cluster detected else 0
    liq_cluster_side: Optional[str] = None  # 'BUY' or 'SELL' dominant side

    # Depth imbalance (single or cross venue placeholder)
    depth_imbalance: Optional[float] = None  # (bid_qty - ask_qty)/(bid_qty+ask_qty)

    # Cross-exchange spread (bps) & deviation flag
    spread_bps: Optional[float] = None
    spread_dev_flag: Optional[int] = None

    # Volume anomaly (Z-score) reused / enhanced
    volume_z: Optional[float] = None
    volume_anom: Optional[int] = None     # 1 if |z| >= configured threshold

    # Timestamp (epoch ms) when last liquidation cluster detected
    last_liq_cluster_ts: Optional[int] = None

    # Internal debug dictionary (kept light)
    meta: Optional[Dict[str, float]] = None


class AlphaV2Features(BaseModel):
    """Sprint 11 Feature Pack v2 composite features."""
    # Multi-timeframe / structure
    hh_break_20: Optional[int] = None  # 1 if broke 20-bar high this bar
    ll_break_20: Optional[int] = None  # 1 if broke 20-bar low this bar
    range_pct_20: Optional[float] = None

    # Anchored VWAP (session)
    sess_vwap: Optional[float] = None
    sess_vwap_upper_1: Optional[float] = None
    sess_vwap_lower_1: Optional[float] = None
    sess_vwap_dev: Optional[float] = None

    # Momentum structure
    adx_slope_5: Optional[float] = None
    rsi: Optional[float] = None  # duplicated convenience (latest RSI)
    bull_div: Optional[int] = None
    bear_div: Optional[int] = None

    # Volatility / squeeze
    bb_kc_ratio: Optional[float] = None
    squeeze_flag: Optional[int] = None

    # Volume / liquidity
    volume_burst: Optional[int] = None

    # Time / calendar
    hour: Optional[int] = None
    session: Optional[str] = None
    week_of_month: Optional[int] = None

    # Attribution snapshot (normalized contributions) (optional keys only)
    attribution: Optional[Dict[str, float]] = None


class WhaleFeatures(BaseModel):
    """Sprint 41 Whale / Smart-Money aggregate feature pack.

    All fields optional & defensive so downstream code remains resilient if a
    particular collector/source is disabled or temporarily unavailable.

    Naming convention: group_metric_window / flags suffixed with _flag.
    Windows (s, m, l) map to short (<=60m), medium (<=6h), long (<=24h) by config.
    """
    # Net exchange flows (USD)
    whale_net_inflow_usd_s: Optional[float] = None
    whale_net_inflow_usd_m: Optional[float] = None
    whale_net_inflow_usd_l: Optional[float] = None
    whale_inflow_z_s: Optional[float] = None
    whale_inflow_z_m: Optional[float] = None
    exch_deposit_burst_flag: Optional[int] = None
    exch_withdrawal_burst_flag: Optional[int] = None

    # Large block / sweep activity (CEX trades aggregation)
    block_trade_count_5m: Optional[int] = None
    block_trade_notional_5m: Optional[float] = None
    block_trade_notional_p99_z: Optional[float] = None
    sweep_sell_flag: Optional[int] = None
    sweep_buy_flag: Optional[int] = None
    iceberg_replenish_score: Optional[float] = None

    # Options anomalies (Deribit public API)
    opt_call_put_volratio_z: Optional[float] = None
    opt_oi_delta_1h_z: Optional[float] = None
    opt_skew_shift_z: Optional[float] = None
    opt_block_trade_flag: Optional[int] = None

    # Smart-money curated wallets pressure metrics
    smart_money_buy_pressure_s: Optional[float] = None
    smart_money_sell_pressure_s: Optional[float] = None
    smart_money_hit_rate_30d: Optional[float] = None

    # Aggregated confidence / meta
    composite_pressure_score: Optional[float] = None
    last_update_ts: Optional[int] = None  # epoch ms when whale snapshot last refreshed
    sources_active: Optional[int] = None  # how many collectors reported in window
    meta: Optional[Dict[str, float]] = None


class MacroFeatures(BaseModel):
    """Sprint 42 Cross-Asset / Macro Fusion feature pack.

    Rolling correlations, macro regime probabilities, synthetic BTC-VIX and
    event flags. All optional for resilience when a source temporarily fails.
    """
    # Rolling correlations (labelled windows)
    btc_spy_corr_30m: Optional[float] = None
    btc_spy_corr_4h: Optional[float] = None
    btc_spy_corr_1d: Optional[float] = None
    btc_spy_corr_1w: Optional[float] = None
    btc_dxy_corr_4h: Optional[float] = None
    eth_gold_corr_1w: Optional[float] = None
    # Correlation trend / z
    btc_spy_corr_trend_1d: Optional[float] = None
    btc_spy_corr_z_1d: Optional[float] = None

    # Volatility proxies
    btc_vix_proxy: Optional[float] = None
    btc_vix_proxy_z: Optional[float] = None
    realized_vol_24h: Optional[float] = None

    # Macro regime classifier outputs
    macro_risk_regime: Optional[str] = None  # risk_on|risk_off|liquidity_squeeze|neutral
    risk_on_prob: Optional[float] = None
    risk_off_prob: Optional[float] = None
    liquidity_squeeze_prob: Optional[float] = None
    macro_extreme_flag: Optional[int] = None

    # Event / flag detectors
    carry_unwind_flag: Optional[int] = None
    dxy_surge_flag: Optional[int] = None
    oil_price_shock_z: Optional[float] = None
    gold_safehaven_flow_z: Optional[float] = None

    # Composite health / diagnostics
    data_fresh_min: Optional[float] = None
    sources_active: Optional[int] = None
    last_refresh_ts: Optional[int] = None
    meta: Optional[Dict[str, float]] = None


class BehaviorFeatures(BaseModel):
        """Sprint 45 Behavioral Finance veto & sizing feature pack.

        Lightweight container for per-bar behavioral context derived from existing
        market + sentiment + whale inputs. All numeric fields optional to maintain
        resilience while upstream collectors incrementally mature.

        Key groups
            * FOMO / Euphoria / Capitulation composite scores (z-normalised)
            * Session & calendar bias flags / quality metrics
            * Final gate outputs (behavior_veto, behavior_size_mult)
        """

        # Composite scores
        beh_fomo_score_z: Optional[float] = None
        beh_euphoria_score_z: Optional[float] = None
        beh_capitulation_score_z: Optional[float] = None

        # Raw component proxies (subset; full list evolves)
        ret_z: Optional[float] = None
        volume_z: Optional[float] = None
        oi_rate_z: Optional[float] = None
        funding_z: Optional[float] = None
        liq_cluster: Optional[int] = None
        liq_notional_z: Optional[float] = None
        wick_body_ratio: Optional[float] = None
        whale_net_inflow_z: Optional[float] = None
        whale_withdrawal_flag: Optional[int] = None
        sent_z_s: Optional[float] = None

        # Session / calendar
        session: Optional[str] = None              # asia|eu|us
        hour_bin: Optional[int] = None             # 0..23 or custom bin id
        dow: Optional[int] = None                  # 0=Mon .. 6=Sun
        is_weekend: Optional[int] = None
        is_holiday: Optional[int] = None
        session_expectancy: Optional[float] = None
        tod_expectancy: Optional[float] = None
        tod_quality: Optional[float] = None        # 0..1 percentile rank among hours
        weekend_penalty: Optional[float] = None
        holiday_penalty: Optional[float] = None

        # Final policy outputs
        behavior_veto: Optional[int] = None        # 1 if hard veto triggered
        behavior_action: Optional[str] = None      # ENTER|DAMPEN|VETO
        behavior_size_mult: Optional[float] = None # applied multiplicatively
        behavior_reason: Optional[str] = None
        flags: Optional[List[str]] = None          # active textual flags
        behavior_veto_prob: Optional[float] = None # calibrated probability of behavioral failure if trade taken
        divergence_flag: Optional[int] = None      # price up while cvd flat/down or vice versa
        oi_purge_flag: Optional[int] = None        # sharp OI drop event

        # Diagnostics
        hysteresis_state: Optional[str] = None
        meta: Optional[Dict[str, float]] = None


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
    alpha_v2: Optional[AlphaV2Features] = None
    # Sprint 11 addition: expose flow metrics inside FV for logging/telemetry
    flow_metrics: Optional[FlowMetricsFeatures] = None
    # Sprint 42 macro fusion pack (injected later when available)
    macro: Optional[MacroFeatures] = None
    # Sprint 41 whale / smart-money feature pack (added for Sprint 43 fusion)
    whales: Optional[WhaleFeatures] = None
    # Sprint 44 pattern engine output (list of active/just-updated patterns this bar)
    patterns: Optional[List["PatternInstance"]] = None  # defined later in file; use forward ref
    # Sprint 45 behavioral finance veto feature pack (attached late in pipeline)
    behavior: Optional[BehaviorFeatures] = None
    # Sprint 46 economic / event risk feature pack (attached late)
    econ: Optional[EconFeatures] = None

class PatternDirection(str, Enum):
    LONG = "long"
    SHORT = "short"

class PatternStage(str, Enum):
    FORMING = "forming"
    CONFIRMED = "confirmed"
    FAILED = "failed"

class PatternType(str, Enum):  # classical + harmonics + structural buckets
    # Classical
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    CUP_HANDLE = "cup_handle"
    ASC_TRIANGLE = "ascending_triangle"
    DESC_TRIANGLE = "descending_triangle"
    SYM_TRIANGLE = "sym_triangle"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    # Harmonics (subset for early scaffolding)
    GARTLEY = "gartley"
    BAT = "bat"
    BUTTERFLY = "butterfly"
    CRAB = "crab"
    CYPHER = "cypher"
    SHARK = "shark"
    # Volume profile / structural events
    VPOC_SHIFT = "vpoc_shift"
    DOUBLE_DISTRIBUTION = "double_distribution"
    SINGLE_PRINT = "single_print"
    VALUE_ROTATION = "value_rotation"
    # Support / Resistance derived setup (e.g., multi-touch level break)
    SR_LEVEL_BREAK = "sr_level_break"

class PatternInstance(BaseModel):
    """Represents a detected pattern candidate life-cycle snapshot.

    This object is persisted inside FeatureVector.patterns and also emitted to
    downstream consumers (scoring / execution policy / reporting). All numeric
    values optional so partially-evaluated formations can be surfaced early with
    explanation & progressive refinement.
    """
    ts: int  # epoch ms of last evaluation
    symbol: Symbol
    timeframe: Timeframe
    pat_type: PatternType
    direction: PatternDirection
    stage: PatternStage
    quality: Optional[float] = None        # raw geometry/confluence composite 0..1
    confidence: Optional[float] = None     # calibrated probability of expected resolution
    age_bars: int = 0                      # bars since first detection
    freshness_bars: int = 0                # bars since last structural update
    mtf_agree_ct: Optional[int] = None     # number of timeframes with similar signal
    confluence: List[str] = []             # tags: lvn, vpoc_shift, divergence, regime_trend_up, etc.
    reason_codes: List[str] = []           # internal rule hits for transparency
    # Key price levels
    neckline_px: Optional[float] = None
    breakout_px: Optional[float] = None
    target1_px: Optional[float] = None
    target2_px: Optional[float] = None
    struct_stop_px: Optional[float] = None
    measured_move_px: Optional[float] = None
    # Harmonic specifics
    fib_fit_err: Optional[float] = None
    harm_prz_score: Optional[float] = None
    # Geometry / regression stats
    channel_r2: Optional[float] = None
    triangle_slope: Optional[float] = None
    # Volume profile deltas
    vpoc_distance_pct: Optional[float] = None
    lvn_confluence_flag: Optional[int] = None
    va_rotation_flag: Optional[int] = None
    # Fractal / regime context
    fractal_dim: Optional[float] = None
    hurst_h: Optional[float] = None
    sr_level_strength: Optional[float] = None
    sr_reaction_score: Optional[float] = None
    # Risk / sizing helpers
    target_confidence: Optional[float] = None
    stop_confidence: Optional[float] = None
    # Meta
    dedup_group: Optional[str] = None      # id for conflict resolver grouping
    hash_id: Optional[str] = None          # stable hash for tracking across bars

    class Config:
        arbitrary_types_allowed = True



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


class QualityDecision(BaseModel):
    """Sprint 18 quality gate evaluation output.

    Contains binning, composite quality score and veto / soft gate outcomes that
    upstream engine layers (sizing, execution planner, transport) can use to
    adjust position size or abstain prior to order placement.
    """
    bin: str
    qscore: float
    blocked: bool
    veto_reasons: List[str] = []        # hard veto short codes
    soft_flags: List[str] = []          # soft gate codes (non-blocking)
    requirements: Dict[str, bool] = {}  # any extra confirmation requirements
    size_multiplier: float = 1.0
    notes: str = ""


class Position(BaseModel):
    """Represents an open position."""
    side: str
    size: float
    entry: float
    risk: float
    cluster: str
    entry_price: float = 0.0
    bars_held: int = 0


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


# ---------------------------------------------------------------------------
# Sprint 46: Economic Calendar / Event Integration Types
# ---------------------------------------------------------------------------

class EconEventClass(str, Enum):
    CPI = "cpi"
    FOMC = "fomc"
    NFP = "nfp"
    GDP = "gdp"
    TREASURY_AUCTION = "treasury_auction"
    SEC_ETF = "sec_etf"
    REGULATORY = "regulatory"
    EXCHANGE_MAINT = "exchange_maint"
    HOLIDAY = "holiday"
    EARNINGS_COIN = "earnings_coin"
    EARNINGS_MSTR = "earnings_mstr"
    EARNINGS_OTHER = "earnings_other"
    OTHER = "other"


class EconSeverity(str, Enum):
    HIGH = "high"
    MED = "med"
    LOW = "low"


class EconEventStatus(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    DONE = "done"
    CANCELLED = "cancelled"


class EconomicEvent(BaseModel):
    """Canonical normalized economic / macro / exchange / earnings event.

    All timestamps are epoch milliseconds UTC. Risk window handling is performed
    downstream (risk engine) but pre/post template minutes can be overridden per
    event (e.g. an exchange maintenance specifying full window explicitly).
    """

    id: str                               # stable hash (source + raw id + start time)
    source: str                           # collector/source identifier
    raw_id: Optional[str] = None          # raw source ID if available
    cls: EconEventClass                   # event class bucket
    title: str
    region: Optional[str] = None          # US|EU|DE|GLOBAL|EXCHANGE
    severity: EconSeverity = EconSeverity.MED
    symbols: Optional[List[str]] = None   # affected symbols (e.g., ["BTCUSDT"] or None=all)
    ts_start: int                         # scheduled start (epoch ms)
    ts_end: Optional[int] = None          # expected end (epoch ms) if known
    status: EconEventStatus = EconEventStatus.SCHEDULED
    risk_pre_min: Optional[int] = None    # override template pre window (minutes)
    risk_post_min: Optional[int] = None   # override template post window (minutes)
    updated_ts: Optional[int] = None      # last refresh (epoch ms)
    notes: Optional[str] = None
    expected: Optional[str] = None        # consensus / expected value textual
    previous: Optional[str] = None
    actual: Optional[str] = None
    surprise_score: Optional[float] = None  # numeric surprise when derivable
    url: Optional[str] = None             # source reference link
    meta: Optional[Dict[str, str]] = None

    def countdown_min(self, now_ms: int) -> Optional[float]:
        if self.status != EconEventStatus.SCHEDULED:
            return 0.0
        return (self.ts_start - now_ms) / 60000.0

    def in_risk_window(self, now_ms: int, pre_min: int, post_min: int) -> bool:
        start = self.ts_start - pre_min * 60000
        end = (self.ts_end or self.ts_start) + post_min * 60000
        return start <= now_ms <= end


class EconWindowSide(str, Enum):
    PRE = "pre"
    LIVE = "live"
    POST = "post"
    OUT = "out"


class EconFeatures(BaseModel):
    """Per-bar economic / event risk feature pack injected into FeatureVector.

    All fields optional/defensive to ensure resilience if the event engine is
    temporarily unavailable. Naming aligned with Sprint 46 spec.
    """
    econ_risk_active: Optional[int] = None            # 1 if inside any active risk window
    econ_risk_class: Optional[str] = None             # dominant highest severity event class
    econ_countdown_min: Optional[float] = None        # minutes to next high/med event (>=0)
    econ_window_side: Optional[str] = None            # pre|live|post|out for dominant event
    econ_severity: Optional[str] = None               # high|med|low for dominant event
    econ_surprise_score: Optional[float] = None       # last processed surprise score (if within window)
    exchange_status_flag: Optional[int] = None        # 1 if any exchange maintenance active
    holiday_penalty: Optional[float] = None           # size penalty multiplier (<=1)
    earnings_flag_coin: Optional[int] = None          # 1 if COIN earnings within active window
    earnings_flag_mstr: Optional[int] = None          # 1 if MSTR earnings within active window
    macro_blackout_prob: Optional[float] = None       # probability based composite (0..1)
    allowed_size_mult_econ: Optional[float] = None    # final size multiplier due to econ policy
    flags: Optional[List[str]] = None                 # textual active flags (e.g., ["CPI_PRE", "FOMC_LIVE"])
    meta: Optional[Dict[str, float]] = None           # lightweight diagnostics (cache age, counts)

# Ensure forward refs resolved (FeatureVector defined later)
