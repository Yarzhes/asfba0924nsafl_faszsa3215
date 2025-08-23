import pytest
from ultra_signals.core.custom_types import EnsembleDecision, PortfolioState, Position, RiskEvent
from ultra_signals.risk.portfolio import evaluate_portfolio

@pytest.fixture
def mock_decision():
    """Provides a default mock EnsembleDecision for a LONG trade."""
    return EnsembleDecision(
        ts=1, symbol="BTC/USDT", tf="1h", decision="LONG", confidence=0.8,
        subsignals=[], vote_detail={}, vetoes=[]
    )

@pytest.fixture
def mock_portfolio_state():
    """Provides a mock PortfolioState."""
    return PortfolioState(
        ts=1, equity=10000.0, positions={},
        exposure={
            "net": {"long": 0.0, "short": 0.0},
            "cluster": {},
            "symbol": {},
        }
    )

@pytest.fixture
def mock_settings():
    """Provides mock portfolio settings."""
    return {
        "portfolio": {
            "max_risk_per_symbol": 0.01,
            "max_positions_per_symbol": 1,
            "max_positions_total": 5,
            "max_net_long_risk": 0.03,
            "max_net_short_risk": 0.03,
        }
    }

def test_evaluate_portfolio_allow_trade(mock_decision, mock_portfolio_state, mock_settings):
    """
    Tests that a trade is allowed when no risk limits are breached.
    """
    allowed, size_scale, events = evaluate_portfolio(mock_decision, mock_portfolio_state, mock_settings)
    assert allowed
    assert size_scale == 1.0
    assert not events

def test_evaluate_portfolio_veto_max_positions_total(mock_decision, mock_portfolio_state, mock_settings):
    """
    Tests that a trade is vetoed if the max total positions limit is reached.
    """
    mock_portfolio_state.positions = {f"POS_{i}": Position(side="LONG", size=1, entry=1, risk=0.01, cluster="c1") for i in range(5)}
    allowed, _, events = evaluate_portfolio(mock_decision, mock_portfolio_state, mock_settings)
    assert not allowed
    assert any(e.reason == "MAX_POSITIONS_TOTAL" for e in events)

def test_evaluate_portfolio_veto_max_positions_per_symbol(mock_decision, mock_portfolio_state, mock_settings):
    """
    Tests that a trade is vetoed if the max positions for that symbol is reached.
    """
    mock_portfolio_state.positions["BTC/USDT"] = Position(side="LONG", size=1, entry=1, risk=0.01, cluster="c1")
    allowed, _, events = evaluate_portfolio(mock_decision, mock_portfolio_state, mock_settings)
    assert not allowed
    assert any(e.reason == "MAX_POSITIONS_PER_SYMBOL" for e in events)

def test_evaluate_portfolio_veto_max_net_long_risk(mock_decision, mock_portfolio_state, mock_settings):
    """
    Tests that a trade is vetoed if it would breach the max net long risk.
    """
    mock_portfolio_state.exposure["net"]["long"] = 0.025 # Already using 2.5% risk
    # The new trade is assumed to have 1% risk, totaling 3.5%
    allowed, _, events = evaluate_portfolio(mock_decision, mock_portfolio_state, mock_settings)
    assert not allowed
    assert any(e.reason == "MAX_NET_LONG_RISK" for e in events)

def test_evaluate_portfolio_no_veto_if_flat(mock_decision, mock_portfolio_state, mock_settings):
    """
    Tests that FLAT decisions are always allowed, regardless of risk limits.
    """
    mock_decision.decision = "FLAT"
    mock_portfolio_state.exposure["net"]["long"] = 0.05 # Clearly over the limit
    allowed, _, _ = evaluate_portfolio(mock_decision, mock_portfolio_state, mock_settings)
    assert allowed