import pytest
from ultra_signals.core.custom_types import EnsembleDecision, SubSignal
from ultra_signals.transport.telegram_formatter import format_message

@pytest.fixture
def mock_decision():
    """Provides a mock EnsembleDecision with vote details."""
    subsignals = [
        SubSignal(ts=1, symbol="X", tf="1h", strategy_id="strat_A_long", direction="LONG", confidence_calibrated=0.8, reasons={}),
        SubSignal(ts=1, symbol="X", tf="1h", strategy_id="strat_B_short", direction="SHORT", confidence_calibrated=0.7, reasons={})
    ]
    return EnsembleDecision(
        ts=1, symbol="BTC/USDT", tf="1h", decision="LONG", confidence=0.65,
        subsignals=subsignals,
        vote_detail={
            "weighted_sum": 0.55,
            "agree": 1,
            "total": 2,
            "profile": "trend",
        },
        vetoes=[]
    )

@pytest.fixture
def mock_vetoed_decision(mock_decision):
    """Provides a mock decision that has been vetoed."""
    mock_decision.decision = "FLAT"
    mock_decision.vetoes = ["VETO_REASON_A"]
    return mock_decision

def test_format_message_includes_vote_summary(mock_decision):
    """
    Tests that the formatted message contains the ensemble vote summary.
    """
    message = format_message(mock_decision, {})
    assert "Ensemble Confidence" in message
    assert "Vote: `1/2`" in message
    assert "Profile: `trend`" in message
    assert "Wgt Sum: `0.550`" in message

def test_format_message_includes_veto_reason(mock_vetoed_decision):
    """
    Tests that the formatted message includes the top veto reason if present.
    """
    message = format_message(mock_vetoed_decision, {})
    assert "VETOED" in message
    assert "VETO\\_REASON\\_A" in message

def test_format_message_includes_subsignal_breakdown(mock_decision):
    """
    Tests that the message includes a breakdown of contributing sub-signals.
    """
    message = format_message(mock_decision, {})
    assert "Contributing Signals" in message
    assert "strat\\_A\\_long" in message
    assert "strat\\_B\\_short" in message