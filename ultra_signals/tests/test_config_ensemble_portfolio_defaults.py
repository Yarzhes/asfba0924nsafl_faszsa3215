import pytest
from pydantic import ValidationError

from ultra_signals.core.config import (
    EnsembleSettings,
    CorrelationSettings,
    PortfolioSettings,
    BrakesSettings,
    SizingSettings,
)
from ultra_signals.core.custom_types import (
    SubSignal,
    EnsembleDecision,
    PortfolioState,
    RiskEvent,
)


def test_ensemble_settings_defaults():
    """Verify EnsembleSettings default values and constraints."""
    settings = EnsembleSettings()
    assert 0.5 <= settings.majority_threshold <= 1.0
    assert settings.veto_trend_flip is True
    assert settings.veto_band_pierce is True

    with pytest.raises(ValidationError):
        EnsembleSettings(majority_threshold=4)


def test_correlation_settings_defaults():
    """Verify CorrelationSettings default values and constraints."""
    settings = CorrelationSettings()
    assert settings.enabled is True
    assert settings.lookback_periods > 1
    assert 0.0 <= settings.cluster_threshold <= 1.0
    assert 0.0 <= settings.hysteresis <= 1.0
    assert settings.refresh_interval_bars > 1

    with pytest.raises(ValidationError):
        CorrelationSettings(lookback_periods=0)
    with pytest.raises(ValidationError):
        CorrelationSettings(cluster_threshold=2)


def test_portfolio_settings_defaults():
    """Verify PortfolioSettings default values and constraints."""
    settings = PortfolioSettings()
    assert settings.max_exposure_per_symbol > 0
    assert settings.max_exposure_per_cluster > 0
    assert settings.max_net_exposure > 0
    assert 0 < settings.max_margin_pct <= 100
    assert settings.max_total_positions > 0

    with pytest.raises(ValidationError):
        PortfolioSettings(max_margin_pct=101)
    with pytest.raises(ValidationError):
        PortfolioSettings(max_total_positions=0)


def test_brakes_settings_defaults():
    """Verify BrakesSettings default values and constraints."""
    settings = BrakesSettings()
    assert settings.min_spacing_sec_cluster >= 0
    assert 0 < settings.daily_loss_soft_limit_pct <= 100
    assert 0 < settings.daily_loss_hard_limit_pct <= 100
    assert settings.streak_cooldown_trades >= 0
    assert settings.streak_cooldown_hours >= 0

    with pytest.raises(ValidationError):
        BrakesSettings(daily_loss_hard_limit_pct=0)


def test_sizing_settings_defaults():
    """Verify SizingSettings default values."""
    settings = SizingSettings()
    assert 0 < settings.vol_risk_scale_pct <= 100
    
    with pytest.raises(ValidationError):
        SizingSettings(vol_risk_scale_pct=0)


def test_subsignal_model():
    """Test SubSignal data model."""
    sub = SubSignal(
        ts=1, symbol="BTCUSDT", tf="5m", strategy_id="test",
        direction="LONG", confidence_calibrated=0.8, reasons={"a": 1}
    )
    assert sub.confidence_calibrated == 0.8
    assert sub.direction == "LONG"


def test_ensemble_decision_model():
    """Test EnsembleDecision data model."""
    sub = SubSignal(
        ts=1, symbol="BTCUSDT", tf="5m", strategy_id="test",
        direction="LONG", confidence_calibrated=0.8, reasons={"a": 1}
    )
    decision = EnsembleDecision(
        ts=1, symbol="BTCUSDT", tf="5m", decision="LONG",
        confidence=0.7, subsignals=[sub], vote_detail={}, vetoes=[]
    )
    assert decision.decision == "LONG"
    assert len(decision.subsignals) == 1


def test_portfolio_state_model():
    """Test PortfolioState data model."""
    state = PortfolioState(ts=1, equity=100.0, positions={}, exposure={})
    assert state.equity == 100.0


def test_risk_event_model():
    """Test RiskEvent data model."""
    event = RiskEvent(
        ts=1, symbol="BTCUSDT", reason="test",
        action="VETO", detail={"msg": "test"}
    )
    assert event.action == "VETO"