import pytest
from unittest.mock import patch, MagicMock
from ultra_signals.apps import backtest_cli
from ultra_signals.backtest.event_runner import MockSignalEngine

@pytest.fixture
def mock_load_settings():
    """Fixture to mock the settings loader."""
    with patch('ultra_signals.apps.backtest_cli.load_settings') as mock_loader:
        # Create a mock settings object that can be configured per test
        mock_settings = MagicMock()
        mock_settings.features.warmup_periods = 10
        mock_settings.model_dump.return_value = {"features": {"warmup_periods": 10}}
        mock_settings.backtest.data.model_dump.return_value = {}
        mock_settings.backtest.model_dump.return_value = {}
        mock_settings.reports.model_dump.return_value = {}
        mock_settings.logging.level = "INFO"
        mock_settings.backtest.data.model_dump.return_value = {}
        mock_settings.backtest.model_dump.return_value = {}
        mock_settings.reports.model_dump.return_value = {}
        mock_loader.return_value = mock_settings
        yield mock_loader

class MockArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@patch('ultra_signals.apps.backtest_cli.DataAdapter')
@patch('ultra_signals.apps.backtest_cli.EventRunner')
def test_handle_run_invokes_runner(mock_runner, mock_adapter, mock_load_settings):
    """Verify that handle_run correctly initializes and invokes the EventRunner."""
    args = MockArgs(config='settings.yaml', start=None, end=None)
    
    # Configure the mock EventRunner to return some trades
    mock_runner.return_value.run.return_value = ([{"pnl": 10}], [])
    # The code checks if trades is not empty, so we need to return a list with a dictionary
    # that has a 'pnl' key.
    mock_runner.return_value.trades = [{"pnl": 10}]
    
    backtest_cli.handle_run(args, mock_load_settings)
    
    mock_adapter.assert_called_once()
    mock_runner.assert_called_once()
    mock_runner.return_value.run.assert_called_once()
    
@patch('ultra_signals.apps.backtest_cli.WalkForwardAnalysis')
def test_handle_wf_invokes_analyzer(mock_wfa, mock_load_settings):
    """Verify that handle_wf correctly initializes and invokes the WalkForwardAnalysis."""
    args = MockArgs(config='settings.yaml')
    
    # Simulate the WFA returning some results
    mock_wfa.return_value.run.return_value = (MagicMock(), MagicMock())
    
    backtest_cli.handle_wf(args, mock_load_settings)
    
    mock_wfa.assert_called_once()
    mock_wfa.return_value.run.assert_called_once()

@patch('ultra_signals.calibration.calibrate.fit_calibration_model')
@patch('ultra_signals.calibration.calibrate.save_model')
@patch('pandas.read_csv')
@patch('ultra_signals.apps.backtest_cli.calculate_brier_score')
def test_handle_cal_invokes_calibration(mock_brier, mock_read_csv, mock_save, mock_fit, mock_load_settings):
    """Verify that handle_cal invokes the calibration logic."""
    args = MockArgs(config='settings.yaml', method='isotonic')
    
    # Mock the predictions file
    import pandas as pd
    mock_df = pd.DataFrame({
        'raw_score': pd.Series([0.1, 0.2, 0.8, 0.9], name='raw_score'),
        'outcome': pd.Series([0, 0, 1, 1], name='outcome')
    })
    mock_read_csv.return_value = mock_df
    mock_brier.return_value = 0.25
    
    backtest_cli.handle_cal(args, mock_load_settings)
    
    mock_read_csv.assert_called_once()
    mock_fit.assert_called_once()
    mock_save.assert_called_once()