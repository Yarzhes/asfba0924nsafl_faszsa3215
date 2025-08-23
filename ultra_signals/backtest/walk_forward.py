# ultra_signals/backtest/walk_forward.py

import yaml
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from ultra_signals.core.timeutils import convert_to_timedelta

class WalkForwardAnalysis:
    """
    Orchestrates a walk-forward analysis based on a YAML configuration file.
    """

    def __init__(self, config_path: str):
        """
        Initializes the WalkForwardAnalysis with a configuration file.

        :param config_path: Path to the YAML configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self._validate_config()

    def _validate_config(self):
        """
        Validates the structure and types of the configuration file.
        """
        required_keys = [
            "walk_forward_config",
            "symbols",
            "version_info",
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: {key}")

        wfc = self.config["walk_forward_config"]
        required_wfc_keys = [
            "analysis_start_date",
            "analysis_end_date",
            "window",
            "data_rules",
        ]
        for key in required_wfc_keys:
            if key not in wfc:
                raise ValueError(f"Missing key in walk_forward_config: {key}")
        
        # Further validation can be added here (e.g., for date formats)

    def _generate_windows(self):
        """
        Generates the rolling windows for the walk-forward analysis.
        """
        wfc = self.config["walk_forward_config"]
        start_date = datetime.fromisoformat(wfc["analysis_start_date"])
        end_date = datetime.fromisoformat(wfc["analysis_end_date"])
        
        train_delta = convert_to_timedelta(wfc["window"]["train_period"])
        test_delta = convert_to_timedelta(wfc["window"]["test_period"])
        advance_delta = convert_to_timedelta(wfc["window"]["advance_by"])
        purge_delta = convert_to_timedelta(wfc["data_rules"]["purge_period"])
        embargo_delta = convert_to_timedelta(wfc["data_rules"]["embargo_period"])

        current_date = start_date
        while current_date + train_delta + test_delta <= end_date:
            train_start = current_date
            train_end = current_date + train_delta
            
            test_start = train_end
            test_end = test_start + test_delta

            yield {
                "train_start": train_start,
                "train_end": train_end - purge_delta,
                "test_start": test_start + embargo_delta,
                "test_end": test_end,
            }
            current_date += advance_delta
    
    def run(self):
        """
        Executes the walk-forward analysis.
        """
        print("Starting Walk-Forward Analysis...")
        print(f"Strategy: {self.config['version_info']['strategy_version']}")
        print(f"Symbols: {self.config['symbols']}")

        for i, window in enumerate(self._generate_windows()):
            print(f"\n--- Running Window {i+1} ---")
            print(f"  Train Period: {window['train_start']} to {window['train_end']}")
            print(f"  Test Period:  {window['test_start']} to {window['test_end']}")
            
            # In a real scenario, you would trigger the backtest here
            # using the window dates.
            # a new backtest instance needs to be created, and then the results from it stored
            
        print("\nWalk-Forward Analysis Complete.")

if __name__ == '__main__':
    # This block demonstrates how to run the WalkForwardAnalysis.
    # It uses the example configuration file in the same directory.
    config_file = "ultra_signals/backtest/walk_forward_config.yaml"
    wfa = WalkForwardAnalysis(config_path=config_file)
    wfa.run()