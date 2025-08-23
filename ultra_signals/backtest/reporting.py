import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from ultra_signals.backtest.metrics import generate_equity_curve

class ReportGenerator:
    """Generates and saves backtest reports and artifacts."""

    def __init__(self, config: Dict[str, Any]):
        self.output_dir = Path(config.get("output_dir", "reports/backtest_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(
        self,
        kpis: Dict[str, Any],
        equity_data: List[Dict], # Changed to list of dicts
        trades: pd.DataFrame
    ):
        """
        Generates a full backtest report, including a summary text file,
        an equity curve plot, and a CSV of all trades.
        """
        # 1. Save KPIs to a text file
        self._save_kpis(kpis)
        
        # 2. Save equity curve plot
        equity_curve = generate_equity_curve(equity_data) # Use the new function
        self._save_equity_curve_plot(equity_curve)
        
        # 3. Save trades to CSV
        self._save_trades_csv(trades)
        
        print(f"Report saved to {self.output_dir}")

    def _save_kpis(self, kpis: Dict[str, Any]):
        """Saves KPIs to a summary.txt file."""
        path = self.output_dir / "summary.txt"
        with open(path, "w") as f:
            f.write("Backtest Performance Summary\n")
            f.write("="*30 + "\n")
            for key, value in kpis.items():
                # Format floats to 2 decimal places, others as is
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")

    def _save_equity_curve_plot(self, equity_curve: pd.Series):
        """Generates and saves a plot of the equity curve."""
        if equity_curve.empty:
            return
            
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        equity_curve.plot(ax=ax, title="Equity Curve", color="blue", linewidth=2)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True)
        plt.tight_layout()
        
        plot_path = self.output_dir / "equity_curve.png"
        plt.savefig(plot_path)
        plt.close(fig)

    def _save_trades_csv(self, trades: pd.DataFrame):
        """Saves the trade log to a CSV file."""
        if trades.empty:
            return
        
        path = self.output_dir / "trades.csv"
        trades.to_csv(path, index=False)