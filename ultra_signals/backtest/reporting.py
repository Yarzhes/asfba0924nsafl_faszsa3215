import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter  # NEW: for summarize_veto_reasons

from ultra_signals.backtest.metrics import generate_equity_curve

class ReportGenerator:
    """Generates and saves backtest reports and artifacts."""

    def __init__(self, config: Dict[str, Any]):
        self.output_dir = Path(config.get("output_dir", "reports/backtest_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        kpis: Dict[str, Any],
        equity_data: List[Dict],          # list of dicts (timestamps & equity)
        trades: pd.DataFrame,
        risk_events: Optional[List[Any]] = None,  # NEW: optional RiskEvents
    ):
        """
        Generates a full backtest report, including a summary text file,
        an equity curve plot, a CSV of all trades, and (optionally) a
        CSV + summary for RiskEvents if provided.
        """
        # 1) Save KPIs
        self._save_kpis(kpis)

        # 2) Save equity curve plot
        equity_curve = generate_equity_curve(equity_data)
        self._save_equity_curve_plot(equity_curve)

        # 3) Save trades CSV
        self._save_trades_csv(trades)

        # 4) Optionally save RiskEvents + append top reasons to summary
        if risk_events:
            df = self._risk_events_to_df(risk_events)
            if not df.empty:
                self._save_risk_events_csv(df)
                self._append_top_vetoes_to_summary(df)

        print(f"Report saved to {self.output_dir}")

    def _save_kpis(self, kpis: Dict[str, Any]):
        """Saves KPIs to a summary.txt file."""
        path = self.output_dir / "summary.txt"
        with open(path, "w") as f:
            f.write("Backtest Performance Summary\n")
            f.write("=" * 30 + "\n")
            for key, value in kpis.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")

    def _save_equity_curve_plot(self, equity_curve: pd.Series):
        """Generates and saves a plot of the equity curve."""
        if equity_curve.empty:
            return
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(12, 8))
        equity_curve.plot(ax=ax, title="Equity Curve", linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True)
        plt.tight_layout()
        plot_path = self.output_dir / "equity_curve.png"
        plt.savefig(plot_path)
        plt.close(fig)

    def _save_trades_csv(self, trades: pd.DataFrame):
        """Saves the trade log to a CSV file."""
        if trades is None or trades.empty:
            return
        path = self.output_dir / "trades.csv"
        trades.to_csv(path, index=False)

    # ---------------- NEW: RiskEvents helpers ----------------

    def _risk_events_to_df(self, risk_events: List[Any]) -> pd.DataFrame:
        """
        Convert a mixed list of RiskEvent-like objects/dicts into a DataFrame.
        Accepts pydantic models (.model_dump), dataclasses (.dict), or dicts.
        """
        rows = []
        for ev in risk_events:
            if ev is None:
                continue
            try:
                if hasattr(ev, "model_dump"):
                    d = ev.model_dump()
                elif hasattr(ev, "dict"):
                    d = ev.dict()
                elif isinstance(ev, dict):
                    d = ev
                else:
                    d = {
                        "ts": getattr(ev, "ts", None),
                        "symbol": getattr(ev, "symbol", None),
                        "reason": getattr(ev, "reason", None),
                        "action": getattr(ev, "action", None),
                        "detail": getattr(ev, "detail", None),
                    }
                rows.append(d)
            except Exception:
                # Last-resort fallback to string
                rows.append({"raw": str(ev)})
        df = pd.DataFrame(rows)

        # Best-effort convert epoch seconds to timestamp
        if "ts" in df.columns and "timestamp" not in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
            except Exception:
                pass
        return df

    def _save_risk_events_csv(self, df: pd.DataFrame):
        path = self.output_dir / "risk_events.csv"
        df.to_csv(path, index=False)

    def _append_top_vetoes_to_summary(self, df: pd.DataFrame, top_n: int = 10):
        """Append a small section with the most common risk/veto reasons."""
        if df is None or df.empty or "reason" not in df.columns:
            return
        counts = df["reason"].value_counts().head(top_n)

        total = int(len(df))
        path = self.output_dir / "summary.txt"
        with open(path, "a") as f:
            f.write("\n\nRisk Event Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Total risk events: {total}\n")
            for reason, count in counts.items():
                pct = (count / total) * 100.0 if total else 0.0
                f.write(f"- {reason}: {count} ({pct:.1f}%)\n")


# =======================
# NEW: Standalone helpers
# =======================

def _events_to_df(rows: Any) -> pd.DataFrame:
    """
    Internal: convert an iterable of RiskEvents (mixed types) OR a DataFrame
    into a DataFrame with at least: ts, symbol, reason, action, detail, (optional) fold.
    """
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        norm = []
        for ev in (rows or []):
            if ev is None:
                continue
            try:
                if hasattr(ev, "model_dump"):
                    d = ev.model_dump()
                elif hasattr(ev, "dict"):
                    d = ev.dict()
                elif isinstance(ev, dict):
                    d = ev
                else:
                    d = {
                        "ts": getattr(ev, "ts", None),
                        "symbol": getattr(ev, "symbol", None),
                        "reason": getattr(ev, "reason", None),
                        "action": getattr(ev, "action", None),
                        "detail": getattr(ev, "detail", None),
                        "fold": getattr(ev, "fold", None),
                    }
                norm.append(d)
            except Exception:
                norm.append({"raw": str(ev)})
        df = pd.DataFrame(norm)

    # Timestamp convenience
    if "ts" in df.columns and "timestamp" not in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
        except Exception:
            pass

    # Ensure expected columns exist
    for col in ["fold", "ts", "symbol", "reason", "action", "detail"]:
        if col not in df.columns:
            df[col] = None

    return df


def write_risk_events_csv(out_path: Path, rows: Any) -> None:
    """
    Write a RiskEvents CSV suitable for WF reports.

    - `rows` can be a list of RiskEvents (any shape) or a DataFrame.
    - The CSV includes columns: fold, ts, symbol, reason, action, detail (and keeps others if present).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _events_to_df(rows)
    # Reorder primary columns first if present
    primary = [c for c in ["fold", "ts", "symbol", "reason", "action", "detail"] if c in df.columns]
    others = [c for c in df.columns if c not in primary]
    df = df[primary + others]
    df.to_csv(out_path, index=False)


def summarize_veto_reasons(rows: Any, top_n: int = 10):
    """
    Return a list of (reason, count) for rows where action == 'VETO' (case-insensitive).
    Accepts a DataFrame or an iterable of RiskEvents/dicts.
    """
    df = _events_to_df(rows)
    if "action" not in df.columns or "reason" not in df.columns or df.empty:
        return []

    mask = df["action"].astype(str).str.upper() == "VETO"
    reasons = df.loc[mask, "reason"].dropna().astype(str)
    return Counter(reasons).most_common(top_n)
