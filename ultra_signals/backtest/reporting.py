import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter  # NEW: for summarize_veto_reasons

from ultra_signals.backtest.metrics import generate_equity_curve, compute_kpis

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
    routing_audit: Optional[pd.DataFrame] = None,  # NEW: meta-router audit rows
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

        # 3b) Advanced sizer visualization artifacts (if fields present)
        try:
            if trades is not None and not trades.empty:
                # Attempt to extract adv_risk_pct, meta_p, R
                cols = trades.columns
                adv_col = 'adv_risk_pct' if 'adv_risk_pct' in cols else None
                meta_col = 'meta_p' if 'meta_p' in cols else None
                r_col = 'R' if 'R' in cols else ('rr' if 'rr' in cols else None)
                if adv_col and trades[adv_col].notna().any():
                    self._plot_adv_size_dist(trades[adv_col].astype(float))
                if adv_col and meta_col and trades[meta_col].notna().any():
                    self._plot_conviction_corr(trades[meta_col].astype(float), trades[adv_col].astype(float))
                if adv_col and r_col and trades[r_col].notna().any():
                    self._plot_size_vs_R(trades[adv_col].astype(float), trades[r_col].astype(float))
                # Clamp metrics summary appended
                self._append_clamp_metrics(trades)
        except Exception:
            pass

        # 4) Optionally save RiskEvents + append top reasons to summary
        if risk_events:
            df = self._risk_events_to_df(risk_events)
            if not df.empty:
                self._save_risk_events_csv(df)
                self._append_top_vetoes_to_summary(df)

        # 5) Optional routing audit export
        if routing_audit is not None and not routing_audit.empty:
            self._save_routing_audit_csv(routing_audit)
            self._append_profile_breakdown(trades, routing_audit)

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
        # Also emit a markdown version (report.md) for downstream batch parser expectations
        md_path = self.output_dir / "report.md"
        try:
            with open(md_path, "w") as md:
                md.write("# Backtest Report\n\n")
                md.write("| Metric | Value |\n|--------|-------|\n")
                for key, value in kpis.items():
                    if isinstance(value, float):
                        md.write(f"| {key} | {value:.4f} |\n")
                    else:
                        md.write(f"| {key} | {value} |\n")
        except Exception:
            pass

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

    # ---------------- Advanced Sizer Plots -----------------
    def _plot_adv_size_dist(self, series: pd.Series):
        try:
            import matplotlib.pyplot as plt
            plt.style.use("seaborn-v0_8-darkgrid")
            fig, ax = plt.subplots(figsize=(8,5))
            series.dropna().plot(kind='hist', bins=20, ax=ax, alpha=0.7, color='steelblue')
            ax.set_title('Advanced Sizer Risk % Distribution')
            ax.set_xlabel('Risk % of Equity')
            ax.set_ylabel('Frequency')
            fig.tight_layout()
            fig.savefig(self.output_dir / 'risk_pct_dist.png')
            plt.close(fig)
        except Exception:
            pass

    def _plot_conviction_corr(self, meta: pd.Series, risk: pd.Series):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            m = meta.dropna(); r = risk.loc[m.index].dropna()
            if m.empty or r.empty:
                return
            # Align indexes after drops
            idx = m.index.intersection(r.index)
            m = m.loc[idx]; r = r.loc[idx]
            if len(idx) < 5:
                return
            corr = float(np.corrcoef(m.values, r.values)[0,1]) if len(m)>1 else 0.0
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(m.values, r.values, alpha=0.6, s=25, color='darkorange')
            ax.set_title(f'Meta Probability vs Risk % (corr={corr:.2f})')
            ax.set_xlabel('Meta p')
            ax.set_ylabel('Risk %')
            fig.tight_layout(); fig.savefig(self.output_dir / 'conviction_corr.png'); plt.close(fig)
        except Exception:
            pass

    def _plot_size_vs_R(self, risk: pd.Series, r_vals: pd.Series):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            rk = risk.dropna(); rv = r_vals.loc[rk.index].dropna()
            if rk.empty or rv.empty:
                return
            idx = rk.index.intersection(rv.index)
            rk = rk.loc[idx]; rv = rv.loc[idx]
            if len(idx) < 5:
                return
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(rk.values, rv.values, alpha=0.6, s=25, color='seagreen')
            ax.set_title('Risk % vs R Multiple')
            ax.set_xlabel('Risk %')
            ax.set_ylabel('R (P/L ÷ Risk)')
            fig.tight_layout(); fig.savefig(self.output_dir / 'size_vs_R_scatter.png'); plt.close(fig)
        except Exception:
            pass

    def _append_clamp_metrics(self, trades: pd.DataFrame):
        """Append clamp reason counts to summary if present (expects a 'adv_reasons' or 'reasons' column)."""
        if trades is None or trades.empty:
            return
        reason_col = None
        for c in ['adv_reasons','reasons']:
            if c in trades.columns:
                reason_col = c; break
        if reason_col is None:
            return
        # Expect list-like or string with comma separation
        reasons_all = []
        for v in trades[reason_col]:
            if v is None:
                continue
            if isinstance(v, list):
                reasons_all.extend([str(x) for x in v])
            else:
                # Split on commas
                try:
                    parts = [p.strip() for p in str(v).split(',') if p.strip()]
                    reasons_all.extend(parts)
                except Exception:
                    pass
        if not reasons_all:
            return
        from collections import Counter
        counts = Counter(reasons_all)
        path = self.output_dir / 'summary.txt'
        try:
            with open(path,'a') as f:
                f.write("\nClamp Reason Counts\n")
                f.write("="*25 + "\n")
                for reason, cnt in counts.most_common():
                    f.write(f"- {reason}: {cnt}\n")
        except Exception:
            pass

    # ---------------- NEW: Routing audit + profile metrics ----------------
    def _save_routing_audit_csv(self, audit: pd.DataFrame):
        # Normalize legacy/new column names
        df = audit.copy()
        # If legacy columns present, keep them; else ensure spec columns exist
        for col in ['ts','symbol','tf','profile_id','version','used_overrides','fall_back_chain']:
            if col not in df.columns:
                df[col] = None
        path = self.output_dir / "routing_audit.csv"
        try:
            df.to_csv(path, index=False)
        except Exception:
            pass

    def _append_profile_breakdown(self, trades: pd.DataFrame, audit: pd.DataFrame):
        """Append per-profile trade counts & PnL aggregates to summary."""
        if trades is None or trades.empty:
            return
        # Expect profile info inside trades.vote_detail (json/dict). Extract profile_id & version.
        prof_ids = []
        versions = []
        if 'vote_detail' in trades.columns:
            import json
            for v in trades['vote_detail']:
                pid = ver = None
                try:
                    d = v if isinstance(v, dict) else json.loads(v)
                    prof = d.get('profile') or {}
                    pid = prof.get('profile_id') or prof.get('profile_id'.upper()) or prof.get('id')
                    ver = prof.get('version')
                except Exception:
                    pass
                prof_ids.append(pid)
                versions.append(ver)
        trades = trades.copy()
        if prof_ids:
            trades['profile_id'] = prof_ids
            trades['profile_version'] = versions
        # Basic aggregation
        agg = None
        rich_rows = []
        if 'pnl' in trades.columns and 'profile_id' in trades.columns:
            for pid, grp in trades.groupby('profile_id'):
                try:
                    k = compute_kpis(grp)
                except Exception:
                    k = {}
                rich_rows.append({
                    'profile_id': pid,
                    'trade_count': len(grp),
                    'total_pnl': float(grp['pnl'].sum()) if 'pnl' in grp.columns else 0.0,
                    'profit_factor': k.get('profit_factor'),
                    'win_rate_pct': k.get('win_rate_pct'),
                    'max_drawdown': k.get('max_drawdown'),
                })
            if rich_rows:
                import pandas as _pd
                agg = _pd.DataFrame(rich_rows)
        path = self.output_dir / 'summary.txt'
        if agg is not None and not agg.empty:
            with open(path, 'a') as f:
                f.write("\nPer-Profile Breakdown\n")
                f.write("="*30 + "\n")
                for _, row in agg.iterrows():
                    f.write(
                        f"- {row['profile_id']}: trades={row['trade_count']} pnl={row['total_pnl']:.2f} pf={row.get('profit_factor')} win%={row.get('win_rate_pct')} maxDD={row.get('max_drawdown')}\n"
                    )

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
