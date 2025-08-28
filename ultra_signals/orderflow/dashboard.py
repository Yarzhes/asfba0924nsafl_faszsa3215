"""Minimal Streamlit + Plotly dashboard to visualize CVD vs price and footprint S/R overlays.

Run with: `streamlit run ultra_signals/orderflow/dashboard.py`
This is intentionally simple and dependency-light: Plotly is optional; falls back to basic prints.
"""
try:
    import streamlit as st
    import plotly.graph_objs as go
    import pandas as pd
except Exception:
    st = None


def run_view(feature_writer=None, symbol='BTCUSDT'):
    """Render a simple dashboard from a FeatureViewWriter or recent in-memory data.
    feature_writer: instance of FeatureViewWriter with query_recent()
    """
    if st is None:
        print("Streamlit or Plotly is not installed. Install streamlit and plotly to use the dashboard.")
        return
    st.title("Orderflow Micro Dashboard")
    st.sidebar.text_input('Symbol', value=symbol)
    df = None
    if feature_writer is not None:
        rows = feature_writer.query_recent(200)
        if rows:
            df = pd.DataFrame(rows)
    if df is None or df.empty:
        st.warning("No data available. Start adapters and writer first.")
        return
    df = df.sort_values('ts')
    df['dt'] = pd.to_datetime(df['ts'], unit='s')

    # CVD / of_micro_score timeseries with price overlay
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['dt'], y=df['of_micro_score'], name='of_micro_score', line=dict(color='blue')))
    if 'price' in df.columns:
        fig.add_trace(go.Scatter(x=df['dt'], y=df['price'], name='price', yaxis='y2', line=dict(color='black')))
    # footprint strength (sum of components if present)
    if 'components' in df.columns:
        df['footprint_strength'] = df['components'].apply(lambda c: sum(c.values()) if isinstance(c, dict) else 0)
        fig.add_trace(go.Bar(x=df['dt'], y=df['footprint_strength'], name='footprint_strength', yaxis='y3', opacity=0.3))
    fig.update_layout(
        title=f"Orderflow micro features for {symbol}",
        yaxis=dict(title='of_micro_score'),
        yaxis2=dict(title='price', overlaying='y', side='right'),
        yaxis3=dict(title='footprint', anchor='x', overlaying='y', side='right', position=0.98)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table of footprint levels
    st.subheader('Recent footprint levels')
    footprints = []
    for r in rows:
        fp = r.get('footprint')
        if fp:
            footprints.append({'ts': r['ts'], 'footprint': fp})
    st.table(footprints[:50])

    # Live refresh controls
    auto = st.sidebar.checkbox('Auto-refresh', value=False)
    interval = st.sidebar.number_input('Refresh interval (s)', min_value=1, max_value=60, value=5)
    if auto:
        st.experimental_rerun()


if __name__ == '__main__':
    # quick local-run convenience
    print('Streamlit dashboard entrypoint — run with `streamlit run`')
