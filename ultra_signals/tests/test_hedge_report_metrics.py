import math
from ultra_signals.portfolio.hedge_report import HedgeReportCollector, HedgeSnapshot

def test_equity_metrics_basic():
    rc = HedgeReportCollector()
    # fabricate equity curves
    hedged = [{"timestamp": i, "equity": 10000*(1+0.001*i)} for i in range(50)]
    unhedged = [{"timestamp": i, "equity": 10000*(1+0.0005*i)} for i in range(50)]
    comp = rc.compare(hedged, unhedged, timeframe='5m')
    assert comp['hedged']['total_return'] > comp['unhedged']['total_return']
    assert 'sharpe_annual' in comp['hedged']
    assert comp['delta']['total_return']['delta'] == comp['hedged']['total_return'] - comp['unhedged']['total_return']
