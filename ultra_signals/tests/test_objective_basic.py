from ultra_signals.calibration.objective import composite_fitness

def test_objective_returns_scalar_and_metrics():
    metrics = {'profit_factor':2.0,'winrate':0.6,'sharpe':1.0,'max_drawdown':-0.05,'trades':80}
    weights = {'profit_factor':0.35,'winrate':0.25,'sharpe':0.15,'max_drawdown':-0.15}
    penalties = {'min_trades':60}
    fitness = composite_fitness(metrics, weights, penalties)
    assert isinstance(fitness, float)
    # Ensure penalty not triggered
    assert fitness > 0
