Orderflow analytics: CVD, orderbook imbalance, tape speed, footprint S/R levels.

This module provides a lightweight OrderflowEngine used by other systems.

Demo
----

Run the CLI demo which streams a simulated feed into the calculator and persists FeatureView records into sqlite:

```powershell
python -m ultra_signals.orderflow.service --duration 10 --interval 1
```

Programmatic usage example is available in the module docstring and shows how to create a `SimulatedFeed`, `OrderflowCalculator`, and `FeatureViewWriter` and run `OrderflowService`.
