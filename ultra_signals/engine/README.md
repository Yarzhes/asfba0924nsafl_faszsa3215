# Engine

This directory houses the core decision-making logic of the system. It consumes the `FeatureVector` and produces a `Signal`.

- `scoring.py`: Takes a `FeatureVector` and calculates weighted scores for each component to determine a potential directional bias.
- `risk_filters.py`: Applies a series of checks (e.g., max spread, avoiding news) to a potential signal to decide if it's safe to trade.
- `entries_exits.py`: Defines the logic for trade entry zones, stop-loss placement, and take-profit targets. Manages the lifecycle of a trade.
- `sizing.py`: Calculates the appropriate position size based on account risk parameters.
- `selectors.py`: Implements logic for dynamically selecting the universe of symbols and timeframes to trade.