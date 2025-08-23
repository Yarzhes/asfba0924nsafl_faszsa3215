# S7-T2: Ensemble Voting & Combination Specification

**Author**: Roo
**Date**: 2025-08-22
**Status**: DRAFT

## 1. Overview

This document specifies the mechanism for combining signals from multiple, individual trading strategies into a single, final trading decision. It defines the voting rules, confidence aggregation, and provides worked examples to illustrate the process.

## 2. Ensemble Voting Mechanism

The core of the ensemble is a weighted-sum voting mechanism that is sensitive to the prevailing market regime.

### 2.1. Per-Regime Weight Profiles

Strategy weights are not static. They adapt based on the current market regime, as determined by the Regime Classifier. This allows the system to favor strategies that are expected to perform well in the current market conditions.

**Weight Profile Schema:**

```yaml
regime_weights:
  TREND_UP:
    breakout_book: 0.4
    liquidation_chaser: 0.3
    momentum_scalper: 0.2
    reversion_trader: 0.1
  TREND_DOWN:
    breakout_book: 0.4
    liquidation_chaser: 0.3
    momentum_scalper: 0.2
    reversion_trader: 0.1
  RANGE:
    breakout_book: 0.1
    liquidation_chaser: 0.2
    momentum_scalper: 0.3
    reversion_trader: 0.4
  VOLATILE:
    breakout_book: 0.2
    liquidation_chaser: 0.4
    momentum_scalper: 0.3
    reversion_trader: 0.1
```

### 2.2. Voting Logic

The voting process calculates a weighted sum of the signals from all active strategies.

**Pseudocode:**

```python
function calculate_ensemble_signal(strategy_signals, regime):
    """
    Calculates the final ensemble signal based on a weighted vote.

    Args:
        strategy_signals (list): A list of signal objects from each strategy.
                                 Each object contains:
                                 - name (str)
                                 - direction (int): 1 for long, -1 for short, 0 for neutral
                                 - conf_calibrated (float): 0.0 to 1.0
                                 - sub_signals (dict): e.g., {'book_flip': True}
        regime (str): The current market regime (e.g., 'TREND_UP', 'RANGE').

    Returns:
        dict: A dictionary containing the final signal direction,
              confidence, and supporting strategies.
    """
    final_vote_score = 0.0
    contributing_long_signals = []
    contributing_short_signals = []

    # Get the weight profile for the current regime
    regime_weight_profile = get_weights_for_regime(regime) # loads from config

    for signal in strategy_signals:
        # 1. Apply Veto Rules
        if is_vetoed(signal):
            continue # Skip this signal if vetoed

        # 2. Get the strategy's weight for the current regime
        weight = regime_weight_profile.get(signal.name, 0.0)

        # 3. Calculate weighted vote
        vote = signal.direction * weight
        final_vote_score += vote

        # 4. Track contributing signals for confidence aggregation
        if signal.direction == 1:
            contributing_long_signals.append(signal)
        elif signal.direction == -1:
            contributing_short_signals.append(signal)

    # 5. Apply Majority Threshold
    MAJORITY_THRESHOLD = 0.5 # Configurable
    final_direction = 0
    if final_vote_score > MAJORITY_THRESHOLD:
        final_direction = 1
    elif final_vote_score < -MAJORITY_THRESHOLD:
        final_direction = -1

    # 6. Aggregate Confidence
    if final_direction == 1:
        final_confidence = aggregate_confidence(contributing_long_signals, regime)
        supporting_strategies = [s.name for s in contributing_long_signals]
    elif final_direction == -1:
        final_confidence = aggregate_confidence(contributing_short_signals, regime)
        supporting_strategies = [s.name for s in contributing_short_signals]
    else:
        final_confidence = 0.0
        supporting_strategies = []

    return {
        'direction': final_direction,
        'confidence': final_confidence,
        'vote_score': final_vote_score,
        'supporting_strategies': supporting_strategies
    }
```

### 2.3. Veto List

The veto list provides a mechanism to invalidate a signal unless specific pre-conditions are met. This acts as a safety filter.

**Veto Rules:**
1.  A `breakout_book` signal is **VETOED** if its `book-flip` sub-signal is not present (`False`).
2.  (Placeholder for future rules)

**Pseudocode:**
```python
function is_vetoed(signal):
    """
    Checks if a signal should be vetoed based on a set of rules.
    """
    if signal.name == 'breakout_book' and not signal.sub_signals.get('book_flip', False):
        return True
    # ... other veto rules ...
    return False
```


## 3. Confidence Aggregation

Once the final direction is determined, the confidence scores of all *contributing* signals (i.e., those voting in the same direction) are blended.

The aggregation method is a **weighted average of the `conf_calibrated` values**, where the weights are the same as those used in the voting process.

**Pseudocode:**
```python
function aggregate_confidence(contributing_signals, regime):
    """
    Aggregates the confidence of signals that contributed to the final vote.

    Args:
        contributing_signals (list): List of signals voting in the winning direction.
        regime (str): The current market regime.

    Returns:
        float: The final, aggregated confidence score.
    """
    weighted_confidence_sum = 0.0
    total_weight_sum = 0.0

    regime_weight_profile = get_weights_for_regime(regime)

    for signal in contributing_signals:
        weight = regime_weight_profile.get(signal.name, 0.0)
        weighted_confidence_sum += signal.conf_calibrated * weight
        total_weight_sum += weight

    if total_weight_sum == 0:
        return 0.0

    return weighted_confidence_sum / total_weight_sum
```

## 4. Worked Examples

In all examples, the `MAJORITY_THRESHOLD` is `0.5`.

### Example 1: Strong Long Signal in a Trending Market

*   **Regime**: `TREND_UP`
*   **Active Signals**:
    *   `breakout_book`: direction=1, conf_calibrated=0.85, sub_signals={'book_flip': True}
    *   `liquidation_chaser`: direction=1, conf_calibrated=0.90
    *   `momentum_scalper`: direction=1, conf_calibrated=0.75
    *   `reversion_trader`: direction=0, conf_calibrated=0.0

**1. Veto Check:**
*   `breakout_book`: `book_flip` is `True`. **Pass.**

**2. Voting Calculation:**
*   Weights for `TREND_UP`: `breakout_book`(0.4), `liquidation_chaser`(0.3), `momentum_scalper`(0.2), `reversion_trader`(0.1)
*   `breakout_book`: 1 * 0.4 = `+0.4`
*   `liquidation_chaser`: 1 * 0.3 = `+0.3`
*   `momentum_scalper`: 1 * 0.2 = `+0.2`
*   `reversion_trader`: 0 * 0.1 = `0.0`
*   **`final_vote_score`**: 0.4 + 0.3 + 0.2 + 0.0 = **`0.9`**

**3. Final Direction:**
*   `final_vote_score` (0.9) > `MAJORITY_THRESHOLD` (0.5).
*   **Final Direction: `1` (Long)**

**4. Confidence Aggregation:**
*   Contributing Signals: `breakout_book`, `liquidation_chaser`, `momentum_scalper`
*   Weighted Confidence Sum: (0.85 * 0.4) + (0.90 * 0.3) + (0.75 * 0.2) = 0.34 + 0.27 + 0.15 = `0.76`
*   Total Weight Sum: 0.4 + 0.3 + 0.2 = `0.9`
*   **`final_confidence`**: 0.76 / 0.9 = **`0.844`**

**Result:**
```
{
  'direction': 1,
  'confidence': 0.844,
  'vote_score': 0.9,
  'supporting_strategies': ['breakout_book', 'liquidation_chaser', 'momentum_scalper']
}
```

---

### Example 2: Conflicting Signals in a Ranging Market

*   **Regime**: `RANGE`
*   **Active Signals**:
    *   `breakout_book`: direction=1, conf_calibrated=0.60, sub_signals={'book_flip': True}
    *   `liquidation_chaser`: direction=0, conf_calibrated=0.0
    *   `momentum_scalper`: direction=-1, conf_calibrated=0.70
    *   `reversion_trader`: direction=-1, conf_calibrated=0.80

**1. Veto Check:**
*   All signals pass veto checks.

**2. Voting Calculation:**
*   Weights for `RANGE`: `breakout_book`(0.1), `liquidation_chaser`(0.2), `momentum_scalper`(0.3), `reversion_trader`(0.4)
*   `breakout_book`: 1 * 0.1 = `+0.1`
*   `liquidation_chaser`: 0 * 0.2 = `0.0`
*   `momentum_scalper`: -1 * 0.3 = `-0.3`
*   `reversion_trader`: -1 * 0.4 = `-0.4`
*   **`final_vote_score`**: 0.1 - 0.3 - 0.4 = **`-0.6`**

**3. Final Direction:**
*   `final_vote_score` (-0.6) < -`MAJORITY_THRESHOLD` (-0.5).
*   **Final Direction: `-1` (Short)**

**4. Confidence Aggregation:**
*   Contributing Signals: `momentum_scalper`, `reversion_trader`
*   Weighted Confidence Sum: (0.70 * 0.3) + (0.80 * 0.4) = 0.21 + 0.32 = `0.53`
*   Total Weight Sum: 0.3 + 0.4 = `0.7`
*   **`final_confidence`**: 0.53 / 0.7 = **`0.757`**

**Result:**
```
{
  'direction': -1,
  'confidence': 0.757,
  'vote_score': -0.6,
  'supporting_strategies': ['momentum_scalper', 'reversion_trader']
}
```

---

### Example 3: Signal Vetoed

*   **Regime**: `TREND_UP`
*   **Active Signals**:
    *   `breakout_book`: direction=1, conf_calibrated=0.95, sub_signals={'book_flip': False}  **<-- Will be vetoed**
    *   `liquidation_chaser`: direction=1, conf_calibrated=0.80
    *   `momentum_scalper`: direction=-1, conf_calibrated=0.60
    *   `reversion_trader`: direction=0, conf_calibrated=0.0

**1. Veto Check:**
*   `breakout_book`: `book_flip` is `False`. **VETOED.** The signal is discarded.

**2. Voting Calculation:**
*   Weights for `TREND_UP`: `liquidation_chaser`(0.3), `momentum_scalper`(0.2), `reversion_trader`(0.1)
*   `breakout_book`: (Vetoed) = `0.0`
*   `liquidation_chaser`: 1 * 0.3 = `+0.3`
*   `momentum_scalper`: -1 * 0.2 = `-0.2`
*   `reversion_trader`: 0 * 0.1 = `0.0`
*   **`final_vote_score`**: 0.3 - 0.2 = **`0.1`**

**3. Final Direction:**
*   `final_vote_score` (0.1) is between `-0.5` and `0.5`.
*   **Final Direction: `0` (Neutral)**

**4. Confidence Aggregation:**
*   Since the final direction is neutral, confidence is not calculated.
*   **`final_confidence`**: **`0.0`**

**Result:**
```
{
  'direction': 0,
  'confidence': 0.0,
  'vote_score': 0.1,
  'supporting_strategies': []
}
```
