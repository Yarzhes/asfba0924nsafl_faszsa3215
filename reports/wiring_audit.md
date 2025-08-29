
# End-to-End Wiring Audit Report

This document outlines the connectivity audit of the signal generation pipeline.

## Connectivity Matrix

| Data Point / Feature      | Producer(s) (Sprint) | Consumer(s) (Sprint) | Status | Notes / Fixes Applied |
|---------------------------|----------------------|----------------------|--------|-----------------------|
| **Feeds (WS/REST)**       | `collectors` (S1)    | `features` (S11)     | [OK]      | Data flows correctly. |
| **FeatureStore**          | `features` (S11)     | `engine` (S2)        | [OK]      | Warmup periods checked. |
| **Alpha Emitters**        | `engine` (S2/S11/S13)| `ensemble` (S4)      | [OK]      | Candidates are generated. |
| **Ensemble/Meta-Scorer**  | `ensemble` (S4/S31)  | `guards` (S8)        | [OK]      | `p_win` threshold verified. |
| **MTF Confirmation**      | `strategy` (S30)     | `guards` (S30)       | [OK]      | Logic enabled in canary profile. |
| **Veto Stack**            | `guards` (S8/S18/...) | `guards` (S8)        | [OK]      | All vetoes are active. |
| **Regime Probability**    | `regime_engine` (S61)| `guards` (S61)       | [X]      | **FIXED**: Was stuck at low confidence. Adjusted sensitivity. |
| **Sizing Eligibility**    | `risk` (S12/S32)     | `risk` (S12)         | [OK]      | Canary checks eligibility only. |
| **Sniper Caps**           | `guards` (S21)       | `guards` (S21)       | [OK]      | Caps raised for canary. |
| **Telegram Emitter**      | `transport` (S5)     | (External)           | [X]      | **FIXED**: Implemented `send_blocked_signals_in_canary` and `send_pre_summary`. |
| **Order Flow Metrics**    | `orderflow` (S14/S51)| `guards` (S18)       | [OK]      | Verified. |
| **VPIN / Kyle's λ**       | `liquidity` (S49/S50)| `guards` (S49/S50)   | [OK]      | Verified. |
| **Funding/OI**            | `derivatives` (S54)  | `guards` (S54)       | [X]      | **FIXED**: Endpoint was not resolving correctly, causing false vetoes. |
| **Cross-Asset**           | `portfolio` (S42)    | `guards` (S42)       | [OK]      | Verified. |
| **Pattern Engine**        | `patterns` (S44)     | `ensemble` (S44)     | [OK]      | Verified. |
