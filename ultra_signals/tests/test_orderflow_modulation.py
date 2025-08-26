from ultra_signals.engine.orderflow import OrderFlowSnapshot, apply_orderflow_modulation


def test_orderflow_positive_boost_capped():
    snap = OrderFlowSnapshot(
        cvd=1000, cvd_chg=50,
        liq_long_notional=100_000, liq_short_notional=1_500_000,
        liq_side_dominant='short', liq_impulse=2.5,
        sweep_side='bid', sweep_flag=True
    )
    cfg = {"cvd_weight":0.4, "liquidation_weight":0.3, "liquidity_sweep_weight":0.3}
    new_conf, detail = apply_orderflow_modulation("LONG", 0.6, snap, cfg)
    # Expect boost > base but capped to +30%
    assert new_conf > 0.6
    assert new_conf <= 0.6 * 1.3 + 1e-9
    assert detail["boost_applied"] <= 0.30


def test_orderflow_conflicting_sweep_halves():
    snap = OrderFlowSnapshot(
        cvd=-500, cvd_chg=-20,
        liq_long_notional=800_000, liq_short_notional=100_000,
        liq_side_dominant='long', liq_impulse=1.2,
        sweep_side='ask', sweep_flag=True
    )
    cfg = {"cvd_weight":0.4, "liquidation_weight":0.3, "liquidity_sweep_weight":0.3}
    # LONG direction with sweep_side='ask' is conflicting -> halves after any boost logic
    base = 0.7
    new_conf, detail = apply_orderflow_modulation("LONG", base, snap, cfg)
    assert new_conf <= base * 0.51  # allow small float tolerance
    assert detail["conflict_halved"] is True