"""Configuration knobs for orderflow engine (defaults)."""
DEFAULT = {
    "depth_levels": [1,5,25],
    "cvd_window": 300,
    "tape_window": 10,
    "tape_z_window": 60,
    "footprint_min_volume": 1000,
}
