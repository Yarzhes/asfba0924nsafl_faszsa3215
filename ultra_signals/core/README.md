# Core

This directory contains the foundational, cross-cutting components of the application.

- `config.py`: Handles loading, validation, and providing access to configuration from `settings.yaml` and environment variables.
- `timeutils.py`: Provides utilities for handling timezones, market session timings, and other time-related calculations.
- `mathutils.py`: Contains common mathematical and numerical functions used across feature computations.
- `feature_store.py`: An in-memory database that stores and manages incoming market data and derived features for each symbol and timeframe.