# Open Interest (OI) Providers

This directory contains modules for fetching Open Interest and liquidation data from various third-party sources. Each provider implements a common interface defined in `base.py`.

- `base.py`: Defines the abstract base class (interface) for all OI providers.
- `coinglass.py`: Implementation for the Coinglass API.
- `coinalyze.py`: Implementation for the Coinalyze API.