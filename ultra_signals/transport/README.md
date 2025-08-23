# Transport

This directory contains modules responsible for sending signals to external services.

- `telegram.py`: Formats the `Signal` object into a human-readable message and sends it to the configured Telegram chat, handling rate limiting and error cases.