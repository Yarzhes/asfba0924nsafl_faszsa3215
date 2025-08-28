Ethereum onchain collector

This folder contains a public-RPC-based Ethereum collector for large ERC-20 transfers.

Usage (simple):

- Create a small runner script or use the CLI helper `bin/run_eth_collector.py`.
- Provide a config dict with keys you need e.g. `rpc`, `token_map`, `redis_url`.

Collector features:
- Polling loop (`poll_loop`) with backoff and block confirmation handling.
- Per-tx grouping to collapse peel chains within the same transaction.
- Optional Redis persistence (set `redis_url`) for processed_blocks and committed txs.
- Price resolution via Coingecko, falling back to 1inch quote when required.

Notes:
- This is a starter implementation focusing on correctness and testability. For
  production, run multiple worker processes, persist state externally, and harden RPC error handling.
