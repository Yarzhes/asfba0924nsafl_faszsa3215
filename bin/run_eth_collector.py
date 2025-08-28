"""Small CLI to run EthereumCollector for testing/demo.

Usage:
    python bin/run_eth_collector.py

This script runs the poll_loop until interrupted. Configure collector via
`cfg` below or load from a JSON file.
"""
import asyncio
import signal
from ultra_signals.onchain.eth_collector import EthereumCollector
from ultra_signals.onchain.collectors import BaseCollector

class DummyStore:
    def whale_add_exchange_flow(self, symbol, direction, usd, ts_ms):
        print('INGEST:', symbol, direction, usd, ts_ms)

async def main():
    store = DummyStore()
    cfg = {
        'rpc': 'https://cloudflare-eth.com',
        'token_map': {},
        'price_ttl_sec': 30,
        'redis_url': None,
        'confirmations': 2,
    }
    ec = EthereumCollector(store, cfg)
    loop = asyncio.get_event_loop()
    stop = asyncio.Event()
    def _stop(*args):
        stop.set()
    loop.add_signal_handler(signal.SIGINT, _stop)
    loop.add_signal_handler(signal.SIGTERM, _stop)
    task = asyncio.create_task(ec.poll_loop())
    await stop.wait()
    task.cancel()

if __name__ == '__main__':
    asyncio.run(main())
