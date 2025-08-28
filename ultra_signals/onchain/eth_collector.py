"""Ethereum RPC collector using public RPC (Cloudflare) and Coingecko for prices.

Notes:
 - Uses public RPC endpoints only; no API keys required.
 - Polite caching and simple rate-limiting for price lookups.
 - Minimal Transfer event decoding: looks for topics[0]==ERC20_TRANSFER_SIG
 - Multi-hop collapse: collapse immediate intra-tx internal transfers (peels) by
   grouping by tx_hash and summing per (from,to) filtered by short TTL.
 - Deduper: short-lived in-memory seen set for tx hashes.
"""
from __future__ import annotations
import asyncio
import time
import json
from typing import Dict, Any, Optional, Tuple, List
import aiohttp
from loguru import logger
from .collectors import BaseCollector

# Standard ERC-20 Transfer signature
ERC20_TRANSFER_SIG = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'


class SimplePriceCache:
    def __init__(self, ttl_sec: int = 30):
        self.ttl = ttl_sec
        self._cache: Dict[str, Tuple[float, float]] = {}  # id -> (price, ts)
        self._lock = asyncio.Lock()
    async def get_price(self, session: aiohttp.ClientSession, coingecko_id: Optional[str], token_address: Optional[str] = None, decimals: int = 18) -> Optional[float]:
        """Try Coingecko first, then 1inch quote fallback to derive USD price.

        Returns price in USD per token unit.
        """
        now = time.time()
        key = coingecko_id or token_address
        if not key:
            return None
        entry = self._cache.get(key)
        if entry and now - entry[1] < self.ttl:
            return entry[0]
        async with self._lock:
            entry = self._cache.get(key)
            if entry and now - entry[1] < self.ttl:
                return entry[0]
            # Coingecko attempt
            if coingecko_id:
                try:
                    url = f'https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd'
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            j = await resp.json()
                            p = j.get(coingecko_id, {}).get('usd')
                            if p is not None:
                                self._cache[key] = (float(p), now)
                                return float(p)
                        else:
                            logger.debug('Coingecko fetch status %s', resp.status)
                except Exception:
                    logger.debug('Coingecko fetch exception')
            # Uniswap V2 subgraph fallback
            if token_address:
                try:
                    # 1inch quote: ask for `amount = 10**decimals` of token -> to WETH amount
                    chain_id = 1
                    weth = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'
                    amount = str(10 ** decimals)
                    url = f'https://api.1inch.io/v5.0/{chain_id}/quote?fromTokenAddress={token_address}&toTokenAddress={weth}&amount={amount}'
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            j = await resp.json()
                            to_amount = int(j.get('toTokenAmount', '0'))
                            # get ETH price from coingecko
                            eth_price = None
                            try:
                                async with session.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd', timeout=10) as r2:
                                    if r2.status == 200:
                                        jd = await r2.json()
                                        eth_price = float(jd.get('ethereum', {}).get('usd', 0.0))
                            except Exception:
                                pass
                            if to_amount and eth_price:
                                # to_amount is in wei for WETH (18 decimals)
                                p = (to_amount / (10 ** 18)) * eth_price / (10 ** decimals)
                                self._cache[key] = (p, now)
                                return p
                except Exception:
                    logger.debug('1inch price fetch failed')
            return None


class EthereumCollector(BaseCollector):
    def __init__(self, feature_store, cfg: Dict[str, Any]):
        super().__init__(feature_store, cfg)
        self.rpc = cfg.get('rpc', 'https://cloudflare-eth.com')
        self.price_cache = SimplePriceCache(ttl_sec=cfg.get('price_ttl_sec', 30))
        self.token_map = cfg.get('token_map') or {}  # contract -> {'symbol': 'USDT', 'decimals':6, 'coingecko_id':'tether'}
        self._tx_group: Dict[str, List[Dict[str, Any]]] = {}  # tx_hash -> list of parsed transfers
        self._seen_tx: Dict[str, float] = {}
        self._dampen_ttl = cfg.get('dedupe_ttl_sec', 60)
        self.confirmations = int(cfg.get('confirmations', 2))
        # processed_blocks: list of dict {number, hash, txs:[tx_hash,...]}
        self.processed_blocks: List[Dict[str, Any]] = []
        # committed txs map tx_hash -> commit_info (block_number, symbol, from, to, usd)
        self.committed: Dict[str, Dict[str, Any]] = {}
        # optional redis config (use redis.asyncio if provided)
        self._redis = None
        self.redis_url = cfg.get('redis_url')
        if self.redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(self.redis_url)
            except Exception:
                logger.warning('Redis async client not available; continuing without persistent dedupe')

    async def _restore_state_from_redis(self):
        if not self._redis:
            return
        try:
            pb = await self._redis.get('onchain:processed_blocks')
            if pb:
                try:
                    self.processed_blocks = json.loads(pb)
                except Exception:
                    logger.debug('Failed parse persisted processed_blocks')
            cm = await self._redis.get('onchain:committed')
            if cm:
                try:
                    self.committed = json.loads(cm)
                except Exception:
                    logger.debug('Failed parse persisted committed')
        except Exception:
            logger.debug('No persisted onchain state or failed to load')

    async def _persist_state(self):
        if not self._redis:
            return
        try:
            await self._redis.set('onchain:processed_blocks', json.dumps(self.processed_blocks))
            await self._redis.set('onchain:committed', json.dumps(self.committed))
        except Exception:
            logger.debug('Persist state to redis failed')

    async def fetch_logs(self, from_block: int, to_block: int) -> List[Dict[str, Any]]:
        payload = {
            'jsonrpc': '2.0',
            'method': 'eth_getLogs',
            'params': [{
                'fromBlock': hex(from_block),
                'toBlock': hex(to_block),
                'topics': [ERC20_TRANSFER_SIG]
            }],
            'id': 1
        }
        async with aiohttp.ClientSession() as s:
            try:
                async with s.post(self.rpc, json=payload, timeout=20) as resp:
                    if resp.status != 200:
                        logger.warning('eth_getLogs failed %s %s', resp.status, await resp.text())
                        return []
                    j = await resp.json()
                    return j.get('result') or []
            except Exception:
                logger.exception('eth_getLogs call failed')
                return []

    @staticmethod
    def _decode_transfer_log(log: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            topics = log.get('topics') or []
            if not topics or topics[0].lower() != ERC20_TRANSFER_SIG:
                return None
            from_addr = '0x' + topics[1][-40:]
            to_addr = '0x' + topics[2][-40:]
            data = log.get('data', '0x0')
            value_raw = int(data, 16)
            contract = log.get('address')
            return {'tx_hash': log.get('transactionHash'), 'contract': contract, 'from': from_addr, 'to': to_addr, 'value_raw': value_raw, 'blockNumber': int(log.get('blockNumber', '0'), 16)}
        except Exception:
            return None

    async def add_block(self, block_number: int, block_hash: str, logs: List[Dict[str, Any]]):
        """Process a new block: group logs, detect reorgs, commit txs after confirmations."""
        now = time.time()
        # detect reorg: if we already have a block with same number but different hash
        existing = next((b for b in self.processed_blocks if b['number'] == block_number), None)
        if existing and existing.get('hash') != block_hash:
            logger.info('Detected reorg at block %s', block_number)
            # rollback blocks above this number (inclusive)
            await self._rollback_from_block(block_number)
        # parse logs into tx groups
        for lg in logs:
            rec = self._decode_transfer_log(lg)
            if not rec:
                continue
            tx = rec['tx_hash']
            self._tx_group.setdefault(tx, []).append(rec)
        # register processed block
        txs = list({tx for lg in logs for tx in [lg.get('transactionHash')] if tx})
        self.processed_blocks.append({'number': block_number, 'hash': block_hash, 'txs': txs, 'ts': now})
        # commit txs that are now confirmed
        await self._commit_confirmed(block_number)

    async def _rollback_from_block(self, block_number: int):
        # find blocks to rollback (those with number >= block_number)
        to_rollback = [b for b in self.processed_blocks if b['number'] >= block_number]
        for b in to_rollback:
            for tx in b.get('txs', []):
                # if tx committed, attempt rollback
                if tx in self.committed:
                    info = self.committed.pop(tx)
                    # attempt to call store revert if available
                    try:
                        if hasattr(self.store, 'whale_revert_exchange_flow'):
                            self.store.whale_revert_exchange_flow(info)
                        else:
                            logger.warning('No revert API on store; cannot rollback %s', tx)
                    except Exception:
                        logger.exception('Error during revert for %s', tx)
                # also remove from redis if present
                if self._redis:
                    try:
                        await self._redis.delete(tx)
                    except Exception:
                        logger.debug('Failed deleting tx from redis')
        # remove from processed_blocks
        self.processed_blocks = [b for b in self.processed_blocks if b['number'] < block_number]

    async def _commit_confirmed(self, current_block_number: int):
        # for each processed block, if block.number + confirmations <= current_block_number, commit its txs
        cutoff = current_block_number - self.confirmations + 1
        to_commit_blocks = [b for b in self.processed_blocks if b['number'] <= cutoff]
        async with aiohttp.ClientSession() as s:
            for b in to_commit_blocks:
                for tx in b.get('txs', []):
                    # dedupe persistent
                    if self._redis:
                        try:
                            seen = await self._redis.get(tx)
                            if seen:
                                continue
                        except Exception:
                            logger.debug('Redis get failed')
                    # collapse grouped transfers for this tx
                    group = self._tx_group.get(tx, [])
                    per_pair: Dict[Tuple[str, str, str], int] = {}
                    for g in group:
                        k = (g['contract'], g['from'].lower(), g['to'].lower())
                        per_pair[k] = per_pair.get(k, 0) + g['value_raw']
                    for (contract, frm, to), raw in per_pair.items():
                        meta = self.token_map.get(contract.lower()) or {}
                        decimals = int(meta.get('decimals', 18))
                        symbol = meta.get('symbol') or contract
                        coid = meta.get('coingecko_id')
                        amount = raw / (10 ** decimals)
                        usd = None
                        if coid or contract:
                            price = await self.price_cache.get_price(s, coid, contract, decimals)
                            if price is not None:
                                usd = amount * price
                        if usd is None:
                            logger.debug('No price for %s; skipping tx %s', contract, tx)
                            continue
                        # commit to store
                        try:
                            self.ingest_transfer('ethereum', symbol, frm, to, usd, int(time.time()*1000))
                            self.committed[tx] = {'tx': tx, 'block': b['number'], 'symbol': symbol, 'from': frm, 'to': to, 'usd': usd}
                            if self._redis:
                                try:
                                    await self._redis.set(tx, 1, ex=self._dampen_ttl)
                                except Exception:
                                    logger.debug('Redis set failed')
                        except Exception:
                            logger.exception('Commit ingest failed for %s', tx)
                    # cleanup grouped tx
                    try:
                        del self._tx_group[tx]
                    except KeyError:
                        pass
                # remove block from processed_blocks once committed fully
                try:
                    self.processed_blocks.remove(b)
                except ValueError:
                    pass

    async def poll_loop(self, start_block: Optional[int] = None, poll_interval: int = 6, backoff_initial: float = 1.0):
        """Simple polling loop that fetches new blocks and processes logs."""
        backoff = backoff_initial
        last_block = start_block
        session = aiohttp.ClientSession()
        # restore redis-persisted state if available before starting
        try:
            if self._redis:
                try:
                    await self._restore_state_from_redis()
                except Exception:
                    logger.debug('Failed restoring state from redis')
        except Exception:
            pass

        try:
            while True:
                try:
                    # get latest block
                    payload = {'jsonrpc':'2.0','method':'eth_blockNumber','params':[],'id':1}
                    async with session.post(self.rpc, json=payload, timeout=10) as resp:
                        j = await resp.json()
                        latest = int(j.get('result', '0'), 16)
                    if last_block is None:
                        last_block = latest - 1
                    if latest > last_block:
                        from_b = last_block + 1
                        to_b = latest
                        # for simplicity fetch logs in the full range
                        logs = await self.fetch_logs(from_b, to_b)
                        # group logs by block for add_block
                        by_block: Dict[int, List[Dict[str, Any]]] = {}
                        for lg in logs:
                            bn = int(lg.get('blockNumber', '0'), 16)
                            by_block.setdefault(bn, []).append(lg)
                        # process blocks in order
                        for bn in sorted(by_block.keys()):
                            # fetch block hash
                            payload_b = {'jsonrpc':'2.0','method':'eth_getBlockByNumber','params':[hex(bn), False],'id':1}
                            async with session.post(self.rpc, json=payload_b, timeout=10) as rb:
                                jb = await rb.json()
                                bh = jb.get('result', {}).get('hash')
                            await self.add_block(bn, bh, by_block[bn])
                            # persist after processing each block
                            await self._persist_state()
                        last_block = latest
                    backoff = backoff_initial
                except Exception:
                    logger.exception('Poll loop error; backing off')
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                await asyncio.sleep(poll_interval)
        finally:
            await session.close()

    async def poll_once(self, last_block: int) -> Optional[int]:
        """Helper for tests: perform one poll iteration from last_block+1 to latest and return latest block number."""
        session = aiohttp.ClientSession()
        try:
            payload = {'jsonrpc':'2.0','method':'eth_blockNumber','params':[],'id':1}
            async with session.post(self.rpc, json=payload, timeout=10) as resp:
                j = await resp.json()
                latest = int(j.get('result', '0'), 16)
            if latest > last_block:
                logs = await self.fetch_logs(last_block + 1, latest)
                by_block: Dict[int, List[Dict[str, Any]]] = {}
                for lg in logs:
                    bn = int(lg.get('blockNumber', '0'), 16)
                    by_block.setdefault(bn, []).append(lg)
                for bn in sorted(by_block.keys()):
                    payload_b = {'jsonrpc':'2.0','method':'eth_getBlockByNumber','params':[hex(bn), False],'id':1}
                    async with session.post(self.rpc, json=payload_b, timeout=10) as rb:
                        jb = await rb.json()
                        bh = jb.get('result', {}).get('hash')
                    await self.add_block(bn, bh, by_block[bn])
                    await self._persist_state()
            return latest
        finally:
            await session.close()


__all__ = ['EthereumCollector', 'SimplePriceCache', 'ERC20_TRANSFER_SIG']
