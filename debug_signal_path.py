#!/usr/bin/env python3
"""
Debug Signal Path - Simple End-to-End Test

This script traces the signal generation path step by step to identify where
signals are being blocked. It focuses on one symbol/timeframe to isolate issues.
"""

import asyncio
import time
from loguru import logger

from ultra_signals.core.config import load_settings
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.data.binance_ws import BinanceWSClient
from ultra_signals.data.funding_provider import FundingProvider
from ultra_signals.features import (
    compute_momentum,
    compute_trend,
    compute_volatility,
    compute_volume_flow_features,
    compute_orderbook_features,
    compute_derivatives_features,
    compute_funding_features,
)
from ultra_signals.core.custom_types import FeatureVector
from ultra_signals.engine.scoring import component_scores
from ultra_signals.engine.entries_exits import make_signal
from ultra_signals.engine.risk_filters import apply_filters


def setup_logging():
    """Setup simple console logging."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


async def debug_signal_generation():
    """Test signal generation path for a single symbol."""
    setup_logging()
    
    settings = load_settings("settings.yaml")
    logger.info("Settings loaded successfully")
    
    # Initialize components
    ws_client = BinanceWSClient(settings)
    test_symbol = "BTCUSDT"
    test_timeframe = "5m"
    
    # Subscribe to just one symbol for debugging
    ws_client.subscribe([test_symbol], [test_timeframe])
    
    funding_provider = FundingProvider(settings.funding_rate_provider.model_dump())
    feature_store = FeatureStore(
        warmup_periods=settings.features.warmup_periods,
        funding_provider=funding_provider,
    )
    
    logger.info(f"Starting debug session for {test_symbol}/{test_timeframe}")
    logger.info(f"Warmup periods required: {settings.features.warmup_periods}")
    
    # Counters for debugging
    total_events = 0
    kline_events = 0
    closed_klines = 0
    warmup_complete = False
    signals_generated = 0
    signals_passed = 0
    
    try:
        funding_task = asyncio.create_task(funding_provider.start())
        
        # Set a timeout for this debug session
        start_time = time.time()
        max_runtime = 120  # 2 minutes
        
        async for event in ws_client.start():
            total_events += 1
            
            # Check timeout
            if time.time() - start_time > max_runtime:
                logger.info("Debug session timeout reached")
                break
            
            # Ingest all events
            feature_store.ingest_event(event)
            
            # Log every 100 events
            if total_events % 100 == 0:
                logger.info(f"Processed {total_events} events so far...")
            
            # Only process closed klines for our test symbol/timeframe
            if (hasattr(event, 'symbol') and event.symbol == test_symbol and 
                hasattr(event, 'timeframe') and event.timeframe == test_timeframe and
                hasattr(event, 'closed') and event.closed):
                
                kline_events += 1
                logger.info(f"Closed kline #{kline_events} for {test_symbol}/{test_timeframe}")
                
                # Check warmup status
                ohlcv = feature_store.get_ohlcv(test_symbol, test_timeframe)
                if ohlcv is None:
                    logger.warning("OHLCV data is None!")
                    continue
                    
                data_length = len(ohlcv)
                logger.info(f"OHLCV length: {data_length}, required: {settings.features.warmup_periods}")
                
                if data_length < settings.features.warmup_periods:
                    logger.info(f"Still warming up: {data_length}/{settings.features.warmup_periods}")
                    continue
                
                if not warmup_complete:
                    warmup_complete = True
                    logger.success("🎉 Warmup complete! Starting signal generation...")
                
                closed_klines += 1
                
                try:
                    # Compute features step by step
                    logger.debug("Computing trend features...")
                    trend_features = compute_trend(ohlcv, settings.features.trend.model_dump())
                    logger.debug(f"Trend features: {trend_features}")
                    
                    logger.debug("Computing momentum features...")
                    momentum_features = compute_momentum(ohlcv, settings.features.momentum.model_dump())
                    logger.debug(f"Momentum features: {momentum_features}")
                    
                    logger.debug("Computing volatility features...")
                    volatility_features = compute_volatility(ohlcv, settings.features.volatility.model_dump())
                    logger.debug(f"Volatility features: {volatility_features}")
                    
                    logger.debug("Computing volume flow features...")
                    volume_features = compute_volume_flow_features(ohlcv, **settings.features.volume_flow.model_dump())
                    logger.debug(f"Volume features: {volume_features}")
                    
                    logger.debug("Computing orderbook features...")
                    orderbook_features = compute_orderbook_features(feature_store, test_symbol)
                    logger.debug(f"Orderbook features: {orderbook_features}")
                    
                    logger.debug("Computing derivatives features...")
                    derivatives_features = compute_derivatives_features(feature_store, test_symbol)
                    logger.debug(f"Derivatives features: {derivatives_features}")
                    
                    logger.debug("Computing funding features...")
                    funding_features = compute_funding_features(feature_store, test_symbol)
                    logger.debug(f"Funding features: {funding_features}")
                    
                    # Create feature vector
                    feature_vector = FeatureVector(
                        symbol=test_symbol,
                        timeframe=test_timeframe,
                        ohlcv=trend_features | momentum_features | volatility_features | volume_features,
                        orderbook=orderbook_features,
                        derivatives=derivatives_features,
                        funding=funding_features,
                    )
                    logger.debug("Feature vector created")
                    
                    # Compute scores
                    logger.debug("Computing component scores...")
                    scores = component_scores(feature_vector, settings.features.model_dump())
                    logger.info(f"Component scores: {scores}")
                    
                    # Calculate weighted final score
                    weights = settings.engine.scoring_weights.model_dump()
                    final_score = sum(scores.get(comp, 0.0) * w for comp, w in weights.items())
                    final_score = max(-1.0, min(1.0, final_score))
                    threshold = settings.engine.thresholds.enter
                    
                    logger.info(f"Final score: {final_score:.4f}, threshold: {threshold}")
                    logger.info(f"Score breakdown: {dict(zip(weights.keys(), [scores.get(k, 0.0) * w for k, w in weights.items()]))}")
                    
                    # Generate signal
                    logger.debug("Generating signal...")
                    signal = make_signal(
                        symbol=test_symbol,
                        timeframe=test_timeframe,
                        component_scores=scores,
                        weights=weights,
                        thresholds=settings.engine.thresholds.model_dump(),
                        features=feature_vector,
                        ohlcv=ohlcv,
                    )
                    
                    signals_generated += 1
                    logger.info(f"✅ Signal #{signals_generated} generated: {signal.decision} (score: {signal.score:.4f}, confidence: {signal.confidence:.2f})")
                    
                    if signal.decision != "NO_TRADE":
                        logger.success(f"🎯 Trade signal: {signal.decision} for {signal.symbol}")
                        
                        # Test risk filters
                        logger.debug("Applying risk filters...")
                        risk_result = apply_filters(signal, feature_store, settings.model_dump())
                        
                        if risk_result.passed:
                            signals_passed += 1
                            logger.success(f"🚀 Signal passed all filters! #{signals_passed}")
                        else:
                            logger.warning(f"❌ Signal blocked: {risk_result.reason}")
                            logger.debug(f"Filter details: {risk_result.details}")
                    
                except Exception as e:
                    logger.error(f"Error in signal generation: {e}", exc_info=True)
                
                # Stop after processing a few closed klines if we've seen signals
                if closed_klines >= 5 and signals_generated > 0:
                    logger.info("Debug session complete - signals detected")
                    break
                elif closed_klines >= 10:
                    logger.warning("Debug session complete - no signals generated")
                    break
    
    except Exception as e:
        logger.error(f"Error in debug session: {e}", exc_info=True)
    finally:
        if 'funding_task' in locals():
            funding_task.cancel()
        await ws_client.stop()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("DEBUG SESSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total events processed: {total_events}")
        logger.info(f"Kline events: {kline_events}")
        logger.info(f"Closed klines processed: {closed_klines}")
        logger.info(f"Warmup completed: {warmup_complete}")
        logger.info(f"Signals generated: {signals_generated}")
        logger.info(f"Signals passed filters: {signals_passed}")
        logger.info("="*60)


if __name__ == "__main__":
    try:
        asyncio.run(debug_signal_generation())
    except KeyboardInterrupt:
        logger.info("Debug session stopped by user")
