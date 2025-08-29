"""
Real-time Trading Signal Runner

This is the main entry point for the Ultra-Signals application. It orchestrates
the flow of data and logic from all the different components:

1.  Loads settings from a YAML file.
2.  Initializes the Binance WebSocket client to stream market data.
3.  Initializes the FeatureStore to hold time-series data.
4.  Enters a main loop that:
    - Listens for kline events from the WebSocket.
    - On each **closed** kline, it updates the FeatureStore.
    - Triggers the feature computation pipeline.
    - Runs the scoring engine to generate a potential signal.
    - Applies risk filters.
    - If a valid signal is produced, it sends a notification via Telegram.
5.  Handles graceful shutdown on keyboard interrupt (Ctrl+C).
"""
import argparse
import asyncio

from loguru import logger
from collections import defaultdict

from ultra_signals.core.config import load_settings, ConfigError
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.events import KlineEvent
from ultra_signals.core.custom_types import FeatureVector
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
    compute_orderbook_features_v2,
    compute_cvd_features,
    BookFlipState,
    CvdState,
)
from ultra_signals.engine.scoring import component_scores
from ultra_signals.engine.entries_exits import make_signal
from ultra_signals.engine.risk_filters import apply_filters
import ultra_signals.engine.sizing as sizing_module
from ultra_signals.transport.telegram import format_message, send_message
from ultra_signals.live.metrics import Metrics


async def main_loop():
    """The main orchestration loop of the application."""
    parser = argparse.ArgumentParser(description="Ultra-Signals Realtime Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="settings.yaml",
        help="Path to the configuration file (default: settings.yaml)",
    )
    args = parser.parse_args()

    # --- 1. Load Settings ---
    try:
        settings = load_settings(args.config)
    except ConfigError as e:
        logger.error(f"Failed to start due to configuration error: {e}")
        return

    # --- 2. Initialize Components ---
    ws_client = BinanceWSClient(settings)
    # Add depth and trade streams to the subscription
    stream_types = settings.runtime.timeframes + ["depth", "aggTrade"]
    ws_client.subscribe(settings.runtime.symbols, stream_types)

    funding_provider = FundingProvider(settings.funding_rate_provider.model_dump())

    # SINGLE FEATURESTORE FIX: create exactly one FeatureStore and guard against duplicates
    feature_store = FeatureStore(
        warmup_periods=settings.features.warmup_periods,
        funding_provider=funding_provider,
    )
    # Guard: shout if another FeatureStore exists in-process
    existing_id = getattr(FeatureStore, "_process_singleton_id", None)
    if existing_id is None:
        FeatureStore._process_singleton_id = id(feature_store)
        logger.debug(
            f"[FeatureStore] initialized as process singleton id={id(feature_store)}"
        )
    elif existing_id != id(feature_store):
        # If some other part of the program created another FeatureStore earlier,
        # fail fast so we don't read from an empty cache elsewhere.
        raise RuntimeError(
            "Multiple FeatureStore instances detected in realtime runner! "
            f"existing_id={existing_id} new_id={id(feature_store)}"
        )
    else:
        logger.debug(
            f"[FeatureStore] reusing existing process singleton id={existing_id}"
        )

    # State management for new features
    book_flip_states = defaultdict(BookFlipState)
    cvd_states = defaultdict(CvdState)

    # lightweight runtime metrics collector (used to surface pre-trade summaries)
    metrics = Metrics()

    logger.info("Application starting up...")
    logger.info(f"Using FeatureStore id={id(feature_store)} for all realtime computations")

    # --- 3. Start Background Tasks & Main Loop ---
    try:
        # Start the funding provider's refresh loop as a background task
        funding_task = asyncio.create_task(funding_provider.start())

        async for event in ws_client.start():
            # Ingest all events, but only trigger feature computation on closed klines
            feature_store.ingest_event(event)

            # EXTRA RUNTIME GUARD: ensure no one swapped the global store under us
            singleton_id_now = getattr(FeatureStore, "_process_singleton_id", None)
            if singleton_id_now != id(feature_store):
                raise RuntimeError(
                    "FeatureStore singleton id changed during runtime. "
                    f"expected={id(feature_store)} got={singleton_id_now}"
                )

            # --- 4. Trigger on ANY Kline Update for Primary Timeframe ---
            if not isinstance(event, KlineEvent):
                continue

            kline = event  # For clarity
            if kline.timeframe != settings.runtime.primary_timeframe:
                continue

            # Process BOTH open and closed klines for real-time signal generation
            status = "CLOSED" if kline.closed else "OPEN"
            logger.debug(
                f"Processing {status} kline for {kline.symbol}/{kline.timeframe} "
                f"(FeatureStore id={id(feature_store)})..."
            )

            # --- 5. Feature Computation ---
            ohlcv = feature_store.get_ohlcv(kline.symbol, kline.timeframe)
            if ohlcv is None or len(ohlcv) < settings.features.warmup_periods:
                logger.info(
                    f"[DEBUG] {kline.symbol} warmup check: ohlcv_len={len(ohlcv) if ohlcv is not None else 0}, required={settings.features.warmup_periods} - SKIPPING"
                )
                continue

            logger.info(f"[DEBUG] {kline.symbol} warmup check: PASSED with {len(ohlcv)} periods")

            # --- 5a. OHLCV-based features ---
            ohlcv_features: dict = {}
            ohlcv_features.update(
                compute_trend(ohlcv, **settings.features.trend.model_dump())
            )
            ohlcv_features.update(
                compute_momentum(ohlcv, **settings.features.momentum.model_dump())
            )
            ohlcv_features.update(
                compute_volatility(ohlcv, **settings.features.volatility.model_dump())
            )
            ohlcv_features.update(
                compute_volume_flow_features(
                    ohlcv, **settings.features.volume_flow.model_dump()
                )
            )

            # --- 5b. Live features (Orderbook, Derivatives) ---
            orderbook_features = compute_orderbook_features(feature_store, kline.symbol)
            # New: compute richer derivatives posture where available (funding, OI, basis)
            try:
                from ultra_signals.features.derivatives_posture import compute_derivatives_posture
                derivatives_features = compute_derivatives_posture(feature_store, kline.symbol)
            except Exception:
                # fallback to legacy lightweight derivatives features
                derivatives_features = compute_derivatives_features(feature_store, kline.symbol)
            funding_features = compute_funding_features(feature_store, kline.symbol)

            # --- 5c. V2 Features (optional) ---
            ob_v2_features = None
            cvd_features = None
            
            # Only compute if configurations exist
            if hasattr(settings.features, 'orderbook_v2'):
                ob_v2_features = compute_orderbook_features_v2(
                    feature_store,
                    kline.symbol,
                    book_flip_states[kline.symbol],
                    settings.features.orderbook_v2.model_dump(),
                )
            
            if hasattr(settings.features, 'cvd'):
                cvd_features = compute_cvd_features(
                    feature_store,
                    kline.symbol,
                    cvd_states[kline.symbol],
                    settings.features.cvd.model_dump(),
                )

            # Merge V2 features into the orderbook features dictionary for now
            if orderbook_features and ob_v2_features:
                orderbook_features.update(ob_v2_features.__dict__)
            if ohlcv_features and cvd_features:
                ohlcv_features.update(cvd_features.__dict__)

            # --- 5d. Assemble final feature vector ---
            feature_vector = FeatureVector(
                symbol=kline.symbol,
                timeframe=kline.timeframe,
                ohlcv=ohlcv_features,
                orderbook=orderbook_features.__dict__ if orderbook_features else None,
                derivatives=derivatives_features,
                funding=funding_features,
            )

            # --- 6. Scoring and Signal Generation ---
            scores = component_scores(
                feature_vector, settings.features.model_dump()
            )
            
            # DEBUG: Log scoring details
            weights = settings.engine.scoring_weights.model_dump() if hasattr(settings.engine.scoring_weights, 'model_dump') else settings.engine.scoring_weights
            final_score = sum(scores.get(comp, 0.0) * w for comp, w in weights.items())
            final_score = max(-1.0, min(1.0, final_score))  # Clip to [-1, 1]
            logger.info(f"[DEBUG] {kline.symbol} scoring: scores={scores}, weights={weights}, final_score={final_score}, threshold={settings.engine.thresholds.enter}")

            # Handle weights and thresholds properly - could be dict or Pydantic model  
            weights = settings.engine.scoring_weights.model_dump() if hasattr(settings.engine.scoring_weights, 'model_dump') else settings.engine.scoring_weights
            thresholds = settings.engine.thresholds.model_dump() if hasattr(settings.engine.thresholds, 'model_dump') else settings.engine.thresholds
            
            signal = make_signal(
                symbol=kline.symbol,
                timeframe=kline.timeframe,
                component_scores=scores,
                weights=weights,
                thresholds=thresholds,
                features=feature_vector,
                ohlcv=ohlcv,
            )
            
            # DEBUG: Log signal generation result
            logger.info(f"[DEBUG] {kline.symbol} signal generated: decision={signal.decision}, score={signal.score:.3f}")

            # --- 7. Risk Filtering and Sizing ---
            # IMPORTANT: pass the SAME feature_store instance into filters
            risk_result = apply_filters(signal, feature_store, settings.model_dump(), metrics=metrics)
            if not risk_result.passed:
                logger.info(
                    f"Signal for {signal.symbol} blocked by risk filter: {risk_result.reason}"
                )
                
                # Track sniper rejections in metrics
                if hasattr(metrics, 'inc_sniper_rejection') and risk_result.reason and 'SNIPER' in risk_result.reason:
                    metrics.inc_sniper_rejection(risk_result.reason)
                
                continue

            sized_signal = sizing_module.determine_position_size(signal, settings.model_dump())

            # --- 8. Transport / Notification ---
            if sized_signal.decision != "NO_TRADE":
                # build a compact pre-trade summary for telemetry/transport
                try:
                    pwin = float(sized_signal.confidence)
                except Exception:
                    pwin = None
                try:
                    regime = None
                    vd = sized_signal.vote_detail or {}
                    if isinstance(vd, dict):
                        rg = vd.get('regime') or {}
                        if isinstance(rg, dict):
                            regime = rg.get('regime_label') or rg.get('label')
                except Exception:
                    regime = None
                pre = {
                    "p_win": pwin,
                    "regime": regime,
                    "veto_count": len(getattr(sized_signal, 'vetoes', []) or []),
                    "lat_ms": metrics.latency_tick_to_decision.snapshot(),
                }
                # expose on metrics and on the decision for transport formatters
                metrics.set_pre_trade_summary(pre)
                try:
                    if isinstance(sized_signal.vote_detail, dict):
                        sized_signal.vote_detail['pre_trade'] = pre
                except Exception:
                    pass

                message = format_message(sized_signal, settings.model_dump())
                print("\n" + message + "\n")  # Also print to console

                if not settings.transport.dry_run:
                    await send_message(message, settings.transport.model_dump())
                else:
                    logger.info(
                        f"DRY RUN: Telegram message would be sent:\n---\n{message}\n---"
                    )

    except asyncio.CancelledError:
        logger.warning("Main loop cancelled.")
    finally:
        if "funding_task" in locals() and not funding_task.done():
            funding_task.cancel()
        await ws_client.stop()
        logger.info("Application shutdown complete.")


def run():
    """Entry point to run the main async loop."""
    # Setup Loguru to be cleaner
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C).")


if __name__ == "__main__":
    run()
