"""
Market Metadata for Tick Size and Price Rounding

This module handles exchange-specific tick sizes and provides price rounding
functionality for consistent price formatting across the system.
"""

from typing import Dict, Optional, Union
from loguru import logger
import numpy as np
import math


# Default tick sizes for common symbols (can be overridden in settings)
DEFAULT_TICK_SIZES = {
    # Major pairs
    "BTCUSDT": 0.1,
    "ETHUSDT": 0.01,
    "BNBUSDT": 0.01,
    "SOLUSDT": 0.01,
    "XRPUSDT": 0.0001,
    "DOGEUSDT": 0.00001,
    "ADAUSDT": 0.0001,
    "AVAXUSDT": 0.01,
    "LINKUSDT": 0.001,
    "TONUSDT": 0.0001,
    "TRXUSDT": 0.00001,
    "DOTUSDT": 0.001,
    "NEARUSDT": 0.001,
    "ATOMUSDT": 0.001,
    "LTCUSDT": 0.01,
    "BCHUSDT": 0.01,
    "ARBUSDT": 0.001,
    "APTUSDT": 0.001,
    "MATICUSDT": 0.0001,
    "SUIUSDT": 0.001,
    
    # Minor pairs
    "1000SHIBUSDT": 0.00000001,
    "SHIBUSDT": 0.00000001,
    "PEPEUSDT": 0.00000001,
    "FLOKIUSDT": 0.00000001,
    "BONKUSDT": 0.00000001,
    "WIFUSDT": 0.0001,
    "JUPUSDT": 0.0001,
    "PYTHUSDT": 0.0001,
    "WLDUSDT": 0.0001,
    "ORDIUSDT": 0.01,
    "MEMEUSDT": 0.00000001,
    "BIGTIMEUSDT": 0.0001,
    "TIAUSDT": 0.001,
    "INJUSDT": 0.001,
    "SEIUSDT": 0.0001,
    "KASUSDT": 0.0001,
    "FETUSDT": 0.0001,
    "RUNEUSDT": 0.001,
    "THETAUSDT": 0.001,
    "ALGOUSDT": 0.0001,
    "VETUSDT": 0.00001,
    "ICPUSDT": 0.001,
    "FILUSDT": 0.001,
    "HBARUSDT": 0.0001,
    "AXSUSDT": 0.001,
    "SANDUSDT": 0.0001,
    "MANAUSDT": 0.0001,
    "GALAUSDT": 0.00001,
    "CHZUSDT": 0.0001,
    "HOTUSDT": 0.00000001,
    "ENJUSDT": 0.001,
    "BATUSDT": 0.0001,
    "ZILUSDT": 0.00001,
    "IOTAUSDT": 0.0001,
    "NEOUSDT": 0.001,
    "QTUMUSDT": 0.001,
    "ONTUSDT": 0.001,
    "ZRXUSDT": 0.0001,
    "OMGUSDT": 0.001,
    "KNCUSDT": 0.001,
    "COMPUSDT": 0.01,
    "MKRUSDT": 0.1,
    "AAVEUSDT": 0.01,
    "UNIUSDT": 0.001,
    "SUSHIUSDT": 0.001,
    "CRVUSDT": 0.001,
    "YFIUSDT": 1.0,
    "SNXUSDT": 0.001,
    "1INCHUSDT": 0.001,
    "ALPHAUSDT": 0.001,
    "ZENUSDT": 0.001,
    "STORJUSDT": 0.001,
    "SKLUSDT": 0.0001,
    "ANKRUSDT": 0.0001,
    "COTIUSDT": 0.0001,
    "CHRUSDT": 0.0001,
    "ALICEUSDT": 0.001,
    "MASKUSDT": 0.001,
    "DYDXUSDT": 0.001,
    "IMXUSDT": 0.001,
    "OPUSDT": 0.001,
    "ARBUSDT": 0.001,
    "MAGICUSDT": 0.0001,
    "STXUSDT": 0.001,
    "CFXUSDT": 0.0001,
    "SUIUSDT": 0.001,
    "APTUSDT": 0.001,
    "BLURUSDT": 0.001,
    "JASMYUSDT": 0.00000001,
    "PEOPLEUSDT": 0.0001,
    "GALUSDT": 0.001,
    "LDOUSDT": 0.001,
    "SSVUSDT": 0.01,
    "RPLUSDT": 0.001,
    "FXSUSDT": 0.001,
    "GMXUSDT": 0.01,
    "RNDRUSDT": 0.001,
    "OCEANUSDT": 0.0001,
    "AGIXUSDT": 0.0001,
    "ROSEUSDT": 0.0001,
    "IOTXUSDT": 0.00001,
    "HIVEUSDT": 0.001,
    "DASHUSDT": 0.01,
    "ZECUSDT": 0.01,
    "XMRUSDT": 0.01,
    "ETCUSDT": 0.01,
    "BTTUSDT": 0.00000001,
    "WINUSDT": 0.00000001,
    "CAKEUSDT": 0.001,
    "BAKEUSDT": 0.001,
    "AUDIOUSDT": 0.0001,
    "DENTUSDT": 0.00000001,
    "CKBUSDT": 0.00001,
    "CTXCUSDT": 0.0001,
    "CTSIUSDT": 0.0001,
    "DUSKUSDT": 0.0001,
    "EGLDUSDT": 0.01,
    "FLMUSDT": 0.0001,
    "FTMUSDT": 0.0001,
    "GRTUSDT": 0.0001,
    "HARDUSDT": 0.001,
    "HBARUSDT": 0.0001,
    "ICXUSDT": 0.001,
    "IRISUSDT": 0.0001,
    "KEYUSDT": 0.00000001,
    "LINAUSDT": 0.00000001,
    "LITUSDT": 0.001,
    "LPTUSDT": 0.001,
    "LRCUSDT": 0.0001,
    "MITHUSDT": 0.0001,
    "MTLUSDT": 0.001,
    "NULSUSDT": 0.001,
    "OGNUSDT": 0.001,
    "ONEUSDT": 0.00001,
    "PONDUSDT": 0.00000001,
    "PUNDIXUSDT": 0.001,
    "QUICKUSDT": 0.01,
    "REEFUSDT": 0.00000001,
    "RSRUSDT": 0.00000001,
    "RVNUSDT": 0.00001,
    "SFPUSDT": 0.001,
    "SLPUSDT": 0.0001,
    "SRMUSDT": 0.001,
    "STMXUSDT": 0.00000001,
    "SXPUSDT": 0.0001,
    "TLMUSDT": 0.0001,
    "TOMOUSDT": 0.001,
    "TRBUSDT": 0.001,
    "TROYUSDT": 0.00000001,
    "TWTUSDT": 0.001,
    "UMAUSDT": 0.001,
    "VTHOUSDT": 0.00000001,
    "WAVESUSDT": 0.001,
    "WRXUSDT": 0.001,
    "XEMUSDT": 0.0001,
    "XLMUSDT": 0.00001,
    "XTZUSDT": 0.001,
    "YFIIUSDT": 1.0,
    "YGGUSDT": 0.001,
    "ZENUSDT": 0.001,
    "ZILUSDT": 0.00001,
}


def get_tick_size(symbol: str, settings: Optional[Dict] = None) -> float:
    """
    Get tick size for a symbol.
    
    Args:
        symbol: Trading symbol
        settings: Application settings (optional)
        
    Returns:
        Tick size for the symbol
    """
    # Check settings override first
    if settings:
        tick_overrides = settings.get("formatting", {}).get("tick_size_overrides", {})
        if symbol in tick_overrides:
            return float(tick_overrides[symbol])
    
    # Use default tick sizes
    return DEFAULT_TICK_SIZES.get(symbol, 0.0001)  # Default to 0.0001


def round_to_tick(price: float, tick_size: float) -> float:
    """
    Round price to nearest tick size.
    
    Args:
        price: Raw price
        tick_size: Tick size
        
    Returns:
        Rounded price
    """
    if not isinstance(price, (int, float)) or not isinstance(tick_size, (int, float)):
        raise ValueError(f"Invalid price or tick_size: {price}, {tick_size}")
    
    if np.isnan(price) or np.isinf(price):
        raise ValueError(f"Invalid price: {price}")
    
    if tick_size <= 0:
        logger.warning(f"Invalid tick size: {tick_size}, using 0.0001")
        tick_size = 0.0001
    
    # Round to nearest tick size
    rounded = round(price / tick_size) * tick_size
    
    # Calculate decimal places for precision
    decimal_places = 0
    temp_tick = tick_size
    while temp_tick < 1 and temp_tick > 0:
        temp_tick *= 10
        decimal_places += 1
    
    # Round to proper decimal places to avoid floating point errors
    rounded = round(rounded, decimal_places)
    
    return rounded


def round_price_for_symbol(price: float, symbol: str, settings: Optional[Dict] = None) -> float:
    """
    Round price for a specific symbol using its tick size.
    
    Args:
        price: Raw price
        symbol: Trading symbol
        settings: Application settings (optional)
        
    Returns:
        Rounded price
    """
    tick_size = get_tick_size(symbol, settings)
    return round_to_tick(price, tick_size)


def format_price_for_display(price: float, symbol: str, settings: Optional[Dict] = None) -> str:
    """
    Format price for display with appropriate decimal places.
    
    Args:
        price: Raw price
        symbol: Trading symbol
        settings: Application settings (optional)
        
    Returns:
        Formatted price string
    """
    tick_size = get_tick_size(symbol, settings)
    
    # Calculate decimal places
    decimal_places = 0
    temp_tick = tick_size
    while temp_tick < 1 and temp_tick > 0:
        temp_tick *= 10
        decimal_places += 1
    
    # Round and format
    rounded_price = round_to_tick(price, tick_size)
    return f"{rounded_price:.{decimal_places}f}"


def validate_tick_size(tick_size: float) -> bool:
    """
    Validate that a tick size is reasonable.
    
    Args:
        tick_size: Tick size to validate
        
    Returns:
        True if tick size is valid
    """
    if not isinstance(tick_size, (int, float)):
        return False
    
    if np.isnan(tick_size) or np.isinf(tick_size):
        return False
    
    if tick_size <= 0:
        return False
    
    if tick_size > 1000:
        return False
    
    return True


def get_symbol_info(symbol: str, settings: Optional[Dict] = None) -> Dict:
    """
    Get comprehensive symbol information including tick size.
    
    Args:
        symbol: Trading symbol
        settings: Application settings (optional)
        
    Returns:
        Dictionary with symbol information
    """
    tick_size = get_tick_size(symbol, settings)
    
    # Calculate decimal places properly
    decimal_places = 0
    temp_tick = tick_size
    while temp_tick < 1 and temp_tick > 0:
        temp_tick *= 10
        decimal_places += 1
    
    return {
        "symbol": symbol,
        "tick_size": tick_size,
        "decimal_places": decimal_places,
        "min_price": tick_size,
        "max_price": 1000000.0,  # Reasonable upper bound
        "price_precision": decimal_places
    }


def batch_round_prices(
    prices: Dict[str, float],
    settings: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Round multiple prices for different symbols.
    
    Args:
        prices: Dictionary mapping symbols to prices
        settings: Application settings (optional)
        
    Returns:
        Dictionary with rounded prices
    """
    rounded_prices = {}
    
    for symbol, price in prices.items():
        rounded_prices[symbol] = round_price_for_symbol(price, symbol, settings)
    
    return rounded_prices
