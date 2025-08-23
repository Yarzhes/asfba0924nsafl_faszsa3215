# -*- coding: utf-8 -*-
"""
Tests for the FundingProvider.
"""
import pytest
from unittest.mock import AsyncMock, patch
from ultra_signals.data.funding_provider import FundingProvider

@pytest.mark.asyncio
async def test_funding_provider_fetches_and_caches_data():
    """
    Tests that the FundingProvider can successfully fetch data from the REST
    endpoint and store it in the cache.
    """
    # 1. Arrange
    mock_config = {"refresh_interval_minutes": 10}
    mock_api_response = [
        {
            "symbol": "BTCUSDT",
            "fundingRate": "0.0001",
            "fundingTime": "1672531200000",
        },
        {
            "symbol": "ETHUSDT",
            "fundingRate": "-0.0002",
            "fundingTime": "1672531200000",
        },
    ]

    provider = FundingProvider(mock_config)

    # 2. Act
    def mock_json():
        return mock_api_response

    with patch(
        "ultra_signals.data.funding_provider.httpx.AsyncClient.get",
        new_callable=AsyncMock,
    ) as mock_get:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = mock_json
        mock_get.return_value = mock_response

        await provider._fetch_all_symbols()

    # 3. Assert
    cached_data = provider.get_history("BTCUSDT")
    # Check BTCUSDT
    btc_data = provider.get_history("BTCUSDT")
    assert btc_data is not None
    assert len(btc_data) == 1
    assert btc_data[0]["funding_rate"] == 0.0001
    assert btc_data[0]["funding_time"] == 1672531200000

    # Check ETHUSDT
    eth_data = provider.get_history("ETHUSDT")
    assert eth_data is not None
    assert len(eth_data) == 1
    assert eth_data[0]["funding_rate"] == -0.0002