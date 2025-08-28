# Redis Configuration for Sniper Mode

## Overview

The sniper mode enforcement can use Redis for distributed counter management across multiple runner processes. This ensures accurate signal counting in production deployments.

## Redis Setup

### Basic Redis Configuration

Add to `settings.yaml`:

```yaml
redis:
  enabled: true
  host: "localhost"
  port: 6379
  db: 0
  password: null  # or set via environment variable
  timeout: 5
```

### Environment Variables

For production, use environment variables:

```bash
export ULTRA_SIGNALS_REDIS__ENABLED=true
export ULTRA_SIGNALS_REDIS__HOST=your-redis-host
export ULTRA_SIGNALS_REDIS__PASSWORD=your-redis-password
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  ultra-signals:
    build: .
    depends_on:
      - redis
    environment:
      - ULTRA_SIGNALS_REDIS__ENABLED=true
      - ULTRA_SIGNALS_REDIS__HOST=redis
    volumes:
      - ./settings.yaml:/app/settings.yaml

volumes:
  redis_data:
```

## Monitoring

### Prometheus Metrics

The following metrics are exported:

- `sniper_hourly_cap_total` - Signals blocked by hourly limit
- `sniper_daily_cap_total` - Signals blocked by daily limit  
- `sniper_mtf_required_total` - Signals blocked by MTF requirement

### Grafana Dashboard

Import the dashboard from `dashboards/sniper-mode-dashboard.json` to monitor:

- Rejection rates over time
- Current signal counts
- Latency metrics
- Orders vs rejections ratio

### Redis Key Structure

Keys used by sniper counters:

- `sniper_signals:hour:{bucket}` - Hourly signal count for time bucket
- `sniper_signals:day:{bucket}` - Daily signal count for time bucket

Keys auto-expire after 2x their window size for cleanup.

## Fallback Behavior

If Redis is unavailable:
- System falls back to in-memory counters automatically
- Logging indicates fallback mode
- Single-process deployments work normally
- Multi-process deployments may have inconsistent counts

## Testing

Run Redis tests:

```bash
# Start Redis locally
docker run -d -p 6379:6379 redis:7-alpine

# Run tests with Redis enabled
python -m pytest ultra_signals/tests/test_sniper_redis.py -v -k "redis_enabled"
```

## Production Considerations

1. **High Availability**: Use Redis Cluster or Sentinel for production
2. **Persistence**: Enable AOF persistence for durability
3. **Monitoring**: Monitor Redis memory usage and key expiration
4. **Security**: Use AUTH and TLS in production environments
5. **Backup**: Regular backups if signal history is critical

## Troubleshooting

### Common Issues

1. **Connection timeouts**: Increase `redis.timeout` in settings
2. **Memory usage**: Monitor Redis memory, keys auto-expire
3. **Network issues**: Check Redis connectivity from runner hosts
4. **Permissions**: Ensure Redis user has read/write access

### Debug Commands

```bash
# Check Redis connectivity
redis-cli -h your-host ping

# View sniper keys
redis-cli -h your-host --scan --pattern "sniper_signals:*"

# Check key TTL
redis-cli -h your-host TTL sniper_signals:hour:12345

# Monitor operations
redis-cli -h your-host MONITOR
```
