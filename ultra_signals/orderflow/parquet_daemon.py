"""Background asyncio daemon to flush FeatureViewWriter batches periodically."""
import asyncio
from typing import Optional


class ParquetDaemon:
    def __init__(self, writer, interval: float = 5.0):
        self.writer = writer
        self.interval = interval
        self._task: Optional[asyncio.Task] = None

    async def _loop(self):
        try:
            while True:
                await asyncio.sleep(self.interval)
                try:
                    # invoke private flush method if available
                    if hasattr(self.writer, '_flush_batch_to_parquet'):
                        self.writer._flush_batch_to_parquet()
                except Exception:
                    pass
        except asyncio.CancelledError:
            pass

    def start(self):
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())

    def stop(self):
        if self._task:
            self._task.cancel()


__all__ = ['ParquetDaemon']
