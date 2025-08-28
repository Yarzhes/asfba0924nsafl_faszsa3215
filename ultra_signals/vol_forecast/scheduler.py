from typing import Callable, Dict, Any
from datetime import datetime, timedelta
import threading

class SimpleRefitScheduler:
    """A tiny scheduler to run refit callbacks per-symbol at a cadence.

    Not a production cron; runs callbacks on a background thread using sleep.
    """

    def __init__(self):
        self.tasks = []
        self._stop = False

    def add_periodic(self, symbol: str, interval_seconds: int, callback: Callable[[str], Any]):
        self.tasks.append({"symbol": symbol, "interval": interval_seconds, "callback": callback, "last_run": None})

    def start(self):
        def loop():
            while not self._stop:
                now = datetime.utcnow()
                for t in self.tasks:
                    lr = t["last_run"]
                    if lr is None or (now - lr).total_seconds() >= t["interval"]:
                        try:
                            t["callback"](t["symbol"])
                        except Exception:
                            pass
                        t["last_run"] = now
                threading.Event().wait(1)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
