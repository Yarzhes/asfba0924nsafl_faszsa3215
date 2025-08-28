from typing import Optional
import aiohttp, asyncio
from .telegram import format_arb_telegram_line


class TelegramSender:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None, dry_run: bool = True):
        self.token = token
        self.chat_id = chat_id
        self.dry_run = dry_run

    async def send(self, feature_set):
        line = format_arb_telegram_line(feature_set)
        if self.dry_run or not self.token or not self.chat_id:
            print("[telegram dry-run]", line)
            return True
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        async with aiohttp.ClientSession() as sess:
            payload = {"chat_id": self.chat_id, "text": line}
            async with sess.post(url, json=payload) as r:
                return r.status == 200
