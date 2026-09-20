"""Point the owner's Telegram bot at the deployed RealAI (webhook mode).

After deploying this repo to Vercel (``vercel --prod``) and setting the
env vars there (REALAI_TELEGRAM_BOT_TOKEN, REALAI_TELEGRAM_CHAT_ID,
optionally REALAI_TELEGRAM_WEBHOOK_SECRET + REALAI_WEB_TOKEN):

    export REALAI_TELEGRAM_BOT_TOKEN=123:ABC
    python examples/register_webhook.py https://your-app.vercel.app/webhook

The shared secret (recommended) must equal the
REALAI_TELEGRAM_WEBHOOK_SECRET env var on Vercel - Telegram will then
send it in ``X-Telegram-Bot-Api-Secret-Token`` on every update and the
/webhook route rejects anything else.

Same repo, same bot: this is the owner's own @BotFather bot - no
third-party services.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    url = argv[1]
    token = os.environ.get("REALAI_TELEGRAM_BOT_TOKEN", "").strip()
    secret = os.environ.get("REALAI_TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not token:
        print("set REALAI_TELEGRAM_BOT_TOKEN first (token from @BotFather)")
        return 1

    api = f"https://api.telegram.org/bot{token}"

    def call(method: str, params: dict) -> dict:
        req = urllib.request.Request(
            api + "/" + method + "?" + urllib.parse.urlencode(params),
            method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(resp.read().decode())
        if not out.get("ok"):
            raise SystemExit(f"telegram error: {out}")
        return out

    # sanity: can we reach the bot?
    me = call("getMe", {})
    print(f"bot: @{me['result'].get('username')}")

    res = call("setWebhook", {
        "url": url,
        "secret_token": secret,
        "drop_pending_updates": "true",
        "allowed_updates": json.dumps(["message"]),
    })
    print("setWebhook ok:", res["result"], "->", url)
    print("telegram -> /webhook -> RealAI agent (owner control + replies)")
    print("try it: send '/status' to the bot in your owner chat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
