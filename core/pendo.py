"""
core/pendo.py
-------------
Server-side Pendo Track Event utility.

Sends track events to the Pendo data API via HTTP POST in a background
thread so tracking never blocks the main application flow.
"""

import requests
import time
import threading

PENDO_TRACK_URL = "https://data.pendo.io/data/track"
PENDO_INTEGRATION_KEY = "e7583a38-ef8d-46bd-b297-c59f8ceea957"


def track(event: str, properties: dict = None, visitor_id: str = "system", account_id: str = "system"):
    """Send a server-side track event to Pendo in a background thread."""
    payload = {
        "type": "track",
        "event": event,
        "visitorId": visitor_id,
        "accountId": account_id,
        "timestamp": int(time.time() * 1000),
        "properties": properties or {},
    }

    def _send():
        try:
            requests.post(
                PENDO_TRACK_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-pendo-integration-key": PENDO_INTEGRATION_KEY,
                },
                timeout=5,
            )
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()
