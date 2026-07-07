"""Minimal Vast.ai helpers for the self-hosted GPU that runs the SLMs.

Vast.ai is a spot GPU marketplace billed by the hour — this is the compute the
bot must earn enough to pay for. These helpers wrap the public REST API to list
cheap offers and report the current hourly burn so the economics layer can be
fed a real rate. Requires VAST_API_KEY; without it the functions return None.
"""
from __future__ import annotations

from ..config import secret

API = "https://console.vast.ai/api/v0"


def _client():
    import httpx

    key = secret("VAST_API_KEY")
    if not key:
        return None
    return httpx.Client(base_url=API, params={"api_key": key}, timeout=20.0)


def cheapest_offers(gpu_name: str = "RTX_3090", limit: int = 5) -> list[dict] | None:
    """Return the cheapest on-demand offers for a GPU type (USD/hr)."""
    client = _client()
    if client is None:
        return None
    q = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "gpu_name": {"eq": gpu_name},
        "order": [["dph_total", "asc"]],
    }
    try:
        r = client.get("/bundles", params={"q": __import__("json").dumps(q)})
        r.raise_for_status()
        offers = r.json().get("offers", [])[:limit]
        return [
            {"id": o.get("id"), "gpu": o.get("gpu_name"),
             "usd_per_hour": o.get("dph_total"), "region": o.get("geolocation")}
            for o in offers
        ]
    except Exception:
        return None
    finally:
        client.close()


def current_hourly_burn() -> float | None:
    """Sum dph_total of the account's running instances (USD/hr)."""
    client = _client()
    if client is None:
        return None
    try:
        r = client.get("/instances")
        r.raise_for_status()
        inst = r.json().get("instances", [])
        return round(sum(float(i.get("dph_total", 0) or 0) for i in inst), 4)
    except Exception:
        return None
    finally:
        client.close()
