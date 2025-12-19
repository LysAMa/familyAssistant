import asyncio
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Dict, Optional

from .eta import fetch_eta
from .weather import fetch_weather


@dataclass
class RouteSubscription:
    id: int
    label: str
    origin: Dict
    dest: Dict
    mode: str
    last_eta: Optional[int] = None
    last_condition: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@lru_cache(maxsize=1)
def _state():
    return {"routes": [], "next_id": 1}


def save_route(label: str, origin: Dict, dest: Dict, mode: str = "driving") -> dict:
    st = _state()
    rid = st["next_id"]
    st["next_id"] += 1
    sub = RouteSubscription(id=rid, label=label or f"Route {rid}", origin=origin, dest=dest, mode=mode)
    st["routes"].append(sub)
    return sub.to_dict()


def list_routes() -> List[dict]:
    return [r.to_dict() for r in _state()["routes"]]


async def check_alerts() -> dict:
    """Check for significant changes on saved routes."""
    st = _state()
    alerts = []
    for sub in st["routes"]:
        eta = await fetch_eta(sub.origin, sub.dest, sub.mode)
        wx = await fetch_weather(sub.origin.get("lat"), sub.origin.get("lon"))
        eta_min = eta.get("etaMinutes")
        cond = (wx.get("current", {}) or {}).get("condition") if isinstance(wx, dict) else None

        msg = None
        if eta_min and sub.last_eta and abs(eta_min - sub.last_eta) >= 10:
            diff = eta_min - sub.last_eta
            msg = f"{sub.label}: ETA a changé de {diff:+} min (maintenant {eta_min} min)."
        elif sub.last_condition and cond and cond.lower() != sub.last_condition.lower():
            msg = f"{sub.label}: météo est passée de {sub.last_condition} à {cond}."

        sub.last_eta = eta_min or sub.last_eta
        if cond:
            sub.last_condition = cond

        if msg:
            alerts.append({"routeId": sub.id, "message": msg, "eta": eta_min, "condition": cond})

    return {"alerts": alerts, "routes": list_routes()}
