import httpx, time
from ..config import get_settings


def _google_error(payload: dict) -> str:
    status = payload.get("status")
    msg = payload.get("error_message")
    if msg:
        return f"Google Directions error ({status}): {msg}"
    return f"Google Directions error ({status})"


async def fetch_google_directions(origin: dict, dest: dict, mode: str = "driving") -> dict:
    """Directions via Google Maps API with traffic for driving."""
    s = get_settings()
    if not s.google_maps_key:
        return {"error": "Missing GOOGLE_MAPS_KEY"}

    params = {
        "origin": f"{origin['lat']},{origin['lon']}",
        "destination": f"{dest['lat']},{dest['lon']}",
        "mode": mode,
        "key": s.google_maps_key,
    }
    if mode == "driving":
        params["departure_time"] = "now"
        params["traffic_model"] = "best_guess"

    url = "https://maps.googleapis.com/maps/api/directions/json"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as exc:
        return {"error": f"Google Directions HTTP error ({exc.response.status_code})"}
    except httpx.RequestError as exc:
        return {"error": f"Google Directions request failed: {exc.__class__.__name__}"}
    except ValueError:
        return {"error": "Google Directions returned invalid JSON"}

    if data.get("status") != "OK":
        return {"error": _google_error(data)}

    route = (data.get("routes") or [{}])[0]
    leg = (route.get("legs") or [{}])[0]

    def meters_to_km(val): return round((val or 0) / 1000, 1)
    def seconds_to_min(val):
        try:
            # Accept ints, floats, numeric strings
            return round(float(val or 0) / 60.0)
        except (TypeError, ValueError):
            # Fallback if val is completely non-numeric
            return 0

    eta_field = leg.get("duration_in_traffic") or leg.get("duration") or {}
    return {
        "provider": "google",
        "mode": mode,
        "distanceKm": meters_to_km((leg.get("distance") or {}).get("value")),
        "etaMinutes": seconds_to_min(eta_field.get("value")),
        "summary": route.get("summary") or leg.get("start_address"),
        "polyline": (route.get("overview_polyline") or {}).get("points"),
        "rawStatus": data.get("status"),
    }
