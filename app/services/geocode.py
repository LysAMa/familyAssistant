import httpx
from ..config import get_settings


async def search_address(query: str, country: str = "CA") -> dict:
    """Lookup address candidates using OpenRouteService geocoding."""
    s = get_settings()
    if not s.ors_key:
        return {"error": "Missing ORS_KEY"}

    params = {
        "api_key": s.ors_key,
        "text": query,
        "boundary.country": country,
        "size": 5,
        "lang": "fr",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.openrouteservice.org/geocode/search", params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as exc:
        return {"error": f"Geocode API error ({exc.response.status_code})"}
    except httpx.RequestError as exc:
        return {"error": f"Geocode request failed: {exc.__class__.__name__}"}
    except ValueError:
        return {"error": "Geocode API returned an invalid response"}

    feats = data.get("features") or []
    results = []
    for f in feats:
        props = f.get("properties") or {}
        coords = (f.get("geometry") or {}).get("coordinates") or []
        if len(coords) >= 2:
            results.append({
                "label": props.get("label") or props.get("name"),
                "lat": coords[1],
                "lon": coords[0],
            })
    return {"results": results}


async def reverse_geocode(lat: float, lon: float) -> dict:
    """Reverse geocoding via ORS."""
    s = get_settings()
    if not s.ors_key:
        return {"error": "Missing ORS_KEY"}

    params = {
        "api_key": s.ors_key,
        "point.lat": lat,
        "point.lon": lon,
        "size": 1,
        "lang": "fr",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.openrouteservice.org/geocode/reverse", params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as exc:
        return {"error": f"Reverse geocode error ({exc.response.status_code})"}
    except httpx.RequestError as exc:
        return {"error": f"Reverse request failed: {exc.__class__.__name__}"}
    except ValueError:
        return {"error": "Reverse geocode returned invalid JSON"}

    feats = data.get("features") or []
    if not feats:
        return {"results": []}
    props = feats[0].get("properties") or {}
    return {"results": [{"label": props.get("label") or props.get("name"), "lat": lat, "lon": lon}]}
