
import httpx
from ..config import get_settings

async def fetch_weather(lat: float, lon: float):
    s = get_settings()
    if not s.openweather_key:
        return {"error": "Missing OPENWEATHER_KEY"}
    params = {
        "lat": lat,
        "lon": lon,
        "units": "metric",
        "lang": "fr",
        "appid": s.openweather_key,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r_now = await client.get("https://api.openweathermap.org/data/2.5/weather", params=params)
            r_now.raise_for_status()
            r_fc = await client.get("https://api.openweathermap.org/data/2.5/forecast", params=params)
            r_fc.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Surface provider status codes as JSON instead of bubbling a 500/plain text error.
        return {"error": f"Weather API error ({exc.response.status_code})"}
    except httpx.RequestError as exc:
        return {"error": f"Weather request failed: {exc.__class__.__name__}"}

    try:
        now = r_now.json()
        fc = r_fc.json()
    except ValueError:
        return {"error": "Weather API returned an invalid response"}
    curr = {
        "tempC": now.get("main", {}).get("temp"),
        "feelsLikeC": now.get("main", {}).get("feels_like"),
        "humidity": now.get("main", {}).get("humidity"),
        "windKmh": round(now.get("wind", {}).get("speed", 0) * 3.6, 1) if now.get("wind") else None,
        "condition": (now.get("weather") or [{}])[0].get("description", ""),
        "icon": (now.get("weather") or [{}])[0].get("icon", ""),
    }
    hourly = []
    for item in (fc.get("list") or [])[:4]:
        hourly.append({
            "t": item["dt"] * 1000,
            "tempC": item.get("main", {}).get("temp"),
            "pop": item.get("pop"),
            "weather": (item.get("weather") or [{}])[0].get("main", ""),
            "icon": (item.get("weather") or [{}])[0].get("icon", ""),
        })
    return {"provider": "openweather", "current": curr, "hourly": hourly}
