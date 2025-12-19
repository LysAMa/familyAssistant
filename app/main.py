
from fastapi import FastAPI, Query, HTTPException
import asyncio
from fastapi.responses import HTMLResponse, FileResponse
from .services.weather import fetch_weather
from .services.eta import fetch_eta
from .services.reason import explain_recommendation
from .services.calendar import fetch_ics, next_event_from_ics
from .services.geocode import search_address, reverse_geocode
from .services.agent import run_agent
from .services.google_maps import fetch_google_directions
from .services.stm_transit import fetch_stm_trip_updates
from .services.context import get_context, get_preferences, update_preferences
from .services.family_agents import delete_family_event, get_family_state
from .services.alerts import save_route, list_routes, check_alerts
from pathlib import Path

app = FastAPI(title="Family Assistant")

@app.get("/", response_class=HTMLResponse)
def index():
    html = Path(__file__).with_name("static").joinpath("index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.get("/manifest.webmanifest")
def manifest():
    f = Path(__file__).with_name("static").joinpath("manifest.webmanifest")
    return FileResponse(f)

@app.get("/sw.js")
def sw():
    f = Path(__file__).with_name("static").joinpath("sw.js")
    return FileResponse(f, media_type="text/javascript")

@app.get("/api/weather")
async def api_weather(lat: float, lon: float):
    return await fetch_weather(lat, lon)

@app.get("/api/eta")
async def api_eta(
    o_lat: float, o_lon: float,
    d_lat: float, d_lon: float,
    mode: str = Query("driving")
):
    origin = {"lat": o_lat, "lon": o_lon}
    dest = {"lat": d_lat, "lon": d_lon}
    return await fetch_eta(origin, dest, mode)

@app.get("/api/eta/all")
async def api_eta_all(
    o_lat: float, o_lon: float,
    d_lat: float, d_lon: float,
):
    origin = {"lat": o_lat, "lon": o_lon}
    dest = {"lat": d_lat, "lon": d_lon}
    modes = ["driving", "cycling", "walking"]
    tasks = [fetch_eta(origin, dest, m) for m in modes]
    results = await asyncio.gather(*tasks)
    legs = []
    for m, res in zip(modes, results):
        if isinstance(res, dict):
            res = {**res, "mode": res.get("mode", m)}
        legs.append(res)
    return {"legs": legs}

@app.get("/api/directions/google")
async def api_google_directions(
    o_lat: float, o_lon: float,
    d_lat: float, d_lon: float,
    mode: str = Query("driving")
):
    origin = {"lat": o_lat, "lon": o_lon}
    dest = {"lat": d_lat, "lon": d_lon}
    return await fetch_google_directions(origin, dest, mode)

@app.post("/api/reason")
async def api_reason(legs: list[dict]):
    return explain_recommendation(legs, weather=None)

@app.post("/api/agent")
async def api_agent(payload: dict):
    try:
        query = payload.get("query") or ""
        origin = payload.get("origin") or {}
        dest = payload.get("dest") or {}
        mode = payload.get("mode", "driving")
        family = payload.get("family")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")
    if not all(k in origin for k in ("lat","lon")) or not all(k in dest for k in ("lat","lon")):
        raise HTTPException(status_code=400, detail="Missing origin/dest")
    return await run_agent(query, origin, dest, mode, family)


@app.delete("/api/family/events")
async def api_family_delete_event(payload: dict | None = None):
    try:
        criteria = payload or {}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")
    res = delete_family_event(criteria)
    return res

@app.get("/api/context")
async def api_context():
    return get_context()

@app.get("/api/family")
async def api_family():
    return get_family_state()

@app.post("/api/preferences")
async def api_preferences(payload: dict):
    preferred_modes = payload.get("preferred_modes")
    weather_sensitivity = payload.get("weather_sensitivity")
    time_flex_minutes = payload.get("time_flex_minutes")
    return update_preferences(preferred_modes, weather_sensitivity, time_flex_minutes)

@app.get("/api/calendar/next")
async def api_calendar_next(ics_url: str):
    try:
        ics = await fetch_ics(ics_url)
    except Exception:
        # Return JSON error instead of a plain text 500 to keep the client-side parser happy.
        raise HTTPException(status_code=502, detail="Calendar service unavailable")
    ev = next_event_from_ics(ics)
    return {"event": ev}

@app.get("/api/geocode")
async def api_geocode(q: str):
    return await search_address(q)

@app.get("/api/geocode/reverse")
async def api_geocode_reverse(lat: float, lon: float):
    return await reverse_geocode(lat, lon)

@app.get("/api/transit/stm")
async def api_transit_stm(stop_id: str | None = None, route_id: str | None = None, limit: int = 5):
    limit = max(1, min(limit, 20))
    return await fetch_stm_trip_updates(stop_id=stop_id, route_id=route_id, limit=limit)

@app.post("/api/routes")
async def api_routes(payload: dict):
    try:
        label = payload.get("label") or "Route"
        origin = payload["origin"]
        dest = payload["dest"]
        mode = payload.get("mode", "driving")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")
    if not all(k in origin for k in ("lat","lon")) or not all(k in dest for k in ("lat","lon")):
        raise HTTPException(status_code=400, detail="Missing origin/dest")
    return save_route(label, origin, dest, mode)

@app.get("/api/routes")
async def api_routes_list():
    return {"routes": list_routes()}

@app.get("/api/alerts")
async def api_alerts():
    return await check_alerts()
