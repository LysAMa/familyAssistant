
# RouteWise (Python MVP)

FastAPI backend + tiny static UI (PWA-ready).
- Weather via OpenWeather
- Driving ETA via OpenRouteService (ORS) + Google Maps (traffic-aware)
- Transit (STM Montréal) via GTFS-RT
- Simple reasoning endpoint

## What it solves and for whom
- This app is to help families organize their schedule

But it could also be:
- Commuters and delivery teams who need one screen that blends live weather, traffic-aware driving ETAs, and transit feeds before leaving.
- City ops/dispatch dashboards that want lightweight, self-hostable route intelligence without wiring together multiple vendor SDKs.
- Product teams prototyping multimodal routing experiences who need a ready-made backend with simple APIs and a minimal PWA front-end.

## Why install this

- Single API surface for weather, multimodal routing, transit GTFS-RT, and basic reasoning—no need to stitch services yourself.
- Runs locally with your own API keys; no external SaaS dependency and easy to fork/extend.
- Tiny UI is PWA-ready so non-technical users can pin it and get real-time updates.
- Simple `/api/agent` endpoint lets you experiment with LLM-powered route explanations if you provide `OPENAI_API_KEY`.

## How we'll attract users

- it is easy to use and intrutive the user only have every event on the schedule for the upcoming day

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your keys (see .env.example)
python run.py
```

Open http://localhost:8000 and click the blue bubble.

## Env

- `OPENWEATHER_KEY` — from https://openweathermap.org/
- `ORS_KEY` — from https://openrouteservice.org/ (Directions)
- `GOOGLE_MAPS_KEY` — from Google Cloud (Directions API enabled, for traffic-aware routes)
- `STM_API_KEY` — from STM open data portal (GTFS-RT); `STM_TRIP_UPDATES_URL` optional override
- `OPENAI_API_KEY` (optional, not used in MVP code path)

## API quick references

- `GET /api/weather?lat=&lon=` — OpenWeather current + short forecast
- `GET /api/eta?o_lat=&o_lon=&d_lat=&d_lon&mode=` — ORS driving/cycling/walking
- `GET /api/directions/google?o_lat=&o_lon=&d_lat=&d_lon&mode=` — Google Maps directions (traffic for driving)
- `GET /api/transit/stm?stop_id=&route_id=&limit=` — STM GTFS-RT trip updates (filtered by stop/route)
- `GET /api/geocode?q=` — ORS geocoder
- `POST /api/reason` — simple heuristic explanation of mode choice
- `POST /api/agent` — LLM reasoning agent: body `{"query": "...", "origin": {"lat":..,"lon":..}, "dest": {"lat":..,"lon":..}, "mode":"driving"}` (uses OpenAI if `OPENAI_API_KEY` set)
- `GET /api/context` — returns preferences + recent history
- `POST /api/preferences` — update preferences, body e.g. `{"preferred_modes":["transit","walking"],"weather_sensitivity":8,"time_flex_minutes":15}`
- `POST /api/routes` — save a route to watch for alerts, body `{"label":"...","origin":{"lat","lon"},"dest":{"lat","lon"},"mode":"driving"}`
- `GET /api/alerts` — check for significant ETA/weather changes on saved routes
