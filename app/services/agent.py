import asyncio
import httpx
from ..config import get_settings
from .weather import fetch_weather
from .eta import fetch_eta, PROFILE_MAP
from .google_maps import fetch_google_directions
from .stm_transit import fetch_stm_trip_updates
from .context import record_query, get_preferences
from .family_agents import orchestrate_family_plan


async def _call_openai(prompt: str, api_key: str) -> str | None:
    """Call OpenAI Chat Completions with a concise prompt."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 180,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def _fallback_recommendation(query: str, data: dict, prefs: dict) -> str:
    wx = data.get("weather", {})
    eta = data.get("eta", {})
    cond = (wx.get("current", {}) or {}).get("condition", "")
    eta_min = eta.get("etaMinutes")
    modes = ", ".join(prefs.get("preferred_modes") or [])
    parts = ["Suggestion basée sur ETA et météo."]
    if eta_min:
        parts.append(f"Trajet estimé: {round(eta_min)} min.")
    if cond:
        parts.append(f"Météo: {cond.lower()}.")
    if wx.get("current", {}).get("tempC") is not None:
        parts.append(f"Température: {round(wx['current']['tempC'])}°C.")
    if data.get("transit", {}).get("updates"):
        parts.append("Transports en commun: prochain passage disponible.")
    if modes:
        parts.append(f"Préférences: {modes}.")
    parts.append(f"Question: {query}")
    return " ".join(parts)


async def run_agent(query: str, origin: dict, dest: dict, mode: str = "driving", family: dict | None = None) -> dict:
    """Aggregate data, orchestrate family logistics, and ask LLM to synthesize a recommendation."""
    s = get_settings()
    prefs = get_preferences()
    # Fetch data concurrently
    tasks = [
        fetch_weather(origin["lat"], origin["lon"]),
        fetch_eta(origin, dest, mode),
        fetch_google_directions(origin, dest, mode),
        fetch_stm_trip_updates(limit=3),
    ]
    weather, eta, google_eta, transit = await asyncio.gather(*tasks)
    eta_choice = eta if not eta.get("error") else google_eta
    family_plan = orchestrate_family_plan(family, weather, eta_choice, prefs, s.openai_api_key)
    data = {
        "weather": weather,
        "eta": eta_choice,
        "transit": transit,
        "preferences": prefs,
        "family_plan": family_plan,
    }

    # Build prompt context
    dest_label = dest.get("label") or f"{dest['lat']},{dest['lon']}"
    temp_c = None
    try:
        temp_c = (weather.get("current") or {}).get("tempC")
    except Exception:
        temp_c = None
    temp_txt = f"{round(temp_c)}°C" if isinstance(temp_c, (int, float)) else "inconnue"
    profile_txt = ", ".join([f"{k}->{v}" for k, v in PROFILE_MAP.items()])

    prompt = (
        f"Demande: {query}\n"
        f"Origine: {origin['lat']},{origin['lon']} -> Destination: {dest_label} ({dest['lat']},{dest['lon']})\n"
        f"Météo actuelle: {temp_txt}, détails: {weather}\n"
        f"Préférences: {prefs}\n"
        f"Météo: {weather}\n"
        f"ETA (ORS/Google): {data['eta']}\n"
        f"Profils disponibles: {profile_txt}\n"
        f"Transit STM: {transit}\n"
        "Donne une recommandation brève (1-3 phrases) du meilleur mode et conseils pratiques."
    )

    rec = None
    if s.openai_api_key:
        rec = await _call_openai(prompt, s.openai_api_key)
    if not rec:
        rec = _fallback_recommendation(query, data, prefs)

    # learn from request
    record_query(query, (eta_choice or {}).get("mode"), (eta_choice or {}).get("etaMinutes"))

    return {
        "provider": "routewise-agent",
        "recommendation": rec,
        "data": data,
    }
