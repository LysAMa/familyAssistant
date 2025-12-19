import httpx
import os
from pathlib import Path
from ..config import get_settings
from .google_maps import fetch_google_directions

from .eta_service import ETAModel

# Resolve dataset directory relative to the project root by default.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "my_msltd_montreal"
ML_DATA_DIR = str(Path(os.environ.get("ML_DATA_DIR", _DEFAULT_DATA_DIR)))

eta_model: ETAModel | None = None  # global


def get_eta_model() -> ETAModel:
    global eta_model
    if eta_model is None:
        eta_model = ETAModel(ML_DATA_DIR)
    return eta_model


PROFILE_MAP = {
    "driving": "driving-car",
    "cycling": "cycling-regular",
    "walking": "foot-walking",
}


async def fetch_eta(origin: dict, dest: dict, mode: str = "driving"):
    s = get_settings()
    if not s.ors_key:
        return {"error": "Missing ORS_KEY"}

    profile = PROFILE_MAP.get(mode, "driving-car")
    start = f"{origin['lon']},{origin['lat']}"
    end = f"{dest['lon']},{dest['lat']}"
    url = f"https://api.openrouteservice.org/v2/directions/{profile}"
    params = {
        "start": start,
        "end": end,
        "api_key": s.ors_key,  # ORS also accepts api_key as query param
    }
    headers = {"Authorization": s.ors_key}

    def _extract_detail(response: httpx.Response) -> str | None:
        """Pull a short provider error message if present."""
        detail: str | None = None
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("message")
                if not detail and isinstance(payload.get("error"), dict):
                    detail = payload["error"].get("message")
        except ValueError:
            text = response.text
            detail = text.strip()[:120] if text else None
        return detail

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as exc:
        detail = _extract_detail(exc.response)
        msg = f"ETA API error ({exc.response.status_code})"
        if detail:
            msg += f": {detail}"
        # If ORS cannot snap to the road network, fall back to Google if available.
        if (
            exc.response.status_code == 404
            and detail
            and "routable point" in detail.lower()
            and s.google_maps_key
        ):
            google = await fetch_google_directions(origin, dest, mode)
            if not google.get("error"):
                google["fallback"] = "google"
                return google
        return {"error": msg}
    except httpx.RequestError as exc:
        return {"error": f"ETA request failed: {exc.__class__.__name__}"}
    except ValueError:
        return {"error": "ETA API returned an invalid response"}

    feat = (data.get("features") or [{}])[0]
    props = feat.get("properties", {})
    summary = props.get("summary", {})

    # Cast safe pour éviter les erreurs de division (strings vs float)
    raw_distance = summary.get("distance", 0) or 0
    raw_duration = summary.get("duration", 0) or 0
    try:
        distance_m = float(raw_distance)
    except (TypeError, ValueError):
        distance_m = 0.0
    try:
        duration_s = float(raw_duration)
    except (TypeError, ValueError):
        duration_s = 0.0

    distance_km = distance_m / 1000.0
    eta_min_ors = round(duration_s / 60.0)

    geometry = feat.get("geometry") or {}
    coords = geometry.get("coordinates") or []  # polyline ORS [ [lon,lat], ... ]

    result: dict = {
        "provider": "ors",
        "mode": mode,
        "distanceKm": round(distance_km, 1),
        "etaMinutes": eta_min_ors,  # ETA ORS (baseline)
        "summary": (
            props.get("segments", [{}])[0]
            .get("steps", [{}])[0]
            .get("name", "Itinéraire ORS")
        ),
        "geometry": geometry,
    }

    # ➕ Ajout ETA basé sur ton modèle Montréal (Bluetooth)
    if coords:
        try:
            ml = get_eta_model()
            eta_sec_ml, eta_min_ml, route_nodes, _ = ml.eta_for_polyline(coords)

            # S’assurer que c’est bien un float avant round()
            eta_model_minutes = round(float(eta_min_ml))
            result["etaModelMinutes"] = eta_model_minutes
            result["etaModelSeconds"] = float(eta_sec_ml)
            result["etaModelProvider"] = "ml-montreal"
            result["etaModelText"] = f"{eta_model_minutes} min (gcrnn_modele)"
            result["etaModelRouteNodes"] = route_nodes  # pratique pour debug

            # 👉 C'est ici que ça "print" l'ETA du modèle GCRNN en minutes
            print(f"[ML ETA] GCRNN Montréal: {eta_model_minutes} minutes")

        except Exception as e:
            result["etaModelError"] = str(e)

    return result
